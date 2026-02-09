"""Model training with cross-validation and experiment logging."""

import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.utils import DATA_DIR, FEATURES_DIR, OUTPUT_DIR, get_logger, timer

log = get_logger(__name__)

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "verbosity": 0,
}

# Columns to target-encode (high-cardinality categoricals)
TARGET_ENCODE_COLS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "NAME_FAMILY_STATUS",
]


def prepare_data(
    train_df: pl.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Polars DataFrame to numpy arrays for XGBoost."""
    # Drop string/categorical columns that XGBoost can't handle directly
    numeric_cols = []
    for c in feature_cols:
        dtype = train_df[c].dtype
        if dtype in (
            pl.Float32, pl.Float64,
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        ):
            numeric_cols.append(c)

    X = train_df.select(numeric_cols).to_numpy().astype(np.float32)
    y = train_df["TARGET"].to_numpy().astype(np.float32)
    return X, y, numeric_cols


def _target_encode_fold(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    col: str,
    target_col: str = "TARGET",
    smoothing: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Target-encode a single column using training fold stats.

    Returns (train_encoded, val_encoded) as numpy arrays.
    """
    global_mean = train_df[target_col].mean()

    # Compute per-category stats from training data only
    stats = (
        train_df
        .group_by(col)
        .agg(
            pl.col(target_col).mean().alias("cat_mean"),
            pl.col(target_col).count().alias("cat_count"),
        )
        .with_columns(
            (
                (pl.col("cat_count") * pl.col("cat_mean") + smoothing * global_mean)
                / (pl.col("cat_count") + smoothing)
            ).alias("te_value")
        )
        .select(col, "te_value")
    )

    # Join onto train and val
    train_encoded = train_df.join(stats, on=col, how="left")["te_value"].fill_null(global_mean).to_numpy().astype(np.float32)
    val_encoded = val_df.join(stats, on=col, how="left")["te_value"].fill_null(global_mean).to_numpy().astype(np.float32)

    return train_encoded, val_encoded


def _target_encode_full(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    col: str,
    target_col: str = "TARGET",
    smoothing: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Target-encode using full training set (for test predictions)."""
    global_mean = train_df[target_col].mean()

    stats = (
        train_df
        .group_by(col)
        .agg(
            pl.col(target_col).mean().alias("cat_mean"),
            pl.col(target_col).count().alias("cat_count"),
        )
        .with_columns(
            (
                (pl.col("cat_count") * pl.col("cat_mean") + smoothing * global_mean)
                / (pl.col("cat_count") + smoothing)
            ).alias("te_value")
        )
        .select(col, "te_value")
    )

    train_encoded = train_df.join(stats, on=col, how="left")["te_value"].fill_null(global_mean).to_numpy().astype(np.float32)
    test_encoded = test_df.join(stats, on=col, how="left")["te_value"].fill_null(global_mean).to_numpy().astype(np.float32)

    return train_encoded, test_encoded


def train_cv(
    train_df: pl.DataFrame,
    params: dict | None = None,
    n_folds: int = 5,
    feature_cols: list[str] | None = None,
    experiment_name: str = "default",
) -> dict:
    """Train XGBoost with stratified K-fold CV.

    Returns dict with: cv_scores, mean_auc, std_auc, models, feature_importances, feature_names
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)

    X, y, numeric_cols = prepare_data(train_df, feature_cols)

    # Identify target-encode columns present in the data
    te_cols_present = [c for c in TARGET_ENCODE_COLS if c in train_df.columns]
    te_col_names = [f"TE_{c}" for c in te_cols_present]
    all_feature_names = numeric_cols + te_col_names

    log.info(f"Training data: {X.shape[0]:,} samples, {len(all_feature_names)} features "
             f"({X.shape[1]} numeric + {len(te_cols_present)} target-encoded)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    models = []
    importances = np.zeros(len(all_feature_names))
    oof_preds = np.full(X.shape[0], np.nan)

    model_params = {k: v for k, v in params.items()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_base, X_val_base = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Add target-encoded features (computed per fold to avoid leakage)
        if te_cols_present:
            fold_train_df = train_df[train_idx]
            fold_val_df = train_df[val_idx]

            te_train_arrays = []
            te_val_arrays = []
            for col in te_cols_present:
                te_tr, te_va = _target_encode_fold(fold_train_df, fold_val_df, col)
                te_train_arrays.append(te_tr.reshape(-1, 1))
                te_val_arrays.append(te_va.reshape(-1, 1))

            X_train = np.hstack([X_train_base] + te_train_arrays)
            X_val = np.hstack([X_val_base] + te_val_arrays)
        else:
            X_train, X_val = X_train_base, X_val_base

        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_preds)
        cv_scores.append(fold_auc)
        oof_preds[val_idx] = val_preds

        importances += model.feature_importances_ / n_folds
        models.append(model)

        try:
            best_iter = model.best_iteration
        except AttributeError:
            best_iter = model_params.get("n_estimators", "?")
        log.info(f"  Fold {fold + 1}/{n_folds}: AUC = {fold_auc:.5f} "
                 f"(best iter: {best_iter})")

    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    log.info(f"CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")

    # Log experiment
    _log_experiment(experiment_name, params, mean_auc, std_auc, len(all_feature_names), cv_scores)

    return {
        "cv_scores": cv_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "models": models,
        "feature_importances": importances,
        "feature_names": all_feature_names,
        "numeric_cols": numeric_cols,
        "te_cols": te_cols_present,
        "oof_preds": oof_preds,
        "params": params,
    }


def train_lightgbm_cv(
    train_df: pl.DataFrame,
    params: dict | None = None,
    n_folds: int = 5,
    feature_cols: list[str] | None = None,
    experiment_name: str = "lightgbm",
) -> dict:
    """Train LightGBM with stratified K-fold CV for blending."""
    import lightgbm as lgb

    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 34,
            "learning_rate": 0.03,
            "n_estimators": 1500,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_samples": 30,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "scale_pos_weight": 2.5,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)

    X, y, numeric_cols = prepare_data(train_df, feature_cols)

    te_cols_present = [c for c in TARGET_ENCODE_COLS if c in train_df.columns]
    te_col_names = [f"TE_{c}" for c in te_cols_present]
    all_feature_names = numeric_cols + te_col_names

    log.info(f"LightGBM training: {X.shape[0]:,} samples, {len(all_feature_names)} features")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    models = []
    importances = np.zeros(len(all_feature_names))
    oof_preds = np.full(X.shape[0], np.nan)

    # Separate early stopping from model params for LightGBM
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_base, X_val_base = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if te_cols_present:
            fold_train_df = train_df[train_idx]
            fold_val_df = train_df[val_idx]

            te_train_arrays = []
            te_val_arrays = []
            for col in te_cols_present:
                te_tr, te_va = _target_encode_fold(fold_train_df, fold_val_df, col)
                te_train_arrays.append(te_tr.reshape(-1, 1))
                te_val_arrays.append(te_va.reshape(-1, 1))

            X_train = np.hstack([X_train_base] + te_train_arrays)
            X_val = np.hstack([X_val_base] + te_val_arrays)
        else:
            X_train, X_val = X_train_base, X_val_base

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_preds)
        cv_scores.append(fold_auc)
        oof_preds[val_idx] = val_preds

        importances += model.feature_importances_ / n_folds
        models.append(model)

        log.info(f"  Fold {fold + 1}/{n_folds}: AUC = {fold_auc:.5f} "
                 f"(best iter: {model.best_iteration_})")

    # Restore early stopping param
    params["early_stopping_rounds"] = early_stopping_rounds

    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    log.info(f"LightGBM CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")

    _log_experiment(experiment_name, params, mean_auc, std_auc, len(all_feature_names), cv_scores)

    return {
        "cv_scores": cv_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "models": models,
        "feature_importances": importances,
        "feature_names": all_feature_names,
        "numeric_cols": numeric_cols,
        "te_cols": te_cols_present,
        "oof_preds": oof_preds,
        "params": params,
        "model_type": "lightgbm",
    }


def find_blend_weight(oof_xgb: np.ndarray, oof_lgb: np.ndarray, y: np.ndarray) -> float:
    """Find optimal XGBoost weight for blending (XGB weight + LGB weight = 1)."""
    best_auc = 0.0
    best_w = 0.5
    for w in np.arange(0.3, 0.71, 0.01):
        blended = w * oof_xgb + (1 - w) * oof_lgb
        auc = roc_auc_score(y, blended)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    log.info(f"Best blend: XGB={best_w:.2f}, LGB={1-best_w:.2f}, AUC={best_auc:.5f}")
    return best_w


def _log_experiment(
    name: str,
    params: dict,
    mean_auc: float,
    std_auc: float,
    n_features: int,
    cv_scores: list[float],
) -> None:
    """Append experiment to output/experiment_log.csv."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "experiment_log.csv"
    file_exists = log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "name", "mean_auc", "std_auc", "n_features",
                "cv_scores", "params",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            name,
            f"{mean_auc:.6f}",
            f"{std_auc:.6f}",
            n_features,
            json.dumps([round(s, 6) for s in cv_scores]),
            json.dumps({k: v for k, v in params.items() if k != "n_jobs"}),
        ])
    log.info(f"Experiment logged: {name}")


def save_model(result: dict, name: str = "best_model") -> Path:
    """Save models and metadata to output/."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / f"{name}.pkl"
    save_data = {
        "models": result["models"],
        "feature_names": result["feature_names"],
        "mean_auc": result["mean_auc"],
        "params": result["params"],
    }
    # Save target encoding info if present
    if "te_cols" in result:
        save_data["te_cols"] = result["te_cols"]
    if "numeric_cols" in result:
        save_data["numeric_cols"] = result["numeric_cols"]
    if "model_type" in result:
        save_data["model_type"] = result["model_type"]

    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    log.info(f"Model saved to {model_path}")
    return model_path


def load_model(name: str = "best_model") -> dict:
    """Load saved model."""
    model_path = OUTPUT_DIR / f"{name}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def tune_hyperparameters(
    train_df: pl.DataFrame,
    n_trials: int = 30,
    n_folds: int = 5,
    feature_cols: list[str] | None = None,
) -> dict:
    """Lightweight hyperparameter search using Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)

    X, y, numeric_cols = prepare_data(train_df, feature_cols)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": 1500,
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "early_stopping_rounds": 50,
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, val_preds))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"Best trial AUC: {study.best_value:.5f}")
    log.info(f"Best params: {study.best_params}")

    # Build final params
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "n_estimators": 1500,
        "early_stopping_rounds": 50,
        **study.best_params,
    }

    return best_params


if __name__ == "__main__":
    from home_credit.features.pipeline import build_all_features

    log.info("=== Building features ===")
    train_df, test_df = build_all_features()

    log.info("\n=== Training baseline model ===")
    result = train_cv(train_df, experiment_name="baseline")
    save_model(result, "baseline_model")

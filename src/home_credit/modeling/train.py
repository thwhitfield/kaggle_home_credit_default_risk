"""Model training with cross-validation and experiment logging."""

import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
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
    log.info(f"Training data: {X.shape[0]:,} samples, {X.shape[1]} features")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    models = []
    importances = np.zeros(X.shape[1])
    oof_preds = np.full(X.shape[0], np.nan)

    # Separate early_stopping_rounds from params for fit()
    fit_params = {}
    model_params = {k: v for k, v in params.items()}
    if "early_stopping_rounds" in model_params:
        fit_params["early_stopping_rounds"] = model_params.pop("early_stopping_rounds")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            **fit_params,
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        fold_auc = roc_auc_score(y_val, val_preds)
        cv_scores.append(fold_auc)
        oof_preds[val_idx] = val_preds

        importances += model.feature_importances_ / n_folds
        models.append(model)

        log.info(f"  Fold {fold + 1}/{n_folds}: AUC = {fold_auc:.5f} "
                 f"(best iter: {model.best_iteration})")

    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    log.info(f"CV AUC: {mean_auc:.5f} +/- {std_auc:.5f}")

    # Log experiment
    _log_experiment(experiment_name, params, mean_auc, std_auc, len(numeric_cols), cv_scores)

    return {
        "cv_scores": cv_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "models": models,
        "feature_importances": importances,
        "feature_names": numeric_cols,
        "oof_preds": oof_preds,
        "params": params,
    }


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
    with open(model_path, "wb") as f:
        pickle.dump({
            "models": result["models"],
            "feature_names": result["feature_names"],
            "mean_auc": result["mean_auc"],
            "params": result["params"],
        }, f)
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
                early_stopping_rounds=50,
            )
            val_preds = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import roc_auc_score
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

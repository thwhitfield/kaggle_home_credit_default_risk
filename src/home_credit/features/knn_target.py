"""KNN target-encoded features: mean target of K nearest neighbors.

For the top predictive numeric features, compute the mean TARGET value of the
K nearest neighbors using those features as the distance metric.

Leakage prevention:
- Uses out-of-fold (OOF) predictions during training: for each fold, the KNN
  is fit on training-fold data only, and predictions are made for the held-out fold.
- For test data, the KNN is fit on the full training set.
- All features are standardized per-fold to ensure distance metrics are comparable.
"""

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from home_credit.utils import get_logger, timer

log = get_logger(__name__)

# Top predictive numeric features to use as KNN distance features.
# Based on SHAP / feature importance analysis from prior experiments.
KNN_FEATURE_SETS = {
    "ext_sources": [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ],
    "ext_sources_age": [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "DAYS_BIRTH",
    ],
    "financial": [
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
    ],
}

DEFAULT_K = 500
DEFAULT_N_FOLDS = 5


def _knn_oof_predict(
    X_full: np.ndarray,
    y_full: np.ndarray,
    k: int,
    n_folds: int,
) -> np.ndarray:
    """Compute out-of-fold KNN target mean predictions.

    For each fold:
    1. Fit KNN on training fold
    2. For each validation sample, predict = mean target of K nearest training neighbors

    Returns array of shape (n_samples,) with OOF predictions.
    """
    oof_preds = np.full(len(y_full), np.nan, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train = y_full[train_idx]

        # Standardize features using training fold stats
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Replace NaN with 0 after scaling (missing values get neutral distance)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

        # Use KNeighborsClassifier with predict_proba to get mean target of neighbors
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            metric="euclidean",
            n_jobs=-1,
        )
        knn.fit(X_train_scaled, y_train)

        # predict_proba[:, 1] gives the fraction of K neighbors that are positive
        # This is equivalent to the mean target value of the K nearest neighbors
        oof_preds[val_idx] = knn.predict_proba(X_val_scaled)[:, 1].astype(np.float32)

        log.info(f"    Fold {fold + 1}/{n_folds}: "
                 f"mean_pred={oof_preds[val_idx].mean():.4f}")

    return oof_preds


def _knn_full_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
) -> np.ndarray:
    """Fit KNN on full training set and predict for test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform",
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(X_train_scaled, y_train)
    return knn.predict_proba(X_test_scaled)[:, 1].astype(np.float32)


def build_knn_target_features(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    k: int = DEFAULT_K,
    n_folds: int = DEFAULT_N_FOLDS,
    feature_sets: dict[str, list[str]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build KNN target-encoded features for train and test.

    Args:
        train_df: Training DataFrame with TARGET column and feature columns.
        test_df: Test DataFrame (no TARGET column).
        k: Number of nearest neighbors.
        n_folds: Number of CV folds for OOF predictions (train only).
        feature_sets: Dict mapping set name -> list of column names.
            Defaults to KNN_FEATURE_SETS.

    Returns:
        (train_knn_df, test_knn_df): DataFrames with SK_ID_CURR and KNN features.
    """
    if feature_sets is None:
        feature_sets = KNN_FEATURE_SETS

    y_train = train_df["TARGET"].to_numpy().astype(np.float32)
    train_ids = train_df["SK_ID_CURR"]
    test_ids = test_df["SK_ID_CURR"]

    train_knn_cols = {"SK_ID_CURR": train_ids}
    test_knn_cols = {"SK_ID_CURR": test_ids}

    for set_name, feature_cols in feature_sets.items():
        # Check which columns actually exist
        available = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
        if len(available) < 2:
            log.warning(f"  Skipping {set_name}: only {len(available)} features available")
            continue

        col_name = f"KNN_TARGET_{set_name.upper()}"
        log.info(f"  Building {col_name} (k={k}, features={available})")

        X_train = train_df.select(available).to_numpy().astype(np.float32)
        X_test = test_df.select(available).to_numpy().astype(np.float32)

        # OOF predictions for training data
        with timer(f"KNN OOF {set_name}", log):
            train_preds = _knn_oof_predict(X_train, y_train, k=k, n_folds=n_folds)

        # Full-train predictions for test data
        with timer(f"KNN test {set_name}", log):
            test_preds = _knn_full_predict(X_train, y_train, X_test, k=k)

        train_knn_cols[col_name] = pl.Series(col_name, train_preds)
        test_knn_cols[col_name] = pl.Series(col_name, test_preds)

        log.info(f"    Train mean: {train_preds.mean():.4f}, "
                 f"Test mean: {test_preds.mean():.4f}")

    train_knn_df = pl.DataFrame(train_knn_cols)
    test_knn_df = pl.DataFrame(test_knn_cols)

    n_feats = len(train_knn_df.columns) - 1  # minus SK_ID_CURR
    log.info(f"  Total KNN target features: {n_feats}")

    return train_knn_df, test_knn_df

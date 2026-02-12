"""Experiment: evaluate ratio features and KNN target-encoded features.

Runs 4 experiments:
1. Baseline (existing features, no changes)
2. + Ratio features only
3. + KNN target features only
4. + Both ratio + KNN features

Uses the existing tuned params from best_model for fair comparison.
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl

from home_credit.features.knn_target import build_knn_target_features
from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.features.ratios import build_ratio_features
from home_credit.modeling.train import DEFAULT_PARAMS, train_cv, save_model
from home_credit.utils import FEATURES_DIR, OUTPUT_DIR, get_logger, timer

log = get_logger(__name__)

# Use best known params for fair comparison
BEST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}


def load_or_build_knn_features(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load cached KNN features or build them."""
    knn_train_path = FEATURES_DIR / "knn_target_train.parquet"
    knn_test_path = FEATURES_DIR / "knn_target_test.parquet"

    if knn_train_path.exists() and knn_test_path.exists():
        log.info("Loading cached KNN target features")
        return pl.read_parquet(knn_train_path), pl.read_parquet(knn_test_path)

    log.info("Building KNN target features (this may take a few minutes)...")
    with timer("KNN target features", log):
        knn_train, knn_test = build_knn_target_features(train_df, test_df)

    knn_train.write_parquet(knn_train_path)
    knn_test.write_parquet(knn_test_path)
    log.info(f"Saved KNN features to {FEATURES_DIR}")

    return knn_train, knn_test


def main():
    log.info("=== New Features Experiment ===")
    log.info("Loading base features...")
    train_df, test_df = build_all_features()

    base_feature_cols = get_feature_columns(train_df)
    log.info(f"Base features: {len(base_feature_cols)}")

    # ================================================================
    # Experiment 1: Baseline (existing features)
    # ================================================================
    log.info("\n=== Experiment 1: Baseline ===")
    result_baseline = train_cv(
        train_df,
        params=BEST_PARAMS.copy(),
        experiment_name="exp_new_feats_baseline",
    )
    baseline_auc = result_baseline["mean_auc"]
    log.info(f"Baseline CV AUC: {baseline_auc:.5f}")

    # ================================================================
    # Experiment 2: + Ratio features
    # ================================================================
    log.info("\n=== Experiment 2: + Ratio Features ===")
    train_with_ratios = build_ratio_features(train_df)
    test_with_ratios = build_ratio_features(test_df)

    ratio_cols = [c for c in train_with_ratios.columns if c not in train_df.columns]
    log.info(f"New ratio features: {len(ratio_cols)}")
    for c in ratio_cols:
        log.info(f"  {c}")

    result_ratios = train_cv(
        train_with_ratios,
        params=BEST_PARAMS.copy(),
        experiment_name="exp_ratio_features",
    )
    ratios_auc = result_ratios["mean_auc"]
    log.info(f"Ratio features CV AUC: {ratios_auc:.5f} (delta: {ratios_auc - baseline_auc:+.5f})")

    # ================================================================
    # Experiment 3: + KNN target features
    # ================================================================
    log.info("\n=== Experiment 3: + KNN Target Features ===")
    knn_train, knn_test = load_or_build_knn_features(train_df, test_df)

    knn_cols = [c for c in knn_train.columns if c != "SK_ID_CURR"]
    log.info(f"KNN target features: {len(knn_cols)}")
    for c in knn_cols:
        log.info(f"  {c}")

    train_with_knn = train_df.join(knn_train, on="SK_ID_CURR", how="left")
    result_knn = train_cv(
        train_with_knn,
        params=BEST_PARAMS.copy(),
        experiment_name="exp_knn_target_features",
    )
    knn_auc = result_knn["mean_auc"]
    log.info(f"KNN target CV AUC: {knn_auc:.5f} (delta: {knn_auc - baseline_auc:+.5f})")

    # ================================================================
    # Experiment 4: + Both ratio + KNN features
    # ================================================================
    log.info("\n=== Experiment 4: + Ratio + KNN Features ===")
    train_both = build_ratio_features(train_df).join(knn_train, on="SK_ID_CURR", how="left")
    test_both = build_ratio_features(test_df).join(knn_test, on="SK_ID_CURR", how="left")

    all_new_cols = ratio_cols + knn_cols
    log.info(f"Total new features: {len(all_new_cols)}")

    result_both = train_cv(
        train_both,
        params=BEST_PARAMS.copy(),
        experiment_name="exp_ratio_plus_knn",
    )
    both_auc = result_both["mean_auc"]
    log.info(f"Both features CV AUC: {both_auc:.5f} (delta: {both_auc - baseline_auc:+.5f})")

    # ================================================================
    # Summary
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("=== EXPERIMENT SUMMARY ===")
    log.info("=" * 60)
    log.info(f"{'Experiment':<35} {'CV AUC':>10} {'Delta':>10}")
    log.info("-" * 60)
    log.info(f"{'1. Baseline':<35} {baseline_auc:>10.5f} {'---':>10}")
    log.info(f"{'2. + Ratio features':<35} {ratios_auc:>10.5f} {ratios_auc - baseline_auc:>+10.5f}")
    log.info(f"{'3. + KNN target features':<35} {knn_auc:>10.5f} {knn_auc - baseline_auc:>+10.5f}")
    log.info(f"{'4. + Ratio + KNN (both)':<35} {both_auc:>10.5f} {both_auc - baseline_auc:>+10.5f}")
    log.info("=" * 60)

    # Save best model if improved
    best_result = max(
        [("baseline", result_baseline, baseline_auc),
         ("ratios", result_ratios, ratios_auc),
         ("knn", result_knn, knn_auc),
         ("both", result_both, both_auc)],
        key=lambda x: x[2],
    )
    best_name, best_result_dict, best_auc_val = best_result
    log.info(f"\nBest experiment: {best_name} (AUC={best_auc_val:.5f})")

    if best_auc_val > baseline_auc:
        save_model(best_result_dict, f"best_model_new_feats_{best_name}")
        log.info(f"Saved improved model as best_model_new_feats_{best_name}")
    else:
        log.info("No improvement over baseline — no model saved")


if __name__ == "__main__":
    main()

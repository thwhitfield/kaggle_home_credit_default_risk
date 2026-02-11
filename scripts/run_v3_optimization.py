"""V3 optimization: enhanced cross-features + tuned LGB + CatBoost + 3-way blend."""

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.modeling.submit import generate_submission, log_submission
from home_credit.modeling.train import (
    _target_encode_full,
    find_blend_weights_3,
    prepare_data,
    save_model,
    train_catboost_cv,
    train_cv,
    train_lightgbm_cv,
)
from home_credit.utils import OUTPUT_DIR, get_logger

log = get_logger(__name__)

# Best XGBoost params (from V2 tuning)
TUNED_XGB_PARAMS = {
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
    "colsample_bytree": 0.7,
    "min_child_weight": 30,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": 2.5,
    "early_stopping_rounds": 50,
}

# Tuned LightGBM params (differentiated from XGB for diversity)
TUNED_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 48,
    "learning_rate": 0.02,
    "n_estimators": 2000,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "scale_pos_weight": 2.5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "early_stopping_rounds": 80,
    "max_bin": 300,
    "min_gain_to_split": 0.02,
}

# CatBoost params
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "subsample": 0.8,
    "colsample_bylevel": 0.7,
    "min_data_in_leaf": 30,
    "random_seed": 42,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "verbose": 0,
    "early_stopping_rounds": 100,
}


def main():
    # === 1. Build features (force rebuild to pick up new cross-table features) ===
    log.info("=== Step 1: Building enhanced features (V3) ===")
    train_df, test_df = build_all_features(use_cache=False)
    feature_cols = get_feature_columns(train_df)
    log.info(f"Feature set: {len(feature_cols)} features")

    # === 2. Train XGBoost ===
    log.info("\n=== Step 2: Training XGBoost ===")
    xgb_result = train_cv(
        train_df,
        params=TUNED_XGB_PARAMS.copy(),
        experiment_name="v3_xgb",
    )
    save_model(xgb_result, "v3_xgb_model")

    # === 3. Feature pruning based on XGB importances ===
    log.info("\n=== Step 3: Feature pruning ===")
    importances = xgb_result["feature_importances"]
    all_names = xgb_result["feature_names"]

    # Keep non-zero importance
    nonzero_mask = importances > 0
    selected = [f for f, keep in zip(all_names, nonzero_mask) if keep]
    log.info(f"Non-zero importance: {len(selected)}/{len(all_names)}")

    # Drop bottom 5% (less aggressive than V2's 10%)
    threshold = np.percentile(importances[importances > 0], 5)
    pruned = [f for f, imp in zip(all_names, importances) if imp > threshold]
    log.info(f"After pruning bottom 5%: {len(pruned)} features")

    numeric_pruned = [c for c in pruned if not c.startswith("TE_")]
    te_pruned = [c.replace("TE_", "") for c in pruned if c.startswith("TE_")]
    log.info(f"Pruned to {len(numeric_pruned)} numeric + {len(te_pruned)} target-encoded")

    # === 4. Retrain XGBoost with pruned features ===
    log.info("\n=== Step 4: Retrain XGBoost (pruned) ===")
    xgb_pruned = train_cv(
        train_df,
        params=TUNED_XGB_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v3_xgb_pruned",
    )
    save_model(xgb_pruned, "v3_xgb_pruned")

    # === 5. Train LightGBM with tuned params ===
    log.info("\n=== Step 5: Training LightGBM (tuned) ===")
    lgb_result = train_lightgbm_cv(
        train_df,
        params=TUNED_LGB_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v3_lgb_tuned",
    )
    save_model(lgb_result, "v3_lgb_model")

    # === 6. Train CatBoost ===
    log.info("\n=== Step 6: Training CatBoost ===")
    cb_result = train_catboost_cv(
        train_df,
        params=CATBOOST_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v3_catboost",
    )
    save_model(cb_result, "v3_cb_model")

    # === 7. Find optimal 3-way blend ===
    log.info("\n=== Step 7: Finding optimal 3-way blend ===")
    _, y, _ = prepare_data(train_df, get_feature_columns(train_df))
    xgb_oof = xgb_pruned["oof_preds"]
    lgb_oof = lgb_result["oof_preds"]
    cb_oof = cb_result["oof_preds"]

    # Individual OOF scores
    xgb_auc = roc_auc_score(y, xgb_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    cb_auc = roc_auc_score(y, cb_oof)
    log.info(f"XGBoost OOF AUC:  {xgb_auc:.5f}")
    log.info(f"LightGBM OOF AUC: {lgb_auc:.5f}")
    log.info(f"CatBoost OOF AUC: {cb_auc:.5f}")

    w_xgb, w_lgb, w_cb = find_blend_weights_3(xgb_oof, lgb_oof, cb_oof, y)

    blended_oof = w_xgb * xgb_oof + w_lgb * lgb_oof + w_cb * cb_oof
    blend_auc = roc_auc_score(y, blended_oof)
    log.info(f"3-way Blended OOF AUC: {blend_auc:.5f}")

    # === 8. Generate blended submission ===
    log.info("\n=== Step 8: Generating blended submission ===")

    xgb_sub = generate_submission(
        test_df, xgb_pruned,
        save_path=OUTPUT_DIR / "submission_v3_xgb.csv",
        train_df=train_df,
    )
    lgb_sub = generate_submission(
        test_df, lgb_result,
        save_path=OUTPUT_DIR / "submission_v3_lgb.csv",
        train_df=train_df,
    )
    cb_sub = generate_submission(
        test_df, cb_result,
        save_path=OUTPUT_DIR / "submission_v3_cb.csv",
        train_df=train_df,
    )

    blended_preds = (
        w_xgb * xgb_sub["TARGET"].to_numpy()
        + w_lgb * lgb_sub["TARGET"].to_numpy()
        + w_cb * cb_sub["TARGET"].to_numpy()
    )

    submission = pl.DataFrame({
        "SK_ID_CURR": test_df["SK_ID_CURR"],
        "TARGET": blended_preds,
    })
    submission.write_csv(OUTPUT_DIR / "submission.csv")

    n_features = len(xgb_pruned["feature_names"])
    message = (
        f"V3 blend: XGB({w_xgb:.2f})+LGB({w_lgb:.2f})+CB({w_cb:.2f}), "
        f"{n_features} features, CV AUC {blend_auc:.4f}"
    )

    log_submission(message, blend_auc, n_features)

    # === Summary ===
    log.info(f"\n{'='*60}")
    log.info("=== V3 OPTIMIZATION SUMMARY ===")
    log.info(f"{'='*60}")
    log.info(f"XGBoost (all):    CV AUC = {xgb_result['mean_auc']:.5f} ({len(all_names)} features)")
    log.info(f"XGBoost (pruned): CV AUC = {xgb_pruned['mean_auc']:.5f} ({len(pruned)} features)")
    log.info(f"LightGBM (tuned): CV AUC = {lgb_result['mean_auc']:.5f}")
    log.info(f"CatBoost:         CV AUC = {cb_result['mean_auc']:.5f}")
    log.info(f"3-way Blend:      CV AUC = {blend_auc:.5f}")
    log.info(f"V2 best:          CV AUC = 0.79298")
    log.info(f"Improvement:      +{(blend_auc - 0.79298)*10000:.1f} bps")
    log.info(f"\nBlend weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, CB={w_cb:.2f}")
    log.info(f"Submission saved to output/submission.csv")
    log.info(f"Description: {message}")


if __name__ == "__main__":
    main()

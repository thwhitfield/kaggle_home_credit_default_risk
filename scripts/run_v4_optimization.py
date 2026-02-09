"""V4 optimization: experimental features (H2-H5) + 4-way blend."""

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.modeling.submit import generate_submission, log_submission
from home_credit.modeling.train import (
    find_blend_weights_4,
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

# LightGBM GBDT params
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

# LightGBM GOSS params (great for diversity)
LGB_GOSS_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "goss",
    "num_leaves": 48,
    "learning_rate": 0.02,
    "n_estimators": 2000,
    "colsample_bytree": 0.6,
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "scale_pos_weight": 2.5,
    "random_state": 123,
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
    # === 1. Build features (will rebuild with experimental features) ===
    log.info("=== Step 1: Building V4 features (with experimental H2-H5) ===")
    train_df, test_df = build_all_features(use_cache=True)
    feature_cols = get_feature_columns(train_df)
    log.info(f"Feature set: {len(feature_cols)} features")

    # === 2. Train XGBoost (full feature set for importance-based pruning) ===
    log.info("\n=== Step 2: Training XGBoost (full) ===")
    xgb_result = train_cv(
        train_df,
        params=TUNED_XGB_PARAMS.copy(),
        experiment_name="v4_xgb_full",
    )
    save_model(xgb_result, "v4_xgb_full")

    # === 3. Feature pruning based on XGB importances ===
    log.info("\n=== Step 3: Feature pruning ===")
    importances = xgb_result["feature_importances"]
    all_names = xgb_result["feature_names"]

    nonzero_mask = importances > 0
    selected = [f for f, keep in zip(all_names, nonzero_mask) if keep]
    log.info(f"Non-zero importance: {len(selected)}/{len(all_names)}")

    threshold = np.percentile(importances[importances > 0], 5)
    pruned = [f for f, imp in zip(all_names, importances) if imp > threshold]
    log.info(f"After pruning bottom 5%: {len(pruned)} features")

    # Count experimental features that survived
    exp_survived = [f for f in pruned if f.startswith(("H2_", "H3_", "H4_", "H5_"))]
    log.info(f"Experimental features surviving pruning: {len(exp_survived)}")
    for f in exp_survived:
        idx = all_names.index(f)
        log.info(f"  {f}: importance={importances[idx]:.4f}")

    numeric_pruned = [c for c in pruned if not c.startswith("TE_")]
    te_pruned = [c.replace("TE_", "") for c in pruned if c.startswith("TE_")]
    log.info(f"Pruned to {len(numeric_pruned)} numeric + {len(te_pruned)} target-encoded")

    # === 4. Train all 4 models on pruned features ===
    log.info("\n=== Step 4: Retrain XGBoost (pruned) ===")
    xgb_pruned = train_cv(
        train_df,
        params=TUNED_XGB_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v4_xgb_pruned",
    )
    save_model(xgb_pruned, "v4_xgb_pruned")

    log.info("\n=== Step 5: Training LightGBM GBDT ===")
    lgb_result = train_lightgbm_cv(
        train_df,
        params=TUNED_LGB_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v4_lgb_gbdt",
    )
    save_model(lgb_result, "v4_lgb_gbdt")

    log.info("\n=== Step 6: Training LightGBM GOSS ===")
    goss_result = train_lightgbm_cv(
        train_df,
        params=LGB_GOSS_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v4_lgb_goss",
    )
    save_model(goss_result, "v4_lgb_goss")

    log.info("\n=== Step 7: Training CatBoost ===")
    cb_result = train_catboost_cv(
        train_df,
        params=CATBOOST_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v4_catboost",
    )
    save_model(cb_result, "v4_cb_model")

    # === 5. Find optimal 4-way blend ===
    log.info("\n=== Step 8: Finding optimal 4-way blend ===")
    _, y, _ = prepare_data(train_df, get_feature_columns(train_df))

    xgb_oof = xgb_pruned["oof_preds"]
    lgb_oof = lgb_result["oof_preds"]
    goss_oof = goss_result["oof_preds"]
    cb_oof = cb_result["oof_preds"]

    # Individual OOF scores
    xgb_auc = roc_auc_score(y, xgb_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    goss_auc = roc_auc_score(y, goss_oof)
    cb_auc = roc_auc_score(y, cb_oof)
    log.info(f"XGBoost OOF AUC:   {xgb_auc:.5f}")
    log.info(f"LGB GBDT OOF AUC:  {lgb_auc:.5f}")
    log.info(f"LGB GOSS OOF AUC:  {goss_auc:.5f}")
    log.info(f"CatBoost OOF AUC:  {cb_auc:.5f}")

    w_xgb, w_lgb, w_goss, w_cb = find_blend_weights_4(
        xgb_oof, lgb_oof, goss_oof, cb_oof, y,
        labels=("XGB", "LGB", "GOSS", "CB"),
    )

    blended_oof = w_xgb * xgb_oof + w_lgb * lgb_oof + w_goss * goss_oof + w_cb * cb_oof
    blend_auc = roc_auc_score(y, blended_oof)
    log.info(f"4-way Blended OOF AUC: {blend_auc:.5f}")

    # === 6. Generate blended submission ===
    log.info("\n=== Step 9: Generating blended submission ===")

    xgb_sub = generate_submission(
        test_df, xgb_pruned,
        save_path=OUTPUT_DIR / "submission_v4_xgb.csv",
        train_df=train_df,
    )
    lgb_sub = generate_submission(
        test_df, lgb_result,
        save_path=OUTPUT_DIR / "submission_v4_lgb.csv",
        train_df=train_df,
    )
    goss_sub = generate_submission(
        test_df, goss_result,
        save_path=OUTPUT_DIR / "submission_v4_goss.csv",
        train_df=train_df,
    )
    cb_sub = generate_submission(
        test_df, cb_result,
        save_path=OUTPUT_DIR / "submission_v4_cb.csv",
        train_df=train_df,
    )

    blended_preds = (
        w_xgb * xgb_sub["TARGET"].to_numpy()
        + w_lgb * lgb_sub["TARGET"].to_numpy()
        + w_goss * goss_sub["TARGET"].to_numpy()
        + w_cb * cb_sub["TARGET"].to_numpy()
    )

    submission = pl.DataFrame({
        "SK_ID_CURR": test_df["SK_ID_CURR"],
        "TARGET": blended_preds,
    })
    submission.write_csv(OUTPUT_DIR / "submission.csv")

    n_features = len(xgb_pruned["feature_names"])
    message = (
        f"V4 blend: XGB({w_xgb:.2f})+LGB({w_lgb:.2f})+GOSS({w_goss:.2f})+CB({w_cb:.2f}), "
        f"{n_features} features, CV AUC {blend_auc:.4f}"
    )

    log_submission(message, blend_auc, n_features)

    # === Summary ===
    log.info(f"\n{'='*60}")
    log.info("=== V4 OPTIMIZATION SUMMARY ===")
    log.info(f"{'='*60}")
    log.info(f"XGBoost (full):   CV AUC = {xgb_result['mean_auc']:.5f} ({len(all_names)} features)")
    log.info(f"XGBoost (pruned): CV AUC = {xgb_pruned['mean_auc']:.5f} ({len(pruned)} features)")
    log.info(f"LGB GBDT:         CV AUC = {lgb_result['mean_auc']:.5f}")
    log.info(f"LGB GOSS:         CV AUC = {goss_result['mean_auc']:.5f}")
    log.info(f"CatBoost:         CV AUC = {cb_result['mean_auc']:.5f}")
    log.info(f"4-way Blend:      CV AUC = {blend_auc:.5f}")
    log.info(f"V3 best:          CV AUC = 0.79387")
    log.info(f"Improvement:      {(blend_auc - 0.79387)*10000:+.1f} bps")
    log.info(f"\nBlend weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, GOSS={w_goss:.2f}, CB={w_cb:.2f}")
    log.info(f"Experimental features surviving: {len(exp_survived)}")
    log.info(f"Submission saved to output/submission.csv")
    log.info(f"Description: {message}")


if __name__ == "__main__":
    main()

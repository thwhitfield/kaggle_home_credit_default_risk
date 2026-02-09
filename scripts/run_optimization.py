"""Run the full optimization pipeline: new features + XGB + LGB + blending + submit."""

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.modeling.submit import (
    generate_submission,
    log_submission,
    submit_to_kaggle,
    sync_kaggle_scores,
)
from home_credit.modeling.train import (
    _target_encode_full,
    find_blend_weight,
    prepare_data,
    save_model,
    train_cv,
    train_lightgbm_cv,
)
from home_credit.utils import OUTPUT_DIR, get_logger

log = get_logger(__name__)

# Best XGBoost params from previous tuning
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


def main():
    # === 1. Build features (new enhanced set) ===
    log.info("=== Step 1: Building enhanced features ===")
    train_df, test_df = build_all_features(use_cache=True)
    feature_cols = get_feature_columns(train_df)
    log.info(f"Feature set: {len(feature_cols)} features")

    # === 2. Train XGBoost with enhanced features + target encoding ===
    log.info("\n=== Step 2: Training XGBoost (tuned params, new features, target encoding) ===")
    xgb_result = train_cv(
        train_df,
        params=TUNED_XGB_PARAMS.copy(),
        experiment_name="v2_xgb_enhanced",
    )
    save_model(xgb_result, "v2_xgb_model")

    # === 3. Feature selection: prune zero/low importance ===
    log.info("\n=== Step 3: Feature pruning ===")
    importances = xgb_result["feature_importances"]
    all_names = xgb_result["feature_names"]
    nonzero_mask = importances > 0
    selected = [f for f, keep in zip(all_names, nonzero_mask) if keep]
    log.info(f"Non-zero importance: {len(selected)}/{len(all_names)}")

    # Drop bottom 10%
    threshold = np.percentile(importances[importances > 0], 10)
    pruned = [f for f, imp in zip(all_names, importances) if imp > threshold]
    log.info(f"After pruning bottom 10%: {len(pruned)} features")

    # Map back to numeric_cols and te_cols
    numeric_pruned = [c for c in pruned if not c.startswith("TE_")]
    te_pruned = [c.replace("TE_", "") for c in pruned if c.startswith("TE_")]
    log.info(f"Pruned to {len(numeric_pruned)} numeric + {len(te_pruned)} target-encoded")

    # === 4. Retrain XGBoost with pruned features ===
    log.info("\n=== Step 4: Retrain XGBoost with pruned features ===")
    xgb_pruned = train_cv(
        train_df,
        params=TUNED_XGB_PARAMS.copy(),
        feature_cols=numeric_pruned,
        experiment_name="v2_xgb_pruned",
    )
    save_model(xgb_pruned, "v2_xgb_pruned")

    # === 5. Train LightGBM with same features ===
    log.info("\n=== Step 5: Training LightGBM ===")
    lgb_result = train_lightgbm_cv(
        train_df,
        feature_cols=numeric_pruned,
        experiment_name="v2_lgb",
    )
    save_model(lgb_result, "v2_lgb_model")

    # === 6. Find optimal blend weight ===
    log.info("\n=== Step 6: Finding optimal blend weight ===")
    _, y, _ = prepare_data(train_df, get_feature_columns(train_df))
    xgb_oof = xgb_pruned["oof_preds"]
    lgb_oof = lgb_result["oof_preds"]

    # Individual OOF scores
    xgb_oof_auc = roc_auc_score(y, xgb_oof)
    lgb_oof_auc = roc_auc_score(y, lgb_oof)
    log.info(f"XGBoost OOF AUC: {xgb_oof_auc:.5f}")
    log.info(f"LightGBM OOF AUC: {lgb_oof_auc:.5f}")

    blend_weight = find_blend_weight(xgb_oof, lgb_oof, y)

    # Blended OOF score
    blended_oof = blend_weight * xgb_oof + (1 - blend_weight) * lgb_oof
    blend_auc = roc_auc_score(y, blended_oof)
    log.info(f"Blended OOF AUC: {blend_auc:.5f}")

    # === 7. Generate blended submission ===
    log.info("\n=== Step 7: Generating blended submission ===")

    # Generate XGBoost predictions
    xgb_sub = generate_submission(
        test_df, xgb_pruned,
        save_path=OUTPUT_DIR / "submission_xgb.csv",
        train_df=train_df,
    )

    # Generate LightGBM predictions
    lgb_sub = generate_submission(
        test_df, lgb_result,
        save_path=OUTPUT_DIR / "submission_lgb.csv",
        train_df=train_df,
    )

    # Blend
    blended_preds = (
        blend_weight * xgb_sub["TARGET"].to_numpy()
        + (1 - blend_weight) * lgb_sub["TARGET"].to_numpy()
    )

    submission = pl.DataFrame({
        "SK_ID_CURR": test_df["SK_ID_CURR"],
        "TARGET": blended_preds,
    })
    submission.write_csv(OUTPUT_DIR / "submission.csv")

    n_features = len(xgb_pruned["feature_names"])
    message = (
        f"V2 blend: XGB({blend_weight:.2f})+LGB({1-blend_weight:.2f}), "
        f"{n_features} features, CV AUC {blend_auc:.4f}"
    )

    log_submission(message, blend_auc, n_features)

    # === Summary ===
    log.info(f"\n{'='*60}")
    log.info(f"=== OPTIMIZATION SUMMARY ===")
    log.info(f"{'='*60}")
    log.info(f"XGBoost (all features):     CV AUC = {xgb_result['mean_auc']:.5f} ({len(all_names)} features)")
    log.info(f"XGBoost (pruned):           CV AUC = {xgb_pruned['mean_auc']:.5f} ({len(pruned)} features)")
    log.info(f"LightGBM:                   CV AUC = {lgb_result['mean_auc']:.5f} ({len(lgb_result['feature_names'])} features)")
    log.info(f"Blended (XGB+LGB):          CV AUC = {blend_auc:.5f}")
    log.info(f"Previous best:              CV AUC = 0.78714")
    log.info(f"Improvement:                +{(blend_auc - 0.78714)*10000:.1f} bps")
    log.info(f"\nSubmission saved to output/submission.csv")
    log.info(f"Description: {message}")
    log.info(f"\nTo submit: uv run python -m home_credit.modeling.submit -m \"{message}\" --submit")


if __name__ == "__main__":
    main()

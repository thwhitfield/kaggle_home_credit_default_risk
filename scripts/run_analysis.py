"""Run SHAP analysis, threshold analysis, and feature importance plots."""

import numpy as np
import polars as pl

from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.modeling.evaluate import (
    plot_feature_importance,
    plot_roc_curve,
    run_shap_analysis,
    threshold_analysis,
)
from home_credit.modeling.train import load_model, prepare_data
from home_credit.utils import OUTPUT_DIR, get_logger

log = get_logger(__name__)


def main():
    log.info("=== Loading data and model ===")
    train_df, _ = build_all_features()
    model_data = load_model("best_model")

    feature_names = model_data["feature_names"]
    models = model_data["models"]

    # Prepare data
    X, y, _ = prepare_data(train_df, feature_names)
    log.info(f"Data: {X.shape[0]:,} samples, {X.shape[1]} features")

    # Feature importance (gain)
    log.info("\n=== Feature importance by gain ===")
    importances = np.mean([m.feature_importances_ for m in models], axis=0)
    plot_feature_importance(
        importances, feature_names, top_n=30,
        title="Top 30 Features (XGBoost Gain)",
        save_path=OUTPUT_DIR / "feature_importance_gain.png",
    )

    # OOF predictions for threshold analysis and ROC
    log.info("\n=== Generating OOF predictions ===")
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_val = X[val_idx]
        oof_preds[val_idx] = models[fold].predict_proba(X_val)[:, 1]

    # ROC curve
    log.info("\n=== ROC curve ===")
    plot_roc_curve(y, oof_preds)

    # Threshold analysis
    log.info("\n=== Threshold analysis ===")
    threshold_analysis(y, oof_preds)

    # SHAP analysis (use first fold model)
    log.info("\n=== SHAP analysis ===")
    run_shap_analysis(models[0], X, feature_names, max_samples=5000)

    log.info("\n=== Analysis complete! ===")
    log.info(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

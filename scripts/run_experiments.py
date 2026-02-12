"""Run the full experimentation pipeline: feature selection + tuning + retrain."""

import numpy as np

from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.modeling.evaluate import plot_feature_importance
from home_credit.modeling.train import (
    DEFAULT_PARAMS,
    load_model,
    prepare_data,
    save_model,
    train_cv,
    tune_hyperparameters,
)
from home_credit.utils import OUTPUT_DIR, get_logger

log = get_logger(__name__)


def main():
    log.info("=== Loading features ===")
    train_df, test_df = build_all_features()

    # ======= ROUND 1: Baseline already done, analyze feature importance =======
    log.info("\n=== Round 1: Feature importance from baseline ===")
    baseline = load_model("baseline_model")

    # Compute average feature importances from saved fold models
    models = baseline["models"]
    importances = np.mean([m.feature_importances_ for m in models], axis=0)
    feature_names = baseline["feature_names"]

    plot_feature_importance(
        importances,
        feature_names,
        save_path=OUTPUT_DIR / "feature_importance_baseline.png",
    )
    nonzero_mask = importances > 0
    selected_features = [f for f, keep in zip(feature_names, nonzero_mask) if keep]
    log.info(f"Keeping {len(selected_features)}/{len(feature_names)} non-zero importance features")

    # ======= ROUND 2: Train with selected features =======
    log.info("\n=== Round 2: Retrain with selected features ===")
    result_r2 = train_cv(
        train_df,
        params=DEFAULT_PARAMS.copy(),
        feature_cols=selected_features,
        experiment_name="round2_feature_selection",
    )
    save_model(result_r2, "round2_model")

    # Further prune: drop bottom 10% by importance
    importances_r2 = result_r2["feature_importances"]
    names_r2 = result_r2["feature_names"]
    threshold = np.percentile(importances_r2[importances_r2 > 0], 10)
    pruned_features = [f for f, imp in zip(names_r2, importances_r2) if imp > threshold]
    log.info(f"After pruning bottom 10%: {len(pruned_features)} features")

    # ======= ROUND 3: Hyperparameter tuning =======
    log.info("\n=== Round 3: Hyperparameter tuning (Optuna, 30 trials) ===")
    best_params = tune_hyperparameters(
        train_df,
        n_trials=30,
        feature_cols=pruned_features,
    )
    log.info(f"Best params from Optuna: {best_params}")

    # Final train with best params
    log.info("\n=== Final training with tuned params ===")
    result_final = train_cv(
        train_df,
        params=best_params,
        feature_cols=pruned_features,
        experiment_name="round3_tuned",
    )
    save_model(result_final, "best_model")

    plot_feature_importance(
        result_final["feature_importances"],
        result_final["feature_names"],
        save_path=OUTPUT_DIR / "feature_importance_tuned.png",
    )

    log.info(f"\n=== Summary ===")
    log.info(f"Baseline CV AUC:     {baseline['mean_auc']:.5f} ({len(feature_names)} features)")
    log.info(f"Round 2 CV AUC:      {result_r2['mean_auc']:.5f} ({len(selected_features)} features)")
    log.info(f"Final tuned CV AUC:  {result_final['mean_auc']:.5f} ({len(pruned_features)} features)")


if __name__ == "__main__":
    main()

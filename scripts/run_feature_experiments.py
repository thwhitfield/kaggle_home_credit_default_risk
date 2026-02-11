"""Feature experiment screening: build candidate features, evaluate by hypothesis.

Runs 8 quick XGB CV evaluations:
  1 baseline (current features only)
  6 per-hypothesis ablations (baseline + Hx)
  1 all-combined (baseline + all hypotheses)

Reports per-hypothesis AUC delta and feature importances.
"""

import numpy as np
import polars as pl

from home_credit.data.loader import load_table
from home_credit.features.experimental import (
    HYPOTHESIS_PREFIX,
    build_all_experimental_features,
    get_hypothesis_features,
)
from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.modeling.train import train_cv
from home_credit.utils import DATA_DIR, FEATURES_DIR, get_logger, timer

log = get_logger(__name__)

# Fast screening params: 500 trees, higher lr for speed
SCREENING_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 30,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": 2.5,
    "early_stopping_rounds": 30,
}


def main():
    # === 1. Load existing features ===
    log.info("=== Step 1: Loading baseline features ===")
    train_df, test_df = build_all_features(use_cache=True)
    baseline_feature_cols = get_feature_columns(train_df)
    log.info(f"Baseline: {len(baseline_feature_cols)} features")

    # === 2. Build experimental features ===
    log.info("\n=== Step 2: Building experimental features ===")
    with timer("Loading raw tables", log):
        installments = load_table("installments_payments", DATA_DIR)
        cc = load_table("credit_card_balance", DATA_DIR)
        pos = load_table("POS_CASH_balance", DATA_DIR)
        bureau = load_table("bureau", DATA_DIR)
        bureau_balance = load_table("bureau_balance", DATA_DIR)
        prev = load_table("previous_application", DATA_DIR)

    # AMT_ANNUITY in bureau is String due to mixed values
    bureau = bureau.with_columns(
        pl.col("AMT_ANNUITY").cast(pl.String).cast(pl.Float64, strict=False)
    )

    with timer("Building experimental features", log):
        exp_feats = build_all_experimental_features(
            installments, cc, pos, bureau, bureau_balance, prev,
        )

    log.info(f"Experimental features: {exp_feats.shape[1] - 1} new features")

    # Cache experimental features
    exp_path = FEATURES_DIR / "experimental.parquet"
    exp_feats.write_parquet(exp_path)
    log.info(f"Cached experimental features to {exp_path}")

    # === 3. Join experimental features onto train ===
    train_exp = train_df.join(exp_feats, on="SK_ID_CURR", how="left")

    # Get hypothesis -> column mapping
    hyp_features = get_hypothesis_features(train_exp)
    for h, cols in sorted(hyp_features.items()):
        log.info(f"  {h} ({HYPOTHESIS_PREFIX[h]}): {len(cols)} features")

    # === 4. Baseline CV score ===
    log.info("\n=== Step 3: Baseline screening run ===")
    baseline_result = train_cv(
        train_df,
        params=SCREENING_PARAMS.copy(),
        experiment_name="exp_baseline",
    )
    baseline_auc = baseline_result["mean_auc"]
    log.info(f"Baseline screening AUC: {baseline_auc:.5f}")

    # === 5. Per-hypothesis ablation ===
    log.info("\n=== Step 4: Per-hypothesis ablation ===")
    hyp_results = {}

    for hyp_name in sorted(hyp_features.keys()):
        hyp_cols = hyp_features[hyp_name]
        log.info(
            f"\n--- {hyp_name}: {HYPOTHESIS_PREFIX[hyp_name]} "
            f"({len(hyp_cols)} features) ---"
        )

        # baseline numeric features + this hypothesis
        ablation_cols = baseline_feature_cols + hyp_cols

        result = train_cv(
            train_exp,
            params=SCREENING_PARAMS.copy(),
            feature_cols=ablation_cols,
            experiment_name=f"exp_ablation_{hyp_name}",
        )

        delta_bps = (result["mean_auc"] - baseline_auc) * 10000
        log.info(f"{hyp_name} AUC: {result['mean_auc']:.5f} (delta: {delta_bps:+.1f} bps)")
        hyp_results[hyp_name] = {
            "auc": result["mean_auc"],
            "delta_bps": delta_bps,
            "n_features": len(hyp_cols),
            "feature_importances": result["feature_importances"],
            "feature_names": result["feature_names"],
        }

    # === 6. All experimental features combined ===
    log.info("\n=== Step 5: All experimental features combined ===")
    all_exp_cols = [c for cols in hyp_features.values() for c in cols]
    all_cols = baseline_feature_cols + all_exp_cols

    all_result = train_cv(
        train_exp,
        params=SCREENING_PARAMS.copy(),
        feature_cols=all_cols,
        experiment_name="exp_all_combined",
    )
    all_delta = (all_result["mean_auc"] - baseline_auc) * 10000
    log.info(f"All combined AUC: {all_result['mean_auc']:.5f} (delta: {all_delta:+.1f} bps)")

    # === 7. Feature importance analysis by hypothesis ===
    log.info(f"\n{'='*70}")
    log.info("=== FEATURE IMPORTANCE ANALYSIS ===")
    log.info(f"{'='*70}")

    importances = all_result["feature_importances"]
    names = all_result["feature_names"]

    for hyp_name in sorted(HYPOTHESIS_PREFIX.keys()):
        hyp_idx = [i for i, n in enumerate(names) if n.startswith(f"{hyp_name}_")]
        if not hyp_idx:
            continue
        hyp_imps = importances[hyp_idx]
        hyp_names = [names[i] for i in hyp_idx]
        total_imp = hyp_imps.sum()
        nonzero = (hyp_imps > 0).sum()
        log.info(
            f"\n{hyp_name} ({HYPOTHESIS_PREFIX[hyp_name]}): "
            f"total_imp={total_imp:.4f}, non-zero={nonzero}/{len(hyp_idx)}"
        )
        # Top features from this hypothesis
        top_idx = np.argsort(hyp_imps)[::-1][:5]
        for i in top_idx:
            if hyp_imps[i] > 0:
                log.info(f"    {hyp_names[i]}: {hyp_imps[i]:.4f}")

    # === 8. Summary table ===
    log.info(f"\n{'='*70}")
    log.info("=== EXPERIMENT SUMMARY ===")
    log.info(f"{'='*70}")
    log.info(
        f"{'Hypothesis':<8} {'Description':<40} {'AUC':>8} {'Delta':>10} {'# Feats':>8}"
    )
    log.info("-" * 74)
    log.info(
        f"{'Baseline':<8} {'Existing features':<40} "
        f"{baseline_auc:>8.5f} {'':>10} {len(baseline_feature_cols):>8}"
    )

    for hyp_name in sorted(hyp_results.keys()):
        r = hyp_results[hyp_name]
        log.info(
            f"{hyp_name:<8} {HYPOTHESIS_PREFIX[hyp_name]:<40} "
            f"{r['auc']:>8.5f} {r['delta_bps']:>+10.1f} {r['n_features']:>8}"
        )

    log.info("-" * 74)
    log.info(
        f"{'ALL':<8} {'All hypotheses combined':<40} "
        f"{all_result['mean_auc']:>8.5f} {all_delta:>+10.1f} {len(all_exp_cols):>8}"
    )

    # Recommendation
    winners = [h for h, r in sorted(hyp_results.items()) if r["delta_bps"] > 0]
    losers = [h for h, r in sorted(hyp_results.items()) if r["delta_bps"] <= 0]
    if winners:
        log.info(f"\nWINNERS (positive delta): {', '.join(winners)}")
        log.info("RECOMMENDATION: Integrate these into the main pipeline")
    if losers:
        log.info(f"NO SIGNAL: {', '.join(losers)}")


if __name__ == "__main__":
    main()

"""Features from credit_card_balance table."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_credit_card_features(cc: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate credit card balance features to SK_ID_CURR level.

    Credit card behavior — utilization, drawing patterns, payment behavior.
    """
    log.info("Building credit card features")

    # Enrich with utilization ratio before aggregating
    enriched = cc.with_columns(
        # Utilization: balance / credit limit (higher = riskier)
        (pl.col("AMT_BALANCE") / (pl.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
        .alias("CC_UTILIZATION"),
        # Payment to minimum ratio (paying just the minimum = risky)
        (pl.col("AMT_PAYMENT_CURRENT") / (pl.col("AMT_INST_MIN_REGULARITY") + 1))
        .alias("CC_PAYMENT_TO_MIN_RATIO"),
        # Drawing ratio
        (pl.col("AMT_DRAWINGS_CURRENT") / (pl.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
        .alias("CC_DRAWING_RATIO"),
    )

    feats = enriched.group_by("SK_ID_CURR").agg(
        # --- Volume ---
        pl.len().alias("CC_COUNT"),
        pl.col("SK_ID_PREV").n_unique().alias("CC_NUM_CARDS"),
        # --- Utilization ---
        pl.col("CC_UTILIZATION").mean().alias("CC_UTILIZATION_MEAN"),
        pl.col("CC_UTILIZATION").max().alias("CC_UTILIZATION_MAX"),
        pl.col("CC_UTILIZATION").std().alias("CC_UTILIZATION_STD"),
        # High utilization months (>90%)
        (pl.col("CC_UTILIZATION") > 0.9).sum().alias("CC_HIGH_UTIL_MONTHS"),
        # --- Balance ---
        pl.col("AMT_BALANCE").mean().alias("CC_BALANCE_MEAN"),
        pl.col("AMT_BALANCE").max().alias("CC_BALANCE_MAX"),
        # --- Payment behavior ---
        pl.col("CC_PAYMENT_TO_MIN_RATIO").mean().alias("CC_PAYMENT_TO_MIN_MEAN"),
        pl.col("CC_PAYMENT_TO_MIN_RATIO").min().alias("CC_PAYMENT_TO_MIN_MIN"),
        # Months where only minimum was paid (ratio close to 1)
        (pl.col("CC_PAYMENT_TO_MIN_RATIO").is_between(0.95, 1.05)).sum()
        .alias("CC_MIN_PAYMENT_MONTHS"),
        # --- Drawings ---
        pl.col("AMT_DRAWINGS_CURRENT").mean().alias("CC_DRAWINGS_MEAN"),
        pl.col("AMT_DRAWINGS_CURRENT").max().alias("CC_DRAWINGS_MAX"),
        pl.col("AMT_DRAWINGS_CURRENT").sum().alias("CC_DRAWINGS_TOTAL"),
        pl.col("CC_DRAWING_RATIO").mean().alias("CC_DRAWING_RATIO_MEAN"),
        # ATM drawings (cash advances — often a distress signal)
        pl.col("AMT_DRAWINGS_ATM_CURRENT").sum().alias("CC_ATM_DRAWINGS_TOTAL"),
        pl.col("AMT_DRAWINGS_ATM_CURRENT").mean().alias("CC_ATM_DRAWINGS_MEAN"),
        pl.col("CNT_DRAWINGS_ATM_CURRENT").sum().alias("CC_ATM_DRAWINGS_COUNT"),
        # POS drawings
        pl.col("AMT_DRAWINGS_POS_CURRENT").sum().alias("CC_POS_DRAWINGS_TOTAL"),
        # --- DPD ---
        pl.col("SK_DPD").mean().alias("CC_DPD_MEAN"),
        pl.col("SK_DPD").max().alias("CC_DPD_MAX"),
        (pl.col("SK_DPD") > 0).sum().alias("CC_DPD_MONTHS"),
        pl.col("SK_DPD_DEF").max().alias("CC_DPD_DEF_MAX"),
        # --- Credit limit ---
        pl.col("AMT_CREDIT_LIMIT_ACTUAL").max().alias("CC_CREDIT_LIMIT_MAX"),
        pl.col("AMT_CREDIT_LIMIT_ACTUAL").mean().alias("CC_CREDIT_LIMIT_MEAN"),
        # --- Receivable ---
        pl.col("AMT_RECEIVABLE_PRINCIPAL").mean().alias("CC_RECEIVABLE_MEAN"),
        pl.col("AMT_TOTAL_RECEIVABLE").mean().alias("CC_TOTAL_RECEIVABLE_MEAN"),
        # --- Time depth ---
        pl.col("MONTHS_BALANCE").min().alias("CC_MONTHS_BALANCE_MIN"),
        # --- Contract status ---
        (pl.col("NAME_CONTRACT_STATUS") == "Active").sum().alias("CC_ACTIVE_MONTHS"),
    )

    # Derived
    feats = feats.with_columns(
        # DPD rate
        (pl.col("CC_DPD_MONTHS") / (pl.col("CC_COUNT") + 1))
        .alias("CC_DPD_RATE"),
        # ATM drawing share (cash advance as share of total drawings)
        (pl.col("CC_ATM_DRAWINGS_TOTAL") / (pl.col("CC_DRAWINGS_TOTAL") + 1))
        .alias("CC_ATM_DRAWING_SHARE"),
        # High utilization rate
        (pl.col("CC_HIGH_UTIL_MONTHS") / (pl.col("CC_COUNT") + 1))
        .alias("CC_HIGH_UTIL_RATE"),
        # Average months per card
        (pl.col("CC_COUNT") / (pl.col("CC_NUM_CARDS") + 1))
        .alias("CC_AVG_MONTHS_PER_CARD"),
    )

    return feats

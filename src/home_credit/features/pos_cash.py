"""Features from POS_CASH_balance table."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_pos_cash_features(pos: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate POS/cash balance features to SK_ID_CURR level.

    POS_CASH tracks monthly snapshots of point-of-sale and cash loans.
    DPD (days past due) is the key risk signal here.
    """
    log.info("Building POS/cash features")

    feats = pos.group_by("SK_ID_CURR").agg(
        # --- Volume ---
        pl.len().alias("POS_COUNT"),
        pl.col("SK_ID_PREV").n_unique().alias("POS_NUM_LOANS"),
        # --- DPD (days past due) ---
        pl.col("SK_DPD").mean().alias("POS_DPD_MEAN"),
        pl.col("SK_DPD").max().alias("POS_DPD_MAX"),
        pl.col("SK_DPD").sum().alias("POS_DPD_SUM"),
        (pl.col("SK_DPD") > 0).sum().alias("POS_DPD_MONTHS_COUNT"),
        # DPD_DEF (days past due — tolerance threshold)
        pl.col("SK_DPD_DEF").mean().alias("POS_DPD_DEF_MEAN"),
        pl.col("SK_DPD_DEF").max().alias("POS_DPD_DEF_MAX"),
        (pl.col("SK_DPD_DEF") > 0).sum().alias("POS_DPD_DEF_COUNT"),
        # --- Contract status ---
        (pl.col("NAME_CONTRACT_STATUS") == "Active").sum().alias("POS_ACTIVE_MONTHS"),
        (pl.col("NAME_CONTRACT_STATUS") == "Completed").sum().alias("POS_COMPLETED_MONTHS"),
        (pl.col("NAME_CONTRACT_STATUS") == "Signed").sum().alias("POS_SIGNED_MONTHS"),
        # --- Remaining installments ---
        pl.col("CNT_INSTALMENT").max().alias("POS_MAX_INSTALMENTS"),
        pl.col("CNT_INSTALMENT_FUTURE").mean().alias("POS_REMAINING_INSTALMENTS_MEAN"),
        pl.col("CNT_INSTALMENT_FUTURE").max().alias("POS_REMAINING_INSTALMENTS_MAX"),
        # --- Time depth ---
        pl.col("MONTHS_BALANCE").min().alias("POS_MONTHS_BALANCE_MIN"),
    )

    # Derived
    feats = feats.with_columns(
        # DPD rate across all POS months
        (pl.col("POS_DPD_MONTHS_COUNT") / (pl.col("POS_COUNT") + 1))
        .alias("POS_DPD_RATE"),
        # Completion rate
        (pl.col("POS_COMPLETED_MONTHS") / (pl.col("POS_COUNT") + 1))
        .alias("POS_COMPLETION_RATE"),
        # Average months per loan
        (pl.col("POS_COUNT") / (pl.col("POS_NUM_LOANS") + 1))
        .alias("POS_AVG_MONTHS_PER_LOAN"),
    )

    return feats

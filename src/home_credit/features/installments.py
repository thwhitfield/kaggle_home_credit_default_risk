"""Payment behavior features from installments_payments table."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_installment_features(installments: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate installment payment behavior to SK_ID_CURR level.

    Key insight: late payments and underpayments are strong default signals.
    """
    log.info("Building installment features")

    # First compute per-row payment behavior
    enriched = installments.with_columns(
        # Days difference: positive = paid late, negative = paid early
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("INS_DAYS_DIFF"),
        # Payment ratio: how much of the expected amount was actually paid
        (pl.col("AMT_PAYMENT") / (pl.col("AMT_INSTALMENT") + 1)).alias("INS_PAYMENT_RATIO"),
        # Underpayment amount
        (pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT")).alias("INS_UNDERPAYMENT"),
    )

    feats = enriched.group_by("SK_ID_CURR").agg(
        # --- Volume ---
        pl.len().alias("INS_COUNT"),
        pl.col("SK_ID_PREV").n_unique().alias("INS_NUM_PREV_LOANS"),
        # --- Timeliness ---
        pl.col("INS_DAYS_DIFF").mean().alias("INS_DAYS_DIFF_MEAN"),
        pl.col("INS_DAYS_DIFF").max().alias("INS_DAYS_DIFF_MAX"),
        pl.col("INS_DAYS_DIFF").std().alias("INS_DAYS_DIFF_STD"),
        # Late payment counts (paid after due date)
        (pl.col("INS_DAYS_DIFF") > 0).sum().alias("INS_LATE_COUNT"),
        (pl.col("INS_DAYS_DIFF") > 7).sum().alias("INS_LATE_7DAYS_COUNT"),
        (pl.col("INS_DAYS_DIFF") > 30).sum().alias("INS_LATE_30DAYS_COUNT"),
        # Early payments
        (pl.col("INS_DAYS_DIFF") < 0).sum().alias("INS_EARLY_COUNT"),
        (pl.col("INS_DAYS_DIFF") < -15).sum().alias("INS_VERY_EARLY_COUNT"),
        # --- Payment amounts ---
        pl.col("INS_PAYMENT_RATIO").mean().alias("INS_PAYMENT_RATIO_MEAN"),
        pl.col("INS_PAYMENT_RATIO").min().alias("INS_PAYMENT_RATIO_MIN"),
        pl.col("INS_PAYMENT_RATIO").std().alias("INS_PAYMENT_RATIO_STD"),
        # Underpayments (paid less than expected)
        (pl.col("INS_UNDERPAYMENT") > 0).sum().alias("INS_UNDERPAYMENT_COUNT"),
        pl.col("INS_UNDERPAYMENT").filter(pl.col("INS_UNDERPAYMENT") > 0).mean()
        .alias("INS_UNDERPAYMENT_MEAN"),
        pl.col("INS_UNDERPAYMENT").filter(pl.col("INS_UNDERPAYMENT") > 0).max()
        .alias("INS_UNDERPAYMENT_MAX"),
        # Total amounts
        pl.col("AMT_INSTALMENT").sum().alias("INS_TOTAL_EXPECTED"),
        pl.col("AMT_PAYMENT").sum().alias("INS_TOTAL_PAID"),
        # Version (number of installment schedule revisions)
        pl.col("NUM_INSTALMENT_VERSION").max().alias("INS_MAX_VERSION"),
        pl.col("NUM_INSTALMENT_VERSION").n_unique().alias("INS_VERSION_VARIETY"),
    )

    # Derived
    feats = feats.with_columns(
        # Late payment rate
        (pl.col("INS_LATE_COUNT") / (pl.col("INS_COUNT") + 1))
        .alias("INS_LATE_RATE"),
        # Overall payment coverage
        (pl.col("INS_TOTAL_PAID") / (pl.col("INS_TOTAL_EXPECTED") + 1))
        .alias("INS_OVERALL_PAYMENT_RATIO"),
        # Average installments per loan
        (pl.col("INS_COUNT") / (pl.col("INS_NUM_PREV_LOANS") + 1))
        .alias("INS_AVG_PER_LOAN"),
    )

    return feats

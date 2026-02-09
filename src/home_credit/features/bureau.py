"""Aggregated features from bureau and bureau_balance tables."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def _bureau_balance_features(bureau_balance: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate bureau_balance to the bureau record level (SK_ID_BUREAU).

    bureau_balance has monthly statuses: 0, 1, 2, 3, 4, 5 (DPD buckets), C (closed), X (unknown).
    """
    log.info("  Aggregating bureau_balance -> SK_ID_BUREAU level")

    status_cols = ["0", "1", "2", "3", "4", "5", "C", "X"]

    # Count months in each status per bureau record
    status_counts = (
        bureau_balance
        .group_by("SK_ID_BUREAU")
        .agg(
            pl.len().alias("BB_MONTHS_COUNT"),
            # Count each status
            *[
                (pl.col("STATUS") == s).sum().alias(f"BB_STATUS_{s}_COUNT")
                for s in status_cols
            ],
            # DPD months (status 1-5 means past due)
            pl.col("STATUS").is_in(["1", "2", "3", "4", "5"]).sum().alias("BB_DPD_MONTHS"),
            # Months balance range (how far back records go)
            pl.col("MONTHS_BALANCE").min().alias("BB_MONTHS_BALANCE_MIN"),
            pl.col("MONTHS_BALANCE").max().alias("BB_MONTHS_BALANCE_MAX"),
        )
    )

    return status_counts.with_columns(
        # DPD rate across the life of the bureau record
        (pl.col("BB_DPD_MONTHS") / (pl.col("BB_MONTHS_COUNT") + 1))
        .alias("BB_DPD_RATE"),
    )


def build_bureau_features(
    bureau: pl.LazyFrame,
    bureau_balance: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregate bureau + bureau_balance features to the applicant level (SK_ID_CURR)."""
    log.info("Building bureau features")

    # First, get bureau_balance features per bureau record
    bb_feats = _bureau_balance_features(bureau_balance)

    # Join bureau_balance features onto bureau
    bureau_enriched = bureau.join(bb_feats, on="SK_ID_BUREAU", how="left")

    # Aggregate to SK_ID_CURR level
    feats = bureau_enriched.group_by("SK_ID_CURR").agg(
        # --- Count of bureau records ---
        pl.len().alias("BUR_COUNT"),
        # Count by credit type
        (pl.col("CREDIT_TYPE") == "Consumer credit").sum().alias("BUR_CONSUMER_COUNT"),
        (pl.col("CREDIT_TYPE") == "Credit card").sum().alias("BUR_CREDIT_CARD_COUNT"),
        (pl.col("CREDIT_TYPE") == "Mortgage").sum().alias("BUR_MORTGAGE_COUNT"),
        (pl.col("CREDIT_TYPE") == "Car loan").sum().alias("BUR_CAR_LOAN_COUNT"),
        (pl.col("CREDIT_TYPE") == "Microloan").sum().alias("BUR_MICROLOAN_COUNT"),
        # Count by status
        (pl.col("CREDIT_ACTIVE") == "Active").sum().alias("BUR_ACTIVE_COUNT"),
        (pl.col("CREDIT_ACTIVE") == "Closed").sum().alias("BUR_CLOSED_COUNT"),
        (pl.col("CREDIT_ACTIVE") == "Bad debt").sum().alias("BUR_BAD_DEBT_COUNT"),
        # --- Credit amounts ---
        pl.col("AMT_CREDIT_SUM").mean().alias("BUR_AMT_CREDIT_MEAN"),
        pl.col("AMT_CREDIT_SUM").sum().alias("BUR_AMT_CREDIT_TOTAL"),
        pl.col("AMT_CREDIT_SUM").max().alias("BUR_AMT_CREDIT_MAX"),
        # Current debt
        pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("BUR_DEBT_MEAN"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUR_DEBT_TOTAL"),
        pl.col("AMT_CREDIT_SUM_DEBT").max().alias("BUR_DEBT_MAX"),
        # Overdue amounts
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUR_OVERDUE_TOTAL"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").max().alias("BUR_OVERDUE_MAX"),
        (pl.col("AMT_CREDIT_SUM_OVERDUE") > 0).sum().alias("BUR_OVERDUE_COUNT"),
        # Credit limit (for revolving credit)
        pl.col("AMT_CREDIT_SUM_LIMIT").mean().alias("BUR_CREDIT_LIMIT_MEAN"),
        # --- Max overdue ---
        pl.col("AMT_CREDIT_MAX_OVERDUE").max().alias("BUR_MAX_OVERDUE_EVER"),
        pl.col("AMT_CREDIT_MAX_OVERDUE").mean().alias("BUR_MAX_OVERDUE_MEAN"),
        # --- Time features ---
        pl.col("DAYS_CREDIT").max().alias("BUR_MOST_RECENT_CREDIT_DAYS"),
        pl.col("DAYS_CREDIT").min().alias("BUR_OLDEST_CREDIT_DAYS"),
        pl.col("DAYS_CREDIT_ENDDATE").max().alias("BUR_LATEST_ENDDATE"),
        pl.col("DAYS_CREDIT_UPDATE").max().alias("BUR_MOST_RECENT_UPDATE"),
        # Number of credit inquiries in last year (DAYS_CREDIT > -365)
        (pl.col("DAYS_CREDIT") > -365).sum().alias("BUR_INQUIRIES_LAST_YEAR"),
        # --- Bureau balance aggregates ---
        pl.col("BB_DPD_MONTHS").sum().alias("BUR_BB_DPD_MONTHS_TOTAL"),
        pl.col("BB_DPD_RATE").mean().alias("BUR_BB_DPD_RATE_MEAN"),
        pl.col("BB_MONTHS_COUNT").sum().alias("BUR_BB_MONTHS_TOTAL"),
        # Status counts aggregated
        pl.col("BB_STATUS_C_COUNT").sum().alias("BUR_BB_STATUS_C_TOTAL"),
        pl.col("BB_STATUS_X_COUNT").sum().alias("BUR_BB_STATUS_X_TOTAL"),
    )

    # Add derived ratios
    feats = feats.with_columns(
        # Active-to-total ratio
        (pl.col("BUR_ACTIVE_COUNT") / (pl.col("BUR_COUNT") + 1))
        .alias("BUR_ACTIVE_RATIO"),
        # Debt-to-credit ratio
        (pl.col("BUR_DEBT_TOTAL") / (pl.col("BUR_AMT_CREDIT_TOTAL") + 1))
        .alias("BUR_DEBT_TO_CREDIT_RATIO"),
        # Overdue rate
        (pl.col("BUR_OVERDUE_COUNT") / (pl.col("BUR_COUNT") + 1))
        .alias("BUR_OVERDUE_RATE"),
        # Credit history length in days
        (pl.col("BUR_MOST_RECENT_CREDIT_DAYS") - pl.col("BUR_OLDEST_CREDIT_DAYS"))
        .alias("BUR_CREDIT_HISTORY_LENGTH"),
    )

    return feats

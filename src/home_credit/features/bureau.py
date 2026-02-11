"""Aggregated features from bureau and bureau_balance tables."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def _bureau_balance_features(bureau_balance: pl.DataFrame) -> pl.DataFrame:
    """Aggregate bureau_balance to the bureau record level (SK_ID_BUREAU).

    bureau_balance has monthly statuses: 0, 1, 2, 3, 4, 5 (DPD buckets), C (closed), X (unknown).
    """
    log.info("  Aggregating bureau_balance -> SK_ID_BUREAU level")

    status_cols = ["0", "1", "2", "3", "4", "5", "C", "X"]

    # Cast STATUS to String to handle any mixed types
    bureau_balance = bureau_balance.with_columns(pl.col("STATUS").cast(pl.String))

    # Count months in each status per bureau record
    status_counts = (
        bureau_balance
        .group_by("SK_ID_BUREAU")
        .agg(
            pl.len().alias("BB_MONTHS_COUNT"),
            # Count each status
            *[
                (pl.col("STATUS") == s).cast(pl.UInt32).sum().alias(f"BB_STATUS_{s}_COUNT")
                for s in status_cols
            ],
            # DPD months (status 1-5 means past due)
            (
                (pl.col("STATUS") == "1")
                | (pl.col("STATUS") == "2")
                | (pl.col("STATUS") == "3")
                | (pl.col("STATUS") == "4")
                | (pl.col("STATUS") == "5")
            ).cast(pl.UInt32).sum().alias("BB_DPD_MONTHS"),
            # Months balance range (how far back records go)
            pl.col("MONTHS_BALANCE").min().alias("BB_MONTHS_BALANCE_MIN"),
            pl.col("MONTHS_BALANCE").max().alias("BB_MONTHS_BALANCE_MAX"),
            # Recent DPD (last 6 months)
            (
                (pl.col("MONTHS_BALANCE") >= -6)
                & (
                    (pl.col("STATUS") == "1")
                    | (pl.col("STATUS") == "2")
                    | (pl.col("STATUS") == "3")
                    | (pl.col("STATUS") == "4")
                    | (pl.col("STATUS") == "5")
                )
            ).cast(pl.UInt32).sum().alias("BB_RECENT_6M_DPD"),
            # Recent month count (for rate calc)
            (pl.col("MONTHS_BALANCE") >= -6).cast(pl.UInt32).sum().alias("BB_RECENT_6M_COUNT"),
        )
    )

    return status_counts.with_columns(
        # DPD rate across the life of the bureau record
        (pl.col("BB_DPD_MONTHS") / (pl.col("BB_MONTHS_COUNT") + 1))
        .alias("BB_DPD_RATE"),
        # Recent DPD rate
        (pl.col("BB_RECENT_6M_DPD") / (pl.col("BB_RECENT_6M_COUNT") + 1))
        .alias("BB_RECENT_6M_DPD_RATE"),
    )


def build_bureau_features(
    bureau: pl.DataFrame,
    bureau_balance: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate bureau + bureau_balance features to the applicant level (SK_ID_CURR)."""
    log.info("Building bureau features")

    # First, get bureau_balance features per bureau record
    bb_feats = _bureau_balance_features(bureau_balance)

    # AMT_ANNUITY is inferred as String/Categorical from CSV due to mixed values — cast to Float64
    bureau = bureau.with_columns(
        pl.col("AMT_ANNUITY").cast(pl.String).cast(pl.Float64, strict=False)
    )

    # Join bureau_balance features onto bureau
    bureau_enriched = bureau.join(bb_feats, on="SK_ID_BUREAU", how="left")

    # --- Main aggregation: all bureau records ---
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
        # Annuity sum across bureau records
        pl.col("AMT_ANNUITY").sum().alias("BUR_ANNUITY_TOTAL"),
        pl.col("AMT_ANNUITY").mean().alias("BUR_ANNUITY_MEAN"),
        # --- Bureau balance aggregates ---
        pl.col("BB_DPD_MONTHS").sum().alias("BUR_BB_DPD_MONTHS_TOTAL"),
        pl.col("BB_DPD_RATE").mean().alias("BUR_BB_DPD_RATE_MEAN"),
        pl.col("BB_MONTHS_COUNT").sum().alias("BUR_BB_MONTHS_TOTAL"),
        # Status counts aggregated
        pl.col("BB_STATUS_C_COUNT").sum().alias("BUR_BB_STATUS_C_TOTAL"),
        pl.col("BB_STATUS_X_COUNT").sum().alias("BUR_BB_STATUS_X_TOTAL"),
        # Recent bureau balance DPD
        pl.col("BB_RECENT_6M_DPD").sum().alias("BUR_BB_RECENT_6M_DPD_TOTAL"),
        pl.col("BB_RECENT_6M_DPD_RATE").mean().alias("BUR_BB_RECENT_6M_DPD_RATE_MEAN"),
    )

    # --- Active credit features only ---
    active_feats = bureau_enriched.filter(
        pl.col("CREDIT_ACTIVE") == "Active"
    ).group_by("SK_ID_CURR").agg(
        pl.col("AMT_CREDIT_SUM").sum().alias("BUR_ACTIVE_CREDIT_TOTAL"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUR_ACTIVE_DEBT_TOTAL"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUR_ACTIVE_OVERDUE_TOTAL"),
        pl.col("AMT_ANNUITY").sum().alias("BUR_ACTIVE_ANNUITY_TOTAL"),
    )

    # --- Recent bureau records (last 1 year) ---
    recent_1y = bureau_enriched.filter(
        pl.col("DAYS_CREDIT") > -365
    ).group_by("SK_ID_CURR").agg(
        pl.len().alias("BUR_RECENT_1Y_COUNT"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BUR_RECENT_1Y_CREDIT_MEAN"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUR_RECENT_1Y_OVERDUE"),
        (pl.col("CREDIT_ACTIVE") == "Active").sum().alias("BUR_RECENT_1Y_ACTIVE"),
    )

    # --- Recent bureau records (last 2 years) ---
    recent_2y = bureau_enriched.filter(
        pl.col("DAYS_CREDIT") > -730
    ).group_by("SK_ID_CURR").agg(
        pl.len().alias("BUR_RECENT_2Y_COUNT"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUR_RECENT_2Y_DEBT"),
        (pl.col("AMT_CREDIT_SUM_OVERDUE") > 0).sum().alias("BUR_RECENT_2Y_OVERDUE_COUNT"),
    )

    # Join active and recent features
    feats = feats.join(active_feats, on="SK_ID_CURR", how="left")
    feats = feats.join(recent_1y, on="SK_ID_CURR", how="left")
    feats = feats.join(recent_2y, on="SK_ID_CURR", how="left")

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
        # Active debt-to-credit ratio
        (pl.col("BUR_ACTIVE_DEBT_TOTAL").fill_null(0)
         / (pl.col("BUR_ACTIVE_CREDIT_TOTAL").fill_null(0) + 1))
        .alias("BUR_ACTIVE_DEBT_TO_CREDIT_RATIO"),
        # Credit type diversity (number of different credit types)
        (
            (pl.col("BUR_CONSUMER_COUNT") > 0).cast(pl.Int8)
            + (pl.col("BUR_CREDIT_CARD_COUNT") > 0).cast(pl.Int8)
            + (pl.col("BUR_MORTGAGE_COUNT") > 0).cast(pl.Int8)
            + (pl.col("BUR_CAR_LOAN_COUNT") > 0).cast(pl.Int8)
            + (pl.col("BUR_MICROLOAN_COUNT") > 0).cast(pl.Int8)
        ).alias("BUR_CREDIT_TYPE_DIVERSITY"),
    )

    return feats

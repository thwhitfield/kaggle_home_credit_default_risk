"""Aggregated features from bureau and bureau_balance tables."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from home_credit.utils import get_logger

log = get_logger(__name__)


def _bureau_balance_features(bureau_balance: DataFrame) -> DataFrame:
    """Aggregate bureau_balance to the bureau record level (SK_ID_BUREAU).

    bureau_balance has monthly statuses: 0, 1, 2, 3, 4, 5 (DPD buckets), C (closed), X (unknown).
    """
    log.info("  Aggregating bureau_balance -> SK_ID_BUREAU level")

    status_cols = ["0", "1", "2", "3", "4", "5", "C", "X"]

    # Cast STATUS to String to handle any mixed types
    bureau_balance = bureau_balance.select("*", F.col("STATUS").cast("string").alias("STATUS_STR")).drop("STATUS").withColumnRenamed("STATUS_STR", "STATUS")

    # Count months in each status per bureau record
    status_counts = (
        bureau_balance
        .groupBy("SK_ID_BUREAU")
        .agg(
            F.count("*").alias("BB_MONTHS_COUNT"),
            # Count each status
            *[
                F.sum(F.when(F.col("STATUS") == s, 1).otherwise(0)).alias(f"BB_STATUS_{s}_COUNT")
                for s in status_cols
            ],
            # DPD months (status 1-5 means past due)
            F.sum(F.when(
                (F.col("STATUS") == "1")
                | (F.col("STATUS") == "2")
                | (F.col("STATUS") == "3")
                | (F.col("STATUS") == "4")
                | (F.col("STATUS") == "5"),
                1
            ).otherwise(0)).alias("BB_DPD_MONTHS"),
            # Months balance range (how far back records go)
            F.min("MONTHS_BALANCE").alias("BB_MONTHS_BALANCE_MIN"),
            F.max("MONTHS_BALANCE").alias("BB_MONTHS_BALANCE_MAX"),
            # Recent DPD (last 6 months)
            F.sum(F.when(
                (F.col("MONTHS_BALANCE") >= -6)
                & (
                    (F.col("STATUS") == "1")
                    | (F.col("STATUS") == "2")
                    | (F.col("STATUS") == "3")
                    | (F.col("STATUS") == "4")
                    | (F.col("STATUS") == "5")
                ),
                1
            ).otherwise(0)).alias("BB_RECENT_6M_DPD"),
            # Recent month count (for rate calc)
            F.sum(F.when(F.col("MONTHS_BALANCE") >= -6, 1).otherwise(0)).alias("BB_RECENT_6M_COUNT"),
        )
    )

    return status_counts.select(
        "*",
        # DPD rate across the life of the bureau record
        (F.col("BB_DPD_MONTHS") / (F.col("BB_MONTHS_COUNT") + 1))
        .alias("BB_DPD_RATE"),
        # Recent DPD rate
        (F.col("BB_RECENT_6M_DPD") / (F.col("BB_RECENT_6M_COUNT") + 1))
        .alias("BB_RECENT_6M_DPD_RATE"),
    )


def build_bureau_features(
    bureau: DataFrame,
    bureau_balance: DataFrame,
) -> DataFrame:
    """Aggregate bureau + bureau_balance features to the applicant level (SK_ID_CURR)."""
    log.info("Building bureau features")

    # First, get bureau_balance features per bureau record
    bb_feats = _bureau_balance_features(bureau_balance)

    # AMT_ANNUITY is inferred as String/Categorical from CSV due to mixed values — cast to Float64
    bureau = bureau.select(
        *[c for c in bureau.columns if c != "AMT_ANNUITY"],
        F.col("AMT_ANNUITY").cast("string").cast("double").alias("AMT_ANNUITY"),
    )

    # Join bureau_balance features onto bureau
    bureau_enriched = bureau.join(bb_feats, "SK_ID_BUREAU", "left")

    # --- Main aggregation: all bureau records ---
    feats = bureau_enriched.groupBy("SK_ID_CURR").agg(
        # --- Count of bureau records ---
        F.count("*").alias("BUR_COUNT"),
        # Count by credit type
        F.sum(F.when(F.col("CREDIT_TYPE") == "Consumer credit", 1).otherwise(0)).alias("BUR_CONSUMER_COUNT"),
        F.sum(F.when(F.col("CREDIT_TYPE") == "Credit card", 1).otherwise(0)).alias("BUR_CREDIT_CARD_COUNT"),
        F.sum(F.when(F.col("CREDIT_TYPE") == "Mortgage", 1).otherwise(0)).alias("BUR_MORTGAGE_COUNT"),
        F.sum(F.when(F.col("CREDIT_TYPE") == "Car loan", 1).otherwise(0)).alias("BUR_CAR_LOAN_COUNT"),
        F.sum(F.when(F.col("CREDIT_TYPE") == "Microloan", 1).otherwise(0)).alias("BUR_MICROLOAN_COUNT"),
        # Count by status
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)).alias("BUR_ACTIVE_COUNT"),
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Closed", 1).otherwise(0)).alias("BUR_CLOSED_COUNT"),
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Bad debt", 1).otherwise(0)).alias("BUR_BAD_DEBT_COUNT"),
        # --- Credit amounts ---
        F.avg("AMT_CREDIT_SUM").alias("BUR_AMT_CREDIT_MEAN"),
        F.sum("AMT_CREDIT_SUM").alias("BUR_AMT_CREDIT_TOTAL"),
        F.max("AMT_CREDIT_SUM").alias("BUR_AMT_CREDIT_MAX"),
        # Current debt
        F.avg("AMT_CREDIT_SUM_DEBT").alias("BUR_DEBT_MEAN"),
        F.sum("AMT_CREDIT_SUM_DEBT").alias("BUR_DEBT_TOTAL"),
        F.max("AMT_CREDIT_SUM_DEBT").alias("BUR_DEBT_MAX"),
        # Overdue amounts
        F.sum("AMT_CREDIT_SUM_OVERDUE").alias("BUR_OVERDUE_TOTAL"),
        F.max("AMT_CREDIT_SUM_OVERDUE").alias("BUR_OVERDUE_MAX"),
        F.sum(F.when(F.col("AMT_CREDIT_SUM_OVERDUE") > 0, 1).otherwise(0)).alias("BUR_OVERDUE_COUNT"),
        # Credit limit (for revolving credit)
        F.avg("AMT_CREDIT_SUM_LIMIT").alias("BUR_CREDIT_LIMIT_MEAN"),
        # --- Max overdue ---
        F.max("AMT_CREDIT_MAX_OVERDUE").alias("BUR_MAX_OVERDUE_EVER"),
        F.avg("AMT_CREDIT_MAX_OVERDUE").alias("BUR_MAX_OVERDUE_MEAN"),
        # --- Time features ---
        F.max("DAYS_CREDIT").alias("BUR_MOST_RECENT_CREDIT_DAYS"),
        F.min("DAYS_CREDIT").alias("BUR_OLDEST_CREDIT_DAYS"),
        F.max("DAYS_CREDIT_ENDDATE").alias("BUR_LATEST_ENDDATE"),
        F.max("DAYS_CREDIT_UPDATE").alias("BUR_MOST_RECENT_UPDATE"),
        # Number of credit inquiries in last year (DAYS_CREDIT > -365)
        F.sum(F.when(F.col("DAYS_CREDIT") > -365, 1).otherwise(0)).alias("BUR_INQUIRIES_LAST_YEAR"),
        # Annuity sum across bureau records
        F.sum("AMT_ANNUITY").alias("BUR_ANNUITY_TOTAL"),
        F.avg("AMT_ANNUITY").alias("BUR_ANNUITY_MEAN"),
        # --- Bureau balance aggregates ---
        F.sum("BB_DPD_MONTHS").alias("BUR_BB_DPD_MONTHS_TOTAL"),
        F.avg("BB_DPD_RATE").alias("BUR_BB_DPD_RATE_MEAN"),
        F.sum("BB_MONTHS_COUNT").alias("BUR_BB_MONTHS_TOTAL"),
        # Status counts aggregated
        F.sum("BB_STATUS_C_COUNT").alias("BUR_BB_STATUS_C_TOTAL"),
        F.sum("BB_STATUS_X_COUNT").alias("BUR_BB_STATUS_X_TOTAL"),
        # Recent bureau balance DPD
        F.sum("BB_RECENT_6M_DPD").alias("BUR_BB_RECENT_6M_DPD_TOTAL"),
        F.avg("BB_RECENT_6M_DPD_RATE").alias("BUR_BB_RECENT_6M_DPD_RATE_MEAN"),
    )

    # --- Active credit features only ---
    active_feats = bureau_enriched.filter(
        F.col("CREDIT_ACTIVE") == "Active"
    ).groupBy("SK_ID_CURR").agg(
        F.sum("AMT_CREDIT_SUM").alias("BUR_ACTIVE_CREDIT_TOTAL"),
        F.sum("AMT_CREDIT_SUM_DEBT").alias("BUR_ACTIVE_DEBT_TOTAL"),
        F.sum("AMT_CREDIT_SUM_OVERDUE").alias("BUR_ACTIVE_OVERDUE_TOTAL"),
        F.sum("AMT_ANNUITY").alias("BUR_ACTIVE_ANNUITY_TOTAL"),
    )

    # --- Recent bureau records (last 1 year) ---
    recent_1y = bureau_enriched.filter(
        F.col("DAYS_CREDIT") > -365
    ).groupBy("SK_ID_CURR").agg(
        F.count("*").alias("BUR_RECENT_1Y_COUNT"),
        F.avg("AMT_CREDIT_SUM").alias("BUR_RECENT_1Y_CREDIT_MEAN"),
        F.sum("AMT_CREDIT_SUM_OVERDUE").alias("BUR_RECENT_1Y_OVERDUE"),
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)).alias("BUR_RECENT_1Y_ACTIVE"),
    )

    # --- Recent bureau records (last 2 years) ---
    recent_2y = bureau_enriched.filter(
        F.col("DAYS_CREDIT") > -730
    ).groupBy("SK_ID_CURR").agg(
        F.count("*").alias("BUR_RECENT_2Y_COUNT"),
        F.sum("AMT_CREDIT_SUM_DEBT").alias("BUR_RECENT_2Y_DEBT"),
        F.sum(F.when(F.col("AMT_CREDIT_SUM_OVERDUE") > 0, 1).otherwise(0)).alias("BUR_RECENT_2Y_OVERDUE_COUNT"),
    )

    # Join active and recent features
    feats = feats.join(active_feats, "SK_ID_CURR", "left")
    feats = feats.join(recent_1y, "SK_ID_CURR", "left")
    feats = feats.join(recent_2y, "SK_ID_CURR", "left")

    # Add derived ratios
    feats = feats.select(
        "*",
        # Active-to-total ratio
        (F.col("BUR_ACTIVE_COUNT") / (F.col("BUR_COUNT") + 1))
        .alias("BUR_ACTIVE_RATIO"),
        # Debt-to-credit ratio
        (F.col("BUR_DEBT_TOTAL") / (F.col("BUR_AMT_CREDIT_TOTAL") + 1))
        .alias("BUR_DEBT_TO_CREDIT_RATIO"),
        # Overdue rate
        (F.col("BUR_OVERDUE_COUNT") / (F.col("BUR_COUNT") + 1))
        .alias("BUR_OVERDUE_RATE"),
        # Credit history length in days
        (F.col("BUR_MOST_RECENT_CREDIT_DAYS") - F.col("BUR_OLDEST_CREDIT_DAYS"))
        .alias("BUR_CREDIT_HISTORY_LENGTH"),
        # Active debt-to-credit ratio
        (F.coalesce(F.col("BUR_ACTIVE_DEBT_TOTAL"), F.lit(0))
         / (F.coalesce(F.col("BUR_ACTIVE_CREDIT_TOTAL"), F.lit(0)) + 1))
        .alias("BUR_ACTIVE_DEBT_TO_CREDIT_RATIO"),
        # Credit type diversity (number of different credit types)
        (
            F.when(F.col("BUR_CONSUMER_COUNT") > 0, 1).otherwise(0)
            + F.when(F.col("BUR_CREDIT_CARD_COUNT") > 0, 1).otherwise(0)
            + F.when(F.col("BUR_MORTGAGE_COUNT") > 0, 1).otherwise(0)
            + F.when(F.col("BUR_CAR_LOAN_COUNT") > 0, 1).otherwise(0)
            + F.when(F.col("BUR_MICROLOAN_COUNT") > 0, 1).otherwise(0)
        ).alias("BUR_CREDIT_TYPE_DIVERSITY"),
    )

    return feats

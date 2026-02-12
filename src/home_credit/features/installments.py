"""Payment behavior features from installments_payments table."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_installment_features(installments: DataFrame) -> DataFrame:
    """Aggregate installment payment behavior to SK_ID_CURR level.

    Key insight: late payments and underpayments are strong default signals.
    """
    log.info("Building installment features")

    # First compute per-row payment behavior
    enriched = installments.select(
        "*",
        (F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT")).alias("INS_DAYS_DIFF"),
        (F.col("AMT_PAYMENT") / (F.col("AMT_INSTALMENT") + 1)).alias("INS_PAYMENT_RATIO"),
        (F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT")).alias("INS_UNDERPAYMENT"),
    )

    # --- Main aggregation: all installments ---
    feats = enriched.groupBy("SK_ID_CURR").agg(
        # --- Volume ---
        F.count("*").alias("INS_COUNT"),
        F.countDistinct("SK_ID_PREV").alias("INS_NUM_PREV_LOANS"),
        # --- Timeliness ---
        F.avg("INS_DAYS_DIFF").alias("INS_DAYS_DIFF_MEAN"),
        F.max("INS_DAYS_DIFF").alias("INS_DAYS_DIFF_MAX"),
        F.stddev("INS_DAYS_DIFF").alias("INS_DAYS_DIFF_STD"),
        # Late payment counts
        F.sum(F.when(F.col("INS_DAYS_DIFF") > 0, 1).otherwise(0)).alias("INS_LATE_COUNT"),
        F.sum(F.when(F.col("INS_DAYS_DIFF") > 7, 1).otherwise(0)).alias("INS_LATE_7DAYS_COUNT"),
        F.sum(F.when(F.col("INS_DAYS_DIFF") > 30, 1).otherwise(0)).alias("INS_LATE_30DAYS_COUNT"),
        # Early payments
        F.sum(F.when(F.col("INS_DAYS_DIFF") < 0, 1).otherwise(0)).alias("INS_EARLY_COUNT"),
        F.sum(F.when(F.col("INS_DAYS_DIFF") < -15, 1).otherwise(0)).alias("INS_VERY_EARLY_COUNT"),
        # --- Payment amounts ---
        F.avg("INS_PAYMENT_RATIO").alias("INS_PAYMENT_RATIO_MEAN"),
        F.min("INS_PAYMENT_RATIO").alias("INS_PAYMENT_RATIO_MIN"),
        F.stddev("INS_PAYMENT_RATIO").alias("INS_PAYMENT_RATIO_STD"),
        # Underpayments
        F.sum(F.when(F.col("INS_UNDERPAYMENT") > 0, 1).otherwise(0)).alias("INS_UNDERPAYMENT_COUNT"),
        F.avg(F.when(F.col("INS_UNDERPAYMENT") > 0, F.col("INS_UNDERPAYMENT"))).alias("INS_UNDERPAYMENT_MEAN"),
        F.max(F.when(F.col("INS_UNDERPAYMENT") > 0, F.col("INS_UNDERPAYMENT"))).alias("INS_UNDERPAYMENT_MAX"),
        # Total amounts
        F.sum("AMT_INSTALMENT").alias("INS_TOTAL_EXPECTED"),
        F.sum("AMT_PAYMENT").alias("INS_TOTAL_PAID"),
        # Version
        F.max("NUM_INSTALMENT_VERSION").alias("INS_MAX_VERSION"),
        F.countDistinct("NUM_INSTALMENT_VERSION").alias("INS_VERSION_VARIETY"),
    )

    # --- Recent installment behavior (last 12 installments by due date) ---
    w = Window.partitionBy("SK_ID_CURR").orderBy(F.col("DAYS_INSTALMENT").desc())
    ranked = enriched.withColumn("_rank", F.row_number().over(w))

    recent_12 = ranked.filter(F.col("_rank") <= 12)
    recent_feats = recent_12.groupBy("SK_ID_CURR").agg(
        F.avg("INS_DAYS_DIFF").alias("INS_RECENT_12_DAYS_DIFF_MEAN"),
        F.stddev("INS_DAYS_DIFF").alias("INS_RECENT_12_DAYS_DIFF_STD"),
        F.sum(F.when(F.col("INS_DAYS_DIFF") > 0, 1).otherwise(0)).alias("INS_RECENT_12_LATE_COUNT"),
        F.sum(F.when(F.col("INS_DAYS_DIFF") > 30, 1).otherwise(0)).alias("INS_RECENT_12_LATE_30_COUNT"),
        F.avg("INS_PAYMENT_RATIO").alias("INS_RECENT_12_PAYMENT_RATIO"),
        F.min("INS_PAYMENT_RATIO").alias("INS_RECENT_12_PAYMENT_RATIO_MIN"),
    )

    # Last installment details (rank == 1)
    last_inst = ranked.filter(F.col("_rank") == 1).select(
        F.col("SK_ID_CURR"),
        F.col("INS_DAYS_DIFF").alias("INS_LAST_DAYS_DIFF"),
        F.col("INS_PAYMENT_RATIO").alias("INS_LAST_PAYMENT_RATIO"),
    )

    feats = feats.join(recent_feats, "SK_ID_CURR", "left")
    feats = feats.join(last_inst, "SK_ID_CURR", "left")

    # Derived
    feats = feats.select(
        "*",
        (F.col("INS_LATE_COUNT") / (F.col("INS_COUNT") + 1)).alias("INS_LATE_RATE"),
        (F.col("INS_TOTAL_PAID") / (F.col("INS_TOTAL_EXPECTED") + 1)).alias("INS_OVERALL_PAYMENT_RATIO"),
        (F.col("INS_COUNT") / (F.col("INS_NUM_PREV_LOANS") + 1)).alias("INS_AVG_PER_LOAN"),
        (F.col("INS_RECENT_12_LATE_COUNT") / 12.0).alias("INS_RECENT_12_LATE_RATE"),
    )

    return feats

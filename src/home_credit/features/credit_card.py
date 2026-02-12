"""Features from credit_card_balance table."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_credit_card_features(cc: DataFrame) -> DataFrame:
    """Aggregate credit card balance features to SK_ID_CURR level.

    Credit card behavior — utilization, drawing patterns, payment behavior.
    """
    log.info("Building credit card features")

    # Enrich with utilization ratio before aggregating
    enriched = cc.select(
        "*",
        # Utilization: balance / credit limit (higher = riskier)
        (F.col("AMT_BALANCE") / (F.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
        .alias("CC_UTILIZATION"),
        # Payment to minimum ratio (paying just the minimum = risky)
        (F.col("AMT_PAYMENT_CURRENT") / (F.col("AMT_INST_MIN_REGULARITY") + 1))
        .alias("CC_PAYMENT_TO_MIN_RATIO"),
        # Drawing ratio
        (F.col("AMT_DRAWINGS_CURRENT") / (F.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
        .alias("CC_DRAWING_RATIO"),
    )

    # --- Main aggregation: all months ---
    feats = enriched.groupBy("SK_ID_CURR").agg(
        # --- Volume ---
        F.count("*").alias("CC_COUNT"),
        F.countDistinct("SK_ID_PREV").alias("CC_NUM_CARDS"),
        # --- Utilization ---
        F.avg("CC_UTILIZATION").alias("CC_UTILIZATION_MEAN"),
        F.max("CC_UTILIZATION").alias("CC_UTILIZATION_MAX"),
        F.stddev("CC_UTILIZATION").alias("CC_UTILIZATION_STD"),
        # High utilization months (>90%)
        F.sum(F.when(F.col("CC_UTILIZATION") > 0.9, 1).otherwise(0)).alias("CC_HIGH_UTIL_MONTHS"),
        # --- Balance ---
        F.avg("AMT_BALANCE").alias("CC_BALANCE_MEAN"),
        F.max("AMT_BALANCE").alias("CC_BALANCE_MAX"),
        # --- Payment behavior ---
        F.avg("CC_PAYMENT_TO_MIN_RATIO").alias("CC_PAYMENT_TO_MIN_MEAN"),
        F.min("CC_PAYMENT_TO_MIN_RATIO").alias("CC_PAYMENT_TO_MIN_MIN"),
        # Months where only minimum was paid (ratio close to 1)
        F.sum(F.when(F.col("CC_PAYMENT_TO_MIN_RATIO").between(0.95, 1.05), 1).otherwise(0))
        .alias("CC_MIN_PAYMENT_MONTHS"),
        # --- Drawings ---
        F.avg("AMT_DRAWINGS_CURRENT").alias("CC_DRAWINGS_MEAN"),
        F.max("AMT_DRAWINGS_CURRENT").alias("CC_DRAWINGS_MAX"),
        F.sum("AMT_DRAWINGS_CURRENT").alias("CC_DRAWINGS_TOTAL"),
        F.avg("CC_DRAWING_RATIO").alias("CC_DRAWING_RATIO_MEAN"),
        # ATM drawings (cash advances — often a distress signal)
        F.sum("AMT_DRAWINGS_ATM_CURRENT").alias("CC_ATM_DRAWINGS_TOTAL"),
        F.avg("AMT_DRAWINGS_ATM_CURRENT").alias("CC_ATM_DRAWINGS_MEAN"),
        F.sum("CNT_DRAWINGS_ATM_CURRENT").alias("CC_ATM_DRAWINGS_COUNT"),
        # POS drawings
        F.sum("AMT_DRAWINGS_POS_CURRENT").alias("CC_POS_DRAWINGS_TOTAL"),
        # --- DPD ---
        F.avg("SK_DPD").alias("CC_DPD_MEAN"),
        F.max("SK_DPD").alias("CC_DPD_MAX"),
        F.sum(F.when(F.col("SK_DPD") > 0, 1).otherwise(0)).alias("CC_DPD_MONTHS"),
        F.max("SK_DPD_DEF").alias("CC_DPD_DEF_MAX"),
        # --- Credit limit ---
        F.max("AMT_CREDIT_LIMIT_ACTUAL").alias("CC_CREDIT_LIMIT_MAX"),
        F.avg("AMT_CREDIT_LIMIT_ACTUAL").alias("CC_CREDIT_LIMIT_MEAN"),
        # --- Receivable ---
        F.avg("AMT_RECEIVABLE_PRINCIPAL").alias("CC_RECEIVABLE_MEAN"),
        F.avg("AMT_TOTAL_RECEIVABLE").alias("CC_TOTAL_RECEIVABLE_MEAN"),
        # --- Time depth ---
        F.min("MONTHS_BALANCE").alias("CC_MONTHS_BALANCE_MIN"),
        # --- Contract status ---
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Active", 1).otherwise(0)).alias("CC_ACTIVE_MONTHS"),
    )

    # --- Recent credit card behavior (last 3 months) ---
    recent_3m = enriched.filter(F.col("MONTHS_BALANCE") >= -3).groupBy("SK_ID_CURR").agg(
        F.avg("CC_UTILIZATION").alias("CC_RECENT_3M_UTIL_MEAN"),
        F.avg("AMT_BALANCE").alias("CC_RECENT_3M_BALANCE_MEAN"),
        F.max("SK_DPD").alias("CC_RECENT_3M_DPD_MAX"),
        F.avg("CC_PAYMENT_TO_MIN_RATIO").alias("CC_RECENT_3M_PAY_MIN_RATIO"),
    )

    # --- Recent credit card behavior (last 12 months) ---
    recent_12m = enriched.filter(F.col("MONTHS_BALANCE") >= -12).groupBy("SK_ID_CURR").agg(
        F.avg("CC_UTILIZATION").alias("CC_RECENT_12M_UTIL_MEAN"),
        F.sum(F.when(F.col("CC_UTILIZATION") > 0.9, 1).otherwise(0)).alias("CC_RECENT_12M_HIGH_UTIL_MONTHS"),
        F.sum(F.when(F.col("SK_DPD") > 0, 1).otherwise(0)).alias("CC_RECENT_12M_DPD_COUNT"),
        F.sum("AMT_DRAWINGS_ATM_CURRENT").alias("CC_RECENT_12M_ATM_TOTAL"),
    )

    feats = feats.join(recent_3m, "SK_ID_CURR", "left")
    feats = feats.join(recent_12m, "SK_ID_CURR", "left")

    # Derived
    feats = feats.select(
        "*",
        # DPD rate
        (F.col("CC_DPD_MONTHS") / (F.col("CC_COUNT") + 1))
        .alias("CC_DPD_RATE"),
        # ATM drawing share (cash advance as share of total drawings)
        (F.col("CC_ATM_DRAWINGS_TOTAL") / (F.col("CC_DRAWINGS_TOTAL") + 1))
        .alias("CC_ATM_DRAWING_SHARE"),
        # High utilization rate
        (F.col("CC_HIGH_UTIL_MONTHS") / (F.col("CC_COUNT") + 1))
        .alias("CC_HIGH_UTIL_RATE"),
        # Average months per card
        (F.col("CC_COUNT") / (F.col("CC_NUM_CARDS") + 1))
        .alias("CC_AVG_MONTHS_PER_CARD"),
        # Utilization trend: recent vs overall (positive = worsening)
        (F.coalesce(F.col("CC_RECENT_3M_UTIL_MEAN"), F.lit(0)) - F.coalesce(F.col("CC_UTILIZATION_MEAN"), F.lit(0)))
        .alias("CC_UTIL_TREND_3M"),
        (F.coalesce(F.col("CC_RECENT_12M_UTIL_MEAN"), F.lit(0)) - F.coalesce(F.col("CC_UTILIZATION_MEAN"), F.lit(0)))
        .alias("CC_UTIL_TREND_12M"),
    )

    return feats

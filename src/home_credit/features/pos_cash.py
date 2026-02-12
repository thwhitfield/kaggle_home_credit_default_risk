"""Features from POS_CASH_balance table."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_pos_cash_features(pos: DataFrame) -> DataFrame:
    """Aggregate POS/cash balance features to SK_ID_CURR level.

    POS_CASH tracks monthly snapshots of point-of-sale and cash loans.
    DPD (days past due) is the key risk signal here.
    """
    log.info("Building POS/cash features")

    # --- Main aggregation ---
    feats = pos.groupBy("SK_ID_CURR").agg(
        # --- Volume ---
        F.count("*").alias("POS_COUNT"),
        F.countDistinct("SK_ID_PREV").alias("POS_NUM_LOANS"),
        # --- DPD ---
        F.avg("SK_DPD").alias("POS_DPD_MEAN"),
        F.max("SK_DPD").alias("POS_DPD_MAX"),
        F.sum("SK_DPD").alias("POS_DPD_SUM"),
        F.sum(F.when(F.col("SK_DPD") > 0, 1).otherwise(0)).alias("POS_DPD_MONTHS_COUNT"),
        # DPD_DEF
        F.avg("SK_DPD_DEF").alias("POS_DPD_DEF_MEAN"),
        F.max("SK_DPD_DEF").alias("POS_DPD_DEF_MAX"),
        F.sum(F.when(F.col("SK_DPD_DEF") > 0, 1).otherwise(0)).alias("POS_DPD_DEF_COUNT"),
        # --- Contract status ---
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Active", 1).otherwise(0)).alias("POS_ACTIVE_MONTHS"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Completed", 1).otherwise(0)).alias("POS_COMPLETED_MONTHS"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Signed", 1).otherwise(0)).alias("POS_SIGNED_MONTHS"),
        # --- Remaining installments ---
        F.max("CNT_INSTALMENT").alias("POS_MAX_INSTALMENTS"),
        F.avg("CNT_INSTALMENT_FUTURE").alias("POS_REMAINING_INSTALMENTS_MEAN"),
        F.max("CNT_INSTALMENT_FUTURE").alias("POS_REMAINING_INSTALMENTS_MAX"),
        # --- Time depth ---
        F.min("MONTHS_BALANCE").alias("POS_MONTHS_BALANCE_MIN"),
    )

    # --- Recent POS behavior (last 6 months) ---
    recent_6m = pos.filter(F.col("MONTHS_BALANCE") >= -6).groupBy("SK_ID_CURR").agg(
        F.max("SK_DPD").alias("POS_RECENT_6M_DPD_MAX"),
        F.avg("SK_DPD").alias("POS_RECENT_6M_DPD_MEAN"),
        F.sum(F.when(F.col("SK_DPD") > 0, 1).otherwise(0)).alias("POS_RECENT_6M_DPD_COUNT"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Active", 1).otherwise(0)).alias("POS_RECENT_6M_ACTIVE"),
    )

    # --- Recent POS behavior (last 12 months) ---
    recent_12m = pos.filter(F.col("MONTHS_BALANCE") >= -12).groupBy("SK_ID_CURR").agg(
        F.max("SK_DPD").alias("POS_RECENT_12M_DPD_MAX"),
        F.sum(F.when(F.col("SK_DPD") > 0, 1).otherwise(0)).alias("POS_RECENT_12M_DPD_COUNT"),
        F.sum(F.when(F.col("SK_DPD_DEF") > 0, 1).otherwise(0)).alias("POS_RECENT_12M_DPD_DEF_COUNT"),
    )

    feats = feats.join(recent_6m, "SK_ID_CURR", "left")
    feats = feats.join(recent_12m, "SK_ID_CURR", "left")

    # Derived
    feats = feats.select(
        "*",
        (F.col("POS_DPD_MONTHS_COUNT") / (F.col("POS_COUNT") + 1)).alias("POS_DPD_RATE"),
        (F.col("POS_COMPLETED_MONTHS") / (F.col("POS_COUNT") + 1)).alias("POS_COMPLETION_RATE"),
        (F.col("POS_COUNT") / (F.col("POS_NUM_LOANS") + 1)).alias("POS_AVG_MONTHS_PER_LOAN"),
        (F.coalesce(F.col("POS_RECENT_6M_DPD_COUNT"), F.lit(0)) / 6.0).alias("POS_RECENT_6M_DPD_RATE"),
    )

    return feats

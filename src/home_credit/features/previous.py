"""Aggregated features from previous_application table."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_previous_application_features(prev: DataFrame) -> DataFrame:
    """Aggregate previous Home Credit applications to SK_ID_CURR level."""
    log.info("Building previous application features")

    # --- Main aggregation: all previous applications ---
    feats = prev.groupBy("SK_ID_CURR").agg(
        # --- Counts by status ---
        F.count("*").alias("PREV_COUNT"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Approved", 1).otherwise(0)).alias("PREV_APPROVED_COUNT"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Refused", 1).otherwise(0)).alias("PREV_REFUSED_COUNT"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Canceled", 1).otherwise(0)).alias("PREV_CANCELED_COUNT"),
        # --- Loan amounts ---
        F.avg("AMT_APPLICATION").alias("PREV_AMT_APPLICATION_MEAN"),
        F.max("AMT_APPLICATION").alias("PREV_AMT_APPLICATION_MAX"),
        F.sum("AMT_APPLICATION").alias("PREV_AMT_APPLICATION_TOTAL"),
        F.avg("AMT_CREDIT").alias("PREV_AMT_CREDIT_MEAN"),
        F.sum("AMT_CREDIT").alias("PREV_AMT_CREDIT_TOTAL"),
        # Application-credit difference
        F.avg(F.col("AMT_APPLICATION") - F.col("AMT_CREDIT")).alias("PREV_APP_CREDIT_DIFF_MEAN"),
        # Annuity
        F.avg("AMT_ANNUITY").alias("PREV_ANNUITY_MEAN"),
        F.max("AMT_ANNUITY").alias("PREV_ANNUITY_MAX"),
        # Down payment
        F.avg("AMT_DOWN_PAYMENT").alias("PREV_DOWN_PAYMENT_MEAN"),
        F.max("AMT_DOWN_PAYMENT").alias("PREV_DOWN_PAYMENT_MAX"),
        # --- Time features ---
        F.max("DAYS_DECISION").alias("PREV_MOST_RECENT_DECISION"),
        F.min("DAYS_DECISION").alias("PREV_OLDEST_DECISION"),
        F.avg("DAYS_FIRST_DRAWING").alias("PREV_DAYS_FIRST_DRAWING_MEAN"),
        F.avg("DAYS_FIRST_DUE").alias("PREV_DAYS_FIRST_DUE_MEAN"),
        F.avg("DAYS_LAST_DUE_1ST_VERSION").alias("PREV_DAYS_LAST_DUE_MEAN"),
        F.avg("DAYS_TERMINATION").alias("PREV_DAYS_TERMINATION_MEAN"),
        # --- Contract types ---
        F.sum(F.when(F.col("NAME_CONTRACT_TYPE") == "Cash loans", 1).otherwise(0)).alias("PREV_CASH_LOAN_COUNT"),
        F.sum(F.when(F.col("NAME_CONTRACT_TYPE") == "Revolving loans", 1).otherwise(0)).alias("PREV_REVOLVING_COUNT"),
        # --- Product complexity ---
        F.countDistinct("PRODUCT_COMBINATION").alias("PREV_PRODUCT_VARIETY"),
        # --- Yield group ---
        F.sum(F.when(F.col("NAME_YIELD_GROUP") == "high", 1).otherwise(0)).alias("PREV_HIGH_YIELD_COUNT"),
        F.sum(F.when(F.col("NAME_YIELD_GROUP") == "low_normal", 1).otherwise(0)).alias("PREV_LOW_YIELD_COUNT"),
        # --- Channel ---
        F.countDistinct("CHANNEL_TYPE").alias("PREV_CHANNEL_VARIETY"),
        # --- Weekday patterns ---
        F.sum(F.when(F.col("WEEKDAY_APPR_PROCESS_START") == "SATURDAY", 1).otherwise(0)).alias("PREV_WEEKEND_APP_COUNT"),
        F.sum(F.when(F.col("WEEKDAY_APPR_PROCESS_START") == "SUNDAY", 1).otherwise(0)).alias("PREV_SUNDAY_APP_COUNT"),
        # Insurance
        F.sum(F.when(F.col("NFLAG_INSURED_ON_APPROVAL") == 1, 1).otherwise(0)).alias("PREV_INSURED_COUNT"),
        # Payment term
        F.avg("CNT_PAYMENT").alias("PREV_PAYMENT_TERM_MEAN"),
        F.max("CNT_PAYMENT").alias("PREV_PAYMENT_TERM_MAX"),
    )

    # --- Approved applications only ---
    approved_feats = prev.filter(
        F.col("NAME_CONTRACT_STATUS") == "Approved"
    ).groupBy("SK_ID_CURR").agg(
        F.avg("AMT_CREDIT").alias("PREV_APPROVED_CREDIT_MEAN"),
        F.max("AMT_CREDIT").alias("PREV_APPROVED_CREDIT_MAX"),
        F.avg("AMT_ANNUITY").alias("PREV_APPROVED_ANNUITY_MEAN"),
        F.max("DAYS_DECISION").alias("PREV_LAST_APPROVED_DAYS"),
        F.avg("AMT_DOWN_PAYMENT").alias("PREV_APPROVED_DOWN_PAYMENT_MEAN"),
    )

    # --- Refused applications ---
    refused_feats = prev.filter(
        F.col("NAME_CONTRACT_STATUS") == "Refused"
    ).groupBy("SK_ID_CURR").agg(
        F.avg("AMT_APPLICATION").alias("PREV_REFUSED_AMT_MEAN"),
        F.max("DAYS_DECISION").alias("PREV_LAST_REFUSED_DAYS"),
    )

    # --- Recent applications (last 1 year) ---
    recent_1y = prev.filter(
        F.col("DAYS_DECISION") > -365
    ).groupBy("SK_ID_CURR").agg(
        F.count("*").alias("PREV_RECENT_1Y_COUNT"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Approved", 1).otherwise(0)).alias("PREV_RECENT_1Y_APPROVED"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Refused", 1).otherwise(0)).alias("PREV_RECENT_1Y_REFUSED"),
    )

    feats = feats.join(approved_feats, "SK_ID_CURR", "left")
    feats = feats.join(refused_feats, "SK_ID_CURR", "left")
    feats = feats.join(recent_1y, "SK_ID_CURR", "left")

    # Derived ratios
    feats = feats.select(
        "*",
        (F.col("PREV_APPROVED_COUNT") / (F.col("PREV_COUNT") + 1)).alias("PREV_APPROVAL_RATE"),
        (F.col("PREV_REFUSED_COUNT") / (F.col("PREV_COUNT") + 1)).alias("PREV_REFUSAL_RATE"),
        (F.col("PREV_CASH_LOAN_COUNT") / (F.col("PREV_COUNT") + 1)).alias("PREV_CASH_LOAN_RATIO"),
        (F.col("PREV_MOST_RECENT_DECISION") - F.col("PREV_OLDEST_DECISION")).alias("PREV_HISTORY_SPAN"),
        (F.coalesce(F.col("PREV_RECENT_1Y_APPROVED"), F.lit(0))
         / (F.coalesce(F.col("PREV_RECENT_1Y_COUNT"), F.lit(0)) + 1)).alias("PREV_RECENT_1Y_APPROVAL_RATE"),
    )

    return feats

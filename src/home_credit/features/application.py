"""Feature engineering from the application_train / application_test tables."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from home_credit.utils import get_logger

log = get_logger(__name__)


def _fill(col_name, default=0):
    """Shorthand for coalesce(col, lit(default))."""
    return F.coalesce(F.col(col_name), F.lit(default))


def _is_null_int(col_name):
    """Return 1 if column is null, else 0."""
    return F.when(F.col(col_name).isNull(), 1).otherwise(0)


def _is_not_null_int(col_name):
    """Return 1 if column is not null, else 0."""
    return F.when(F.col(col_name).isNotNull(), 1).otherwise(0)


def _replace_zero_with_null(col_name):
    """Replace 0 with null."""
    return F.when(F.col(col_name) == 0, None).otherwise(F.col(col_name))


def build_application_features(app: DataFrame) -> DataFrame:
    """Create domain-informed features from the application table.

    These stay at the applicant level (SK_ID_CURR) — no aggregation needed.
    """
    log.info("Building application features")

    # Document flag columns
    doc_cols = [
        "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
    ]

    # Building info columns for null count
    building_cols = [
        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG",
        "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
        "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG",
        "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
    ]

    # Clipped family members (min 1)
    fam = F.greatest(F.coalesce(F.col("CNT_FAM_MEMBERS"), F.lit(1)), F.lit(1))

    ext1 = _fill("EXT_SOURCE_1")
    ext2 = _fill("EXT_SOURCE_2")
    ext3 = _fill("EXT_SOURCE_3")
    ext_mean = (ext1 + ext2 + ext3) / F.lit(3)

    days_birth_safe = _replace_zero_with_null("DAYS_BIRTH")

    result = app.select(
        "*",
        # --- Income / credit ratios ---
        (F.col("AMT_CREDIT") / (F.col("AMT_INCOME_TOTAL") + 1)).alias("APP_CREDIT_TO_INCOME_RATIO"),
        (F.col("AMT_ANNUITY") / (F.col("AMT_INCOME_TOTAL") + 1)).alias("APP_ANNUITY_TO_INCOME_RATIO"),
        (F.col("AMT_CREDIT") / (F.coalesce(F.col("AMT_GOODS_PRICE"), F.lit(1)) + 1)).alias("APP_CREDIT_TO_GOODS_RATIO"),
        (F.col("AMT_ANNUITY") / (F.col("AMT_CREDIT") + 1)).alias("APP_ANNUITY_TO_CREDIT_RATIO"),
        (F.col("AMT_GOODS_PRICE") / (F.col("AMT_INCOME_TOTAL") + 1)).alias("APP_GOODS_TO_INCOME_RATIO"),
        (F.col("AMT_INCOME_TOTAL") - F.coalesce(F.col("AMT_ANNUITY"), F.lit(0))).alias("APP_INCOME_AFTER_ANNUITY"),
        (F.col("AMT_INCOME_TOTAL") / fam).alias("APP_INCOME_PER_FAMILY_MEMBER"),
        (F.col("AMT_CREDIT") / fam).alias("APP_CREDIT_PER_FAMILY_MEMBER"),
        (F.col("CNT_CHILDREN") / fam).alias("APP_CHILDREN_RATIO"),
        # --- Age and employment ---
        (F.col("DAYS_BIRTH") / -365.25).alias("APP_AGE_YEARS"),
        (F.col("DAYS_EMPLOYED") / -365.25).alias("APP_EMPLOYMENT_YEARS"),
        (F.col("DAYS_EMPLOYED") / days_birth_safe).alias("APP_EMPLOYMENT_TO_AGE_RATIO"),
        F.when(F.col("DAYS_EMPLOYED") == 365243, 1).otherwise(0).alias("APP_DAYS_EMPLOYED_ANOMALY"),
        # --- External source features ---
        ext_mean.alias("APP_EXT_SOURCE_MEAN"),
        (ext1 * ext2 * ext3).alias("APP_EXT_SOURCE_PRODUCT"),
        _is_null_int("EXT_SOURCE_1").alias("APP_EXT_SOURCE_1_MISSING"),
        _is_null_int("EXT_SOURCE_2").alias("APP_EXT_SOURCE_2_MISSING"),
        _is_null_int("EXT_SOURCE_3").alias("APP_EXT_SOURCE_3_MISSING"),
        # Pairwise interactions
        (ext2 * ext3).alias("APP_EXT_SOURCE_2x3"),
        (ext1 * ext2).alias("APP_EXT_SOURCE_1x2"),
        (ext1 * ext3).alias("APP_EXT_SOURCE_1x3"),
        # Pairwise differences
        (ext2 - ext3).alias("APP_EXT_SOURCE_2_MINUS_3"),
        (ext1 - ext2).alias("APP_EXT_SOURCE_1_MINUS_2"),
        # Std dev approximation
        F.sqrt(
            (F.pow(ext1, 2) + F.pow(ext2, 2) + F.pow(ext3, 2)) / 3
            - F.pow((ext1 + ext2 + ext3) / 3, 2)
        ).alias("APP_EXT_SOURCE_STD"),
        # Weighted mean
        (ext1 * 0.15 + ext2 * 0.45 + ext3 * 0.40).alias("APP_EXT_SOURCE_WEIGHTED_MEAN"),
        # Count of non-null EXT_SOURCE values
        (_is_not_null_int("EXT_SOURCE_1") + _is_not_null_int("EXT_SOURCE_2")
         + _is_not_null_int("EXT_SOURCE_3")).alias("APP_EXT_SOURCE_COUNT"),
        # Squared terms
        F.pow(ext2, 2).alias("APP_EXT_SOURCE_2_SQ"),
        F.pow(ext3, 2).alias("APP_EXT_SOURCE_3_SQ"),
        # EXT_SOURCE crossed with age
        (ext2 * (F.col("DAYS_BIRTH") / -365.25)).alias("APP_EXT2_x_AGE"),
        (ext3 * (F.col("DAYS_BIRTH") / -365.25)).alias("APP_EXT3_x_AGE"),
        # EXT_SOURCE crossed with credit amount
        (ext2 * 1e6 / (F.col("AMT_CREDIT") + 1)).alias("APP_EXT2_PER_CREDIT"),
        # --- Document flags: count ---
        sum(F.col(c) for c in doc_cols).alias("APP_DOCUMENT_COUNT"),
        # --- Contact reachability ---
        (F.col("FLAG_CONT_MOBILE") + F.col("FLAG_EMAIL")
         + F.col("FLAG_PHONE") + F.col("FLAG_WORK_PHONE")).alias("APP_CONTACT_REACHABILITY"),
        # --- Address mismatch score ---
        (F.col("REG_REGION_NOT_LIVE_REGION") + F.col("REG_REGION_NOT_WORK_REGION")
         + F.col("LIVE_REGION_NOT_WORK_REGION") + F.col("REG_CITY_NOT_LIVE_CITY")
         + F.col("REG_CITY_NOT_WORK_CITY") + F.col("LIVE_CITY_NOT_WORK_CITY")).alias("APP_ADDRESS_MISMATCH_SCORE"),
        # --- Social circle default rates ---
        (_fill("DEF_30_CNT_SOCIAL_CIRCLE") / (_fill("OBS_30_CNT_SOCIAL_CIRCLE") + 1)).alias("APP_SOCIAL_DEF_30_RATE"),
        (_fill("DEF_60_CNT_SOCIAL_CIRCLE") / (_fill("OBS_60_CNT_SOCIAL_CIRCLE") + 1)).alias("APP_SOCIAL_DEF_60_RATE"),
        # --- Credit inquiry acceleration ---
        (_fill("AMT_REQ_CREDIT_BUREAU_QRT") + _fill("AMT_REQ_CREDIT_BUREAU_MON") * 3).alias("APP_RECENT_INQUIRIES_WEIGHTED"),
        # Total credit bureau inquiries
        (_fill("AMT_REQ_CREDIT_BUREAU_HOUR") + _fill("AMT_REQ_CREDIT_BUREAU_DAY")
         + _fill("AMT_REQ_CREDIT_BUREAU_WEEK") + _fill("AMT_REQ_CREDIT_BUREAU_MON")
         + _fill("AMT_REQ_CREDIT_BUREAU_QRT") + _fill("AMT_REQ_CREDIT_BUREAU_YEAR")).alias("APP_TOTAL_INQUIRIES"),
        # --- Days features ---
        (F.col("DAYS_REGISTRATION") / days_birth_safe).alias("APP_REGISTRATION_TO_AGE_RATIO"),
        (F.col("DAYS_ID_PUBLISH") / days_birth_safe).alias("APP_ID_PUBLISH_TO_AGE_RATIO"),
        (F.col("DAYS_LAST_PHONE_CHANGE") / days_birth_safe).alias("APP_PHONE_CHANGE_TO_AGE_RATIO"),
        (F.col("DAYS_REGISTRATION") / -365.25).alias("APP_REGISTRATION_YEARS"),
        (F.col("DAYS_ID_PUBLISH") / -365.25).alias("APP_ID_PUBLISH_YEARS"),
        (F.col("DAYS_LAST_PHONE_CHANGE") / -365.25).alias("APP_PHONE_CHANGE_YEARS"),
        # --- Building/housing info null count ---
        sum(_is_null_int(c) for c in building_cols).alias("APP_BUILDING_INFO_NULL_COUNT"),
    )

    # Fix anomalous DAYS_EMPLOYED: replace 365243 with null
    result = result.withColumn(
        "DAYS_EMPLOYED",
        F.when(F.col("DAYS_EMPLOYED") == 365243, None).otherwise(F.col("DAYS_EMPLOYED"))
    )

    return result

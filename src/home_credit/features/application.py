"""Feature engineering from the application_train / application_test tables."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_application_features(app: pl.DataFrame) -> pl.DataFrame:
    """Create domain-informed features from the application table.

    These stay at the applicant level (SK_ID_CURR) — no aggregation needed.
    """
    log.info("Building application features")

    return app.with_columns(
        # --- Income / credit ratios ---
        (pl.col("AMT_CREDIT") / (pl.col("AMT_INCOME_TOTAL") + 1))
        .alias("APP_CREDIT_TO_INCOME_RATIO"),
        (pl.col("AMT_ANNUITY") / (pl.col("AMT_INCOME_TOTAL") + 1))
        .alias("APP_ANNUITY_TO_INCOME_RATIO"),
        (pl.col("AMT_CREDIT") / (pl.col("AMT_GOODS_PRICE").fill_null(1) + 1))
        .alias("APP_CREDIT_TO_GOODS_RATIO"),
        (pl.col("AMT_ANNUITY") / (pl.col("AMT_CREDIT") + 1))
        .alias("APP_ANNUITY_TO_CREDIT_RATIO"),
        (pl.col("AMT_GOODS_PRICE") / (pl.col("AMT_INCOME_TOTAL") + 1))
        .alias("APP_GOODS_TO_INCOME_RATIO"),
        # Income remaining after annuity payment
        (pl.col("AMT_INCOME_TOTAL") - pl.col("AMT_ANNUITY").fill_null(0))
        .alias("APP_INCOME_AFTER_ANNUITY"),
        # Income per family member
        (pl.col("AMT_INCOME_TOTAL") / (pl.col("CNT_FAM_MEMBERS").fill_null(1).clip(1, None)))
        .alias("APP_INCOME_PER_FAMILY_MEMBER"),
        # Credit per family member
        (pl.col("AMT_CREDIT") / (pl.col("CNT_FAM_MEMBERS").fill_null(1).clip(1, None)))
        .alias("APP_CREDIT_PER_FAMILY_MEMBER"),
        # Children ratio
        (pl.col("CNT_CHILDREN") / (pl.col("CNT_FAM_MEMBERS").fill_null(1).clip(1, None)))
        .alias("APP_CHILDREN_RATIO"),
        # --- Age and employment ---
        (pl.col("DAYS_BIRTH") / -365.25).alias("APP_AGE_YEARS"),
        (pl.col("DAYS_EMPLOYED") / -365.25).alias("APP_EMPLOYMENT_YEARS"),
        (
            pl.col("DAYS_EMPLOYED")
            / pl.col("DAYS_BIRTH").replace(0, None)
        ).alias("APP_EMPLOYMENT_TO_AGE_RATIO"),
        # Flag anomalous DAYS_EMPLOYED (365243 = not employed placeholder)
        (pl.col("DAYS_EMPLOYED") == 365243).cast(pl.Int8).alias("APP_DAYS_EMPLOYED_ANOMALY"),
        # --- External source features ---
        (
            pl.col("EXT_SOURCE_1").fill_null(0)
            + pl.col("EXT_SOURCE_2").fill_null(0)
            + pl.col("EXT_SOURCE_3").fill_null(0)
        )
        .truediv(3)
        .alias("APP_EXT_SOURCE_MEAN"),
        (
            pl.col("EXT_SOURCE_1").fill_null(0)
            * pl.col("EXT_SOURCE_2").fill_null(0)
            * pl.col("EXT_SOURCE_3").fill_null(0)
        ).alias("APP_EXT_SOURCE_PRODUCT"),
        pl.col("EXT_SOURCE_1").is_null().cast(pl.Int8).alias("APP_EXT_SOURCE_1_MISSING"),
        pl.col("EXT_SOURCE_2").is_null().cast(pl.Int8).alias("APP_EXT_SOURCE_2_MISSING"),
        pl.col("EXT_SOURCE_3").is_null().cast(pl.Int8).alias("APP_EXT_SOURCE_3_MISSING"),
        # --- EXT_SOURCE pairwise interactions ---
        (pl.col("EXT_SOURCE_2").fill_null(0) * pl.col("EXT_SOURCE_3").fill_null(0))
        .alias("APP_EXT_SOURCE_2x3"),
        (pl.col("EXT_SOURCE_1").fill_null(0) * pl.col("EXT_SOURCE_2").fill_null(0))
        .alias("APP_EXT_SOURCE_1x2"),
        (pl.col("EXT_SOURCE_1").fill_null(0) * pl.col("EXT_SOURCE_3").fill_null(0))
        .alias("APP_EXT_SOURCE_1x3"),
        # Pairwise differences (captures relative standing between scoring systems)
        (pl.col("EXT_SOURCE_2").fill_null(0) - pl.col("EXT_SOURCE_3").fill_null(0))
        .alias("APP_EXT_SOURCE_2_MINUS_3"),
        (pl.col("EXT_SOURCE_1").fill_null(0) - pl.col("EXT_SOURCE_2").fill_null(0))
        .alias("APP_EXT_SOURCE_1_MINUS_2"),
        # Variance/disagreement among the scores
        (
            (
                (pl.col("EXT_SOURCE_1").fill_null(0).pow(2)
                 + pl.col("EXT_SOURCE_2").fill_null(0).pow(2)
                 + pl.col("EXT_SOURCE_3").fill_null(0).pow(2)) / 3
                - ((pl.col("EXT_SOURCE_1").fill_null(0)
                    + pl.col("EXT_SOURCE_2").fill_null(0)
                    + pl.col("EXT_SOURCE_3").fill_null(0)) / 3).pow(2)
            ).sqrt()
        ).alias("APP_EXT_SOURCE_STD"),
        # Weighted mean (EXT_SOURCE_2 and _3 are more predictive per SHAP)
        (
            pl.col("EXT_SOURCE_1").fill_null(0) * 0.15
            + pl.col("EXT_SOURCE_2").fill_null(0) * 0.45
            + pl.col("EXT_SOURCE_3").fill_null(0) * 0.40
        ).alias("APP_EXT_SOURCE_WEIGHTED_MEAN"),
        # Count of non-null EXT_SOURCE values
        (
            pl.col("EXT_SOURCE_1").is_not_null().cast(pl.Int8)
            + pl.col("EXT_SOURCE_2").is_not_null().cast(pl.Int8)
            + pl.col("EXT_SOURCE_3").is_not_null().cast(pl.Int8)
        ).alias("APP_EXT_SOURCE_COUNT"),
        # Squared terms (capture non-linear effects)
        (pl.col("EXT_SOURCE_2").fill_null(0).pow(2)).alias("APP_EXT_SOURCE_2_SQ"),
        (pl.col("EXT_SOURCE_3").fill_null(0).pow(2)).alias("APP_EXT_SOURCE_3_SQ"),
        # EXT_SOURCE crossed with age
        (pl.col("EXT_SOURCE_2").fill_null(0) * (pl.col("DAYS_BIRTH") / -365.25))
        .alias("APP_EXT2_x_AGE"),
        (pl.col("EXT_SOURCE_3").fill_null(0) * (pl.col("DAYS_BIRTH") / -365.25))
        .alias("APP_EXT3_x_AGE"),
        # EXT_SOURCE crossed with credit amount
        (pl.col("EXT_SOURCE_2").fill_null(0) * 1e6 / (pl.col("AMT_CREDIT") + 1))
        .alias("APP_EXT2_PER_CREDIT"),
        # --- Document flags: count how many documents were provided ---
        pl.sum_horizontal([
            pl.col(c) for c in [
                "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4",
                "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7",
                "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
                "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
                "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
                "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
                "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
            ]
        ]).alias("APP_DOCUMENT_COUNT"),
        # --- Social surroundings contact reachability ---
        pl.sum_horizontal([
            pl.col(c) for c in [
                "FLAG_CONT_MOBILE", "FLAG_EMAIL", "FLAG_PHONE",
                "FLAG_WORK_PHONE",
            ]
        ]).alias("APP_CONTACT_REACHABILITY"),
        # --- Address mismatch score ---
        pl.sum_horizontal([
            pl.col("REG_REGION_NOT_LIVE_REGION"),
            pl.col("REG_REGION_NOT_WORK_REGION"),
            pl.col("LIVE_REGION_NOT_WORK_REGION"),
            pl.col("REG_CITY_NOT_LIVE_CITY"),
            pl.col("REG_CITY_NOT_WORK_CITY"),
            pl.col("LIVE_CITY_NOT_WORK_CITY"),
        ]).alias("APP_ADDRESS_MISMATCH_SCORE"),
        # --- Social circle default rates ---
        (pl.col("DEF_30_CNT_SOCIAL_CIRCLE").fill_null(0)
         / (pl.col("OBS_30_CNT_SOCIAL_CIRCLE").fill_null(0) + 1))
        .alias("APP_SOCIAL_DEF_30_RATE"),
        (pl.col("DEF_60_CNT_SOCIAL_CIRCLE").fill_null(0)
         / (pl.col("OBS_60_CNT_SOCIAL_CIRCLE").fill_null(0) + 1))
        .alias("APP_SOCIAL_DEF_60_RATE"),
        # --- Credit inquiry acceleration (recent vs older) ---
        (pl.col("AMT_REQ_CREDIT_BUREAU_QRT").fill_null(0)
         + pl.col("AMT_REQ_CREDIT_BUREAU_MON").fill_null(0) * 3)
        .alias("APP_RECENT_INQUIRIES_WEIGHTED"),
        # Total credit bureau inquiries
        pl.sum_horizontal([
            pl.col("AMT_REQ_CREDIT_BUREAU_HOUR").fill_null(0),
            pl.col("AMT_REQ_CREDIT_BUREAU_DAY").fill_null(0),
            pl.col("AMT_REQ_CREDIT_BUREAU_WEEK").fill_null(0),
            pl.col("AMT_REQ_CREDIT_BUREAU_MON").fill_null(0),
            pl.col("AMT_REQ_CREDIT_BUREAU_QRT").fill_null(0),
            pl.col("AMT_REQ_CREDIT_BUREAU_YEAR").fill_null(0),
        ]).alias("APP_TOTAL_INQUIRIES"),
        # --- Days features ---
        (pl.col("DAYS_REGISTRATION") / pl.col("DAYS_BIRTH").replace(0, None))
        .alias("APP_REGISTRATION_TO_AGE_RATIO"),
        (pl.col("DAYS_ID_PUBLISH") / pl.col("DAYS_BIRTH").replace(0, None))
        .alias("APP_ID_PUBLISH_TO_AGE_RATIO"),
        (pl.col("DAYS_LAST_PHONE_CHANGE") / pl.col("DAYS_BIRTH").replace(0, None))
        .alias("APP_PHONE_CHANGE_TO_AGE_RATIO"),
        # Absolute time features (years)
        (pl.col("DAYS_REGISTRATION") / -365.25).alias("APP_REGISTRATION_YEARS"),
        (pl.col("DAYS_ID_PUBLISH") / -365.25).alias("APP_ID_PUBLISH_YEARS"),
        (pl.col("DAYS_LAST_PHONE_CHANGE") / -365.25).alias("APP_PHONE_CHANGE_YEARS"),
        # --- Building/housing info null count (missingness pattern) ---
        pl.sum_horizontal([
            pl.col(c).is_null().cast(pl.Int8) for c in [
                "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
                "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG",
                "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
                "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG",
                "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
            ]
        ]).alias("APP_BUILDING_INFO_NULL_COUNT"),
    ).with_columns(
        # Fix anomalous DAYS_EMPLOYED: replace 365243 with null
        pl.when(pl.col("DAYS_EMPLOYED") == 365243)
        .then(None)
        .otherwise(pl.col("DAYS_EMPLOYED"))
        .alias("DAYS_EMPLOYED"),
    )

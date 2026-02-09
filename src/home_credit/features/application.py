"""Feature engineering from the application_train / application_test tables."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_application_features(app: pl.LazyFrame) -> pl.LazyFrame:
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
        # --- Days features ---
        (pl.col("DAYS_REGISTRATION") / pl.col("DAYS_BIRTH").replace(0, None))
        .alias("APP_REGISTRATION_TO_AGE_RATIO"),
        (pl.col("DAYS_ID_PUBLISH") / pl.col("DAYS_BIRTH").replace(0, None))
        .alias("APP_ID_PUBLISH_TO_AGE_RATIO"),
        (pl.col("DAYS_LAST_PHONE_CHANGE") / pl.col("DAYS_BIRTH").replace(0, None))
        .alias("APP_PHONE_CHANGE_TO_AGE_RATIO"),
    ).with_columns(
        # Fix anomalous DAYS_EMPLOYED: replace 365243 with null
        pl.when(pl.col("DAYS_EMPLOYED") == 365243)
        .then(None)
        .otherwise(pl.col("DAYS_EMPLOYED"))
        .alias("DAYS_EMPLOYED"),
    )

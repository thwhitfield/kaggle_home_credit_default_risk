"""Aggregated features from previous_application table."""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)


def build_previous_application_features(prev: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate previous Home Credit applications to SK_ID_CURR level."""
    log.info("Building previous application features")

    feats = prev.group_by("SK_ID_CURR").agg(
        # --- Counts by status ---
        pl.len().alias("PREV_COUNT"),
        (pl.col("NAME_CONTRACT_STATUS") == "Approved").sum().alias("PREV_APPROVED_COUNT"),
        (pl.col("NAME_CONTRACT_STATUS") == "Refused").sum().alias("PREV_REFUSED_COUNT"),
        (pl.col("NAME_CONTRACT_STATUS") == "Canceled").sum().alias("PREV_CANCELED_COUNT"),
        # --- Loan amounts ---
        pl.col("AMT_APPLICATION").mean().alias("PREV_AMT_APPLICATION_MEAN"),
        pl.col("AMT_APPLICATION").max().alias("PREV_AMT_APPLICATION_MAX"),
        pl.col("AMT_APPLICATION").sum().alias("PREV_AMT_APPLICATION_TOTAL"),
        pl.col("AMT_CREDIT").mean().alias("PREV_AMT_CREDIT_MEAN"),
        pl.col("AMT_CREDIT").sum().alias("PREV_AMT_CREDIT_TOTAL"),
        # How much the final credit differed from what was applied for
        (pl.col("AMT_APPLICATION") - pl.col("AMT_CREDIT")).mean()
        .alias("PREV_APP_CREDIT_DIFF_MEAN"),
        # Annuity
        pl.col("AMT_ANNUITY").mean().alias("PREV_ANNUITY_MEAN"),
        pl.col("AMT_ANNUITY").max().alias("PREV_ANNUITY_MAX"),
        # Down payment
        pl.col("AMT_DOWN_PAYMENT").mean().alias("PREV_DOWN_PAYMENT_MEAN"),
        pl.col("AMT_DOWN_PAYMENT").max().alias("PREV_DOWN_PAYMENT_MAX"),
        # --- Time features ---
        pl.col("DAYS_DECISION").max().alias("PREV_MOST_RECENT_DECISION"),
        pl.col("DAYS_DECISION").min().alias("PREV_OLDEST_DECISION"),
        # Days from application to first draw
        pl.col("DAYS_FIRST_DRAWING").mean().alias("PREV_DAYS_FIRST_DRAWING_MEAN"),
        pl.col("DAYS_FIRST_DUE").mean().alias("PREV_DAYS_FIRST_DUE_MEAN"),
        pl.col("DAYS_LAST_DUE_1ST_VERSION").mean().alias("PREV_DAYS_LAST_DUE_MEAN"),
        pl.col("DAYS_TERMINATION").mean().alias("PREV_DAYS_TERMINATION_MEAN"),
        # --- Contract types ---
        (pl.col("NAME_CONTRACT_TYPE") == "Cash loans").sum().alias("PREV_CASH_LOAN_COUNT"),
        (pl.col("NAME_CONTRACT_TYPE") == "Revolving loans").sum().alias("PREV_REVOLVING_COUNT"),
        # --- Product complexity ---
        pl.col("PRODUCT_COMBINATION").n_unique().alias("PREV_PRODUCT_VARIETY"),
        # --- Yield group ---
        (pl.col("NAME_YIELD_GROUP") == "high").sum().alias("PREV_HIGH_YIELD_COUNT"),
        (pl.col("NAME_YIELD_GROUP") == "low_normal").sum().alias("PREV_LOW_YIELD_COUNT"),
        # --- Channel ---
        pl.col("CHANNEL_TYPE").n_unique().alias("PREV_CHANNEL_VARIETY"),
        # --- Weekday/hour patterns ---
        (pl.col("WEEKDAY_APPR_PROCESS_START") == "SATURDAY").sum()
        .alias("PREV_WEEKEND_APP_COUNT"),
        (pl.col("WEEKDAY_APPR_PROCESS_START") == "SUNDAY").sum()
        .alias("PREV_SUNDAY_APP_COUNT"),
        # Insurance
        (pl.col("NFLAG_INSURED_ON_APPROVAL") == 1).sum().alias("PREV_INSURED_COUNT"),
    )

    # Derived ratios
    feats = feats.with_columns(
        # Approval rate
        (pl.col("PREV_APPROVED_COUNT") / (pl.col("PREV_COUNT") + 1))
        .alias("PREV_APPROVAL_RATE"),
        # Refusal rate
        (pl.col("PREV_REFUSED_COUNT") / (pl.col("PREV_COUNT") + 1))
        .alias("PREV_REFUSAL_RATE"),
        # Cash vs revolving ratio
        (pl.col("PREV_CASH_LOAN_COUNT") / (pl.col("PREV_COUNT") + 1))
        .alias("PREV_CASH_LOAN_RATIO"),
        # How recently they applied (history span)
        (pl.col("PREV_MOST_RECENT_DECISION") - pl.col("PREV_OLDEST_DECISION"))
        .alias("PREV_HISTORY_SPAN"),
    )

    return feats

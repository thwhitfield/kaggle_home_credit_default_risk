"""Orchestrate all feature engineering: build, join, and save."""

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from home_credit.data.loader import load_table
from home_credit.features.application import build_application_features
from home_credit.features.bureau import build_bureau_features
from home_credit.features.credit_card import build_credit_card_features
from home_credit.features.experimental import build_winning_experimental_features
from home_credit.features.installments import build_installment_features
from home_credit.features.pos_cash import build_pos_cash_features
from home_credit.features.previous import build_previous_application_features
from home_credit.utils import DATA_DIR, FEATURES_DIR, get_logger, timer

log = get_logger(__name__)


def _add_cross_table_features(df: DataFrame) -> DataFrame:
    """Add interaction features that combine signals across tables."""
    log.info("Adding cross-table interaction features")

    cols = set(df.columns)

    # --- Bureau debt relative to current application ---
    if "BUR_DEBT_TOTAL" in cols and "AMT_CREDIT" in cols:
        df = df.withColumn(
            "CROSS_EXISTING_DEBT_TO_NEW_CREDIT",
            F.coalesce(F.col("BUR_DEBT_TOTAL"), F.lit(0)) / (F.col("AMT_CREDIT") + 1)
        )

    # --- Total credit exposure relative to income ---
    if "BUR_AMT_CREDIT_TOTAL" in cols and "AMT_INCOME_TOTAL" in cols:
        df = df.withColumn(
            "CROSS_TOTAL_CREDIT_TO_INCOME",
            (F.coalesce(F.col("BUR_AMT_CREDIT_TOTAL"), F.lit(0)) + F.col("AMT_CREDIT"))
            / (F.col("AMT_INCOME_TOTAL") + 1)
        )

    # --- Bureau annuity burden combined with current annuity ---
    if "BUR_ACTIVE_ANNUITY_TOTAL" in cols and "AMT_ANNUITY" in cols:
        df = df.withColumn(
            "CROSS_TOTAL_ANNUITY_TO_INCOME",
            (F.coalesce(F.col("BUR_ACTIVE_ANNUITY_TOTAL"), F.lit(0))
             + F.coalesce(F.col("AMT_ANNUITY"), F.lit(0)))
            / (F.col("AMT_INCOME_TOTAL") + 1)
        )

    # --- Refusal rate crossed with credit amount ---
    if "PREV_REFUSAL_RATE" in cols:
        df = df.withColumn(
            "CROSS_REFUSAL_RATE_x_CREDIT",
            F.coalesce(F.col("PREV_REFUSAL_RATE"), F.lit(0)) * F.col("AMT_CREDIT")
        )

    # --- External score crossed with late payment behavior ---
    if "INS_LATE_RATE" in cols and "EXT_SOURCE_2" in cols:
        df = df.withColumn(
            "CROSS_EXT2_x_LATE_RATE",
            F.coalesce(F.col("EXT_SOURCE_2"), F.lit(0))
            * F.coalesce(F.col("INS_LATE_RATE"), F.lit(0))
        )

    # --- Is this a first-time applicant? ---
    if "PREV_COUNT" in cols:
        df = df.withColumn(
            "CROSS_FIRST_TIME_APPLICANT",
            F.when(F.col("PREV_COUNT").isNull(), 1).otherwise(0)
        )
    if "BUR_COUNT" in cols:
        df = df.withColumn(
            "CROSS_NO_BUREAU_HISTORY",
            F.when(F.col("BUR_COUNT").isNull(), 1).otherwise(0)
        )

    # --- Credit amount vs previous approved amount (escalation risk) ---
    if "PREV_APPROVED_CREDIT_MEAN" in cols:
        df = df.withColumn(
            "CROSS_CREDIT_VS_PREV_APPROVED",
            F.col("AMT_CREDIT")
            / (F.coalesce(F.col("PREV_APPROVED_CREDIT_MEAN"), F.lit(0)) + 1)
        )

    # --- Combined DPD risk score ---
    dpd_exprs = []
    for c in ["BUR_BB_DPD_RATE_MEAN", "POS_DPD_RATE", "CC_DPD_RATE", "INS_LATE_RATE"]:
        if c in cols:
            dpd_exprs.append(F.coalesce(F.col(c), F.lit(0)))
    if len(dpd_exprs) >= 2:
        combined = dpd_exprs[0]
        for expr in dpd_exprs[1:]:
            combined = combined + expr
        df = df.withColumn("CROSS_COMBINED_DPD_SCORE", combined)

    # --- EXT_SOURCE × bureau delinquency ---
    if "EXT_SOURCE_3" in cols and "BUR_BB_DPD_RATE_MEAN" in cols:
        df = df.withColumn(
            "CROSS_EXT3_ADJUSTED_BY_DPD",
            F.coalesce(F.col("EXT_SOURCE_3"), F.lit(0))
            * (1 - F.coalesce(F.col("BUR_BB_DPD_RATE_MEAN"), F.lit(0)))
        )
    if "EXT_SOURCE_2" in cols and "BUR_OVERDUE_RATE" in cols:
        df = df.withColumn(
            "CROSS_EXT2_ADJUSTED_BY_OVERDUE",
            F.coalesce(F.col("EXT_SOURCE_2"), F.lit(0))
            * (1 - F.coalesce(F.col("BUR_OVERDUE_RATE"), F.lit(0)))
        )

    # --- Income × active debt load ---
    if "BUR_ACTIVE_DEBT_TOTAL" in cols and "AMT_INCOME_TOTAL" in cols:
        df = df.withColumn(
            "CROSS_ACTIVE_DEBT_TO_INCOME",
            F.coalesce(F.col("BUR_ACTIVE_DEBT_TOTAL"), F.lit(0))
            / (F.col("AMT_INCOME_TOTAL") + 1)
        )

    # --- Age × credit card utilization ---
    if "CC_UTILIZATION_MEAN" in cols and "DAYS_BIRTH" in cols:
        df = df.withColumn(
            "CROSS_CC_UTIL_PER_AGE",
            F.coalesce(F.col("CC_UTILIZATION_MEAN"), F.lit(0))
            / ((F.col("DAYS_BIRTH") / -365.25) + 1)
        )

    # --- Employment stability × payment behavior ---
    if "INS_LATE_RATE" in cols and "DAYS_EMPLOYED" in cols:
        df = df.withColumn(
            "CROSS_LATE_RATE_x_EMPLOYMENT",
            F.coalesce(F.col("INS_LATE_RATE"), F.lit(0))
            * F.greatest(F.lit(0), F.least(F.lit(50),
              F.coalesce(F.col("DAYS_EMPLOYED"), F.lit(0)) / -365.25))
        )

    # --- Thin file indicator ---
    supp_null_exprs = []
    for c in ["BUR_COUNT", "PREV_COUNT", "INS_COUNT", "POS_COUNT", "CC_COUNT"]:
        if c in cols:
            supp_null_exprs.append(F.when(F.col(c).isNull(), 1).otherwise(0))
    if supp_null_exprs:
        combined = supp_null_exprs[0]
        for expr in supp_null_exprs[1:]:
            combined = combined + expr
        df = df.withColumn("CROSS_THIN_FILE_SCORE", combined)

    # --- Recent behavior risk score ---
    recent_dpd = []
    for c, divisor in [
        ("BUR_BB_RECENT_6M_DPD_RATE_MEAN", None),
        ("POS_RECENT_6M_DPD_RATE", None),
        ("INS_RECENT_12_LATE_RATE", None),
        ("CC_RECENT_12M_DPD_COUNT", 12.0),
    ]:
        if c in cols:
            expr = F.coalesce(F.col(c), F.lit(0))
            if divisor is not None:
                expr = expr / divisor
            recent_dpd.append(expr)
    if len(recent_dpd) >= 2:
        combined = recent_dpd[0]
        for expr in recent_dpd[1:]:
            combined = combined + expr
        df = df.withColumn("CROSS_RECENT_DPD_SCORE", combined)

    # --- Worsening behavior score ---
    if "CC_UTIL_TREND_12M" in cols and "INS_RECENT_12_PAYMENT_RATIO" in cols:
        df = df.withColumn(
            "CROSS_WORSENING_BEHAVIOR_SCORE",
            F.coalesce(F.col("CC_UTIL_TREND_12M"), F.lit(0))
            - F.coalesce(F.col("INS_RECENT_12_PAYMENT_RATIO"), F.lit(1)) + 1
        )

    # --- Previous app count × EXT_SOURCE ---
    if "PREV_COUNT" in cols and "APP_EXT_SOURCE_MEAN" in cols:
        df = df.withColumn(
            "CROSS_EXPERIENCE_x_SCORE",
            F.log1p(F.coalesce(F.col("PREV_COUNT"), F.lit(0)))
            * F.coalesce(F.col("APP_EXT_SOURCE_MEAN"), F.lit(0))
        )

    return df


def build_all_features(
    data_dir: Path = DATA_DIR,
    features_dir: Path = FEATURES_DIR,
    use_cache: bool = True,
) -> tuple[DataFrame, DataFrame]:
    """Build all features and return (train_df, test_df) with TARGET column.

    Returns DataFrames ready for model training (all features joined).
    """
    features_dir.mkdir(parents=True, exist_ok=True)

    # --- Application features (both train and test) ---
    with timer("Building application train features", log):
        app_train = build_application_features(
            app=load_table("application_train", data_dir),
        )
    with timer("Building application test features", log):
        app_test = build_application_features(
            app=load_table("application_test", data_dir),
        )

    # --- Supplementary table features (shared between train and test) ---
    with timer("Building bureau features", log):
        bureau_feats = build_bureau_features(
            bureau=load_table("bureau", data_dir),
            bureau_balance=load_table("bureau_balance", data_dir),
        )

    with timer("Building previous application features", log):
        prev_feats = build_previous_application_features(
            prev=load_table("previous_application", data_dir),
        )

    with timer("Building installment features", log):
        installment_feats = build_installment_features(
            installments=load_table("installments_payments", data_dir),
        )

    with timer("Building POS/cash features", log):
        pos_feats = build_pos_cash_features(
            pos=load_table("POS_CASH_balance", data_dir),
        )

    with timer("Building credit card features", log):
        cc_feats = build_credit_card_features(
            cc=load_table("credit_card_balance", data_dir),
        )

    # --- Experimental features ---
    with timer("Building experimental features", log):
        exp_feats = build_winning_experimental_features(
            installments=load_table("installments_payments", data_dir),
            cc=load_table("credit_card_balance", data_dir),
            pos=load_table("POS_CASH_balance", data_dir),
            bureau=load_table("bureau", data_dir),
            bureau_balance=load_table("bureau_balance", data_dir),
            prev=load_table("previous_application", data_dir),
        )

    # --- Join all supplementary features onto application ---
    supp_tables = [bureau_feats, prev_feats, installment_feats, pos_feats, cc_feats, exp_feats]

    def join_all(app_df: DataFrame) -> DataFrame:
        result = app_df
        for feats_df in supp_tables:
            result = result.join(feats_df, "SK_ID_CURR", "left")
        result = _add_cross_table_features(result)
        return result

    with timer("Joining all features", log):
        train_df = join_all(app_train)
        test_df = join_all(app_test)

    train_count = train_df.count()
    test_count = test_df.count()
    log.info(f"Final train shape: ({train_count}, {len(train_df.columns)})")
    log.info(f"Final test shape: ({test_count}, {len(test_df.columns)})")

    # Save final datasets
    final_train_path = str(features_dir / "train_final.parquet")
    final_test_path = str(features_dir / "test_final.parquet")
    train_df.write.mode("overwrite").parquet(final_train_path)
    test_df.write.mode("overwrite").parquet(final_test_path)
    log.info("Saved final feature sets to parquet")

    return train_df, test_df


def get_feature_columns(df: DataFrame) -> list[str]:
    """Get feature columns (everything except SK_ID_CURR and TARGET)."""
    exclude = {"SK_ID_CURR", "TARGET"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    train_df, test_df = build_all_features(use_cache=False)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"\nTrain: {train_count:,} rows x {len(train_df.columns)} features")
    print(f"Test:  {test_count:,} rows x {len(test_df.columns)} features")
    feature_cols = get_feature_columns(train_df)
    print(f"Feature columns: {len(feature_cols)}")

    # Show nulls
    for c in feature_cols:
        null_count = train_df.filter(F.col(c).isNull()).count()
        null_pct = null_count / train_count * 100
        if null_pct > 50:
            print(f"  {c}: {null_pct:.1f}%")

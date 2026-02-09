"""Orchestrate all feature engineering: build, join, and save."""

from pathlib import Path

import polars as pl

from home_credit.data.loader import load_table
from home_credit.features.application import build_application_features
from home_credit.features.bureau import build_bureau_features
from home_credit.features.credit_card import build_credit_card_features
from home_credit.features.installments import build_installment_features
from home_credit.features.pos_cash import build_pos_cash_features
from home_credit.features.previous import build_previous_application_features
from home_credit.utils import DATA_DIR, FEATURES_DIR, get_logger, timer

log = get_logger(__name__)


def _build_and_save(name: str, builder, features_dir: Path, **kwargs) -> pl.DataFrame:
    """Build a feature set, save to parquet, and return it."""
    parquet_path = features_dir / f"{name}.parquet"
    if parquet_path.exists():
        log.info(f"Loading cached {name} features from {parquet_path}")
        return pl.read_parquet(parquet_path)

    with timer(f"Building {name} features", log):
        feats = builder(**kwargs)
    feats.write_parquet(parquet_path)
    log.info(f"Saved {name} features: {feats.shape[1]} columns, {feats.shape[0]:,} rows")
    return feats


def _add_cross_table_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add interaction features that combine signals across tables."""
    log.info("Adding cross-table interaction features")

    exprs = []

    # --- Bureau debt relative to current application ---
    if "BUR_DEBT_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
        exprs.append(
            (pl.col("BUR_DEBT_TOTAL").fill_null(0) / (pl.col("AMT_CREDIT") + 1))
            .alias("CROSS_EXISTING_DEBT_TO_NEW_CREDIT")
        )

    # --- Total credit exposure relative to income ---
    if "BUR_AMT_CREDIT_TOTAL" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        exprs.append(
            ((pl.col("BUR_AMT_CREDIT_TOTAL").fill_null(0) + pl.col("AMT_CREDIT"))
             / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("CROSS_TOTAL_CREDIT_TO_INCOME")
        )

    # --- Bureau annuity burden combined with current annuity ---
    if "BUR_ACTIVE_ANNUITY_TOTAL" in df.columns and "AMT_ANNUITY" in df.columns:
        exprs.append(
            ((pl.col("BUR_ACTIVE_ANNUITY_TOTAL").fill_null(0)
              + pl.col("AMT_ANNUITY").fill_null(0))
             / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("CROSS_TOTAL_ANNUITY_TO_INCOME")
        )

    # --- Refusal rate crossed with credit amount ---
    if "PREV_REFUSAL_RATE" in df.columns:
        exprs.append(
            (pl.col("PREV_REFUSAL_RATE").fill_null(0) * pl.col("AMT_CREDIT"))
            .alias("CROSS_REFUSAL_RATE_x_CREDIT")
        )

    # --- External score crossed with late payment behavior ---
    if "INS_LATE_RATE" in df.columns and "EXT_SOURCE_2" in df.columns:
        exprs.append(
            (pl.col("EXT_SOURCE_2").fill_null(0) * pl.col("INS_LATE_RATE").fill_null(0))
            .alias("CROSS_EXT2_x_LATE_RATE")
        )

    # --- Is this a first-time applicant? (no previous applications) ---
    if "PREV_COUNT" in df.columns:
        exprs.append(
            pl.col("PREV_COUNT").is_null().cast(pl.Int8)
            .alias("CROSS_FIRST_TIME_APPLICANT")
        )
    if "BUR_COUNT" in df.columns:
        exprs.append(
            pl.col("BUR_COUNT").is_null().cast(pl.Int8)
            .alias("CROSS_NO_BUREAU_HISTORY")
        )

    # --- Credit amount vs previous approved amount (escalation risk) ---
    if "PREV_APPROVED_CREDIT_MEAN" in df.columns:
        exprs.append(
            (pl.col("AMT_CREDIT") / (pl.col("PREV_APPROVED_CREDIT_MEAN").fill_null(0) + 1))
            .alias("CROSS_CREDIT_VS_PREV_APPROVED")
        )

    # --- Combined DPD risk score (across bureau, POS, CC, installments) ---
    dpd_cols = []
    if "BUR_BB_DPD_RATE_MEAN" in df.columns:
        dpd_cols.append(pl.col("BUR_BB_DPD_RATE_MEAN").fill_null(0))
    if "POS_DPD_RATE" in df.columns:
        dpd_cols.append(pl.col("POS_DPD_RATE").fill_null(0))
    if "CC_DPD_RATE" in df.columns:
        dpd_cols.append(pl.col("CC_DPD_RATE").fill_null(0))
    if "INS_LATE_RATE" in df.columns:
        dpd_cols.append(pl.col("INS_LATE_RATE").fill_null(0))
    if len(dpd_cols) >= 2:
        exprs.append(
            pl.sum_horizontal(dpd_cols).alias("CROSS_COMBINED_DPD_SCORE")
        )

    if exprs:
        df = df.with_columns(exprs)

    return df


def build_all_features(
    data_dir: Path = DATA_DIR,
    features_dir: Path = FEATURES_DIR,
    use_cache: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build all features and return (train_df, test_df) with TARGET column.

    Returns DataFrames ready for model training (all features joined).
    """
    features_dir.mkdir(parents=True, exist_ok=True)

    final_train_path = features_dir / "train_final.parquet"
    final_test_path = features_dir / "test_final.parquet"

    if use_cache and final_train_path.exists() and final_test_path.exists():
        log.info("Loading cached final feature sets")
        return pl.read_parquet(final_train_path), pl.read_parquet(final_test_path)

    # --- Application features (both train and test) ---
    app_train = _build_and_save(
        "app_train", build_application_features, features_dir,
        app=load_table("application_train", data_dir),
    )
    app_test = _build_and_save(
        "app_test", build_application_features, features_dir,
        app=load_table("application_test", data_dir),
    )

    # --- Supplementary table features (shared between train and test) ---
    bureau_feats = _build_and_save(
        "bureau", build_bureau_features, features_dir,
        bureau=load_table("bureau", data_dir),
        bureau_balance=load_table("bureau_balance", data_dir),
    )

    prev_feats = _build_and_save(
        "previous", build_previous_application_features, features_dir,
        prev=load_table("previous_application", data_dir),
    )

    installment_feats = _build_and_save(
        "installments", build_installment_features, features_dir,
        installments=load_table("installments_payments", data_dir),
    )

    pos_feats = _build_and_save(
        "pos_cash", build_pos_cash_features, features_dir,
        pos=load_table("POS_CASH_balance", data_dir),
    )

    cc_feats = _build_and_save(
        "credit_card", build_credit_card_features, features_dir,
        cc=load_table("credit_card_balance", data_dir),
    )

    # --- Join all supplementary features onto application ---
    supp_tables = [bureau_feats, prev_feats, installment_feats, pos_feats, cc_feats]

    def join_all(app_df: pl.DataFrame) -> pl.DataFrame:
        result = app_df
        for feats_df in supp_tables:
            result = result.join(feats_df, on="SK_ID_CURR", how="left")
        # Add cross-table interaction features
        result = _add_cross_table_features(result)
        return result

    with timer("Joining all features", log):
        train_df = join_all(app_train)
        test_df = join_all(app_test)

    log.info(f"Final train shape: {train_df.shape}")
    log.info(f"Final test shape: {test_df.shape}")

    # Save final datasets
    train_df.write_parquet(final_train_path)
    test_df.write_parquet(final_test_path)
    log.info("Saved final feature sets to parquet")

    return train_df, test_df


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get feature columns (everything except SK_ID_CURR and TARGET)."""
    exclude = {"SK_ID_CURR", "TARGET"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    train_df, test_df = build_all_features(use_cache=False)
    print(f"\nTrain: {train_df.shape[0]:,} rows x {train_df.shape[1]} features")
    print(f"Test:  {test_df.shape[0]:,} rows x {test_df.shape[1]} features")
    feature_cols = get_feature_columns(train_df)
    print(f"Feature columns: {len(feature_cols)}")

    # Show nulls
    null_pcts = {
        c: train_df[c].null_count() / train_df.shape[0] * 100
        for c in feature_cols
    }
    high_null = {k: v for k, v in null_pcts.items() if v > 50}
    if high_null:
        print(f"\nHigh null features (>50%): {len(high_null)}")
        for k, v in sorted(high_null.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k}: {v:.1f}%")

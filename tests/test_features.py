"""Tests for feature engineering modules."""

import pytest
from pyspark.sql import DataFrame, functions as F

from home_credit.utils import DATA_DIR, FEATURES_DIR, get_spark_session


@pytest.fixture(scope="module")
def check_data_exists():
    if not (DATA_DIR / "application_train.csv").exists():
        pytest.skip("Data not downloaded")


@pytest.fixture(scope="module")
def check_features_exist():
    if not (FEATURES_DIR / "train_final.parquet").exists():
        pytest.skip("Features not built — run `python -m home_credit.features.pipeline` first")


class TestApplicationFeatures:
    def test_builds_without_error(self, check_data_exists):
        from home_credit.data.loader import load_table
        from home_credit.features.application import build_application_features

        df = load_table("application_train")
        result = build_application_features(df)
        assert result.count() > 0
        assert "APP_CREDIT_TO_INCOME_RATIO" in result.columns
        assert "APP_AGE_YEARS" in result.columns
        assert "APP_EXT_SOURCE_MEAN" in result.columns

    def test_no_all_null_columns(self, check_data_exists):
        from home_credit.data.loader import load_table
        from home_credit.features.application import build_application_features

        result = build_application_features(load_table("application_train"))
        total = result.count()
        new_cols = [c for c in result.columns if c.startswith("APP_")]
        for col in new_cols:
            null_count = result.filter(F.col(col).isNull()).count()
            assert null_count < total, f"{col} is all null"


class TestBureauFeatures:
    def test_builds_without_error(self, check_data_exists):
        from home_credit.data.loader import load_table
        from home_credit.features.bureau import build_bureau_features

        result = build_bureau_features(
            load_table("bureau"),
            load_table("bureau_balance"),
        )
        assert result.count() > 0
        assert "BUR_COUNT" in result.columns
        assert "BUR_DEBT_TO_CREDIT_RATIO" in result.columns

    def test_aggregates_to_sk_id_curr(self, check_data_exists):
        from home_credit.data.loader import load_table
        from home_credit.features.bureau import build_bureau_features

        result = build_bureau_features(
            load_table("bureau"),
            load_table("bureau_balance"),
        )
        n_unique = result.select(F.countDistinct("SK_ID_CURR")).collect()[0][0]
        total = result.count()
        assert n_unique == total, "Not unique per SK_ID_CURR"


class TestFinalFeatureSet:
    def test_train_has_target(self, check_features_exist):
        spark = get_spark_session()
        df = spark.read.parquet(str(FEATURES_DIR / "train_final.parquet"))
        assert "TARGET" in df.columns
        n_unique = df.select(F.countDistinct("TARGET")).collect()[0][0]
        assert n_unique == 2

    def test_test_has_no_target(self, check_features_exist):
        spark = get_spark_session()
        df = spark.read.parquet(str(FEATURES_DIR / "test_final.parquet"))
        assert "TARGET" not in df.columns

    def test_no_all_null_features(self, check_features_exist):
        spark = get_spark_session()
        df = spark.read.parquet(str(FEATURES_DIR / "train_final.parquet"))
        total = df.count()
        exclude = {"SK_ID_CURR", "TARGET"}
        feature_cols = [c for c in df.columns if c not in exclude]
        all_null = [c for c in feature_cols if df.filter(F.col(c).isNull()).count() == total]
        assert len(all_null) == 0, f"All-null features: {all_null}"

    def test_train_test_column_alignment(self, check_features_exist):
        spark = get_spark_session()
        train = spark.read.parquet(str(FEATURES_DIR / "train_final.parquet"))
        test = spark.read.parquet(str(FEATURES_DIR / "test_final.parquet"))
        train_feats = set(train.columns) - {"TARGET"}
        test_feats = set(test.columns)
        assert train_feats == test_feats, (
            f"Mismatch: train-only={train_feats - test_feats}, test-only={test_feats - train_feats}"
        )

    def test_feature_count_reasonable(self, check_features_exist):
        spark = get_spark_session()
        df = spark.read.parquet(str(FEATURES_DIR / "train_final.parquet"))
        n_features = len([c for c in df.columns if c not in {"SK_ID_CURR", "TARGET"}])
        assert n_features >= 100, f"Only {n_features} features, expected 100+"
        assert n_features < 1000, f"Too many features ({n_features}), possible duplication"

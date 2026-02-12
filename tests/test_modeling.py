"""Tests for modeling modules."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from pyspark.sql import functions as F

from home_credit.utils import FEATURES_DIR, OUTPUT_DIR, get_spark_session


@pytest.fixture(scope="module")
def check_features_exist():
    if not (FEATURES_DIR / "train_final.parquet").exists():
        pytest.skip("Features not built")


@pytest.fixture(scope="module")
def small_train(check_features_exist):
    """Load a small sample for fast tests."""
    spark = get_spark_session()
    df = spark.read.parquet(str(FEATURES_DIR / "train_final.parquet"))
    return df.orderBy(F.rand(seed=42)).limit(1000)


class TestTraining:
    def test_prepare_data(self, small_train):
        from home_credit.features.pipeline import get_feature_columns
        from home_credit.modeling.train import prepare_data

        feature_cols = get_feature_columns(small_train)
        X, y, numeric_cols = prepare_data(small_train, feature_cols)
        assert X.shape[0] == 1000
        assert X.shape[1] > 50
        assert y.shape[0] == 1000
        assert set(np.unique(y)) == {0.0, 1.0}

    def test_train_small_sample(self, small_train):
        from home_credit.modeling.train import train_cv

        result = train_cv(
            small_train,
            params={
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 3,
                "n_estimators": 10,
                "learning_rate": 0.3,
                "random_state": 42,
                "n_jobs": 1,
                "verbosity": 0,
            },
            n_folds=2,
            experiment_name="test_run",
        )
        assert len(result["cv_scores"]) == 2
        assert 0.0 < result["mean_auc"] < 1.0
        assert len(result["models"]) == 2
        assert len(result["feature_names"]) > 0


class TestSubmission:
    def test_submission_format(self):
        sub_path = OUTPUT_DIR / "submission.csv"
        if not sub_path.exists():
            pytest.skip("No submission file generated yet")

        df = pd.read_csv(sub_path)
        assert list(df.columns) == ["SK_ID_CURR", "TARGET"]
        assert len(df) > 40_000
        assert df["TARGET"].min() >= 0
        assert df["TARGET"].max() <= 1
        assert df["SK_ID_CURR"].nunique() == len(df), "Duplicate SK_ID_CURR"

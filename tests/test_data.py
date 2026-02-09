"""Tests for data download and loading."""

import polars as pl
import pytest

from home_credit.data.download import EXPECTED_FILES
from home_credit.data.loader import load_table, scan_table
from home_credit.utils import DATA_DIR


@pytest.fixture(scope="module")
def check_data_exists():
    """Skip all tests in this module if data hasn't been downloaded."""
    if not (DATA_DIR / "application_train.csv").exists():
        pytest.skip("Data not downloaded — run `python -m home_credit.data.download` first")


class TestDataLoading:
    def test_data_files_exist(self, check_data_exists):
        for f in EXPECTED_FILES:
            assert (DATA_DIR / f).exists(), f"Missing: {f}"

    def test_scan_application_train(self, check_data_exists):
        lf = scan_table("application_train")
        assert isinstance(lf, pl.LazyFrame)
        df = lf.select("SK_ID_CURR", "TARGET").collect()
        assert df.shape[0] > 300_000
        assert "SK_ID_CURR" in df.columns
        assert "TARGET" in df.columns

    def test_scan_application_test(self, check_data_exists):
        lf = scan_table("application_test")
        df = lf.select("SK_ID_CURR").collect()
        assert df.shape[0] > 40_000

    def test_load_table_returns_dataframe(self, check_data_exists):
        df = load_table("sample_submission")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] > 0

    def test_load_optimizes_dtypes(self, check_data_exists):
        df = load_table("application_train")
        # TARGET should be downcast from Int64 to Int8 (values 0 and 1)
        assert df["TARGET"].dtype in (pl.Int8, pl.Int16, pl.Int32)

    def test_scan_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            scan_table("nonexistent_table")

    def test_bureau_has_join_keys(self, check_data_exists):
        df = scan_table("bureau").select("SK_ID_CURR", "SK_ID_BUREAU").collect()
        assert "SK_ID_CURR" in df.columns
        assert "SK_ID_BUREAU" in df.columns

    def test_bureau_balance_has_join_key(self, check_data_exists):
        df = scan_table("bureau_balance").select("SK_ID_BUREAU").collect()
        assert "SK_ID_BUREAU" in df.columns

    def test_previous_application_has_join_keys(self, check_data_exists):
        df = scan_table("previous_application").select("SK_ID_CURR", "SK_ID_PREV").collect()
        assert "SK_ID_CURR" in df.columns
        assert "SK_ID_PREV" in df.columns

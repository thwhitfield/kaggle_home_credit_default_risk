"""Load CSVs with PySpark, handle dtypes, reduce memory."""

from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
)

from home_credit.utils import DATA_DIR, get_logger, get_spark_session

log = get_logger(__name__)


def load_table(name: str, data_dir: Path = DATA_DIR) -> DataFrame:
    """Load a CSV table as a Spark DataFrame with optimized dtypes."""
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    spark = get_spark_session()
    log.info(f"Loading {name}")
    df = spark.read.csv(str(path), header=True, inferSchema=True)
    df = _optimize_dtypes(df)
    row_count = df.count()
    n_cols = len(df.columns)
    log.info(f"  {name}: {row_count:,} rows x {n_cols} cols")
    return df


def _optimize_dtypes(df: DataFrame) -> DataFrame:
    """Downcast numeric columns to save memory."""
    for col_name, dtype in df.dtypes:
        if dtype == "double":
            df = df.withColumn(col_name, F.col(col_name).cast(FloatType()))
        elif dtype == "bigint":
            # Spark doesn't easily let us inspect min/max without an action,
            # so we cast to IntegerType which covers most cases for this dataset
            df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
    return df


# Convenience loaders for each table
def load_application_train(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("application_train", data_dir)


def load_application_test(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("application_test", data_dir)


def load_bureau(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("bureau", data_dir)


def load_bureau_balance(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("bureau_balance", data_dir)


def load_previous_application(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("previous_application", data_dir)


def load_installments_payments(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("installments_payments", data_dir)


def load_pos_cash_balance(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("POS_CASH_balance", data_dir)


def load_credit_card_balance(data_dir: Path = DATA_DIR) -> DataFrame:
    return load_table("credit_card_balance", data_dir)

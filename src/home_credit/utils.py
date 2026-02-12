"""Shared helpers: logging, timing, conversion utilities."""

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FEATURES_DIR = DATA_DIR / "features"

_spark_session = None


def get_spark_session() -> SparkSession:
    """Get or create a SparkSession. Configurable via environment variables."""
    global _spark_session
    if _spark_session is not None:
        return _spark_session

    master = os.environ.get("SPARK_MASTER", "local[*]")
    app_name = os.environ.get("SPARK_APP_NAME", "home_credit")
    driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "8g")

    _spark_session = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    # Reduce Spark's verbose logging
    _spark_session.sparkContext.setLogLevel("WARN")
    return _spark_session


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timer(description: str, logger: logging.Logger | None = None):
    """Context manager that logs elapsed time."""
    log = logger or get_logger("timer")
    log.info(f"Starting: {description}")
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    log.info(f"Finished: {description} ({elapsed:.1f}s)")


def ensure_dirs():
    """Create output and feature directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def to_pandas_for_model(df: DataFrame) -> "pd.DataFrame":
    """Convert Spark DataFrame to pandas, suitable for sklearn/xgboost."""
    return df.toPandas()


def to_numpy_for_model(df: DataFrame) -> "np.ndarray":
    """Convert Spark DataFrame to numpy array."""
    return df.toPandas().to_numpy()

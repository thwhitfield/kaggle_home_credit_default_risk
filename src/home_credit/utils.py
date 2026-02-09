"""Shared helpers: logging, timing, conversion utilities."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FEATURES_DIR = DATA_DIR / "features"


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


def to_pandas_for_model(df: pl.DataFrame) -> "pd.DataFrame":
    """Convert Polars DataFrame to pandas, suitable for sklearn/xgboost."""
    return df.to_pandas()


def to_numpy_for_model(df: pl.DataFrame) -> "np.ndarray":
    """Convert Polars DataFrame to numpy array."""
    return df.to_numpy()

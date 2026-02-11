"""Load CSVs with Polars, handle dtypes, reduce memory."""

from pathlib import Path

import polars as pl

from home_credit.utils import DATA_DIR, get_logger

log = get_logger(__name__)


def scan_table(name: str, data_dir: Path = DATA_DIR) -> pl.LazyFrame:
    """Scan a CSV lazily. Use .collect() only when needed."""
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pl.scan_csv(path, try_parse_dates=True)


def load_table(name: str, data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """Load a table eagerly with optimized dtypes."""
    log.info(f"Loading {name}")
    df = scan_table(name, data_dir).collect()
    df = _optimize_dtypes(df)
    log.info(f"  {name}: {df.shape[0]:,} rows x {df.shape[1]} cols, {df.estimated_size('mb'):.1f} MB")
    return df


def _optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast numeric columns to save memory."""
    exprs = []
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if dtype == pl.Float64:
            exprs.append(pl.col(col_name).cast(pl.Float32))
        elif dtype == pl.Int64:
            series = df[col_name]
            min_val = series.min()
            max_val = series.max()
            if min_val is not None and max_val is not None:
                if min_val >= -128 and max_val <= 127:
                    exprs.append(pl.col(col_name).cast(pl.Int8))
                elif min_val >= -32768 and max_val <= 32767:
                    exprs.append(pl.col(col_name).cast(pl.Int16))
                elif min_val >= -(2**31) and max_val <= 2**31 - 1:
                    exprs.append(pl.col(col_name).cast(pl.Int32))
                else:
                    exprs.append(pl.col(col_name))
            else:
                exprs.append(pl.col(col_name))
        elif dtype == pl.String:
            n_unique = df[col_name].n_unique()
            n_total = df[col_name].len()
            if n_unique < n_total * 0.5:
                exprs.append(pl.col(col_name).cast(pl.Categorical))
            else:
                exprs.append(pl.col(col_name))
        else:
            exprs.append(pl.col(col_name))

    if exprs:
        df = df.select(exprs)
    return df


# Convenience loaders for each table
def load_application_train(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("application_train", data_dir)


def load_application_test(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("application_test", data_dir)


def load_bureau(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("bureau", data_dir)


def load_bureau_balance(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("bureau_balance", data_dir)


def load_previous_application(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("previous_application", data_dir)


def load_installments_payments(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("installments_payments", data_dir)


def load_pos_cash_balance(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("POS_CASH_balance", data_dir)


def load_credit_card_balance(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    return load_table("credit_card_balance", data_dir)

"""Ratio and interaction features between key monetary columns and top correlated features.

Creates:
- Monetary ratios (credit/income, annuity/income, payment/balance, etc.)
- Products and divisions between top correlated numeric features
  (EXT_SOURCE_*, AMT_*, DAYS_* columns)

These features are computed on the joined dataset (after all tables are joined),
since they need columns from both application and supplementary tables.
"""

import polars as pl

from home_credit.utils import get_logger

log = get_logger(__name__)

# --- Monetary ratio definitions ---
# (numerator, denominator, name) — denominator gets +1 to avoid division by zero
MONETARY_RATIOS = [
    # Credit / balance ratios from supplementary tables
    ("BUR_AMT_CREDIT_TOTAL", "AMT_INCOME_TOTAL", "RATIO_BUR_CREDIT_TO_INCOME"),
    ("BUR_DEBT_TOTAL", "AMT_INCOME_TOTAL", "RATIO_BUR_DEBT_TO_INCOME"),
    ("BUR_DEBT_TOTAL", "BUR_AMT_CREDIT_TOTAL", "RATIO_BUR_DEBT_TO_CREDIT"),
    ("BUR_ACTIVE_ANNUITY_TOTAL", "AMT_INCOME_TOTAL", "RATIO_BUR_ANNUITY_TO_INCOME"),
    ("BUR_ACTIVE_DEBT_TOTAL", "BUR_ACTIVE_ANNUITY_TOTAL", "RATIO_BUR_ACTIVE_DEBT_TO_ANNUITY"),

    # Previous application amounts vs current
    ("PREV_AMT_APPLICATION_MEAN", "AMT_CREDIT", "RATIO_PREV_APP_AMT_TO_CREDIT"),
    ("PREV_APPROVED_CREDIT_MEAN", "AMT_INCOME_TOTAL", "RATIO_PREV_APPROVED_TO_INCOME"),
    ("PREV_APPROVED_CREDIT_MAX", "AMT_CREDIT", "RATIO_PREV_MAX_APPROVED_TO_CREDIT"),

    # Installments payment vs expected
    ("INS_TOTAL_PAID", "INS_TOTAL_EXPECTED", "RATIO_INS_PAID_TO_EXPECTED"),
    ("INS_UNDERPAYMENT_MEAN", "AMT_CREDIT", "RATIO_INS_UNDERPAID_TO_CREDIT"),

    # Credit card balance/limit
    ("CC_BALANCE_MEAN", "AMT_INCOME_TOTAL", "RATIO_CC_BALANCE_TO_INCOME"),
    ("CC_ATM_DRAWINGS_TOTAL", "AMT_INCOME_TOTAL", "RATIO_CC_ATM_TO_INCOME"),

    # Annuity burden relative to different measures
    ("AMT_ANNUITY", "APP_INCOME_AFTER_ANNUITY", "RATIO_ANNUITY_TO_REMAINING_INCOME"),
    ("AMT_ANNUITY", "AMT_GOODS_PRICE", "RATIO_ANNUITY_TO_GOODS"),

    # Payment behavior ratios
    ("INS_LATE_30DAYS_COUNT", "INS_COUNT", "RATIO_SEVERE_LATE_TO_TOTAL_INS"),
]

# --- Top correlated feature interactions ---
# Products and divisions between features that are individually most predictive.
# Based on common Kaggle findings: EXT_SOURCE_* are by far the most important,
# followed by DAYS_BIRTH, DAYS_EMPLOYED, AMT_* columns, and bureau aggregates.
INTERACTION_PAIRS = [
    # EXT_SOURCE crossed with monetary
    ("EXT_SOURCE_2", "AMT_CREDIT", "div", "INTER_EXT2_DIV_CREDIT"),
    ("EXT_SOURCE_3", "AMT_CREDIT", "div", "INTER_EXT3_DIV_CREDIT"),
    ("EXT_SOURCE_2", "AMT_ANNUITY", "div", "INTER_EXT2_DIV_ANNUITY"),
    ("EXT_SOURCE_3", "AMT_ANNUITY", "div", "INTER_EXT3_DIV_ANNUITY"),
    ("EXT_SOURCE_2", "AMT_INCOME_TOTAL", "mul", "INTER_EXT2_x_INCOME"),
    ("EXT_SOURCE_3", "AMT_INCOME_TOTAL", "mul", "INTER_EXT3_x_INCOME"),

    # EXT_SOURCE crossed with time
    ("EXT_SOURCE_2", "DAYS_EMPLOYED", "mul", "INTER_EXT2_x_EMPLOYED"),
    ("EXT_SOURCE_3", "DAYS_EMPLOYED", "mul", "INTER_EXT3_x_EMPLOYED"),
    ("EXT_SOURCE_1", "DAYS_BIRTH", "mul", "INTER_EXT1_x_BIRTH"),

    # EXT_SOURCE crossed with bureau metrics
    ("EXT_SOURCE_2", "BUR_OVERDUE_RATE", "mul", "INTER_EXT2_x_OVERDUE_RATE"),
    ("EXT_SOURCE_3", "INS_LATE_RATE", "mul", "INTER_EXT3_x_LATE_RATE"),
    ("EXT_SOURCE_2", "BUR_DEBT_TOTAL", "div", "INTER_EXT2_DIV_BUR_DEBT"),

    # Age × monetary
    ("DAYS_BIRTH", "AMT_CREDIT", "div", "INTER_AGE_DIV_CREDIT"),
    ("DAYS_BIRTH", "AMT_ANNUITY", "div", "INTER_AGE_DIV_ANNUITY"),
    ("DAYS_BIRTH", "AMT_INCOME_TOTAL", "div", "INTER_AGE_DIV_INCOME"),

    # Employment × monetary
    ("DAYS_EMPLOYED", "AMT_CREDIT", "div", "INTER_EMPLOYED_DIV_CREDIT"),
    ("DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "div", "INTER_EMPLOYED_DIV_INCOME"),

    # Cross-table interaction products
    ("INS_LATE_RATE", "BUR_OVERDUE_RATE", "mul", "INTER_INS_LATE_x_BUR_OVERDUE"),
    ("INS_LATE_RATE", "POS_DPD_RATE", "mul", "INTER_INS_LATE_x_POS_DPD"),
    ("CC_UTILIZATION_MEAN", "BUR_DEBT_TO_CREDIT_RATIO", "mul", "INTER_CC_UTIL_x_BUR_DTI"),

    # EXT_SOURCE min/max (captures worst/best score)
    ("EXT_SOURCE_2", "EXT_SOURCE_3", "min", "INTER_EXT_SOURCE_23_MIN"),
    ("EXT_SOURCE_2", "EXT_SOURCE_3", "max", "INTER_EXT_SOURCE_23_MAX"),
]


def build_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add ratio and interaction features to a joined DataFrame.

    Operates on the full joined dataset (application + all supplementary tables).
    Only creates features where both columns exist in the DataFrame.
    """
    log.info("Building ratio and interaction features")
    exprs = []

    # --- Monetary ratios ---
    for num, den, name in MONETARY_RATIOS:
        if num in df.columns and den in df.columns:
            exprs.append(
                (pl.col(num).fill_null(0) / (pl.col(den).fill_null(0) + 1))
                .alias(name)
            )

    n_ratios = len(exprs)
    log.info(f"  Monetary ratios: {n_ratios}")

    # --- Feature interactions (products, divisions, min, max) ---
    n_interactions = 0
    for col_a, col_b, op, name in INTERACTION_PAIRS:
        if col_a not in df.columns or col_b not in df.columns:
            continue

        a = pl.col(col_a).fill_null(0)
        b = pl.col(col_b).fill_null(0)

        if op == "mul":
            exprs.append((a * b).alias(name))
        elif op == "div":
            exprs.append((a / (b + 1)).alias(name))
        elif op == "min":
            exprs.append(pl.min_horizontal(a, b).alias(name))
        elif op == "max":
            exprs.append(pl.max_horizontal(a, b).alias(name))

        n_interactions += 1

    log.info(f"  Feature interactions: {n_interactions}")
    log.info(f"  Total new ratio/interaction features: {len(exprs)}")

    if exprs:
        df = df.with_columns(exprs)

    return df

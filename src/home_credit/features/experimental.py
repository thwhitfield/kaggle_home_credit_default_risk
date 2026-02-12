"""Experimental feature hypotheses for feature discovery screening.

Each feature is prefixed with its hypothesis tag (H1_, H2_, etc.)
so we can easily group and ablate by hypothesis during screening.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from home_credit.utils import get_logger

log = get_logger(__name__)

HYPOTHESIS_PREFIX = {
    "H1": "Per-loan behavioral variance",
    "H2": "Behavioral trajectory / slope",
    "H3": "Time since last bad event",
    "H4": "Untapped raw columns",
    "H5": "Per-credit-type bureau aggregations",
    "H6": "Application frequency / velocity",
}


# ---------------------------------------------------------------------------
# H1: Per-Loan Behavioral Variance
# ---------------------------------------------------------------------------

def build_h1_features(
    installments: DataFrame,
    cc: DataFrame,
    pos: DataFrame,
) -> dict[str, DataFrame]:
    """Per-loan stats aggregated across loans to capture behavioral consistency."""
    log.info("  H1: Building per-loan variance features")
    result = {}

    # --- Installments: per-loan late rate / payment ratio ---
    ins_enriched = installments.select(
        "*",
        (F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT")).alias("_days_diff"),
        (F.col("AMT_PAYMENT") / (F.col("AMT_INSTALMENT") + 1)).alias("_pay_ratio"),
    )
    ins_per_loan = ins_enriched.groupBy("SK_ID_CURR", "SK_ID_PREV").agg(
        F.avg(F.when(F.col("_days_diff") > 0, 1.0).otherwise(0.0)).alias("loan_late_rate"),
        F.avg("_pay_ratio").alias("loan_pay_ratio"),
        F.avg("_days_diff").alias("loan_days_diff_mean"),
    )
    ins_variance = ins_per_loan.groupBy("SK_ID_CURR").agg(
        F.stddev("loan_late_rate").alias("H1_INS_PERLOAN_LATE_RATE_STD"),
        F.max("loan_late_rate").alias("H1_INS_WORST_LOAN_LATE_RATE"),
        F.min("loan_late_rate").alias("H1_INS_BEST_LOAN_LATE_RATE"),
        (F.max("loan_late_rate") - F.min("loan_late_rate"))
            .alias("H1_INS_LOAN_LATE_RATE_SPREAD"),
        F.stddev("loan_pay_ratio").alias("H1_INS_PERLOAN_PAY_RATIO_STD"),
        F.min("loan_pay_ratio").alias("H1_INS_WORST_LOAN_PAY_RATIO"),
        F.stddev("loan_days_diff_mean").alias("H1_INS_PERLOAN_DAYS_DIFF_STD"),
    )
    result["ins"] = ins_variance

    # --- Credit card: per-card utilization / DPD ---
    cc_enriched = cc.select(
        "*",
        (F.col("AMT_BALANCE") / (F.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_util"),
    )
    cc_per_card = cc_enriched.groupBy("SK_ID_CURR", "SK_ID_PREV").agg(
        F.avg("_util").alias("card_util_mean"),
        F.max("_util").alias("card_util_max"),
        F.avg(F.when(F.col("SK_DPD") > 0, 1.0).otherwise(0.0)).alias("card_dpd_rate"),
    )
    cc_variance = cc_per_card.groupBy("SK_ID_CURR").agg(
        F.stddev("card_util_mean").alias("H1_CC_PERLOAN_UTIL_STD"),
        F.max("card_util_mean").alias("H1_CC_WORST_CARD_UTIL"),
        (F.max("card_util_mean") - F.min("card_util_mean"))
            .alias("H1_CC_CARD_UTIL_SPREAD"),
        F.max("card_util_max").alias("H1_CC_WORST_CARD_PEAK_UTIL"),
        F.stddev("card_dpd_rate").alias("H1_CC_PERLOAN_DPD_RATE_STD"),
        F.max("card_dpd_rate").alias("H1_CC_WORST_CARD_DPD_RATE"),
    )
    result["cc"] = cc_variance

    # --- POS/cash: per-loan DPD ---
    pos_per_loan = pos.groupBy("SK_ID_CURR", "SK_ID_PREV").agg(
        F.avg(F.when(F.col("SK_DPD") > 0, 1.0).otherwise(0.0)).alias("loan_dpd_rate"),
        F.max("SK_DPD").alias("loan_dpd_max"),
    )
    pos_variance = pos_per_loan.groupBy("SK_ID_CURR").agg(
        F.stddev("loan_dpd_rate").alias("H1_POS_PERLOAN_DPD_RATE_STD"),
        F.max("loan_dpd_rate").alias("H1_POS_WORST_LOAN_DPD_RATE"),
        (F.max("loan_dpd_rate") - F.min("loan_dpd_rate"))
            .alias("H1_POS_LOAN_DPD_SPREAD"),
        F.max("loan_dpd_max").alias("H1_POS_WORST_LOAN_DPD_MAX"),
        F.stddev("loan_dpd_max").alias("H1_POS_PERLOAN_DPD_MAX_STD"),
    )
    result["pos"] = pos_variance

    return result


# ---------------------------------------------------------------------------
# H2: Behavioral Trajectory / Slope
# ---------------------------------------------------------------------------

def _split_half_trajectory(
    df: DataFrame,
    id_col: str,
    time_col: str,
    min_records: int = 4,
) -> tuple[DataFrame, DataFrame]:
    """Split each group into older half and recent half by time rank.

    Returns (older_half, recent_half) DataFrames.
    Only includes groups with >= min_records rows.
    """
    w = Window.partitionBy(id_col).orderBy(time_col)
    w_count = Window.partitionBy(id_col)
    ranked = df.withColumn("_rank", F.row_number().over(w)) \
               .withColumn("_total", F.count("*").over(w_count)) \
               .filter(F.col("_total") >= min_records)

    older = ranked.filter(F.col("_rank") <= F.col("_total") / 2)
    recent = ranked.filter(F.col("_rank") > F.col("_total") / 2)
    return older, recent


def build_h2_features(
    installments: DataFrame,
    bureau_balance: DataFrame,
    bureau: DataFrame,
    pos: DataFrame,
    cc: DataFrame,
) -> dict[str, DataFrame]:
    """Behavioral trajectory: recent half vs older half of history."""
    log.info("  H2: Building trajectory features")
    result = {}

    # --- Installments trajectory ---
    ins_enriched = installments.select(
        "*",
        (F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT")).alias("_days_diff"),
        (F.col("AMT_PAYMENT") / (F.col("AMT_INSTALMENT") + 1)).alias("_pay_ratio"),
    )
    ins_older, ins_recent = _split_half_trajectory(
        ins_enriched, "SK_ID_CURR", "DAYS_INSTALMENT"
    )
    ins_older_stats = ins_older.groupBy("SK_ID_CURR").agg(
        F.avg(F.when(F.col("_days_diff") > 0, 1.0).otherwise(0.0)).alias("older_late_rate"),
        F.avg("_pay_ratio").alias("older_pay_ratio"),
    )
    ins_recent_stats = ins_recent.groupBy("SK_ID_CURR").agg(
        F.avg(F.when(F.col("_days_diff") > 0, 1.0).otherwise(0.0)).alias("recent_late_rate"),
        F.avg("_pay_ratio").alias("recent_pay_ratio"),
    )
    ins_traj = ins_recent_stats.join(ins_older_stats, "SK_ID_CURR", "inner")
    ins_traj = ins_traj.select(
        "SK_ID_CURR",
        (F.col("recent_late_rate") - F.col("older_late_rate"))
            .alias("H2_INS_LATE_RATE_TRAJECTORY"),
        (F.col("recent_pay_ratio") - F.col("older_pay_ratio"))
            .alias("H2_INS_PAY_RATIO_TRAJECTORY"),
        (
            (F.col("recent_late_rate") > F.col("older_late_rate") * 2)
            & (F.col("recent_late_rate") > 0.1)
        ).cast("int").alias("H2_INS_WORSENING_FLAG"),
    )
    result["ins"] = ins_traj

    # --- Bureau balance trajectory ---
    # Join bureau_balance with bureau to get SK_ID_CURR
    bb_with_curr = bureau_balance.join(
        bureau.select("SK_ID_BUREAU", "SK_ID_CURR"), "SK_ID_BUREAU", "left"
    )
    bb_with_curr = bb_with_curr.withColumn("STATUS", F.col("STATUS").cast("string"))
    bb_older, bb_recent = _split_half_trajectory(
        bb_with_curr, "SK_ID_CURR", "MONTHS_BALANCE"
    )
    bb_older_stats = bb_older.groupBy("SK_ID_CURR").agg(
        F.avg(F.when(F.col("STATUS").isin("1", "2", "3", "4", "5"), 1.0).otherwise(0.0))
            .alias("older_dpd_rate"),
        F.avg(F.when(F.col("STATUS") == "C", 1.0).otherwise(0.0))
            .alias("older_closed_rate"),
    )
    bb_recent_stats = bb_recent.groupBy("SK_ID_CURR").agg(
        F.avg(F.when(F.col("STATUS").isin("1", "2", "3", "4", "5"), 1.0).otherwise(0.0))
            .alias("recent_dpd_rate"),
        F.avg(F.when(F.col("STATUS") == "C", 1.0).otherwise(0.0))
            .alias("recent_closed_rate"),
    )
    bb_traj = bb_recent_stats.join(bb_older_stats, "SK_ID_CURR", "inner")
    bb_traj = bb_traj.select(
        "SK_ID_CURR",
        (F.col("recent_dpd_rate") - F.col("older_dpd_rate"))
            .alias("H2_BUR_BB_DPD_TRAJECTORY"),
        (F.col("recent_closed_rate") - F.col("older_closed_rate"))
            .alias("H2_BUR_BB_CLOSED_TRAJECTORY"),
        (
            (F.col("recent_dpd_rate") > F.col("older_dpd_rate") * 2)
            & (F.col("recent_dpd_rate") > 0.05)
        ).cast("int").alias("H2_BUR_BB_WORSENING_FLAG"),
    )
    result["bb"] = bb_traj

    # --- POS/cash trajectory ---
    pos_older, pos_recent = _split_half_trajectory(
        pos, "SK_ID_CURR", "MONTHS_BALANCE"
    )
    pos_older_stats = pos_older.groupBy("SK_ID_CURR").agg(
        F.avg(F.when(F.col("SK_DPD") > 0, 1.0).otherwise(0.0)).alias("older_dpd_rate"),
        F.max("SK_DPD").alias("older_dpd_max"),
    )
    pos_recent_stats = pos_recent.groupBy("SK_ID_CURR").agg(
        F.avg(F.when(F.col("SK_DPD") > 0, 1.0).otherwise(0.0)).alias("recent_dpd_rate"),
        F.max("SK_DPD").alias("recent_dpd_max"),
    )
    pos_traj = pos_recent_stats.join(pos_older_stats, "SK_ID_CURR", "inner")
    pos_traj = pos_traj.select(
        "SK_ID_CURR",
        (F.col("recent_dpd_rate") - F.col("older_dpd_rate"))
            .alias("H2_POS_DPD_TRAJECTORY"),
        (F.col("recent_dpd_max") - F.col("older_dpd_max"))
            .alias("H2_POS_DPD_MAX_TRAJECTORY"),
        (
            (F.col("recent_dpd_rate") > F.col("older_dpd_rate") * 2)
            & (F.col("recent_dpd_rate") > 0.05)
        ).cast("int").alias("H2_POS_WORSENING_FLAG"),
    )
    result["pos"] = pos_traj

    # --- Credit card trajectory ---
    cc_enriched = cc.select(
        "*",
        (F.col("AMT_BALANCE") / (F.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_util"),
        (F.col("AMT_DRAWINGS_CURRENT") / (F.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_draw_ratio"),
    )
    cc_older, cc_recent = _split_half_trajectory(
        cc_enriched, "SK_ID_CURR", "MONTHS_BALANCE"
    )
    cc_older_stats = cc_older.groupBy("SK_ID_CURR").agg(
        F.avg("_util").alias("older_util"),
        F.avg(F.when(F.col("SK_DPD") > 0, 1.0).otherwise(0.0)).alias("older_dpd_rate"),
        F.avg("_draw_ratio").alias("older_draw_ratio"),
    )
    cc_recent_stats = cc_recent.groupBy("SK_ID_CURR").agg(
        F.avg("_util").alias("recent_util"),
        F.avg(F.when(F.col("SK_DPD") > 0, 1.0).otherwise(0.0)).alias("recent_dpd_rate"),
        F.avg("_draw_ratio").alias("recent_draw_ratio"),
    )
    cc_traj = cc_recent_stats.join(cc_older_stats, "SK_ID_CURR", "inner")
    cc_traj = cc_traj.select(
        "SK_ID_CURR",
        (F.col("recent_util") - F.col("older_util"))
            .alias("H2_CC_UTIL_TRAJECTORY"),
        (F.col("recent_dpd_rate") - F.col("older_dpd_rate"))
            .alias("H2_CC_DPD_TRAJECTORY"),
        (F.col("recent_draw_ratio") - F.col("older_draw_ratio"))
            .alias("H2_CC_DRAWING_TRAJECTORY"),
    )
    result["cc"] = cc_traj

    return result


# ---------------------------------------------------------------------------
# H3: Time-Since-Last-Bad-Event
# ---------------------------------------------------------------------------

def build_h3_features(
    installments: DataFrame,
    bureau: DataFrame,
    bureau_balance: DataFrame,
    pos: DataFrame,
    cc: DataFrame,
) -> dict[str, DataFrame]:
    """Recency of bad events (null = never had a bad event)."""
    log.info("  H3: Building time-since-last-event features")
    result = {}

    # --- Installments: days since last late / severe late ---
    ins_enriched = installments.select(
        "*",
        (F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT")).alias("_days_diff"),
    )
    ins_last_late = (
        ins_enriched.filter(F.col("_days_diff") > 0)
        .groupBy("SK_ID_CURR")
        .agg(
            # max DAYS_INSTALMENT = closest to 0 = most recent
            F.max("DAYS_INSTALMENT").alias("H3_DAYS_SINCE_LAST_LATE_INS"),
        )
    )
    ins_last_severe = (
        ins_enriched.filter(F.col("_days_diff") > 30)
        .groupBy("SK_ID_CURR")
        .agg(
            F.max("DAYS_INSTALMENT").alias("H3_DAYS_SINCE_LAST_SEVERE_LATE_INS"),
        )
    )
    ins_h3 = ins_last_late.join(ins_last_severe, "SK_ID_CURR", "full")
    result["ins"] = ins_h3

    # --- Bureau: days since last overdue ---
    bur_last_overdue = (
        bureau.filter(F.col("CREDIT_DAY_OVERDUE") > 0)
        .groupBy("SK_ID_CURR")
        .agg(
            F.max("DAYS_CREDIT").alias("H3_DAYS_SINCE_LAST_BUREAU_OVERDUE"),
        )
    )
    # Bureau balance: months since last DPD
    bb_with_curr = bureau_balance.join(
        bureau.select("SK_ID_BUREAU", "SK_ID_CURR"), "SK_ID_BUREAU", "left"
    )
    bb_with_curr = bb_with_curr.withColumn("STATUS", F.col("STATUS").cast("string"))
    bb_dpd = bb_with_curr.filter(
        F.col("STATUS").isin("1", "2", "3", "4", "5")
    )
    bb_last_dpd = (
        bb_dpd.groupBy("SK_ID_CURR")
        .agg(
            F.max("MONTHS_BALANCE").alias("H3_MONTHS_SINCE_LAST_BB_DPD"),
        )
    )
    bur_h3 = bur_last_overdue.join(bb_last_dpd, "SK_ID_CURR", "full")
    result["bur"] = bur_h3

    # --- POS/cash: months since last DPD ---
    pos_last_dpd = (
        pos.filter(F.col("SK_DPD") > 0)
        .groupBy("SK_ID_CURR")
        .agg(
            F.max("MONTHS_BALANCE").alias("H3_MONTHS_SINCE_LAST_POS_DPD"),
        )
    )
    pos_last_dpd_def = (
        pos.filter(F.col("SK_DPD_DEF") > 0)
        .groupBy("SK_ID_CURR")
        .agg(
            F.max("MONTHS_BALANCE").alias("H3_MONTHS_SINCE_LAST_POS_DPD_DEF"),
        )
    )
    pos_h3 = pos_last_dpd.join(pos_last_dpd_def, "SK_ID_CURR", "full")
    result["pos"] = pos_h3

    # --- Credit card: months since last DPD / high util ---
    cc_last_dpd = (
        cc.filter(F.col("SK_DPD") > 0)
        .groupBy("SK_ID_CURR")
        .agg(
            F.max("MONTHS_BALANCE").alias("H3_MONTHS_SINCE_LAST_CC_DPD"),
        )
    )
    cc_enriched = cc.select(
        "*",
        (F.col("AMT_BALANCE") / (F.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_util"),
    )
    cc_last_high_util = (
        cc_enriched.filter(F.col("_util") > 0.9)
        .groupBy("SK_ID_CURR")
        .agg(
            F.max("MONTHS_BALANCE").alias("H3_MONTHS_SINCE_LAST_HIGH_CC_UTIL"),
        )
    )
    cc_h3 = cc_last_dpd.join(cc_last_high_util, "SK_ID_CURR", "full")
    result["cc"] = cc_h3

    return result


# ---------------------------------------------------------------------------
# H4: Untapped Raw Columns
# ---------------------------------------------------------------------------

def build_h4_features(
    bureau: DataFrame,
    prev: DataFrame,
) -> dict[str, DataFrame]:
    """Features from raw columns that are currently completely unused."""
    log.info("  H4: Building untapped raw column features")
    result = {}

    # --- Bureau: CNT_CREDIT_PROLONG and CREDIT_DAY_OVERDUE ---
    bur_untapped = bureau.groupBy("SK_ID_CURR").agg(
        F.sum("CNT_CREDIT_PROLONG").alias("H4_BUR_PROLONG_TOTAL"),
        F.max("CNT_CREDIT_PROLONG").alias("H4_BUR_PROLONG_MAX"),
        F.sum(F.when(F.col("CNT_CREDIT_PROLONG") > 0, 1).otherwise(0))
            .alias("H4_BUR_PROLONG_COUNT"),
        F.max("CREDIT_DAY_OVERDUE").alias("H4_BUR_CURRENT_OVERDUE_MAX"),
        F.sum("CREDIT_DAY_OVERDUE").alias("H4_BUR_CURRENT_OVERDUE_SUM"),
    )
    result["bur"] = bur_untapped

    # --- Previous: interest rates, client type, rejection reason, loan purpose ---
    prev_untapped = prev.groupBy("SK_ID_CURR").agg(
        # Interest rates (higher = lender assessed as riskier)
        F.avg("RATE_INTEREST_PRIMARY").alias("H4_PREV_INTEREST_RATE_MEAN"),
        F.max("RATE_INTEREST_PRIMARY").alias("H4_PREV_INTEREST_RATE_MAX"),
        # Client type: Repeater vs New
        F.avg(F.when(F.col("NAME_CLIENT_TYPE").cast("string") == "Repeater", 1.0).otherwise(0.0))
            .alias("H4_PREV_REPEATER_RATE"),
        F.avg(F.when(F.col("NAME_CLIENT_TYPE").cast("string") == "New", 1.0).otherwise(0.0))
            .alias("H4_PREV_NEW_CLIENT_RATE"),
        # Rejection reasons (only non-null for refused apps)
        F.sum(F.when(F.col("CODE_REJECT_REASON").cast("string") == "HC", 1).otherwise(0))
            .alias("H4_PREV_REJECT_HC_COUNT"),
        F.sum(F.when(F.col("CODE_REJECT_REASON").cast("string") == "SCOFR", 1).otherwise(0))
            .alias("H4_PREV_REJECT_SCOFR_COUNT"),
        # Loan purpose diversity
        F.countDistinct("NAME_CASH_LOAN_PURPOSE")
            .alias("H4_PREV_LOAN_PURPOSE_DIVERSITY"),
    )
    result["prev"] = prev_untapped

    return result


# ---------------------------------------------------------------------------
# H5: Per-Credit-Type Bureau Aggregations
# ---------------------------------------------------------------------------

def build_h5_features(bureau: DataFrame) -> dict[str, DataFrame]:
    """Break down bureau financials by credit type (consumer, card, car)."""
    log.info("  H5: Building per-credit-type bureau features")

    # Cast CREDIT_TYPE to string for comparison
    bureau = bureau.withColumn("CREDIT_TYPE", F.col("CREDIT_TYPE").cast("string"))

    dfs = []
    for credit_type, prefix in [
        ("Consumer credit", "CONSUMER"),
        ("Credit card", "CARD"),
        ("Car loan", "CAR"),
    ]:
        type_df = (
            bureau.filter(F.col("CREDIT_TYPE") == credit_type)
            .groupBy("SK_ID_CURR")
            .agg(
                F.sum("AMT_CREDIT_SUM_DEBT")
                    .alias(f"H5_BUR_{prefix}_DEBT_TOTAL"),
                F.sum("AMT_CREDIT_SUM_OVERDUE")
                    .alias(f"H5_BUR_{prefix}_OVERDUE_TOTAL"),
                F.sum(F.when(F.col("CREDIT_ACTIVE").cast("string") == "Active", 1).otherwise(0))
                    .alias(f"H5_BUR_{prefix}_ACTIVE_COUNT"),
            )
        )
        dfs.append(type_df)

    # Join all type-specific DataFrames
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.join(df, "SK_ID_CURR", "full")

    return {"bur_types": combined}


# ---------------------------------------------------------------------------
# H6: Application Frequency / Velocity
# ---------------------------------------------------------------------------

def build_h6_features(
    prev: DataFrame,
    bureau: DataFrame,
) -> dict[str, DataFrame]:
    """Application frequency, acceleration, and inter-application gaps."""
    log.info("  H6: Building frequency/velocity features")
    result = {}

    # --- Previous applications: apps per year and acceleration ---
    prev_vel = prev.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("_total"),
        F.sum(F.when(F.col("DAYS_DECISION") > -365, 1).otherwise(0)).alias("_recent_1y"),
        F.max("DAYS_DECISION").alias("_most_recent"),
        F.min("DAYS_DECISION").alias("_oldest"),
    )
    prev_vel = prev_vel.select(
        "*",
        (F.abs(F.col("_most_recent") - F.col("_oldest")) / 365.25 + 1)
            .alias("_history_years"),
    )
    prev_vel = prev_vel.select(
        "SK_ID_CURR",
        (F.col("_total") / F.col("_history_years"))
            .alias("H6_PREV_APPS_PER_YEAR"),
        (F.col("_recent_1y") / (F.col("_total") / F.col("_history_years") + 0.1))
            .alias("H6_PREV_APPLICATION_ACCELERATION"),
    )

    # --- Previous applications: inter-application gap stats ---
    prev_sorted = prev.orderBy("SK_ID_CURR", "DAYS_DECISION")
    w = Window.partitionBy("SK_ID_CURR").orderBy("DAYS_DECISION")
    prev_gaps = (
        prev_sorted.withColumn(
            "_gap_days",
            F.col("DAYS_DECISION") - F.lag("DAYS_DECISION", 1).over(w)
        )
        .filter(F.col("_gap_days").isNotNull())
        .groupBy("SK_ID_CURR")
        .agg(
            F.min("_gap_days").alias("H6_PREV_MIN_APP_GAP_DAYS"),
            F.avg("_gap_days").alias("H6_PREV_MEAN_APP_GAP_DAYS"),
            F.stddev("_gap_days").alias("H6_PREV_APP_GAP_STD"),
        )
    )

    prev_h6 = prev_vel.join(prev_gaps, "SK_ID_CURR", "full")
    result["prev"] = prev_h6

    # --- Bureau: credits per year ---
    bur_vel = bureau.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("_total"),
        F.max("DAYS_CREDIT").alias("_most_recent"),
        F.min("DAYS_CREDIT").alias("_oldest"),
    )
    bur_vel = bur_vel.select(
        "*",
        (F.abs(F.col("_most_recent") - F.col("_oldest")) / 365.25 + 1)
            .alias("_history_years"),
    )
    bur_vel = bur_vel.select(
        "SK_ID_CURR",
        (F.col("_total") / F.col("_history_years"))
            .alias("H6_BUR_CREDITS_PER_YEAR"),
    )
    result["bur"] = bur_vel

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_all_experimental_features(
    installments: DataFrame,
    cc: DataFrame,
    pos: DataFrame,
    bureau: DataFrame,
    bureau_balance: DataFrame,
    prev: DataFrame,
) -> DataFrame:
    """Build all experimental features, return single DF keyed on SK_ID_CURR.

    Each feature column is prefixed with its hypothesis tag (H1_, H2_, etc.).
    """
    log.info("Building all experimental features")

    builders = [
        ("H1", build_h1_features, {
            "installments": installments, "cc": cc, "pos": pos,
        }),
        ("H2", build_h2_features, {
            "installments": installments, "bureau_balance": bureau_balance,
            "bureau": bureau, "pos": pos, "cc": cc,
        }),
        ("H3", build_h3_features, {
            "installments": installments, "bureau": bureau,
            "bureau_balance": bureau_balance, "pos": pos, "cc": cc,
        }),
        ("H4", build_h4_features, {"bureau": bureau, "prev": prev}),
        ("H5", build_h5_features, {"bureau": bureau}),
        ("H6", build_h6_features, {"prev": prev, "bureau": bureau}),
    ]

    all_dfs: list[DataFrame] = []
    for name, builder, kwargs in builders:
        log.info(f"  {name}: {HYPOTHESIS_PREFIX[name]}")
        result_dfs = builder(**kwargs)
        for df in result_dfs.values():
            all_dfs.append(df)

    # Join all on SK_ID_CURR using successive full outer joins
    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.join(df, "SK_ID_CURR", "full")

    feature_cols = [c for c in combined.columns if c != "SK_ID_CURR"]
    log.info(f"Total experimental features: {len(feature_cols)}")

    return combined


def build_winning_experimental_features(
    installments: DataFrame,
    cc: DataFrame,
    pos: DataFrame,
    bureau: DataFrame,
    bureau_balance: DataFrame,
    prev: DataFrame,
) -> DataFrame:
    """Build only the winning hypotheses (H2, H3, H4, H5) for production use.

    Returns a single DataFrame keyed on SK_ID_CURR with ~41 features.
    """
    log.info("Building winning experimental features (H2, H3, H4, H5)")

    # AMT_ANNUITY in bureau needs casting
    bureau = bureau.withColumn(
        "AMT_ANNUITY", F.col("AMT_ANNUITY").cast("string").cast("double")
    )

    builders = [
        ("H2", build_h2_features, {
            "installments": installments, "bureau_balance": bureau_balance,
            "bureau": bureau, "pos": pos, "cc": cc,
        }),
        ("H3", build_h3_features, {
            "installments": installments, "bureau": bureau,
            "bureau_balance": bureau_balance, "pos": pos, "cc": cc,
        }),
        ("H4", build_h4_features, {"bureau": bureau, "prev": prev}),
        ("H5", build_h5_features, {"bureau": bureau}),
    ]

    all_dfs: list[DataFrame] = []
    for name, builder, kwargs in builders:
        log.info(f"  {name}: {HYPOTHESIS_PREFIX[name]}")
        result_dfs = builder(**kwargs)
        for df in result_dfs.values():
            all_dfs.append(df)

    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.join(df, "SK_ID_CURR", "full")

    feature_cols = [c for c in combined.columns if c != "SK_ID_CURR"]
    log.info(f"Winning experimental features: {len(feature_cols)}")

    return combined


def get_hypothesis_features(df: DataFrame) -> dict[str, list[str]]:
    """Return mapping of hypothesis tag -> list of feature column names."""
    result = {}
    for prefix in HYPOTHESIS_PREFIX:
        cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
        if cols:
            result[prefix] = cols
    return result

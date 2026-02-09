"""Experimental feature hypotheses for feature discovery screening.

Each feature is prefixed with its hypothesis tag (H1_, H2_, etc.)
so we can easily group and ablate by hypothesis during screening.
"""

import polars as pl

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
    installments: pl.DataFrame,
    cc: pl.DataFrame,
    pos: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Per-loan stats aggregated across loans to capture behavioral consistency."""
    log.info("  H1: Building per-loan variance features")
    result = {}

    # --- Installments: per-loan late rate / payment ratio ---
    ins_enriched = installments.with_columns(
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("_days_diff"),
        (pl.col("AMT_PAYMENT") / (pl.col("AMT_INSTALMENT") + 1)).alias("_pay_ratio"),
    )
    ins_per_loan = ins_enriched.group_by(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        (pl.col("_days_diff") > 0).mean().alias("loan_late_rate"),
        pl.col("_pay_ratio").mean().alias("loan_pay_ratio"),
        pl.col("_days_diff").mean().alias("loan_days_diff_mean"),
    )
    ins_variance = ins_per_loan.group_by("SK_ID_CURR").agg(
        pl.col("loan_late_rate").std().alias("H1_INS_PERLOAN_LATE_RATE_STD"),
        pl.col("loan_late_rate").max().alias("H1_INS_WORST_LOAN_LATE_RATE"),
        pl.col("loan_late_rate").min().alias("H1_INS_BEST_LOAN_LATE_RATE"),
        (pl.col("loan_late_rate").max() - pl.col("loan_late_rate").min())
            .alias("H1_INS_LOAN_LATE_RATE_SPREAD"),
        pl.col("loan_pay_ratio").std().alias("H1_INS_PERLOAN_PAY_RATIO_STD"),
        pl.col("loan_pay_ratio").min().alias("H1_INS_WORST_LOAN_PAY_RATIO"),
        pl.col("loan_days_diff_mean").std().alias("H1_INS_PERLOAN_DAYS_DIFF_STD"),
    )
    result["ins"] = ins_variance

    # --- Credit card: per-card utilization / DPD ---
    cc_enriched = cc.with_columns(
        (pl.col("AMT_BALANCE") / (pl.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_util"),
    )
    cc_per_card = cc_enriched.group_by(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        pl.col("_util").mean().alias("card_util_mean"),
        pl.col("_util").max().alias("card_util_max"),
        (pl.col("SK_DPD") > 0).mean().alias("card_dpd_rate"),
    )
    cc_variance = cc_per_card.group_by("SK_ID_CURR").agg(
        pl.col("card_util_mean").std().alias("H1_CC_PERLOAN_UTIL_STD"),
        pl.col("card_util_mean").max().alias("H1_CC_WORST_CARD_UTIL"),
        (pl.col("card_util_mean").max() - pl.col("card_util_mean").min())
            .alias("H1_CC_CARD_UTIL_SPREAD"),
        pl.col("card_util_max").max().alias("H1_CC_WORST_CARD_PEAK_UTIL"),
        pl.col("card_dpd_rate").std().alias("H1_CC_PERLOAN_DPD_RATE_STD"),
        pl.col("card_dpd_rate").max().alias("H1_CC_WORST_CARD_DPD_RATE"),
    )
    result["cc"] = cc_variance

    # --- POS/cash: per-loan DPD ---
    pos_per_loan = pos.group_by(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        (pl.col("SK_DPD") > 0).mean().alias("loan_dpd_rate"),
        pl.col("SK_DPD").max().alias("loan_dpd_max"),
    )
    pos_variance = pos_per_loan.group_by("SK_ID_CURR").agg(
        pl.col("loan_dpd_rate").std().alias("H1_POS_PERLOAN_DPD_RATE_STD"),
        pl.col("loan_dpd_rate").max().alias("H1_POS_WORST_LOAN_DPD_RATE"),
        (pl.col("loan_dpd_rate").max() - pl.col("loan_dpd_rate").min())
            .alias("H1_POS_LOAN_DPD_SPREAD"),
        pl.col("loan_dpd_max").max().alias("H1_POS_WORST_LOAN_DPD_MAX"),
        pl.col("loan_dpd_max").std().alias("H1_POS_PERLOAN_DPD_MAX_STD"),
    )
    result["pos"] = pos_variance

    return result


# ---------------------------------------------------------------------------
# H2: Behavioral Trajectory / Slope
# ---------------------------------------------------------------------------

def _split_half_trajectory(
    df: pl.DataFrame,
    id_col: str,
    time_col: str,
    min_records: int = 4,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split each group into older half and recent half by time rank.

    Returns (older_half, recent_half) DataFrames.
    Only includes groups with >= min_records rows.
    """
    ranked = df.with_columns(
        pl.col(time_col).rank("ordinal").over(id_col).alias("_rank"),
        pl.len().over(id_col).alias("_total"),
    ).filter(pl.col("_total") >= min_records)

    older = ranked.filter(pl.col("_rank") <= pl.col("_total") / 2)
    recent = ranked.filter(pl.col("_rank") > pl.col("_total") / 2)
    return older, recent


def build_h2_features(
    installments: pl.DataFrame,
    bureau_balance: pl.DataFrame,
    bureau: pl.DataFrame,
    pos: pl.DataFrame,
    cc: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Behavioral trajectory: recent half vs older half of history."""
    log.info("  H2: Building trajectory features")
    result = {}

    # --- Installments trajectory ---
    ins_enriched = installments.with_columns(
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("_days_diff"),
        (pl.col("AMT_PAYMENT") / (pl.col("AMT_INSTALMENT") + 1)).alias("_pay_ratio"),
    )
    ins_older, ins_recent = _split_half_trajectory(
        ins_enriched, "SK_ID_CURR", "DAYS_INSTALMENT"
    )
    ins_older_stats = ins_older.group_by("SK_ID_CURR").agg(
        (pl.col("_days_diff") > 0).mean().alias("older_late_rate"),
        pl.col("_pay_ratio").mean().alias("older_pay_ratio"),
    )
    ins_recent_stats = ins_recent.group_by("SK_ID_CURR").agg(
        (pl.col("_days_diff") > 0).mean().alias("recent_late_rate"),
        pl.col("_pay_ratio").mean().alias("recent_pay_ratio"),
    )
    ins_traj = ins_recent_stats.join(ins_older_stats, on="SK_ID_CURR", how="inner")
    ins_traj = ins_traj.select(
        "SK_ID_CURR",
        (pl.col("recent_late_rate") - pl.col("older_late_rate"))
            .alias("H2_INS_LATE_RATE_TRAJECTORY"),
        (pl.col("recent_pay_ratio") - pl.col("older_pay_ratio"))
            .alias("H2_INS_PAY_RATIO_TRAJECTORY"),
        (
            (pl.col("recent_late_rate") > pl.col("older_late_rate") * 2)
            & (pl.col("recent_late_rate") > 0.1)
        ).cast(pl.Int8).alias("H2_INS_WORSENING_FLAG"),
    )
    result["ins"] = ins_traj

    # --- Bureau balance trajectory ---
    # Join bureau_balance with bureau to get SK_ID_CURR
    bb_with_curr = bureau_balance.join(
        bureau.select("SK_ID_BUREAU", "SK_ID_CURR"), on="SK_ID_BUREAU", how="left"
    )
    bb_with_curr = bb_with_curr.with_columns(pl.col("STATUS").cast(pl.String))
    bb_older, bb_recent = _split_half_trajectory(
        bb_with_curr, "SK_ID_CURR", "MONTHS_BALANCE"
    )
    bb_older_stats = bb_older.group_by("SK_ID_CURR").agg(
        (
            (pl.col("STATUS") == "1") | (pl.col("STATUS") == "2")
            | (pl.col("STATUS") == "3") | (pl.col("STATUS") == "4")
            | (pl.col("STATUS") == "5")
        ).mean().alias("older_dpd_rate"),
        (pl.col("STATUS") == "C").mean().alias("older_closed_rate"),
    )
    bb_recent_stats = bb_recent.group_by("SK_ID_CURR").agg(
        (
            (pl.col("STATUS") == "1") | (pl.col("STATUS") == "2")
            | (pl.col("STATUS") == "3") | (pl.col("STATUS") == "4")
            | (pl.col("STATUS") == "5")
        ).mean().alias("recent_dpd_rate"),
        (pl.col("STATUS") == "C").mean().alias("recent_closed_rate"),
    )
    bb_traj = bb_recent_stats.join(bb_older_stats, on="SK_ID_CURR", how="inner")
    bb_traj = bb_traj.select(
        "SK_ID_CURR",
        (pl.col("recent_dpd_rate") - pl.col("older_dpd_rate"))
            .alias("H2_BUR_BB_DPD_TRAJECTORY"),
        (pl.col("recent_closed_rate") - pl.col("older_closed_rate"))
            .alias("H2_BUR_BB_CLOSED_TRAJECTORY"),
        (
            (pl.col("recent_dpd_rate") > pl.col("older_dpd_rate") * 2)
            & (pl.col("recent_dpd_rate") > 0.05)
        ).cast(pl.Int8).alias("H2_BUR_BB_WORSENING_FLAG"),
    )
    result["bb"] = bb_traj

    # --- POS/cash trajectory ---
    pos_older, pos_recent = _split_half_trajectory(
        pos, "SK_ID_CURR", "MONTHS_BALANCE"
    )
    pos_older_stats = pos_older.group_by("SK_ID_CURR").agg(
        (pl.col("SK_DPD") > 0).mean().alias("older_dpd_rate"),
        pl.col("SK_DPD").max().alias("older_dpd_max"),
    )
    pos_recent_stats = pos_recent.group_by("SK_ID_CURR").agg(
        (pl.col("SK_DPD") > 0).mean().alias("recent_dpd_rate"),
        pl.col("SK_DPD").max().alias("recent_dpd_max"),
    )
    pos_traj = pos_recent_stats.join(pos_older_stats, on="SK_ID_CURR", how="inner")
    pos_traj = pos_traj.select(
        "SK_ID_CURR",
        (pl.col("recent_dpd_rate") - pl.col("older_dpd_rate"))
            .alias("H2_POS_DPD_TRAJECTORY"),
        (pl.col("recent_dpd_max") - pl.col("older_dpd_max"))
            .alias("H2_POS_DPD_MAX_TRAJECTORY"),
        (
            (pl.col("recent_dpd_rate") > pl.col("older_dpd_rate") * 2)
            & (pl.col("recent_dpd_rate") > 0.05)
        ).cast(pl.Int8).alias("H2_POS_WORSENING_FLAG"),
    )
    result["pos"] = pos_traj

    # --- Credit card trajectory ---
    cc_enriched = cc.with_columns(
        (pl.col("AMT_BALANCE") / (pl.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_util"),
        (pl.col("AMT_DRAWINGS_CURRENT") / (pl.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_draw_ratio"),
    )
    cc_older, cc_recent = _split_half_trajectory(
        cc_enriched, "SK_ID_CURR", "MONTHS_BALANCE"
    )
    cc_older_stats = cc_older.group_by("SK_ID_CURR").agg(
        pl.col("_util").mean().alias("older_util"),
        (pl.col("SK_DPD") > 0).mean().alias("older_dpd_rate"),
        pl.col("_draw_ratio").mean().alias("older_draw_ratio"),
    )
    cc_recent_stats = cc_recent.group_by("SK_ID_CURR").agg(
        pl.col("_util").mean().alias("recent_util"),
        (pl.col("SK_DPD") > 0).mean().alias("recent_dpd_rate"),
        pl.col("_draw_ratio").mean().alias("recent_draw_ratio"),
    )
    cc_traj = cc_recent_stats.join(cc_older_stats, on="SK_ID_CURR", how="inner")
    cc_traj = cc_traj.select(
        "SK_ID_CURR",
        (pl.col("recent_util") - pl.col("older_util"))
            .alias("H2_CC_UTIL_TRAJECTORY"),
        (pl.col("recent_dpd_rate") - pl.col("older_dpd_rate"))
            .alias("H2_CC_DPD_TRAJECTORY"),
        (pl.col("recent_draw_ratio") - pl.col("older_draw_ratio"))
            .alias("H2_CC_DRAWING_TRAJECTORY"),
    )
    result["cc"] = cc_traj

    return result


# ---------------------------------------------------------------------------
# H3: Time-Since-Last-Bad-Event
# ---------------------------------------------------------------------------

def build_h3_features(
    installments: pl.DataFrame,
    bureau: pl.DataFrame,
    bureau_balance: pl.DataFrame,
    pos: pl.DataFrame,
    cc: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Recency of bad events (null = never had a bad event)."""
    log.info("  H3: Building time-since-last-event features")
    result = {}

    # --- Installments: days since last late / severe late ---
    ins_enriched = installments.with_columns(
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("_days_diff"),
    )
    ins_last_late = (
        ins_enriched.filter(pl.col("_days_diff") > 0)
        .group_by("SK_ID_CURR")
        .agg(
            # max DAYS_INSTALMENT = closest to 0 = most recent
            pl.col("DAYS_INSTALMENT").max().alias("H3_DAYS_SINCE_LAST_LATE_INS"),
        )
    )
    ins_last_severe = (
        ins_enriched.filter(pl.col("_days_diff") > 30)
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("DAYS_INSTALMENT").max().alias("H3_DAYS_SINCE_LAST_SEVERE_LATE_INS"),
        )
    )
    ins_h3 = ins_last_late.join(ins_last_severe, on="SK_ID_CURR", how="outer_coalesce")
    result["ins"] = ins_h3

    # --- Bureau: days since last overdue ---
    bur_last_overdue = (
        bureau.filter(pl.col("CREDIT_DAY_OVERDUE") > 0)
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("DAYS_CREDIT").max().alias("H3_DAYS_SINCE_LAST_BUREAU_OVERDUE"),
        )
    )
    # Bureau balance: months since last DPD
    bb_with_curr = bureau_balance.join(
        bureau.select("SK_ID_BUREAU", "SK_ID_CURR"), on="SK_ID_BUREAU", how="left"
    )
    bb_with_curr = bb_with_curr.with_columns(pl.col("STATUS").cast(pl.String))
    bb_dpd = bb_with_curr.filter(
        (pl.col("STATUS") == "1") | (pl.col("STATUS") == "2")
        | (pl.col("STATUS") == "3") | (pl.col("STATUS") == "4")
        | (pl.col("STATUS") == "5")
    )
    bb_last_dpd = (
        bb_dpd.group_by("SK_ID_CURR")
        .agg(
            pl.col("MONTHS_BALANCE").max().alias("H3_MONTHS_SINCE_LAST_BB_DPD"),
        )
    )
    bur_h3 = bur_last_overdue.join(bb_last_dpd, on="SK_ID_CURR", how="outer_coalesce")
    result["bur"] = bur_h3

    # --- POS/cash: months since last DPD ---
    pos_last_dpd = (
        pos.filter(pl.col("SK_DPD") > 0)
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("MONTHS_BALANCE").max().alias("H3_MONTHS_SINCE_LAST_POS_DPD"),
        )
    )
    pos_last_dpd_def = (
        pos.filter(pl.col("SK_DPD_DEF") > 0)
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("MONTHS_BALANCE").max().alias("H3_MONTHS_SINCE_LAST_POS_DPD_DEF"),
        )
    )
    pos_h3 = pos_last_dpd.join(pos_last_dpd_def, on="SK_ID_CURR", how="outer_coalesce")
    result["pos"] = pos_h3

    # --- Credit card: months since last DPD / high util ---
    cc_last_dpd = (
        cc.filter(pl.col("SK_DPD") > 0)
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("MONTHS_BALANCE").max().alias("H3_MONTHS_SINCE_LAST_CC_DPD"),
        )
    )
    cc_enriched = cc.with_columns(
        (pl.col("AMT_BALANCE") / (pl.col("AMT_CREDIT_LIMIT_ACTUAL") + 1))
            .alias("_util"),
    )
    cc_last_high_util = (
        cc_enriched.filter(pl.col("_util") > 0.9)
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("MONTHS_BALANCE").max().alias("H3_MONTHS_SINCE_LAST_HIGH_CC_UTIL"),
        )
    )
    cc_h3 = cc_last_dpd.join(cc_last_high_util, on="SK_ID_CURR", how="outer_coalesce")
    result["cc"] = cc_h3

    return result


# ---------------------------------------------------------------------------
# H4: Untapped Raw Columns
# ---------------------------------------------------------------------------

def build_h4_features(
    bureau: pl.DataFrame,
    prev: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Features from raw columns that are currently completely unused."""
    log.info("  H4: Building untapped raw column features")
    result = {}

    # --- Bureau: CNT_CREDIT_PROLONG and CREDIT_DAY_OVERDUE ---
    bur_untapped = bureau.group_by("SK_ID_CURR").agg(
        pl.col("CNT_CREDIT_PROLONG").sum().alias("H4_BUR_PROLONG_TOTAL"),
        pl.col("CNT_CREDIT_PROLONG").max().alias("H4_BUR_PROLONG_MAX"),
        (pl.col("CNT_CREDIT_PROLONG") > 0).sum().alias("H4_BUR_PROLONG_COUNT"),
        pl.col("CREDIT_DAY_OVERDUE").max().alias("H4_BUR_CURRENT_OVERDUE_MAX"),
        pl.col("CREDIT_DAY_OVERDUE").sum().alias("H4_BUR_CURRENT_OVERDUE_SUM"),
    )
    result["bur"] = bur_untapped

    # --- Previous: interest rates, client type, rejection reason, loan purpose ---
    prev_untapped = prev.group_by("SK_ID_CURR").agg(
        # Interest rates (higher = lender assessed as riskier)
        pl.col("RATE_INTEREST_PRIMARY").mean().alias("H4_PREV_INTEREST_RATE_MEAN"),
        pl.col("RATE_INTEREST_PRIMARY").max().alias("H4_PREV_INTEREST_RATE_MAX"),
        # Client type: Repeater vs New
        (pl.col("NAME_CLIENT_TYPE").cast(pl.String) == "Repeater").mean()
            .alias("H4_PREV_REPEATER_RATE"),
        (pl.col("NAME_CLIENT_TYPE").cast(pl.String) == "New").mean()
            .alias("H4_PREV_NEW_CLIENT_RATE"),
        # Rejection reasons (only non-null for refused apps)
        (pl.col("CODE_REJECT_REASON").cast(pl.String) == "HC").sum()
            .alias("H4_PREV_REJECT_HC_COUNT"),
        (pl.col("CODE_REJECT_REASON").cast(pl.String) == "SCOFR").sum()
            .alias("H4_PREV_REJECT_SCOFR_COUNT"),
        # Loan purpose diversity
        pl.col("NAME_CASH_LOAN_PURPOSE").n_unique()
            .alias("H4_PREV_LOAN_PURPOSE_DIVERSITY"),
    )
    result["prev"] = prev_untapped

    return result


# ---------------------------------------------------------------------------
# H5: Per-Credit-Type Bureau Aggregations
# ---------------------------------------------------------------------------

def build_h5_features(bureau: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Break down bureau financials by credit type (consumer, card, car)."""
    log.info("  H5: Building per-credit-type bureau features")

    # Cast CREDIT_TYPE to string for comparison
    bureau = bureau.with_columns(pl.col("CREDIT_TYPE").cast(pl.String))

    dfs = []
    for credit_type, prefix in [
        ("Consumer credit", "CONSUMER"),
        ("Credit card", "CARD"),
        ("Car loan", "CAR"),
    ]:
        type_df = (
            bureau.filter(pl.col("CREDIT_TYPE") == credit_type)
            .group_by("SK_ID_CURR")
            .agg(
                pl.col("AMT_CREDIT_SUM_DEBT").sum()
                    .alias(f"H5_BUR_{prefix}_DEBT_TOTAL"),
                pl.col("AMT_CREDIT_SUM_OVERDUE").sum()
                    .alias(f"H5_BUR_{prefix}_OVERDUE_TOTAL"),
                (pl.col("CREDIT_ACTIVE").cast(pl.String) == "Active").sum()
                    .alias(f"H5_BUR_{prefix}_ACTIVE_COUNT"),
            )
        )
        dfs.append(type_df)

    # Join all type-specific DataFrames
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.join(df, on="SK_ID_CURR", how="outer_coalesce")

    return {"bur_types": combined}


# ---------------------------------------------------------------------------
# H6: Application Frequency / Velocity
# ---------------------------------------------------------------------------

def build_h6_features(
    prev: pl.DataFrame,
    bureau: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Application frequency, acceleration, and inter-application gaps."""
    log.info("  H6: Building frequency/velocity features")
    result = {}

    # --- Previous applications: apps per year and acceleration ---
    prev_vel = prev.group_by("SK_ID_CURR").agg(
        pl.len().alias("_total"),
        (pl.col("DAYS_DECISION") > -365).sum().alias("_recent_1y"),
        pl.col("DAYS_DECISION").max().alias("_most_recent"),
        pl.col("DAYS_DECISION").min().alias("_oldest"),
    ).with_columns(
        ((pl.col("_most_recent") - pl.col("_oldest")).abs() / 365.25 + 1)
            .alias("_history_years"),
    ).with_columns(
        (pl.col("_total") / pl.col("_history_years"))
            .alias("H6_PREV_APPS_PER_YEAR"),
        (pl.col("_recent_1y") / (pl.col("_total") / pl.col("_history_years") + 0.1))
            .alias("H6_PREV_APPLICATION_ACCELERATION"),
    ).select("SK_ID_CURR", "H6_PREV_APPS_PER_YEAR", "H6_PREV_APPLICATION_ACCELERATION")

    # --- Previous applications: inter-application gap stats ---
    prev_sorted = prev.sort(["SK_ID_CURR", "DAYS_DECISION"])
    prev_gaps = (
        prev_sorted.with_columns(
            (pl.col("DAYS_DECISION") - pl.col("DAYS_DECISION").shift(1).over("SK_ID_CURR"))
                .alias("_gap_days")
        )
        .filter(pl.col("_gap_days").is_not_null())
        .group_by("SK_ID_CURR")
        .agg(
            pl.col("_gap_days").min().alias("H6_PREV_MIN_APP_GAP_DAYS"),
            pl.col("_gap_days").mean().alias("H6_PREV_MEAN_APP_GAP_DAYS"),
            pl.col("_gap_days").std().alias("H6_PREV_APP_GAP_STD"),
        )
    )

    prev_h6 = prev_vel.join(prev_gaps, on="SK_ID_CURR", how="outer_coalesce")
    result["prev"] = prev_h6

    # --- Bureau: credits per year ---
    bur_vel = bureau.group_by("SK_ID_CURR").agg(
        pl.len().alias("_total"),
        pl.col("DAYS_CREDIT").max().alias("_most_recent"),
        pl.col("DAYS_CREDIT").min().alias("_oldest"),
    ).with_columns(
        ((pl.col("_most_recent") - pl.col("_oldest")).abs() / 365.25 + 1)
            .alias("_history_years"),
    ).with_columns(
        (pl.col("_total") / pl.col("_history_years"))
            .alias("H6_BUR_CREDITS_PER_YEAR"),
    ).select("SK_ID_CURR", "H6_BUR_CREDITS_PER_YEAR")
    result["bur"] = bur_vel

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_all_experimental_features(
    installments: pl.DataFrame,
    cc: pl.DataFrame,
    pos: pl.DataFrame,
    bureau: pl.DataFrame,
    bureau_balance: pl.DataFrame,
    prev: pl.DataFrame,
) -> pl.DataFrame:
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

    all_dfs: list[pl.DataFrame] = []
    for name, builder, kwargs in builders:
        log.info(f"  {name}: {HYPOTHESIS_PREFIX[name]}")
        result_dfs = builder(**kwargs)
        for df in result_dfs.values():
            all_dfs.append(df)

    # Join all on SK_ID_CURR using successive outer joins
    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.join(df, on="SK_ID_CURR", how="outer_coalesce")

    feature_cols = [c for c in combined.columns if c != "SK_ID_CURR"]
    log.info(f"Total experimental features: {len(feature_cols)}")

    return combined


def build_winning_experimental_features(
    installments: pl.DataFrame,
    cc: pl.DataFrame,
    pos: pl.DataFrame,
    bureau: pl.DataFrame,
    bureau_balance: pl.DataFrame,
    prev: pl.DataFrame,
) -> pl.DataFrame:
    """Build only the winning hypotheses (H2, H3, H4, H5) for production use.

    Returns a single DataFrame keyed on SK_ID_CURR with ~41 features.
    """
    log.info("Building winning experimental features (H2, H3, H4, H5)")

    # AMT_ANNUITY in bureau needs casting
    bureau = bureau.with_columns(
        pl.col("AMT_ANNUITY").cast(pl.String).cast(pl.Float64, strict=False)
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

    all_dfs: list[pl.DataFrame] = []
    for name, builder, kwargs in builders:
        log.info(f"  {name}: {HYPOTHESIS_PREFIX[name]}")
        result_dfs = builder(**kwargs)
        for df in result_dfs.values():
            all_dfs.append(df)

    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.join(df, on="SK_ID_CURR", how="outer_coalesce")

    feature_cols = [c for c in combined.columns if c != "SK_ID_CURR"]
    log.info(f"Winning experimental features: {len(feature_cols)}")

    return combined


def get_hypothesis_features(df: pl.DataFrame) -> dict[str, list[str]]:
    """Return mapping of hypothesis tag -> list of feature column names."""
    result = {}
    for prefix in HYPOTHESIS_PREFIX:
        cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
        if cols:
            result[prefix] = cols
    return result

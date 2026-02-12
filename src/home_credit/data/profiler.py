"""Profile datasets: schema summaries, missingness, distributions."""

from pathlib import Path

from pyspark.sql import functions as F

from home_credit.utils import DATA_DIR, OUTPUT_DIR, get_logger, get_spark_session

log = get_logger(__name__)

TABLE_NAMES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
]


def profile_table(name: str, data_dir: Path = DATA_DIR) -> str:
    """Generate a markdown profile for a single table."""
    spark = get_spark_session()
    path = data_dir / f"{name}.csv"
    df = spark.read.csv(str(path), header=True, inferSchema=True)

    n_rows = df.count()
    n_cols = len(df.columns)

    lines = [f"### {name}", f"- **Rows**: {n_rows:,}", f"- **Columns**: {n_cols}", ""]

    lines.append("| Column | Dtype | Missing % | Unique | Min | Max | Mean |")
    lines.append("|--------|-------|-----------|--------|-----|-----|------|")

    numeric_types = {"int", "bigint", "smallint", "tinyint", "float", "double"}

    for col_name, dtype in df.dtypes:
        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = null_count / n_rows * 100 if n_rows > 0 else 0
        n_unique = df.select(F.countDistinct(col_name)).collect()[0][0]

        if dtype in numeric_types:
            stats = df.select(
                F.min(col_name).alias("min_val"),
                F.max(col_name).alias("max_val"),
                F.avg(col_name).alias("mean_val"),
            ).collect()[0]
            min_val = str(stats["min_val"]) if stats["min_val"] is not None else "—"
            max_val = str(stats["max_val"]) if stats["max_val"] is not None else "—"
            mean_val = f"{stats['mean_val']:.2f}" if stats["mean_val"] is not None else "—"
        else:
            min_val = "—"
            max_val = "—"
            mean_val = "—"

        lines.append(
            f"| {col_name} | {dtype} | {null_pct:.1f}% | {n_unique:,} "
            f"| {min_val} | {max_val} | {mean_val} |"
        )

    lines.append("")
    return "\n".join(lines)


def profile_all(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> str:
    """Profile all tables and write output/data_profile.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = ["# Data Profile — Home Credit Default Risk", ""]

    sections.append("## Table Relationships")
    sections.append("")
    sections.append("All tables join to the main application table via `SK_ID_CURR`:")
    sections.append("")
    sections.append("```")
    sections.append("application_train / application_test")
    sections.append("├── bureau (SK_ID_CURR) ── one applicant has many bureau records")
    sections.append("│   └── bureau_balance (SK_ID_BUREAU) ── monthly status for each bureau record")
    sections.append("├── previous_application (SK_ID_CURR) ── previous loan applications at Home Credit")
    sections.append("│   ├── installments_payments (SK_ID_PREV) ── payment history for previous loans")
    sections.append("│   ├── POS_CASH_balance (SK_ID_PREV) ── monthly POS/cash loan snapshots")
    sections.append("│   └── credit_card_balance (SK_ID_PREV) ── monthly credit card snapshots")
    sections.append("```")
    sections.append("")
    sections.append("**Join keys:**")
    sections.append("- `SK_ID_CURR`: Links all tables to the main application (applicant level)")
    sections.append("- `SK_ID_BUREAU`: Links `bureau` → `bureau_balance`")
    sections.append(
        "- `SK_ID_PREV`: Links `previous_application` → `installments_payments`, "
        "`POS_CASH_balance`, `credit_card_balance`"
    )
    sections.append("")
    sections.append("---")
    sections.append("")

    sections.append("## Table Profiles")
    sections.append("")

    for name in TABLE_NAMES:
        log.info(f"Profiling {name}")
        try:
            sections.append(profile_table(name, data_dir))
            sections.append("---")
            sections.append("")
        except Exception:
            log.warning(f"Table {name} not found, skipping")
            sections.append(f"### {name}\n\n*File not found*\n\n---\n")

    spark = get_spark_session()
    train_path = data_dir / "application_train.csv"
    if train_path.exists():
        df = spark.read.csv(str(train_path), header=True, inferSchema=True).select("TARGET")
        total = df.count()
        counts = df.groupBy("TARGET").count().orderBy("TARGET").collect()
        sections.append("## Target Distribution (application_train)")
        sections.append("")
        for row in counts:
            val, count = row["TARGET"], row["count"]
            pct = count / total * 100
            sections.append(f"- TARGET={val}: {count:,} ({pct:.1f}%)")
        sections.append("")

    report = "\n".join(sections)
    out_path = output_dir / "data_profile.md"
    out_path.write_text(report)
    log.info(f"Data profile saved to {out_path}")
    return report


if __name__ == "__main__":
    profile_all()

"""Profile datasets: schema summaries, missingness, distributions."""

from pathlib import Path

import polars as pl

from home_credit.utils import DATA_DIR, OUTPUT_DIR, get_logger

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
    path = data_dir / f"{name}.csv"
    df = pl.read_csv(path, n_rows=0)  # schema only
    schema = dict(df.schema)

    df = pl.scan_csv(path).collect()
    n_rows = df.shape[0]
    n_cols = df.shape[1]

    lines = [f"### {name}", f"- **Rows**: {n_rows:,}", f"- **Columns**: {n_cols}", ""]

    # Column details
    lines.append("| Column | Dtype | Missing % | Unique | Min | Max | Mean |")
    lines.append("|--------|-------|-----------|--------|-----|-----|------|")

    for col_name in df.columns:
        col = df[col_name]
        dtype = str(schema.get(col_name, col.dtype))
        null_pct = col.null_count() / n_rows * 100 if n_rows > 0 else 0
        n_unique = col.n_unique()

        if col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
            min_val = f"{col.min()}"
            max_val = f"{col.max()}"
            mean_val = f"{col.mean():.2f}" if col.mean() is not None else "—"
        else:
            min_val = "—"
            max_val = "—"
            mean_val = "—"

        lines.append(
            f"| {col_name} | {dtype} | {null_pct:.1f}% | {n_unique:,} | {min_val} | {max_val} | {mean_val} |"
        )

    lines.append("")
    return "\n".join(lines)


def profile_all(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> str:
    """Profile all tables and write output/data_profile.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = ["# Data Profile — Home Credit Default Risk", ""]

    # Table relationships
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

    # Profile each table
    sections.append("## Table Profiles")
    sections.append("")

    for name in TABLE_NAMES:
        log.info(f"Profiling {name}")
        try:
            sections.append(profile_table(name, data_dir))
            sections.append("---")
            sections.append("")
        except FileNotFoundError:
            log.warning(f"Table {name} not found, skipping")
            sections.append(f"### {name}\n\n*File not found*\n\n---\n")

    # Target distribution
    train_path = data_dir / "application_train.csv"
    if train_path.exists():
        df = pl.read_csv(train_path, columns=["TARGET"])
        counts = df["TARGET"].value_counts().sort("TARGET")
        sections.append("## Target Distribution (application_train)")
        sections.append("")
        for row in counts.iter_rows():
            val, count = row
            pct = count / df.shape[0] * 100
            sections.append(f"- TARGET={val}: {count:,} ({pct:.1f}%)")
        sections.append("")

    report = "\n".join(sections)
    out_path = output_dir / "data_profile.md"
    out_path.write_text(report)
    log.info(f"Data profile saved to {out_path}")
    return report


if __name__ == "__main__":
    profile_all()

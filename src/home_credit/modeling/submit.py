"""Generate predictions and submit to Kaggle."""

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from home_credit.features.pipeline import build_all_features, get_feature_columns
from home_credit.modeling.train import load_model, prepare_data
from home_credit.utils import OUTPUT_DIR, get_logger

log = get_logger(__name__)

COMPETITION = "home-credit-default-risk"


def generate_submission(
    test_df: pl.DataFrame,
    model_data: dict,
    save_path: Path | None = None,
) -> pl.DataFrame:
    """Generate submission DataFrame from test data and trained models."""
    if save_path is None:
        save_path = OUTPUT_DIR / "submission.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = model_data["feature_names"]
    models = model_data["models"]

    # Prepare test features (only numeric columns the model was trained on)
    available = [c for c in feature_names if c in test_df.columns]
    missing = [c for c in feature_names if c not in test_df.columns]
    if missing:
        log.warning(f"Missing {len(missing)} features in test data, filling with 0")
        for c in missing:
            test_df = test_df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(c))

    X_test = test_df.select(feature_names).to_numpy().astype(np.float32)

    # Average predictions across CV folds
    preds = np.zeros(X_test.shape[0])
    for model in models:
        preds += model.predict_proba(X_test)[:, 1] / len(models)

    submission = pl.DataFrame({
        "SK_ID_CURR": test_df["SK_ID_CURR"],
        "TARGET": preds,
    })

    submission.write_csv(save_path)
    log.info(f"Submission saved to {save_path} ({submission.shape[0]:,} rows)")
    return submission


def submit_to_kaggle(message: str, submission_path: Path | None = None) -> None:
    """Submit to Kaggle via CLI."""
    if submission_path is None:
        submission_path = OUTPUT_DIR / "submission.csv"

    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    venv_bin = Path(sys.executable).parent
    kaggle_bin = shutil.which("kaggle", path=str(venv_bin)) or shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError("kaggle CLI not found")

    log.info(f"Submitting {submission_path} to {COMPETITION}")
    subprocess.run(
        [
            kaggle_bin, "competitions", "submit",
            "-c", COMPETITION,
            "-f", str(submission_path),
            "-m", message,
        ],
        check=True,
    )
    log.info("Submission successful!")


def fetch_kaggle_scores() -> pl.DataFrame:
    """Fetch all submission scores from Kaggle.

    Returns a DataFrame with columns: fileName, date, description, status,
    publicScore, privateScore.
    """
    venv_bin = Path(sys.executable).parent
    kaggle_bin = shutil.which("kaggle", path=str(venv_bin)) or shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError("kaggle CLI not found")

    result = subprocess.run(
        [kaggle_bin, "competitions", "submissions", "-c", COMPETITION, "--csv"],
        capture_output=True, text=True, check=True,
    )
    # Parse the CSV output from kaggle CLI
    import io
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if not rows:
        log.warning("No submissions found on Kaggle")
        return pl.DataFrame()

    return pl.DataFrame(rows)


def sync_kaggle_scores() -> pl.DataFrame:
    """Fetch scores from Kaggle and rewrite submission_log.csv with them.

    Matches Kaggle submissions to local log entries by description/message.
    Returns the updated log as a DataFrame.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "submission_log.csv"

    # Fetch from Kaggle
    kaggle_df = fetch_kaggle_scores()
    if kaggle_df.is_empty():
        log.info("No Kaggle submissions to sync")
        if log_path.exists():
            return pl.read_csv(log_path)
        return pl.DataFrame()

    log.info(f"Fetched {kaggle_df.shape[0]} submissions from Kaggle")

    # Read existing local log
    if log_path.exists():
        local_df = pl.read_csv(log_path)
    else:
        local_df = pl.DataFrame({
            "timestamp": [], "message": [], "cv_score": [],
            "n_features": [], "kaggle_public_lb": [], "kaggle_private_lb": [],
        })

    # Ensure columns exist
    if "kaggle_private_lb" not in local_df.columns:
        local_df = local_df.with_columns(pl.lit("").alias("kaggle_private_lb"))

    # Build a lookup from kaggle description -> scores
    kaggle_lookup = {}
    for row in kaggle_df.iter_rows(named=True):
        desc = row.get("description", "")
        public = row.get("publicScore", "")
        private = row.get("privateScore", "")
        if desc:
            kaggle_lookup[desc] = (public, private)

    # Update local log entries by matching message to description
    new_public = []
    new_private = []
    for row in local_df.iter_rows(named=True):
        msg = row["message"]
        if msg in kaggle_lookup:
            pub, priv = kaggle_lookup[msg]
            new_public.append(pub)
            new_private.append(priv)
        else:
            new_public.append(row.get("kaggle_public_lb", ""))
            new_private.append(row.get("kaggle_private_lb", ""))

    updated = local_df.with_columns(
        pl.Series("kaggle_public_lb", new_public),
        pl.Series("kaggle_private_lb", new_private),
    )

    # Write back
    updated.write_csv(log_path)
    log.info(f"Updated {log_path} with Kaggle scores")

    for row in updated.iter_rows(named=True):
        pub = row["kaggle_public_lb"]
        priv = row["kaggle_private_lb"]
        cv = row["cv_score"]
        msg = row["message"][:60]
        log.info(f"  CV={cv}  Public={pub}  Private={priv}  | {msg}")

    return updated


def log_submission(
    message: str,
    cv_score: float,
    n_features: int,
    kaggle_score: str = "",
) -> None:
    """Log submission to output/submission_log.csv."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "submission_log.csv"
    file_exists = log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "message", "cv_score", "n_features",
                "kaggle_public_lb", "kaggle_private_lb",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            message,
            f"{cv_score:.6f}",
            n_features,
            kaggle_score,
            "",
        ])


def main():
    parser = argparse.ArgumentParser(description="Generate and submit predictions")
    parser.add_argument("--message", "-m", help="Submission description")
    parser.add_argument("--model", default="best_model", help="Model name to load")
    parser.add_argument("--submit", action="store_true", help="Actually submit to Kaggle")
    parser.add_argument(
        "--sync-scores", action="store_true",
        help="Fetch scores from Kaggle and update submission_log.csv",
    )
    args = parser.parse_args()

    # Sync-only mode
    if args.sync_scores:
        sync_kaggle_scores()
        return

    if not args.message:
        parser.error("--message is required when generating a submission")

    # Load model
    model_data = load_model(args.model)
    log.info(f"Loaded model: {args.model} (CV AUC: {model_data['mean_auc']:.5f})")

    # Build features (will use cache)
    _, test_df = build_all_features()

    # Generate submission
    submission = generate_submission(test_df, model_data)

    # Log
    log_submission(
        args.message,
        model_data["mean_auc"],
        len(model_data["feature_names"]),
    )

    if args.submit:
        submit_to_kaggle(args.message)
        # Auto-sync scores after submitting (with a delay for processing)
        log.info("Waiting a few seconds for Kaggle to process...")
        import time
        time.sleep(10)
        sync_kaggle_scores()
    else:
        log.info("Submission file generated. Use --submit to actually submit to Kaggle.")


if __name__ == "__main__":
    main()

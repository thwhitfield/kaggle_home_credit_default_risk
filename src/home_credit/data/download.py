"""Download and extract Kaggle competition data."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from home_credit.utils import DATA_DIR, get_logger

log = get_logger(__name__)

COMPETITION = "home-credit-default-risk"

EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "sample_submission.csv",
]


def download_data(data_dir: Path = DATA_DIR) -> None:
    """Download competition data via Kaggle API if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing = [f for f in EXPECTED_FILES if (data_dir / f).exists()]
    if len(existing) == len(EXPECTED_FILES):
        log.info("All data files already present, skipping download.")
        return

    log.info(f"Downloading {COMPETITION} data to {data_dir}")
    # Find kaggle CLI next to the current Python interpreter (works inside venv)
    venv_bin = Path(sys.executable).parent
    kaggle_bin = shutil.which("kaggle", path=str(venv_bin)) or shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError("kaggle CLI not found. Install it with: uv pip install kaggle")
    subprocess.run(
        [kaggle_bin, "competitions", "download", "-c", COMPETITION, "-p", str(data_dir)],
        check=True,
    )

    # Extract any zip files
    for zf in data_dir.glob("*.zip"):
        log.info(f"Extracting {zf.name}")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(data_dir)
        zf.unlink()
        log.info(f"Removed {zf.name}")

    # Verify
    missing = [f for f in EXPECTED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files after download: {missing}")

    log.info("All data files downloaded and extracted successfully.")


if __name__ == "__main__":
    download_data()

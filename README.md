# Home Credit Default Risk

A complete ML pipeline for the [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition — predict whether a loan applicant will default using application data and credit history across 8 interconnected tables.

**CV AUC: 0.787 | 220 engineered features | XGBoost**

## Quick Start

```bash
# Install dependencies
uv sync

# Download competition data (requires Kaggle API credentials)
uv run python -m home_credit.data.download

# Profile the data
uv run python -m home_credit.data.profiler

# Build features
uv run python -m home_credit.features.pipeline

# Train model
uv run python -m home_credit.modeling.train

# Generate submission
uv run python -m home_credit.modeling.submit -m "description" --model best_model

# Submit to Kaggle
uv run python -m home_credit.modeling.submit -m "description" --model best_model --submit
```

## Setup

**Prerequisites:** Python >= 3.11, [uv](https://docs.astral.sh/uv/)

```bash
git clone <this-repo>
cd kaggle_home_credit_default_risk
uv sync
```

**Kaggle API:** Place your `kaggle.json` in `~/.kaggle/` with permissions `chmod 600`.

## Project Structure

```
├── pyproject.toml
├── src/home_credit/
│   ├── data/
│   │   ├── download.py      # Kaggle API download + extraction
│   │   ├── loader.py        # Polars CSV loading with dtype optimization
│   │   └── profiler.py      # Schema summaries, missingness, distributions
│   ├── features/
│   │   ├── application.py   # Income ratios, external scores, age/employment
│   │   ├── bureau.py        # Credit history from other institutions
│   │   ├── previous.py      # Previous Home Credit applications
│   │   ├── installments.py  # Payment behavior (late/early, underpayments)
│   │   ├── pos_cash.py      # POS/cash loan DPD and contract status
│   │   ├── credit_card.py   # Utilization, drawings, payment-to-minimum
│   │   └── pipeline.py      # Orchestrates all feature engineering
│   ├── modeling/
│   │   ├── train.py         # XGBoost CV training, Optuna tuning, experiment logging
│   │   ├── evaluate.py      # SHAP, threshold analysis, feature importance plots
│   │   └── submit.py        # Kaggle submission generation and upload
│   └── utils.py             # Logging, timing, paths
├── tests/
│   ├── test_data.py         # Data loading and schema tests
│   ├── test_features.py     # Feature engineering validation
│   └── test_modeling.py     # Training and submission format tests
├── scripts/
│   ├── run_experiments.py   # Full experimentation pipeline
│   └── run_analysis.py      # SHAP + threshold + importance analysis
├── data/                    # .gitignored — raw CSVs and feature parquets
└── output/                  # .gitignored — models, plots, reports
```

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests validate data loading, feature engineering correctness, model training on small samples, and submission file format.

## Key Results

| Experiment | CV AUC | Features |
|-----------|--------|----------|
| Baseline (default params) | 0.7857 | 266 |
| Feature selection | 0.7854 | 245 |
| **Final (tuned)** | **0.7871** | **220** |

Top predictive features: external credit scores (EXT_SOURCE_*), credit card utilization, payment timeliness, employment duration, and previous application refusal rates.

See `output/final_report.md` for detailed analysis.

## Full Pipeline

Run the complete pipeline end-to-end:

```bash
# 1. Download data
uv run python -m home_credit.data.download

# 2. Profile data (optional)
uv run python -m home_credit.data.profiler

# 3. Build all features (cached to parquet)
uv run python -m home_credit.features.pipeline

# 4. Train + tune (writes to output/)
uv run python scripts/run_experiments.py

# 5. Analysis (SHAP, threshold, importance plots)
uv run python scripts/run_analysis.py

# 6. Submit
uv run python -m home_credit.modeling.submit -m "Final tuned model" --model best_model --submit
```

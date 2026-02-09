# CLAUDE.md — Project Context

## Project

Home Credit Default Risk — Kaggle competition. Predict loan default from application data + 7 supplementary credit history tables.

## Stack

- **Data:** Polars (lazy evaluation, parquet caching)
- **Modeling:** XGBoost (5-fold stratified CV)
- **Tuning:** Optuna
- **Analysis:** SHAP, matplotlib/seaborn
- **Deps:** managed with `uv`, src layout package `home_credit`

## Current Best Score

- **CV AUC:** 0.7871 (5-fold stratified)
- **Features:** 220 (after importance-based pruning from 282)
- **Model:** XGBoost with tuned hyperparameters (depth=5, lr=0.03, 1500 trees)

## How to Run

```bash
uv sync
uv run python -m home_credit.data.download   # download data
uv run python -m home_credit.features.pipeline  # build features (cached to parquet)
uv run python -m home_credit.modeling.train    # train baseline
uv run python -m home_credit.modeling.submit -m "msg" --model best_model --submit
uv run pytest tests/ -v
```

## Key Files

- `src/home_credit/features/pipeline.py` — Orchestrates feature building and joining
- `src/home_credit/modeling/train.py` — CV training, Optuna tuning, experiment logging
- `src/home_credit/modeling/submit.py` — Submission generation and Kaggle upload (CLI)
- `scripts/run_experiments.py` — Full feature selection + tuning pipeline
- `scripts/run_analysis.py` — SHAP + threshold analysis
- `output/experiment_log.csv` — All experiment results
- `data/features/` — Cached parquet feature sets

## Architecture

All supplementary tables are aggregated to `SK_ID_CURR` level. Features are built independently per table, then left-joined onto the application table. Feature sets are cached as parquet files in `data/features/`.

Feature pipeline skips recomputation if parquet cache exists. To rebuild, delete `data/features/` and rerun.

## Known Issues

- Optuna tuning with 30 trials on 5-fold CV is very slow (~2+ hours). Consider reducing to 15 trials or using 3-fold CV for tuning.
- ~72% of applicants don't have credit card data, so CC_* features are null for most rows. XGBoost handles this natively.
- XGBoost 3.x moved `early_stopping_rounds` to the constructor (not `fit()`).

## Next Steps for Improvement

- Add interaction features (EXT_SOURCE_2 * EXT_SOURCE_3, etc.)
- Target encoding for ORGANIZATION_TYPE, OCCUPATION_TYPE
- Time-weighted aggregations (recent behavior matters more)
- Try LightGBM for blending
- Null importance feature selection

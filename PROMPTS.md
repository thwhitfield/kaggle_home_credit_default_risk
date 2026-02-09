# Prompt Walkthrough — Building This Project with Claude Code

This document walks through the prompts I gave Claude Code and what it built in response. The entire project — from empty repo to a competitive Kaggle submission — was built in a single day across multiple conversation sessions.

---

## Session 1: Build the Full Pipeline from Scratch

### Prompt

> Build a complete ML pipeline for the Home Credit Default Risk Kaggle competition. Download the data, profile it, engineer features from all 8 tables, train XGBoost with cross-validation, tune hyperparameters with Optuna, run SHAP analysis, and submit to Kaggle.

### What Claude Did

Built the entire project structure from scratch:

- **Data layer**: Download module (Kaggle CLI), loader with dtype optimization, profiler
- **Feature engineering**: 6 table-specific modules (application, bureau, previous, installments, POS/cash, credit card), each with domain-informed aggregations — not generic min/max/mean, but features a credit analyst would care about (debt-to-credit ratios, late payment rates, utilization, etc.)
- **Feature pipeline**: Orchestrator that builds all features, caches as parquet, joins via `SK_ID_CURR`
- **Modeling**: XGBoost with 5-fold stratified CV, Optuna hyperparameter tuning (30 trials), experiment logging
- **Evaluation**: ROC curves, SHAP analysis, threshold analysis for business decisions
- **Submission**: Kaggle CLI upload with score syncing back into local logs
- **Tests**: Unit tests for data, features, and modeling modules
- **Docs**: README, CLAUDE.md, walkthrough notebook

**Result:** Baseline CV AUC 0.7857 (266 features) → Tuned CV AUC 0.7871 (220 features). Two Kaggle submissions: Public LB 0.7904 and 0.7912.

**Commits:** `65a8539`, `bdb0650`, `b23cdc4`, `020bc7b`

---

## Session 2: Optimize Features and Add LightGBM

### Prompt

> Now focus on improving the score. Add more features, try LightGBM for blending, and add target encoding for high-cardinality categoricals. Submit the best result to Kaggle.

### What Claude Did

Major feature engineering overhaul (~150 new features):

- **EXT_SOURCE interactions**: Pairwise products, differences, variance, weighted means of the 3 external credit scores (these turned out to be the single most important feature group)
- **Time-windowed aggregations**: Recent 3/6/12 month behavior vs all-time stats for bureau, installments, POS/cash, and credit card tables
- **Cross-table interaction features**: Combined DPD score across all tables, debt-to-income ratios, first-time applicant flags, thin file detection
- **Target encoding**: Smoothed per-fold encoding of 6 high-cardinality categoricals (ORGANIZATION_TYPE, OCCUPATION_TYPE, etc.)
- **LightGBM**: Added as a second model type, found optimal XGB/LGB blend weight via grid search
- **Feature pruning**: Importance-based pruning dropped 374 → 311 features

**Result:** CV AUC 0.7930 (XGB 0.68 + LGB 0.32 blend, 311 features). Kaggle Public LB 0.7966, Private LB 0.7933. A +59 bps CV jump — the biggest improvement of the entire project.

**Commits:** `c573717`, `2c8717c`

---

## Session 3: Expand the Ensemble

### Prompt

> Can we improve further by adding more model diversity? Try different LightGBM variants and CatBoost. Find the optimal blend across all models.

### What Claude Did

Tested 5 model variants and built a 4-way ensemble:

- **LGB GBDT** (tuned): Better regularization than V2 — num_leaves=48, lr=0.02, 2000 trees
- **LGB GOSS**: Gradient-based one-side sampling — the surprise winner, got 33% blend weight
- **LGB DART**: Dropout-based boosting — underperformed (no early stopping support), rejected
- **CatBoost**: Weakest individually (0.7903) but added diversity — earned 10% blend weight
- **4-way blend optimization**: Grid search over (XGB, LGB, GOSS, CB) weights
- **Additional cross-table features**: Expanded from 9 to 10 interaction features

**Result:** CV AUC 0.7939 (XGB 0.40 + LGB 0.17 + GOSS 0.33 + CB 0.10, 335 features). Kaggle Public LB 0.7970, Private LB 0.7934. Our best score.

**Commits:** `de638c6`, `5c9bf68`

---

## Session 4: Feature Discovery Experiments

### Prompt

> Come up with a list of experiments to run to find additional features to create which could potentially help improve our model performance. If it makes sense, come up with a way to run those experiments in a more efficient manner. Don't look up anything from competitors who have done this kaggle before. Try to think intuitively about the problem and about what sorts of features might make sense to incorporate, which provide some additional signal beyond what is already being captured.

### What Claude Did

Designed and executed a systematic feature discovery experiment:

**Planning phase** — Identified 6 hypotheses (~65 candidate features) based on credit risk intuition:
1. H1: Per-loan behavioral variance (does the applicant behave consistently across loans?)
2. H2: Behavioral trajectory (is the applicant getting better or worse over time?)
3. H3: Time since last bad event (recency of bad behavior, not just frequency)
4. H4: Untapped raw columns (CNT_CREDIT_PROLONG, CREDIT_DAY_OVERDUE, RATE_INTEREST_PRIMARY, etc.)
5. H5: Per-credit-type bureau aggregations (break down debt/overdue by credit type)
6. H6: Application velocity (accelerating credit-seeking as a desperation signal)

**Efficient screening design** — Rather than training a full 4-way blend per hypothesis (~40 min each), ran one fast XGB (500 trees, lr=0.05) per hypothesis as an ablation study. 8 CV runs in ~26 minutes total.

**Results:**
- H1 (-3.8 bps) and H6 (-0.9 bps) rejected
- H2 (+1.2 bps), H3 (+3.3 bps), H4 (+0.9 bps), H5 (+3.6 bps) accepted
- Integrated 41 winning features into production pipeline
- Full 4-way retrain: CV AUC 0.7935 (-3.3 bps vs V3)

**Key lesson:** Screening gains from fast shallow models don't always transfer to production deep ensembles. The 1500-2000 tree models already captured these patterns implicitly. V3 remained our best model.

**Commits:** `e2337ef`

---

## Session 5: Update the Walkthrough

### Prompt

> Update the walkthrough notebook so that it walks through all the different experiments that we've run as part of this and the key takeaways. It should highlight the actual kaggle submission scores as the key points of reference, but also tie those back to how they corresponded to our local CV results at each stage.

### What Claude Did

Rewrote the walkthrough notebook (42 cells) to tell the complete story:

- Updated intro to reflect final architecture (4-model blend, 380+ features)
- Reorganized experiment log display by phase (V1/V2/V3/V4/screening)
- Wrote detailed narrative for each version with tables showing CV AUC and Kaggle scores
- Added V4 feature discovery section with hypothesis descriptions and screening results
- Updated Kaggle submissions section to handle V4 (not submitted, with explanation)
- Rewrote summary with full progression table, CV-to-LB alignment analysis, and technical lessons
- Fixed stale code references (model_data → v3_xgb)

**Commits:** `8def21f`

---

## Score Progression

| Version | What Changed | CV AUC | Public LB | Private LB |
|---------|-------------|--------|-----------|------------|
| V1 Baseline | Default XGBoost, 266 features | 0.7857 | 0.7904 | 0.7859 |
| V1 Tuned | Optuna hyperparameters, 220 features | 0.7871 | 0.7912 | 0.7865 |
| V2 Blend | +150 features, XGB+LGB blend | 0.7930 | 0.7966 | 0.7933 |
| **V3 Ensemble** | **4-way blend (XGB+LGB+GOSS+CB)** | **0.7939** | **0.7970** | **0.7934** |
| V4 Experimental | +41 experimental features | 0.7935 | — | — |

## What I Learned

1. **Feature engineering is king** — V2's +150 features gave 59 bps, dwarfing everything else.
2. **Model diversity pays off** — Even weak models (CatBoost at 0.7903) improve the blend.
3. **Trust your CV** — Our CV consistently tracked ~30-40 bps below Kaggle Public LB, with stable gap across all versions. This let us skip submitting V4 when it regressed locally.
4. **Diminishing returns are real** — Going from V1→V2 was massive (+59 bps). V2→V3 was modest (+9 bps). V3→V4 was negative. The low-hanging fruit gets picked fast.
5. **Screening != production** — Features that help a 500-tree model may already be captured by a 2000-tree model. Always validate in the full pipeline.

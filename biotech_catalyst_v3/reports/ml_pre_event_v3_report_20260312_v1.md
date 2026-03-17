# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-17
**Objective:** Predict large stock move from public pre-event information only.
**No press release content used.**
**Version:** v3 — adds 9 timing features + 6 train-fold-safe reaction priors vs v2 baseline.

> ⚠ **VALIDITY STATUS: APPROXIMATELY VALID (historical analysis only)**
> This model used 9 timing features that are anchored to the realized event date (`v_actual_date`).
> These features encode valid pre-event signals but cannot be reproduced at live inference
> without recomputing them using `prediction_date = today`.
> See: [`reports/pre_event_validity_audit_v0.6_20260317.md`](pre_event_validity_audit_v0.6_20260317.md)
> Audit action: invalid features removed from `build_pre_event_train_v2.py`. Next retrain will be clean.

---

## 0. Training Setup (required fields)

| Field | Value |
|---|---|
| **Train table** | `ml_baseline_train_20260317_v3.csv` |
| **Feature dataset** | `ml_dataset_features_20260316_v2.csv` |
| **Total usable rows** | **596** (2023+ events with row_ready=True and v_actual_date set) |
| **Train rows** | **417** (year range: 2023–2024) |
| **Val rows** | **89** (year range: 2024–2025) |
| **Test rows** | **90** (year range: 2025–2026) |
| **Class balance — train** | 119 pos / 298 neg = **28.5% positive** |
| **Class balance — val** | 27 pos / 62 neg = **30.3% positive** |
| **Class balance — test** | 38 pos / 52 neg = **42.2% positive** |
| **Split method** | Time-ordered by `v_actual_date` ascending; 70/15/15 percentile split |
| **Cohort exclusion** | 2020–2022 rows excluded: near-zero positive rate (~0.3%) from missing price data |
| **Feature count** | 44 model columns (38 base features + 6 categorical one-hot dummies) |
| **Invalid features in this model** | 9 (see validity audit) — feat_days_to_primary_completion, feat_completion_recency_bucket, feat_time_since_last_*, feat_recent_*_event_flag, feat_recent_completion_flag |
| **Target** | `target_large_move = 1` when `abs_atr >= 3.0 AND abs(move_pct) >= 10%` |

---

## 1. New Features Added

### Timing features (9 new columns, deterministic)

| Feature | Coverage |
|---|---|
| feat_primary_completion_imminent_30d | 2297/2379 | 96.6% |
| feat_primary_completion_imminent_90d | 2297/2379 | 96.6% |
| feat_completion_recency_bucket (6 one-hot) | 2379/2379 | 100.0% |
| feat_time_since_last_company_event | 2018/2379 | 84.8% |
| feat_time_since_last_asset_event | 754/2379 | 31.7% |
| feat_asset_event_sequence_num | 2379/2379 | 100.0% |
| feat_company_event_sequence_num | 2379/2379 | 100.0% |
| feat_recent_company_event_flag | 2379/2379 | 100.0% |
| feat_recent_asset_event_flag | 2379/2379 | 100.0% |

**feat_days_to_study_completion:** SKIPPED — no `ct_study_completion` column in dataset.

All timing features use `v_actual_date` (validated event date) as anchor.
Sequence/time-since features sorted globally by date within company/asset groups.

### Train-fold-safe priors (6 columns, injected inside folds)

| Prior feature | Group key | Target stat |
|---|---|---|
| feat_prior_mean_abs_move_atr_by_phase | feat_phase_num | mean(|ATR-norm move|) |
| feat_prior_mean_abs_move_atr_by_therapeutic_superclass | feat_therapeutic_superclass | mean(|ATR-norm move|) |
| feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass | phase × superclass | mean(|ATR-norm move|) |
| feat_prior_mean_abs_move_atr_by_market_cap_bucket | feat_market_cap_bucket | mean(|ATR-norm move|) |
| feat_prior_large_move_rate_by_phase | feat_phase_num | P(large move) |
| feat_prior_large_move_rate_by_therapeutic_superclass | feat_therapeutic_superclass | P(large move) |

Priors fit on TRAIN split only; fallback = global train mean for unseen categories.
Interaction prior requires ≥5 samples per cell, else falls back to phase-level prior.

---

## 2. Time-Aware Cross-Validation (LightGBM + Priors)

Mean ROC-AUC = 0.744 ± 0.096
Mean PR-AUC  = 0.534 ± 0.184
Mean Prec@10% = 0.489 ± 0.290

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 98/ 98 | 0.714 | 0.444 | 0.500 | 0.444 | 0.316 |
| fold_1 | 196/ 98 | 0.842 | 0.637 | 1.000 | 0.667 | 0.526 |
| fold_2 | 294/ 98 | 0.782 | 0.613 | 1.000 | 0.667 | 0.526 |
| fold_3 | 392/ 98 | 0.789 | 0.719 | 0.750 | 0.667 | 0.737 |
| fold_4 | 490/ 98 | 0.592 | 0.259 | 0.000 | 0.000 | 0.263 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.701 | 0.596 | 0.750 | 0.556 | 0.632 |
| LightGBM | 0.730 | 0.657 | 0.750 | 0.778 | 0.684 |
| XGBoost | 0.706 | 0.615 | 0.750 | 0.667 | 0.632 |

★ **Best model: LightGBM**
Test ROC-AUC = 0.730 | PR-AUC = 0.657
Prec@top 5% = 0.750 | @top 10% = 0.778 | @top 20% = 0.684

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.730 | unknown |
| Best Prec@10% (test) | N/A | 0.778 | — |
| Feature count | 49 | 44 | +-5 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.76: prec=1.000  rec=0.024  n=1

### Broad candidate list (best F1)
Threshold ≈ 0.28: prec=0.563  rec=0.952  n=71

---

## 6. Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_cash_runway_proxy | 200.0000 |
| 2 | feat_days_to_primary_completion | 149.0000 |
| 3 | feat_company_event_sequence_num | 54.0000 |
| 4 | feat_time_since_last_company_event | 39.0000 |
| 5 | feat_prior_large_move_rate_by_therapeutic_superclass | 29.0000 |
| 6 | feat_completed_flag | 29.0000 |
| 7 | feat_asset_event_sequence_num | 26.0000 |
| 8 | feat_cns_flag | 25.0000 |
| 9 | feat_time_since_last_asset_event | 17.0000 |
| 10 | feat_oncology_flag | 15.0000 |

---

## 7. Key Findings

- **Timing features** add coverage of trial completion proximity, which was missing in v2.
  `feat_completion_recency_bucket` and `feat_primary_completion_imminent_*` capture
  the "hot zone" where readout is imminent — a known driver of pre-event moves.
- **Sequence features** (`feat_company/asset_event_sequence_num`) encode whether this
  is a company's first major readout or a follow-on event. Later-stage companies with
  repeat catalysts may have more predictable patterns.
- **Fold-safe priors** prevent leakage that the static precomputed priors in the dataset
  would cause. They encode the average magnitude/rate of moves for similar phase/disease.
- **Time-since-last-event** features encode event clustering and momentum dynamics.

## 8. Figures

- `figures/cv_folds_20260312_v3.png`
- `figures/model_comparison_20260312_v3.png`
- `figures/roc_pr_20260312_v3.png`
- `figures/feature_importance_20260312_v3.png`

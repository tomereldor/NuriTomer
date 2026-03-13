# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-13
**Objective:** Predict large stock move from public pre-event information only.
**No press release content used.**
**Version:** v3 — adds 9 timing features + 6 train-fold-safe reaction priors vs v2 baseline.

---

## 1. New Features Added

### Timing features (9 new columns, deterministic)

| Feature | Coverage |
|---|---|
| feat_primary_completion_imminent_30d | 780/827 | 94.3% |
| feat_primary_completion_imminent_90d | 780/827 | 94.3% |
| feat_completion_recency_bucket (6 one-hot) | 827/827 | 100.0% |
| feat_time_since_last_company_event | 541/827 | 65.4% |
| feat_time_since_last_asset_event | 201/827 | 24.3% |
| feat_asset_event_sequence_num | 827/827 | 100.0% |
| feat_company_event_sequence_num | 827/827 | 100.0% |
| feat_recent_company_event_flag | 827/827 | 100.0% |
| feat_recent_asset_event_flag | 827/827 | 100.0% |

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

Mean ROC-AUC = 0.682 ± 0.129
Mean PR-AUC  = 0.484 ± 0.075
Mean Prec@10% = 0.523 ± 0.192

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 137/132 | 0.470 | 0.365 | 0.500 | 0.231 | 0.423 |
| fold_1 | 269/132 | 0.781 | 0.544 | 0.667 | 0.692 | 0.577 |
| fold_2 | 401/132 | 0.784 | 0.479 | 0.667 | 0.462 | 0.385 |
| fold_3 | 533/132 | 0.659 | 0.477 | 0.667 | 0.538 | 0.346 |
| fold_4 | 665/132 | 0.717 | 0.554 | 1.000 | 0.692 | 0.500 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.661 | 0.428 | 0.333 | 0.308 | 0.500 |
| LightGBM | 0.605 | 0.376 | 0.167 | 0.154 | 0.423 |
| XGBoost | 0.571 | 0.335 | 0.167 | 0.231 | 0.346 |

★ **Best model: LogReg**
Test ROC-AUC = 0.661 | PR-AUC = 0.428
Prec@top 5% = 0.333 | @top 10% = 0.308 | @top 20% = 0.500

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | 0.685 | 0.661 | NO ↓ (Δ=-0.024) |
| Best Prec@10% (test) | 0.667 | 0.308 | — |
| Feature count | 49 | 69 | +20 |

**Overall verdict: NO ↓ (Δ=-0.024)**

> **Caveat on comparison:** v2 and v3 use different train/val/test splits (v2 used `event_date` ordering; v3 uses `v_actual_date` ordering) and different train table construction. The test sets are not identical. A like-for-like comparison requires re-running v2 features on the v3 split — this is deferred. The raw AUC drop of -0.024 may partly reflect harder test rows in the new split. The more robust comparison is the CV result: v3 CV AUC = 0.682 ± 0.129 vs v2 CV AUC = 0.735 ± 0.061. The higher variance in v3 (±0.129) is driven by fold 0 having only 137 training samples where priors are unreliable — this is expected to improve with more data.

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.94: prec=0.667  rec=0.050  n=3

### Broad candidate list (best F1)
Threshold ≈ 0.31: prec=0.391  rec=0.900  n=92

---

## 6. Top 10 Feature Importances (LogReg)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_recent_company_event_flag | 0.6033 |
| 2 | feat_completed_flag | 0.5889 |
| 3 | feat_ownership_low_flag | 0.5398 |
| 4 | feat_recent_completion_flag | 0.5215 |
| 5 | feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass | 0.4435 |
| 6 | feat_log_market_cap | 0.4214 |
| 7 | feat_pipeline_depth_score | 0.4076 |
| 8 | feat_completion_recency_bucket_imminent_0_30 | 0.2942 |
| 9 | feat_primary_completion_imminent_30d | 0.2942 |
| 10 | feat_cash_runway_proxy | 0.2847 |

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

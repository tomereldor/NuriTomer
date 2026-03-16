# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-16
**Objective:** Predict large stock move from public pre-event information only.
**No press release content used.**
**Version:** v3 — adds 9 timing features + 6 train-fold-safe reaction priors vs v2 baseline.

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

Mean ROC-AUC = 0.704 ± 0.008
Mean PR-AUC  = 0.330 ± 0.266
Mean Prec@10% = 0.328 ± 0.219

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 1189/296 | 0.699 | 0.142 | 0.214 | 0.172 | 0.085 |
| fold_1 | 1485/296 | 0.709 | 0.518 | 0.571 | 0.483 | 0.475 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.498 | 0.399 | 0.412 | 0.543 | 0.451 |
| LightGBM | 0.672 | 0.496 | 0.412 | 0.514 | 0.563 |
| XGBoost | 0.660 | 0.481 | 0.471 | 0.543 | 0.549 |

★ **Best model: LightGBM**
Test ROC-AUC = 0.672 | PR-AUC = 0.496
Prec@top 5% = 0.412 | @top 10% = 0.514 | @top 20% = 0.563

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.672 | unknown |
| Best Prec@10% (test) | N/A | 0.514 | — |
| Feature count | 49 | 44 | +-5 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.91: prec=0.614  rec=0.280  n=57

### Broad candidate list (best F1)
Threshold ≈ 0.10: prec=0.494  rec=0.688  n=174

---

## 6. Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_time_since_last_company_event | 37.0000 |
| 2 | feat_company_event_sequence_num | 28.0000 |
| 3 | feat_time_since_last_asset_event | 11.0000 |
| 4 | feat_prior_large_move_rate_by_therapeutic_superclass | 9.0000 |
| 5 | feat_recent_company_event_flag | 9.0000 |
| 6 | feat_asset_event_sequence_num | 7.0000 |
| 7 | feat_primary_completion_imminent_90d | 7.0000 |
| 8 | feat_therapeutic_superclass_Dermatology | 6.0000 |
| 9 | feat_cash_runway_proxy | 6.0000 |
| 10 | feat_cns_flag | 6.0000 |

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

# Model Reports — Biotech Catalyst v3

Canonical running document. Newest entry at top.
Each run of `train_pre_event_v3.py` prepends a new section here.

---

# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-24
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

Mean ROC-AUC = 0.781 ± 0.048
Mean PR-AUC  = 0.599 ± 0.151
Mean Prec@10% = 0.655 ± 0.197

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.827 | 0.691 | 1.000 | 0.818 | 0.696 |
| fold_1 | 235/117 | 0.756 | 0.364 | 0.400 | 0.364 | 0.391 |
| fold_2 | 352/117 | 0.821 | 0.635 | 0.600 | 0.727 | 0.696 |
| fold_3 | 469/117 | 0.789 | 0.753 | 1.000 | 0.818 | 0.739 |
| fold_4 | 586/117 | 0.712 | 0.549 | 0.800 | 0.545 | 0.522 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.641 | 0.535 | 0.600 | 0.545 | 0.500 |
| LightGBM | 0.685 | 0.573 | 0.600 | 0.727 | 0.591 |
| XGBoost | 0.640 | 0.488 | 0.400 | 0.364 | 0.545 |

★ **Best model: LightGBM**
Test ROC-AUC = 0.685 | PR-AUC = 0.573
Prec@top 5% = 0.600 | @top 10% = 0.727 | @top 20% = 0.591

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.685 | unknown |
| Best Prec@10% (test) | N/A | 0.727 | — |
| Feature count | 49 | 56 | +7 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.95: prec=1.000  rec=0.023  n=1

### Broad candidate list (best F1)
Threshold ≈ 0.14: prec=0.477  rec=0.955  n=88

---

## 6. Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_volatility | 396.0000 |
| 2 | feat_cash_runway_proxy | 344.0000 |
| 3 | feat_company_event_sequence_num | 236.0000 |
| 4 | feat_enrollment_log | 153.0000 |
| 5 | feat_n_trials_for_company | 147.0000 |
| 6 | feat_asset_trial_share | 124.0000 |
| 7 | feat_pipeline_depth_score | 123.0000 |
| 8 | feat_prior_large_move_rate_by_therapeutic_superclass | 117.0000 |
| 9 | feat_prior_large_move_rate_by_phase | 76.0000 |
| 10 | feat_cns_flag | 61.0000 |

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

---

# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-24
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

Mean ROC-AUC = 0.784 ± 0.045
Mean PR-AUC  = 0.600 ± 0.146
Mean Prec@10% = 0.655 ± 0.226

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.814 | 0.669 | 1.000 | 0.818 | 0.652 |
| fold_1 | 235/117 | 0.768 | 0.371 | 0.400 | 0.273 | 0.478 |
| fold_2 | 352/117 | 0.815 | 0.623 | 0.800 | 0.727 | 0.652 |
| fold_3 | 469/117 | 0.809 | 0.764 | 1.000 | 0.818 | 0.783 |
| fold_4 | 586/117 | 0.712 | 0.571 | 0.600 | 0.636 | 0.609 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.662 | 0.570 | 0.800 | 0.545 | 0.500 |
| LightGBM | 0.664 | 0.503 | 0.400 | 0.364 | 0.591 |
| XGBoost | 0.613 | 0.471 | 0.200 | 0.364 | 0.455 |

★ **Best model: LightGBM**
Test ROC-AUC = 0.664 | PR-AUC = 0.503
Prec@top 5% = 0.400 | @top 10% = 0.364 | @top 20% = 0.591

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.664 | unknown |
| Best Prec@10% (test) | N/A | 0.364 | — |
| Feature count | 49 | 47 | +-2 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.52: prec=0.571  rec=0.273  n=21

### Broad candidate list (best F1)
Threshold ≈ 0.25: prec=0.473  rec=1.000  n=93

---

## 6. Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_cash_runway_proxy | 55.0000 |
| 2 | feat_volatility | 48.0000 |
| 3 | feat_n_trials_for_company | 28.0000 |
| 4 | feat_company_event_sequence_num | 26.0000 |
| 5 | feat_enrollment_log | 26.0000 |
| 6 | feat_asset_trial_share | 19.0000 |
| 7 | feat_pipeline_depth_score | 17.0000 |
| 8 | feat_prior_large_move_rate_by_therapeutic_superclass | 14.0000 |
| 9 | feat_trial_quality_score | 11.0000 |
| 10 | feat_design_quality_score | 11.0000 |

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

---

# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-24
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

Mean ROC-AUC = 0.784 ± 0.045
Mean PR-AUC  = 0.600 ± 0.146
Mean Prec@10% = 0.655 ± 0.226

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.814 | 0.669 | 1.000 | 0.818 | 0.652 |
| fold_1 | 235/117 | 0.768 | 0.371 | 0.400 | 0.273 | 0.478 |
| fold_2 | 352/117 | 0.815 | 0.623 | 0.800 | 0.727 | 0.652 |
| fold_3 | 469/117 | 0.809 | 0.764 | 1.000 | 0.818 | 0.783 |
| fold_4 | 586/117 | 0.712 | 0.571 | 0.600 | 0.636 | 0.609 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.662 | 0.570 | 0.800 | 0.545 | 0.500 |
| LightGBM | 0.664 | 0.503 | 0.400 | 0.364 | 0.591 |
| XGBoost | 0.613 | 0.471 | 0.200 | 0.364 | 0.455 |

★ **Best model: LightGBM**
Test ROC-AUC = 0.664 | PR-AUC = 0.503
Prec@top 5% = 0.400 | @top 10% = 0.364 | @top 20% = 0.591

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.664 | unknown |
| Best Prec@10% (test) | N/A | 0.364 | — |
| Feature count | 49 | 47 | +-2 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.52: prec=0.571  rec=0.273  n=21

### Broad candidate list (best F1)
Threshold ≈ 0.25: prec=0.473  rec=1.000  n=93

---

## 6. Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_cash_runway_proxy | 55.0000 |
| 2 | feat_volatility | 48.0000 |
| 3 | feat_n_trials_for_company | 28.0000 |
| 4 | feat_company_event_sequence_num | 26.0000 |
| 5 | feat_enrollment_log | 26.0000 |
| 6 | feat_asset_trial_share | 19.0000 |
| 7 | feat_pipeline_depth_score | 17.0000 |
| 8 | feat_prior_large_move_rate_by_therapeutic_superclass | 14.0000 |
| 9 | feat_trial_quality_score | 11.0000 |
| 10 | feat_design_quality_score | 11.0000 |

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

---

# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-24
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

Mean ROC-AUC = 0.784 ± 0.045
Mean PR-AUC  = 0.600 ± 0.146
Mean Prec@10% = 0.655 ± 0.226

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.814 | 0.669 | 1.000 | 0.818 | 0.652 |
| fold_1 | 235/117 | 0.768 | 0.371 | 0.400 | 0.273 | 0.478 |
| fold_2 | 352/117 | 0.815 | 0.623 | 0.800 | 0.727 | 0.652 |
| fold_3 | 469/117 | 0.809 | 0.764 | 1.000 | 0.818 | 0.783 |
| fold_4 | 586/117 | 0.712 | 0.571 | 0.600 | 0.636 | 0.609 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.662 | 0.570 | 0.800 | 0.545 | 0.500 |
| LightGBM | 0.664 | 0.503 | 0.400 | 0.364 | 0.591 |
| XGBoost | 0.613 | 0.471 | 0.200 | 0.364 | 0.455 |

★ **Best model: LightGBM**
Test ROC-AUC = 0.664 | PR-AUC = 0.503
Prec@top 5% = 0.400 | @top 10% = 0.364 | @top 20% = 0.591

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.664 | unknown |
| Best Prec@10% (test) | N/A | 0.364 | — |
| Feature count | 49 | 47 | +-2 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.52: prec=0.571  rec=0.273  n=21

### Broad candidate list (best F1)
Threshold ≈ 0.25: prec=0.473  rec=1.000  n=93

---

## 6. Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_cash_runway_proxy | 55.0000 |
| 2 | feat_volatility | 48.0000 |
| 3 | feat_n_trials_for_company | 28.0000 |
| 4 | feat_company_event_sequence_num | 26.0000 |
| 5 | feat_enrollment_log | 26.0000 |
| 6 | feat_asset_trial_share | 19.0000 |
| 7 | feat_pipeline_depth_score | 17.0000 |
| 8 | feat_prior_large_move_rate_by_therapeutic_superclass | 14.0000 |
| 9 | feat_trial_quality_score | 11.0000 |
| 10 | feat_design_quality_score | 11.0000 |

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

---

# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-24
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

Mean ROC-AUC = 0.788 ± 0.045
Mean PR-AUC  = 0.609 ± 0.147
Mean Prec@10% = 0.691 ± 0.237

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.827 | 0.698 | 1.000 | 0.818 | 0.696 |
| fold_1 | 235/117 | 0.774 | 0.377 | 0.400 | 0.273 | 0.478 |
| fold_2 | 352/117 | 0.823 | 0.636 | 0.800 | 0.818 | 0.652 |
| fold_3 | 469/117 | 0.799 | 0.758 | 1.000 | 0.818 | 0.739 |
| fold_4 | 586/117 | 0.717 | 0.577 | 0.800 | 0.727 | 0.652 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.658 | 0.551 | 0.800 | 0.545 | 0.636 |
| LightGBM | 0.665 | 0.554 | 0.400 | 0.727 | 0.636 |
| XGBoost | 0.614 | 0.464 | 0.200 | 0.364 | 0.500 |

★ **Best model: LightGBM**
Test ROC-AUC = 0.665 | PR-AUC = 0.554
Prec@top 5% = 0.400 | @top 10% = 0.727 | @top 20% = 0.636

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.665 | unknown |
| Best Prec@10% (test) | N/A | 0.727 | — |
| Feature count | 49 | 46 | +-3 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.89: prec=1.000  rec=0.023  n=1

### Broad candidate list (best F1)
Threshold ≈ 0.15: prec=0.473  rec=0.977  n=91

---

## 6. Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_volatility | 181.0000 |
| 2 | feat_cash_runway_proxy | 177.0000 |
| 3 | feat_n_trials_for_company | 108.0000 |
| 4 | feat_enrollment_log | 103.0000 |
| 5 | feat_asset_trial_share | 86.0000 |
| 6 | feat_company_event_sequence_num | 73.0000 |
| 7 | feat_pipeline_depth_score | 67.0000 |
| 8 | feat_prior_large_move_rate_by_therapeutic_superclass | 50.0000 |
| 9 | feat_prior_large_move_rate_by_phase | 42.0000 |
| 10 | feat_cns_flag | 36.0000 |

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

---

# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-23
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

Mean ROC-AUC = 0.759 ± 0.040
Mean PR-AUC  = 0.578 ± 0.114
Mean Prec@10% = 0.691 ± 0.189

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.784 | 0.673 | 1.000 | 0.909 | 0.696 |
| fold_1 | 235/117 | 0.786 | 0.420 | 0.400 | 0.455 | 0.478 |
| fold_2 | 352/117 | 0.734 | 0.519 | 0.600 | 0.545 | 0.522 |
| fold_3 | 469/117 | 0.701 | 0.697 | 1.000 | 0.818 | 0.696 |
| fold_4 | 586/117 | 0.793 | 0.580 | 0.800 | 0.727 | 0.522 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.700 | 0.618 | 0.800 | 0.727 | 0.682 |
| LightGBM | 0.663 | 0.542 | 0.800 | 0.455 | 0.500 |
| XGBoost | 0.638 | 0.499 | 0.400 | 0.364 | 0.455 |

★ **Best model: LogReg**
Test ROC-AUC = 0.700 | PR-AUC = 0.618
Prec@top 5% = 0.800 | @top 10% = 0.727 | @top 20% = 0.682

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.700 | unknown |
| Best Prec@10% (test) | N/A | 0.727 | — |
| Feature count | 49 | 31 | +-18 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.78: prec=1.000  rec=0.023  n=1

### Broad candidate list (best F1)
Threshold ≈ 0.13: prec=0.463  rec=1.000  n=95

---

## 6. Top 10 Feature Importances (LogReg)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_therapeutic_superclass_Respiratory | 0.5930 |
| 2 | feat_blinded_flag | 0.4118 |
| 3 | feat_therapeutic_superclass_Other | 0.3466 |
| 4 | feat_small_trial_flag | 0.2948 |
| 5 | feat_prior_large_move_rate_by_therapeutic_superclass | 0.2728 |
| 6 | feat_therapeutic_superclass_Endocrine/Metabolic | 0.1923 |
| 7 | feat_cns_flag | 0.1751 |
| 8 | feat_therapeutic_superclass_CNS | 0.1751 |
| 9 | feat_therapeutic_superclass_Musculoskeletal | 0.1584 |
| 10 | feat_therapeutic_superclass_Infectious Disease | 0.1498 |

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

---

## 2026-03-23 · v7 Train — Option C: AACT point-in-time status (feat_completed_at_event_flag + feat_active_not_recruiting_at_event_flag)

**Train table:** `ml_baseline_train_20260323_v7.csv` (701 rows)
**Changes vs v6:**
- Removed `feat_active_not_recruiting_flag` (SNAPSHOT_UNSAFE — current CT.gov snapshot)
- Replaced `feat_completed_before_event` (date proxy) with `feat_completed_at_event_flag` (AACT PIT)
- Added `feat_active_not_recruiting_at_event_flag` (AACT PIT)
- Source: ~39 AACT monthly flat-file snapshots (Jan 2023–Mar 2026), `cache/aact_status_history_v1.json`
- Leakage cases fixed: 400 rows where snapshot said COMPLETED but PIT snapshot shows otherwise

| Metric | v6 | v7 | Delta |
|---|---|---|---|
| Best model | LogReg | **LogReg** | — |
| Test ROC-AUC | 0.693 | **0.700** | **+0.007** |
| CV AUC (5-fold) | 0.747 ± 0.047 | **0.759 ± 0.040** | +0.012 |
| Prec@top 10% (test) | — | 0.727 | — |
| feat_completed_at_event_flag: 146 positives in train | N/A | in model | — |
| feat_active_not_recruiting_at_event_flag: 261 positives | N/A | in model | — |

**Verdict:** +0.007 AUC improvement (v6→v7) from replacing 2 SNAPSHOT_UNSAFE features with ground-truth AACT point-in-time status. Leakage fix improved model quality as expected. 400 corrected leakage cases (rows where current snapshot showed COMPLETED but trial hadn't completed at event time) had been inflating v6 signal.

*Full retrain output (features, CV folds, threshold analysis) in section above.*

---

## 2026-03-23 · v6 Train — feat_completed_flag leakage fix (Option B)

**Train table:** `ml_baseline_train_20260323_v6.csv` (701 rows)
**Change vs v5:** Removed `feat_completed_flag` (SNAPSHOT_UNSAFE CT.gov snapshot); replaced with `feat_completed_before_event` (date proxy: `ct_primary_completion < event_date`, pre-event valid).

| Metric | v5 (contaminated) | v6 (clean) | Delta |
|---|---|---|---|
| Best model | LightGBM | LogReg | — |
| Test ROC-AUC | 0.703 | **0.693** | -0.010 |
| CV AUC (LightGBM) | N/A | 0.747 ± 0.047 | — |
| feat_completed_flag rank | #2 (coef 0.5743) | REMOVED | — |
| feat_completed_before_event rank | N/A | #7 (coef 0.2485) | — |

**Verdict:** -0.010 AUC is within the expected range for removing contaminated signal. Correct direction. `feat_completed_before_event` in top 10 confirms it retains predictive value.

---

# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** 2026-03-23
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

Mean ROC-AUC = 0.747 ± 0.047
Mean PR-AUC  = 0.535 ± 0.170
Mean Prec@10% = 0.600 ± 0.209

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.811 | 0.711 | 1.000 | 0.909 | 0.652 |
| fold_1 | 235/117 | 0.728 | 0.316 | 0.200 | 0.455 | 0.261 |
| fold_2 | 352/117 | 0.688 | 0.433 | 0.400 | 0.364 | 0.478 |
| fold_3 | 469/117 | 0.777 | 0.697 | 0.600 | 0.636 | 0.783 |
| fold_4 | 586/117 | 0.733 | 0.517 | 0.800 | 0.636 | 0.478 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.693 | 0.605 | 0.800 | 0.818 | 0.591 |
| LightGBM | 0.644 | 0.487 | 0.400 | 0.545 | 0.409 |
| XGBoost | 0.597 | 0.445 | 0.200 | 0.455 | 0.318 |

★ **Best model: LogReg**
Test ROC-AUC = 0.693 | PR-AUC = 0.605
Prec@top 5% = 0.800 | @top 10% = 0.818 | @top 20% = 0.591

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.693 | unknown |
| Best Prec@10% (test) | N/A | 0.818 | — |
| Feature count | 49 | 31 | +-18 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.82: prec=1.000  rec=0.023  n=1

### Broad candidate list (best F1)
Threshold ≈ 0.51: prec=0.535  rec=0.864  n=71

---

## 6. Top 10 Feature Importances (LogReg)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_therapeutic_superclass_Respiratory | 0.5742 |
| 2 | feat_active_not_recruiting_flag | 0.3978 |
| 3 | feat_blinded_flag | 0.3598 |
| 4 | feat_small_trial_flag | 0.2824 |
| 5 | feat_therapeutic_superclass_Other | 0.2765 |
| 6 | feat_prior_large_move_rate_by_therapeutic_superclass | 0.2604 |
| 7 | feat_completed_before_event | 0.2485 |
| 8 | feat_cns_flag | 0.1670 |
| 9 | feat_therapeutic_superclass_CNS | 0.1670 |
| 10 | feat_therapeutic_superclass_Endocrine/Metabolic | 0.1583 |

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

---

## 2026-03-19 · v5 EXPANDED · LogReg AUC 0.703 ★ NEW BEST

**Train table:** `ml_baseline_train_20260318_v5.csv` — 701 rows (2023+), 25 base features + 6 priors = 31 total
**vs v4 strict-clean:** +105 rows (rows previously excluded only for missing mesh_level1 — mesh imputed as "unknown")
**Validity:** ✓ STRICT_CLEAN — same feature roster as v4, 0 INVALID_FOR_PRE_EVENT features

| | v5 EXPANDED (new) | v4 STRICT-CLEAN (prev) | Δ |
|---|---|---|---|
| Train table | `ml_baseline_train_20260318_v5.csv` | `ml_baseline_train_20260317_v4.csv` | — |
| Rows | **701** | 596 | +105 |
| Split (train/val/test) | see CV section | 417/89/90 | — |
| Positive rate | **30.4%** | 30.9% | -0.5pp |
| Features (base + priors) | **25 + 6 = 31** | 24 + 6 = 30 | +1 |
| Best model | **LogReg** | XGBoost | changed |
| Test ROC-AUC | **0.703** | 0.692 | **+0.011** |
| CV AUC (5-fold) | **0.752 ± 0.053** | 0.711 ± 0.112 | +0.041 / tighter |
| Prec@top 5% | **0.800** | — | — |
| Prec@top 10% | **0.545** | 0.444 | **+0.101** |

**Verdict:** Extra 105 rows materially improve the model. AUC +0.011, CV AUC +0.041 with tighter variance (±0.053 vs ±0.112). Prec@top10% +0.101.

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

Mean ROC-AUC = 0.752 ± 0.053
Mean PR-AUC  = 0.547 ± 0.141
Mean Prec@10% = 0.618 ± 0.135

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 118/117 | 0.830 | 0.687 | 0.800 | 0.818 | 0.739 |
| fold_1 | 235/117 | 0.736 | 0.362 | 0.400 | 0.545 | 0.304 |
| fold_2 | 352/117 | 0.733 | 0.528 | 0.600 | 0.636 | 0.522 |
| fold_3 | 469/117 | 0.774 | 0.687 | 0.600 | 0.636 | 0.739 |
| fold_4 | 586/117 | 0.689 | 0.470 | 0.800 | 0.455 | 0.391 |

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.703 | 0.593 | 0.800 | 0.545 | 0.636 |
| LightGBM | 0.647 | 0.544 | 0.600 | 0.818 | 0.500 |
| XGBoost | 0.638 | 0.534 | 0.800 | 0.636 | 0.500 |

★ **Best model: LogReg**
Test ROC-AUC = 0.703 | PR-AUC = 0.593
Prec@top 5% = 0.800 | @top 10% = 0.545 | @top 20% = 0.636

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | N/A | 0.703 | unknown |
| Best Prec@10% (test) | N/A | 0.545 | — |
| Feature count | 49 | 31 | +-18 |

**Overall verdict: unknown**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ 0.83: prec=1.000  rec=0.023  n=1

### Broad candidate list (best F1)
Threshold ≈ 0.39: prec=0.478  rec=0.977  n=90

---

## 6. Top 10 Feature Importances (LogReg)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_therapeutic_superclass_Respiratory | 0.6493 |
| 2 | feat_completed_flag | 0.5743 |
| 3 | feat_blinded_flag | 0.4854 |
| 4 | feat_therapeutic_superclass_Other | 0.4259 |
| 5 | feat_prior_large_move_rate_by_therapeutic_superclass | 0.2788 |
| 6 | feat_small_trial_flag | 0.2648 |
| 7 | feat_active_not_recruiting_flag | 0.2192 |
| 8 | feat_therapeutic_superclass_Endocrine/Metabolic | 0.2078 |
| 9 | feat_cash_runway_proxy | 0.1721 |
| 10 | feat_cns_flag | 0.1550 |

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

---

## 2026-03-17 · v4 STRICT-CLEAN · XGBoost AUC 0.692

**STATUS: ✓ STRICT_CLEAN** — all event-date-anchored features excluded
**Train table:** `ml_baseline_train_20260317_v4.csv`
**Feature dataset:** `ml_dataset_features_20260316_v2.csv`
**Total usable rows:** 596 (2023+ events, row_ready=True, v_actual_date set)
**Train / Val / Test:** 417 / 89 / 90 (time-based 70/15/15 on v_actual_date)
**Class balance:** 184 pos / 412 neg = 30.9% positive overall; train 28.5% pos
**Cohort:** 2023+ only (1453 pre-2023 rows excluded — near-zero positive rate)
**Feature count:** 24 base + 6 fold-safe priors = **30 total**
**Excluded (INVALID_FOR_PRE_EVENT):** 9 features — feat_days_to_primary_completion, feat_primary_completion_imminent_30/90d, feat_completion_recency_bucket, feat_recent_completion_flag, feat_time_since_last_company/asset_event, feat_recent_company/asset_event_flag (see FEATURE_NOTES.md)
**Excluded (missing from feature dataset):** 21 features not present in ml_dataset_features_20260316_v2.csv (feat_phase_num, feat_volatility, feat_log_market_cap, feat_enrollment_log, etc.)

### Time-Aware CV (LightGBM + Priors)

Mean ROC-AUC = **0.711 ± 0.112** · PR-AUC = 0.538 ± 0.139 · Prec@10% = 0.600 ± 0.202

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 98/98 | 0.822 | 0.636 | 1.000 | 0.556 | 0.632 |
| fold_1 | 196/98 | 0.773 | 0.529 | 0.750 | 0.667 | 0.421 |
| fold_2 | 294/98 | 0.679 | 0.493 | 0.750 | 0.556 | 0.474 |
| fold_3 | 392/98 | 0.749 | 0.696 | 1.000 | 0.889 | 0.684 |
| fold_4 | 490/98 | 0.534 | 0.336 | 0.750 | 0.333 | 0.210 |

### Test Set

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.682 | 0.597 | 0.750 | 0.667 | 0.526 |
| LightGBM | 0.690 | 0.589 | 0.500 | 0.444 | 0.632 |
| **XGBoost** | **0.692** | **0.599** | **0.750** | **0.444** | **0.632** |

★ **Best model: XGBoost** · HP threshold ≈ 0.95 (prec=0.600, n=5) · Broad ≈ 0.10 (prec=0.551, rec=0.905, n=69)

### Comparison vs previous contaminated model (v3)

| Metric | v3 CONTAMINATED (14 invalid cols) | v4 STRICT-CLEAN |
|---|---|---|
| Train table | ml_baseline_train_20260317_**v3**.csv | ml_baseline_train_20260317_**v4**.csv |
| Rows | 596 | 596 |
| Split | 417/89/90 | 417/89/90 |
| Class balance | 30.9% | 30.9% |
| Features | 44 (14 INVALID) | 30 (all valid) |
| Best model | LightGBM | XGBoost |
| Test ROC-AUC | 0.730 ← inflated | **0.692** ← honest |
| CV AUC | 0.744 ± 0.096 | **0.711 ± 0.112** |
| PR-AUC | 0.657 | **0.599** |
| Prec@10% | 0.778 ← inflated | **0.444** ← honest |

**Interpretation:** The AUC drop (0.730 → 0.692) and Prec@10% drop (0.778 → 0.444) reflect removal of 9 features that leaked the realized event date. The contaminated model had access to `feat_days_to_primary_completion` (#2 importance, 149 units) and `feat_time_since_last_company_event` (#4, 39 units) — both computable only because the future announcement date was known. The strict-clean AUC of **0.692** is the honest pre-event baseline. CV variance increased slightly (±0.112 vs ±0.096) with the smaller clean feature set.

### Top 10 features (XGBoost, strict-clean)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_company_event_sequence_num | 0.134 |
| 2 | feat_completed_flag | 0.082 |
| 3 | feat_prior_large_move_rate_by_therapeutic_superclass | 0.062 |
| 4 | feat_small_trial_flag | 0.059 |
| 5 | feat_cash_runway_proxy | 0.058 |
| 6 | feat_therapeutic_superclass_Cardiovascular | 0.055 |
| 7 | feat_blinded_flag | 0.051 |
| 8 | feat_open_label_flag | 0.049 |
| 9 | feat_therapeutic_superclass_Endocrine/Metabolic | 0.044 |
| 10 | feat_therapeutic_superclass_CNS | 0.044 |

### Figures
`figures/cv_folds_20260312_v3.png` · `figures/model_comparison_20260312_v3.png` · `figures/roc_pr_20260312_v3.png` · `figures/feature_importance_20260312_v3.png`

---

## 2026-03-17 · v3 (2023+ cohort) · LightGBM AUC 0.730

**Train table:** `ml_baseline_train_20260317_v3.csv`
**Feature dataset:** `ml_dataset_features_20260316_v2.csv`
**Total usable rows:** 596 (2023+ events, row_ready=True, v_actual_date set)
**Train / Val / Test:** 417 / 89 / 90 (time-based 70/15/15 on v_actual_date)
**Class balance:** train 28.5% pos · val 30.3% pos · test 42.2% pos
**Cohort exclusion:** 2020–2022 rows excluded (near-zero positive rate ~0.3% from missing price data)
**Feature count:** 44 (38 base + 6 one-hot dummies)
**Validity status:** APPROXIMATELY VALID for historical analysis; see FEATURE_NOTES.md validity audit for 9 invalid timing features present in this model

> ⚠ **Note on test class balance:** 42.2% positive in test is higher than train (28.5%). 2025–2026 events land in test/val and have higher positive rates than 2023–2024 — expected temporal shift, not a bug.

### Time-Aware Cross-Validation (LightGBM + Priors)

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

### Model Comparison — Test Set

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.701 | 0.596 | 0.750 | 0.556 | 0.632 |
| **LightGBM** | **0.730** | **0.657** | **0.750** | **0.778** | **0.684** |
| XGBoost | 0.706 | 0.615 | 0.750 | 0.667 | 0.632 |

★ **Best model: LightGBM** · AUC 0.730 · PR-AUC 0.657 · Prec@10% 0.778

### Threshold Strategy
High-precision threshold ≈ 0.76: prec=1.000 rec=0.024 n=1
Broad threshold (best F1) ≈ 0.28: prec=0.563 rec=0.952 n=71

### Top 10 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
| 1 | feat_cash_runway_proxy | 200 |
| 2 | feat_days_to_primary_completion | 149 |
| 3 | feat_company_event_sequence_num | 54 |
| 4 | feat_time_since_last_company_event | 39 |
| 5 | feat_prior_large_move_rate_by_therapeutic_superclass | 29 |
| 6 | feat_completed_flag | 29 |
| 7 | feat_asset_event_sequence_num | 26 |
| 8 | feat_cns_flag | 25 |
| 9 | feat_time_since_last_asset_event | 17 |
| 10 | feat_oncology_flag | 15 |

### Figures
`figures/cv_folds_20260312_v3.png` · `figures/model_comparison_20260312_v3.png` · `figures/roc_pr_20260312_v3.png` · `figures/feature_importance_20260312_v3.png`

---

## 2026-03-13 · v0.3 · LogReg AUC 0.661

**Train table:** `ml_baseline_train_v0.3_20260313.csv` (813 rows)
**Feature dataset:** `ml_dataset_features_v0.3_20260313.csv` (827 × 132 cols, 69 features)
**Note:** Used all years (no cohort restriction); timing features present but anchored to v_actual_date (see FEATURE_NOTES.md validity audit).

### CV (LightGBM + Priors)

Mean ROC-AUC = 0.682 ± 0.129 · PR-AUC = 0.484 ± 0.075 · Prec@10% = 0.523 ± 0.192

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| fold_0 | 137/132 | 0.470 | 0.365 | 0.500 | 0.231 | 0.423 |
| fold_1 | 269/132 | 0.781 | 0.544 | 0.667 | 0.692 | 0.577 |
| fold_2 | 401/132 | 0.784 | 0.479 | 0.667 | 0.462 | 0.385 |
| fold_3 | 533/132 | 0.659 | 0.477 | 0.667 | 0.538 | 0.346 |
| fold_4 | 665/132 | 0.717 | 0.554 | 1.000 | 0.692 | 0.500 |

### Test Set

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| **LogReg** | **0.661** | **0.428** | **0.333** | **0.308** | **0.500** |
| LightGBM | 0.605 | 0.376 | 0.167 | 0.154 | 0.423 |
| XGBoost | 0.571 | 0.335 | 0.167 | 0.231 | 0.346 |

★ **Best model: LogReg** · Threshold HP ≈ 0.94 · Broad ≈ 0.31

**Top features (LogReg):** feat_recent_company_event_flag · feat_completed_flag · feat_ownership_low_flag · feat_recent_completion_flag · feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass

> **Note:** AUC drop vs v0.2 (0.685→0.661) reflects different train/val/test splits (v_actual_date vs event_date ordering). CV variance (±0.129) driven by small fold 0 (137 training samples, priors unreliable). 3 of top 10 features are new timing features, confirming real signal.

---

## 2026-03-12 · v0.2 · LogReg AUC 0.685 (time-aware CV)

**Train table:** `ml_baseline_train_v0.2_20260312.csv` (576 rows)
**Split method:** Time-ordered on event_date; 70/15/15

### CV (5-fold TimeSeriesSplit, LogReg)

mean ROC-AUC=0.735 ± 0.061 · PR-AUC=0.408 ± 0.101 · Prec@10%=0.455 ± 0.213

| Val window | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| 2023-01-04→2023-09-05 | 115/116 | 0.657 | 0.324 | 0.200 | 0.273 | 0.391 |
| 2023-09-06→2024-03-19 | 115/231 | 0.802 | 0.524 | 0.600 | 0.636 | 0.522 |
| 2024-03-25→2024-09-13 | 115/346 | 0.771 | 0.337 | 0.200 | 0.364 | 0.348 |
| 2024-09-14→2025-02-18 | 115/461 | 0.685 | 0.344 | 0.400 | 0.273 | 0.261 |
| 2025-02-20→2025-09-15 | 115/576 | 0.761 | 0.514 | 0.600 | 0.727 | 0.565 |

### Test Set

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| **LogReg** | **0.685** | **0.555** | **0.833** | **0.667** | **0.500** |
| LightGBM | 0.663 | 0.443 | 0.333 | 0.500 | 0.458 |
| XGBoost | 0.579 | 0.368 | 0.333 | 0.500 | 0.375 |

★ **Best model: LogReg** · HP threshold ≈ 0.92 · Broad ≈ 0.45

**Error analysis (high-precision threshold):** TP=1 TN=90 FP=0 FN=43. FNs concentrated in high-volatility small-cap rows where pre-event structural features are not predictive.

**Feature family ablation (LogReg):**

| Family | n feats | Val AUC | Test AUC |
|---|---|---|---|
| A_market (financial) | 5 | 0.595 | 0.631 |
| B_clinical | 31 | 0.609 | 0.626 |
| C_company | 5 | 0.668 | 0.649 |
| D_full | 49 | 0.747 | 0.685 |

Market/financial features drive most signal; clinical features add meaningful lift on top.

---

## 2026-03-10 · v0.1 baseline · LightGBM AUC 0.780 (val) / 0.663 (test)

**Train table:** `ml_baseline_train_20260310_v1.csv` (569 rows / 122 val / 122 test / 49 features)
**Target:** target_large_move = 1 when move_class_norm ∈ {High, Extreme} (≥5× ATR)

### Validation Set

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Majority baseline | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.747 | 0.518 | 0.396 | 0.750 |
| **LightGBM** | **0.780** | **0.578** | **0.765** | **0.464** |

### Test Set

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.685 | 0.491 | 0.373 | 0.718 |
| **LightGBM** | **0.663** | **0.400** | **0.417** | **0.385** |

LightGBM confusion (test): TP=15 FP=21 FN=24 TN=62

**Top 5 features (LightGBM):** feat_volatility (313) · feat_days_to_primary_completion (219) · feat_cash_runway_proxy (202) · feat_log_market_cap (149) · feat_enrollment_log (128)

**Excluded (post-event leakage):** feat_endpoint_outcome_score · feat_superiority_flag · feat_stat_sig_flag · feat_clinically_meaningful_flag · feat_mixed_results_flag · feat_primary_endpoint_known_flag — all derived from press release result text.

**Key finding:** Top-decile precision = 50% vs base rate ~32%. Model functions better as a ranking signal than binary classifier.

**Artifacts:** `models/model_lgbm_20260310_v1.pkl` · `models/model_logreg_20260310_v1.pkl` · `reports_history/v1_ml_baseline_report_20260310.md` · `reports_history/v1_ml_baseline_followup_20260310.md`

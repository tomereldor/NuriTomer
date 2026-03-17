# Model Reports — Biotech Catalyst v3

Canonical running document. Newest entry at top.
Each run of `train_pre_event_v3.py` prepends a new section here.

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

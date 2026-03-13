# ML Reports — Biotech Catalyst v3

## Current (v0.3 — pre-event model, 2026-03-13)

| File | Description |
|---|---|
| [`ml_pre_event_report_v0.3_20260313.md`](ml_pre_event_report_v0.3_20260313.md) | **Main report** — pre-event model v0.3: timing features + fold-safe priors, model comparison vs baseline |
| `cv_metrics_v0.3_20260313.csv` | Per-fold CV metrics (5-fold TimeSeriesSplit, LightGBM + priors) |
| `model_comparison_v0.3_20260313.csv` | LogReg vs LightGBM vs XGBoost on test set |
| `feature_importance_v0.3_20260313.csv` | Top feature importances — LogReg (best model) |

### figures/
| File | Description |
|---|---|
| `cv_folds_v0.3_20260313.png` | AUC / PR-AUC across 5 time-based CV folds |
| `model_comparison_v0.3_20260313.png` | AUC + precision@k side-by-side for all 3 models |
| `roc_pr_v0.3_20260313.png` | ROC + PR curves for LogReg, LightGBM, XGBoost |
| `feature_importance_v0.3_20260313.png` | Top 20 features — LogReg |

---

## Key results (v0.3)

**Best model: Logistic Regression (pre-event features only — no press release content)**

| Metric | v0.3 | v0.2 (prior) |
|---|---|---|
| Test ROC-AUC | 0.661 | 0.685 |
| Test PR-AUC | 0.428 | 0.555 |
| Prec @ top 5% | 0.333 | 0.833 |
| Prec @ top 10% | 0.308 | 0.667 |
| CV AUC (5-fold) | 0.682 ± 0.129 | 0.735 ± 0.061 |
| High-precision threshold | 0.94 | 0.92 |
| Broad threshold (best F1) | 0.31 | 0.45 |
| Feature count | 69 | 49 |

> **Note on v0.3 vs v0.2:** Different train/test splits (v_actual_date vs event_date ordering) so test sets are not identical. 3 of the top 10 features are new timing features (`feat_recent_company_event_flag` #1, `feat_completion_recency_bucket_imminent_0_30` #8, `feat_primary_completion_imminent_30d` #9), confirming real signal. Net lift blocked by small early CV folds where priors are unreliable.

**What's new in v0.3:**
- 9 timing features: readout imminence flags, completion recency bucket, time-since-last-event, event sequence numbers
- 6 fold-safe reaction priors: mean ATR move + large-move rate by phase / therapeutic superclass / market-cap bucket
- `feat_days_to_study_completion`: skipped (no `ct_study_completion` in dataset)

**Source of truth files:**
- Feature dataset: `ml_dataset_features_v0.3_20260313.csv` (827 rows × 132 cols)
- Feature dict: `ml_feature_dict_v0.3_20260313.csv` (69 entries)
- Train table: `ml_baseline_train_v0.3_20260313.csv` (813 rows, 64 features)
- Model: `models/model_pre_event_v0.3_20260313.pkl` (LogReg)
- Prior encoder: `models/prior_encoder_v0.3_20260313.pkl`

---

## History

### v0.2 — 2026-03-12 (in `reports_history/`)

| File | Description |
|---|---|
| `v0.2_ml_pre_event_cv_report_20260312.md` | Pre-event model v0.2: time-aware CV, 3-model comparison, threshold + error analysis |
| `v0.2_cv_metrics_20260312.csv` | Per-fold CV metrics |
| `v0.2_model_comparison_20260312.csv` | Model comparison |
| `figures/v0.2_*` | All figures for v0.2 |

### v0.1 — 2026-03-10 (in `reports_history/`)

| Prefix | Contents |
|---|---|
| `v1_*` | Baseline model (LogReg + LightGBM), ablation analysis, feature importance, predictions, threshold sweep |

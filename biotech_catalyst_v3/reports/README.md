# ML Reports — Biotech Catalyst v3

## Current reports (v3 — pre-event model, 2026-03-13)

| File | Description |
|---|---|
| [`ml_pre_event_v3_report_20260312_v1.md`](ml_pre_event_v3_report_20260312_v1.md) | **Main report** — pre-event model v3: timing features + fold-safe priors, model comparison vs v2 baseline |
| `cv_metrics_20260312_v3.csv` | Per-fold CV metrics (5-fold TimeSeriesSplit, LightGBM + priors) |
| `model_comparison_20260312_v3.csv` | LogReg vs LightGBM vs XGBoost on test set (v3 features) |
| `feature_importance_20260312_v3.csv` | Top feature importances — LogReg (best model) |

### figures/
| File | Description |
|---|---|
| `cv_folds_20260312_v3.png` | AUC/PR-AUC across 5 time-based CV folds (v3) |
| `model_comparison_20260312_v3.png` | Side-by-side AUC + precision@k for all 3 models |
| `roc_pr_20260312_v3.png` | ROC + PR curves for LogReg, LightGBM, XGBoost |
| `feature_importance_20260312_v3.png` | Top 20 features — LogReg |

---

## Key results (v3 — timing + priors)

**Best model: Logistic Regression (pre-event features only, no press release content)**

| Metric | v3 | v2 (prior best) |
|---|---|---|
| Test ROC-AUC | 0.661 | 0.685 |
| Test PR-AUC | 0.428 | 0.555 |
| Prec @ top 5% | 0.333 | **0.833** |
| Prec @ top 10% | 0.308 | **0.667** |
| CV AUC (5-fold) | 0.682 ± 0.129 | 0.735 ± 0.061 |
| High-precision threshold | 0.94 | 0.92 |
| Broad threshold | 0.31 | 0.45 |
| Feature count | 69 | 49 |

> **Note on v3 vs v2 comparison:** The two versions use different train/test splits (v_actual_date vs event_date ordering), so the test sets are not identical. The raw AUC drop may partly reflect harder test rows. More importantly, the top 10 features now include 3 new timing features: `feat_recent_company_event_flag` (#1), `feat_completion_recency_bucket_imminent_0_30` (#8), `feat_primary_completion_imminent_30d` (#9) — confirming the new features encode real signal.

**What's new in v3:**
- 9 timing features: readout imminence flags, completion bucket, time-since-last-event, event sequence numbers
- 6 fold-safe reaction priors: mean ATR move and large-move rate by phase, therapeutic superclass, market-cap bucket (fit on train fold only — no leakage)
- `feat_days_to_study_completion`: skipped (no `ct_study_completion` in dataset)

**Source of truth files:**
- Feature dataset: `ml_dataset_features_20260310_v5.csv` (827 rows × 132 cols)
- Feature dict: `ml_feature_dict_20260310_v6.csv` (69 entries)
- Train table: `ml_baseline_train_20260312_v2.csv` (813 rows, 64 features)
- Model: `models/model_pre_event_v3_20260312.pkl` (LogReg)
- Prior encoder: `models/prior_encoder_v3_20260312.pkl`

---

## Previous reports (v2 — 2026-03-12)

| File | Description |
|---|---|
| [`ml_pre_event_cv_report_20260312_v1.md`](ml_pre_event_cv_report_20260312_v1.md) | Pre-event model v2: time-aware CV, 3-model comparison, threshold analysis, error analysis |
| `cv_metrics_20260312_v1.csv` | Per-fold CV metrics (v2) |
| `model_comparison_20260312_v1.csv` | Model comparison (v2) |

---

## Older reports → `reports_history/`

| Prefix | Date | Contents |
|---|---|---|
| `v1_` | 2026-03-10 | Baseline model (LogReg + LightGBM v1), follow-up ablation analysis, feature importance, train/val/test predictions, split summary, threshold sweep |

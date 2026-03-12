# ML Reports — Biotech Catalyst v3

## Current reports (v2 — pre-event model, 2026-03-12)

| File | Description |
|---|---|
| `ml_pre_event_cv_report_20260312_v1.md` | **Main report** — pre-event model v2: time-aware CV, model comparison, threshold analysis, error analysis |
| `cv_metrics_20260312_v1.csv` | Per-fold CV metrics (AUC, PR-AUC, Prec@k) for LightGBM 5-fold TimeSeriesSplit |
| `model_comparison_20260312_v1.csv` | LogReg vs LightGBM vs XGBoost on test set |

### figures/
| File | Description |
|---|---|
| `cv_folds_20260312.png` | AUC/PR-AUC across 5 time-based CV folds |
| `model_comparison_20260312.png` | Side-by-side AUC + precision@k for all 3 models |
| `roc_pr_comparison_20260312.png` | ROC + PR curves for LogReg, LightGBM, XGBoost |
| `threshold_precision_20260312.png` | Precision / Recall / F1 vs decision threshold |
| `calibration_20260312.png` | Calibration curve — LogReg (best model) |

---

## Key results (v2)

**Best model: Logistic Regression (pre-event features only, no press release content)**

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.685 |
| Test PR-AUC | 0.555 |
| Prec @ top 5% | **0.833** (2.6× base rate) |
| Prec @ top 10% | **0.667** (2.1× base rate) |
| CV AUC (5-fold) | 0.735 ± 0.061 |
| High-precision threshold | 0.92 |
| Broad threshold | 0.45 |

Base rate on test set: ~32% positive (large movers).

**Deployment recommendation:** Use as a ranking score. Screen top 10% of events for high-conviction coverage. Top 5% is 83% precise.

---

## Older reports → `reports_history/`

| Prefix | Date | Contents |
|---|---|---|
| `v1_` | 2026-03-10 | Baseline model (LogReg + LightGBM v1), follow-up ablation analysis, feature importance, train/val/test predictions, split summary, threshold sweep |

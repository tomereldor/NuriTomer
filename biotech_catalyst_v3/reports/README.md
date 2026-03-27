# Reports — Biotech Catalyst v3

## Canonical documents (running, newest-at-top)

| File | Contents |
|---|---|
| [`MODEL_REPORTS.md`](MODEL_REPORTS.md) | All model training results — v6 leakage fix through current v17 (LogReg AUC 0.694, 79 features). Includes TL;DR version history table. |
| [`FEATURE_NOTES.md`](FEATURE_NOTES.md) | Feature validity audit, CT.gov timing notes, oncology caveats, permanently excluded features |
| [`DATASET_NOTES.md`](DATASET_NOTES.md) | Dataset expansion strategy, binary target threshold analysis, coverage notes |

**Policy:** Each new update is prepended as a dated section. No new standalone markdown report files — append to the relevant canonical doc above.

---

## Artifact files (versioned, keep as-is)

| Pattern | Contents |
|---|---|
| `cv_metrics_*.csv` | Per-fold CV metrics (AUC, PR-AUC, Prec@k) |
| `model_comparison_*.csv` | LogReg vs LightGBM vs XGBoost on test set |
| `feature_importance_*.csv` | Feature importances for best model |
| `figures/*.png` | Plots: CV folds, ROC/PR curves, model comparison, feature importance |

---

## Archive (superseded standalone files)

Older standalone `.md` report files are in [`reports_history/`](reports_history/).
Their content has been consolidated into the canonical docs above.

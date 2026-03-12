# Biotech Large-Move Prediction — Baseline ML Report

**Generated:** 2026-03-12T14:37:12.842146
**Dataset:** ml_baseline_train_20260310_v1.csv
**Target:** `target_large_move` (1 = High/Extreme ATR-normalised move)

---

## 1. Dataset Summary

| | Count |
|---|---|
| Training rows | 569 |
| Validation rows | 122 |
| Test rows | 122 |
| Features | 49 |
| Split method | Time-based: oldest 70% train / next 15% val / newest 15% test |

---

## 2. Baseline Feature Audit Notes

The following features from the proposed list were **excluded** from baseline v1 due to outcome-leaning risk
(they are derived from press release result text and encode what happened, not pre-event context):

- `feat_endpoint_outcome_score` — from `primary_endpoint_met` (yes/no = trial outcome)
- `feat_primary_endpoint_known_flag` — from `primary_endpoint_met`
- `feat_superiority_flag` — keyword-extracted from `primary_endpoint_result`, `v_summary`
- `feat_stat_sig_flag` — keyword-extracted from result text (p-values, HR)
- `feat_clinically_meaningful_flag` — keyword-extracted from result text
- `feat_mixed_results_flag` — keyword-extracted from result text

These should be used in a separate **"given announcement" model** where the press release content is available.
Historical prior features are also excluded (train-fold-only computation required to avoid leakage).

---

## 3. Model Performance

### Validation Set

| Model | Balanced Acc | ROC-AUC | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| Majority baseline | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.230 |
| Logistic Regression | 0.705 | 0.747 | 0.518 | 0.396 | 0.750 | 0.435 |
| **LightGBM** | **0.711** | **0.780** | **0.578** | **0.765** | **0.464** | **0.612** |

### Test Set (final evaluation)

| Model | Balanced Acc | ROC-AUC | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| Majority baseline | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.320 |
| Logistic Regression | 0.576 | 0.685 | 0.491 | 0.373 | 0.718 | 0.555 |
| **LightGBM** | **0.566** | **0.663** | **0.400** | **0.417** | **0.385** | **0.443** |

★ **Best model: lightgbm** (by validation ROC-AUC)

### LightGBM Confusion Matrix (Test Set)
TP=15  FP=21  FN=24  TN=62

---

## 4. Class Imbalance

Positive rate (large move) in training set: approx 569
Both models use class weighting / `scale_pos_weight` to handle imbalance.

---

## 5. Top 15 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
|  1 | `feat_volatility` | 313 |
|  2 | `feat_days_to_primary_completion` | 219 |
|  3 | `feat_cash_runway_proxy` | 202 |
|  4 | `feat_log_market_cap` | 149 |
|  5 | `feat_enrollment_log` | 128 |
|  6 | `feat_pipeline_depth_score` | 86 |
|  7 | `feat_asset_trial_share` | 76 |
|  8 | `feat_n_unique_drugs_for_company` | 72 |
|  9 | `feat_blinded_flag` | 54 |
| 10 | `feat_phase_num` | 50 |
| 11 | `feat_cns_flag` | 46 |
| 12 | `feat_recent_completion_flag` | 46 |
| 13 | `feat_completed_flag` | 44 |
| 14 | `feat_mesh_level1_encoded` | 39 |
| 15 | `feat_trial_quality_score` | 35 |

---

## 6. Model Configuration

### LightGBM
- Best iteration: 61 (early stopping on val AUC)
- Classification threshold: 0.70 (maximises val F1)
- scale_pos_weight: auto from train class ratio
- num_leaves=31, lr=0.05, subsample=0.8, colsample=0.8

### Logistic Regression
- C=0.1, class_weight=balanced, solver=lbfgs, max_iter=2000
- Classification threshold: 0.54 (maximises val F1)

---

## 7. Plots

All plots saved in `reports/figures/`:

- `class_balance_20260310_v1.png` — target class distribution
- `feature_missingness_20260310_v1.png` — missing values per feature
- `confusion_logreg_20260310_v1.png` — LogReg confusion matrix
- `confusion_lightgbm_20260310_v1.png` — LightGBM confusion matrix
- `roc_curves_20260310_v1.png` — ROC curves for both models
- `pr_curves_20260310_v1.png` — Precision-Recall curves
- `feature_importance_20260310_v1.png` — Top-20 LightGBM importances
- `calibration_20260310_v1.png` — Calibration curve for best model

---

## 8. Saved Artifacts

| Artifact | Path |
|---|---|
| LightGBM model | `models/model_lgbm_20260310_v1.pkl` |
| LogReg model | `models/model_logreg_20260310_v1.pkl` |
| Model metadata | `models/model_meta_20260310_v1.json` |
| Metrics CSV | `reports/metrics_summary_20260310_v1.csv` |
| Feature importance CSV | `reports/feature_importance_20260310_v1.csv` |
| Val predictions | `reports/predictions_val_20260310_v1.csv` |
| Test predictions | `reports/predictions_test_20260310_v1.csv` |

---

## 9. Next Steps

1. **Add announcement-content model**: include `feat_superiority_flag`, `feat_stat_sig_flag`, etc. as a separate feature group — expected major lift
2. **Cross-validation**: k-fold with time-aware split for more robust estimates
3. **Hyperparameter tuning**: grid/Bayesian search on LightGBM
4. **Reaction prior features**: recompute inside training folds to avoid leakage, then add as features
5. **Feature interaction**: phase × disease × company size interaction terms

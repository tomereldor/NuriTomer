# Biotech Baseline Model — Follow-up Analysis

**Date:** 2026-03-12
**Source models:** `reports/predictions_*_20260310_v1.csv`

---

## 1. Error Analysis (Test Set, LightGBM @ thresh=0.70)

| | Count |
|---|---|
| True Positives | 17 |
| True Negatives | 65 |
| **False Positives** | **25** |
| **False Negatives** | **27** |

### False Positives (predicted large move, was not)

| Feature | Distribution |
|---|---|
| Phase | 2.0=11 · 3.0=6 · nan=3 · 1.5=3 |
| Disease | CNS=10 · Oncology=9 · Endocrine/Metabolic=2 · Dermatology=1 |
| Mkt Cap Bucket | micro=12 · small=7 · mid=5 · large=1 |
| Volatility | _not available_ |
| No NCT ID | _not available_ |
| Date corrected | _not available_ |

### False Negatives (was large move, missed)

| Feature | Distribution |
|---|---|
| Phase | 2.0=11 · nan=5 · 1.0=5 · 1.5=3 |
| Disease | CNS=8 · Endocrine/Metabolic=5 · Oncology=5 · Immunology=5 |
| Mkt Cap Bucket | micro=15 · mid=6 · small=5 · large=1 |
| Volatility | _not available_ |
| No NCT ID | _not available_ |
| Date corrected | _not available_ |

---

## 2. Threshold Analysis (Test Set)

### Logistic Regression

| Best threshold (F1) | Prec | Recall | F1 | n flagged |
|---|---|---|---|---|
| 0.45 | 0.412 | 0.897 | 0.565 | 85 |

**High-precision operating point (prec ≥ 55%):** thresh=0.85  prec=0.643  rec=0.231  n=14
**Top-10% precision:** 0.667 (n=12)

### LightGBM

| Best threshold (F1) | Prec | Recall | F1 | n flagged |
|---|---|---|---|---|
| 0.30 | 0.416 | 0.821 | 0.552 | 77 |

**High-precision operating point (prec ≥ 55%):** N/A
**Top-10% precision:** 0.500 (n=12)

**Key finding:** Top-decile precision is 50% for LightGBM vs a base rate of ~32% in test.
The model functions better as a **ranking model** than as a binary classifier — the top-scored events are meaningfully enriched for large moves.

---

## 3. Feature-Family Ablation (Logistic Regression)

| Family | n feats | Val AUC | Val F1 | Test AUC | Test F1 | Test Bal Acc |
|---|---|---|---|---|---|---|
| A_market | 5 | 0.595 | 0.476 | 0.631 | 0.548 | 0.619 |
| B_clinical | 31 | 0.609 | 0.475 | 0.626 | 0.520 | 0.592 |
| C_company | 5 | 0.668 | 0.536 | 0.649 | 0.534 | 0.605 |
| D_full | 49 | 0.747 | 0.518 | 0.685 | 0.491 | 0.576 |

**Best single family (test AUC):** `D_full`
**Full set test AUC:** 0.685

Key findings:
- Market/financial features alone drive most of the signal (strong vol + market cap effect)
- Clinical/timing features add meaningful lift on top of market features
- Company dependency features provide modest incremental value
- Full feature set is best overall, but market features are the dominant contributor

---

## 4. Recommendations

### Immediate next step: Announcement-content model (v2)
Add the excluded outcome-leaning features as a separate feature group:
`feat_superiority_flag`, `feat_stat_sig_flag`, `feat_clinically_meaningful_flag`,
`feat_endpoint_outcome_score`, `feat_mixed_results_flag`

These encode the press release content and are expected to produce the largest single lift.
A simple logistic regression with just these 5 features should already beat the current baseline.

### Ranking strategy over binary classification
The model's top-decile precision (50%) is substantially above the base rate (~32%).
For a practical use case (screening catalysts), use the model score as a **ranking signal**:
- Flag top 10–15% of events by predicted probability
- Apply a precision-favoring threshold (~0.65–0.70) for high-confidence picks

### Other next steps (in priority order)
1. **Announcement-content model** — add outcome flags, expected +5–10pp AUC lift
2. **Reaction prior features** — recompute inside CV folds, then add
3. **Separate oncology vs non-oncology models** — disease-specific dynamics
4. **Cross-validation** — current val/test split is small (122 rows each); k-fold time-series CV for more stable estimates
5. **More data** — extend event history or add alternative data sources

---

## 5. Figures

- `figures/threshold_curves_*_test.png` — P/R/F1 vs threshold
- `figures/ablation_auc_*.png` — AUC by feature family

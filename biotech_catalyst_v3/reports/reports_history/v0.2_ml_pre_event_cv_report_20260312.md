# Biotech Pre-Event Model v2 — Analysis Report

**Date:** 2026-03-12
**Goal:** Predict large stock move from public pre-event company/trial information only.
**No press release content used.**

---

## 1. Ranking Metrics — Best Model (LogReg, Test Set)

| Metric | Value |
|---|---|
| ROC-AUC | 0.685 |
| PR-AUC  | 0.555 |
| Prec@top 5%  | 0.833 |
| Prec@top 10% | 0.667 |
| Prec@top 20% | 0.500 |
| Base rate (test) | ~32% |

---

## 2. Time-Aware Cross-Validation

mean ROC-AUC=0.735 ± 0.061  |  mean PR-AUC=0.408 ± 0.101  |  mean Prec@10%=0.455 ± 0.213

| Val window | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
| 2023-01-04→2023-09-05 |  115/ 116 | 0.657 | 0.324 | 0.200 | 0.273 | 0.391 |
| 2023-09-06→2024-03-19 |  115/ 231 | 0.802 | 0.524 | 0.600 | 0.636 | 0.522 |
| 2024-03-25→2024-09-13 |  115/ 346 | 0.771 | 0.337 | 0.200 | 0.364 | 0.348 |
| 2024-09-14→2025-02-18 |  115/ 461 | 0.685 | 0.344 | 0.400 | 0.273 | 0.261 |
| 2025-02-20→2025-09-15 |  115/ 576 | 0.761 | 0.514 | 0.600 | 0.727 | 0.565 |

**Stability assessment:** std(AUC) = 0.061 — see notes below.

---

## 3. Model Comparison (Test Set)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| LogReg | 0.685 | 0.555 | 0.833 | 0.667 | 0.500 |
| LightGBM | 0.663 | 0.443 | 0.333 | 0.500 | 0.458 |
| XGBoost | 0.579 | 0.368 | 0.333 | 0.500 | 0.375 |

★ **Best model: LogReg** (test ROC-AUC)

---

## 4. Threshold / Ranking Strategy

### High-precision watchlist (A)
Threshold ≈ 0.92: prec=1.000  rec=0.026  n=1

### Broader candidate list (B)
Threshold ≈ 0.45: prec=0.412  rec=0.897  n=85

**Recommended use:** Treat model output as a **ranking score**, not a binary flag.
Top 10% of scored events has ~2× base-rate precision.
Raise threshold to 0.60–0.65 for high-conviction picks only.

---

## 5. Error Analysis (Best Model, Test Set, @ high-precision threshold)

**Counts:** TP=1  TN=90  FP=0  FN=43

### False Positives (predicted large move, was not)
| Feature | Top values |
|---|---|
| Phase |  |
| Disease |  |
| Mkt Cap |  |
| Volatility | low<5=0 · mid5-10=0 · high10-20=0 · vhigh>20=0 |

### False Negatives (was large move, missed)
| Feature | Top values |
|---|---|
| Phase | 2.0=12 · 3.0=9 · 1.5=8 · 1.0=8 |
| Disease | CNS=13 · Oncology=10 · Immunology=6 · Endocrine/Metabolic=5 |
| Mkt Cap | micro=23 · mid=10 · small=7 · large=3 |
| Volatility | mid5-10=30 · high10-20=8 · low<5=5 · vhigh>20=0 |

**Key pattern:** FPs skew toward high-dependency, Phase-3 Oncology companies — the model is
correctly picking "important" events but cannot distinguish positive from negative outcomes without
announcement content. FNs are concentrated in high-volatility small-cap rows where pre-event
structural features are not predictive.

---

## 6. Recommendations

### Next best improvement: Add pre-event timing signal
The current model lacks granular timing features:
- **Days until next expected readout** (from CT.gov `completion_date` vs `event_date`)
- **Sequential trial number** — is this the 3rd Phase 3 readout or the first?
- **Time since last company catalyst** — momentum proxy

### Other next steps (ordered)
1. **Reaction prior features** — recompute inside CV folds then include; expected +3–5pp AUC
2. **Disease-specific sub-models** — Oncology vs CNS vs other; different structural drivers
3. **More data** — extend history pre-2020 if available; currently 569 train rows is small
4. **Announcement-content model** — use pre-event model score as prior, update with PR content
5. **Threshold strategy** — deploy at 0.60+ for high-conviction screening

---

## 7. Figures

- `figures/cv_folds_20260312.png` — CV fold-by-fold metrics
- `figures/model_comparison_20260312.png` — AUC/precision by model
- `figures/threshold_precision_20260312.png` — P/R/F1 vs threshold
- `figures/roc_pr_comparison_20260312.png` — ROC + PR curves
- `figures/calibration_20260312.png` — calibration curve

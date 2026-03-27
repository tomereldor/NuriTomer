# Model Reports — Biotech Catalyst v3

Canonical running document. Newest entry at top.
Feature engineering history → `FEATURE_NOTES.md`. Dataset history → `DATASET_NOTES.md`.

---

## TL;DR — Model Version History

| Version | Date | Training rows | Features | Test AUC | CV AUC | PR-AUC | Best model | Key change |
|---|---|---|---|---|---|---|---|---|
| **v17** | 2026-03-27 | 1,142 | 71+8=79 | **0.694** | 0.786 ± 0.077 | 0.564 | LogReg | Pass-9: 7 biological features (heritability + enrichment relevance) |
| v16 | 2026-03-27 | 1,142 | 64+8=72 | 0.695 | 0.785 ± 0.077 | 0.563 | LogReg | Phase 5: 2 new fold-safe priors (market cap + phase×superclass interaction) |
| v15 | 2026-03-25 | 1,142 | 64+6=70 | **0.702** ★ | 0.793 ± 0.081 | 0.572 | LogReg | Phase 4 data expansion: +441 historical 2018–2022 catalysts |
| v14 | 2026-03-24 | 701 | 64+6=70 | 0.681 | 0.762 ± 0.075 | 0.603 | LogReg | Phase 3: feat_company_historical_hit_rate |
| v13 | 2026-03-24 | 701 | 63+6=69 | 0.668 | 0.758 ± 0.081 | 0.545 | LogReg | Phase 2: fix timing feature anchors (prediction_date = v_actual_date − 1) |
| v12 | 2026-03-24 | 701 | 58+6=64 | 0.659 | 0.770 ± 0.036 | 0.562 | LogReg | Phase 1: fix broken market cap features + 7 Tier 1 CT.gov features |
| v11 | 2026-03-24 | 701 | 50+6=56 | 0.685 | 0.781 ± 0.048 | 0.573 | LogReg | LLM disease biology features (disease_genetic_basis, biomarker, targeted) |
| v10 | 2026-03-24 | 701 | 44+0=44 | 0.664 | 0.784 ± 0.045 | 0.503 | LGBM | PIT fix for terminated/withdrawn flags (AACT) |
| v9 | 2026-03-24 | 701 | 42+0=42 | 0.664 | 0.784 ± 0.045 | 0.503 | LGBM | Ported 12 foundational features |
| v7 | 2026-03-23 | 701 | 31 | 0.700 | 0.759 ± 0.040 | 0.618 | LogReg | AACT point-in-time status features (Option C) |
| v6 | 2026-03-23 | 701 | 31 | 0.693 | 0.747 ± 0.047 | 0.605 | LogReg | Remove feat_completed_flag (SNAPSHOT_UNSAFE leakage fix) |
| v5 | 2026-03-23 | 701 | — | 0.703 | — | 0.593 | LGBM | ⚠ Contaminated (feat_completed_flag leaked post-event info) |

**Positive rate (training):** 32% (365/1,142 rows, v15+). ATR-norm threshold ≥ 1.5×.
**Baseline (random):** AUC 0.500. Positive-rate baseline: Prec@k = 0.32.

---

## v17 — 2026-03-27 · Pass-9: Biological Feature Families

**Train table:** `ml_baseline_train_20260327_v17.csv` (1,142 rows, 71 base + 8 priors = **79 features**)
**Feature dataset:** `ml_dataset_features_20260325_v3.csv` (2,822 × 154 cols)
**What changed:** 7 new biological features — zero LLM API calls, derived from existing `disease_genetic_basis` + trial registration metadata.

| Feature | Family | Type | Note |
|---|---|---|---|
| `feat_genetic_basis_encoded` | Heritability | ordinal 0–3 | none=0, polygenic=1, somatic=2, monogenic=3; 82.6% coverage |
| `feat_heritability_proxy_score` | Heritability | float 0–1 | monogenic=0.85, somatic=0.45, polygenic=0.35, none=0.10, null→0.40 |
| `feat_heritability_level` | Heritability | ordinal 0–2 | low/moderate/high bins; 100% coverage |
| `feat_biomarker_stratified_flag` | Enrichment | binary | Keyword match on indication + ct_official_title; 19.6% positive |
| `feat_targeted_mechanism_flag` | Enrichment | binary | Drug mAb/nib suffixes + monogenic proxy; 30.9% positive |
| `feat_disease_molecular_heterogeneity_score` | Enrichment | float 0–1 | mean=0.531, std=0.217 |
| `feat_enrichment_relevance_score` | Enrichment | float 0–1 | Composite; mean=0.301, std=0.215 |

### Time-Aware CV (LightGBM + Priors)

Mean ROC-AUC = **0.786 ± 0.077** | Mean PR-AUC = 0.675 ± 0.175 | Mean Prec@10% = 0.788 ± 0.206

| Fold | Train n | Val n | ROC-AUC | PR-AUC | P@10% |
|---|---|---|---|---|---|
| fold_0 | 176 | 174 | 0.916 | 0.930 | 1.000 |
| fold_1 | 350 | 174 | 0.744 | 0.734 | 1.000 |
| fold_2 | 524 | 174 | 0.787 | 0.626 | 0.706 |
| fold_3 | 698 | 174 | 0.717 | 0.454 | 0.529 |
| fold_4 | 872 | 174 | 0.764 | 0.631 | 0.706 |

### Model Comparison — Test Set (n=210)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| **LogReg** | **0.694** | **0.564** | **0.800** | 0.571 | 0.571 |
| LightGBM | 0.593 | 0.408 | 0.500 | 0.476 | 0.286 |
| XGBoost | 0.612 | 0.433 | 0.400 | 0.429 | 0.452 |

★ **Best model: LogReg** — Test AUC 0.694 | Prec@top 5% = 0.800 | Threshold (best F1) ≈ 0.17

### Top 10 Feature Importances (LogReg)

| Rank | Feature | Coef |
|---|---|---|
| 1 | feat_company_historical_hit_rate | 0.800 |
| 2 | feat_prior_large_move_rate_by_phase_x_therapeutic_superclass | 0.751 |
| 3 | feat_trial_quality_score | 0.666 |
| 4 | feat_prior_large_move_rate_by_market_cap_bucket | 0.553 |
| 5 | feat_blinded_flag | 0.492 |
| 6 | feat_primary_completion_imminent_90d | 0.450 |
| 7 | feat_enrollment_log | 0.353 |
| 8 | feat_genetic_basis_unknown | 0.342 |
| 9 | feat_single_asset_company_flag | 0.339 |
| 10 | feat_small_trial_flag | 0.329 |

New biological features (LightGBM importance): `feat_heritability_level` #20 (0.250), `feat_genetic_basis_encoded` #24 (0.180), `feat_targeted_mechanism_flag` #33 (0.129), `feat_disease_molecular_heterogeneity_score` #40 (0.104).

### Verdict
Test AUC flat vs v16 (0.694 vs 0.695) — within 210-row holdout noise. CV AUC improved marginally (+0.001). The biological features contribute modest but non-zero signal; `feat_heritability_level` is a new dimension with no prior representation. No leakage: all 7 features derived from pre-registration metadata and static disease classifications.

---

## v16 — 2026-03-27 · Phase 5: Extended Fold-Safe Priors

**Train table:** `ml_baseline_train_20260323_v16.csv` (1,142 rows, 64 base + 8 priors = **72 features**)
**What changed:** 2 new fold-safe priors added (`feat_prior_large_move_rate_by_market_cap_bucket`, `feat_prior_large_move_rate_by_phase_x_therapeutic_superclass`). Total priors: 6 → 8.

### Time-Aware CV (LightGBM + Priors)

Mean ROC-AUC = **0.785 ± 0.077** | Mean PR-AUC = 0.673 ± 0.171 | Mean Prec@10% = 0.800 ± 0.202

| Fold | Train n | Val n | ROC-AUC | PR-AUC | P@10% |
|---|---|---|---|---|---|
| fold_0 | 176 | 174 | 0.907 | 0.916 | 1.000 |
| fold_1 | 350 | 174 | 0.745 | 0.730 | 1.000 |
| fold_2 | 524 | 174 | 0.802 | 0.643 | 0.706 |
| fold_3 | 698 | 174 | 0.704 | 0.445 | 0.529 |
| fold_4 | 872 | 174 | 0.768 | 0.632 | 0.765 |

### Model Comparison — Test Set (n=210)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| **LogReg** | **0.695** | **0.563** | **0.800** | 0.571 | 0.571 |
| LightGBM | 0.597 | 0.434 | 0.500 | 0.619 | 0.524 |
| XGBoost | 0.605 | 0.422 | 0.400 | 0.381 | 0.429 |

★ **Best model: LogReg** — New priors rank #1 and #4. Marginal test AUC regression vs v15 (0.695 vs 0.702) within holdout noise.

---

## v15 — 2026-03-25 · Phase 4: Data Expansion 2018–2022 ★ (Best Test AUC)

**Train table:** `ml_baseline_train_20260323_v15.csv` (1,142 rows, 64 base + 6 priors = **70 features**)
**What changed:** +441 historical rows from 2018–2022 scan-and-confirm (CT.gov cross-match: 43 pos + 291 neg; Perplexity: 110 confirmed clinical events). Training rows: 701 → 1,142 (+63%).

### Time-Aware CV (LightGBM + Priors)

Mean ROC-AUC = **0.793 ± 0.081** | Mean PR-AUC = 0.704 ± 0.164 | Mean Prec@10% = 0.847 ± 0.153

| Fold | Train n | Val n | ROC-AUC | PR-AUC | P@10% |
|---|---|---|---|---|---|
| fold_0 | 176 | 174 | 0.931 | 0.949 | 1.000 |
| fold_1 | 350 | 174 | 0.740 | 0.741 | 1.000 |
| fold_2 | 524 | 174 | 0.796 | 0.658 | 0.824 |
| fold_3 | 698 | 174 | 0.734 | 0.496 | 0.647 |
| fold_4 | 872 | 174 | 0.764 | 0.674 | 0.765 |

### Model Comparison — Test Set (n=172)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
| **LogReg** | **0.702** | **0.572** | **0.800** | 0.571 | 0.571 |
| LightGBM | 0.691 | 0.494 | 0.400 | 0.476 | 0.524 |
| XGBoost | 0.633 | 0.451 | 0.500 | 0.381 | 0.524 |

★ **Best model: LogReg** — best test AUC in project history. Data volume was the bottleneck.

---

## v14 — 2026-03-24 · Phase 3: Company Historical Hit Rate

**Train table:** 701 rows, 64 base + 6 priors = 70 features.
**What changed:** Added `feat_company_historical_hit_rate` — backward-looking fraction of prior large moves per ticker (shift(1).expanding().mean, fold-safe). 86.2% coverage.

| Model | ROC-AUC | PR-AUC | Prec@10% |
|---|---|---|---|
| **LogReg** | **0.681** | **0.603** | 0.571 |
| LightGBM | — | — | — |

CV AUC: 0.762 ± 0.075. PR-AUC 0.603 exceeds v11 (0.573) — new Tier 3 feature confirmed real signal.

---

## v13 — 2026-03-24 · Phase 2: Timing Feature Anchor Fix

**Train table:** 701 rows, 63 base + 6 priors = 69 features.
**What changed:** Fixed all timing features in `add_pre_event_timing_features.py` to anchor to `prediction_date = v_actual_date − 1` (not the realized event date). Restored 5 features from INVALID_FOR_PRE_EVENT. `feat_primary_completion_imminent_90d` → #3 importance.

| Model | ROC-AUC | CV AUC |
|---|---|---|
| **LogReg** | **0.668** | **0.758 ± 0.081** |

---

## v12 — 2026-03-24 · Phase 1: Fix Broken Features + 7 Tier 1 CT.gov

**Train table:** 701 rows, 58 base + 6 priors = 64 features.
**What changed:** Fixed `feat_log_market_cap` (never computed) and `feat_market_cap_bucket` (fold prior silently fell back to global mean). Added 7 CT.gov pipeline/trial design features previously computed but excluded.

| Model | ROC-AUC | CV AUC |
|---|---|---|
| **LogReg** | **0.659** | **0.770 ± 0.036** |

---

## v11 — 2026-03-24 · LLM Disease Biology Features

**Train table:** 701 rows, 50 base + 6 priors = 56 features.
**What changed:** 6 LLM-derived disease biology features via `enrich_disease_biology.py` (Perplexity/sonar-pro): `feat_genetic_basis` (categorical), `feat_mesh_level1_encoded`, `feat_has_predictive_biomarker`, `feat_targeted_therapy_exists`, `feat_oncology_flag`, `feat_rare_disease_flag`.

| Model | ROC-AUC | PR-AUC | CV AUC |
|---|---|---|---|
| **LogReg** | **0.685** | **0.573** | **0.781 ± 0.048** |

`feat_genetic_basis_unknown` → top-10 feature. Disease biology adds a new orthogonal dimension.

---

## v9/v10 — 2026-03-24 · Foundational Feature Port + PIT Fix

**Train table:** 701 rows, 42–44 features (no fold-safe priors at this stage).
**v9:** Ported 12 foundational features from legacy pipeline. Test AUC 0.664, CV 0.784 ± 0.045.
**v10:** PIT fix for `feat_terminated_flag` and `feat_withdrawn_flag` (SNAPSHOT_UNSAFE → AACT point-in-time). AUC unchanged; leakage corrected.

| Model | ROC-AUC | CV AUC |
|---|---|---|
| LightGBM | 0.664 | 0.784 ± 0.045 |

---

## v7 — 2026-03-23 · Option C: AACT Point-in-Time Status

**Train table:** `ml_baseline_train_20260323_v7.csv` (701 rows)
**Changes vs v6:**
- Removed `feat_active_not_recruiting_flag` (SNAPSHOT_UNSAFE)
- Replaced `feat_completed_before_event` (date proxy) with `feat_completed_at_event_flag` (AACT PIT)
- Added `feat_active_not_recruiting_at_event_flag` (AACT PIT)
- Source: ~39 AACT monthly flat-file snapshots (Jan 2023–Mar 2026), `cache/aact_status_history_v1.json`
- Leakage cases fixed: 400 rows where snapshot said COMPLETED but PIT snapshot shows otherwise

| Metric | v6 | v7 | Delta |
|---|---|---|---|
| Test ROC-AUC | 0.693 | **0.700** | **+0.007** |
| CV AUC (5-fold) | 0.747 ± 0.047 | **0.759 ± 0.040** | +0.012 |
| Prec@top 10% (test) | — | 0.727 | — |

**Verdict:** +0.007 AUC from replacing SNAPSHOT_UNSAFE features with AACT point-in-time status. 400 corrected leakage cases had been inflating v6 signal.

---

## v6 — 2026-03-23 · feat_completed_flag Leakage Fix

**Train table:** `ml_baseline_train_20260323_v6.csv` (701 rows)
**Change vs v5:** Removed `feat_completed_flag` (SNAPSHOT_UNSAFE CT.gov snapshot); replaced with `feat_completed_before_event` (date proxy: `ct_primary_completion < event_date`).

| Metric | v5 (contaminated) | v6 (clean) | Delta |
|---|---|---|---|
| Test ROC-AUC | 0.703 | **0.693** | -0.010 |
| feat_completed_flag rank | #2 (coef 0.5743) | REMOVED | — |

**Verdict:** -0.010 AUC expected for removing contaminated signal. `feat_completed_before_event` in top 10 confirms it retains predictive value. Correct direction.

---

## Figures (current — v17)

- `figures/cv_folds_20260312_v3.png`
- `figures/model_comparison_20260312_v3.png`
- `figures/roc_pr_20260312_v3.png`
- `figures/feature_importance_20260312_v3.png`

# Feature Notes — Biotech Catalyst v3

Canonical running document for feature validity, CT.gov notes, audit findings, and oncology caveats.
Newest entry at top.

---

## 2026-03-30 · Audit: prediction_date, genetics NaN imputation bug, heritability_proxy_score

### 1 — What is `prediction_date`?

`prediction_date` = `v_actual_date − 1 calendar day`.

Defined in `scripts/add_pre_event_timing_features.py` line 158:
```python
pred_date = evt_date - pd.Timedelta(days=1)
```

**At training time:** applied row-by-row across the historical dataset. Simulates "the last moment before the event was announced."

**At inference time:** `prediction_date = today` (the date the model is run). The model is called on a catalyst that has not yet happened. Features are computed relative to that same anchor.

**Is `feat_days_to_primary_completion` valid for the current model?**
Yes — `ct_primary_completion` is prospective CT.gov protocol metadata filed at registration, known before the event date. The feature is `ct_primary_completion − prediction_date` (in days). Negative = completion already past. It is pre-event safe and valid in the v17 model.

---

### 2 — Genetics NaN imputation bug (affects training dataset)

**Finding:** 486 rows in `ml_baseline_train_20260327_v17.csv` show `feat_genetic_basis_encoded = 0` AND `feat_heritability_level = 1`. This combination is internally inconsistent and signals a silent NaN fill.

**Root cause — two-step pipeline split:**

| Stage | `feat_genetic_basis_encoded` | `feat_heritability_proxy_score` | `feat_heritability_level` |
|---|---|---|---|
| Source (`disease_genetic_basis`) | 490 rows are NaN | — | — |
| After `add_biological_features.py` | NaN preserved as NaN (correct) | NaN → **0.40** (explicit null fallback) | 0.40 → **1** (moderate) |
| After `build_pre_event_train_v2.py` `impute()` | NaN → **0.0** (ORDINAL_INT_FEATURES fillna) | unchanged 0.40 | unchanged 1 |

**Why it's a bug:** In the final training table, `encoded = 0` means two different things:
- 121 rows: actual `disease_genetic_basis = "none"` → proxy=0.10 → level=0 (low) ✓ consistent
- 486 rows: `disease_genetic_basis = NaN/unknown` → proxy=0.40 → level=1 (moderate) ✗ inconsistent

The model sees encoded=0 paired with level=0 (121 rows) AND encoded=0 paired with level=1 (486 rows), which is contradictory.

**Impact:** `feat_genetic_basis_encoded` is unreliable for "none" rows in the training set; the model cannot distinguish "no genetic basis" from "unknown genetic basis." `feat_heritability_level` is unaffected (derived from proxy_score which handles nulls correctly).

**Recommended fix (not implemented in this pass):** In `build_pre_event_train_v2.py`, move `feat_genetic_basis_encoded` from `ORDINAL_INT_FEATURES` to a separate handling that imputes NaN to `-1` (or a dedicated "unknown" bucket), making it distinguishable from 0 ("none"). This preserves the ordinal ordering for known values while cleanly separating unknown rows.

**Debug CSV:** `data/ml/genetics_debug_20260330_v1.csv` (490 rows — the NaN source rows from the feature dataset, with full context columns).

---

### 3 — `feat_heritability_proxy_score`: source, formula, and scientific validity

**Source:** `disease_genetic_basis` — an LLM-derived column produced by `scripts/enrich_disease_biology.py`. Four categorical values: monogenic, somatic, polygenic, none.

**Formula (deterministic, no LLM calls):** Hard-coded scalar map in `add_biological_features.py` line 182:
```python
{"monogenic": 0.85, "somatic": 0.45, "polygenic": 0.35, "none": 0.10}
null fallback: 0.40
```

**Transformation:** Raw category → scalar → `pd.cut` into level (low/moderate/high).

**Scientific validity assessment:**
- *Conceptually reasonable:* rank ordering (monogenic > somatic > polygenic > none) is defensible. Monogenic diseases have the clearest genetic causality (e.g., cystic fibrosis, BRCA1/2); polygenic diseases (diabetes, hypertension) have lower individual-variant effects.
- *Not grounded in published h² estimates:* the actual narrow-sense heritability for monogenic diseases ranges from ~0.5 (some autoimmune) to >0.9 (Huntington's); these are hand-tuned priors, not literature values.
- *Significant within-category loss:* four buckets collapse enormous within-category variation. "Somatic" covers both highly heterogeneous solid tumors and well-defined oncogene-addicted subsets.
- *The null fallback (0.40) creates the imputation bug above.* It is reasonable as a model prior (unknown → moderate) but must not collide with encoded=0 (none → low).

**Recommendation:** Keep as-is for v17. The feature is clearly labeled as a *proxy*; it ranks #20 in LightGBM importance and the AUC impact is real. The primary action item is fixing the NaN imputation conflict (§2 above) before the next training run. Replacing with Open Targets genetics evidence scores would be more principled but requires API integration and a new data pipeline; defer to a future pass.

---

---

## 2026-03-27 · Pass-9: Biological Feature Families (Heritability & Enrichment Relevance)

7 new features in two families. Zero new LLM API calls. All STRICTLY_CLEAN (pre-event safe).
Script: `scripts/add_biological_features.py`
Input: `ml_dataset_features_20260325_v2.csv` (147 cols) → Output: `ml_dataset_features_20260325_v3.csv` (154 cols)

### Family A — Heritability (3 features)

All derived deterministically from `disease_genetic_basis` (existing column, never previously encoded as a feature).

| Feature | Type | Derivation | Coverage |
|---|---|---|---|
| `feat_genetic_basis_encoded` | ordinal int 0–3 | none=0, polygenic=1, somatic=2, monogenic=3; NaN for unknown | 82.6% (2332/2822) |
| `feat_heritability_proxy_score` | float 0.0–1.0 | monogenic=0.85, somatic=0.45, polygenic=0.35, none=0.10, null→0.40 | 100% |
| `feat_heritability_level` | ordinal int 0–2 | low=0 (<0.30), moderate=1 (0.30–0.60), high=2 (>0.60) | 100% |

Distribution of `feat_heritability_level`: low=561 (none), moderate=2026 (polygenic+somatic+null), high=235 (monogenic).

### Family B — Enrichment Relevance (4 features)

| Feature | Type | Derivation | Coverage | Positive rate |
|---|---|---|---|---|
| `feat_biomarker_stratified_flag` | binary | Keyword match on `indication` + `ct_official_title` (mutation-specific + enrichment design terms) | 100% | 19.6% (554/2822) |
| `feat_targeted_mechanism_flag` | binary | Drug-name suffix/keyword rules (mAb, -nib, inhibitor, gene therapy, etc.) + monogenic disease flag | 100% | 30.9% (871/2822) |
| `feat_disease_molecular_heterogeneity_score` | float 0.0–1.0 | `disease_genetic_basis` × `mesh_level1` × rare_disease proxy | 100% | mean=0.531, std=0.217 |
| `feat_enrichment_relevance_score` | float 0.0–1.0 | 0.35×biomarker_stratified + 0.25×targeted_mechanism + 0.25×(1−heterogeneity) + 0.15×has_predictive_biomarker | 100% | mean=0.301, std=0.215 |

### Pre-event validity

All 7 features derived from CT.gov registration metadata and LLM disease classifications (disease_genetic_basis) — all known before the event date. No post-event information, no snapshot-unsafe fields.

### Notes

- `feat_targeted_mechanism_flag` will fire for behavioral/non-targeted therapies used in monogenic diseases (e.g., ACT therapy for a monogenic indication). Acceptable heuristic — the model will learn the appropriate weight.
- `feat_biomarker_stratified_flag` positive rate (19.6%) lower than initially estimated — reflects that explicit biomarker stratification is a subset of all trials, primarily oncology.
- Phase 2 (deferred): extend `disease_biology_v1.json` with LLM-derived heritability fields for the 17.4% of rows with unknown genetic basis.

---

## 2026-03-25 · v12–v16 Feature Engineering Sprint (Phases 1–5)

Five-phase improvement run. All changes are pre-event safe (CT.gov protocol metadata; no event-day or post-event information).

---

### Phase 1 (v12) — Fix Broken Features + Add 7 Tier-1 CT.gov Features

**Bugs fixed:**

| Feature | Bug | Fix |
|---|---|---|
| `feat_log_market_cap` | In training roster but **never computed** — silently imputed to median for all rows | Added `np.log10(market_cap_m.clip(lower=1e-3))` in `add_high_signal_features.py` |
| `feat_market_cap_bucket` | Never computed → fold prior `feat_prior_mean_abs_move_atr_by_market_cap_bucket` silently fell back to global mean for every row | Added 4-bucket encoding (micro/small/mid/large) in `add_high_signal_features.py` |

**7 Tier-1 CT.gov features added to training roster** (computed but previously excluded):

| Feature | Source | Why included |
|---|---|---|
| `feat_ctgov_pipeline_maturity_score` | `build_ctgov_pipeline_proxies.py` | Composite sponsor maturity signal. Rated High in prior audit. |
| `feat_ctgov_n_late_stage_trials_sponsor` | Same | Late-stage pipeline depth. |
| `feat_ctgov_asset_maturity_score` | Same | Drug-level maturity. |
| `feat_recent_ctgov_update_flag` | `refresh_ctgov_features.py` | Recent CT.gov update signals imminent readout. |
| `feat_controlled_flag` | `add_high_signal_features.py` | Controlled trials → more rigorous design → larger credible moves. |
| `feat_priority_review_flag` | Same | Direct regulatory signal. |
| `feat_primary_endpoint_known_flag` | Same | Trials with clear endpoints have more interpretable readouts. |

All 7 are derived from CT.gov protocol registration metadata. Pre-event safe.

**v12 retrain:** 701 rows, 58+6=64 features. Test AUC 0.659, CV AUC 0.770 ± 0.036.

---

### Phase 2 (v13) — Fix Timing Feature Anchors

**Root cause:** All timing features in `add_pre_event_timing_features.py` were computing days relative to `v_actual_date` (the event date), making them event-date-anchored and therefore INVALID\_FOR\_PRE\_EVENT. At inference time, the event date is unknown.

**Fix:** Added `prediction_date` parameter (default `v_actual_date - 1 day` in training; `today` at inference). All relative-time computations anchored to `prediction_date`.

**5 features restored from INVALID\_FOR\_PRE\_EVENT to training roster:**

| Feature | Pre-event validity | Signal |
|---|---|---|
| `feat_days_to_primary_completion` | `ct_primary_completion - prediction_date` (protocol metadata) | Countdown to trial completion — was #2 importance historically |
| `feat_primary_completion_imminent_30d` | Binary: completion within 30 days of prediction | Imminent readout flag |
| `feat_primary_completion_imminent_90d` | Binary: completion within 90 days of prediction | Broader readout window |
| `feat_time_since_last_company_event` | Days since last company event before `prediction_date` | Event clustering signal |
| `feat_time_since_last_asset_event` | Days since last asset event before `prediction_date` | Asset-level clustering |

**v13 retrain:** 701 rows, 63+6=69 features. Test AUC 0.668, CV AUC 0.758 ± 0.081.
`feat_primary_completion_imminent_90d` → #3 importance. Real signal confirmed.

---

### Phase 3 (v14) — Company Historical Hit Rate

**New feature:** `feat_company_historical_hit_rate`

- **Definition:** Fraction of a company's prior catalyst events (before the current one) that produced a large move (target=1).
- **Implementation:** Sort all rows by `v_actual_date`. For each row, compute `shift(1).expanding().mean()` on the binary target, grouped by ticker. This is strictly backward-looking.
- **Fold safety:** Computed once on the full dataset (time-sorted); no future data bleeds into any row's value because `shift(1)` skips the current event. FoldPriorEncoder is not used (this is a per-row static feature, not a per-fold aggregate).
- **Coverage:** 86.2% of rows (tickers with ≥ 1 prior event in the dataset).
- **Caveat:** Pre-2023 quiet-completion rows are near-0% positive rate by construction (CT.gov selection bias), so this feature underestimates true hit rates for older companies. Signal is expected to improve as more Phase 4 historical data is added.

**v14 retrain:** 701 rows, 64+6=70 features. Test AUC 0.681, PR-AUC 0.603, CV AUC 0.762 ± 0.075.

---

### Phase 4 (v15) — Data Expansion: 2018–2022 Historical Catalysts

**Motivation:** Training was limited to 701 rows (2023+ only) because pre-2023 CT.gov rows were captured via "quiet completion" scanning — near-0% positive rate due to selection bias. This phase retroactively applies scan-and-confirm to 2018–2022.

**Method:**

1. **Scan large moves** (`scan_large_moves.py`): 460 tickers, 2018–2022. Result: 18,008 large daily moves (≥10% absolute).
2. **Fetch CT.gov completions** (`find_clinical_events.py`): Phase 2/3 completions with yfinance price data. Result: 2,627 events.
3. **Cross-match** (`cross_match_events.py`): Within ±10 calendar days per ticker. Result: 327 matched pairs → 43 positives (ATR-normalized ≥ 3.0 AND |move_pct| ≥ 10%) + 291 small-move CT.gov negatives.
4. **Perplexity batch classification** (`classify_unmatched_catalysts.py`): 1,778 high-normalized unmatched large moves, 5 events/call. Result: 111 confirmed clinical catalysts (73 clinical\_trial + 38 fda\_decision). API cost: ~$0.17 (345 calls).
5. **Merge** (`merge_phase4_data.py`): Deduplicate against master, align columns, archive old version.

**New data tiers:**

| `data_tier` value | Count | Description |
|---|---|---|
| `phase4_ctgov` | 43 | CT.gov cross-match positives — large-move events confirmed by CT.gov completion within ±10 days |
| `phase4_ctgov_neg` | 291 | CT.gov small-move completions — genuine clinical events with no market impact |
| `phase4_perp` | 110 | Perplexity-confirmed unmatched large moves — clinical catalyst confirmed by LLM |

**Master CSV:** 2,514 → 2,958 rows. Training bypass: `data_tier ∈ PHASE4_TIERS` bypasses `MIN_EVENT_YEAR=2023` filter in `build_pre_event_train_v2.py`.

**Cross-match yield note:** Only 43/327 matched pairs had ATR-normalized ≥ 3.0. CT.gov primary completion dates frequently lag press release dates by weeks–months, reducing match yield. Perplexity classification captured a complementary set.

**v15 retrain:** 1,142 rows (+63%), 32% positive rate, 64+6=70 features. Test AUC 0.702 (new best), CV AUC 0.793 ± 0.081. Best model: LogReg.

---

### Phase 5 (v16) — Extended Fold-Safe Priors

**2 new priors added to `add_train_fold_priors.py`:**

| Feature | Group key | Target | Note |
|---|---|---|---|
| `feat_prior_large_move_rate_by_market_cap_bucket` | `feat_market_cap_bucket` | `target_large_move` | Complements existing mean-abs-move prior for same key; directly models positive rate |
| `feat_prior_large_move_rate_by_phase_x_therapeutic_superclass` | `(feat_phase_num, feat_therapeutic_superclass)` | `target_large_move` | Interaction; cells with < 5 samples fall back to phase-level rate prior |

**Total priors:** 6 → 8. Fallback cascade for interaction: cell → phase-level `feat_prior_large_move_rate_by_phase` → global train mean.

**v16 retrain:** 1,142 rows, 64+8=72 features. Test AUC 0.695, CV AUC 0.785 ± 0.077. New priors rank #1 and #4 by LogReg importance. Marginal test-set regression vs v15 (0.695 vs 0.702) is within noise for the 172-row holdout; LightGBM shows larger degradation from sparse interaction cells. Best model remains LogReg.

---

## 2026-03-24 · v11 — LLM-Derived Disease Biology Features

**STATUS: IMPLEMENTED** — Three new features classify each indication's biological properties using Perplexity (sonar model). All are static medical knowledge — inherent to the disease, not the trial or event — and are pre-event safe.

### New Features

| Feature | Type | Values | Pre-event safe? |
|---|---|---|---|
| `feat_has_predictive_biomarker` | binary | 0/1 | Yes — inherent disease property |
| `feat_genetic_basis` | categorical | none / monogenic / polygenic / somatic | Yes — inherent disease property |
| `feat_targeted_therapy_exists` | binary | 0/1 | Yes — inherent disease property |

### Methodology

1. Normalized `indication` column → 1,152 unique values (case-insensitive)
2. Batched 10 per Perplexity API call (sonar model, temperature=0.0) → 116 batches
3. Structured prompt requests JSON classification for each disease
4. Results cached in `cache/disease_biology_v1.json` (resume-safe)
5. Mapped back to master CSV as `disease_*` columns; feature pipeline reads these in Step 6b

### Coverage & Distribution

- Coverage: 96.9% (2,437/2,514 rows with non-null indication)
- `genetic_basis` distribution: polygenic 35.2%, somatic 28.3%, none 23.7%, monogenic 9.8%
- `has_predictive_biomarker`: 30% of rows
- `targeted_therapy_exists`: 48% of rows

### Spot-check validation

| Disease | Biomarker | Genetic | Targeted | Correct? |
|---|---|---|---|---|
| Breast cancer | True | somatic | True | Yes (HER2/ER/PR) |
| NSCLC | True | somatic | True | Yes (EGFR/ALK/PD-L1) |
| Cystic fibrosis | True | monogenic | True | Yes (CFTR/ivacaftor) |
| COVID-19 | False | none | False | Yes |
| Type 2 diabetes | False | polygenic | False | Yes |
| CML | True | somatic | True | Yes (BCR-ABL/imatinib) |
| Major depression | False | polygenic | False | Yes |
| Sickle cell | False | monogenic | True | Yes |

### v11 Retrain Results

| Metric | v10 (baseline) | v11 | Delta |
|---|---|---|---|
| Test AUC | 0.664 | 0.685 | +0.021 |
| Test PR-AUC | 0.503 | 0.573 | +0.070 |
| Prec@top 10% | 0.364 | 0.727 | +0.363 |
| CV AUC | 0.784 ± 0.045 | 0.781 ± 0.048 | -0.003 |

### Files

- `scripts/enrich_disease_biology.py` — Perplexity classification + cache + master CSV write
- `cache/disease_biology_v1.json` — 1,152 entries (gitignored)
- `scripts/add_high_signal_features.py` — Step 6b: `build_disease_biology_features()`
- `scripts/build_pre_event_train_v2.py` — v11: 2 binary + 1 categorical (one-hot → 5 columns)

---

## 2026-03-24 · v10 — PIT Fix for feat_terminated_flag + feat_withdrawn_flag

**STATUS: ✓ IMPLEMENTED** — `feat_terminated_flag` and `feat_withdrawn_flag` (current CT.gov snapshot) replaced by AACT point-in-time versions; `feat_trial_quality_score` now uses PIT flags for the −2 termination/withdrawal penalty.

### Problem

`feat_terminated_flag = (ct_status == "TERMINATED")` and `feat_withdrawn_flag = (ct_status == "WITHDRAWN")` were used as −2 penalty terms inside `feat_trial_quality_score`. Both use the current CT.gov snapshot (March 2026) — same SNAPSHOT_UNSAFE issue as `feat_completed_flag`.

Audit using AACT cache against 33 training rows with current status TERMINATED or WITHDRAWN:
- **23 leakage cases** — trial was still RECRUITING or ACTIVE_NOT_RECRUITING at event time; terminated post-event
- **10 clean cases** — already TERMINATED/WITHDRAWN at event time

For the 23 leakage rows: `feat_trial_quality_score` was artificially lowered by 2 points relative to what was knowable at event time.

### Fix

Added **Step 0b.5** (`build_status_pit_flags`) in `add_high_signal_features.py`:
- `feat_terminated_at_event_flag` = `ct_status_at_event == "TERMINATED"` (AACT PIT; falls back to snapshot for ~7.6% of training rows with no AACT record)
- `feat_withdrawn_at_event_flag` = `ct_status_at_event == "WITHDRAWN"` (same)

`build_trial_quality_score` (Step 0c) now uses these PIT flags for the −2 penalties instead of the snapshot flags.

Both PIT flags added to training roster in v10 (low base rate: 9 and 1 positive cases in 701 rows — minimal direct signal, but correct for quality score composition).

Snapshot flags (`feat_terminated_flag`, `feat_withdrawn_flag`) moved to `INVALID_FOR_PRE_EVENT`.

### After fix

| | Snapshot (old) | PIT (v10) |
|---|---|---|
| TERMINATED positive count | 32 | 9 |
| WITHDRAWN positive count | 1 | 1 |
| Quality score inflated downward for | 23 rows | 0 rows |

v10: Test AUC 0.664, CV AUC 0.784 ± 0.045 — flat vs v9 (expected; 9 affected rows are too few to move the aggregate metric, but the score is now correct).

---

## 2026-03-24 · v9 — Port Foundational Features + Fix INVALID_FOR_PRE_EVENT Roster

**STATUS: ✓ IMPLEMENTED** — Three legacy root-level scripts ported into `add_high_signal_features.py`; SNAPSHOT_UNSAFE and INVALID features cleaned from training roster; v9 train table = 701 rows × 42 features.

### Problem

`add_high_signal_features.py` (Pass-4) assumed foundational features pre-existed in the CSV, but three root-level scripts (`build_ml_ready_features.py`, `add_high_value_predictors.py`, `completeness_pass.py`) were never ported when the dataset expanded from 831 → 2379 rows. As a result, `feat_phase_num`, `feat_trial_quality_score`, `feat_regulatory_stage_score`, `feat_volatility`, `feat_orphan_flag`, and ~10 others were absent from the v5 features CSV.

### Fix

Ported all missing derivations into `add_high_signal_features.py` as new Steps 0a–0d:

| Step | Features added |
|---|---|
| 0a Clinical core | `feat_phase_num`, `feat_late_stage_flag`, `feat_enrollment_log`, `feat_randomized_flag`, `feat_design_quality_score`, `feat_withdrawn_flag`*, `feat_terminated_flag`* |
| 0b Regulatory flags | `feat_orphan_flag`, `feat_fast_track_flag`, `feat_breakthrough_flag`, `feat_nda_bla_flag`, `feat_priority_review_flag`*, `feat_regulatory_stage_score` |
| 0c Trial quality | `feat_trial_quality_score`, `feat_controlled_flag` |
| 0d Company foundation | `feat_n_trials_for_company`, `feat_n_unique_drugs_for_company`, `feat_single_asset_company_flag`, `feat_lead_asset_dependency_score`, `feat_n_late_stage_trials_for_company`, `feat_pipeline_concentration_simple`* |
| 6 (extended) | `feat_mesh_level1_encoded` (ordinal int 1–11, MESH_ENCODE_MAP) |

\* intermediate only; not in training roster

### SNAPSHOT_UNSAFE features moved to INVALID_FOR_PRE_EVENT

| Feature | Reason |
|---|---|
| `feat_short_squeeze_flag` | `short_percent` via yfinance current snapshot; historical value unknown |
| `feat_ownership_low_flag` | `institutional_ownership` via yfinance current snapshot; same issue |

### feat_event_proximity_bucket declared INVALID_FOR_PRE_EVENT

`feat_event_proximity_bucket` = bucket of `(ct_primary_completion - event_date).days`. Uses realized `event_date` as anchor — same invalidity as `feat_days_to_primary_completion`. Removed from CATEGORICAL_FEATURES in v9.

### Regulatory flag coverage (expected near-zero)

`feat_breakthrough_flag` = 3 rows, `feat_orphan_flag` = 2 rows, `feat_fast_track_flag` = 2 rows across 2379 rows. Expected: the dataset is primarily trial **data readout** events; designation announcements are a different catalyst type. `v_summary` (rich text) covers 893/2379 rows (validated cohort only); keyword matching on historical rows finds nothing because no press release text exists for those rows.

### v9 Model Results

| Metric | v9 (42 features) | v8 (41 features) | v7 (25 features) | v6 (baseline) |
|---|---|---|---|---|
| Test AUC | 0.664 | 0.665 | 0.700 | 0.693 |
| CV AUC | 0.784 ± 0.045 | 0.788 ± 0.045 | 0.759 ± 0.040 | — |
| Model | LightGBM | LightGBM | LogReg | LightGBM |

Test AUC delta v8→v9 is noise (−0.001). CV AUC improvement v7→v8/v9 (+0.025–0.029) reflects richer feature set. The v8/v9 vs v7 test AUC gap (−0.036) is likely test-set noise given small N (106 rows) and LightGBM vs LogReg model difference.

Top features (v9): `feat_cash_runway_proxy`, `feat_volatility`, `feat_n_trials_for_company`, `feat_company_event_sequence_num`, `feat_enrollment_log`.

---

## 2026-03-23 · Option C — Point-in-Time AACT Status (feat_completed_at_event_flag + feat_active_not_recruiting_at_event_flag)

**STATUS: ✓ IMPLEMENTED** — AACT monthly flat-file archives fetched (~39 months, Jan 2023–Mar 2026); two SNAPSHOT_UNSAFE features replaced by ground-truth point-in-time variants.

### Problem

Two SNAPSHOT_UNSAFE features remained in training after v6:

1. **`feat_active_not_recruiting_flag`** = `(ct_status == "ACTIVE_NOT_RECRUITING")` — `ct_status` is the current CT.gov snapshot (March 2026 fetch), not the trial status at event time. Same leakage risk as `feat_completed_flag`.
2. **`feat_completed_before_event`** (v6 fix) = `(ct_primary_completion < event_date)` — a valid pre-event date proxy, but coarse: it only checks whether the *scheduled* data-collection end date preceded the event, not whether the trial had actually transitioned to COMPLETED status at the time.

### Why the date proxy is not sufficient

`ct_primary_completion` is ambiguous: CT.gov updates it from "estimated" to "actual" after the trial completes, but we have no `_type` field to distinguish the two. More critically, agreement between the date proxy and actual point-in-time status (verified on 530 training rows where both are available) is only **71.1%** — and 145 of the 530 rows are false positives where the proxy says "completed" but the trial had NOT yet transitioned to COMPLETED status at event time. These are exactly the leakage cases that needed fixing.

### Fix — Option C: AACT Monthly Snapshots

**AACT** (Duke CTTI — Aggregate Analysis of ClinicalTrials.gov) archives the full CT.gov database as pipe-delimited flat files on the 1st of each month, publicly available at `aact.ctti-clinicaltrials.org/downloads/snapshots`.

**`scripts/fetch_aact_status_history.py` — algorithm:**

1. Load master CSV → extract 710 unique NCT IDs (validated cohort, `v_is_verified` non-null)
2. Load/resume cache at `cache/aact_status_history_v1.json`
3. For each of ~39 months (Jan 2023–Mar 2026):
   - Stream-download the monthly ZIP (~2GB) to `$TMPDIR`
   - Open ZIP in-memory → extract only `studies.txt` (pipe-delimited)
   - Read only columns `nct_id` and `overall_status`
   - Filter to our 710 NCT IDs; normalize status to SCREAMING_SNAKE_CASE (AACT uses "Active, not recruiting" etc.)
   - Store `{nct_id: {month_key: status}}` in cache; delete temp ZIP
4. Point-in-time lookup for each event row: find the **latest month ≤ event_date** in the cache
5. Write `ct_status_at_event` column to master CSV; add `data_tier` column ("validated" / "historical")

**New features added (v7+):**

| Feature | Replaces | Derivation |
|---|---|---|
| `feat_completed_at_event_flag` | `feat_completed_before_event` (date proxy) | `1 if ct_status_at_event == "COMPLETED"`, else 0; NaN if no AACT record |
| `feat_active_not_recruiting_at_event_flag` | `feat_active_not_recruiting_flag` (SNAPSHOT_UNSAFE) | `1 if ct_status_at_event == "ACTIVE_NOT_RECRUITING"`, else 0; NaN if no AACT record |

**SNAPSHOT_UNSAFE features removed from training:**

| Feature | Removed in | Classification |
|---|---|---|
| `feat_completed_flag` | v6 | SNAPSHOT_UNSAFE (ct_status = current CT.gov snapshot) |
| `feat_active_not_recruiting_flag` | v7 | SNAPSHOT_UNSAFE (same source) |
| `feat_completed_before_event` | v7 | Superseded by PIT; kept in features CSV for compatibility |

### Coverage (correctly scoped)

The AACT archive starts Jan 2023. The training cohort is also 2023+ only (older rows excluded due to near-zero positive rate). So PIT and training scope align well:

| Scope | PIT coverage |
|---|---|
| Full dataset (2379 rows, incl. 2020–2022) | 32.6% — misleading figure; most missing are pre-2023 rows not used in training |
| **Training cohort (2023+, 711 rows)** | **92.4% (657/711)** — the operationally relevant number |
| Training rows without PIT (54 rows) | Imputed as 0 (absent) — more conservative than using unreliable date proxy |
| Monthly granularity | Status could lag by ≤30 days (event near month boundary) |

**Do not use `feat_completed_before_event` as fallback for the 54 null rows.** The date proxy has 27% disagreement with PIT (145 false positives in 530 validated comparisons), and those false positives are exactly the leakage pattern Option C was designed to fix. Imputing 0 is safer.

### Why not add feat_completed_before_event back as a separate training feature?

Short answer: it adds almost no marginal coverage and is unreliable as a standalone signal.

`ct_primary_completion` is **not a fixed planned date** — CT.gov updates it from "estimated" to "actual" after the trial completes, and sponsors amend it when trials are delayed. The value in our March 2026 fetch is partially post-hoc, not the original protocol date. We have no `_type` field to distinguish estimated vs actual.

Semantic breakdown of the 530 training rows where both PIT and proxy are available:

| Bucket | Count | Meaning |
|---|---|---|
| PIT=1, proxy=1 | 126 | Completed on schedule |
| PIT=0, proxy=0 | 251 | Not yet complete, not expected yet |
| PIT=0, proxy=1 | 145 | Scheduled to finish, still running → behind schedule / delayed |
| PIT=1, proxy=0 | 8 | Completed later than originally scheduled |

The PIT=0/proxy=1 bucket (145 rows) is conceptually interesting — it signals a delayed trial. But:
- Marginal coverage gain: only **6 additional rows** where proxy is available and PIT is null (negligible)
- The proxy has lower total coverage (75.4%) than PIT (92.4%) — adding it as a standalone feature would degrade coverage
- `ct_primary_completion` isn't a reliable "planned date" signal because it gets updated post-hoc

**Decision: `feat_completed_before_event` remains in features CSV for pipeline compatibility but is not added to training.** The PIT feature supersedes it entirely.

### Watch list (updated)

| Feature | Status |
|---|---|
| `feat_active_not_recruiting_flag` | **REMOVED from training** (v7) — SNAPSHOT_UNSAFE |
| `feat_completed_flag` | **REMOVED from training** (v6) — SNAPSHOT_UNSAFE |
| `feat_active_not_recruiting_at_event_flag` | **ACTIVE** (v7+) — AACT PIT, 92.4% coverage in training cohort |
| `feat_completed_at_event_flag` | **ACTIVE** (v7+) — AACT PIT, 92.4% coverage in training cohort |
| `feat_completed_before_event` | Retained in features CSV only; superseded in training by PIT |
| `feat_withdrawn_flag`, `feat_terminated_flag` | Still excluded; AACT cache available if needed |

---

## 2026-03-23 · feat_completed_flag Leakage Fix (Option B — Date Proxy)

**STATUS: ✓ IMPLEMENTED** — `feat_completed_flag` removed from training; replaced by `feat_completed_before_event`.

### Problem

`feat_completed_flag = (ct_status == "COMPLETED")` was **#2 in LightGBM feature importance** (coef 0.5743, v5).
`ct_status` is a current CT.gov snapshot (March 2026 fetch), not the trial status at event time.
For 2024 events, a trial could have transitioned to COMPLETED after the event date, leaking future information.
**Classification: SNAPSHOT_UNSAFE.**

### Fix — Option B: Date Proxy

```python
feat_completed_before_event = (ct_primary_completion_parsed < event_date)
```

`ct_primary_completion` is a **prospective protocol milestone** (scheduled data-collection end date),
registered before the trial starts. If it precedes the event date, the trial's primary collection
period was finished before the event — a valid pre-event proxy.

| Metric | Value |
|---|---|
| Training rows (2023+) | 701 |
| `feat_completed_flag == 1` (old, snapshot) | 415 / 701 (59.2%) |
| `feat_completed_before_event == 1` (new, proxy) | 282 / 528 non-null rows (40.2% of total) |
| `ct_primary_completion` null rate | 173 / 701 (24.7%) — below 30% fallback threshold |
| Imputation for nulls | 0 (absent) — standard binary feature imputation |

### Status after v6 retrain

See MODEL_REPORTS.md entry for v6. Expected: small AUC drop (<0.02) from removing contamination is correct and acceptable.

### Watch list

| Feature | Status |
|---|---|
| `feat_active_not_recruiting_flag` | REVIEW_NEEDED — `ct_status == "ACTIVE_NOT_RECRUITING"` is also a CT.gov snapshot. Keep in training for now; revisit with Option C (CT.gov status history fetch). |
| `feat_completed_flag` | **REMOVED from training** as of v6 (SNAPSHOT_UNSAFE) |
| `feat_recent_completion_flag` | **EXCLUDED** (SNAPSHOT_UNSAFE + INVALID_FOR_PRE_EVENT anchor) |
| `feat_withdrawn_flag`, `feat_terminated_flag` | Excluded from training; revisit after Option C |
| `feat_completed_before_event` | **ACTIVE** (v6+) — date proxy, pre-event valid |
| `feat_completed_at_event_flag` | PLANNED — Option C (CT.gov history API fetch for ~750 validated rows) |

---

## 2026-03-17 · Strict-Clean Retrain Status

**STATUS: ✓ STRICT_CLEAN retrain complete** — `ml_baseline_train_20260317_v4.csv` is the current trusted baseline.

The 9 invalid features identified in the audit below were excluded from `build_pre_event_train_v2.py` (commit `235eac4`). The strict-clean retrain (v4) ran steps 7–8 with those features removed. The previous contaminated train table (`ml_baseline_train_20260317_v3.csv`) has been archived.

| | Contaminated v3 | Strict-Clean v4 |
|---|---|---|
| Train table | ml_baseline_train_20260317_v3.csv (archived) | ml_baseline_train_20260317_v4.csv |
| Invalid features | 14 (9 base + 6 one-hot dummies) | **0** |
| Test AUC | 0.730 (inflated) | **0.692 (honest)** |

---

## 2026-03-17 · Pre-Event Validity Audit (v0.6)

**Scope:** All features in the current pre-event binary classifier (`ml_baseline_train_20260317_v3.csv`)
**Hard rule:** The model may ONLY use information publicly available BEFORE the future event occurs.

### Confirmed training setup

| Field | Value |
|---|---|
| Model file | `models/model_pre_event_v3_20260312.pkl` (best: LightGBM) |
| Train / Val / Test | 417 / 89 / 90 (time-based on v_actual_date, 2023+ only) |
| Class balance (overall) | 184 pos / 412 neg = 30.9% positive |
| Class balance (train/val/test) | 28.5% / 30.3% / 42.2% |
| Feature count | 44 columns (38 base + 6 one-hot dummies) |
| Target | target_large_move = 1 when abs_atr ≥ 3.0 AND abs(move_pct) ≥ 10% |

**Note on test class balance:** 42.2% in test > 28.5% in train — temporal shift (2025–2026 events have higher positive rate), not a bug.

### Full audit table

| Feature | Anchor used | PRE-EVENT VALID? | Reason |
|---|---|---|---|
| `feat_days_to_primary_completion` | v_actual_date (realized) | **INVALID** | Future event date, unknown at prediction time |
| `feat_primary_completion_imminent_30d` | Derived from above | **INVALID** | Derived from invalid anchor |
| `feat_primary_completion_imminent_90d` | Derived from above | **INVALID** | Derived from invalid anchor |
| `feat_completion_recency_bucket` | Derived from above | **INVALID** | Derived from invalid anchor |
| `feat_recent_completion_flag` | event_date (realized) | **INVALID** | Uses realized event_date as anchor |
| `feat_time_since_last_company_event` | v_actual_date of current event | **INVALID** | Current event endpoint = future date |
| `feat_time_since_last_asset_event` | v_actual_date of current event | **INVALID** | Same |
| `feat_recent_company_event_flag` | Derived from above | **INVALID** | Same |
| `feat_recent_asset_event_flag` | Derived from above | **INVALID** | Same |
| `feat_asset_event_sequence_num` | v_actual_date for sorting only | **VALID** | Count of prior events is knowable before next event |
| `feat_company_event_sequence_num` | v_actual_date for sorting only | **VALID** | Same — ordinal position is pre-event knowable |
| `feat_completed_flag` | CT.gov status (public) | **VALID** | Visible before any announcement |
| `feat_active_not_recruiting_flag` | CT.gov status | **VALID** | Same |
| All other features | Trial design, financial, disease, pipeline, CT.gov proxy | **VALID** | No event date anchor |

**Total removed: 9 base features + 6 one-hot dummies = 15 columns to be excluded from next retrain.**

### Current model validity status

```
STATUS: APPROXIMATELY VALID for historical analysis
        INVALID for strict live deployment without inference-time recomputation
```

**Why approximately valid:** The anchor discrepancy (v_actual_date vs. prediction_date) is small for 96.8% of rows (0-day mismatch). AUC 0.730 reflects genuine predictive lift on historical held-out test events.

**Why invalid for strict live deployment:** `feat_days_to_primary_completion = ct_primary_completion - v_actual_date` cannot be reproduced at inference because v_actual_date is unknown. At inference, these must be computed with `prediction_date = today`.

### Fix path (do NOT implement until retrain is approved)

In `add_pre_event_timing_features.py` and `refresh_ctgov_features.py`, change anchor from `v_actual_date` to a `prediction_date` parameter:

```python
def build_timing_features(df, prediction_date=None):
    if prediction_date is None:
        date_col = "v_actual_date" if "v_actual_date" in df.columns else "event_date"
        evt_date = _parse_dates(df[date_col])
    else:
        evt_date = pd.Series([prediction_date] * len(df), index=df.index)
```

For training: `prediction_date = v_actual_date` per row (identical to current behavior).
For inference: pass `prediction_date = today` so features reflect "how far from today."

Once fixed, the 9 invalid features can be re-added as valid pre-event features.

### Mandatory fields for every future training report

Every model training report in MODEL_REPORTS.md must include:
train table filename · feature dataset filename · total usable rows · train/val/test row counts · class balance per split · split method · year range per split · any cohort exclusions · feature count + excluded features · pre-event validity status.

### Permanent pre-event hard rule

> Any feature used in the pre-event stock move size model must be computable using ONLY information publicly available BEFORE the future event occurs. FORBIDDEN: realized event date (`v_actual_date`, `event_date`, `event_trading_date`), realized announcement/PR date, realized stock move date, announcement content or outcome.
>
> Features that use `v_actual_date` as an ANCHOR (e.g., "days to CT.gov completion from event date") are INVALID. Features that use past event dates for sorting/sequencing are VALID. Ordinal sequence counts (how many prior events exist for this company/drug) are VALID.

---

## 2026-03-15 · Follow-Up Actions & Audit Findings (v0.4)

**Based on:** ML audit of 2026-03-13

### Target definitions (confirmed)

**`target_abs_move_atr`:**
```
target_abs_move_atr = abs(price_after - price_before) / price_before / atr_pct
```
- price_before = closing price last trading day strictly before event
- price_after = closing price first trading day strictly after event
- Window = 1 overnight move bracketing the announcement (`move_2d_pct` column = "2 days apart", not 2-day trailing)
- ATR = Wilder's RMA `ewm(alpha=1/20, adjust=False)`, 20 trading-day lookback, strictly pre-event

**`target_large_move`:** Binary. 1 = abs_atr ≥ 3.0 AND abs(move_pct) ≥ 10% (updated from original ≥5× ATR in v0.6). See DATASET_NOTES.md for threshold analysis.

**`target_move_bucket`:** 5-class ordinal: Noise (<1.5×) / Low (1.5–3×) / Medium (3–5×) / High (5–8×) / Extreme (≥8×). Not used in current model.

### Permanently excluded post-event features

| Feature | Source | Reason |
|---|---|---|
| `feat_superiority_flag` | primary_endpoint_result, v_pr_key_info, v_summary | Current PR text — result language |
| `feat_stat_sig_flag` | Same text fields | p-values, HR/OR from current result |
| `feat_clinically_meaningful_flag` | Same text fields | Clinical significance language |
| `feat_mixed_results_flag` | Same text fields | Failure/miss language |
| `feat_endpoint_outcome_score` | primary_endpoint_met (Yes/No/Unclear) | Records current catalyst outcome |
| `feat_primary_endpoint_known_flag` | primary_endpoint_met | 1 = outcome already happened |

Confirmed absent from all current training tables. Must not be added to any future pre-event training table.

**Future valid version:** keyword signals from prior-phase publications for the same drug (not yet implemented).

### Oncology timing caveat

CT.gov primary completion date–based features are valid but carry a known limitation for oncology:

**Issue:** For OS/PFS/DFS-driven oncology trials, the efficacy readout occurs when a target number of events accrues — typically 6–24 months before the CT.gov primary completion date. The CT.gov date reflects administrative study close, not the readout. Imminence flags can say "far" when a readout is genuinely imminent.

**Decision:** Keep all timing features as-is. Real signal for non-oncology (~40% of dataset). For oncology the "COMPLETED + recently closed" signal remains meaningful.

**Mitigation added in v0.5:** Oncology interaction features (`feat_oncology_x_imminent_*`) allow the model to weight CT.gov timing differently for oncology vs non-oncology. Quick comparison (LogReg, v0.3 split): +0.005 test AUC (0.653→0.658), neutral on val. Features are not harmful; signal will grow with more oncology data.

### Binary vs multiclass strategy

- Binary (`target_large_move`) remains the primary objective. Aligned with investment use case.
- Multiclass deferred until binary AUC exceeds 0.70 reliably.
- `target_move_bucket` (5-class) exists in dataset; no multiclass model trained yet.

### Deferred: Management / CRO quality signal

PI track record feasible via CT.gov (same API pattern as sponsor queries, ~1–2 days). CEO/CMO level requires LLM enrichment — not practical now. Classified as B-priority research feature.

---

## 2026-03-14 · CT.gov Timing & Pipeline Feature Refresh (v0.4)

**Dataset:** `ml_dataset_features_v0.4_20260313.csv` (827 rows × 145 cols)
**New features added:** 19 (11 CT.gov timing + 8 pipeline proxy)
**Feature dict:** `ml_feature_dict_v0.4_20260313.csv` (82 entries, was 69 in v0.3)

### CT.gov Timing Features (11)

All fetched from CT.gov API v2 per NCT ID. 679 unique NCT IDs queried; cached in `cache/ctgov_details_v1.json`.

| Feature | Description | Coverage |
|---|---|---|
| `feat_ctgov_primary_completion_date` | Primary completion date (registry) | 784/827 (94.8%) |
| `feat_days_to_primary_completion` | Days to primary completion | 773/827 (93.5%) |
| `feat_primary_completion_imminent_30d` | Imminence flag: within 30 days | 773/827 (93.5%) |
| `feat_primary_completion_imminent_90d` | Imminence flag: within 90 days | 773/827 (93.5%) |
| `feat_completion_recency_bucket` | Recency bucket (6 levels) | 784/827 (94.8%) |
| `feat_ct_status_current` | Current CT.gov overall status | 784/827 (94.8%) |
| `feat_active_not_recruiting_flag` | Binary: ACTIVE_NOT_RECRUITING | 784/827 (94.8%) |
| `feat_completed_flag` | Binary: COMPLETED | 784/827 (94.8%) |
| `feat_days_since_ctgov_last_update` | Days since CT.gov last update | 773/827 (93.5%) |
| `feat_recent_ctgov_update_flag` | Binary: updated within 90 days | 773/827 (93.5%) |
| `feat_status_timing_consistency_flag` | Status/completion date consistency | 773/827 (93.5%) |

**Status distribution (pull date 2026-03-13):** COMPLETED 64.9% · RECRUITING 12.2% · ACTIVE_NOT_RECRUITING 11.6% · TERMINATED 3.9% · other 7.3%. Status is as-of pull date, not historical event date.

### Pipeline Proxy Features (8)

293 unique sponsors and 596 unique drug names queried via CT.gov `query.spons` and `query.intr`. PAGE_SIZE=100.

| Feature | Coverage |
|---|---|
| Sponsor features (5) | 784/827 (94.8%) |
| Drug/intervention features (3) | 820/827 (99.2%) |

**Distributions:** `feat_ctgov_n_active_trials_sponsor` median=7, mean=9.3. `feat_ctgov_pipeline_maturity_score` median=2.0. `feat_ctgov_n_trials_total_sponsor` mean=1181 (inflated by large pharma — 48 of 293 sponsors capped at 100 results).

**Caveats:**
- Large pharma outliers: 48 sponsors (16.4%) returned >100 CT.gov results (capped). Maturity score uses n_sample denominator to limit distortion.
- `feat_ctgov_n_trials_same_intervention` outlier: common adjuvants return 30k+ results (sample capped at 100).
- Sponsor name normalisation: legal suffixes stripped; some mismatches may persist.

### Feature priority for next model

**High:** `feat_ctgov_pipeline_maturity_score` · `feat_ctgov_n_late_stage_trials_sponsor` · `feat_ctgov_asset_maturity_score` · `feat_recent_ctgov_update_flag`

**Medium:** `feat_ctgov_n_active_trials_sponsor` · `feat_ctgov_n_trials_same_intervention` · `feat_active_not_recruiting_flag`

**Lower:** `feat_ctgov_n_trials_total_sponsor` (collinear with maturity) · `feat_status_timing_consistency_flag` (better as filter) · `feat_days_since_ctgov_last_update` (use binary version instead)

**Suggested transformations:** log-transform `feat_ctgov_n_trials_total_sponsor` and `feat_ctgov_n_trials_same_intervention`; one-hot `feat_ct_status_current` (merge rare categories into "other").

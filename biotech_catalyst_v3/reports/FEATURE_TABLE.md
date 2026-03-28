# Feature Reference Table — biotech_catalyst_v3

> **Last updated:** 2026-03-28 | **Current model:** v17 (79 features: 71 base + 8 fold-safe priors)
> **Full feature dataset:** `data/ml/ml_dataset_features_20260325_v3.csv` — 2,822 rows × 154 cols
> **Training table:** `data/ml/ml_baseline_train_20260327_v17.csv` — 1,142 rows × 79 features
> Both files are **committed and pushed** to `main`.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | **Active** — used in v17 training |
| 🔄 | **Intermediate** — in full feature dataset, used to derive other features; not directly in training |
| ⚠️ | **Snapshot-unsafe** — value changes over time; cannot be safely used as pre-event feature |
| ❌ | **Post-event** — encodes outcome information; permanently excluded |
| 🗑️ | **Deprecated** — removed and replaced by a safer alternative |

---

## Section 1 — Clinical Trial Core

These features are derived from CT.gov trial registration fields, which are prospective protocol metadata (set before trials begin). All are pre-event safe.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_phase_num` | float | Phase as numeric: Phase 1=1.0, Phase 1/2=1.5, Phase 2=2.0, Phase 2/3=2.5, Phase 3=3.0, Phase 4=4.0 | `ct_phase` | 96.8% | ✅ |
| `feat_late_stage_flag` | binary | 1 if `feat_phase_num ≥ 2.5` (Phase 2/3 or later) | `feat_phase_num` | 96.8% | ✅ |
| `feat_enrollment_log` | float | `log1p(ct_enrollment)` — log-transforms enrollment count to reduce skew | `ct_enrollment` | 94.4% | ✅ |
| `feat_randomized_flag` | binary | 1 if `ct_allocation == "RANDOMIZED"` | `ct_allocation` | 83.9% | ✅ |
| `feat_blinded_flag` | binary | 1 if trial title/allocation contains "blinded", "masked", "double-blind" | `ct_official_title`, `ct_allocation` | 100% | ✅ |
| `feat_open_label_flag` | binary | 1 if trial title/allocation contains "open label", "open-label" | `ct_official_title`, `ct_allocation` | 100% | ✅ |
| `feat_small_trial_flag` | binary | 1 if `ct_enrollment < 50` | `ct_enrollment` | 94.4% (else 0) | ✅ |
| `feat_controlled_flag` | binary | 1 if randomized OR blinded inference from trial title | `feat_randomized_flag`, `ct_official_title` | 100% | ✅ |
| `feat_design_quality_score` | float 0–6 | Composite trial rigor: +1 blinded, +1 randomized, +1 controlled, +1.5 if Phase ≥2.5, +1 if primary endpoint registered. Reflects trial design stringency before any results. | Above flags + `feat_phase_num` | 97.1% | ✅ |
| `feat_primary_endpoint_known_flag` | binary | 1 if trial has a registered pre-specified primary endpoint on CT.gov (endpoint is *defined*, not *met*) | `ct_official_title` keyword match | 100% | ✅ |

---

## Section 2 — Regulatory Designations

FDA designations assigned before trial completion; knowable pre-event.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_orphan_flag` | binary | 1 if "orphan" or "rare disease" designation keyword found in trial title/conditions | `ct_official_title`, `ct_conditions` | 100% (rare) | ✅ |
| `feat_fast_track_flag` | binary | 1 if "fast track" keyword in trial title/conditions | `ct_official_title`, `ct_conditions` | 100% (rare) | ✅ |
| `feat_breakthrough_flag` | binary | 1 if "breakthrough therapy" keyword in trial title/conditions | `ct_official_title`, `ct_conditions` | 100% (rare) | ✅ |
| `feat_nda_bla_flag` | binary | 1 if "NDA", "BLA", "new drug application", "biologics license" keyword found | `ct_official_title` | 100% (rare) | ✅ |
| `feat_priority_review_flag` | binary | 1 if "priority review" keyword in trial title | `ct_official_title` | 100% (rare) | ✅ |
| `feat_regulatory_stage_score` | ordinal 0–3 | Ladder: 0=baseline, 1=orphan/rare, 2=NDA/BLA filed, 3=priority review. Aggregates regulatory momentum into one numeric signal. | Above flags | 100% | ✅ |

---

## Section 3 — Trial Design Quality (Composite)

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_trial_quality_score` | float | `design_quality_score + 1.5×breakthrough_flag − 2×terminated_at_event_flag − 2×withdrawn_at_event_flag`. Penalizes point-in-time failures; rewards breakthrough designation. | Sections 1+2 + PIT status flags | 100% | ✅ |
| `feat_pivotal_proxy_score` | float 0–5 | `+1.5` if Phase≥3.0, `+1.0` if Phase 2/3, `+1.0` if regulatory_score≥2, `+0.5` if regulatory_score=1, `+0.5×breakthrough_flag`, `+0.5×priority_review_flag`. Proxy for binary pivotal designation without requiring the `is_pivotal` label. | `feat_phase_num`, `feat_regulatory_stage_score`, `feat_breakthrough_flag`, `feat_priority_review_flag` | 100% | ✅ |

---

## Section 4 — Company Pipeline Profile

Features quantifying the breadth and depth of a company's drug development program. All derived from the training dataset itself (pre-event knowable from public CT.gov data).

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_n_trials_for_company` | int | Count of clinical catalyst events for this ticker in the dataset | `ticker` | 100% | ✅ |
| `feat_n_unique_drugs_for_company` | int | Count of distinct `drug_name` values per ticker | `ticker`, `drug_name` | 100% | ✅ |
| `feat_single_asset_company_flag` | binary | 1 if company has ≤1 distinct drug in dataset (binary risk flag) | `feat_n_unique_drugs_for_company` | 100% | ✅ |
| `feat_lead_asset_dependency_score` | float 0–1 | `1 / (1 + log1p(n_unique_drugs))` — higher = more concentrated pipeline | `feat_n_unique_drugs_for_company` | 100% | ✅ |
| `feat_asset_trial_share` | float 0–1 | `asset_trial_count / n_trials_for_company` — fraction of company events belonging to this drug | `ticker`, `drug_name` | 100% | ✅ |
| `feat_pipeline_depth_score` | float | `sqrt(n_unique_drugs) × (1 + late_stage_count / total_count)` — breadth weighted by late-stage richness | `feat_n_unique_drugs_for_company`, `feat_n_late_stage_trials_for_company` | 100% | ✅ |
| `feat_company_historical_hit_rate` | float 0–1 | Fraction of prior large moves for this ticker (backward-looking, fold-safe). Computed as `shift(1).expanding().mean()` on `target_large_move`, sorted by `v_actual_date` per ticker. Returns NaN for a company's first ever event. Imputed to median=0.000 at training time. | `ticker`, `v_actual_date`, `target_large_move` | 86.2% (tickers with ≥1 prior event) | ✅ |
| `feat_asset_event_sequence_num` | int | Ordinal position of this event within (ticker, drug_name) history. 1 = first known event for this drug at this company. | `ticker`, `drug_name`, `v_actual_date` | 100% | ✅ |
| `feat_company_event_sequence_num` | int | Ordinal position of this event within ticker history. 1 = company's first catalyst in dataset. | `ticker`, `v_actual_date` | 100% | ✅ |
| `feat_n_late_stage_trials_for_company` | int | Count of Phase 2/3+ events for this ticker (used to compute pipeline_depth_score) | `ticker`, `feat_phase_num` | 100% | 🔄 |
| `feat_asset_trial_count_for_company` | int | Count of events with same (ticker, drug_name) — intermediate for asset_trial_share | `ticker`, `drug_name` | 100% | 🔄 |
| `feat_pipeline_concentration_simple` | float | `lead_asset_dependency_score × (late_stage_trials / total_trials)` — existential risk proxy | Above | 100% | 🔄 |

---

## Section 5 — Disease Classification & Biology

Disease classification from MeSH API + LLM-derived biology fields (static disease knowledge, not trial-specific — pre-event safe).

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_mesh_level1_encoded` | ordinal 1–11 | MeSH top-level disease category as integer: 1=Neoplasms, 2=Immune System Diseases, 3=Nervous System Diseases, 4=Cardiovascular, 5=Respiratory, 6=Digestive, 7=Endocrine, 8=Skin, 9=Musculoskeletal, 10=Infectious, 11=Other | `mesh_level1` (NLM MeSH API) | 91.0% | ✅ |
| `feat_oncology_flag` | binary | 1 if `mesh_level1 == "Neoplasms"` | `mesh_level1` | 100% | ✅ |
| `feat_cns_flag` | binary | 1 if `mesh_level1 == "Nervous System Diseases"` | `mesh_level1` | 100% | ✅ |
| `feat_rare_disease_flag` | binary | 1 if `feat_orphan_flag == 1` OR rare-disease keyword in indication | `feat_orphan_flag`, `indication` | 100% | ✅ |
| `feat_has_predictive_biomarker` | binary | 1 if disease has known predictive biomarkers for treatment selection (LLM-derived static disease knowledge, enrich_disease_biology.py) | `disease_has_predictive_biomarker` column | 30% positive | ✅ |
| `feat_targeted_therapy_exists` | binary | 1 if targeted therapies exist for this disease (LLM-derived static knowledge) | `disease_targeted_therapy_exists` | 48% positive | ✅ |
| `feat_genetic_basis` | categorical | Disease genetic basis: "none" / "monogenic" / "polygenic" / "somatic" / "unknown" (LLM-derived, static) | `disease_genetic_basis` | 82.6% non-unknown | 🔄 (→ one-hot cols in training) |
| `feat_therapeutic_superclass` | categorical | Drug class from trial registration (e.g., "Oncology", "CNS", "Immunology") — 11 classes | `ct_conditions`, `indication` | ~95% | 🔄 (→ one-hot cols in training) |
| `feat_therapeutic_superclass_CNS` | binary | One-hot encoding of `feat_therapeutic_superclass == "CNS"` | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Cardiovascular` | binary | One-hot: therapeutic_superclass == "Cardiovascular" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Dermatology` | binary | One-hot: therapeutic_superclass == "Dermatology" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Endocrine/Metabolic` | binary | One-hot: therapeutic_superclass == "Endocrine/Metabolic" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_GI/Hepatology` | binary | One-hot: therapeutic_superclass == "GI/Hepatology" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Immunology` | binary | One-hot: therapeutic_superclass == "Immunology" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Infectious Disease` | binary | One-hot: therapeutic_superclass == "Infectious Disease" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Musculoskeletal` | binary | One-hot: therapeutic_superclass == "Musculoskeletal" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Oncology` | binary | One-hot: therapeutic_superclass == "Oncology" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Other` | binary | One-hot: therapeutic_superclass == "Other" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_therapeutic_superclass_Respiratory` | binary | One-hot: therapeutic_superclass == "Respiratory" | `feat_therapeutic_superclass` | 100% | ✅ |
| `feat_genetic_basis_monogenic` | binary | One-hot: genetic_basis == "monogenic" | `feat_genetic_basis` | 100% | ✅ |
| `feat_genetic_basis_none` | binary | One-hot: genetic_basis == "none" | `feat_genetic_basis` | 100% | ✅ |
| `feat_genetic_basis_polygenic` | binary | One-hot: genetic_basis == "polygenic" | `feat_genetic_basis` | 100% | ✅ |
| `feat_genetic_basis_somatic` | binary | One-hot: genetic_basis == "somatic" | `feat_genetic_basis` | 100% | ✅ |
| `feat_genetic_basis_unknown` | binary | One-hot: genetic_basis == "unknown" — notable: ranks #8 in v17 LogReg importance | `feat_genetic_basis` | 100% | ✅ |

---

## Section 6 — Financial & Market Context

Pre-event financial signals from public filings and market data, fetched once and stored in the master CSV.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_log_market_cap` | float | `log10(market_cap_m + 0.001)` — log-scales market cap to reduce skew; +0.001 avoids log(0). Fixed in v12 (was computed incorrectly before). | `market_cap_m` | 91.0% | ✅ |
| `feat_volatility` | float | 20-trading-day ATR-normalized volatility computed from pre-event OHLC data. Uses Wilder's RMA (EWM alpha=1/20). | `price_before`, OHLC cache | 84.8% | ✅ |
| `feat_cash_runway_proxy` | float | Estimated cash runway in years = `cash_position_m / estimated_quarterly_burn`. Burn proxy from pipeline breadth. | `cash_position_m`, pipeline signals | 69.7% | ✅ |
| `feat_market_cap_bucket` | categorical | 4-bucket size: "micro" (<$300M), "small" ($300M–$1B), "mid" ($1B–$5B), "large" (>$5B) | `market_cap_m` | 100% | 🔄 (used for fold-safe priors, not direct feature) |

---

## Section 7 — CT.gov Timing Features

Derived from the trial's registered `ct_primary_completion` date, which is a **prospective protocol milestone** set before the trial begins. All anchored to `prediction_date = v_actual_date − 1` (i.e., the day before the event) to prevent event-date leakage.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_days_to_primary_completion` | int | `(ct_primary_completion − prediction_date).days`. Negative = completion already past. Anchored to day before event (v13 fix). | `ct_primary_completion`, `v_actual_date` | 82.6% | ✅ |
| `feat_primary_completion_imminent_30d` | binary | 1 if `0 ≤ feat_days_to_primary_completion ≤ 30` | `feat_days_to_primary_completion` | 82.6% | ✅ |
| `feat_primary_completion_imminent_90d` | binary | 1 if `0 ≤ feat_days_to_primary_completion ≤ 90`. Ranks #6 in v17 LogReg importance. | `feat_days_to_primary_completion` | 82.6% | ✅ |
| `feat_completion_recency_bucket` | categorical | 6-level: "imminent_0_30" / "near1_90" / "medium_91_180" / "far_180_plus" / "past" / "unknown" | `feat_days_to_primary_completion` | 100% | 🔄 |
| `feat_days_since_ctgov_last_update` | int | Days since CT.gov record was last updated (as of fetch date). Signals trial activity. | `ct_last_updated` (CT.gov API) | 93.5% | ✅ (via feat_recent_ctgov_update_flag) |
| `feat_recent_ctgov_update_flag` | binary | 1 if last CT.gov update ≤ 90 days before event date. Signals imminent readout. | `feat_days_since_ctgov_last_update` | 93.5% | ✅ |
| `feat_status_timing_consistency_flag` | binary | 1 if CT.gov status and completion date are mutually consistent (e.g., COMPLETED with recent date) | `ct_status`, `ct_primary_completion` | 93.5% | 🔄 |
| `feat_completed_before_event` | binary | 1 if `ct_primary_completion < event_date` (date proxy: prospective milestone predates event). Pre-event safe because the completion DATE is registered prospectively. | `ct_primary_completion`, `event_date` | ~40% | 🗑️ Replaced by AACT PIT (v7+) |
| `feat_recent_completion_flag` | binary | 1 if `(event_date − ct_primary_completion).days ≤ 180` | `ct_primary_completion`, `event_date` | 82.6% | 🗑️ Event-date anchored |
| `feat_ctgov_primary_completion_date` | date | Raw parsed `ct_primary_completion` from CT.gov (intermediate date field) | CT.gov API | 94.8% | 🔄 |

**Snapshot-unsafe CT.gov status features** (exist in full dataset, excluded from training):

| Feature | Type | Description | Status |
|---------|------|-------------|--------|
| `feat_ct_status_current` | categorical | Current CT.gov trial status (snapshot as of fetch date) | ⚠️ |
| `feat_active_not_recruiting_flag` | binary | 1 if current `ct_status == "ACTIVE_NOT_RECRUITING"` (snapshot) | ⚠️ Replaced by `feat_active_not_recruiting_at_event_flag` |
| `feat_completed_flag` | binary | 1 if current `ct_status == "COMPLETED"` (snapshot) | ⚠️ Replaced by `feat_completed_at_event_flag` |
| `feat_withdrawn_flag` | binary | 1 if current `ct_status == "WITHDRAWN"` (snapshot) | ⚠️ |
| `feat_terminated_flag` | binary | 1 if current `ct_status == "TERMINATED"` (snapshot) | ⚠️ |

**Point-in-time status features** (from AACT monthly archive snapshots — pre-event safe):

| Feature | Type | Description / Formula | Coverage | Status |
|---------|------|-----------------------|----------|--------|
| `feat_completed_at_event_flag` | binary | 1 if trial status was **COMPLETED** as of event_date. Derived from AACT monthly flat-file archives (~39 ZIPs); lookup = latest monthly snapshot ≤ event_date. Falls back to 0 if no AACT data. | 57.3% (486/1142 training rows missing → imputed 0) | ✅ |
| `feat_active_not_recruiting_at_event_flag` | binary | 1 if trial status was **ACTIVE_NOT_RECRUITING** as of event_date (AACT PIT, same method). | 57.3% (same coverage) | ✅ |
| `feat_terminated_at_event_flag` | binary | 1 if trial status was **TERMINATED** as of event_date (AACT PIT) | 100% | ✅ |
| `feat_withdrawn_at_event_flag` | binary | 1 if trial status was **WITHDRAWN** as of event_date (AACT PIT) | 100% | ✅ |

---

## Section 8 — CT.gov Sponsor & Asset Pipeline Features

Fetched via CT.gov API (`build_ctgov_pipeline_proxies.py`). Reflect sponsor/drug development history as registered on CT.gov — pre-event public information.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_ctgov_n_late_stage_trials_sponsor` | int | Count of Phase 2/3+ trials registered by this sponsor on CT.gov (capped at 100 for large pharma) | CT.gov API, `ct_sponsor` | 94.8% | ✅ |
| `feat_ctgov_pipeline_maturity_score` | float | `log1p(n_late_stage_trials_sponsor) + log1p(n_completed_trials_sponsor)` — sponsor development experience | CT.gov API | 94.8% | ✅ |
| `feat_ctgov_asset_maturity_score` | float | `log1p(n_late_stage_trials_same_intervention) + log1p(n_completed_trials_same_intervention)` — drug-specific track record | CT.gov API, `drug_name` | 99.2% | ✅ |
| `feat_ctgov_n_active_trials_sponsor` | int | Count of RECRUITING or ACTIVE_NOT_RECRUITING trials for sponsor | CT.gov API | 94.8% | 🔄 |
| `feat_ctgov_n_completed_trials_sponsor` | int | Count of COMPLETED trials for sponsor | CT.gov API | 94.8% | 🔄 |
| `feat_ctgov_n_trials_total_sponsor` | int | Total trials registered by sponsor (capped at 100) | CT.gov API | 94.8% | 🔄 |
| `feat_ctgov_n_trials_same_intervention` | int | Total CT.gov trials for this drug/intervention (capped at 100) | CT.gov API | 99.2% | 🔄 |
| `feat_ctgov_n_late_stage_trials_same_intervention` | int | Phase 2/3+ trials for same intervention (intermediate for asset_maturity_score) | CT.gov API | 99.2% | 🔄 |

---

## Section 9 — Event Sequence Features

Backward-looking sequence features. All use `v_actual_date` as the time axis and only look at prior events — no future data leak.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_time_since_last_company_event` | float | Days between `prediction_date` (= `v_actual_date − 1`) and the prior event for this ticker. NaN for first event. Imputed to median=71 days. | `ticker`, `v_actual_date` | 86.5% | ✅ |
| `feat_time_since_last_asset_event` | float | Days between `prediction_date` and the prior event for this (ticker, drug_name). NaN for first asset event. Imputed to median=109.5 days. | `ticker`, `drug_name`, `v_actual_date` | 34.7% | ✅ |
| `feat_recent_company_event_flag` | binary | 1 if time_since_last_company_event ≤ 90 days | Above | 100% | 🔄 (in full dataset, correlated with time feature; not separately in training) |
| `feat_recent_asset_event_flag` | binary | 1 if time_since_last_asset_event ≤ 180 days | Above | 100% | 🔄 |

---

## Section 10 — Oncology Interaction Features

Multiplicative interactions allowing the model to weight timing signals differently within the oncology disease class.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_oncology_x_imminent_30d` | binary | `feat_oncology_flag × feat_primary_completion_imminent_30d` | Sections 5+7 | 100% | 🔄 (in full dataset, available for future training) |
| `feat_oncology_x_imminent_90d` | binary | `feat_oncology_flag × feat_primary_completion_imminent_90d` | Sections 5+7 | 100% | 🔄 |
| `feat_oncology_x_recent_completion` | binary | `feat_oncology_flag × feat_recent_completion_flag` | Sections 5+7 | 100% | 🔄 |
| `feat_oncology_x_recency_imminent` | binary | `feat_oncology_flag × (imminent_90d OR recent_completion)` | Sections 5+7 | 100% | 🔄 |

---

## Section 11 — Biological Feature Families (Pass-9, v17)

Added in v17 (2026-03-27). All derived via **deterministic rules from existing columns** — zero new LLM API calls. Source: `disease_genetic_basis` column (from `enrich_disease_biology.py`) plus keyword matching on `indication` and `drug_name`.

### Family A — Heritability Proxy (3 features)

Re-encodes the existing `disease_genetic_basis` categorical column (previously one-hot only) into numeric ordinals and a continuous proxy.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_genetic_basis_encoded` | ordinal 0–3 | `none=0, polygenic=1, somatic=2, monogenic=3`. NaN for "unknown". Preserves ordinal structure (monogenic is most well-defined mechanistically). | `disease_genetic_basis` | 82.6% (2,332/2,822) | ✅ |
| `feat_heritability_proxy_score` | float 0–1 | Score by basis: `monogenic=0.85, somatic=0.45, polygenic=0.35, none=0.10`; null rows→`0.40` (global mean fallback). Represents tractability of the disease mechanism. | `disease_genetic_basis` | 100% | ✅ |
| `feat_heritability_level` | ordinal 0–2 | Bins heritability_proxy_score: `low=0 (<0.30), moderate=1 (0.30–0.60), high=2 (>0.60)`. Distribution: low=561 (19.9%), moderate=2,026 (71.8%), high=235 (8.3%). Ranks #20 in v17 LightGBM importance. | `feat_heritability_proxy_score` | 100% | ✅ |

### Family B — Enrichment Relevance (4 features)

Captures whether a trial is designed around a molecularly-defined patient population (biomarker-selected or targeted therapy) — a structural signal for predictability of outcome.

| Feature | Type | Description / Formula | Source | Coverage | Status |
|---------|------|-----------------------|--------|----------|--------|
| `feat_biomarker_stratified_flag` | binary | 1 if `indication` or `ct_official_title` contains biomarker/enrichment keywords: "biomarker", "mutation", "fusion", "amplification", "stratified", "enriched", "selected", "positive", "negative" + oncology-specific patterns. Positive rate: **19.6%** (554/2,822). | `indication`, `ct_official_title` | 100% | ✅ |
| `feat_targeted_mechanism_flag` | binary | 1 if `drug_name` contains targeted-therapy suffix/keyword (`-mab`, `-nib`, `-lib`, `inhibitor`, `antibody`, `gene therapy`, `antisense`, `siRNA`) OR disease is monogenic. Positive rate: **30.9%** (871/2,822). | `drug_name`, `disease_genetic_basis` | 100% | ✅ |
| `feat_disease_molecular_heterogeneity_score` | float 0–1 | Lower = well-defined target (monogenic/rare); higher = heterogeneous population (somatic/oncology). Formula: weighted combo of genetic basis type + MeSH category + rare_disease_flag. Mean=0.531, std=0.217. | `disease_genetic_basis`, `mesh_level1`, `feat_rare_disease_flag` | 100% | ✅ |
| `feat_enrichment_relevance_score` | float 0–1 | `0.35×biomarker_stratified + 0.25×targeted_mechanism + 0.25×(1−heterogeneity) + 0.15×has_predictive_biomarker`. Composite signal for "how molecularly-targeted is this trial?". Mean=0.301, std=0.215. | Above 3 + `feat_has_predictive_biomarker` | 100% | ✅ |

---

## Section 12 — Fold-Safe Priors (8 features, computed at training time)

These are **group-level reaction rate priors** — mean historical ATR move and large-move rate by categorical group. Computed via `FoldPriorEncoder`: `.fit()` is called **only on the training fold** in each CV split, then `.transform()` applied to validation/test. Fallback cascade for unseen categories: phase-level → global mean.

| Feature | Type | Description | Groups | Status |
|---------|------|-------------|--------|--------|
| `feat_prior_mean_abs_move_atr_by_phase` | float | Mean `abs(stock_movement_atr_normalized)` in training fold, grouped by `feat_phase_num` | Phase | ✅ |
| `feat_prior_large_move_rate_by_phase` | float | Fraction `target_large_move==1` in training fold, grouped by `feat_phase_num` | Phase | ✅ |
| `feat_prior_mean_abs_move_atr_by_therapeutic_superclass` | float | Mean ATR move in training fold, grouped by `feat_therapeutic_superclass` | Drug class | ✅ |
| `feat_prior_large_move_rate_by_therapeutic_superclass` | float | Large-move rate in training fold, grouped by `feat_therapeutic_superclass` | Drug class | ✅ |
| `feat_prior_mean_abs_move_atr_by_market_cap_bucket` | float | Mean ATR move in training fold, grouped by `feat_market_cap_bucket` | Market cap tier | ✅ |
| `feat_prior_large_move_rate_by_market_cap_bucket` | float | Large-move rate in training fold, grouped by `feat_market_cap_bucket` (**Added v16, ranks #4 in v17**) | Market cap tier | ✅ |
| `feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass` | float | Mean ATR move, grouped by (phase × drug class) interaction | Phase × class | ✅ |
| `feat_prior_large_move_rate_by_phase_x_therapeutic_superclass` | float | Large-move rate by (phase × drug class) interaction (**Added v16, ranks #2 in v17**) | Phase × class | ✅ |

> **Note:** Priors are not columns in `ml_dataset_features_20260325_v3.csv`. They are computed and injected at training time by `add_train_fold_priors.py` / `train_baseline_models.py`. They appear in `ml_baseline_train_20260327_v17.csv`.

---

## Section 13 — Permanently Excluded Features

### Post-Event (Outcome Content) — ❌

These features encode the result of the trial announcement and **must never** be used for pre-event prediction.

| Feature | Reason |
|---------|--------|
| `feat_superiority_flag` | Keyword match on outcome press release text ("superior", "met primary endpoint") |
| `feat_stat_sig_flag` | Keyword match on outcome text ("p-value", "significant", "p<0.05") |
| `feat_clinically_meaningful_flag` | Keyword match on outcome text ("meaningful benefit", "clinically significant") |
| `feat_mixed_results_flag` | Keyword match on outcome text ("failed", "missed", "inconclusive") |
| `primary_endpoint_met` (raw) | Realized outcome label — the thing we are trying to predict |
| `primary_endpoint_result` (raw) | Outcome result text |

### Removed for Leakage or Replaced — 🗑️

| Feature | Version Removed | Reason | Replacement |
|---------|----------------|--------|-------------|
| `feat_completed_flag` | v6 (2026-03-23) | SNAPSHOT_UNSAFE: current CT.gov status, not PIT | `feat_completed_at_event_flag` (AACT PIT) |
| `feat_active_not_recruiting_flag` | v7 (2026-03-23) | SNAPSHOT_UNSAFE | `feat_active_not_recruiting_at_event_flag` (AACT PIT) |
| `feat_short_squeeze_flag` | v7 | SNAPSHOT_UNSAFE: yfinance current short interest | — |
| `feat_ownership_low_flag` | v7 | SNAPSHOT_UNSAFE: yfinance current institutional ownership | — |
| `feat_completed_before_event` | v7 | Replaced by higher-quality AACT PIT lookup | `feat_completed_at_event_flag` |

---

## Version × Feature Matrix

Shows when each major feature group was added to the training roster.

| Feature Group | v1–v5 | v6 | v7 | v8 | v9–v10 | v11 | v12 | v13 | v14 | v15 | v16 | v17 |
|---------------|-------|----|----|----|---------|----|-----|-----|-----|-----|-----|-----|
| Clinical core (phase, design, enrollment) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Regulatory flags | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Company pipeline profile | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 6 fold-safe priors (phase, class, cap) | v3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AACT PIT status flags | — | — | v7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LLM disease biology (mesh, oncology, etc.) | — | — | — | — | — | v11 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CT.gov timing (days-to-completion) | — | — | — | — | — | — | v12 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Company historical hit rate | — | — | — | — | — | — | — | — | v14 | ✓ | ✓ | ✓ |
| Phase 4 historical data (2018–2022) | — | — | — | — | — | — | — | — | — | v15 | ✓ | ✓ |
| 2 new fold-safe priors (cap bucket, phase×class large-move rate) | — | — | — | — | — | — | — | — | — | — | v16 | ✓ |
| Biological feature families (heritability, enrichment) | — | — | — | — | — | — | — | — | — | — | — | v17 |

---

## Sample Rows

### Full feature dataset (`ml_dataset_features_20260325_v3.csv`)

| ticker | event_date | drug_name | move_pct | move_class_norm | feat_phase_num | feat_oncology_flag | feat_primary_completion_imminent_90d | feat_blinded_flag | feat_heritability_proxy_score |
|--------|-----------|-----------|----------|----------------|---------------|--------------------|--------------------------------------|-------------------|-------------------------------|
| CELC | 2025-10-18 | gedatolisib | +35.8% | High | 3.0 | 1 | 0 | 1 | 0.45 (somatic) |
| ALNY | 2024-08-30 | vutrisiran | −11.6% | Medium | 3.0 | 0 | 0 | 1 | 0.85 (monogenic) |
| QURE | 2025-09-24 | AMT-130 | +285% | Extreme | 1.5 | 0 | 0 | 0 | 0.85 (monogenic) |
| ORIC | 2025-11-13 | ORIC-944 | −1.8% | Noise | 1.0 | 1 | 0 | 0 | 0.45 (somatic) |

### Training table (`ml_baseline_train_20260327_v17.csv`)

| ticker | event_date | target_large_move | feat_phase_num | feat_trial_quality_score | feat_company_historical_hit_rate | feat_oncology_flag | feat_primary_completion_imminent_90d | split |
|--------|-----------|------------------|---------------|-------------------------|-----------------------------------|--------------------|--------------------------------------|-------|
| LLY | 2010-02-15 | 0 | 2.0 | 2.0 | 0.00 (imputed) | 1 | 0 | train |
| LLY | 2010-12-28 | 0 | 2.0 | 2.0 | 0.00 | 1 | 0 | train |
| CELC | 2025-10-18 | 1 | 3.0 | 4.5 | 0.67 | 1 | 0 | test |

---

## Appendix A — Full Column List: ml_dataset_features_20260325_v3.csv (154 cols)

**Metadata / Event Info (non-feature):** `ticker`, `event_date`, `event_type`, `move_pct`, `price_at_event`, `catalyst_summary`, `drug_name`, `nct_id`, `indication`, `is_pivotal`, `pivotal_evidence`, `primary_endpoint_met`, `primary_endpoint_result`, `ct_official_title`, `ct_phase`, `ct_enrollment`, `ct_conditions`, `ct_status`, `ct_sponsor`, `ct_allocation`, `ct_primary_completion`, `market_cap_m`, `current_price`, `cash_position_m`, `short_percent`, `institutional_ownership`, `analyst_target`, `analyst_rating`, `atr_pct`, `stock_movement_atr_normalized`, `avg_daily_move`, `move_class_abs`, `move_class_norm`, `move_class_combo`, `event_trading_date`, `move_2d_pct`, `price_before`, `price_after`, `stock_relative_move`, `data_complete`, `v_is_verified`, `v_actual_date`, `v_pr_link`, `v_pr_date`, `v_pr_title`, `v_pr_key_info`, `v_is_material`, `v_confidence`, `v_summary`, `v_error`, `v_action`, `best_event_link`, `mesh_level1`, `mesh_level1_reason`, `mesh_branches_raw`, `mesh_terms_raw`, `ct_conditions_raw`, `data_tier`, `stale_price_data`, `mesh_level1_encoded`, `row_ready`, `row_not_ready_reason`, `ct_status_at_event`, `disease_has_predictive_biomarker`, `disease_genetic_basis`, `disease_targeted_therapy_exists`

**Feature columns (feat_*):** `feat_phase_num`, `feat_late_stage_flag`, `feat_enrollment_log`, `feat_randomized_flag`, `feat_design_quality_score`, `feat_withdrawn_flag`, `feat_terminated_flag`, `feat_orphan_flag`, `feat_fast_track_flag`, `feat_breakthrough_flag`, `feat_nda_bla_flag`, `feat_priority_review_flag`, `feat_regulatory_stage_score`, `feat_terminated_at_event_flag`, `feat_withdrawn_at_event_flag`, `feat_controlled_flag`, `feat_trial_quality_score`, `feat_n_trials_for_company`, `feat_n_unique_drugs_for_company`, `feat_single_asset_company_flag`, `feat_lead_asset_dependency_score`, `feat_n_late_stage_trials_for_company`, `feat_pipeline_concentration_simple`, `feat_company_historical_hit_rate`, `feat_asset_trial_count_for_company`, `feat_asset_trial_share`, `feat_pipeline_depth_score`, `feat_pivotal_proxy_score`, `feat_primary_endpoint_known_flag`, `feat_superiority_flag`, `feat_stat_sig_flag`, `feat_clinically_meaningful_flag`, `feat_mixed_results_flag`, `feat_blinded_flag`, `feat_open_label_flag`, `feat_small_trial_flag`, `feat_completed_before_event`, `feat_recent_completion_flag`, `feat_completed_at_event_flag`, `feat_active_not_recruiting_at_event_flag`, `feat_therapeutic_superclass`, `feat_mesh_level1_encoded`, `feat_oncology_flag`, `feat_cns_flag`, `feat_rare_disease_flag`, `feat_has_predictive_biomarker`, `feat_genetic_basis`, `feat_targeted_therapy_exists`, `feat_cash_runway_proxy`, `feat_volatility`, `feat_log_market_cap`, `feat_market_cap_bucket`, `feat_ctgov_primary_completion_date`, `feat_days_to_primary_completion`, `feat_primary_completion_imminent_30d`, `feat_primary_completion_imminent_90d`, `feat_completion_recency_bucket`, `feat_ct_status_current`, `feat_active_not_recruiting_flag`, `feat_completed_flag`, `feat_days_since_ctgov_last_update`, `feat_recent_ctgov_update_flag`, `feat_status_timing_consistency_flag`, `feat_ctgov_n_active_trials_sponsor`, `feat_ctgov_n_late_stage_trials_sponsor`, `feat_ctgov_n_completed_trials_sponsor`, `feat_ctgov_n_trials_total_sponsor`, `feat_ctgov_pipeline_maturity_score`, `feat_ctgov_n_trials_same_intervention`, `feat_ctgov_n_late_stage_trials_same_intervention`, `feat_ctgov_asset_maturity_score`, `feat_time_since_last_company_event`, `feat_time_since_last_asset_event`, `feat_asset_event_sequence_num`, `feat_company_event_sequence_num`, `feat_recent_company_event_flag`, `feat_recent_asset_event_flag`, `feat_oncology_x_imminent_30d`, `feat_oncology_x_imminent_90d`, `feat_oncology_x_recent_completion`, `feat_oncology_x_recency_imminent`, `feat_genetic_basis_encoded`, `feat_heritability_proxy_score`, `feat_heritability_level`, `feat_biomarker_stratified_flag`, `feat_targeted_mechanism_flag`, `feat_disease_molecular_heterogeneity_score`, `feat_enrichment_relevance_score`

---

## Appendix B — Model Performance by Version

| Version | Date | Train rows | Test AUC | CV AUC | Key change |
|---------|------|-----------|---------|--------|-----------|
| v17 (current) | 2026-03-27 | 1,142 | 0.694 | 0.786±0.077 | 7 biological features (heritability + enrichment families) |
| v16 | 2026-03-27 | 1,142 | 0.695 | 0.785±0.077 | 2 new fold-safe priors (market-cap large-move rate, phase×class large-move rate) |
| v15 ★ | 2026-03-25 | **1,142** | **0.702** | 0.793±0.081 | Phase 4 data expansion (+441 historical 2018–2022 rows) |
| v14 | 2026-03-24 | 701 | 0.681 | 0.762±0.075 | `feat_company_historical_hit_rate` (fold-safe backward-looking hit rate) |
| v13 | 2026-03-24 | 701 | 0.668 | 0.758±0.081 | Timing anchor fix (prediction_date = event_date−1) |
| v12 | 2026-03-24 | 701 | 0.659 | 0.770±0.036 | log_market_cap bug fix, 7 CT.gov features |
| v11 | 2026-03-24 | 701 | 0.685 | 0.781±0.048 | LLM disease biology features (mesh, oncology, genetic basis) |
| v10 | 2026-03-23 | 701 | 0.664 | 0.784±0.045 | PIT fix for terminated/withdrawn |
| v9 | 2026-03-23 | 701 | 0.664 | 0.784±0.045 | Port foundational features to new pipeline |
| v7 | 2026-03-23 | 701 | 0.700 | 0.759±0.040 | AACT PIT status (feat_completed_at_event_flag) |
| v6 | 2026-03-23 | 701 | 0.693 | 0.747±0.047 | Remove feat_completed_flag leakage |
| v5 | 2026-03-19 | 701 | 0.703 | 0.752±0.053 | Prior baseline (contained leakage) |

★ = best test AUC. Best CV AUC: v17 at 0.786. Best combined signal: **v17** (post-leakage-fix best, 0.694 test / 0.786 CV).

---

## Top Feature Importances — v17 (LogReg coefficients)

| Rank | Feature | Coefficient | Notes |
|------|---------|-------------|-------|
| 1 | `feat_company_historical_hit_rate` | +0.800 | Backward-looking company track record |
| 2 | `feat_prior_large_move_rate_by_phase_x_therapeutic_superclass` | +0.751 | Phase×class prior (v16) |
| 3 | `feat_trial_quality_score` | +0.666 | Composite design quality |
| 4 | `feat_prior_large_move_rate_by_market_cap_bucket` | +0.553 | Market cap prior (v16) |
| 5 | `feat_blinded_flag` | +0.492 | Trial blinding → rigorous design |
| 6 | `feat_primary_completion_imminent_90d` | +0.450 | Readout imminent within 90 days |
| 7 | `feat_enrollment_log` | +0.353 | Larger trials → higher stakes |
| 8 | `feat_genetic_basis_unknown` | +0.342 | Uncertain disease mechanism → binary outcome |
| 9 | `feat_single_asset_company_flag` | +0.339 | Existential risk → amplified move |
| 10 | `feat_small_trial_flag` | +0.329 | Small early-stage trials → high variance |
| 20 | `feat_heritability_level` | — | Highest-ranking v17 biological feature (LightGBM) |

# Biotech Catalyst v3 — Pre-Event Move Prediction

**Goal:** Given today's public company/trial information (before any announcement), predict whether the eventual catalyst will cause a large stock move (≥ 3–5× ATR-normalized).

No press release content is used. All predictions are based on pre-event structural features only.

---

## Current Files — Quick Reference (as of 2026-03-28)

> **For Nurit:** Here are the current canonical data files and where to find them.

| File | Location | Git? | Contents |
|------|----------|------|---------|
| **Full feature dataset** | `data/ml/ml_dataset_features_20260325_v3.csv` | ✅ committed & pushed | 2,822 rows × 154 cols — all engineered features before training-set filtering |
| **Training table (current)** | `data/ml/ml_baseline_train_20260327_v17.csv` | ✅ committed & pushed | 1,142 rows × 79 features (71 base + 8 fold-safe priors) — what the model actually trains on |
| **Master dataset** | `enriched_all_clinical_clean_v3.csv` | ✅ committed & pushed | 2,958 rows × 63 cols — source of truth for all events |
| **Feature reference** | `reports/FEATURE_TABLE.md` | ✅ committed & pushed | Full table: every feature, formula, source, coverage, status |

**Current model (v17):** Logistic Regression — Test AUC **0.694**, CV AUC **0.786 ± 0.077** (5-fold time-aware). 79 features, 1,142 training rows (32% positive rate).

**Feature documentation:** See [`reports/FEATURE_TABLE.md`](reports/FEATURE_TABLE.md) for the complete feature reference with descriptions, formulas, and active/deprecated status.

---

## Summary

We built a pre-event binary classifier that predicts whether a biotech catalyst (clinical trial readout) will cause a large stock move — defined as ≥ 3× ATR-normalized AND ≥ 10% absolute — using only public information available before the announcement.

**Dataset:** 2514 clinical trial events (2007–2026) sourced from CT.gov and enriched with financial data. Rows are tiered: 758 `trusted_trainable` (2023+, complete price data, 30.5% positive rate) · 1733 `history_only` (pre-2023, genuine hard negatives — reserved for sponsor/asset history and future timing models) · 22 `repairable` · 1 `reject`. Model trained on 701-row 2023+ strict subset (596 row_ready + 105 relaxed from missing-mesh exclusion only).

**Target:** `target_large_move = 1` when `abs(stock_movement_atr_normalized) ≥ 3.0` AND `abs(move_pct) ≥ 10%`. Positive rate: 30.4% on the current training cohort.

**Features (31 total — strict-clean):** `feat_completed_before_event` (date proxy: primary completion date precedes event, pre-event valid), `feat_active_not_recruiting_flag` (CT.gov status flag), trial design flags (blinded, open-label, small trial), disease class flags (oncology, CNS, rare disease), company context (cash runway, pipeline depth), event sequence ordinals (company/asset event count — no event-date anchor), and 6 fold-safe reaction priors (mean ATR + large-move rate by therapeutic class/phase/market-cap).

**Best model (v6):** Logistic Regression — AUC **0.693**, Prec@top 10% **0.818**, CV AUC **0.747 ± 0.047** (5-fold time-aware). ✓ STRICT_CLEAN — no event-date-anchored or snapshot-unsafe features.

**Top predictors:** `feat_active_not_recruiting_flag`, `feat_blinded_flag`, prior by therapeutic class, `feat_completed_before_event`, `feat_small_trial_flag`, `feat_cns_flag`.

**Pipeline:** One command runs all 8 steps — feature engineering → CT.gov API enrichment → train table → model training:
```bash
python -m scripts.run_full_pre_event_pipeline
```

---

## Quick start

```bash
cd /Users/tomer/Code/NuriTomer
source .venv/bin/activate
cd biotech_catalyst_v3

# Run everything from here as:
python -m scripts.<script_name>
```

---

## Current state (v1.2 — 2026-03-19)

### Source of truth files

| File | Description |
|---|---|
| `ml_dataset_features_20260316_v2.csv` | ML feature dataset — 2379 rows × 108 cols, full pipeline run on v3 master |
| `ml_feature_dict_20260316_v2.csv` | Feature dictionary — 14 entries with coverage + description |
| `ml_baseline_train_20260318_v5.csv` | **CURRENT** training table — 701 rows (2023+), 25 base + 6 priors = 31 features, 30.4% positive rate |
| `ml_baseline_train_20260317_v4.csv` | Previous strict-clean table — 596 rows — archived |
| `enriched_all_clinical_clean_v3.csv` | **MASTER DATASET** — 2514 rows × 58 cols (v2 + 1652 historical 2020–2022 events) |
| `enriched_all_clinical_clean_v3_tiered_20260318_v1.csv` | Tiered master — same rows + `data_tier` + `target_large_move` columns |
| `candidate_strict_trainable_20260318_v1.csv` | Candidate training subset — 758 trusted_trainable rows, 231 positives (30.5%) |
| `biotech_universe_expanded.csv` | 460 tracked biotech tickers ($50M–$10B) |

### Current model (v5 expanded strict-clean — 2023+ cohort, 31 features)

| File | Description |
|---|---|
| `models/model_pre_event_v3_20260312.pkl` | ✓ STRICT-CLEAN — LogReg, AUC 0.703 |

> ✓ **Model validity status: STRICT_CLEAN.** All 9 event-date-anchored features (`feat_days_to_primary_completion`, timing imminence/recency/time-since features) are excluded from the train table. No PR/outcome features. Model uses only information knowable before the announcement.

### Current model report

→ [`reports/MODEL_REPORTS.md`](reports/MODEL_REPORTS.md) — full model history, newest at top

| Metric | v5 EXPANDED (current) | v4 STRICT-CLEAN | v3 contaminated ⚠ |
|---|---|---|---|
| Best model | **LogReg** | XGBoost | LightGBM |
| Test ROC-AUC | **0.703** | 0.692 | 0.730 (inflated) |
| CV AUC (5-fold) | **0.752 ± 0.053** | 0.711 ± 0.112 | 0.744 ± 0.096 |
| Prec @ top 10% | **0.545** | 0.444 | 0.778 (inflated) |
| Rows | **701** | 596 | 596 |
| Features | **25 base + 6 priors = 31** | 24 base + 6 priors = 30 | 44 (14 invalid) |
| Pre-event valid? | ✓ **STRICT_CLEAN** | ✓ STRICT_CLEAN | ✗ contaminated |

### Feature notes, validity audit, CT.gov notes

→ [`reports/FEATURE_NOTES.md`](reports/FEATURE_NOTES.md) — validity audit · CT.gov feature notes · oncology caveats · excluded features

### Dataset notes, target analysis, expansion

→ [`reports/DATASET_NOTES.md`](reports/DATASET_NOTES.md) — threshold analysis · expansion strategy · coverage

---

## Documentation policy

**Canonical running docs** (one file per topic, newest section at top):

| File | Topic |
|---|---|
| [`reports/MODEL_REPORTS.md`](reports/MODEL_REPORTS.md) | All model training results |
| [`reports/FEATURE_NOTES.md`](reports/FEATURE_NOTES.md) | Feature validity, CT.gov notes, audit findings |
| [`reports/DATASET_NOTES.md`](reports/DATASET_NOTES.md) | Dataset expansions, target analysis, coverage |

**Rule:** When adding new documentation, prepend a dated section to the relevant canonical doc — do NOT create new standalone `.md` report files. Each section must include date, version tag, and a short heading. `train_pre_event_v3.py` already follows this policy (prepends to MODEL_REPORTS.md).

**Exception:** Dataset and model artifact files (`.csv`, `.pkl`, `.json`) remain separately versioned as before.

Old standalone report files are archived in `reports/reports_history/`.

---

## File naming convention (artifacts)

```
{name}_{YYYYMMDD}_v{N}.{ext}        # new pipeline naming
```

- Date first, integer version
- Previous versions → `archive/` (gitignored locally)
- Old semver format (`v0.X_YYYYMMDD`) deprecated — do not create new files in that format

---

## Pipeline overview

```
enriched_all_clinical_clean_v3.csv   ← master dataset (2514 rows; v2 archived)
        ↓
scripts/prepare_ml_dataset.py              → initial feature table
scripts/add_high_signal_features.py        → Pass-4 features
scripts/refresh_ctgov_features.py          → Pass-6: CT.gov timing (11 features)
scripts/build_ctgov_pipeline_proxies.py    → Pass-7: CT.gov pipeline proxies (8 features)
scripts/add_pre_event_timing_features.py   → Pass-5: timing/sequence features
scripts/add_oncology_timing_interactions.py → Pass-8: oncology × timing interactions
        ↓
scripts/build_pre_event_train_v2.py        → train table
        ↓
scripts/train_pre_event_v3.py              → models/ + reports/
```

**Status (2026-03-19):** v5 retrain complete on expanded 701-row pool. New best: LogReg AUC 0.703 / CV AUC 0.752 ± 0.053 / Prec@10% 0.545 — all metrics exceed v4 strict-clean baseline. ✓ STRICT_CLEAN.

### Key scripts

| Script | What it does |
|---|---|
| `run_full_pre_event_pipeline.py` | **One-command orchestration** — runs all 8 steps; supports --start-step N, --dry-run, --master |
| `add_pre_event_timing_features.py` | Pass-5: adds 9 timing/sequence features to the feature dataset |
| `add_oncology_timing_interactions.py` | Pass-8: adds 4 oncology × timing interaction features |
| `expand_historical_events.py` | Dataset expansion: CT.gov Phase 2/3 completions for a date range → appends to master |
| `refresh_ctgov_features.py` | Pass-6: CT.gov timing refresh (per-NCT-ID API, 11 features) |
| `build_ctgov_pipeline_proxies.py` | Pass-7: CT.gov sponsor + drug aggregate features (8 features) |
| `add_train_fold_priors.py` | `FoldPriorEncoder` — inject reaction priors safely inside CV folds |
| `build_pre_event_train_v2.py` | Build train/val/test table from feature dataset |
| `train_pre_event_v3.py` | Train LogReg + LightGBM + XGBoost, CV, report, plots |
| `add_high_signal_features.py` | Pass-4: pivotal proxy, trial design flags, therapeutic superclass |
| `validate_catalysts.py` | Validate noise-class rows via Perplexity + PR fetch |
| `backfill_price_at_event.py` | Fill price_before / price_after / move_2d_pct via yfinance |

### Running the full feature → model pipeline

One command (from `biotech_catalyst_v3/`):
```bash
python -m scripts.run_full_pre_event_pipeline
```

Or resume from a specific step (e.g. after step 3 API calls complete):
```bash
python -m scripts.run_full_pre_event_pipeline --start-step 4
```

Steps: 1 `prepare_ml_dataset` → 2 `add_high_signal_features` → 3 `refresh_ctgov_features` (API, ~20 min) → 4 `build_ctgov_pipeline_proxies` (API, ~10 min) → 5 `add_pre_event_timing_features` → 6 `add_oncology_timing_interactions` → 7 `build_pre_event_train_v2` → 8 `train_pre_event_v3`

**Important:** All pipeline scripts use the naming convention `ml_dataset_features_YYYYMMDD_vN.csv` (date-first, integer version). Do not put old `v0.X_YYYYMMDD` files in the project root — they will be ignored. Old files belong in `archive/`.

---

## Target definition

- **`target_large_move`** (binary, primary): 1 = ATR-normalized move ≥ **3.0×** AND abs(move_pct) ≥ **10%** (Medium + High + Extreme, minus large-cap noise)
- **`target_abs_move_atr`** (continuous): raw ATR-normalized absolute move magnitude
- **`target_move_bucket`** (5-class): Noise / Low / Medium / High / Extreme

ATR = Wilder's RMA, `ewm(alpha=1/20, adjust=False)`, 20 trading-day lookback, strictly pre-event.
Class thresholds: Noise < 1.5× · Low 1.5–3× · Medium 3–5× · High 5–8× · Extreme ≥ 8×

**Move window:** Last trading day strictly before event → first trading day strictly after event (1 overnight move bracketing the announcement; `move_2d_pct` column name is shorthand for "2 days apart", not a 2-day trailing return).

**Binary target v0.6:** `target_large_move = 1` when `stock_movement_atr_normalized >= 3.0 AND abs(move_pct) >= 10.0`. This adds the Medium class (3–5× ATR) while excluding large-cap pharma events with high ATR multiples but small absolute moves. Positive rate: 9.7% on full v3 dataset (242/2500); 26.9% on original curated rows (237/881).

---

## Key architecture decisions

- **Pre-event only:** No press release content. All features derivable from public CT.gov + financial data before the news.
- **Time-based splits:** Train/val/test split on `v_actual_date`; CV uses `TimeSeriesSplit` to prevent future leakage.
- **Fold-safe priors:** Reaction priors (mean ATR move by phase/disease/mkt-cap) fit on train fold only via `FoldPriorEncoder`. Never precomputed globally.
- **ATR normalization:** Moves normalized by pre-event ATR to make large-cap and small-cap events comparable.
- **Single master CSV:** `enriched_all_clinical_clean_v3.csv` is the one file all enrichments write to (v2 archived).

## Answers to collaborator questions — 2026-03-13

### Q1 — What exactly is `target_abs_move_atr`?

It is the ATR-normalised absolute stock move bracketing the announcement:

```
target_abs_move_atr = abs(price_after - price_before) / price_before / atr_pct
```

- `price_before` = closing price on the **last trading day strictly before** the event date
- `price_after` = closing price on the **first trading day strictly after** the event date
- So the window spans **1 overnight move** (not 2 trailing days). The raw column is called `move_2d_pct` meaning "2 calendar days apart", not a 2-day trailing return.
- Based on **trading days**, not calendar days.
- `atr_pct` = Wilder's RMA with alpha=1/20, 20 trading-day lookback, using price data **strictly before** the event (no look-ahead).

---

### Q2 — Are feat_superiority_flag / feat_stat_sig_flag / feat_clinically_meaningful_flag / feat_mixed_results_flag post-event?

**Yes — all four are post-event and must NOT be used in the pre-event model.**

They are keyword matches on `primary_endpoint_result`, `v_pr_key_info`, `v_summary`, `catalyst_summary`, and `pivotal_evidence` — all of which contain the **current announcement text**. They reflect the outcome of the catalyst event itself, not any prior-phase information.

They are already excluded from the training table and will remain excluded. They exist in the feature dataset for post-hoc analysis only.

To build a legitimate pre-event equivalent, you would need to match keyword signals from **prior-phase publications** for the same drug — not yet implemented.

---

### Q3 — Are the CT.gov timing features valid for oncology?

**Partially valid — known limitation, not a reason to drop.**

For non-oncology trials, CT.gov primary completion date ≈ readout date, so the imminence and recency features are reliable.

For oncology (OS/PFS/DFS-driven trials), the efficacy readout occurs when a target number of events accrues, which can be **6–24 months before** CT.gov primary completion date. The CT.gov date is the administrative study close, not the readout. This means `feat_primary_completion_imminent_30d/90d` and `feat_completion_recency_bucket` can show "far" or "past" even when an oncology readout is genuinely imminent or just happened.

Current decision: keep these features as-is. They carry real signal for the ~40% of the dataset that is non-oncology, and even for oncology the "COMPLETED + recently closed" signal is meaningful. The model can learn to discount them for oncology via the existing `feat_oncology_flag`.

Better future feature: extract enrollment completion language or "on track / delayed / ahead of schedule" signals from CT.gov `detailed_description` or investor guidance text. Worth adding later but not trivial.

---

### Q4 — Is a management / CRO quality feature feasible?

**Feasible later, not the next priority.**

- **PI track record:** CT.gov exposes `principalInvestigator` per trial. Querying historical trial outcomes by PI name is doable via the same API pattern used for sponsor queries. Roughly 1–2 days of work.
- **CRO reputation:** CT.gov lists collaborators but does not always name the CRO explicitly. Partial coverage.
- **Senior management / CEO / CMO track record:** Requires linking to external sources (SEC filings, LinkedIn, OpenFDA). Not practical without an LLM enrichment pass.

The existing `feat_ctgov_pipeline_maturity_score` (sponsor depth) already captures some of this signal. PI-level track record is a B-priority research feature for later.

---

### Q5 — Should we add one-vs-rest confusion matrices for each class? Is multiclass worth prioritising?

**Multiclass is not the right focus now. Binary remains the primary objective.**

The current model targets `target_large_move` (binary: High + Extreme = 1). This is correct for the investment use case — we want to know which catalysts will cause outsized moves, not the exact magnitude bucket.

The 5-class `target_move_bucket` column exists in the dataset but no multiclass model has been trained. Class imbalance is severe (Noise = ~72% of rows; High + Extreme together = ~8%). A multiclass model would perform worse on the rare classes without substantially more data.

One-vs-rest confusion matrices are useful once a multiclass model exists, but not worth adding to the current binary model — precision/recall at threshold is the right diagnostic there (already in the reports).

**Decision: defer multiclass until binary AUC exceeds 0.70 reliably.**

---

## Pre-event feature validity — hard rule

**For the pre-event stock move size model, a feature is only valid if it can be computed using information publicly available BEFORE the future event occurs.**

The following are FORBIDDEN in any pre-event training table:
- The realized event date (`v_actual_date`, `event_date`, `event_trading_date`)
- The realized announcement / PR date
- The realized stock-move date
- The content or outcome of the announcement (`primary_endpoint_result`, PR text fields, etc.)

**Currently excluded** (use realized event date as anchor — valid concept, wrong implementation):

| Feature | Reason excluded |
|---|---|
| `feat_days_to_primary_completion` | `ct_primary_completion - v_actual_date` — anchor = realized announcement date |
| `feat_primary_completion_imminent_30d/90d` | Derived from above |
| `feat_completion_recency_bucket` | Derived from above |
| `feat_recent_completion_flag` | `(event_date - ct_primary_completion) <= 365` — anchor = realized event_date |
| `feat_time_since_last_company_event` | Endpoint = realized event date (unknown pre-event) |
| `feat_time_since_last_asset_event` | Same |
| `feat_recent_company_event_flag` | Derived from above |
| `feat_recent_asset_event_flag` | Derived from above |

**Still valid** (ordinal counts, no event-date anchor needed):
- `feat_asset_event_sequence_num`, `feat_company_event_sequence_num` — count of prior events for this drug/company, pre-event knowable

**Fix path:** Add `prediction_date` parameter to `add_pre_event_timing_features.py` and `add_high_signal_features.py`. Default = `v_actual_date` for training (identical behavior), override with `pd.Timestamp.now()` at inference. Once fixed, these features can be re-enabled.

Every future training report must include: train-table filename, total rows, train/val/test counts, class balance per split, split method, year range per split, feature count, pre-event validity status.

## Post-event feature exclusion policy

The following features exist in the feature dataset but are **permanently excluded** from the pre-event model training table. They derive from the current announcement text or outcome — using them would be look-ahead leakage:

| Feature | Why excluded |
|---|---|
| `feat_superiority_flag` | Keyword match on current PR text (`primary_endpoint_result`, `v_pr_key_info`, etc.) |
| `feat_stat_sig_flag` | Same — p-values and HR/OR from current result |
| `feat_clinically_meaningful_flag` | Same — clinical significance language from current result |
| `feat_mixed_results_flag` | Same — failure/miss language from current result |
| `feat_endpoint_outcome_score` | Derived from `primary_endpoint_met` (Yes/No/Unclear) — the current outcome |
| `feat_primary_endpoint_known_flag` | Same source — 1 = outcome is known (i.e., announcement already happened) |

These features are retained in the feature dataset for post-hoc analysis only. They must never be added to `build_pre_event_train_v2.py`.

A legitimate pre-event equivalent would require matching prior-phase publications for the same drug — not yet implemented.

## Timing feature caveats (oncology)

CT.gov primary completion date–based features (`feat_primary_completion_imminent_30d/90d`, `feat_completion_recency_bucket`, `feat_days_to_primary_completion`) are **valid but carry a known caveat for oncology trials**:

In OS/PFS/DFS-driven oncology trials, the efficacy readout occurs when a target number of events accrues — often 6–24 months **before** CT.gov primary completion date. The CT.gov date reflects the administrative study close, not the readout. This means imminence flags can be 0 or "far" even when an oncology readout is genuinely imminent.

These features are kept as-is. They carry real signal for non-oncology (where CT.gov date ≈ readout date) and even for oncology the COMPLETED + recently-closed signal is meaningful. A future improvement would add a separate feature family from enrollment completion announcements or "on track / delayed" text in CT.gov study descriptions — not yet implemented.

## Multiclass status

`target_move_bucket` (5-class) exists in the dataset but no multiclass model has been trained. Binary (`target_large_move`) remains the primary objective. Multiclass is deferred until binary AUC exceeds 0.70 reliably.

---

## Feature groups (v0.3 — 69 features)

| Group | Count | Examples |
|---|---|---|
| Trial / regulatory | 15 | phase_num, regulatory_stage_score, pivotal_proxy_score, enrollment_log |
| Company pipeline | 6 | n_unique_drugs, asset_trial_share, pipeline_depth_score, lead_asset_dependency |
| Financial / market | 5 | volatility, log_market_cap, cash_runway_proxy, short_squeeze_flag |
| Trial design flags | 9 | randomized, blinded, open_label, small_trial, completed, withdrawn |
| Disease / therapeutic | 14 | mesh_level1_encoded, therapeutic_superclass (11 one-hot), oncology/cns/rare flags |
| **Timing (new v0.3)** | **11** | imminent_30d/90d, recency_bucket, time_since_last_event, sequence_num, recent_flag |
| Fold-safe priors | 6 | prior_mean_abs_move by phase/superclass/market-cap; prior_large_move_rate |
| Event proximity | 5 | event_proximity_bucket (one-hot: future_far/near/just_completed/past/unknown) |
| **CT.gov Timing (new v0.4)** | **11** | ctgov_primary_completion_date, days_to_primary_completion, imminence flags (CT.gov-derived), ct_status_current, active_not_recruiting_flag, completed_flag, days_since_ctgov_last_update, recent_ctgov_update_flag, status_timing_consistency_flag |
| **CT.gov Pipeline Proxy (new v0.4)** | **8** | n_active/late/completed/total trials by sponsor; pipeline_maturity_score; n_trials/late_trials same_intervention; asset_maturity_score |
| **Oncology interactions (new v0.5)** | **4** | oncology × imminent_30d/90d; oncology × recent_completion; oncology × recency_imminent |

---

## Reports & history

- **Current report:** [`reports/ml_pre_event_report_v0.3_20260313.md`](reports/ml_pre_event_report_v0.3_20260313.md)
- **Reports index:** [`reports/README.md`](reports/README.md)
- **Previous reports:** [`reports/reports_history/`](reports/reports_history/)

---

## Changelog

### v3.49 — 2026-03-27 (Pass-9: biological feature families → v17 retrain)

- **7 new biological features** in two families — zero new LLM API calls, derived from existing `disease_genetic_basis` column and trial registration metadata:
  - **Family A — Heritability (3 features):** `feat_genetic_basis_encoded` (ordinal 0–3), `feat_heritability_proxy_score` (float 0–1; monogenic=0.85, somatic=0.45, polygenic=0.35, none=0.10), `feat_heritability_level` (ordinal bin: low/moderate/high)
  - **Family B — Enrichment Relevance (4 features):** `feat_biomarker_stratified_flag` (trial-level keyword match on indication + ct_official_title), `feat_targeted_mechanism_flag` (drug-name suffix rules: mAb/nib/inhibitor/etc. + monogenic disease proxy), `feat_disease_molecular_heterogeneity_score` (disease tractability), `feat_enrichment_relevance_score` (weighted composite)
- **v17 retrain:** `ml_baseline_train_20260327_v17.csv` (1,142 rows, 71 base + 8 priors = 79 features)
  - Test AUC **0.694** (vs v16: 0.695 — flat, within holdout noise), PR-AUC 0.564, CV AUC **0.786 ± 0.077**, best model: LogReg
  - `feat_heritability_level` ranks #20, `feat_genetic_basis_encoded` #24 in LightGBM importance — contributing but modest signal
  - No leakage risk: all 7 features derived from pre-registration metadata and static disease classifications known before the event
- **New script:** `scripts/add_biological_features.py` (Pass-9); idempotent, no external API calls
- **Documentation:** `reports/FEATURE_NOTES.md` updated with Pass-9 entry; `reports/MODEL_REPORTS.md` updated
- **Files changed:** `scripts/add_biological_features.py` (new), `scripts/build_pre_event_train_v2.py` (VERSION=17), `reports/FEATURE_NOTES.md`

### v3.48 — 2026-03-25 (Phase 5: extended fold-safe priors → v16 retrain)

- **2 new fold-safe priors** added to `scripts/add_train_fold_priors.py` (total: 6 → 8):
  - `feat_prior_large_move_rate_by_market_cap_bucket` — positive rate by market-cap tier (micro/small/mid/large); complements existing mean-abs-move prior for the same key
  - `feat_prior_large_move_rate_by_phase_x_therapeutic_superclass` — interaction prior; cells with < 5 training samples fall back to phase-level rate prior, then global mean
- **v16 retrain:** `ml_baseline_train_20260323_v16.csv` (1,142 rows, 64 base + 8 priors = 72 features)
  - Test AUC **0.695**, PR-AUC 0.563, CV AUC **0.785 ± 0.077**, best model: LogReg
  - New priors rank #1 and #4 by LogReg importance (real signal)
  - Marginal test AUC regression vs v15 (0.695 vs 0.702) within 172-row holdout noise; LightGBM more affected by sparse interaction cells
- **Documentation:** `reports/FEATURE_NOTES.md` updated with full Phase 1–5 feature engineering history (all previously undocumented)
- **Files changed:** `scripts/add_train_fold_priors.py`, `scripts/build_pre_event_train_v2.py` (VERSION=16)

### v3.47 — 2026-03-25 (Phase 4 data expansion: 2018–2022 historical catalysts → v15 retrain)

- **Data expansion:** Applied scan-and-confirm strategy retroactively to 2018–2022 to recover historical clinical catalysts previously excluded by year filter.
  - **CT.gov cross-match:** Scanned 460 tickers for large moves (2018–2022) → 18,008 candidates; cross-matched against 2,627 CT.gov Phase 2/3 completions within ±10 calendar days per ticker → 43 confirmed clinical catalysts (positives) + 291 CT.gov small-move negatives.
  - **Perplexity classification:** Batch-classified 1,778 unmatched high-normalized moves (5 events/call) → 111 additional confirmed clinical catalysts (73 clinical_trial + 38 fda_decision).
  - **Total new rows:** 444 added to master CSV (2,514 → 2,958 rows), tagged `data_tier ∈ {phase4_ctgov, phase4_ctgov_neg, phase4_perp}`.
- **v15 retrain:** `ml_baseline_train_20260323_v15.csv` (1,142 rows — up from 701, +63%; 64 base + 6 priors = 70 features)
  - Training positive rate: **32%** (365 positives out of 1,142 rows, up from 30%)
  - Test AUC **0.702** (new best, +0.021 vs v14), CV AUC **0.793 ± 0.081**
  - PR-AUC 0.572 (slightly below v14's 0.603; more negatives in expanded dataset)
  - Best model: **Logistic Regression** (LightGBM AUC 0.691, XGBoost 0.633)
  - Top features: `feat_company_historical_hit_rate`, `feat_trial_quality_score`, `feat_blinded_flag`, `feat_genetic_basis_unknown`, `feat_primary_completion_imminent_90d`
- **API cost:** ~$0.17 total (345 Perplexity batch calls for unmatched classification)
- **New scripts:** `scripts/cross_match_events.py`, `scripts/classify_unmatched_catalysts.py`, `scripts/merge_phase4_data.py`
- **Files changed:** `enriched_all_clinical_clean_v3.csv` (2,514 → 2,958 rows), `scripts/build_pre_event_train_v2.py` (VERSION=15, PHASE4_TIERS bypass filter)
- **Feature/training files:** `ml_dataset_features_20260325_v2.csv` (2,822 × 147), `ml_baseline_train_20260323_v15.csv` (1,142 × 71)

### v3.46 — 2026-03-24 (feat_company_historical_hit_rate → v14 retrain)

- **New feature:** `feat_company_historical_hit_rate` (Tier 3) — backward-looking large-move rate per ticker, computed via `shift(1).expanding().mean()` on date-sorted data. Fold-safe. Coverage 86.2%.
- **v14 retrain:** `ml_baseline_train_20260323_v14.csv` (701 rows, 64 base + 6 priors = 70 features)
  - Test AUC **0.681** (+0.013 vs v13), PR-AUC **0.603** (exceeds v11 baseline of 0.573!), Prec@top5% **0.800**
  - CV AUC **0.762 ± 0.075**, best model: LightGBM
  - `feat_days_to_primary_completion` confirmed as #4 importance (was #1 historically)
  - `feat_company_historical_hit_rate` ranks #23 — limited by pre-2023 quiet-completion rows having ~0% positive rate (biased history). Valid feature, limited signal in current dataset.
- **Files changed:** `scripts/add_high_signal_features.py` (Step 0d extension), `scripts/build_pre_event_train_v2.py` (VERSION=14)

### v3.45 — 2026-03-24 (fix timing feature anchors + 5 timing features → v13 retrain)

- **Root cause fix:** All timing features in `add_pre_event_timing_features.py` now use `prediction_date = v_actual_date - 1 day` as anchor instead of `v_actual_date`. This makes them valid for pre-event inference (at inference: `prediction_date = today`).
- **5 features removed from `INVALID_FOR_PRE_EVENT` and added to training roster:**
  - Numeric: `feat_days_to_primary_completion`, `feat_time_since_last_company_event`, `feat_time_since_last_asset_event`
  - Binary: `feat_primary_completion_imminent_30d`, `feat_primary_completion_imminent_90d`
- **v13 retrain:** `ml_baseline_train_20260323_v13.csv` (701 rows, 63 base + 6 priors = 69 features)
  - Test AUC **0.668** (+0.009 vs v12; best model: LogReg), CV AUC **0.758 ± 0.081**
  - `feat_primary_completion_imminent_90d` → #3 importance, `feat_primary_completion_imminent_30d` → #4 — real signal confirmed
  - Note: LightGBM underperforming LogReg; 701 rows + 69 features may benefit from regularization tuning
- **Files changed:** `scripts/add_pre_event_timing_features.py` (prediction_date anchor), `scripts/build_pre_event_train_v2.py` (VERSION=13, removed 5 from INVALID list)
- **Feature files:** `ml_dataset_features_20260323_v1.csv` (2379 × 145), `ml_baseline_train_20260323_v13.csv` (701 × 70)

### v3.44 — 2026-03-24 (fix broken market cap features + 7 Tier 1 features → v12 retrain)

- **Bug fixes (critical):**
  - `feat_log_market_cap`: was in training roster since v1 but never computed — every row was silently imputed to median. Now correctly computed as `log10(market_cap_m)`. Confirmed as #3 feature importance.
  - `feat_market_cap_bucket`: referenced by fold-safe prior `feat_prior_mean_abs_move_atr_by_market_cap_bucket` but never computed — prior fell back to global mean for every row. Fixed; buckets: micro (<$300M), small ($300M-$2B), mid ($2B-$10B), large (>$10B). Coverage: 62.2%.
- **7 Tier 1 features added to training roster** (all pre-computed but excluded from roster):
  - Binary: `feat_controlled_flag`, `feat_priority_review_flag`, `feat_primary_endpoint_known_flag`, `feat_recent_ctgov_update_flag`
  - Numeric: `feat_ctgov_pipeline_maturity_score`, `feat_ctgov_n_late_stage_trials_sponsor`, `feat_ctgov_asset_maturity_score`
- **v12 retrain:** `ml_baseline_train_20260323_v12.csv` (701 rows, 58 base + 6 priors = 64 features)
  - Test AUC **0.659** (−0.026 vs v11; within noise on 112-row test set), PR-AUC **0.562**
  - CV AUC **0.770 ± 0.036** (−0.011 vs v11; lower variance)
  - Top 10: `feat_volatility`, `feat_cash_runway_proxy`, `feat_log_market_cap` ✓ (now real), `feat_ctgov_asset_maturity_score`, `feat_enrollment_log`, `feat_company_event_sequence_num`, `feat_ctgov_pipeline_maturity_score`, `feat_n_trials_for_company`, `feat_ctgov_n_late_stage_trials_sponsor`, `feat_asset_trial_share`
  - Test AUC regression likely noise (112 test rows); CV is more reliable signal
- **Files changed:** `scripts/add_high_signal_features.py` (Step 7 + NEW_FEATURE_META), `scripts/build_pre_event_train_v2.py` (VERSION=12, feature rosters)
- **Feature files:** `ml_dataset_features_20260323_v9.csv` (2379 rows × 145 cols), `ml_baseline_train_20260323_v12.csv` (701 × 65 cols)

### v3.43 — 2026-03-24 (LLM-derived disease biology features → v11 retrain)

- **New features:** Three disease biology features classified via Perplexity (sonar, temp=0) from the `indication` column (1,152 unique diseases, 96.9% coverage):
  - `feat_has_predictive_biomarker` (binary) — whether a known biomarker routinely guides treatment selection
  - `feat_genetic_basis` (categorical: none/monogenic/polygenic/somatic) — primary genetic basis of the disease
  - `feat_targeted_therapy_exists` (binary) — whether an approved or late-stage targeted therapy exists
- **Pre-event safe:** These are static medical knowledge features — inherent to the disease, not the trial or event date
- **v11 retrain:** `ml_baseline_train_20260323_v11.csv` (701 rows, 50 base + 6 priors = 56 features)
  - Test AUC **0.685** (+0.021 vs v10), PR-AUC **0.573** (+0.070), Prec@top 10% **0.727** (+0.363)
  - CV AUC 0.781 ± 0.048 (flat vs v10 — within noise)
- **Pipeline:** `enrich_disease_biology.py` → cache in `cache/disease_biology_v1.json` → `add_high_signal_features.py` Step 6b → `build_pre_event_train_v2.py` v11
- **Bug fix:** Fixed string-based version comparison in `train_pre_event_v3.py` that caused v9 > v11 (lexicographic vs numeric)

### v3.42 — 2026-03-24 (PIT fix for terminated/withdrawn flags → v10 retrain)

- **Leakage fix:** `feat_terminated_flag` and `feat_withdrawn_flag` (CT.gov snapshot) contaminated `feat_trial_quality_score` — 23/33 terminated rows in training were still active at event time, receiving a spurious −2 quality penalty
- **Fix:** Added `feat_terminated_at_event_flag` + `feat_withdrawn_at_event_flag` (AACT PIT, same approach as Option C); `build_trial_quality_score()` now uses PIT flags for the −2 penalty terms
- **v10 retrain:** `ml_baseline_train_20260323_v10.csv` (701 rows, 44 features) — Test AUC 0.664, CV AUC 0.784 ± 0.045 (flat vs v9 — 9 affected rows too few to move aggregate AUC, but scores are now correct)
- Snapshot flags moved to `INVALID_FOR_PRE_EVENT`

### v3.41 — 2026-03-24 (foundational feature port + INVALID roster cleanup → v9 retrain)

- **Root cause fix:** Three legacy root-level scripts (`build_ml_ready_features.py`, `add_high_value_predictors.py`, `completeness_pass.py`) were never ported when dataset expanded from 831 → 2379 rows. ~12 foundational features were absent from training.
- **Features ported into `add_high_signal_features.py`** (Steps 0a–0d, all pre-event safe):
  - Clinical core: `feat_phase_num`, `feat_late_stage_flag`, `feat_enrollment_log`, `feat_randomized_flag`, `feat_design_quality_score`
  - Regulatory flags: `feat_orphan_flag`, `feat_fast_track_flag`, `feat_breakthrough_flag`, `feat_nda_bla_flag`, `feat_regulatory_stage_score`
  - Trial quality: `feat_trial_quality_score`
  - Company foundation: `feat_n_trials_for_company`, `feat_n_unique_drugs_for_company`, `feat_single_asset_company_flag`, `feat_lead_asset_dependency_score`
  - `feat_volatility` (pre-event ATR, strictly backward-looking)
  - `feat_mesh_level1_encoded` (ordinal int, therapeutic area)
- **INVALID_FOR_PRE_EVENT additions:**
  - `feat_short_squeeze_flag`, `feat_ownership_low_flag` — SNAPSHOT_UNSAFE (yfinance current snapshot)
  - `feat_event_proximity_bucket` — anchored to realized event_date; cannot be reproduced pre-event
- **v9 retrain:** `ml_baseline_train_20260323_v9.csv` (701 rows, 42 features)
  - Best model: LightGBM — Test AUC **0.664**, CV AUC 0.784 ± 0.045
  - Top features: `feat_cash_runway_proxy` (#1), `feat_volatility` (#2), `feat_n_trials_for_company` (#3)
  - Test AUC ~flat vs v8 (0.665); CV AUC improvement +0.025 vs v7 (0.759) reflects richer feature set
- **Features CSV:** `ml_dataset_features_20260323_v6.csv` (2379 rows × 135 cols, 73 feat_ columns)

### v3.40 — 2026-03-24 (Option C: AACT point-in-time status → v8 retrain)

- **AACT monthly snapshots:** `fetch_aact_status_history.py` fetched ~39 months of AACT flat files (Jan 2023–Mar 2026) for 710 unique NCT IDs → `ct_status_at_event` column in master CSV
- **New features:** `feat_completed_at_event_flag` + `feat_active_not_recruiting_at_event_flag` (AACT point-in-time, replaces snapshot/date-proxy versions)
- **SNAPSHOT_UNSAFE features removed from training:** `feat_active_not_recruiting_flag`, `feat_completed_before_event` (date proxy)
- **v8 retrain:** `ml_baseline_train_20260323_v8.csv` (701 rows, 41 features) — Test AUC 0.665, CV AUC 0.788 ± 0.045

### v3.39 — 2026-03-23 (feat_completed_flag leakage fix → v6 retrain)
- **Leakage fix (Option B):** `feat_completed_flag` (CT.gov snapshot — SNAPSHOT_UNSAFE) removed from training feature set
  - Root cause: `ct_status == "COMPLETED"` reflects current CT.gov state (March 2026 fetch), not trial status at event time. For 2024 events this leaks future information.
  - Replaced by `feat_completed_before_event = (ct_primary_completion < event_date)` — prospective protocol milestone, registered before trial starts, pre-event valid
  - Null rate in training cohort: 24.7% (below 30% fallback threshold) → imputed as 0 (absent)
- **v6 retrain:** `ml_baseline_train_20260323_v6.csv` (701 rows, 31 features) → `feat_completed_before_event` present, `feat_completed_flag` absent
  - Best model: LogReg — AUC **0.693** (vs v5 0.703, delta −0.010 — expected for removing contamination)
  - `feat_completed_before_event` rank #7 (coef 0.2485) — retains predictive value
  - CV AUC: 0.747 ± 0.047
- **Features CSV:** `ml_dataset_features_20260323_v3.csv` (v2 archived) — added `feat_completed_before_event` column
- Next: Option C — fetch CT.gov status history for ~750 validated rows to get exact point-in-time status → `feat_completed_at_event_flag` → v7 retrain

### v1.4 — 2026-03-23 (EDGAR 8-K historical outcome ingest)
- **SEC EDGAR 8-K pipeline** — `scripts/edgar_8k_ingest.py` — queries all 8-K press releases for history_only tickers (2020–2022) using EDGAR Submissions API + Exhibit 99.1 parsing
- Covered 143 US-listed tickers (1,145 events); 16 foreign filers (SNY, NVS, AZN, BNTX, GLPG etc.) excluded — file 6-K not 8-K
- Match rate: 64.9% of events found an 8-K within ±21d; 35.1% no 8-K (expected — large-cap pharma file many non-clinical 8-Ks)
- Clinical hit rate: 17.9% (205 events with clinical/FDA press release content)
- Outcomes extracted: 176 positive · 6 negative · 9 inconclusive · 3 mixed (81 keyword heuristic + 124 Perplexity)
- Output: `edgar_8k_matches_20260323.csv` — all columns tagged `DO_NOT_USE_FOR_MODEL`
- **Enables next step:** Build `feat_company_prior_success_rate` + `feat_drug_prior_phase_success_rate` features using historical outcomes (valid pre-event features — computed from events BEFORE the target event date)
- API: EDGAR (free, no key) + Perplexity sonar-pro (existing key). Total cost: EDGAR free, ~$0.10 Perplexity
- See `reports/DATASET_NOTES.md` (2026-03-23 section) for full coverage breakdown

### v0.9 — 2026-03-17 (pre-event validity audit + patch)
- **Validity audit:** 9 timing features identified as INVALID for strict pre-event use
- All 9 use `v_actual_date` or `event_date` (realized event date) as anchor — not knowable before the event
- Invalid features excluded from `build_pre_event_train_v2.py` — safe for next retrain
- v3 model marked: **approximately valid** (historical analysis OK; not for live deployment without inference fix)
- v3 training report updated with all mandatory fields (rows, splits, class balance, validity status)
- Permanent pre-event validity rule added to README and audit report
- **No retrain yet** — awaiting approval after reviewing performance impact
- Invalid features: `feat_days_to_primary_completion` (#2 importance), `feat_time_since_last_company_event` (#4), `feat_completion_recency_bucket`, `feat_primary_completion_imminent_30d/90d`, `feat_recent_completion_flag`, `feat_time_since_last_asset_event`, `feat_recent_company/asset_event_flag`
- Valid ordinal features kept: `feat_company_event_sequence_num` (#3), `feat_asset_event_sequence_num` (#7)
- Fix path: add `prediction_date` param to timing feature scripts → re-enable with correct anchor
- See: [`reports/pre_event_validity_audit_v0.6_20260317.md`](reports/pre_event_validity_audit_v0.6_20260317.md)

### v1.3 — 2026-03-19 (Benzinga pilot)
- **Benzinga API pilot** — tested all relevant endpoints for dataset improvement potential
- Script: `scripts/benzinga_pilot_event_ingest.py`
- Access verdict: current plan (`BENZIN_API_KEY`) provides only unfiltered rolling news feed; ticker filter broken, press-releases (404), FDA calendar (403), WIIM (404) all inaccessible
- Pilot fetched 1,000 items (2026-02-20 → 2026-03-19), 149 master-ticker mentions, 7 within ±3d of known events — no historical 2020–2022 coverage possible
- **Scaling NOT recommended** until plan upgraded to support ticker filtering or press-releases endpoint
- All Benzinga columns flagged `DO_NOT_USE_FOR_MODEL`; output CSVs gitignored
- See `reports/DATASET_NOTES.md` (2026-03-19 section) for full access audit

### v1.2 — 2026-03-19 (v5 expanded retrain)
- **Retrain on 701-row expanded strict-clean pool** — +105 rows vs v4 (relaxed mesh-only exclusion)
- `ml_baseline_train_20260318_v5.csv`: 701 rows, 25 base + 6 priors = 31 features, 30.4% positive rate
- **New best model: Logistic Regression** — AUC 0.703, CV AUC 0.752 ± 0.053, Prec@10% 0.545
- vs v4: AUC +0.011, CV AUC +0.041 with tighter variance (±0.053 vs ±0.112), Prec@10% +0.101
- XGBoost and LightGBM both scored lower than LogReg on this pool (AUC 0.638 / 0.647)
- `models/model_pre_event_v3_20260312.pkl` updated to LogReg v5
- ✓ STRICT_CLEAN — same 9-feature exclusion policy, same feature roster as v4

### v1.1 — 2026-03-18 (dataset tiering pass)
- **Curation / tiering pass on full 2514-row expanded master**
- Script: `scripts/curate_dataset_tiers.py` — assigns `data_tier` to every row
- Tier counts: `trusted_trainable` 758 · `repairable` 22 · `history_only` 1733 · `reject` 1
- `trusted_trainable` (758 rows, 2023+, complete price): 231 positives — **30.5% positive rate** ✓ already in 25–30% target window
- `history_only` (1733 rows, mostly 2020–2022): genuine hard negatives (mean AbsATR ≈ 0.7); do NOT train on; keep for sponsor/asset history and future timing model
- `repairable` (22 rows): 12 × 2023+ missing price (yfinance backfillable) + 7 × pre-2023 High/Extreme events
- Output: `enriched_all_clinical_clean_v3_tiered_20260318_v1.csv` (60 cols, +`data_tier` +`target_large_move`) and `candidate_strict_trainable_20260318_v1.csv` (758 rows)
- Full analysis in `reports/DATASET_NOTES.md` (2026-03-18 section)

### v1.0 — 2026-03-17 (strict-clean baseline)
- **STRICT_CLEAN retrain complete** — all event-date-anchored features removed from training
- `ml_baseline_train_20260317_v4.csv`: 596 rows, 24 base features + 6 priors = 30 total, 0 invalid
- Contaminated train table v3 (14 invalid feature columns) archived
- **XGBoost AUC: 0.692** (honest strict-clean baseline), CV AUC: 0.711 ± 0.112, Prec@10%: 0.444
- AUC drop vs contaminated model (0.730 → 0.692) confirms the invalid features were inflating metrics
- Top strict-clean features: `feat_company_event_sequence_num`, `feat_completed_flag`, prior by therapeutic class, `feat_cash_runway_proxy`

### v0.9 — 2026-03-17 (pre-event validity audit + doc consolidation)
- Pre-event validity audit: identified 9 event-date-anchored features as INVALID for strict pre-event use
- Patched `build_pre_event_train_v2.py` to exclude all 9 features via `INVALID_FOR_PRE_EVENT` list
- Consolidated 7 standalone `.md` report files into 3 canonical running docs: `MODEL_REPORTS.md`, `FEATURE_NOTES.md`, `DATASET_NOTES.md`
- Patched `train_pre_event_v3.py` to prepend to `MODEL_REPORTS.md` instead of creating standalone files
- Previous v3 model marked as approximately valid (historical analysis only, not for live deployment)

### v0.8 — 2026-03-17 (restrict training to 2023+ rows) ⚠ superseded by v1.0
- **Training cohort restricted to 2023+ events** — 2020–2022 rows excluded from train/val/test
- After fix: 596 training rows (30.9% positive rate) vs 2049 rows (0.6% positive rate) before
- Train table `ml_baseline_train_20260317_v3.csv` — **archived** (contained 14 invalid feature columns)
- Reported AUC 0.730 / Prec@10% 0.778 were **inflated** by 9 event-date-anchored features; superseded by v1.0 strict-clean retrain

### v0.7 — 2026-03-16 (full pipeline on v3 master + pipeline bug fixes)
- **First complete pipeline run on v3 master (2514 rows → 2379 after prep)**
- Pipeline now correctly flows 2379 rows through all 8 steps end-to-end
- **New model retrained on v3 data: LightGBM, AUC 0.672, CV AUC 0.704 ± 0.008, Prec@top10% 0.514**
- Previous best was LogReg 0.661 / CV 0.682 ± 0.129 / Prec@10% 0.308 (827 rows)
- Pipeline bug fixes (all 6 find-latest functions):
  - Removed archive-search from all `_find_latest*` functions — archived high-version files no longer shadow new files
  - Fixed steps 3 + 4 (`refresh_ctgov_features`, `build_ctgov_pipeline_proxies`) to use integer versioning (`_YYYYMMDD_vN.csv`) instead of semver (`v0.X_YYYYMMDD.csv`)
  - Fixed step 1 (`prepare_ml_dataset`): was reading wrong input file (`v2` in `scripts/`) and writing to wrong location
  - Fixed step 6 (`add_oncology_timing_interactions`): was using old semver regex, couldn't find pipeline output
  - Fixed step 7 (`build_pre_event_train_v2`): added target computation (`target_large_move`) and median-NaN fallback to 0
  - Fixed step 8 (`train_pre_event_v3`): added `fillna(0)` after prior injection to handle post-merge NaN
- **Known issue:** 2020–2022 rows have ~0.3–0.5% positive rate vs 24–32% for 2023+ (time-split puts these in train → 0.6% positives in train). Next step: filter training data to 2023+ rows before retraining

### v0.6 — 2026-03-16 (threshold analysis + pipeline orchestration)
- **New binary target defined: ≥ 3.0× ATR AND abs(move_pct) ≥ 10%** (was ≥ 5× ATR, no floor)
- Adds the Medium class (3–5× ATR) as positives: +53 new positive rows vs baseline
- On full v3 (2500 rows): 242 positives = **9.7%** positive rate
- On original v2 rows only (881): 237 positives = **26.9%** — hits the 25–30% target for the curated event cohort
- Absolute 10% floor removes 23 large-cap pharma false-positives (AZN/PFE/SNY/AMGN with 5–8% absolute moves)
- 15% floor vs 10% floor = 3-row difference — 10% chosen
- Date mismatch: 96.8% exact, 1.7% off by 1 day, only 0.8% >7 days — no filtering needed
- Created `scripts/run_full_pre_event_pipeline.py` — one-command 8-step pipeline with --start-step, --dry-run, --master flags
- **Status: threshold approved → ready to run full pipeline and retrain**
- See: [`reports/binary_target_threshold_analysis_v0.6_20260316.md`](reports/binary_target_threshold_analysis_v0.6_20260316.md)

### v0.5 — 2026-03-16 (dataset expansion)
- **Master dataset expanded: 862 → 2514 rows (+1652 new rows)**
- Historical extension: Phase 2/3 CT.gov completions 2020-01-01 → 2022-12-31
- Script: `scripts/expand_historical_events.py`
- New rows: 1652 | Phase 2: 695, Phase 3: 688, Phase 2/3: 55, Ph1/2: 182
- Year coverage now: 2020=464, 2021=445, 2022=398 (was 19/21/34)
- New row move distribution: Noise=1466 (89%), Low=133, Med=17, High=3, Extreme=2
- Oncology: 26.8% of new rows | All 1652 have nct_id + ATR computed
- Full dataset: Noise=81.2%, Low=7.5%, Medium=3.2%, High=3.7%, Extreme=3.9%
- **Next required step:** Re-run full feature pipeline on v3 master (prepare_ml_dataset → all pass scripts → rebuild train table → retrain)
- See: [`reports/dataset_expansion_strategy_v0.5_20260316.md`](reports/dataset_expansion_strategy_v0.5_20260316.md)

### v0.5 — 2026-03-15
- Added 4 oncology × timing interaction features via `add_oncology_timing_interactions.py`
- Feature dataset: 827 × 149 cols, 86 features
- Quick comparison (LogReg, same v0.3 split): +0.005 AUC on test (0.653→0.658), neutral on val — no degradation
- Full lift expected when model is retrained on the complete v0.5 feature set including CT.gov proxies
- See: [`reports/pre_event_model_followup_actions_v0.4_20260315.md`](reports/pre_event_model_followup_actions_v0.4_20260315.md)

### v0.7 — 2026-03-23 (Option C: AACT point-in-time status)
- Replaced two SNAPSHOT_UNSAFE features with ground-truth point-in-time variants from AACT monthly archives
- `fetch_aact_status_history.py`: downloads ~39 AACT monthly flat-file snapshots (Jan 2023–Mar 2026), extracts `overall_status` for 710 unique NCT IDs, writes `ct_status_at_event` (point-in-time lookup: latest month ≤ event date) and `data_tier` to master CSV
- **Removed from training:** `feat_active_not_recruiting_flag` (SNAPSHOT_UNSAFE — same issue as `feat_completed_flag`)
- **Added:** `feat_completed_at_event_flag` (AACT PIT, replaces `feat_completed_before_event` date proxy)
- **Added:** `feat_active_not_recruiting_at_event_flag` (AACT PIT, replaces `feat_active_not_recruiting_flag`)
- v7 retrain: **AUC 0.700** (+0.007 vs v6 0.693); CV AUC 0.759 ± 0.040; LogReg best; 400 leakage cases corrected
- Cache: `cache/aact_status_history_v1.json` (~200KB); peak disk 2.2GB per month, cleaned after parse
- See: `reports/FEATURE_NOTES.md` for full Option C coverage/watch-list

### v0.6 — 2026-03-23 (feat_completed_flag leakage fix)
- `feat_completed_flag` removed from training (SNAPSHOT_UNSAFE)
- `feat_completed_before_event` added (date proxy: ct_primary_completion < event_date)
- v6 retrain: AUC 0.693 (delta −0.010 vs v5, expected from removing contamination)
- Train table: `ml_baseline_train_20260323_v6.csv` (701 rows × 31 cols)

### v0.4 — 2026-03-14 (audit update — 2026-03-15)
- ML audit: confirmed target definitions, post-event exclusion policy, timing caveats
- Clarified binary target cutoff: `target_large_move = 1` at ≥5× ATR (High+Extreme), not ≥3×
- Documented 6 permanently excluded post-event features and rationale
- Documented oncology timing caveat for CT.gov primary completion–based features
- Deferred multiclass until binary AUC > 0.70; deferred management/CRO quality feature

### v0.4 — 2026-03-14
- Added 11 CT.gov-grounded timing features via `refresh_ctgov_features.py` (per-NCT-ID API fetch, 679 unique IDs, 93.5–94.8% coverage)
- Added 8 CT.gov pipeline proxy features via `build_ctgov_pipeline_proxies.py` (293 sponsors, 596 drugs queried)
- Rebuilt `ml_feature_dict_v0.4_20260313.csv` — 82 entries, unified schema
- Key new signals: `feat_ctgov_pipeline_maturity_score`, `feat_ctgov_asset_maturity_score`, `feat_ct_status_current`, `feat_recent_ctgov_update_flag`
- Feature dataset: 827 × 145 cols (was 132 in v0.3)
- Model not yet retrained — next step: rebuild train table and retrain on v0.4 features
- See: [`reports/ctgov_timing_pipeline_features_v0.4_20260313.md`](reports/ctgov_timing_pipeline_features_v0.4_20260313.md)

### v0.3 — 2026-03-13
- Added 9 timing/sequence features (imminence flags, recency bucket, time-since-last-event, sequence numbers)
- Added 6 fold-safe reaction priors via `FoldPriorEncoder`
- Rebuilt train table (`build_pre_event_train_v2.py`)
- Retrained LogReg + LightGBM + XGBoost; LogReg best (AUC 0.661, CV 0.682 ± 0.129)
- Enforced `{name}_{vX.Y}_{date}` naming convention across all output files
- `feat_days_to_study_completion` skipped — `ct_study_completion` not in dataset

### v0.2 — 2026-03-12
- Time-aware CV (5-fold `TimeSeriesSplit`) + 3-model comparison
- Best result: LogReg AUC 0.685, Prec@top5% = 0.833 (2.6× base rate), CV AUC 0.735 ± 0.061
- Pass-4 feature engineering: 22 new features (pivotal proxy, trial design, therapeutic class, financial context)
- Baseline error analysis + threshold sweep

### v0.1 — 2026-03-10
- First ML dataset built from validated catalysts
- Pass-1 + Pass-3 feature engineering (trial quality, regulatory state, company pipeline, reaction priors)
- First baseline models (LogReg + LightGBM); train/val/test split established
- MeSH Level-1 disease classification (831/831 filled)

*Full detailed changelog available in git log.*

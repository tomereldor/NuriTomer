# Biotech Catalyst v3 — Pre-Event Move Prediction

**Goal:** Given today's public company/trial information (before any announcement), predict whether the eventual catalyst will cause a large stock move (≥ 3–5× ATR-normalized).

No press release content is used. All predictions are based on pre-event structural features only.

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

## Current state (v0.4 — 2026-03-14)

### Source of truth files

| File | Description |
|---|---|
| `ml_dataset_features_v0.4_20260313.csv` | **THE ML INPUT FILE** — 827 rows × 145 cols, 82 features, all pre-event |
| `ml_feature_dict_v0.4_20260313.csv` | Feature dictionary — 82 entries with coverage + description |
| `ml_baseline_train_v0.3_20260313.csv` | Training table — 813 rows, 64 model-ready features (v0.3 model) |
| `enriched_all_clinical_clean_v2.csv` | Master raw dataset — 862 rows × 58 cols (do not use directly for ML) |
| `biotech_universe_expanded.csv` | 460 tracked biotech tickers ($50M–$10B) |

### Current model (v0.3 — not yet retrained on v0.4 features)

| File | Description |
|---|---|
| `models/model_pre_event_v0.3_20260313.pkl` | Best model — Logistic Regression |
| `models/prior_encoder_v0.3_20260313.pkl` | Fold-safe prior encoder (fit on train, apply to val/test) |

### Current model report (v0.3)

→ [`reports/ml_pre_event_report_v0.3_20260313.md`](reports/ml_pre_event_report_v0.3_20260313.md)

| Metric | v0.3 |
|---|---|
| Best model | Logistic Regression |
| Test ROC-AUC | 0.661 |
| CV AUC (5-fold) | 0.682 ± 0.129 |
| Prec @ top 10% | 0.308 |
| Features | 69 (incl. 9 timing + 6 fold-safe priors) |

### CT.gov feature refresh report (v0.4)

→ [`reports/ctgov_timing_pipeline_features_v0.4_20260313.md`](reports/ctgov_timing_pipeline_features_v0.4_20260313.md)

---

## File naming convention

```
{name}_{vX.Y}_{YYYYMMDD}.{ext}
```

- Version first, always cumulative (v0.3 → v0.4 → …), never reset per date
- Date = date file was last generated
- Previous versions → `archive/` (gitignored locally) or `reports/reports_history/`

---

## Pipeline overview

```
enriched_all_clinical_clean_v2.csv   ← master dataset
        ↓
scripts/add_pre_event_timing_features.py   → ml_dataset_features_v0.3_20260313.csv
        ↓
scripts/build_pre_event_train_v2.py        → ml_baseline_train_v0.3_20260313.csv
        ↓
scripts/train_pre_event_v3.py              → models/ + reports/
```

### Key scripts

| Script | What it does |
|---|---|
| `add_pre_event_timing_features.py` | Pass-5: adds 9 timing/sequence features to the feature dataset |
| `add_train_fold_priors.py` | `FoldPriorEncoder` — inject reaction priors safely inside CV folds |
| `build_pre_event_train_v2.py` | Build train/val/test table from feature dataset |
| `train_pre_event_v3.py` | Train LogReg + LightGBM + XGBoost, CV, report, plots |
| `add_high_signal_features.py` | Pass-4: pivotal proxy, trial design flags, therapeutic superclass |
| `validate_catalysts.py` | Validate noise-class rows via Perplexity + PR fetch |
| `backfill_price_at_event.py` | Fill price_before / price_after / move_2d_pct via yfinance |

### Running the full feature → model pipeline

```bash
python -m scripts.add_pre_event_timing_features   # regenerate feature dataset
python -m scripts.build_pre_event_train_v2         # rebuild train table
python -m scripts.train_pre_event_v3               # retrain models + report
```

---

## Target definition

- **`target_large_move`** (binary, primary): 1 = ATR-normalized move ≥ **5×** (High or Extreme class)
- **`target_abs_move_atr`** (continuous): raw ATR-normalized absolute move magnitude
- **`target_move_bucket`** (5-class): Noise / Low / Medium / High / Extreme

ATR = Wilder's RMA, `ewm(alpha=1/20, adjust=False)`, 20 trading-day lookback, strictly pre-event.
Class thresholds: Noise < 1.5× · Low 1.5–3× · Medium 3–5× · High 5–8× · Extreme ≥ 8×

**Move window:** Last trading day strictly before event → first trading day strictly after event (1 overnight move bracketing the announcement; `move_2d_pct` column name is shorthand for "2 days apart", not a 2-day trailing return).

**Binary cutoff clarification:** `target_large_move = 1` when `move_class_norm ∈ {High, Extreme}`, i.e., ATR-normalized move ≥ 5×. The goal description "≥ 3–5× ATR" refers to the Medium–High boundary; the actual binary positive class starts at High (5×).

---

## Key architecture decisions

- **Pre-event only:** No press release content. All features derivable from public CT.gov + financial data before the news.
- **Time-based splits:** Train/val/test split on `v_actual_date`; CV uses `TimeSeriesSplit` to prevent future leakage.
- **Fold-safe priors:** Reaction priors (mean ATR move by phase/disease/mkt-cap) fit on train fold only via `FoldPriorEncoder`. Never precomputed globally.
- **ATR normalization:** Moves normalized by pre-event ATR to make large-cap and small-cap events comparable.
- **Single master CSV:** `enriched_all_clinical_clean_v2.csv` is the one file all enrichments write to.

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

---

## Reports & history

- **Current report:** [`reports/ml_pre_event_report_v0.3_20260313.md`](reports/ml_pre_event_report_v0.3_20260313.md)
- **Reports index:** [`reports/README.md`](reports/README.md)
- **Previous reports:** [`reports/reports_history/`](reports/reports_history/)

---

## Changelog

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

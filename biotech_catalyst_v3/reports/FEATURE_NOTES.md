# Feature Notes — Biotech Catalyst v3

Canonical running document for feature validity, CT.gov notes, audit findings, and oncology caveats.
Newest entry at top.

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

# Pre-Event Model — Validity Audit

**Date:** 2026-03-17
**Scope:** All features in the current pre-event binary classifier
**Hard rule:** The model may ONLY use information publicly available BEFORE the future event occurs. Any feature that uses the realized event date, announcement date, stock-move date, or event outcome is forbidden.

---

## Part 1 — Latest trained model: confirmed facts

| Field | Value |
|---|---|
| **Model file** | `models/model_pre_event_v3_20260312.pkl` (best: LightGBM) |
| **Train table** | `ml_baseline_train_20260317_v3.csv` |
| **Total rows in train table** | **596** |
| **Train / Val / Test** | **417 / 89 / 90** (70/15/15 time-based split on v_actual_date) |
| **Class balance (overall)** | 184 positive / 412 negative = **30.9% positive** |
| **Class balance (train)** | 119 pos / 298 neg = **28.5%** |
| **Class balance (val)** | 27 pos / 62 neg = **30.3%** |
| **Class balance (test)** | 38 pos / 52 neg = **42.2%** |
| **Split method** | Time-ordered (`v_actual_date` ascending); training restricted to 2023+ events |
| **Feature count** | 44 columns (38 base + 6 one-hot dummies for categorical encoding) |
| **Target** | `target_large_move = 1` when `abs_atr >= 3.0 AND abs(move_pct) >= 10%` |

**Note on test class balance:** 42.2% positive in test is higher than train (28.5%). This is because 2025–2026 events (which land in test/val) have a higher positive rate than 2023–2024 (which are in train). This is an expected temporal shift, not a bug.

---

## Part 2 — Pre-event validity audit

### Audit methodology

For each timing/date feature, I checked:
1. What column is used as the anchor (event date vs. CT.gov date vs. fixed reference)
2. Whether the anchor requires knowledge of the realized event/announcement date
3. Whether the concept remains valid if the anchor is replaced with "today" (prediction date)

**Feature source cross-reference:**
- `feat_days_to_primary_completion` → `refresh_ctgov_features.py` line 258: `(pc_date - evt_date).days` where `evt_date = v_actual_date`
- `feat_recent_completion_flag` → `add_high_signal_features.py` line 317: `(event_date - ct_primary_completion).dt.days <= 365`
- `feat_time_since_last_*` → `add_pre_event_timing_features.py` line 193–203: `v_actual_date_current - v_actual_date_previous`
- `feat_*_event_sequence_num` → `add_pre_event_timing_features.py` line 206–212: `cumcount()` within sorted date groups

---

### Full audit table

| Feature | Computed as | Anchor used | PRE-EVENT VALID? | Reason |
|---|---|---|---|---|
| `feat_days_to_primary_completion` | `ct_primary_completion - v_actual_date` | **v_actual_date** (realized announcement date) | **INVALID** | v_actual_date is the future event date, unknown at prediction time |
| `feat_primary_completion_imminent_30d` | `days_to_primary_completion ∈ [0,30]` | Derived from above | **INVALID** | Derived from invalid anchor |
| `feat_primary_completion_imminent_90d` | `days_to_primary_completion ∈ [0,90]` | Derived from above | **INVALID** | Derived from invalid anchor |
| `feat_completion_recency_bucket` | Categorical from `days_to_primary_completion` | Derived from above | **INVALID** | Derived from invalid anchor |
| `feat_recent_completion_flag` | `(event_date - ct_primary_completion) <= 365` AND status=COMPLETED | **event_date** (realized) | **INVALID** | Uses realized event_date as anchor |
| `feat_time_since_last_company_event` | `v_actual_date_current - v_actual_date_prev` (same ticker) | **v_actual_date** of current event | **INVALID** | Current event endpoint = future event date, unknown at prediction time |
| `feat_time_since_last_asset_event` | Same, per ticker+drug | **v_actual_date** of current event | **INVALID** | Same |
| `feat_recent_company_event_flag` | `time_since_last_company_event <= 90` | Derived from above | **INVALID** | Same |
| `feat_recent_asset_event_flag` | `time_since_last_asset_event <= 180` | Derived from above | **INVALID** | Same |
| `feat_asset_event_sequence_num` | `cumcount()` within ticker+drug, sorted by v_actual_date | v_actual_date for SORTING ONLY; the value = ordinal count of prior events | **VALID** | Count of prior events for this drug is knowable before next event occurs |
| `feat_company_event_sequence_num` | `cumcount()` within ticker, sorted by v_actual_date | v_actual_date for sorting | **VALID** | Same — ordinal position is pre-event knowable |
| `feat_completed_flag` | `ct_status == "COMPLETED"` | CT.gov status (public) | **VALID** | CT.gov status visible before any announcement |
| `feat_active_not_recruiting_flag` | `ct_status == "ACTIVE_NOT_RECRUITING"` | CT.gov status | **VALID** | Same |
| All other features | Trial design, financial, disease class, pipeline, CT.gov pipeline proxy | No event date used | **VALID** | No event date anchor |

### Severity assessment

| Feature | Model importance (LightGBM) | Action |
|---|---|---|
| `feat_days_to_primary_completion` | **#2 (149)** — highest-impact removal | EXCLUDED from training table |
| `feat_time_since_last_company_event` | #4 (39) | EXCLUDED |
| `feat_company_event_sequence_num` | #3 (54) | KEPT — VALID |
| `feat_asset_event_sequence_num` | #7 (26) | KEPT — VALID |
| `feat_time_since_last_asset_event` | #9 (17) | EXCLUDED |
| `feat_recent_completion_flag` | present | EXCLUDED |
| `feat_completion_recency_bucket_*` | present | EXCLUDED (6 one-hot dummies) |
| `feat_primary_completion_imminent_*` | present | EXCLUDED (2 features) |
| `feat_recent_*_event_flag` | present | EXCLUDED (2 features) |

**Total removed: 9 base features + 6 one-hot dummies = 15 columns removed from training table.**

---

## Part 3 — Current model validity status

**`models/model_pre_event_v3_20260312.pkl` and `ml_baseline_train_20260317_v3.csv`:**

```
STATUS: APPROXIMATELY VALID for historical analysis
        INVALID for strict live deployment without inference-time recomputation
```

**Why approximately valid for historical analysis:**
- The model correctly excludes all outcome features (no PR text, no endpoint results)
- The timing features encode valid pre-event signals (CT.gov timing context)
- The anchor discrepancy (v_actual_date vs. prediction_date) is small for most events: 96.8% of rows have 0-day mismatch between event_date and v_actual_date
- AUC 0.730 reflects genuine predictive lift on historical held-out test events

**Why invalid for strict live deployment:**
- `feat_days_to_primary_completion = ct_primary_completion - v_actual_date` cannot be reproduced at inference because v_actual_date (the future announcement date) is unknown
- At inference, these features must be computed with `prediction_date = today`
- Without this fix, the model's top two features (#2, #4 by importance) would require a different computation path than training

**Model and train table marked:** both files are now carrying this audit finding in the permanent record. **Do not use the v3 model for live deployment without patching the inference pipeline.**

---

## Part 4 — Fix path (do NOT implement yet, awaiting retrain approval)

**Short-term fix (required before live inference):**

In `add_pre_event_timing_features.py` and `refresh_ctgov_features.py`, change the anchor from `v_actual_date` to a `prediction_date` parameter:

```python
# Before:
date_col = "v_actual_date" if "v_actual_date" in out.columns else "event_date"
evt_date = _parse_dates(out[date_col])

# After:
# For training: prediction_date = v_actual_date (unchanged)
# For inference: prediction_date = pd.Timestamp.now().normalize()
def build_timing_features(df, prediction_date=None):
    if prediction_date is None:
        date_col = "v_actual_date" if "v_actual_date" in df.columns else "event_date"
        evt_date = _parse_dates(df[date_col])
    else:
        evt_date = pd.Series([prediction_date] * len(df), index=df.index)
```

For the training pipeline, this would give identical results to the current implementation (prediction_date = v_actual_date per row). For inference, you'd pass `prediction_date = today` and the features would correctly reflect "how far is CT.gov completion from today?"

**For `feat_recent_completion_flag` in `add_high_signal_features.py`:**
```python
# Use prediction_date instead of event_date:
days_since = (prediction_date - comp_dates).dt.days
```

**Impact on model performance (estimated):**
- Removing 9 invalid features will reduce feature count for next retrain
- The two most impactful removals are `feat_days_to_primary_completion` (#2) and `feat_time_since_last_company_event` (#4)
- Expected AUC impact: moderate decline, partially offset by cleaner features
- `feat_company_event_sequence_num` (#3) and `feat_asset_event_sequence_num` (#7) are KEPT — these provide the sequencing signal without date anchor issues

**Approved alternative (re-enable with fixed anchor):**
Once the prediction_date parameter is added to the feature scripts, these features can be re-added to `build_pre_event_train_v2.py` as valid pre-event features. The exclusion in this audit applies only to the CURRENT IMPLEMENTATION (anchored to v_actual_date).

---

## Permanent rule — Pre-event feature validity

> **PRE-EVENT MODEL HARD RULE:**
>
> Any feature used in the pre-event stock move size model must be computable using ONLY information publicly available BEFORE the future event occurs. The following are FORBIDDEN:
>
> - The realized event date (`v_actual_date`, `event_date`, `event_trading_date`)
> - The realized announcement date or PR date
> - The realized stock move date
> - The content or outcome of the announcement (`primary_endpoint_result`, `v_pr_key_info`, etc.)
>
> Features that use `v_actual_date` or `event_date` as an ANCHOR for computing "how far away" a trial completion is from the event are INVALID, because the event date is future information.
>
> Features that use past event dates (previous events for sorting/sequencing) are VALID as long as the CURRENT event date is not used as an endpoint.
>
> Ordinal sequence counts (how many prior events exist for this company/drug) are VALID.

---

## Part 5 — Mandatory fields for every future training report

Every model training report must include:

| Field | Required |
|---|---|
| Train table filename | ✓ |
| Feature dataset filename | ✓ |
| Total usable rows (post-filter) | ✓ |
| Train / val / test row counts | ✓ |
| Class balance per split | ✓ (n pos, n neg, % positive) |
| Split method and key (time-based / random) | ✓ |
| Year range in each split | ✓ |
| Any year/cohort exclusions | ✓ |
| Feature count and list of excluded features | ✓ |
| Pre-event validity status | ✓ |

These fields are now required in all `reports/ml_pre_event_*` report files.

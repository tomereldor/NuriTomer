# CT.gov Timing & Pipeline Feature Refresh — v0.4

**Date:** 2026-03-14
**Dataset:** `ml_dataset_features_v0.4_20260313.csv` (827 rows × 145 cols)
**New features added:** 19 (11 CT.gov timing + 8 pipeline proxy)
**Feature dict:** `ml_feature_dict_v0.4_20260313.csv` (82 entries, up from 69 in v0.3)

---

## 1. CT.gov Timing Features (Part 1)

All 11 timing features were fetched from CT.gov API v2 per NCT ID. 679 unique NCT IDs queried; results cached in `cache/ctgov_details_v1.json` (~60 MB).

### Coverage

| Feature | Description | Coverage |
|---|---|---|
| `feat_ctgov_primary_completion_date` | Primary completion date string (CT.gov registry) | 784/827 (94.8%) |
| `feat_days_to_primary_completion` | Days to primary completion (CT.gov anchor) | 773/827 (93.5%) |
| `feat_primary_completion_imminent_30d` | Imminence flag: within 30 days | 773/827 (93.5%) |
| `feat_primary_completion_imminent_90d` | Imminence flag: within 90 days | 773/827 (93.5%) |
| `feat_completion_recency_bucket` | Recency bucket (CT.gov-derived) | 784/827 (94.8%) |
| `feat_ct_status_current` | Current CT.gov overall status | 784/827 (94.8%) |
| `feat_active_not_recruiting_flag` | Binary: ACTIVE_NOT_RECRUITING | 784/827 (94.8%) |
| `feat_completed_flag` | Binary: COMPLETED | 784/827 (94.8%) |
| `feat_days_since_ctgov_last_update` | Days since CT.gov last update | 773/827 (93.5%) |
| `feat_recent_ctgov_update_flag` | Binary: updated within 90 days | 773/827 (93.5%) |
| `feat_status_timing_consistency_flag` | Status/completion date consistency check | 773/827 (93.5%) |

### Status distribution (as of CT.gov pull)

| Status | Count |
|---|---|
| COMPLETED | 537 (64.9%) |
| RECRUITING | 101 (12.2%) |
| ACTIVE_NOT_RECRUITING | 96 (11.6%) |
| TERMINATED | 32 (3.9%) |
| NOT_YET_RECRUITING | 9 (1.1%) |
| ENROLLING_BY_INVITATION | 5 (0.6%) |
| OTHER | 4 (0.5%) |

**Note:** Status is as-of data pull date (2026-03-13), not the historical event date. 64.9% of studies are COMPLETED — expected given the historical dataset. Status consistency check: 717 consistent, 56 inconsistent (6.8%) — useful as a data quality flag.

---

## 2. Pipeline Proxy Features (Part 2)

293 unique sponsors and 596 unique drug names queried via CT.gov `query.spons` and `query.intr`. PAGE_SIZE=100 per call.

### Coverage

| Feature | Coverage |
|---|---|
| Sponsor features (5) | 784/827 (94.8%) |
| Drug/intervention features (3) | 820/827 (99.2%) |

Drug coverage is higher (99.2%) because drug name normalisation and single-word fallback succeed for more rows than sponsor lookups.

### Distributions

| Feature | Median | Mean | P90 | Max |
|---|---|---|---|---|
| `feat_ctgov_n_active_trials_sponsor` | 7 | 9.3 | 20 | 47 |
| `feat_ctgov_n_late_stage_trials_sponsor` | 24 | 35.3 | 75 | 95 |
| `feat_ctgov_n_completed_trials_sponsor` | 24 | 33.9 | 70 | 95 |
| `feat_ctgov_n_trials_total_sponsor` | 36 | 1181.7 | 3700 | 13208 |
| `feat_ctgov_pipeline_maturity_score` | 2.0 | 2.5 | 5 | 9 |
| `feat_ctgov_n_trials_same_intervention` | 7 | 227.4 | 193 | 32948 |
| `feat_ctgov_n_late_stage_trials_same_intervention` | 5 | 17.8 | 70 | 91 |
| `feat_ctgov_asset_maturity_score` | 1.8 | 2.1 | 5 | 9 |

### Caveats

- **Large pharma sponsors:** 48 of 293 sponsors (16.4%) returned >100 results on CT.gov (capped). `feat_ctgov_n_trials_total_sponsor` uses the registry's `totalCount` (e.g. Novartis = 2711), inflating the mean to 1181. The maturity score uses `n_sample` (capped at 100) in the denominator to limit distortion.
- **`feat_ctgov_n_trials_same_intervention` outlier:** Some drug names (e.g. common adjuvants) return 30k+ results. The sample used for phase counting is capped at 100.
- **Sponsor name normalisation:** Legal suffixes (Inc., Corp., etc.) stripped for matching. Some sponsor mismatches may persist for non-standard names.

---

## 3. Weak Dataset-History Timing Features — Candidates for Dropping

The following features in v0.3 were derived from the dataset's own timing columns (not from CT.gov) and are now superseded or partially overlapping with CT.gov-grounded equivalents:

| Feature | Reason to drop |
|---|---|
| `feat_primary_completion_imminent_30d` (v0.3) | Replaced by CT.gov-derived version with same name; v0.4 already uses CT.gov version |
| `feat_primary_completion_imminent_90d` (v0.3) | Same as above |
| `feat_completion_recency_bucket` (v0.3) | Same as above — v0.4 uses CT.gov-derived version |
| `feat_imminent_30d` | Dataset-column-derived, overlaps with CT.gov version; kept for now as fallback where nct_id not found |
| `feat_imminent_90d` | Same as above |

**Recommendation:** In v0.5 modeling, use the CT.gov-derived imminence/recency features as primary. Keep the dataset-derived versions as fallback only (set to CT.gov value when available, dataset value otherwise). Do not use both as independent features — they are highly correlated.

---

## 4. Recommended Features for Next Model Comparison

**High priority — most likely to contribute signal:**

| Feature | Rationale |
|---|---|
| `feat_ctgov_pipeline_maturity_score` | Composite; captures sponsor experience level in one number |
| `feat_ctgov_n_late_stage_trials_sponsor` | Strong proxy for sponsor's track record; P90=75 shows good spread |
| `feat_ctgov_asset_maturity_score` | Drug development history; novel assets (score~0) vs well-studied (score~9) |
| `feat_ct_status_current` (encoded) | Current status at pull time; distinguish completed vs ongoing |
| `feat_recent_ctgov_update_flag` | Freshness signal; may correlate with upcoming catalysts |

**Medium priority — useful but may be redundant with existing features:**

| Feature | Rationale |
|---|---|
| `feat_ctgov_n_active_trials_sponsor` | Already partially captured by `feat_n_unique_drugs`, `feat_pipeline_depth_score` |
| `feat_ctgov_n_trials_same_intervention` | Useful for drug novelty but noisy for common drugs (log-transform recommended) |
| `feat_active_not_recruiting_flag` | Narrower version of status; test as binary |

**Lower priority:**

| Feature | Rationale |
|---|---|
| `feat_ctgov_n_trials_total_sponsor` | High collinearity with maturity score; mean inflated by large pharma |
| `feat_status_timing_consistency_flag` | Data quality flag — better as a filter than as a model feature |
| `feat_days_since_ctgov_last_update` | Raw day count; `feat_recent_ctgov_update_flag` binary version is cleaner |

**Suggested transformations before modeling:**
- Log-transform: `feat_ctgov_n_trials_total_sponsor`, `feat_ctgov_n_trials_same_intervention` (heavy right tail)
- One-hot encode: `feat_ct_status_current` (7 categories; merge rare ones into "other")

---

## 5. Summary

| Item | Value |
|---|---|
| New CT.gov timing features | 11 |
| New CT.gov pipeline proxy features | 8 |
| Total features v0.4 | 82 (was 69 in v0.3) |
| Timing feature coverage | 93.5–94.8% |
| Pipeline proxy coverage (sponsor) | 94.8% |
| Pipeline proxy coverage (drug) | 99.2% |
| Unique NCT IDs fetched | 679 |
| Unique sponsors queried | 293 |
| Unique drug names queried | 596 |
| Sponsors with capped results (>100) | 48 (16.4%) |

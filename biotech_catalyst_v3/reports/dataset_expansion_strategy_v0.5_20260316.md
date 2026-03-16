# Dataset Expansion Strategy — Pre-Event Model

**Date:** 2026-03-16
**Current master:** `enriched_all_clinical_clean_v2.csv` — 862 rows, 2007–2026
**Current ML training set:** 813 rows (after filter), 86 features

---

## Current state assessment

**Year distribution problem:**

| Period | Rows | % |
|---|---|---|
| 2007–2019 | 43 | 5.0% |
| 2020–2022 | 74 | 8.6% |
| 2023–2026 | 734 | 85.2% |

85% of the dataset comes from the last 3 years. The model has seen very few market regimes, rate environments, and company cohorts. This limits generalization severely.

**Class distribution:**

| Class | Count | % |
|---|---|---|
| Noise | 563 | 65.3% |
| Low | 54 | 6.3% |
| Medium | 60 | 7.0% |
| High | 82 | 9.5% |
| Extreme | 92 | 10.7% |

Target class (High + Extreme) = 174 positive examples (20.2%). Reasonable positive rate but small absolute count.

**Key bottleneck:** The model is underfit for events from 2020–2022 (sparse) and has almost no data from earlier years. Temporal diversity is the main limitation for generalization — not raw row count or class balance.

---

## Expansion path assessment

### A — Historical extension backward (2020–2022)

**Mechanism:** Re-run CT.gov-first event collection for Phase 2/3 COMPLETED trials 2020-01-01 to 2022-12-31. Map sponsor → ticker via existing universe. Compute stock moves + ATR using existing utils. Same pipeline, different time window.

| Dimension | Assessment |
|---|---|
| Model value | **Very high** — adds temporal diversity, different macro regimes (COVID recovery, rate hikes), different company cohort |
| Implementation difficulty | **Low** — `find_clinical_events.py` already exists; just needs date range change + master schema integration |
| Expected data gain | +200–400 new rows (rough: ~800 Phase 2/3 completions/year × 20% ticker match = 160–250/year × 2 years) |
| Recommendation | **Priority 1 — implement now** |

**Note:** CT.gov primary_completion_date is used as the event date proxy. For oncology, this may lag the actual readout. This introduces noise on individual rows but is acceptable at the dataset scale.

### B — Better event-first collection (CT.gov-grounded)

**Mechanism:** Already the default approach via `find_clinical_events.py`. Current dataset is built CT.gov-first. No structural change needed — this is Path A applied historically.

| Dimension | Assessment |
|---|---|
| Model value | Already implemented |
| Recommendation | **Continue as-is (same as A)** |

### C — Hard-negative expansion

**Mechanism:** Collect Phase 2/3 COMPLETED trials where the move at completion was definitively small (<1.5× ATR). Specifically target confirmed "no event" rows — trials that completed quietly with no announcement.

| Dimension | Assessment |
|---|---|
| Model value | **Medium** — helps model distinguish "this drug/sponsor/design pattern tends not to move" |
| Implementation difficulty | **Medium** — requires additional filtering to avoid including already-covered events |
| Expected data gain | +100–200 rows of clean low-ATR events |
| Recommendation | **Priority 3 — after historical extension** |

**Note:** The historical extension (Path A) will naturally include many Noise-class rows (CT.gov completions with no announcement). This partially covers Path C without extra work.

### D — More medium-move examples

**Mechanism:** Specifically target events with 3–5× ATR moves (the boundary between Medium and High). These are the hardest to classify correctly.

| Dimension | Assessment |
|---|---|
| Model value | **Medium** — better calibration around the boundary |
| Implementation difficulty | **High** — no clean CT.gov filter for move magnitude; requires measuring first and then filtering |
| Expected data gain | +30–50 medium-class rows |
| Recommendation | **Priority 4 — low priority; medium rows will appear naturally in Path A** |

### E — More oncology and late-stage

**Mechanism:** Explicitly filter CT.gov for oncology Phase 3+ trials.

| Dimension | Assessment |
|---|---|
| Model value | **Low-Medium** — oncology is already ~28% of dataset; the timing mismatch caveat makes these rows noisier anyway |
| Implementation difficulty | **Low** — add filter to Path A |
| Recommendation | **Include in Path A naturally; no separate pass needed** |

---

## Decision

**Implement Path A: historical extension 2020–2022.**

Script: `scripts/expand_historical_events.py`

This is the single best value-for-effort expansion:
- Uses existing infrastructure (CT.gov API, universe file, ohlc_cache, volatility utils)
- Adds temporal diversity across different market environments
- Includes natural distribution of Noise/Low/Medium/High rows (unlike pure "large move" collection)
- No manual validation needed for initial pass (same quality tier as existing CT.gov-derived rows)

---

## Post-expansion pipeline

After the master CSV is expanded, the feature pipeline must be re-run:

```bash
# 1. Re-run feature generation on the expanded master
python -m scripts.prepare_ml_dataset --input enriched_all_clinical_clean_v3.csv

# 2. Run feature passes in order
python -m scripts.add_high_signal_features
python -m scripts.refresh_ctgov_features
python -m scripts.build_ctgov_pipeline_proxies
python -m scripts.add_pre_event_timing_features
python -m scripts.add_oncology_timing_interactions

# 3. Rebuild train table and retrain
python -m scripts.build_pre_event_train_v2
python -m scripts.train_pre_event_v3
```

This is the correct next step after the expansion pass.

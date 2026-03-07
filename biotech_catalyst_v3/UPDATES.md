# Dataset Updates Log

> Written for subject matter experts and collaborators.
> Focus: **what changed in the data**, what the output files now contain, and what it means.
> Updated with every meaningful data change.

---

## Current Primary File

**`enriched_all_clinical_validated.csv`** — use this for analysis
- 2,175 rows × 52 columns + 11 validation columns (v_*)
- All rows are Clinical Data events (clinical trial readouts, drug results, endpoint announcements)
- Covers Jan 2023 – Dec 2025 across 460 biotech tickers

---

## Update History (newest first)

---

### 2026-03-07 — False Positive Interpretation & ML Strategy

**What false positives actually mean:**

A false positive row says: "On this date, this stock moved because of clinical trial data." But when we verify it, there was no clinical news that day. What the row actually contains is **a random trading day** — the stock's normal background price fluctuation, with no event behind it.

This is a critical distinction for ML. The noise class is supposed to represent: *"a real clinical readout happened but the market barely reacted."* That is a meaningful, learnable signal — a drug that missed endpoints, a trial in an already-crowded indication, results in line with consensus expectations. A model trained on those rows learns what separates a consequential trial from a non-consequential one.

But a false positive row represents something entirely different: *no event occurred at all.* It's just a quiet Tuesday. Training on it as though it were a "clinical event with no market reaction" teaches the model a corrupted lesson.

**Why so many false positives exist:**

The dataset was built largely from ClinicalTrials.gov "completion dates." CT.gov records when a trial administratively closes — but this does not always correspond to a public data announcement. A trial can mark itself complete months before results are ever published or presented at a conference. In those cases, no press release exists, nothing moved the market, and the row is noise in the truest sense.

**The ML problem if we don't fix this:**

| Scenario | Effect on model |
|----------|----------------|
| Train on false positives as "Low move" class | Model learns that random market days = no reaction to clinical data. Completely wrong. |
| ~940 false positives in 2,175 rows | ~43% of the entire dataset may be bad labels |
| False positives concentrated in the "Low" / "Noise" class | Negative class is severely corrupted; positive class (High moves) is largely clean |
| Model trained on corrupted data | Will likely fail to distinguish real muted reactions from noise — low precision on the Low class |

**What we should do — recommended approach:**

**Option A (Recommended): Remove false positives, keep confirmed real events**
Run the full validation on all 1,571 noise rows. Remove any row where `v_action = FLAG_FALSE_POSITIVE`. Keep rows where `v_action = OK` (confirmed real event, no market reaction) and `v_action = DATE_FIXED` (real event, corrected date).

Result: a dataset where every row corresponds to a confirmed public clinical data announcement. The model learns to predict market reaction *given that something actually happened* — which is the right question to ask.

Estimated clean dataset size after full validation (extrapolating from 60% FP rate):
- ~630 confirmed real noise rows (40% of 1,571)
- ~375 high-move rows (largely already verified)
- ~1,005 total rows — smaller but trustworthy

**Option B: Keep false positives, relabel them as "control days"**
Instead of treating them as "clinical events with no reaction," relabel them as a third class: random non-event trading days. This reframes the model question to: *"Is this even a real clinical event at all?"* — a useful pre-filter question but different from the original goal.

**Option C (not recommended): Use only high-move rows**
The 375 high-move events are well-verified (large moves are easy to confirm — something clearly happened). You could use just those as the positive class with no negative class, for a ranking/anomaly task. But this loses the ability to train a binary classifier.

**Action items:**
1. Run full validation: `python -m scripts.validate_catalysts` (~40 min for 1,571 rows)
2. Run fix script: `python -m scripts.fix_validated_rows --input enriched_all_clinical_validated.csv --remove-false-positives`
3. Re-export clean dataset with `data_complete = True` and `v_action` not `FLAG_FALSE_POSITIVE`

**Filter for ML-ready clean dataset (after full validation):**
```
data_complete == True
AND v_action in [OK, DATE_FIXED]  ← confirmed real events only
AND move_class_combo != "Medium"  ← clear High vs Low label only
```

---

### 2026-03-06 — Noise Row Validation (v3.4)

**File affected:** `enriched_all_clinical_validated.csv` (new file)

**What happened:**
72% of rows in the dataset have very small stock moves (< 1.5× the stock's normal daily volatility). We call these "noise" rows. The concern was: are these real clinical data events that just didn't move the market, or are they fabricated — the AI incorrectly attributed a clinical catalyst to a day when nothing happened?

We ran a verification pass on 5 rows as a test. For each row, an AI (Perplexity) was asked to find a press release or news source confirming the clinical event on that date. It then fetched the actual press release page and extracted the date, title, and key information.

**Test results (5 rows):**
- 3 rows = **False positives** — no clinical news found on that date (SNY, AZN, KPTI)
- 2 rows = **Wrong date** — real event found but attributed to wrong date (IMRX: off by 9 months; CYTK: off by 4 days)
- 0 rows = confirmed correct

**Bottom line:** ~60% false positive rate in the noise class (preliminary, 5-row sample). This means a significant portion of the 1,571 noise rows are likely not real clinical data events and should be excluded from ML training.

**New columns added to every row (after validation is run):**

| Column | What it tells you |
|--------|------------------|
| `v_action` | What to do: `OK` (verified) · `FIX_DATE` (real event, wrong date) · `FLAG_FALSE_POSITIVE` (not real) · `FLAG_ERROR` (couldn't check) |
| `v_pr_title` | Title of the press release found |
| `v_pr_date` | Date on the press release |
| `v_pr_key_info` | Key excerpt from the press release |
| `v_pr_link` | URL of the press release |
| `v_summary` | AI explanation of why it was verified or rejected |
| `v_confidence` | AI confidence: High / Medium / Low |

**Status:** Full validation of all 1,571 noise rows still pending (~40 min run). Only 5 rows have v_* columns filled so far. Run `python -m scripts.validate_catalysts` to complete.

---

### 2026-03-05 — Column Cleanup + Drug/Indication Fill (v3.3 post-processing)

**File affected:** `enriched_all_clinical.csv`

**What changed in the data:**

| Field | Before | After |
|-------|--------|-------|
| `drug_name` fill rate | 22% | 99% |
| `indication` fill rate | 5% | 97% |
| `ct_phase` format | Mixed (`PHASE3`, `Phase3`, `phase 3`) | Standardized (`Phase 3`, `Phase 1/2`, etc.) |
| Column count | 61 columns | 41 columns |
| `price_at_event` fill | ~74% | ~93% (400 more rows filled) |

**How:** `drug_name` and `indication` were derived from the ClinicalTrials.gov fields already in the data (`ct_conditions`, trial title) — no new API calls needed. 20 redundant or empty columns were dropped.

**Column renamed:** `normalized_move` → `stock_movement_atr_normalized` (same data, clearer name)

**What this means for analysis:** Drug name and indication are now available for nearly all rows, making it possible to filter by disease area or drug type without gaps.

---

### 2026-03-04 — Data Quality Fixes + NCT-Complete Subset

**Files affected:** `enriched_all_clinical.csv`, `enriched_all_clinical_with_nct.csv` (new)

**What changed:**

- **Dropped 4 empty rows** (BPMC ×3, SAGE ×1) — these had no move percentage and no financial data; effectively blank entries
- **Added `data_complete` column** — True/False flag indicating whether a row has everything needed for ML:
  - Requires: `move_pct`, `market_cap_m`, `current_price`, `atr_pct` all present
  - Result: **2,139 / 2,175 rows (98.3%) are ML-ready**
  - Remaining 36 incomplete: 27 missing move_pct (CT.gov completions where yfinance had no data), 3 missing market cap (network timeouts on AZN/NRC), 6 missing ATR only
- **Fixed `event_trading_date`** — 536 rows were missing this. Filled from `event_date`; weekend events advanced to next Monday.
- **Cleared false error messages** — 13 rows had "Financial fetch error" text despite having valid financial data. Cleared.

**New file — `enriched_all_clinical_with_nct.csv`:**
- Subset of 1,778 rows that have a confirmed ClinicalTrials.gov NCT ID
- 81.6% of the full dataset
- Use this if your analysis requires a specific trial reference (e.g. to look up trial details, enrollment, sponsor)
- `event_trading_date` is 100% filled in this file

---

### 2026-03-04 — ATR Volatility Classification Overhaul

**Files affected:** All 6 enriched CSVs

**What changed:**
The move classification system was rebuilt. Every event is now classified on three scales:

| Column | What it measures | Categories |
|--------|-----------------|------------|
| `move_class_norm` | Move vs. the stock's own daily volatility (ATR-normalized) | Noise · Low · Medium · High · Extreme |
| `move_class_abs` | Raw % move, regardless of stock volatility | VeryLow · Low · Medium · High · VeryHigh |
| `move_class_combo` | **Primary ML label** — must be significant on BOTH scales | Low · Medium · High |

**ATR thresholds (normalized):**
- **Noise** — move < 1.5× normal daily swing (stock barely reacted)
- **Low** — 1.5–3× (noticeable but not dramatic)
- **Medium** — 3–5× (clearly event-driven)
- **High** — 5–8× (major reaction)
- **Extreme** — ≥8× (exceptional, very rare)

**Why this matters:** A 5% move on a volatile small-cap (ATR = 8%) is noise. A 5% move on a stable large-cap (ATR = 1%) is a big deal. The ATR-normalized column accounts for this. Use `move_class_norm` for most analyses.

**ATR methodology:** Wilder's 20-day smoothed average (same as TradingView). Computed using only the 20 trading days *before* the event — no look-ahead bias.

**Distribution in current dataset:**
- Noise: 1,571 rows (72%)
- Low: 231 rows (11%)
- Medium: 144 rows (7%)
- High: 108 rows (5%)
- Extreme: 121 rows (6%)

---

### 2026-03-01 — Dataset Expansion: 104 → 2,175 events (v3.3)

**File affected:** `enriched_all_clinical.csv` (rebuilt from scratch)

**What changed:**
The dataset grew 21× in size. Previously it only captured events where the stock moved ≥30% — by definition excluding all cases where clinical data was released but the market barely reacted (the "noise" cases). This made it impossible to train an ML model to predict *whether* a move would happen.

**New approach:**
1. Expanded the biotech universe from ~111 tickers → **460 tickers** ($50M–$10B market cap)
2. Pulled all Phase 2/3 trial completion dates from ClinicalTrials.gov and matched each to the stock's price move on that date — regardless of move size
3. Also kept the high-move scan (≥10%) as a second source to capture events ClinicalTrials.gov missed

**Result:**

| Metric | Before | After |
|--------|--------|-------|
| Events | 104 | 2,175 |
| Tickers | ~111 | 460 |
| Low-move events | 105 | 1,602 |
| High-move events | 105 | 375 |
| Gainers / Losers | Skewed high | 1,121 / 1,054 (balanced) |
| Market cap coverage | 98% | 99.7% |
| Date range | 2024–2025 | Jan 2023 – Dec 2025 |

**Columns in this dataset:** ticker, event_date, event_trading_date, move_pct, price_at_event, price_before, price_after, move_2d_pct, event_type (Gainer/Loser), catalyst_type, catalyst_summary, drug_name, indication, nct_id, ct_phase, ct_enrollment, ct_conditions, ct_sponsor, ct_status, ct_allocation, ct_primary_completion, atr_pct, avg_daily_move, stock_movement_atr_normalized, move_class_norm, move_class_abs, move_class_combo, market_cap_m, current_price, cash_position_m, short_percent, institutional_ownership, analyst_target, analyst_rating, is_pivotal, primary_endpoint_met, primary_endpoint_result, data_complete, press_release_url

---

### 2026-02-17 — Initial Dataset: 210 balanced events (v3.2)

**Files:** `enriched_high_moves.csv` (265 events), `ml_dataset_clinical.csv` (210 events)

Starting point. 265 high-move events (≥30% moves) confirmed as Clinical Data catalysts, enriched with trial details and financials. Balanced ML dataset of 105 high-move + 105 low-move events.

---

## Key Files — Quick Reference

| File | Rows | Use for |
|------|------|---------|
| `enriched_all_clinical_validated.csv` | 2,175 | **Primary** — includes validation flags |
| `enriched_all_clinical.csv` | 2,175 | Same data, without v_* columns |
| `enriched_all_clinical_with_nct.csv` | 1,778 | Analysis requiring confirmed trial ID |
| `enriched_high_moves.csv` | 265 | High-move-only subset (≥30% moves) |
| `ml_dataset_clinical.csv` | 210 | Original balanced ML dataset (v3.2) |
| `biotech_universe_expanded.csv` | 460 | The 460 biotech tickers tracked |

## Recommended Filters for Clean ML Use

```
data_complete == True          → keeps 2,139 rows (98.3%)
move_class_combo != "Medium"   → removes ambiguous cases (use Low vs High as binary label)
v_action not in [FLAG_FALSE_POSITIVE, FLAG_ERROR]  → removes suspected hallucinations (after full validation)
```

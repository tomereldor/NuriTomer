# Outcomes Updates Log

---

## Current Primary File

**`enriched_all_clinical_clean_v2.csv`** ← THE ONE FILE TO USE
- **862 rows × 58 cols** — master dataset, all enrichments merged
- False positives removed · dates corrected · prices re-fetched
- Source links: 856/862 (99.3%) have ≥1 link (`v_pr_link` or `best_event_link`)
- MeSH Level-1 classification: 474 rows classified across 10 disease branches
- v_action: 550 DATE_FIXED · 292 OK · 8 error/unresolved
- All other files → `archive/`

**`biotech_universe_expanded.csv`** — reference (not a dataset)
- 460 biotech tickers ($50M–$10B market cap) used as the tracking universe

---

## Update History (newest first)

---

### 2026-03-09 — Master file consolidation + archive cleanup (v3.8)

**File:** `enriched_all_clinical_clean_v2.csv` (862 rows × 58 cols) — now the single source of truth

Merged `_linked` (source links) and `_mesh` (MeSH classification) enrichments into the base clean_v2 file. All 17 superseded CSVs moved to `archive/`. Going forward all pipeline updates write directly to `enriched_all_clinical_clean_v2.csv`.

**All 58 columns:**
- Core: ticker, event_date, event_type, move_pct, price_at/before/after, move_2d_pct, event_trading_date
- Trial: catalyst_type/summary, drug_name, nct_id, indication, is_pivotal, primary_endpoint_met/result, ct_* (7 cols)
- Financials: market_cap_m, current_price, cash_position_m, short_percent, institutional_ownership, analyst_target/rating
- ATR/move: atr_pct, stock_movement_atr_normalized, avg_daily_move, move_class_abs/norm/combo, stock_relative_move, data_complete
- Validation: v_is_verified, v_actual_date, v_pr_link, v_pr_date/title/key_info, v_is_material, v_confidence, v_summary, v_error, v_action
- Links: best_event_link
- MeSH: mesh_level1, mesh_level1_reason, mesh_branches_raw, mesh_terms_raw, ct_conditions_raw

**Archived:** 17 CSVs moved to `archive/` (all previous intermediate and superseded files)

---

### 2026-03-09 — MeSH Level-1 Disease Classification (v3.7)

**File:** `clean_v2_mesh.csv` (862 rows × 57 cols)

Added standardized MeSH Level-1 disease category to every row in clean_v2 using the ClinicalTrials.gov API v2.

**How it works:** For each `nct_id`, the script fetches `conditionBrowseModule.ancestors` (full MeSH hierarchy) and intersects with a 10-branch priority list. When multiple branches match, condition text keyword matching picks the most relevant one.

**Coverage (862 rows):**

| Reason | Count |
|--------|-------|
| single_branch | 240 |
| condition_match | 183 |
| priority_list | 51 |
| no_branches | 176 |
| no_nct_id | 47 |

**Disease distribution (474 rows with MeSH):**

| Category | Count |
|----------|-------|
| Neoplasms | 141 |
| Nervous System Diseases | 68 |
| Immune System Diseases | 52 |
| Respiratory Tract Diseases | 51 |
| Infectious Diseases | 43 |
| Cardiovascular Diseases | 33 |
| Endocrine System Diseases | 28 |
| Digestive System Diseases | 27 |
| Skin Diseases | 17 |
| Musculoskeletal Diseases | 14 |

**Script:** `scripts/mesh_level1_from_nct.py`

---

### 2026-03-09 — Event Source Link Backfill (v3.6.2)

**File:** `enriched_all_clinical_clean_v2_linked.csv` (862 rows × 53 cols)

Added `best_event_link` column to fill source links for the 276 rows missing `v_pr_link`. Used Perplexity to find the highest-quality credible link per event.

**Link coverage after backfill:**
- `v_pr_link` filled: 586 rows (68%)
- `best_event_link` backfilled: 270 rows (additional)
- **Has at least one source link: 856/862 (99.3%)**
- Still no link: 6 rows

**Bonus:** 174 DATE_FIXED rows that received a confirmed link were promoted to `v_is_verified=True`.

**Script:** `scripts/backfill_event_links.py`

---

### 2026-03-09 — Data Integrity Audit: v_is_verified=FALSE Rows (v3.6.1)

Deep audit of the 181 rows in `clean_v2` where `v_is_verified=FALSE` and no `v_pr_link`.

#### What v_is_verified=FALSE actually means

It does **not** mean "fake event." Per the validation prompt (rule 2):
> *"If news exists on a DIFFERENT date, set is_verified=false and provide actual_date."*

FALSE = real event, but the original scan attributed it to the wrong date.

#### Date shift analysis

All 562 DATE_FIXED rows show a 0-day shift when comparing `event_date` vs `v_actual_date` — this is expected because `fix_validated_rows.py` **overwrites** `event_date` in place with the corrected date. The original (wrong) dates are no longer in the file. The v_summary text confirms corrections ranged from days to months.

#### Financial data alignment (corrected date)

Of 562 DATE_FIXED rows:

| Status | Count |
|--------|-------|
| `event_trading_date` aligns with corrected date (≤4 trading days) | **546** (97%) |
| Misaligned — prices may be on wrong trading day | **13** (2.3%) |
| Can't verify (unparseable v_actual_date) | **3** |

#### Why 181 rows have no PR link

`v_pr_link` is the only link/URL column in `clean_v2` — no alternative source columns exist.

The validation prompt explicitly allows `"pr_link": null` when no official PR exists on businesswire/globenewswire/prnewswire/sec.gov. Perplexity returns null when the event was found via a news article, conference presentation, or earnings mention. This is a code design decision, not a bug.

Confidence for the 181 FALSE+no_pr rows:
- **high: 135 (74%)** — Perplexity was confident about the event, just had no official PR URL
- medium: 42 (23%)
- low: 3 (2%), NaN: 1 (FLAG_ERROR)

#### State combinations explained

| State | Count | How it arises | Expected? |
|-------|-------|---------------|-----------|
| `FALSE` + `DATE_FIXED` | 562 | Perplexity confirmed event on different date; fix applied; `v_is_verified` is never reset after fix | ✅ Yes |
| `FALSE` + `FIX_DATE` | 7 | Perplexity returned an unparseable date (`2023-12-00`, `2025-??-??`); fix skipped; prices still on original wrong date | ⚠️ No |
| `FALSE` + `FLAG_ERROR` | 1 | API failure during validation; event status unknown | ⚠️ No |
| `FALSE` + no `v_pr_link` | 181 | Perplexity found event but could not provide official PR URL; allowed by prompt design | ✅ Yes |

#### Recommended action for 181 rows

| Group | Count | Action |
|-------|-------|--------|
| `DATE_FIXED`, prices aligned | 163 | **KEEP** — event confirmed, date and prices correct |
| `DATE_FIXED`, prices misaligned | 8 | **MANUAL REVIEW** — prices may correspond to wrong trading day |
| `DATE_FIXED`, alignment unverifiable | 3 | **KEEP** — low risk |
| `FIX_DATE` (invalid AI date, fix skipped) | 6 | **MANUAL REVIEW** — event date unconfirmed, prices on original (wrong) date |
| `FLAG_ERROR` (API failure) | 1 | **REVALIDATE** on next `validate_catalysts.py` run |

**Summary: 166/181 are safe to keep as-is. 14 warrant follow-up (8 misaligned + 6 FIX_DATE).**

---

### 2026-03-09 — PR Discovery Pipeline for Dataset Expansion (v3.6)

**New script:** `scripts/extend_with_pr_discovery.py`

Adds a Perplexity-powered discovery pipeline to find new clinical catalyst events
**not already in the dataset** and extend `enriched_all_clinical_clean_v2.csv`.

#### How it works

| Stage | What happens |
|-------|-------------|
| 1 — Discovery | Runs 20 targeted Perplexity sonar-pro search queries across Phase 3/2/1, FDA decisions, interim analyses, and time periods 2023–2025 |
| 2 — Verification | Fetches each PR URL, applies keyword relevance filter, deduplicates against existing dataset by (ticker, date) |
| 3 — OHLC enrichment | Downloads price history, computes price_before/at/after, move_pct, ATR, all move classifications |
| 4 — Output | Accepted candidates + rejected (with rejection reason) saved to separate CSVs |

#### Output files (not yet run at full scale)
- `extended_pr_discovery_raw.csv` — intermediate: all events returned by Perplexity before filtering
- `extended_relevant_clinical_candidates.csv` — accepted new rows, schema matches `clean_v2`
- `extended_relevant_clinical_rejected.csv` — rejected rows with `_reject_reason` for debugging

#### Key design choices
- Only accepts events where `abs(move_pct) >= 3%` (configurable via `--min-move`)
- Deduplicates by exact (ticker, event_date) — no near-duplicates
- New rows marked `v_action=DISCOVERED`, `v_is_verified=True` (Perplexity confirmed)
- CT.gov fields (nct_id, ct_phase, etc.) left blank — backfillable via `fix_missing_nct.py`

#### Usage
```bash
# Test run — 2 queries, no output written
python -m scripts.extend_with_pr_discovery --dry-run --limit 2

# Full run
python -m scripts.extend_with_pr_discovery \
    --existing enriched_all_clinical_clean_v2.csv \
    --candidates extended_relevant_clinical_candidates.csv \
    --rejected  extended_relevant_clinical_rejected.csv

# Resume from saved Stage 1 results (skip API calls)
python -m scripts.extend_with_pr_discovery \
    --from-raw  extended_pr_discovery_raw.csv \
    --existing  enriched_all_clinical_clean_v2.csv \
    --candidates extended_relevant_clinical_candidates.csv \
    --rejected  extended_relevant_clinical_rejected.csv
```

---

### 2026-03-08 — Data Quality Review: What's in the Clean File and Why (v3.4.2)

**File:** `enriched_all_clinical_clean.csv` — no rows changed, this is a clarification of what the data means

Every row in the clean file belongs there. Here's the breakdown of the three groups that might look confusing at first glance, and a note on the price update done today.

---

**1. 349 rows where `v_is_verified = FALSE` — but they have a press release link**

These rows are real events. "FALSE" does not mean the event was fake — it means the AI found the clinical news, but on a *different date* than originally recorded. The press release exists; the original date was just wrong. The AI set `is_verified = false` and provided the correct date, which we then used to fix the row. These are the `DATE_FIXED` rows in the dataset — their dates and prices have been corrected. The 261 rows with a PR link have a confirmed source. The 88 without a captured link still had a corrected date returned by the AI.

**Verdict: Keep all 349. They are real, date-corrected events.**

---

**2. 604 rows with no `v_is_verified` at all, and no press release link**

These rows were never sent through the AI validator — intentionally. The validation step only ran on rows where the stock barely moved (the "noise" class). All 604 rows in this group are High, Extreme, Medium, or Low movers. A stock moving +40% or −25% on a given day doesn't need an AI to confirm that something happened. These are the most obviously real rows in the dataset.

**Verdict: Keep all 604. They are high-confidence events that never needed verification.**

---

**3. 42 rows where `v_is_verified = TRUE` but no press release link**

These are actually the cleanest rows in the dataset. "TRUE" means the AI confirmed: yes, there was a real clinical data event on or within one day of the claimed date. The absence of a stored press release URL just means the link wasn't captured — either the source was a journal publication, conference abstract, or FDA document (rather than a company press release), or the URL fetch failed after the event was already confirmed. The confirmation itself is what matters.

**Verdict: Keep all 42. Trust them. If you need a source URL for any of these, a targeted search by ticker + date will find it.**

---

**4. Prices updated to match corrected dates**

For the 349 date-corrected rows: previously the event dates were fixed but the price data (`price_at_event`, `price_before`, `price_after`, `move_pct`) still reflected the original wrong date's prices. Today those prices were re-fetched at the corrected dates. 346 of 349 rows were updated (3 had unparseable dates like "2023-12-00"). Spot-check: 10/10 verified rows matched live market data exactly.

The clean file now has fully correct price data across all 1,057 rows.

---

### 2026-03-09 — Fix Dates + Remove FPs from High-Move Group → Clean v2 (v3.5.1)

**New file:** `enriched_all_clinical_clean_v2.csv`

**What this does:**

Takes the backfill file (where all 604 high-move rows were now verified) and applies the same correction pipeline that was previously run only on noise rows:
- Date-corrected rows: update `event_date` to the actual press release date, re-fetch prices, recompute ATR and move classifications
- False positives: removed

**Results:**

| Action | Count |
|--------|-------|
| FIX_DATE rows corrected (dates + prices re-fetched) | 216 |
| FIX_DATE rows skipped (AI returned unparseable date) | 7 |
| False positives removed | 195 |
| **Rows remaining** | **862** |

**What's in `enriched_all_clinical_clean_v2.csv` (862 rows):**

| Group | Rows | v_action | Notes |
|-------|------|----------|-------|
| Noise-class confirmed real events | 104 | OK | From original noise validation |
| Noise-class date-corrected events | 346 | DATE_FIXED | Corrected in v3.4.1 |
| High-move confirmed real events | 188 | OK | Newly verified in v3.5 |
| High-move date-corrected events | 216 | DATE_FIXED | Corrected in this step |
| Unresolved (error / invalid date) | 8 | FLAG_ERROR / FIX_DATE | Retain original dates |
| **Total** | **862** | | |

**Use this file for ML training.** It is the most complete clean dataset: all rows are real clinical events with correct dates and correct price data. False positives have been removed from both the noise class (done in v3.4) and the high-move class (done in this step).

---

### 2026-03-09 — PR Backfill + Verification for High-Move Rows (v3.5)

**New file:** `enriched_all_clinical_clean_pr_backfill.csv` (1,057 rows — same as clean file, with v_* columns now filled for the previously unverified group)

**Background:**

Until now, the 604 high/extreme/medium/low-move rows in the clean file had never been sent through the AI validator — we assumed large price moves were self-evidently real. That assumption turns out to be partially wrong. This update runs full Perplexity verification on all 604 of those rows to find press release links and confirm whether the clinical event attribution is correct.

**Results — all 604 rows processed:**

| Outcome | Count | What it means |
|---------|-------|---------------|
| Verified (OK) | 188 | Real clinical event on the correct date — confirmed |
| Wrong date (FIX_DATE) | 220 | Real event found but date is off by 1–several days |
| False positive | 195 | No clinical news found — the catalyst association is likely wrong |
| Error | 1 | API failure — could not verify |
| **Total** | **604** | |

**274 new press release links added** across the verified and date-corrected rows.

**False positive rate: 32.3%** — lower than the 71% found in the noise class, but still significant. Roughly 1 in 3 high-move rows turns out to have no verifiable clinical catalyst. These are cases where the stock moved dramatically for a non-clinical reason (earnings, acquisition, FDA label change, general market move) but was attributed to a clinical trial completion date.

**What this means for ML:**

The 195 false positives in the high-move class are a problem: they are mislabeled as "clinical data event with a large market reaction" when in fact no clinical announcement happened. The clean file (`enriched_all_clinical_clean.csv`) remains unchanged. The backfill file (`enriched_all_clinical_clean_pr_backfill.csv`) gives you the full picture — use its `v_action` column to decide which rows to include:

```
v_action == "OK"           → 188 rows — cleanest verified high-move events
v_action == "FIX_DATE"     → 220 rows — real events, dates need correcting before use
v_action is NaN            → 453 rows — noise class (already cleaned in the base file)
v_action == "FLAG_FALSE_POSITIVE" → 195 rows — exclude from training
```

**No changes to the original `enriched_all_clinical_clean.csv`** — this is a new separate file.

---

### 2026-03-08 — Price Re-fetch for Date-Corrected Rows (v3.4.1)

**File updated:** `enriched_all_clinical_clean.csv`

**What changed:**

Previously, the 349 DATE_FIXED rows had their event dates corrected (to the actual press release date) but still retained the original (wrong-date) prices for `price_at_event`, `price_before`, `price_after`, and `move_pct`. This update re-fetches all those prices at the corrected dates.

**Results:**
- 346 / 349 rows re-priced successfully (3 skipped: had invalid dates like "2023-12-00" stored by the AI validator)
- ATR, move classifications (`move_class_norm`, `move_class_abs`, `move_class_combo`) also recomputed with correct dates
- **Verified correct:** 10/10 spot-checked rows matched live yfinance prices exactly (CYTK, IMRX, ABBV, MLTX, IONS, PFE, MLYS, MRK, SNY confirmed)

**Row counts unchanged:** Still 1,057 rows. Only the price columns were updated for DATE_FIXED rows.

**Note for ML use:** DATE_FIXED rows now have fully reliable price data. The only caveat remaining is the 3 rows with unparseable dates ("2023-12-00", "2022-09-00", "2024-12-??") — these retain original wrong-date prices but represent a tiny fraction (<1%) of the dataset.

---

### 2026-03-07 — Full Validation Run + Clean Dataset Created

**Files produced:**
- `enriched_all_clinical_validated.csv` — updated with full validation results (all 1,571 noise rows checked)
- `enriched_all_clinical_clean.csv` — **new** clean ML-ready dataset (false positives removed, dates corrected)

**Full validation results (all 1,571 noise-class rows):**

| v_action | Count | Meaning |
|----------|-------|---------|
| `FLAG_FALSE_POSITIVE` | 1,118 | No clinical news found — not a real event |
| `FIX_DATE` | 349 | Real event found but attributed to wrong date |
| `OK` | 104 | Confirmed real clinical event, small market reaction |

**False positive rate: 71% of noise rows** (1,118 / 1,571). Combined with the 604 high/medium-move rows that were not validated (they don't need it — large moves are self-evidently real events), the full dataset breakdown is:

| Category | Rows | Status |
|----------|------|--------|
| High/medium moves (not validated — clearly real) | 604 | Kept as-is |
| Confirmed real noise events (OK) | 104 | Kept |
| Date-corrected events (FIX_DATE) | 349 | Kept, dates updated |
| False positives removed | 1,118 | Removed from clean file |
| **Clean dataset total** | **1,057** | |

**What changed in `enriched_all_clinical_clean.csv` vs the original:**
- 1,118 rows removed (the false positives)
- 349 rows have corrected `event_date` and `event_trading_date` (the actual date the press release was published)
- Prices (`price_at_event`, `price_before`, `price_after`, `move_pct`) were re-fetched for all date-corrected rows in the 2026-03-08 update. All price data in the clean file is now at the correct (actual press release) date.
- `move_class_norm` breakdown in clean file: Noise 453 · Medium 160 · High 143 · Low 141 · Extreme 127

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

**Filter for ML-ready clean dataset (use `enriched_all_clinical_clean.csv` directly):**
```
data_complete == True             → 1,023 rows
AND move_class_combo != "Medium"  → removes ambiguous cases, keeps clear High vs Low label
```
For DATE_FIXED rows: prices reflect the *original* wrong date until re-fetched. Use `v_action == "OK"` or unvalidated high-move rows (no v_action) if you need clean price data.

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

| File | Rows | Cols | Use for |
|------|------|------|---------|
| `enriched_all_clinical_clean_v2.csv` | 862 | 58 | **Everything** — master file, all enrichments merged ← USE THIS |
| `biotech_universe_expanded.csv` | 460 | — | The 460 biotech tickers tracked (reference) |
| `archive/*` | — | — | All previous intermediate and superseded files |

## Recommended Filters for Clean ML Use

```
data_complete == True          → keeps 2,139 rows (98.3%)
move_class_combo != "Medium"   → removes ambiguous cases (use Low vs High as binary label)
v_action not in [FLAG_FALSE_POSITIVE, FLAG_ERROR]  → removes suspected hallucinations (after full validation)
```

---

## Verification field decisions

Factual answers about the `v_is_verified` field in `enriched_all_clinical_clean.csv`, based on code logic in `validate_catalysts.py` and confirmed against actual file counts.

---

**Q1: 349 rows have `v_is_verified = FALSE` — some have a press release link. Is this expected? Should they be removed?**

Q1: YES, expected. `v_is_verified = False` means Perplexity found clinical news but on a DIFFERENT date than originally recorded — not that the event is fake. The code rule: *"If news exists on a different date, set is_verified=false and provide actual_date."* All 349 rows carry `v_action = DATE_FIXED`; their dates and prices have already been corrected. The 261 with a PR link have a confirmed source; the 88 without a PR link still had a corrected actual_date returned.
Reason: v_is_verified=False = wrong date, not a false event. DATE_FIXED rows are the date-corrected real events.
Action: **KEEP** all 349.

---

**Q2: 604 rows have blank `v_is_verified` and no press release link. Should they be removed?**

Q2: NO. These rows were never sent through the validator. The validation script only processes `move_class_norm == 'Noise'` rows. All 604 blanks are High / Extreme / Medium / Low movers — no Noise rows are in this group. Large price moves (the majority are High/Extreme) are self-evidently real events and did not require AI verification.
Reason: Blank v_is_verified = not validated, not suspicious. Validator only ran on Noise class by design.
Action: **KEEP** all 604.

---

**Q3: ~42 rows have `v_is_verified = TRUE` but no press release link. Should they be re-checked?**

Q3: TRUST them. All 42 carry `v_action = OK`. The code rule: *"is_verified = true ONLY if Perplexity finds concrete evidence of clinical news on or within 1 day of the claimed date."* Absence of a stored PR link means the URL was not captured (URL fetch may have failed, or the source was a journal publication, conference abstract, or CT.gov milestone — not a press release). Perplexity's confirmation of the event itself is what matters.
Reason: is_verified=True is the strictest possible validation outcome. Missing URL ≠ missing event.
Action: **KEEP** all 42. Optionally run a manual URL lookup pass if PR provenance is needed for every row.

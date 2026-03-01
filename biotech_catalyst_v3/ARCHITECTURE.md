# Biotech Catalyst Analyzer v3 — Architecture

## What This Project Does

This tool finds **stock moves** in biotech companies, figures out **why** they happened, and produces a **balanced ML dataset** for predicting whether clinical trial data will cause a significant stock move.

The pipeline now handles both extremes:
- **High-move events** — stocks that jumped 30%+ or dropped 20%+ on clinical data
- **Low-move events** — stocks that moved within normal volatility (< 1.5x ATR) despite clinical data being released

The result is a balanced binary classification dataset: "Did this clinical data event cause a significant move?"

---

## Latest Output

**`enriched_all_clinical.csv`** — Full Clinical Data dataset (primary output, v3.3)

- **2,179 Clinical Data events** across 460 biotech tickers (21× expansion from v3.2's 104)
- move_class_combo: 1,602 Low / 375 High / 165 Medium
- Gainers / Losers: 1,121 / 1,058 (well-balanced)
- market_cap filled: 2,172/2,179 (99.7%)
- ATR filled: 2,142/2,179 (98.3%)
- Move range: -98.5% to +196.1% | Avg absolute move: 13.9%
- Time range: Jan 2023 – Dec 2025
- 52 columns per event (all CT.gov, financial, ATR, and classification columns)

**Supporting datasets:**

| File | Rows | Description |
|------|------|-------------|
| `enriched_all_clinical.csv` | 2,179 | **PRIMARY** — Clinical Data events only, fully enriched |
| `enriched_large_moves.csv` | 2,339 | All catalyst types merged (Clinical + Regulatory + Earnings + etc.) |
| `enriched_clinical_events.csv` | 1,939 | CT.gov-sourced events enriched (265 original + 1,674 new) |
| `enriched_high_moves.csv` | 265 | Original high-move events with ATR (v3.1 baseline) |
| `large_moves_clinical.csv` | 400 | High-move events confirmed clinical by Perplexity |
| `large_moves_filtered.csv` | 5,661 | Pre-filtered candidates (15–200% moves) for Perplexity review |
| `large_moves_new.csv` | 15,667 | Raw ≥10% moves from universe scan |
| `clinical_events_new.csv` | 1,674 | CT.gov completions with stock moves (news-first) |
| `biotech_universe_expanded.csv` | 460 | Validated biotech universe ($50M–$10B market cap) |

Intermediate/archived files are in `archive/` with date suffixes.

---

## Directory Structure

```
biotech_catalyst_v3/
│
├── enriched_all_clinical.csv         # PRIMARY OUTPUT — 2,179 Clinical Data events
├── enriched_large_moves.csv          # All catalyst types merged (2,339 events)
├── enriched_clinical_events.csv      # CT.gov-sourced events enriched
├── enriched_high_moves.csv           # Original 265 high-move events (v3.1 baseline)
├── large_moves_clinical.csv          # 400 high-move events confirmed clinical
├── large_moves_filtered.csv          # 5,661 candidates (15–200% moves)
├── large_moves_new.csv               # 15,667 raw ≥10% moves from universe scan
├── clinical_events_new.csv           # 1,674 CT.gov completion events
├── biotech_universe_expanded.csv     # 460 validated biotech tickers
├── archive/                          # Dated intermediate/superseded files
├── .env                              # API keys (gitignored)
│
├── clients/                          # API clients
│   ├── clinicaltrials_client.py      #   ClinicalTrials.gov API v2 client
│   └── financial_client.py           #   FinancialData fetcher (yfinance)
│
├── utils/                            # Shared utilities
│   ├── data_quality.py               #   Quality threshold, catalyst classification, date validation
│   └── volatility.py                 #   ATR + avg_daily_move, classify_move() (abs/norm/combo)
│
├── scripts/                          # Pipeline scripts
│   ├── expand_company_universe.py    #   Build 460-ticker universe (SPDR XBI + Nasdaq screener)
│   ├── find_clinical_events.py       #   CT.gov completions → stock moves (news-first approach)
│   ├── scan_large_moves.py           #   Scan universe for ≥10% moves not in existing dataset
│   ├── filter_to_clinical.py         #   Perplexity: confirm Clinical Data catalyst, add drug_name
│   ├── incremental_enrich.py         #   Enrich new events; deduplicates + partial-save resume
│   ├── full_pipeline_fix.py          #   Orchestrates all fix steps in order
│   ├── fix_existing_data.py          #   Apply quality/ATR/URL improvements to CSV
│   ├── fix_missing_nct.py            #   Backfill missing ClinicalTrials.gov NCT IDs
│   ├── find_press_release_urls.py    #   Find missing URLs via Perplexity API
│   ├── extract_low_move_clinical.py  #   [Legacy v3.2] Extract ATR-normalized low-move events
│   ├── enrich_clinical_fields.py     #   [Legacy v3.2] Enrich indication/phase/endpoint
│   ├── backfill_financials.py        #   Backfill missing financial data via yfinance
│   ├── create_ml_dataset.py          #   Combine high+low into balanced ML dataset
│   └── extract_low_moves.py          #   [Legacy v3.1] Extract 3-10% moves by raw percentage
│
├── batch_scanner.py                  # Market scanner (yfinance batch download)
├── batch_enrichment.py               # Main enrichment pipeline
├── incremental_scanner.py            # Finds NEW events not in existing data
├── perplexity_scanner.py             # Uses Perplexity AI to find biotech catalysts
├── finnhub_scanner.py                # Uses Finnhub API for market data
├── scan_with_delays.py               # Rate-limited scanner variant
├── find_more_events.py               # Discovers additional events via Perplexity search
│
├── cleanup.sh                        # Run this to remove stale files
└── ARCHITECTURE.md                   # This file
```

---

## How the Pipeline Works

### Phase D: Universe Expansion + Full Dataset (v3.3 — current)

```
SPDR XBI XLSX + Nasdaq screener API
        │
        ▼
expand_company_universe.py  ──→  biotech_universe_expanded.csv
(validate with yfinance,          (460 tickers, $50M–$10B)
 $50M–$10B cap range)                      │
                                           ├─────────────────────────────────┐
                                           ▼                                 ▼
                              find_clinical_events.py           scan_large_moves.py
                              (CT.gov Phase 2/3 completions     (≥10% moves, all 460 tickers,
                               → sponsor → ticker → move)        deduped vs existing data)
                                           │                                 │
                                           ▼                                 ▼
                              clinical_events_new.csv          large_moves_new.csv
                              (1,674 events, all move sizes)   (15,667 moves)
                                           │                                 │
                                           │                    large_moves_filtered.csv
                                           │                    (5,661 rows, 15–200% moves)
                                           │                                 │
                                           │                    filter_to_clinical.py
                                           │                    (Perplexity confirms clinical,
                                           │                     adds drug_name + summary)
                                           │                                 │
                                           │                    large_moves_clinical.csv
                                           │                    (400 confirmed clinical events)
                                           │                                 │
                                           └──────────────┬──────────────────┘
                                                          ▼
                                              incremental_enrich.py
                                              (CT.gov + yfinance + ATR,
                                               deduplication + partial save)
                                                          │
                                                          ▼
                                              enriched_all_clinical.csv
                                              (2,179 Clinical Data events)
```

### Phase A: High-Move Events (existing)

```
batch_scanner.py  ──→  raw_moves_*.csv  ──→  batch_enrichment.py  ──→  enriched_final.csv
                       (>30% moves)           (AI + ClinicalTrials     (265 events,
                                               + yfinance)              14 catalyst types)
                                                      │
                                                      ▼
                                          fix_existing_data.py  ──→  enriched_high_moves.csv
                                          (quality + ATR + URLs)      (265 events + ATR cols)
```

### Phase B: Low-Move Events

```
enriched_high_moves.csv
        │
        │  (extract ticker universe)
        ▼
extract_low_move_clinical.py  ──→  low_move_candidates.csv
(scan for <1.5x ATR moves,         (300 events, 63 tickers)
 ≥2% absolute)                            │
                                          ▼
                               filter_to_clinical.py  ──→  low_move_clinical.csv
                               (Perplexity API confirms     (120 Clinical Data events)
                                clinical data catalyst)            │
                                                                   ▼
                                                      fix_missing_nct.py ──→ NCT IDs (101/120)
                                                      fix_existing_data.py ──→ quality + ATR
                                                      backfill_financials.py ──→ market cap, cash, etc.
                                                      enrich_clinical_fields.py ──→ indication, phase,
                                                                                    is_pivotal, endpoints
                                                                   │
                                                                   ▼
                                                          low_move_enriched.csv
                                                          (120 events, 98% field coverage)
```

### Phase C: Balanced ML Dataset

```
enriched_high_moves.csv          low_move_enriched.csv
(105 Clinical Data, High-move)   (120 Clinical Data, Low-move)
        │                                │
        └──────────┬─────────────────────┘
                   ▼
          create_ml_dataset.py
          (filter to Clinical Data,
           balance classes, shuffle)
                   │
                   ▼
          ml_dataset_clinical.csv
          (210 events: 105 High + 105 Low)
```

---

## Key Columns in the Output

| Column | What It Means |
|--------|---------------|
| `ticker` | Stock symbol (e.g., SRRK, VKTX) |
| `event_date` | When the move happened |
| `move_pct` | How much the stock moved (e.g., +165%, -3%) |
| `move_class` | **ML target**: `High` (>3x ATR) or `Low` (<1.5x ATR) |
| `catalyst_type` | Why it moved (all rows = Clinical Data in ML dataset) |
| `catalyst_summary` | Plain-English description of the event |
| `drug_name` | The drug involved |
| `nct_id` | ClinicalTrials.gov trial ID (e.g., NCT04120493) |
| `phase` / `ct_phase` | Clinical trial phase |
| `is_pivotal` | Is this a pivotal/registrational trial? |
| `ct_sponsor` | Who runs the trial |
| `ct_enrollment` | How many patients in the trial |
| `ct_conditions` | Disease/indication |
| `ct_allocation` | Randomized vs non-randomized |
| `market_cap_m` | Company size in millions |
| `cash_position_m` | Cash on hand in millions |
| `atr_pct` | Average True Range as % (stock's normal daily volatility) |
| `normalized_move` | Move size / ATR (how many "normal days" this move equals) |
| `move_magnitude` | Low (<1.5x ATR), Medium (1.5-3x), High (>3x) |
| `data_quality_score` | 0-1 score of how complete the data is |

---

## What's Working vs What Needs Rebuilding

### Working Now
- Full pipeline from universe expansion → scan → filter → enrich → final dataset
- `expand_company_universe.py` — live ETF + Nasdaq screener fetch, yfinance validation
- `find_clinical_events.py` — CT.gov API v2 with `AREA[CompletionDate]RANGE[...]`, paginated
- `scan_large_moves.py` — batch yfinance scan with ATR + deduplication
- `filter_to_clinical.py` — Perplexity clinical confirmation with timeout handling + partial saves
- `incremental_enrich.py` — unified enrichment with dedup, partial saves, resume support
- `clients/financial_client.py` — real `FinancialDataFetcher` using yfinance
- `utils/volatility.py` — ATR + `avg_daily_move` + three-way `move_class_combo` classification
- ClinicalTrials.gov API client with smart prioritized search
- Financial data backfill via yfinance
- All post-processing scripts (quality, catalyst fix, NCT backfill, URLs)
- All legacy market scanners (batch, incremental, Perplexity, Finnhub)

### Needs Attention
- **AICatalystResearcher** — The Perplexity-based class in `batch_enrichment.py` for researching what caused each move. `perplexity_scanner.py` has a working reference. (Clinical fields enrichment is now covered by `enrich_clinical_fields.py`.)
- **`enriched_all_clinical.csv` class balance** — 73% Low / 17% High / 8% Medium. If training a classifier, down-sample Low or up-sample High/Medium before fitting.

---

## Quick Commands

```bash
export PERPLEXITY_API_KEY="$(grep PERPLEXITY_API_KEY .env | cut -d= -f2)"

# === v3.3 Full Pipeline (Universe Expansion → Final Dataset) ===

# Step 1: Expand biotech universe (fetches SPDR XBI + Nasdaq screener, validates with yfinance)
python3 -m scripts.expand_company_universe --min-cap 50 --max-cap 10000

# Step 2a: Find clinical events via CT.gov (news-first, all move sizes)
python3 -m scripts.find_clinical_events --start 2023-01-01 --end 2025-12-31

# Step 2b: Scan universe for large moves (≥10%, skips already-known events)
python3 -m scripts.scan_large_moves --min-move 10 --start 2023-01-01

# Step 3: Filter large moves to confirmed clinical events via Perplexity
python3 -u -m scripts.filter_to_clinical \
  --input large_moves_filtered.csv --output large_moves_clinical.csv --target-count 400

# Step 4a: Enrich CT.gov events
python3 -u -m scripts.incremental_enrich \
  --new clinical_events_new.csv --existing enriched_high_moves.csv \
  --output enriched_clinical_events.csv

# Step 4b: Enrich confirmed large-move clinical events (merges with step 4a output)
python3 -u -m scripts.incremental_enrich \
  --new large_moves_clinical.csv --existing enriched_clinical_events.csv \
  --output enriched_large_moves.csv

# Step 5: Extract Clinical Data only
python3 -c "
import pandas as pd
df = pd.read_csv('enriched_large_moves.csv')
df[df['catalyst_type']=='Clinical Data'].to_csv('enriched_all_clinical.csv', index=False)
print(f'Saved {len(df[df[\"catalyst_type\"]==\"Clinical Data\"])} Clinical Data events')
"

# === v3.1/v3.2 Legacy Pipeline ===
python3 -m scripts.full_pipeline_fix --input enriched_final.csv
python3 scripts/extract_low_move_clinical.py --input enriched_high_moves.csv
python3 scripts/create_ml_dataset.py

# === Individual tools ===
python3 -m scripts.fix_missing_nct --input <file.csv>
python3 -m scripts.backfill_financials --input <file.csv> --output <file.csv>
python3 -u scripts/enrich_clinical_fields.py --input <file.csv>
python3 -m scripts.find_press_release_urls --input <file.csv>
python3 batch_scanner.py --min-move 30 --start-date 2024-01-01
```

---

## Change Log

### v3.3 — Universe Expansion + Full Dataset (2026-03-01)

**What changed:** Expanded the biotech universe from 111 → 460 tickers and rebuilt the entire dataset collection pipeline to produce a 21× larger Clinical Data dataset.

**Why:** The original dataset had only 104 Clinical Data events sourced from high-move scans of a small ticker universe. This severely limited ML training. Two structural problems were fixed:
1. The universe was too small (111 tickers, manually curated) — missing hundreds of relevant small/mid-cap biotechs
2. The scan-for-big-moves approach only captured positive examples (stocks that moved significantly) — it couldn't find events where clinical data was released but the stock barely moved

**New scripts:**
- `scripts/expand_company_universe.py` — Downloads live holdings from SPDR XBI XLSX and Nasdaq screener API (fallback for ARKG/IBB which block direct CSV access). Validates each ticker with yfinance. Produces `biotech_universe_expanded.csv` with 460 tickers, $50M–$10B market cap range.
- `scripts/find_clinical_events.py` — News-first approach: fetches Phase 2/3 trial completions from CT.gov API v2, maps sponsor → ticker via universe name matching, then fetches the stock move around each completion date. Captures all move sizes (Low/Medium/High) from the same event type.
- `scripts/scan_large_moves.py` — Scans all 460 universe tickers for ≥10% single-day moves not already in the dataset. Computes ATR and normalized move for each event. Output: 15,667 raw qualifying moves.
- `scripts/incremental_enrich.py` — Unified enrichment: deduplicates on (ticker, event_date), enriches with CT.gov details + yfinance financials + ATR classification, partial-saves every N rows for crash recovery, supports resume from `_partial.csv`.

**Modified scripts:**
- `clients/financial_client.py` — New file: real `FinancialDataFetcher` using yfinance replacing the stub in `batch_enrichment.py`
- `batch_enrichment.py` — Imports real `FinancialDataFetcher`; fetches financials for every event (was every 5th)
- `utils/volatility.py` — Added `avg_daily_move` to `calculate_atr`; new `classify_move()` returns three classification columns: `move_class_abs` (VeryLow/Low/Medium/High/VeryHigh), `move_class_norm` (Normal/Elevated/High/Extreme), `move_class_combo` (the ML label: Low if <15% AND <2x ATR; High if ≥30% AND ≥3.5x ATR; else Medium)
- `scripts/filter_to_clinical.py` — Fixed `row['date']` → `row['event_date']` column name bug; added timeout/network exception handling (no more crashes on slow Perplexity responses); added partial-save every 10 clinical hits; added `flush=True` for unbuffered log output

**Results:**

| Metric | Before (v3.2) | After (v3.3) |
|--------|---------------|--------------|
| Biotech universe | 111 tickers | 460 tickers |
| Clinical Data events | 104 | 2,179 |
| Low-move events | 105 | 1,602 |
| High-move events | 105 | 375 |
| Gainers/Losers balance | skewed | 1,121 / 1,058 |
| Market cap coverage | 98% | 99.7% |

**Key design decisions:**
- CT.gov `filter.advanced=AREA[CompletionDate]RANGE[start,end]` (not `filter.completionDate` which doesn't exist in API v2)
- `nextPageToken` pagination loop to handle >1,000 CT.gov results
- Pre-filter large_moves_new.csv to 15–200% range before Perplexity (reduces API calls from 15k → 5.6k)
- `move_class_combo` as the primary ML label: combines absolute and ATR-normalized thresholds so volatility-adjusted signals aren't thrown away

---

### v3.2.1 — Low-Move Enrichment Fix (2026-02-21)

**What changed:** Fixed the low-move pipeline which was producing events with 0% fill rate on financial and AI-researched clinical fields.

**Problem:** The low-move pipeline (v3.2) skipped two enrichment steps that the high-move data had received from the original `batch_enrichment.py` run:
1. **Financial data** — `market_cap_m`, `current_price`, `cash_position_m`, `short_percent`, `institutional_ownership`, `analyst_target`, `analyst_rating`, `cash_runway_months` were all empty (0/120)
2. **AI-researched clinical fields** — `indication`, `phase`, `is_pivotal`, `pivotal_evidence`, `primary_endpoint_met`, `primary_endpoint_result` were all empty (0/120)

**Fix:**
1. Ran `backfill_financials.py` on `low_move_enriched.csv` — filled 118/120 rows with yfinance data
2. Created `scripts/enrich_clinical_fields.py` — new script that queries Perplexity API to fill clinical trial details (indication, phase, is_pivotal, endpoint results). Filled 120/120 rows.
3. Rebuilt `ml_dataset_clinical.csv` with fully enriched data

**Before → After (low-move data):**
| Field | Before | After |
|-------|--------|-------|
| indication | 0% | 98% |
| phase | 0% | 98% |
| is_pivotal | 0% | 99% |
| primary_endpoint_met | 0% | 99% |
| market_cap_m | 0% | 98% |
| current_price | 0% | 98% |
| cash_position_m | 0% | 98% |
| data_quality_score | 0% | 100% (mean: 0.83) |

**New script:** `scripts/enrich_clinical_fields.py` — Uses Perplexity `sonar` model to research each clinical event and extract structured fields. Saves progress every 20 rows. Can be run on any enriched CSV with `drug_name` and `catalyst_summary` columns.

**Pipeline update:** Step 4 of the low-move pipeline now includes `backfill_financials.py` and `enrich_clinical_fields.py` after NCT backfill.

**Files cleaned:** Moved 6 intermediate/superseded files to `archive/` with date suffixes. Only latest CSVs remain in the working directory.

---

### v3.2 — Balanced ML Dataset Pipeline (2026-02-17)

**What changed:** Added a complete pipeline for extracting low-move clinical events and combining them with high-move events into a balanced ML dataset.

**Why:** The original dataset only contained high-move events (30%+ moves). Training an ML model to predict whether clinical data will cause a significant move requires negative examples too — events where clinical data was released but the stock barely moved.

**New scripts:**
- `scripts/extract_low_move_clinical.py` — Scans the same biotech ticker universe for days where the stock moved < 1.5x ATR (within normal volatility) but at least 2% absolute. Uses ATR normalization instead of raw percentage thresholds, which accounts for each stock's baseline volatility.
- `scripts/filter_to_clinical.py` — Takes the raw low-move candidates and queries Perplexity API to confirm which ones coincide with actual clinical trial data releases. Filters from ~300 candidates down to ~120 confirmed Clinical Data events.
- `scripts/create_ml_dataset.py` — Combines high-move Clinical Data events with low-move Clinical Data events, balances the classes (min of both), shuffles, and outputs the final dataset.

**Key design decisions:**
- ATR-normalized thresholds instead of raw percentages: a 5% move on a stock with 2% ATR is significant, but a 5% move on a stock with 10% ATR is noise. The threshold of < 1.5x ATR captures the latter.
- Same ticker universe for both classes to avoid confounding stock-level differences with the move prediction task.
- Perplexity API as the clinical data oracle, since there's no structured database mapping every clinical data release to a specific date and ticker.

**New data files:**
- `ml_dataset_clinical.csv` — Primary ML output (210 events, 105 High + 105 Low)
- `enriched_high_moves.csv` — enriched_final.csv + ATR columns
- `low_move_enriched.csv` — Fully enriched low-move Clinical Data events
- `.env` — API keys (gitignored)

**Schema changes:**
- Added `move_class` column to ML dataset (`High` or `Low`)
- Low-move events use `event_date` (renamed from `date`) and `price_at_event` (renamed from `close`) to match high-move schema

---

### v3.1 — Post-Processing & Quality Pipeline (previous)

**What changed:** Added quality scoring, catalyst reclassification, ATR normalization, NCT backfill, and press release URL discovery.

**New scripts:** `fix_existing_data.py`, `fix_missing_nct.py`, `find_press_release_urls.py`, `full_pipeline_fix.py`

**New utils:** `data_quality.py`, `volatility.py`

**New columns:** `data_quality_score`, `data_quality_threshold_passed`, `is_valid_date`, `atr_pct`, `normalized_move`, `move_magnitude`, `press_release_url`

---

### v3.0 — Initial Pipeline (previous)

Batch scanner + Perplexity enrichment + ClinicalTrials.gov lookup + financial data. Produced `enriched_final.csv` with 265 events across ~130 tickers.

---
---

## Previous Architecture (v3.1, archived 2026-02-17)

> The following is the previous version of this document, preserved for reference.

---

## What This Project Does

This tool finds **big stock moves** in biotech companies and figures out **why** they happened. When a biotech stock jumps 30% or drops 20%, there's usually a reason — a clinical trial result, an FDA decision, an earnings surprise. This pipeline:

1. **Scans** the market for big price moves
2. **Researches** what caused each move (using AI + public APIs)
3. **Enriches** each event with clinical trial data, financial data, and quality metrics
4. **Outputs** a structured dataset ready for analysis or ML

---

## Latest Output

**`enriched_final.csv`** — This is the current best dataset.

- 265 events across ~130 biotech tickers
- 40+ columns per event (price data, catalyst details, clinical trial info, financials, quality scores, ATR normalization)
- Covers moves from Jan 2024 to present
- NCT ID coverage: 99% of Clinical Data events (104/105)
- 14 catalyst type categories (Clinical Data, Regulatory, Earnings, Analyst, Acquisition/M&A, etc.)

---

## Directory Structure

```
biotech_catalyst_v3/
│
├── enriched_final.csv              # LATEST OUTPUT — use this
├── enriched_new_batch.csv          # Base enriched data (before fixes)
├── raw_moves_filtered.csv          # Input: filtered price moves to enrich
├── raw_moves_new.csv               # Input: latest raw market moves
├── enriched_new_events.csv         # Additional events found via Perplexity
├── full_ticker_list.csv            # The ~130 biotech tickers we track
├── nct_search_results.csv          # Audit log from NCT ID backfill
├── nct_search_trace.json           # Detailed NCT search trace log
├── url_search_log.json             # Audit log from press release URL search
│
├── clients/                       # API clients
│   └── clinicaltrials_client.py   #   ClinicalTrials.gov API v2 client
│
├── utils/                         # Shared utilities
│   ├── data_quality.py            #   Quality threshold, catalyst classification, date validation
│   └── volatility.py              #   ATR calculation, move normalization
│
├── scripts/                       # One-off and maintenance scripts
│   ├── full_pipeline_fix.py       #   Orchestrates all fix steps in order
│   ├── fix_existing_data.py       #   Apply all improvements to existing CSV
│   ├── fix_missing_nct.py         #   Backfill missing ClinicalTrials.gov NCT IDs
│   ├── find_press_release_urls.py #   Find missing URLs via Perplexity API
│   └── extract_low_moves.py       #   Extract 3-10% moves for ML training balance
│
├── batch_scanner.py               # Market scanner (yfinance batch download)
├── batch_enrichment.py            # Main enrichment pipeline (needs AI researcher rebuilt)
├── incremental_scanner.py         # Finds NEW events not in existing data
├── perplexity_scanner.py          # Uses Perplexity AI to find biotech catalysts
├── finnhub_scanner.py             # Uses Finnhub API for market data
├── scan_with_delays.py            # Rate-limited scanner variant
├── find_more_events.py            # Discovers additional events via Perplexity search
│
├── cleanup.sh                     # Run this to remove stale files
└── ARCHITECTURE.md                # This file
```

---

## How the Pipeline Works

### Step 1: Scan for Big Moves

Multiple scanners look for biotech stocks that moved significantly:

```
batch_scanner.py          →  raw_moves_*.csv
incremental_scanner.py    →  raw_moves_new.csv
perplexity_scanner.py     →  finds events via AI search
```

**What it does:** Downloads daily price data for ~130 biotech tickers using yfinance. Flags any day where a stock moved more than 30% up or down.

### Step 2: Enrich Each Event

For each big move, the pipeline gathers three types of data:

```
┌─────────────────────────────────────────────────┐
│              For each price move:                │
│                                                  │
│  1. AI Research (Perplexity API)                 │
│     → What caused the move?                      │
│     → Drug name, indication, phase               │
│     → Is this a pivotal trial?                   │
│                                                  │
│  2. Clinical Trial Lookup (ClinicalTrials.gov)   │
│     → NCT ID (the trial's unique identifier)     │
│     → Official title, enrollment, status          │
│     → Sponsor, phase, completion date            │
│                                                  │
│  3. Financial Data (yfinance)                    │
│     → Market cap, cash position                  │
│     → Short interest, institutional ownership    │
│     → Analyst targets                            │
│                                                  │
│  → Combined into enriched_*.csv                  │
└─────────────────────────────────────────────────┘
```

### Step 3: Post-Processing (scripts/)

After enrichment, several scripts clean and improve the data:

- **fix_missing_nct.py** — Uses a prioritized search strategy to find the correct ClinicalTrials.gov NCT ID for each drug. Tries the drug name first, then known aliases (e.g., "atebimetinib" → "IMM-1-104"), then filters by indication, phase, and sponsor.

- **fix_existing_data.py** — Applies five improvements:
  1. **Quality threshold** — Adds `data_quality_threshold_passed` (True if score >= 0.7)
  2. **Catalyst classification** — Classifies into 14 categories (Clinical Data, Regulatory, Earnings, Analyst, Acquisition/M&A, Insider Trading, Corporate Action, Technical/Momentum, Pipeline Update, Legal, etc.) Separates "Unknown" (couldn't find cause) from specific categories.
  3. **Error logging** — Flags missing financial data and invalid dates
  4. **ATR normalization** — Calculates how unusual a move is relative to the stock's normal volatility
  5. **Press release URLs** — Uses Perplexity API to find official press release/news URLs for events missing them

- **find_press_release_urls.py** — Standalone script to find missing press release URLs using Perplexity's `sonar-pro` model with web search. Queries for official sources (businesswire, globenewswire, prnewswire, sec.gov, IR sites). Rate-limited, saves progress periodically, writes audit log to `url_search_log.json`.

- **full_pipeline_fix.py** — Orchestrates all fix steps in order: NCT backfill → quality/catalyst/errors/ATR/URLs → verification. Supports `--skip-nct`, `--skip-atr`, `--skip-urls` flags.

- **extract_low_moves.py** — Finds 3-10% moves (smaller moves) for ML model training. You need both big moves and normal moves to train a model that predicts outcomes.

---

## Key Columns in the Output

| Column | What It Means |
|--------|---------------|
| `ticker` | Stock symbol (e.g., SRRK, VKTX) |
| `event_date` | When the big move happened |
| `move_pct` | How much the stock moved (e.g., +361%, -30%) |
| `catalyst_type` | Why it moved: Clinical Data, Regulatory, Earnings, Partnership, Financing, Analyst, Acquisition/M&A, Corporate Action, Technical/Momentum, Insider Trading, Pipeline Update, Legal, Unknown |
| `catalyst_summary` | Plain-English description of the event |
| `drug_name` | The drug involved (if clinical) |
| `nct_id` | ClinicalTrials.gov trial ID (e.g., NCT04120493) |
| `phase` | Clinical trial phase (Phase 1, 2, 3) |
| `is_pivotal` | Is this a pivotal/registrational trial? |
| `ct_sponsor` | Who runs the trial |
| `ct_enrollment` | How many patients in the trial |
| `market_cap_m` | Company size in millions |
| `cash_position_m` | Cash on hand in millions |
| `data_quality_score` | 0-1 score of how complete the data is |
| `data_quality_threshold_passed` | True if quality >= 0.7 |
| `atr_pct` | Average True Range as % (stock's normal daily volatility) |
| `normalized_move` | Move size / ATR (how many "normal days" this move equals) |
| `move_magnitude` | Low (<1.5x ATR), Medium (1.5-3x), High (>3x) |

---

## What's Working vs What Needs Rebuilding

### Working Now
- ClinicalTrials.gov API client with smart prioritized search
- All post-processing scripts (quality threshold, catalyst fix, ATR, NCT backfill)
- All market scanners (batch, incremental, Perplexity, Finnhub)
- Low-move extraction for ML training data

### Needs Rebuilding
- **AICatalystResearcher** — The Perplexity-based class that researches what caused each move. `perplexity_scanner.py` has a working reference implementation. Needs to be extracted into a proper module.
- **FinancialDataFetcher** — The yfinance-based financial data fetcher. Straightforward to rebuild.
- These are marked as `TODO` stubs in `batch_enrichment.py`.

---

## Quick Commands

```bash
# Run full pipeline (NCT fix + all improvements + verification)
python -m scripts.full_pipeline_fix --input enriched_final.csv

# Skip slow steps
python -m scripts.full_pipeline_fix --input enriched_final.csv --skip-atr --skip-urls

# Apply improvements only (skip NCT fix)
python -m scripts.fix_existing_data --input enriched_final.csv --output enriched_final.csv

# Find missing press release URLs only
python -m scripts.find_press_release_urls --input enriched_final.csv
python -m scripts.find_press_release_urls --dry-run  # preview what would be searched

# Fix missing NCT IDs in existing data
python -m scripts.fix_missing_nct --input enriched_final.csv

# Extract low-move events for ML training
python -m scripts.extract_low_moves --min-move 3 --max-move 10

# Scan market for new big moves
python batch_scanner.py --min-move 30 --start-date 2024-01-01

# Clean up stale files
chmod +x cleanup.sh && ./cleanup.sh
```

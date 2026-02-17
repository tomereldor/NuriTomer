# Biotech Catalyst Analyzer v3 — Architecture

## What This Project Does

This tool finds **stock moves** in biotech companies, figures out **why** they happened, and produces a **balanced ML dataset** for predicting whether clinical trial data will cause a significant stock move.

The pipeline now handles both extremes:
- **High-move events** — stocks that jumped 30%+ or dropped 20%+ on clinical data
- **Low-move events** — stocks that moved within normal volatility (< 1.5x ATR) despite clinical data being released

The result is a balanced binary classification dataset: "Did this clinical data event cause a significant move?"

---

## Latest Output

**`ml_dataset_clinical.csv`** — Balanced ML dataset (primary output)

- 210 events: 105 High-move + 105 Low-move Clinical Data events
- 80 unique biotech tickers
- 41 columns per event
- NCT ID coverage: 194/210 (92%)
- Trial phase coverage: 190/210 (90%)
- Time range: Jan 2024 – Dec 2025

**Supporting datasets:**

| File | Rows | Description |
|------|------|-------------|
| `enriched_high_moves.csv` | 265 | All high-move events (all catalyst types) with ATR |
| `enriched_final.csv` | 265 | Original enriched dataset (pre-ATR, archived) |
| `low_move_enriched.csv` | 120 | Low-move Clinical Data events, fully enriched |
| `low_move_candidates.csv` | 300 | Raw low-move candidates (pre-clinical filter) |

---

## Directory Structure

```
biotech_catalyst_v3/
│
├── ml_dataset_clinical.csv           # PRIMARY OUTPUT — balanced ML dataset
├── enriched_high_moves.csv           # High-move events with ATR normalization
├── enriched_final.csv                # Original enriched dataset (archived)
├── low_move_enriched.csv             # Low-move Clinical Data, fully enriched
├── low_move_clinical.csv             # Low-move Clinical Data (pre-enrichment)
├── low_move_clinical_nct_fixed.csv   # Low-move after NCT backfill
├── low_move_candidates.csv           # Raw low-move candidates (pre-filter)
├── nct_search_results.csv            # Audit log from NCT ID backfill
├── nct_search_trace.json             # Detailed NCT search trace log
├── .env                              # API keys (gitignored)
│
├── clients/                          # API clients
│   └── clinicaltrials_client.py      #   ClinicalTrials.gov API v2 client
│
├── utils/                            # Shared utilities
│   ├── data_quality.py               #   Quality threshold, catalyst classification, date validation
│   └── volatility.py                 #   ATR calculation, move normalization
│
├── scripts/                          # Pipeline scripts
│   ├── full_pipeline_fix.py          #   Orchestrates all fix steps in order
│   ├── fix_existing_data.py          #   Apply quality/ATR/URL improvements to CSV
│   ├── fix_missing_nct.py            #   Backfill missing ClinicalTrials.gov NCT IDs
│   ├── find_press_release_urls.py    #   Find missing URLs via Perplexity API
│   ├── extract_low_moves.py          #   [Legacy] Extract 3-10% moves by raw percentage
│   ├── extract_low_move_clinical.py  #   [New] Extract ATR-normalized low-move events
│   ├── filter_to_clinical.py         #   [New] Filter candidates to Clinical Data via Perplexity
│   ├── create_ml_dataset.py          #   [New] Combine high+low into balanced ML dataset
│   └── backfill_financials.py        #   Backfill missing financial data
│
├── batch_scanner.py                  # Market scanner (yfinance batch download)
├── batch_enrichment.py               # Main enrichment pipeline (AI researcher TODO)
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

### Phase B: Low-Move Events (new)

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
                                                      fix_missing_nct.py  ──→  NCT enrichment
                                                      fix_existing_data.py ──→ low_move_enriched.csv
                                                      (quality + ATR)          (120 events, 101 NCTs)
```

### Phase C: Balanced ML Dataset (new)

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
- Full pipeline from scan → enrich → ATR → ML dataset
- ClinicalTrials.gov API client with smart prioritized search
- ATR-normalized low-move extraction + Perplexity clinical filter
- All post-processing scripts (quality, catalyst fix, ATR, NCT backfill, URLs)
- All market scanners (batch, incremental, Perplexity, Finnhub)
- Balanced ML dataset creation

### Needs Rebuilding
- **AICatalystResearcher** — The Perplexity-based class in `batch_enrichment.py` that researches what caused each move. `perplexity_scanner.py` has a working reference. Needs extraction into a proper module.
- **FinancialDataFetcher** — The yfinance-based financial data fetcher. Marked as `TODO` in `batch_enrichment.py`.
- **Financial enrichment for low-move events** — `low_move_enriched.csv` has no financial data (market_cap, cash_position, etc). Need to run `backfill_financials.py` or rebuild the fetcher.

---

## Quick Commands

```bash
# === Full pipeline (high-move) ===
python3 -m scripts.full_pipeline_fix --input enriched_final.csv
python3 -m scripts.full_pipeline_fix --input enriched_final.csv --skip-atr --skip-urls

# === Low-move pipeline ===
# Step 1: Add ATR to high-move data
python3 -m scripts.fix_existing_data --input enriched_final.csv --output enriched_high_moves.csv

# Step 2: Extract low-move candidates (ATR-normalized)
python3 scripts/extract_low_move_clinical.py --input enriched_high_moves.csv

# Step 3: Filter to Clinical Data via Perplexity
export PERPLEXITY_API_KEY="$(grep PERPLEXITY_API_KEY .env | cut -d= -f2)"
python3 scripts/filter_to_clinical.py --input low_move_candidates.csv --output low_move_clinical.csv

# Step 4: Enrich (NCT + quality + ATR)
python3 -m scripts.fix_missing_nct --input low_move_clinical.csv --all-types
python3 -m scripts.fix_existing_data --input low_move_clinical_nct_fixed.csv --output low_move_enriched.csv

# Step 5: Create balanced ML dataset
python3 scripts/create_ml_dataset.py

# === Individual tools ===
python3 -m scripts.fix_missing_nct --input <file.csv>
python3 -m scripts.find_press_release_urls --input <file.csv>
python3 -m scripts.extract_low_moves --min-move 3 --max-move 10  # legacy raw-pct version
python3 batch_scanner.py --min-move 30 --start-date 2024-01-01
```

---

## Change Log

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

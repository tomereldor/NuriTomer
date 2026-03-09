"""
Extend with PR Discovery
========================
Extends the biotech catalyst dataset with NEW, RELEVANT clinical catalyst
events discovered via Perplexity sonar-pro search queries.

Pipeline
--------
Stage 1 — Discovery
    Run targeted Perplexity search queries for clinical catalyst events.
    Each query returns a JSON list of {ticker, event_date, pr_url, ...}.
    Saves intermediate results to a raw CSV (resumable via --from-raw).

Stage 2 — Verification + Deduplication
    For each candidate: fetch PR URL via fetch_pr_details().
    Apply relevance keyword filter.
    Deduplicate against existing dataset by (ticker, event_date).

Stage 3 — OHLC Enrichment
    Download OHLC for all new tickers.
    Compute price_before, price_at_event, price_after, move_pct, move_2d_pct.
    Compute ATR and move classifications.
    Optionally filter by minimum move_pct (default 3%).

Stage 4 — Output
    Save accepted candidates and rejected rows as separate CSVs.
    Candidates are ready for manual review and optional merge into clean_v2.

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3

    # Dry run — show which queries would run, no API calls
    python -m scripts.extend_with_pr_discovery --dry-run --limit 3

    # Full run
    python -m scripts.extend_with_pr_discovery \\
        --existing enriched_all_clinical_clean_v2.csv \\
        --candidates extended_relevant_clinical_candidates.csv \\
        --rejected  extended_relevant_clinical_rejected.csv

    # Resume from saved discovery raw (skip Stage 1 API calls)
    python -m scripts.extend_with_pr_discovery \\
        --from-raw  extended_pr_discovery_raw.csv \\
        --existing  enriched_all_clinical_clean_v2.csv \\
        --candidates extended_relevant_clinical_candidates.csv \\
        --rejected  extended_relevant_clinical_rejected.csv
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_catalysts import _load_dotenv, fetch_pr_details
from utils.ohlc_cache import load_ohlc_bulk, date_range_for_events
from utils.volatility import compute_atr_for_ticker, classify_move

_load_dotenv()

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL     = "https://api.perplexity.ai/chat/completions"
MODEL              = "sonar-pro"
RATE_LIMIT_DELAY   = 2.5   # seconds between discovery API calls


# ============================================================================
# Discovery queries
# 20 targeted queries across catalyst types, phases, and time periods
# ============================================================================

DISCOVERY_QUERIES = [
    # Phase 3 topline by quarter — highest signal events
    ("Phase3_Q1_2024", "biotech Phase 3 topline results January February March 2024"),
    ("Phase3_Q2_2024", "biotech Phase 3 topline results April May June 2024"),
    ("Phase3_Q3_2024", "biotech Phase 3 topline results July August September 2024"),
    ("Phase3_Q4_2024", "biotech Phase 3 topline results October November December 2024"),
    ("Phase3_Q1_2025", "biotech Phase 3 topline results January February March 2025"),
    ("Phase3_Q2_2025", "biotech Phase 3 topline results April May June 2025"),
    ("Phase3_Q3_2025", "biotech Phase 3 topline results July August September 2025"),
    ("Phase3_Q4_2025", "biotech Phase 3 topline results October November December 2025"),
    # Phase 3 topline 2023 — year not covered by quarterly queries above
    ("Phase3_H1_2023", "biotech Phase 3 topline primary endpoint results H1 2023"),
    ("Phase3_H2_2023", "biotech Phase 3 topline primary endpoint results H2 2023"),
    # Phase 2 readouts
    ("Phase2_2024", "biotech Phase 2b clinical data readout positive 2024 NASDAQ NYSE"),
    ("Phase2_2025", "biotech Phase 2b clinical data readout positive 2025 NASDAQ NYSE"),
    # FDA decisions
    ("FDA_2024", "FDA approval rejection PDUFA complete response letter biotech 2024"),
    ("FDA_2025", "FDA approval rejection PDUFA complete response letter biotech 2025"),
    # Interim analyses — high probability of large stock move
    ("Interim_2024_2025", "biotech interim analysis clinical trial data readout stock large move 2024 2025"),
    # Phase 1 big movers
    ("Phase1_2024_2025", "Phase 1 clinical trial data readout biotech stock surge 30 percent 2024 2025"),
    # By indication — oncology is largest bucket
    ("Oncology_2024", "oncology cancer Phase 3 clinical trial primary endpoint results 2024 NASDAQ NYSE"),
    ("Oncology_2025", "oncology cancer Phase 3 clinical trial primary endpoint results 2025 NASDAQ NYSE"),
    # Big stock moves — catches events not covered by specific phase queries
    ("BigGainer_2024_2025", "biotech NASDAQ stock surge 50 percent clinical trial data press release 2024 2025"),
    ("BigLoser_2024_2025", "biotech NASDAQ stock crash clinical trial failed primary endpoint 2024 2025"),
]


# ============================================================================
# Relevance filter
# ============================================================================

INCLUDE_KEYWORDS = [
    "topline", "top-line", "primary endpoint", "interim", "phase 3", "phase iii",
    "phase 2", "phase ii", "phase 1", "phase i", "clinical results", "data readout",
    "data read-out", "fda approv", "fda reject", "fda refus", "pdufa", "nda", "bla",
    "pivotal", "efficacy data", "clinical trial result", "clinical study result",
    "met primary", "announced results", "reported results", "data from",
]

EXCLUDE_KEYWORDS = [
    "investor conference", "earnings call", "quarterly result", "quarterly financial",
    "financial result", "annual meeting", "appoints", "board of director",
    "stock offering", "public offering", "shelf registration", "collaboration agreement",
    "license agreement", "strategic update",
]

# Pattern to extract ticker from "(NASDAQ: XXXX)" or "(NYSE: XXXX)" in PR text
_EXCHANGE_PATTERN = re.compile(r'\((?:NASDAQ|NYSE)[:\s]+([A-Z]{1,5})\)', re.IGNORECASE)


def _is_relevant(text: str) -> bool:
    """Return True if text passes the clinical catalyst relevance filter."""
    t = text.lower()
    return any(kw in t for kw in INCLUDE_KEYWORDS) and not any(kw in t for kw in EXCLUDE_KEYWORDS)


def _extract_ticker_from_pr(text: str, fallback: str = "") -> str:
    """Extract confirmed ticker from exchange pattern in PR body text."""
    m = _EXCHANGE_PATTERN.search(text)
    return m.group(1).upper() if m else (fallback.upper() if fallback else "")


# ============================================================================
# Perplexity discovery call
# ============================================================================

def _normalize_event_date(raw: str) -> str:
    """
    Convert various date formats to YYYY-MM-DD.
    Handles: 'Q1 2024' → '2024-02-15', 'January 2024' → '2024-01-15',
    'March 6, 2024' → '2024-03-06', '2024-03-06' → '2024-03-06'.
    Returns '' on failure.
    """
    raw = str(raw).strip()
    if not raw:
        return ""

    # Already ISO format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', raw):
        return raw

    # "Q1 2024" / "Q2 2025" etc.
    m = re.match(r'Q([1-4])\s+(\d{4})', raw, re.IGNORECASE)
    if m:
        q, year = int(m.group(1)), m.group(2)
        mid_month = {1: "02", 2: "05", 3: "08", 4: "11"}[q]
        return f"{year}-{mid_month}-15"

    # Try pandas to_datetime for everything else
    try:
        ts = pd.to_datetime(raw, dayfirst=False, errors="raise")
        return ts.strftime("%Y-%m-%d")
    except Exception:
        pass

    # "Month YYYY" → use day 15
    m = re.match(r'(\w+)\s+(\d{4})', raw)
    if m:
        try:
            ts = pd.to_datetime(f"{m.group(1)} 15 {m.group(2)}")
            return ts.strftime("%Y-%m-%d")
        except Exception:
            pass

    return ""


def _call_perplexity_search(
    query: str,
    max_retries: int = 3,
) -> Tuple[Optional[List[dict]], List[str], str]:
    """
    Run a discovery query via Perplexity sonar-pro.

    Returns:
        (events_list, citation_urls, error_str)
        events_list: list of event dicts from JSON response, or None on error
        citation_urls: source URLs from Perplexity citations field
    """
    if not PERPLEXITY_API_KEY:
        return None, [], "PERPLEXITY_API_KEY not set in environment"

    system = "Return ONLY valid JSON, no markdown, no explanation."

    prompt = f"""Find real biotech clinical trial result events for: {query}

Return a JSON array of up to 5 objects. Use exactly this format:
[
  {{
    "ticker": "XXXX",
    "company": "Company Name Inc.",
    "event_date": "2024-03-15",
    "drug_name": "drug name",
    "indication": "target disease",
    "phase": "Phase 3",
    "headline": "Company announces topline results...",
    "pr_url": "https://www.businesswire.com/...",
    "primary_endpoint_met": true,
    "summary": "One sentence describing the outcome."
  }}
]

Include: Phase 1/2/3 topline/interim results, FDA approvals/rejections, PDUFA outcomes.
Exclude: earnings, conferences, management changes, stock offerings, licensing deals.
Use null for pr_url if unknown. Return JSON array only:"""

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 3000,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=90)

            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                return None, [], f"HTTP {resp.status_code}: {resp.text[:100]}"

            data      = resp.json()
            citations = data.get("citations", [])
            content   = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
            )

            if not content:
                return None, citations, "Empty response"

            # Strip markdown fences if present
            for fence in ("```json", "```"):
                if content.startswith(fence):
                    content = content[len(fence):]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Attempt to recover from truncated JSON by closing the array
                # after the last complete object
                last_brace = content.rfind("},")
                if last_brace != -1:
                    try:
                        parsed = json.loads(content[: last_brace + 1] + "]")
                    except json.JSONDecodeError as e2:
                        return None, citations, f"JSON parse error (even after repair): {e2}"
                else:
                    return None, citations, "JSON parse error: could not repair truncated response"

            # Normalise: model sometimes wraps list in a dict
            if isinstance(parsed, dict):
                for key in ("events", "results", "data", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    parsed = [parsed]

            if not isinstance(parsed, list):
                return None, citations, f"Expected list, got {type(parsed).__name__}"

            return parsed, citations, ""

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None, [], "Timeout"
        except Exception as e:
            return None, [], str(e)[:120]

    return None, [], "Max retries exceeded"


# ============================================================================
# OHLC price helpers (mirror of fix_validated_rows.py)
# ============================================================================

def _close_on_date(ohlc: pd.DataFrame, date_str: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (close_price, trading_date_str) on or up to 4 days after date_str."""
    if ohlc is None or ohlc.empty:
        return None, None
    target = pd.Timestamp(date_str).normalize()
    for delta in range(5):
        check = target + pd.Timedelta(days=delta)
        mask  = ohlc.index.normalize() == check
        if mask.any():
            val = ohlc.loc[mask, "Close"].iloc[-1]
            return (round(float(val), 4) if pd.notna(val) else None), check.strftime("%Y-%m-%d")
    return None, None


def _close_before(ohlc: pd.DataFrame, date_str: str) -> Optional[float]:
    """Last closing price strictly before date_str."""
    if ohlc is None or ohlc.empty:
        return None
    pre = ohlc[ohlc.index.normalize() < pd.Timestamp(date_str).normalize()]
    if pre.empty:
        return None
    val = pre["Close"].iloc[-1]
    return round(float(val), 4) if pd.notna(val) else None


def _close_after(ohlc: pd.DataFrame, date_str: str) -> Optional[float]:
    """First closing price strictly after date_str."""
    if ohlc is None or ohlc.empty:
        return None
    post = ohlc[ohlc.index.normalize() > pd.Timestamp(date_str).normalize()]
    if post.empty:
        return None
    val = post["Close"].iloc[0]
    return round(float(val), 4) if pd.notna(val) else None


# ============================================================================
# OHLC enrichment
# ============================================================================

def _enrich_with_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Download OHLC and compute price metrics + ATR for all candidates."""
    if df.empty:
        return df

    tickers = df["ticker"].str.upper().unique().tolist()
    print(f"\nDownloading OHLC for {len(tickers)} tickers ...")

    ohlc_start, ohlc_end = date_range_for_events(df)
    ohlc_cache = load_ohlc_bulk(tickers, ohlc_start, ohlc_end, events_df=df)

    for idx, row in df.iterrows():
        ticker   = str(row.get("ticker", "")).upper()
        date_str = str(row.get("event_date", ""))
        ohlc     = ohlc_cache.get(ticker)

        if ohlc is None:
            df.at[idx, "data_complete"] = False
            continue

        price_at, trading_date = _close_on_date(ohlc, date_str)
        price_before           = _close_before(ohlc, date_str)
        price_after            = _close_after(ohlc, date_str)

        df.at[idx, "price_at_event"]    = price_at
        df.at[idx, "price_before"]      = price_before
        df.at[idx, "price_after"]       = price_after
        df.at[idx, "event_trading_date"] = trading_date or date_str

        if price_before and price_before > 0 and price_at:
            move_pct = round((price_at - price_before) / price_before * 100, 2)
            df.at[idx, "move_pct"]   = move_pct
            df.at[idx, "event_type"] = "Gainer" if move_pct > 0 else "Loser"

        if price_before and price_before > 0 and price_after:
            df.at[idx, "move_2d_pct"] = round(
                (price_after - price_before) / price_before * 100, 2
            )

        effective_date = trading_date or date_str
        atr = compute_atr_for_ticker(ohlc, effective_date)
        if atr.get("atr_pct"):
            df.at[idx, "atr_pct"]        = atr["atr_pct"]
            df.at[idx, "avg_daily_move"] = atr.get("avg_daily_move")

            move_val = df.at[idx, "move_pct"]
            if pd.notna(move_val):
                cls = classify_move(float(move_val), atr["atr_pct"])
                df.at[idx, "stock_movement_atr_normalized"] = cls["normalized_move"]
                df.at[idx, "move_class_abs"]                = cls["move_class_abs"]
                df.at[idx, "move_class_norm"]               = cls["move_class_norm"]
                df.at[idx, "move_class_combo"]              = cls["move_class_combo"]

    return df


# ============================================================================
# Main pipeline
# ============================================================================

def run_discovery(
    existing_file:   str,
    candidates_file: str,
    rejected_file:   str,
    raw_file:        str  = "extended_pr_discovery_raw.csv",
    from_raw:        str  = None,
    limit:           int  = None,
    dry_run:         bool = False,
    min_move_pct:    float = 3.0,
) -> pd.DataFrame:

    # ----- Load existing dataset for deduplication -----
    print(f"Loading existing dataset: {existing_file}")
    existing    = pd.read_csv(existing_file)
    exist_dates = pd.to_datetime(existing["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    existing_keys = set(zip(existing["ticker"].str.upper(), exist_dates))
    print(f"  {len(existing):,} existing rows, {len(existing_keys):,} unique (ticker, date) keys")

    # =========================================================================
    # Stage 1 — Discovery
    # =========================================================================
    if from_raw:
        print(f"\nSkipping Stage 1 — loading raw results from: {from_raw}")
        raw_df = pd.read_csv(from_raw)
        print(f"  {len(raw_df):,} raw candidates")

    else:
        queries = DISCOVERY_QUERIES[:limit] if limit else DISCOVERY_QUERIES
        print(f"\n=== Stage 1: Discovery ({len(queries)} queries) ===")

        if dry_run:
            for i, (label, query) in enumerate(queries):
                print(f"  [{i+1}/{len(queries)}] {label}: {query[:70]}")
            print("\n[DRY RUN] No API calls made.")
            return pd.DataFrame()

        all_events: List[dict] = []

        for i, (label, query) in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] {label}")
            print(f"  Query: {query[:80]}...")

            events, citations, error = _call_perplexity_search(query)

            if error:
                print(f"  ERROR: {error}")
            elif events:
                print(f"  Found {len(events)} events ({len(citations)} citations)")
                for ev in events:
                    ev["_query_label"] = label
                all_events.extend(events)
            else:
                print("  No events returned")

            time.sleep(RATE_LIMIT_DELAY)

        if not all_events:
            print("\nNo events discovered.")
            return pd.DataFrame()

        raw_df = pd.DataFrame(all_events)
        raw_df.to_csv(raw_file, index=False)
        print(f"\nRaw discovery saved → {raw_file} ({len(raw_df):,} rows)")

    # =========================================================================
    # Stage 2 — Verification + Deduplication
    # =========================================================================
    print(f"\n=== Stage 2: Verification + Dedup ({len(raw_df):,} raw candidates) ===")

    accepted: List[dict] = []
    rejected: List[dict] = []
    seen_in_batch: set   = set()

    for _, row in raw_df.iterrows():
        ticker        = str(row.get("ticker", "")).strip().upper()
        event_date_raw = str(row.get("event_date", "")).strip()
        pr_url         = str(row.get("pr_url", "") or "").strip()
        headline       = str(row.get("headline", "") or "").strip()
        summary_txt    = str(row.get("summary", "") or "").strip()

        def _reject(reason: str, extra: dict = None):
            r = {**row.to_dict(), "_reject_reason": reason}
            if extra:
                r.update(extra)
            rejected.append(r)

        # --- Validate / normalise date ---
        event_date = _normalize_event_date(event_date_raw)
        if not event_date:
            _reject(f"invalid_date:{event_date_raw}")
            continue

        # pr_details will be populated lazily (once) below
        pr_details: Dict[str, str] = {
            "pr_date": "", "pr_title": "", "pr_key_info": "", "pr_fetch_error": ""
        }
        pr_already_fetched = False

        def _fetch_pr_once() -> None:
            nonlocal pr_details, pr_already_fetched
            if pr_already_fetched or not pr_url.startswith("http"):
                return
            print(f"  Fetching PR: {ticker} {event_date} — {pr_url[:70]}...", flush=True)
            pr_details = fetch_pr_details(pr_url)
            pr_already_fetched = True
            time.sleep(0.5)

        # --- Validate ticker ---
        # Note: str(None) → "NONE", so also reject those sentinel values
        if not ticker or ticker in ("NONE", "NULL", "N/A", "NA", "TBD") \
                or len(ticker) > 5 or not ticker.isalpha():
            # Try to recover ticker from PR text before rejecting
            if pr_url.startswith("http"):
                _fetch_pr_once()
                pr_text_tmp = pr_details.get("pr_key_info", "") + " " + pr_details.get("pr_title", "")
                recovered = _extract_ticker_from_pr(pr_text_tmp)
                if recovered and len(recovered) <= 5 and recovered.isalpha() \
                        and recovered not in ("NONE", "NULL"):
                    ticker = recovered
                    print(f"  Recovered ticker from PR: {recovered}", flush=True)
                else:
                    _reject(f"invalid_ticker:{ticker}")
                    continue
            else:
                _reject(f"invalid_ticker:{ticker}")
                continue

        # --- Deduplicate against existing dataset ---
        if (ticker, event_date) in existing_keys:
            _reject("already_in_dataset")
            continue

        # --- Deduplicate within this batch ---
        batch_key = (ticker, event_date)
        if batch_key in seen_in_batch:
            _reject("batch_duplicate")
            continue
        seen_in_batch.add(batch_key)

        # --- Fetch and parse the press release (if not already done above) ---
        _fetch_pr_once()

        # --- Relevance filter ---
        combined = " ".join([headline, summary_txt, pr_details.get("pr_key_info", "")])
        if not _is_relevant(combined):
            _reject("relevance_filter", {"_pr_title": pr_details.get("pr_title", "")})
            continue

        # --- Attempt ticker confirmation from PR text ---
        pr_text = pr_details.get("pr_key_info", "") + " " + pr_details.get("pr_title", "")
        confirmed_ticker = _extract_ticker_from_pr(pr_text, fallback=ticker)
        if confirmed_ticker and confirmed_ticker != ticker:
            print(f"    Ticker: AI said {ticker}, PR says {confirmed_ticker} → using {confirmed_ticker}")
            ticker = confirmed_ticker

        # --- Determine catalyst_type column value ---
        ct_raw = str(row.get("catalyst_type", "")).lower()
        if "fda" in ct_raw or "approval" in ct_raw or "rejection" in ct_raw:
            catalyst_type_col = "FDA Decision"
        else:
            catalyst_type_col = "Clinical Data"

        accepted.append({
            # Core event fields
            "ticker":               ticker,
            "event_date":           event_date,
            "event_type":           None,   # filled after OHLC
            "move_pct":             None,
            "price_at_event":       None,
            # Catalyst metadata
            "catalyst_type":        catalyst_type_col,
            "catalyst_summary":     summary_txt or headline,
            "drug_name":            str(row.get("drug_name", "") or ""),
            "nct_id":               "",
            "indication":           str(row.get("indication", "") or ""),
            "is_pivotal":           None,
            "pivotal_evidence":     "",
            "primary_endpoint_met": row.get("primary_endpoint_met"),
            "primary_endpoint_result": "",
            # CT.gov fields — blank, backfillable via fix_missing_nct.py
            "ct_official_title":    "",
            "ct_phase":             str(row.get("phase", "") or ""),
            "ct_enrollment":        None,
            "ct_conditions":        "",
            "ct_status":            "",
            "ct_sponsor":           "",
            "ct_allocation":        "",
            "ct_primary_completion": "",
            # Financials — blank (stale anyway; can be backfilled)
            "market_cap_m":            None,
            "current_price":           None,
            "cash_position_m":         None,
            "short_percent":           None,
            "institutional_ownership": None,
            "analyst_target":          None,
            "analyst_rating":          "",
            # ATR / move — filled after OHLC
            "atr_pct":                      None,
            "stock_movement_atr_normalized": None,
            "avg_daily_move":               None,
            "move_class_abs":               None,
            "move_class_norm":              None,
            "move_class_combo":             None,
            # Trading columns — filled after OHLC
            "event_trading_date": event_date,
            "move_2d_pct":        None,
            "price_before":       None,
            "price_after":        None,
            "stock_relative_move": None,
            "data_complete":      True,
            # Validation columns
            "v_is_verified": True,
            "v_actual_date": "",
            "v_pr_link":     pr_url,
            "v_pr_date":     pr_details.get("pr_date", ""),
            "v_pr_title":    pr_details.get("pr_title", ""),
            "v_pr_key_info": pr_details.get("pr_key_info", ""),
            "v_is_material": row.get("is_material"),
            "v_confidence":  str(row.get("confidence", "medium") or "medium"),
            "v_summary":     summary_txt,
            "v_error":       pr_details.get("pr_fetch_error", ""),
            "v_action":      "DISCOVERED",
            # Internal (excluded from final output)
            "_query_label":  str(row.get("_query_label", "")),
        })

    print(f"\n  Accepted: {len(accepted):,}  |  Rejected: {len(rejected):,}")

    if not accepted:
        print("No accepted candidates — nothing to enrich.")
        if rejected:
            pd.DataFrame(rejected).to_csv(rejected_file, index=False)
            print(f"Rejected saved → {rejected_file}")
        return pd.DataFrame()

    # =========================================================================
    # Stage 3 — OHLC Enrichment
    # =========================================================================
    print(f"\n=== Stage 3: OHLC Enrichment ({len(accepted):,} candidates) ===")
    candidates_df = pd.DataFrame(accepted)
    candidates_df = _enrich_with_ohlc(candidates_df)

    # Apply minimum move filter (keep rows without OHLC — can't filter what we don't have)
    has_move   = candidates_df["move_pct"].notna()
    big_enough = has_move & (candidates_df["move_pct"].abs() >= min_move_pct)
    no_ohlc    = ~has_move

    final_df        = candidates_df[big_enough | no_ohlc].copy()
    small_move_df   = candidates_df[has_move & ~big_enough].copy()

    if not small_move_df.empty:
        small_move_df["_reject_reason"] = f"move_pct < {min_move_pct}%"
        rejected.extend(small_move_df.to_dict("records"))

    print(f"\n  After move filter (>= {min_move_pct}%): {len(final_df):,} kept, "
          f"{len(small_move_df):,} rejected for small/zero move")

    # =========================================================================
    # Stage 4 — Output
    # =========================================================================
    # Align column order to existing dataset; exclude internal _* columns
    existing_cols = list(existing.columns)
    final_cols    = [c for c in existing_cols if c in final_df.columns] + \
                    [c for c in final_df.columns
                     if c not in existing_cols and not c.startswith("_")]
    final_df = final_df[[c for c in final_cols if c in final_df.columns]]

    final_df.to_csv(candidates_file, index=False)
    print(f"\nCandidates saved → {candidates_file} ({len(final_df):,} rows)")

    if rejected:
        rejected_out = pd.DataFrame(rejected)
        rejected_out.to_csv(rejected_file, index=False)
        print(f"Rejected saved    → {rejected_file} ({len(rejected_out):,} rows)")

    # ----- Summary -----
    print(f"\n{'='*60}")
    print("DISCOVERY SUMMARY")
    print(f"{'='*60}")
    if not from_raw:
        print(f"Queries run:          {len(DISCOVERY_QUERIES[:limit] if limit else DISCOVERY_QUERIES):,}")
        print(f"Raw events found:     {len(raw_df):,}")
    print(f"After dedup/filter:   {len(accepted):,}")
    print(f"After move filter:    {len(final_df):,}")
    if "move_class_norm" in final_df.columns:
        mc = final_df["move_class_norm"].value_counts()
        if not mc.empty:
            print("Move class breakdown:")
            for cls, cnt in mc.items():
                print(f"  {cls:<10}  {cnt:,}")
    if "v_action" in final_df.columns:
        print(f"\nAll new rows marked v_action=DISCOVERED")
    print(f"\nNext step: review {candidates_file}, then merge into clean_v2 if satisfied.")

    return final_df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extend dataset with new clinical catalyst events via Perplexity discovery"
    )
    parser.add_argument(
        "--existing", default="enriched_all_clinical_clean_v2.csv",
        help="Existing clean dataset (for deduplication)",
    )
    parser.add_argument(
        "--candidates", default="extended_relevant_clinical_candidates.csv",
        help="Output CSV for accepted new candidates",
    )
    parser.add_argument(
        "--rejected", default="extended_relevant_clinical_rejected.csv",
        help="Output CSV for rejected candidates (debug)",
    )
    parser.add_argument(
        "--raw", default="extended_pr_discovery_raw.csv",
        help="Intermediate raw discovery output (saved after Stage 1)",
    )
    parser.add_argument(
        "--from-raw", default=None,
        help="Skip Stage 1 and load raw results from this CSV",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of discovery queries (useful for testing)",
    )
    parser.add_argument(
        "--min-move", type=float, default=3.0,
        help="Minimum abs(move_pct) to keep a candidate (default: 3.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print query list without making any API calls",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    run_discovery(
        existing_file   = args.existing,
        candidates_file = args.candidates,
        rejected_file   = args.rejected,
        raw_file        = args.raw,
        from_raw        = args.from_raw,
        limit           = args.limit,
        dry_run         = args.dry_run,
        min_move_pct    = args.min_move,
    )

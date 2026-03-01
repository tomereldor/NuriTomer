"""Find clinical trial result events using a news-first approach.

Pipeline:
  1. Fetch completed Phase 2/3 trials from ClinicalTrials.gov in a date range
  2. Map each trial's sponsor to a stock ticker (via universe file)
  3. Get the stock move on/around the completion date
  4. Output: CSV of clinical events with stock moves (all move sizes)

This captures Low/Medium/High moves for the same event type, enabling
balanced ML training that the original scan-for-big-moves approach can't provide.

Usage:
    python -m scripts.find_clinical_events
    python -m scripts.find_clinical_events --start 2023-01-01 --end 2025-12-31
"""

import os
import sys
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# ClinicalTrials.gov helpers
# ---------------------------------------------------------------------------

def _fetch_ct_page(params: dict):
    """Fetch one page of CT.gov results. Returns (studies, next_page_token)."""
    url = "https://clinicaltrials.gov/api/v2/studies"
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            print(f"  CT.gov HTTP {r.status_code}")
            return [], None
        data = r.json()
        return data.get("studies", []), data.get("nextPageToken")
    except Exception as e:
        print(f"  CT.gov fetch error: {e}")
        return [], None


def _parse_study(study: dict) -> dict:
    """Extract fields we need from a CT.gov study JSON."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    cond = proto.get("conditionsModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    arms = proto.get("armsInterventionsModule", {})

    interventions = [i.get("name", "") for i in arms.get("interventions", [])]

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", ""),
        "phase": "/".join(design.get("phases", [])),
        "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
        "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "conditions": "; ".join(cond.get("conditions", [])),
        "enrollment": design.get("enrollmentInfo", {}).get("count", 0) or 0,
        "interventions": "; ".join(interventions[:3]),
    }


def get_trial_completions(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Phase 2 and Phase 3 trials that completed in [start_date, end_date].

    Uses CT.gov API v2 with RANGE filter and automatic pagination.
    """
    # CT.gov v2 uses filter.advanced with AREA syntax for date ranges
    date_filter = f"AREA[CompletionDate]RANGE[{start_date},{end_date}]"

    all_studies = []
    for phase in ["PHASE2", "PHASE3"]:
        print(f"  Fetching {phase} completions {start_date}–{end_date}...")
        params = {
            "query.term": f"AREA[Phase]({phase})",
            "filter.overallStatus": "COMPLETED",
            "filter.advanced": date_filter,
            "pageSize": 1000,
            "format": "json",
        }

        page = 0
        while True:
            studies, next_token = _fetch_ct_page(params)
            all_studies.extend(studies)
            page += 1

            if not next_token:
                break
            params["pageToken"] = next_token
            time.sleep(0.5)

        print(f"    {phase}: {len(all_studies)} cumulative")
        time.sleep(1)

    if not all_studies:
        return pd.DataFrame()

    rows = [_parse_study(s) for s in all_studies]
    df = pd.DataFrame(rows)

    # Drop rows without completion dates or NCT IDs
    df = df[df["nct_id"].str.startswith("NCT", na=False)]
    df = df[df["completion_date"].notna() & (df["completion_date"] != "")]
    df = df.drop_duplicates("nct_id")

    print(f"  Total unique completed trials: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Sponsor → Ticker mapping
# ---------------------------------------------------------------------------

KNOWN_SPONSORS = {
    "pfizer": "PFE", "novartis": "NVS", "roche": "RHHBY", "merck": "MRK",
    "bristol-myers squibb": "BMY", "abbvie": "ABBV", "eli lilly": "LLY",
    "johnson & johnson": "JNJ", "astrazeneca": "AZN", "sanofi": "SNY",
    "gilead": "GILD", "amgen": "AMGN", "biogen": "BIIB", "vertex": "VRTX",
    "regeneron": "REGN", "moderna": "MRNA", "biontech": "BNTX",
    "alnylam": "ALNY", "ionis": "IONS", "bluebird": "BLUE",
    "blueprint medicines": "BPMC", "sage therapeutics": "SAGE",
}


def load_universe_mapping(universe_file: str):
    """
    Build name→ticker, ticker set, and ticker→market_cap from the universe CSV.
    Returns (name_to_ticker dict, ticker set, ticker_to_cap dict).
    """
    if not os.path.exists(universe_file):
        return {}, set(), {}

    df = pd.read_csv(universe_file)
    tickers = set(df["ticker"].str.upper().unique())
    name_map: dict = {}
    ticker_to_cap: dict = {}

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        name = str(row.get("name", "")).lower().strip()
        cap = row.get("market_cap_m")
        if pd.notna(cap):
            ticker_to_cap[ticker] = round(float(cap), 1)
        if name and name != "nan":
            name_map[name] = ticker
            first_word = name.split()[0] if name.split() else ""
            if len(first_word) > 3:
                name_map.setdefault(first_word, ticker)

    return name_map, tickers, ticker_to_cap


def map_sponsor_to_ticker(sponsor: str, name_map: dict, ticker_set: set):
    """Try to resolve a sponsor string to a stock ticker."""
    if not sponsor:
        return None

    sponsor_lower = sponsor.lower().strip()

    # 1. Known hard-coded mappings
    for key, ticker in KNOWN_SPONSORS.items():
        if key in sponsor_lower:
            return ticker

    # 2. Direct name match in universe
    if sponsor_lower in name_map:
        return name_map[sponsor_lower]

    # 3. Partial match: sponsor is a substring of a universe name, or vice versa
    for name, ticker in name_map.items():
        if len(name) > 4 and (name in sponsor_lower or sponsor_lower in name):
            return ticker

    # 4. If sponsor looks like an all-caps ticker symbol itself
    clean = sponsor.replace(" ", "").replace(",", "").replace(".", "")
    if clean.isupper() and 1 < len(clean) <= 5 and clean in ticker_set:
        return clean

    return None


# ---------------------------------------------------------------------------
# Stock move calculation
# ---------------------------------------------------------------------------

def get_stock_move(ticker: str, event_date: str, window_days: int = 7) -> dict:
    """
    Get 1-day stock move around event_date.

    Looks for the first trading day on or after event_date (within window_days)
    and computes: close(T) / close(T-1) - 1.
    """
    try:
        event_dt = pd.to_datetime(event_date)
        start = (event_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (event_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return {"move_pct": None, "error": "Insufficient price data"}

        # Flatten multi-level columns (yfinance >= 0.2.31)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)

        # Find first trading day >= event_date
        future = df.index[df.index >= event_dt]
        if len(future) == 0:
            return {"move_pct": None, "error": "No trading day after event date"}

        event_idx = df.index.get_loc(future[0])
        if event_idx == 0:
            return {"move_pct": None, "error": "No prior trading day in window"}

        prev_close = df["Close"].iloc[event_idx - 1]
        event_close = df["Close"].iloc[event_idx]

        # Handle Series (older yfinance)
        prev_close = float(prev_close.iloc[0]) if hasattr(prev_close, "iloc") else float(prev_close)
        event_close = float(event_close.iloc[0]) if hasattr(event_close, "iloc") else float(event_close)

        if prev_close == 0:
            return {"move_pct": None, "error": "Zero prev_close"}

        move_1d = (event_close - prev_close) / prev_close * 100

        # Optional 2-day move (T+1)
        move_2d = None
        if event_idx + 1 < len(df):
            next_close = df["Close"].iloc[event_idx + 1]
            next_close = float(next_close.iloc[0]) if hasattr(next_close, "iloc") else float(next_close)
            move_2d = (next_close - prev_close) / prev_close * 100

        return {
            "move_pct": round(move_1d, 2),
            "move_2d_pct": round(move_2d, 2) if move_2d is not None else None,
            "price_before": round(prev_close, 2),
            "price_after": round(event_close, 2),
            "event_trading_date": future[0].strftime("%Y-%m-%d"),
        }

    except Exception as e:
        return {"move_pct": None, "error": str(e)[:80]}


def classify_move_simple(move_pct) -> str:
    """Simple absolute move classifier (no ATR). High ≥15%, Medium 5–15%, Low <5%."""
    if move_pct is None:
        return "Unknown"
    abs_m = abs(move_pct)
    if abs_m >= 15:
        return "High"
    elif abs_m >= 5:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_clinical_events(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    universe_file: str = "biotech_universe_expanded.csv",
    existing_file: str = "enriched_high_moves.csv",
    output_file: str = "clinical_events_new.csv",
):
    """
    Fetch CT.gov completed Phase 2/3 trials, map to tickers, get stock moves.
    Skips events already in existing_file (matched on ticker + nct_id).
    """

    # Load universe
    name_map, ticker_set, ticker_to_cap = load_universe_mapping(universe_file)
    print(f"Universe: {len(ticker_set)} tickers loaded")

    # Load existing to skip duplicates
    existing_keys: set = set()
    if os.path.exists(existing_file):
        existing = pd.read_csv(existing_file)
        # Key on (ticker, nct_id) — skip if we already have this trial
        for _, row in existing.iterrows():
            nct = str(row.get("nct_id", "")).strip()
            if nct and nct.startswith("NCT"):
                existing_keys.add((str(row["ticker"]).upper(), nct))
        print(f"Existing events with NCT IDs: {len(existing_keys)}")

    # Fetch trial completions
    print(f"\nFetching trial completions {start_date} → {end_date}...")
    trials = get_trial_completions(start_date, end_date)

    if trials.empty:
        print("No trials found. Exiting.")
        return pd.DataFrame()

    # Map sponsors → tickers and get stock moves
    print(f"\nMapping {len(trials)} trials to tickers and fetching price moves...")
    events = []
    no_ticker = 0
    no_price = 0

    for i, (_, trial) in enumerate(trials.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(trials)} | events found: {len(events)}")

        ticker = map_sponsor_to_ticker(trial["sponsor"], name_map, ticker_set)
        if not ticker:
            no_ticker += 1
            continue

        # Skip already-known events
        if (ticker, trial["nct_id"]) in existing_keys:
            continue

        # Use primary_completion_date if available (closer to results announcement)
        date = trial["primary_completion_date"] or trial["completion_date"]
        if not date:
            continue

        move = get_stock_move(ticker, date)

        events.append({
            "ticker": ticker,
            "event_date": date,
            "event_trading_date": move.get("event_trading_date"),
            "nct_id": trial["nct_id"],
            "title": trial["title"],
            "phase": trial["phase"],
            "sponsor": trial["sponsor"],
            "conditions": trial["conditions"],
            "enrollment": trial["enrollment"],
            "interventions": trial["interventions"],
            "market_cap_m": ticker_to_cap.get(ticker),
            "move_pct": move.get("move_pct"),
            "move_2d_pct": move.get("move_2d_pct"),
            "price_before": move.get("price_before"),
            "price_after": move.get("price_after"),
            "move_class": classify_move_simple(move.get("move_pct")),
            "event_type": "Gainer" if (move.get("move_pct") or 0) > 0 else "Loser",
            "catalyst_type": "Clinical Data",
            "price_error": move.get("error", ""),
        })

        if move.get("move_pct") is None:
            no_price += 1

        time.sleep(0.15)

    df = pd.DataFrame(events)

    if df.empty:
        print("\nNo events found — check universe file and date range.")
        return df

    # Save all (including no-price rows for inspection)
    df.to_csv(output_file, index=False)

    df_valid = df[df["move_pct"].notna()]

    print(f"\n{'='*55}")
    print(f"CLINICAL EVENTS FOUND")
    print(f"{'='*55}")
    print(f"Trials searched:        {len(trials)}")
    print(f"  No ticker match:      {no_ticker}")
    print(f"  Skipped (existing):   {len(existing_keys)}")
    print(f"Events recorded:        {len(df)}")
    print(f"  With valid move data: {len(df_valid)}")
    print(f"  No price data:        {no_price}")
    if len(df_valid) > 0:
        print(f"\nMove class distribution:")
        print(df_valid["move_class"].value_counts().to_string())
        print(f"\nTop gainers:")
        print(df_valid.nlargest(5, "move_pct")[["ticker", "event_date", "move_pct", "conditions"]].to_string(index=False))
        print(f"\nTop losers:")
        print(df_valid.nsmallest(5, "move_pct")[["ticker", "event_date", "move_pct", "conditions"]].to_string(index=False))
    print(f"\nSaved to: {output_file}")

    return df_valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find clinical trial events (news-first)")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--universe", default="biotech_universe_expanded.csv")
    parser.add_argument("--existing", default="enriched_high_moves.csv")
    parser.add_argument("--output", default="clinical_events_new.csv")
    args = parser.parse_args()

    find_clinical_events(
        start_date=args.start,
        end_date=args.end,
        universe_file=args.universe,
        existing_file=args.existing,
        output_file=args.output,
    )

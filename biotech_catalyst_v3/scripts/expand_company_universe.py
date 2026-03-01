"""Expand biotech company universe from multiple sources.

Sources (in priority order):
  1. SPDR XBI holdings - fetched live from ssga.com XLSX (currently 144 holdings)
  2. Nasdaq stock screener - all biotech/pharma/drug listed stocks (~800 tickers)
  3. Existing dataset tickers (always included)
  4. ClinicalTrials.gov sponsors (logged for coverage info, hard to map to tickers)

The Nasdaq screener is a superset of IBB and ARKG since both are fully listed on
US exchanges. No separate IBB/ARKG download needed.

Usage:
    python -m scripts.expand_company_universe
    python -m scripts.expand_company_universe --min-cap 50 --max-cap 10000
"""

import os
import sys
import io
import time
import requests
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Warrant/rights suffixes to exclude (e.g. AIMDW, RCKTW)
_WARRANT_SUFFIXES = ("W", "R", "U", "Z")

SPDR_XBI_URL = (
    "https://www.ssga.com/us/en/intermediary/etfs/library-content"
    "/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx"
)
NASDAQ_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=5000&download=true"
)
NASDAQ_SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
}


# ---------------------------------------------------------------------------
# Source 1: SPDR XBI holdings (live XLSX download)
# ---------------------------------------------------------------------------

def get_xbi_holdings() -> list:
    """Fetch current XBI holdings from SPDR XLSX. Returns list of ticker strings."""
    try:
        print("  Fetching XBI holdings from SPDR...")
        resp = requests.get(SPDR_XBI_URL, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), skiprows=4, engine="openpyxl")
        tickers = (
            df["Ticker"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )
        tickers = [t for t in tickers if _is_valid_ticker(t)]
        print(f"  XBI: {len(tickers)} holdings")
        return tickers
    except Exception as e:
        print(f"  XBI fetch failed: {e} — using empty list")
        return []


# ---------------------------------------------------------------------------
# Source 2: Nasdaq biotech/pharma screener
# ---------------------------------------------------------------------------

def get_nasdaq_biotech_tickers() -> list:
    """
    Fetch all biotech/pharma/drug tickers from Nasdaq stock screener.
    Covers XBI, IBB, ARKG universes and more.
    """
    KEYWORDS = ["biotech", "pharma", "drug", "life science"]
    try:
        print("  Fetching Nasdaq biotech/pharma screener...")
        resp = requests.get(NASDAQ_SCREENER_URL, timeout=30,
                            headers=NASDAQ_SCREENER_HEADERS)
        resp.raise_for_status()
        rows = resp.json().get("data", {}).get("rows", [])
        tickers = [
            r["symbol"].strip().upper()
            for r in rows
            if any(k in r.get("industry", "").lower() for k in KEYWORDS)
            and _is_valid_ticker(r.get("symbol", ""))
        ]
        tickers = sorted(set(tickers))
        print(f"  Nasdaq screener: {len(tickers)} biotech/pharma tickers")
        return tickers
    except Exception as e:
        print(f"  Nasdaq screener failed: {e} — using empty list")
        return []


def _is_valid_ticker(symbol: str) -> bool:
    """Accept 1-5 char alphabetic tickers, reject warrants/rights/units."""
    s = symbol.strip().upper()
    if not s or not s.isalpha() or len(s) > 5:
        return False
    # Reject common warrant/rights suffixes (e.g. RCKTW, SRZN W)
    if len(s) >= 2 and s.endswith(_WARRANT_SUFFIXES) and s[:-1].isalpha():
        return False
    return True


# ---------------------------------------------------------------------------
# Source 3: Existing dataset tickers (always include)
# ---------------------------------------------------------------------------

def load_existing_tickers(file: str = "enriched_high_moves.csv") -> set:
    if os.path.exists(file):
        df = pd.read_csv(file)
        return set(df["ticker"].str.upper().unique())
    return set()


# ---------------------------------------------------------------------------
# Source 4: ClinicalTrials.gov sponsors (informational)
# ---------------------------------------------------------------------------

def get_clinicaltrials_sponsors(limit: int = 500) -> list:
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": "AREA[Phase](PHASE2 OR PHASE3)",
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
        "pageSize": min(limit, 1000),
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            return []
        sponsors = set()
        for study in r.json().get("studies", []):
            name = (study.get("protocolSection", {})
                    .get("sponsorCollaboratorsModule", {})
                    .get("leadSponsor", {})
                    .get("name", ""))
            if name:
                sponsors.add(name)
        return sorted(sponsors)
    except Exception as e:
        print(f"  CT.gov error: {e}")
        return []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_biotech(ticker: str, min_cap: float, max_cap: float):
    """Validate ticker via yfinance. Returns dict or None if filtered out."""
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return None

        mkt_cap_m = (info.get("marketCap") or 0) / 1e6
        if mkt_cap_m < min_cap or (max_cap > 0 and mkt_cap_m > max_cap):
            return None

        sector = (info.get("sector") or "").lower()
        industry = (info.get("industry") or "").lower()
        if sector and "health" not in sector:
            if not any(k in industry for k in ["biotech", "pharma", "drug", "diagnostic", "life science", "medical"]):
                return None

        return {
            "ticker": ticker,
            "name": info.get("shortName", ""),
            "market_cap_m": round(mkt_cap_m, 1),
            "current_price": round(float(price), 2),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "employees": info.get("fullTimeEmployees") or 0,
            "country": info.get("country", ""),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def expand_universe(
    min_cap: float = 50,
    max_cap: float = 10000,
    existing_file: str = "enriched_high_moves.csv",
    output_file: str = "biotech_universe_expanded.csv",
):
    existing = load_existing_tickers(existing_file)
    print(f"Existing dataset tickers: {len(existing)}\n")

    print("Gathering ticker candidates...")
    all_tickers: set = set(existing)

    xbi = get_xbi_holdings()
    all_tickers.update(xbi)

    nasdaq = get_nasdaq_biotech_tickers()
    all_tickers.update(nasdaq)

    print(f"\nTotal unique candidates: {len(all_tickers)}")
    print(f"  From existing dataset: {len(existing)}")
    print(f"  From XBI:             {len(xbi)}")
    print(f"  From Nasdaq screener: {len(nasdaq)}")

    print("\nFetching CT.gov sponsors (coverage info)...")
    sponsors = get_clinicaltrials_sponsors()
    print(f"  {len(sponsors)} active Phase 2/3 sponsors found")

    print(f"\nValidating {len(all_tickers)} candidates (${min_cap}M – ${max_cap}M mkt cap)...")
    valid = []
    for i, ticker in enumerate(sorted(all_tickers)):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(all_tickers)} | valid: {len(valid)}")
        result = validate_biotech(ticker, min_cap, max_cap)
        if result:
            result["is_existing"] = ticker in existing
            valid.append(result)
        time.sleep(0.1)

    df = pd.DataFrame(valid).sort_values("market_cap_m", ascending=False).reset_index(drop=True)
    df.to_csv(output_file, index=False)

    new_count = (~df["is_existing"]).sum()
    print(f"\n{'='*55}")
    print(f"UNIVERSE EXPANSION COMPLETE")
    print(f"{'='*55}")
    print(f"Total validated:  {len(df)}")
    print(f"  Existing:       {df['is_existing'].sum()}")
    print(f"  New:            {new_count}")
    if len(df):
        print(f"Market cap range: ${df['market_cap_m'].min():.0f}M – ${df['market_cap_m'].max():.0f}M")
    print(f"Saved to: {output_file}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expand biotech company universe")
    parser.add_argument("--min-cap", type=float, default=50, help="Min market cap $M (default 50)")
    parser.add_argument("--max-cap", type=float, default=10000, help="Max market cap $M (default 10000)")
    parser.add_argument("--existing-file", default="enriched_high_moves.csv")
    parser.add_argument("--output", default="biotech_universe_expanded.csv")
    args = parser.parse_args()

    expand_universe(
        min_cap=args.min_cap,
        max_cap=args.max_cap,
        existing_file=args.existing_file,
        output_file=args.output,
    )

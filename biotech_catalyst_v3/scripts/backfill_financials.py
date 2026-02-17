"""
Backfill missing financial data using yfinance.

The original FinancialDataFetcher in batch_enrichment.py was never implemented,
so ~113/265 rows are missing market_cap_m, current_price, cash_position_m, etc.
This script fills them in using yfinance Ticker.info.

Usage:
    python -m scripts.backfill_financials
    python -m scripts.backfill_financials --input enriched_final.csv --output enriched_final.csv
    python -m scripts.backfill_financials --dry-run
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Map from yfinance info keys to our CSV columns
FIELD_MAP = {
    "market_cap_m": ("marketCap", 1e6),           # yf returns raw, we want millions
    "current_price": (["currentPrice", "regularMarketPrice"], 1),
    "cash_position_m": ("totalCash", 1e6),
    "short_percent": ("shortPercentOfFloat", 1),   # already 0-1
    "institutional_ownership": ("heldPercentInstitutions", 1),
    "analyst_target": ("targetMeanPrice", 1),
}


def _get_field(info: dict, key_or_keys, divisor: float):
    """Extract a value from yfinance info dict, dividing by divisor."""
    keys = key_or_keys if isinstance(key_or_keys, list) else [key_or_keys]
    for k in keys:
        val = info.get(k)
        if val is not None:
            try:
                return round(float(val) / divisor, 2)
            except (ValueError, TypeError):
                continue
    return None


def _get_analyst_rating(info: dict) -> str:
    """Extract analyst recommendation string."""
    return info.get("recommendationKey", "") or ""


def _get_cash_runway(info: dict) -> int:
    """Estimate cash runway in months from totalCash and quarterly burn."""
    cash = info.get("totalCash")
    # Try operating cash flow (negative = burning cash)
    ocf = info.get("operatingCashflow")
    if cash and ocf and ocf < 0:
        quarterly_burn = abs(ocf) / 4
        if quarterly_burn > 0:
            months = int(cash / (quarterly_burn / 3))
            return max(0, months)
    return None


def fetch_financials_for_ticker(ticker: str) -> dict:
    """Fetch all financial fields for a single ticker. Returns dict of column->value."""
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None:
            return {"_error": f"No data returned for {ticker}"}
    except Exception as e:
        return {"_error": str(e)}

    result = {}
    for col, (key_or_keys, divisor) in FIELD_MAP.items():
        result[col] = _get_field(info, key_or_keys, divisor)

    result["analyst_rating"] = _get_analyst_rating(info)
    result["cash_runway_months"] = _get_cash_runway(info)

    return result


def backfill_financials(
    input_file: str = "enriched_final.csv",
    output_file: str = "enriched_final.csv",
    dry_run: bool = False,
    save_every: int = 20,
    delay: float = 0.3,
):
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Find rows missing ANY of the main 3 financial columns
    main_cols = ["market_cap_m", "current_price", "cash_position_m"]
    missing_mask = df[main_cols].isna().any(axis=1)
    missing_idx = df[missing_mask].index.tolist()

    already_complete = len(df) - len(missing_idx)
    print(f"Already complete (all 3 main financial cols): {already_complete}/{len(df)}")
    print(f"Need backfill: {len(missing_idx)}")

    if dry_run:
        # Get unique tickers
        tickers = df.loc[missing_idx, "ticker"].unique()
        print(f"\nUnique tickers to fetch: {len(tickers)}")
        print(f"  {sorted(tickers)[:20]}{'...' if len(tickers) > 20 else ''}")
        return

    # Group by ticker so we only fetch each ticker once
    ticker_data_cache = {}
    unique_tickers = df.loc[missing_idx, "ticker"].unique()
    print(f"\nFetching data for {len(unique_tickers)} unique tickers...")

    for i, ticker in enumerate(sorted(unique_tickers)):
        print(f"  [{i+1}/{len(unique_tickers)}] {ticker}...", end=" ", flush=True)
        result = fetch_financials_for_ticker(ticker)

        if "_error" in result:
            print(f"ERROR: {result['_error'][:60]}")
        else:
            filled = sum(1 for v in result.values() if v is not None and v != "")
            print(f"OK ({filled} fields)")

        ticker_data_cache[ticker] = result
        time.sleep(delay)

    # Apply cached data to all missing rows
    filled_count = 0
    for idx in missing_idx:
        ticker = df.at[idx, "ticker"]
        data = ticker_data_cache.get(ticker, {})

        if "_error" in data:
            continue

        row_filled = False
        for col, val in data.items():
            if col.startswith("_"):
                continue
            if val is not None and pd.isna(df.at[idx, col]):
                df.at[idx, col] = val
                row_filled = True

        if row_filled:
            filled_count += 1

    # Recalculate data_quality_score for affected rows
    print(f"\nRecalculating quality scores for updated rows...")
    _recalc_quality_scores(df)

    # Save
    df.to_csv(output_file, index=False)

    # Summary
    after_missing = df[main_cols].isna().any(axis=1).sum()
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Rows updated:          {filled_count}/{len(missing_idx)}")
    print(f"Still missing financials: {after_missing}/{len(df)}")
    for col in main_cols:
        filled = df[col].notna().sum()
        print(f"  {col:25s} {filled:3d}/{len(df)}")
    print(f"Output:                {output_file}")


def _recalc_quality_scores(df: pd.DataFrame):
    """Recalculate data_quality_score based on field completeness."""
    scored_cols = [
        "catalyst_summary", "catalyst_type", "drug_name", "nct_id",
        "phase", "ct_sponsor", "ct_enrollment",
        "market_cap_m", "current_price", "cash_position_m",
        "press_release_url",
    ]
    existing = [c for c in scored_cols if c in df.columns]

    for idx in df.index:
        filled = 0
        for col in existing:
            val = df.at[idx, col]
            if pd.notna(val) and str(val).strip() not in ("", "Unknown"):
                filled += 1
        df.at[idx, "data_quality_score"] = round(filled / len(existing), 2)

    # Recalc threshold
    if "data_quality_threshold_passed" in df.columns:
        df["data_quality_threshold_passed"] = df["data_quality_score"] >= 0.7


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill missing financial data")
    parser.add_argument("--input", default="enriched_final.csv")
    parser.add_argument("--output", default="enriched_final.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between API calls")
    args = parser.parse_args()

    backfill_financials(
        input_file=args.input,
        output_file=args.output,
        dry_run=args.dry_run,
        delay=args.delay,
    )

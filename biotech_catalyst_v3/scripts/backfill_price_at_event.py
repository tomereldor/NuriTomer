"""Backfill price_at_event, price_before, price_after, move_2d_pct via yfinance.

What this script does
---------------------
For rows missing price_at_event, price_before, price_after, or move_2d_pct:
  - Downloads OHLC data for each ticker using the existing bulk downloader
    (with local parquet cache, so tickers already cached are instant)
  - price_at_event  = closing price on event_trading_date
  - price_before    = closing price on the last trading day BEFORE the event
  - price_after     = closing price on the first trading day AFTER the event
  - move_2d_pct     = (price_after - price_before) / price_before * 100

Coverage
--------
After running cleanup_columns.py, ~400 rows already have price_at_event filled
from the close column. This script fills the remaining ~1,670 rows that require
a live yfinance lookup.

Requires
--------
  pip install yfinance pandas pyarrow requests
  Active internet connection

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3
    python -m scripts.backfill_price_at_event

    # Only backfill price_at_event (skip 2-day move)
    python -m scripts.backfill_price_at_event --skip-2d

    # Force re-download even if cached
    python -m scripts.backfill_price_at_event --force-refresh
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ohlc_cache import load_ohlc_bulk, date_range_for_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_close_on_date(ohlc_df: pd.DataFrame, target_date: str) -> float | None:
    """Return the closing price on or nearest-to target_date (looks forward up to 3 days)."""
    if ohlc_df is None or ohlc_df.empty:
        return None
    target = pd.Timestamp(target_date).normalize()
    for delta in range(4):
        check_date = target + pd.Timedelta(days=delta)
        mask = ohlc_df.index.normalize() == check_date
        if mask.any():
            val = ohlc_df.loc[mask, "Close"].iloc[-1]
            return round(float(val), 4) if pd.notna(val) else None
    return None


def _get_close_before(ohlc_df: pd.DataFrame, target_date: str) -> float | None:
    """Return the last closing price strictly before target_date."""
    if ohlc_df is None or ohlc_df.empty:
        return None
    target = pd.Timestamp(target_date).normalize()
    pre = ohlc_df[ohlc_df.index.normalize() < target]
    if pre.empty:
        return None
    val = pre["Close"].iloc[-1]
    return round(float(val), 4) if pd.notna(val) else None


def _get_close_after(ohlc_df: pd.DataFrame, target_date: str) -> float | None:
    """Return the first closing price strictly after target_date."""
    if ohlc_df is None or ohlc_df.empty:
        return None
    target = pd.Timestamp(target_date).normalize()
    post = ohlc_df[ohlc_df.index.normalize() > target]
    if post.empty:
        return None
    val = post["Close"].iloc[0]
    return round(float(val), 4) if pd.notna(val) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def backfill_prices(
    input_file: str = "enriched_all_clinical.csv",
    skip_2d: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    print(f"Loading {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    col_price = "price_at_event"
    col_before = "price_before"
    col_after  = "price_after"
    col_2d     = "move_2d_pct"

    # Ensure columns exist
    for col in [col_price, col_before, col_after, col_2d]:
        if col not in df.columns:
            df[col] = None

    # Rows that need at least one price field filled
    needs_price = df[col_price].isna()
    needs_2d    = (~skip_2d) & df[col_2d].isna()
    needs_work  = needs_price | needs_2d
    to_fill     = df[needs_work].copy()

    print(f"  Missing price_at_event:  {needs_price.sum():,}")
    print(f"  Missing move_2d_pct:     {needs_2d.sum():,}")
    print(f"  Rows needing any fill:   {needs_work.sum():,}")

    if to_fill.empty:
        print("Nothing to backfill.")
        return df

    # Use event_trading_date if available, else event_date
    date_col = "event_trading_date" if "event_trading_date" in to_fill.columns else "event_date"
    to_fill["_date"] = to_fill[date_col].fillna(to_fill["event_date"])

    # Bulk OHLC download (uses cache if available)
    ohlc_start, ohlc_end = date_range_for_events(
        to_fill.rename(columns={"_date": "event_date"})
    )
    tickers = to_fill["ticker"].str.upper().unique().tolist()
    print(f"\nDownloading OHLC for {len(tickers)} tickers ({ohlc_start} → {ohlc_end}) ...")
    ohlc_cache = load_ohlc_bulk(
        tickers, ohlc_start, ohlc_end,
        events_df=to_fill.rename(columns={"_date": "event_date"}),
        force_refresh=force_refresh,
    )

    # Fill row by row from cache
    filled_price = filled_2d = 0
    for idx, row in to_fill.iterrows():
        ticker = str(row["ticker"]).upper()
        edate  = str(row["_date"])
        ohlc   = ohlc_cache.get(ticker)
        if ohlc is None:
            continue

        if needs_price[idx]:
            price = _get_close_on_date(ohlc, edate)
            if price is not None:
                df.at[idx, col_price] = price
                filled_price += 1

        if needs_2d[idx]:
            pb = _get_close_before(ohlc, edate)
            pa = _get_close_after(ohlc, edate)
            if pb is not None:
                df.at[idx, col_before] = pb
            if pa is not None:
                df.at[idx, col_after] = pa
            if pb is not None and pa is not None and pb > 0:
                df.at[idx, col_2d] = round((pa - pb) / pb * 100, 2)
                filled_2d += 1

    df.to_csv(input_file, index=False)

    print(f"\n{'='*55}")
    print("BACKFILL COMPLETE")
    print(f"{'='*55}")
    print(f"price_at_event filled: {filled_price:,} new rows  "
          f"(total: {df[col_price].notna().sum():,}/{len(df):,})")
    if not skip_2d:
        print(f"move_2d_pct filled:    {filled_2d:,} new rows  "
              f"(total: {df[col_2d].notna().sum():,}/{len(df):,})")
    print(f"Saved → {input_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill price columns via yfinance OHLC")
    parser.add_argument("--input", default="enriched_all_clinical.csv")
    parser.add_argument("--skip-2d", action="store_true",
                        help="Skip filling price_before / price_after / move_2d_pct")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-download OHLC even if cached")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    backfill_prices(args.input, args.skip_2d, args.force_refresh)

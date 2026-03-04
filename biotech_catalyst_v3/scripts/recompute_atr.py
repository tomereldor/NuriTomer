"""Recompute ATR and move classifications for existing enriched CSVs.

Uses bulk OHLC download (one call per ~50 tickers) + vectorized Wilder ATR (20-day).
Overwrites atr_pct, atr_value, avg_daily_move, normalized_move, and all move_class_* columns.

Usage:
    # Recompute on the main dataset (default)
    python -m scripts.recompute_atr

    # Specify input/output
    python -m scripts.recompute_atr --input enriched_all_clinical.csv --output enriched_all_clinical.csv

    # Force re-download OHLC even if cached
    python -m scripts.recompute_atr --force-refresh
"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ohlc_cache import load_ohlc_bulk, date_range_for_events
from utils.volatility import batch_enrich_atr


def recompute_atr(
    input_file: str = "enriched_all_clinical.csv",
    output_file: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    output_file = output_file or input_file

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  {len(df)} rows, {len(df['ticker'].unique())} unique tickers")

    # Date range: earliest event minus 45 calendar days → latest event
    ohlc_start, ohlc_end = date_range_for_events(df)
    tickers = df["ticker"].str.upper().unique().tolist()

    print(f"Bulk OHLC download: {len(tickers)} tickers ({ohlc_start} → {ohlc_end})")
    ohlc_cache = load_ohlc_bulk(
        tickers, ohlc_start, ohlc_end,
        events_df=df,           # enables per-ticker date ranges
        force_refresh=force_refresh,
    )

    print("Computing Wilder ATR (20-day) and reclassifying moves...")
    df = batch_enrich_atr(df, ohlc_cache)

    df.to_csv(output_file, index=False)
    print(f"\nSaved → {output_file}")

    # Summary
    if "move_class_combo" in df.columns:
        dist = df["move_class_combo"].value_counts().to_dict()
        print(f"move_class_combo: {dist}")
    if "move_class_norm" in df.columns:
        dist = df["move_class_norm"].value_counts().to_dict()
        print(f"move_class_norm:  {dist}")
    atr_filled = df["atr_pct"].notna().sum()
    print(f"ATR filled:       {atr_filled}/{len(df)} ({100*atr_filled/len(df):.1f}%)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute ATR for existing enriched CSVs")
    parser.add_argument("--input", default="enriched_all_clinical.csv")
    parser.add_argument("--output", default=None, help="Default: overwrite input")
    parser.add_argument("--force-refresh", action="store_true", help="Re-download OHLC even if cached")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    recompute_atr(args.input, args.output, args.force_refresh)

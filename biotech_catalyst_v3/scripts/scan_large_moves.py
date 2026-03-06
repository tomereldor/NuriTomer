"""Scan expanded biotech universe for large stock moves (≥10%).

Complementary to the news-first approach in find_clinical_events.py.
Catches catalysts that don't appear in ClinicalTrials.gov (FDA decisions,
partnerships, earnings) and events where CT.gov sponsor → ticker mapping failed.

Output is a raw list of moves to be filtered by catalyst type in a subsequent step.

Usage:
    python -m scripts.scan_large_moves
    python -m scripts.scan_large_moves --min-move 10 --start 2023-01-01
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATR_LOOKBACK = 20


def load_tickers(universe_file: str, existing_file: str):
    """
    Load tickers from universe file.
    Returns (ticker_list, existing_keys, ticker_to_cap dict).
    """
    if not os.path.exists(universe_file):
        raise FileNotFoundError(
            f"Universe file not found: {universe_file}\n"
            "Run scripts/expand_company_universe.py first."
        )

    universe = pd.read_csv(universe_file)
    tickers = universe["ticker"].str.upper().unique().tolist()
    ticker_to_cap = {
        str(r["ticker"]).upper(): round(float(r["market_cap_m"]), 1)
        for _, r in universe.iterrows()
        if pd.notna(r.get("market_cap_m"))
    }

    existing_keys: set = set()
    if os.path.exists(existing_file):
        existing = pd.read_csv(existing_file)
        existing_keys = set(zip(existing["ticker"].str.upper(), existing["event_date"].astype(str)))

    return tickers, existing_keys, ticker_to_cap


def scan_ticker(
    ticker: str,
    start: str,
    end: str,
    min_move: float,
    existing_keys: set,
    market_cap_m=None,
) -> list:
    """Download price history for one ticker and return qualifying move events."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return []

        # Flatten multi-level columns (yfinance >= 0.2.31)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["return_pct"] = df["Close"].pct_change() * 100

        # ATR for normalization context
        df["prev_close"] = df["Close"].shift(1)
        df["TR"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["prev_close"]),
                abs(df["Low"] - df["prev_close"]),
            ),
        )
        df["atr_20"] = df["TR"].rolling(ATR_LOOKBACK).mean()
        df["atr_pct"] = df["atr_20"] / df["Close"] * 100

        large = df[df["return_pct"].abs() >= min_move].dropna(subset=["return_pct"])

        events = []
        for date, row in large.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            if (ticker, date_str) in existing_keys:
                continue

            # Scalar extraction (handle Series from older yfinance)
            def _scalar(val):
                return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)

            ret = _scalar(row["return_pct"])
            close = _scalar(row["Close"])
            vol = int(_scalar(row["Volume"])) if "Volume" in row and pd.notna(row["Volume"]) else 0
            atr = _scalar(row["atr_pct"]) if pd.notna(row.get("atr_pct", None)) else None

            events.append({
                "ticker": ticker,
                "event_date": date_str,
                "market_cap_m": market_cap_m,
                "move_pct": round(ret, 2),
                "close": round(close, 2),
                "volume": vol,
                "atr_pct": round(atr, 2) if atr else None,
                "normalized_move": round(abs(ret) / atr, 2) if atr else None,
                "event_type": "Gainer" if ret > 0 else "Loser",
                "catalyst_type": "",  # To be filled by filter_to_clinical.py
            })

        return events

    except Exception:
        return []


def scan_large_moves(
    universe_file: str = "biotech_universe_expanded.csv",
    existing_file: str = "enriched_high_moves.csv",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    min_move_pct: float = 10.0,
    output_file: str = "large_moves_new.csv",
):
    """Scan all universe tickers for moves ≥ min_move_pct not already in dataset."""

    tickers, existing_keys, ticker_to_cap = load_tickers(universe_file, existing_file)
    print(f"Scanning {len(tickers)} tickers for moves ≥{min_move_pct}%")
    print(f"Date range: {start_date} → {end_date}")
    print(f"Skipping {len(existing_keys)} existing (ticker, date) pairs\n")

    all_moves = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(tickers)} | moves found: {len(all_moves)}")

        events = scan_ticker(ticker, start_date, end_date, min_move_pct, existing_keys,
                             market_cap_m=ticker_to_cap.get(ticker))
        all_moves.extend(events)
        time.sleep(0.1)

    if not all_moves:
        print("No qualifying moves found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_moves)
    df = df.sort_values("move_pct", key=abs, ascending=False).reset_index(drop=True)
    df.to_csv(output_file, index=False)

    gainers = (df["event_type"] == "Gainer").sum()
    losers = (df["event_type"] == "Loser").sum()

    print(f"\n{'='*55}")
    print(f"LARGE MOVES SCAN COMPLETE")
    print(f"{'='*55}")
    print(f"New qualifying moves: {len(df)}")
    print(f"  Gainers: {gainers}")
    print(f"  Losers:  {losers}")
    print(f"\nTop 10 moves:")
    print(df.head(10)[["ticker", "event_date", "move_pct", "atr_pct", "normalized_move"]].to_string(index=False))
    print(f"\nSaved to: {output_file}")
    print(f"\nNext: run filter_to_clinical.py --input {output_file} to keep Clinical Data events")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan biotech universe for large stock moves")
    parser.add_argument("--universe", default="biotech_universe_expanded.csv")
    parser.add_argument("--existing", default="enriched_high_moves.csv")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--min-move", type=float, default=10.0, help="Min absolute move %")
    parser.add_argument("--output", default="large_moves_new.csv")
    args = parser.parse_args()

    scan_large_moves(
        universe_file=args.universe,
        existing_file=args.existing,
        start_date=args.start,
        end_date=args.end,
        min_move_pct=args.min_move,
        output_file=args.output,
    )

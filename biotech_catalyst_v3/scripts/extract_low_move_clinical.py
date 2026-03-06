"""Extract ATR-normalized low-move events for ML training."""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATR_LOOKBACK = 20


def calculate_atr(prices_df, period=ATR_LOOKBACK):
    """Calculate ATR for a price dataframe."""
    df = prices_df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['prev_close']),
            abs(df['Low'] - df['prev_close'])
        )
    )
    df['ATR'] = df['TR'].rolling(period).mean()
    df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
    return df


def extract_low_moves(
    tickers: list,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    max_normalized_move: float = 1.5,  # < 1.5x ATR = "low"
    min_abs_move_pct: float = 2.0,     # At least 2% to be interesting
    target_count: int = 300,           # Get extra, filter later
) -> pd.DataFrame:
    """Find low-move events (normalized_move < 1.5x ATR)."""

    all_events = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(tickers)}")

        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) < 20:
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Calculate ATR
            df = calculate_atr(df)

            # Calculate daily return
            df['return_pct'] = df['Close'].pct_change() * 100
            df['abs_return'] = df['return_pct'].abs()

            # Calculate normalized move
            df['normalized_move'] = df['abs_return'] / df['ATR_pct']

            # Filter: low normalized move but meaningful absolute move
            low_moves = df[
                (df['normalized_move'] < max_normalized_move) &
                (df['abs_return'] >= min_abs_move_pct) &
                (df['ATR_pct'].notna())
            ]

            for date, row in low_moves.iterrows():
                all_events.append({
                    'ticker': ticker,
                    'date': date.strftime('%Y-%m-%d'),
                    'move_pct': round(float(row['return_pct']), 2),
                    'atr_pct': round(float(row['ATR_pct']), 2),
                    'normalized_move': round(float(row['normalized_move']), 2),
                    'close': round(float(row['Close']), 2),
                    'event_type': 'Gainer' if row['return_pct'] > 0 else 'Loser',
                })
        except Exception as e:
            continue

        time.sleep(0.1)

    events_df = pd.DataFrame(all_events)

    if events_df.empty:
        print("\nNo low-move events found!")
        return events_df

    # Sort by normalized_move ascending (lowest = most "normal")
    events_df = events_df.sort_values('normalized_move')

    print(f"\nFound {len(events_df)} low-move events")
    return events_df.head(target_count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract ATR-normalized low-move events")
    parser.add_argument("--input", default="enriched_high_moves.csv",
                        help="High-move CSV to extract ticker universe from")
    parser.add_argument("--output", default="low_move_candidates.csv")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--max-normalized-move", type=float, default=1.5)
    parser.add_argument("--min-abs-move", type=float, default=2.0)
    parser.add_argument("--target-count", type=int, default=300)
    args = parser.parse_args()

    # Load tickers from high-move dataset (same universe)
    high_moves = pd.read_csv(args.input)
    tickers = high_moves['ticker'].unique().tolist()

    print(f"Scanning {len(tickers)} tickers for low-move events...")
    print(f"  ATR threshold: < {args.max_normalized_move}x ATR")
    print(f"  Min absolute move: {args.min_abs_move}%")

    events = extract_low_moves(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        max_normalized_move=args.max_normalized_move,
        min_abs_move_pct=args.min_abs_move,
        target_count=args.target_count,
    )

    events.to_csv(args.output, index=False)
    print(f"Saved {len(events)} candidates to {args.output}")
    print("\nNext: Run filter_to_clinical.py to keep only Clinical Data events")

"""
Extract low-move events (3-10% moves) for ML training balance.

Usage:
    python -m scripts.extract_low_moves
    python -m scripts.extract_low_moves --min-move 3 --max-move 10 --output low_move_events.csv
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_scanner import BIOTECH_TICKERS


def extract_low_moves(
    tickers: list,
    start_date: str = "2024-01-01",
    min_move_pct: float = 3.0,
    max_move_pct: float = 10.0,
    output_file: str = "low_move_events_raw.csv",
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Scan tickers for moves in the min_move_pct to max_move_pct range.
    These are "normal catalyst" events - clinical data that moved the stock
    modestly, useful as negatives/baseline for ML models.
    """
    print("=" * 60)
    print(f"EXTRACTING LOW-MOVE EVENTS ({min_move_pct}-{max_move_pct}% moves)")
    print(f"Date range: {start_date} to present")
    print(f"Tickers: {len(tickers)}")
    print("=" * 60)

    all_events = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(
            f"\nBatch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}: "
            f"{len(batch)} tickers"
        )

        try:
            data = yf.download(
                " ".join(batch),
                start=start_date,
                progress=False,
                threads=True,
                timeout=60,
            )

            if data.empty:
                print("  No data returned")
                time.sleep(5)
                continue

            # Get close prices
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data.xs("Close", level=0, axis=1)
            elif "Close" in data.columns and len(batch) == 1:
                close_data = data[["Close"]].copy()
                close_data.columns = [batch[0]]
            else:
                print("  Unexpected data format")
                continue

            for ticker in batch:
                if ticker not in close_data.columns:
                    continue

                prices = close_data[ticker].dropna()
                if len(prices) < 2:
                    continue

                returns = prices.pct_change()
                abs_returns = returns.abs() * 100

                # Filter to moves within the target range
                low_moves = returns[
                    (abs_returns >= min_move_pct) & (abs_returns <= max_move_pct)
                ]

                for date, ret in low_moves.items():
                    if pd.isna(ret):
                        continue
                    move_type = "Gainer" if ret > 0 else "Loser"
                    all_events.append(
                        {
                            "Ticker": ticker,
                            "Date": date.strftime("%Y-%m-%d"),
                            "Type": move_type,
                            "Move_%": round(ret * 100, 2),
                            "Price_Event": (
                                round(prices.loc[date], 2)
                                if date in prices.index
                                else 0
                            ),
                        }
                    )

            print(f"  Total events so far: {len(all_events)}")

        except Exception as e:
            print(f"  Error: {str(e)[:80]}")

        time.sleep(3)

    df = pd.DataFrame(all_events)

    if not df.empty:
        df["Abs_Move"] = df["Move_%"].abs()
        df = df.sort_values("Abs_Move", ascending=False)
        df = df.drop(columns=["Abs_Move"])
        df.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total low-move events: {len(df)}")
        print(f"Unique tickers: {df['Ticker'].nunique()}")
        print(f"Gainers: {(df['Type'] == 'Gainer').sum()}")
        print(f"Losers: {(df['Type'] == 'Loser').sum()}")
        print(f"Move range: {df['Move_%'].min():.1f}% to {df['Move_%'].max():.1f}%")
        print(f"\nSaved to: {output_file}")
        print(
            "\nNext step: Run enrichment pipeline on these events,"
            " then filter for catalyst_type == 'Clinical Data'"
        )
    else:
        print("\nNo events found!")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract low-move events for ML training")
    parser.add_argument("--min-move", type=float, default=3.0)
    parser.add_argument("--max-move", type=float, default=10.0)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--output", type=str, default="low_move_events_raw.csv")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    extract_low_moves(
        BIOTECH_TICKERS,
        start_date=args.start_date,
        min_move_pct=args.min_move,
        max_move_pct=args.max_move,
        output_file=args.output,
        batch_size=args.batch_size,
    )

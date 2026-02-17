"""ATR and volatility metrics for move normalization."""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from typing import Dict, Optional


def calculate_atr(ticker: str, event_date: str, lookback: int = 14) -> Dict:
    """
    Calculate ATR metrics for a stock BEFORE the event.

    Returns dict with:
        atr_pct:   ATR as percentage of price (typical daily volatility)
        atr_value: Raw ATR in dollars
    """
    try:
        event_dt = pd.to_datetime(event_date)
        end_date = event_dt - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback + 15)

        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
        )

        if df.empty or len(df) < lookback:
            return {"atr_pct": None, "atr_value": None}

        # Flatten multi-level columns if present (yfinance >= 0.2.31)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["prev_close"] = df["Close"].shift(1)
        df["TR"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["prev_close"]),
                abs(df["Low"] - df["prev_close"]),
            ),
        )

        atr = df["TR"].rolling(lookback).mean().iloc[-1]
        price = df["Close"].iloc[-1]

        # Handle scalar vs Series (yfinance version differences)
        atr_val = float(atr.iloc[0]) if hasattr(atr, "iloc") else float(atr)
        price_val = float(price.iloc[0]) if hasattr(price, "iloc") else float(price)

        if price_val == 0:
            return {"atr_pct": None, "atr_value": None}

        atr_pct = (atr_val / price_val) * 100

        return {
            "atr_pct": round(atr_pct, 2),
            "atr_value": round(atr_val, 4),
        }

    except Exception as e:
        print(f"  ATR error for {ticker}: {e}")
        return {"atr_pct": None, "atr_value": None}


def calculate_normalized_move(move_pct: float, atr_pct: float) -> Optional[float]:
    """Calculate move normalized by ATR (how many ATRs the move represents)."""
    if atr_pct is None or atr_pct == 0:
        return None
    return round(abs(move_pct) / atr_pct, 2)


def classify_move_magnitude(normalized_move: Optional[float]) -> str:
    """
    Classify move based on ATR normalization.

    < 1.5x ATR = Low    (within normal volatility)
    1.5-3.0x   = Medium (notable)
    > 3.0x     = High   (significant outlier)
    """
    if normalized_move is None:
        return "Unknown"
    if normalized_move < 1.5:
        return "Low"
    elif normalized_move < 3.0:
        return "Medium"
    else:
        return "High"


def enrich_with_atr(df: pd.DataFrame, batch_size: int = 25) -> pd.DataFrame:
    """
    Add ATR metrics to dataframe.

    Adds columns: atr_pct, normalized_move, move_magnitude
    """
    df = df.copy()
    df["atr_pct"] = None
    df["normalized_move"] = None
    df["move_magnitude"] = None

    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % batch_size == 0:
            print(f"  ATR progress: {i + 1}/{total}")

        atr_data = calculate_atr(row["ticker"], str(row["event_date"]))
        df.at[idx, "atr_pct"] = atr_data["atr_pct"]

        if atr_data["atr_pct"]:
            norm = calculate_normalized_move(row["move_pct"], atr_data["atr_pct"])
            df.at[idx, "normalized_move"] = norm
            df.at[idx, "move_magnitude"] = classify_move_magnitude(norm)

    return df

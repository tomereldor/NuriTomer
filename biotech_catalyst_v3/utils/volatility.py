"""ATR and volatility metrics for move normalization.

ATR approach:
    - Wilder's ATR (RMA): ewm(alpha=1/lookback, adjust=False) — same as TradingView / pandas_ta default
    - 20 trading-day lookback (≈ 1 calendar month)
    - Computed from pre-downloaded OHLC data (no per-row yfinance calls)

Classification thresholds:
    move_class_norm  (pure ATR multiple, 5 levels):
        < 1.5x ATR  → Noise    (normal daily variation)
        1.5–3x ATR  → Low      (slightly elevated)
        3–5x ATR    → Medium   (clear catalyst signal)
        5–8x ATR    → High     (strong catalyst)
        ≥ 8x ATR    → Extreme  (defining event: Phase 3, FDA, etc.)

    move_class_abs   (absolute %, 5 levels — unchanged from v1 for backward compat):
        < 10%  → VeryLow
        10–15% → Low
        15–30% → Medium
        30–50% → High
        ≥ 50%  → VeryHigh

    move_class_combo (ML label, dual AND requirement):
        Low    → abs < 15%  AND atr_norm < 2.5x   (within noise for this stock)
        High   → abs ≥ 20%  AND atr_norm ≥ 3.5x   (meaningful AND statistically unusual)
        Medium → everything else

Statistical basis: Wilder ATR ≈ 0.8σ of daily returns, so 3.5x ATR ≈ 4.4σ
    → by chance ~0.001% of days, strong signal for catalyst detection.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

ATR_LOOKBACK = 20  # trading days (≈ 1 calendar month)


# ---------------------------------------------------------------------------
# Core ATR computation (from pre-fetched OHLC — no yfinance calls)
# ---------------------------------------------------------------------------

def compute_atr_for_ticker(
    ohlc_df: pd.DataFrame,
    event_date: str,
    lookback: int = ATR_LOOKBACK,
) -> Dict:
    """
    Compute Wilder's ATR for a ticker at a specific event date.

    Uses only price data STRICTLY BEFORE the event (no look-ahead bias).

    Args:
        ohlc_df:    DataFrame with columns Open/High/Low/Close, DatetimeIndex
        event_date: "YYYY-MM-DD" — event day is excluded
        lookback:   Number of trading days for ATR (default: 20)

    Returns:
        {"atr_pct": float, "atr_value": float, "avg_daily_move": float}
        All values are None on failure.
    """
    try:
        event_dt = pd.Timestamp(event_date).normalize()
        pre = ohlc_df[ohlc_df.index < event_dt].copy()

        if len(pre) < lookback:
            return {"atr_pct": None, "atr_value": None, "avg_daily_move": None}

        # Only need lookback + buffer rows for EWM initialization
        pre = pre.tail(lookback + 10)

        pre["_prev_close"] = pre["Close"].shift(1)
        pre["_TR"] = np.maximum(
            pre["High"] - pre["Low"],
            np.maximum(
                (pre["High"] - pre["_prev_close"]).abs(),
                (pre["Low"] - pre["_prev_close"]).abs(),
            ),
        )

        # Wilder's RMA: alpha = 1/lookback, same as TradingView ATR
        atr_series = pre["_TR"].ewm(alpha=1 / lookback, adjust=False, min_periods=lookback).mean()
        atr_val = atr_series.iloc[-1]
        price_val = pre["Close"].iloc[-1]

        # Unwrap pandas scalar if needed
        if hasattr(atr_val, "item"):
            atr_val = atr_val.item()
        if hasattr(price_val, "item"):
            price_val = price_val.item()

        if not np.isfinite(atr_val) or not np.isfinite(price_val) or price_val == 0:
            return {"atr_pct": None, "atr_value": None, "avg_daily_move": None}

        atr_pct = (atr_val / price_val) * 100
        avg_daily_move = float(pre["Close"].pct_change().abs().mean() * 100)

        return {
            "atr_pct": round(float(atr_pct), 2),
            "atr_value": round(float(atr_val), 4),
            "avg_daily_move": round(avg_daily_move, 2),
        }

    except Exception as e:
        return {"atr_pct": None, "atr_value": None, "avg_daily_move": None}


# ---------------------------------------------------------------------------
# Move classification
# ---------------------------------------------------------------------------

def classify_move(move_pct: float, atr_pct: float) -> dict:
    """
    Classify a move using absolute %, ATR multiple, and combined ML label.

    Returns keys:
        normalized_move:  abs(move_pct) / atr_pct  (how many ATRs the move is)
        move_class_abs:   VeryLow / Low / Medium / High / VeryHigh
        move_class_norm:  Noise / Low / Medium / High / Extreme
        move_class_combo: Low / Medium / High  (ML label)
    """
    abs_move = abs(move_pct)
    norm = abs_move / atr_pct if atr_pct else None

    # --- Absolute class ---
    if abs_move < 10:
        abs_c = "VeryLow"
    elif abs_move < 15:
        abs_c = "Low"
    elif abs_move < 30:
        abs_c = "Medium"
    elif abs_move < 50:
        abs_c = "High"
    else:
        abs_c = "VeryHigh"

    # --- ATR-normalized class ---
    if norm is None:
        norm_c = "Unknown"
    elif norm < 1.5:
        norm_c = "Noise"
    elif norm < 3.0:
        norm_c = "Low"
    elif norm < 5.0:
        norm_c = "Medium"
    elif norm < 8.0:
        norm_c = "High"
    else:
        norm_c = "Extreme"

    # --- Combo ML label (dual AND requirement) ---
    # Low:    clearly small in both absolute and relative terms
    # High:   meaningfully large AND statistically unusual for this stock
    # Medium: everything in between
    if abs_move < 15 and (norm is None or norm < 2.5):
        combo = "Low"
    elif abs_move >= 20 and (norm is not None and norm >= 3.5):
        combo = "High"
    else:
        combo = "Medium"

    return {
        "normalized_move": round(norm, 2) if norm is not None else None,
        "move_class_abs": abs_c,
        "move_class_norm": norm_c,
        "move_class_combo": combo,
    }


# ---------------------------------------------------------------------------
# Batch ATR enrichment (vectorized over a DataFrame)
# ---------------------------------------------------------------------------

def batch_enrich_atr(
    df: pd.DataFrame,
    ohlc_cache: Dict[str, pd.DataFrame],
    lookback: int = ATR_LOOKBACK,
) -> pd.DataFrame:
    """
    Add ATR columns to all rows using pre-fetched OHLC cache.

    No yfinance calls — all data comes from ohlc_cache.
    Existing atr_pct values are overwritten for rows where OHLC is available.

    Adds/updates columns:
        atr_pct, atr_value, avg_daily_move,
        normalized_move, move_class_abs, move_class_norm, move_class_combo
    """
    df = df.copy()
    atr_cols = [
        "atr_pct", "atr_value", "avg_daily_move", "normalized_move",
        "move_class_abs", "move_class_norm", "move_class_combo",
    ]
    for col in atr_cols:
        if col not in df.columns:
            df[col] = None

    hits = 0
    for idx, row in df.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        event_date = str(row.get("event_trading_date") or row.get("event_date") or "")
        move_pct = row.get("move_pct")

        ohlc = ohlc_cache.get(ticker)
        if ohlc is None or not event_date or pd.isna(move_pct):
            continue

        atr = compute_atr_for_ticker(ohlc, event_date, lookback)
        df.at[idx, "atr_pct"] = atr.get("atr_pct")
        df.at[idx, "atr_value"] = atr.get("atr_value")
        df.at[idx, "avg_daily_move"] = atr.get("avg_daily_move")

        if atr.get("atr_pct"):
            cls = classify_move(float(move_pct), atr["atr_pct"])
            for k, v in cls.items():
                df.at[idx, k] = v
            hits += 1

    print(f"  ATR computed: {hits}/{len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Legacy row-by-row function (kept for backward compatibility)
# Prefer batch_enrich_atr + ohlc_cache for any bulk processing.
# ---------------------------------------------------------------------------

def calculate_atr(ticker: str, event_date: str, lookback: int = ATR_LOOKBACK) -> Dict:
    """
    [DEPRECATED for bulk use] Calculate ATR via a single yfinance download.

    Use batch_enrich_atr() + load_ohlc_bulk() instead to avoid
    per-row network calls.
    """
    try:
        event_dt = pd.to_datetime(event_date)
        end_date = event_dt - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback + 15)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

        if raw.empty or len(raw) < lookback:
            return {"atr_pct": None, "atr_value": None, "avg_daily_move": None}

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        return compute_atr_for_ticker(raw, event_date, lookback)

    except Exception as e:
        print(f"  ATR error for {ticker}: {e}")
        return {"atr_pct": None, "atr_value": None, "avg_daily_move": None}


# ---------------------------------------------------------------------------
# Thin wrappers kept for backward compatibility
# ---------------------------------------------------------------------------

def calculate_normalized_move(move_pct: float, atr_pct: float) -> Optional[float]:
    if atr_pct is None or atr_pct == 0:
        return None
    return round(abs(move_pct) / atr_pct, 2)


def classify_move_magnitude(normalized_move: Optional[float]) -> str:
    """Classify move magnitude by ATR multiple (legacy wrapper)."""
    if normalized_move is None:
        return "Unknown"
    if normalized_move < 1.5:
        return "Noise"
    elif normalized_move < 3.0:
        return "Low"
    elif normalized_move < 5.0:
        return "Medium"
    elif normalized_move < 8.0:
        return "High"
    else:
        return "Extreme"


def enrich_with_atr(
    df: pd.DataFrame,
    ohlc_cache: Optional[Dict] = None,
    batch_size: int = 25,
    lookback: int = ATR_LOOKBACK,
) -> pd.DataFrame:
    """
    Add ATR metrics to a DataFrame.

    If ohlc_cache is provided, uses batch_enrich_atr (fast, no network calls).
    Otherwise falls back to row-by-row yfinance downloads (slow, legacy).
    """
    if ohlc_cache is not None:
        return batch_enrich_atr(df, ohlc_cache, lookback)

    # Legacy fallback
    df = df.copy()
    for col in ["atr_pct", "atr_value", "avg_daily_move", "normalized_move",
                "move_magnitude", "move_class_abs", "move_class_norm", "move_class_combo"]:
        if col not in df.columns:
            df[col] = None

    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % batch_size == 0:
            print(f"  ATR progress: {i + 1}/{total}")

        atr_data = calculate_atr(row["ticker"], str(row["event_date"]), lookback)
        df.at[idx, "atr_pct"] = atr_data.get("atr_pct")
        df.at[idx, "atr_value"] = atr_data.get("atr_value")
        df.at[idx, "avg_daily_move"] = atr_data.get("avg_daily_move")

        if atr_data.get("atr_pct"):
            move_pct = row.get("move_pct")
            if pd.notna(move_pct):
                cls = classify_move(float(move_pct), atr_data["atr_pct"])
                for k, v in cls.items():
                    df.at[idx, k] = v
                df.at[idx, "move_magnitude"] = classify_move_magnitude(cls.get("normalized_move"))

    return df

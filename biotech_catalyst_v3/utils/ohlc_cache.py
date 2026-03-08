"""Bulk OHLC downloader with local parquet cache.

Architecture:
    1. Compute per-ticker needed date range from events_df (avoids requesting
       19 years of data for a ticker with only 2024 events)
    2. Load from cache (data/ohlc/<TICKER>.parquet) where available
    3. Bulk-download uncached tickers in small chunks via yf.download()
       — uses a requests.Session with proper User-Agent to avoid 429s
    4. Fallback: any tickers that still fail get retried individually via
       Ticker.history() (different URL pattern, more resilient)
    5. Cache each ticker to parquet on first successful download

Zero per-row yfinance calls — one bulk download covers the whole batch.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "data" / "ohlc"
CHUNK_SIZE = 20  # smaller chunks = less likely to trigger 429


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def _make_session() -> requests.Session:
    """Create a session with browser-like headers to avoid Yahoo Finance 429s."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _covers_range(df: pd.DataFrame, start: str, end: str) -> bool:
    """Return True if cached data covers the needed date range."""
    if df is None or df.empty:
        return False
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    return (
        idx[0].date() <= pd.Timestamp(start).date()
        and idx[-1].date() >= pd.Timestamp(end).date()
    )


def _normalize_df(df: pd.DataFrame, ticker: str = "") -> Optional[pd.DataFrame]:
    """Flatten MultiIndex, strip timezone, drop all-NaN rows."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        # Try extracting ticker slice first
        level0 = set(df.columns.get_level_values(0))
        level1 = set(df.columns.get_level_values(1))
        if ticker.upper() in level0:
            df = df[ticker.upper()].copy()
        elif ticker.upper() in level1:
            df = df.xs(ticker.upper(), axis=1, level=1).copy()
        else:
            # Flatten by taking level 0 (works for single-ticker downloads)
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    if df.empty:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def _ticker_date_range(
    ticker: str,
    events_df: pd.DataFrame,
    lookback_days: int = 45,
) -> tuple:
    """Compute (start, end) covering all events for a specific ticker."""
    rows = events_df[events_df["ticker"].str.upper() == ticker]
    dates = pd.to_datetime(rows["event_date"], format="mixed", errors="coerce").dropna()
    if dates.empty:
        return None, None
    start = (dates.min() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    return start, end


# ---------------------------------------------------------------------------
# Individual fallback download
# ---------------------------------------------------------------------------

def _download_single(
    ticker: str,
    start: str,
    end: str,
    session: requests.Session = None,
) -> Optional[pd.DataFrame]:
    """Download one ticker via Ticker.history() — more resilient than bulk download."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True)
        return _normalize_df(df, ticker)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_ohlc_bulk(
    tickers: List[str],
    start: str,
    end: str,
    events_df: Optional[pd.DataFrame] = None,
    threads: int = 4,
    force_refresh: bool = False,
    chunk_delay: float = 3.0,
    retries: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLC data for many tickers, using disk cache where available.

    Args:
        tickers:       Ticker symbols (any case)
        start:         Global fallback start "YYYY-MM-DD"
        end:           Global fallback end "YYYY-MM-DD"
        events_df:     If provided, uses per-ticker date ranges (avoids
                       downloading decades of data for tickers with recent events)
        threads:       Parallel threads for yf.download
        force_refresh: Re-download even if cached
        chunk_delay:   Seconds to sleep between bulk chunks
        retries:       Retry attempts per chunk before falling back to individual

    Returns:
        {TICKER: DataFrame(Open, High, Low, Close, Volume)}
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [t.upper() for t in tickers]
    session = _make_session()

    result: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []

    # --- Cache check (uses per-ticker range if events_df provided) ---
    for ticker in tickers:
        t_start, t_end = (
            _ticker_date_range(ticker, events_df) if events_df is not None
            else (start, end)
        )
        if t_start is None:
            t_start, t_end = start, end

        if not force_refresh:
            path = _cache_path(ticker)
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    if _covers_range(df, t_start, t_end):
                        result[ticker] = df
                        continue
                except Exception:
                    pass
        to_download.append(ticker)

    if not to_download:
        print(f"  OHLC: {len(result)}/{len(tickers)} from cache (nothing to download)")
        return result

    print(f"  OHLC: {len(result)} cached | {len(to_download)} to download")

    # --- Bulk download in small chunks ---
    failed_bulk: List[str] = []

    for chunk_idx, chunk_start_i in enumerate(range(0, len(to_download), CHUNK_SIZE)):
        chunk = to_download[chunk_start_i : chunk_start_i + CHUNK_SIZE]
        n_total = len(to_download)
        label = f"{chunk_start_i + 1}–{min(chunk_start_i + CHUNK_SIZE, n_total)}/{n_total}"
        print(f"    chunk {label} ...")

        if chunk_idx > 0:
            time.sleep(chunk_delay)

        # Use the tightest common date range for this chunk
        if events_df is not None:
            chunk_starts, chunk_ends = [], []
            for t in chunk:
                ts, te = _ticker_date_range(t, events_df)
                if ts:
                    chunk_starts.append(ts)
                    chunk_ends.append(te)
            c_start = min(chunk_starts) if chunk_starts else start
            c_end = max(chunk_ends) if chunk_ends else end
        else:
            c_start, c_end = start, end

        raw = None
        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(
                    chunk if len(chunk) > 1 else chunk[0],
                    start=c_start,
                    end=c_end,
                    auto_adjust=True,
                    threads=threads,
                    progress=False,
                    group_by="ticker",
                )
                break
            except Exception as e:
                print(f"      attempt {attempt}/{retries}: {type(e).__name__}")
                if attempt < retries:
                    time.sleep(chunk_delay * attempt * 2)

        # Parse results and track what succeeded
        got: set = set()
        if raw is not None and not raw.empty:
            for ticker in chunk:
                df = _normalize_df(raw if len(chunk) == 1 else raw, ticker)
                if df is not None and not df.empty:
                    result[ticker] = df
                    got.add(ticker)
                    try:
                        df.to_parquet(_cache_path(ticker))
                    except Exception:
                        pass

        # Queue failures for individual fallback
        for ticker in chunk:
            if ticker not in got:
                failed_bulk.append(ticker)

    # --- Individual fallback for tickers that failed bulk download ---
    if failed_bulk:
        print(f"  Fallback: downloading {len(failed_bulk)} failed tickers individually...")
        for i, ticker in enumerate(failed_bulk):
            if i > 0 and i % 10 == 0:
                time.sleep(chunk_delay)
                print(f"    {i}/{len(failed_bulk)} ...")

            t_start, t_end = (
                _ticker_date_range(ticker, events_df) if events_df is not None
                else (start, end)
            )
            if t_start is None:
                t_start, t_end = start, end

            df = _download_single(ticker, t_start, t_end, session)
            if df is not None and not df.empty:
                result[ticker] = df
                try:
                    df.to_parquet(_cache_path(ticker))
                except Exception:
                    pass

    n_ok = len(result)
    n_fail = len(tickers) - n_ok
    print(f"  OHLC ready: {n_ok}/{len(tickers)} tickers" +
          (f" ({n_fail} unavailable — likely delisted)" if n_fail else ""))
    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def date_range_for_events(
    events_df: pd.DataFrame,
    lookback_calendar_days: int = 45,
) -> tuple:
    """Global (start, end) covering all events with ATR buffer."""
    dates = pd.to_datetime(events_df["event_date"], format="mixed", errors="coerce").dropna()
    start = (dates.min() - pd.Timedelta(days=lookback_calendar_days)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    return start, end

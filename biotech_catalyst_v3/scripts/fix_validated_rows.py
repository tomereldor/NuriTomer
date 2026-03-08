"""
Fix Validated Rows
==================
Reads a CSV produced by validate_catalysts.py (which contains v_action columns)
and applies the recommended corrections automatically.

Actions handled
---------------
FIX_DATE
    The event is real but was attributed to the wrong date.
    Steps:
      1. Update event_date → v_actual_date
      2. Find the actual trading day on/after v_actual_date in OHLC data
      3. Re-fetch price_at_event, price_before, price_after
      4. Recompute move_pct, move_2d_pct, event_type
      5. Recompute atr_pct, avg_daily_move (Wilder 20-day RMA, strictly pre-event)
      6. Recompute stock_movement_atr_normalized, move_class_norm, move_class_abs,
         move_class_combo
      7. Mark v_action → DATE_FIXED

FLAG_FALSE_POSITIVE
    No clinical news found — the catalyst association is likely hallucinated.
    Default: set data_complete = False  (row stays but is excluded from ML)
    With --remove: physically delete the row

FLAG_ERROR
    API / network failure during validation — skip (re-run validate_catalysts.py).

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3

    # Dry run — show what would change without writing
    python -m scripts.fix_validated_rows --input enriched_all_clinical_validated.csv --dry-run

    # Apply fixes (overwrites input by default)
    python -m scripts.fix_validated_rows --input enriched_all_clinical_validated.csv

    # Apply fixes and remove false positives instead of flagging them
    python -m scripts.fix_validated_rows --input enriched_all_clinical_validated.csv --remove-false-positives

    # Save to a separate file
    python -m scripts.fix_validated_rows \\
        --input  enriched_all_clinical_validated.csv \\
        --output enriched_all_clinical_fixed.csv
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ohlc_cache import load_ohlc_bulk, date_range_for_events
from utils.volatility import compute_atr_for_ticker, classify_move


# ============================================================================
# Price helpers (same logic as backfill_price_at_event.py)
# ============================================================================

def _get_close_on_date(ohlc: pd.DataFrame, date_str: str) -> Optional[float]:
    """Closing price on date_str, looking forward up to 4 days (weekends/holidays)."""
    if ohlc is None or ohlc.empty:
        return None
    target = pd.Timestamp(date_str).normalize()
    for delta in range(5):
        mask = ohlc.index.normalize() == target + pd.Timedelta(days=delta)
        if mask.any():
            val = ohlc.loc[mask, "Close"].iloc[-1]
            return round(float(val), 4) if pd.notna(val) else None
    return None


def _get_trading_date(ohlc: pd.DataFrame, date_str: str) -> Optional[str]:
    """Return the actual trading date string for the close found by _get_close_on_date."""
    if ohlc is None or ohlc.empty:
        return None
    target = pd.Timestamp(date_str).normalize()
    for delta in range(5):
        check = target + pd.Timedelta(days=delta)
        mask  = ohlc.index.normalize() == check
        if mask.any():
            return check.strftime("%Y-%m-%d")
    return None


def _get_close_before(ohlc: pd.DataFrame, date_str: str) -> Optional[float]:
    """Last closing price strictly before date_str."""
    if ohlc is None or ohlc.empty:
        return None
    target = pd.Timestamp(date_str).normalize()
    pre    = ohlc[ohlc.index.normalize() < target]
    if pre.empty:
        return None
    val = pre["Close"].iloc[-1]
    return round(float(val), 4) if pd.notna(val) else None


def _get_close_after(ohlc: pd.DataFrame, date_str: str) -> Optional[float]:
    """First closing price strictly after date_str."""
    if ohlc is None or ohlc.empty:
        return None
    target = pd.Timestamp(date_str).normalize()
    post   = ohlc[ohlc.index.normalize() > target]
    if post.empty:
        return None
    val = post["Close"].iloc[0]
    return round(float(val), 4) if pd.notna(val) else None


# ============================================================================
# Core fix functions
# ============================================================================

def fix_date_rows(
    df: pd.DataFrame,
    ohlc_cache: dict,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    For each FIX_DATE row:
      - Update event_date / event_trading_date
      - Re-fetch prices and recompute move_pct / move_2d_pct / event_type
      - Recompute ATR and move classifications
      - Set v_action = 'DATE_FIXED'
    """
    mask     = df["v_action"] == "FIX_DATE"
    fix_rows = df[mask]
    n        = len(fix_rows)

    if n == 0:
        print("  No FIX_DATE rows.")
        return df

    print(f"  FIX_DATE rows: {n}")
    if dry_run:
        date_col = "event_trading_date" if "event_trading_date" in df.columns else "event_date"
        print(fix_rows[["ticker", date_col, "v_actual_date", "move_pct",
                         "v_pr_title"]].to_string(index=False))
        return df

    fixed = skipped = 0

    for idx, row in fix_rows.iterrows():
        new_date = str(row.get("v_actual_date", "")).strip()
        if not new_date or new_date == "nan":
            skipped += 1
            continue
        # Skip rows with invalid dates (e.g. "2023-12-00")
        try:
            pd.Timestamp(new_date)
        except Exception:
            print(f"    SKIP invalid date '{new_date}' for {row.get('ticker', '?')}")
            skipped += 1
            continue

        ticker = str(row.get("ticker", "")).upper()
        ohlc   = ohlc_cache.get(ticker)

        # ── 1. Update dates ──────────────────────────────────────────────
        df.at[idx, "event_date"] = new_date
        trading_date = _get_trading_date(ohlc, new_date) if ohlc is not None else None
        if trading_date:
            df.at[idx, "event_trading_date"] = trading_date

        effective_date = trading_date or new_date

        # ── 2. Re-fetch prices ───────────────────────────────────────────
        price_at     = _get_close_on_date(ohlc, new_date)
        price_before = _get_close_before(ohlc, new_date)
        price_after  = _get_close_after(ohlc, new_date)

        if price_at     is not None: df.at[idx, "price_at_event"] = price_at
        if price_before is not None: df.at[idx, "price_before"]   = price_before
        if price_after  is not None: df.at[idx, "price_after"]    = price_after

        # ── 3. Recompute move_pct / event_type ──────────────────────────
        if price_before and price_before > 0 and price_at:
            move_pct = round((price_at - price_before) / price_before * 100, 2)
            df.at[idx, "move_pct"]    = move_pct
            df.at[idx, "event_type"]  = "Gainer" if move_pct > 0 else "Loser"

        # ── 4. Recompute move_2d_pct ────────────────────────────────────
        if price_before and price_before > 0 and price_after:
            df.at[idx, "move_2d_pct"] = round(
                (price_after - price_before) / price_before * 100, 2
            )

        # ── 5. Recompute ATR (strictly pre-event) ───────────────────────
        if ohlc is not None:
            atr = compute_atr_for_ticker(ohlc, effective_date)
            if atr.get("atr_pct"):
                df.at[idx, "atr_pct"]        = atr["atr_pct"]
                df.at[idx, "avg_daily_move"]  = atr.get("avg_daily_move")

                # ── 6. Recompute move classifications ────────────────────
                move_pct_val = df.at[idx, "move_pct"]
                if pd.notna(move_pct_val):
                    cls = classify_move(float(move_pct_val), atr["atr_pct"])
                    df.at[idx, "stock_movement_atr_normalized"] = cls["normalized_move"]
                    df.at[idx, "move_class_abs"]   = cls["move_class_abs"]
                    df.at[idx, "move_class_norm"]  = cls["move_class_norm"]
                    df.at[idx, "move_class_combo"] = cls["move_class_combo"]

        # ── 7. Mark as fixed ─────────────────────────────────────────────
        df.at[idx, "v_action"] = "DATE_FIXED"
        fixed += 1

        price_str = f"${price_at:.2f}" if price_at else "price N/A"
        print(f"    {ticker} {new_date} ({price_str})  "
              f"move={df.at[idx, 'move_pct']:+.1f}%  "
              f"norm={df.at[idx, 'stock_movement_atr_normalized']:.2f}× ATR")

    print(f"  Fixed: {fixed}  |  Skipped (no actual_date): {skipped}")
    return df


def fix_false_positive_rows(
    df: pd.DataFrame,
    remove: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    For each FLAG_FALSE_POSITIVE row:
      - remove=False (default): set data_complete=False, mark v_action=FLAGGED
      - remove=True:            drop the row entirely
    """
    mask = df["v_action"] == "FLAG_FALSE_POSITIVE"
    n    = mask.sum()

    if n == 0:
        print("  No FLAG_FALSE_POSITIVE rows.")
        return df

    action_str = "Remove" if remove else "Set data_complete=False"
    print(f"  FLAG_FALSE_POSITIVE rows: {n}  →  {action_str}")

    date_col = "event_trading_date" if "event_trading_date" in df.columns else "event_date"
    if dry_run:
        print(df[mask][["ticker", date_col, "move_pct", "v_summary"]].to_string(index=False))
        return df

    if remove:
        df = df[~mask].reset_index(drop=True)
        print(f"    Removed {n} rows.")
    else:
        df.loc[mask, "data_complete"] = False
        df.loc[mask, "v_action"]      = "FLAGGED"
        print(f"    Marked {n} rows as data_complete=False (kept in dataset).")

    return df


# ============================================================================
# Main
# ============================================================================

def fix_validated_rows(
    input_file:            str,
    output_file:           str  = None,
    remove_false_positives: bool = False,
    dry_run:               bool = False,
    force_ohlc_refresh:    bool = False,
) -> pd.DataFrame:
    output_file = output_file or input_file

    print(f"\nLoading {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    if "v_action" not in df.columns:
        print("\nERROR: 'v_action' column not found. "
              "Run validate_catalysts.py first.")
        return df

    # Count actions
    action_counts = df["v_action"].value_counts().to_dict()
    print("\nv_action distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action:<25}  {count:,}")

    fix_date_df  = df[df["v_action"] == "FIX_DATE"]
    fp_df        = df[df["v_action"] == "FLAG_FALSE_POSITIVE"]

    if fix_date_df.empty and fp_df.empty:
        print("\nNothing to fix.")
        if not dry_run:
            df.to_csv(output_file, index=False)
        return df

    # ── Download OHLC for all tickers that need date fixes ──────────────────
    ohlc_cache: dict = {}
    if not fix_date_df.empty:
        print(f"\nDownloading OHLC for {fix_date_df['ticker'].nunique()} "
              f"tickers (date-fix rows)...")

        # Build a temp df with v_actual_date as event_date for date_range_for_events
        tmp = fix_date_df.copy()
        tmp["event_date"] = tmp["v_actual_date"].fillna(tmp["event_date"])

        ohlc_start, ohlc_end = date_range_for_events(tmp)
        tickers = fix_date_df["ticker"].str.upper().unique().tolist()
        ohlc_cache = load_ohlc_bulk(
            tickers, ohlc_start, ohlc_end,
            events_df=tmp,
            force_refresh=force_ohlc_refresh,
        )

    # ── Apply fixes ──────────────────────────────────────────────────────────
    print("\n── FIX_DATE ─────────────────────────────────────────────────────")
    df = fix_date_rows(df, ohlc_cache, dry_run=dry_run)

    print("\n── FLAG_FALSE_POSITIVE ──────────────────────────────────────────")
    df = fix_false_positive_rows(df, remove=remove_false_positives, dry_run=dry_run)

    # ── Summary ──────────────────────────────────────────────────────────────
    if not dry_run:
        df.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print("FIX SUMMARY")
        print(f"{'='*60}")
        new_counts = df["v_action"].value_counts().to_dict()
        date_fixed  = new_counts.get("DATE_FIXED", 0)
        flagged     = new_counts.get("FLAGGED", 0)
        remaining   = df["data_complete"].eq(False).sum() if "data_complete" in df.columns else 0

        print(f"Date corrections applied:     {date_fixed:,}")
        print(f"False positives flagged:      {flagged:,}")
        print(f"Rows with data_complete=False:{remaining:,}")
        print(f"Total rows remaining:         {len(df):,}")
        print(f"\nSaved -> {output_file}")
    else:
        print("\n[DRY RUN] No changes written.")

    return df


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply corrections from validate_catalysts.py output"
    )
    parser.add_argument("--input",  required=True,
                        help="Validated CSV from validate_catalysts.py")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: overwrite input)")
    parser.add_argument("--remove-false-positives", action="store_true",
                        help="Delete FLAG_FALSE_POSITIVE rows instead of flagging them")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing anything")
    parser.add_argument("--force-ohlc-refresh", action="store_true",
                        help="Re-download OHLC even if cached")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    fix_validated_rows(
        input_file             = args.input,
        output_file            = args.output,
        remove_false_positives = args.remove_false_positives,
        dry_run                = args.dry_run,
        force_ohlc_refresh     = args.force_ohlc_refresh,
    )

"""
expand_historical_events.py
============================
Dataset expansion — historical extension pass.

Fetches Phase 2/3 COMPLETED trials from CT.gov for a given date range,
maps each trial's lead sponsor to a stock ticker via the universe file,
computes the stock move and Wilder ATR around the completion date, then
appends new rows to the master clinical-event CSV.

This is the primary dataset-growth mechanism for pre-2023 periods where the
original collection (scan_large_moves + extend_with_pr_discovery) is sparse.

Design decisions:
  - Uses CT.gov primary_completion_date as event date proxy.  For oncology
    event-driven trials this may lag the actual readout by months, which
    introduces per-row noise but is acceptable at dataset scale.
  - Phase filter: PHASE2 + PHASE3 only (matching existing dataset).
  - Universe restriction: sponsor must map to a ticker in biotech_universe_expanded.csv.
  - Dedup: skips (ticker, nct_id) pairs already in the master.
  - ATR: Wilder RMA ewm(alpha=1/20, adjust=False), 20 trading-day lookback,
    strictly pre-event — identical to existing recompute_atr.py / volatility.py.
  - Output schema: matches enriched_all_clinical_clean_v2.csv columns exactly;
    fields that require manual or LLM enrichment are left as NaN.

Usage (from biotech_catalyst_v3/):
    python -m scripts.expand_historical_events
    python -m scripts.expand_historical_events --start 2020-01-01 --end 2022-12-31
    python -m scripts.expand_historical_events --start 2018-01-01 --end 2019-12-31
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.find_clinical_events import (
    get_trial_completions,
    map_sponsor_to_ticker,
    load_universe_mapping,
)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR  = os.path.join(BASE_DIR, "archive")

MASTER_V2    = os.path.join(ARCHIVE_DIR, "enriched_all_clinical_clean_v2.csv")
MASTER_OUT   = os.path.join(BASE_DIR, "enriched_all_clinical_clean_v3.csv")
UNIVERSE_CSV = os.path.join(BASE_DIR, "biotech_universe_expanded.csv")

# ATR parameters — must match volatility.py
ATR_LOOKBACK = 20       # trading days
ATR_ALPHA    = 1 / ATR_LOOKBACK

# Move classification thresholds — must match volatility.py
THRESHOLDS = [
    (1.5, "Noise"),
    (3.0, "Low"),
    (5.0, "Medium"),
    (8.0, "High"),
    (float("inf"), "Extreme"),
]

# Minimum rows needed to compute a reliable ATR
MIN_PRE_ROWS = 15


# ---------------------------------------------------------------------------
# ATR + move computation (inline, no utils import to avoid circular deps)
# ---------------------------------------------------------------------------

def _compute_atr_and_move(ticker: str, event_date: str) -> dict:
    """
    Download OHLC for ticker, compute Wilder ATR, and calculate the
    ATR-normalised move bracketing event_date.

    Returns dict with keys:
        price_before, price_after, move_pct, move_2d_pct,
        atr_pct, stock_movement_atr_normalized, move_class_norm,
        event_trading_date, price_at_event
    or {error: ...} on failure.
    """
    try:
        event_dt = pd.Timestamp(event_date)
        # Download 60 calendar days before + 10 after to ensure enough pre-event bars
        dl_start = (event_dt - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        dl_end   = (event_dt + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

        raw = yf.download(
            ticker, start=dl_start, end=dl_end,
            progress=False, auto_adjust=True
        )
        if raw.empty or len(raw) < 3:
            return {"error": "insufficient_data"}

        # Flatten multi-level columns (yfinance >= 0.2.31)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        # Pre-event price series (strictly before event)
        pre = raw[raw.index < event_dt].copy()
        if len(pre) < MIN_PRE_ROWS:
            return {"error": f"only_{len(pre)}_pre_event_bars"}

        # True Range
        pre = pre.copy()
        pre["_prev_close"] = pre["Close"].shift(1)
        pre["_TR"] = pre[["High", "Low", "_prev_close"]].apply(
            lambda r: max(
                r["High"] - r["Low"],
                abs(r["High"] - r["_prev_close"]),
                abs(r["Low"]  - r["_prev_close"]),
            ),
            axis=1,
        )
        pre = pre.dropna(subset=["_TR"])
        if len(pre) < MIN_PRE_ROWS:
            return {"error": "not_enough_TR_rows"}

        # Wilder RMA
        atr_series = pre["_TR"].ewm(alpha=ATR_ALPHA, adjust=False, min_periods=ATR_LOOKBACK).mean()
        atr_val    = atr_series.iloc[-1]
        last_pre_close = pre["Close"].iloc[-1]

        if last_pre_close <= 0 or atr_val <= 0:
            return {"error": "zero_price_or_atr"}

        atr_pct = (atr_val / last_pre_close) * 100

        # Post-event prices
        post = raw[raw.index > event_dt]
        if post.empty:
            return {"error": "no_post_event_data"}

        price_after = float(post["Close"].iloc[0])
        price_before = float(last_pre_close)
        event_trading_date = post.index[0].strftime("%Y-%m-%d")

        if price_before == 0:
            return {"error": "zero_price_before"}

        move_pct  = round((price_after - price_before) / price_before * 100, 2)
        move_2d_pct = None
        if len(post) >= 2:
            p2 = float(post["Close"].iloc[1])
            move_2d_pct = round((p2 - price_before) / price_before * 100, 2)

        norm = abs(move_pct) / atr_pct

        # Classify
        move_class_norm = "Extreme"
        for threshold, label in THRESHOLDS:
            if norm < threshold:
                move_class_norm = label
                break

        # Absolute % classification (simple)
        abs_m = abs(move_pct)
        if abs_m < 3:
            move_class_abs = "VeryLow"
        elif abs_m < 7:
            move_class_abs = "Low"
        elif abs_m < 15:
            move_class_abs = "Medium"
        elif abs_m < 25:
            move_class_abs = "High"
        else:
            move_class_abs = "VeryHigh"

        return {
            "price_before":               round(price_before, 4),
            "price_after":                round(price_after, 4),
            "price_at_event":             round(price_after, 4),
            "move_pct":                   move_pct,
            "move_2d_pct":                move_2d_pct,
            "atr_pct":                    round(atr_pct, 4),
            "stock_movement_atr_normalized": round(norm, 4),
            "move_class_norm":            move_class_norm,
            "move_class_abs":             move_class_abs,
            "move_class_combo":           move_class_norm,
            "event_trading_date":         event_trading_date,
            "event_type":                 "Gainer" if move_pct >= 0 else "Loser",
        }

    except Exception as e:
        return {"error": str(e)[:80]}


# ---------------------------------------------------------------------------
# Build a row in the master schema
# ---------------------------------------------------------------------------

def _build_master_row(trial: dict, ticker: str, move: dict, market_cap_m) -> dict:
    """Assemble a row that matches enriched_all_clinical_clean_v2.csv schema."""

    # Use primary_completion_date as event anchor
    event_date = trial.get("primary_completion_date") or trial.get("completion_date", "")

    # Drug name: first intervention if available
    drug_name = trial.get("interventions", "").split(";")[0].strip() or None

    # Phase normalisation: "PHASE2/PHASE3" → "Phase 2/3" etc.
    phase_raw = trial.get("phase", "")
    phase_map = {
        "PHASE1": "Phase 1", "PHASE2": "Phase 2", "PHASE3": "Phase 3",
        "PHASE4": "Phase 4", "PHASE2/PHASE3": "Phase 2/3",
        "PHASE3/PHASE4": "Phase 3/4", "NA": "N/A", "": "",
    }
    ct_phase = phase_map.get(phase_raw.upper().replace(" ", ""), phase_raw)

    return {
        # Identity
        "ticker":                       ticker,
        "event_date":                   event_date,
        "event_type":                   move.get("event_type", ""),
        "move_pct":                     move.get("move_pct"),
        "price_at_event":               move.get("price_at_event"),
        "catalyst_type":                "Clinical Data",
        "catalyst_summary":             None,

        # Drug / trial
        "drug_name":                    drug_name,
        "nct_id":                       trial.get("nct_id"),
        "indication":                   trial.get("conditions", "").split(";")[0].strip() or None,
        "is_pivotal":                   None,
        "pivotal_evidence":             None,
        "primary_endpoint_met":         None,
        "primary_endpoint_result":      None,

        # CT.gov
        "ct_official_title":            trial.get("title"),
        "ct_phase":                     ct_phase,
        "ct_enrollment":                trial.get("enrollment") or None,
        "ct_conditions":                trial.get("conditions") or None,
        "ct_status":                    "COMPLETED",
        "ct_sponsor":                   trial.get("sponsor"),
        "ct_allocation":                None,
        "ct_primary_completion":        trial.get("primary_completion_date") or trial.get("completion_date"),

        # Financials
        "market_cap_m":                 market_cap_m,
        "current_price":                None,
        "cash_position_m":              None,
        "short_percent":                None,
        "institutional_ownership":      None,
        "analyst_target":               None,
        "analyst_rating":               None,

        # Price / ATR
        "atr_pct":                      move.get("atr_pct"),
        "stock_movement_atr_normalized": move.get("stock_movement_atr_normalized"),
        "avg_daily_move":               move.get("atr_pct"),   # approx proxy
        "move_class_abs":               move.get("move_class_abs"),
        "move_class_norm":              move.get("move_class_norm"),
        "move_class_combo":             move.get("move_class_combo"),

        # Trading details
        "event_trading_date":           move.get("event_trading_date"),
        "move_2d_pct":                  move.get("move_2d_pct"),
        "price_before":                 move.get("price_before"),
        "price_after":                  move.get("price_after"),
        "stock_relative_move":          None,
        "data_complete":                None,

        # Validation columns (unfilled — manual or LLM pass later)
        "v_is_verified":                None,
        "v_actual_date":                event_date,
        "v_pr_link":                    None,
        "v_pr_date":                    None,
        "v_pr_title":                   None,
        "v_pr_key_info":                None,
        "v_is_material":                None,
        "v_confidence":                 None,
        "v_summary":                    None,
        "v_error":                      None,
        "v_action":                     None,

        # Links
        "best_event_link":              None,

        # MeSH (unfilled — run finalize_mesh.py or mesh_level1_from_nct.py after)
        "mesh_level1":                  None,
        "mesh_level1_reason":           None,
        "mesh_branches_raw":            None,
        "mesh_terms_raw":               None,
        "ct_conditions_raw":            None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def expand_historical_events(
    start_date: str = "2020-01-01",
    end_date:   str = "2022-12-31",
) -> pd.DataFrame:

    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # --- Load master ---
    if not os.path.exists(MASTER_V2):
        raise FileNotFoundError(f"Master dataset not found: {MASTER_V2}")
    master = pd.read_csv(MASTER_V2)
    print(f"Master: {len(master)} rows × {len(master.columns)} cols")

    # Build dedup key: (ticker, nct_id)
    existing_keys = set()
    for _, row in master.iterrows():
        nct = str(row.get("nct_id", "")).strip()
        tkr = str(row.get("ticker", "")).strip().upper()
        if nct.startswith("NCT") and tkr:
            existing_keys.add((tkr, nct))
    print(f"Existing (ticker, nct_id) pairs: {len(existing_keys)}")

    # --- Load universe ---
    name_map, ticker_set, ticker_to_cap = load_universe_mapping(UNIVERSE_CSV)
    print(f"Universe: {len(ticker_set)} tickers")

    # --- Fetch CT.gov completions ---
    print(f"\nFetching Phase 2/3 completions {start_date} → {end_date} ...")
    trials = get_trial_completions(start_date, end_date)
    if trials.empty:
        print("No trials found. Exiting.")
        return pd.DataFrame()
    print(f"Trials found: {len(trials)}")

    # --- Process each trial ---
    print("\nMapping to tickers + computing ATR moves ...")
    new_rows = []
    skipped_no_ticker = 0
    skipped_existing  = 0
    skipped_no_price  = 0
    errors = []

    for i, (_, trial) in enumerate(trials.iterrows(), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(trials)} | new rows: {len(new_rows)}")

        ticker = map_sponsor_to_ticker(
            trial["sponsor"], name_map, ticker_set
        )
        if not ticker:
            skipped_no_ticker += 1
            continue

        nct = str(trial.get("nct_id", "")).strip()
        if (ticker, nct) in existing_keys:
            skipped_existing += 1
            continue

        date = trial.get("primary_completion_date") or trial.get("completion_date", "")
        if not date:
            skipped_no_ticker += 1
            continue

        move = _compute_atr_and_move(ticker, date)

        if "error" in move:
            skipped_no_price += 1
            errors.append((ticker, nct, date, move["error"]))
            continue

        cap = ticker_to_cap.get(ticker)
        row = _build_master_row(trial.to_dict(), ticker, move, cap)
        new_rows.append(row)

        time.sleep(0.1)  # rate-limit yfinance

    if not new_rows:
        print("\nNo new rows produced.")
        return pd.DataFrame()

    new_df = pd.DataFrame(new_rows)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"EXPANSION RESULTS  ({start_date} → {end_date})")
    print(f"{'='*60}")
    print(f"Trials fetched:          {len(trials)}")
    print(f"  Skipped (no ticker):   {skipped_no_ticker}")
    print(f"  Skipped (existing):    {skipped_existing}")
    print(f"  Skipped (no price):    {skipped_no_price}")
    print(f"  NEW rows added:        {len(new_rows)}")
    print()
    print(f"Move class distribution (ATR-normalised):")
    print(new_df["move_class_norm"].value_counts().to_string())
    print()
    print(f"Event type:")
    print(new_df["event_type"].value_counts().to_string())
    print()
    print(f"Phase distribution:")
    print(new_df["ct_phase"].value_counts().to_string())
    print()
    year_col = pd.to_datetime(new_df["event_date"], errors="coerce").dt.year
    print(f"Year distribution:")
    print(year_col.value_counts().sort_index().to_string())
    print()

    # Oncology proxy: conditions containing cancer/oncol/tumor/neoplas/leukemia
    onco_kw = r"cancer|oncol|tumor|tumour|neoplas|leukemia|lymphoma|myeloma|carcinoma|sarcoma|glioma|melanoma"
    is_onco = (
        new_df["ct_conditions"].str.lower().str.contains(onco_kw, na=False) |
        new_df["indication"].str.lower().str.contains(onco_kw, na=False)
    )
    print(f"Oncology rows (proxy):  {is_onco.sum()} / {len(new_df)} ({is_onco.mean()*100:.1f}%)")
    print(f"Rows with nct_id:       {new_df['nct_id'].notna().sum()}")
    print(f"Rows with ATR filled:   {new_df['atr_pct'].notna().sum()}")
    print(f"Immediately ML-usable:  {new_df['move_class_norm'].notna().sum()} (have move_class_norm)")

    # --- Append and save ---
    # Align columns to master
    master_cols = master.columns.tolist()
    for col in master_cols:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[master_cols]

    # Dedup new_df against master by nct_id only (master is authoritative — preserve all its rows)
    master_ncts = set(master["nct_id"].dropna().astype(str).str.strip())
    new_df_clean = new_df[~new_df["nct_id"].astype(str).str.strip().isin(master_ncts)].copy()
    print(f"New rows after nct_id dedup: {len(new_df_clean)} (removed {len(new_df)-len(new_df_clean)} nct_id overlaps)")
    combined = pd.concat([master, new_df_clean], ignore_index=True)

    # Archive old master if a v3 already exists
    if os.path.exists(MASTER_OUT):
        from shutil import copy
        copy(MASTER_OUT, os.path.join(ARCHIVE_DIR, "enriched_all_clinical_clean_v3_prev.csv"))

    combined.to_csv(MASTER_OUT, index=False)
    print(f"\nSaved: enriched_all_clinical_clean_v3.csv")
    print(f"  Previous master: {len(master)} rows")
    print(f"  New rows added:  {len(new_df)}")
    print(f"  Combined total:  {len(combined)} rows")

    new_year = pd.to_datetime(combined["v_actual_date"], errors="coerce").dt.year
    print(f"\nFull dataset year distribution:")
    print(new_year.value_counts().sort_index().to_string())

    print(f"\nFull dataset move_class_norm:")
    print(combined["move_class_norm"].value_counts().to_string())

    return new_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand master clinical event dataset with historical CT.gov completions"
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2022-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    expand_historical_events(start_date=args.start, end_date=args.end)

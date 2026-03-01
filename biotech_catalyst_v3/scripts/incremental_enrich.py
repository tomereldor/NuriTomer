"""Incrementally enrich new events without re-processing existing data.

Takes a file of raw events (ticker + event_date + nct_id + move_pct) and
enriches each with:
  - ClinicalTrials.gov details (if nct_id present)
  - Financial data (market cap, cash, etc.) via yfinance
  - ATR normalization and move classification

Then merges with the existing enriched dataset, skipping any duplicates.

Usage:
    python -m scripts.incremental_enrich
    python -m scripts.incremental_enrich --new clinical_events_new.csv --existing enriched_high_moves.csv
    python -m scripts.incremental_enrich --new large_moves_new.csv --output enriched_all.csv
"""

import os
import sys
import time
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.clinicaltrials_client import ClinicalTrialsClient
from clients.financial_client import FinancialDataFetcher
from utils.volatility import calculate_atr, classify_move
from utils.data_quality import add_quality_threshold, fix_catalyst_type


# ---------------------------------------------------------------------------
# Single-event enrichment
# ---------------------------------------------------------------------------

def enrich_event(row: pd.Series, ct_client: ClinicalTrialsClient, fin_fetcher: FinancialDataFetcher) -> dict:
    """Enrich one event row with CT.gov, financial, and ATR data."""
    result = row.to_dict()

    ticker = str(result.get("ticker", "")).upper()
    nct_id = str(result.get("nct_id", "")).strip()

    # ------------------------------------------------------------------
    # ClinicalTrials.gov details
    # ------------------------------------------------------------------
    if nct_id and nct_id.startswith("NCT"):
        try:
            trial = ct_client.fetch_trial_details(nct_id)
            if trial:
                result["ct_official_title"] = trial.official_title
                result["ct_phase"] = trial.phase
                result["ct_enrollment"] = trial.enrollment
                result["ct_conditions"] = trial.conditions
                result["ct_status"] = trial.status
                result["ct_sponsor"] = trial.sponsor
                result["ct_allocation"] = trial.allocation
                result["ct_primary_completion"] = trial.primary_completion_date
        except Exception as e:
            result.setdefault("errors", "")
            result["errors"] = f"{result['errors']}; CT error: {str(e)[:60]}".strip("; ")

    # ------------------------------------------------------------------
    # Financial data
    # ------------------------------------------------------------------
    try:
        fin = fin_fetcher.fetch(ticker)
        result["market_cap_m"] = fin.market_cap_m
        result["current_price"] = fin.current_price
        result["cash_position_m"] = fin.cash_position_m
        result["cash_runway_months"] = fin.cash_runway_months
        result["short_percent"] = fin.short_percent
        result["institutional_ownership"] = fin.institutional_ownership
        result["analyst_target"] = fin.analyst_target
        result["analyst_rating"] = fin.analyst_rating
        if fin.error:
            result.setdefault("errors", "")
            result["errors"] = f"{result['errors']}; {fin.error}".strip("; ")
        if fin.missing_fields:
            result.setdefault("errors", "")
            result["errors"] = f"{result['errors']}; Missing: {fin.missing_fields}".strip("; ")
    except Exception as e:
        result.setdefault("errors", "")
        result["errors"] = f"{result['errors']}; Fin error: {str(e)[:60]}".strip("; ")

    # ------------------------------------------------------------------
    # ATR normalization
    # ------------------------------------------------------------------
    event_date = str(result.get("event_trading_date") or result.get("event_date") or "")
    move_pct = result.get("move_pct")

    if event_date and pd.notna(move_pct):
        try:
            atr = calculate_atr(ticker, event_date)
            result["atr_pct"] = atr.get("atr_pct")
            result["avg_daily_move"] = atr.get("avg_daily_move")

            if atr.get("atr_pct"):
                cls = classify_move(float(move_pct), atr["atr_pct"])
                result.update(cls)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Column alignment helper
# ---------------------------------------------------------------------------

def _align_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure both DataFrames have the same columns (fill missing with None)."""
    all_cols = list(dict.fromkeys(list(df_a.columns) + list(df_b.columns)))
    for col in all_cols:
        if col not in df_a.columns:
            df_a[col] = None
        if col not in df_b.columns:
            df_b[col] = None
    return df_a[all_cols], df_b[all_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def incremental_enrich(
    new_events_file: str = "clinical_events_new.csv",
    existing_file: str = "enriched_high_moves.csv",
    output_file: str = "enriched_all.csv",
    save_every: int = 20,
    delay: float = 0.2,
):
    """
    Enrich new_events_file and merge with existing_file → output_file.

    Deduplication key: (ticker, event_date).
    Partial results are written every save_every rows.
    """

    # ------------------------------------------------------------------
    # Load existing enriched data
    # ------------------------------------------------------------------
    if os.path.exists(existing_file):
        existing = pd.read_csv(existing_file)
        existing_keys = set(
            zip(existing["ticker"].str.upper(), existing["event_date"].astype(str))
        )
        print(f"Existing events: {len(existing)}")
    else:
        existing = pd.DataFrame()
        existing_keys = set()
        print("No existing file — starting fresh.")

    # ------------------------------------------------------------------
    # Load new candidates and filter to truly new
    # ------------------------------------------------------------------
    new_raw = pd.read_csv(new_events_file)
    new_raw["_key"] = list(zip(new_raw["ticker"].str.upper(), new_raw["event_date"].astype(str)))
    truly_new = new_raw[~new_raw["_key"].isin(existing_keys)].drop(columns=["_key"]).reset_index(drop=True)

    print(f"New candidates:   {len(new_raw)}")
    print(f"Already in dataset (skipped): {len(new_raw) - len(truly_new)}")
    print(f"To enrich:        {len(truly_new)}")

    if truly_new.empty:
        print("Nothing new to enrich.")
        if not existing.empty:
            existing.to_csv(output_file, index=False)
            print(f"Existing data copied to {output_file}")
        return existing

    # ------------------------------------------------------------------
    # Check for partial results from a previous interrupted run
    # ------------------------------------------------------------------
    partial_results = []
    partial_keys: set = set()
    partial_file = output_file.replace(".csv", "_partial.csv")

    if os.path.exists(partial_file):
        partial_df = pd.read_csv(partial_file)
        # Only keep partial rows that are not in existing
        partial_new = partial_df[~partial_df.apply(
            lambda r: (str(r["ticker"]).upper(), str(r["event_date"])) in existing_keys, axis=1
        )]
        partial_results = partial_new.to_dict("records")
        partial_keys = set(zip(partial_new["ticker"].str.upper(), partial_new["event_date"].astype(str)))
        print(f"Resuming from partial: {len(partial_results)} already enriched")

    # ------------------------------------------------------------------
    # Initialize clients
    # ------------------------------------------------------------------
    ct_client = ClinicalTrialsClient(rate_limit=0.3)
    fin_fetcher = FinancialDataFetcher()

    # ------------------------------------------------------------------
    # Enrich row by row
    # ------------------------------------------------------------------
    enriched = list(partial_results)
    skipped_partial = 0

    for i, (_, row) in enumerate(truly_new.iterrows()):
        row_key = (str(row["ticker"]).upper(), str(row["event_date"]))

        # Skip if already done in a partial run
        if row_key in partial_keys:
            skipped_partial += 1
            continue

        if (i + 1) % 10 == 0:
            print(f"  Enriching {i+1}/{len(truly_new)} | done: {len(enriched) - skipped_partial}")

        result = enrich_event(row, ct_client, fin_fetcher)
        enriched.append(result)
        time.sleep(delay)

        # Save partial progress
        if len(enriched) % save_every == 0:
            pd.DataFrame(enriched).to_csv(partial_file, index=False)

    enriched_df = pd.DataFrame(enriched)

    # ------------------------------------------------------------------
    # Merge with existing
    # ------------------------------------------------------------------
    if not existing.empty:
        existing, enriched_df = _align_columns(existing, enriched_df)
        combined = pd.concat([existing, enriched_df], ignore_index=True)
    else:
        combined = enriched_df

    # Post-processing
    if "data_quality_score" in combined.columns:
        combined = add_quality_threshold(combined, threshold=0.7)
    combined["catalyst_type"] = combined.apply(fix_catalyst_type, axis=1)

    combined.to_csv(output_file, index=False)

    # Clean up partial file on success
    if os.path.exists(partial_file):
        os.remove(partial_file)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    new_added = len(enriched_df)
    print(f"\n{'='*55}")
    print(f"INCREMENTAL ENRICHMENT COMPLETE")
    print(f"{'='*55}")
    print(f"Previously: {len(existing)} events")
    print(f"Added:      {new_added} events")
    print(f"Total:      {len(combined)} events")
    if "catalyst_type" in combined.columns:
        clinical = (combined["catalyst_type"] == "Clinical Data").sum()
        print(f"Clinical Data events: {clinical}/{len(combined)}")
    if "move_class_combo" in combined.columns:
        print(f"move_class_combo dist: {combined['move_class_combo'].value_counts().to_dict()}")
    print(f"Saved to: {output_file}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally enrich new events")
    parser.add_argument("--new", default="clinical_events_new.csv", help="New events file to enrich")
    parser.add_argument("--existing", default="enriched_high_moves.csv", help="Existing enriched dataset")
    parser.add_argument("--output", default="enriched_all.csv", help="Combined output file")
    parser.add_argument("--save-every", type=int, default=20, help="Save partial results every N rows")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between API calls")
    args = parser.parse_args()

    incremental_enrich(
        new_events_file=args.new,
        existing_file=args.existing,
        output_file=args.output,
        save_every=args.save_every,
        delay=args.delay,
    )

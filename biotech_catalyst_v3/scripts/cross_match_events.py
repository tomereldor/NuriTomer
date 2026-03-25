"""Phase 4 (A2): Cross-match large stock moves with CT.gov completions.

For each large move (≥10%, 2018-2022), check if a CT.gov Phase 2/3 completion
exists for the same ticker within ±WINDOW_DAYS calendar days.

Strategy:
  - Matched large moves → confirmed clinical catalysts (positive training examples)
  - CT.gov completions with |move_pct| < 10% → natural negatives (clinical + no large move)
  - Unmatched large moves → saved for optional Perplexity classification

Outputs:
  data/confirmed_catalysts_2018_2022.csv   — positives + negatives ready for enrichment
  data/unmatched_large_moves_2018_2022.csv — large moves without CT.gov match

Usage:
    python -m scripts.cross_match_events
    python -m scripts.cross_match_events --window 10 --neg-sample 1000
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WINDOW_DAYS = 10   # ±10 calendar days for cross-match


# ---------------------------------------------------------------------------
# CT.gov API enrichment
# ---------------------------------------------------------------------------

def fetch_ctgov_details(nct_id: str) -> dict:
    """Fetch official title, allocation, primary_completion, sponsor from CT.gov API."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        r = requests.get(url, params={"format": "json"}, timeout=30)
        if r.status_code != 200:
            return {}
        data = r.json()
        proto = data.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        # allocation: RANDOMIZED | NON_RANDOMIZED | NA
        alloc_mod = design.get("enrollmentInfo", {})  # not here
        alloc = (
            design.get("designInfo", {}).get("allocation", "")
            or ""
        )

        return {
            "ct_official_title": ident.get("officialTitle") or ident.get("briefTitle", ""),
            "ct_sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "ct_allocation": alloc,
            "ct_primary_completion": (
                status_mod.get("primaryCompletionDateStruct", {}).get("date", "")
                or status_mod.get("completionDateStruct", {}).get("date", "")
            ),
        }
    except Exception as e:
        return {}


# ---------------------------------------------------------------------------
# Move class helpers (mirror of existing pipeline)
# ---------------------------------------------------------------------------

MOVE_CLASS_ABS_BINS = [0, 5, 10, 20, float("inf")]
MOVE_CLASS_ABS_LABELS = ["Low", "Medium", "High", "Extreme"]

NORM_BINS  = [0, 1.5, 3.0, 5.0, 8.0, float("inf")]
NORM_LABELS = ["Noise", "Low", "Medium", "High", "Extreme"]


def compute_move_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Add move_class_abs, move_class_norm, stock_movement_atr_normalized."""
    abs_move = df["move_pct"].abs()

    df["move_class_abs"] = pd.cut(
        abs_move, bins=MOVE_CLASS_ABS_BINS, labels=MOVE_CLASS_ABS_LABELS, right=False
    ).astype(str)

    # ATR-normalised move (only where atr_pct is available and nonzero)
    atr = df["atr_pct"].replace(0, np.nan)
    df["stock_movement_atr_normalized"] = (abs_move / atr).round(3)

    df["move_class_norm"] = pd.cut(
        df["stock_movement_atr_normalized"],
        bins=NORM_BINS, labels=NORM_LABELS, right=False
    ).astype(str)
    df.loc[df["stock_movement_atr_normalized"].isna(), "move_class_norm"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cross_match(
    large_moves_file: str = "data/large_moves_2018_2022.csv",
    ctgov_file: str = "data/ctgov_completions_2018_2022.csv",
    output_catalysts: str = "data/confirmed_catalysts_2018_2022.csv",
    output_unmatched: str = "data/unmatched_large_moves_2018_2022.csv",
    window_days: int = WINDOW_DAYS,
    neg_sample: int = 1000,       # max CT.gov small-move negatives to include
    enrich_ctgov: bool = True,    # fetch full CT.gov details (requires network)
    enrich_delay: float = 0.3,    # seconds between CT.gov API calls
):
    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("Loading large_moves...", flush=True)
    lm = pd.read_csv(large_moves_file)
    lm["event_date"] = pd.to_datetime(lm["event_date"])
    lm["ticker"] = lm["ticker"].str.upper()
    print(f"  {len(lm):,} large moves loaded")

    print("Loading ctgov_completions...", flush=True)
    cg = pd.read_csv(ctgov_file)
    cg["event_date"] = pd.to_datetime(cg["event_date"], errors="coerce")
    cg["ticker"] = cg["ticker"].str.upper()
    cg = cg.dropna(subset=["event_date"])
    print(f"  {len(cg):,} CT.gov completions loaded")

    # ------------------------------------------------------------------
    # Build CT.gov lookup: ticker → list of rows
    # ------------------------------------------------------------------
    from collections import defaultdict
    ctgov_by_ticker: dict = defaultdict(list)
    for _, row in cg.iterrows():
        ctgov_by_ticker[row["ticker"]].append(row)

    # ------------------------------------------------------------------
    # Cross-match
    # ------------------------------------------------------------------
    print(f"\nCross-matching (window ±{window_days} days)...", flush=True)

    matched_rows = []
    unmatched_rows = []
    matched_nct_ids = set()

    for _, move in lm.iterrows():
        ticker = move["ticker"]
        move_date = move["event_date"]

        best_match = None
        best_diff = float("inf")

        for ct_row in ctgov_by_ticker.get(ticker, []):
            diff = abs((ct_row["event_date"] - move_date).days)
            if diff <= window_days and diff < best_diff:
                best_diff = diff
                best_match = ct_row

        if best_match is not None:
            matched_nct_ids.add(best_match["nct_id"])
            matched_rows.append({
                "ticker": ticker,
                "event_date": move_date.strftime("%Y-%m-%d"),
                "v_actual_date": move_date.strftime("%Y-%m-%d"),
                "event_type": move["event_type"],
                "move_pct": move["move_pct"],
                "atr_pct": move["atr_pct"],
                "normalized_move": move["normalized_move"],
                "close": move.get("close"),
                "market_cap_m": move.get("market_cap_m"),
                # CT.gov metadata
                "nct_id": best_match["nct_id"],
                "drug_name": (best_match.get("interventions", "") or "").split(";")[0].strip(),
                "indication": best_match.get("conditions", ""),
                "ct_conditions": best_match.get("conditions", ""),
                "ct_phase": best_match.get("phase", ""),
                "ct_enrollment": best_match.get("enrollment", np.nan),
                "ct_status": "COMPLETED",
                "ct_sponsor": best_match.get("sponsor", ""),
                # CT.gov details to be enriched below
                "ct_official_title": "",
                "ct_allocation": "",
                "ct_primary_completion": "",
                # Labels and flags
                "catalyst_type": "Clinical Data",
                "v_is_material": True,
                "data_tier": "ctgov_crossmatch",
                "match_type": "ctgov_cross_match",
                "days_offset": best_diff,
            })
        else:
            unmatched_rows.append({
                "ticker": ticker,
                "event_date": move_date.strftime("%Y-%m-%d"),
                "move_pct": move["move_pct"],
                "atr_pct": move["atr_pct"],
                "normalized_move": move["normalized_move"],
                "event_type": move["event_type"],
                "market_cap_m": move.get("market_cap_m"),
                "match_type": "unmatched",
                "catalyst_type": "",
            })

    n_matched = len(matched_rows)
    n_unmatched = len(unmatched_rows)
    print(f"  Matched:   {n_matched:,} large moves confirmed as clinical catalysts")
    print(f"  Unmatched: {n_unmatched:,} large moves without CT.gov match")

    # ------------------------------------------------------------------
    # CT.gov small-move negatives (|move_pct| < 10%, not in matched set)
    # ------------------------------------------------------------------
    cg_negs = cg[
        (cg["move_pct"].abs() < 10.0) &
        (~cg["nct_id"].isin(matched_nct_ids)) &
        cg["move_pct"].notna()
    ].copy()

    if len(cg_negs) > neg_sample:
        cg_negs = cg_negs.sample(n=neg_sample, random_state=42)

    print(f"  CT.gov negatives sampled: {len(cg_negs):,} (from {len(cg[cg['move_pct'].abs() < 10.0]):,} available)")

    neg_rows = []
    for _, row in cg_negs.iterrows():
        neg_rows.append({
            "ticker": row["ticker"],
            "event_date": row["event_date"].strftime("%Y-%m-%d"),
            "v_actual_date": row["event_date"].strftime("%Y-%m-%d"),
            "event_type": row.get("event_type", ""),
            "move_pct": row.get("move_pct"),
            "atr_pct": np.nan,   # not computed for small-move events
            "normalized_move": np.nan,
            "close": row.get("price_after"),
            "market_cap_m": row.get("market_cap_m"),
            "nct_id": row.get("nct_id", ""),
            "drug_name": (row.get("interventions", "") or "").split(";")[0].strip(),
            "indication": row.get("conditions", ""),
            "ct_conditions": row.get("conditions", ""),
            "ct_phase": row.get("phase", ""),
            "ct_enrollment": row.get("enrollment", np.nan),
            "ct_status": "COMPLETED",
            "ct_sponsor": row.get("sponsor", ""),
            "ct_official_title": "",
            "ct_allocation": "",
            "ct_primary_completion": "",
            "catalyst_type": "Clinical Data",
            "v_is_material": True,
            "data_tier": "ctgov_completion",
            "match_type": "ctgov_neg_sample",
            "days_offset": 0,
        })

    # ------------------------------------------------------------------
    # Optionally enrich matched events with full CT.gov details
    # ------------------------------------------------------------------
    if enrich_ctgov and n_matched > 0:
        unique_nct_ids = list({r["nct_id"] for r in matched_rows if r.get("nct_id")})
        print(f"\nFetching CT.gov details for {len(unique_nct_ids)} unique NCT IDs...", flush=True)

        nct_details: dict = {}
        for i, nct_id in enumerate(unique_nct_ids):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(unique_nct_ids)} fetched", flush=True)
            details = fetch_ctgov_details(nct_id)
            if details:
                nct_details[nct_id] = details
            time.sleep(enrich_delay)

        print(f"  Retrieved details for {len(nct_details)} NCT IDs")

        # Apply enrichment to matched rows
        for row in matched_rows:
            nct_id = row.get("nct_id", "")
            if nct_id in nct_details:
                d = nct_details[nct_id]
                row["ct_official_title"] = d.get("ct_official_title", "")
                row["ct_sponsor"]        = d.get("ct_sponsor", "") or row["ct_sponsor"]
                row["ct_allocation"]     = d.get("ct_allocation", "")
                row["ct_primary_completion"] = d.get("ct_primary_completion", "")

    # ------------------------------------------------------------------
    # Assemble output DataFrames
    # ------------------------------------------------------------------
    all_catalysts = matched_rows + neg_rows
    df_catalysts = pd.DataFrame(all_catalysts)

    # Compute move classes
    df_catalysts = compute_move_classes(df_catalysts)

    # target_large_move: 1 if |move_pct| >= 10 AND |stock_movement_atr_normalized| >= 3
    # For rows without ATR, fall back to |move_pct| >= 10 only (conservative)
    abs_move = df_catalysts["move_pct"].abs()
    has_atr = df_catalysts["stock_movement_atr_normalized"].notna()
    df_catalysts["target_large_move"] = (
        (abs_move >= 10.0) & (
            (~has_atr) |   # no ATR → rely on move_pct only (large moves are already ≥10%)
            (df_catalysts["stock_movement_atr_normalized"] >= 3.0)
        )
    ).astype(int)
    # For negatives (|move_pct| < 10%), force target = 0
    df_catalysts.loc[abs_move < 10.0, "target_large_move"] = 0

    df_catalysts = df_catalysts.sort_values("event_date").reset_index(drop=True)
    df_catalysts.to_csv(output_catalysts, index=False)

    # Save unmatched
    df_unmatched = pd.DataFrame(unmatched_rows).sort_values("move_pct", key=abs, ascending=False)
    df_unmatched.to_csv(output_unmatched, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pos = df_catalysts["target_large_move"].sum()
    n_neg = len(df_catalysts) - n_pos
    pos_rate = n_pos / len(df_catalysts) * 100 if len(df_catalysts) > 0 else 0

    print(f"\n{'='*60}")
    print(f"CROSS-MATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Confirmed catalysts:")
    print(f"  Positives (large move + clinical): {n_pos:,}")
    print(f"  Negatives (clinical + small move):  {n_neg:,}")
    print(f"  Total:                              {len(df_catalysts):,}")
    print(f"  Positive rate:                      {pos_rate:.1f}%")
    print(f"\nUnmatched large moves saved:          {len(df_unmatched):,}")
    print(f"\nSaved to: {output_catalysts}")
    print(f"Unmatched: {output_unmatched}")

    if n_matched > 0:
        print(f"\nTop 10 matched positives (by |move_pct|):")
        top = df_catalysts[df_catalysts["target_large_move"] == 1].nlargest(10, "move_pct", keep="all")
        print(top[["ticker", "event_date", "move_pct", "atr_pct", "ct_phase", "indication"]].to_string(index=False))

    return df_catalysts, df_unmatched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-match large moves with CT.gov completions")
    parser.add_argument("--large-moves",  default="data/large_moves_2018_2022.csv")
    parser.add_argument("--ctgov",        default="data/ctgov_completions_2018_2022.csv")
    parser.add_argument("--output",       default="data/confirmed_catalysts_2018_2022.csv")
    parser.add_argument("--unmatched",    default="data/unmatched_large_moves_2018_2022.csv")
    parser.add_argument("--window",       type=int,   default=WINDOW_DAYS)
    parser.add_argument("--neg-sample",   type=int,   default=1000)
    parser.add_argument("--no-enrich",    action="store_true", help="Skip CT.gov API enrichment")
    args = parser.parse_args()

    cross_match(
        large_moves_file=args.large_moves,
        ctgov_file=args.ctgov,
        output_catalysts=args.output,
        output_unmatched=args.unmatched,
        window_days=args.window,
        neg_sample=args.neg_sample,
        enrich_ctgov=not args.no_enrich,
    )

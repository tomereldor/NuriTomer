"""
completeness_pass.py
====================
Final completeness pass over the latest ML feature dataset.

Three recovery stages (cheapest-first):

  Stage 1 — ClinicalTrials.gov (deterministic, free API)
    Fill missing clinical fields for rows that have nct_id:
      ct_allocation (120 rows), ct_phase (10 rows), ct_enrollment, ct_conditions,
      ct_status, ct_sponsor, ct_primary_completion, ct_official_title
    Also derive indication from ct_conditions where indication is missing.

  Stage 2 — Finance / yfinance (deterministic, bulk cached)
    Fill price_before, price_after, move_2d_pct for ~196 rows.

  Stage 3 — Perplexity (LLM, narrow whitelist only)
    Fill: indication, is_pivotal, pivotal_evidence,
          primary_endpoint_met, primary_endpoint_result
    Only targets rows that:
      - have catalyst_summary OR v_summary (source text available)
      - are still missing one of the above fields
      - prioritise Phase 3 > Phase 2/3 > Phase 2 > other
    Default limit: 300 rows (configurable via --perplexity-limit)

After all fills: re-derive all downstream feature columns.
Saves back to the SAME file (in-place update, no version bump).

Usage:
    cd biotech_catalyst_v3
    python -m scripts.completeness_pass
    python -m scripts.completeness_pass --skip-perplexity
    python -m scripts.completeness_pass --perplexity-limit 100
    python -m scripts.completeness_pass --input ml_dataset_features_20260310_v2.csv
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.clinicaltrials_client import ClinicalTrialsClient
from utils.ohlc_cache import load_ohlc_bulk, date_range_for_events

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_FILE = "ml_dataset_features_20260310_v2.csv"

PHASE_MAP = {
    "Phase 1 (Early)": 0.5,
    "Phase 1":         1.0,
    "Phase 1/2":       1.5,
    "Phase 2":         2.0,
    "Phase 2/3":       2.5,
    "Phase 3":         3.0,
    "Phase 4":         4.0,
}

MESH_ENCODE_MAP = {
    "Neoplasms":                  1,
    "Nervous System Diseases":    2,
    "Immune System Diseases":     3,
    "Endocrine System Diseases":  4,
    "Respiratory Tract Diseases": 5,
    "Infectious Diseases":        6,
    "Cardiovascular Diseases":    7,
    "Digestive System Diseases":  8,
    "Skin Diseases":              9,
    "Musculoskeletal Diseases":   10,
    "Other / Non-Disease":        11,
}

PROXIMITY_BUCKETS = [
    (365,           float("inf"),  "future_far"),
    (0,             365,           "future_near"),
    (-180,          0,             "just_completed"),
    (float("-inf"), -180,          "past"),
]

PRIORITY_COLS = [
    "mesh_level1_encoded", "indication", "ct_phase", "ct_enrollment",
    "ct_conditions", "ct_status", "ct_sponsor", "ct_allocation",
    "ct_primary_completion", "price_before", "price_after", "move_2d_pct",
    "primary_endpoint_met", "primary_endpoint_result", "is_pivotal", "pivotal_evidence",
]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Stage 1: ClinicalTrials.gov
# ---------------------------------------------------------------------------

def fill_from_ctgov(df: pd.DataFrame) -> tuple:
    """Fetch missing clinical fields for rows that have nct_id."""
    client = ClinicalTrialsClient()

    # Clinical columns the client can fill
    CT_COLS = [
        "ct_phase", "ct_enrollment", "ct_conditions", "ct_status",
        "ct_sponsor", "ct_allocation", "ct_primary_completion", "ct_official_title",
    ]

    has_nct = df["nct_id"].notna()
    missing_any = df[CT_COLS].isna().any(axis=1)
    candidates = df.index[has_nct & missing_any].tolist()

    print(f"  Candidates: {len(candidates)} rows with nct_id but missing clinical field(s)")

    counts = {c: 0 for c in CT_COLS}
    errors = 0

    for i, idx in enumerate(candidates):
        nct_id = str(df.at[idx, "nct_id"]).strip()
        result = client.fetch_trial_details(nct_id)

        if result is None:
            errors += 1
            time.sleep(0.5)
            continue

        fills = {
            "ct_phase":              result.phase,
            "ct_enrollment":         result.enrollment,
            "ct_conditions":         result.conditions,
            "ct_status":             result.status,
            "ct_sponsor":            result.sponsor,
            "ct_allocation":         result.allocation,
            "ct_primary_completion": result.primary_completion_date,
            "ct_official_title":     result.official_title or result.title,
        }

        for col, val in fills.items():
            if pd.isna(df.at[idx, col]) and val:
                df.at[idx, col] = val
                counts[col] += 1

        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(candidates)}] done, errors so far: {errors}")

        time.sleep(0.5)

    # Derive indication from ct_conditions where indication still null
    ind_from_ct = 0
    for idx in df.index[df["indication"].isna() & df["ct_conditions"].notna()]:
        first = str(df.at[idx, "ct_conditions"]).split("|")[0].strip()
        if first:
            df.at[idx, "indication"] = first
            ind_from_ct += 1

    counts["indication_from_ct_conditions"] = ind_from_ct
    counts["fetch_errors"] = errors
    return df, counts


# ---------------------------------------------------------------------------
# Stage 2: Finance (price_before / price_after / move_2d_pct)
# ---------------------------------------------------------------------------

def _close_before(ohlc_df, target_date):
    if ohlc_df is None or ohlc_df.empty:
        return None
    target = pd.Timestamp(target_date).normalize()
    pre = ohlc_df[ohlc_df.index.normalize() < target]
    if pre.empty:
        return None
    val = pre["Close"].iloc[-1]
    return round(float(val), 4) if pd.notna(val) else None


def _close_after(ohlc_df, target_date):
    if ohlc_df is None or ohlc_df.empty:
        return None
    target = pd.Timestamp(target_date).normalize()
    post = ohlc_df[ohlc_df.index.normalize() > target]
    if post.empty:
        return None
    val = post["Close"].iloc[0]
    return round(float(val), 4) if pd.notna(val) else None


def fill_prices(df: pd.DataFrame) -> tuple:
    """Fill price_before, price_after, move_2d_pct via yfinance OHLC cache."""
    missing_mask = (
        df["price_before"].isna() |
        df["price_after"].isna()  |
        df["move_2d_pct"].isna()
    )
    to_fill = df[missing_mask].copy()
    print(f"  Candidates: {len(to_fill)} rows missing price_before / price_after / move_2d_pct")

    if to_fill.empty:
        return df, {"filled_price_before": 0, "filled_price_after": 0, "filled_move_2d_pct": 0}

    date_col = "event_trading_date" if "event_trading_date" in to_fill.columns else "event_date"
    to_fill["_date"] = to_fill[date_col].fillna(to_fill["event_date"])

    # Build minimal temp df with ticker + event_date for ohlc_cache helpers
    tmp = to_fill[["ticker", "_date"]].rename(columns={"_date": "event_date"})
    ohlc_start, ohlc_end = date_range_for_events(tmp)
    tickers = to_fill["ticker"].str.upper().unique().tolist()
    print(f"  Downloading OHLC for {len(tickers)} tickers ({ohlc_start} → {ohlc_end})")

    ohlc = load_ohlc_bulk(
        tickers, ohlc_start, ohlc_end,
        events_df=tmp,
    )

    filled_before = filled_after = filled_2d = 0
    for idx, row in to_fill.iterrows():
        ticker  = str(row["ticker"]).upper()
        edate   = str(row["_date"])
        ohlc_df = ohlc.get(ticker)
        if ohlc_df is None:
            continue

        pb = _close_before(ohlc_df, edate)
        pa = _close_after(ohlc_df, edate)

        if pb is not None and pd.isna(df.at[idx, "price_before"]):
            df.at[idx, "price_before"] = pb
            filled_before += 1
        if pa is not None and pd.isna(df.at[idx, "price_after"]):
            df.at[idx, "price_after"] = pa
            filled_after += 1

        pb_final = df.at[idx, "price_before"]
        pa_final = df.at[idx, "price_after"]
        if pd.notna(pb_final) and pd.notna(pa_final) and pb_final > 0:
            if pd.isna(df.at[idx, "move_2d_pct"]):
                df.at[idx, "move_2d_pct"] = round((pa_final - pb_final) / pb_final * 100, 2)
                filled_2d += 1

    return df, {
        "filled_price_before": filled_before,
        "filled_price_after":  filled_after,
        "filled_move_2d_pct":  filled_2d,
    }


# ---------------------------------------------------------------------------
# Stage 3: Perplexity (narrow — only high-value fields with source text)
# ---------------------------------------------------------------------------

def _perplexity_call(ticker, date, drug, summary, ct_phase, ct_conditions, api_key, model):
    context = f"Drug: {drug}" if drug else ""
    if ct_phase:
        context += f", Trial phase: {ct_phase}"
    if ct_conditions:
        context += f", Conditions: {ct_conditions}"

    prompt = (
        f"For biotech stock {ticker} around {date}, clinical trial data was released.\n"
        f"{context}\n"
        f"Summary: {summary}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "indication": "specific disease/condition (e.g. non-small cell lung cancer)",\n'
        '  "is_pivotal": true or false,\n'
        '  "pivotal_evidence": "brief reason if pivotal, else null",\n'
        '  "primary_endpoint_met": "Yes, No, Unclear, or Mixed",\n'
        '  "primary_endpoint_result": "one-sentence key result"\n'
        "}"
    )

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return {"_error": f"HTTP {resp.status_code}"}
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"```json|```", "", content).strip()
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            start = content.index("{")
            end   = content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception:
            return {"_error": "parse_error"}
    except Exception as e:
        return {"_error": str(e)[:80]}


def fill_perplexity(df: pd.DataFrame, limit: int = 300) -> tuple:
    """
    Perplexity pass — narrow whitelist of high-value fields only.

    Target selection:
      - Have source text (catalyst_summary OR v_summary)
      - Still missing at least one of: indication, primary_endpoint_met, is_pivotal
      - Phase priority: Phase 3 → Phase 2/3 → Phase 2 → other
      - Hard limit on total calls (--perplexity-limit)
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    model   = os.environ.get("PERPLEXITY_MODEL", "sonar")

    if not api_key:
        print("  PERPLEXITY_API_KEY not set — skipping Perplexity stage")
        return df, {"skipped": "no_api_key"}

    has_text = (
        df["catalyst_summary"].notna() |
        df.get("v_summary", pd.Series(dtype="object")).notna()
    )
    missing_target = (
        df["indication"].isna()          |
        df["primary_endpoint_met"].isna() |
        df["is_pivotal"].isna()
    )

    eligible = df[has_text & missing_target].copy()
    phase_rank = {"Phase 3": 0, "Phase 2/3": 1, "Phase 2": 2}
    eligible["_phase_rank"] = eligible["ct_phase"].map(phase_rank).fillna(3)
    eligible = eligible.sort_values("_phase_rank").head(limit)

    print(f"  Candidates: {len(eligible)} rows (limit={limit})")
    if eligible.empty:
        return df, {}

    counts = {
        "indication": 0, "primary_endpoint_met": 0,
        "primary_endpoint_result": 0, "is_pivotal": 0,
        "pivotal_evidence": 0, "errors": 0,
    }

    for i, (idx, row) in enumerate(eligible.iterrows()):
        ticker  = str(row.get("ticker", ""))
        date    = str(row.get("event_date", ""))
        drug    = str(row.get("drug_name", "")) if pd.notna(row.get("drug_name")) else ""
        summary = (
            row.get("v_summary")
            if pd.notna(row.get("v_summary"))
            else row.get("catalyst_summary", "")
        )
        ct_phase = str(row.get("ct_phase", "")) if pd.notna(row.get("ct_phase")) else ""
        ct_conds = str(row.get("ct_conditions", "")) if pd.notna(row.get("ct_conditions")) else ""

        result = _perplexity_call(ticker, date, drug, summary, ct_phase, ct_conds, api_key, model)

        if "_error" in result:
            counts["errors"] += 1
            if (i + 1) % 10 == 0 or counts["errors"] <= 3:
                print(f"    [{i+1}/{len(eligible)}] {ticker} {date} ERR: {result['_error']}")
            time.sleep(2)
            continue

        field_map = {
            "indication":              ("indication",              lambda x: str(x)),
            "is_pivotal":              ("is_pivotal",              lambda x: str(x).capitalize()),
            "pivotal_evidence":        ("pivotal_evidence",        lambda x: str(x)),
            "primary_endpoint_met":    ("primary_endpoint_met",    lambda x: str(x)),
            "primary_endpoint_result": ("primary_endpoint_result", lambda x: str(x)),
        }
        for json_key, (col, cast) in field_map.items():
            val = result.get(json_key)
            if val is not None and str(val).lower() not in ("null", "none", "") and pd.isna(df.at[idx, col]):
                df.at[idx, col] = cast(val)
                counts[col] += 1

        if (i + 1) % 25 == 0:
            print(
                f"    [{i+1}/{len(eligible)}] {ticker} {date} → "
                f"ep={result.get('primary_endpoint_met','?')} "
                f"pivotal={result.get('is_pivotal','?')}"
            )
        time.sleep(1.5)

    return df, counts


# ---------------------------------------------------------------------------
# Re-derive all feature columns from updated source data
# ---------------------------------------------------------------------------

def _parse_partial_date(val):
    if not val or pd.isna(val):
        return None
    val = str(val).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return pd.to_datetime(val, format=fmt)
        except (ValueError, TypeError):
            pass
    return None


def _proximity_bucket(d):
    if pd.isna(d):
        return "unknown"
    for lo, hi, label in PROXIMITY_BUCKETS:
        if lo <= d < hi:
            return label
    return "unknown"


def _endpoint_outcome_score(row) -> float:
    """Endpoint outcome only — no is_pivotal dependency. Range -1 to +1."""
    ep = str(row.get("primary_endpoint_met", "")).strip().lower()
    return {"yes": 1.0, "no": -1.0, "unclear": 0.0}.get(ep, 0.0)


def rederive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-apply all feature derivations over the full dataset so that
    newly filled source columns propagate into their downstream features.
    We recompute each feature fully rather than patching only nulls —
    this ensures consistency if any source value changed.
    """
    enroll    = pd.to_numeric(df["ct_enrollment"], errors="coerce")
    phase_num = df["ct_phase"].map(PHASE_MAP)

    # ── Clinical ─────────────────────────────────────────────────────────
    df["feat_phase_num"]         = phase_num
    df["feat_late_stage_flag"]   = (phase_num >= 2.5).astype(float)
    df.loc[phase_num.isna(), "feat_late_stage_flag"] = np.nan

    df["feat_enrollment_log"]    = np.log1p(enroll)

    df["feat_randomized_flag"]   = np.where(
        df["ct_allocation"].isna(), np.nan,
        (df["ct_allocation"] == "RANDOMIZED").astype(float),
    )

    df["feat_active_not_recruiting_flag"] = np.where(
        df["ct_status"].isna(), np.nan,
        (df["ct_status"] == "ACTIVE_NOT_RECRUITING").astype(float),
    )

    # ── mesh_level1_encoded (raw column) — fill from mesh_level1 map ────
    df["mesh_level1_encoded"]    = df["mesh_level1"].map(MESH_ENCODE_MAP)
    df["feat_mesh_level1_encoded"] = df["mesh_level1"].map(MESH_ENCODE_MAP)

    # ── Timing ───────────────────────────────────────────────────────────
    event_dates      = pd.to_datetime(df["event_date"], errors="coerce")
    completion_dates = pd.to_datetime(
        df["ct_primary_completion"].apply(_parse_partial_date), errors="coerce"
    )
    days_delta = (completion_dates - event_dates).dt.days
    df["feat_days_to_primary_completion"] = days_delta
    df["feat_event_proximity_bucket"]     = days_delta.apply(_proximity_bucket)

    # ── Status flags ─────────────────────────────────────────────────────
    df["feat_withdrawn_flag"]  = (df["ct_status"] == "WITHDRAWN").astype(int)
    df["feat_terminated_flag"] = (df["ct_status"] == "TERMINATED").astype(int)

    # ── Design quality score ─────────────────────────────────────────────
    # +2 RANDOMIZED | +2 enroll>300 | +1 enroll 101–300
    # +1 Phase3     | +0.5 Phase2/3 | -1 NON_RANDOMIZED
    design = pd.Series(0.0, index=df.index)
    design += (df["ct_allocation"] == "RANDOMIZED").astype(float)       * 2.0
    design -= (df["ct_allocation"] == "NON_RANDOMIZED").astype(float)   * 1.0
    design += (enroll > 300).fillna(False).astype(float)                * 2.0
    design += ((enroll > 100) & (enroll <= 300)).fillna(False).astype(float) * 1.0
    design += (phase_num >= 3.0).fillna(False).astype(float)            * 1.0
    design += (phase_num == 2.5).fillna(False).astype(float)            * 0.5
    all_missing = df["ct_allocation"].isna() & enroll.isna() & phase_num.isna()
    design[all_missing] = np.nan
    df["feat_design_quality_score"] = design

    # ── Trial quality score ──────────────────────────────────────────────
    # = design + 1.0×blinded + 0.5×breakthrough − 2×withdrawn − 2×terminated
    title_lower   = df["ct_official_title"].fillna("").str.lower()
    blinded_title = title_lower.str.contains(
        r"placebo[- ]control|double[- ]blind|triple[- ]blind|controlled study|controlled trial",
        regex=True, na=False,
    )
    breakthrough  = df.get("feat_breakthrough_flag", pd.Series(0, index=df.index)).astype(float)
    quality = (
        design.fillna(0.0)
        + blinded_title.astype(float)
        + breakthrough * 0.5
        - df["feat_withdrawn_flag"].astype(float)  * 2.0
        - df["feat_terminated_flag"].astype(float) * 2.0
    )
    quality[all_missing & df["ct_status"].isna()] = np.nan
    df["feat_trial_quality_score"] = quality

    df["feat_controlled_flag"] = (
        (df["feat_randomized_flag"] == 1) | blinded_title
    ).astype(int)

    # ── Company-level (re-aggregate from scratch) ────────────────────────
    late_mask = (
        (df["feat_late_stage_flag"] == 1) |
        (phase_num >= 3.0).fillna(False)
    )
    df["_late_row"] = late_mask.astype(int)
    df["feat_n_late_stage_trials_for_company"] = (
        df.groupby("ticker")["_late_row"].transform("sum").astype(int)
    )
    df.drop(columns=["_late_row"], inplace=True)

    late_frac = (
        df["feat_n_late_stage_trials_for_company"]
        / df["feat_n_trials_for_company"].replace(0, np.nan)
    ).fillna(0.0)
    df["feat_pipeline_concentration_simple"] = (
        df["feat_lead_asset_dependency_score"] * late_frac
    ).round(4)

    # ── Endpoint score (benefits from newly filled endpoint/pivotal cols) ─
    df["feat_endpoint_outcome_score"] = df.apply(_endpoint_outcome_score, axis=1)

    # ── Reaction priors — recompute on full dataset ──────────────────────
    abs_move = df["stock_movement_atr_normalized"].abs()
    df["feat_prior_mean_abs_move_atr_by_phase"] = (
        abs_move.groupby(df["feat_phase_num"]).transform("mean")
    )
    df["feat_prior_mean_abs_move_atr_by_therapeutic_superclass"] = (
        abs_move.groupby(df["mesh_level1"]).transform("mean")
    )
    cross_key = df["feat_phase_num"].astype(str) + "__" + df["mesh_level1"].astype(str)
    df["feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass"] = (
        abs_move.groupby(cross_key).transform("mean")
    )
    df["feat_prior_mean_abs_move_atr_by_market_cap_bucket"] = (
        abs_move.groupby(df["feat_market_cap_bucket"]).transform("mean")
    )

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Final completeness pass — fill missing values in the ML feature dataset"
    )
    parser.add_argument("--input",             default=DATA_FILE)
    parser.add_argument("--skip-ctgov",        action="store_true")
    parser.add_argument("--skip-finance",      action="store_true")
    parser.add_argument("--skip-perplexity",   action="store_true")
    parser.add_argument("--perplexity-limit",  type=int, default=300,
                        help="Max rows to send to Perplexity (default 300)")
    args = parser.parse_args()

    _load_dotenv()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded: {args.input}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # Snapshot before
    before = {
        c: int(df[c].isna().sum())
        for c in PRIORITY_COLS if c in df.columns
    }
    print("\nMissing BEFORE:")
    for c, n in before.items():
        print(f"  {c:<35}  {n:>4} / {len(df)}  ({n/len(df)*100:.1f}%)")

    ctgov_counts      = {}
    finance_counts    = {}
    perplexity_counts = {}

    # ── Stage 1: CT.gov ───────────────────────────────────────────────────
    if not args.skip_ctgov:
        print("\n── Stage 1: ClinicalTrials.gov ──")
        df, ctgov_counts = fill_from_ctgov(df)
        filled_by_ct = {k: v for k, v in ctgov_counts.items() if v}
        print(f"  Filled: {filled_by_ct}")
        df.to_csv(args.input, index=False)
        print(f"  Saved checkpoint after CT.gov.")

    # ── Stage 2: Finance ──────────────────────────────────────────────────
    if not args.skip_finance:
        print("\n── Stage 2: Finance (yfinance) ──")
        df, finance_counts = fill_prices(df)
        print(f"  Filled: {finance_counts}")
        df.to_csv(args.input, index=False)
        print(f"  Saved checkpoint after finance.")

    # ── Stage 3: Perplexity ───────────────────────────────────────────────
    if not args.skip_perplexity:
        print("\n── Stage 3: Perplexity ──")
        df, perplexity_counts = fill_perplexity(df, limit=args.perplexity_limit)
        filled_by_pl = {k: v for k, v in perplexity_counts.items() if v}
        print(f"  Filled: {filled_by_pl}")
        df.to_csv(args.input, index=False)
        print(f"  Saved checkpoint after Perplexity.")

    # ── Re-derive features ────────────────────────────────────────────────
    print("\n── Re-deriving feature columns ──")
    df = rederive_features(df)
    print("  Done.")

    # ── Save in-place ─────────────────────────────────────────────────────
    df.to_csv(args.input, index=False)
    print(f"\nSaved in-place: {args.input}")

    # ── Summary ──────────────────────────────────────────────────────────
    after = {
        c: int(df[c].isna().sum())
        for c in PRIORITY_COLS if c in df.columns
    }

    total_filled = sum(before[c] - after.get(c, before[c]) for c in before)

    print(f"\n{'═'*72}")
    print("COMPLETENESS PASS SUMMARY")
    print(f"{'═'*72}")
    print(f"File:          {args.input}  (updated in-place)")
    print(f"Rows × Cols:   {df.shape[0]} × {df.shape[1]}")
    print(f"\n{'Column':<35}  {'Before':>7}  {'After':>7}  {'Filled':>7}")
    print(f"{'─'*35}  {'─'*7}  {'─'*7}  {'─'*7}")
    for c in PRIORITY_COLS:
        if c not in before:
            continue
        b, a = before[c], after.get(c, before[c])
        tag = f"  ← {b-a}" if b != a else ""
        print(f"  {c:<33}  {b:>7}  {a:>7}  {b-a:>7}{tag}")

    print(f"\nTotal cells filled: {total_filled}")
    print("\nFilled by source:")
    if ctgov_counts:
        print(f"  CT.gov:     {ctgov_counts}")
    if finance_counts:
        print(f"  Finance:    {finance_counts}")
    if perplexity_counts:
        print(f"  Perplexity: {perplexity_counts}")

    remaining = [(c, a) for c, a in after.items() if a > 0]
    if remaining:
        print("\nRemaining missing (priority columns):")
        for c, n in remaining:
            pct = n / len(df) * 100
            print(f"  {c:<35}  {n:>4}  ({pct:.1f}%)")
    else:
        print("\nAll priority columns fully filled.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()

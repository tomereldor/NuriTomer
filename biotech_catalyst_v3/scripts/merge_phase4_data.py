"""Phase 4 (A3): Merge confirmed 2018-2022 catalysts into master CSV.

Sources:
  data/confirmed_catalysts_2018_2022.csv  — CT.gov cross-matched events
  data/perplexity_confirmed_2018_2022.csv — Perplexity-confirmed clinical events

New data_tier values (used to bypass MIN_EVENT_YEAR filter in training):
  "phase4_ctgov"     — cross-matched large moves with CT.gov completion (positives)
  "phase4_ctgov_neg" — CT.gov completion small-move events (negatives)
  "phase4_perp"      — Perplexity-confirmed clinical catalysts (positives)

After running this script, re-run the feature engineering + training pipeline:
    python -m scripts.add_high_signal_features
    python -m scripts.add_pre_event_timing_features
    python -m scripts.build_pre_event_train_v2
    python -m scripts.train_pre_event_v3

Usage:
    python -m scripts.merge_phase4_data
    python -m scripts.merge_phase4_data --max-ctgov-neg 600 --dry-run
"""

import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MASTER_CSV     = "enriched_all_clinical_clean_v3.csv"
ARCHIVE_DIR    = "archive"
CATALYSTS_FILE = "data/confirmed_catalysts_2018_2022.csv"
PERPLEXITY_FILE = "data/perplexity_confirmed_2018_2022.csv"


# ---------------------------------------------------------------------------
# Phase normalization
# ---------------------------------------------------------------------------

_PHASE_MAP = {
    # CT.gov API ENUM → master CSV format
    "PHASE1":           "Phase 1",
    "PHASE2":           "Phase 2",
    "PHASE3":           "Phase 3",
    "PHASE4":           "Phase 4",
    "PHASE1/PHASE2":    "Phase 1/2",
    "PHASE2/PHASE3":    "Phase 2/3",
    "EARLY_PHASE1":     "Phase 1 (Early)",
    "NA":               "",
    # Perplexity text variants
    "phase 1":          "Phase 1",
    "phase 2":          "Phase 2",
    "phase 3":          "Phase 3",
    "phase 4":          "Phase 4",
    "phase 1/2":        "Phase 1/2",
    "phase 2/3":        "Phase 2/3",
    "phase1":           "Phase 1",
    "phase2":           "Phase 2",
    "phase3":           "Phase 3",
    "nda":              "Phase 3",   # NDA implies Phase 3 complete
    "bla":              "Phase 3",
    "nda/bla":          "Phase 3",
    "phase 2b":         "Phase 2",
    "phase 2b/3":       "Phase 2/3",
    "phase2b":          "Phase 2",
    "phase2b/3":        "Phase 2/3",
    "ind":              "",
    "approval":         "Phase 3",
    "clinical trial":   "",
    "":                 "",
}


def normalize_phase(raw) -> str:
    if not raw or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    raw_str = str(raw).strip()
    # Direct lookup (case-insensitive)
    lower = raw_str.lower()
    if lower in _PHASE_MAP:
        return _PHASE_MAP[lower]
    if raw_str in _PHASE_MAP:
        return _PHASE_MAP[raw_str]
    # Partial normalisation: already looks like "Phase X"
    if raw_str.startswith("Phase "):
        return raw_str
    return raw_str  # pass through unknown


# ---------------------------------------------------------------------------
# Move class helpers
# ---------------------------------------------------------------------------

_ABS_BINS   = [0, 5, 10, 20, float("inf")]
_ABS_LABELS = ["Low", "Medium", "High", "Extreme"]
_NORM_BINS  = [0, 1.5, 3.0, 5.0, 8.0, float("inf")]
_NORM_LABELS = ["Noise", "Low", "Medium", "High", "Extreme"]


def add_move_classes(df: pd.DataFrame) -> pd.DataFrame:
    abs_m = df["move_pct"].abs()
    df["move_class_abs"] = pd.cut(
        abs_m, bins=_ABS_BINS, labels=_ABS_LABELS, right=False
    ).astype(str)
    df.loc[df["move_pct"].isna(), "move_class_abs"] = np.nan

    atr = df["atr_pct"].replace(0, np.nan)
    df["stock_movement_atr_normalized"] = (abs_m / atr).round(3)

    df["move_class_norm"] = pd.cut(
        df["stock_movement_atr_normalized"],
        bins=_NORM_BINS, labels=_NORM_LABELS, right=False
    ).astype(str)
    df.loc[df["stock_movement_atr_normalized"].isna(), "move_class_norm"] = np.nan

    df["move_class_combo"] = df["move_class_abs"].fillna("?") + "/" + df["move_class_norm"].fillna("?")
    return df


# ---------------------------------------------------------------------------
# Price imputation
# ---------------------------------------------------------------------------

def impute_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate price_before/price_after from close and move_pct."""
    has_close = df.get("close", pd.Series(dtype=float)).notna()
    has_move  = df["move_pct"].notna()

    mask = has_close & has_move
    close = df.get("close", pd.Series(dtype=float))
    move  = df["move_pct"] / 100.0  # fractional

    # close = price_after (closing price on event day)
    # close = price_before * (1 + move_pct)
    df.loc[mask, "price_after"]  = close[mask].round(2)
    df.loc[mask, "price_before"] = (close[mask] / (1 + move[mask])).round(2)
    df.loc[mask, "price_at_event"] = df.loc[mask, "price_after"]
    return df


# ---------------------------------------------------------------------------
# Build rows from CT.gov cross-match output
# ---------------------------------------------------------------------------

def build_ctgov_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert confirmed_catalysts CSV rows to master CSV format."""
    rows = []
    for _, r in df.iterrows():
        is_pos = r.get("target_large_move", 0) == 1
        tier   = "phase4_ctgov" if is_pos else "phase4_ctgov_neg"

        rows.append({
            "ticker":          r["ticker"],
            "event_date":      r["event_date"],
            "v_actual_date":   r["event_date"],
            "event_type":      r.get("event_type", "Gainer" if (r.get("move_pct", 0) or 0) > 0 else "Loser"),
            "move_pct":        r.get("move_pct"),
            "atr_pct":         r.get("atr_pct"),      # NaN for CT.gov negatives
            "close":           r.get("close"),
            "market_cap_m":    r.get("market_cap_m"),
            "drug_name":       str(r.get("drug_name", "") or "").split(";")[0].strip(),
            "nct_id":          r.get("nct_id", ""),
            "indication":      r.get("indication", "") or "",
            "ct_conditions":   r.get("ct_conditions", "") or r.get("indication", "") or "",
            "ct_phase":        normalize_phase(r.get("ct_phase", "")),
            "ct_enrollment":   r.get("ct_enrollment"),
            "ct_status":       "COMPLETED",
            "ct_official_title": r.get("ct_official_title", "") or "",
            "ct_allocation":   r.get("ct_allocation", "") or "",
            "ct_primary_completion": r.get("ct_primary_completion", "") or "",
            "ct_sponsor":      r.get("ct_sponsor", "") or "",
            "catalyst_type":   "Clinical Data",
            "v_is_material":   True,
            "v_is_verified":   True,
            "v_confidence":    "high",
            "data_tier":       tier,
        })

    result = pd.DataFrame(rows)
    result = add_move_classes(result)
    result = impute_prices(result)
    return result


# ---------------------------------------------------------------------------
# Build rows from Perplexity output
# ---------------------------------------------------------------------------

def build_perplexity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert perplexity_confirmed CSV rows to master CSV format."""
    rows = []
    for _, r in df.iterrows():
        catalyst_type = (
            "Clinical Data" if r.get("catalyst_type") == "clinical_trial"
            else "Regulatory"
        )
        rows.append({
            "ticker":        r["ticker"],
            "event_date":    r["event_date"],
            "v_actual_date": r["event_date"],
            "event_type":    "Gainer" if (r.get("move_pct", 0) or 0) > 0 else "Loser",
            "move_pct":      r.get("move_pct"),
            "atr_pct":       r.get("atr_pct"),
            "close":         None,
            "market_cap_m":  r.get("market_cap_m"),
            "drug_name":     str(r.get("drug", "") or "").strip(),
            "nct_id":        "",
            "indication":    str(r.get("indication", "") or "").strip(),
            "ct_conditions": str(r.get("indication", "") or "").strip(),
            "ct_phase":      normalize_phase(r.get("phase", "")),
            "ct_enrollment": np.nan,
            "ct_status":     "COMPLETED",
            "ct_official_title": "",
            "ct_allocation": "",
            "ct_primary_completion": "",
            "ct_sponsor":    "",
            "catalyst_type": catalyst_type,
            "v_is_material": True,
            "v_is_verified": True,
            "v_confidence":  r.get("confidence", "medium"),
            "data_tier":     "phase4_perp",
        })

    result = pd.DataFrame(rows)
    result = add_move_classes(result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def merge_phase4(
    master_csv: str = MASTER_CSV,
    catalysts_file: str = CATALYSTS_FILE,
    perplexity_file: str = PERPLEXITY_FILE,
    max_ctgov_neg: int = 600,
    dry_run: bool = False,
):
    # ------------------------------------------------------------------
    # Load sources
    # ------------------------------------------------------------------
    print("Loading master CSV...", flush=True)
    master = pd.read_csv(master_csv)
    print(f"  Master: {master.shape[0]} rows × {master.shape[1]} cols")

    print("Loading confirmed_catalysts...", flush=True)
    cats = pd.read_csv(catalysts_file)
    print(f"  Catalysts: {len(cats)} rows "
          f"({cats.get('target_large_move', pd.Series()).sum()} positives)")

    print("Loading perplexity_confirmed...", flush=True)
    perp = pd.read_csv(perplexity_file)
    print(f"  Perplexity: {len(perp)} confirmed clinical events")

    # ------------------------------------------------------------------
    # Build new rows
    # ------------------------------------------------------------------

    # CT.gov cross-match POSITIVES only (large moves confirmed as clinical)
    ctgov_pos = cats[cats.get("target_large_move", pd.Series(0, index=cats.index)) == 1].copy()
    print(f"\nCT.gov cross-match positives: {len(ctgov_pos)}")
    df_ctgov_pos = build_ctgov_rows(ctgov_pos)

    # CT.gov NEGATIVES (small-move clinical completions, sample)
    ctgov_neg = cats[cats.get("target_large_move", pd.Series(0, index=cats.index)) == 0].copy()
    if len(ctgov_neg) > max_ctgov_neg:
        ctgov_neg = ctgov_neg.sample(n=max_ctgov_neg, random_state=42)
    print(f"CT.gov negatives (sampled): {len(ctgov_neg)}")
    df_ctgov_neg = build_ctgov_rows(ctgov_neg)

    # Perplexity confirmed (all clinical catalysts)
    print(f"Perplexity positives: {len(perp)}")
    df_perp = build_perplexity_rows(perp)

    # ------------------------------------------------------------------
    # Dedup against master CSV
    # ------------------------------------------------------------------
    existing_keys = set(zip(master["ticker"].str.upper(), master["event_date"].astype(str)))

    def dedup(df: pd.DataFrame, label: str) -> pd.DataFrame:
        df["ticker"] = df["ticker"].str.upper()
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")
        before = len(df)
        df = df[~df.apply(lambda r: (r["ticker"], r["event_date"]) in existing_keys, axis=1)]
        dropped = before - len(df)
        if dropped:
            print(f"  {label}: dropped {dropped} duplicates already in master")
        return df

    df_ctgov_pos = dedup(df_ctgov_pos, "CT.gov positives")
    df_ctgov_neg = dedup(df_ctgov_neg, "CT.gov negatives")
    df_perp      = dedup(df_perp,      "Perplexity")

    new_rows = pd.concat([df_ctgov_pos, df_ctgov_neg, df_perp], ignore_index=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pos = (
        (new_rows["stock_movement_atr_normalized"] >= 3.0) &
        (new_rows["move_pct"].abs() >= 10.0)
    ).sum()
    n_neg = len(new_rows) - n_pos

    print(f"\n{'='*60}")
    print(f"NEW ROWS SUMMARY")
    print(f"{'='*60}")
    print(f"CT.gov cross-match positives: {len(df_ctgov_pos)}")
    print(f"CT.gov negatives:             {len(df_ctgov_neg)}")
    print(f"Perplexity positives:         {len(df_perp)}")
    print(f"  Total new rows:             {len(new_rows)}")
    print(f"  Of which target=1:          {n_pos}")
    print(f"  Of which target=0:          {n_neg}")
    print(f"\nExisting master rows:         {len(master)}")
    print(f"New master total:             {len(master) + len(new_rows)}")

    print(f"\nData tier distribution:")
    print(new_rows["data_tier"].value_counts().to_string())

    print(f"\nCT.gov phase distribution (new rows):")
    phase_counts = new_rows[new_rows["ct_phase"].notna() & (new_rows["ct_phase"] != "")]["ct_phase"].value_counts()
    print(phase_counts.to_string())

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return new_rows

    # ------------------------------------------------------------------
    # Archive master CSV and write updated version
    # ------------------------------------------------------------------
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    archive_path = os.path.join(ARCHIVE_DIR, f"enriched_all_clinical_clean_v3_{today}_pre_phase4.csv")
    shutil.copy2(master_csv, archive_path)
    print(f"\nMaster CSV archived to: {archive_path}")

    # Align new_rows columns to master (fill missing with NaN)
    for col in master.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan

    # Drop extra columns not in master
    extra = [c for c in new_rows.columns if c not in master.columns]
    if extra:
        new_rows = new_rows.drop(columns=extra)

    # Ensure column order matches master
    new_rows = new_rows[master.columns]

    # Combine and save
    combined = pd.concat([master, new_rows], ignore_index=True)
    combined.to_csv(master_csv, index=False)

    print(f"Updated master CSV: {len(combined)} rows × {len(combined.columns)} cols")
    print(f"  data_tier distribution:")
    print(combined["data_tier"].value_counts(dropna=False).to_string())

    return new_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Phase 4 data into master CSV")
    parser.add_argument("--master",          default=MASTER_CSV)
    parser.add_argument("--catalysts",       default=CATALYSTS_FILE)
    parser.add_argument("--perplexity",      default=PERPLEXITY_FILE)
    parser.add_argument("--max-ctgov-neg",   type=int, default=600)
    parser.add_argument("--dry-run",         action="store_true")
    args = parser.parse_args()

    merge_phase4(
        master_csv=args.master,
        catalysts_file=args.catalysts,
        perplexity_file=args.perplexity,
        max_ctgov_neg=args.max_ctgov_neg,
        dry_run=args.dry_run,
    )

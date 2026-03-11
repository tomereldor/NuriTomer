"""
build_ml_ready_features.py
==========================
Transform ml_dataset_mesh_final.csv into ML-ready features.

Steps:
  1. Target variables  (target_*)
  2. Clinical features (feat_phase_num, feat_late_stage_flag,
                        feat_endpoint_positive_score,
                        feat_mesh_level1_encoded)
  3. Financial/market  (feat_log_market_cap, feat_short_squeeze_flag,
                        feat_ownership_low_flag, feat_volatility)
  4. Event timing      (feat_days_to_primary_completion,
                        feat_event_proximity_bucket,
                        feat_active_not_recruiting_flag)
  5. Trial design quality (feat_design_quality_score)
  6. Save versioned output:
       ml_dataset_features_YYYYMMDD_vN.csv
       ml_feature_dict_YYYYMMDD_vN.csv

Usage:
    python3 build_ml_ready_features.py
    python3 build_ml_ready_features.py --input ml_dataset_mesh_final.csv --date 20260310
"""

import argparse
import glob
import math
import os
import re
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DEFAULT = "ml_dataset_mesh_final.csv"
TODAY = date.today().strftime("%Y%m%d")

PHASE_MAP = {
    "Phase 1 (Early)": 0.5,
    "Phase 1":         1.0,
    "Phase 1/2":       1.5,
    "Phase 2":         2.0,
    "Phase 2/3":       2.5,
    "Phase 3":         3.0,
    "Phase 4":         4.0,
}

MOVE_BUCKET_MAP = {
    "Noise":   0,
    "Low":     1,
    "Medium":  2,
    "High":    3,
    "Extreme": 4,
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

# Proximity bucket thresholds (days to primary completion, positive = future)
PROXIMITY_BUCKETS = [
    (365,   float("inf"),  "future_far"),     # > 1 year away
    (0,     365,           "future_near"),    # 0 – 12 months away
    (-180,  0,             "just_completed"), # 0 – 6 months past
    (float("-inf"), -180,  "past"),           # > 6 months past
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_partial_date(val: str):
    """Parse YYYY-MM-DD, YYYY-MM, or YYYY → first day of period."""
    if not val or pd.isna(val):
        return None
    val = str(val).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return pd.to_datetime(val, format=fmt)
        except (ValueError, TypeError):
            pass
    return None


def _next_version(date_str: str, base_dir: str, prefix: str) -> int:
    """Return the next version integer for files matching prefix_DATE_vN.csv."""
    pattern = os.path.join(base_dir, f"{prefix}_{date_str}_v*.csv")
    existing = glob.glob(pattern)
    if not existing:
        return 1
    nums = []
    for f in existing:
        m = re.search(r"_v(\d+)\.csv$", f)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def _endpoint_outcome_score(row) -> float:
    """
    Endpoint outcome only — does NOT use is_pivotal.
      primary_endpoint_met : Yes=+1, No=-1, Unclear/missing=0
    Range: -1 to +1
    Pivotal/importance signal is captured separately by feat_regulatory_stage_score.
    """
    ep = str(row.get("primary_endpoint_met", "")).strip().lower()
    return {"yes": 1.0, "no": -1.0, "unclear": 0.0}.get(ep, 0.0)


def _proximity_bucket(days: float) -> str:
    if pd.isna(days):
        return "unknown"
    for lo, hi, label in PROXIMITY_BUCKETS:
        if lo <= days < hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Target variables."""
    df["target_abs_move_atr"] = df["stock_movement_atr_normalized"]
    df["target_move_bucket"] = df["move_class_norm"].map(MOVE_BUCKET_MAP)
    df["target_large_move"] = df["move_class_norm"].isin(["High", "Extreme"]).astype(int)
    return df


def build_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Clinical fields."""
    # Phase numeric
    df["feat_phase_num"] = df["ct_phase"].map(PHASE_MAP)

    # Late-stage flag: Phase 2/3 (2.5), Phase 3 (3.0), Phase 4 (4.0)
    df["feat_late_stage_flag"] = (df["feat_phase_num"] >= 2.5).astype(float)
    df.loc[df["feat_phase_num"].isna(), "feat_late_stage_flag"] = float("nan")

    # Endpoint outcome score — outcome only, no is_pivotal dependency
    df["feat_endpoint_outcome_score"] = df.apply(_endpoint_outcome_score, axis=1)

    # MeSH Level-1 integer encoding — fully derived from mesh_level1 (831/831)
    df["feat_mesh_level1_encoded"] = df["mesh_level1"].map(MESH_ENCODE_MAP)

    return df


def build_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Financial / market features."""
    df["feat_log_market_cap"] = np.log10(df["market_cap_m"].clip(lower=1e-3))

    df["feat_short_squeeze_flag"] = (df["short_percent"] >= 20.0).astype(float)
    df.loc[df["short_percent"].isna(), "feat_short_squeeze_flag"] = float("nan")

    df["feat_ownership_low_flag"] = (df["institutional_ownership"] < 30.0).astype(float)
    df.loc[df["institutional_ownership"].isna(), "feat_ownership_low_flag"] = float("nan")

    df["feat_volatility"] = df["atr_pct"]

    return df


def build_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Event timing."""
    event_dates = pd.to_datetime(df["event_date"], errors="coerce")

    completion_dates = df["ct_primary_completion"].apply(_parse_partial_date)
    completion_dates = pd.to_datetime(completion_dates, errors="coerce")

    days_delta = (completion_dates - event_dates).dt.days
    df["feat_days_to_primary_completion"] = days_delta

    df["feat_event_proximity_bucket"] = days_delta.apply(_proximity_bucket)

    df["feat_active_not_recruiting_flag"] = (
        df["ct_status"] == "ACTIVE_NOT_RECRUITING"
    ).astype(float)
    df.loc[df["ct_status"].isna(), "feat_active_not_recruiting_flag"] = float("nan")

    return df


def build_design_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Trial design quality score (0–6).

    Points:
      +2  ct_allocation == RANDOMIZED
      +2  ct_enrollment > 300
      +1  ct_enrollment 101–300
      +1  feat_phase_num >= 3.0
      +0.5 feat_phase_num == 2.5 (Phase 2/3)
      -1  ct_allocation == NON_RANDOMIZED
    """
    score = pd.Series(0.0, index=df.index)

    rand_mask   = df["ct_allocation"] == "RANDOMIZED"
    nonrand_mask = df["ct_allocation"] == "NON_RANDOMIZED"
    score += rand_mask.astype(float) * 2.0
    score -= nonrand_mask.astype(float) * 1.0

    enroll = pd.to_numeric(df["ct_enrollment"], errors="coerce")
    score += (enroll > 300).fillna(False).astype(float) * 2.0
    score += ((enroll > 100) & (enroll <= 300)).fillna(False).astype(float) * 1.0

    phase_num = df.get("feat_phase_num", pd.Series(dtype=float))
    score += (phase_num >= 3.0).fillna(False).astype(float) * 1.0
    score += (phase_num == 2.5).fillna(False).astype(float) * 0.5

    # NaN out if all key inputs are missing
    all_missing = df["ct_allocation"].isna() & enroll.isna() & phase_num.isna()
    score[all_missing] = float("nan")

    df["feat_design_quality_score"] = score
    return df


# ---------------------------------------------------------------------------
# Feature dictionary builder
# ---------------------------------------------------------------------------

FEATURE_META = [
    # (name, type, description, source_columns)
    ("target_abs_move_atr",              "target",  "ATR-normalised absolute stock move (primary regression target)",        "stock_movement_atr_normalized"),
    ("target_move_bucket",               "target",  "Ordinal move class: 0=Noise 1=Low 2=Med 3=High 4=Extreme",             "move_class_norm"),
    ("target_large_move",                "target",  "Binary: 1 if move_class_norm in {High, Extreme}",                      "move_class_norm"),
    ("feat_phase_num",                   "feat",    "Trial phase as float: 0.5=EarlyI 1=I 1.5=I/II 2=II 2.5=II/III 3=III 4=IV", "ct_phase"),
    ("feat_late_stage_flag",             "feat",    "1 if phase_num >= 2.5 (Phase 2/3, 3, or 4)",                           "ct_phase"),
    ("feat_endpoint_outcome_score",      "feat",    "Endpoint outcome only: Yes=+1 No=-1 Unclear/missing=0. Range -1 to +1. Pivotal importance captured separately by feat_regulatory_stage_score.", "primary_endpoint_met"),
    ("feat_mesh_level1_encoded",         "feat",    "MeSH Level-1 integer: 1=Neoplasms … 10=Musculoskeletal 11=Other",      "mesh_level1"),
    ("feat_log_market_cap",              "feat",    "log10(market_cap_m) — log-scale market cap in USD millions",            "market_cap_m"),
    ("feat_short_squeeze_flag",          "feat",    "1 if short_percent >= 20% (elevated short interest)",                  "short_percent"),
    ("feat_ownership_low_flag",          "feat",    "1 if institutional_ownership < 30%",                                   "institutional_ownership"),
    ("feat_volatility",                  "feat",    "atr_pct — 20-day ATR as % of price (Wilder RMA)",                      "atr_pct"),
    ("feat_days_to_primary_completion",  "feat",    "Days from event_date to ct_primary_completion (positive = future)",     "event_date, ct_primary_completion"),
    ("feat_event_proximity_bucket",      "feat",    "Categorical: future_far / future_near / just_completed / past / unknown", "feat_days_to_primary_completion"),
    ("feat_active_not_recruiting_flag",  "feat",    "1 if ct_status == ACTIVE_NOT_RECRUITING",                              "ct_status"),
    ("feat_design_quality_score",        "feat",    "Trial quality: +2 randomized, +2 enroll>300, +1 enroll>100, +1 Ph3, +0.5 Ph2/3, -1 non-rand", "ct_allocation, ct_enrollment, feat_phase_num"),
]


def build_feature_dict(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, ftype, desc, src in FEATURE_META:
        if name not in df.columns:
            continue
        col = df[name]
        rows.append({
            "feature_name":   name,
            "type":           ftype,
            "description":    desc,
            "source_columns": src,
            "dtype":          str(col.dtype),
            "n_valid":        int(col.notna().sum()),
            "n_null":         int(col.isna().sum()),
            "pct_valid":      round(col.notna().mean() * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ML-ready feature set from mesh-final dataset")
    parser.add_argument("--input",  default=INPUT_DEFAULT)
    parser.add_argument("--date",   default=TODAY, help="Date tag for output files (YYYYMMDD)")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    # ---- Load ----
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {args.input}: {df.shape[0]} rows × {df.shape[1]} cols")

    # ---- Build features ----
    df = build_targets(df)
    df = build_clinical_features(df)
    df = build_financial_features(df)
    df = build_timing_features(df)
    df = build_design_quality(df)

    feat_cols   = sorted(c for c in df.columns if c.startswith("feat_"))
    target_cols = sorted(c for c in df.columns if c.startswith("target_"))
    print(f"Added {len(target_cols)} target columns: {target_cols}")
    print(f"Added {len(feat_cols)} feature columns: {feat_cols}")

    # ---- Coverage report ----
    print("\nFeature coverage:")
    for col in target_cols + feat_cols:
        n_valid = df[col].notna().sum()
        print(f"  {col:<45} {n_valid:>4}/{len(df)} ({n_valid/len(df)*100:.0f}%)")

    # ---- Auto-versioned output ----
    ver = _next_version(args.date, args.outdir, "ml_dataset_features")
    data_path = os.path.join(args.outdir, f"ml_dataset_features_{args.date}_v{ver}.csv")
    dict_path = os.path.join(args.outdir, f"ml_feature_dict_{args.date}_v{ver}.csv")

    df.to_csv(data_path, index=False)
    print(f"\nSaved feature dataset : {data_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    feat_dict = build_feature_dict(df)
    feat_dict.to_csv(dict_path, index=False)
    print(f"Saved feature dictionary: {dict_path}  ({len(feat_dict)} entries)")


if __name__ == "__main__":
    main()

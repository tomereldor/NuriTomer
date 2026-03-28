"""
add_oncology_timing_interactions.py
=====================================
Pass-8: Add oncology-aware timing interaction features.

Background:
  CT.gov primary completion date–based features (imminence flags, recency bucket)
  are valid pre-event features for non-oncology trials but are systematically
  misleading for oncology trials where the readout is event-driven (OS/PFS/DFS).
  In those trials the CT.gov primary completion date can lag the readout by
  6–24 months, so imminence flags can say "far" when a readout is genuinely
  imminent.

  Adding interaction terms lets the model learn separate weights for
  oncology vs non-oncology timing without changing the base features.

Features added (4):
  feat_oncology_x_imminent_30d       — oncology AND primary_completion_imminent_30d
  feat_oncology_x_imminent_90d       — oncology AND primary_completion_imminent_90d
  feat_oncology_x_recent_completion  — oncology AND recent_completion_flag
  feat_oncology_x_recency_imminent   — oncology AND completion_recency_bucket = imminent_0_30

Input:  latest ml_dataset_features_YYYYMMDD_vN.csv  (new pipeline naming)
Output: ml_dataset_features_YYYYMMDD_v(N+1).csv

Usage (from biotech_catalyst_v3/):
    python -m scripts.add_oncology_timing_interactions
"""

import glob
import os
import re
import shutil

import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
ML_DATA_DIR = os.path.join(BASE_DIR, "data", "ml")

NEW_FEAT_COLS = [
    "feat_oncology_x_imminent_30d",
    "feat_oncology_x_imminent_90d",
    "feat_oncology_x_recent_completion",
    "feat_oncology_x_recency_imminent",
]

NEW_FEAT_META = [
    ("feat_oncology_x_imminent_30d",
     "CT.gov Pipeline Proxy",
     "Interaction: oncology=1 AND primary_completion_imminent_30d=1",
     "feat_oncology_flag × feat_primary_completion_imminent_30d",
     "deterministic"),
    ("feat_oncology_x_imminent_90d",
     "CT.gov Pipeline Proxy",
     "Interaction: oncology=1 AND primary_completion_imminent_90d=1",
     "feat_oncology_flag × feat_primary_completion_imminent_90d",
     "deterministic"),
    ("feat_oncology_x_recent_completion",
     "CT.gov Pipeline Proxy",
     "Interaction: oncology=1 AND recent_completion_flag=1",
     "feat_oncology_flag × feat_recent_completion_flag",
     "deterministic"),
    ("feat_oncology_x_recency_imminent",
     "CT.gov Pipeline Proxy",
     "Interaction: oncology=1 AND completion_recency_bucket = imminent_0_30",
     "feat_oncology_flag × (feat_completion_recency_bucket == 'imminent_0_30')",
     "deterministic"),
]


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------

def _find_latest_features(base_dir, archive_dir):
    """Return (path, version_int, date_tag) for highest-version ml_dataset_features file in base_dir only."""
    candidates = glob.glob(os.path.join(base_dir, "ml_dataset_features_*.csv"))
    best, best_v, best_date = None, -1, None
    for f in candidates:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, best_date = v, f, m.group(1)
    return best, best_v, best_date


def _find_latest_dict(base_dir, archive_dir):
    """Return path of highest-version ml_feature_dict file in base_dir only."""
    candidates = glob.glob(os.path.join(base_dir, "ml_feature_dict_*.csv"))
    best, best_v = None, -1
    for f in candidates:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best = v, f
    return best


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def add_oncology_timing_interactions(df):
    """Add 4 deterministic oncology × timing interaction features."""
    onc = df["feat_oncology_flag"].fillna(0).astype(float)

    def _safe_binary(col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df[col].fillna(0).astype(float)

    imm30  = _safe_binary("feat_primary_completion_imminent_30d")
    imm90  = _safe_binary("feat_primary_completion_imminent_90d")
    recent = _safe_binary("feat_recent_completion_flag")

    if "feat_completion_recency_bucket" in df.columns:
        recency_imminent = (
            df["feat_completion_recency_bucket"] == "imminent_0_30"
        ).astype(float)
    else:
        recency_imminent = pd.Series(np.nan, index=df.index)

    df["feat_oncology_x_imminent_30d"]      = onc * imm30
    df["feat_oncology_x_imminent_90d"]      = onc * imm90
    df["feat_oncology_x_recent_completion"] = onc * recent
    df["feat_oncology_x_recency_imminent"]  = onc * recency_imminent

    return df


# ---------------------------------------------------------------------------
# Feature dictionary update
# ---------------------------------------------------------------------------

def update_feature_dict(df, old_dict_path, out_path):
    if old_dict_path and os.path.exists(old_dict_path):
        fdict = pd.read_csv(old_dict_path)
    else:
        raise FileNotFoundError(f"Feature dict not found: {old_dict_path}")

    # Drop any existing entries for these features (idempotent)
    fdict = fdict[~fdict["feature_name"].isin(NEW_FEAT_COLS)].copy()

    new_rows = []
    for feat_name, feat_group, desc, src_cols, src_type in NEW_FEAT_META:
        n_valid = int(df[feat_name].notna().sum())
        n_null  = int(df[feat_name].isna().sum())
        pct_v   = round(n_valid / len(df) * 100, 1)

        row = {c: "" for c in fdict.columns}
        row.update({
            "feature_name":          feat_name,
            "role":                  "feature",
            "feature_group":         feat_group,
            "data_type":             "float",
            "source_columns":        src_cols,
            "source_system":         "derived",
            "calculation_logic":     desc,
            "plain_english_definition": desc,
            "why_it_matters": (
                "Allows the model to learn separate weights for oncology vs "
                "non-oncology CT.gov timing — mitigates the oncology mismatch "
                "where readout precedes primary completion date by months"
            ),
            "validity_rule":  "0 or 1",
            "notes":          "Pass-8 oncology timing interaction",
            "stage":          "pre_event",
            "feature_type":   src_type,
            "description":    desc,
            "source_type":    src_type,
            "n_valid":        n_valid,
            "n_null":         n_null,
            "pct_valid":      round(pct_v / 100, 4),
            "valid_count":    n_valid,
            "null_count":     n_null,
            "valid_pct":      pct_v,
        })
        new_rows.append(row)

    new_df  = pd.DataFrame(new_rows, columns=fdict.columns)
    full    = pd.concat([fdict, new_df], ignore_index=True)
    full.to_csv(out_path, index=False)
    return len(full)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # --- find inputs ---
    feat_path, feat_v, date_tag = _find_latest_features(ML_DATA_DIR, ARCHIVE_DIR)
    dict_path                   = _find_latest_dict(ML_DATA_DIR, ARCHIVE_DIR)

    if feat_path is None:
        raise FileNotFoundError("No ml_dataset_features_YYYYMMDD_vN.csv found in " + ML_DATA_DIR)
    if dict_path is None:
        raise FileNotFoundError("No ml_feature_dict_YYYYMMDD_vN.csv found in " + ML_DATA_DIR)

    print(f"Input : {os.path.basename(feat_path)}  (v{feat_v})")
    df = pd.read_csv(feat_path)
    print(f"Shape : {df.shape[0]} rows × {df.shape[1]} cols")

    # --- add features ---
    df = add_oncology_timing_interactions(df)

    for col in NEW_FEAT_COLS:
        n     = df[col].notna().sum()
        pct   = n / len(df) * 100
        mean  = df[col].mean()
        print(f"  {col}: {n}/{len(df)} ({pct:.1f}%) | mean={mean:.3f}")

    # --- output paths (increment version, keep date_tag from input) ---
    new_v         = feat_v + 1
    out_feat_name = f"ml_dataset_features_{date_tag}_v{new_v}.csv"
    out_dict_name = f"ml_feature_dict_{date_tag}_v{new_v}.csv"
    os.makedirs(ML_DATA_DIR, exist_ok=True)
    out_feat_path = os.path.join(ML_DATA_DIR, out_feat_name)
    out_dict_path = os.path.join(ML_DATA_DIR, out_dict_name)

    # --- archive the previous latest ---
    if os.path.dirname(feat_path) == ML_DATA_DIR:
        shutil.move(feat_path, os.path.join(ARCHIVE_DIR, os.path.basename(feat_path)))
        print(f"Archived: {os.path.basename(feat_path)}")

    if dict_path and os.path.dirname(dict_path) == ML_DATA_DIR:
        shutil.move(dict_path, os.path.join(ARCHIVE_DIR, os.path.basename(dict_path)))
        print(f"Archived: {os.path.basename(dict_path)}")

    # --- save feature dataset ---
    df.to_csv(out_feat_path, index=False)
    print(f"\nSaved : {out_feat_name}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # --- update feature dict ---
    archived_dict = os.path.join(ARCHIVE_DIR, os.path.basename(dict_path)) if dict_path else None
    n_entries = update_feature_dict(df, archived_dict, out_dict_path)
    print(f"Dict  : {n_entries} entries → {out_dict_name}")
    print("\nDone — oncology timing interaction features added.")


if __name__ == "__main__":
    main()

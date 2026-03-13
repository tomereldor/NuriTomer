"""
add_pre_event_timing_features.py
==================================
Part 1 — Add deterministic pre-event timing and sequence features.

Reads  : latest ml_dataset_features_*_vN.csv  (checks BASE_DIR and archive/)
Outputs: ml_dataset_features_*_v(N+1).csv   (new features appended)
         ml_feature_dict_*_v(N+1).csv        (updated dictionary)
Archives: superseded files to archive/

New features added:
  1.  feat_days_to_primary_completion      — revalidated
  2.  feat_days_to_study_completion        — SKIP (column not in dataset)
  3.  feat_primary_completion_imminent_30d — binary flag
  4.  feat_primary_completion_imminent_90d — binary flag
  5.  feat_completion_recency_bucket       — categorical (6 levels)
  6.  feat_time_since_last_company_event   — days since prior event for same ticker
  7.  feat_time_since_last_asset_event     — days since prior event for same ticker+drug
  8.  feat_asset_event_sequence_num        — ordinal position within ticker+drug history
  9.  feat_company_event_sequence_num      — ordinal position within ticker history
  10. feat_recent_company_event_flag       — 1 if prior company event within 90 days
  11. feat_recent_asset_event_flag         — 1 if prior asset event within 180 days

Notes:
- All features are deterministic; no LLMs, no external fetches.
- feat_days_to_study_completion skipped: no ct_study_completion column available.
- Sequence/time-since features use v_actual_date for ordering (validated dates).

Usage (from biotech_catalyst_v3/):
    python -m scripts.add_pre_event_timing_features
"""

import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------

def _find_latest_version(base_dir, archive_dir, prefix):
    """Return (path, version_int, date_tag) for the highest-version file across
    both base_dir and archive_dir."""
    candidates = (
        glob.glob(os.path.join(base_dir, f"{prefix}_*.csv")) +
        glob.glob(os.path.join(archive_dir, f"{prefix}_*.csv"))
    )
    best, best_v, date_tag = None, 0, None
    for f in candidates:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, date_tag = v, f, m.group(1)
    return best, best_v, date_tag


def _next_version_in_basedir(base_dir, archive_dir, prefix, date_tag):
    """Return max version seen across both dirs + 1."""
    all_files = (
        glob.glob(os.path.join(base_dir, f"{prefix}_{date_tag}_v*.csv")) +
        glob.glob(os.path.join(archive_dir, f"{prefix}_{date_tag}_v*.csv"))
    )
    nums = [int(re.search(r"_v(\d+)\.csv$", f).group(1))
            for f in all_files if re.search(r"_v(\d+)\.csv$", f)]
    return max(nums) + 1 if nums else 1


def _parse_dates(series):
    return pd.to_datetime(series, errors="coerce")


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

NEW_FEATURES_META = [
    ("feat_days_to_primary_completion",
     "Days from event date to ct_primary_completion date; negative = event after completion",
     "ct_primary_completion, v_actual_date",
     "deterministic"),
    ("feat_primary_completion_imminent_30d",
     "1 if primary completion is within next 0–30 days from event date; 0 otherwise",
     "feat_days_to_primary_completion",
     "deterministic"),
    ("feat_primary_completion_imminent_90d",
     "1 if primary completion is within next 0–90 days from event date; 0 otherwise",
     "feat_days_to_primary_completion",
     "deterministic"),
    ("feat_completion_recency_bucket",
     "Categorical: imminent_0_30 / near1_90 / medium_91_180 / far_180_plus / past / unknown",
     "feat_days_to_primary_completion",
     "deterministic"),
    ("feat_time_since_last_company_event",
     "Days since prior clinical event for same ticker; NaN for first company event",
     "ticker, v_actual_date",
     "deterministic"),
    ("feat_time_since_last_asset_event",
     "Days since prior clinical event for same ticker+drug_name; NaN for first asset event",
     "ticker, drug_name, v_actual_date",
     "deterministic"),
    ("feat_asset_event_sequence_num",
     "Ordinal position of this event within ticker+drug_name history (1=first known)",
     "ticker, drug_name, v_actual_date",
     "deterministic"),
    ("feat_company_event_sequence_num",
     "Ordinal position of this event within the company (ticker) history",
     "ticker, v_actual_date",
     "deterministic"),
    ("feat_recent_company_event_flag",
     "1 if this company had a prior catalyst within the last 90 days; else 0",
     "feat_time_since_last_company_event",
     "deterministic"),
    ("feat_recent_asset_event_flag",
     "1 if this ticker+drug had a prior event within 180 days; else 0",
     "feat_time_since_last_asset_event",
     "deterministic"),
]

NEW_FEAT_NAMES = [m[0] for m in NEW_FEATURES_META]


# ---------------------------------------------------------------------------
# Feature builders  (all computed on a working copy, returned with original index)
# ---------------------------------------------------------------------------

def build_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all timing/sequence features.
    Returns a new DataFrame with same index and ALL original + new columns.
    """
    out = df.copy()

    # ── Best event date ───────────────────────────────────────────────────────
    date_col = "v_actual_date" if "v_actual_date" in out.columns else "event_date"
    evt_date = _parse_dates(out[date_col])

    # ── Primary completion date ───────────────────────────────────────────────
    prim_comp = _parse_dates(out["ct_primary_completion"]) if "ct_primary_completion" in out.columns else pd.NaT

    # ── 1. feat_days_to_primary_completion ────────────────────────────────────
    days_to_pc = (prim_comp - evt_date).dt.days
    # Preserve prior values where recomputed is null
    if "feat_days_to_primary_completion" in out.columns:
        days_to_pc = days_to_pc.combine_first(out["feat_days_to_primary_completion"])
    out["feat_days_to_primary_completion"] = days_to_pc
    dtpc = out["feat_days_to_primary_completion"]

    # ── 3. feat_primary_completion_imminent_30d ───────────────────────────────
    imminent30 = np.where(dtpc.isna(), np.nan,
                          np.where((dtpc >= 0) & (dtpc <= 30), 1.0, 0.0))
    out["feat_primary_completion_imminent_30d"] = imminent30

    # ── 4. feat_primary_completion_imminent_90d ───────────────────────────────
    imminent90 = np.where(dtpc.isna(), np.nan,
                          np.where((dtpc >= 0) & (dtpc <= 90), 1.0, 0.0))
    out["feat_primary_completion_imminent_90d"] = imminent90

    # ── 5. feat_completion_recency_bucket ─────────────────────────────────────
    def _bucket(d):
        if pd.isna(d):
            return "unknown"
        if d < 0:
            return "past"
        if d <= 30:
            return "imminent_0_30"
        if d <= 90:
            return "near1_90"
        if d <= 180:
            return "medium_91_180"
        return "far_180_plus"

    out["feat_completion_recency_bucket"] = dtpc.map(_bucket)

    # ── Sequence / time-since features require sorted order ──────────────────
    # Add a temp sort key, compute features, then sort back to original order
    out["_orig_idx"] = np.arange(len(out))
    out["_evt_date"] = evt_date
    out["_ticker"]   = out["ticker"].fillna("__unk__")
    out["_drug"]     = out["drug_name"].fillna("__unk__") if "drug_name" in out.columns else "__unk__"
    out["_comp_grp"] = out["_ticker"]
    out["_asset_grp"] = out["_ticker"] + "||" + out["_drug"]

    sorted_df = out.sort_values("_evt_date").copy()

    # ── 6. feat_time_since_last_company_event ────────────────────────────────
    sorted_df["_prev_comp"] = sorted_df.groupby("_comp_grp")["_evt_date"].shift(1)
    sorted_df["feat_time_since_last_company_event"] = (
        sorted_df["_evt_date"] - sorted_df["_prev_comp"]
    ).dt.days

    # ── 7. feat_time_since_last_asset_event ──────────────────────────────────
    sorted_df["_prev_asset"] = sorted_df.groupby("_asset_grp")["_evt_date"].shift(1)
    sorted_df["feat_time_since_last_asset_event"] = (
        sorted_df["_evt_date"] - sorted_df["_prev_asset"]
    ).dt.days

    # ── 8. feat_asset_event_sequence_num ─────────────────────────────────────
    sorted_df["feat_asset_event_sequence_num"] = (
        sorted_df.groupby("_asset_grp").cumcount() + 1
    )

    # ── 9. feat_company_event_sequence_num ───────────────────────────────────
    sorted_df["feat_company_event_sequence_num"] = (
        sorted_df.groupby("_comp_grp").cumcount() + 1
    )

    # ── 10. feat_recent_company_event_flag ───────────────────────────────────
    sorted_df["feat_recent_company_event_flag"] = np.where(
        sorted_df["feat_time_since_last_company_event"].isna(), 0.0,
        np.where(sorted_df["feat_time_since_last_company_event"] <= 90, 1.0, 0.0)
    )

    # ── 11. feat_recent_asset_event_flag ─────────────────────────────────────
    sorted_df["feat_recent_asset_event_flag"] = np.where(
        sorted_df["feat_time_since_last_asset_event"].isna(), 0.0,
        np.where(sorted_df["feat_time_since_last_asset_event"] <= 180, 1.0, 0.0)
    )

    # ── Merge sequence/time-since cols back to original order ────────────────
    seq_cols = [
        "feat_time_since_last_company_event",
        "feat_time_since_last_asset_event",
        "feat_asset_event_sequence_num",
        "feat_company_event_sequence_num",
        "feat_recent_company_event_flag",
        "feat_recent_asset_event_flag",
    ]
    # sorted_df has _orig_idx; use it to align back
    patch = sorted_df.set_index("_orig_idx")[seq_cols]
    out = out.set_index("_orig_idx")
    for col in seq_cols:
        out[col] = patch[col]
    out = out.reset_index(drop=True)

    # ── Clean temp cols ───────────────────────────────────────────────────────
    tmp_cols = ["_evt_date", "_ticker", "_drug", "_comp_grp", "_asset_grp"]
    out = out.drop(columns=[c for c in tmp_cols if c in out.columns], errors="ignore")

    return out


# ---------------------------------------------------------------------------
# Feature dictionary update
# ---------------------------------------------------------------------------

def update_feature_dict(df, old_dict_path, new_path):
    if old_dict_path and os.path.exists(old_dict_path):
        fdict = pd.read_csv(old_dict_path)
    else:
        fdict = pd.DataFrame(columns=[
            "feature_name", "stage", "feature_type", "description",
            "source_columns", "source_type", "n_valid", "n_null", "pct_valid",
        ])

    # Remove stale entries for features we're (re)creating
    fdict = fdict[~fdict["feature_name"].isin(NEW_FEAT_NAMES)]

    new_rows = []
    for feat_name, desc, src_cols, src_type in NEW_FEATURES_META:
        if feat_name not in df.columns:
            continue
        n_valid = int(df[feat_name].notna().sum())
        n_null  = int(df[feat_name].isna().sum())
        new_rows.append({
            "feature_name":   feat_name,
            "stage":          "pass5_timing",
            "feature_type":   "feat",
            "description":    desc,
            "source_columns": src_cols,
            "source_type":    src_type,
            "n_valid":        n_valid,
            "n_null":         n_null,
            "pct_valid":      round(n_valid / len(df) * 100, 1),
        })

    fdict = pd.concat([fdict, pd.DataFrame(new_rows)], ignore_index=True)
    fdict.to_csv(new_path, index=False)
    print(f"Feature dict: {len(fdict)} entries → {os.path.basename(new_path)}")
    return fdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # ── Find source (check both base and archive) ─────────────────────────────
    src_path, src_v, date_tag = _find_latest_version(BASE_DIR, ARCHIVE_DIR, "ml_dataset_features")
    if not src_path:
        print("ERROR: no ml_dataset_features_*.csv found", file=sys.stderr)
        sys.exit(1)

    # If it's in archive, copy to base dir first to work from it
    if src_path.startswith(ARCHIVE_DIR):
        dst = os.path.join(BASE_DIR, os.path.basename(src_path))
        shutil.copy(src_path, dst)
        src_path = dst
        print(f"Restored from archive: {os.path.basename(src_path)}")

    print(f"Input  : {os.path.basename(src_path)}  (v{src_v})")

    df = pd.read_csv(src_path)
    print(f"Shape  : {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Find existing feature dict ────────────────────────────────────────────
    old_dict_path, _, _ = _find_latest_version(BASE_DIR, ARCHIVE_DIR, "ml_feature_dict")

    # ── Report study completion availability ──────────────────────────────────
    study_comp_col = next((c for c in df.columns if "study_completion" in c.lower()), None)
    if study_comp_col:
        print(f"  ct_study_completion found ({study_comp_col}) — consider adding in future pass")
    else:
        print("  feat_days_to_study_completion: SKIPPED — no ct_study_completion in dataset")

    # ── Build timing features ─────────────────────────────────────────────────
    print("\nBuilding timing features...")
    df_new = build_timing_features(df)

    cols_before = set(df.columns)
    cols_after  = set(df_new.columns)
    added   = sorted(c for c in cols_after - cols_before if c in NEW_FEAT_NAMES)
    updated = sorted(c for c in cols_before & set(NEW_FEAT_NAMES))

    print(f"\nNew feature columns added  : {len(added)}")
    for c in added:
        print(f"  + {c}")
    print(f"Feature columns revalidated: {len(updated)}")
    for c in updated:
        print(f"  ~ {c}")

    # ── Archive current versions (base dir only) ──────────────────────────────
    for prefix in ["ml_dataset_features", "ml_feature_dict"]:
        base_files = glob.glob(os.path.join(BASE_DIR, f"{prefix}_*.csv"))
        for f in base_files:
            if re.search(r"_v\d+\.csv$", f):
                dest = os.path.join(ARCHIVE_DIR, os.path.basename(f))
                if not os.path.exists(dest):
                    shutil.move(f, dest)
                    print(f"Archived: archive/{os.path.basename(f)}")
                else:
                    os.remove(f)
                    print(f"Removed duplicate: {os.path.basename(f)}")

    # ── Save new feature dataset ──────────────────────────────────────────────
    new_feat_v = _next_version_in_basedir(BASE_DIR, ARCHIVE_DIR, "ml_dataset_features", date_tag)
    out_feat   = os.path.join(BASE_DIR, f"ml_dataset_features_{date_tag}_v{new_feat_v}.csv")
    df_new.to_csv(out_feat, index=False)
    print(f"\nSaved  : {os.path.basename(out_feat)}  "
          f"({df_new.shape[0]} rows × {df_new.shape[1]} cols)")

    # ── Save new feature dictionary ───────────────────────────────────────────
    new_dict_v = _next_version_in_basedir(BASE_DIR, ARCHIVE_DIR, "ml_feature_dict", date_tag)
    out_dict   = os.path.join(BASE_DIR, f"ml_feature_dict_{date_tag}_v{new_dict_v}.csv")
    update_feature_dict(df_new, old_dict_path, out_dict)

    # ── Coverage summary ──────────────────────────────────────────────────────
    print("\n── New feature coverage ─────────────────────────────────────────────")
    for feat, *_ in NEW_FEATURES_META:
        if feat in df_new.columns:
            n_valid = df_new[feat].notna().sum()
            pct = n_valid / len(df_new) * 100
            print(f"  {feat:<52}  {n_valid:4d}/{len(df_new)}  ({pct:.1f}%)")
        else:
            print(f"  {feat:<52}  NOT ADDED")

    print("\nDone.")


if __name__ == "__main__":
    main()

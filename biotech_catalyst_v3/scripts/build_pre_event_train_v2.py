"""
build_pre_event_train_v2.py
============================
Part 3 — Build refreshed pre-event training table with new timing features.

Input  : latest ml_dataset_features_*_vN.csv
Output : ml_baseline_train_20260312_v2.csv
         ml_baseline_train_dict_20260312_v2.csv

Changes vs v1:
- Adds 10 new timing/sequence features (from add_pre_event_timing_features.py)
- Does NOT include the fold-safe priors (those are injected inside CV folds)
- Same training filter: row_ready & v_actual_date notna
- Same split: 70/15/15 time-based
- Same imputation/encoding logic

Usage (from biotech_catalyst_v3/):
    python -m scripts.build_pre_event_train_v2
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

DATE_TAG = "20260317"
VERSION  = 3

# Only train on events from 2023+ (2020-2022 rows have near-zero positive rate
# due to missing price data, which would make the train split almost label-free)
MIN_EVENT_YEAR = 2023

# ---------------------------------------------------------------------------
# Feature roster
# ---------------------------------------------------------------------------

# Kept from v1
NUMERIC_FEATURES_V1 = [
    "feat_phase_num",
    "feat_regulatory_stage_score",
    "feat_pivotal_proxy_score",
    "feat_design_quality_score",
    "feat_trial_quality_score",
    "feat_enrollment_log",
    "feat_days_to_primary_completion",
    "feat_n_unique_drugs_for_company",
    "feat_asset_trial_share",
    "feat_pipeline_depth_score",
    "feat_lead_asset_dependency_score",
    "feat_n_trials_for_company",
    "feat_volatility",
    "feat_log_market_cap",
    "feat_cash_runway_proxy",
]

BINARY_FEATURES_V1 = [
    "feat_late_stage_flag",
    "feat_active_not_recruiting_flag",
    "feat_completed_flag",
    "feat_recent_completion_flag",
    "feat_orphan_flag",
    "feat_breakthrough_flag",
    "feat_fast_track_flag",
    "feat_nda_bla_flag",
    "feat_randomized_flag",
    "feat_blinded_flag",
    "feat_open_label_flag",
    "feat_small_trial_flag",
    "feat_oncology_flag",
    "feat_cns_flag",
    "feat_rare_disease_flag",
    "feat_single_asset_company_flag",
    "feat_short_squeeze_flag",
    "feat_ownership_low_flag",
]

# New timing features (Part 1)
NEW_NUMERIC_FEATURES = [
    "feat_days_to_primary_completion",         # already in v1, but revalidated
    "feat_time_since_last_company_event",      # NaN for first event → median impute
    "feat_time_since_last_asset_event",        # NaN for first asset event → median impute
    "feat_asset_event_sequence_num",           # ordinal
    "feat_company_event_sequence_num",         # ordinal
]

NEW_BINARY_FEATURES = [
    "feat_primary_completion_imminent_30d",    # 0 safe for unknown
    "feat_primary_completion_imminent_90d",    # 0 safe for unknown
    "feat_recent_company_event_flag",          # 0 safe
    "feat_recent_asset_event_flag",            # 0 safe
]

NEW_CATEGORICAL_FEATURES = [
    "feat_completion_recency_bucket",          # 6 levels → one-hot
]

# Combined
NUMERIC_FEATURES = sorted(set(NUMERIC_FEATURES_V1 + NEW_NUMERIC_FEATURES))
BINARY_FEATURES  = sorted(set(BINARY_FEATURES_V1 + NEW_BINARY_FEATURES))
CATEGORICAL_FEATURES = [
    "feat_therapeutic_superclass",
    "feat_event_proximity_bucket",
] + NEW_CATEGORICAL_FEATURES

ORDINAL_INT_FEATURES = [
    "feat_mesh_level1_encoded",
]

ALL_FEATURE_COLS = (NUMERIC_FEATURES + BINARY_FEATURES +
                    CATEGORICAL_FEATURES + ORDINAL_INT_FEATURES)

TARGET_COL    = "target_large_move"
METADATA_COLS = ["ticker", "event_date", "drug_name", "nct_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_feat(base_dir, archive_dir):
    candidates = glob.glob(os.path.join(base_dir, "ml_dataset_features_*.csv"))
    best, best_v = None, 0
    for f in candidates:
        m = re.search(r"_v(\d+)\.csv$", f)
        if m and int(m.group(1)) > best_v:
            best_v, best = int(m.group(1)), f
    return best


def impute(df, feat_cols):
    df = df.copy()
    imputation_log = []
    for col in feat_cols:
        if col not in df.columns:
            continue
        n_miss = df[col].isna().sum()
        if n_miss == 0:
            imputation_log.append((col, "none", 0))
            continue
        if col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("unknown")
            imputation_log.append((col, "unknown", int(n_miss)))
        elif col in BINARY_FEATURES or col in ORDINAL_INT_FEATURES:
            df[col] = df[col].fillna(0.0)
            imputation_log.append((col, "0 (absent)", int(n_miss)))
        else:
            med = df[col].median()
            fill_val = med if pd.notna(med) else 0.0
            df[col] = df[col].fillna(fill_val)
            imputation_log.append((col, f"median={fill_val:.4f}", int(n_miss)))
    return df, imputation_log


def encode_categoricals(df, cat_cols):
    for col in cat_cols:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def time_split(df, date_col="_split_date", train_pct=0.70, val_pct=0.15):
    """Assign split labels based on sorted date percentiles."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n  = len(df)
    tr_end = int(n * train_pct)
    va_end = int(n * (train_pct + val_pct))
    split = ["train"] * tr_end + ["val"] * (va_end - tr_end) + ["test"] * (n - va_end)
    df["split"] = split
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    src_path = _find_latest_feat(BASE_DIR, ARCHIVE_DIR)
    if not src_path:
        print("ERROR: no ml_dataset_features_*.csv found", file=sys.stderr)
        sys.exit(1)
    print(f"Source: {os.path.basename(src_path)}")

    df_raw = pd.read_csv(src_path)
    print(f"Raw: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # ── Compute binary target if missing ─────────────────────────────────────
    # target_large_move = 1 when abs_move_atr >= 3.0 AND abs(move_pct) >= 10%
    if TARGET_COL not in df_raw.columns:
        abs_atr = df_raw["stock_movement_atr_normalized"].abs()
        abs_pct = df_raw["move_pct"].abs()
        df_raw[TARGET_COL] = ((abs_atr >= 3.0) & (abs_pct >= 10.0)).astype(int)
        print(f"Computed {TARGET_COL}: {int(df_raw[TARGET_COL].sum())} positives "
              f"({df_raw[TARGET_COL].mean():.1%})")

    # ── Training filter ───────────────────────────────────────────────────────
    mask = df_raw["row_ready"].astype(bool) & df_raw["v_actual_date"].notna()
    df   = df_raw[mask].copy()
    print(f"After row_ready filter: {len(df)} rows")

    # Restrict to MIN_EVENT_YEAR+ (2020-2022 rows have ~0% positives due to
    # missing price data; time-split puts them all in train → 0.6% positive rate)
    df["_event_year"] = pd.to_datetime(df["v_actual_date"], errors="coerce").dt.year
    before_year_filter = len(df)
    df = df[df["_event_year"] >= MIN_EVENT_YEAR].copy()
    df = df.drop(columns=["_event_year"])
    print(f"After year >= {MIN_EVENT_YEAR} filter: {len(df)} rows "
          f"(excluded {before_year_filter - len(df)} pre-{MIN_EVENT_YEAR} rows)")

    # ── Use validated date for splitting ─────────────────────────────────────
    df["_split_date"] = pd.to_datetime(df["v_actual_date"], errors="coerce")
    df = time_split(df, "_split_date")
    df = df.drop(columns=["_split_date"])
    print(f"Split: {df['split'].value_counts().to_dict()}")

    # ── Check feature availability ────────────────────────────────────────────
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    missing   = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Missing features (will skip): {missing}")

    keep_cols   = METADATA_COLS + [TARGET_COL] + available
    df_before   = df[[c for c in keep_cols if c in df.columns]].copy()

    # ── Impute ────────────────────────────────────────────────────────────────
    df_imp, imp_log = impute(df_before, available)

    # ── One-hot encode categoricals ───────────────────────────────────────────
    df_enc = encode_categoricals(df_imp, CATEGORICAL_FEATURES)

    # Add back split column
    df_enc["split"] = df["split"].values

    feat_cols_final = [
        c for c in df_enc.columns
        if c not in set(METADATA_COLS) | {TARGET_COL, "split"}
    ]
    print(f"Final feature count: {len(feat_cols_final)}")

    # ── Archive superseded v1 train table ─────────────────────────────────────
    for prefix in ["ml_baseline_train", "ml_baseline_train_dict"]:
        old_files = glob.glob(os.path.join(BASE_DIR, f"{prefix}_*.csv"))
        for f in old_files:
            dest = os.path.join(ARCHIVE_DIR, os.path.basename(f))
            if not os.path.exists(dest):
                shutil.move(f, dest)
                print(f"Archived: archive/{os.path.basename(f)}")
            else:
                os.remove(f)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_data = os.path.join(BASE_DIR, f"ml_baseline_train_{DATE_TAG}_v{VERSION}.csv")
    out_dict = os.path.join(BASE_DIR, f"ml_baseline_train_dict_{DATE_TAG}_v{VERSION}.csv")

    final_order = METADATA_COLS + [TARGET_COL] + feat_cols_final + ["split"]
    final_order = [c for c in final_order if c in df_enc.columns]
    df_enc[final_order].to_csv(out_data, index=False)

    # Build dict
    imp_map = {col: (method, miss) for col, method, miss in imp_log}
    dict_rows = []
    for col in final_order:
        role = "metadata" if col in METADATA_COLS else \
               "target"   if col == TARGET_COL    else \
               "split"    if col == "split"        else "feature"
        orig = next((c for c in CATEGORICAL_FEATURES if col.startswith(c + "_")), col)
        method, miss = imp_map.get(orig, ("none", 0))
        dict_rows.append({
            "column_name":    col,
            "role":           role,
            "source_feature": orig,
            "dtype":          str(df_enc[col].dtype) if col in df_enc.columns else "?",
            "imputation_used": method,
            "missing_before": int(miss),
            "missing_after":  int(df_enc[col].isna().sum()) if col in df_enc.columns else 0,
            "note":           (
                "one-hot encoded" if orig in CATEGORICAL_FEATURES else
                "ordinal int"     if col in ORDINAL_INT_FEATURES   else
                "binary flag"     if col in BINARY_FEATURES        else
                "numeric"         if col in NUMERIC_FEATURES       else "other"
            ),
        })
    pd.DataFrame(dict_rows).to_csv(out_dict, index=False)

    print(f"\nSaved: {os.path.basename(out_data)}  ({len(df_enc)} rows × {len(final_order)} cols)")
    print(f"Saved: {os.path.basename(out_dict)}  ({len(dict_rows)} entries)")
    print(f"\nTarget distribution: {df_enc[TARGET_COL].value_counts().to_dict()}")
    print(f"Positive rate: {df_enc[TARGET_COL].mean():.1%}")

    # Feature summary
    print(f"\nNew timing features included:")
    all_new = NEW_NUMERIC_FEATURES + NEW_BINARY_FEATURES + NEW_CATEGORICAL_FEATURES
    for feat in all_new:
        found_cols = [c for c in feat_cols_final if c.startswith(feat)]
        if found_cols:
            print(f"  ✓ {feat}  ({len(found_cols)} column(s))")
        else:
            print(f"  ✗ {feat}  (NOT IN DATASET)")


if __name__ == "__main__":
    main()

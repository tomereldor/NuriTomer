"""
build_baseline_training_table.py
=================================
Build the final baseline training table for biotech large-move prediction.

Input  : ml_dataset_features_20260310_v4.csv
Output : ml_baseline_train_20260310_v1.csv  (or v2, v3 if earlier exists)
         ml_baseline_train_dict_20260310_v1.csv

Training filter : df[row_ready & v_actual_date.notna()]
Target          : target_large_move  (binary: 1 = High or Extreme ATR-normalised move)

Usage (from biotech_catalyst_v3/):
    python -m scripts.build_baseline_training_table
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
ML_DATA_DIR = os.path.join(BASE_DIR, "data", "ml")

# ---------------------------------------------------------------------------
# Feature roster
# ---------------------------------------------------------------------------

# Outcome-leaning features excluded from baseline v1:
# feat_endpoint_outcome_score     → derived from primary_endpoint_met (yes/no = trial outcome)
# feat_primary_endpoint_known_flag → derived from primary_endpoint_met
# feat_superiority_flag           → keyword-extracted from outcome text (primary_endpoint_result, v_summary)
# feat_stat_sig_flag              → keyword-extracted from outcome/result text
# feat_clinically_meaningful_flag → keyword-extracted from outcome text
# feat_mixed_results_flag         → keyword-extracted from outcome/result text
# These encode WHAT HAPPENED — valid for a "given announcement" model but
# excluded from v1 to keep it clean as a pre-announcement baseline.

NUMERIC_FEATURES = [
    # Trial / regulatory
    "feat_phase_num",
    "feat_regulatory_stage_score",
    "feat_pivotal_proxy_score",
    "feat_design_quality_score",
    "feat_trial_quality_score",
    "feat_enrollment_log",
    "feat_days_to_primary_completion",
    # Company pipeline
    "feat_n_unique_drugs_for_company",
    "feat_asset_trial_share",
    "feat_pipeline_depth_score",
    "feat_lead_asset_dependency_score",
    # Financial
    "feat_volatility",
    "feat_log_market_cap",
    "feat_cash_runway_proxy",
]

BINARY_FEATURES = [
    # Clinical flags
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
    # Company
    "feat_single_asset_company_flag",
    # Financial
    "feat_short_squeeze_flag",
    "feat_ownership_low_flag",
]

CATEGORICAL_FEATURES = [
    "feat_therapeutic_superclass",   # 11 categories → one-hot
    "feat_event_proximity_bucket",   # 5 categories → one-hot
]

ORDINAL_INT_FEATURES = [
    "feat_mesh_level1_encoded",      # 1–11, keep as ordinal int
]

ALL_FEATURE_COLS = (NUMERIC_FEATURES + BINARY_FEATURES +
                    CATEGORICAL_FEATURES + ORDINAL_INT_FEATURES)

TARGET_COL   = "target_large_move"
METADATA_COLS = ["ticker", "event_date", "drug_name", "nct_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_version_file(base_dir, prefix):
    files = glob.glob(os.path.join(base_dir, f"{prefix}_*.csv"))
    best, best_v, date_tag = None, 0, None
    for f in files:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, date_tag = v, f, m.group(1)
    return best, best_v, date_tag


def _next_version(base_dir, prefix, date_tag):
    files = glob.glob(os.path.join(base_dir, f"{prefix}_{date_tag}_v*.csv"))
    nums = [int(re.search(r"_v(\d+)\.csv$", f).group(1))
            for f in files if re.search(r"_v(\d+)\.csv$", f)]
    return max(nums) + 1 if nums else 1


# ---------------------------------------------------------------------------
# Pre-training audit
# ---------------------------------------------------------------------------

def pre_training_audit(df, feature_cols):
    n = len(df)
    print("\n=== PRE-TRAINING AUDIT ===")
    print(f"Training rows: {n}")
    print(f"Target balance: {df[TARGET_COL].value_counts().to_dict()}")
    imbal = df[TARGET_COL].mean()
    print(f"  Positive rate (large move): {imbal:.1%}")

    issues = []
    print(f"\n{'Feature':<48} {'Null%':>7}  {'Var':>8}  {'Status'}")
    print("-" * 80)
    for col in feature_cols:
        if col not in df.columns:
            print(f"  {col:<46} {'NOT IN DF':>7}")
            issues.append((col, "NOT IN DF"))
            continue
        null_pct = df[col].isna().mean() * 100
        try:
            var = df[col].var(numeric_only=True) if df[col].dtype != object else None
        except Exception:
            var = None
        var_str = f"{var:.4f}" if var is not None else "categ."
        status = "OK"
        if null_pct > 30:
            status = "HIGH NULL"
        elif var is not None and var < 1e-6:
            status = "ZERO VAR"
        print(f"  {col:<46} {null_pct:>6.1f}%  {var_str:>8}  {status}")
        if status != "OK":
            issues.append((col, status))

    print(f"\nExcluded (outcome-leaning, not in proposed list):")
    excluded_outcome = [
        "feat_endpoint_outcome_score",
        "feat_primary_endpoint_known_flag",
        "feat_superiority_flag",
        "feat_stat_sig_flag",
        "feat_clinically_meaningful_flag",
        "feat_mixed_results_flag",
    ]
    for col in excluded_outcome:
        print(f"  {col}  ← outcome-leaning: derived from press release result text")

    print(f"\nExcluded (historical priors — train-only, leakage risk):")
    prior_cols = [c for c in df.columns if "prior_mean" in c]
    for c in prior_cols:
        print(f"  {c}  ← computed on full dataset; recompute inside folds before use")

    if issues:
        print(f"\nIssues found: {issues}")
    return issues


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute(df, feature_cols):
    """Apply safe imputation per feature type."""
    imputation_log = []
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            continue
        n_miss = df[col].isna().sum()
        if n_miss == 0:
            imputation_log.append((col, "none", 0))
            continue

        if col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("unknown")
            imputation_log.append((col, "unknown", n_miss))
        elif col in BINARY_FEATURES or col == "feat_mesh_level1_encoded":
            # 0 is semantically safe for all binary flags (unknown = assume absent)
            df[col] = df[col].fillna(0.0)
            imputation_log.append((col, "0 (absent)", n_miss))
        else:  # numeric
            med = df[col].median()
            df[col] = df[col].fillna(med)
            imputation_log.append((col, f"median={med:.4f}", n_miss))

    return df, imputation_log


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df, cat_cols):
    """One-hot encode categorical features, drop first to avoid multicollinearity."""
    for col in cat_cols:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


# ---------------------------------------------------------------------------
# Build train dict
# ---------------------------------------------------------------------------

def build_train_dict(df_before, df_after, imputation_log, feature_cols_expanded):
    imp_map = {col: (method, miss) for col, method, miss in imputation_log}
    rows = []

    # Metadata
    for col in METADATA_COLS:
        rows.append({
            "column_name": col, "role": "metadata",
            "source_feature": col, "dtype": str(df_after[col].dtype) if col in df_after.columns else "str",
            "imputation_used": "none", "missing_before": 0, "missing_after": 0,
            "note": "traceability only — do not pass to model",
        })

    # Target
    rows.append({
        "column_name": TARGET_COL, "role": "target",
        "source_feature": TARGET_COL, "dtype": str(df_after[TARGET_COL].dtype),
        "imputation_used": "none",
        "missing_before": int(df_before[TARGET_COL].isna().sum()),
        "missing_after": int(df_after[TARGET_COL].isna().sum()),
        "note": "binary: 1=High/Extreme ATR-normalised move, 0=Noise/Low/Medium",
    })

    # Features
    for col in feature_cols_expanded:
        src = col  # for one-hot columns, original is the prefix
        orig_src = next(
            (c for c in CATEGORICAL_FEATURES if col.startswith(c + "_")), col
        )
        method, miss = imp_map.get(orig_src, ("none", 0))
        miss_after = int(df_after[col].isna().sum()) if col in df_after.columns else 0

        rows.append({
            "column_name": col, "role": "feature",
            "source_feature": orig_src, "dtype": str(df_after[col].dtype) if col in df_after.columns else "?",
            "imputation_used": method,
            "missing_before": int(miss),
            "missing_after": miss_after,
            "note": (
                "one-hot encoded" if orig_src in CATEGORICAL_FEATURES else
                "ordinal integer" if col in ORDINAL_INT_FEATURES else
                "binary flag" if col in BINARY_FEATURES else
                "numeric"
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    src_path, _, date_tag = _latest_version_file(ML_DATA_DIR, "ml_dataset_features")
    if not src_path:
        print("ERROR: no ml_dataset_features_*.csv found", file=sys.stderr)
        sys.exit(1)

    print(f"Source: {os.path.basename(src_path)}")
    df_raw = pd.read_csv(src_path)
    print(f"Raw: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # ---- Training filter ----
    mask = df_raw["row_ready"].astype(bool) & df_raw["v_actual_date"].notna()
    df = df_raw[mask].copy()
    print(f"After filter (row_ready & v_actual_date notna): {len(df)} rows")

    # ---- Pre-training audit ----
    pre_training_audit(df, ALL_FEATURE_COLS)

    # ---- Subset to needed columns ----
    avail_features = [c for c in ALL_FEATURE_COLS if c in df.columns]
    keep_cols = METADATA_COLS + [TARGET_COL] + avail_features
    df_before = df[keep_cols].copy()

    # ---- Impute ----
    df_imp, imp_log = impute(df_before, avail_features)

    # ---- Encode categoricals ----
    df_enc = encode_categoricals(df_imp, CATEGORICAL_FEATURES)

    # Final feature columns after encoding
    feat_cols_final = [
        c for c in df_enc.columns
        if c not in METADATA_COLS and c != TARGET_COL
    ]
    print(f"\nFinal feature columns: {len(feat_cols_final)}")
    print(f"  {feat_cols_final}")

    # ---- Archive superseded baseline train files ----
    for prefix in ["ml_baseline_train", "ml_baseline_train_dict"]:
        old_path, old_v, _ = _latest_version_file(ML_DATA_DIR, prefix)
        if old_path and os.path.exists(old_path):
            dest = os.path.join(ARCHIVE_DIR, os.path.basename(old_path))
            shutil.move(old_path, dest)
            print(f"Archived: archive/{os.path.basename(old_path)}")

    # ---- Save ----
    os.makedirs(ML_DATA_DIR, exist_ok=True)
    new_v = _next_version(ML_DATA_DIR, "ml_baseline_train", date_tag)
    out_data = os.path.join(ML_DATA_DIR, f"ml_baseline_train_{date_tag}_v{new_v}.csv")
    out_dict = os.path.join(ML_DATA_DIR, f"ml_baseline_train_dict_{date_tag}_v{new_v}.csv")

    # Keep metadata in final file for traceability, but after target
    final_cols = METADATA_COLS + [TARGET_COL] + feat_cols_final
    df_enc[final_cols].to_csv(out_data, index=False)

    train_dict = build_train_dict(df_before, df_enc, imp_log, feat_cols_final)
    train_dict.to_csv(out_dict, index=False)

    print(f"\nSaved : {os.path.basename(out_data)}  ({len(df_enc)} rows × {len(final_cols)} cols)")
    print(f"Saved : {os.path.basename(out_dict)}  ({len(train_dict)} entries)")
    print(f"\nTarget distribution:\n{df_enc[TARGET_COL].value_counts().to_dict()}")
    print(f"Positive rate: {df_enc[TARGET_COL].mean():.1%}")


if __name__ == "__main__":
    main()

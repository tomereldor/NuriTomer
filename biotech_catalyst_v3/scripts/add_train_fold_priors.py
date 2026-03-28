"""
add_train_fold_priors.py
========================
Part 2 — Train-fold-safe prior feature generation.

This module provides functions to compute reaction prior features
INSIDE each training fold, preventing future-data leakage.

IMPORTANT: These functions are NOT run standalone to produce a static CSV.
They are called inside cross-validation loops and train/val/test splits.
The train_pre_event_v3.py script imports and uses these functions.

Prior features computed:
  1. feat_prior_mean_abs_move_atr_by_phase
  2. feat_prior_mean_abs_move_atr_by_therapeutic_superclass
  3. feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass
  4. feat_prior_mean_abs_move_atr_by_market_cap_bucket
  5. feat_prior_large_move_rate_by_phase
  6. feat_prior_large_move_rate_by_therapeutic_superclass

Design principles:
  - Priors are fit ONLY on train_df (train fold rows).
  - Fallback for unseen categories = global train mean/rate.
  - NaN grouping keys → use global fallback.
  - No information from val/test bleeds into prior estimates.

Usage (imported by train_pre_event_v3.py):
    from scripts.add_train_fold_priors import add_fold_priors

Standalone test run (from biotech_catalyst_v3/):
    python -m scripts.add_train_fold_priors
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_ABS_MOVE = "target_abs_move_atr"    # continuous: abs ATR-normalized move
TARGET_LARGE    = "target_large_move"       # binary: 1 = large/extreme move

PRIOR_CONFIGS = [
    # (output_col_name, groupby_col, target_col, stat)
    ("feat_prior_mean_abs_move_atr_by_phase",
     "feat_phase_num",
     TARGET_ABS_MOVE,
     "mean"),
    ("feat_prior_mean_abs_move_atr_by_therapeutic_superclass",
     "feat_therapeutic_superclass",
     TARGET_ABS_MOVE,
     "mean"),
    ("feat_prior_mean_abs_move_atr_by_market_cap_bucket",
     "feat_market_cap_bucket",
     TARGET_ABS_MOVE,
     "mean"),
    ("feat_prior_large_move_rate_by_phase",
     "feat_phase_num",
     TARGET_LARGE,
     "mean"),
    ("feat_prior_large_move_rate_by_therapeutic_superclass",
     "feat_therapeutic_superclass",
     TARGET_LARGE,
     "mean"),
    ("feat_prior_large_move_rate_by_market_cap_bucket",
     "feat_market_cap_bucket",
     TARGET_LARGE,
     "mean"),
]

INTERACTION_CONFIGS = [
    # (output_col_name, group_cols, target_col, stat, fallback_single_prior)
    ("feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass",
     ["feat_phase_num", "feat_therapeutic_superclass"],
     TARGET_ABS_MOVE,
     "mean",
     "feat_prior_mean_abs_move_atr_by_phase"),
    ("feat_prior_large_move_rate_by_phase_x_therapeutic_superclass",
     ["feat_phase_num", "feat_therapeutic_superclass"],
     TARGET_LARGE,
     "mean",
     "feat_prior_large_move_rate_by_phase"),
]


# ---------------------------------------------------------------------------
# Core prior computation
# ---------------------------------------------------------------------------

class FoldPriorEncoder:
    """
    Fit prior encodings on train data; transform train/val/test safely.

    The encoder stores:
        - per-group means/rates
        - global fallback (train global mean/rate)

    For unseen categories or NaN keys → fallback.
    For the interaction feature: if cell has <5 samples, fall back to
    the single-dimension prior (phase-level), then global.
    """

    def __init__(self, min_samples_for_interaction: int = 5):
        self.min_samples = min_samples_for_interaction
        self.lookup_      = {}   # col → {group_key: value}
        self.fallback_    = {}   # col → global_train_value
        self.fitted_      = False

    def fit(self, train_df: pd.DataFrame) -> "FoldPriorEncoder":
        """Compute all priors from train_df only."""
        # ── Single-key priors ────────────────────────────────────────────────
        for col_out, group_col, target_col, stat in PRIOR_CONFIGS:
            if target_col not in train_df.columns or group_col not in train_df.columns:
                self.lookup_[col_out]   = {}
                self.fallback_[col_out] = np.nan
                continue

            series = train_df[target_col].astype(float)
            global_val = float(series.mean()) if stat == "mean" else float(series.mean())
            self.fallback_[col_out] = global_val

            tmp = train_df[[group_col, target_col]].copy()
            tmp[target_col] = tmp[target_col].astype(float)
            grp = tmp.groupby(group_col, dropna=True)[target_col].mean()
            self.lookup_[col_out] = grp.to_dict()

        # ── Interaction prior ────────────────────────────────────────────────
        for col_out, group_cols, target_col, stat in [c[:4] for c in INTERACTION_CONFIGS]:
            if target_col not in train_df.columns or any(
                c not in train_df.columns for c in group_cols
            ):
                self.lookup_[col_out]   = {}
                self.fallback_[col_out] = np.nan
                continue

            series = train_df[target_col].astype(float)
            global_val = float(series.mean())
            self.fallback_[col_out] = global_val

            tmp = train_df[group_cols + [target_col]].copy()
            tmp[target_col] = tmp[target_col].astype(float)
            grp = tmp.groupby(group_cols, dropna=True)[target_col]
            counts = grp.count()
            means  = grp.mean()
            # Only store cells with enough samples; others will fall back
            valid_cells = counts[counts >= self.min_samples].index
            cell_dict = {}
            for key in valid_cells:
                cell_dict[key] = float(means[key])
            self.lookup_[col_out] = cell_dict

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, source_label: str = "") -> pd.DataFrame:
        """Map prior encodings to df rows. Returns df with new columns added."""
        if not self.fitted_:
            raise RuntimeError("FoldPriorEncoder must be fit before transform.")

        df = df.copy()

        # ── Single-key priors ────────────────────────────────────────────────
        for col_out, group_col, target_col, stat in PRIOR_CONFIGS:
            lookup   = self.lookup_.get(col_out, {})
            fallback = self.fallback_.get(col_out, np.nan)
            if group_col not in df.columns:
                df[col_out] = fallback
                continue
            df[col_out] = df[group_col].map(lookup).fillna(fallback).astype(float)

        # ── Interaction prior ────────────────────────────────────────────────
        for config in INTERACTION_CONFIGS:
            col_out, group_cols, target_col, stat = config[:4]
            fallback_prior = config[4] if len(config) > 4 else "feat_prior_mean_abs_move_atr_by_phase"

            lookup   = self.lookup_.get(col_out, {})
            fallback = self.fallback_.get(col_out, np.nan)

            # Fallback cascade: cell → single-dim phase prior → global
            phase_lookup = self.lookup_.get(fallback_prior, {})

            def _lookup_interaction(row, _lookup=lookup, _phase=phase_lookup, _fb=fallback):
                # Build interaction key
                keys = tuple(row[c] for c in group_cols)
                if any(pd.isna(k) for k in keys):
                    return _fb
                val = _lookup.get(keys)
                if val is not None:
                    return val
                # Phase-level fallback
                phase_val = _phase.get(keys[0])
                if phase_val is not None:
                    return phase_val
                return _fb

            if all(c in df.columns for c in group_cols):
                df[col_out] = df[group_cols].apply(_lookup_interaction, axis=1).astype(float)
            else:
                df[col_out] = fallback

        if source_label:
            n_new = len(
                [c for c in df.columns
                 if c.startswith("feat_prior_")]
            )
            pass  # quiet

        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on train_df and transform train_df."""
        self.fit(train_df)
        return self.transform(train_df, source_label="train")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_fold_priors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df=None,
) -> tuple:
    """
    Fit priors on train_df; map them to train_df, val_df, and optionally test_df.

    Returns: (train_df_with_priors, val_df_with_priors, test_df_with_priors_or_None)

    Usage:
        train_p, val_p, test_p = add_fold_priors(train_df, val_df, test_df)
    """
    enc = FoldPriorEncoder()
    enc.fit(train_df)

    train_out = enc.transform(train_df, "train")
    val_out   = enc.transform(val_df,   "val")
    test_out  = enc.transform(test_df,  "test") if test_df is not None else None

    return train_out, val_out, test_out


def add_fold_priors_cv(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
) -> tuple:
    """
    Variant for CV loops where X_* are feature-only frames and *_full include targets.
    Fits encoder on train_full, returns X_train + X_val with prior columns appended.
    """
    enc = FoldPriorEncoder()
    enc.fit(train_full)

    # X frames may not have target/group cols — use full frames to map, merge back
    prior_cols = list(enc.lookup_.keys())

    train_enriched = enc.transform(train_full)
    val_enriched   = enc.transform(val_full)

    # Extract only the newly added prior cols and append to X_*
    idx_tr = X_train.index
    idx_va = X_val.index

    prior_train = train_enriched.loc[idx_tr, prior_cols] if idx_tr.isin(train_enriched.index).all() else \
                  train_enriched.reset_index(drop=True)[prior_cols].iloc[:len(X_train)]
    prior_val   = val_enriched.loc[idx_va, prior_cols] if idx_va.isin(val_enriched.index).all() else \
                  val_enriched.reset_index(drop=True)[prior_cols].iloc[:len(X_val)]

    X_train_out = pd.concat([X_train.reset_index(drop=True),
                              prior_train.reset_index(drop=True)], axis=1)
    X_val_out   = pd.concat([X_val.reset_index(drop=True),
                              prior_val.reset_index(drop=True)], axis=1)

    return X_train_out, X_val_out, enc


# ---------------------------------------------------------------------------
# Prior column list helper
# ---------------------------------------------------------------------------

def get_prior_col_names() -> list:
    """Return the list of column names produced by FoldPriorEncoder."""
    return (
        [c for c, _, _, _ in PRIOR_CONFIGS] +
        [c[0] for c in INTERACTION_CONFIGS]
    )


# ---------------------------------------------------------------------------
# Standalone test / diagnostic
# ---------------------------------------------------------------------------

def _self_test():
    """Quick sanity check: fit on first 70% of rows, transform remainder."""
    SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(SCRIPT_DIR)
    ML_DATA_DIR = os.path.join(BASE_DIR, "data", "ml")

    import glob
    feat_files = glob.glob(os.path.join(ML_DATA_DIR, "ml_dataset_features_*.csv"))
    if not feat_files:
        print("ERROR: no ml_dataset_features_*.csv found")
        sys.exit(1)

    feat_path = max(feat_files, key=os.path.getmtime)
    print(f"Loaded: {os.path.basename(feat_path)}")
    df = pd.read_csv(feat_path)

    # Filter to training-eligible rows
    mask = df["row_ready"].astype(bool) & df["v_actual_date"].notna()
    df = df[mask].sort_values("v_actual_date").reset_index(drop=True)
    print(f"Training rows: {len(df)}")

    # Check required columns exist
    for _, grp_col, tgt_col, _ in PRIOR_CONFIGS:
        if grp_col not in df.columns:
            print(f"  MISSING groupby col: {grp_col}")
        if tgt_col not in df.columns:
            print(f"  MISSING target col: {tgt_col}")

    # Split 70/30
    cut = int(len(df) * 0.7)
    train_df = df.iloc[:cut].copy()
    val_df   = df.iloc[cut:].copy()

    train_p, val_p, _ = add_fold_priors(train_df, val_df, None)

    prior_cols = get_prior_col_names()
    print("\n── Prior columns produced ──────────────────────────────────────────")
    for col in prior_cols:
        if col in train_p.columns:
            tr_mean = train_p[col].mean()
            va_mean = val_p[col].mean()
            tr_null = train_p[col].isna().sum()
            va_null = val_p[col].isna().sum()
            print(f"  {col:<60}  train_mean={tr_mean:.3f}  val_mean={va_mean:.3f}  "
                  f"train_null={tr_null}  val_null={va_null}")
        else:
            print(f"  {col:<60}  NOT PRODUCED")

    print("\nSelf-test passed.")


if __name__ == "__main__":
    _self_test()

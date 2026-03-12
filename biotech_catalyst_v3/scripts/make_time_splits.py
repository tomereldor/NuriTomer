"""
make_time_splits.py
===================
Add time-based split assignments to the baseline training table.

Input  : ml_baseline_train_*_v?.csv
Output : same file with a 'split' column added (train/val/test)
         reports/split_summary_20260310_v1.csv

Split logic:
    Sort by event_date (ascending).
    train = oldest 70%
    val   = next 15%
    test  = newest 15%

This mirrors deployment: the model must generalise to future events.

Usage (from biotech_catalyst_v3/):
    python -m scripts.make_time_splits
"""

import glob
import os
import re
import sys

import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")


def _latest(base_dir, prefix):
    files = glob.glob(os.path.join(base_dir, f"{prefix}_*.csv"))
    best, best_v, date_tag = None, 0, None
    for f in files:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, date_tag = v, f, m.group(1)
    return best, best_v, date_tag


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_path, train_v, date_tag = _latest(BASE_DIR, "ml_baseline_train")
    if not train_path or "dict" in train_path:
        print("ERROR: no ml_baseline_train_*.csv found", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {os.path.basename(train_path)}")
    df = pd.read_csv(train_path, parse_dates=["event_date"])
    print(f"Loaded: {df.shape[0]} rows")

    # Time-based split
    df = df.sort_values("event_date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    df["split"] = "test"
    df.loc[:train_end - 1, "split"] = "train"
    df.loc[train_end:val_end - 1, "split"] = "val"

    # Save in-place (add split column)
    df.to_csv(train_path, index=False)
    print(f"Added 'split' column → {os.path.basename(train_path)}")

    # Split summary
    target_col = "target_large_move"
    summary_rows = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_pos = int(sub[target_col].sum())
        n_neg = int(len(sub) - n_pos)
        summary_rows.append({
            "split":       split,
            "n_rows":      len(sub),
            "date_min":    str(sub["event_date"].min())[:10],
            "date_max":    str(sub["event_date"].max())[:10],
            "n_positive":  n_pos,
            "n_negative":  n_neg,
            "positive_pct": round(n_pos / len(sub) * 100, 1),
        })
        print(f"  {split:5s}: {len(sub):4d} rows | "
              f"{str(sub['event_date'].min())[:10]} → {str(sub['event_date'].max())[:10]} | "
              f"pos={n_pos} ({n_pos/len(sub)*100:.1f}%)")

    summary = pd.DataFrame(summary_rows)
    out_path = os.path.join(REPORTS_DIR, f"split_summary_{date_tag}_v{train_v}.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved split summary: reports/{os.path.basename(out_path)}")


if __name__ == "__main__":
    main()

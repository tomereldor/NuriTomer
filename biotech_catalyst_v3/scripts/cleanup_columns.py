"""Clean up and normalize enriched_all_clinical.csv (no network required).

What this script does
---------------------
1.  Fills price_at_event from the `close` column (batch-3 rows, ~400 rows)
2.  Derives drug_name from `interventions` for rows that are missing it
3.  Derives indication from `ct_conditions` for rows that are missing it
4.  Normalises ct_phase to human-readable values (PHASE3 → Phase 3, etc.)
5.  Renames `normalized_move` → `stock_movement_atr_normalized`
6.  Drops redundant / stale columns
7.  Saves the result in place (creates a .bak backup first)

Columns dropped
---------------
Exact duplicates of ct_* columns (confirmed 100% match):
    title, sponsor, conditions, enrollment

Superseded / messy / user-requested deletions:
    phase               → use ct_phase (now normalised)
    interventions       → used to derive drug_name then dropped
    move_class          → old classification scheme, superseded by move_class_combo
    move_magnitude      → superseded by move_class_norm
    cash_runway_months  → requested
    data_quality_score  → 5% fill, stale old-pipeline artefact
    timestamp           → 5% fill, pipeline-internal
    is_valid_date       → 5% fill, pipeline-internal
    errors              → 6% fill, pipeline-internal
    data_quality_threshold_passed → redundant with data_complete
    press_release_url   → 0% fill
    atr_value           → 0% fill (never computed)
    price_error         → 1% fill, pipeline-internal
    abs_move            → duplicate of abs(move_pct)
    close               → used to fill price_at_event then dropped
    volume              → only 18% fill, partial old-pipeline artefact

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3
    python -m scripts.cleanup_columns

    # Specify a different file
    python -m scripts.cleanup_columns --input enriched_all_clinical.csv
"""

import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PHASE_MAP = {
    "PHASE1":        "Phase 1",
    "PHASE2":        "Phase 2",
    "PHASE3":        "Phase 3",
    "PHASE4":        "Phase 4",
    "PHASE1/PHASE2": "Phase 1/2",
    "PHASE2/PHASE3": "Phase 2/3",
    "EARLY_PHASE1":  "Phase 1 (Early)",
}

# Lower-case strings that indicate a control arm, not a drug
_CONTROL_KEYWORDS = {
    "placebo", "vehicle", "saline", "control",
    "matching placebo", "matching tablet", "matching capsule",
}


def _normalise_phase(val):
    if pd.isna(val):
        return val
    return _PHASE_MAP.get(str(val).strip(), str(val).strip())


def _extract_drug_name(interventions_str):
    """Return the first non-control item from a semicolon-separated list."""
    if pd.isna(interventions_str):
        return None
    items = [s.strip() for s in str(interventions_str).split(";") if s.strip()]
    for item in items:
        if not any(kw in item.lower() for kw in _CONTROL_KEYWORDS):
            return item
    return items[0] if items else None


def _extract_indication(conditions_str):
    """Return the first condition from a semicolon-separated list."""
    if pd.isna(conditions_str):
        return None
    return str(conditions_str).split(";")[0].strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COLS_TO_DROP = [
    # exact duplicates of ct_* columns
    "title", "sponsor", "conditions", "enrollment",
    # superseded column (messy free-form phase labels)
    "phase",
    # used to derive drug_name, then dropped
    "interventions",
    # old/superseded classification
    "move_class", "move_magnitude",
    # user-requested removals
    "cash_runway_months",
    # stale old-pipeline artefacts (5% fill)
    "data_quality_score", "timestamp", "is_valid_date",
    # pipeline-internal / diagnostic
    "errors", "data_quality_threshold_passed", "press_release_url",
    "price_error",
    # never computed
    "atr_value",
    # partial old-pipeline batch data
    "abs_move", "close", "volume",
]


def cleanup_columns(input_file: str = "enriched_all_clinical.csv") -> pd.DataFrame:
    print(f"Loading {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # 1. Fill price_at_event from `close` (batch-3 rows have close but not
    #    price_at_event — both represent the closing price on event_date)
    # ------------------------------------------------------------------
    if "close" in df.columns and "price_at_event" in df.columns:
        mask = df["price_at_event"].isna() & df["close"].notna()
        df.loc[mask, "price_at_event"] = df.loc[mask, "close"]
        print(f"  price_at_event filled from close: {mask.sum()} rows")

    # ------------------------------------------------------------------
    # 2. Derive drug_name from interventions where missing
    # ------------------------------------------------------------------
    if "interventions" in df.columns and "drug_name" in df.columns:
        mask = df["drug_name"].isna() & df["interventions"].notna()
        df.loc[mask, "drug_name"] = df.loc[mask, "interventions"].map(_extract_drug_name)
        print(f"  drug_name derived from interventions: {mask.sum()} rows")

    # ------------------------------------------------------------------
    # 3. Derive indication from ct_conditions where missing
    # ------------------------------------------------------------------
    if "ct_conditions" in df.columns and "indication" in df.columns:
        mask = df["indication"].isna() & df["ct_conditions"].notna()
        df.loc[mask, "indication"] = df.loc[mask, "ct_conditions"].map(_extract_indication)
        print(f"  indication derived from ct_conditions: {mask.sum()} rows")

    # ------------------------------------------------------------------
    # 4. Normalise ct_phase to human-readable format
    # ------------------------------------------------------------------
    if "ct_phase" in df.columns:
        df["ct_phase"] = df["ct_phase"].map(_normalise_phase)
        print(f"  ct_phase normalised: {df['ct_phase'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 5. Rename normalized_move → stock_movement_atr_normalized
    # ------------------------------------------------------------------
    if "normalized_move" in df.columns:
        df = df.rename(columns={"normalized_move": "stock_movement_atr_normalized"})
        print("  Renamed: normalized_move → stock_movement_atr_normalized")

    # ------------------------------------------------------------------
    # 6. Drop redundant / stale columns
    # ------------------------------------------------------------------
    present = [c for c in COLS_TO_DROP if c in df.columns]
    absent  = [c for c in COLS_TO_DROP if c not in df.columns]
    df = df.drop(columns=present)
    print(f"  Dropped {len(present)} columns: {present}")
    if absent:
        print(f"  (Already absent — skipped: {absent})")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    bak = input_file.replace(".csv", ".bak.csv")
    shutil.copy(input_file, bak)
    print(f"\n  Backup saved → {bak}")

    df.to_csv(input_file, index=False)
    print(f"  Saved → {input_file}  ({len(df):,} rows × {len(df.columns)} columns)")

    # Summary
    print(f"\n{'='*60}")
    print("COLUMN FILL RATES AFTER CLEANUP")
    print(f"{'='*60}")
    for col in df.columns:
        n = df[col].notna().sum()
        pct = 100 * n / len(df)
        bar = "#" * int(pct / 5)
        print(f"  {col:<45s}  {pct:5.1f}%  {bar}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up and normalise enriched CSV columns")
    parser.add_argument("--input", default="enriched_all_clinical.csv",
                        help="CSV file to clean up (overwritten in place)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cleanup_columns(args.input)

"""
Apply all pipeline improvements to existing enriched_new_batch.csv.

Issues addressed:
  1. Add data_quality_threshold_passed column (>= 0.7)
  2. Fix catalyst_type: separate Unknown vs Other
  3. Flag missing financials in errors column + validate dates
  4. Add ATR normalization (atr_pct, normalized_move, move_magnitude)

Usage:
    python -m scripts.fix_existing_data
    python -m scripts.fix_existing_data --input enriched_new_batch.csv --skip-atr
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_quality import (
    add_quality_threshold,
    fix_catalyst_type,
    flag_date_issues,
    flag_missing_financials,
)


def fix_existing_data(
    input_file: str = "enriched_new_batch.csv",
    output_file: str = "enriched_improved.csv",
    skip_atr: bool = False,
    skip_urls: bool = False,
):
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # ------------------------------------------------------------------
    # Issue 1: Quality threshold
    # ------------------------------------------------------------------
    print("\n--- Issue 1: Quality threshold ---")
    df = add_quality_threshold(df, threshold=0.7)
    passed = df["data_quality_threshold_passed"].sum()
    print(f"  Passed (>= 0.7): {passed}/{len(df)} ({passed/len(df)*100:.0f}%)")

    # ------------------------------------------------------------------
    # Issue 2: Fix catalyst_type
    # ------------------------------------------------------------------
    print("\n--- Issue 2: Catalyst type reclassification ---")
    before = df["catalyst_type"].value_counts()
    df["catalyst_type"] = df.apply(fix_catalyst_type, axis=1)
    after = df["catalyst_type"].value_counts()

    # Show changes
    unknown_before = before.get("Unknown", 0)
    unknown_after = (df["catalyst_type"] == "Unknown").sum()
    other_before = before.get("Other", 0)
    other_after = df["catalyst_type"].str.startswith("Other").sum()
    print(f"  Unknown: {unknown_before} -> {unknown_after}")
    print(f"  Other (incl. Other: <title>): {other_before} -> {other_after}")
    print(f"  Distribution:")
    # Group Other: variants for display
    type_counts = df["catalyst_type"].apply(
        lambda x: "Other: ..." if str(x).startswith("Other:") else x
    ).value_counts()
    for t, c in type_counts.items():
        print(f"    {t}: {c}")

    # ------------------------------------------------------------------
    # Issue 3: Financial error logging + date validation
    # ------------------------------------------------------------------
    print("\n--- Issue 3: Error logging + date validation ---")
    errors_before = df["errors"].fillna("").str.len().gt(0).sum()

    df = flag_missing_financials(df)
    df = flag_date_issues(df)

    errors_after = df["errors"].fillna("").str.len().gt(0).sum()
    invalid_dates = (~df["is_valid_date"]).sum()
    print(f"  Rows with errors: {errors_before} -> {errors_after}")
    print(f"  Invalid dates flagged: {invalid_dates}")

    # ------------------------------------------------------------------
    # Issue 4: ATR normalization
    # ------------------------------------------------------------------
    if not skip_atr:
        print("\n--- Issue 4: ATR normalization ---")
        print("  This fetches historical data for each row; may take a few minutes...")
        from utils.volatility import enrich_with_atr

        df = enrich_with_atr(df)

        has_atr = df["atr_pct"].notna().sum()
        print(f"  ATR computed: {has_atr}/{len(df)}")
        if has_atr > 0:
            mag = df["move_magnitude"].value_counts()
            print(f"  Move magnitude distribution:")
            for m, c in mag.items():
                print(f"    {m}: {c}")
    else:
        print("\n--- Issue 4: ATR normalization (SKIPPED) ---")

    # ------------------------------------------------------------------
    # Issue 5: Find missing press release URLs
    # ------------------------------------------------------------------
    if not skip_urls:
        print("\n--- Issue 5: Find missing press release URLs ---")
        from scripts.find_press_release_urls import find_press_release_urls

        # Save intermediate so URL finder reads the updated data
        df.to_csv(output_file, index=False)
        find_press_release_urls(
            input_file=output_file,
            output_file=output_file,
        )
        # Reload after URL finder writes
        df = pd.read_csv(output_file)
    else:
        print("\n--- Issue 5: Press release URLs (SKIPPED) ---")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    cols_added = []
    if "data_quality_threshold_passed" in df.columns:
        cols_added.append("data_quality_threshold_passed")
    if "is_valid_date" in df.columns:
        cols_added.append("is_valid_date")
    if "atr_pct" in df.columns:
        cols_added.extend(["atr_pct", "normalized_move", "move_magnitude"])
    if cols_added:
        print(f"New columns: {', '.join(cols_added)}")
    has_url = (df["press_release_url"].notna() & (df["press_release_url"] != "")).sum()
    print(f"Press release URLs: {has_url}/{len(df)}")
    print(f"Total rows: {len(df)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix existing enriched data")
    parser.add_argument("--input", default="enriched_new_batch.csv")
    parser.add_argument("--output", default="enriched_improved.csv")
    parser.add_argument("--skip-atr", action="store_true", help="Skip ATR calculation (slow)")
    parser.add_argument("--skip-urls", action="store_true", help="Skip press release URL finding (slow)")
    args = parser.parse_args()

    fix_existing_data(
        input_file=args.input,
        output_file=args.output,
        skip_atr=args.skip_atr,
        skip_urls=args.skip_urls,
    )

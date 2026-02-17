"""
Run all data fixes in the correct order.

This is the single script to produce the final enriched dataset:
  1. Fix missing NCT IDs (with ticker fallbacks + detailed trace log)
  2. Apply all improvements (quality threshold, catalyst type, errors, ATR)
  3. Verify results

Usage:
    python -m scripts.full_pipeline_fix
    python -m scripts.full_pipeline_fix --input enriched_new_batch.csv --skip-atr
    python -m scripts.full_pipeline_fix --skip-nct   # skip NCT fix, just re-run improvements
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_full_pipeline(
    input_file: str = "enriched_new_batch.csv",
    output_file: str = "enriched_final.csv",
    skip_nct: bool = False,
    skip_atr: bool = False,
    skip_urls: bool = False,
):
    base_dir = os.path.dirname(input_file) or "."

    # ------------------------------------------------------------------
    # Step 1: Fix missing NCT IDs
    # ------------------------------------------------------------------
    if not skip_nct:
        print("\n" + "=" * 60)
        print("STEP 1: Fix missing NCT IDs")
        print("=" * 60)

        from scripts.fix_missing_nct import fix_missing_nct

        nct_output = os.path.join(base_dir, "enriched_nct_fixed.csv")
        fix_missing_nct(
            input_file=input_file,
            output_file=nct_output,
            dry_run=False,
            all_types=True,
        )
        step2_input = nct_output
    else:
        print("\n--- Skipping NCT fix ---")
        step2_input = input_file

    # ------------------------------------------------------------------
    # Step 2: Apply all improvements
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Apply improvements (quality, catalyst type, errors, ATR, URLs)")
    print("=" * 60)

    from scripts.fix_existing_data import fix_existing_data

    fix_existing_data(
        input_file=step2_input,
        output_file=output_file,
        skip_atr=skip_atr,
        skip_urls=skip_urls,
    )

    # Clean up intermediate file
    if not skip_nct and os.path.exists(nct_output) and nct_output != output_file:
        os.remove(nct_output)

    # ------------------------------------------------------------------
    # Step 3: Verify
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    df = pd.read_csv(output_file)
    clinical = df[df["catalyst_type"] == "Clinical Data"]
    has_nct = (
        clinical["nct_id"].notna()
        & (clinical["nct_id"] != "")
        & clinical["nct_id"].astype(str).str.startswith("NCT")
    ).sum()

    has_pr_url = (
        df["press_release_url"].notna() & (df["press_release_url"] != "")
    ).sum()

    print(f"Total rows:                {len(df)}")
    print(f"Clinical Data rows:        {len(clinical)}")
    print(f"  With NCT ID:            {has_nct}/{len(clinical)}")
    print(f"  Missing NCT ID:         {len(clinical) - has_nct}")
    print(f"Press release URLs:        {has_pr_url}/{len(df)}")
    print(f"Quality threshold passed:  {df.get('data_quality_threshold_passed', pd.Series(dtype=bool)).sum()}/{len(df)}")

    if "atr_pct" in df.columns:
        has_atr = df["atr_pct"].notna().sum()
        print(f"ATR computed:              {has_atr}/{len(df)}")
        mag = df["move_magnitude"].value_counts().to_dict()
        print(f"Move magnitude:            {mag}")

    print(f"\nCatalyst type distribution:")
    type_counts = df["catalyst_type"].apply(
        lambda x: "Other: ..." if str(x).startswith("Other:") else x
    ).value_counts()
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    # Show remaining missing NCT IDs
    missing_nct = clinical[
        clinical["nct_id"].isna() | (clinical["nct_id"] == "")
    ]
    if len(missing_nct) > 0:
        print(f"\nStill missing NCT IDs ({len(missing_nct)} rows):")
        for _, r in missing_nct.iterrows():
            drug = str(r.get("drug_name", "")) if pd.notna(r.get("drug_name")) else "(none)"
            print(f"  {r['ticker']:6s} | {drug[:40]:40s} | {str(r.get('phase', ''))}")
        print(f"\nSee nct_search_trace.json for detailed search logs.")

    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full pipeline fix")
    parser.add_argument("--input", default="enriched_new_batch.csv")
    parser.add_argument("--output", default="enriched_final.csv")
    parser.add_argument("--skip-nct", action="store_true")
    parser.add_argument("--skip-atr", action="store_true")
    parser.add_argument("--skip-urls", action="store_true", help="Skip press release URL finding")
    args = parser.parse_args()

    run_full_pipeline(
        input_file=args.input,
        output_file=args.output,
        skip_nct=args.skip_nct,
        skip_atr=args.skip_atr,
        skip_urls=args.skip_urls,
    )

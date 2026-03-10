"""
Re-run NCT ID search for rows with missing nct_id.
Saves detailed JSON trace log for every search.

Usage:
    python -m scripts.fix_missing_nct
    python -m scripts.fix_missing_nct --input enriched_new_batch.csv --dry-run
    python -m scripts.fix_missing_nct --all-types  # Search all rows, not just Clinical Data
"""

import json
import os
import re
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.clinicaltrials_client import ClinicalTrialsClient


def extract_design_keywords(summary: str) -> list:
    """Extract study-design keywords from a catalyst summary."""
    if not summary or pd.isna(summary):
        return []

    keywords = []
    summary_lower = str(summary).lower()

    design_terms = [
        "randomized", "placebo-controlled", "double-blind", "open-label",
        "single-arm", "phase 1", "phase 2", "phase 3", "pivotal",
        "dose-escalation", "dose-expansion", "registrational",
    ]
    for term in design_terms:
        if term in summary_lower:
            keywords.append(term)

    # Extract trial name pattern (e.g., "SAPPHIRE study", "VENTURE trial")
    trial_match = re.search(r'\b([A-Z]{4,10})\s*(?:study|trial)\b', str(summary), re.I)
    if trial_match:
        keywords.append(trial_match.group(1))

    return keywords


def fix_missing_nct(
    input_file: str = "enriched_new_batch.csv",
    output_file: str = None,
    dry_run: bool = False,
    all_types: bool = False,
):
    """Find and fill missing NCT IDs. Saves JSON trace log."""
    df = pd.read_csv(input_file)

    # Identify rows with missing nct_id
    if all_types:
        # Search any row that has a drug_name but no NCT ID
        has_drug = df["drug_name"].notna() & (df["drug_name"] != "") & (df["drug_name"] != "nan")
        missing_mask = (df["nct_id"].isna() | (df["nct_id"] == "")) & has_drug
        # Also include Clinical Data rows with no drug (we'll use ticker fallbacks)
        clinical_no_drug = (df["catalyst_type"] == "Clinical Data") & (df["nct_id"].isna() | (df["nct_id"] == ""))
        missing_mask = missing_mask | clinical_no_drug
    else:
        clinical_mask = df["catalyst_type"] == "Clinical Data"
        missing_mask = clinical_mask & (df["nct_id"].isna() | (df["nct_id"] == ""))

    missing = df[missing_mask]

    total_clinical = (df["catalyst_type"] == "Clinical Data").sum()
    has_nct = df["nct_id"].notna() & (df["nct_id"] != "") & df["nct_id"].astype(str).str.startswith("NCT")

    print(f"Total rows: {len(df)}")
    print(f"Clinical Data rows: {total_clinical}")
    print(f"Rows with NCT ID: {has_nct.sum()}")
    print(f"Rows to search: {len(missing)}")

    if missing.empty:
        print("Nothing to fix!")
        return

    client = ClinicalTrialsClient(rate_limit=0.5)
    detailed_logs = []
    found_count = 0

    for i, (idx, row) in enumerate(missing.iterrows()):
        ticker = row["ticker"]
        drug = str(row["drug_name"]) if pd.notna(row.get("drug_name")) else ""
        indication = str(row["indication"]) if pd.notna(row.get("indication")) else ""
        phase = str(row["phase"]) if pd.notna(row.get("phase")) else ""
        summary = str(row["catalyst_summary"]) if pd.notna(row.get("catalyst_summary")) else ""
        sponsor = str(row["ct_sponsor"]) if pd.notna(row.get("ct_sponsor")) else ""

        design_kw = extract_design_keywords(summary)

        print(f"\n[{i+1}/{len(missing)}] {ticker:6s} | {drug[:40] if drug != 'nan' else '(no drug)'}")

        # Clean up nan strings
        drug_clean = drug if drug and drug != "nan" else ""
        indication_clean = indication if indication != "nan" else None
        phase_clean = phase if phase != "nan" else None
        sponsor_clean = sponsor if sponsor != "nan" else None

        nct_id, log = client.search_nct_prioritized(
            drug_name=drug_clean,
            indication=indication_clean,
            phase=phase_clean,
            study_design_keywords=design_kw or None,
            sponsor=sponsor_clean,
            ticker=ticker,
        )

        detailed_logs.append({
            "ticker": ticker,
            "event_date": str(row.get("event_date", "")),
            "catalyst_type": str(row.get("catalyst_type", "")),
            "drug_name_input": drug,
            "nct_id_found": nct_id,
            **log,
        })

        if nct_id:
            found_count += 1
            print(f"  + Found: {nct_id}  ({log.get('result')})")
            if not dry_run:
                df.loc[idx, "nct_id"] = nct_id
                # Backfill ct_ columns from trial details
                trial = client.fetch_trial_details(nct_id)
                if trial:
                    ct_fields = {
                        "ct_official_title": trial.official_title,
                        "ct_phase": trial.phase,
                        "ct_enrollment": trial.enrollment,
                        "ct_conditions": trial.conditions,
                        "ct_status": trial.status,
                        "ct_sponsor": trial.sponsor,
                        "ct_allocation": trial.allocation,
                        "ct_primary_completion": trial.primary_completion_date,
                    }
                    for col, val in ct_fields.items():
                        if pd.isna(row.get(col)) or str(row.get(col)) in ("", "nan"):
                            df.loc[idx, col] = val
        else:
            reason = log.get("rejection_reason", log.get("result", "unknown"))
            print(f"  x Not found: {reason}")

    # Save detailed JSON trace log
    base_dir = os.path.dirname(input_file) or "."
    json_log_file = os.path.join(base_dir, "nct_search_trace.json")
    with open(json_log_file, "w") as f:
        json.dump(detailed_logs, f, indent=2, default=str)
    print(f"\nDetailed trace log: {json_log_file}")

    # Save summary CSV log too
    summary_rows = []
    for entry in detailed_logs:
        summary_rows.append({
            "ticker": entry["ticker"],
            "event_date": entry["event_date"],
            "drug_name": entry["drug_name_input"],
            "nct_id_found": entry["nct_id_found"],
            "result": entry["result"],
            "rejection_reason": entry.get("rejection_reason", ""),
            "queries_tried": len(entry.get("search_queries", [])),
            "final_candidates": len(entry.get("final_candidates", [])),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(base_dir, "data", "nct_search_results.csv"), index=False
    )

    # Save updated CSV
    if not dry_run:
        out = output_file or input_file.replace(".csv", "_nct_fixed.csv")
        df.to_csv(out, index=False)
        print(f"Updated CSV saved: {out}")

    # Final report
    total_nct_after = (
        df["nct_id"].notna() & (df["nct_id"] != "") & df["nct_id"].astype(str).str.startswith("NCT")
    ).sum()
    print(f"\n{'='*50}")
    print(f"RESULTS: Found {found_count}/{len(missing)} new NCT IDs")
    print(f"Total NCT IDs now: {total_nct_after}/{len(df)} rows")
    if dry_run:
        print("(dry run - no files modified)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix missing NCT IDs")
    parser.add_argument("--input", default="enriched_new_batch.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-types", action="store_true",
                        help="Search all rows with drug names, not just Clinical Data")
    args = parser.parse_args()

    fix_missing_nct(
        input_file=args.input,
        output_file=args.output,
        dry_run=args.dry_run,
        all_types=args.all_types,
    )

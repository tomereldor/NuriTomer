"""
Enrich low-move clinical events with AI-researched fields via Perplexity.

Fills: indication, phase, is_pivotal, pivotal_evidence,
       primary_endpoint_met, primary_endpoint_result

Usage:
    export PERPLEXITY_API_KEY="..."
    python3 -u scripts/enrich_clinical_fields.py --input low_move_enriched.csv
"""

import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


def research_clinical_event(ticker: str, date: str, drug: str, summary: str, ct_phase: str = "", ct_conditions: str = "") -> dict:
    """Ask Perplexity for clinical trial details about this event."""

    context = f"Drug: {drug}" if drug else ""
    if ct_phase:
        context += f", Trial phase from ClinicalTrials.gov: {ct_phase}"
    if ct_conditions:
        context += f", Conditions: {ct_conditions}"

    prompt = f"""For the biotech stock {ticker} around {date}, there was clinical trial data released.
{context}
Summary: {summary}

Answer in JSON only with these fields:
{{
  "indication": "the specific disease/condition being treated (e.g. 'non-small cell lung cancer', 'atopic dermatitis')",
  "phase": "trial phase as written (e.g. 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 1/2', 'Phase 2/3')",
  "is_pivotal": true or false (is this a pivotal/registrational trial that could lead to FDA approval?),
  "pivotal_evidence": "brief reason if pivotal, or null",
  "primary_endpoint_met": "Yes, No, Unclear, or Mixed",
  "primary_endpoint_result": "one sentence describing the key result (e.g. 'Met primary endpoint with 45% ORR vs 20% for control')"
}}"""

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            return {"_error": f"HTTP {resp.status_code}"}

        content = resp.json()["choices"][0]["message"]["content"]
        # Strip markdown fences
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except Exception:
            return {"_error": "parse_error"}
    except Exception as e:
        return {"_error": str(e)[:80]}


def enrich_clinical_fields(
    input_file: str = "low_move_enriched.csv",
    output_file: str = None,
    delay: float = 1.0,
    save_every: int = 20,
):
    if output_file is None:
        output_file = input_file

    if not PERPLEXITY_API_KEY:
        print("ERROR: PERPLEXITY_API_KEY not set")
        sys.exit(1)

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Fields to fill
    target_fields = ["indication", "phase", "is_pivotal", "pivotal_evidence",
                     "primary_endpoint_met", "primary_endpoint_result"]

    # Ensure columns exist
    for col in target_fields:
        if col not in df.columns:
            df[col] = None

    # Find rows needing enrichment (missing indication OR phase)
    needs_enrichment = df["indication"].isna() | (df["indication"].astype(str).isin(["", "nan"]))
    indices = df[needs_enrichment].index.tolist()

    print(f"Rows needing enrichment: {len(indices)}/{len(df)}")

    filled = 0
    errors = 0

    for i, idx in enumerate(indices):
        row = df.loc[idx]
        ticker = row["ticker"]
        date = str(row["event_date"])
        drug = str(row.get("drug_name", "")) if pd.notna(row.get("drug_name")) else ""
        summary = str(row.get("catalyst_summary", "")) if pd.notna(row.get("catalyst_summary")) else ""
        ct_phase = str(row.get("ct_phase", "")) if pd.notna(row.get("ct_phase")) else ""
        ct_conditions = str(row.get("ct_conditions", "")) if pd.notna(row.get("ct_conditions")) else ""

        print(f"[{i+1}/{len(indices)}] {ticker:6s} {date} {drug[:30]:30s}", end=" ", flush=True)

        result = research_clinical_event(ticker, date, drug, summary, ct_phase, ct_conditions)

        if "_error" in result:
            print(f"ERR: {result['_error'][:40]}")
            errors += 1
        else:
            for field in target_fields:
                val = result.get(field)
                if val is not None and str(val).lower() != "null":
                    df.at[idx, field] = val
            phase_str = result.get("phase", "?")
            endpoint = result.get("primary_endpoint_met", "?")
            print(f"{phase_str} | endpoint={endpoint}")
            filled += 1

        if (i + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
            print(f"  [saved progress to {output_file}]")

        time.sleep(delay)

    # Final save
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"ENRICHMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Filled:  {filled}/{len(indices)}")
    print(f"Errors:  {errors}/{len(indices)}")
    for col in target_fields:
        non_empty = ((df[col].notna()) & (df[col].astype(str) != "") & (df[col].astype(str) != "nan")).sum()
        print(f"  {col:30s}  {non_empty:3d}/{len(df)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich clinical fields via Perplexity")
    parser.add_argument("--input", default="low_move_enriched.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    enrich_clinical_fields(
        input_file=args.input,
        output_file=args.output,
        delay=args.delay,
    )

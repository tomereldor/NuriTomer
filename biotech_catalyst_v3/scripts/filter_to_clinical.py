"""Filter low-move candidates to Clinical Data events via Perplexity."""

import pandas as pd
import requests
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


def check_if_clinical(ticker: str, date: str) -> dict:
    """Ask Perplexity if this was a clinical data event."""

    prompt = f"""Was there clinical trial data or results announced for {ticker} on or around {date}?

Answer in JSON only:
{{"is_clinical": true/false, "drug_name": "name or null", "summary": "one sentence or null"}}"""

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=45
        )
    except requests.exceptions.Timeout:
        return {"is_clinical": False, "error": "timeout"}
    except requests.exceptions.RequestException as e:
        return {"is_clinical": False, "error": f"request_error: {str(e)[:60]}"}

    if resp.status_code != 200:
        return {"is_clinical": False, "error": resp.status_code}

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception:
        return {"is_clinical": False, "error": "parse_error"}


def filter_clinical_events(input_file: str, output_file: str, target_count: int = 120):
    """Filter candidates to confirmed Clinical Data events."""

    if not PERPLEXITY_API_KEY:
        print("ERROR: PERPLEXITY_API_KEY environment variable not set")
        sys.exit(1)

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} candidates from {input_file}")

    # Resume from partial save if it exists
    partial_file = output_file.replace(".csv", "_partial.csv")
    clinical_events = []
    if os.path.exists(partial_file):
        existing = pd.read_csv(partial_file)
        clinical_events = existing.to_dict("records")
        print(f"Resumed from partial save: {len(clinical_events)} events already found")

    date_col = 'event_date' if 'event_date' in df.columns else 'date'

    for i, (_, row) in enumerate(df.iterrows()):
        if len(clinical_events) >= target_count:
            break

        print(f"[{i+1}/{len(df)}] {row['ticker']} {row[date_col]}...", end=" ", flush=True)

        result = check_if_clinical(row['ticker'], row[date_col])

        if result.get('error'):
            print(f"ERROR ({result['error']})", flush=True)
        elif result.get('is_clinical'):
            row_dict = row.to_dict()
            row_dict['drug_name'] = result.get('drug_name')
            row_dict['catalyst_summary'] = result.get('summary')
            row_dict['catalyst_type'] = 'Clinical Data'
            clinical_events.append(row_dict)
            print(f"CLINICAL ({result.get('drug_name', 'unknown')})", flush=True)
            # Partial save every 10 clinical hits
            if len(clinical_events) % 10 == 0:
                pd.DataFrame(clinical_events).to_csv(partial_file, index=False)
        else:
            print("Not clinical", flush=True)

        time.sleep(1.0)

    result_df = pd.DataFrame(clinical_events)
    result_df.to_csv(output_file, index=False)
    if os.path.exists(partial_file):
        os.remove(partial_file)

    print(f"\nFound {len(result_df)} Clinical Data events")
    print(f"Saved to {output_file}")
    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter low-move candidates to Clinical Data")
    parser.add_argument("--input", default="low_move_candidates.csv")
    parser.add_argument("--output", default="low_move_clinical.csv")
    parser.add_argument("--target-count", type=int, default=120)
    args = parser.parse_args()

    filter_clinical_events(
        input_file=args.input,
        output_file=args.output,
        target_count=args.target_count,
    )

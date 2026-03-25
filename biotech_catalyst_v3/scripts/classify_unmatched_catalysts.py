"""Phase 4 (A2 cont.): Classify unmatched large moves via Perplexity.

For large biotech stock moves that didn't match a CT.gov completion date,
ask Perplexity to identify the catalyst type (clinical_trial, fda_decision,
partnership, earnings, other).

Keep: clinical_trial, fda_decision (both are clinical/regulatory events)
Drop: partnership, earnings, other

Caches results in cache/catalyst_classification_cache_v1.json for resume.

Usage:
    python -m scripts.classify_unmatched_catalysts
    python -m scripts.classify_unmatched_catalysts --max-events 500 --min-normalized 3.0
    python -m scripts.classify_unmatched_catalysts --dry-run  # show plan, no API calls
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".env")
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL     = "https://api.perplexity.ai/chat/completions"
MODEL              = "sonar"
BATCH_SIZE         = 5    # events per Perplexity call
CACHE_FILE         = "cache/catalyst_classification_cache_v1.json"

CLINICAL_TYPES = {"clinical_trial", "fda_decision"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache(cache_file: str) -> dict:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_file: str):
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def cache_key(ticker: str, event_date: str) -> str:
    return f"{ticker}|{event_date}"


# ---------------------------------------------------------------------------
# Perplexity call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a biotech event analyst. Given a list of biotech stock move events "
    "(ticker, date, move%), identify the most likely cause for each from public news. "
    "Respond ONLY with valid JSON, no markdown, no preamble."
)


def build_batch_prompt(events: list) -> str:
    lines = []
    for i, e in enumerate(events):
        lines.append(
            f"{i+1}. {e['ticker']} moved {e['move_pct']:+.1f}% on {e['event_date']}"
        )

    events_text = "\n".join(lines)

    return f"""For each biotech stock event below, identify the catalyst from public news sources.

{events_text}

For each event, return a JSON array with one object per event:
{{
  "ticker": "<TICKER>",
  "event_date": "<DATE>",
  "catalyst_type": "<clinical_trial|fda_decision|partnership|earnings|other>",
  "drug": "<drug name or empty string>",
  "indication": "<disease/condition or empty string>",
  "phase": "<Phase 1/2/3/NDA/BLA or empty string>",
  "confidence": "<high|medium|low>"
}}

Only use these catalyst_type values: clinical_trial, fda_decision, partnership, earnings, other.
Return a JSON array of {len(events)} objects."""


def call_perplexity_batch(events: list, max_retries: int = 3) -> list:
    """Call Perplexity for a batch of events. Returns list of classification dicts."""
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY not set in environment")

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(events)},
        ],
        "temperature": 0.0,
        "max_tokens": 1000,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  Perplexity HTTP {resp.status_code}: {resp.text[:120]}")
                return []

            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")

            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            # Sometimes wrapped in {"results": [...]}
            for key in ("results", "events", "data"):
                if key in parsed:
                    return parsed[key]
            return []

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}", flush=True)
            if attempt == max_retries - 1:
                return []
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}", flush=True)
            if attempt == max_retries - 1:
                return []
        time.sleep(2)

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def classify_unmatched(
    unmatched_file: str = "data/unmatched_large_moves_2018_2022.csv",
    output_file: str = "data/perplexity_confirmed_2018_2022.csv",
    cache_file: str = CACHE_FILE,
    min_normalized: float = 3.0,
    max_events: int = 2000,
    batch_size: int = BATCH_SIZE,
    dry_run: bool = False,
):
    # ------------------------------------------------------------------
    # Load and filter candidates
    # ------------------------------------------------------------------
    df = pd.read_csv(unmatched_file)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y-%m-%d")

    # Filter to high-quality candidates
    candidates = df[
        (df["normalized_move"] >= min_normalized) &
        df["normalized_move"].notna()
    ].copy()

    # Sort by |normalized_move| descending — highest-quality events first
    candidates = candidates.sort_values("normalized_move", ascending=False)

    if len(candidates) > max_events:
        candidates = candidates.head(max_events)

    print(f"Candidates (normalized >= {min_normalized}): {len(candidates):,}")
    print(f"  Max events: {max_events:,}")
    print(f"  Will use:   {len(candidates):,}")

    if dry_run:
        print(f"\n[DRY RUN] Would make ~{len(candidates) // batch_size + 1} Perplexity calls")
        print(f"Example batch prompt:")
        sample = candidates.head(batch_size).to_dict("records")
        print(build_batch_prompt(sample))
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Load cache
    # ------------------------------------------------------------------
    cache = load_cache(cache_file)
    cached_keys = {k for k in cache}
    print(f"Cache: {len(cache):,} events already classified")

    # Build list of events needing classification
    to_classify = [
        r.to_dict()
        for _, r in candidates.iterrows()
        if cache_key(r["ticker"], r["event_date"]) not in cached_keys
    ]
    print(f"To classify: {len(to_classify):,}")

    if not to_classify:
        print("All candidates already cached.")
    else:
        print(f"\nClassifying in batches of {batch_size}...", flush=True)
        total_calls = 0
        clinical_found = 0

        for i in range(0, len(to_classify), batch_size):
            batch = to_classify[i : i + batch_size]

            if (i // batch_size + 1) % 10 == 0:
                print(
                    f"  Batch {i//batch_size+1}/{len(to_classify)//batch_size+1} | "
                    f"clinical found so far: {clinical_found}",
                    flush=True,
                )

            results = call_perplexity_batch(batch)
            total_calls += 1

            # Map results back by ticker+date
            result_map = {
                cache_key(r.get("ticker", ""), r.get("event_date", "")): r
                for r in results
                if isinstance(r, dict)
            }

            for batch_idx, event in enumerate(batch):
                key = cache_key(event["ticker"], event["event_date"])
                if key in result_map:
                    classification = result_map[key]
                else:
                    # Fallback: try positional match
                    if batch_idx < len(results) and isinstance(results[batch_idx], dict):
                        classification = results[batch_idx]
                    else:
                        classification = {"catalyst_type": "error", "confidence": "low"}

                cache[key] = {
                    "ticker": event["ticker"],
                    "event_date": event["event_date"],
                    "move_pct": event["move_pct"],
                    "atr_pct": event.get("atr_pct"),
                    "normalized_move": event.get("normalized_move"),
                    "market_cap_m": event.get("market_cap_m"),
                    **classification,
                }

                if classification.get("catalyst_type") in CLINICAL_TYPES:
                    clinical_found += 1

            save_cache(cache, cache_file)
            time.sleep(1.5)  # ~40 calls/min, well within Perplexity rate limits

        print(f"\nClassification complete: {total_calls} API calls made")

    # ------------------------------------------------------------------
    # Build output from cache (for all candidates, not just new ones)
    # ------------------------------------------------------------------
    candidate_keys = {
        cache_key(r["ticker"], r["event_date"])
        for _, r in candidates.iterrows()
    }

    rows = []
    for key, result in cache.items():
        if key in candidate_keys:
            rows.append(result)

    df_all = pd.DataFrame(rows)

    if df_all.empty:
        print("No results in cache for candidates.")
        return df_all

    # Filter to clinical catalysts only
    df_clinical = df_all[df_all["catalyst_type"].isin(CLINICAL_TYPES)].copy()

    # Summary
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total classified: {len(df_all):,}")
    print(f"\nCatalyst type distribution:")
    print(df_all["catalyst_type"].value_counts().to_string())
    print(f"\nClinical catalysts confirmed: {len(df_clinical):,}")

    if len(df_clinical) > 0:
        clinical_rate = len(df_clinical) / len(df_all) * 100
        print(f"Clinical rate: {clinical_rate:.1f}%")
        print(f"\nTop 10 clinical catalysts (by move_pct):")
        print(
            df_clinical.nlargest(10, "move_pct")[
                ["ticker", "event_date", "move_pct", "normalized_move",
                 "catalyst_type", "drug", "indication", "phase"]
            ].to_string(index=False)
        )

    # Save confirmed clinical catalysts
    df_clinical.to_csv(output_file, index=False)
    print(f"\nSaved {len(df_clinical)} confirmed clinical events to: {output_file}")

    return df_clinical


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify unmatched large moves via Perplexity")
    parser.add_argument("--input",          default="data/unmatched_large_moves_2018_2022.csv")
    parser.add_argument("--output",         default="data/perplexity_confirmed_2018_2022.csv")
    parser.add_argument("--cache",          default=CACHE_FILE)
    parser.add_argument("--min-normalized", type=float, default=3.0)
    parser.add_argument("--max-events",     type=int,   default=2000)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()

    classify_unmatched(
        unmatched_file=args.input,
        output_file=args.output,
        cache_file=args.cache,
        min_normalized=args.min_normalized,
        max_events=args.max_events,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

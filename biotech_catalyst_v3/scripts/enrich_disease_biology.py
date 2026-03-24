"""
enrich_disease_biology.py
==========================
LLM-derived disease biology features from indication text.

Uses Perplexity (sonar model) to classify each unique indication into three
biology dimensions that influence clinical trial success rates:

  1. has_predictive_biomarker (bool)  — routine biomarker guides treatment selection
  2. genetic_basis (str)             — none / monogenic / polygenic / somatic
  3. targeted_therapy_exists (bool)  — approved or late-stage targeted therapy exists

These are static medical knowledge features — inherent to the disease, not the
trial or event — and are pre-event safe (disease biology doesn't change with
the event date).

Output columns written to master CSV:
  disease_has_predictive_biomarker  (0/1)
  disease_genetic_basis             (none/monogenic/polygenic/somatic)
  disease_targeted_therapy_exists   (0/1)

Cache: cache/disease_biology_v1.json — resume-safe, saves after each batch.

Usage (from biotech_catalyst_v3/):
    # Test with 2 batches (20 indications)
    python -m scripts.enrich_disease_biology --limit 20

    # Full run (~115 batches, ~2-3 min)
    python -m scripts.enrich_disease_biology

    # Dry run — show unique indications and cache status
    python -m scripts.enrich_disease_biology --dry-run
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# .env loader (reused from validate_catalysts.py)
# ============================================================================

def _load_dotenv() -> None:
    """Load key=value pairs from .env at repo root into os.environ."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(repo_root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar"  # cheaper than sonar-pro; sufficient for factual medical knowledge

BATCH_SIZE = 10
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MASTER_CSV = os.path.join(BASE_DIR, "enriched_all_clinical_clean_v3.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "disease_biology_v1.json")

VALID_GENETIC_BASIS = {"none", "monogenic", "polygenic", "somatic"}

# Columns written to master CSV
OUT_COLS = [
    "disease_has_predictive_biomarker",
    "disease_genetic_basis",
    "disease_targeted_therapy_exists",
]


# ============================================================================
# Cache management
# ============================================================================

def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ============================================================================
# Perplexity API
# ============================================================================

SYSTEM_PROMPT = (
    "You are a medical knowledge assistant. Classify diseases based on "
    "established medical and scientific knowledge. Always respond with "
    "valid JSON only, no markdown, no preamble."
)

CLASSIFICATION_PROMPT = """For each disease/condition listed below, provide three classifications based on
established medical/scientific knowledge:

1. has_predictive_biomarker (true/false): Is there a known predictive biomarker
   routinely used to guide treatment selection for this disease?
   - true examples: HER2 in breast cancer, PD-L1 in NSCLC, BRCA in ovarian cancer, CFTR in cystic fibrosis
   - false examples: COVID-19, obesity, major depressive disorder, osteoarthritis

2. genetic_basis (string): What is the primary genetic basis of this disease?
   - "none": primarily infectious, environmental, or idiopathic (e.g. COVID-19, trauma)
   - "monogenic": caused by mutations in a single gene (e.g. cystic fibrosis, sickle cell, Huntington's)
   - "polygenic": influenced by multiple genes (e.g. type 2 diabetes, schizophrenia, Alzheimer's)
   - "somatic": caused by acquired somatic mutations (e.g. most cancers, myelodysplastic syndromes)

3. targeted_therapy_exists (true/false): Is there at least one approved or late-stage (Phase 3+)
   therapy that targets a specific validated molecular target (gene/protein/pathway) for this disease?
   - true examples: CML (imatinib->BCR-ABL), breast cancer (trastuzumab->HER2), CF (ivacaftor->CFTR)
   - false examples: major depressive disorder, chronic pain, obesity (prior to GLP-1 era)

Respond with ONLY a valid JSON array, one object per disease, in the same order as listed:
[{{"disease": "...", "has_predictive_biomarker": true/false,
  "genetic_basis": "none|monogenic|polygenic|somatic",
  "targeted_therapy_exists": true/false}}, ...]

Diseases:
{numbered_list}"""


def call_perplexity(prompt: str) -> tuple:
    """Call Perplexity API. Returns (parsed_json, error_string)."""
    if not PERPLEXITY_API_KEY:
        return None, "PERPLEXITY_API_KEY not set in environment"

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(PERPLEXITY_URL, headers=headers,
                                 json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text[:120]}"

            content = (resp.json()
                       .get("choices", [{}])[0]
                       .get("message", {})
                       .get("content", ""))
            if not content:
                return None, "Empty response"

            # Strip markdown fences
            content = content.strip()
            for fence in ("```json", "```"):
                if content.startswith(fence):
                    content = content[len(fence):]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                return json.loads(content), ""
            except json.JSONDecodeError as e:
                return None, f"JSON parse error: {e}\nContent: {content[:200]}"

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None, "Timeout"
        except Exception as e:
            return None, str(e)[:120]

    return None, "Max retries exceeded"


def classify_batch(indications: list) -> tuple:
    """Classify a batch of indications. Returns (results_dict, error_string).

    results_dict maps normalized indication -> {has_predictive_biomarker, genetic_basis, targeted_therapy_exists}
    """
    numbered = "\n".join(f"{i+1}. {ind}" for i, ind in enumerate(indications))
    prompt = CLASSIFICATION_PROMPT.format(numbered_list=numbered)

    parsed, error = call_perplexity(prompt)
    if error:
        return {}, error
    if not isinstance(parsed, list):
        return {}, f"Expected JSON array, got {type(parsed).__name__}"

    results = {}
    for i, item in enumerate(parsed):
        if i >= len(indications):
            break
        ind = indications[i]

        biomarker = bool(item.get("has_predictive_biomarker", False))
        genetic = str(item.get("genetic_basis", "none")).lower().strip()
        targeted = bool(item.get("targeted_therapy_exists", False))

        # Validate genetic_basis
        if genetic not in VALID_GENETIC_BASIS:
            genetic = "none"

        results[ind] = {
            "has_predictive_biomarker": biomarker,
            "genetic_basis": genetic,
            "targeted_therapy_exists": targeted,
        }

    return results, ""


# ============================================================================
# Main pipeline
# ============================================================================

def run(limit: int = None, dry_run: bool = False):
    print(f"Loading {MASTER_CSV} ...")
    df = pd.read_csv(MASTER_CSV)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    # Extract and normalize indications
    ind_raw = df["indication"].dropna().str.strip().str.lower()
    unique_inds = sorted(ind_raw.unique())
    print(f"  Indications: {ind_raw.count()}/{len(df)} non-null, "
          f"{len(unique_inds)} unique (case-insensitive)")

    # Load cache
    cache = load_cache()
    cached_count = sum(1 for ind in unique_inds if ind in cache)
    uncached = [ind for ind in unique_inds if ind not in cache]
    print(f"  Cache: {cached_count} cached, {len(uncached)} remaining")

    if dry_run:
        print(f"\n[DRY RUN] Would classify {len(uncached)} indications "
              f"in {(len(uncached) + BATCH_SIZE - 1) // BATCH_SIZE} batches")
        if uncached:
            print(f"  First 10: {uncached[:10]}")
        return

    if limit:
        uncached = uncached[:limit]
        print(f"  Limited to first {limit} uncached indications")

    if not uncached:
        print("\nAll indications already cached — skipping API calls.")
    else:
        n_batches = (len(uncached) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nClassifying {len(uncached)} indications in {n_batches} batches ...")

        errors = 0
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            batch = uncached[start:start + BATCH_SIZE]

            print(f"  Batch {batch_idx + 1}/{n_batches} ({len(batch)} indications) ...",
                  end=" ", flush=True)

            results, error = classify_batch(batch)
            if error:
                errors += 1
                print(f"ERROR: {error[:80]}")
            else:
                cache.update(results)
                save_cache(cache)
                print(f"OK ({len(results)} classified)")

            if batch_idx < n_batches - 1:
                time.sleep(RATE_LIMIT_DELAY)

        print(f"\nDone. Cache now has {len(cache)} entries. Errors: {errors}")

    # Map cache to master CSV
    print("\nWriting columns to master CSV ...")
    ind_norm = df["indication"].fillna("").str.strip().str.lower()

    df["disease_has_predictive_biomarker"] = ind_norm.map(
        lambda x: cache.get(x, {}).get("has_predictive_biomarker")
    ).astype("Int64")

    df["disease_genetic_basis"] = ind_norm.map(
        lambda x: cache.get(x, {}).get("genetic_basis")
    )

    df["disease_targeted_therapy_exists"] = ind_norm.map(
        lambda x: cache.get(x, {}).get("targeted_therapy_exists")
    ).astype("Int64")

    # Coverage report
    for col in OUT_COLS:
        nn = df[col].notna().sum()
        print(f"  {col}: {nn}/{len(df)} non-null ({nn/len(df)*100:.1f}%)")

    # Distribution for genetic_basis
    if df["disease_genetic_basis"].notna().any():
        print(f"\n  genetic_basis distribution:")
        dist = df["disease_genetic_basis"].value_counts(dropna=False)
        for val, cnt in dist.items():
            label = val if pd.notna(val) else "NaN"
            print(f"    {label:>10}: {cnt:>5} ({cnt/len(df)*100:.1f}%)")

    # Save
    df.to_csv(MASTER_CSV, index=False)
    print(f"\nSaved: {os.path.basename(MASTER_CSV)}  "
          f"({len(df)} rows × {len(df.columns)} cols)")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify disease biology features via Perplexity"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max indications to classify (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats and exit without API calls")
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run)

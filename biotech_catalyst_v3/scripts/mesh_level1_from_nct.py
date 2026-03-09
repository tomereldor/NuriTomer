"""
mesh_level1_from_nct.py
=======================
Append MeSH Level-1 disease category to a CSV that has an nct_id column.

Fetches from ClinicalTrials.gov API v2 (no auth required).

Output columns added
--------------------
  mesh_level1         Top-level MeSH branch name (e.g. "Neoplasms")
  mesh_level1_reason  How it was chosen (see reasons below)
  mesh_branches_raw   Pipe-joined list of all non-All branches found
  mesh_terms_raw      Pipe-joined MeSH term names
  ct_conditions_raw   Pipe-joined conditions from protocolSection

Reasons
-------
  single_branch    — exactly one non-All branch; used it directly
  condition_match  — multiple branches; one matched the condition text
  priority_list    — multiple branches; fell back to priority order
  no_branches      — API returned no usable branches
  no_nct_id        — row has no NCT ID
  fetch_error      — network / API failure

Usage
-----
  python mesh_level1_from_nct.py --input enriched_all_clinical_clean_v2.csv \\
                                  --output clean_v2_mesh.csv
"""

import argparse
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

# ── ClinicalTrials.gov v2 ──────────────────────────────────────────────────
CT_URL      = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
TIMEOUT     = 15
RETRY_DELAY = 4
MAX_RETRIES = 3

# ── Priority order when multiple branches exist ────────────────────────────
BRANCH_PRIORITY = [
    "Neoplasms",
    "Immune System Diseases",
    "Nervous System Diseases",
    "Cardiovascular Diseases",
    "Respiratory Tract Diseases",
    "Digestive System Diseases",
    "Endocrine System Diseases",
    "Skin Diseases",
    "Musculoskeletal Diseases",
    "Infectious Diseases",
]

# Simple keywords to match each branch against condition text
_BRANCH_KEYWORDS: Dict[str, list] = {
    "Neoplasms":                ["cancer", "tumor", "tumour", "carcinoma", "lymphoma",
                                 "leukemia", "leukaemia", "melanoma", "sarcoma", "neoplasm"],
    "Immune System Diseases":   ["autoimmune", "immune", "lupus", "rheumatoid", "psoriasis",
                                 "crohn", "colitis", "atopic", "allerg"],
    "Nervous System Diseases":  ["neuro", "alzheimer", "parkinson", "multiple sclerosis",
                                 "epilepsy", "stroke", "dementia", "als", "migraine"],
    "Cardiovascular Diseases":  ["heart", "cardiac", "coronary", "hypertension", "atrial",
                                 "vascular", "stroke", "cardiomyopathy"],
    "Respiratory Tract Diseases": ["lung", "respiratory", "asthma", "copd", "pulmonary",
                                    "bronch", "fibrosis"],
    "Digestive System Diseases":  ["liver", "hepat", "gastro", "intestin", "bowel",
                                    "colon", "pancrea", "gastrointestinal"],
    "Endocrine System Diseases":  ["diabet", "thyroid", "endocrine", "insulin", "obesity",
                                    "metabolic", "adrenal"],
    "Skin Diseases":              ["skin", "derma", "eczema", "acne", "rash", "psoriasis"],
    "Musculoskeletal Diseases":   ["muscle", "bone", "joint", "arthritis", "osteo",
                                    "spinal", "skeletal"],
    "Infectious Diseases":        ["infect", "virus", "viral", "bacteria", "fungal",
                                    "hiv", "hepatitis", "covid", "rsv"],
}


# ── API fetch with retry ───────────────────────────────────────────────────

def _fetch_study(nct_id: str) -> Optional[dict]:
    url = CT_URL.format(nct_id=nct_id.strip().upper())
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None  # NCT not found — treat as missing, not error
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return None
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return "ERROR"


# ── MeSH extraction ───────────────────────────────────────────────────────

def _extract_mesh(data: dict) -> Tuple[str, str, str, str, str]:
    """
    Returns (mesh_level1, reason, branches_raw, terms_raw, conditions_raw).

    API v2 uses:
      conditionBrowseModule.meshes    — direct MeSH terms
      conditionBrowseModule.ancestors — full hierarchy including top-level branches
    Branches are derived by intersecting ancestors with BRANCH_PRIORITY list.
    """
    derived   = data.get("derivedSection", {})
    protocol  = data.get("protocolSection", {})
    browse    = derived.get("conditionBrowseModule", {})

    # Raw conditions from protocolSection
    conditions: list = protocol.get("conditionsModule", {}).get("conditions", [])
    conditions_raw   = " | ".join(conditions)
    condition_text   = conditions_raw.lower()

    # Direct MeSH terms (e.g. "Lung Neoplasms")
    meshes: list = browse.get("meshes", [])
    terms_raw = " | ".join(m.get("term", "") for m in meshes if m.get("term"))

    # CT.gov uses the actual MeSH node names in ancestors, which sometimes
    # differ from our branch labels.  Map the known divergences.
    _ANCESTOR_ALIASES = {
        "Infections":                   "Infectious Diseases",
        "Heart Diseases":               "Cardiovascular Diseases",
        "Vascular Diseases":            "Cardiovascular Diseases",
        "Musculoskeletal and Connective Tissue Diseases": "Musculoskeletal Diseases",
        "Musculoskeletal Diseases":     "Musculoskeletal Diseases",
        "Gastrointestinal Diseases":    "Digestive System Diseases",
        "Hepatobiliary Diseases":       "Digestive System Diseases",
        "Nervous System Diseases":      "Nervous System Diseases",
    }

    # Derive branches: ancestor terms that map to our priority list
    ancestors: list = browse.get("ancestors", [])
    ancestor_names  = {a.get("term", "") for a in ancestors if a.get("term")}
    branch_set: set = set()
    for name in ancestor_names:
        # Direct match
        if name in BRANCH_PRIORITY:
            branch_set.add(name)
        # Alias match
        elif name in _ANCESTOR_ALIASES:
            branch_set.add(_ANCESTOR_ALIASES[name])
    # Preserve priority order
    branches     = [b for b in BRANCH_PRIORITY if b in branch_set]
    branches_raw = " | ".join(branches)

    if not branches:
        return "", "no_branches", branches_raw, terms_raw, conditions_raw

    if len(branches) == 1:
        return branches[0], "single_branch", branches_raw, terms_raw, conditions_raw

    # Multiple branches — try to match condition text first
    for branch in branches:
        keywords = _BRANCH_KEYWORDS.get(branch, [branch.lower()])
        if any(kw in condition_text for kw in keywords):
            return branch, "condition_match", branches_raw, terms_raw, conditions_raw

    # Fall back to priority list
    for priority_branch in BRANCH_PRIORITY:
        if priority_branch in branches:
            return priority_branch, "priority_list", branches_raw, terms_raw, conditions_raw

    # Use first branch if nothing matched
    return branches[0], "priority_list", branches_raw, terms_raw, conditions_raw


# ── Main ──────────────────────────────────────────────────────────────────

def run(
    input_file:  str,
    output_file: str,
    delay:       float = 0.4,
    save_every:  int   = 25,
    overwrite:   bool  = False,
) -> None:
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows from {input_file}")

    # Initialise output columns if missing
    for col in ["mesh_level1", "mesh_level1_reason", "mesh_branches_raw",
                "mesh_terms_raw", "ct_conditions_raw"]:
        if col not in df.columns:
            df[col] = ""

    # Determine rows to process
    if overwrite:
        target_mask = df["nct_id"].notna() & (df["nct_id"].astype(str).str.strip() != "")
    else:
        already_done = df["mesh_level1"].notna() & (df["mesh_level1"].astype(str).str.strip() != "")
        target_mask  = (
            df["nct_id"].notna()
            & (df["nct_id"].astype(str).str.strip() != "")
            & ~already_done
        )

    target_idx = df[target_mask].index.tolist()
    print(f"Rows with nct_id: {target_mask.sum():,} | "
          f"Already filled: {df['mesh_level1'].notna().sum():,} | "
          f"To fetch: {len(target_idx):,}")

    # Mark rows without nct_id
    no_nct = df["nct_id"].isna() | (df["nct_id"].astype(str).str.strip() == "")
    df.loc[no_nct & (df["mesh_level1_reason"] == ""), "mesh_level1_reason"] = "no_nct_id"

    # In-run cache keyed by nct_id
    cache: Dict[str, dict] = {}
    filled = 0
    errors = 0

    for i, idx in enumerate(target_idx):
        nct_id = str(df.at[idx, "nct_id"]).strip()

        if nct_id in cache:
            data = cache[nct_id]
        else:
            data = _fetch_study(nct_id)
            cache[nct_id] = data
            time.sleep(delay)

        print(f"[{i+1}/{len(target_idx)}] {nct_id} ...", end=" ", flush=True)

        if data == "ERROR" or data is None:
            df.at[idx, "mesh_level1_reason"] = "fetch_error" if data == "ERROR" else "no_nct_id"
            errors += 1
            print("fetch_error" if data == "ERROR" else "not_found")
            continue

        mesh, reason, branches_raw, terms_raw, conditions_raw = _extract_mesh(data)
        df.at[idx, "mesh_level1"]         = mesh
        df.at[idx, "mesh_level1_reason"]  = reason
        df.at[idx, "mesh_branches_raw"]   = branches_raw
        df.at[idx, "mesh_terms_raw"]      = terms_raw
        df.at[idx, "ct_conditions_raw"]   = conditions_raw

        if mesh:
            filled += 1
        print(f"{mesh or '(no branches)'}  [{reason}]")

        if (i + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
            print(f"  [checkpoint: {filled} filled, {errors} errors]")

    df.to_csv(output_file, index=False)

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"Fetched:      {len(target_idx):,}")
    print(f"Filled:       {filled:,}")
    print(f"No branches:  {(df['mesh_level1_reason']=='no_branches').sum():,}")
    print(f"Errors:       {errors:,}")
    print(f"No NCT ID:    {(df['mesh_level1_reason']=='no_nct_id').sum():,}")
    if filled:
        print(f"\nmesh_level1 distribution:")
        print(df["mesh_level1"].value_counts().to_string())
    print(f"\nOutput → {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append MeSH Level-1 from ClinicalTrials.gov")
    parser.add_argument("--input",      required=True,  help="Input CSV with nct_id column")
    parser.add_argument("--output",     required=True,  help="Output CSV path")
    parser.add_argument("--delay",      type=float, default=0.4,
                        help="Seconds between API calls (default: 0.4)")
    parser.add_argument("--save-every", type=int,   default=25,
                        help="Checkpoint frequency (default: 25)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-fetch even rows that already have mesh_level1")
    args = parser.parse_args()

    run(
        input_file  = args.input,
        output_file = args.output,
        delay       = args.delay,
        save_every  = args.save_every,
        overwrite   = args.overwrite,
    )

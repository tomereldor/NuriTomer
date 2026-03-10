#!/usr/bin/env python3
"""
recover_mesh.py
===============
Deterministic mesh_level1 recovery pass for ml_dataset_v1.csv.

Splits missing-mesh rows into two groups:

  Group A — rows WITH  nct_id  (75 rows)
  Group B — rows WITHOUT nct_id (46 rows)

Group A recovery hierarchy (per row, stops at first success):
  A1. mesh_terms_raw column  → exact branch name match
  A2. mesh_terms_raw column  → keyword match
  A3. ct_conditions / indication columns → keyword match
  A4. ClinicalTrials.gov API (fresh call)
        browseBranches  → first branch name
        meshes          → exact + keyword match on terms
        ancestors       → exact + keyword match on terms
        conditions      → NLM MeSH Lookup on each condition term

Group B recovery hierarchy (per row, stops at first success):
  B1. catalyst_summary    → keyword match
  B2. catalyst_summary    → extract disease terms → NLM MeSH Lookup
  B3. drug_name           → NLM MeSH Lookup

Output columns added:
  mesh_recovery_method  — which step resolved the row
  mesh_source_term      — the term that was matched or queried

Usage:
    python3 recover_mesh.py
    python3 recover_mesh.py --input ml_dataset_v1.csv --output ml_dataset_mesh_recovered.csv
    python3 recover_mesh.py --skip-api   # use only local columns + keyword matching
"""

import argparse
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN  = os.path.join(SCRIPT_DIR, "ml_dataset_v1.csv")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "ml_dataset_mesh_recovered.csv")

# ── API ────────────────────────────────────────────────────────────────────────
CT_URL     = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
NLM_SEARCH = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
NLM_DETAIL = "https://id.nlm.nih.gov/mesh/{uid}.json"
TIMEOUT     = 15
RETRY_DELAY = 4
MAX_RETRIES = 3
API_DELAY   = 0.35   # seconds between outbound calls

# ── MeSH branch definitions ────────────────────────────────────────────────────
BRANCH_PRIORITY: List[str] = [
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

BRANCH_SET = set(BRANCH_PRIORITY)

# Aliases used by CT.gov ancestor terms that differ from our branch labels
ANCESTOR_ALIASES: Dict[str, str] = {
    "Infections":                                     "Infectious Diseases",
    "Heart Diseases":                                 "Cardiovascular Diseases",
    "Vascular Diseases":                              "Cardiovascular Diseases",
    "Musculoskeletal and Connective Tissue Diseases": "Musculoskeletal Diseases",
    "Musculoskeletal Diseases":                       "Musculoskeletal Diseases",
    "Gastrointestinal Diseases":                      "Digestive System Diseases",
    "Hepatobiliary Diseases":                         "Digestive System Diseases",
    "Nervous System Diseases":                        "Nervous System Diseases",
    "Mental Disorders":                               "Nervous System Diseases",
    "Hemic and Lymphatic Diseases":                   "Immune System Diseases",
    "Immune System Diseases":                         "Immune System Diseases",
    "Neoplasms":                                      "Neoplasms",
    "Skin and Connective Tissue Diseases":            "Skin Diseases",
    "Nutritional and Metabolic Diseases":             "Endocrine System Diseases",
    "Endocrine System Diseases":                      "Endocrine System Diseases",
}

# Disease-keyword → branch (used for text matching — ordered by priority)
KEYWORDS: Dict[str, List[str]] = {
    "Neoplasms": [
        "cancer", "tumor", "tumour", "carcinoma", "lymphoma", "leukemia",
        "leukaemia", "melanoma", "sarcoma", "neoplasm", "glioma", "myeloma",
        "adenocarcinoma", "blastoma", "malignant", "oncolog",
    ],
    "Immune System Diseases": [
        "autoimmune", "lupus", "rheumatoid", "psoriasis", "crohn", "colitis",
        "atopic", "allerg", "immunodeficien", "myositis", "sjogren",
        "vasculitis", "sickle cell", "hemophilia", "haemophilia", "anemia",
        "anaemia", "immune thrombocytopenia", "itp",
    ],
    "Nervous System Diseases": [
        "neuro", "alzheimer", "parkinson", "multiple sclerosis", "epilep",
        "seizure", "stroke", "dementia", "als", "migraine", "depression",
        "anxiety", "schizophrenia", "bipolar", "huntington", "spinal muscular",
        "psychiatric", "rett", "dravet", "neuropath", "psychosis",
        "encephalitis", "sleep", "somnolen", "hypersomnia", "fragile x",
        "phelan", "angelman", "prader", "mental disorder", "intellectual",
        "autism", "adhd", "sanfilippo", "gaucher", "niemann", "leukodystrophy",
    ],
    "Cardiovascular Diseases": [
        "heart", "cardiac", "coronary", "hypertension", "atrial",
        "cardiomyopathy", "heart failure", "arrhythmia", "atherosclerosis",
        "thrombosis", "vascular", "aortic",
    ],
    "Respiratory Tract Diseases": [
        "lung", "respiratory", "asthma", "copd", "pulmonary", "bronch",
        "fibrosis", "emphysema", "cough",
    ],
    "Digestive System Diseases": [
        "liver", "hepat", "gastro", "intestin", "bowel", "colon", "pancrea",
        "gastrointestinal", "nash", "nafld", "cirrhosis", "ibd", "mash",
        "esophag", "crohn",
    ],
    "Endocrine System Diseases": [
        "diabet", "thyroid", "endocrine", "insulin", "obesity", "metabolic",
        "adrenal", "pcos", "acromegal", "hyperlipid", "osteoporosis",
        "growth hormone", "hypercholesterol",
    ],
    "Skin Diseases": [
        "skin", "derma", "eczema", "acne", "rash", "alopecia", "vitiligo",
        "pemphigus", "prurigo", "dermatitis", "ichthyosis",
    ],
    "Musculoskeletal Diseases": [
        "muscle", "muscular dystrophy", "bone", "joint", "arthritis", "osteo",
        "spinal", "skeletal", "duchenne", "becker", "spondylitis", "myopathy",
        "dystrophy", "osteosarcoma", "fibromyalgia",
    ],
    "Infectious Diseases": [
        "infect", "virus", "viral", "bacteria", "fungal", "hiv", "hepatitis",
        "covid", "rsv", "tuberculosis", "sepsis", "malaria", "influenza",
    ],
}

# NLM MeSH tree-number prefix (first 3 chars) → branch
TREE_PREFIX_MAP: Dict[str, str] = {
    "C01": "Infectious Diseases",
    "C02": "Infectious Diseases",
    "C03": "Infectious Diseases",
    "C04": "Neoplasms",
    "C05": "Musculoskeletal Diseases",
    "C06": "Digestive System Diseases",
    "C07": "Digestive System Diseases",
    "C08": "Respiratory Tract Diseases",
    "C09": "Respiratory Tract Diseases",
    "C10": "Nervous System Diseases",
    "C11": "Nervous System Diseases",   # Eye Diseases
    "C12": "Endocrine System Diseases", # Urogenital
    "C13": "Endocrine System Diseases", # Female Genital
    "C14": "Cardiovascular Diseases",
    "C15": "Immune System Diseases",    # Hemic and Lymphatic
    "C16": "Nervous System Diseases",   # Congenital / Hereditary
    "C17": "Skin Diseases",
    "C18": "Endocrine System Diseases", # Nutritional and Metabolic
    "C19": "Endocrine System Diseases",
    "C20": "Immune System Diseases",
    "C21": "Infectious Diseases",       # Disorders of Environmental Origin
    "C23": "Nervous System Diseases",   # Pathological Conditions (catch-all → NS)
    "F03": "Nervous System Diseases",   # Mental Disorders
}

# ── Utility helpers ────────────────────────────────────────────────────────────

def _is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() == ""


def _http_get(url: str, params: dict = None):
    """GET with retry logic. Returns parsed JSON, None (not found), or 'ERROR'."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
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


# ── Core matching logic ────────────────────────────────────────────────────────

def match_branch_exact(text: str) -> Optional[str]:
    """Return a branch if the text IS (or contains) an exact branch name."""
    if _is_empty(text):
        return None
    for branch in BRANCH_PRIORITY:
        if branch.lower() in text.lower():
            return branch
    # Also check aliases
    for alias, branch in ANCESTOR_ALIASES.items():
        if alias.lower() in text.lower():
            return branch
    return None


def match_branch_keywords(text: str) -> Optional[str]:
    """Return the first matching branch via keyword scan of text."""
    if _is_empty(text):
        return None
    t = text.lower()
    for branch in BRANCH_PRIORITY:
        if any(kw in t for kw in KEYWORDS.get(branch, [])):
            return branch
    return None


def match_terms_list(terms_raw) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a pipe-separated string of MeSH terms (e.g. mesh_terms_raw),
    return (branch, matched_term) or (None, None).
    Tries exact branch match first, then keyword match.
    """
    if _is_empty(terms_raw):
        return None, None
    terms = [t.strip() for t in str(terms_raw).split("|") if t.strip()]
    # Pass 1: exact branch name present in any term
    for term in terms:
        branch = match_branch_exact(term)
        if branch:
            return branch, term
    # Pass 2: keyword match on combined text
    combined = " ".join(terms)
    branch = match_branch_keywords(combined)
    if branch:
        return branch, combined[:80]
    return None, None


def nlm_lookup(term: str, nlm_cache: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Query NLM MeSH Lookup API for `term`.
    Returns (branch, term_used) or (None, None).
    Caches results by term.
    """
    term = term.strip()
    if not term:
        return None, None
    key = term.lower()
    if key in nlm_cache:
        return nlm_cache[key]

    # Search for descriptor
    resp = _http_get(NLM_SEARCH, params={"label": term, "match": "contains", "limit": 1})
    time.sleep(API_DELAY)

    if not resp or resp == "ERROR" or not isinstance(resp, list) or not resp:
        nlm_cache[key] = (None, None)
        return None, None

    resource_url = resp[0].get("resource", "")
    uid = resource_url.rstrip("/").split("/")[-1]
    if not uid:
        nlm_cache[key] = (None, None)
        return None, None

    detail = _http_get(NLM_DETAIL.format(uid=uid))
    time.sleep(API_DELAY)

    if not detail or detail == "ERROR" or not isinstance(detail, dict):
        nlm_cache[key] = (None, None)
        return None, None

    tree_entries = detail.get("treeNumber") or []
    if isinstance(tree_entries, str):
        tree_entries = [tree_entries]

    for entry in tree_entries:
        # Entry is a URI like "http://id.nlm.nih.gov/mesh/C10.597.606..."
        # Extract the path segment after the last "/"
        raw = str(entry).rstrip("/").split("/")[-1]
        prefix = raw[:3]
        branch = TREE_PREFIX_MAP.get(prefix)
        if branch:
            nlm_cache[key] = (branch, term)
            return branch, term

    nlm_cache[key] = (None, None)
    return None, None


# ── ClinicalTrials.gov API helpers ─────────────────────────────────────────────

def fetch_ct_study(nct_id: str, ct_cache: dict):
    """Fetch CT.gov v2 study JSON with in-memory cache."""
    key = str(nct_id).strip().upper()
    if key in ct_cache:
        return ct_cache[key]
    data = _http_get(CT_URL.format(nct_id=key))
    ct_cache[key] = data
    time.sleep(API_DELAY)
    return data


def resolve_from_ct_data(data: dict, nlm_cache: dict, skip_api: bool) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Given a CT.gov study JSON, attempt to resolve mesh_level1.
    Returns (branch, method_suffix, source_term).
    Method suffixes: 'browse_branches', 'meshes_exact', 'meshes_keyword',
                     'ancestors_exact', 'ancestors_keyword', 'conditions_nlm'
    """
    if not data or data == "ERROR":
        return None, None, None

    derived  = data.get("derivedSection", {})
    browse   = derived.get("conditionBrowseModule", {})
    protocol = data.get("protocolSection", {})

    # A4a — browseBranches (present in some studies)
    bb = browse.get("browseBranches", [])
    for entry in bb:
        name = entry.get("name", "")
        if name and name != "All":
            branch = match_branch_exact(name)
            if branch:
                return branch, "browse_branches", name

    # A4b — meshes (direct MeSH terms, e.g. "Rett Syndrome", "Nervous System Diseases")
    meshes = browse.get("meshes", [])
    mesh_terms = [m.get("term", "") for m in meshes if m.get("term")]
    if mesh_terms:
        combined = " | ".join(mesh_terms)
        branch, matched = match_terms_list(combined)
        if branch:
            suffix = "meshes_exact" if match_branch_exact(matched or "") else "meshes_keyword"
            return branch, suffix, matched or combined[:80]

    # A4c — ancestors
    ancestors   = browse.get("ancestors", [])
    anc_terms   = [a.get("term", "") for a in ancestors if a.get("term")]
    if anc_terms:
        combined = " | ".join(anc_terms)
        branch, matched = match_terms_list(combined)
        if branch:
            suffix = "ancestors_exact" if match_branch_exact(matched or "") else "ancestors_keyword"
            return branch, suffix, matched or combined[:80]

    # A4d — conditions → NLM lookup (requires API)
    if not skip_api:
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        for cond in conditions:
            if not cond:
                continue
            branch, term = nlm_lookup(cond, nlm_cache)
            if branch:
                return branch, "conditions_nlm", cond

    return None, None, None


# ── Disease-term extraction from free text ─────────────────────────────────────

# Patterns that signal a disease name follows
_DISEASE_PATTERNS = [
    r"(?:Phase\s+\d[/\d]*\s+(?:trial|study)\s+(?:in|for|of)\s+)([\w\s,\-\(\)]+?)(?:\s+(?:patients|participants|subjects|on|at|in|with|who)|[.,;]|$)",
    r"(?:in\s+(?:patients\s+with|subjects\s+with)\s+)([\w\s,\-\(\)]+?)(?:\s+(?:at|who|with|on)|[.,;]|$)",
    r"(?:for\s+(?:the\s+treatment\s+of|treating)\s+)([\w\s,\-\(\)]+?)(?:\s+(?:at|who|with|on|in|showed)|[.,;]|$)",
    r"(?:treating\s+)([\w\s,\-\(\)]+?)(?:\s+(?:at|who|with|showed|patients)|[.,;]|$)",
    r"(?:Designation\s+for\s+[\w\s\-]+\s+(?:in|for)\s+)([\w\s,\-\(\)]+?)(?:\s+(?:patients|Type|based)|[.,;]|$)",
]

def extract_disease_terms(text: str) -> List[str]:
    """
    Extract likely disease terms from free text using regex patterns.
    Returns a deduplicated list of candidate terms (max 5).
    """
    if _is_empty(text):
        return []

    candidates: List[str] = []

    for pattern in _DISEASE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            term = m.group(1).strip().rstrip(".,;")
            if 5 < len(term) < 80:
                candidates.append(term)

    # Deduplicate preserving order
    seen: set = set()
    unique: List[str] = []
    for t in candidates:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique[:5]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(input_file: str, output_file: str, skip_api: bool = False) -> None:
    SEP = "=" * 64

    df = pd.read_csv(input_file, low_memory=False)
    print(f"\n{SEP}")
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    if skip_api:
        print("[--skip-api] CT.gov and NLM API calls disabled")
    print(SEP)

    # ── Identify target rows ───────────────────────────────────────────────────
    def is_filled(series):
        return series.fillna("").astype(str).str.strip().ne("")

    mesh_filled_before = int(is_filled(df["mesh_level1"]).sum())
    mesh_missing_mask  = ~is_filled(df["mesh_level1"])
    has_nct_mask       = (
        df["nct_id"].fillna("").astype(str).str.strip().str.upper().str.startswith("NCT")
    )

    group_a_idx = df[mesh_missing_mask &  has_nct_mask].index.tolist()
    group_b_idx = df[mesh_missing_mask & ~has_nct_mask].index.tolist()

    print(f"\nmesh_level1 filled (before): {mesh_filled_before}")
    print(f"mesh_level1 missing (total): {len(df) - mesh_filled_before}")
    print(f"  Group A — has nct_id, missing mesh: {len(group_a_idx)}")
    print(f"  Group B — no  nct_id, missing mesh: {len(group_b_idx)}")

    # ── Ensure output columns exist ────────────────────────────────────────────
    for col in ["mesh_recovery_method", "mesh_source_term"]:
        if col not in df.columns:
            df[col] = ""

    # Shared API caches
    ct_cache:  dict = {}   # nct_id  → CT.gov JSON
    nlm_cache: dict = {}   # term    → (branch, term)

    # Recovery counts by method
    method_counts: Dict[str, int] = {}

    def _record(idx: int, branch: str, method: str, source_term: str) -> None:
        df.at[idx, "mesh_level1"]          = branch
        df.at[idx, "mesh_level1_reason"]   = method
        df.at[idx, "mesh_recovery_method"] = method
        df.at[idx, "mesh_source_term"]     = source_term
        method_counts[method] = method_counts.get(method, 0) + 1

    # ── GROUP A ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"GROUP A — {len(group_a_idx)} rows with nct_id, missing mesh_level1")
    print(SEP)

    a_resolved = 0
    a_failed   = []

    for i, idx in enumerate(group_a_idx):
        nct_id    = str(df.at[idx, "nct_id"]).strip()
        branch    = None
        method    = None
        src_term  = None

        # A1 — mesh_terms_raw exact branch name (already in local data)
        if "mesh_terms_raw" in df.columns:
            b, t = match_terms_list(df.at[idx, "mesh_terms_raw"])
            if b:
                branch, method, src_term = b, "A1_terms_raw_exact", t

        # A2 — indication + ct_conditions keyword match (no API)
        if not branch:
            for col in ("indication", "ct_conditions"):
                if col in df.columns and not _is_empty(df.at[idx, col]):
                    b = match_branch_keywords(str(df.at[idx, col]))
                    if b:
                        branch, method, src_term = b, f"A2_{col}_keyword", str(df.at[idx, col])[:80]
                        break

        # A3 — fresh CT.gov API call
        if not branch and not skip_api:
            data = fetch_ct_study(nct_id, ct_cache)
            b, meth_suffix, t = resolve_from_ct_data(data, nlm_cache, skip_api)
            if b:
                branch, method, src_term = b, f"A3_ctgov_{meth_suffix}", t

        if branch:
            _record(idx, branch, method, src_term or "")
            a_resolved += 1
            status = f"✓ {branch[:32]:<32} [{method}]"
        else:
            a_failed.append(idx)
            status = "✗ unresolved"

        if (i + 1) % 10 == 0 or (i + 1) == len(group_a_idx):
            print(f"  [{i+1:>3}/{len(group_a_idx)}] {nct_id}  {status}")
        else:
            print(f"  [{i+1:>3}/{len(group_a_idx)}] {nct_id}  {status}")

    print(f"\nGroup A: resolved {a_resolved}/{len(group_a_idx)}")
    if a_failed:
        tickers = df.loc[a_failed, "ticker"].tolist()
        print(f"  Unresolved ({len(a_failed)}): {tickers}")

    # ── GROUP B ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"GROUP B — {len(group_b_idx)} rows without nct_id, missing mesh_level1")
    print(SEP)

    b_resolved = 0
    b_failed   = []

    for i, idx in enumerate(group_b_idx):
        ticker   = df.at[idx, "ticker"]
        branch   = None
        method   = None
        src_term = None

        # B1 — indication keyword match
        if "indication" in df.columns and not _is_empty(df.at[idx, "indication"]):
            ind = str(df.at[idx, "indication"])
            b = match_branch_keywords(ind)
            if b:
                branch, method, src_term = b, "B1_indication_keyword", ind[:80]

        # B1b — indication → NLM lookup
        if not branch and not skip_api and "indication" in df.columns:
            ind = df.at[idx, "indication"]
            if not _is_empty(ind):
                b, t = nlm_lookup(str(ind), nlm_cache)
                if b:
                    branch, method, src_term = b, "B1b_indication_nlm", str(ind)[:80]

        # B2 — catalyst_summary keyword match
        if not branch and "catalyst_summary" in df.columns:
            cs = df.at[idx, "catalyst_summary"]
            if not _is_empty(cs):
                b = match_branch_keywords(str(cs))
                if b:
                    branch, method, src_term = b, "B2_summary_keyword", str(cs)[:80]

        # B2b — extract disease terms from catalyst_summary → NLM lookup
        if not branch and not skip_api and "catalyst_summary" in df.columns:
            cs = df.at[idx, "catalyst_summary"]
            if not _is_empty(cs):
                disease_terms = extract_disease_terms(str(cs))
                for term in disease_terms:
                    b, t = nlm_lookup(term, nlm_cache)
                    if b:
                        branch, method, src_term = b, "B2b_summary_nlm", term
                        break

        # B3 — drug_name → NLM lookup
        if not branch and not skip_api and "drug_name" in df.columns:
            dn = df.at[idx, "drug_name"]
            if not _is_empty(dn):
                b, t = nlm_lookup(str(dn), nlm_cache)
                if b:
                    branch, method, src_term = b, "B3_drug_name_nlm", str(dn)[:80]

        if branch:
            _record(idx, branch, method, src_term or "")
            b_resolved += 1
            status = f"✓ {branch[:32]:<32} [{method}]"
        else:
            b_failed.append(idx)
            status = "✗ unresolved"

        print(f"  [{i+1:>3}/{len(group_b_idx)}] {ticker:<8}  {status}")

    print(f"\nGroup B: resolved {b_resolved}/{len(group_b_idx)}")
    if b_failed:
        tickers = df.loc[b_failed, "ticker"].tolist()
        print(f"  Unresolved ({len(b_failed)}): {tickers}")

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv(output_file, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    mesh_filled_after  = int(is_filled(df["mesh_level1"]).sum())
    total_recovered    = mesh_filled_after - mesh_filled_before
    still_missing      = len(df) - mesh_filled_after

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"  Rows with mesh_level1 before:  {mesh_filled_before}")
    print(f"  Rows recovered this pass:      {total_recovered}")
    print(f"    Group A recovered:           {a_resolved}")
    print(f"    Group B recovered:           {b_resolved}")
    print(f"  Rows still missing mesh:       {still_missing}")
    print(f"  Total rows in output:          {len(df)}")

    if method_counts:
        print(f"\n  Breakdown by recovery method:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {count:>4}  {method}")

    if still_missing > 0:
        missing_rows = df[~is_filled(df["mesh_level1"])][["ticker", "nct_id", "indication"]].head(20)
        print(f"\n  Still-unresolved sample (first {min(20, still_missing)}):")
        print(missing_rows.to_string(index=False))

    print(f"\n  Output → {output_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deterministic mesh_level1 recovery pass"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_IN,
        help="Input CSV  (default: ml_dataset_v1.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT,
        help="Output CSV (default: ml_dataset_mesh_recovered.csv)",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip CT.gov and NLM API calls; keyword matching only",
    )
    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        skip_api=args.skip_api,
    )

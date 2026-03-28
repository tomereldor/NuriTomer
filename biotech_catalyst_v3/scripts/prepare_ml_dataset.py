#!/usr/bin/env python3
"""
prepare_ml_dataset.py
=====================
Single-pass ML preparation pipeline for enriched_all_clinical_clean_v3.csv.

Steps:
  1. Load dataset
  2. Detect missing critical ML fields
  3. Fix / drop move_pct = 0 anomalies
  4. Recover mesh_level1 via 4-level fallback hierarchy
  5. Encode mesh_level1 → deterministic numeric feature
  6. Flag unusable rows  (row_ready column)
  7. Deduplicate  (ticker + event_date + catalyst_summary)
  8. Drop zero-variance columns (catalyst_type)
  9. Save ml_dataset_features_YYYYMMDD_v1.csv  (project root, feeds step 2)

Usage:
    python prepare_ml_dataset.py
    python prepare_ml_dataset.py --input enriched_all_clinical_clean_v3.csv
    python prepare_ml_dataset.py --skip-api     # offline — skips CT.gov and NLM calls
    python prepare_ml_dataset.py --output ml_dataset_features_20260316_v1.csv
"""

import argparse
import os
import time
from datetime import date
from typing import Dict, Optional

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ML_DATA_DIR = os.path.join(PROJECT_DIR, "data", "ml")
DEFAULT_IN  = os.path.join(PROJECT_DIR, "enriched_all_clinical_clean_v3.csv")
_DATE_TAG   = date.today().strftime("%Y%m%d")
DEFAULT_OUT = os.path.join(ML_DATA_DIR, f"ml_dataset_features_{_DATE_TAG}_v1.csv")

# ── API settings ───────────────────────────────────────────────────────────────
CT_URL      = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
NLM_SEARCH  = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
NLM_DETAIL  = "https://id.nlm.nih.gov/mesh/{uid}.json"
TIMEOUT     = 15
RETRY_DELAY = 4
MAX_RETRIES = 3
API_DELAY   = 0.4   # seconds between calls

# ── MeSH category definitions ──────────────────────────────────────────────────
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

# Fixed deterministic encoding for all known categories
MESH_ENCODING: Dict[str, int] = {
    "Neoplasms":                  1,
    "Immune System Diseases":     2,
    "Nervous System Diseases":    3,
    "Cardiovascular Diseases":    4,
    "Respiratory Tract Diseases": 5,
    "Digestive System Diseases":  6,
    "Endocrine System Diseases":  7,
    "Skin Diseases":              8,
    "Musculoskeletal Diseases":   9,
    "Infectious Diseases":        10,
}

_BRANCH_KEYWORDS: Dict[str, list] = {
    "Neoplasms": [
        "cancer", "tumor", "tumour", "carcinoma", "lymphoma", "leukemia",
        "leukaemia", "melanoma", "sarcoma", "neoplasm", "glioma", "myeloma",
        "adenocarcinoma", "blastoma", "oncology", "malignant",
    ],
    "Immune System Diseases": [
        "autoimmune", "immune", "lupus", "rheumatoid", "psoriasis", "crohn",
        "colitis", "atopic", "allerg", "immunodeficiency", "myositis",
        "sjogren", "vasculitis",
    ],
    "Nervous System Diseases": [
        "neuro", "alzheimer", "parkinson", "multiple sclerosis", "epilepsy",
        "stroke", "dementia", "als", "migraine", "depression", "anxiety",
        "schizophrenia", "bipolar", "huntington", "spinal muscular",
        "psychiatric", "rett", "dravet", "neuropathy", "psychosis",
    ],
    "Cardiovascular Diseases": [
        "heart", "cardiac", "coronary", "hypertension", "atrial",
        "vascular", "cardiomyopathy", "heart failure", "arrhythmia",
        "atherosclerosis", "thrombosis",
    ],
    "Respiratory Tract Diseases": [
        "lung", "respiratory", "asthma", "copd", "pulmonary", "bronch",
        "fibrosis", "idiopathic pulmonary", "emphysema",
    ],
    "Digestive System Diseases": [
        "liver", "hepat", "gastro", "intestin", "bowel", "colon",
        "pancrea", "gastrointestinal", "nash", "nafld", "cirrhosis",
        "ibd", "mash",
    ],
    "Endocrine System Diseases": [
        "diabet", "thyroid", "endocrine", "insulin", "obesity",
        "metabolic", "adrenal", "pcos", "acromegal", "hyperlipid",
    ],
    "Skin Diseases": [
        "skin", "derma", "eczema", "acne", "rash", "alopecia",
        "vitiligo", "pemphigus", "prurigo",
    ],
    "Musculoskeletal Diseases": [
        "muscle", "bone", "joint", "arthritis", "osteo", "spinal",
        "skeletal", "duchenne", "spondylitis", "myopathy", "dystrophy",
    ],
    "Infectious Diseases": [
        "infect", "virus", "viral", "bacteria", "fungal", "hiv",
        "hepatitis", "covid", "rsv", "tuberculosis", "sepsis",
    ],
}

_ANCESTOR_ALIASES: Dict[str, str] = {
    "Infections":                                   "Infectious Diseases",
    "Heart Diseases":                               "Cardiovascular Diseases",
    "Vascular Diseases":                            "Cardiovascular Diseases",
    "Musculoskeletal and Connective Tissue Diseases": "Musculoskeletal Diseases",
    "Musculoskeletal Diseases":                     "Musculoskeletal Diseases",
    "Gastrointestinal Diseases":                    "Digestive System Diseases",
    "Hepatobiliary Diseases":                       "Digestive System Diseases",
    "Nervous System Diseases":                      "Nervous System Diseases",
    "Mental Disorders":                             "Nervous System Diseases",
}

# MeSH tree-number prefix (first 3 chars) → category
# Covers the C-branch (Diseases) and F03 (Mental Disorders)
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
    "C11": "Nervous System Diseases",
    "C14": "Cardiovascular Diseases",
    "C15": "Immune System Diseases",
    "C16": "Nervous System Diseases",
    "C17": "Skin Diseases",
    "C18": "Endocrine System Diseases",
    "C19": "Endocrine System Diseases",
    "C20": "Immune System Diseases",
    "F03": "Nervous System Diseases",
}


# ── Utility helpers ────────────────────────────────────────────────────────────

def _is_empty(val) -> bool:
    """Return True if val is None, NaN, or blank string."""
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() == ""


def count_missing(series: pd.Series) -> int:
    """Count NaN + blank-string values in a Series."""
    return (series.fillna("").astype(str).str.strip() == "").sum()


def _http_get(url: str, params: dict = None):
    """GET with retry logic. Returns parsed JSON (dict or list), None (404), or 'ERROR'."""
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


# ── Step 4 recovery helpers ────────────────────────────────────────────────────

def _mesh_from_branches_raw(branches_raw) -> Optional[str]:
    """4.1 — extract first valid category from pipe-joined mesh_branches_raw."""
    if _is_empty(branches_raw):
        return None
    for part in str(branches_raw).split("|"):
        candidate = part.strip()
        if candidate in BRANCH_PRIORITY:
            return candidate
    return None


def _mesh_from_terms_raw(terms_raw) -> Optional[str]:
    """4.2 — keyword-match pipe-joined mesh_terms_raw against known categories."""
    if _is_empty(terms_raw):
        return None
    text = str(terms_raw).lower()
    for branch in BRANCH_PRIORITY:
        if any(kw in text for kw in _BRANCH_KEYWORDS.get(branch, [])):
            return branch
    return None


def _mesh_from_nct_api(nct_id: str, cache: dict) -> Optional[str]:
    """4.3 — query ClinicalTrials.gov API and extract top-level MeSH branch."""
    nct_key = str(nct_id).strip().upper()
    if nct_key in cache:
        data = cache[nct_key]
    else:
        data = _http_get(CT_URL.format(nct_id=nct_key))
        cache[nct_key] = data
        time.sleep(API_DELAY)

    if not data or data == "ERROR":
        return None

    derived  = data.get("derivedSection", {})
    browse   = derived.get("conditionBrowseModule", {})
    protocol = data.get("protocolSection", {})

    ancestors      = browse.get("ancestors", [])
    ancestor_names = {a.get("term", "") for a in ancestors if a.get("term")}
    branch_set: set = set()
    for name in ancestor_names:
        if name in BRANCH_PRIORITY:
            branch_set.add(name)
        elif name in _ANCESTOR_ALIASES:
            branch_set.add(_ANCESTOR_ALIASES[name])

    branches = [b for b in BRANCH_PRIORITY if b in branch_set]
    if not branches:
        return None
    if len(branches) == 1:
        return branches[0]

    # Multiple branches — resolve by condition keyword match
    conditions    = protocol.get("conditionsModule", {}).get("conditions", [])
    condition_text = " ".join(conditions).lower()
    for branch in branches:
        if any(kw in condition_text for kw in _BRANCH_KEYWORDS.get(branch, [])):
            return branch

    return branches[0]  # priority-ordered fallback


def _mesh_from_indication(indication, cache: dict, skip_api: bool) -> Optional[str]:
    """
    4.4 — derive MeSH category from indication text.

    First tries local keyword matching (always runs).
    Then falls back to NLM MeSH Lookup API (skipped when skip_api=True).
    """
    if _is_empty(indication):
        return None

    text = str(indication).strip().lower()

    # Local keyword match — no network call
    for branch in BRANCH_PRIORITY:
        if any(kw in text for kw in _BRANCH_KEYWORDS.get(branch, [])):
            return branch

    if skip_api:
        return None

    # NLM MeSH Lookup API
    cache_key = f"nlm:{text}"
    if cache_key in cache:
        return cache[cache_key]

    result = None
    search_resp = _http_get(NLM_SEARCH, params={"label": text, "match": "contains", "limit": 1})
    time.sleep(API_DELAY)

    if search_resp and search_resp != "ERROR" and isinstance(search_resp, list) and search_resp:
        resource_url = search_resp[0].get("resource", "")
        uid = resource_url.rstrip("/").split("/")[-1]
        if uid:
            detail = _http_get(NLM_DETAIL.format(uid=uid))
            time.sleep(API_DELAY)
            if detail and detail != "ERROR" and isinstance(detail, dict):
                # treeNumber may appear as "treeNumber" or "treeNumberList"
                tree_numbers = detail.get("treeNumberList") or detail.get("treeNumber") or []
                if isinstance(tree_numbers, str):
                    tree_numbers = [tree_numbers]
                for tree in tree_numbers:
                    prefix = str(tree)[:3]
                    cat = TREE_PREFIX_MAP.get(prefix)
                    if cat:
                        result = cat
                        break

    cache[cache_key] = result
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(input_file: str, output_file: str, skip_api: bool = False) -> None:
    SEP = "=" * 62

    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 1 — Load dataset")
    print(SEP)

    df = pd.read_csv(input_file, low_memory=False)
    initial_rows = len(df)
    print(f"  Rows:    {initial_rows:,}")
    print(f"  Columns: {len(df.columns)}")

    # ── STEP 2: Detect missing critical ML fields ──────────────────────────────
    print(f"\n{SEP}")
    print("STEP 2 — Missing critical ML fields")
    print(SEP)

    critical_cols = ["nct_id", "mesh_level1", "indication", "move_pct", "atr_pct", "market_cap_m"]
    for col in critical_cols:
        if col not in df.columns:
            print(f"  {'[MISSING COL]':<22} {col}")
            continue
        n = count_missing(df[col])
        pct = n / len(df) * 100
        flag = " ← NOTE: identifier, not dropped" if col == "nct_id" else ""
        print(f"  {col:<22} {n:>4} missing  ({pct:.1f}%){flag}")

    # Drop rows missing BOTH move_pct AND atr_pct
    missing_both = df["move_pct"].isna() & df["atr_pct"].isna()
    n_drop_no_signal = int(missing_both.sum())
    if n_drop_no_signal:
        print(f"\n  Dropping {n_drop_no_signal} rows missing both move_pct AND atr_pct")
        df = df[~missing_both].copy()
    else:
        print(f"\n  No rows missing both move_pct AND atr_pct — nothing dropped")

    # ── STEP 3: move_pct = 0 anomaly ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 3 — move_pct = 0 anomaly")
    print(SEP)

    zero_mask = df["move_pct"] == 0
    print(f"  Rows with move_pct == 0: {zero_mask.sum()}")

    if "stale_price_data" not in df.columns:
        df["stale_price_data"] = False

    n_fixed = 0
    n_stale = 0
    stale_idx = []

    for idx in df[zero_mask].index:
        pb = df.at[idx, "price_before"] if "price_before" in df.columns else None
        pa = df.at[idx, "price_after"]  if "price_after"  in df.columns else None

        pb_valid = pd.notna(pb) and float(pb) != 0
        pa_valid = pd.notna(pa)

        if pb_valid and pa_valid:
            if float(pb) != float(pa):
                # Recalculate the correct move
                df.at[idx, "move_pct"] = round((float(pa) - float(pb)) / float(pb) * 100, 4)
                n_fixed += 1
            else:
                # Identical prices — stale market data
                stale_idx.append(idx)
                n_stale += 1
        else:
            # No usable prices — treat as stale
            stale_idx.append(idx)
            n_stale += 1

    df.loc[stale_idx, "stale_price_data"] = True

    print(f"  Recalculated (prices differ, calc was wrong): {n_fixed}")
    print(f"  Marked stale  (price_before == price_after):  {n_stale}")
    if stale_idx:
        stale_tickers = df.loc[stale_idx, "ticker"].tolist()
        print(f"  Stale tickers: {stale_tickers}")

    # ── STEP 4: Recover mesh_level1 ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 4 — Recover mesh_level1")
    print(SEP)

    if "mesh_level1" not in df.columns:
        df["mesh_level1"] = ""
    if "mesh_level1_reason" not in df.columns:
        df["mesh_level1_reason"] = ""

    # Identify rows that need recovery
    needs_mesh = df["mesh_level1"].fillna("").astype(str).str.strip() == ""
    print(f"  Rows missing mesh_level1 before recovery: {needs_mesh.sum()}")
    if skip_api:
        print("  [--skip-api] CT.gov API calls disabled; NLM API calls disabled")
        print("               keyword matching on indication still active")

    r_branches  = 0
    r_terms     = 0
    r_nct_api   = 0
    r_indication = 0
    r_still_missing = 0

    nct_cache: dict = {}   # nct_id  → raw CT.gov API response
    nlm_cache: dict = {}   # "nlm:term" → resolved category (or None)

    target_indices = df[needs_mesh].index.tolist()

    for i, idx in enumerate(target_indices):
        cat = None

        # 4.1 — mesh_branches_raw
        if cat is None and "mesh_branches_raw" in df.columns:
            cat = _mesh_from_branches_raw(df.at[idx, "mesh_branches_raw"])
            if cat:
                df.at[idx, "mesh_level1_reason"] = "branches_raw_fallback"
                r_branches += 1

        # 4.2 — mesh_terms_raw keyword match
        if cat is None and "mesh_terms_raw" in df.columns:
            cat = _mesh_from_terms_raw(df.at[idx, "mesh_terms_raw"])
            if cat:
                df.at[idx, "mesh_level1_reason"] = "terms_raw_fallback"
                r_terms += 1

        # 4.3 — ClinicalTrials.gov API (requires valid nct_id)
        if cat is None and not skip_api and "nct_id" in df.columns:
            nct_val = df.at[idx, "nct_id"]
            if not _is_empty(nct_val) and str(nct_val).strip().upper().startswith("NCT"):
                cat = _mesh_from_nct_api(str(nct_val), nct_cache)
                if cat:
                    df.at[idx, "mesh_level1_reason"] = "nct_api_fallback"
                    r_nct_api += 1

        # 4.4 — indication keyword + NLM MeSH Lookup API
        if cat is None and "indication" in df.columns:
            ind_val = df.at[idx, "indication"]
            cat = _mesh_from_indication(ind_val, nlm_cache, skip_api=skip_api)
            if cat:
                method = "indication_keyword" if skip_api else "indication_nlm_api"
                df.at[idx, "mesh_level1_reason"] = method
                r_indication += 1

        if cat:
            df.at[idx, "mesh_level1"] = cat
        else:
            r_still_missing += 1

        # Progress indicator every 50 rows
        if (i + 1) % 50 == 0 or (i + 1) == len(target_indices):
            print(f"  [{i+1}/{len(target_indices)}] recovered so far: "
                  f"{r_branches + r_terms + r_nct_api + r_indication}")

    total_recovered = r_branches + r_terms + r_nct_api + r_indication
    n_mesh_filled = (df["mesh_level1"].fillna("").astype(str).str.strip() != "").sum()

    print(f"\n  Recovery breakdown:")
    print(f"    4.1 mesh_branches_raw:       {r_branches:>4} rows")
    print(f"    4.2 mesh_terms_raw keywords: {r_terms:>4} rows")
    print(f"    4.3 ClinicalTrials.gov API:  {r_nct_api:>4} rows")
    print(f"    4.4 indication / NLM API:    {r_indication:>4} rows")
    print(f"    ─────────────────────────────────")
    print(f"    Total recovered:             {total_recovered:>4} rows")
    print(f"    Still missing:               {r_still_missing:>4} rows")
    print(f"  mesh_level1 filled after recovery: {n_mesh_filled} / {len(df)}")

    # ── STEP 5: Encode mesh_level1 → numeric feature ──────────────────────────
    print(f"\n{SEP}")
    print("STEP 5 — Encode mesh_level1 → mesh_level1_encoded")
    print(SEP)

    # Build encoding: start from fixed map, extend for any novel categories
    dynamic_encoding: Dict[str, int] = dict(MESH_ENCODING)
    next_code = max(MESH_ENCODING.values()) + 1

    def encode(val) -> Optional[int]:
        nonlocal next_code
        if _is_empty(val):
            return None
        v = str(val).strip()
        if v not in dynamic_encoding:
            dynamic_encoding[v] = next_code
            print(f"  [NEW CATEGORY] '{v}' assigned code {next_code}")
            next_code += 1
        return dynamic_encoding[v]

    df["mesh_level1_encoded"] = df["mesh_level1"].apply(encode)

    print("  Encoding map (category → code → row count):")
    dist = df["mesh_level1"].value_counts()
    for cat, code in sorted(dynamic_encoding.items(), key=lambda x: x[1]):
        count = int(dist.get(cat, 0))
        print(f"    {code:>3}  {cat:<37} {count:>4} rows")

    n_null_enc = int(df["mesh_level1_encoded"].isna().sum())
    print(f"\n  mesh_level1_encoded null: {n_null_enc} rows")

    # ── STEP 6: Flag unusable rows ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 6 — Flag unusable rows  (row_ready column)")
    print(SEP)

    df["row_ready"]            = True
    df["row_not_ready_reason"] = ""

    def _flag(mask: pd.Series, reason: str) -> int:
        """Flag rows matching mask as not_ready. Return count flagged."""
        unflagged = mask & df["row_ready"]
        df.loc[unflagged, "row_ready"]            = False
        df.loc[mask & (df["row_not_ready_reason"] == ""), "row_not_ready_reason"] = reason
        return int(mask.sum())

    c_stale    = _flag(df["stale_price_data"] == True,          "stale_price_data")
    c_no_move  = _flag(df["move_pct"].isna(),                   "missing_move_pct")
    c_no_mesh  = _flag(
        df["mesh_level1"].fillna("").astype(str).str.strip() == "",
        "missing_mesh_level1",
    )

    print(f"  stale_price_data:       {c_stale:>4} rows  → row_ready = False")
    print(f"  missing_move_pct:       {c_no_move:>4} rows  → row_ready = False")
    print(f"  missing_mesh_level1:    {c_no_mesh:>4} rows  → row_ready = False")
    print(f"\n  row_ready = True:   {int(df['row_ready'].sum()):>5}")
    print(f"  row_ready = False:  {int((~df['row_ready']).sum()):>5}")

    # ── STEP 7: Deduplicate ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 7 — Deduplicate  (ticker + event_date + catalyst_summary)")
    print(SEP)

    before_dedup = len(df)

    # Score each row by completeness; keep the richest duplicate
    df["_nonnull"] = df.notna().sum(axis=1)
    df["_cs_key"]  = (
        df["catalyst_summary"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = (
        df.sort_values("_nonnull", ascending=False)
          .drop_duplicates(subset=["ticker", "event_date", "_cs_key"], keep="first")
          .drop(columns=["_nonnull", "_cs_key"])
          .reset_index(drop=True)
    )

    rows_removed_dedup = before_dedup - len(df)
    print(f"  Before: {before_dedup:,}")
    print(f"  After:  {len(df):,}")
    print(f"  Removed {rows_removed_dedup} duplicate rows")

    # ── STEP 8: Drop zero-variance columns ────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 8 — Drop zero-variance columns")
    print(SEP)

    cols_to_drop = [c for c in ["catalyst_type"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped: {cols_to_drop}")
    print(f"  (catalyst_type had 100% 'Clinical Data' — zero predictive value)")

    # ── STEP 9: Save ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 9 — Save ML-ready dataset")
    print(SEP)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    # Final stats
    n_ready_final     = int(df["row_ready"].sum())
    n_not_ready_final = int((~df["row_ready"]).sum())
    rows_removed_total = initial_rows - len(df)

    print(f"""
  ┌─────────────────────────────────────────────┐
  │              FINAL SUMMARY                  │
  ├─────────────────────────────────────────────┤
  │  Initial rows:             {initial_rows:>6,}           │
  │  Dropped (no move+atr):    {n_drop_no_signal:>6}           │
  │  Deduplicated:             {rows_removed_dedup:>6}           │
  │  Total rows removed:       {rows_removed_total:>6}           │
  │                                             │
  │  mesh_level1 recovered:    {total_recovered:>6}           │
  │  mesh_level1 still null:   {r_still_missing:>6}           │
  │                                             │
  │  row_ready = True:         {n_ready_final:>6,}           │
  │  row_ready = False:        {n_not_ready_final:>6}           │
  │                                             │
  │  Final rows:               {len(df):>6,}           │
  │  Final columns:            {len(df.columns):>6}           │
  └─────────────────────────────────────────────┘

  Output → {output_file}
""")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML preparation pipeline for enriched_all_clinical_clean_v3.csv"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_IN,
        help=f"Input CSV  (default: enriched_all_clinical_clean_v3.csv)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT,
        help=f"Output CSV (default: ml_dataset_v1.csv)",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip ClinicalTrials.gov and NLM MeSH API calls (offline mode); "
             "keyword matching on indication still runs",
    )
    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        skip_api=args.skip_api,
    )

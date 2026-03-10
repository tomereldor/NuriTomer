#!/usr/bin/env python3
"""
finalize_mesh.py
================
Final mesh_level1 resolution pass for rows still missing after recover_mesh.py.

Staged approach to minimise LLM usage:
  Step 1 — Non-disease trial detection  (rule-based)
  Step 2 — Acronym dictionary           (word-boundary regex)
  Step 3 — Mechanism / term keywords    (substring scan)
  Step 4 — LLM inference                (Claude claude-haiku-4-5 — only rows unresolved after 1-3)

Input:   ml_dataset_mesh_recovered.csv
Output:  ml_dataset_mesh_final.csv

New column added: mesh_resolution_method
  nondisease | acronym | mechanism | llm

Usage:
    python3 finalize_mesh.py
    python3 finalize_mesh.py --skip-llm   # stops after Step 3
    python3 finalize_mesh.py --input ml_dataset_mesh_recovered.csv
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
DEFAULT_IN  = SCRIPT_DIR / "ml_dataset_mesh_recovered.csv"
DEFAULT_OUT = SCRIPT_DIR / "ml_dataset_mesh_final.csv"
DOTENV      = SCRIPT_DIR.parent / ".env"

ALLOWED_CATEGORIES = [
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
    "Other / Non-Disease",
]

# ── Step 1 — Non-disease detection ────────────────────────────────────────────
# Exact-match the indication field (stripped, lowercased)
NON_DISEASE_INDICATIONS = {
    "safety", "pharmacokinetic", "pharmacokinetics", "pharmakokinetic",
    "pk", "pk study", "bioavailability", "bioequivalence",
    "healthy volunteers", "healthy volunteer", "healthy subjects",
    "healthy adults", "normal volunteers", "normal healthy",
    "renal impairment",   # pure PK/dose-finding in renal population
}

# Substrings in ANY text field → non-disease
NON_DISEASE_SUBSTRINGS = [
    "healthy volunteer",
    "healthy subject",
    "normal volunteer",
    "bioequivalence study",
    "bioavailability study",
    "pharmacokinetic study",
    "pk/pd study",
    "no clinical trial data or results were announced",  # wrong-date placeholder rows
]

# ── Step 2 — Acronym dictionary ────────────────────────────────────────────────
# Keys are UPPERCASE; matched as whole words (\b...\b) to avoid false substrings.
# Values are MeSH Level-1 categories.
ACRONYM_MAP = {
    # ── Neoplasms ──────────────────────────────────────────────────────────
    "AML":    "Neoplasms",   # Acute Myeloid Leukemia
    "CML":    "Neoplasms",   # Chronic Myeloid Leukemia
    "CLL":    "Neoplasms",   # Chronic Lymphocytic Leukemia
    "NSCLC":  "Neoplasms",
    "SCLC":   "Neoplasms",
    "HCC":    "Neoplasms",   # Hepatocellular Carcinoma
    "RCC":    "Neoplasms",   # Renal Cell Carcinoma
    "GBM":    "Neoplasms",   # Glioblastoma Multiforme
    "NHL":    "Neoplasms",
    "DLBCL":  "Neoplasms",
    "MDS":    "Neoplasms",   # Myelodysplastic Syndrome
    "NMIBC":  "Neoplasms",   # Non-Muscle-Invasive Bladder Cancer
    "MIBC":   "Neoplasms",
    "TNBC":   "Neoplasms",   # Triple-Negative Breast Cancer
    "mCRPC":  "Neoplasms",
    "CRC":    "Neoplasms",   # Colorectal Cancer (context: oncology dataset)
    "PDAC":   "Neoplasms",   # Pancreatic Ductal Adenocarcinoma
    "HNSCC":  "Neoplasms",
    "MCC":    "Neoplasms",   # Merkel Cell Carcinoma
    "UCB":    "Neoplasms",   # Urothelial Carcinoma of Bladder
    # ── Cardiovascular ────────────────────────────────────────────────────
    "HCM":    "Cardiovascular Diseases",  # Hypertrophic Cardiomyopathy
    "DCM":    "Cardiovascular Diseases",
    "AFib":   "Cardiovascular Diseases",
    "HFpEF":  "Cardiovascular Diseases",
    "HFrEF":  "Cardiovascular Diseases",
    "PAH":    "Cardiovascular Diseases",  # Pulmonary Arterial Hypertension
    "CAD":    "Cardiovascular Diseases",
    # ── Nervous System (incl. eye → C11) ──────────────────────────────────
    "ALS":    "Nervous System Diseases",
    "SMA":    "Nervous System Diseases",
    "DMD":    "Nervous System Diseases",
    "BMD":    "Nervous System Diseases",
    "PTSD":   "Nervous System Diseases",
    "MDD":    "Nervous System Diseases",
    "GAD":    "Nervous System Diseases",
    "ADHD":   "Nervous System Diseases",
    "TBI":    "Nervous System Diseases",
    "PKU":    "Nervous System Diseases",  # Phenylketonuria (metabolic → CNS)
    "HD":     "Nervous System Diseases",  # Huntington's Disease
    "AMD":    "Nervous System Diseases",  # Age-related Macular Degeneration (C11)
    "PCED":   "Nervous System Diseases",  # Persistent Corneal Epithelial Defect
    "FSGS":   "Immune System Diseases",   # Focal Segmental Glomerulosclerosis
    # ── Immune ────────────────────────────────────────────────────────────
    "LAD":    "Immune System Diseases",   # Leukocyte Adhesion Deficiency
    "HSCT":   "Immune System Diseases",
    "GvHD":   "Immune System Diseases",
    "ITP":    "Immune System Diseases",   # Immune Thrombocytopenia
    # ── Endocrine / Metabolic ─────────────────────────────────────────────
    "CKD":    "Endocrine System Diseases",
    "ESRD":   "Endocrine System Diseases",
    "PKD":    "Endocrine System Diseases",
    "T1D":    "Endocrine System Diseases",
    "T2D":    "Endocrine System Diseases",
    "T2DM":   "Endocrine System Diseases",
    "T1DM":   "Endocrine System Diseases",
    "FCS":    "Endocrine System Diseases",  # Familial Chylomicronemia Syndrome
    "HoFH":   "Endocrine System Diseases",  # Homozygous Familial Hypercholesterolemia
    "HeFH":   "Endocrine System Diseases",
    # ── Respiratory ───────────────────────────────────────────────────────
    "COPD":   "Respiratory Tract Diseases",
    "IPF":    "Respiratory Tract Diseases",
    "PAP":    "Respiratory Tract Diseases",
    # ── Digestive ─────────────────────────────────────────────────────────
    "NASH":   "Digestive System Diseases",
    "NAFLD":  "Digestive System Diseases",
    "MASH":   "Digestive System Diseases",
    "PSC":    "Digestive System Diseases",  # Primary Sclerosing Cholangitis
    # ── Infectious ────────────────────────────────────────────────────────
    "COVID":  "Infectious Diseases",
    "COVID-19": "Infectious Diseases",
    "SARS-CoV-2": "Infectious Diseases",
    "HIV":    "Infectious Diseases",
    "HBV":    "Infectious Diseases",
    "HCV":    "Infectious Diseases",
    "RSV":    "Infectious Diseases",
    "CMV":    "Infectious Diseases",
}

# ── Step 3 — Mechanism / term keywords ────────────────────────────────────────
# Ordered list of (substring_lower, category).
# Earlier entries take priority.
MECHANISM_TERMS = [
    # Neoplasms — mechanism signals
    ("checkpoint inhibitor",    "Neoplasms"),
    ("car-t",                   "Neoplasms"),
    ("pd-1",                    "Neoplasms"),
    ("pd-l1",                   "Neoplasms"),
    ("kras",                    "Neoplasms"),
    ("egfr inhibitor",          "Neoplasms"),
    ("her2",                    "Neoplasms"),
    ("braf",                    "Neoplasms"),
    ("bcg-unresponsive",        "Neoplasms"),   # bladder cancer context
    ("bladder cancer",          "Neoplasms"),
    ("non-muscle-invasive",     "Neoplasms"),
    ("complete response rate",  "Neoplasms"),   # oncology endpoint language
    # Cardiovascular
    ("obstructive hcm",         "Cardiovascular Diseases"),
    ("myocarditis",             "Cardiovascular Diseases"),
    ("heart failure",           "Cardiovascular Diseases"),
    ("atrial fibrillation",     "Cardiovascular Diseases"),
    ("ablation catheter",       "Cardiovascular Diseases"),
    ("laromestrocel",           "Cardiovascular Diseases"),  # cardiac stem cell therapy
    ("intracardiac",            "Cardiovascular Diseases"),
    ("vascular access",         "Cardiovascular Diseases"),   # HUMA dialysis access graft
    # Nervous System — disease signals
    ("amyloid",                 "Nervous System Diseases"),
    ("p-tau",                   "Nervous System Diseases"),
    ("tau protein",             "Nervous System Diseases"),
    ("kcc2",                    "Nervous System Diseases"),   # OVID's OV350 target
    ("intracranial hemorrhage", "Nervous System Diseases"),
    ("intracranial",            "Nervous System Diseases"),
    ("rns system",              "Nervous System Diseases"),   # NeuroPace epilepsy device
    ("stargardt",               "Nervous System Diseases"),   # retinal dystrophy (C11)
    ("retinal",                 "Nervous System Diseases"),
    ("corneal",                 "Nervous System Diseases"),   # eye → C11
    ("macular",                 "Nervous System Diseases"),
    ("dry eye",                 "Nervous System Diseases"),
    ("uveiti",                  "Nervous System Diseases"),
    ("geographic atrophy",      "Nervous System Diseases"),
    ("ophthalm",                "Nervous System Diseases"),
    ("ocular",                  "Nervous System Diseases"),
    ("glaucoma",                "Nervous System Diseases"),
    ("refractive",              "Nervous System Diseases"),   # ametropia, myopia
    ("phenylketonuria",         "Nervous System Diseases"),
    ("sanfilippo",              "Nervous System Diseases"),
    # Immune
    ("il-17",                   "Immune System Diseases"),
    ("il-23",                   "Immune System Diseases"),
    ("tnf-alpha",               "Immune System Diseases"),
    ("tnf alpha",               "Immune System Diseases"),
    ("jak inhibitor",           "Immune System Diseases"),
    ("kidney transplant",       "Immune System Diseases"),   # rejection = immune
    ("transplant rejection",    "Immune System Diseases"),
    ("leukocyte adhesion",      "Immune System Diseases"),
    ("glomerulosclerosis",      "Immune System Diseases"),
    ("glomerulo",               "Immune System Diseases"),
    # Endocrine / Metabolic
    ("glp-1",                   "Endocrine System Diseases"),
    ("sglt2",                   "Endocrine System Diseases"),
    ("glycogen storage",        "Endocrine System Diseases"),
    ("chylomicronemia",         "Endocrine System Diseases"),
    ("hyperoxaluria",           "Endocrine System Diseases"),  # rare metabolic (C18)
    ("oxalic acid",             "Endocrine System Diseases"),
    ("low-back pain",           "Musculoskeletal Diseases"),
    ("low back pain",           "Musculoskeletal Diseases"),
    ("back pain",               "Musculoskeletal Diseases"),
    ("osteoarthritis",          "Musculoskeletal Diseases"),
    ("knee osteoarthritis",     "Musculoskeletal Diseases"),
    # Skin
    ("glabellar",               "Skin Diseases"),
    ("facial volume",           "Skin Diseases"),
    ("platysma",                "Skin Diseases"),
    ("botulinum",               "Skin Diseases"),
    ("hyaluronic acid",         "Skin Diseases"),
    ("dermal filler",           "Skin Diseases"),
    # Digestive
    ("functional constipation", "Digestive System Diseases"),
    ("constipation",            "Digestive System Diseases"),
    # Respiratory
    ("chronic cough",           "Respiratory Tract Diseases"),
    ("refractory chronic",      "Respiratory Tract Diseases"),
    # Infectious
    ("meningococcal",           "Infectious Diseases"),
    ("pneumococcal",            "Infectious Diseases"),
    ("monkeypox",               "Infectious Diseases"),
    ("antiviral",               "Infectious Diseases"),
    # Endocrine / Renal (full-phrase additions)
    ("chronic kidney disease",  "Endocrine System Diseases"),
    ("kidney disease",          "Endocrine System Diseases"),
    ("renal disease",           "Endocrine System Diseases"),
    ("renal failure",           "Endocrine System Diseases"),
    # Eye (full-phrase)
    ("eye pain",                "Nervous System Diseases"),
    ("eye inflammation",        "Nervous System Diseases"),
    ("inflammation eye",        "Nervous System Diseases"),
    # Other
    ("veterinary",              "Other / Non-Disease"),
    ("animal model",            "Other / Non-Disease"),
    ("herniated disc",          "Musculoskeletal Diseases"),  # PETS - canine disc
    ("disc herniation",         "Musculoskeletal Diseases"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    if DOTENV.exists():
        for line in DOTENV.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _concat(*fields) -> str:
    """Combine multiple row fields into a single lowercase string for matching."""
    parts = []
    for f in fields:
        if f and str(f).strip().lower() not in ("nan", "none", ""):
            parts.append(str(f).strip())
    return " ".join(parts).lower()


def _acronym_search(text: str) -> Optional[Tuple[str, str]]:
    """
    Scan text for whole-word acronym matches.
    Returns (category, matched_acronym) for the first hit, or None.
    """
    for acronym, category in ACRONYM_MAP.items():
        pattern = r"\b" + re.escape(acronym) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return category, acronym
    return None


def _mechanism_search(text: str) -> Optional[Tuple[str, str]]:
    """
    Scan text for mechanism/term substrings.
    Returns (category, matched_term) for the first hit (priority-ordered), or None.
    """
    for term, category in MECHANISM_TERMS:
        if term in text:
            return category, term
    return None


# ── LLM inference (Step 4) ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical trial classification engine.

Given a drug name, indication, and trial summary, classify the trial into exactly one
MeSH Level-1 disease category for machine learning purposes.

Respond with ONLY a JSON object — no other text:
{
  "mesh_level1": "<category>",
  "mesh_inference_reason": "<brief explanation, max 15 words>"
}

Allowed categories (use exact spelling):
Neoplasms
Immune System Diseases
Nervous System Diseases
Cardiovascular Diseases
Respiratory Tract Diseases
Digestive System Diseases
Endocrine System Diseases
Skin Diseases
Musculoskeletal Diseases
Infectious Diseases
Other / Non-Disease"""


def _build_user_prompt(row: pd.Series) -> str:
    drug    = str(row.get("drug_name",        "")).strip() or "unknown"
    ind     = str(row.get("indication",       "")).strip() or "unknown"
    summary = str(row.get("catalyst_summary", "")).strip() or "none"
    return f"Drug: {drug}\nIndication: {ind}\nSummary: {summary[:400]}"


def _llm_classify(rows: pd.DataFrame) -> dict:
    """
    Call Claude claude-haiku-4-5 for each row.
    Returns {index: (category, reason)} for successfully classified rows.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  [WARN] ANTHROPIC_API_KEY not set — skipping LLM step.")
        print("         Add it to .env as: ANTHROPIC_API_KEY=sk-ant-...")
        return {}

    try:
        import anthropic
    except ImportError:
        print("\n  [WARN] anthropic package not installed — skipping LLM step.")
        print("         Install: pip3 install anthropic")
        return {}

    client  = anthropic.Anthropic(api_key=api_key)
    results = {}

    for idx, row in rows.iterrows():
        prompt = _build_user_prompt(row)
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # Strip markdown fences if model wrapped in ```json```
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            parsed = json.loads(raw)
            category = parsed.get("mesh_level1", "").strip()
            reason   = parsed.get("mesh_inference_reason", "").strip()
            if category not in ALLOWED_CATEGORIES:
                # Fuzzy-fix: partial match against allowed list
                for allowed in ALLOWED_CATEGORIES:
                    if allowed.lower() in category.lower() or category.lower() in allowed.lower():
                        category = allowed
                        break
                else:
                    category = "Other / Non-Disease"
            results[idx] = (category, reason)
            print(f"    [{idx}] {row.get('ticker','')} → {category}  ({reason[:60]})")
        except Exception as e:
            print(f"    [{idx}] {row.get('ticker','')} → LLM error: {e}")
            results[idx] = ("Other / Non-Disease", f"llm_error: {str(e)[:40]}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_file: str, output_file: str, skip_llm: bool = False) -> None:
    SEP = "=" * 64
    _load_dotenv()

    df = pd.read_csv(input_file, low_memory=False)
    total_rows = len(df)

    def is_filled(col):
        return df[col].fillna("").astype(str).str.strip().ne("")

    mesh_before = int(is_filled("mesh_level1").sum())
    target_mask = ~is_filled("mesh_level1")
    target_idx  = df[target_mask].index.tolist()

    print(f"\n{SEP}")
    print(f"Loaded: {total_rows:,} rows  |  mesh filled: {mesh_before}  |  target: {len(target_idx)}")
    print(SEP)

    # Ensure output columns exist
    for col in ["mesh_resolution_method", "mesh_recovery_method", "mesh_source_term",
                "mesh_inference_reason"]:
        if col not in df.columns:
            df[col] = ""

    counts = {"nondisease": 0, "acronym": 0, "mechanism": 0, "llm": 0}

    def _apply(idx: int, category: str, method: str, source: str, reason: str = "") -> None:
        df.at[idx, "mesh_level1"]            = category
        df.at[idx, "mesh_resolution_method"] = method
        df.at[idx, "mesh_source_term"]       = source
        if reason:
            df.at[idx, "mesh_inference_reason"] = reason
        counts[method] = counts.get(method, 0) + 1

    # ── STEP 1: Non-disease detection ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 1 — Non-disease trial detection")
    print(SEP)

    still_missing = []
    for idx in target_idx:
        row = df.loc[idx]

        # Check indication exact match
        ind_raw = str(row.get("indication", "")).strip().lower()
        if ind_raw in NON_DISEASE_INDICATIONS:
            _apply(idx, "Other / Non-Disease", "nondisease", f"indication={ind_raw}")
            print(f"  [{idx}] {row.get('ticker','')}  indication='{ind_raw}' → Other / Non-Disease")
            continue

        # Check substrings across all text fields
        combined = _concat(
            row.get("indication", ""),
            row.get("catalyst_summary", ""),
            row.get("ct_conditions", ""),
        )
        matched_sub = None
        for sub in NON_DISEASE_SUBSTRINGS:
            if sub in combined:
                matched_sub = sub
                break
        if matched_sub:
            _apply(idx, "Other / Non-Disease", "nondisease", f"text='{matched_sub[:60]}'")
            print(f"  [{idx}] {row.get('ticker','')}  substring match → Other / Non-Disease")
            continue

        still_missing.append(idx)

    print(f"\n  Step 1 resolved: {counts['nondisease']}  |  remaining: {len(still_missing)}")

    # ── STEP 2: Acronym dictionary ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 2 — Acronym dictionary")
    print(SEP)

    after_acronym = []
    for idx in still_missing:
        row = df.loc[idx]
        text = _concat(
            row.get("indication", ""),
            row.get("catalyst_summary", ""),
            row.get("ct_conditions", ""),
            row.get("drug_name", ""),
        )
        result = _acronym_search(text)
        if result:
            category, matched = result
            _apply(idx, category, "acronym", matched)
            print(f"  [{idx}] {row.get('ticker',''):<8} '{matched}' → {category}")
        else:
            after_acronym.append(idx)

    print(f"\n  Step 2 resolved: {counts['acronym']}  |  remaining: {len(after_acronym)}")

    # ── STEP 3: Mechanism / term keywords ─────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 3 — Mechanism / term keywords")
    print(SEP)

    after_mechanism = []
    for idx in after_acronym:
        row = df.loc[idx]
        text = _concat(
            row.get("indication", ""),
            row.get("catalyst_summary", ""),
            row.get("ct_conditions", ""),
            row.get("drug_name", ""),
        )
        result = _mechanism_search(text)
        if result:
            category, matched = result
            _apply(idx, category, "mechanism", matched)
            print(f"  [{idx}] {row.get('ticker',''):<8} '{matched}' → {category}")
        else:
            after_mechanism.append(idx)

    print(f"\n  Step 3 resolved: {counts['mechanism']}  |  remaining: {len(after_mechanism)}")

    # ── STEP 4: LLM inference ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"STEP 4 — LLM inference  ({len(after_mechanism)} rows)")
    print(SEP)

    if skip_llm:
        print("  [--skip-llm] Skipping LLM step.")
    elif after_mechanism:
        llm_rows = df.loc[after_mechanism]
        llm_results = _llm_classify(llm_rows)
        for idx, (category, reason) in llm_results.items():
            _apply(idx, category, "llm", "", reason)
        counts["llm"] = len(llm_results)
        print(f"\n  Step 4 resolved: {counts['llm']}")

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv(output_file, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    mesh_after  = int(is_filled("mesh_level1").sum())
    total_resolved = mesh_after - mesh_before
    still_null  = total_rows - mesh_after

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"  Rows before (filled):       {mesh_before}")
    print(f"  Target rows:                {len(target_idx)}")
    print(f"")
    print(f"  Resolved by Step 1 (non-disease):  {counts.get('nondisease', 0)}")
    print(f"  Resolved by Step 2 (acronym):      {counts.get('acronym',    0)}")
    print(f"  Resolved by Step 3 (mechanism):    {counts.get('mechanism',  0)}")
    print(f"  Resolved by Step 4 (llm):          {counts.get('llm',        0)}")
    print(f"  ─────────────────────────────────────")
    print(f"  Total resolved this pass:   {total_resolved}")
    print(f"  Still missing:              {still_null}")
    print(f"  mesh_level1 fill rate:      {mesh_after}/{total_rows}  ({mesh_after/total_rows*100:.1f}%)")

    if still_null > 0:
        leftover = df[~is_filled("mesh_level1")][["ticker", "indication", "drug_name"]].head(20)
        print(f"\n  Still unresolved ({still_null}):")
        print(leftover.to_string(index=False))

    # mesh distribution
    print(f"\n  mesh_level1 distribution (final):")
    dist = df["mesh_level1"].value_counts()
    for cat, n in dist.items():
        print(f"    {n:>4}  {cat}")

    print(f"\n  Output → {output_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final mesh_level1 resolution (Steps 1-4)")
    parser.add_argument("--input",    default=str(DEFAULT_IN),  help="Input CSV")
    parser.add_argument("--output",   default=str(DEFAULT_OUT), help="Output CSV")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Stop after Step 3 (no LLM calls)")
    args = parser.parse_args()

    main(input_file=args.input, output_file=args.output, skip_llm=args.skip_llm)

"""
add_biological_features.py
==========================
Pass-9: Add biological feature families — heritability & enrichment relevance.

Background:
  The v16 model (AUC 0.695) uses disease-level binary flags (feat_has_predictive_biomarker,
  feat_targeted_therapy_exists) but has never encoded disease_genetic_basis as a model feature.
  This pass adds 7 deterministic features from two families:

  Family A — Heritability (3 features):
    feat_genetic_basis_encoded           — ordinal int 0-3 from disease_genetic_basis
    feat_heritability_proxy_score        — float 0.0-1.0 proxy for heritability degree
    feat_heritability_level              — ordinal bin 0-2 (low/moderate/high)

  Family B — Enrichment Relevance (4 features):
    feat_biomarker_stratified_flag       — trial-level keyword match (indication + ct_official_title)
    feat_targeted_mechanism_flag         — drug-name suffix/keyword rules
    feat_disease_molecular_heterogeneity_score — disease-level tractability proxy
    feat_enrichment_relevance_score      — weighted composite of the above

All 7 features are STRICTLY_CLEAN (pre-event safe): derived from CT.gov registration
metadata and LLM-derived disease classifications, all known before the event date.
Zero new LLM API calls required.

Input:  latest ml_dataset_features_YYYYMMDD_vN.csv
Output: ml_dataset_features_YYYYMMDD_v(N+1).csv

Usage (from biotech_catalyst_v3/):
    python -m scripts.add_biological_features
"""

import glob
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")

NEW_FEAT_COLS = [
    "feat_genetic_basis_encoded",
    "feat_heritability_proxy_score",
    "feat_heritability_level",
    "feat_biomarker_stratified_flag",
    "feat_targeted_mechanism_flag",
    "feat_disease_molecular_heterogeneity_score",
    "feat_enrichment_relevance_score",
]

# (name, stage, feature_type, description, source_columns, source_type)
NEW_FEAT_META = [
    (
        "feat_genetic_basis_encoded",
        "pass9_biological",
        "ordinal",
        "Ordinal encoding of disease_genetic_basis: none=0, polygenic=1, somatic=2, monogenic=3; NaN for unknown",
        "disease_genetic_basis",
        "deterministic_mapping",
    ),
    (
        "feat_heritability_proxy_score",
        "pass9_biological",
        "float",
        "Heritability proxy 0.0-1.0: monogenic=0.85, somatic=0.45, polygenic=0.35, none=0.10, null=0.40",
        "disease_genetic_basis",
        "deterministic_mapping",
    ),
    (
        "feat_heritability_level",
        "pass9_biological",
        "ordinal",
        "Ordinal bin of heritability_proxy_score: low=0 (<0.30), moderate=1 (0.30-0.60), high=2 (>0.60)",
        "feat_heritability_proxy_score",
        "deterministic_mapping",
    ),
    (
        "feat_biomarker_stratified_flag",
        "pass9_biological",
        "binary",
        "1 if indication or ct_official_title contains biomarker/mutation-specific trial design keywords",
        "indication, ct_official_title",
        "deterministic_keyword",
    ),
    (
        "feat_targeted_mechanism_flag",
        "pass9_biological",
        "binary",
        "1 if drug_name contains targeted-therapy suffix/keyword (mAb, -nib, inhibitor, gene therapy, etc.) or disease is monogenic",
        "drug_name, disease_genetic_basis",
        "deterministic_keyword",
    ),
    (
        "feat_disease_molecular_heterogeneity_score",
        "pass9_biological",
        "float",
        "Disease tractability proxy 0.0-1.0: lower = well-defined target (monogenic); higher = heterogeneous (somatic/oncology)",
        "disease_genetic_basis, mesh_level1, feat_rare_disease_flag",
        "deterministic_mapping",
    ),
    (
        "feat_enrichment_relevance_score",
        "pass9_biological",
        "float",
        "Composite enrichment relevance 0.0-1.0: 0.35×biomarker_stratified + 0.25×targeted_mechanism + 0.25×(1-heterogeneity) + 0.15×has_predictive_biomarker",
        "feat_biomarker_stratified_flag, feat_targeted_mechanism_flag, feat_disease_molecular_heterogeneity_score, feat_has_predictive_biomarker",
        "deterministic_mapping",
    ),
]

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# Biomarker stratification — trial-level keywords (indication + ct_official_title)
_BIOMARKER_KEYWORDS = [
    # Mutation/gene-specific
    "egfr", "her2", "her-2", "brca", "alk", "ros1", "kras", "braf", "met",
    "ret", "ntrk", "flt3", "npm1", "idh1", "idh2", "pik3ca", "erbb2",
    "abl", "jak2", "pten",
    # Biomarker-specific
    "pd-l1", "msi-h", "msi high", "msi-high", "tmb-h", "tmb high", "tmb-high",
    "hla", "cr1",
    # Trial enrichment design
    "positive", "mutant", "fusion", "stratified", "selected",
    "enriched", "biomarker", "genotype", "amplification", "deficient",
]

# Targeted mechanism — drug_name suffixes (without leading hyphen; applied to each
# whitespace/comma/semicolon-separated token of the drug name, lowercased)
_TARGETED_SUFFIXES = (
    "mab", "nib", "tinib", "inib", "zumab", "ximab", "umab",
    "asin", "enib", "rafenib",
)

_TARGETED_KEYWORDS = [
    "inhibitor", "gene therapy", "sirna", "antisense", "adc", "bispecific",
    "car-t", "cart", "oligonucleotide", "crispr", "gene editing", "mrna",
    "antisense oligonucleotide",
]


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------

def _find_latest_features(base_dir):
    """Return (path, version_int, date_tag) for highest-version ml_dataset_features file."""
    candidates = glob.glob(os.path.join(base_dir, "ml_dataset_features_*.csv"))
    best, best_v, best_date = None, -1, None
    for f in candidates:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, best_date = v, f, m.group(1)
    return best, best_v, best_date


def _find_latest_dict(base_dir):
    """Return path of highest-version ml_feature_dict file."""
    candidates = glob.glob(os.path.join(base_dir, "ml_feature_dict_*.csv"))
    best, best_v = None, -1
    for f in candidates:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best = v, f
    return best


# ---------------------------------------------------------------------------
# Family A — Heritability features
# ---------------------------------------------------------------------------

_GENETIC_BASIS_ORDINAL = {"none": 0, "polygenic": 1, "somatic": 2, "monogenic": 3}
_HERITABILITY_PROXY    = {"monogenic": 0.85, "somatic": 0.45, "polygenic": 0.35, "none": 0.10}
_HERITABILITY_NULL_FALLBACK = 0.40


def _add_heritability_features(df):
    """Add feat_genetic_basis_encoded, feat_heritability_proxy_score, feat_heritability_level."""
    # Source column: feat_genetic_basis (has 'unknown' for nulls) or disease_genetic_basis
    if "feat_genetic_basis" in df.columns:
        raw = df["feat_genetic_basis"].str.lower().str.strip()
        # treat 'unknown' as null for the ordinal encoding
        raw_clean = raw.replace("unknown", np.nan)
    elif "disease_genetic_basis" in df.columns:
        raw_clean = df["disease_genetic_basis"].str.lower().str.strip()
    else:
        raw_clean = pd.Series(np.nan, index=df.index)

    # feat_genetic_basis_encoded: ordinal 0-3, NaN for unknown
    df["feat_genetic_basis_encoded"] = raw_clean.map(_GENETIC_BASIS_ORDINAL).astype("float64")

    # feat_heritability_proxy_score: float 0-1, null → 0.40 fallback
    df["feat_heritability_proxy_score"] = (
        raw_clean.map(_HERITABILITY_PROXY).fillna(_HERITABILITY_NULL_FALLBACK)
    )

    # feat_heritability_level: ordinal bin 0-2
    score = df["feat_heritability_proxy_score"]
    df["feat_heritability_level"] = pd.cut(
        score,
        bins=[-np.inf, 0.30, 0.60, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype("float64")

    return df


# ---------------------------------------------------------------------------
# Family B — Enrichment Relevance features
# ---------------------------------------------------------------------------

def _biomarker_stratified(df):
    """Binary: indication or ct_official_title contains a biomarker/enrichment keyword."""
    ind   = df["indication"].fillna("").str.lower()      if "indication"      in df.columns else pd.Series("", index=df.index)
    title = df["ct_official_title"].fillna("").str.lower() if "ct_official_title" in df.columns else pd.Series("", index=df.index)
    combined = ind + " " + title

    flag = pd.Series(0, index=df.index, dtype="float64")
    for kw in _BIOMARKER_KEYWORDS:
        flag = flag.where(~combined.str.contains(kw, regex=False), 1.0)
    return flag


def _targeted_mechanism(df):
    """Binary: drug_name has targeted-therapy suffix/keyword, or disease is monogenic."""
    import re as _re
    drug_lower = df["drug_name"].fillna("").str.lower() if "drug_name" in df.columns else pd.Series("", index=df.index)

    def _has_suffix(drug_str):
        # Split on spaces, commas, semicolons, slashes, parens to get individual tokens
        tokens = _re.split(r"[\s,;/()\[\]]+", drug_str)
        # Strip trailing punctuation from each token
        tokens = [t.rstrip(".:!?-") for t in tokens if t]
        return any(tok.endswith(sfx) for tok in tokens for sfx in _TARGETED_SUFFIXES)

    suffix_hit  = drug_lower.apply(_has_suffix)
    keyword_hit = drug_lower.apply(lambda s: any(kw in s for kw in _TARGETED_KEYWORDS))

    # monogenic diseases → likely a gene-targeting therapy
    if "feat_genetic_basis" in df.columns:
        monogenic = (df["feat_genetic_basis"].str.lower().str.strip() == "monogenic")
    elif "disease_genetic_basis" in df.columns:
        monogenic = (df["disease_genetic_basis"].str.lower().str.strip() == "monogenic")
    else:
        monogenic = pd.Series(False, index=df.index)

    return (suffix_hit | keyword_hit | monogenic).astype("float64")


def _molecular_heterogeneity(df):
    """Molecular heterogeneity score 0-1: higher = more heterogeneous (somatic/oncology)."""
    if "feat_genetic_basis" in df.columns:
        raw = df["feat_genetic_basis"].str.lower().str.strip().replace("unknown", np.nan)
    elif "disease_genetic_basis" in df.columns:
        raw = df["disease_genetic_basis"].str.lower().str.strip()
    else:
        raw = pd.Series(np.nan, index=df.index)

    mesh = df["mesh_level1"].fillna("") if "mesh_level1" in df.columns else pd.Series("", index=df.index)
    onc_flag = df["feat_oncology_flag"].fillna(0).astype(float) if "feat_oncology_flag" in df.columns else pd.Series(0.0, index=df.index)
    rare_flag = df["feat_rare_disease_flag"].fillna(0).astype(float) if "feat_rare_disease_flag" in df.columns else pd.Series(0.0, index=df.index)

    # Base score from genetic_basis
    def _base(basis):
        if pd.isna(basis):
            return 0.50
        if basis == "monogenic":
            return 0.10
        if basis == "somatic":
            return 0.75   # will be adjusted by oncology flag
        if basis == "polygenic":
            return 0.55
        if basis == "none":
            return 0.40
        return 0.50

    base = raw.map(_base)

    # somatic + non-oncology → lower
    is_somatic = (raw == "somatic")
    is_non_onc = (onc_flag == 0)
    base = base.where(~(is_somatic & is_non_onc), 0.50)

    # MeSH adjustment
    is_neoplasm  = mesh.str.lower() == "neoplasms"
    is_infect    = mesh.str.lower() == "infectious diseases"

    score = base.copy()
    score = score.where(~is_neoplasm, (base + 0.10).clip(upper=1.0))
    score = score.where(~is_infect,   (base - 0.15).clip(lower=0.0))
    score = score.where(~(rare_flag == 1), (score - 0.10).clip(lower=0.0))

    return score.astype("float64")


def _add_enrichment_relevance_features(df):
    """Add 4 enrichment relevance features."""
    df["feat_biomarker_stratified_flag"]          = _biomarker_stratified(df)
    df["feat_targeted_mechanism_flag"]             = _targeted_mechanism(df)
    df["feat_disease_molecular_heterogeneity_score"] = _molecular_heterogeneity(df)

    bio  = df["feat_biomarker_stratified_flag"]
    tgt  = df["feat_targeted_mechanism_flag"]
    het  = df["feat_disease_molecular_heterogeneity_score"]
    pbm  = df["feat_has_predictive_biomarker"].fillna(0).astype(float) if "feat_has_predictive_biomarker" in df.columns else pd.Series(0.0, index=df.index)

    df["feat_enrichment_relevance_score"] = (
        0.35 * bio
        + 0.25 * tgt
        + 0.25 * (1.0 - het)
        + 0.15 * pbm
    ).astype("float64")

    return df


# ---------------------------------------------------------------------------
# Feature dictionary update
# ---------------------------------------------------------------------------

def update_feature_dict(df, old_dict_path, out_path):
    if old_dict_path and os.path.exists(old_dict_path):
        fdict = pd.read_csv(old_dict_path)
    else:
        raise FileNotFoundError(f"Feature dict not found: {old_dict_path}")

    # Drop any existing entries for these features (idempotent)
    fdict = fdict[~fdict["feature_name"].isin(NEW_FEAT_COLS)].copy()

    new_rows = []
    for feat_name, stage, ftype, desc, src_cols, src_type in NEW_FEAT_META:
        n_valid = int(df[feat_name].notna().sum())
        n_null  = int(df[feat_name].isna().sum())
        pct_v   = round(n_valid / len(df) * 100, 1)

        row = {c: "" for c in fdict.columns}
        row.update({
            "feature_name":   feat_name,
            "stage":          stage,
            "feature_type":   ftype,
            "description":    desc,
            "source_columns": src_cols,
            "source_type":    src_type,
            "n_valid":        n_valid,
            "n_null":         n_null,
            "pct_valid":      pct_v,
        })
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows, columns=fdict.columns)
    full   = pd.concat([fdict, new_df], ignore_index=True)
    full.to_csv(out_path, index=False)
    return len(full)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # --- find inputs ---
    feat_path, feat_v, date_tag = _find_latest_features(BASE_DIR)
    dict_path                   = _find_latest_dict(BASE_DIR)

    if feat_path is None:
        raise FileNotFoundError("No ml_dataset_features_YYYYMMDD_vN.csv found in " + BASE_DIR)
    if dict_path is None:
        raise FileNotFoundError("No ml_feature_dict_YYYYMMDD_vN.csv found in " + BASE_DIR)

    print(f"Input : {os.path.basename(feat_path)}  (v{feat_v})")
    df = pd.read_csv(feat_path)
    print(f"Shape : {df.shape[0]} rows × {df.shape[1]} cols")

    # --- idempotency check ---
    already_present = [c for c in NEW_FEAT_COLS if c in df.columns]
    if already_present:
        print(f"[idempotent] Features already present: {already_present} — skipping recomputation.")
        return

    # --- add features ---
    df = _add_heritability_features(df)
    df = _add_enrichment_relevance_features(df)

    # --- spot-check distributions ---
    print("\nFeature distributions:")
    for col in NEW_FEAT_COLS:
        n   = df[col].notna().sum()
        pct = n / len(df) * 100
        if df[col].dtype in (float, "float64") and df[col].nunique() > 5:
            print(f"  {col}: {n}/{len(df)} ({pct:.1f}%) | mean={df[col].mean():.3f} | std={df[col].std():.3f}")
        else:
            vc = df[col].value_counts(dropna=False).head(5).to_dict()
            print(f"  {col}: {n}/{len(df)} ({pct:.1f}%) | vals={vc}")

    # --- output paths ---
    new_v         = feat_v + 1
    out_feat_name = f"ml_dataset_features_{date_tag}_v{new_v}.csv"
    out_dict_name = f"ml_feature_dict_{date_tag}_v{new_v}.csv"
    out_feat_path = os.path.join(BASE_DIR, out_feat_name)
    out_dict_path = os.path.join(BASE_DIR, out_dict_name)

    # --- archive previous latest ---
    if os.path.dirname(os.path.abspath(feat_path)) == os.path.abspath(BASE_DIR):
        import shutil
        shutil.move(feat_path, os.path.join(ARCHIVE_DIR, os.path.basename(feat_path)))
        print(f"\nArchived: {os.path.basename(feat_path)}")

    if dict_path and os.path.dirname(os.path.abspath(dict_path)) == os.path.abspath(BASE_DIR):
        import shutil
        shutil.move(dict_path, os.path.join(ARCHIVE_DIR, os.path.basename(dict_path)))
        print(f"Archived: {os.path.basename(dict_path)}")

    # --- save feature dataset ---
    df.to_csv(out_feat_path, index=False)
    print(f"\nSaved : {out_feat_name}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # --- update feature dict ---
    archived_dict = os.path.join(ARCHIVE_DIR, os.path.basename(dict_path)) if dict_path else None
    n_entries = update_feature_dict(df, archived_dict, out_dict_path)
    print(f"Dict  : {n_entries} entries → {out_dict_name}")
    print("\nDone — biological features (pass9) added.")


if __name__ == "__main__":
    main()

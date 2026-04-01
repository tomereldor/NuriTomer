"""
enrich_opentargets_genetics.py
==============================
Replace LLM-derived genetics features with Open Targets Platform evidence.

Queries the Open Targets GraphQL API (free, no key required) to map clinical
trial indications to EFO IDs and fetch per-disease genetic evidence scores.

Produces 6 new feature columns:
  feat_ot_genetic_basis           — ordinal: none=0, polygenic=1, somatic=2, monogenic=3; -1=unknown
  feat_ot_genetic_evidence_score  — float 0-1: max(monogenic, gwas, cancer_somatic) signal
  feat_ot_n_genetic_targets       — int: targets with curated monogenic evidence (in top 100)
  feat_ot_monogenic_signal        — float 0-1: max score from Orphanet/G2P/ClinGen/PanelApp
  feat_ot_gwas_signal             — float 0-1: max score from GWAS Catalog / Gene Burden
  feat_ot_somatic_signal          — float 0-1: max score from IntOGen / Cancer Gene Census

Datasource design rationale (validated empirically, OT API v26.x):
  - `gene2phenotype`, `orphanet`, `clingen`, `genomics_england`: curated rare/monogenic DBs.
    These ONLY contain single-gene disease associations → clean monogenic signal.
  - `gwas_credible_sets`: GWAS Catalog credible sets (renamed from `ot_genetics_portal` in v26.x).
    Used alone for polygenic signal — `gene_burden` excluded because CF (monogenic) has
    gene_burden=0.83 for CFTR from rare-variant burden tests, which would give a false polygenic signal.
  - `intogen`, `cancer_gene_census`: cancer driver databases → true somatic/cancer signal.
    NOTE: `eva_somatic` is intentionally EXCLUDED from somatic signal — it includes
    non-cancer somatic variants (e.g. KCNJ11 in T2D insulinomas) that contaminate the signal.

Usage (from biotech_catalyst_v3/):
    python -m scripts.enrich_opentargets_genetics              # Phase 1: prototype (top 50)
    python -m scripts.enrich_opentargets_genetics --n 100      # prototype with custom N
    python -m scripts.enrich_opentargets_genetics --full       # Phase 2: all diseases
    python -m scripts.enrich_opentargets_genetics --full --write-features  # + update features

Output (--full --write-features):
    ml_dataset_features_YYYYMMDD_v(N+1).csv  with 6 new feat_ot_* columns
    cache/opentargets_efo_mapping_v1.json    EFO ID cache (gitignored)
    cache/opentargets_genetics_v1.json       evidence scores cache (gitignored)
"""

import argparse
import glob
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
ML_DATA_DIR = os.path.join(BASE_DIR, "data", "ml")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")

OT_API_URL          = "https://api.platform.opentargets.org/api/v4/graphql"
EFO_CACHE_PATH      = os.path.join(CACHE_DIR, "opentargets_efo_mapping_v1.json")
GENETICS_CACHE_PATH = os.path.join(CACHE_DIR, "opentargets_genetics_v1.json")

# ---------------------------------------------------------------------------
# Datasource groupings
# ---------------------------------------------------------------------------

# Curated rare/monogenic disease databases (do NOT include common risk alleles)
MONOGENIC_CURATED_SRC = {"gene2phenotype", "orphanet", "clingen", "genomics_england"}

# GWAS Catalog → polygenic signal
# NOTE: as of OT API v26.x, GWAS Catalog is "gwas_credible_sets" (renamed from "ot_genetics_portal")
# gene_burden intentionally excluded: CF (monogenic) has gene_burden=0.83 because CFTR is studied
# in rare-variant burden tests — high gene_burden does NOT imply polygenic disease.
GWAS_SRC = {"gwas_credible_sets"}

# Cancer driver gene databases — IntOGen ONLY
# cancer_gene_census excluded: it includes cancer pathway genes that are ALSO used as
# therapeutic targets in non-cancer diseases (JAK2/JAK3/FGFR3 in ulcerative colitis via
# JAK inhibitors → false somatic signal). IntOGen comes exclusively from tumor somatic
# mutation burden studies (TCGA etc.) and is clean for this purpose.
# eva_somatic also excluded — captures non-cancer ClinVar somatic variants (e.g. KCNJ11 in T2D).
CANCER_SRC = {"intogen"}

OT_FEATURE_COLS = [
    "feat_ot_genetic_basis",
    "feat_ot_genetic_evidence_score",
    "feat_ot_n_genetic_targets",
    "feat_ot_monogenic_signal",
    "feat_ot_gwas_signal",
    "feat_ot_somatic_signal",
]

# Null fallback score for diseases with no OT data (same as current heritability proxy null)
_NULL_EVIDENCE_SCORE = 0.40

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

EFO_SEARCH_QUERY = """
query SearchDisease($term: String!) {
  search(queryString: $term, entityNames: ["disease"], page: {index: 0, size: 3}) {
    hits { id name }
  }
}
"""

DISEASE_EVIDENCE_QUERY = """
query DiseaseEvidence($efoId: String!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: 0, size: 25}) {
      count
      rows {
        target { approvedSymbol }
        score
        datasourceScores { id score }
      }
    }
  }
}
"""

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _make_ssl_context():
    ctx = ssl.create_default_context()
    return ctx


def graphql_request(query, variables=None, retries=2):
    """Execute a GraphQL query. Returns (data_dict, error_str_or_None)."""
    ctx = _make_ssl_context()
    payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OT_API_URL,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                result = json.loads(resp.read())
            if "errors" in result:
                return None, str(result["errors"])
            return result.get("data"), None
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            err = f"HTTP {e.code}: {body}"
        except Exception as e:
            err = str(e)
        if attempt < retries - 1:
            time.sleep(1.0 * (attempt + 1))
    return None, err


# ---------------------------------------------------------------------------
# EFO mapping
# ---------------------------------------------------------------------------

def search_efo(indication, efo_cache):
    """Map indication → (efo_id, ot_name). Uses cache; makes API call on miss."""
    key = str(indication).lower().strip()
    if key in efo_cache:
        entry = efo_cache[key]
        return entry.get("efo_id"), entry.get("ot_name")

    data, err = graphql_request(EFO_SEARCH_QUERY, {"term": indication})
    if err or not data:
        efo_cache[key] = {"efo_id": None, "ot_name": None, "error": str(err)}
        return None, None

    hits = data.get("search", {}).get("hits", [])
    if not hits:
        efo_cache[key] = {"efo_id": None, "ot_name": None}
        return None, None

    top = hits[0]
    efo_cache[key] = {"efo_id": top["id"], "ot_name": top["name"]}
    return top["id"], top["name"]


# ---------------------------------------------------------------------------
# Evidence fetch and aggregation
# ---------------------------------------------------------------------------

def fetch_evidence(efo_id, genetics_cache):
    """Fetch OT evidence for an EFO ID. Uses cache; makes API call on miss."""
    if efo_id in genetics_cache:
        return genetics_cache[efo_id]

    data, err = graphql_request(DISEASE_EVIDENCE_QUERY, {"efoId": efo_id})
    if err or not data or not data.get("disease"):
        result = {"error": str(err), "count": 0, "rows": []}
        genetics_cache[efo_id] = result
        return result

    d = data["disease"]
    assoc = d.get("associatedTargets", {})
    result = {
        "disease_name": d.get("name"),
        "count": assoc.get("count", 0),
        "rows": assoc.get("rows", []),
    }
    genetics_cache[efo_id] = result
    return result


def aggregate_evidence(evidence):
    """
    Aggregate per-target datasource scores into per-disease signal scores.

    Returns dict with:
      mono_signal   — max score across MONOGENIC_CURATED_SRC
      gwas_signal   — max score across GWAS_SRC
      somatic_signal — max score across CANCER_SRC (intogen + CGC only)
      n_mono_targets — count of targets with any MONOGENIC_CURATED_SRC evidence
    """
    rows = evidence.get("rows", [])
    mono_max = gwas_max = somatic_max = 0.0
    n_mono = 0

    for row in rows:
        has_mono = False
        for ds in row.get("datasourceScores", []):
            src = ds.get("id", "")
            score = ds.get("score") or 0.0
            if src in MONOGENIC_CURATED_SRC:
                mono_max = max(mono_max, score)
                has_mono = True
            elif src in GWAS_SRC:
                gwas_max = max(gwas_max, score)
            elif src in CANCER_SRC:
                somatic_max = max(somatic_max, score)
        if has_mono:
            n_mono += 1

    return {
        "mono_signal":     round(mono_max, 6),
        "gwas_signal":     round(gwas_max, 6),
        "somatic_signal":  round(somatic_max, 6),
        "n_mono_targets":  n_mono,
    }


def classify_genetic_basis(mono, gwas, somatic, n_mono):
    """
    Classify disease into genetic category based on OT evidence signals.

    Returns:
        3 = monogenic  — curated single-gene disease (Orphanet/G2P/ClinGen/PanelApp)
        2 = somatic    — cancer driver (IntOGen / Cancer Gene Census dominant)
        1 = polygenic  — complex trait (GWAS Catalog OR many curated genes)
        0 = none       — low evidence across all sources
       -1 = unknown    — no OT data available

    gwas here = gwas_credible_sets ONLY (excludes gene_burden to avoid false-positive
    polygenic labels for single-gene disorders with gene burden evidence, e.g. CF).

    Threshold gwas > 0.7 for polygenic: chosen to separate genuine GWAS-driven diseases
    (T2D=0.942, psoriasis=0.940, Alzheimer=0.953) from monogenic diseases with moderate
    GWAS signal (CF gwas_credible_sets=0.512 from a single CFTR GWAS entry).

    Logic validated against:
        CF (mono), sickle cell (mono), Huntington (mono),
        breast cancer (somatic), T2D (poly), RA (poly), psoriasis (poly),
        Alzheimer's (poly), asthma (poly)
    """
    max_all = max(mono, gwas, somatic)

    # No meaningful data → unknown
    if max_all < 0.05:
        return -1

    # Cancer/somatic: IntOGen or Cancer Gene Census dominates
    if somatic > 0.3:
        return 2

    # Polygenic: strong GWAS Catalog signal (many independent loci across the genome)
    # Threshold 0.7 separates genuine multi-locus GWAS diseases from single-gene GWAS entries
    if gwas > 0.7:
        return 1

    # Monogenic: curated rare-disease databases, few associated genes, low GWAS
    # n_mono ≤ 10 guards against complex diseases with many curated genes (e.g. RA n=53)
    if mono > 0.5 and n_mono <= 10 and somatic < 0.15:
        return 3

    # Polygenic: moderate GWAS OR many curated genes (complex disease with monogenic forms)
    if gwas > 0.3 or n_mono > 10:
        return 1

    # Weak curated monogenic signal with few targets
    if mono > 0.1 and n_mono > 0:
        return 3

    # Residual low signal
    if max_all > 0.1:
        return 0

    return 0


# ---------------------------------------------------------------------------
# Per-indication feature computation
# ---------------------------------------------------------------------------

_NULL_FEATURES = {
    "feat_ot_genetic_basis":          -1,
    "feat_ot_genetic_evidence_score":  _NULL_EVIDENCE_SCORE,
    "feat_ot_n_genetic_targets":       0,
    "feat_ot_monogenic_signal":        0.0,
    "feat_ot_gwas_signal":             0.0,
    "feat_ot_somatic_signal":          0.0,
    "ot_efo_id":                       None,
    "ot_efo_name":                     None,
}


def compute_features_for_indication(indication, efo_cache, genetics_cache, sleep_sec=0.65):
    """Map one indication string → full OT feature dict. Rate-limited."""
    if not indication or (isinstance(indication, float) and np.isnan(indication)):
        return dict(_NULL_FEATURES)

    efo_id, ot_name = search_efo(str(indication), efo_cache)
    time.sleep(sleep_sec)

    if not efo_id:
        return dict(_NULL_FEATURES)

    evidence = fetch_evidence(efo_id, genetics_cache)
    time.sleep(sleep_sec)

    if evidence.get("count", 0) == 0:
        # EFO found but no associations in OT → unknown (coverage gap, not "none")
        return {**dict(_NULL_FEATURES), "ot_efo_id": efo_id, "ot_efo_name": ot_name}

    scores = aggregate_evidence(evidence)
    category = classify_genetic_basis(
        scores["mono_signal"], scores["gwas_signal"],
        scores["somatic_signal"], scores["n_mono_targets"]
    )
    evidence_score = max(scores["mono_signal"], scores["gwas_signal"], scores["somatic_signal"])
    if evidence_score < 0.01:
        evidence_score = _NULL_EVIDENCE_SCORE

    return {
        "feat_ot_genetic_basis":          category,
        "feat_ot_genetic_evidence_score":  round(evidence_score, 6),
        "feat_ot_n_genetic_targets":       scores["n_mono_targets"],
        "feat_ot_monogenic_signal":        scores["mono_signal"],
        "feat_ot_gwas_signal":             scores["gwas_signal"],
        "feat_ot_somatic_signal":          scores["somatic_signal"],
        "ot_efo_id":                       efo_id,
        "ot_efo_name":                     ot_name,
    }


# ---------------------------------------------------------------------------
# Indication lookup builder
# ---------------------------------------------------------------------------

def build_indication_lookup(indications, efo_cache, genetics_cache, verbose=True):
    """
    For each unique indication, compute OT features. Returns dict:
        {indication_str: feature_dict}
    """
    unique = [i for i in sorted(set(indications)) if i and not (isinstance(i, float) and np.isnan(i))]
    lookup = {}
    n = len(unique)
    t0 = time.time()

    for i, ind in enumerate(unique):
        ind_str = str(ind)
        feats = compute_features_for_indication(ind_str, efo_cache, genetics_cache)
        lookup[ind_str] = feats

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            mapped = sum(1 for v in lookup.values() if v.get("ot_efo_id"))
            print(
                f"  [{i+1}/{n}]  mapped={mapped}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    return lookup


# ---------------------------------------------------------------------------
# Feature file update
# ---------------------------------------------------------------------------

def _find_latest_features(base_dir):
    candidates = glob.glob(os.path.join(base_dir, "ml_dataset_features_*.csv"))
    best, best_v, best_date = None, -1, None
    for f in candidates:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, best_date = v, f, m.group(1)
    return best, best_v, best_date


def write_features_to_dataset(lookup, verbose=True):
    """
    Add feat_ot_* columns to the latest ml_dataset_features CSV.
    Archives the previous version and saves as vN+1.
    """
    import shutil

    feat_path, feat_v, date_tag = _find_latest_features(ML_DATA_DIR)
    if feat_path is None:
        raise FileNotFoundError("No ml_dataset_features file found in " + ML_DATA_DIR)

    if verbose:
        print(f"\nReading: {os.path.basename(feat_path)}")
    df = pd.read_csv(feat_path)
    if verbose:
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")

    # Idempotency: skip if already present
    already = [c for c in OT_FEATURE_COLS if c in df.columns]
    if already:
        print(f"[idempotent] OT features already present: {already} — skipping.")
        return feat_path

    # Map features via indication lookup
    ind_col = df["indication"] if "indication" in df.columns else pd.Series("", index=df.index)

    for feat in OT_FEATURE_COLS:
        null_val = _NULL_FEATURES[feat]
        df[feat] = ind_col.map(
            lambda ind, _f=feat, _null=null_val: (
                lookup.get(str(ind), _NULL_FEATURES).get(_f, _null)
                if pd.notna(ind) else _null
            )
        )

    # Save EFO metadata as non-feature columns (useful for debugging)
    df["ot_efo_id"]   = ind_col.map(lambda ind: lookup.get(str(ind), {}).get("ot_efo_id")   if pd.notna(ind) else None)
    df["ot_efo_name"] = ind_col.map(lambda ind: lookup.get(str(ind), {}).get("ot_efo_name") if pd.notna(ind) else None)

    new_v         = feat_v + 1
    out_feat_name = f"ml_dataset_features_{date_tag}_v{new_v}.csv"
    out_feat_path = os.path.join(ML_DATA_DIR, out_feat_name)
    os.makedirs(ML_DATA_DIR, exist_ok=True)

    # Archive old version
    if os.path.dirname(os.path.abspath(feat_path)) == os.path.abspath(ML_DATA_DIR):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        shutil.move(feat_path, os.path.join(ARCHIVE_DIR, os.path.basename(feat_path)))
        if verbose:
            print(f"Archived: archive/{os.path.basename(feat_path)}")

    df.to_csv(out_feat_path, index=False)
    if verbose:
        print(f"Saved:    {out_feat_name}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        _print_feature_distributions(df)

    return out_feat_path


def _print_feature_distributions(df):
    print("\nOT feature distributions:")
    for col in OT_FEATURE_COLS:
        if col not in df.columns:
            continue
        n = df[col].notna().sum()
        pct = n / len(df) * 100
        if df[col].dtype in (float, "float64") and df[col].nunique() > 5:
            print(f"  {col}: {n}/{len(df)} ({pct:.1f}%)  mean={df[col].mean():.3f}  std={df[col].std():.3f}")
        else:
            vc = df[col].value_counts(dropna=False).head(6).to_dict()
            print(f"  {col}: {n}/{len(df)} ({pct:.1f}%)  vals={vc}")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def validation_report(lookup, df, llm_col="disease_genetic_basis"):
    """
    Print a validation report comparing OT classifications against LLM baseline.
    """
    _LLM_TO_INT = {"none": 0, "polygenic": 1, "somatic": 2, "monogenic": 3}
    _INT_TO_LABEL = {-1: "unknown", 0: "none", 1: "polygenic", 2: "somatic", 3: "monogenic"}

    # Build per-indication comparison table
    records = []
    for ind_str, feats in lookup.items():
        rows = df[df["indication"].astype(str) == ind_str]
        if rows.empty:
            continue
        llm_raw = rows[llm_col].iloc[0] if llm_col in rows.columns else None
        llm_int = _LLM_TO_INT.get(str(llm_raw).lower(), -1) if pd.notna(llm_raw) else -1
        records.append({
            "indication":    ind_str,
            "n_rows":        len(rows),
            "ot_basis":      feats["feat_ot_genetic_basis"],
            "llm_basis":     llm_int,
            "ot_label":      _INT_TO_LABEL.get(feats["feat_ot_genetic_basis"], "?"),
            "llm_label":     _INT_TO_LABEL.get(llm_int, "?"),
            "ot_efo_id":     feats.get("ot_efo_id", ""),
            "ot_mono":       feats["feat_ot_monogenic_signal"],
            "ot_gwas":       feats["feat_ot_gwas_signal"],
            "ot_somatic":    feats["feat_ot_somatic_signal"],
            "ot_evidence":   feats["feat_ot_genetic_evidence_score"],
            "ot_n_mono":     feats["feat_ot_n_genetic_targets"],
        })

    if not records:
        print("No records to compare.")
        return

    cmp = pd.DataFrame(records)
    n_total  = len(cmp)
    n_mapped = int(((cmp["ot_efo_id"] != "") & cmp["ot_efo_id"].notna()).sum())

    print("\n" + "="*70)
    print("OPEN TARGETS GENETICS — VALIDATION REPORT")
    print("="*70)
    print(f"\nIndications processed:  {n_total}")
    n_efo_mapped = ((cmp["ot_efo_id"].notna()) & (cmp["ot_efo_id"] != "")).sum()
    print(f"EFO mapped:             {n_efo_mapped}/{n_total}  ({100.0 * n_efo_mapped / n_total:.1f}%)")

    # OT classification distribution
    print("\nOT classification distribution:")
    for label, val in [("monogenic", 3), ("somatic", 2), ("polygenic", 1), ("none", 0), ("unknown", -1)]:
        ct = (cmp["ot_basis"] == val).sum()
        print(f"  {label:12s} ({val:>2}):  {ct:4d}  ({100.0*ct/n_total:.1f}%)")

    # Agreement with LLM
    both_known = cmp[(cmp["ot_basis"] >= 0) & (cmp["llm_basis"] >= 0)]
    if len(both_known) > 0:
        agree = (both_known["ot_basis"] == both_known["llm_basis"]).sum()
        print(f"\nAgreement with LLM (where both classified): {agree}/{len(both_known)} ({100.0*agree/len(both_known):.1f}%)")

        print("\nDisagreements (OT vs LLM):")
        disagree = both_known[both_known["ot_basis"] != both_known["llm_basis"]].sort_values("n_rows", ascending=False)
        for _, row in disagree.head(20).iterrows():
            print(f"  {row['indication'][:40]:<40}  OT={row['ot_label']:10s}  LLM={row['llm_label']:10s}  "
                  f"(mono={row['ot_mono']:.2f} gwas={row['ot_gwas']:.2f} somatic={row['ot_somatic']:.2f}  n={row['ot_n_mono']})")

    # NaN comparison
    llm_nan = (cmp["llm_basis"] < 0).sum()
    ot_nan  = (cmp["ot_basis"] < 0).sum()
    print(f"\nUnknown rate:  LLM={llm_nan}/{n_total} ({100.0*llm_nan/n_total:.1f}%)  "
          f"OT={ot_nan}/{n_total} ({100.0*ot_nan/n_total:.1f}%)")

    # Score distribution comparison
    print("\nfeat_ot_genetic_evidence_score distribution:")
    mapped = cmp[cmp["ot_efo_id"].notna() & (cmp["ot_efo_id"] != "")]
    if len(mapped) > 0:
        q = mapped["ot_evidence"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  p10={q[0.10]:.3f}  p25={q[0.25]:.3f}  p50={q[0.50]:.3f}  p75={q[0.75]:.3f}  p90={q[0.90]:.3f}")
        print(f"  unique values: {mapped['ot_evidence'].nunique()} (vs 5 in LLM proxy)")

    # Show unknown diseases
    unknowns = cmp[cmp["ot_basis"] == -1].sort_values("n_rows", ascending=False)
    if len(unknowns) > 0:
        print(f"\nUnknown diseases ({len(unknowns)}):")
        for _, row in unknowns.head(15).iterrows():
            efo = row["ot_efo_id"] if row["ot_efo_id"] else "no EFO"
            print(f"  {row['indication'][:45]:<45}  n_rows={row['n_rows']:3d}  efo={efo}")

    print("="*70)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_cache(path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache, path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enrich dataset with Open Targets genetics features")
    parser.add_argument("--full",           action="store_true", help="Process all unique indications (Phase 2)")
    parser.add_argument("--n",              type=int, default=50, help="Number of indications for prototype mode (default: 50)")
    parser.add_argument("--write-features", action="store_true", help="Write OT features back to ml_dataset_features CSV (requires --full)")
    parser.add_argument("--sleep",          type=float, default=0.65, help="Sleep seconds between API calls (default: 0.65)")
    args = parser.parse_args()

    if args.write_features and not args.full:
        print("WARNING: --write-features requires --full. Ignoring --write-features.")
        args.write_features = False

    # Find and load feature dataset
    feat_path, feat_v, date_tag = _find_latest_features(ML_DATA_DIR)
    if feat_path is None:
        print(f"ERROR: No ml_dataset_features file found in {ML_DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Feature dataset: {os.path.basename(feat_path)}")
    df = pd.read_csv(feat_path, usecols=["indication", "disease_genetic_basis"])
    print(f"Rows: {len(df)}, Unique indications: {df['indication'].nunique()}, NaN: {df['indication'].isna().sum()}")

    # Select indications to process
    ind_counts = df["indication"].value_counts(dropna=True)
    if args.full:
        selected = list(ind_counts.index)
        print(f"\nMode: FULL — processing all {len(selected)} unique indications")
        print(f"Estimated time: {2 * len(selected) * args.sleep / 60:.0f}–{3 * len(selected) * args.sleep / 60:.0f} minutes")
    else:
        n = min(args.n, len(ind_counts))
        selected = list(ind_counts.head(n).index)
        print(f"\nMode: PROTOTYPE — processing top {n} indications by frequency")

    # Load caches
    efo_cache      = _load_cache(EFO_CACHE_PATH)
    genetics_cache = _load_cache(GENETICS_CACHE_PATH)

    n_cached = sum(1 for s in selected if s.lower().strip() in efo_cache)
    print(f"Cache hits: {n_cached}/{len(selected)} EFO mappings already cached")

    print(f"\nProcessing {len(selected)} indications (sleep={args.sleep}s between calls)...")
    t0 = time.time()

    lookup = {}
    for i, ind in enumerate(selected):
        ind_str = str(ind)
        # Reuse cached EFO + genetics data if available
        feats = compute_features_for_indication(
            ind_str, efo_cache, genetics_cache, sleep_sec=args.sleep
        )
        lookup[ind_str] = feats

        if (i + 1) % 10 == 0 or (i + 1) == len(selected):
            mapped = sum(1 for v in lookup.values() if v.get("ot_efo_id"))
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            eta = (len(selected) - i - 1) / rate
            print(f"  [{i+1:4d}/{len(selected)}]  mapped={mapped:4d}  "
                  f"elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s", flush=True)

        # Save caches periodically
        if (i + 1) % 50 == 0:
            _save_cache(efo_cache, EFO_CACHE_PATH)
            _save_cache(genetics_cache, GENETICS_CACHE_PATH)

    # Final cache save
    _save_cache(efo_cache, EFO_CACHE_PATH)
    _save_cache(genetics_cache, GENETICS_CACHE_PATH)
    print(f"\nCaches saved: {EFO_CACHE_PATH}")

    # Validation report (compare vs LLM baseline)
    df_subset = df[df["indication"].isin(selected)]
    validation_report(lookup, df_subset)

    # Write features to dataset (Phase 2 only)
    if args.write_features:
        print("\n--- Writing OT features to feature dataset ---")
        # Reload full dataframe (we only loaded 2 cols above)
        write_features_to_dataset(lookup, verbose=True)
        print("\nDone — OT genetics features written to feature dataset.")
        print("Next: run python -m scripts.build_pre_event_train_v2 to rebuild training table.")
    else:
        print("\nPrototype complete. Run with --full --write-features to update the feature dataset.")

    elapsed_total = time.time() - t0
    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")


if __name__ == "__main__":
    main()

"""
build_ctgov_pipeline_proxies.py
================================
Part 2 — Add CT.gov sponsor and intervention aggregate pipeline features.

For each row, queries CT.gov by sponsor name and drug name to compute:

Sponsor-level (7 features):
  feat_ctgov_n_active_trials_sponsor        — count active/recruiting interventional trials
  feat_ctgov_n_late_stage_trials_sponsor     — count Phase 2/3 or 3 or 4 trials
  feat_ctgov_n_completed_trials_sponsor      — count completed interventional trials
  feat_ctgov_n_trials_total_sponsor          — total interventional trials (any status)
  feat_ctgov_pipeline_maturity_score         — composite: (late_stage + 0.5*active) / sqrt(total+1)

Intervention/asset-level:
  feat_ctgov_n_trials_same_intervention      — all interventional studies for same drug
  feat_ctgov_n_late_stage_trials_same_intervention — Phase 2/3 or 3/4 for same drug
  feat_ctgov_asset_maturity_score            — composite: late_stage_drug / sqrt(total_drug + 1)

Caching:
  cache/ctgov_sponsor_v1.json       — keyed by normalised sponsor name
  cache/ctgov_intervention_v1.json  — keyed by normalised drug name

Strategy:
  For each unique sponsor/drug: one CT.gov search call returning up to 100 results
  (most small/mid-cap biotechs have far fewer). Large-cap sponsors with >100 results
  are handled conservatively (counts capped, noted in report).
  Rows without ct_sponsor or drug_name are skipped (features set to NaN).

Usage (from biotech_catalyst_v3/):
    python -m scripts.build_ctgov_pipeline_proxies
"""

import glob
import json
import os
import re
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
BASE_DIR           = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR        = os.path.join(BASE_DIR, "archive")
ML_DATA_DIR        = os.path.join(BASE_DIR, "data", "ml")
CACHE_DIR          = os.path.join(BASE_DIR, "cache")
SPONSOR_CACHE_FILE = os.path.join(CACHE_DIR, "ctgov_sponsor_v1.json")
DRUG_CACHE_FILE    = os.path.join(CACHE_DIR, "ctgov_intervention_v1.json")

BASE_URL   = "https://clinicaltrials.gov/api/v2/studies"
RATE_LIMIT = 0.35
MAX_RETRIES = 3
PAGE_SIZE   = 100   # max per page; enough for most small biotechs

LATE_STAGE_PHASES = {"PHASE2", "PHASE3", "PHASE4", "PHASE2/PHASE3", "PHASE3/PHASE4"}
ACTIVE_STATUSES   = {
    "RECRUITING", "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING",
}

NEW_FEAT_COLS = [
    "feat_ctgov_n_active_trials_sponsor",
    "feat_ctgov_n_late_stage_trials_sponsor",
    "feat_ctgov_n_completed_trials_sponsor",
    "feat_ctgov_n_trials_total_sponsor",
    "feat_ctgov_pipeline_maturity_score",
    "feat_ctgov_n_trials_same_intervention",
    "feat_ctgov_n_late_stage_trials_same_intervention",
    "feat_ctgov_asset_maturity_score",
]

NEW_FEAT_META = [
    ("feat_ctgov_n_active_trials_sponsor",
     "Count of active/recruiting interventional trials for the same CT.gov lead sponsor",
     "ct_sponsor → CT.gov query", "deterministic"),
    ("feat_ctgov_n_late_stage_trials_sponsor",
     "Count of Phase 2/3 or Phase 3 or Phase 4 interventional trials for the sponsor",
     "ct_sponsor → CT.gov query", "deterministic"),
    ("feat_ctgov_n_completed_trials_sponsor",
     "Count of completed interventional trials for the sponsor",
     "ct_sponsor → CT.gov query", "deterministic"),
    ("feat_ctgov_n_trials_total_sponsor",
     "Total interventional trials on CT.gov for this sponsor (any status/phase, up to 100)",
     "ct_sponsor → CT.gov query", "deterministic"),
    ("feat_ctgov_pipeline_maturity_score",
     "Sponsor maturity = (n_late_stage + 0.5 * n_active) / sqrt(n_total + 1); higher = deeper late-stage pipeline",
     "feat_ctgov_n_late_stage_trials_sponsor, feat_ctgov_n_active_trials_sponsor", "deterministic"),
    ("feat_ctgov_n_trials_same_intervention",
     "Count of all interventional trials mentioning the same drug/intervention",
     "drug_name → CT.gov query.intr", "deterministic"),
    ("feat_ctgov_n_late_stage_trials_same_intervention",
     "Count of Phase 2/3+ trials for the same drug/intervention",
     "drug_name → CT.gov query.intr", "deterministic"),
    ("feat_ctgov_asset_maturity_score",
     "Drug maturity = n_late_stage_drug / sqrt(n_total_drug + 1); higher = well-developed asset",
     "feat_ctgov_n_trials_same_intervention, feat_ctgov_n_late_stage_trials_same_intervention", "deterministic"),
]


# ---------------------------------------------------------------------------
# Versioning helpers (same pattern as Part 1)
# ---------------------------------------------------------------------------

def _find_latest(base_dir, prefix, exclude_dict=True):
    """Return (path, version_int, date_tag) for highest-version file in base_dir only."""
    candidates = glob.glob(os.path.join(base_dir, f"{prefix}_*.csv"))
    if exclude_dict:
        candidates = [f for f in candidates if "dict" not in os.path.basename(f)]
    best, best_v, best_date = None, -1, None
    for f in candidates:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, best_date = v, f, m.group(1)
    return best, best_v, best_date


def _find_latest_dict(base_dir):
    """Return path of highest-version dict file in base_dir only."""
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
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, path: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# CT.gov aggregate query
# ---------------------------------------------------------------------------

def _normalise_sponsor(name: str) -> str:
    """Strip legal suffixes for more reliable matching."""
    if not name:
        return ""
    n = str(name).strip()
    n = re.sub(
        r'\s*,?\s*(Inc\.?|Corp\.?|Ltd\.?|LLC\.?|GmbH|AG|NV|BV|SA|SE|PLC|L\.?P\.?)\.?\s*$',
        "", n, flags=re.IGNORECASE
    ).strip()
    return n


def _is_late_stage(phases: list) -> bool:
    """True if any phase label indicates Phase 2/3 or later."""
    for p in phases:
        p_up = p.upper().replace(" ", "").replace("/", "/")
        # Normalise e.g. "Phase 3" → "PHASE3"
        p_norm = re.sub(r"[^A-Z0-9/]", "", p_up)
        if p_norm in {"PHASE2", "PHASE3", "PHASE4", "PHASE2/PHASE3", "PHASE3/PHASE4"}:
            return True
        # Also catch "2/3", "3", "4" alone
        if re.fullmatch(r"(2|3|4|2/3|3/4)", p_norm):
            return True
    return False


def _query_ctgov(params: dict, session: requests.Session) -> dict:
    """Single CT.gov API call with retries; returns parsed JSON dict."""
    params["format"] = "json"
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(RATE_LIMIT)
            return resp.json()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return {}
    return {}


def _parse_studies(data: dict) -> list:
    """Extract a list of (phases, status) tuples from CT.gov response."""
    out = []
    for s in data.get("studies", []):
        proto  = s.get("protocolSection", {})
        design = proto.get("designModule", {})
        status = proto.get("statusModule", {}).get("overallStatus", "")
        phases = design.get("phases", [])
        out.append((phases, status))
    return out


def get_sponsor_counts(
    sponsor: str, cache: dict, session: requests.Session
) -> dict:
    """
    Return sponsor-level counts for interventional trials.
    Uses cache keyed by normalised sponsor name.
    """
    key = _normalise_sponsor(sponsor)
    if not key:
        return {"n_total": 0, "n_active": 0, "n_late": 0, "n_completed": 0, "capped": False}
    if key in cache:
        return cache[key]

    # Query CT.gov — up to PAGE_SIZE results
    data = _query_ctgov(
        {
            "query.spons": key,
            "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
            "countTotal": "true",
            "pageSize": PAGE_SIZE,
            "fields": "protocolSection.statusModule.overallStatus,protocolSection.designModule.phases",
        },
        session,
    )

    studies = _parse_studies(data)
    total_on_ctgov = data.get("totalCount") or len(studies)
    capped = total_on_ctgov > PAGE_SIZE

    n_active    = sum(1 for _, s in studies if s in ACTIVE_STATUSES)
    n_late      = sum(1 for p, _ in studies if _is_late_stage(p))
    n_completed = sum(1 for _, s in studies if s == "COMPLETED")

    result = {
        "n_total":     int(total_on_ctgov),  # true total from CT.gov
        "n_sample":    len(studies),          # what we actually downloaded
        "n_active":    int(n_active),
        "n_late":      int(n_late),
        "n_completed": int(n_completed),
        "capped":      capped,
    }
    cache[key] = result
    return result


def get_intervention_counts(
    drug_name: str, cache: dict, session: requests.Session
) -> dict:
    """
    Return drug/intervention-level counts.
    Tries exact match; if none found tries first word (generic name prefix).
    """
    key = str(drug_name).strip().lower() if drug_name else ""
    if not key:
        return {"n_total": 0, "n_late": 0}
    if key in cache:
        return cache[key]

    def _query_drug(term):
        data = _query_ctgov(
            {
                "query.intr": term,
                "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
                "countTotal": "true",
                "pageSize": PAGE_SIZE,
                "fields": "protocolSection.statusModule.overallStatus,protocolSection.designModule.phases",
            },
            session,
        )
        return data

    data = _query_drug(drug_name)
    # If nothing, try first word (handles "drug (CODE)" patterns)
    if not data.get("studies") and " " in str(drug_name):
        data = _query_drug(drug_name.split()[0])

    studies        = _parse_studies(data)
    total_on_ctgov = data.get("totalCount") or len(studies)

    n_late = sum(1 for p, _ in studies if _is_late_stage(p))

    result = {
        "n_total": int(total_on_ctgov),
        "n_late":  int(n_late),
        "capped":  total_on_ctgov > PAGE_SIZE,
    }
    cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Score computations
# ---------------------------------------------------------------------------

def _pipeline_maturity(n_late: float, n_active: float, n_total: float) -> float:
    """Higher → more mature late-stage pipeline relative to total size."""
    return round((n_late + 0.5 * n_active) / (np.sqrt(n_total + 1)), 4)


def _asset_maturity(n_late: float, n_total: float) -> float:
    return round(n_late / (np.sqrt(n_total + 1)), 4)


# ---------------------------------------------------------------------------
# Feature dictionary update
# ---------------------------------------------------------------------------

def update_feature_dict(df: pd.DataFrame, old_dict_path, out_path: str):
    if old_dict_path and os.path.exists(old_dict_path):
        fdict = pd.read_csv(old_dict_path)
    else:
        fdict = pd.DataFrame(columns=[
            "feature_name", "stage", "feature_type", "description",
            "source_columns", "source_type", "n_valid", "n_null", "pct_valid",
        ])

    new_names = [m[0] for m in NEW_FEAT_META]
    fdict = fdict[~fdict["feature_name"].isin(new_names)]

    rows = []
    for feat_name, desc, src_cols, src_type in NEW_FEAT_META:
        if feat_name not in df.columns:
            continue
        n_valid = int(df[feat_name].notna().sum())
        n_null  = int(df[feat_name].isna().sum())
        rows.append({
            "feature_name":   feat_name,
            "stage":          "pass7_ctgov_pipeline",
            "feature_type":   "feat",
            "description":    desc,
            "source_columns": src_cols,
            "source_type":    src_type,
            "n_valid":        n_valid,
            "n_null":         n_null,
            "pct_valid":      round(n_valid / len(df) * 100, 1),
        })

    fdict = pd.concat([fdict, pd.DataFrame(rows)], ignore_index=True)
    fdict.to_csv(out_path, index=False)
    print(f"Feature dict: {len(fdict)} entries → {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Find latest feature dataset ───────────────────────────────────────────
    os.makedirs(ML_DATA_DIR, exist_ok=True)
    src_path, src_v, date_tag = _find_latest(ML_DATA_DIR, "ml_dataset_features")
    if not src_path:
        print("ERROR: no ml_dataset_features_YYYYMMDD_vN.csv found in " + ML_DATA_DIR, file=sys.stderr)
        sys.exit(1)
    print(f"Input : {os.path.basename(src_path)}  (v{src_v})")
    df = pd.read_csv(src_path)
    print(f"Shape : {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Load caches ───────────────────────────────────────────────────────────
    sponsor_cache = _load_cache(SPONSOR_CACHE_FILE)
    drug_cache    = _load_cache(DRUG_CACHE_FILE)
    session       = requests.Session()

    # ── Unique sponsors ───────────────────────────────────────────────────────
    sponsor_col = "ct_sponsor" if "ct_sponsor" in df.columns else None
    drug_col    = "drug_name"  if "drug_name"  in df.columns else None

    uniq_sponsors  = (
        df[sponsor_col].dropna().unique().tolist() if sponsor_col else []
    )
    uniq_drugs = (
        df[drug_col].dropna().unique().tolist() if drug_col else []
    )

    # Filter to sponsors not already cached
    to_fetch_sponsors = [
        s for s in uniq_sponsors
        if _normalise_sponsor(s) not in sponsor_cache and _normalise_sponsor(s)
    ]
    to_fetch_drugs = [
        d for d in uniq_drugs
        if str(d).strip().lower() not in drug_cache and str(d).strip()
    ]

    print(f"\nSponsors : {len(uniq_sponsors)} unique, "
          f"{len(uniq_sponsors) - len(to_fetch_sponsors)} cached, "
          f"{len(to_fetch_sponsors)} to fetch")
    print(f"Drug names: {len(uniq_drugs)} unique, "
          f"{len(uniq_drugs) - len(to_fetch_drugs)} cached, "
          f"{len(to_fetch_drugs)} to fetch")

    # ── Fetch sponsor counts ──────────────────────────────────────────────────
    print(f"\nFetching sponsor pipeline counts ...")
    for i, sponsor in enumerate(to_fetch_sponsors):
        if i % 25 == 0 and i > 0:
            _save_cache(sponsor_cache, SPONSOR_CACHE_FILE)
            print(f"  ... {i}/{len(to_fetch_sponsors)} sponsors fetched")
        get_sponsor_counts(sponsor, sponsor_cache, session)

    _save_cache(sponsor_cache, SPONSOR_CACHE_FILE)
    print(f"Sponsor cache: {len(sponsor_cache)} entries saved")

    # ── Fetch drug/intervention counts ────────────────────────────────────────
    print(f"\nFetching drug/intervention counts ...")
    for i, drug in enumerate(to_fetch_drugs):
        if i % 50 == 0 and i > 0:
            _save_cache(drug_cache, DRUG_CACHE_FILE)
            print(f"  ... {i}/{len(to_fetch_drugs)} drugs fetched")
        get_intervention_counts(drug, drug_cache, session)

    _save_cache(drug_cache, DRUG_CACHE_FILE)
    print(f"Drug cache: {len(drug_cache)} entries saved")

    # ── Build row-level features ──────────────────────────────────────────────
    print("\nBuilding pipeline proxy features ...")
    rows = []
    for _, row in df.iterrows():
        sponsor  = row.get(sponsor_col, "") if sponsor_col else ""
        drug     = row.get(drug_col,    "") if drug_col    else ""
        out      = {}

        # ── Sponsor features ──────────────────────────────────────────────────
        if sponsor and not pd.isna(sponsor):
            sc = get_sponsor_counts(str(sponsor), sponsor_cache, session)
            out["feat_ctgov_n_active_trials_sponsor"]    = float(sc.get("n_active",    0))
            out["feat_ctgov_n_late_stage_trials_sponsor"] = float(sc.get("n_late",     0))
            out["feat_ctgov_n_completed_trials_sponsor"] = float(sc.get("n_completed", 0))
            out["feat_ctgov_n_trials_total_sponsor"]     = float(sc.get("n_total",     0))
            out["feat_ctgov_pipeline_maturity_score"]    = _pipeline_maturity(
                sc.get("n_late", 0), sc.get("n_active", 0), sc.get("n_total", 0)
            )
        else:
            for c in ["feat_ctgov_n_active_trials_sponsor",
                      "feat_ctgov_n_late_stage_trials_sponsor",
                      "feat_ctgov_n_completed_trials_sponsor",
                      "feat_ctgov_n_trials_total_sponsor",
                      "feat_ctgov_pipeline_maturity_score"]:
                out[c] = np.nan

        # ── Drug features ─────────────────────────────────────────────────────
        if drug and not pd.isna(drug):
            dc = get_intervention_counts(str(drug), drug_cache, session)
            out["feat_ctgov_n_trials_same_intervention"]             = float(dc.get("n_total", 0))
            out["feat_ctgov_n_late_stage_trials_same_intervention"]  = float(dc.get("n_late",  0))
            out["feat_ctgov_asset_maturity_score"]                   = _asset_maturity(
                dc.get("n_late", 0), dc.get("n_total", 0)
            )
        else:
            for c in ["feat_ctgov_n_trials_same_intervention",
                      "feat_ctgov_n_late_stage_trials_same_intervention",
                      "feat_ctgov_asset_maturity_score"]:
                out[c] = np.nan

        rows.append(out)

    feat_df = pd.DataFrame(rows)

    # ── Merge into df ─────────────────────────────────────────────────────────
    existing = [c for c in NEW_FEAT_COLS if c in df.columns]
    df = df.drop(columns=existing)
    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # ── Coverage ──────────────────────────────────────────────────────────────
    print("\n── CT.gov pipeline feature coverage ─────────────────────────────────")
    for col in NEW_FEAT_COLS:
        if col in df.columns:
            n = df[col].notna().sum()
            med = df[col].dropna().median() if df[col].dtype != object else "—"
            print(f"  {col:<52}  {n:4d}/{len(df)}  median={med}")

    # ── Large sponsors check ──────────────────────────────────────────────────
    capped = {k: v for k, v in sponsor_cache.items() if v.get("capped")}
    if capped:
        print(f"\n  Note: {len(capped)} sponsors have >100 trials on CT.gov "
              f"(counts capped at first 100 sample):")
        for k in list(capped)[:5]:
            print(f"    {k}: {capped[k].get('n_total')} total")

    # ── Archive and save ──────────────────────────────────────────────────────
    new_v    = src_v + 1
    out_name = f"ml_dataset_features_{date_tag}_v{new_v}.csv"
    out_path = os.path.join(ML_DATA_DIR, out_name)

    dest = os.path.join(ARCHIVE_DIR, os.path.basename(src_path))
    shutil.move(src_path, dest)
    print(f"Archived: archive/{os.path.basename(src_path)}")

    df.to_csv(out_path, index=False)
    print(f"\nSaved : {out_name}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # ── Update feature dict ───────────────────────────────────────────────────
    old_dict = _find_latest_dict(ML_DATA_DIR)
    if old_dict:
        shutil.move(old_dict, os.path.join(ARCHIVE_DIR, os.path.basename(old_dict)))

    out_dict_name = f"ml_feature_dict_{date_tag}_v{new_v}.csv"
    update_feature_dict(df, os.path.join(ARCHIVE_DIR, os.path.basename(old_dict)) if old_dict else None,
                        os.path.join(ML_DATA_DIR, out_dict_name))
    print(f"Saved : {out_dict_name}")

    print("\nDone — CT.gov pipeline proxy features added.")
    return out_name, out_dict_name


if __name__ == "__main__":
    main()

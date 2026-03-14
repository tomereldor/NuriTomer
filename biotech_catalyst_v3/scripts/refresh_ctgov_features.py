"""
refresh_ctgov_features.py
==========================
Part 1 — Replace weak dataset-history timing proxies with CT.gov-grounded features.

For each row with a valid nct_id, queries ClinicalTrials.gov v2 API and
refreshes/adds the following row-level features:

  feat_ctgov_primary_completion_date  — raw date string from CT.gov (metadata)
  feat_days_to_primary_completion     — recomputed from CT.gov date vs event date
  feat_primary_completion_imminent_30d — 1 if within 30 days of event
  feat_primary_completion_imminent_90d — 1 if within 90 days of event
  feat_completion_recency_bucket      — categorical (imminent_0_30 / near_31_90 /
                                         medium_91_180 / far_180_plus / past / unknown)
  feat_ct_status_current              — normalized CT.gov overallStatus
  feat_active_not_recruiting_flag     — 1 if ACTIVE_NOT_RECRUITING
  feat_completed_flag                 — 1 if COMPLETED / PRIMARY_COMPLETION_COMPLETED
  feat_days_since_ctgov_last_update   — days from lastUpdatePostDate to event date
  feat_recent_ctgov_update_flag       — 1 if CT.gov updated within 90 days before event
  feat_status_timing_consistency_flag — 1 if status/timing combo looks plausible

Caches all CT.gov responses to cache/ctgov_details_v1.json.
Rows without nct_id are skipped.
Uses v_actual_date if available, else event_date as anchor.

Usage (from biotech_catalyst_v3/):
    python -m scripts.refresh_ctgov_features
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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
CACHE_FILE  = os.path.join(CACHE_DIR, "ctgov_details_v1.json")

BASE_URL    = "https://clinicaltrials.gov/api/v2/studies"
RATE_LIMIT  = 0.35   # seconds between API calls
MAX_RETRIES = 3

NEW_FEAT_COLS = [
    "feat_ctgov_primary_completion_date",
    "feat_days_to_primary_completion",
    "feat_primary_completion_imminent_30d",
    "feat_primary_completion_imminent_90d",
    "feat_completion_recency_bucket",
    "feat_ct_status_current",
    "feat_active_not_recruiting_flag",
    "feat_completed_flag",
    "feat_days_since_ctgov_last_update",
    "feat_recent_ctgov_update_flag",
    "feat_status_timing_consistency_flag",
]

# CT.gov status → normalised string
STATUS_MAP = {
    "RECRUITING":                   "recruiting",
    "ACTIVE_NOT_RECRUITING":        "active_not_recruiting",
    "COMPLETED":                    "completed",
    "TERMINATED":                   "terminated",
    "WITHDRAWN":                    "withdrawn",
    "NOT_YET_RECRUITING":           "not_yet_recruiting",
    "ENROLLING_BY_INVITATION":      "enrolling_by_invitation",
    "SUSPENDED":                    "suspended",
    "UNKNOWN_STATUS":               "unknown",
    "AVAILABLE":                    "available",
    "NO_LONGER_AVAILABLE":          "no_longer_available",
    "TEMPORARILY_NOT_AVAILABLE":    "temporarily_not_available",
}

COMPLETED_STATUSES = {
    "COMPLETED", "PRIMARY_COMPLETION_COMPLETED",
}

ACTIVE_STATUSES = {
    "RECRUITING", "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING",
}


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------

def _find_latest(base_dir, archive_dir, prefix):
    candidates = (
        glob.glob(os.path.join(base_dir,    f"{prefix}_*.csv")) +
        glob.glob(os.path.join(archive_dir, f"{prefix}_*.csv"))
    )
    candidates = [f for f in candidates if "dict" not in os.path.basename(f)]
    best, best_v = None, -1
    for f in candidates:
        m = re.search(r"_v(\d+(?:\.\d+)?)_", f)
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                v = 0
            if v > best_v:
                best_v, best = v, f
    return best, best_v


def _find_latest_dict(base_dir, archive_dir, prefix):
    candidates = (
        glob.glob(os.path.join(base_dir,    f"{prefix}_*.csv")) +
        glob.glob(os.path.join(archive_dir, f"{prefix}_*.csv"))
    )
    best, best_v = None, -1
    for f in candidates:
        m = re.search(r"_v(\d+(?:\.\d+)?)_", f)
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                v = 0
            if v > best_v:
                best_v, best = v, f
    return best


def _next_semver(current_version_float: float) -> str:
    """0.3 → '0.4',  0.9 → '0.10',  1.0 → '1.1'."""
    major = int(current_version_float)
    minor = round((current_version_float - major) * 10)
    return f"{major}.{minor + 1}"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# CT.gov fetch
# ---------------------------------------------------------------------------

def fetch_study_raw(nct_id: str, cache: dict, session: requests.Session) -> dict:
    """Return raw CT.gov JSON for nct_id, using cache. Returns None on failure."""
    if nct_id in cache:
        return cache[nct_id]

    url = f"{BASE_URL}/{nct_id}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params={"format": "json"}, timeout=30)
            if resp.status_code == 404:
                cache[nct_id] = None
                return None
            resp.raise_for_status()
            data = resp.json()
            cache[nct_id] = data
            time.sleep(RATE_LIMIT)
            return data
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED {nct_id}: {e}")
                cache[nct_id] = None
                return None
    return None


def extract_fields(raw: dict) -> dict:
    """Extract timing/status fields from raw CT.gov study JSON."""
    if raw is None:
        return {}
    proto      = raw.get("protocolSection", {})
    status_mod = proto.get("statusModule", {})

    def _date(key):
        return status_mod.get(key, {}).get("date", "")

    return {
        "primary_completion_date":  _date("primaryCompletionDateStruct"),
        "completion_date":          _date("completionDateStruct"),
        "last_update_post_date":    _date("lastUpdatePostDateStruct"),
        "study_first_post_date":    _date("studyFirstPostDateStruct"),
        "overall_status":           status_mod.get("overallStatus", ""),
    }


# ---------------------------------------------------------------------------
# Date parsing (CT.gov uses "2024-01", "2024-01-15", "January 2024", etc.)
# ---------------------------------------------------------------------------

def _parse_ctgov_date(d) -> pd.Timestamp:
    if not d or not isinstance(d, str):
        return pd.NaT
    d = d.strip()
    for fmt in ["%Y-%m-%d", "%Y-%m", "%B %Y", "%Y"]:
        try:
            return pd.to_datetime(d, format=fmt)
        except Exception:
            pass
    return pd.to_datetime(d, errors="coerce")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _completion_bucket(days) -> str:
    if pd.isna(days):
        return "unknown"
    if days < 0:
        return "past"
    if days <= 30:
        return "imminent_0_30"
    if days <= 90:
        return "near_31_90"
    if days <= 180:
        return "medium_91_180"
    return "far_180_plus"


def _status_consistency(status: str, days_to_completion) -> float:
    """
    1 = plausible, 0 = inconsistent, NaN = unknown.
    Heuristics:
    - COMPLETED but completion is >1 year in the future → inconsistent
    - RECRUITING but completion is >3 years in the past  → inconsistent
    - TERMINATED / WITHDRAWN but we have an event → mild inconsistency flagged
    - Otherwise consistent
    """
    if not status or pd.isna(days_to_completion):
        return np.nan
    d = float(days_to_completion)
    if status == "COMPLETED" and d > 365:
        return 0.0
    if status in ACTIVE_STATUSES and d < -1095:   # 3 years past
        return 0.0
    if status in {"TERMINATED", "WITHDRAWN"}:
        return 0.0
    return 1.0


def compute_row_features(fields: dict, evt_date: pd.Timestamp) -> dict:
    """Given extracted CT.gov fields and the event date, compute all new features."""
    out = {}

    # ── Primary completion ────────────────────────────────────────────────────
    pc_date = _parse_ctgov_date(fields.get("primary_completion_date", ""))
    out["feat_ctgov_primary_completion_date"] = (
        fields.get("primary_completion_date", "") or ""
    )

    if pd.notna(pc_date) and pd.notna(evt_date):
        days = (pc_date - evt_date).days
        out["feat_days_to_primary_completion"]     = float(days)
        out["feat_primary_completion_imminent_30d"] = 1.0 if 0 <= days <= 30  else 0.0
        out["feat_primary_completion_imminent_90d"] = 1.0 if 0 <= days <= 90  else 0.0
        out["feat_completion_recency_bucket"]       = _completion_bucket(days)
    else:
        out["feat_days_to_primary_completion"]     = np.nan
        out["feat_primary_completion_imminent_30d"] = np.nan
        out["feat_primary_completion_imminent_90d"] = np.nan
        out["feat_completion_recency_bucket"]       = "unknown"

    # ── Status ────────────────────────────────────────────────────────────────
    raw_status = fields.get("overall_status", "")
    out["feat_ct_status_current"] = STATUS_MAP.get(raw_status, raw_status.lower() if raw_status else "")
    out["feat_active_not_recruiting_flag"] = 1.0 if raw_status == "ACTIVE_NOT_RECRUITING" else 0.0
    out["feat_completed_flag"]             = 1.0 if raw_status in COMPLETED_STATUSES   else 0.0

    # ── Last update ───────────────────────────────────────────────────────────
    lu_date = _parse_ctgov_date(fields.get("last_update_post_date", ""))
    if pd.notna(lu_date) and pd.notna(evt_date):
        days_since = (evt_date - lu_date).days
        out["feat_days_since_ctgov_last_update"] = float(days_since)
        out["feat_recent_ctgov_update_flag"]     = 1.0 if 0 <= days_since <= 90 else 0.0
    else:
        out["feat_days_since_ctgov_last_update"] = np.nan
        out["feat_recent_ctgov_update_flag"]     = np.nan

    # ── Consistency ───────────────────────────────────────────────────────────
    out["feat_status_timing_consistency_flag"] = _status_consistency(
        raw_status, out["feat_days_to_primary_completion"]
    )

    return out


# ---------------------------------------------------------------------------
# Feature dictionary update
# ---------------------------------------------------------------------------

NEW_FEAT_META = [
    ("feat_ctgov_primary_completion_date",
     "Raw primary completion date string from CT.gov (metadata, not for model)",
     "nct_id → CT.gov primaryCompletionDateStruct", "deterministic"),
    ("feat_days_to_primary_completion",
     "Days from event date to CT.gov primary completion date; negative = event after completion",
     "nct_id → CT.gov primaryCompletionDateStruct, v_actual_date", "deterministic"),
    ("feat_primary_completion_imminent_30d",
     "1 if CT.gov primary completion is within 0–30 days of event date",
     "feat_days_to_primary_completion", "deterministic"),
    ("feat_primary_completion_imminent_90d",
     "1 if CT.gov primary completion is within 0–90 days of event date",
     "feat_days_to_primary_completion", "deterministic"),
    ("feat_completion_recency_bucket",
     "Bucket: imminent_0_30 / near_31_90 / medium_91_180 / far_180_plus / past / unknown",
     "feat_days_to_primary_completion", "deterministic"),
    ("feat_ct_status_current",
     "Normalised CT.gov overallStatus at time of data fetch",
     "nct_id → CT.gov overallStatus", "deterministic"),
    ("feat_active_not_recruiting_flag",
     "1 if CT.gov status is ACTIVE_NOT_RECRUITING",
     "feat_ct_status_current", "deterministic"),
    ("feat_completed_flag",
     "1 if CT.gov status is COMPLETED",
     "feat_ct_status_current", "deterministic"),
    ("feat_days_since_ctgov_last_update",
     "Days between CT.gov lastUpdatePostDate and event date; positive = update was before event",
     "nct_id → CT.gov lastUpdatePostDateStruct, v_actual_date", "deterministic"),
    ("feat_recent_ctgov_update_flag",
     "1 if CT.gov was updated within 90 days before the event date (activity signal)",
     "feat_days_since_ctgov_last_update", "deterministic"),
    ("feat_status_timing_consistency_flag",
     "1 if status/timing combo looks plausible; 0 if inconsistent (e.g. COMPLETED but far future date)",
     "feat_ct_status_current, feat_days_to_primary_completion", "deterministic"),
]


def update_feature_dict(df: pd.DataFrame, old_dict_path, out_path: str):
    if old_dict_path and os.path.exists(old_dict_path):
        fdict = pd.read_csv(old_dict_path)
    else:
        fdict = pd.DataFrame(columns=[
            "feature_name", "stage", "feature_type", "description",
            "source_columns", "source_type", "n_valid", "n_null", "pct_valid",
        ])

    # Remove stale entries for features we're creating/updating
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
            "stage":          "pass6_ctgov_timing",
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
    return fdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Find source ───────────────────────────────────────────────────────────
    src_path, src_v = _find_latest(BASE_DIR, ARCHIVE_DIR, "ml_dataset_features")
    if not src_path:
        print("ERROR: no ml_dataset_features_*.csv found", file=sys.stderr)
        sys.exit(1)

    # If source is in archive, restore it
    if ARCHIVE_DIR in src_path:
        dst = os.path.join(BASE_DIR, os.path.basename(src_path))
        shutil.copy(src_path, dst)
        src_path = dst
        print(f"Restored from archive: {os.path.basename(src_path)}")

    print(f"Input : {os.path.basename(src_path)}  (v{src_v})")
    df = pd.read_csv(src_path)
    print(f"Shape : {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Best event date ───────────────────────────────────────────────────────
    date_col = "v_actual_date" if "v_actual_date" in df.columns else "event_date"
    df["_evt_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # ── Filter to rows with nct_id ────────────────────────────────────────────
    mask_nct = df["nct_id"].notna() & (df["nct_id"].str.strip() != "")
    n_with   = mask_nct.sum()
    n_uniq   = df.loc[mask_nct, "nct_id"].nunique()
    print(f"Rows with nct_id: {n_with}/{len(df)}  ({n_uniq} unique IDs)")

    # ── Load cache ────────────────────────────────────────────────────────────
    cache  = load_cache()
    n_cached = sum(1 for k in df.loc[mask_nct, "nct_id"].unique() if k in cache)
    print(f"Cache: {len(cache)} total entries, {n_cached}/{n_uniq} already cached for this dataset")

    # ── Fetch missing NCT IDs ─────────────────────────────────────────────────
    session   = requests.Session()
    to_fetch  = [nct for nct in df.loc[mask_nct, "nct_id"].unique() if nct not in cache]
    print(f"Fetching {len(to_fetch)} uncached NCT IDs from CT.gov ...")

    for i, nct_id in enumerate(to_fetch):
        if i % 50 == 0 and i > 0:
            save_cache(cache)
            print(f"  ... {i}/{len(to_fetch)} fetched, cache saved")
        fetch_study_raw(nct_id, cache, session)

    save_cache(cache)
    print(f"Fetching complete. Cache now: {len(cache)} entries")

    # ── Compute features row-by-row ───────────────────────────────────────────
    print("\nComputing CT.gov timing features ...")
    feat_records = []
    for _, row in df.iterrows():
        nct_id   = row.get("nct_id")
        evt_date = row["_evt_date"]

        if not nct_id or pd.isna(nct_id) or str(nct_id).strip() == "":
            feat_records.append({c: np.nan for c in NEW_FEAT_COLS})
            continue

        raw    = cache.get(str(nct_id))
        fields = extract_fields(raw) if raw else {}
        feats  = compute_row_features(fields, evt_date)

        # Fill missing keys
        for c in NEW_FEAT_COLS:
            if c not in feats:
                feats[c] = np.nan
        feat_records.append(feats)

    feat_df = pd.DataFrame(feat_records)

    # ── Merge features into df ────────────────────────────────────────────────
    # Drop existing stale versions of these columns first
    cols_to_drop = [c for c in NEW_FEAT_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop + ["_evt_date"])
    df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    # ── Coverage report ───────────────────────────────────────────────────────
    print("\n── CT.gov timing feature coverage ──────────────────────────────────")
    for col in NEW_FEAT_COLS:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col:<48}  {n:4d}/{len(df)}  ({n/len(df)*100:.1f}%)")

    # ── Build new version tag ─────────────────────────────────────────────────
    new_v    = _next_semver(src_v)
    date_tag = "20260313"
    new_name = f"ml_dataset_features_v{new_v}_{date_tag}.csv"
    out_path = os.path.join(BASE_DIR, new_name)

    # ── Archive current version in base dir ──────────────────────────────────
    for f in glob.glob(os.path.join(BASE_DIR, "ml_dataset_features_v*.csv")):
        dest = os.path.join(ARCHIVE_DIR, os.path.basename(f))
        if not os.path.exists(dest):
            shutil.move(f, dest)
        else:
            os.remove(f)
        print(f"Archived: archive/{os.path.basename(f)}")

    df.to_csv(out_path, index=False)
    print(f"\nSaved : {new_name}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # ── Update feature dict ───────────────────────────────────────────────────
    old_dict = _find_latest_dict(BASE_DIR, ARCHIVE_DIR, "ml_feature_dict")
    for f in glob.glob(os.path.join(BASE_DIR, "ml_feature_dict_v*.csv")):
        dest = os.path.join(ARCHIVE_DIR, os.path.basename(f))
        if not os.path.exists(dest):
            shutil.move(f, dest)
        else:
            os.remove(f)

    new_dict_name = f"ml_feature_dict_v{new_v}_{date_tag}.csv"
    out_dict_path = os.path.join(BASE_DIR, new_dict_name)
    update_feature_dict(df, old_dict, out_dict_path)
    print(f"Saved : {new_dict_name}")

    print("\nDone — CT.gov timing features refreshed.")
    return new_name, new_dict_name


if __name__ == "__main__":
    main()

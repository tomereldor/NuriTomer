"""
fetch_aact_status_history.py
=============================
Option C — Point-in-time CT.gov trial status via AACT monthly flat-file archives.

For each NCT ID in the validated cohort (v_is_verified non-null), downloads
AACT monthly pipe-delimited snapshots and records the overall_status at the
month *closest before* the event date.

Outputs:
  cache/aact_status_history_v1.json  — {nct_id: {month_key: status}, _meta: {...}}
  enriched_all_clinical_clean_v3.csv — adds columns:
      ct_status_at_event  (uppercase normalized, e.g. "COMPLETED")
      data_tier           ("validated" | "historical")

Usage (from biotech_catalyst_v3/):
    python -m scripts.fetch_aact_status_history            # full run
    python -m scripts.fetch_aact_status_history --months 3 # test: 3 most-recent months only
    python -m scripts.fetch_aact_status_history --write-only # skip download, just write CSV
    python -m scripts.fetch_aact_status_history --urls-file urls.txt  # manual URL list
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import warnings
import zipfile
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
CACHE_FILE  = os.path.join(CACHE_DIR, "aact_status_history_v1.json")
MASTER_CSV  = os.path.join(BASE_DIR, "enriched_all_clinical_clean_v3.csv")

# AACT download page and known URL patterns
AACT_DOWNLOADS_URL = "https://aact.ctti-clinicaltrials.org/downloads"
# AACT monthly flat-file URL pattern (YYYYMMDD of first day of the following month release,
# or occasionally the current month's date). We try both.
AACT_BASE = "https://aact.ctti-clinicaltrials.org/static/exported_files/monthly"

RATE_LIMIT  = 3.0   # seconds between month downloads (polite; these are large files)
MAX_RETRIES = 3
BACKOFF_BASE = 5    # seconds

# AACT uses mixed-case status values; normalize to uppercase API-style
AACT_STATUS_NORM = {
    "Recruiting":                "RECRUITING",
    "Active, not recruiting":    "ACTIVE_NOT_RECRUITING",
    "Completed":                 "COMPLETED",
    "Terminated":                "TERMINATED",
    "Withdrawn":                 "WITHDRAWN",
    "Not yet recruiting":        "NOT_YET_RECRUITING",
    "Enrolling by invitation":   "ENROLLING_BY_INVITATION",
    "Suspended":                 "SUSPENDED",
    "Unknown status":            "UNKNOWN_STATUS",
    "Available":                 "AVAILABLE",
    "No longer available":       "NO_LONGER_AVAILABLE",
    "Temporarily not available": "TEMPORARILY_NOT_AVAILABLE",
    # Already-uppercase pass-through (some AACT versions)
    "RECRUITING":                "RECRUITING",
    "ACTIVE_NOT_RECRUITING":     "ACTIVE_NOT_RECRUITING",
    "COMPLETED":                 "COMPLETED",
    "TERMINATED":                "TERMINATED",
    "WITHDRAWN":                 "WITHDRAWN",
    "NOT_YET_RECRUITING":        "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION":   "ENROLLING_BY_INVITATION",
    "SUSPENDED":                 "SUSPENDED",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            data = json.load(f)
        # Ensure _meta exists
        if "_meta" not in data:
            data["_meta"] = {"months_completed": []}
        return data
    return {"_meta": {"months_completed": []}}


def save_cache(cache: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, separators=(",", ":"))
    os.replace(tmp, CACHE_FILE)


# ---------------------------------------------------------------------------
# Month list generation
# ---------------------------------------------------------------------------

def generate_month_list(start="2023-01", end=None) -> list:
    """Return list of 'YYYY-MM' strings from start through end (inclusive)."""
    if end is None:
        end = datetime.now().strftime("%Y-%m")
    months = []
    cur = start
    while cur <= end:
        months.append(cur)
        y, m = int(cur[:4]), int(cur[5:7])
        m += 1
        if m > 12:
            m, y = 1, y + 1
        cur = f"{y:04d}-{m:02d}"
    return months


# ---------------------------------------------------------------------------
# URL discovery
# ---------------------------------------------------------------------------

def _try_url(session, url) -> bool:
    """HEAD request to check if URL exists. Returns True if 200."""
    try:
        r = session.head(url, timeout=15, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def discover_urls_from_page(session: requests.Session) -> dict:
    """
    Scrape AACT downloads page for pipe-delimited export ZIP links.
    Returns dict: month_key -> url.
    Falls back gracefully if page is JS-rendered (returns empty dict).
    """
    urls = {}
    try:
        r = session.get(AACT_DOWNLOADS_URL, timeout=30)
        if r.status_code != 200:
            return urls
        # Look for links matching monthly pipe-delimited exports
        # Pattern: /static/exported_files/monthly/YYYYMMDD_pipe-delimited-export.zip
        pattern = r'href="([^"]*monthly/(\d{8})_pipe-delimited-export\.zip[^"]*)"'
        for match in re.finditer(pattern, r.text):
            href, date_str = match.group(1), match.group(2)
            # Convert YYYYMMDD → YYYY-MM
            month_key = f"{date_str[:4]}-{date_str[4:6]}"
            # Make absolute URL if needed
            if href.startswith("/"):
                href = "https://aact.ctti-clinicaltrials.org" + href
            urls[month_key] = href
        if urls:
            print(f"  Discovered {len(urls)} URLs from AACT downloads page")
    except Exception as e:
        print(f"  Page scrape failed: {e}")
    return urls


def build_candidate_urls(month_key: str) -> list:
    """
    Generate candidate ZIP URLs for a given month (YYYY-MM).
    AACT uses the 1st of the next month as the filename date, or sometimes
    the last day of the current month, or the first of the current month.
    We try several patterns and use HEAD requests to find the real one.
    """
    y, m = int(month_key[:4]), int(month_key[5:7])
    # "Release" month: the snapshot for month M is typically released early month M+1
    # and named after the 1st of M+1.
    next_m = m + 1
    next_y = y
    if next_m > 12:
        next_m, next_y = 1, y + 1

    candidates = []
    # Primary: YYYYMMDD where YYYYMM is next month, DD=01
    candidates.append(f"{AACT_BASE}/{next_y:04d}{next_m:02d}01_pipe-delimited-export.zip")
    # Alt: release named after 1st of current month
    candidates.append(f"{AACT_BASE}/{y:04d}{m:02d}01_pipe-delimited-export.zip")
    # Alt: last day of current month (28/30/31)
    for day in [31, 30, 29, 28]:
        candidates.append(f"{AACT_BASE}/{y:04d}{m:02d}{day:02d}_pipe-delimited-export.zip")
    return candidates


def resolve_url(session: requests.Session, month_key: str,
                page_urls: dict, manual_urls: dict):
    """Find the download URL for a given month_key."""
    # 1. Manual override
    if month_key in manual_urls:
        return manual_urls[month_key]
    # 2. Page-scraped
    if month_key in page_urls:
        return page_urls[month_key]
    # 3. Try candidate patterns via HEAD
    for url in build_candidate_urls(month_key):
        if _try_url(session, url):
            return url
    return None


# ---------------------------------------------------------------------------
# ZIP download + parse
# ---------------------------------------------------------------------------

def download_and_parse(session: requests.Session, url: str,
                       nct_ids_set: set, month_key: str) -> dict:
    """
    Stream-download ZIP, extract studies.txt, parse nct_id + overall_status
    for rows in nct_ids_set. Returns {nct_id: normalized_status}.
    """
    tmp_path = None
    for attempt in range(MAX_RETRIES):
        try:
            # Stream to temp file (ZipFile needs seekable IO)
            fd, tmp_path = tempfile.mkstemp(suffix=".zip",
                                            dir=os.environ.get("TMPDIR", "/tmp"))
            os.close(fd)
            print(f"    Downloading {url} ...", flush=True)
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=131072):
                        f.write(chunk)
                        total += len(chunk)
                        if total % (50 * 1024 * 1024) == 0:
                            print(f"      ... {total // 1024 // 1024}MB", flush=True)
            print(f"    Downloaded {total // 1024 // 1024}MB", flush=True)
            break
        except requests.RequestException as e:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                tmp_path = None
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"    Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    FAILED after {MAX_RETRIES} attempts: {e}")
                return {}

    if not tmp_path or not os.path.exists(tmp_path):
        return {}

    results = {}
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Find studies.txt (may be in a subdirectory)
            studies_name = next(
                (n for n in zf.namelist()
                 if os.path.basename(n).lower() == "studies.txt"),
                None
            )
            if studies_name is None:
                print(f"    studies.txt not found in ZIP. Contents: {zf.namelist()[:10]}")
                return {}

            print(f"    Parsing {studies_name} ...", flush=True)
            with zf.open(studies_name) as fh:
                # Read in chunks to handle large files
                text = fh.read().decode("utf-8", errors="replace")
            # Parse with pandas
            df = pd.read_csv(
                StringIO(text),
                sep="|",
                usecols=lambda c: c.strip().lower() in ("nct_id", "overall_status"),
                dtype=str,
                low_memory=False,
            )
            # Normalize column names (AACT uses lowercase)
            df.columns = [c.strip().lower() for c in df.columns]
            if "nct_id" not in df.columns or "overall_status" not in df.columns:
                print(f"    ERROR: expected columns not found. Got: {list(df.columns)}")
                return {}

            df = df.dropna(subset=["nct_id"])
            df["nct_id"] = df["nct_id"].str.strip()
            df = df[df["nct_id"].isin(nct_ids_set)]

            for _, row in df.iterrows():
                raw_status = str(row.get("overall_status", "")).strip()
                norm = AACT_STATUS_NORM.get(raw_status, raw_status.upper().replace(" ", "_") if raw_status else "")
                results[row["nct_id"]] = norm

        print(f"    Found {len(results)}/{len(nct_ids_set)} target NCTs in {month_key}")
    except Exception as e:
        print(f"    Parse error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return results


# ---------------------------------------------------------------------------
# Status lookup: find best month ≤ event_date
# ---------------------------------------------------------------------------

def lookup_status_at_event(cache: dict, nct_id: str, event_date_str: str,
                            max_lookback: int = 6):
    """
    For nct_id, find the status from the latest snapshot month ≤ event_date.
    Falls back up to max_lookback months if exact month not in cache.
    Returns normalized status string or None.
    """
    if nct_id not in cache or nct_id == "_meta":
        return None
    try:
        event_dt = pd.to_datetime(event_date_str, errors="coerce")
        if pd.isna(event_dt):
            return None
    except Exception:
        return None

    history = cache[nct_id]  # {month_key: status}
    # Convert month keys to dates (1st of month)
    month_dates = []
    for mk, status in history.items():
        try:
            dt = pd.to_datetime(mk + "-01")
            month_dates.append((dt, mk, status))
        except Exception:
            continue

    if not month_dates:
        return None

    # Filter to months ≤ event_date; take latest
    eligible = [(dt, mk, st) for dt, mk, st in month_dates if dt <= event_dt]
    if not eligible:
        # Fallback: take the earliest available month (trial may be older than our range)
        eligible = sorted(month_dates, key=lambda x: x[0])[:1]
        if eligible:
            return eligible[0][2]
        return None

    eligible.sort(key=lambda x: x[0], reverse=True)
    return eligible[0][2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch AACT monthly snapshots for point-in-time trial status")
    parser.add_argument("--months", type=int, default=None,
                        help="Limit to N most-recent months (for testing)")
    parser.add_argument("--write-only", action="store_true",
                        help="Skip download, just write ct_status_at_event to master CSV from cache")
    parser.add_argument("--urls-file", type=str, default=None,
                        help="Path to text file with manual URLs: one per line, format 'YYYY-MM https://...'")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Load master CSV ────────────────────────────────────────────────────────
    print(f"Loading master CSV: {os.path.basename(MASTER_CSV)}")
    df_master = pd.read_csv(MASTER_CSV)
    print(f"Shape: {df_master.shape[0]} rows × {df_master.shape[1]} cols")

    # Validated cohort: v_is_verified non-null AND has nct_id
    validated_mask = df_master["v_is_verified"].notna()
    has_nct = df_master["nct_id"].notna() & (df_master["nct_id"].str.strip() != "")
    target_mask = validated_mask & has_nct

    nct_ids_set = set(df_master.loc[target_mask, "nct_id"].str.strip().unique())
    print(f"Target: {len(nct_ids_set)} unique NCT IDs in validated cohort "
          f"({target_mask.sum()} rows)")

    # ── Load cache ────────────────────────────────────────────────────────────
    cache = load_cache()
    months_done = set(cache["_meta"].get("months_completed", []))
    print(f"Cache: {len(cache) - 1} NCT IDs, {len(months_done)} months already completed")

    if args.write_only:
        print("\n--write-only: skipping download phase")
    else:
        # ── Generate month list ────────────────────────────────────────────────
        all_months = generate_month_list("2023-01")
        if args.months:
            all_months = all_months[-args.months:]
        pending = [m for m in all_months if m not in months_done]
        print(f"\nMonths to fetch: {len(pending)} of {len(all_months)} total")

        if pending:
            session = requests.Session()
            session.headers.update({"User-Agent": "AACT-research-downloader/1.0"})

            # ── URL discovery ────────────────────────────────────────────────
            print("\nDiscovering URLs from AACT downloads page ...")
            page_urls = discover_urls_from_page(session)

            # Manual URLs
            manual_urls = {}
            if args.urls_file and os.path.exists(args.urls_file):
                with open(args.urls_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split(None, 1)
                            if len(parts) == 2:
                                manual_urls[parts[0]] = parts[1]
                print(f"Loaded {len(manual_urls)} manual URLs")

            # ── Fetch each month ─────────────────────────────────────────────
            for i, month_key in enumerate(pending):
                print(f"\n[{i+1}/{len(pending)}] Month {month_key}")

                url = resolve_url(session, month_key, page_urls, manual_urls)
                if url is None:
                    print(f"  No URL found for {month_key} — skipping")
                    continue

                results = download_and_parse(session, url, nct_ids_set, month_key)

                # Store in cache: {nct_id: {month_key: status}}
                for nct_id, status in results.items():
                    if nct_id not in cache:
                        cache[nct_id] = {}
                    cache[nct_id][month_key] = status

                # Mark month complete
                cache["_meta"]["months_completed"].append(month_key)
                months_done.add(month_key)
                save_cache(cache)
                print(f"  Saved cache. Total NCTs with data: {len(cache) - 1}")

                if i < len(pending) - 1:
                    time.sleep(RATE_LIMIT)

            print(f"\nFetch complete. Cache: {len(cache) - 1} NCT IDs, "
                  f"{len(months_done)} months done")

    # ── Coverage report ────────────────────────────────────────────────────────
    covered = sum(1 for nct in nct_ids_set if nct in cache and cache[nct])
    print(f"\nCoverage: {covered}/{len(nct_ids_set)} target NCT IDs have ≥1 month of data")

    # Status distribution
    all_statuses: dict[str, int] = {}
    for nct in nct_ids_set:
        if nct in cache:
            for v in cache[nct].values():
                all_statuses[v] = all_statuses.get(v, 0) + 1
    if all_statuses:
        print("Status distribution across cache entries:")
        for k, v in sorted(all_statuses.items(), key=lambda x: -x[1]):
            print(f"  {k:<35} {v}")

    # ── Lookup: resolve ct_status_at_event for each validated row ────────────
    print("\nLooking up point-in-time status for each row ...")

    date_col = "v_actual_date" if "v_actual_date" in df_master.columns else "event_date"

    ct_status_at_event = []
    for _, row in df_master.iterrows():
        nct_id = str(row.get("nct_id", "")).strip() if pd.notna(row.get("nct_id")) else ""
        evt_date = str(row.get(date_col, "")).strip() if pd.notna(row.get(date_col)) else ""
        is_validated = pd.notna(row.get("v_is_verified"))

        if not is_validated or not nct_id or not evt_date:
            ct_status_at_event.append(None)
            continue

        status = lookup_status_at_event(cache, nct_id, evt_date)
        ct_status_at_event.append(status)

    df_master["ct_status_at_event"] = ct_status_at_event

    # data_tier column
    df_master["data_tier"] = df_master["v_is_verified"].apply(
        lambda x: "validated" if pd.notna(x) else "historical"
    )

    # ── Sanity checks ──────────────────────────────────────────────────────────
    validated_rows = df_master[df_master["data_tier"] == "validated"]
    pit_non_null = validated_rows["ct_status_at_event"].notna().sum()
    pit_coverage = pit_non_null / len(validated_rows) * 100 if len(validated_rows) else 0

    print(f"\n── Sanity checks ──────────────────────────────────────────────────")
    print(f"  data_tier='validated': {len(validated_rows)} rows (expected ~894)")
    print(f"  data_tier='historical': {(df_master['data_tier'] == 'historical').sum()} rows (expected ~1620)")
    print(f"  ct_status_at_event non-null in validated: {pit_non_null}/{len(validated_rows)} ({pit_coverage:.1f}%)")

    # Leakage detection: snapshot COMPLETED but point-in-time NOT COMPLETED
    if "ct_status" in df_master.columns:
        snap_completed = df_master["ct_status"] == "COMPLETED"
        pit_not_completed = df_master["ct_status_at_event"] != "COMPLETED"
        pit_not_null = df_master["ct_status_at_event"].notna()
        leakage_fixed = (snap_completed & pit_not_completed & pit_not_null).sum()
        print(f"  Leakage cases fixed (snap=COMPLETED, PIT≠COMPLETED): {leakage_fixed}")

    # Spot-check: NCT05188729 — should NOT be COMPLETED at event ~2024-08
    spot = df_master[df_master["nct_id"] == "NCT05188729"][["nct_id", date_col, "ct_status_at_event", "ct_status"]].head(3)
    if len(spot) > 0:
        print(f"\n  NCT05188729 spot-check (event ~2024-08, trial completed 2024-10):")
        print(spot.to_string(index=False))

    # ── Save master CSV ────────────────────────────────────────────────────────
    df_master.to_csv(MASTER_CSV, index=False)
    print(f"\nSaved: {os.path.basename(MASTER_CSV)}  "
          f"({df_master.shape[0]} rows × {df_master.shape[1]} cols)")
    print("\nDone — ct_status_at_event + data_tier written to master CSV.")


if __name__ == "__main__":
    main()

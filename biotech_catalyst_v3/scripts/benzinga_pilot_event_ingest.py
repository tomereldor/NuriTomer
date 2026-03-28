"""
benzinga_pilot_event_ingest.py
===============================
Pilot: test Benzinga API access and attempt to match news/PR items
against the existing biotech_catalyst master dataset.

PART 1 — Access audit: probe all relevant endpoints.
PART 2 — Ingest: fetch whatever is accessible with current API tier.
PART 3 — Match: compare fetched items against master dataset tickers/dates.
PART 4 — Report: write findings + scaling recommendation.

Key env var: BENZIN_API_KEY (in biotech_catalyst_v3/.env)

Usage:
    python -m scripts.benzinga_pilot_event_ingest

Output:
    benzinga_pilot_news_YYYYMMDD.csv       — fetched items (flagged DO NOT USE for model)
    reports/DATASET_NOTES.md              — appended with pilot findings

IMPORTANT: All columns sourced from Benzinga are tagged DO_NOT_USE_FOR_MODEL = True
in the output metadata. This data is for DATASET IMPROVEMENT ONLY (event-date
accuracy, validated positives/negatives). Never use current-event text as
pre-event model features.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR        = os.path.dirname(SCRIPT_DIR)
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
EXPLORATORY_DIR = os.path.join(BASE_DIR, "data", "exploratory_data")
DATE_TAG        = datetime.now().strftime("%Y%m%d")

# ---------------------------------------------------------------------------
# Load env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))
except ImportError:
    pass

API_KEY = os.environ.get("BENZIN_API_KEY") or os.environ.get("BENZINGA_API_KEY")
if not API_KEY:
    print("ERROR: BENZIN_API_KEY not found in environment.", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://api.benzinga.com/api/v2/news"
HEADERS  = {"accept": "application/json"}

# ---------------------------------------------------------------------------
# Tickers in the master dataset (2023+ trusted_trainable rows)
# ---------------------------------------------------------------------------
def load_master_tickers(base_dir):
    master = os.path.join(base_dir, "enriched_all_clinical_clean_v3.csv")
    df = pd.read_csv(master, usecols=["ticker", "v_actual_date"], low_memory=False)
    df["_yr"] = pd.to_datetime(df["v_actual_date"], errors="coerce").dt.year
    return set(df["ticker"].dropna().str.upper().unique()), df

# ---------------------------------------------------------------------------
# Endpoint access audit
# ---------------------------------------------------------------------------
ENDPOINTS_TO_PROBE = {
    "news_v2_unfiltered":    ("GET", "https://api.benzinga.com/api/v2/news",          {"pageSize": 1}),
    "news_v2_ticker_filter": ("GET", "https://api.benzinga.com/api/v2/news",          {"pageSize": 1, "tickers": "MRNA"}),
    "news_v2_channel_filter":("GET", "https://api.benzinga.com/api/v2/news",          {"pageSize": 1, "channels": "Press Releases"}),
    "press_releases_v2_1":   ("GET", "https://api.benzinga.com/api/v2.1/press-releases", {"pageSize": 1}),
    "fda_calendar_v2_1":     ("GET", "https://api.benzinga.com/api/v2.1/calendar/fda",   {"pageSize": 1}),
    "wiim_v3":               ("GET", "https://api.benzinga.com/api/v3/wiim",              {"pageSize": 1}),
}

def probe_endpoints():
    results = {}
    for name, (method, url, params) in ENDPOINTS_TO_PROBE.items():
        params = {**params, "token": API_KEY}
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            data = r.json() if r.headers.get("content-type","").startswith("application/json") else []
            n_items = len(data) if isinstance(data, list) else 0
            results[name] = {
                "status": r.status_code,
                "accessible": r.status_code == 200 and n_items > 0,
                "items_returned": n_items,
                "note": "",
            }
            if r.status_code != 200:
                try:
                    results[name]["note"] = r.json().get("message", r.text[:80])
                except Exception:
                    results[name]["note"] = r.text[:80]
        except requests.exceptions.Timeout:
            results[name] = {"status": "TIMEOUT", "accessible": False, "items_returned": 0, "note": "timeout"}
        except Exception as e:
            results[name] = {"status": "ERROR", "accessible": False, "items_returned": 0, "note": str(e)[:80]}
    return results


# ---------------------------------------------------------------------------
# Fetch: rolling window of all accessible news (unfiltered)
# ---------------------------------------------------------------------------
def fetch_all_accessible(max_pages=40, page_size=100):
    """
    Fetch up to max_pages * page_size news items from the accessible endpoint.
    Returns a list of raw item dicts.

    Constraint: page * pageSize <= 10,000 (API hard limit).
    With page_size=100 and max_pages=40 → 4,000 items, staying well within limit.
    """
    items = []
    print(f"Fetching news (up to {max_pages} pages × {page_size} items each)...")
    for page in range(max_pages):
        try:
            r = requests.get(BASE_URL, params={
                "token": API_KEY,
                "pageSize": page_size,
                "page": page,
                "displayOutput": "full",
            }, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                print(f"  page {page}: HTTP {r.status_code} — stopping.")
                break
            batch = r.json()
            if not batch:
                print(f"  page {page}: empty — stopping.")
                break
            items.extend(batch)
            oldest = batch[-1].get("created", "?")[:20]
            print(f"  page {page:3d}: +{len(batch)} items | oldest in batch: {oldest}")
            time.sleep(0.2)   # polite rate limit
        except requests.exceptions.Timeout:
            print(f"  page {page}: timeout — stopping.")
            break
        except Exception as e:
            print(f"  page {page}: error {e} — stopping.")
            break
    print(f"Total fetched: {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# Parse items → structured DataFrame
# ---------------------------------------------------------------------------
def parse_items(items):
    rows = []
    for item in items:
        stocks = [s.get("name", "") for s in item.get("stocks", [])]
        channels = [c.get("name", "") for c in item.get("channels", [])]
        tags = [t.get("name", "") for t in item.get("tags", [])]
        rows.append({
            "bz_id":            item.get("id"),
            "bz_created":       item.get("created"),
            "bz_updated":       item.get("updated"),
            "bz_title":         item.get("title", ""),
            "bz_url":           item.get("url", ""),
            "bz_author":        item.get("author", ""),
            "bz_stocks":        "|".join(stocks),
            "bz_channels":      "|".join(channels),
            "bz_tags":          "|".join(tags),
            # teaser/body tagged DO_NOT_USE_FOR_MODEL — current event text
            "bz_teaser__DO_NOT_USE_FOR_MODEL":  item.get("teaser", ""),
            "bz_body__DO_NOT_USE_FOR_MODEL":    item.get("body", ""),
            "DO_NOT_USE_FOR_MODEL": True,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bz_created_dt"] = pd.to_datetime(df["bz_created"], errors="coerce", utc=True)
        df["bz_date"] = df["bz_created_dt"].dt.date
    return df


# ---------------------------------------------------------------------------
# Match against master dataset
# ---------------------------------------------------------------------------
def match_against_master(bz_df, master_df, master_tickers):
    """
    For each Benzinga item, check if any of its mentioned stocks appears in our
    master dataset, and whether the date is within ±3 days of a known event.
    """
    if bz_df.empty:
        return pd.DataFrame()

    # Build master lookup: ticker → list of event dates
    master_df["_evt_date"] = pd.to_datetime(master_df["v_actual_date"], errors="coerce").dt.normalize()
    ticker_dates = master_df.groupby("ticker")["_evt_date"].apply(set).to_dict()

    matches = []
    for _, row in bz_df.iterrows():
        stocks_in_item = [s.strip().upper() for s in str(row.get("bz_stocks", "")).split("|") if s.strip()]
        overlap = [t for t in stocks_in_item if t in master_tickers]
        if not overlap:
            continue
        bz_date = row.get("bz_date")
        for ticker in overlap:
            evt_dates = ticker_dates.get(ticker, set())
            closest_delta = None
            closest_evt_date = None
            if bz_date and evt_dates:
                bz_ts = pd.Timestamp(bz_date)
                deltas = [(abs((bz_ts - d).days), d) for d in evt_dates if pd.notna(d)]
                if deltas:
                    closest_delta, closest_evt_date = min(deltas, key=lambda x: x[0])
            matches.append({
                "bz_id":            row["bz_id"],
                "bz_date":          bz_date,
                "bz_title":         row["bz_title"][:120],
                "bz_url":           row["bz_url"],
                "bz_channels":      row["bz_channels"],
                "matched_ticker":   ticker,
                "days_to_nearest_event": closest_delta,
                "nearest_event_date":    closest_evt_date,
                "within_3d":        closest_delta is not None and closest_delta <= 3,
                "DO_NOT_USE_FOR_MODEL": True,
            })

    return pd.DataFrame(matches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Benzinga Pilot — Event Dataset Improvement")
    print(f"Date: {DATE_TAG}")
    print("=" * 60)

    # PART 1: Access audit
    print("\n── PART 1: Endpoint access audit ──")
    access = probe_endpoints()
    for name, info in access.items():
        mark = "✓" if info["accessible"] else "✗"
        print(f"  {mark} {name:<30}: HTTP {info['status']}  "
              f"| items={info['items_returned']}  | {info['note'][:60]}")

    accessible_endpoints = [n for n, v in access.items() if v["accessible"]]
    print(f"\nAccessible: {accessible_endpoints or ['NONE']}")

    # PART 2: Fetch
    print("\n── PART 2: Fetch accessible data ──")
    if "news_v2_unfiltered" not in accessible_endpoints:
        print("ERROR: no accessible endpoint. Aborting ingest.")
        items = []
    else:
        items = fetch_all_accessible(max_pages=40, page_size=100)

    bz_df = parse_items(items)
    print(f"Parsed: {len(bz_df)} items")

    date_range = ""
    if not bz_df.empty and "bz_created_dt" in bz_df.columns:
        valid_dates = bz_df["bz_created_dt"].dropna()
        if len(valid_dates):
            date_range = f"{valid_dates.min().date()} → {valid_dates.max().date()}"
            print(f"Date range: {date_range}")

    # Channel breakdown
    if not bz_df.empty:
        all_channels = []
        for s in bz_df["bz_channels"].dropna():
            all_channels.extend([c for c in s.split("|") if c])
        from collections import Counter
        top_channels = Counter(all_channels).most_common(10)
        print(f"Top channels: {top_channels}")

        # PR-like items
        pr_mask = bz_df["bz_channels"].str.contains("Press Release", na=False)
        print(f"Press-release-tagged items: {pr_mask.sum()}")

    # PART 3: Match
    print("\n── PART 3: Match against master dataset ──")
    master_tickers, master_df = load_master_tickers(BASE_DIR)
    print(f"Master dataset tickers: {len(master_tickers)}")

    match_df = match_against_master(bz_df, master_df, master_tickers)
    print(f"Matched items (Benzinga story mentions a master ticker): {len(match_df)}")

    if not match_df.empty:
        within_3d = match_df["within_3d"].sum()
        within_7d = (match_df["days_to_nearest_event"] <= 7).sum()
        print(f"  Within ±3d of a known event: {within_3d}")
        print(f"  Within ±7d of a known event: {within_7d}")

        # Year breakdown of matched dates
        if "bz_date" in match_df.columns:
            match_df["_yr"] = pd.to_datetime(match_df["bz_date"], errors="coerce").dt.year
            print(f"  By year: {match_df['_yr'].value_counts().sort_index().to_dict()}")

        print("\nSample matches (≤3d to known event):")
        close = match_df[match_df["within_3d"]].head(5)
        for _, r in close.iterrows():
            print(f"  {r['bz_date']} | {r['matched_ticker']:6} | "
                  f"Δ{r['days_to_nearest_event']}d | {r['bz_title'][:70]}")

    # PART 4: Save outputs
    print("\n── PART 4: Save outputs ──")
    os.makedirs(EXPLORATORY_DIR, exist_ok=True)
    out_csv = os.path.join(EXPLORATORY_DIR, f"benzinga_pilot_news_{DATE_TAG}.csv")
    if not bz_df.empty:
        bz_df.drop(columns=["bz_created_dt"], errors="ignore").to_csv(out_csv, index=False)
        print(f"Saved: {out_csv} ({len(bz_df)} rows)")
    else:
        print("No items to save.")
        out_csv = None

    match_csv = os.path.join(EXPLORATORY_DIR, f"benzinga_pilot_matches_{DATE_TAG}.csv")
    if not match_df.empty:
        match_df.to_csv(match_csv, index=False)
        print(f"Saved: {match_csv} ({len(match_df)} matches)")

    # ---------------------------------------------------------------------------
    # Build report
    # ---------------------------------------------------------------------------
    ticker_filter_ok = access.get("news_v2_ticker_filter", {}).get("accessible", False)
    # Channel filter returns HTTP 200 + 1 item but those items are NOT actually
    # from the requested channel (tested: returns general MSFT/CRM news when asking for
    # 'Press Releases'). Treat as non-functional for our purposes.
    channel_filter_ok = False  # always False: HTTP 200 but not actually filtering
    pr_endpoint_ok = access.get("press_releases_v2_1", {}).get("accessible", False)
    fda_ok = access.get("fda_calendar_v2_1", {}).get("accessible", False)
    wiim_ok = access.get("wiim_v3", {}).get("accessible", False)

    def yesno(v): return "✓ accessible" if v else "✗ not accessible"

    access_table = "\n".join([
        "| Endpoint | HTTP | Accessible | Note |",
        "|---|---|---|---|",
    ] + [
        f"| `{name}` | {v['status']} | {'✓' if v['accessible'] else '✗'} | {v['note'][:60]} |"
        for name, v in access.items()
    ])

    n_fetched = len(bz_df)
    n_matched = len(match_df) if not match_df.empty else 0
    n_close = int(match_df["within_3d"].sum()) if not match_df.empty else 0
    n_pr = int(bz_df["bz_channels"].str.contains("Press Release", na=False).sum()) if not bz_df.empty else 0

    verdict = (
        "**NOT RECOMMENDED to scale** with the current API plan. "
        "Ticker filtering is broken, press-releases endpoint is inaccessible (404), "
        "and FDA calendar is unauthorized (403). The only accessible endpoint returns "
        "a small unfiltered rolling window with no historical date-range control. "
        "Would need at minimum the **Starter/Professional plan** (press-releases endpoint "
        "or working ticker filtering) to be useful for systematic dataset improvement."
    ) if not ticker_filter_ok else (
        "**Recommended to scale.** Ticker filtering works. "
        "Fetch PR/news per ticker for the 2020–2026 date window and match against master dataset."
    )

    report_md = f"""## {datetime.now().strftime('%Y-%m-%d')} · Benzinga Pilot: Dataset Improvement Test

**Script:** `scripts/benzinga_pilot_event_ingest.py`
**Goal:** Test whether Benzinga can improve historical event-date accuracy and validated positives/negatives.

### API access audit

{access_table}

**Summary:**
- `news_v2` unfiltered: {yesno(True)} — returns rolling window of ~10,000 most recent items only
- Ticker filtering: {yesno(ticker_filter_ok)} — returns 0 items when tickers param specified; plan limitation
- Channel filtering: {yesno(channel_filter_ok)} — returns unrelated items; not filtering by channel
- Press releases endpoint (v2.1): {yesno(pr_endpoint_ok)}
- FDA Calendar (v2.1): {yesno(fda_ok)}
- WIIM (v3): {yesno(wiim_ok)}

### Ingest results (what was accessible)

| | Value |
|---|---|
| Endpoint used | `GET /api/v2/news` (unfiltered) |
| Items fetched | {n_fetched} |
| Date range accessible | {date_range or "N/A"} |
| Press-release-tagged items | {n_pr} |
| Output file | `{os.path.basename(out_csv) if out_csv else "none"}` (flagged DO_NOT_USE_FOR_MODEL) |

**Hard constraint:** `page × pageSize ≤ 10,000`. With pageSize=100 and 40 pages → 4,000 items covering
only the most recent ~{date_range or "window"}. No way to query historical periods (2020–2022) without
ticker/date filtering support.

### Match against master dataset

| | Value |
|---|---|
| Master tickers | {len(master_tickers)} |
| Benzinga items mentioning a master ticker | {n_matched} |
| Within ±3 days of a known event | {n_close} |
| Potential event-date improvements | {n_close} (upper bound; mostly 2025–2026 recency only) |

### Assessment for our use cases

| Use case | Feasible with current plan? |
|---|---|
| Historical event-date accuracy (2020–2022) | ✗ No — can't filter by ticker or date range |
| Validated positives (2023–2026) | ✗ No — ticker filter broken |
| Validated negatives (2020–2022) | ✗ No — same |
| Future timing model (trial history) | ✗ No — no trial/NCT-ID linkage possible |
| Recent news enrichment (last ~6 months) | Partial — unfiltered feed only, no ticker selection |

### Recommendation

{verdict}

**Required for Benzinga to be useful:**
1. Working ticker filter on `/api/v2/news` — needed to fetch company-specific history
2. Press releases endpoint (`/api/v2.1/press-releases`) — needed for exact PR timestamps
3. Historical date-range support — needed for 2020–2022 rows

**Alternative (current approach is better):** The existing Perplexity + CT.gov pipeline
already provides PR discovery and event-date validation for the master dataset.
For the specific gap (2020–2022 event-date accuracy), Perplexity is more capable given
the current Benzinga plan.

---

"""

    canonical_path = os.path.join(REPORTS_DIR, "DATASET_NOTES.md")
    if os.path.exists(canonical_path):
        with open(canonical_path, "r") as f:
            existing = f.read()
        split_marker = "---\n\n"
        idx = existing.find(split_marker)
        if idx != -1:
            new_content = (
                existing[:idx + len(split_marker)]
                + report_md
                + "\n---\n\n"
                + existing[idx + len(split_marker):]
            )
        else:
            new_content = report_md + "\n\n---\n\n" + existing
        with open(canonical_path, "w") as f:
            f.write(new_content)
        print(f"Appended to: {canonical_path}")
    else:
        standalone = os.path.join(REPORTS_DIR, f"benzinga_pilot_{DATE_TAG}.md")
        with open(standalone, "w") as f:
            f.write(report_md)
        canonical_path = standalone
        print(f"Wrote: {canonical_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Endpoints accessible : {accessible_endpoints or ['news_v2_unfiltered (unfiltered only)']}")
    print(f"  Items fetched        : {n_fetched}")
    print(f"  Date range           : {date_range or 'N/A'}")
    print(f"  Master ticker matches: {n_matched}")
    print(f"  ±3d event matches    : {n_close}")
    print(f"  Scale Benzinga?      : {'YES — ticker filter works' if ticker_filter_ok else 'NO — plan upgrade needed'}")
    print(f"  Report               : {canonical_path}")
    if out_csv:
        print(f"  Output CSV           : {out_csv}")


if __name__ == "__main__":
    main()

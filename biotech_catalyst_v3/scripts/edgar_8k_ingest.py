"""
edgar_8k_ingest.py
==================
Query SEC EDGAR for 8-K filings (press releases) for biotech tickers
in the master dataset, matched by ticker + date proximity.

What this does
--------------
Phase 1 — Ticker→CIK lookup:
    Fetch SEC's company_tickers.json once, map all 160 unique tickers
    from history_only rows (2020–2022) to their CIK numbers.

Phase 2 — 8-K filing discovery:
    For each ticker, fetch filings from the EDGAR Submissions API.
    Filter to 8-Ks filed within ±21 days of any event_date in our dataset.
    Paginate older filings via the filings.files sub-array if needed.

Phase 3 — Clinical relevance filter + text extraction:
    For each matched 8-K, attempt to download Exhibit 99.1 (press release).
    Filter using clinical/FDA keyword list.

Phase 4 — Outcome extraction (two-stage):
    Stage A — Keyword heuristics: fast, free, covers clear-cut cases.
    Stage B — Perplexity API: for ambiguous cases (only if --no-llm not set).
    Extracts: outcome_polarity, primary_endpoint_met, key_finding, confidence.

Phase 5 — Output + report:
    Write edgar_8k_matches_YYYYMMDD.csv (all columns tagged DO_NOT_USE_FOR_MODEL).
    Prepend reports/DATASET_NOTES.md with coverage statistics.

Key design choices
------------------
- DO NOT USE any edgar_* columns as direct pre-event model features.
  They are for data quality, event-date validation, and building
  drug/company historical outcome rate features in a separate pipeline.
- Keyword heuristics first, Perplexity only for ambiguous cases
  (saves API credits; clear clinical outcomes don't need LLM).
- Resumable: if output CSV already exists, skip already-processed rows.
- EDGAR rate limit: 10 req/s. We use 0.15s delay (safe margin).

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3

    # Show plan (dry-run)
    python -m scripts.edgar_8k_ingest --dry-run

    # Test 20 rows (no LLM)
    python -m scripts.edgar_8k_ingest --limit 20 --no-llm

    # Full run
    python -m scripts.edgar_8k_ingest

    # Full run, skip Perplexity
    python -m scripts.edgar_8k_ingest --no-llm

Output
------
    edgar_8k_matches_YYYYMMDD.csv   — matched rows with edgar_* columns
    reports/DATASET_NOTES.md        — prepended with pilot findings

IMPORTANT: All edgar_* columns are tagged DO_NOT_USE_FOR_MODEL = True.
           They are enrichment signals for data quality ONLY.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATE_TAG    = datetime.now().strftime("%Y%m%d")

MASTER_CSV = os.path.join(BASE_DIR, "enriched_all_clinical_clean_v3.csv")
TIERED_CSV = os.path.join(BASE_DIR, "enriched_all_clinical_clean_v3_tiered_20260318_v1.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, f"edgar_8k_matches_{DATE_TAG}.csv")


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Load key=value pairs from .env at repo root into os.environ."""
    for env_path in [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(os.path.dirname(BASE_DIR), ".env"),
    ]:
        if not os.path.exists(env_path):
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val


_load_dotenv()

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL     = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL   = os.environ.get("PERPLEXITY_MODEL", "sonar-pro")

# ---------------------------------------------------------------------------
# EDGAR API constants
# ---------------------------------------------------------------------------
# SEC requires a User-Agent identifying the app and a contact email.
EDGAR_HEADERS = {
    "User-Agent": "biotech-catalyst-research/1.0 (research@nuritomer.com)",
    "Accept": "application/json",
}
EDGAR_DELAY   = 0.15   # seconds between EDGAR requests (10 req/s limit)
PERPLEXITY_DELAY = 1.5  # seconds between Perplexity calls

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL     = "https://data.sec.gov/submissions/CIK{cik10}.json"
FILING_INDEX_URL    = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}-index.json"
FILING_DOC_URL      = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

MATCH_WINDOW_DAYS = 21   # ±21 days to match 8-K to event_date
MIN_YEAR          = 2020
MAX_YEAR          = 2022

# ---------------------------------------------------------------------------
# Clinical relevance keywords (at least one must appear in 8-K text)
# ---------------------------------------------------------------------------
CLINICAL_KEYWORDS = [
    r"clinical trial", r"phase\s*[123i]+(ii|iii)?",
    r"pivotal", r"primary endpoint", r"primary outcome", r"primary efficacy",
    r"topline", r"top-line", r"data readout", r"interim analysis",
    r"trial results", r"study results", r"efficacy", r"safety data",
    r"pdufa", r"nda\b", r"bla\b", r"fda appro", r"fda accept",
    r"breakthrough therapy", r"fast track", r"orphan drug",
    r"statistically significant", r"met.{0,20}primary", r"failed.{0,20}meet",
    r"positive (results|data|outcome)", r"negative (results|data|outcome)",
]
_CLINICAL_RE = re.compile(
    "|".join(CLINICAL_KEYWORDS), re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Keyword-based outcome heuristics
# ---------------------------------------------------------------------------
_POSITIVE_RE = re.compile(
    r"met.{0,30}(primary endpoint|primary outcome|primary efficacy)"
    r"|statistically significant.{0,60}(primary|efficac|endpoint)"
    r"|(primary endpoint|primary outcome|primary efficacy).{0,60}(met|achieved|demonstrated|reached)"
    r"|fda (approv|accept|granted|cleared)"
    r"|positive (results|outcome|topline|top-line)"
    r"|breakthrough therapy designation"
    r"|grant(ed|s) (approval|clearance)",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"(did not|failed to|not).{0,30}(meet|achieve|demonstrate).{0,30}(primary|endpoint|outcome)"
    r"|failed to meet.{0,20}primary"
    r"|negative (results|outcome|topline|top-line)"
    r"|not statistically significant"
    r"|(fda|agency) (rejected|refused|issued complete response|issued a complete response)"
    r"|complete response letter",
    re.IGNORECASE,
)
_ENDPOINT_MET_RE   = re.compile(r"met.{0,40}primary endpoint|primary endpoint.{0,40}met|met its primary", re.IGNORECASE)
_ENDPOINT_NOTMET_RE = re.compile(r"(did not|failed to).{0,40}(meet|achieve).{0,40}primary", re.IGNORECASE)


def _keyword_outcome(text: str) -> Dict[str, str]:
    """Fast keyword-based outcome extraction. Returns empty strings if ambiguous."""
    pos = bool(_POSITIVE_RE.search(text))
    neg = bool(_NEGATIVE_RE.search(text))

    if pos and not neg:
        polarity = "positive"
        conf     = "high"
    elif neg and not pos:
        polarity = "negative"
        conf     = "high"
    elif pos and neg:
        polarity = "inconclusive"
        conf     = "medium"
    else:
        polarity = ""          # Unknown — send to Perplexity
        conf     = ""

    if _ENDPOINT_MET_RE.search(text):
        ep_met = "yes"
    elif _ENDPOINT_NOTMET_RE.search(text):
        ep_met = "no"
    else:
        ep_met = "unknown"

    return {"outcome_polarity": polarity, "primary_endpoint_met": ep_met, "confidence": conf}


# ---------------------------------------------------------------------------
# Perplexity LLM extraction
# ---------------------------------------------------------------------------
_EXTRACT_SYSTEM = (
    "You are a biotech analyst extracting structured outcomes from SEC 8-K press releases. "
    "Return ONLY valid JSON with no extra text."
)

_EXTRACT_USER = """Analyze this biotech press release and return a JSON object with exactly these fields:
- "outcome_polarity": "positive" (trial succeeded / drug approved) | "negative" (failed / rejected) | "inconclusive" (mixed / ongoing / unclear) | "unknown" (no clinical content found)
- "primary_endpoint_met": "yes" | "no" | "mixed" | "unknown"
- "key_finding": one sentence, max 60 words, describing the main result
- "confidence": "high" (explicit clear language) | "medium" (implied) | "low" (ambiguous)

Return ONLY the JSON object, no preamble.

Press release text (first 1500 chars):
{text}"""


def _perplexity_extract(text: str) -> Dict[str, str]:
    """Call Perplexity to extract outcome from press release text."""
    if not PERPLEXITY_API_KEY:
        return {"outcome_polarity": "", "primary_endpoint_met": "unknown",
                "key_finding": "", "confidence": "", "llm_error": "No PERPLEXITY_API_KEY"}

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": _EXTRACT_USER.format(text=text[:1500])},
        ],
        "temperature": 0.0,
        "max_tokens": 250,
    }
    try:
        r = requests.post(
            PERPLEXITY_URL,
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        data = json.loads(raw)
        return {
            "outcome_polarity":     str(data.get("outcome_polarity", "unknown")),
            "primary_endpoint_met": str(data.get("primary_endpoint_met", "unknown")),
            "key_finding":          str(data.get("key_finding", "")),
            "confidence":           str(data.get("confidence", "low")),
            "llm_error":            "",
        }
    except Exception as e:
        return {"outcome_polarity": "unknown", "primary_endpoint_met": "unknown",
                "key_finding": "", "confidence": "", "llm_error": str(e)[:120]}


# ---------------------------------------------------------------------------
# EDGAR API helpers
# ---------------------------------------------------------------------------
def _edgar_get(url: str, as_json: bool = True):
    """GET from EDGAR with required User-Agent and rate-limit delay."""
    time.sleep(EDGAR_DELAY)
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        return r.json() if as_json else r.text
    except Exception as e:
        return None


def build_ticker_cik_map(tickers: List[str]) -> Dict[str, str]:
    """
    Fetch SEC company_tickers.json and return {TICKER: zero-padded-CIK-10} for tickers in set.
    SEC returns {idx: {cik_str, ticker, title}} — no authentication required.
    """
    print("Fetching company_tickers.json from SEC...")
    data = _edgar_get(COMPANY_TICKERS_URL)
    if not data:
        print("ERROR: Could not fetch company_tickers.json")
        return {}

    ticker_set = {t.upper() for t in tickers}
    mapping = {}
    for entry in data.values():
        t = entry.get("ticker", "").upper()
        if t in ticker_set:
            cik = str(entry["cik_str"])
            mapping[t] = cik.zfill(10)

    found    = len(mapping)
    missing  = ticker_set - set(mapping.keys())
    print(f"  CIK lookup: {found}/{len(ticker_set)} tickers matched")
    if missing:
        print(f"  Not found in EDGAR: {sorted(missing)}")
    return mapping


def fetch_8k_filings(cik10: str) -> List[Dict]:
    """
    Fetch all 8-K filings for a CIK from the EDGAR submissions API.
    Paginates through filings.files if the recent window doesn't cover MIN_YEAR.
    Returns list of {accessionNumber, filingDate, form, primaryDocument}.
    """
    url  = SUBMISSIONS_URL.format(cik10=cik10)
    data = _edgar_get(url)
    if not data:
        return []

    def _extract_8ks(filing_data: dict) -> List[Dict]:
        """Pull 8-K rows from a filings block."""
        recent = filing_data.get("filings", {}).get("recent", {})
        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
        accs   = recent.get("accessionNumber", [])
        docs   = recent.get("primaryDocument", [])
        result = []
        for i, form in enumerate(forms):
            if form in ("8-K", "8-K/A"):
                result.append({
                    "accessionNumber": accs[i] if i < len(accs) else "",
                    "filingDate":      dates[i] if i < len(dates) else "",
                    "form":            form,
                    "primaryDocument": docs[i] if i < len(docs) else "",
                })
        return result

    all_8ks = _extract_8ks(data)

    # Check if earliest 8-K in recent window is before MIN_YEAR; if so, done.
    dates_in_recent = [r["filingDate"] for r in all_8ks if r["filingDate"]]
    earliest_recent = min(dates_in_recent, default="9999")

    # Paginate through filings.files if we need older data
    if earliest_recent > f"{MIN_YEAR}-01-01":
        older_files = data.get("filings", {}).get("files", [])
        for file_entry in older_files:
            fname = file_entry.get("name", "")
            if not fname:
                continue
            old_url  = f"https://data.sec.gov/submissions/{fname}"
            old_data = _edgar_get(old_url)
            if not old_data:
                continue
            old_8ks = []
            forms = old_data.get("form", [])
            dates = old_data.get("filingDate", [])
            accs  = old_data.get("accessionNumber", [])
            docs  = old_data.get("primaryDocument", [])
            for i, form in enumerate(forms):
                if form in ("8-K", "8-K/A"):
                    filing_date = dates[i] if i < len(dates) else ""
                    old_8ks.append({
                        "accessionNumber": accs[i] if i < len(accs) else "",
                        "filingDate":      filing_date,
                        "form":            form,
                        "primaryDocument": docs[i] if i < len(docs) else "",
                    })
            all_8ks.extend(old_8ks)
            # If this file's dates go back before MIN_YEAR, no need to fetch more
            file_dates = [r["filingDate"] for r in old_8ks if r["filingDate"]]
            if file_dates and min(file_dates) < f"{MIN_YEAR}-01-01":
                break

    return all_8ks


def fetch_exhibit_text(cik10: str, accession: str, primary_doc: str = "") -> str:
    """
    Fetch Exhibit 99.1 (press release) from an 8-K filing.

    Strategy:
      1. Fetch the primary 8-K form document (XBRL wrapper or simple HTML).
      2. Parse it for exhibit links — look for any href matching ex99 / 99.1 patterns.
      3. Fetch the exhibit and return its plain text (first 3000 chars).
      4. If no exhibit link found, try using the primary document text directly.

    Returns plain text or '' on failure.
    """
    cik_bare   = cik10.lstrip("0") or "0"
    acc_nodash = accession.replace("-", "")
    base_dir   = f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{acc_nodash}"

    if not primary_doc:
        return ""

    primary_url = f"{base_dir}/{primary_doc}"
    raw = _edgar_get(primary_url, as_json=False)
    if not raw:
        return ""

    # Parse primary doc HTML
    try:
        soup = BeautifulSoup(raw, "html.parser")
    except Exception:
        return ""

    # Look for Exhibit 99.1 link in the wrapper document
    ex99_href = None
    _EX99_RE = re.compile(r"ex[\-_]?99[\-_]?1|exhibit[\s_\-]?99", re.IGNORECASE)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or not href:
            continue
        if _EX99_RE.search(href) or _EX99_RE.search(a.get_text(strip=True)):
            # Resolve relative URL
            if href.startswith("http"):
                ex99_href = href
            else:
                ex99_href = f"{base_dir}/{href.lstrip('/')}"
            break

    if ex99_href:
        ex_raw = _edgar_get(ex99_href, as_json=False)
        if ex_raw:
            try:
                ex_soup = BeautifulSoup(ex_raw, "html.parser")
                for tag in ex_soup(["script", "style"]):
                    tag.decompose()
                text = ex_soup.get_text(" ", strip=True)
            except Exception:
                text = ex_raw
            text = re.sub(r"\s+", " ", text).strip()
            return text[:3000]

    # Fallback: use primary document text (works for older non-XBRL 8-Ks)
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    # Filter out XBRL-only content (very short or no alphabetic sentences)
    if len(text) < 200 or not re.search(r"[A-Z][a-z]{5,}", text):
        return ""
    return text[:3000]


# ---------------------------------------------------------------------------
# Main matching logic
# ---------------------------------------------------------------------------
def match_8ks_to_events(
    events: pd.DataFrame,
    filings: List[Dict],
    window_days: int = MATCH_WINDOW_DAYS,
) -> Optional[Dict]:
    """
    Find the best 8-K match for any event in this ticker's event list.
    Returns the closest match (by date delta) per event row, or None.
    """
    results = []
    for _, row in events.iterrows():
        evt_date_str = row.get("event_date") or row.get("v_actual_date")
        if not evt_date_str:
            continue
        try:
            evt_date = pd.to_datetime(evt_date_str).date()
        except Exception:
            continue

        best = None
        best_delta = window_days + 1
        for f in filings:
            fd_str = f.get("filingDate", "")
            if not fd_str:
                continue
            try:
                fd = datetime.strptime(fd_str, "%Y-%m-%d").date()
            except Exception:
                continue
            delta = abs((fd - evt_date).days)
            if delta <= window_days and delta < best_delta:
                best_delta = delta
                best = f

        results.append({
            "row_idx":    row.name,
            "ticker":     row.get("ticker", ""),
            "event_date": str(evt_date_str),
            "match":      best,
            "match_days": best_delta if best else None,
        })
    return results


# ---------------------------------------------------------------------------
# Build result row
# ---------------------------------------------------------------------------
def build_result_row(
    ticker: str,
    cik10: str,
    event_date: str,
    row_idx: int,
    match: Dict,
    match_days: int,
    text: str,
    is_clinical: bool,
    outcome: Dict,
    method: str,
) -> Dict:
    return {
        "ticker":                   ticker,
        "event_date":               event_date,
        "row_idx":                  row_idx,
        # EDGAR metadata — DO NOT USE FOR MODEL
        "edgar_cik":                cik10,
        "edgar_8k_date":            match.get("filingDate", ""),
        "edgar_8k_accession":       match.get("accessionNumber", ""),
        "edgar_8k_form":            match.get("form", ""),
        "edgar_8k_match_days":      match_days,
        "edgar_is_clinical":        is_clinical,
        "edgar_text_excerpt":       text[:1200] if text else "",
        # Extracted outcome — DO NOT USE FOR MODEL
        "edgar_outcome_polarity":   outcome.get("outcome_polarity", ""),
        "edgar_primary_endpoint_met": outcome.get("primary_endpoint_met", "unknown"),
        "edgar_key_finding":        outcome.get("key_finding", ""),
        "edgar_llm_confidence":     outcome.get("confidence", ""),
        "edgar_method":             method,
        "edgar_llm_error":          outcome.get("llm_error", ""),
        # Explicit flag — these columns must never enter a feature matrix
        "DO_NOT_USE_FOR_MODEL":     True,
    }


# ---------------------------------------------------------------------------
# DATASET_NOTES.md updater
# ---------------------------------------------------------------------------
def prepend_dataset_notes(report_text: str) -> None:
    notes_path = os.path.join(REPORTS_DIR, "DATASET_NOTES.md")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    existing = ""
    if os.path.exists(notes_path):
        with open(notes_path, "r") as f:
            existing = f.read()
    with open(notes_path, "w") as f:
        f.write(report_text + "\n\n" + existing)
    print(f"Prepended DATASET_NOTES.md at {notes_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EDGAR 8-K ingest for biotech catalyst dataset")
    parser.add_argument("--dry-run",  action="store_true", help="Show plan only, no API calls")
    parser.add_argument("--limit",    type=int, default=0, help="Process at most N rows (0=all)")
    parser.add_argument("--no-llm",   action="store_true", help="Skip Perplexity; keyword heuristics only")
    parser.add_argument("--year-min", type=int, default=MIN_YEAR, help=f"Min event year (default {MIN_YEAR})")
    parser.add_argument("--year-max", type=int, default=MAX_YEAR, help=f"Max event year (default {MAX_YEAR})")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load master dataset
    # ------------------------------------------------------------------
    csv_path = TIERED_CSV if os.path.exists(TIERED_CSV) else MASTER_CSV
    print(f"Loading dataset from: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Derive event year
    df["_event_year"] = pd.to_datetime(df["event_date"], errors="coerce").dt.year

    # Target rows: history_only (or pre-MIN_YEAR if no data_tier)
    if "data_tier" in df.columns:
        target_mask = df["data_tier"] == "history_only"
    else:
        target_mask = df["_event_year"] < 2023

    # Year window filter
    year_mask = (
        (df["_event_year"] >= args.year_min) &
        (df["_event_year"] <= args.year_max)
    )
    df_target = df[target_mask & year_mask & df["ticker"].notna()].copy()

    print(f"\nTarget rows ({args.year_min}–{args.year_max} history_only): {len(df_target)}")
    unique_tickers = sorted(df_target["ticker"].str.upper().unique())
    print(f"Unique tickers: {len(unique_tickers)}")

    if args.limit > 0:
        # Limit to first N rows (by ticker to keep groups intact)
        n_tickers = max(1, args.limit // (len(df_target) // max(len(unique_tickers), 1)))
        limited_tickers = unique_tickers[:n_tickers]
        df_target = df_target[df_target["ticker"].str.upper().isin(limited_tickers)]
        print(f"Limiting to {len(limited_tickers)} tickers → {len(df_target)} rows")

    if args.dry_run:
        print("\n[DRY-RUN] Plan:")
        print(f"  1. Fetch company_tickers.json → map {len(unique_tickers)} tickers to CIK")
        print(f"  2. Fetch submissions + 8-K lists for each CIK ({len(unique_tickers)} requests)")
        print(f"  3. Match 8-Ks to {len(df_target)} rows (±{MATCH_WINDOW_DAYS}d window)")
        print(f"  4. Fetch Exhibit 99.1 for each matched 8-K")
        print(f"  5. Keyword heuristics → Perplexity for ambiguous (unless --no-llm)")
        print(f"  6. Write {os.path.basename(OUTPUT_CSV)}")
        return

    # ------------------------------------------------------------------
    # Resume: skip rows already processed
    # ------------------------------------------------------------------
    already_done = set()
    if os.path.exists(OUTPUT_CSV):
        prev = pd.read_csv(OUTPUT_CSV, usecols=["ticker", "event_date"], low_memory=False)
        already_done = set(zip(prev["ticker"].str.upper(), prev["event_date"].astype(str)))
        print(f"Resuming: {len(already_done)} (ticker, date) pairs already processed")

    # ------------------------------------------------------------------
    # Phase 1: Ticker→CIK lookup
    # ------------------------------------------------------------------
    ticker_cik = build_ticker_cik_map(unique_tickers)

    # ------------------------------------------------------------------
    # Phase 2–4: Per-ticker processing
    # ------------------------------------------------------------------
    all_results = []
    save_every  = 25   # checkpoint frequency

    tickers_with_cik = [t for t in unique_tickers if t in ticker_cik]
    print(f"\nProcessing {len(tickers_with_cik)} tickers with known CIK...")

    for t_idx, ticker in enumerate(tickers_with_cik):
        cik10   = ticker_cik[ticker]
        t_rows  = df_target[df_target["ticker"].str.upper() == ticker]

        # Filter already-done
        t_rows = t_rows[~t_rows.apply(
            lambda r: (ticker.upper(), str(r["event_date"])) in already_done, axis=1
        )]
        if t_rows.empty:
            continue

        print(f"\n[{t_idx+1}/{len(tickers_with_cik)}] {ticker} (CIK {cik10}) — {len(t_rows)} events")

        # Fetch 8-K filings
        filings = fetch_8k_filings(cik10)
        if not filings:
            print(f"  No filings found for {ticker}")
            continue

        # Filter filings to year window (±1 year buffer)
        filings = [
            f for f in filings
            if f.get("filingDate", "") >= f"{args.year_min - 1}-01-01"
            and f.get("filingDate", "") <= f"{args.year_max + 1}-12-31"
        ]
        print(f"  {len(filings)} 8-Ks in {args.year_min-1}–{args.year_max+1} window")

        # Match 8-Ks to events
        matches = match_8ks_to_events(t_rows, filings)

        for m in matches:
            if not m["match"]:
                # No 8-K found within window
                print(f"    {m['event_date']} → no 8-K match")
                all_results.append({
                    "ticker":                   ticker,
                    "event_date":               m["event_date"],
                    "row_idx":                  m["row_idx"],
                    "edgar_cik":                cik10,
                    "edgar_8k_date":            "",
                    "edgar_8k_accession":       "",
                    "edgar_8k_form":            "",
                    "edgar_8k_match_days":      None,
                    "edgar_is_clinical":        False,
                    "edgar_text_excerpt":       "",
                    "edgar_outcome_polarity":   "no_8k_found",
                    "edgar_primary_endpoint_met": "unknown",
                    "edgar_key_finding":        "",
                    "edgar_llm_confidence":     "",
                    "edgar_method":             "no_match",
                    "edgar_llm_error":          "",
                    "DO_NOT_USE_FOR_MODEL":     True,
                })
                continue

            filing    = m["match"]
            match_days = m["match_days"]
            accession  = filing["accessionNumber"]

            print(f"    {m['event_date']} → 8-K {filing['filingDate']} (Δ{match_days}d, {accession})")

            # Fetch exhibit text
            text       = fetch_exhibit_text(cik10, accession, primary_doc=filing.get("primaryDocument", ""))
            is_clinical = bool(text and _CLINICAL_RE.search(text))

            if not is_clinical:
                print(f"      Not clinical — skipping LLM")
                outcome = {
                    "outcome_polarity": "not_clinical",
                    "primary_endpoint_met": "unknown",
                    "key_finding": "",
                    "confidence": "",
                    "llm_error": "",
                }
                method = "keyword_filter"
            else:
                # Stage A: keyword heuristics
                kw = _keyword_outcome(text)
                if kw["outcome_polarity"]:
                    outcome = {**kw, "key_finding": "", "llm_error": ""}
                    method  = "keyword"
                    print(f"      Keyword → {kw['outcome_polarity']} (conf: {kw['confidence']})")
                elif args.no_llm:
                    outcome = {
                        "outcome_polarity": "ambiguous",
                        "primary_endpoint_met": kw["primary_endpoint_met"],
                        "key_finding": "",
                        "confidence": "",
                        "llm_error": "skipped (--no-llm)",
                    }
                    method = "keyword_ambiguous"
                    print(f"      Ambiguous (no LLM)")
                else:
                    # Stage B: Perplexity
                    print(f"      Ambiguous → calling Perplexity...")
                    time.sleep(PERPLEXITY_DELAY)
                    outcome = _perplexity_extract(text)
                    method  = "perplexity"
                    print(f"      Perplexity → {outcome['outcome_polarity']} (conf: {outcome['confidence']})")

            all_results.append(build_result_row(
                ticker, cik10, m["event_date"], m["row_idx"],
                filing, match_days, text, is_clinical, outcome, method
            ))

        # Checkpoint save (every save_every new results accumulated)
        if all_results and len(all_results) % save_every == 0:
            _save_results(all_results, OUTPUT_CSV)

    # ------------------------------------------------------------------
    # Final save + report
    # ------------------------------------------------------------------
    if not all_results:
        print("\nNo results to save.")
        return

    df_out = _save_results(all_results, OUTPUT_CSV)
    _print_and_report(df_out, args)


def _save_results(results: List[Dict], path: str) -> pd.DataFrame:
    df_new = pd.DataFrame(results)
    if os.path.exists(path):
        df_existing = pd.read_csv(path, low_memory=False)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
        # Deduplicate
        df_out = df_out.drop_duplicates(subset=["ticker", "event_date", "edgar_8k_accession"])
    else:
        df_out = df_new
    df_out.to_csv(path, index=False)
    print(f"\nSaved {len(df_out)} rows → {os.path.basename(path)}")
    return df_out


def _print_and_report(df_out: pd.DataFrame, args) -> None:
    total         = len(df_out)
    matched       = (df_out["edgar_8k_date"] != "").sum()
    clinical      = df_out["edgar_is_clinical"].sum()
    positive      = (df_out["edgar_outcome_polarity"] == "positive").sum()
    negative      = (df_out["edgar_outcome_polarity"] == "negative").sum()
    inconclusive  = (df_out["edgar_outcome_polarity"] == "inconclusive").sum()
    no_match      = (df_out["edgar_outcome_polarity"] == "no_8k_found").sum()
    not_clinical  = (df_out["edgar_outcome_polarity"] == "not_clinical").sum()
    keyword_rows  = (df_out["edgar_method"] == "keyword").sum()
    perplexity_rows = (df_out["edgar_method"] == "perplexity").sum()

    report = f"""---
## {datetime.now().strftime('%Y-%m-%d')} · EDGAR 8-K Ingest Pilot

**Script:** `scripts/edgar_8k_ingest.py`
**Output:** `edgar_8k_matches_{DATE_TAG}.csv` ({total} rows)
**Target:** history_only rows {args.year_min}–{args.year_max} ({total} events, {df_out['ticker'].nunique()} tickers)

### Coverage

| Metric | Count | % of target |
|---|---|---|
| Events processed | {total} | 100% |
| 8-K found within ±21d | {matched} | {matched/total*100:.1f}% |
| Clinical 8-K (keyword filter) | {clinical} | {clinical/total*100:.1f}% |
| No 8-K found | {no_match} | {no_match/total*100:.1f}% |
| 8-K found but not clinical | {not_clinical} | {not_clinical/total*100:.1f}% |

### Outcome extraction (clinical matches only)

| Outcome | Count |
|---|---|
| positive | {positive} |
| negative | {negative} |
| inconclusive | {inconclusive} |
| ambiguous / unknown | {clinical - positive - negative - inconclusive} |

Extraction method: {keyword_rows} keyword · {perplexity_rows} Perplexity

### Key findings

- **Match rate:** {matched/total*100:.1f}% of events have an 8-K within ±21 days
- **Clinical rate (among matched):** {clinical/max(matched,1)*100:.1f}% of matched 8-Ks contain clinical content
- All `edgar_*` columns are tagged `DO_NOT_USE_FOR_MODEL = True`

### Next steps

1. Build drug/company historical outcome rate features from this table
   (pre-event valid: computed from events *before* the target event date)
2. Use `edgar_8k_date` to validate/correct `event_date` where Δ > 3 days
3. Cross-check `edgar_outcome_polarity = no_8k_found` rows — may be false positives

---"""

    print("\n" + "="*60)
    print(report)
    prepend_dataset_notes(report)


if __name__ == "__main__":
    main()

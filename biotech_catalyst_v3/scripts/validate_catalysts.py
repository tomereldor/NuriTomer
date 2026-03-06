"""
Biotech Catalyst Validation Script
====================================
Validates low-move ("noise") rows via Perplexity API to identify
rows where the clinical catalyst may be hallucinated or misattributed.

What this does
--------------
1. Identifies "noise" rows: events where the stock barely moved despite
   an alleged clinical data catalyst (move_class_norm == 'Noise',
   i.e. < 1.5× ATR — within the stock's normal daily variation).
2. For each noise row, asks Perplexity to verify whether clinical news
   actually existed on or around that date.  Perplexity returns a PR link.
3. Fetches the press release URL and extracts: PR date, PR title, and
   key content excerpt (~500 chars).
4. Flags false positives (no news found), date corrections, and errors.

Output columns (all prefixed v_)
---------------------------------
  v_is_verified    bool    Perplexity confirmed clinical news on/near event date
  v_actual_date    str     Correct event date if different from event_trading_date
  v_pr_link        str     Official press release URL found by Perplexity
  v_pr_date        str     Publication date extracted from the PR page
  v_pr_title       str     Title extracted from the PR page
  v_pr_key_info    str     First ~500 chars of PR body (key excerpt)
  v_is_material    bool    Major data readout (vs minor update)
  v_confidence     str     high / medium / low
  v_summary        str     One-sentence Perplexity summary of what happened
  v_error          str     Error message if anything failed

ATR methodology (from utils/volatility.py)
------------------------------------------
  Wilder's RMA: ewm(alpha=1/20, adjust=False) — same as TradingView default.
  Lookback: 20 trading days (≈ 1 calendar month) STRICTLY before the event.
  atr_pct = (ATR value / last pre-event closing price) × 100.
  Noise threshold: move < 1.5× ATR.

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3

    # Preview which rows would be validated
    python -m scripts.validate_catalysts --dry-run

    # Test with 20 rows
    python -m scripts.validate_catalysts --limit 20

    # Full validation run (saves every 25 rows, resumable)
    python -m scripts.validate_catalysts

    # Generate cleanup report from a validated CSV
    python -m scripts.validate_catalysts --report
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# .env loader
# ============================================================================

def _load_dotenv() -> None:
    """Load key=value pairs from .env at repo root into os.environ."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path  = os.path.join(repo_root, ".env")
    if not os.path.exists(env_path):
        return
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


# ============================================================================
# Configuration
# ============================================================================

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL     = "https://api.perplexity.ai/chat/completions"
MODEL              = "sonar-pro"
RATE_LIMIT_DELAY   = 1.5   # seconds between Perplexity calls

ATR_NOISE_THRESHOLD = 1.5  # matches utils/volatility.py Noise threshold

PR_FETCH_TIMEOUT = 15      # seconds for press release HTTP fetch
PR_KEY_INFO_CHARS = 600    # how many body chars to store in v_pr_key_info

# Browser-like headers to avoid bot blocking on PR sites
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class ValidationResult:
    # Perplexity verification fields
    is_verified:  bool = False
    actual_date:  str  = ""
    pr_link:      str  = ""
    is_material:  bool = False
    confidence:   str  = ""
    summary:      str  = ""
    # PR page extraction fields
    pr_date:      str  = ""
    pr_title:     str  = ""
    pr_key_info:  str  = ""
    # Error
    error:        str  = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Press release fetching + parsing
# ============================================================================

# Date patterns to search for in PR text
_DATE_PATTERNS = [
    r'\b(\w+ \d{1,2},?\s+20\d{2})\b',              # "March 6, 2026" / "March 6 2026"
    r'\b(20\d{2}-\d{2}-\d{2})\b',                   # "2026-03-06"
    r'\b(\d{1,2}/\d{1,2}/20\d{2})\b',               # "3/6/2026"
]


def _extract_date_from_text(text: str) -> str:
    """Try to find a publication date in free text. Returns first match or ''."""
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""


def _clean_text(text: str, max_chars: int = PR_KEY_INFO_CHARS) -> str:
    """Collapse whitespace and truncate."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars] if len(text) > max_chars else text


def _parse_businesswire(soup: BeautifulSoup) -> Tuple[str, str, str]:
    """Returns (date, title, key_info) for businesswire.com pages."""
    title = ""
    date  = ""
    body  = ""

    h1 = soup.find("h1", class_=re.compile(r"hl-summary|bwHeadline", re.I))
    if not h1:
        h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    ts = soup.find(class_=re.compile(r"bwTimestamp|bwDateline|release-date", re.I))
    if not ts:
        ts = soup.find("time")
    if ts:
        date = ts.get("datetime", "") or ts.get_text(" ", strip=True)

    story = soup.find(class_=re.compile(r"bw-release-story|bwBodyText|release-body", re.I))
    if story:
        body = _clean_text(story.get_text(" ", strip=True))

    return date, title, body


def _parse_globenewswire(soup: BeautifulSoup) -> Tuple[str, str, str]:
    """Returns (date, title, key_info) for globenewswire.com pages."""
    title = ""
    date  = ""
    body  = ""

    h1 = soup.find("h1", class_=re.compile(r"article-headline|title", re.I))
    if not h1:
        h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    ts = soup.find(class_=re.compile(r"article-source-info__date|article-date|date", re.I))
    if not ts:
        ts = soup.find("time")
    if ts:
        date = ts.get("datetime", "") or ts.get_text(" ", strip=True)

    article = soup.find(class_=re.compile(r"article-body|press-release-body", re.I))
    if article:
        body = _clean_text(article.get_text(" ", strip=True))

    return date, title, body


def _parse_prnewswire(soup: BeautifulSoup) -> Tuple[str, str, str]:
    """Returns (date, title, key_info) for prnewswire.com pages."""
    title = ""
    date  = ""
    body  = ""

    h1 = soup.find("h1", class_=re.compile(r"release-headline|title", re.I))
    if not h1:
        h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    ts = soup.find(class_=re.compile(r"release-date|date-time", re.I))
    if not ts:
        ts = soup.find("time")
    if ts:
        date = ts.get("datetime", "") or ts.get_text(" ", strip=True)

    release = soup.find(class_=re.compile(r"release-body|article-body", re.I))
    if release:
        body = _clean_text(release.get_text(" ", strip=True))

    return date, title, body


def _parse_sec(soup: BeautifulSoup) -> Tuple[str, str, str]:
    """Returns (date, title, key_info) for SEC EDGAR pages."""
    title = ""
    date  = ""
    body  = ""

    # SEC filing pages have the company name + form type in the title
    if soup.title:
        title = soup.title.get_text(" ", strip=True)

    # Look for filing date in the page
    text = soup.get_text(" ", strip=True)
    date = _extract_date_from_text(text)
    body = _clean_text(text)

    return date, title, body


def _parse_generic(soup: BeautifulSoup) -> Tuple[str, str, str]:
    """Generic fallback: og tags → h1 → body text."""
    title = ""
    date  = ""
    body  = ""

    # Open Graph / Twitter meta tags
    og_title = soup.find("meta", property="og:title")
    og_desc  = soup.find("meta", property="og:description")
    pub_time = (soup.find("meta", property="article:published_time") or
                soup.find("meta", attrs={"name": "pubdate"}) or
                soup.find("meta", attrs={"name": "date"}))

    if og_title:
        title = og_title.get("content", "")
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)

    if pub_time:
        date = pub_time.get("content", "")
    if not date:
        ts = soup.find("time")
        if ts:
            date = ts.get("datetime", "") or ts.get_text(" ", strip=True)

    if og_desc:
        body = og_desc.get("content", "")
    if not body:
        # Strip nav/footer and take main content
        for tag in soup(["nav", "footer", "header", "script", "style"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main:
            body = _clean_text(main.get_text(" ", strip=True))

    return date, title, body


def fetch_pr_details(url: str) -> Dict[str, str]:
    """
    Fetch a press release URL and extract date, title, and key body excerpt.

    Returns dict with keys: pr_date, pr_title, pr_key_info, pr_fetch_error.
    All values are strings; pr_fetch_error is '' on success.
    """
    result = {"pr_date": "", "pr_title": "", "pr_key_info": "", "pr_fetch_error": ""}

    if not url or not url.startswith("http"):
        result["pr_fetch_error"] = "No valid URL"
        return result

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=PR_FETCH_TIMEOUT,
                            allow_redirects=True)
        if resp.status_code != 200:
            result["pr_fetch_error"] = f"HTTP {resp.status_code}"
            return result

        soup = BeautifulSoup(resp.text, "lxml")
        domain = urlparse(url).netloc.lower()

        if "businesswire" in domain:
            date, title, body = _parse_businesswire(soup)
        elif "globenewswire" in domain:
            date, title, body = _parse_globenewswire(soup)
        elif "prnewswire" in domain:
            date, title, body = _parse_prnewswire(soup)
        elif "sec.gov" in domain:
            date, title, body = _parse_sec(soup)
        else:
            date, title, body = _parse_generic(soup)

        # If date not found by parser, scan first 2000 chars of body
        if not date and body:
            date = _extract_date_from_text(body[:2000])

        result["pr_date"]     = date.strip()[:100]
        result["pr_title"]    = title.strip()[:300]
        result["pr_key_info"] = body.strip()[:PR_KEY_INFO_CHARS]

    except requests.exceptions.Timeout:
        result["pr_fetch_error"] = "Timeout"
    except Exception as e:
        result["pr_fetch_error"] = str(e)[:100]

    return result


# ============================================================================
# Identify noise rows
# ============================================================================

def identify_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows flagged as Noise (< 1.5× ATR) that warrant verification."""
    df = df.copy()

    nm_col = (
        "stock_movement_atr_normalized" if "stock_movement_atr_normalized" in df.columns
        else "normalized_move" if "normalized_move" in df.columns
        else None
    )

    if "move_class_norm" in df.columns:
        is_noise = df["move_class_norm"] == "Noise"
    elif nm_col:
        is_noise = df[nm_col].notna() & (df[nm_col] < ATR_NOISE_THRESHOLD)
    else:
        has_atr  = df["atr_pct"].notna() & (df["atr_pct"] > 0)
        nm       = df["move_pct"].abs() / df["atr_pct"].where(has_atr, other=float("nan"))
        is_noise = has_atr & (nm < ATR_NOISE_THRESHOLD)

    noise_df = df[is_noise].copy()

    print(f"Total rows:       {len(df):,}")
    print(f"Rows with ATR:    {df['atr_pct'].notna().sum():,}  "
          f"({100*df['atr_pct'].notna().sum()/len(df):.1f}%)")
    print(f"Noise candidates: {len(noise_df):,}  ({100*len(noise_df)/len(df):.1f}%)")
    if nm_col:
        print(f"Median ATR-norm:  {df[nm_col].median():.2f}×")

    return noise_df


# ============================================================================
# Perplexity verification
# ============================================================================

def build_verification_prompt(row: pd.Series) -> str:
    ticker  = row.get("ticker", "")
    date    = row.get("event_trading_date") or row.get("event_date", "")
    drug    = row.get("drug_name", "") or "unknown drug"
    nct_id  = str(row.get("nct_id", "") or "")
    trial   = str(row.get("ct_official_title", "") or "")
    phase   = str(row.get("ct_phase", "") or "")
    summary = str(row.get("catalyst_summary", "") or "")[:200]

    ctx = []
    if nct_id.startswith("NCT"):
        ctx.append(f"NCT ID: {nct_id}")
    if drug and drug != "unknown drug":
        ctx.append(f"Drug: {drug}")
    if trial:
        ctx.append(f"Trial: {trial[:120]}")
    if phase:
        ctx.append(f"Phase: {phase}")
    context = "; ".join(ctx) if ctx else "no specific drug/trial info"

    return f"""VERIFICATION TASK: Determine whether a specific biotech clinical-data event actually occurred.

CLAIMED EVENT:
- Ticker:  {ticker}
- Date:    {date}
- Context: {context}
- Claimed catalyst: "{summary}"

IMPORTANT: Be SKEPTICAL. The claimed catalyst may be WRONG or HALLUCINATED.

Respond with ONLY a valid JSON object (no markdown, no other text):

{{
    "is_verified": true/false,
    "actual_date": "YYYY-MM-DD or null",
    "pr_link": "direct URL to official press release on businesswire/globenewswire/prnewswire/sec.gov or null",
    "is_material": true/false,
    "confidence": "high/medium/low",
    "summary": "one sentence: what actually happened, or 'No clinical news found for this date'"
}}

RULES:
1. is_verified = true ONLY if you find concrete evidence of clinical news on or within 1 day of {date}.
2. If news exists on a DIFFERENT date, set is_verified=false and provide actual_date.
3. If NO clinical news exists for {ticker} around this date, say so in summary.
4. pr_link = the direct URL to the official company press release (not a news article), or null.
5. is_material = true only for significant data readouts (Phase 2/3 results, FDA decisions).

JSON only:"""


def call_perplexity(prompt: str, max_retries: int = 3) -> Tuple[Optional[dict], str]:
    if not PERPLEXITY_API_KEY:
        return None, "PERPLEXITY_API_KEY not set in environment"

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a skeptical biotech research assistant. "
                    "Verify claims — do not assume they are true. "
                    "Always respond with valid JSON only, no markdown, no preamble."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(PERPLEXITY_URL, headers=headers,
                                 json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text[:120]}"

            content = (resp.json()
                       .get("choices", [{}])[0]
                       .get("message", {})
                       .get("content", ""))
            if not content:
                return None, "Empty response"

            content = content.strip()
            for fence in ("```json", "```"):
                if content.startswith(fence):
                    content = content[len(fence):]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                return json.loads(content), ""
            except json.JSONDecodeError as e:
                return None, f"JSON parse error: {e}"

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None, "Timeout"
        except Exception as e:
            return None, str(e)[:120]

    return None, "Max retries exceeded"


def verify_row(row: pd.Series) -> ValidationResult:
    """Call Perplexity to verify event, then fetch PR details if a link is returned."""
    parsed, error = call_perplexity(build_verification_prompt(row))

    if error:
        return ValidationResult(error=error)
    if not parsed:
        return ValidationResult(error="No response parsed")

    result = ValidationResult(
        is_verified = parsed.get("is_verified", False),
        actual_date = parsed.get("actual_date") or "",
        pr_link     = parsed.get("pr_link") or "",
        is_material = parsed.get("is_material", False),
        confidence  = parsed.get("confidence", "low"),
        summary     = parsed.get("summary", ""),
    )

    # Fetch PR page and extract date, title, key info
    if result.pr_link:
        pr = fetch_pr_details(result.pr_link)
        result.pr_date    = pr["pr_date"]
        result.pr_title   = pr["pr_title"]
        result.pr_key_info = pr["pr_key_info"]
        if pr["pr_fetch_error"]:
            result.error = f"PR fetch: {pr['pr_fetch_error']}"

    return result


# ============================================================================
# Main pipeline
# ============================================================================

# All verification columns written to the CSV
V_COLS = [
    "v_is_verified", "v_actual_date",
    "v_pr_link", "v_pr_date", "v_pr_title", "v_pr_key_info",
    "v_is_material", "v_confidence", "v_summary", "v_error",
]


def validate_dataset(
    input_file:    str,
    output_file:   str  = None,
    limit:         int  = None,
    dry_run:       bool = False,
    skip_verified: bool = True,
    save_every:    int  = 25,
) -> pd.DataFrame:
    print(f"\nLoading {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows × {len(df.columns)} columns\n")

    noise_df = identify_noise_rows(df)
    if noise_df.empty:
        print("\nNo noise candidates — dataset looks clean.")
        return df

    if limit:
        noise_df = noise_df.head(limit)
        print(f"Processing first {limit} noise candidates (--limit)\n")

    if dry_run:
        date_col  = "event_trading_date" if "event_trading_date" in noise_df.columns else "event_date"
        show_cols = [c for c in ["ticker", date_col, "move_pct", "atr_pct",
                                  "move_class_norm", "drug_name", "nct_id"]
                     if c in noise_df.columns]
        print("\n[DRY RUN] Rows that would be verified:")
        print(noise_df[show_cols].to_string(index=False))
        return df

    for col in V_COLS:
        if col not in df.columns:
            df[col] = None

    output_file = output_file or input_file.replace(".csv", "_validated.csv")
    date_col    = "event_trading_date" if "event_trading_date" in noise_df.columns else "event_date"

    print(f"Verifying {len(noise_df)} rows via Perplexity + PR fetch ...")
    print(f"Output: {output_file}\n")

    verified_count  = 0
    false_pos_count = 0
    date_fix_count  = 0
    error_count     = 0
    pr_fetched      = 0

    for i, (idx, row) in enumerate(noise_df.iterrows()):
        if skip_verified and pd.notna(df.at[idx, "v_is_verified"]):
            continue

        ticker = row.get("ticker", "?")
        date   = row.get(date_col, "?")
        move   = row.get("move_pct", 0)

        print(f"[{i+1}/{len(noise_df)}] {ticker} {date} ({move:+.1f}%) ...",
              end=" ", flush=True)

        result = verify_row(row)

        df.at[idx, "v_is_verified"] = result.is_verified
        df.at[idx, "v_actual_date"] = result.actual_date
        df.at[idx, "v_pr_link"]     = result.pr_link
        df.at[idx, "v_pr_date"]     = result.pr_date
        df.at[idx, "v_pr_title"]    = result.pr_title
        df.at[idx, "v_pr_key_info"] = result.pr_key_info
        df.at[idx, "v_is_material"] = result.is_material
        df.at[idx, "v_confidence"]  = result.confidence
        df.at[idx, "v_summary"]     = result.summary
        df.at[idx, "v_error"]       = result.error

        if result.pr_link:
            pr_fetched += 1

        if result.error and not result.is_verified:
            error_count += 1
            status = f"ERROR: {result.error[:40]}"
        elif result.is_verified:
            verified_count += 1
            pr_info = f" | PR: {result.pr_title[:40]}..." if result.pr_title else ""
            status  = f"VERIFIED ({result.confidence}){pr_info}"
        elif result.actual_date and result.actual_date != str(date):
            date_fix_count += 1
            status = f"WRONG DATE -> {result.actual_date}"
        else:
            false_pos_count += 1
            status = "FALSE POSITIVE"

        print(status, flush=True)
        time.sleep(RATE_LIMIT_DELAY)

        if (i + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
            print(f"  [Saved -> {output_file}]")

    df.to_csv(output_file, index=False)

    processed = verified_count + false_pos_count + date_fix_count + error_count
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Noise candidates:    {len(noise_df):,}")
    print(f"Processed:           {processed:,}")
    print(f"  Verified:          {verified_count:,}")
    print(f"  Date corrections:  {date_fix_count:,}")
    print(f"  False positives:   {false_pos_count:,}")
    print(f"  Errors:            {error_count:,}")
    print(f"PR pages fetched:    {pr_fetched:,}")
    if processed > 0:
        fp_rate = 100 * false_pos_count / processed
        print(f"False positive rate: {fp_rate:.1f}%")
        if fp_rate > 20:
            print("  => High — consider removing or re-enriching flagged rows.")
        elif fp_rate > 5:
            print("  => Moderate — review flagged rows.")
        else:
            print("  => Low — dataset looks reliable.")
    print(f"\nSaved -> {output_file}")
    return df


# ============================================================================
# Cleanup report
# ============================================================================

def generate_cleanup_report(df: pd.DataFrame,
                            output_path: str = "validation_report.csv") -> None:
    if "v_is_verified" not in df.columns:
        print("No verification data found. Run validate_dataset first.")
        return

    date_col = "event_trading_date" if "event_trading_date" in df.columns else "event_date"

    false_positives  = df[df["v_is_verified"] == False].copy()
    date_corrections = df[
        df["v_actual_date"].notna() &
        (df["v_actual_date"] != df[date_col].astype(str))
    ].copy()

    show = [c for c in ["ticker", date_col, "move_pct", "atr_pct", "move_class_norm",
                         "drug_name", "v_is_verified", "v_actual_date",
                         "v_pr_link", "v_pr_date", "v_pr_title", "v_pr_key_info",
                         "v_summary"]
            if c in df.columns]

    fp = false_positives[show].copy();  fp["action"] = "REMOVE_OR_REVIEW"
    dc = date_corrections[show].copy(); dc["action"] = "FIX_DATE"

    pd.concat([fp, dc], ignore_index=True).to_csv(output_path, index=False)
    print(f"\nCleanup report -> {output_path}")
    print(f"  False positives to remove: {len(false_positives):,}")
    print(f"  Date corrections needed:   {len(date_corrections):,}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate biotech catalyst noise rows via Perplexity + PR fetch"
    )
    parser.add_argument("--input",         default="enriched_all_clinical.csv")
    parser.add_argument("--output",        default=None)
    parser.add_argument("--limit",         type=int,   default=None,
                        help="Max noise rows to process (for testing)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Show which rows would be validated, then exit")
    parser.add_argument("--report",        action="store_true",
                        help="Generate cleanup report from a validated CSV")
    parser.add_argument("--atr-threshold", type=float, default=ATR_NOISE_THRESHOLD,
                        help=f"Noise threshold in ATR multiples (default: {ATR_NOISE_THRESHOLD})")
    parser.add_argument("--save-every",    type=int,   default=25,
                        help="Save progress every N rows (default: 25)")
    args = parser.parse_args()

    ATR_NOISE_THRESHOLD = args.atr_threshold
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.report:
        generate_cleanup_report(pd.read_csv(args.input),
                                args.output or "validation_report.csv")
    else:
        validate_dataset(
            input_file    = args.input,
            output_file   = args.output,
            limit         = args.limit,
            dry_run       = args.dry_run,
            save_every    = args.save_every,
        )

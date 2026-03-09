"""
Backfill Event Source Links
============================
Finds the best credible source link for rows missing v_pr_link in clean_v2.

Reuses _query_perplexity() and _load_api_key() from find_press_release_urls.py.

New vs existing script:
  - Targets v_pr_link (not press_release_url)
  - Writes to best_event_link (new column, never overwrites v_pr_link)
  - Tiered URL preference: official PR > regulatory > IR/CT.gov > reputable news
  - Updates v_is_verified=True for DATE_FIXED rows where a link is confirmed

v_is_verified update rule:
  DATE_FIXED + link found  → True   (event confirmed by new source, date was corrected)
  OK + link found          → leave  (already True, no change needed)
  FIX_DATE / FLAG_ERROR    → leave  (event date still unconfirmed)

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3

    # Dry run — show target rows
    python -m scripts.backfill_event_links --dry-run

    # Full run
    python -m scripts.backfill_event_links \\
        --input  enriched_all_clinical_clean_v2.csv \\
        --output enriched_all_clinical_clean_v2_linked.csv
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.find_press_release_urls import _query_perplexity, _load_api_key


# ── URL tier ranking (lower number = more preferred) ────────────────────────
_TIERS = [
    # Tier 1 — official PR wires
    (1, ["businesswire.com", "globenewswire.com", "prnewswire.com"]),
    # Tier 2 — regulatory / SEC
    (2, ["sec.gov", "fda.gov", "ema.europa.eu"]),
    # Tier 3 — IR sites, CT.gov, journals
    (3, ["clinicaltrials.gov", "pubmed.ncbi", "nejm.org", "nature.com",
         "ir.", "investor.", "newsroom.", "clinicaltrialresults"]),
    # Tier 4 — reputable biotech / financial news
    (4, ["biospace.com", "fiercebiotech.com", "statnews.com", "endpoints.news",
         "reuters.com", "bloomberg.com", "wsj.com", "seekingalpha.com"]),
]

_SKIP_DOMAINS = [
    "google.com", "bing.com", "yahoo.com",
    "twitter.com", "x.com", "reddit.com", "wikipedia.org",
]


def _url_tier(url: str) -> int:
    u = url.lower()
    for tier, domains in _TIERS:
        if any(d in u for d in domains):
            return tier
    return 99


def _pick_best_url(content_url: str, citations: list) -> str:
    """Return the highest-tier URL from content + all citations."""
    candidates = []
    if content_url and content_url.startswith("http"):
        candidates.append(content_url)
    for c in citations:
        if isinstance(c, str) and c.startswith("http"):
            candidates.append(c)

    # Remove junk domains
    candidates = [u for u in candidates if not any(s in u.lower() for s in _SKIP_DOMAINS)]
    if not candidates:
        return ""

    # Sort by (tier, url_length) — shortest URL at same tier is usually cleaner
    return min(candidates, key=lambda u: (_url_tier(u), len(u)))


# ── Main ────────────────────────────────────────────────────────────────────

def backfill_event_links(
    input_file:  str   = "enriched_all_clinical_clean_v2.csv",
    output_file: str   = "enriched_all_clinical_clean_v2_linked.csv",
    delay:       float = 1.5,
    save_every:  int   = 10,
    dry_run:     bool  = False,
) -> pd.DataFrame:

    df = pd.read_csv(input_file)

    # Rows missing v_pr_link
    no_pr = df["v_pr_link"].isna() | (
        df["v_pr_link"].astype(str).str.strip().isin(["", "nan", "None"])
    )
    target_idx = df[no_pr].index.tolist()

    print(f"Loaded {len(df):,} rows from {input_file}")
    print(f"Missing v_pr_link: {len(target_idx)} rows to process")
    print(f"  DATE_FIXED: {(df.loc[target_idx,'v_action']=='DATE_FIXED').sum()}"
          f"  |  OK: {(df.loc[target_idx,'v_action']=='OK').sum()}"
          f"  |  Other: {(~df.loc[target_idx,'v_action'].isin(['DATE_FIXED','OK'])).sum()}")

    if dry_run:
        print("\n[DRY RUN] First 10 target rows:")
        print(df.loc[target_idx[:10],
                     ["ticker", "event_date", "v_action", "v_is_verified",
                      "catalyst_summary"]].to_string())
        return df

    if "best_event_link" not in df.columns:
        df["best_event_link"] = ""

    api_key = _load_api_key()
    filled = 0
    verified_updated = 0

    for i, idx in enumerate(target_idx):
        row     = df.loc[idx]
        ticker  = str(row.get("ticker", ""))
        date    = str(row.get("event_date", ""))
        summary = str(row.get("catalyst_summary") or row.get("v_summary") or "")

        print(f"[{i+1}/{len(target_idx)}] {ticker} {date} ...", end=" ", flush=True)

        result = _query_perplexity(ticker, date, summary, api_key)
        best   = _pick_best_url(result["url"], result["citations"])

        if best:
            df.at[idx, "best_event_link"] = best
            filled += 1

            # Update v_is_verified for DATE_FIXED rows only
            if df.at[idx, "v_action"] == "DATE_FIXED":
                df.at[idx, "v_is_verified"] = True
                verified_updated += 1
                print(f"CONFIRMED (tier {_url_tier(best)}) → v_is_verified=True  {best[:70]}")
            else:
                print(f"FOUND (tier {_url_tier(best)})  {best[:70]}")
        else:
            print("not found")

        if (i + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
            print(f"  [checkpoint: {filled} found, {verified_updated} verified updated]")

        time.sleep(delay)

    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print("BACKFILL SUMMARY")
    print(f"{'='*60}")
    print(f"Target rows:          {len(target_idx):,}")
    print(f"best_event_link filled: {filled:,}")
    print(f"Unresolved:           {len(target_idx) - filled:,}")
    print(f"v_is_verified → True: {verified_updated:,}  (DATE_FIXED rows with new link)")
    print(f"\nOutput → {output_file}")

    return df


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill best_event_link for rows missing v_pr_link"
    )
    parser.add_argument("--input",  default="enriched_all_clinical_clean_v2.csv")
    parser.add_argument("--output", default="enriched_all_clinical_clean_v2_linked.csv")
    parser.add_argument("--delay",      type=float, default=1.5,
                        help="Seconds between Perplexity calls (default: 1.5)")
    parser.add_argument("--save-every", type=int,   default=10,
                        help="Checkpoint save frequency (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print target rows without making API calls")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    backfill_event_links(
        input_file  = args.input,
        output_file = args.output,
        delay       = args.delay,
        save_every  = args.save_every,
        dry_run     = args.dry_run,
    )

"""
Find missing press release URLs using Perplexity API (sonar-pro with web search).

Usage:
    python -m scripts.find_press_release_urls
    python -m scripts.find_press_release_urls --input enriched_final.csv --output enriched_final.csv
    python -m scripts.find_press_release_urls --dry-run   # show what would be searched
"""

import os
import re
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key from .env
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_api_key() -> str:
    """Load Perplexity API key from .env file."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("PERPLEXITY_API_KEY="):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not key:
        raise RuntimeError(f"PERPLEXITY_API_KEY not found in {ENV_PATH} or environment")
    return key


def _extract_url(text: str) -> str:
    """Extract first plausible press release URL from API response text."""
    # Match URLs from common press release / IR domains
    url_pattern = re.compile(
        r'https?://[^\s<>"\')\]]+(?:businesswire|globenewswire|prnewswire|sec\.gov|ir\.|investor\.|newsroom)[^\s<>"\')\]]*'
    )
    match = url_pattern.search(text)
    if match:
        url = match.group(0).rstrip(".,;:)")
        return url

    # Fallback: any URL
    general_pattern = re.compile(r'https?://[^\s<>"\')\]]+')
    match = general_pattern.search(text)
    if match:
        url = match.group(0).rstrip(".,;:)")
        # Skip search engine URLs, social media, etc.
        skip_domains = ["google.com", "bing.com", "yahoo.com", "twitter.com", "x.com", "reddit.com"]
        if not any(d in url for d in skip_domains):
            return url

    return ""


def _query_perplexity(ticker: str, event_date: str, summary: str, api_key: str) -> dict:
    """Query Perplexity API for a press release URL. Returns dict with url, raw_response, citations."""
    summary_truncated = summary[:200] if summary else ""

    prompt = (
        f"Find the official press release or news article URL for this biotech stock event:\n"
        f"- Ticker: {ticker}\n"
        f"- Date: {event_date}\n"
        f"- Event: {summary_truncated}\n\n"
        f"Return ONLY the most relevant URL from official sources like: "
        f"ir.company.com, businesswire.com, globenewswire.com, prnewswire.com, sec.gov, "
        f"or reputable financial news sites. "
        f"If you cannot find a specific URL, respond with 'NO_URL_FOUND'."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
    }

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])

        # Try to extract URL from response content first
        url = _extract_url(content)

        # If no URL in content, check citations
        if not url and citations:
            for cite in citations:
                cite_str = cite if isinstance(cite, str) else str(cite)
                candidate = _extract_url(cite_str)
                if candidate:
                    url = candidate
                    break
            # If citations are plain URLs
            if not url:
                for cite in citations:
                    if isinstance(cite, str) and cite.startswith("http"):
                        url = cite
                        break

        return {"url": url, "raw_response": content[:500], "citations": citations}

    except requests.exceptions.HTTPError as e:
        return {"url": "", "raw_response": f"HTTP Error: {e}", "citations": []}
    except Exception as e:
        return {"url": "", "raw_response": f"Error: {e}", "citations": []}


def find_press_release_urls(
    input_file: str = "enriched_final.csv",
    output_file: str = "enriched_final.csv",
    dry_run: bool = False,
    save_every: int = 10,
    delay: float = 1.0,
):
    """Find and populate missing press_release_url values using Perplexity API."""
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Identify rows missing URLs
    missing_mask = df["press_release_url"].isna() | (df["press_release_url"] == "")
    missing_idx = df[missing_mask].index.tolist()
    already_has = len(df) - len(missing_idx)
    print(f"Already have URLs: {already_has}/{len(df)}")
    print(f"Missing URLs: {len(missing_idx)}")

    if dry_run:
        print("\n--- DRY RUN: would search for these rows ---")
        for i, idx in enumerate(missing_idx[:10]):
            row = df.loc[idx]
            print(f"  [{i+1}] {row['ticker']} | {row['event_date']} | {str(row.get('catalyst_summary', ''))[:80]}")
        if len(missing_idx) > 10:
            print(f"  ... and {len(missing_idx) - 10} more")
        return

    api_key = _load_api_key()
    print(f"\nSearching for {len(missing_idx)} URLs via Perplexity API...")

    # Progress tracking
    found_count = 0
    error_count = 0
    search_log = []

    for i, idx in enumerate(missing_idx):
        row = df.loc[idx]
        ticker = str(row.get("ticker", ""))
        event_date = str(row.get("event_date", ""))
        summary = str(row.get("catalyst_summary", "")) if pd.notna(row.get("catalyst_summary")) else ""

        print(f"  [{i+1}/{len(missing_idx)}] {ticker} {event_date}...", end=" ", flush=True)

        result = _query_perplexity(ticker, event_date, summary, api_key)

        if result["url"]:
            df.at[idx, "press_release_url"] = result["url"]
            found_count += 1
            print(f"FOUND: {result['url'][:80]}")
        else:
            print("not found")

        search_log.append({
            "idx": int(idx),
            "ticker": ticker,
            "event_date": event_date,
            "url_found": result["url"],
            "raw_response": result["raw_response"][:200],
        })

        # Save progress periodically
        if (i + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
            print(f"  [saved progress: {found_count} found so far]")

        # Rate limiting
        time.sleep(delay)

    # Final save
    df.to_csv(output_file, index=False)

    # Save search log
    log_path = os.path.join(os.path.dirname(output_file) or ".", "url_search_log.json")
    with open(log_path, "w") as f:
        json.dump(search_log, f, indent=2)

    # Summary
    total_urls = (df["press_release_url"].notna() & (df["press_release_url"] != "")).sum()
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"URLs found this run:  {found_count}/{len(missing_idx)}")
    print(f"Total URLs now:       {total_urls}/{len(df)} ({total_urls/len(df)*100:.0f}%)")
    print(f"Search log:           {log_path}")
    print(f"Output:               {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find missing press release URLs")
    parser.add_argument("--input", default="enriched_final.csv")
    parser.add_argument("--output", default="enriched_final.csv")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be searched")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    args = parser.parse_args()

    find_press_release_urls(
        input_file=args.input,
        output_file=args.output,
        dry_run=args.dry_run,
        save_every=args.save_every,
        delay=args.delay,
    )

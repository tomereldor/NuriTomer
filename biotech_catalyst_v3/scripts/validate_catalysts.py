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
   actually existed on or around that date.
3. Flags false positives (no news found), date corrections, and errors.

Column names
------------
The primary CSV uses:
  - stock_movement_atr_normalized  (ATR-normalized move; alias: normalized_move)
  - move_class_norm                (Noise / Low / Medium / High / Extreme)
  - event_trading_date             (trading day of the event, preferred over event_date)

ATR methodology (from utils/volatility.py)
------------------------------------------
  Wilder's RMA: ewm(alpha=1/20, adjust=False) — same as TradingView default.
  Lookback: 20 trading days (≈ 1 calendar month) STRICTLY before the event date.
  atr_pct = (ATR value / last pre-event closing price) × 100.
  Noise threshold: move < 1.5× ATR (stock moved less than 1.5 "normal days").

Usage
-----
    cd /Users/tomer/Code/NuriTomer/biotech_catalyst_v3

    # Dry run — see which rows would be validated
    python -m scripts.validate_catalysts --input enriched_all_clinical.csv --dry-run

    # Validate first 20 noise rows (for testing)
    python -m scripts.validate_catalysts --input enriched_all_clinical.csv --limit 20

    # Full validation run
    python -m scripts.validate_catalysts --input enriched_all_clinical.csv \
        --output enriched_all_clinical_validated.csv

    # Generate cleanup report from a previously validated CSV
    python -m scripts.validate_catalysts --input enriched_all_clinical_validated.csv --report
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Configuration
# ============================================================================

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_URL    = "https://api.perplexity.ai/chat/completions"
MODEL             = "sonar-pro"
RATE_LIMIT_DELAY  = 1.5  # seconds between API calls

# Noise classification threshold (matches utils/volatility.py ATR_NOISE_THRESHOLD)
ATR_NOISE_THRESHOLD = 1.5  # moves < 1.5× ATR are flagged as noise


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class ValidationResult:
    is_verified: bool  = False
    actual_date: str   = ""
    pr_link:     str   = ""
    is_material: bool  = False   # True = major data readout; False = minor update
    confidence:  str   = ""      # "high" / "medium" / "low"
    summary:     str   = ""
    error:       str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Identify noise rows
# ============================================================================

def identify_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows that are likely noise and warrant verification.

    Primary filter: move_class_norm == 'Noise'  (< 1.5× ATR, pre-computed)
    Fallback:       if move_class_norm is missing, recompute from atr_pct.

    All rows in enriched_all_clinical.csv are catalyst_type == 'Clinical Data',
    so that filter is not needed here.
    """
    df = df.copy()

    # Resolve the normalized move column (new name or backward-compat alias)
    nm_col = (
        'stock_movement_atr_normalized' if 'stock_movement_atr_normalized' in df.columns
        else 'normalized_move' if 'normalized_move' in df.columns
        else None
    )

    # Use pre-computed move_class_norm if available
    if 'move_class_norm' in df.columns:
        is_noise = df['move_class_norm'] == 'Noise'
    elif nm_col:
        is_noise = df[nm_col].notna() & (df[nm_col] < ATR_NOISE_THRESHOLD)
    else:
        # Fallback: recompute from raw columns
        has_atr = df['atr_pct'].notna() & (df['atr_pct'] > 0)
        nm = df['move_pct'].abs() / df['atr_pct'].where(has_atr, other=float('nan'))
        is_noise = has_atr & (nm < ATR_NOISE_THRESHOLD)

    df['_is_noise'] = is_noise
    noise_df = df[df['_is_noise']].copy()

    total      = len(df)
    n_noise    = len(noise_df)
    n_with_atr = df['atr_pct'].notna().sum()

    print(f"Total rows:          {total:,}")
    print(f"Rows with ATR:       {n_with_atr:,}  ({100*n_with_atr/total:.1f}%)")
    print(f"Noise candidates:    {n_noise:,}  ({100*n_noise/total:.1f}%)")
    if nm_col:
        median_nm = df[nm_col].median()
        print(f"Median ATR-norm:     {median_nm:.2f}×  (overall dataset)")

    return noise_df


# ============================================================================
# Perplexity verification prompt
# ============================================================================

def build_verification_prompt(row: pd.Series) -> str:
    """
    Skeptical prompt: asks Perplexity to VERIFY (not assume) the catalyst exists.
    """
    ticker   = row.get('ticker', '')
    date     = row.get('event_trading_date') or row.get('event_date', '')
    drug     = row.get('drug_name', '') or 'unknown drug'
    nct_id   = str(row.get('nct_id', '') or '')
    trial    = row.get('ct_official_title', '') or ''
    phase    = row.get('ct_phase', '') or ''
    summary  = str(row.get('catalyst_summary', '') or '')[:200]

    ctx_parts = []
    if nct_id.startswith('NCT'):
        ctx_parts.append(f"NCT ID: {nct_id}")
    if drug and drug != 'unknown drug':
        ctx_parts.append(f"Drug: {drug}")
    if trial:
        ctx_parts.append(f"Trial: {trial[:120]}")
    if phase:
        ctx_parts.append(f"Phase: {phase}")
    context = "; ".join(ctx_parts) if ctx_parts else "no specific drug/trial info"

    prompt = f"""VERIFICATION TASK: Determine whether a specific biotech clinical-data event actually occurred.

CLAIMED EVENT:
- Ticker: {ticker}
- Date:   {date}
- Context: {context}
- Claimed catalyst: "{summary}"

IMPORTANT: Be SKEPTICAL. The claimed catalyst may be WRONG or HALLUCINATED.

Respond with ONLY a valid JSON object (no markdown, no other text):

{{
    "is_verified": true/false,
    "actual_date": "YYYY-MM-DD or null",
    "pr_link": "official press-release URL or null",
    "is_material": true/false,
    "confidence": "high/medium/low",
    "summary": "one sentence: what actually happened, or 'No clinical news found for this date'"
}}

RULES:
1. is_verified = true ONLY if you find concrete evidence of clinical news on or within 1 day of {date}.
2. If news exists but on a DIFFERENT date, set is_verified=false and put the correct date in actual_date.
3. If NO clinical news exists for {ticker} around this date, say so clearly in summary.
4. pr_link = official company press release (businesswire, globenewswire, prnewswire, SEC) or null.
5. is_material = true only for significant data readouts (Phase 2/3 results, FDA decisions).

JSON only:"""

    return prompt


# ============================================================================
# Perplexity API call
# ============================================================================

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
                    "Your job is to VERIFY claims, not assume they are true. "
                    "Always respond with valid JSON only — no markdown, no preamble."
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
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text[:120]}"

            content = (resp.json()
                       .get('choices', [{}])[0]
                       .get('message', {})
                       .get('content', ''))
            if not content:
                return None, "Empty response"

            # Strip optional markdown fences
            content = content.strip()
            for fence in ('```json', '```'):
                if content.startswith(fence):
                    content = content[len(fence):]
            if content.endswith('```'):
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
            return None, "Request timeout"
        except Exception as e:
            return None, str(e)[:120]

    return None, "Max retries exceeded"


def verify_row(row: pd.Series) -> ValidationResult:
    prompt = build_verification_prompt(row)
    parsed, error = call_perplexity(prompt)

    if error:
        return ValidationResult(error=error)
    if not parsed:
        return ValidationResult(error="No response parsed")

    return ValidationResult(
        is_verified = parsed.get('is_verified', False),
        actual_date = parsed.get('actual_date') or "",
        pr_link     = parsed.get('pr_link') or "",
        is_material = parsed.get('is_material', False),
        confidence  = parsed.get('confidence', 'low'),
        summary     = parsed.get('summary', ''),
        error       = "",
    )


# ============================================================================
# Main validation pipeline
# ============================================================================

def validate_dataset(
    input_file:   str,
    output_file:  str  = None,
    limit:        int  = None,
    dry_run:      bool = False,
    skip_verified: bool = True,
    save_every:   int  = 25,
) -> pd.DataFrame:
    """
    Validate noise rows via Perplexity and save v_* verification columns.

    Partial progress is saved to <output_file> every save_every rows so the
    run is resumable: rows with v_is_verified already filled are skipped.
    """
    print(f"\nLoading {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"  {len(df):,} rows × {len(df.columns)} columns\n")

    noise_df = identify_noise_rows(df)

    if noise_df.empty:
        print("\nNo noise candidates found — dataset looks clean.")
        return df

    if limit:
        noise_df = noise_df.head(limit)
        print(f"Processing first {limit} noise candidates (--limit)")

    if dry_run:
        date_col = 'event_trading_date' if 'event_trading_date' in noise_df.columns else 'event_date'
        show_cols = ['ticker', date_col, 'move_pct', 'atr_pct',
                     'move_class_norm', 'drug_name', 'nct_id']
        show_cols = [c for c in show_cols if c in noise_df.columns]
        print("\n[DRY RUN] Rows that would be verified:")
        print(noise_df[show_cols].to_string(index=False))
        return df

    # Ensure verification columns exist
    v_cols = ['v_is_verified', 'v_actual_date', 'v_pr_link',
              'v_is_material', 'v_confidence', 'v_summary', 'v_error']
    for col in v_cols:
        if col not in df.columns:
            df[col] = None

    output_file = output_file or input_file.replace('.csv', '_validated.csv')

    print(f"\nVerifying {len(noise_df)} noise rows via Perplexity API ...")
    print(f"Output: {output_file}\n")

    verified_count    = 0
    false_pos_count   = 0
    date_fix_count    = 0
    error_count       = 0

    date_col = 'event_trading_date' if 'event_trading_date' in noise_df.columns else 'event_date'

    for i, (idx, row) in enumerate(noise_df.iterrows()):
        # Resume: skip if already verified
        if skip_verified and pd.notna(df.at[idx, 'v_is_verified']):
            continue

        ticker = row.get('ticker', '?')
        date   = row.get(date_col, '?')
        move   = row.get('move_pct', 0)

        print(f"[{i+1}/{len(noise_df)}] {ticker} {date} ({move:+.1f}%) ...", end=" ", flush=True)

        result = verify_row(row)

        df.at[idx, 'v_is_verified'] = result.is_verified
        df.at[idx, 'v_actual_date'] = result.actual_date
        df.at[idx, 'v_pr_link']     = result.pr_link
        df.at[idx, 'v_is_material'] = result.is_material
        df.at[idx, 'v_confidence']  = result.confidence
        df.at[idx, 'v_summary']     = result.summary
        df.at[idx, 'v_error']       = result.error

        if result.error:
            error_count += 1
            print(f"ERROR: {result.error[:50]}")
        elif result.is_verified:
            verified_count += 1
            print(f"VERIFIED ({result.confidence})")
        elif result.actual_date and result.actual_date != str(date):
            date_fix_count += 1
            print(f"WRONG DATE -> {result.actual_date}")
        else:
            false_pos_count += 1
            print(f"FALSE POSITIVE")

        time.sleep(RATE_LIMIT_DELAY)

        if (i + 1) % save_every == 0:
            df.to_csv(output_file, index=False)
            print(f"  [Saved progress -> {output_file}]")

    df.to_csv(output_file, index=False)

    # Summary
    processed = verified_count + false_pos_count + date_fix_count + error_count
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Noise candidates:  {len(noise_df):,}")
    print(f"Processed:         {processed:,}")
    print(f"  Verified:        {verified_count:,}")
    print(f"  Date corrections:{date_fix_count:,}")
    print(f"  False positives: {false_pos_count:,}")
    print(f"  Errors:          {error_count:,}")

    if processed > 0:
        fp_rate = 100 * false_pos_count / processed
        print(f"\nFalse positive rate: {fp_rate:.1f}%")
        if fp_rate > 20:
            print("  => High false positive rate — consider removing or re-enriching these rows.")
        elif fp_rate > 5:
            print("  => Moderate false positive rate — review flagged rows.")
        else:
            print("  => Low false positive rate — dataset looks reliable.")

    print(f"\nSaved -> {output_file}")
    return df


# ============================================================================
# Cleanup report
# ============================================================================

def generate_cleanup_report(df: pd.DataFrame,
                            output_path: str = "validation_report.csv") -> None:
    if 'v_is_verified' not in df.columns:
        print("No verification data found. Run validate_dataset first.")
        return

    date_col = 'event_trading_date' if 'event_trading_date' in df.columns else 'event_date'

    false_positives = df[df['v_is_verified'] == False].copy()
    date_corrections = df[
        df['v_actual_date'].notna() &
        (df['v_actual_date'] != df[date_col].astype(str))
    ].copy()

    show_cols = [c for c in
                 ['ticker', date_col, 'move_pct', 'atr_pct', 'move_class_norm',
                  'drug_name', 'v_is_verified', 'v_actual_date', 'v_pr_link', 'v_summary']
                 if c in df.columns]

    fp_report   = false_positives[show_cols].copy()
    fp_report['action'] = 'REMOVE_OR_REVIEW'

    dc_report   = date_corrections[show_cols].copy()
    dc_report['action'] = 'FIX_DATE'

    report = pd.concat([fp_report, dc_report], ignore_index=True)
    report.to_csv(output_path, index=False)

    print(f"\nCleanup Report -> {output_path}")
    print(f"  False positives to remove: {len(false_positives):,}")
    print(f"  Date corrections needed:   {len(date_corrections):,}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate biotech catalyst noise rows via Perplexity API"
    )
    parser.add_argument("--input",  default="enriched_all_clinical.csv",
                        help="Input CSV (default: enriched_all_clinical.csv)")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: <input>_validated.csv)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max noise rows to process (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print which rows would be validated, then exit")
    parser.add_argument("--report", action="store_true",
                        help="Generate cleanup report from an already-validated CSV")
    parser.add_argument("--atr-threshold", type=float, default=ATR_NOISE_THRESHOLD,
                        help=f"ATR multiple below which a row is 'noise' (default: {ATR_NOISE_THRESHOLD})")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save progress every N rows (default: 25)")
    args = parser.parse_args()

    ATR_NOISE_THRESHOLD = args.atr_threshold

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.report:
        df = pd.read_csv(args.input)
        generate_cleanup_report(df, args.output or "validation_report.csv")
    else:
        validate_dataset(
            input_file  = args.input,
            output_file = args.output,
            limit       = args.limit,
            dry_run     = args.dry_run,
            save_every  = args.save_every,
        )

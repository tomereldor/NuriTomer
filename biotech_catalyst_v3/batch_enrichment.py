"""
Batch enrichment for new events.
Processes events and saves progress periodically.

NOTE: This script requires the following components to be implemented:
  - AICatalystResearcher  (Perplexity-based catalyst research)
  - FinancialDataFetcher  (yfinance + Perplexity financial data)
  - EnrichedEvent / ClinicalTrialData / FinancialData dataclasses
  - calculate_quality_score function

These were previously in pipeline.py but that file was a broken skeleton.
They need to be rebuilt in a proper module before this script can run.
The working parts of the pipeline are in:
  - clients/clinicaltrials_client.py  (ClinicalTrials.gov API - WORKING)
  - utils/data_quality.py             (quality threshold, catalyst fix - WORKING)
  - utils/volatility.py               (ATR normalization - WORKING)
  - scripts/fix_existing_data.py      (post-processing - WORKING)
  - scripts/fix_missing_nct.py        (NCT backfill - WORKING)
"""

import pandas as pd
import time
from datetime import datetime
from dataclasses import dataclass, asdict, field
import os
import sys

from utils.data_quality import (
    add_quality_threshold,
    fix_catalyst_type,
    validate_event_date,
    flag_missing_financials,
)
from clients.clinicaltrials_client import ClinicalTrialsClient
from clients.financial_client import FinancialDataFetcher

# TODO: Set your Perplexity API key as env var or replace here
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


# ---------------------------------------------------------------------------
# Data models (minimal stubs — expand when rebuilding the AI researcher)
# ---------------------------------------------------------------------------
@dataclass
class ClinicalTrialData:
    official_title: str = ""
    phase: str = ""
    enrollment: int = 0
    conditions: str = ""
    status: str = ""
    sponsor: str = ""
    allocation: str = ""
    primary_completion_date: str = ""
    error: str = ""


@dataclass
class FinancialData:
    market_cap_m: float = None
    current_price: float = None
    cash_position_m: float = None
    cash_runway_months: int = None
    short_percent: float = None
    institutional_ownership: float = None
    analyst_target: float = None
    analyst_rating: str = ""
    error: str = ""
    missing_fields: str = ""


@dataclass
class EnrichedEvent:
    ticker: str = ""
    event_date: str = ""
    event_type: str = ""
    move_pct: float = 0.0
    price_at_event: float = 0.0
    catalyst_type: str = ""
    catalyst_summary: str = ""
    drug_name: str = ""
    nct_id: str = ""
    indication: str = ""
    phase: str = ""
    is_pivotal: bool = False
    pivotal_evidence: str = ""
    press_release_url: str = ""
    primary_endpoint_met: str = ""
    primary_endpoint_result: str = ""
    ct_official_title: str = ""
    ct_phase: str = ""
    ct_enrollment: int = 0
    ct_conditions: str = ""
    ct_status: str = ""
    ct_sponsor: str = ""
    ct_allocation: str = ""
    ct_primary_completion: str = ""
    market_cap_m: float = None
    current_price: float = None
    cash_position_m: float = None
    cash_runway_months: int = None
    short_percent: float = None
    institutional_ownership: float = None
    analyst_target: float = None
    analyst_rating: str = ""
    data_quality_score: float = 0.0
    errors: str = ""
    timestamp: str = ""


def calculate_quality_score(ai_result, ct_data, fin_data) -> float:
    """Simple quality score based on data completeness."""
    score = 0.0
    total = 0.0

    # AI research fields
    for attr in ['catalyst_type', 'summary', 'drug_name']:
        total += 1
        val = getattr(ai_result, attr, None)
        if val and str(val) not in ('', 'Unknown', 'None'):
            score += 1

    # Clinical trial fields
    for attr in ['official_title', 'phase', 'conditions', 'sponsor']:
        total += 1
        val = getattr(ct_data, attr, None)
        if val and str(val) not in ('', 'None'):
            score += 1

    # Financial fields
    for attr in ['market_cap_m', 'current_price', 'cash_position_m']:
        total += 1
        val = getattr(fin_data, attr, None)
        if val is not None:
            score += 1

    return round(score / total, 2) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# TODO: Implement these classes to restore full pipeline functionality
# ---------------------------------------------------------------------------
class AICatalystResearcher:
    """TODO: Rebuild Perplexity-based catalyst researcher."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def research(self, ticker, date, move_pct):
        raise NotImplementedError(
            "AICatalystResearcher needs to be rebuilt. "
            "See perplexity_scanner.py for reference implementation."
        )



def enrich_events_batch(
    input_file: str = 'raw_moves_filtered.csv',
    output_file: str = 'enriched_new_batch.csv',
    existing_file: str = 'enriched_catalysts_all.csv',
    batch_size: int = 10,
    start_from: int = 0
):
    """
    Enrich events in batches, saving progress periodically.
    """
    # Load events to enrich
    events_df = pd.read_csv(input_file)
    print(f"Loaded {len(events_df)} events to enrich")

    # Load existing enriched events to avoid duplicates
    existing_keys = set()
    if os.path.exists(existing_file):
        existing = pd.read_csv(existing_file)
        existing_keys = set(zip(existing['ticker'].str.upper(), existing['event_date']))
        print(f"Loaded {len(existing_keys)} existing events to skip")

    # Filter out already enriched events
    events_df['key'] = list(zip(events_df['Ticker'].str.upper(), events_df['Date']))
    events_df = events_df[~events_df['key'].isin(existing_keys)].drop(columns=['key'])
    print(f"Events to process after filtering: {len(events_df)}")

    if events_df.empty:
        print("No new events to enrich!")
        return pd.DataFrame()

    # Load any partial results
    results = []
    if os.path.exists(output_file):
        partial = pd.read_csv(output_file)
        results = partial.to_dict('records')
        processed_keys = set(zip(partial['ticker'].str.upper(), partial['event_date']))
        events_df['key'] = list(zip(events_df['Ticker'].str.upper(), events_df['Date']))
        events_df = events_df[~events_df['key'].isin(processed_keys)].drop(columns=['key'])
        print(f"Resumed from {len(results)} already processed events")
        print(f"Remaining events: {len(events_df)}")

    if events_df.empty:
        print("All events already processed!")
        return pd.DataFrame(results)

    # Skip to start_from if specified
    if start_from > 0:
        events_df = events_df.iloc[start_from:]
        print(f"Starting from event {start_from}")

    # Initialize modules
    ai_researcher = AICatalystResearcher(PERPLEXITY_API_KEY)
    ct_client = ClinicalTrialsClient(rate_limit=0.5)
    financial_fetcher = FinancialDataFetcher()

    total = len(events_df)
    start_time = time.time()

    for idx, (_, row) in enumerate(events_df.iterrows()):
        ticker = row['Ticker']
        date = row['Date']
        move_pct = row['Move_%']

        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (total - idx - 1) / rate if rate > 0 else 0

        print(f"\n[{idx+1}/{total}] {ticker} on {date} ({move_pct:+.1f}%) - ETA: {remaining/60:.1f}min")

        errors = []

        # Date validation (Issue 3)
        is_valid, date_err = validate_event_date(date)
        if not is_valid:
            errors.append(date_err)
            print(f"   Date issue: {date_err}")

        # AI Research
        try:
            ai_result = ai_researcher.research(ticker, date, move_pct)
            if ai_result.error:
                errors.append(ai_result.error)
        except Exception as e:
            print(f"   AI Error: {str(e)[:50]}")
            ai_result = type('obj', (object,), {
                'catalyst_type': 'Unknown', 'summary': '', 'drug_name': None,
                'nct_id': None, 'indication': None, 'phase': None,
                'is_pivotal': False, 'pivotal_evidence': None,
                'press_release_url': None, 'primary_endpoint_met': None,
                'primary_endpoint_result': None, 'error': str(e)
            })()
            errors.append(str(e))

        time.sleep(1.0)

        # NCT ID lookup (prioritized search via ClinicalTrials.gov API)
        nct_id = getattr(ai_result, 'nct_id', None)
        drug_name = getattr(ai_result, 'drug_name', None)
        if (not nct_id or not str(nct_id).startswith('NCT')) and drug_name:
            try:
                indication = getattr(ai_result, 'indication', None)
                phase = getattr(ai_result, 'phase', None)
                nct_id, search_log = ct_client.search_nct_prioritized(
                    drug_name=drug_name,
                    indication=indication,
                    phase=phase,
                )
                if nct_id:
                    print(f"   Found NCT: {nct_id} ({search_log.get('result')})")
                else:
                    print(f"   NCT not found ({search_log.get('result')})")
            except Exception as e:
                print(f"   NCT search error: {e}")

        # Fetch CT.gov data
        ct_data = ClinicalTrialData()
        if nct_id and str(nct_id).startswith('NCT'):
            try:
                trial = ct_client.fetch_trial_details(nct_id)
                if trial:
                    ct_data = ClinicalTrialData(
                        official_title=trial.official_title,
                        phase=trial.phase,
                        enrollment=trial.enrollment,
                        conditions=trial.conditions,
                        status=trial.status,
                        sponsor=trial.sponsor,
                        allocation=trial.allocation,
                        primary_completion_date=trial.primary_completion_date,
                    )
            except Exception as e:
                ct_data.error = f"CT.gov fetch error: {e}"
                errors.append(ct_data.error)

        # Fetch financial data for EVERY event
        fin_data = FinancialData()
        try:
            fin_data = financial_fetcher.fetch(ticker)
            if fin_data.error:
                errors.append(fin_data.error)
            if fin_data.missing_fields:
                errors.append(f"Missing: {fin_data.missing_fields}")
        except Exception as e:
            errors.append(f"Financial fetch error: {e}")
        time.sleep(0.3)

        # Calculate quality score
        quality_score = calculate_quality_score(ai_result, ct_data, fin_data)

        # Build enriched event
        enriched = EnrichedEvent(
            ticker=ticker,
            event_date=date,
            event_type=row.get('Type', 'Gainer' if move_pct > 0 else 'Loser'),
            move_pct=move_pct,
            price_at_event=row.get('Price_Event', 0),

            catalyst_type=getattr(ai_result, 'catalyst_type', 'Unknown'),
            catalyst_summary=getattr(ai_result, 'summary', ''),
            drug_name=getattr(ai_result, 'drug_name', '') or '',
            nct_id=nct_id or '',
            indication=getattr(ai_result, 'indication', '') or '',
            phase=getattr(ai_result, 'phase', '') or ct_data.phase or '',
            is_pivotal=getattr(ai_result, 'is_pivotal', False),
            pivotal_evidence=getattr(ai_result, 'pivotal_evidence', None),
            press_release_url=getattr(ai_result, 'press_release_url', '') or '',
            primary_endpoint_met=getattr(ai_result, 'primary_endpoint_met', '') or '',
            primary_endpoint_result=getattr(ai_result, 'primary_endpoint_result', '') or '',

            ct_official_title=ct_data.official_title,
            ct_phase=ct_data.phase,
            ct_enrollment=ct_data.enrollment,
            ct_conditions=ct_data.conditions,
            ct_status=ct_data.status,
            ct_sponsor=ct_data.sponsor,
            ct_allocation=ct_data.allocation,
            ct_primary_completion=ct_data.primary_completion_date,

            market_cap_m=fin_data.market_cap_m,
            current_price=fin_data.current_price,
            cash_position_m=fin_data.cash_position_m,
            cash_runway_months=fin_data.cash_runway_months,
            short_percent=fin_data.short_percent,
            institutional_ownership=fin_data.institutional_ownership,
            analyst_target=fin_data.analyst_target,
            analyst_rating=fin_data.analyst_rating,

            data_quality_score=quality_score,
            errors='; '.join(errors) if errors else '',
            timestamp=datetime.now().isoformat(),
        )

        results.append(asdict(enriched))
        print(f"   Catalyst: {getattr(ai_result, 'catalyst_type', 'Unknown')}, Quality: {quality_score:.2f}")

        # Save progress every batch_size events
        if (idx + 1) % batch_size == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            print(f"\n   >>> SAVED {len(results)} events to {output_file}")

    # Final save with post-processing
    results_df = pd.DataFrame(results)

    # Issue 1: Quality threshold
    results_df = add_quality_threshold(results_df, threshold=0.7)

    # Issue 2: Fix catalyst_type classification
    results_df['catalyst_type'] = results_df.apply(fix_catalyst_type, axis=1)

    # Issue 3: Flag any remaining missing financials
    results_df = flag_missing_financials(results_df)

    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"Total enriched: {len(results_df)}")
    passed = results_df.get('data_quality_threshold_passed', pd.Series(dtype=bool)).sum()
    print(f"Quality threshold passed: {passed}/{len(results_df)}")
    print(f"Catalyst types: {results_df['catalyst_type'].value_counts().to_dict()}")
    print(f"Saved to: {output_file}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch enrichment")
    parser.add_argument('--input', type=str, default='raw_moves_filtered.csv')
    parser.add_argument('--output', type=str, default='enriched_new_batch.csv')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--start-from', type=int, default=0)

    args = parser.parse_args()

    enrich_events_batch(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        start_from=args.start_from
    )

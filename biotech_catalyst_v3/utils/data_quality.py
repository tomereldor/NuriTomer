"""Data quality threshold and catalyst type classification utilities."""

import re
import pandas as pd
from datetime import timedelta
from typing import Tuple


# ----------------------------------------------------------------
# Issue 1: Quality threshold
# ----------------------------------------------------------------
def add_quality_threshold(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Add boolean data_quality_threshold_passed column."""
    df["data_quality_threshold_passed"] = df["data_quality_score"] >= threshold
    return df


# ----------------------------------------------------------------
# Issue 2: Catalyst type classification
# ----------------------------------------------------------------
UNKNOWN_PATTERNS = [
    "unable to identify",
    "no specific news",
    "no specific catalyst",
    "could not determine",
    "could not find",
    "not identified",
    "insufficient search results",
    "search results do not contain",
    "unable to determine",
    "unspecified catalyst",
    "unknown catalyst",
    "not found in available",
    "no specific catalyst, news",
    "not identified in available sources",
]


def _categorize_from_summary(summary_lower: str) -> str:
    """Try to assign a specific category based on summary keywords. Returns category or empty string."""
    # Acquisition / M&A
    if any(t in summary_lower for t in [
        "acquire", "acquisition", "merger", "buyout", "takeover",
        "agreed to acquire", "all-cash deal",
    ]):
        return "Acquisition/M&A"

    # Analyst activity
    if any(t in summary_lower for t in [
        "analyst", "rating", "upgrade", "downgrade", "initiated coverage",
        "initiating coverage", "price target", "reiterated", "outperform",
        "overweight", "buy rating",
    ]):
        return "Analyst"

    # Insider trading
    if any(t in summary_lower for t in [
        "insider", "ceo purchase", "director purchase", "insider stock purchase",
        "insider buying", "insider sale",
    ]):
        return "Insider Trading"

    # Corporate actions
    if any(t in summary_lower for t in [
        "reverse stock split", "nasdaq deficiency", "delisting", "stock split",
        "investor conference", "name change", "restructuring",
    ]):
        return "Corporate Action"

    # Pipeline updates (discontinuation, preclinical, etc.)
    if any(t in summary_lower for t in [
        "discontinued", "discontinuation", "halted", "preclinical",
        "pipeline", "shelved", "terminated program",
    ]):
        return "Pipeline Update"

    # Legal / investigation
    if any(t in summary_lower for t in [
        "investigation", "lawsuit", "litigation", "subpoena", "federal",
        "sec ", "settlement", "fraud",
    ]):
        return "Legal"

    # Technical / momentum (no fundamental catalyst)
    if any(t in summary_lower for t in [
        "rsi", "momentum", "52-week high", "short squeeze", "volatility",
        "oversold", "overbought", "activist investor", "optimistic comments",
    ]):
        return "Technical/Momentum"

    # Clinical data (broader catch)
    clinical_terms = [
        "phase 1", "phase 2", "phase 3", "clinical trial",
        "endpoint", "efficacy", "orr", "pfs", "data readout",
        "topline", "top-line", "clinical data",
    ]
    if any(t in summary_lower for t in clinical_terms):
        return "Clinical Data"

    # Regulatory
    if any(t in summary_lower for t in ["fda", "approval", "cleared", "accepted", "granted", "ema"]):
        return "Regulatory"

    # Earnings
    if any(t in summary_lower for t in ["earnings", "revenue", "quarter", "q1", "q2", "q3", "q4", "eps"]):
        return "Earnings"

    # Partnership
    if any(t in summary_lower for t in ["partnership", "collaboration", "license", "agreement", "deal"]):
        return "Partnership"

    # Financing
    if any(t in summary_lower for t in ["offering", "financing", "raise", "capital", "atm", "dilution"]):
        return "Financing"

    return ""


def fix_catalyst_type(row) -> str:
    """
    Properly classify catalyst_type:
    - Keep specific types as-is (Clinical Data, Regulatory, etc.)
    - Re-categorize 'Other: ...' rows into specific types when possible
    - 'Unknown': When we genuinely couldn't identify a catalyst
    """
    current_type = str(row.get("catalyst_type", ""))
    summary = str(row.get("catalyst_summary", ""))
    summary_lower = summary.lower()

    # If already a specific known category (not "Other: ..." or "Unknown"), keep it
    known_categories = {
        "Clinical Data", "Regulatory", "Earnings", "Partnership",
        "Financing", "Clinical Safety Event", "Analyst", "Acquisition/M&A",
        "Insider Trading", "Corporate Action", "Pipeline Update",
        "Technical/Momentum", "Legal",
    }
    if current_type in known_categories:
        return current_type

    # Check if it's actually unknown (couldn't find catalyst)
    if any(pattern in summary_lower for pattern in UNKNOWN_PATTERNS):
        return "Unknown"

    # Empty / NaN summary -> Unknown
    if not summary or summary == "nan" or len(summary.strip()) == 0:
        return "Unknown"

    # Try to categorize from summary keywords
    category = _categorize_from_summary(summary_lower)
    if category:
        return category

    # Still uncategorized with a real summary -> keep as descriptive "Other: ..."
    title = summary[:60].split(".")[0].strip()
    if title:
        return f"Other: {title}"

    return current_type or "Unknown"


def classify_catalyst_type_from_summary(summary: str) -> Tuple[str, str]:
    """
    Classify catalyst type from a summary string.
    Returns (catalyst_type, short_title).
    Useful when re-categorizing or for new enrichment.
    """
    if not summary or summary == "nan":
        return "Unknown", ""

    summary_lower = summary.lower()

    if any(p in summary_lower for p in UNKNOWN_PATTERNS):
        return "Unknown", ""

    title = summary[:60].split(".")[0].strip()

    category = _categorize_from_summary(summary_lower)
    if category:
        return category, title

    return f"Other: {title}", title


# ----------------------------------------------------------------
# Issue 3: Date validation
# ----------------------------------------------------------------
def validate_event_date(event_date: str, max_days_back: int = 730) -> Tuple[bool, str]:
    """
    Validate event date is in the past and within range.
    Returns (is_valid, error_message or empty string).
    """
    try:
        event_dt = pd.to_datetime(event_date)
        today = pd.Timestamp.now()
        cutoff = today - timedelta(days=max_days_back)

        if event_dt > today:
            return False, f"Future event: {event_date}"
        if event_dt < cutoff:
            return False, f"Event older than {max_days_back} days: {event_date}"

        return True, ""

    except Exception as e:
        return False, f"Invalid date: {event_date} ({e})"


def flag_date_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_valid_date column and log date issues in errors."""
    df["is_valid_date"] = True
    for idx, row in df.iterrows():
        is_valid, err = validate_event_date(str(row["event_date"]))
        if not is_valid:
            df.at[idx, "is_valid_date"] = False
            existing = str(row.get("errors", "")) if pd.notna(row.get("errors")) else ""
            if existing:
                df.at[idx, "errors"] = f"{existing}; {err}"
            else:
                df.at[idx, "errors"] = err
    return df


def flag_missing_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Log errors for rows that have missing financials but no error recorded."""
    fin_cols = ["market_cap_m", "current_price", "cash_position_m"]
    for idx, row in df.iterrows():
        missing = [c for c in fin_cols if pd.isna(row.get(c))]
        if not missing:
            continue

        existing_err = str(row.get("errors", "")) if pd.notna(row.get("errors")) else ""
        # Don't re-log if already noted
        if "financial" in existing_err.lower() or "partial data" in existing_err.lower():
            continue

        err_msg = f"Missing financials: {', '.join(missing)}"
        if existing_err:
            df.at[idx, "errors"] = f"{existing_err}; {err_msg}"
        else:
            df.at[idx, "errors"] = err_msg
    return df

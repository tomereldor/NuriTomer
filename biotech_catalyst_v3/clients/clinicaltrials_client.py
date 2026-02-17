"""ClinicalTrials.gov API v2 client with prioritized search and trace logging."""

import re
import time
import requests
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TrialResult:
    nct_id: str
    title: str
    phase: str
    status: str
    conditions: str
    interventions: str
    sponsor: str
    enrollment: int
    brief_summary: str
    study_design: str
    # CT.gov detail fields (populated on fetch)
    official_title: str = ""
    allocation: str = ""
    primary_completion_date: str = ""


class ClinicalTrialsClient:
    """
    Client for ClinicalTrials.gov API v2.

    Supports two modes:
    1. search_nct_prioritized() - Find the best NCT ID with full trace log
    2. fetch_trial_details()    - Get full trial details by NCT ID

    API docs: https://clinicaltrials.gov/data-api/api
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    # Known drug name -> code aliases for tricky lookups
    DRUG_ALIASES = {
        "atebimetinib": ["IMM-1-104"],
        "solnerstotug": ["SNS-101"],
        "cibotercept": ["KER-012"],
        "giroctocogene fitelparvovec": ["SB-525", "PF-07055480"],
        "isaralgagene civaparvovec": ["ST-920"],
        "ozureprubart": ["RPT-193"],
        "relutrigine": ["PRAX-628"],
        "rosnilimab": ["ANB032"],
        "zelnecirnon": ["CALY-002"],
        "zeleciment rostadirsen": ["DYNE-251"],
        # Added for remaining missing drugs
        "vobra duo": ["MGD024", "MGD-024", "vobramitamab duocarmazine"],
        "sep-786": ["SEP786"],
        "elevidys": ["SRP-9001", "delandistrogene moxeparvovec"],
        "vafseo": ["vadadustat", "AKB-6548"],
        "luvelta": ["luvelta", "bremelanotide"],  # Sutro's ADC
        "alto-100": ["ALTO-100", "NV-5138"],
    }

    # Ticker -> known lead drug for cases where AI didn't extract drug name
    TICKER_DRUG_FALLBACKS = {
        "IVVD": ["VYD222", "adintrevimab"],
        "VERA": ["atacicept", "VERA-101", "MAU868"],
        "STOK": ["STK-001", "zorevunersen"],
        "KYTX": ["KYV-101", "brexucabtagene autoleucel"],
        "MGNX": ["MGD024", "vobramitamab duocarmazine"],
        "SEPN": ["SEP-786", "SEP786"],
    }

    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Public: prioritized NCT search with detailed trace log
    # ------------------------------------------------------------------
    def search_nct_prioritized(
        self,
        drug_name: str,
        indication: str = None,
        phase: str = None,
        study_design_keywords: List[str] = None,
        sponsor: str = None,
        ticker: str = None,
    ) -> Tuple[Optional[str], Dict]:
        """
        Search for the best-matching NCT ID using staged filtering.

        Returns (nct_id or None, detailed_log dict).

        The log dict contains full trace info:
        - input: all search parameters
        - search_queries: every API query attempted
        - stages: what happened at each filtering stage
        - final_candidates: top candidates with details
        - result: outcome category
        - rejection_reason: why search failed (if it did)
        """
        log: Dict = {
            "input": {
                "drug_name": drug_name,
                "indication": indication,
                "phase": phase,
                "keywords": study_design_keywords,
                "sponsor": sponsor,
                "ticker": ticker,
            },
            "search_queries": [],
            "stages": [],
            "candidates": 0,
            "final_candidates": [],
            "result": "not_found",
            "rejection_reason": None,
        }

        # If no drug name, try ticker fallback
        if not drug_name and ticker and ticker in self.TICKER_DRUG_FALLBACKS:
            drug_name = self.TICKER_DRUG_FALLBACKS[ticker][0]
            log["input"]["drug_name_from_fallback"] = drug_name

        if not drug_name:
            log["result"] = "no_drug_name"
            log["rejection_reason"] = "No drug name provided and no ticker fallback available"
            return None, log

        # Stage 1: Search by drug name
        log["search_queries"].append(f"query.intr={drug_name}")
        candidates = self._search_by_intervention(drug_name)
        log["stages"].append({
            "stage": "drug_search",
            "query": drug_name,
            "hits": len(candidates),
            "nct_ids": [c.nct_id for c in candidates[:5]],
        })
        log["candidates"] = len(candidates)

        if not candidates:
            # Try drug-name variations (codes, aliases)
            variations = self._get_drug_variations(drug_name)
            log["variations_tried"] = variations

            for var in variations:
                log["search_queries"].append(f"query.intr={var}")
                candidates = self._search_by_intervention(var)
                if candidates:
                    log["stages"].append({
                        "stage": "variation_search",
                        "query": var,
                        "hits": len(candidates),
                        "nct_ids": [c.nct_id for c in candidates[:5]],
                    })
                    break

        # If still nothing and we have a ticker fallback, try those
        if not candidates and ticker and ticker in self.TICKER_DRUG_FALLBACKS:
            for fallback_drug in self.TICKER_DRUG_FALLBACKS[ticker]:
                if fallback_drug.lower() == drug_name.lower():
                    continue
                log["search_queries"].append(f"query.intr={fallback_drug} (ticker_fallback)")
                candidates = self._search_by_intervention(fallback_drug)
                if candidates:
                    log["stages"].append({
                        "stage": "ticker_fallback_search",
                        "query": fallback_drug,
                        "hits": len(candidates),
                        "nct_ids": [c.nct_id for c in candidates[:5]],
                    })
                    break

        if not candidates:
            log["result"] = "no_candidates"
            log["rejection_reason"] = (
                f"No trials found for '{drug_name}'"
                + (f" or variations {self._get_drug_variations(drug_name)}" if self._get_drug_variations(drug_name) else "")
            )
            return None, log

        if len(candidates) == 1:
            log["result"] = "single_match"
            log["final_candidates"] = [self._candidate_summary(candidates[0])]
            return candidates[0].nct_id, log

        # Stage 2: Filter by indication
        if indication and len(candidates) > 1:
            before_count = len(candidates)
            filtered = self._filter_by_indication(candidates, indication)
            rejected_ids = [c.nct_id for c in candidates if c not in (filtered or [])]
            if filtered:
                candidates = filtered
            log["stages"].append({
                "stage": "indication_filter",
                "indication": indication,
                "before": before_count,
                "after": len(filtered) if filtered else before_count,
                "applied": bool(filtered),
                "rejected": rejected_ids[:5],
            })

        # Stage 3: Filter by phase
        if phase and len(candidates) > 1:
            before_count = len(candidates)
            filtered = self._filter_by_phase(candidates, phase)
            rejected_ids = [c.nct_id for c in candidates if c not in (filtered or [])]
            if filtered:
                candidates = filtered
            log["stages"].append({
                "stage": "phase_filter",
                "phase": phase,
                "before": before_count,
                "after": len(filtered) if filtered else before_count,
                "applied": bool(filtered),
                "rejected": rejected_ids[:5],
            })

        # Stage 4: Filter by study-design keywords
        if study_design_keywords and len(candidates) > 1:
            before_count = len(candidates)
            filtered = self._filter_by_design_keywords(candidates, study_design_keywords)
            if filtered:
                candidates = filtered
            log["stages"].append({
                "stage": "design_filter",
                "keywords": study_design_keywords,
                "before": before_count,
                "after": len(filtered) if filtered else before_count,
                "applied": bool(filtered),
            })

        # Stage 5: Filter by sponsor
        if sponsor and len(candidates) > 1:
            before_count = len(candidates)
            filtered = self._filter_by_sponsor(candidates, sponsor)
            if filtered:
                candidates = filtered
            log["stages"].append({
                "stage": "sponsor_filter",
                "sponsor": sponsor,
                "before": before_count,
                "after": len(filtered) if filtered else before_count,
                "applied": bool(filtered),
            })

        # Score remaining candidates and return top hit
        candidates = self._score_and_sort(candidates)
        log["final_candidates"] = [self._candidate_summary(c) for c in candidates[:5]]
        log["result"] = "best_match"

        return candidates[0].nct_id if candidates else None, log

    # ------------------------------------------------------------------
    # Public: fetch full trial details by NCT ID
    # ------------------------------------------------------------------
    def fetch_trial_details(self, nct_id: str) -> Optional[TrialResult]:
        """Fetch a single trial by NCT ID. Returns None on 404."""
        url = f"{self.BASE_URL}/{nct_id}"
        try:
            resp = self.session.get(url, params={"format": "json"}, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return self._parse_study(resp.json())
        except requests.RequestException as e:
            print(f"  Error fetching {nct_id}: {e}")
            return None
        finally:
            time.sleep(self.rate_limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _candidate_summary(self, c: TrialResult) -> Dict:
        """Compact summary of a trial candidate for logging."""
        return {
            "nct_id": c.nct_id,
            "title": c.title[:100],
            "phase": c.phase,
            "status": c.status,
            "sponsor": c.sponsor,
            "enrollment": c.enrollment,
            "conditions": c.conditions[:80],
        }

    def _search_by_intervention(self, drug_name: str, limit: int = 20) -> List[TrialResult]:
        """Search ClinicalTrials.gov by intervention/drug name."""
        params = {
            "query.intr": drug_name,
            "pageSize": limit,
            "format": "json",
        }
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = [self._parse_study(s) for s in data.get("studies", [])]
            time.sleep(self.rate_limit)
            return results
        except Exception as e:
            print(f"  Search error for '{drug_name}': {e}")
            return []

    def _parse_study(self, study: dict) -> TrialResult:
        """Parse a single study JSON into TrialResult."""
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        desc = proto.get("descriptionModule", {})
        cond = proto.get("conditionsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        interventions = [i.get("name", "") for i in arms.get("interventions", [])]
        phases = design.get("phases", [])
        design_info = design.get("designInfo", {})

        return TrialResult(
            nct_id=ident.get("nctId", ""),
            title=ident.get("officialTitle", "") or ident.get("briefTitle", ""),
            phase="/".join(phases) if phases else "",
            status=status_mod.get("overallStatus", ""),
            conditions="; ".join(cond.get("conditions", [])),
            interventions="; ".join(interventions),
            sponsor=sponsor_mod.get("leadSponsor", {}).get("name", ""),
            enrollment=design.get("enrollmentInfo", {}).get("count", 0) or 0,
            brief_summary=desc.get("briefSummary", ""),
            study_design=str(design_info),
            official_title=ident.get("officialTitle", ""),
            allocation=design_info.get("allocation", ""),
            primary_completion_date=(
                status_mod.get("primaryCompletionDateStruct", {}).get("date", "")
            ),
        )

    def _get_drug_variations(self, drug_name: str) -> List[str]:
        """Generate alternative search terms for a drug name."""
        variations: List[str] = []

        drug_lower = drug_name.lower()
        for known, alts in self.DRUG_ALIASES.items():
            if known in drug_lower:
                variations.extend(alts)

        # Extract code from parentheses: "drug (CODE-123)"
        paren_match = re.search(r'\(([A-Z]{2,5}[-]?\d{2,5})\)', drug_name)
        if paren_match:
            variations.append(paren_match.group(1))

        # Extract standalone drug-code patterns
        codes = re.findall(r'\b([A-Z]{2,5}[-]\d{2,5})\b', drug_name)
        variations.extend(codes)

        # If drug name has multiple words, try just the first (generic name)
        parts = drug_name.split()
        if len(parts) > 1:
            variations.append(parts[0])

        return list(dict.fromkeys(variations))  # dedupe, preserve order

    def _filter_by_indication(
        self, candidates: List[TrialResult], indication: str
    ) -> List[TrialResult]:
        terms = [t.strip().lower() for t in re.split(r'[,\s]+', indication.lower()) if len(t.strip()) > 3]
        return [c for c in candidates if any(t in c.conditions.lower() for t in terms)]

    def _filter_by_phase(
        self, candidates: List[TrialResult], phase: str
    ) -> List[TrialResult]:
        phase_norm = phase.lower().replace("phase", "").replace(" ", "").strip()
        filtered = []
        for c in candidates:
            c_phase = c.phase.lower().replace("phase", "").replace(" ", "")
            if phase_norm in c_phase or c_phase in phase_norm:
                filtered.append(c)
        return filtered

    def _filter_by_design_keywords(
        self, candidates: List[TrialResult], keywords: List[str]
    ) -> List[TrialResult]:
        filtered = []
        for c in candidates:
            searchable = f"{c.brief_summary} {c.study_design} {c.title}".lower()
            if any(kw.lower() in searchable for kw in keywords):
                filtered.append(c)
        return filtered

    def _filter_by_sponsor(
        self, candidates: List[TrialResult], sponsor: str
    ) -> List[TrialResult]:
        sponsor_lower = sponsor.lower()
        sponsor_clean = re.sub(
            r'\s*(inc\.?|corp\.?|therapeutics|pharma|biosciences?|pharmaceuticals?)\s*',
            '', sponsor_lower, flags=re.I
        ).strip()
        return [
            c for c in candidates
            if sponsor_clean in c.sponsor.lower() or sponsor_lower in c.sponsor.lower()
        ]

    def _score_and_sort(self, candidates: List[TrialResult]) -> List[TrialResult]:
        """Rank candidates by relevance heuristics."""
        def score(c: TrialResult) -> float:
            s = 0.0
            status_l = c.status.lower()
            if "recruiting" in status_l:
                s += 10
            elif "active" in status_l:
                s += 8
            elif "completed" in status_l:
                s += 5
            s += min(c.enrollment / 50, 5)
            if "3" in c.phase:
                s += 4
            elif "2" in c.phase:
                s += 2
            return s

        return sorted(candidates, key=score, reverse=True)

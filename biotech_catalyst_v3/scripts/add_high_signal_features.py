"""
add_high_signal_features.py
============================
Pass-4 deterministic feature engineering — high-signal features for baseline ML.

Reads  : latest ml_dataset_features_*_vN.csv
Outputs: ml_dataset_features_*_v(N+1).csv
         ml_feature_dict_*_v(N+1).csv
Archives: superseded latest files to archive/

New features added (Steps 0a–0d are foundational; Steps 1–7 are higher-order composites):

  Step 0a — Clinical protocol core (from ct_phase, ct_enrollment, ct_allocation)
    feat_phase_num, feat_late_stage_flag, feat_enrollment_log,
    feat_randomized_flag, feat_design_quality_score,
    feat_withdrawn_flag*, feat_terminated_flag*  (*intermediate only, not in training)
  Step 0b — Regulatory designation flags (keyword match on press release + CT.gov text)
    feat_orphan_flag, feat_fast_track_flag, feat_breakthrough_flag,
    feat_nda_bla_flag, feat_priority_review_flag*, feat_regulatory_stage_score
  Step 0b.5 — PIT status flags for terminated/withdrawn (AACT point-in-time; same approach as Option C)
    feat_terminated_at_event_flag, feat_withdrawn_at_event_flag
    Falls back to snapshot flags (feat_terminated_flag / feat_withdrawn_flag) when ct_status_at_event is null
  Step 0c — Trial quality score (requires 0a + 0b + 0b.5)
    feat_trial_quality_score, feat_controlled_flag
  Step 0d — Company foundation features
    feat_n_trials_for_company, feat_n_unique_drugs_for_company,
    feat_single_asset_company_flag, feat_lead_asset_dependency_score,
    feat_n_late_stage_trials_for_company, feat_pipeline_concentration_simple*
  Step 1 — Company asset depth
    feat_asset_trial_count_for_company, feat_asset_trial_share,
    feat_pipeline_depth_score
  Step 2 — Pivotal proxy
    feat_pivotal_proxy_score
  Step 3 — Outcome / evidence flags
    feat_primary_endpoint_known_flag, feat_superiority_flag,
    feat_stat_sig_flag, feat_clinically_meaningful_flag, feat_mixed_results_flag
  Step 4 — Trial design flags
    feat_blinded_flag, feat_open_label_flag, feat_small_trial_flag
  Step 5 — Trial status / timing
    feat_completed_before_event, feat_recent_completion_flag (SNAPSHOT_UNSAFE — excluded from training)
    feat_completed_at_event_flag, feat_active_not_recruiting_at_event_flag (AACT point-in-time)
  Step 6 — Disease structure
    feat_therapeutic_superclass, feat_mesh_level1_encoded (ordinal int),
    feat_oncology_flag, feat_cns_flag, feat_rare_disease_flag
  Step 7 — Financial context
    feat_volatility, feat_cash_runway_proxy

Usage (from biotech_catalyst_v3/):
    python -m scripts.add_high_signal_features
"""

import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
MASTER_CSV  = os.path.join(BASE_DIR, "enriched_all_clinical_clean_v3.csv")

# ---------------------------------------------------------------------------
# Constants / keyword lists
# ---------------------------------------------------------------------------

MESH_TO_SUPERCLASS = {
    "Neoplasms":                  "Oncology",
    "Nervous System Diseases":    "CNS",
    "Immune System Diseases":     "Immunology",
    "Endocrine System Diseases":  "Endocrine/Metabolic",
    "Respiratory Tract Diseases": "Respiratory",
    "Infectious Diseases":        "Infectious Disease",
    "Cardiovascular Diseases":    "Cardiovascular",
    "Digestive System Diseases":  "GI/Hepatology",
    "Skin Diseases":              "Dermatology",
    "Musculoskeletal Diseases":   "Musculoskeletal",
    "Other / Non-Disease":        "Other",
}

# Ordinal int encoding — same values as completeness_pass.py MESH_ENCODE_MAP
MESH_ENCODE_MAP = {
    "Neoplasms":                  1,
    "Nervous System Diseases":    2,
    "Immune System Diseases":     3,
    "Endocrine System Diseases":  4,
    "Respiratory Tract Diseases": 5,
    "Infectious Diseases":        6,
    "Cardiovascular Diseases":    7,
    "Digestive System Diseases":  8,
    "Skin Diseases":              9,
    "Musculoskeletal Diseases":   10,
    "Other / Non-Disease":        11,
}

PIVOTAL_KEYWORDS = [
    "pivotal", "registrational", "registration study", "registration trial",
    "phase 3", "phase iii", "confirmatory", "approval-enabling",
    "pdufa", " nda ", " bla ", "new drug application", "biologics license",
]

SUPERIORITY_KEYWORDS = [
    "met primary endpoint", "met its primary endpoint", "achieved primary endpoint",
    "positive topline", "positive phase", "positive results",
    "demonstrated superiority", "statistically significant improvement",
    "significantly reduced", "significantly improved", "significantly better",
    "non-inferior", "noninferior", "met non-inferiority",
    "superior efficacy", "superior to",
]

STAT_SIG_KEYWORDS = [
    "p<0.05", "p < 0.05", "p=0.0", "p = 0.0", "p<0.01", "p < 0.01",
    "p<0.001", "p < 0.001", "statistically significant", "(p=", "(p <",
    "hazard ratio", "hr=", "hr =", "odds ratio", "confidence interval",
    "95% ci", "95% confidence", "log-rank", "nominal p",
]

CLINICALLY_MEANINGFUL_KEYWORDS = [
    "clinically meaningful", "clinically significant", "clinically relevant",
    "durable response", "durable remission", "complete response", "complete remission",
    "overall survival benefit", "survival benefit", "disease-free survival",
    "event-free survival", "meaningful improvement", "meaningful reduction",
    "transformative", "unprecedented efficacy", "profound",
]

MIXED_KEYWORDS = [
    "mixed results", "did not meet", "failed to meet", "missed primary",
    "did not achieve", "was not met", "not statistically significant",
    "not significant", "nominal significance", "trend toward",
    "subgroup only", "exploratory only", "numerically improved but",
    "failed to demonstrate", "p>0.05", "p > 0.05", "no significant difference",
]

BLINDED_KEYWORDS = [
    "double-blind", "double blind", "single-blind", "single blind",
    "blinded", "masking", "triple-blind", "triple blind",
]

OPEN_LABEL_KEYWORDS = [
    "open-label", "open label", "unblinded", "non-blinded",
]

RARE_DISEASE_KEYWORDS = [
    "rare disease", "orphan disease", "ultra-rare", "ultra rare",
    "rare pediatric", "unmet medical need",
]

# Regulatory designation keywords (Steps 0b) — press release + CT.gov title
REGULATORY_TEXT_FIELDS = [
    "pivotal_evidence", "v_pr_key_info", "v_summary",
    "catalyst_summary", "v_pr_title", "ct_official_title",
]

ORPHAN_KEYWORDS = [
    "orphan drug", "orphan designation", "orphan disease",
    "rare disease designation", "orphan status",
]
FAST_TRACK_KEYWORDS = [
    "fast track", "fast-track", "fast track designation",
]
BREAKTHROUGH_KEYWORDS = [
    "breakthrough therapy", "breakthrough designation",
    "breakthrough therapy designation", "breakthrough",
]
NDA_BLA_KEYWORDS = [
    "nda", "bla", "snda", "sbla",
    "new drug application", "biologics license application",
    "marketing authorization", "maa",
    "submitted to fda", "regulatory submission",
    "seeking approval", "approval application",
    "filed with the fda", "filing with the fda",
]
PRIORITY_REVIEW_KEYWORDS = [
    "priority review", "priority review voucher",
    "pdufa", "pdufa date", "pdufa action date",
    "approval expected", "approval decision expected",
]
PIVOTAL_REG_KEYWORDS = [
    "pivotal", "registrational", "registration study", "registration trial",
]

# Phase → numeric mapping (prospective protocol metadata, pre-event safe)
PHASE_MAP = {
    "Phase 1 (Early)": 0.5,
    "Phase 1":         1.0,
    "Phase 1/2":       1.5,
    "Phase 2":         2.0,
    "Phase 2/3":       2.5,
    "Phase 3":         3.0,
    "Phase 4":         4.0,
}

# Text columns used for outcome evidence extraction
EVIDENCE_FIELDS = [
    "primary_endpoint_result", "v_pr_key_info", "v_summary",
    "v_pr_title", "catalyst_summary", "pivotal_evidence",
]

# Text columns used for trial design extraction
DESIGN_FIELDS = [
    "ct_official_title", "v_pr_title", "v_summary", "catalyst_summary",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_version_file(base_dir, prefix):
    """Return (path, version_int, date_tag) for highest-version matching file."""
    files = glob.glob(os.path.join(base_dir, f"{prefix}_*.csv"))
    best, best_v, best_date = None, 0, None
    for f in files:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v = v
                best = f
                best_date = m.group(1)
    return best, best_v, best_date


def _keyword_hit(texts, keywords):
    """1 if any keyword found in the combined lowercase text, else 0."""
    combined = " ".join(
        str(t).lower() for t in texts
        if t and str(t).strip().lower() not in ("nan", "none", "")
    )
    return int(any(kw in combined for kw in keywords))


def _row_texts(row, fields):
    return [row.get(f, "") for f in fields if f in row.index]


def _combined_text_series(df, cols):
    """Vectorized: return lowercase combined text Series for all present columns."""
    present = [c for c in cols if c in df.columns]
    return (
        df[present].fillna("").astype(str)
        .apply(lambda r: " ".join(r), axis=1)
        .str.lower()
    )


def _keyword_flag_series(text_series, keywords):
    """Vectorized: 1 if any keyword found in text Series, else 0."""
    pattern = "|".join(re.escape(k) for k in keywords)
    return text_series.str.contains(pattern, regex=True, na=False).astype(int)


# ---------------------------------------------------------------------------
# Step 0a — Clinical protocol core features
# ---------------------------------------------------------------------------

def build_clinical_core(df):
    """
    Clinical protocol features from ct_phase, ct_enrollment, ct_allocation.

    All source columns are prospective protocol metadata registered at trial
    inception and immutable once set — pre-event safe.

    Exception: feat_withdrawn_flag / feat_terminated_flag use ct_status (current
    CT.gov snapshot). Mild SNAPSHOT_UNSAFE risk — used only as intermediates
    for feat_trial_quality_score, NOT directly in the training feature roster.
    """
    enroll    = pd.to_numeric(df.get("ct_enrollment"), errors="coerce")
    phase_num = df.get("ct_phase", pd.Series(dtype=str)).map(PHASE_MAP)
    ct_status = df.get("ct_status", pd.Series(dtype=str))
    ct_alloc  = df.get("ct_allocation", pd.Series(dtype=str))

    df["feat_phase_num"]       = phase_num
    df["feat_late_stage_flag"] = (phase_num >= 2.5).astype(float)
    df.loc[phase_num.isna(), "feat_late_stage_flag"] = float("nan")

    df["feat_enrollment_log"]  = np.log1p(enroll)

    df["feat_randomized_flag"] = np.where(
        ct_alloc.isna(), np.nan,
        (ct_alloc == "RANDOMIZED").astype(float),
    )

    # feat_design_quality_score (0–6):
    # +2 RANDOMIZED | +2 enroll>300 | +1 enroll 101–300
    # +1 Phase 3    | +0.5 Phase 2/3 | -1 NON_RANDOMIZED
    design = pd.Series(0.0, index=df.index)
    design += (ct_alloc == "RANDOMIZED").astype(float)     * 2.0
    design -= (ct_alloc == "NON_RANDOMIZED").astype(float) * 1.0
    design += (enroll > 300).fillna(False).astype(float)   * 2.0
    design += ((enroll > 100) & (enroll <= 300)).fillna(False).astype(float) * 1.0
    design += (phase_num >= 3.0).fillna(False).astype(float) * 1.0
    design += (phase_num == 2.5).fillna(False).astype(float) * 0.5
    all_missing = ct_alloc.isna() & enroll.isna() & phase_num.isna()
    design[all_missing] = np.nan
    df["feat_design_quality_score"] = design

    # Intermediate snapshot flags — NOT in training roster directly
    df["feat_withdrawn_flag"]  = (ct_status == "WITHDRAWN").astype(int)
    df["feat_terminated_flag"] = (ct_status == "TERMINATED").astype(int)

    n = len(df)
    ph_nn = int(phase_num.notna().sum())
    print(f"  feat_phase_num: {ph_nn}/{n} non-null")
    print(f"  feat_late_stage_flag (>=Ph2/3): {int((df['feat_late_stage_flag']==1).sum())} rows")
    print(f"  feat_design_quality_score: mean={design.mean():.2f}, null={int(design.isna().sum())}")
    return df


# ---------------------------------------------------------------------------
# Step 0b — Regulatory designation flags
# ---------------------------------------------------------------------------

def build_regulatory_flags(df):
    """
    Keyword match on press release + CT.gov title text — pre-event safe.

    Regulatory designations (orphan, breakthrough, fast track) are publicly
    announced before the data event and captured in contemporaneous press releases.
    """
    text = _combined_text_series(df, REGULATORY_TEXT_FIELDS)

    df["feat_orphan_flag"]          = _keyword_flag_series(text, ORPHAN_KEYWORDS)
    df["feat_fast_track_flag"]      = _keyword_flag_series(text, FAST_TRACK_KEYWORDS)
    df["feat_breakthrough_flag"]    = _keyword_flag_series(text, BREAKTHROUGH_KEYWORDS)
    df["feat_nda_bla_flag"]         = _keyword_flag_series(text, NDA_BLA_KEYWORDS)
    df["feat_priority_review_flag"] = _keyword_flag_series(text, PRIORITY_REVIEW_KEYWORDS)

    # feat_regulatory_stage_score: ordinal regulatory ladder (0–3)
    # 0 = none | 1 = pivotal/registrational | 2 = NDA/BLA | 3 = priority review/PDUFA
    pivotal_hit = _keyword_flag_series(text, PIVOTAL_REG_KEYWORDS)
    score = pd.Series(0, index=df.index, dtype=int)
    score = np.where(pivotal_hit                     == 1, np.maximum(score, 1), score)
    score = np.where(df["feat_nda_bla_flag"]         == 1, np.maximum(score, 2), score)
    score = np.where(df["feat_priority_review_flag"] == 1, np.maximum(score, 3), score)
    df["feat_regulatory_stage_score"] = score.astype(int)

    n = len(df)
    print(f"  orphan={df['feat_orphan_flag'].sum()}, "
          f"breakthrough={df['feat_breakthrough_flag'].sum()}, "
          f"fast_track={df['feat_fast_track_flag'].sum()}, "
          f"nda_bla={df['feat_nda_bla_flag'].sum()}")
    score_arr = df["feat_regulatory_stage_score"]
    print(f"  reg_stage_score: 0={int((score_arr==0).sum())}, 1={int((score_arr==1).sum())}, "
          f"2={int((score_arr==2).sum())}, 3={int((score_arr==3).sum())}")
    return df


# ---------------------------------------------------------------------------
# Step 0b.5 — PIT terminated / withdrawn flags (AACT point-in-time)
# ---------------------------------------------------------------------------

def build_status_pit_flags(df):
    """
    feat_terminated_at_event_flag and feat_withdrawn_at_event_flag —
    point-in-time versions using ct_status_at_event (AACT monthly snapshot,
    same source as feat_completed_at_event_flag from Option C).

    For rows without AACT data (ct_status_at_event null), falls back to the
    current CT.gov snapshot flags (feat_terminated_flag / feat_withdrawn_flag).
    This is safe because for 92.4% of the training cohort (2023+) AACT data
    is available. The 7.6% fallback rows are all trials with no AACT record,
    so leakage risk is limited to those rows.

    Audit: 23/33 terminated rows in training were still active at event time —
    they were terminated AFTER the event. The snapshot flags inject a -2
    quality-score penalty for these 23 rows that should not be there.
    These PIT flags fix that in feat_trial_quality_score (Step 0c).
    """
    ct_pit = df.get("ct_status_at_event", pd.Series(dtype=str))

    # PIT flag — NaN where no AACT record
    term_pit = (ct_pit == "TERMINATED").astype(float)
    term_pit[ct_pit.isna()] = float("nan")

    with_pit = (ct_pit == "WITHDRAWN").astype(float)
    with_pit[ct_pit.isna()] = float("nan")

    # Fall back to snapshot for rows with no AACT record
    snap_term = df.get("feat_terminated_flag", pd.Series(0, index=df.index)).fillna(0)
    snap_with = df.get("feat_withdrawn_flag",  pd.Series(0, index=df.index)).fillna(0)

    df["feat_terminated_at_event_flag"] = term_pit.fillna(snap_term)
    df["feat_withdrawn_at_event_flag"]  = with_pit.fillna(snap_with)

    pit_avail = ct_pit.notna().sum()
    print(f"  terminated_at_event={int(df['feat_terminated_at_event_flag'].sum())} "
          f"(snapshot had {int(snap_term.sum())}), "
          f"withdrawn_at_event={int(df['feat_withdrawn_at_event_flag'].sum())} "
          f"— PIT available for {pit_avail}/{len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Step 0c — Trial quality score (requires 0a + 0b + 0b.5)
# ---------------------------------------------------------------------------

def build_trial_quality_score(df):
    """
    feat_trial_quality_score and feat_controlled_flag.

    Must run after build_clinical_core (0a), build_regulatory_flags (0b),
    and build_status_pit_flags (0b.5).

    Uses feat_terminated_at_event_flag / feat_withdrawn_at_event_flag (PIT)
    for the -2 penalties — fixes the SNAPSHOT_UNSAFE contamination where 23/33
    terminated rows in training were still active at event time.
    """
    ct_alloc  = df.get("ct_allocation", pd.Series(dtype=str))
    enroll    = pd.to_numeric(df.get("ct_enrollment"), errors="coerce")
    phase_num = df.get("feat_phase_num", pd.Series(dtype=float))
    ct_status = df.get("ct_status", pd.Series(dtype=str))

    title_lower   = df.get("ct_official_title", pd.Series(dtype=str)).fillna("").str.lower()
    blinded_title = title_lower.str.contains(
        r"placebo[- ]control|double[- ]blind|triple[- ]blind|controlled study|controlled trial",
        regex=True, na=False,
    )

    randomized = df.get("feat_randomized_flag", pd.Series(0.0, index=df.index)).fillna(0)
    df["feat_controlled_flag"] = ((randomized == 1) | blinded_title).astype(int)

    design = df.get("feat_design_quality_score",
                    pd.Series(0.0, index=df.index)).fillna(0.0)
    breakthrough = df.get("feat_breakthrough_flag",
                          pd.Series(0, index=df.index)).fillna(0).astype(float)
    quality = (
        design
        + blinded_title.astype(float)
        + breakthrough * 0.5
        - df.get("feat_withdrawn_at_event_flag",  pd.Series(0, index=df.index)).fillna(0).astype(float) * 2.0
        - df.get("feat_terminated_at_event_flag", pd.Series(0, index=df.index)).fillna(0).astype(float) * 2.0
    )
    all_missing = ct_alloc.isna() & enroll.isna() & phase_num.isna() & ct_status.isna()
    quality[all_missing] = np.nan
    df["feat_trial_quality_score"] = quality

    print(f"  feat_trial_quality_score: mean={quality.mean():.2f}, null={int(quality.isna().sum())}")
    print(f"  feat_controlled_flag: {int(df['feat_controlled_flag'].sum())} rows")
    return df


# ---------------------------------------------------------------------------
# Step 0d — Company foundation features
# ---------------------------------------------------------------------------

def build_company_foundation(df):
    """
    Company-level pipeline structure features.

    Dataset aggregations representing the company's overall pipeline profile —
    knowable pre-event (how many drugs, how late-stage is the pipeline).
    Requires feat_late_stage_flag from build_clinical_core.
    """
    # feat_n_trials_for_company: total events in dataset for this ticker
    df["feat_n_trials_for_company"] = (
        df.groupby("ticker")["ticker"].transform("count").astype(int)
    )

    # feat_n_unique_drugs_for_company: distinct drug_names per ticker
    drug_nunique = df.groupby("ticker")["drug_name"].transform(
        lambda s: s.dropna().nunique()
    )
    df["feat_n_unique_drugs_for_company"] = drug_nunique.astype(int)

    # feat_single_asset_company_flag: ≤1 distinct drug
    df["feat_single_asset_company_flag"] = (
        df["feat_n_unique_drugs_for_company"] <= 1
    ).astype(int)

    # feat_lead_asset_dependency_score: deterministic pipeline concentration proxy
    def _dep_score(n):
        if n <= 1: return 1.0
        if n == 2: return 0.7
        if n <= 4: return 0.4
        return 0.2

    df["feat_lead_asset_dependency_score"] = (
        df["feat_n_unique_drugs_for_company"].apply(_dep_score)
    )

    # feat_n_late_stage_trials_for_company: requires feat_late_stage_flag
    late_flag = df.get("feat_late_stage_flag", pd.Series(0.0, index=df.index)).fillna(0)
    phase_num = df.get("feat_phase_num", pd.Series(dtype=float))
    late_mask = (late_flag == 1) | (phase_num >= 3.0).fillna(False)
    df["_late_row"] = late_mask.astype(int)
    df["feat_n_late_stage_trials_for_company"] = (
        df.groupby("ticker")["_late_row"].transform("sum").astype(int)
    )
    df.drop(columns=["_late_row"], inplace=True)

    # feat_pipeline_concentration_simple: existential risk proxy (intermediate)
    late_frac = (
        df["feat_n_late_stage_trials_for_company"]
        / df["feat_n_trials_for_company"].replace(0, np.nan)
    ).fillna(0.0)
    df["feat_pipeline_concentration_simple"] = (
        df["feat_lead_asset_dependency_score"] * late_frac
    ).round(4)

    n = len(df)
    print(f"  n_trials_for_company: mean={df['feat_n_trials_for_company'].mean():.1f}")
    print(f"  n_unique_drugs: mean={df['feat_n_unique_drugs_for_company'].mean():.1f}, "
          f"single_asset_events={df['feat_single_asset_company_flag'].sum()}")
    print(f"  n_late_stage_for_company: mean={df['feat_n_late_stage_trials_for_company'].mean():.1f}")
    return df


# ---------------------------------------------------------------------------
# Step 1 — Company asset depth
# ---------------------------------------------------------------------------

def build_company_asset_features(df):
    drug_norm = df["drug_name"].str.strip().str.lower().fillna("unknown")
    df["_drug_norm"] = drug_norm

    # feat_asset_trial_count_for_company: events sharing the same ticker + drug
    df["feat_asset_trial_count_for_company"] = (
        df.groupby(["ticker", "_drug_norm"])["_drug_norm"]
        .transform("count")
        .astype(float)
    )

    # feat_asset_trial_share
    n_trials = df.get("feat_n_trials_for_company", pd.Series(np.nan, index=df.index))
    df["feat_asset_trial_share"] = (
        df["feat_asset_trial_count_for_company"] / n_trials.replace(0, np.nan)
    ).round(4)

    # feat_pipeline_depth_score: breadth + late-stage richness composite
    # = min(n_late_stage, 5) * 0.6 + log1p(n_unique_drugs) * 0.4
    late   = df.get("feat_n_late_stage_trials_for_company",
                    pd.Series(0.0, index=df.index)).fillna(0)
    n_drugs = df.get("feat_n_unique_drugs_for_company",
                     pd.Series(1.0, index=df.index)).fillna(1)
    df["feat_pipeline_depth_score"] = (
        np.minimum(late, 5) * 0.6 + np.log1p(n_drugs) * 0.4
    ).round(4)

    df.drop(columns=["_drug_norm"], inplace=True)
    print(f"  asset_trial_share coverage: {df['feat_asset_trial_share'].notna().sum()}/{len(df)}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Pivotal proxy score (0–5)
# ---------------------------------------------------------------------------

def build_pivotal_proxy(df):
    """
    feat_pivotal_proxy_score — combines structured signals + text.

      +1.5  Phase 3 or 4 (feat_phase_num >= 3.0)
      +1.0  Phase 2/3     (feat_phase_num == 2.5)
      +1.0  NDA/BLA filed (feat_regulatory_stage_score >= 2)
      +0.5  Pivotal language in reg score (feat_regulatory_stage_score == 1)
      +0.5  feat_breakthrough_flag
      +0.5  feat_priority_review_flag
      +0.5  pivotal keywords in evidence/summary text

    100% coverage (all inputs default to 0 for missing).
    """
    score = pd.Series(0.0, index=df.index)

    phase = df.get("feat_phase_num", pd.Series(dtype=float))
    score += (phase >= 3.0).fillna(False).astype(float) * 1.5
    score += (phase == 2.5).fillna(False).astype(float) * 1.0

    reg = df.get("feat_regulatory_stage_score",
                 pd.Series(0.0, index=df.index)).fillna(0)
    score += (reg >= 2).astype(float) * 1.0
    score += (reg == 1).astype(float) * 0.5

    score += df.get("feat_breakthrough_flag",
                    pd.Series(0, index=df.index)).fillna(0).astype(float) * 0.5
    score += df.get("feat_priority_review_flag",
                    pd.Series(0, index=df.index)).fillna(0).astype(float) * 0.5

    text_cols = [f for f in ["pivotal_evidence", "v_summary", "v_pr_key_info",
                              "catalyst_summary"] if f in df.columns]
    pivotal_hit = df[text_cols].apply(
        lambda row: _keyword_hit(list(row), PIVOTAL_KEYWORDS), axis=1
    )
    score += pivotal_hit.astype(float) * 0.5

    df["feat_pivotal_proxy_score"] = score.round(3)
    print(f"  pivotal_proxy_score: mean={score.mean():.2f}, "
          f">=2.0: {(score >= 2.0).sum()} rows")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Outcome / evidence flags
# ---------------------------------------------------------------------------

def build_outcome_flags(df):
    avail = [f for f in EVIDENCE_FIELDS if f in df.columns]

    # feat_primary_endpoint_known_flag
    ep = df.get("primary_endpoint_met", pd.Series(dtype=str))
    df["feat_primary_endpoint_known_flag"] = (
        ep.str.strip().str.lower().isin(["yes", "no"])
    ).astype(int)

    # Keyword flags across all evidence text fields
    flags = {
        "feat_superiority_flag":          SUPERIORITY_KEYWORDS,
        "feat_stat_sig_flag":             STAT_SIG_KEYWORDS,
        "feat_clinically_meaningful_flag": CLINICALLY_MEANINGFUL_KEYWORDS,
        "feat_mixed_results_flag":        MIXED_KEYWORDS,
    }
    for col, keywords in flags.items():
        df[col] = df[avail].apply(
            lambda row: _keyword_hit(list(row), keywords), axis=1
        )

    n = len(df)
    for col in ["feat_primary_endpoint_known_flag"] + list(flags.keys()):
        print(f"  {col}: {df[col].sum()} ({df[col].sum()/n*100:.0f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Trial design flags
# ---------------------------------------------------------------------------

def build_design_flags(df):
    avail = [f for f in DESIGN_FIELDS if f in df.columns]

    df["feat_blinded_flag"] = df[avail].apply(
        lambda row: _keyword_hit(list(row), BLINDED_KEYWORDS), axis=1
    )
    df["feat_open_label_flag"] = df[avail].apply(
        lambda row: _keyword_hit(list(row), OPEN_LABEL_KEYWORDS), axis=1
    )

    enroll = pd.to_numeric(df.get("ct_enrollment"), errors="coerce")
    df["feat_small_trial_flag"] = (enroll < 50).astype(float)
    df.loc[enroll.isna(), "feat_small_trial_flag"] = float("nan")

    n = len(df)
    print(f"  blinded={df['feat_blinded_flag'].sum()} ({df['feat_blinded_flag'].sum()/n*100:.0f}%), "
          f"open_label={df['feat_open_label_flag'].sum()} ({df['feat_open_label_flag'].sum()/n*100:.0f}%), "
          f"small_trial={df['feat_small_trial_flag'].sum()}")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Trial status / timing
# ---------------------------------------------------------------------------

def build_timing_flags(df):
    event_dates = pd.to_datetime(df["event_date"], errors="coerce")
    status      = df.get("ct_status", pd.Series(dtype=str))

    # feat_completed_before_event: pre-event valid proxy for trial completion.
    # Uses ct_primary_completion (prospective protocol milestone registered before
    # trial starts) rather than ct_status (current CT.gov snapshot, SNAPSHOT_UNSAFE).
    # True if the scheduled primary data-collection end date preceded the event date.
    comp_dates = pd.to_datetime(
        df.get("ct_primary_completion", pd.Series(dtype=str)), errors="coerce"
    )
    df["feat_completed_before_event"] = (comp_dates < event_dates).astype(float)
    df.loc[comp_dates.isna(), "feat_completed_before_event"] = float("nan")

    # feat_recent_completion_flag: SNAPSHOT_UNSAFE — uses ct_status (current
    # CT.gov snapshot) combined with realized event_date anchor. Retained for
    # pipeline compatibility but EXCLUDED from training feature roster.
    days_since = (event_dates - comp_dates).dt.days
    df["feat_recent_completion_flag"] = (
        (status == "COMPLETED") & (days_since >= 0) & (days_since <= 365)
    ).astype(float)
    df.loc[status.isna(), "feat_recent_completion_flag"] = float("nan")

    # feat_completed_at_event_flag: point-in-time AACT monthly snapshot status.
    # Uses ct_status_at_event (written by fetch_aact_status_history.py) which
    # reflects the trial status in the AACT archive for the month closest-before
    # the event date. Replaces feat_completed_before_event (date proxy) in training
    # when available. NaN rows imputed as 0 (absent/unknown → not completed).
    ct_status_pit = df.get("ct_status_at_event", pd.Series(dtype=str))
    df["feat_completed_at_event_flag"] = (ct_status_pit == "COMPLETED").astype(float)
    df.loc[ct_status_pit.isna(), "feat_completed_at_event_flag"] = float("nan")

    # feat_active_not_recruiting_at_event_flag: point-in-time version.
    # Replaces feat_active_not_recruiting_flag (SNAPSHOT_UNSAFE) in training.
    df["feat_active_not_recruiting_at_event_flag"] = (
        ct_status_pit == "ACTIVE_NOT_RECRUITING"
    ).astype(float)
    df.loc[ct_status_pit.isna(), "feat_active_not_recruiting_at_event_flag"] = float("nan")

    pit_complete = (ct_status_pit == "COMPLETED").sum()
    pit_anr = (ct_status_pit == "ACTIVE_NOT_RECRUITING").sum()
    pit_nonnull = ct_status_pit.notna().sum()
    print(f"  completed_before_event={int(df['feat_completed_before_event'].sum())}, "
          f"recent_completion={df['feat_recent_completion_flag'].sum()}")
    print(f"  pit_status non-null={pit_nonnull}: completed_at_event={pit_complete}, "
          f"active_not_recruiting_at_event={pit_anr}")
    return df


# ---------------------------------------------------------------------------
# Step 6 — Disease structure
# ---------------------------------------------------------------------------

def build_disease_features(df):
    # feat_therapeutic_superclass: human-readable string version of mesh_level1
    df["feat_therapeutic_superclass"] = (
        df["mesh_level1"].map(MESH_TO_SUPERCLASS).fillna("Other")
    )

    # feat_mesh_level1_encoded: ordinal int (1–11) matching MESH_ENCODE_MAP
    # Protocol metadata — safe, immutable.
    df["feat_mesh_level1_encoded"] = df["mesh_level1"].map(MESH_ENCODE_MAP)

    df["feat_oncology_flag"] = (df["mesh_level1"] == "Neoplasms").astype(int)
    df["feat_cns_flag"]      = (df["mesh_level1"] == "Nervous System Diseases").astype(int)

    # feat_rare_disease_flag: orphan designation OR rare keywords in indication/mesh
    orphan = df.get("feat_orphan_flag",
                    pd.Series(0, index=df.index)).fillna(0)
    ind_text  = df.get("indication",  pd.Series("", index=df.index)).fillna("").str.lower()
    mesh_text = df.get("mesh_level1", pd.Series("", index=df.index)).fillna("").str.lower()
    rare_text_hit = (ind_text + " " + mesh_text).apply(
        lambda t: int(any(kw in t for kw in RARE_DISEASE_KEYWORDS))
    )
    df["feat_rare_disease_flag"] = ((orphan == 1) | (rare_text_hit == 1)).astype(int)

    n = len(df)
    print(f"  oncology={df['feat_oncology_flag'].sum()} ({df['feat_oncology_flag'].sum()/n*100:.0f}%), "
          f"cns={df['feat_cns_flag'].sum()} ({df['feat_cns_flag'].sum()/n*100:.0f}%), "
          f"rare={df['feat_rare_disease_flag'].sum()} ({df['feat_rare_disease_flag'].sum()/n*100:.0f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Financial context
# ---------------------------------------------------------------------------

def build_financial_context(df):
    """
    feat_cash_runway_proxy = cash_position_m / market_cap_m (SNAPSHOT_UNSAFE — not in training)
    feat_volatility = atr_pct (pre-event 20-day ATR — strictly backward-looking, SAFE)
    """
    cash = pd.to_numeric(df.get("cash_position_m"), errors="coerce")
    mcap = pd.to_numeric(df.get("market_cap_m"),    errors="coerce").replace(0, np.nan)
    df["feat_cash_runway_proxy"] = (cash / mcap).round(4).clip(upper=5.0)

    # feat_volatility: pre-event ATR. atr_pct is computed using OHLC data strictly
    # before the event date (Wilder's RMA, 20-day lookback) — pre-event safe.
    atr = pd.to_numeric(df.get("atr_pct"), errors="coerce")
    df["feat_volatility"] = atr

    nn = df["feat_cash_runway_proxy"].notna().sum()
    above1 = (df["feat_cash_runway_proxy"] > 1.0).sum()
    vol_nn = df["feat_volatility"].notna().sum()
    print(f"  cash_runway_proxy: {nn}/{len(df)} non-null, {above1} rows with cash > market_cap")
    print(f"  feat_volatility (atr_pct): {vol_nn}/{len(df)} non-null")
    return df


# ---------------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------------

NEW_FEATURE_META = [
    # (feature_name, stage, feature_type, description, source_columns, source_type)
    # ── Step 0a: Clinical protocol core ──────────────────────────────────────
    ("feat_phase_num",                             "pass4_0a", "feat",
     "Numeric phase (0.5=Ph1E, 1=Ph1, 1.5=Ph1/2, 2=Ph2, 2.5=Ph2/3, 3=Ph3, 4=Ph4)",
     "ct_phase", "deterministic"),
    ("feat_late_stage_flag",                       "pass4_0a", "feat",
     "1 if feat_phase_num >= 2.5 (Phase 2/3, 3, or 4)",
     "feat_phase_num", "deterministic"),
    ("feat_enrollment_log",                        "pass4_0a", "feat",
     "log1p(ct_enrollment) — log-scaled planned enrollment",
     "ct_enrollment", "deterministic"),
    ("feat_randomized_flag",                       "pass4_0a", "feat",
     "1 if ct_allocation == RANDOMIZED; NaN if ct_allocation missing",
     "ct_allocation", "deterministic"),
    ("feat_design_quality_score",                  "pass4_0a", "feat",
     "+2 RANDOMIZED +2 enroll>300 +1 enroll 101-300 +1 Ph3 +0.5 Ph2/3 -1 NON_RANDOMIZED",
     "ct_allocation, ct_enrollment, ct_phase", "deterministic"),
    ("feat_withdrawn_flag",                        "pass4_0a", "feat",
     "1 if ct_status == WITHDRAWN (intermediate; mild SNAPSHOT_UNSAFE risk; not in training roster)",
     "ct_status", "deterministic"),
    ("feat_terminated_flag",                       "pass4_0a", "feat",
     "1 if ct_status == TERMINATED (intermediate; mild SNAPSHOT_UNSAFE risk; not in training roster)",
     "ct_status", "deterministic"),
    # ── Step 0b: Regulatory designation flags ────────────────────────────────
    ("feat_orphan_flag",                           "pass4_0b", "feat",
     "1 if orphan drug/designation keywords found in press release + CT.gov text",
     "pivotal_evidence, v_pr_key_info, v_summary, catalyst_summary, ct_official_title",
     "keyword"),
    ("feat_fast_track_flag",                       "pass4_0b", "feat",
     "1 if fast track designation keywords found",
     "pivotal_evidence, v_pr_key_info, v_summary, catalyst_summary, ct_official_title",
     "keyword"),
    ("feat_breakthrough_flag",                     "pass4_0b", "feat",
     "1 if breakthrough therapy designation keywords found",
     "pivotal_evidence, v_pr_key_info, v_summary, catalyst_summary, ct_official_title",
     "keyword"),
    ("feat_nda_bla_flag",                          "pass4_0b", "feat",
     "1 if NDA/BLA/regulatory submission keywords found",
     "pivotal_evidence, v_pr_key_info, v_summary, catalyst_summary, ct_official_title",
     "keyword"),
    ("feat_priority_review_flag",                  "pass4_0b", "feat",
     "1 if priority review/PDUFA keywords found (intermediate for reg_stage_score)",
     "pivotal_evidence, v_pr_key_info, v_summary, catalyst_summary, ct_official_title",
     "keyword"),
    ("feat_regulatory_stage_score",                "pass4_0b", "feat",
     "0=none 1=pivotal/registrational 2=NDA/BLA 3=priority review/PDUFA",
     "feat_nda_bla_flag, feat_priority_review_flag, keyword match", "deterministic"),
    # ── Step 0c: Trial quality score ─────────────────────────────────────────
    ("feat_controlled_flag",                       "pass4_0c", "feat",
     "1 if randomized OR blinded/controlled-study language in trial title",
     "feat_randomized_flag, ct_official_title", "deterministic"),
    ("feat_trial_quality_score",                   "pass4_0c", "feat",
     "design_score + 1×blinded_title + 0.5×breakthrough - 2×withdrawn - 2×terminated",
     "feat_design_quality_score, feat_breakthrough_flag, feat_withdrawn_flag, "
     "feat_terminated_flag, ct_official_title", "deterministic"),
    # ── Step 0d: Company foundation ──────────────────────────────────────────
    ("feat_n_trials_for_company",                  "pass4_0d", "feat",
     "Total dataset events for this ticker (company pipeline activity proxy)",
     "ticker", "deterministic"),
    ("feat_n_unique_drugs_for_company",             "pass4_0d", "feat",
     "Distinct drug_names per ticker (pipeline breadth)",
     "drug_name, ticker", "deterministic"),
    ("feat_single_asset_company_flag",             "pass4_0d", "feat",
     "1 if n_unique_drugs_for_company <= 1 (single-asset company)",
     "feat_n_unique_drugs_for_company", "deterministic"),
    ("feat_lead_asset_dependency_score",           "pass4_0d", "feat",
     "1.0 (1 drug) | 0.7 (2) | 0.4 (3-4) | 0.2 (5+) — pipeline concentration proxy",
     "feat_n_unique_drugs_for_company", "deterministic"),
    ("feat_n_late_stage_trials_for_company",       "pass4_0d", "feat",
     "Count of late-stage (>=Ph2/3) trials for this company in dataset",
     "feat_late_stage_flag, feat_phase_num, ticker", "deterministic"),
    ("feat_pipeline_concentration_simple",         "pass4_0d", "feat",
     "lead_asset_dep × late_frac — existential risk proxy (intermediate)",
     "feat_lead_asset_dependency_score, feat_n_late_stage_trials_for_company, "
     "feat_n_trials_for_company", "deterministic"),
    # ── Step 1: Company asset depth ──────────────────────────────────────────
    ("feat_asset_trial_count_for_company",         "pass4", "feat",
     "Count of dataset events for this drug_name within this ticker",
     "drug_name, ticker", "deterministic"),
    ("feat_asset_trial_share",                     "pass4", "feat",
     "asset_trial_count / n_trials_for_company — fraction of company events on this drug",
     "feat_asset_trial_count_for_company, feat_n_trials_for_company", "deterministic"),
    ("feat_pipeline_depth_score",                  "pass4", "feat",
     "min(late_stage_trials,5)*0.6 + log1p(n_unique_drugs)*0.4 — pipeline breadth/depth composite",
     "feat_n_late_stage_trials_for_company, feat_n_unique_drugs_for_company", "deterministic"),
    ("feat_pivotal_proxy_score",                   "pass4", "feat",
     "0-5 pivotal importance: +1.5 Ph3, +1.0 Ph2/3 or NDA, +0.5 each BTD/priority/text keywords",
     "feat_phase_num, feat_regulatory_stage_score, feat_breakthrough_flag, "
     "feat_priority_review_flag, pivotal_evidence, v_summary, v_pr_key_info", "deterministic"),
    ("feat_primary_endpoint_known_flag",           "pass4", "feat",
     "1 if primary_endpoint_met is Yes or No (outcome is known, not null/unclear)",
     "primary_endpoint_met", "deterministic"),
    ("feat_superiority_flag",                      "pass4", "feat",
     "1 if superiority / met-primary / positive-result keywords found in evidence text",
     "primary_endpoint_result, v_pr_key_info, v_summary, catalyst_summary, pivotal_evidence",
     "keyword"),
    ("feat_stat_sig_flag",                         "pass4", "feat",
     "1 if statistical significance keywords (p-value, HR, CI) found in evidence text",
     "primary_endpoint_result, v_pr_key_info, v_summary", "keyword"),
    ("feat_clinically_meaningful_flag",            "pass4", "feat",
     "1 if clinically meaningful / durable / complete response keywords found",
     "primary_endpoint_result, v_pr_key_info, v_summary", "keyword"),
    ("feat_mixed_results_flag",                    "pass4", "feat",
     "1 if mixed/partial/did-not-meet keywords found in evidence text",
     "primary_endpoint_result, v_pr_key_info, v_summary, catalyst_summary", "keyword"),
    ("feat_blinded_flag",                          "pass4", "feat",
     "1 if double/single-blind or masking keywords in trial title or design text",
     "ct_official_title, v_pr_title, v_summary", "keyword"),
    ("feat_open_label_flag",                       "pass4", "feat",
     "1 if open-label or unblinded keywords in trial title or design text",
     "ct_official_title, v_pr_title, v_summary", "keyword"),
    ("feat_small_trial_flag",                      "pass4", "feat",
     "1 if ct_enrollment < 50 (early-stage / feasibility trial)",
     "ct_enrollment", "deterministic"),
    ("feat_completed_flag",                        "pass4", "feat",
     "1 if ct_status == COMPLETED at time of event",
     "ct_status", "deterministic"),
    ("feat_recent_completion_flag",                "pass4", "feat",
     "1 if ct_status==COMPLETED and primary completion within 12 months before event_date",
     "ct_status, ct_primary_completion, event_date", "deterministic"),
    ("feat_terminated_at_event_flag",              "pass4_pit", "feat",
     "1 if AACT monthly snapshot status == TERMINATED at closest month ≤ event date "
     "(point-in-time; fixes SNAPSHOT_UNSAFE feat_terminated_flag). "
     "Falls back to feat_terminated_flag when ct_status_at_event is null.",
     "ct_status_at_event, feat_terminated_flag", "deterministic"),
    ("feat_withdrawn_at_event_flag",               "pass4_pit", "feat",
     "1 if AACT monthly snapshot status == WITHDRAWN at closest month ≤ event date "
     "(point-in-time; fixes SNAPSHOT_UNSAFE feat_withdrawn_flag). "
     "Falls back to feat_withdrawn_flag when ct_status_at_event is null.",
     "ct_status_at_event, feat_withdrawn_flag", "deterministic"),
    ("feat_completed_at_event_flag",               "pass4_pit", "feat",
     "1 if AACT monthly snapshot status == COMPLETED at closest month ≤ event date "
     "(point-in-time; replaces feat_completed_before_event date proxy in v7+)",
     "ct_status_at_event", "deterministic"),
    ("feat_active_not_recruiting_at_event_flag",   "pass4_pit", "feat",
     "1 if AACT monthly snapshot status == ACTIVE_NOT_RECRUITING at closest month ≤ event date "
     "(point-in-time; replaces feat_active_not_recruiting_flag snapshot in v7+)",
     "ct_status_at_event", "deterministic"),
    ("feat_mesh_level1_encoded",                   "pass4", "feat",
     "Ordinal int encoding of mesh_level1 (1=Oncology … 11=Other); same map as completeness_pass.py",
     "mesh_level1", "deterministic"),
    ("feat_therapeutic_superclass",                "pass4", "feat",
     "Human-readable therapeutic area from mesh_level1: Oncology/CNS/Immunology/etc.",
     "mesh_level1", "deterministic"),
    ("feat_oncology_flag",                         "pass4", "feat",
     "1 if mesh_level1 == Neoplasms (oncology trial)",
     "mesh_level1", "deterministic"),
    ("feat_cns_flag",                              "pass4", "feat",
     "1 if mesh_level1 == Nervous System Diseases",
     "mesh_level1", "deterministic"),
    ("feat_rare_disease_flag",                     "pass4", "feat",
     "1 if orphan designation OR rare disease keywords in indication/mesh",
     "feat_orphan_flag, indication, mesh_level1", "deterministic"),
    ("feat_cash_runway_proxy",                     "pass4", "feat",
     "cash_position_m / market_cap_m — cash coverage ratio; >1.0 = cash exceeds market cap (existential signal). "
     "SNAPSHOT_UNSAFE (yfinance current market_cap) — not in training roster.",
     "cash_position_m, market_cap_m", "deterministic"),
    ("feat_volatility",                            "pass4", "feat",
     "atr_pct — pre-event 20-day Wilder ATR as fraction of price; strictly backward-looking, pre-event safe",
     "atr_pct", "deterministic"),
]


def build_feature_dict(df):
    meta_lookup = {
        name: (desc, src, stage, src_type)
        for name, stage, ftype, desc, src, src_type in NEW_FEATURE_META
    }
    feat_cols   = [c for c in df.columns if c.startswith("feat_")]
    target_cols = [c for c in df.columns if c.startswith("target_")]
    rows = []
    for col in target_cols + feat_cols:
        n_valid = df[col].notna().sum()
        n_total = len(df)
        meta = meta_lookup.get(col)
        rows.append({
            "feature_name":   col,
            "stage":          meta[2] if meta else "pass1/pass3",
            "feature_type":   "target" if col.startswith("target_") else "feat",
            "description":    meta[0] if meta else "",
            "source_columns": meta[1] if meta else "",
            "source_type":    meta[3] if meta else "deterministic",
            "n_valid":        int(n_valid),
            "n_null":         int(n_total - n_valid),
            "pct_valid":      round(n_valid / n_total * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    data_path, data_v, date_tag = _latest_version_file(BASE_DIR, "ml_dataset_features")
    dict_path, dict_v, _        = _latest_version_file(BASE_DIR, "ml_feature_dict")

    if not data_path:
        print("ERROR: no ml_dataset_features_*.csv found in " + BASE_DIR, file=sys.stderr)
        sys.exit(1)

    print(f"Input : {os.path.basename(data_path)}  (v{data_v})")
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Inject ct_status_at_event from master CSV ─────────────────────────────
    # Needed for feat_completed_at_event_flag + feat_active_not_recruiting_at_event_flag.
    # Drop stale column first if re-running.
    if "ct_status_at_event" in df.columns:
        df = df.drop(columns=["ct_status_at_event"])
    if os.path.exists(MASTER_CSV):
        master = pd.read_csv(MASTER_CSV, usecols=["nct_id", "event_date", "drug_name",
                                                   "ct_status_at_event"])
        pit_map = (master
                   .dropna(subset=["nct_id", "event_date"])
                   .drop_duplicates(subset=["nct_id", "event_date", "drug_name"])
                   [["nct_id", "event_date", "drug_name", "ct_status_at_event"]])
        df = df.merge(pit_map, on=["nct_id", "event_date", "drug_name"], how="left")
        pit_nn = df["ct_status_at_event"].notna().sum()
        print(f"Injected ct_status_at_event from master CSV: {pit_nn}/{len(df)} non-null")
    else:
        print(f"WARNING: master CSV not found at {MASTER_CSV}; "
              f"feat_completed_at_event_flag will be all-NaN")
    print()

    print("Step 0a: Clinical protocol core features")
    df = build_clinical_core(df)
    print("\nStep 0b: Regulatory designation flags")
    df = build_regulatory_flags(df)
    print("\nStep 0b.5: PIT terminated/withdrawn flags")
    df = build_status_pit_flags(df)
    print("\nStep 0c: Trial quality score")
    df = build_trial_quality_score(df)
    print("\nStep 0d: Company foundation features")
    df = build_company_foundation(df)
    print("\nStep 1: Company asset depth")
    df = build_company_asset_features(df)
    print("\nStep 2: Pivotal proxy score")
    df = build_pivotal_proxy(df)
    print("\nStep 3: Outcome / evidence flags")
    df = build_outcome_flags(df)
    print("\nStep 4: Trial design flags")
    df = build_design_flags(df)
    print("\nStep 5: Trial status / timing")
    df = build_timing_flags(df)
    print("\nStep 6: Disease structure")
    df = build_disease_features(df)
    print("\nStep 7: Financial context")
    df = build_financial_context(df)

    new_feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"\nTotal feat_ columns: {len(new_feat_cols)}")

    # Archive old files
    new_v = data_v + 1
    for src, label in [(data_path, "dataset"), (dict_path, "dict")]:
        if src and os.path.exists(src):
            dest = os.path.join(ARCHIVE_DIR, os.path.basename(src))
            shutil.move(src, dest)
            print(f"Archived {label}: archive/{os.path.basename(src)}")

    # Save v(N+1)
    new_data_path = os.path.join(BASE_DIR, f"ml_dataset_features_{date_tag}_v{new_v}.csv")
    new_dict_path = os.path.join(BASE_DIR, f"ml_feature_dict_{date_tag}_v{new_v}.csv")

    df.to_csv(new_data_path, index=False)
    feat_dict = build_feature_dict(df)
    feat_dict.to_csv(new_dict_path, index=False)

    print(f"\nSaved : {os.path.basename(new_data_path)}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"Saved : {os.path.basename(new_dict_path)}  ({len(feat_dict)} entries)")

    # Coverage report
    new_names = [m[0] for m in NEW_FEATURE_META]
    n = len(df)
    print(f"\n{'Feature':<52} {'Coverage':>9}  Source")
    print("-" * 75)
    for col in new_names:
        if col not in df.columns:
            print(f"  {col:<50} {'NOT BUILT':>9}")
            continue
        nn = df[col].notna().sum()
        meta = next((m for m in NEW_FEATURE_META if m[0] == col), None)
        src_type = meta[5] if meta else ""
        print(f"  {col:<50} {nn/n*100:>8.1f}%  {src_type}")

    print("\nNo Perplexity usage. All features deterministic or keyword-based.")


if __name__ == "__main__":
    main()

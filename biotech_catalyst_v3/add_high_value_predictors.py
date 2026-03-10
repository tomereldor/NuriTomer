"""
add_high_value_predictors.py
============================
Pass-3 feature engineering: company-dependency / existential-risk proxies,
regulatory-state flags, trial-quality metrics, and historical reaction priors.

Input  (default): ml_dataset_features_20260310_v1.csv   (831 rows × 80 cols)
Output           : ml_dataset_features_20260310_v2.csv   (831 rows × ~103 cols)
                   ml_feature_dict_20260310_v2.csv

The script is fully deterministic — no LLMs, no external API calls.

Usage:
    cd biotech_catalyst_v3
    python add_high_value_predictors.py
    python add_high_value_predictors.py --input ml_dataset_features_20260310_v1.csv --date 20260310
"""

import argparse
import glob
import os
import re
import shutil
import sys
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DEFAULT = "ml_dataset_features_20260310_v1.csv"
TODAY = date.today().strftime("%Y%m%d")
ARCHIVE_DIR = "archive"

# Text columns searched for all keyword flags (combined case-insensitively)
TEXT_COLS = [
    "pivotal_evidence",
    "v_pr_key_info",
    "v_summary",
    "catalyst_summary",
    "v_pr_title",
    "ct_official_title",
]

# ── Regulatory keyword lists (printed at runtime) ──────────────────────────
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
PIVOTAL_KEYWORDS = [
    "pivotal", "registrational", "registration study", "registration trial",
]

# Market-cap bucket edges (USD millions)
CAP_BINS   = [0, 300, 1000, 5000, float("inf")]
CAP_LABELS = ["micro", "small", "mid", "large"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_version(date_str: str, base_dir: str, prefix: str) -> int:
    """Return the next available vN integer for files matching prefix_DATE_vN.csv."""
    pattern = os.path.join(base_dir, f"{prefix}_{date_str}_v*.csv")
    existing = glob.glob(pattern)
    if not existing:
        return 1
    nums = []
    for f in existing:
        m = re.search(r"_v(\d+)\.csv$", f)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def _combined_text(df: pd.DataFrame, cols: list) -> pd.Series:
    """Join text columns into one lowercase string per row for keyword search."""
    present = [c for c in cols if c in df.columns]
    return (
        df[present].fillna("").astype(str)
        .apply(lambda r: " ".join(r), axis=1)
        .str.lower()
    )


def _keyword_flag(text: pd.Series, keywords: list) -> pd.Series:
    """Return 1 if any keyword found in text, else 0 (no NaN — all rows are searchable)."""
    pattern = "|".join(re.escape(k) for k in keywords)
    return text.str.contains(pattern, regex=True, na=False).astype(int)


def _archive_file(path: str, archive_dir: str) -> str:
    """Move file to archive_dir; return destination path."""
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, os.path.basename(path))
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(path, dest)
    return dest


# ---------------------------------------------------------------------------
# Step 1 — Company dependency / existential risk proxies
# ---------------------------------------------------------------------------

def build_company_dependency(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Step 1: Company dependency / existential risk proxies ──")

    # 1. Total trials per company (ticker-level count of rows)
    df["feat_n_trials_for_company"] = (
        df.groupby("ticker")["ticker"].transform("count").astype(int)
    )

    # 2. Distinct drug names per company
    drug_nunique = df.groupby("ticker")["drug_name"].transform(
        lambda s: s.dropna().nunique()
    )
    df["feat_n_unique_drugs_for_company"] = drug_nunique.astype(int)

    # 3. Single-asset flag: 1 if only 1 unique drug in entire pipeline
    df["feat_single_asset_company_flag"] = (
        df["feat_n_unique_drugs_for_company"] <= 1
    ).astype(int)

    # 4. Lead-asset dependency score — deterministic mapping
    # Mapping: 1 drug → 1.0 | 2 → 0.7 | 3–4 → 0.4 | 5+ → 0.2
    def _dep_score(n: int) -> float:
        if n <= 1: return 1.0
        if n == 2: return 0.7
        if n <= 4: return 0.4
        return 0.2

    df["feat_lead_asset_dependency_score"] = (
        df["feat_n_unique_drugs_for_company"].apply(_dep_score)
    )
    print("  feat_lead_asset_dependency_score mapping:")
    print("    1 drug  → 1.0  (single-asset; this trial is the entire company)")
    print("    2 drugs → 0.7  (near-single; this drug is the clear lead)")
    print("    3–4     → 0.4  (pipeline company; important but not only bet)")
    print("    5+      → 0.2  (diversified; single trial has lower existential weight)")

    # 5. Late-stage trials per company
    #    Late = feat_late_stage_flag == 1  OR  feat_phase_num >= 3.0
    if "feat_late_stage_flag" in df.columns:
        late_mask = (df["feat_late_stage_flag"] == 1)
    elif "feat_phase_num" in df.columns:
        late_mask = (df["feat_phase_num"] >= 3.0).fillna(False)
    else:
        late_mask = pd.Series(False, index=df.index)

    if "feat_phase_num" in df.columns:
        late_mask = late_mask | (df["feat_phase_num"] >= 3.0).fillna(False)

    df["_late_row"] = late_mask.astype(int)
    df["feat_n_late_stage_trials_for_company"] = (
        df.groupby("ticker")["_late_row"].transform("sum").astype(int)
    )
    df.drop(columns=["_late_row"], inplace=True)

    # 6. Pipeline concentration (simple)
    #    Formula:
    #      feat_pipeline_concentration_simple =
    #          feat_lead_asset_dependency_score
    #        × (feat_n_late_stage_trials_for_company / feat_n_trials_for_company)
    #
    #    High → few drugs AND most trials are late-stage → single existential late-stage bet
    #    Low  → diversified drugs OR mostly early-stage → any one trial is less decisive
    late_frac = (
        df["feat_n_late_stage_trials_for_company"]
        / df["feat_n_trials_for_company"].replace(0, np.nan)
    ).fillna(0.0)
    df["feat_pipeline_concentration_simple"] = (
        df["feat_lead_asset_dependency_score"] * late_frac
    ).round(4)

    print("\n  feat_pipeline_concentration_simple formula:")
    print("    = feat_lead_asset_dependency_score")
    print("      × (feat_n_late_stage_trials_for_company / feat_n_trials_for_company)")
    print("  Interpretation:")
    print("    High → few drugs, mostly late-stage  → existential, high-conviction move expected")
    print("    Low  → diversified or early pipeline → this readout is one of many bets")

    return df


# ---------------------------------------------------------------------------
# Step 2 — Regulatory-state features
# ---------------------------------------------------------------------------

def build_regulatory_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Step 2: Regulatory-state features ──")

    text = _combined_text(df, TEXT_COLS)

    df["feat_orphan_flag"]          = _keyword_flag(text, ORPHAN_KEYWORDS)
    df["feat_fast_track_flag"]      = _keyword_flag(text, FAST_TRACK_KEYWORDS)
    df["feat_breakthrough_flag"]    = _keyword_flag(text, BREAKTHROUGH_KEYWORDS)
    df["feat_nda_bla_flag"]         = _keyword_flag(text, NDA_BLA_KEYWORDS)
    df["feat_priority_review_flag"] = _keyword_flag(text, PRIORITY_REVIEW_KEYWORDS)

    # feat_regulatory_stage_score — ordinal regulatory ladder
    # Scoring (cumulative-max; higher tier overrides lower):
    #   0 = no regulatory language detected
    #   1 = pivotal / registrational  (trial is designed for approval pathway)
    #   2 = NDA / BLA submitted or pending
    #   3 = priority review / imminent FDA decision (PDUFA date visible)
    pivotal_hit  = _keyword_flag(text, PIVOTAL_KEYWORDS)
    score = pd.Series(0, index=df.index, dtype=int)
    score = np.where(pivotal_hit                    == 1, np.maximum(score, 1), score)
    score = np.where(df["feat_nda_bla_flag"]        == 1, np.maximum(score, 2), score)
    score = np.where(df["feat_priority_review_flag"] == 1, np.maximum(score, 3), score)
    df["feat_regulatory_stage_score"] = score.astype(int)

    print(f"  Columns searched (case-insensitive, combined per row):")
    print(f"    {TEXT_COLS}")
    print(f"  feat_orphan_flag          keywords: {ORPHAN_KEYWORDS}")
    print(f"  feat_fast_track_flag      keywords: {FAST_TRACK_KEYWORDS}")
    print(f"  feat_breakthrough_flag    keywords: {BREAKTHROUGH_KEYWORDS}")
    print(f"  feat_nda_bla_flag         keywords: {NDA_BLA_KEYWORDS}")
    print(f"  feat_priority_review_flag keywords: {PRIORITY_REVIEW_KEYWORDS}")
    print( "  feat_regulatory_stage_score scoring (cumulative-max):")
    print(f"    0 = none detected")
    print(f"    1 = pivotal/registrational keywords: {PIVOTAL_KEYWORDS}")
    print(f"    2 = NDA/BLA (from feat_nda_bla_flag)")
    print(f"    3 = priority review / PDUFA (from feat_priority_review_flag)")

    return df


# ---------------------------------------------------------------------------
# Step 3 — Trial quality / credibility features
# ---------------------------------------------------------------------------

def build_trial_quality(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Step 3: Trial quality / credibility features ──")

    enroll = pd.to_numeric(df["ct_enrollment"], errors="coerce")

    # 1. Log enrollment
    df["feat_enrollment_log"] = np.log1p(enroll)

    # 2. Randomized flag
    df["feat_randomized_flag"] = np.where(
        df["ct_allocation"].isna(),
        np.nan,
        (df["ct_allocation"] == "RANDOMIZED").astype(float),
    )

    # 3. Controlled flag: RANDOMIZED _or_ blinded/placebo language in title
    title_lower = df["ct_official_title"].fillna("").str.lower()
    blinded_title = title_lower.str.contains(
        r"placebo[- ]control|double[- ]blind|triple[- ]blind|controlled study|controlled trial",
        regex=True, na=False,
    )
    df["feat_controlled_flag"] = (
        (df["feat_randomized_flag"] == 1) | blinded_title
    ).astype(int)

    # 4. Withdrawn / terminated flags
    df["feat_withdrawn_flag"]  = (df["ct_status"] == "WITHDRAWN").astype(int)
    df["feat_terminated_flag"] = (df["ct_status"] == "TERMINATED").astype(int)

    # 5. Composite trial quality score
    #    Formula:
    #      feat_trial_quality_score =
    #          feat_design_quality_score          (from pass1; see below for its components)
    #        + 1.0  × blinded_title               (double-blind / placebo-controlled)
    #        + 0.5  × feat_breakthrough_flag
    #        − 2.0  × feat_withdrawn_flag
    #        − 2.0  × feat_terminated_flag
    #
    #    feat_design_quality_score components (already in dataset):
    #      +2  RANDOMIZED  |  +2  enroll > 300  |  +1  enroll 101–300
    #      +1  Phase 3     |  +0.5  Phase 2/3   |  −1  NON_RANDOMIZED
    #
    #    Theoretical range:  −5  to  +9.5

    if "feat_design_quality_score" in df.columns:
        base = df["feat_design_quality_score"].fillna(0.0)
    else:
        base = pd.Series(0.0, index=df.index)

    breakthrough = df.get("feat_breakthrough_flag", pd.Series(0, index=df.index)).astype(float)

    quality = (
        base
        + blinded_title.astype(float)
        + breakthrough * 0.5
        - df["feat_withdrawn_flag"].astype(float)  * 2.0
        - df["feat_terminated_flag"].astype(float) * 2.0
    )

    # NaN where all inputs are unavailable
    both_missing = df["ct_allocation"].isna() & df["ct_status"].isna() & enroll.isna()
    quality[both_missing] = np.nan
    df["feat_trial_quality_score"] = quality

    print("  feat_trial_quality_score formula:")
    print("    = feat_design_quality_score  (pass1 base)")
    print("      + 1.0  if ct_official_title contains double-blind / placebo-controlled")
    print("      + 0.5  if feat_breakthrough_flag == 1")
    print("      − 2.0  if ct_status == WITHDRAWN")
    print("      − 2.0  if ct_status == TERMINATED")
    print("    feat_design_quality_score base (already computed in pass1):")
    print("      +2 RANDOMIZED  | +2 enroll>300 | +1 enroll 101–300")
    print("      +1 Phase3      | +0.5 Phase2/3 | −1 NON_RANDOMIZED")
    print("    Blinded-title regex:")
    print("      placebo[- ]control | double[- ]blind | triple[- ]blind")
    print("      | controlled study | controlled trial")

    return df


# ---------------------------------------------------------------------------
# Step 4 — Historical reaction prior features
# ---------------------------------------------------------------------------

def build_reaction_priors(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── Step 4: Historical reaction prior features ──")
    print("  ⚠  LEAKAGE WARNING: priors are computed on the full dataset.")
    print("     Before model training, recompute these on the train split only.")
    print("     Included now for exploration and prototyping.")

    abs_move = df["stock_movement_atr_normalized"].abs()

    # 1. Prior by phase
    df["feat_prior_mean_abs_move_atr_by_phase"] = (
        abs_move.groupby(df["feat_phase_num"]).transform("mean")
    )

    # 2. Prior by therapeutic superclass (mesh_level1)
    df["feat_prior_mean_abs_move_atr_by_therapeutic_superclass"] = (
        abs_move.groupby(df["mesh_level1"]).transform("mean")
    )

    # 3. Cross: phase × therapeutic superclass
    cross_key = (
        df["feat_phase_num"].astype(str) + "__" + df["mesh_level1"].astype(str)
    )
    df["feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass"] = (
        abs_move.groupby(cross_key).transform("mean")
    )

    # 4. Market-cap bucket
    df["feat_market_cap_bucket"] = pd.cut(
        df["market_cap_m"],
        bins=CAP_BINS,
        labels=CAP_LABELS,
        right=False,
    ).astype(str)
    df["feat_market_cap_bucket"] = df["feat_market_cap_bucket"].replace("nan", np.nan)

    # 5. Prior by market-cap bucket
    df["feat_prior_mean_abs_move_atr_by_market_cap_bucket"] = (
        abs_move.groupby(df["feat_market_cap_bucket"]).transform("mean")
    )

    print(f"\n  Market-cap bucket edges: {CAP_BINS[:-1]} USD M  →  labels: {CAP_LABELS}")
    print("  Market-cap distribution in dataset:")
    print(df["feat_market_cap_bucket"].value_counts().to_string(header=False))

    print("\n  Phase priors (mean |ATR-normalised move|):")
    phase_tbl = (
        abs_move.groupby(df["feat_phase_num"]).agg(["mean", "count"])
        .rename(columns={"mean": "mean_abs_move_atr", "count": "n"})
        .round(3)
    )
    print(phase_tbl.to_string())

    print("\n  Therapeutic superclass priors (mean |ATR-normalised move|):")
    mesh_tbl = (
        abs_move.groupby(df["mesh_level1"]).agg(["mean", "count"])
        .rename(columns={"mean": "mean_abs_move_atr", "count": "n"})
        .sort_values("mean_abs_move_atr", ascending=False)
        .round(3)
    )
    print(mesh_tbl.to_string())

    print("\n  Market-cap bucket priors (mean |ATR-normalised move|):")
    cap_tbl = (
        abs_move.groupby(df["feat_market_cap_bucket"]).agg(["mean", "count"])
        .rename(columns={"mean": "mean_abs_move_atr", "count": "n"})
        .round(3)
    )
    print(cap_tbl.to_string())

    return df


# ---------------------------------------------------------------------------
# Feature meta for the updated dictionary
# ---------------------------------------------------------------------------

# Each entry: (name, feature_type, source_columns, transformation_logic, why_important)
NEW_FEATURE_META = [
    # ── Step 1: Company dependency ──────────────────────────────────────────
    (
        "feat_n_trials_for_company",
        "company_risk",
        "ticker",
        "count(rows) per ticker",
        "Total trial count for this company. Diversified pipelines (many trials) make "
        "any single readout less decisive. Low count = concentrated pipeline risk = "
        "bigger expected swing on any single data release.",
    ),
    (
        "feat_n_unique_drugs_for_company",
        "company_risk",
        "ticker, drug_name",
        "nunique(drug_name) per ticker, ignoring nulls",
        "Distinct drug count per company. The fewer the drugs, the more investor risk "
        "is concentrated in this one trial. Single-drug companies face existential outcomes.",
    ),
    (
        "feat_single_asset_company_flag",
        "company_risk",
        "feat_n_unique_drugs_for_company",
        "1 if feat_n_unique_drugs_for_company <= 1, else 0",
        "Binary existential-risk flag. Single-asset companies have zero pipeline fallback. "
        "A positive readout can double/triple the stock; a failure can cause a 70–90%% drop. "
        "This is the strongest binary predictor of extreme move magnitude.",
    ),
    (
        "feat_lead_asset_dependency_score",
        "company_risk",
        "feat_n_unique_drugs_for_company",
        "1 drug→1.0 | 2→0.7 | 3–4→0.4 | 5+→0.2",
        "Continuous existential-weight score. Captures the non-linear reduction in trial "
        "importance as pipeline diversifies. Strongly predictive of ATR-normalised move "
        "magnitude because investor risk-on/risk-off scales with company dependency.",
    ),
    (
        "feat_n_late_stage_trials_for_company",
        "company_risk",
        "ticker, feat_late_stage_flag, feat_phase_num",
        "count(rows per ticker where feat_late_stage_flag==1 or feat_phase_num>=3)",
        "Late-stage trial count per company. Companies with many Phase 3 assets have "
        "demonstrated execution; any one readout is less make-or-break than for a "
        "company whose sole late-stage trial is the current catalyst.",
    ),
    (
        "feat_pipeline_concentration_simple",
        "company_risk",
        "feat_lead_asset_dependency_score, feat_n_late_stage_trials_for_company, feat_n_trials_for_company",
        "feat_lead_asset_dependency_score × (feat_n_late_stage_for_company / feat_n_trials_for_company)",
        "Composite existential-risk proxy. High value = few drugs AND most trials are "
        "late-stage = this trial is both the lead asset and the company's only late-stage "
        "bet. For inference: high concentration + positive readout → extreme move; "
        "high concentration + failure → catastrophic/binary outcome.",
    ),
    # ── Step 2: Regulatory state ────────────────────────────────────────────
    (
        "feat_orphan_flag",
        "regulatory",
        ", ".join(TEXT_COLS),
        f"1 if combined text contains any of: {ORPHAN_KEYWORDS}",
        "Orphan Drug designation signals rare disease with smaller patient population, "
        "faster FDA review track, and typically higher pricing power and lower competition. "
        "These dynamics produce outsized stock reactions relative to trial size because "
        "commercial success probability is higher if the drug works.",
    ),
    (
        "feat_fast_track_flag",
        "regulatory",
        ", ".join(TEXT_COLS),
        f"1 if combined text contains any of: {FAST_TRACK_KEYWORDS}",
        "FDA Fast Track signals serious / unmet medical need and enables rolling submission, "
        "accelerating the path from trial data to approval. Companies highlight this "
        "because it compresses time-to-market risk, making a positive readout even more "
        "immediately valuable.",
    ),
    (
        "feat_breakthrough_flag",
        "regulatory",
        ", ".join(TEXT_COLS),
        f"1 if combined text contains any of: {BREAKTHROUGH_KEYWORDS}",
        "Breakthrough Therapy designation is the FDA's strongest signal: the drug is "
        "substantially better than existing treatment. Breakthrough drugs command premium "
        "valuations and attract acquisition interest. A positive readout for a Breakthrough "
        "drug can trigger immediate M&A speculation, amplifying the stock move.",
    ),
    (
        "feat_nda_bla_flag",
        "regulatory",
        ", ".join(TEXT_COLS),
        f"1 if combined text contains any of: {NDA_BLA_KEYWORDS}",
        "NDA/BLA submission or planned submission means the company is at or near the "
        "regulatory finish line. The market re-prices the stock to reflect the probability "
        "that FDA approval follows. Positive trial data with an NDA already filed compresses "
        "approval probability toward 1, creating a step-change in valuation.",
    ),
    (
        "feat_priority_review_flag",
        "regulatory",
        ", ".join(TEXT_COLS),
        f"1 if combined text contains any of: {PRIORITY_REVIEW_KEYWORDS}",
        "Priority Review (6-month vs standard 10-month FDA review) signals an imminent "
        "approval decision with a visible PDUFA date. This concentrates the entire "
        "risk/reward into a single near-term binary event, producing the largest moves "
        "in biotech. A PDUFA date in the press release is a key inference signal.",
    ),
    (
        "feat_regulatory_stage_score",
        "regulatory",
        "feat_nda_bla_flag, feat_priority_review_flag, pivotal_evidence",
        "0=no regulatory language | 1=pivotal/registrational | 2=NDA/BLA | 3=priority review",
        "Ordinal regulatory ladder score. Higher score = closer to approval = higher option "
        "value of a positive result. A pivotal Phase 3 trial with priority review (score=3) "
        "will move the stock far more than an exploratory Phase 2 (score=0) on equivalent "
        "data quality. Critical for distinguishing 'trial result' from 'approval catalyst'.",
    ),
    # ── Step 3: Trial quality ───────────────────────────────────────────────
    (
        "feat_enrollment_log",
        "trial_quality",
        "ct_enrollment",
        "log1p(ct_enrollment) = log(ct_enrollment + 1)",
        "Log-scale enrollment size. Large trials produce statistically robust, "
        "FDA-grade evidence; investors react with higher conviction. Log scale prevents "
        "very large trial outliers from dominating while preserving the key signal that "
        "n=20 Phase 1 and n=1,000 Phase 3 trials have very different evidential weight.",
    ),
    (
        "feat_randomized_flag",
        "trial_quality",
        "ct_allocation",
        "1 if ct_allocation == RANDOMIZED, NaN if allocation missing",
        "Randomized trials produce causally interpretable efficacy results. Non-randomized "
        "single-arm studies face FDA skepticism and require historical control comparisons "
        "that are harder to validate. Randomisation is the primary determinant of how "
        "convincing the data is to regulators and investors.",
    ),
    (
        "feat_controlled_flag",
        "trial_quality",
        "ct_allocation, ct_official_title",
        "1 if RANDOMIZED or ct_official_title matches (placebo-control|double-blind|triple-blind|controlled study|controlled trial)",
        "Controlled design (active placebo arm) allows direct efficacy comparison without "
        "relying on external benchmarks. Double-blind eliminates performance bias. "
        "More controlled trials produce more credible data which investors price with "
        "higher conviction in either direction (larger moves, both up and down).",
    ),
    (
        "feat_withdrawn_flag",
        "trial_quality",
        "ct_status",
        "1 if ct_status == WITHDRAWN",
        "Trial was withdrawn before enrollment began. A catalyst event from a withdrawn "
        "trial is anomalous — may reflect early signal, protocol failure, or regulatory "
        "interaction. Flagged so the model treats this status as a credibility qualifier.",
    ),
    (
        "feat_terminated_flag",
        "trial_quality",
        "ct_status",
        "1 if ct_status == TERMINATED",
        "Trial terminated early. This is high-variance: early termination can signal "
        "either (a) safety failure (catastrophically negative) or (b) overwhelming "
        "interim efficacy (extremely positive — DSMB stops trial because treatment works). "
        "The direction is in the endpoint result; knowing termination occurred is "
        "essential context for calibrating the expected move size.",
    ),
    (
        "feat_trial_quality_score",
        "trial_quality",
        "feat_design_quality_score, feat_controlled_flag, feat_breakthrough_flag, feat_withdrawn_flag, feat_terminated_flag",
        "feat_design_quality_score + 1.0×blinded_title + 0.5×breakthrough − 2.0×withdrawn − 2.0×terminated",
        "Composite trial credibility score. Combines design quality (randomized, controlled, "
        "blinded, enrollment, phase) with regulatory designation (breakthrough) and adverse "
        "status (terminated/withdrawn). Higher score = more credible, regulatory-grade evidence "
        "= higher conviction in the market reaction. Key input for distinguishing exploratory "
        "signals from registration-quality data.",
    ),
    # ── Step 4: Reaction priors ─────────────────────────────────────────────
    (
        "feat_prior_mean_abs_move_atr_by_phase",
        "reaction_prior",
        "feat_phase_num, stock_movement_atr_normalized",
        "mean(abs(stock_movement_atr_normalized)) grouped by feat_phase_num, full dataset",
        "Historical average ATR-normalised absolute move per trial phase. Phase 1 data "
        "releases produce structurally different market reactions than Phase 3 pivotal data. "
        "This prior encodes that baseline expectation. "
        "⚠ Recompute on train-only split before deployment to prevent label leakage.",
    ),
    (
        "feat_prior_mean_abs_move_atr_by_therapeutic_superclass",
        "reaction_prior",
        "mesh_level1, stock_movement_atr_normalized",
        "mean(abs(stock_movement_atr_normalized)) grouped by mesh_level1, full dataset",
        "Therapeutic area prior. Oncology trials historically move biotech stocks more "
        "than gastrointestinal or musculoskeletal trials, reflecting structural investor "
        "attention and valuation norms by disease area. "
        "⚠ Recompute on train-only split before deployment to prevent label leakage.",
    ),
    (
        "feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass",
        "reaction_prior",
        "feat_phase_num, mesh_level1, stock_movement_atr_normalized",
        "mean(abs(stock_movement_atr_normalized)) grouped by (feat_phase_num × mesh_level1), full dataset",
        "Cross prior for phase × therapeutic area. A Phase 3 oncology trial is categorically "
        "different from a Phase 3 cardiovascular trial in expected market reaction. "
        "This cross-group prior captures that interaction more precisely than either "
        "phase or therapeutic area alone. "
        "⚠ Recompute on train-only split before deployment to prevent label leakage.",
    ),
    (
        "feat_market_cap_bucket",
        "reaction_prior",
        "market_cap_m",
        f"pd.cut(market_cap_m, bins={CAP_BINS}, labels={CAP_LABELS}, right=False)",
        "Market cap tier: micro(<$300M), small($300M–$1B), mid($1B–$5B), large($5B+). "
        "Micro-cap companies move exponentially more than large-cap on equivalent data "
        "because their entire enterprise value is the pipeline. Important for interaction "
        "features, stratification, and as a categorical group variable.",
    ),
    (
        "feat_prior_mean_abs_move_atr_by_market_cap_bucket",
        "reaction_prior",
        "feat_market_cap_bucket, stock_movement_atr_normalized",
        "mean(abs(stock_movement_atr_normalized)) grouped by feat_market_cap_bucket, full dataset",
        "Market cap tier prior. Captures non-linear threshold effects in investor "
        "behavior: below $300M market cap, stocks react in qualitatively different ways "
        "than mid/large-cap stocks on the same trial outcome. Complements the continuous "
        "feat_log_market_cap by adding the group-level baseline reaction. "
        "⚠ Recompute on train-only split before deployment to prevent label leakage.",
    ),
]


# ---------------------------------------------------------------------------
# Feature dictionary builder
# ---------------------------------------------------------------------------

def build_feature_dict(df: pd.DataFrame, old_dict_path) -> pd.DataFrame:
    """
    Load and normalize the previous feature dict (pass1 features),
    then append new pass3 features.
    """
    if old_dict_path and os.path.exists(old_dict_path):
        old = pd.read_csv(old_dict_path)
        # Normalise old schema → new unified schema
        col_rename = {
            "type":        "feature_type",
            "description": "transformation_logic",
            "n_valid":     "coverage",
            "n_null":      "missing_count",
            "pct_valid":   "_pct_valid_drop",  # will drop
        }
        old = old.rename(columns={k: v for k, v in col_rename.items() if k in old.columns})
        old = old.drop(columns=["_pct_valid_drop"], errors="ignore")
        if "stage" not in old.columns:
            old["stage"] = "pass1"
        for c in ["feature_name", "stage", "feature_type", "source_columns",
                  "transformation_logic", "dtype", "coverage", "missing_count"]:
            if c not in old.columns:
                old[c] = ""
        old = old[[
            "feature_name", "stage", "feature_type", "source_columns",
            "transformation_logic", "dtype", "coverage", "missing_count",
        ]]
    else:
        old = pd.DataFrame(columns=[
            "feature_name", "stage", "feature_type", "source_columns",
            "transformation_logic", "dtype", "coverage", "missing_count",
        ])

    # Build new rows
    new_rows = []
    for name, ftype, src, logic, why in NEW_FEATURE_META:
        if name not in df.columns:
            continue
        col = df[name]
        new_rows.append({
            "feature_name":       name,
            "stage":              "pass3",
            "feature_type":       ftype,
            "source_columns":     src,
            "transformation_logic": logic,
            "dtype":              str(col.dtype),
            "coverage":           int(col.notna().sum()),
            "missing_count":      int(col.isna().sum()),
        })

    new_df = pd.DataFrame(new_rows)
    return pd.concat([old, new_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add high-value predictors (pass 3)")
    parser.add_argument("--input",  default=INPUT_DEFAULT,
                        help="Input feature CSV (default: ml_dataset_features_20260310_v1.csv)")
    parser.add_argument("--date",   default=TODAY,
                        help="Date tag for output files (YYYYMMDD, default: today)")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (default: current directory)")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(args.outdir, input_path)
    if not os.path.exists(input_path):
        # Try relative to cwd
        if not os.path.exists(args.input):
            print(f"ERROR: input not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        input_path = args.input

    df = pd.read_csv(input_path)
    cols_before  = df.shape[1]
    input_feats  = set(c for c in df.columns if c.startswith("feat_"))
    print(f"Input : {input_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    # ── Locate previous feature dict ──────────────────────────────────────
    m = re.search(r"_v(\d+)\.csv$", os.path.basename(input_path))
    in_ver = int(m.group(1)) if m else 1
    old_dict_name = re.sub(r"ml_dataset_features", "ml_feature_dict",
                           os.path.basename(input_path))
    old_dict_path = os.path.join(os.path.dirname(input_path) or args.outdir, old_dict_name)
    if not os.path.exists(old_dict_path):
        cands = glob.glob(os.path.join(args.outdir,
                          f"ml_feature_dict_{args.date}_v{in_ver}.csv"))
        old_dict_path = cands[0] if cands else None

    # ── Build features ────────────────────────────────────────────────────
    df = build_company_dependency(df)
    df = build_regulatory_features(df)
    df = build_trial_quality(df)
    df = build_reaction_priors(df)

    new_feat_cols = sorted(
        c for c in df.columns
        if c.startswith("feat_") and c not in input_feats
    )
    cols_after = df.shape[1]

    # ── Versioned output paths ────────────────────────────────────────────
    ver      = _next_version(args.date, args.outdir, "ml_dataset_features")
    data_out = os.path.join(args.outdir, f"ml_dataset_features_{args.date}_v{ver}.csv")
    dict_out = os.path.join(args.outdir, f"ml_feature_dict_{args.date}_v{ver}.csv")

    df.to_csv(data_out, index=False)

    feat_dict = build_feature_dict(df, old_dict_path)
    feat_dict.to_csv(dict_out, index=False)

    # ── Archive superseded files ──────────────────────────────────────────
    archive_base = os.path.join(args.outdir, ARCHIVE_DIR)
    archived = []
    if os.path.exists(input_path) and os.path.abspath(input_path) != os.path.abspath(data_out):
        archived.append((input_path, _archive_file(input_path, archive_base)))
    if old_dict_path and os.path.exists(old_dict_path) and \
            os.path.abspath(old_dict_path) != os.path.abspath(dict_out):
        archived.append((old_dict_path, _archive_file(old_dict_path, archive_base)))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"Input  : {input_path}")
    print(f"Output : {data_out}")
    print(f"Rows   : {df.shape[0]}")
    print(f"Columns: {cols_before} → {cols_after}  (+{cols_after - cols_before} columns)")
    print(f"\nNew features added ({len(new_feat_cols)}):")
    print(f"  {'Feature':<55}  {'Valid':>5}  {'Null':>4}")
    print(f"  {'─'*55}  {'─'*5}  {'─'*4}")
    for c in new_feat_cols:
        n_valid = int(df[c].notna().sum())
        n_null  = int(df[c].isna().sum())
        print(f"  {c:<55}  {n_valid:>5}  {n_null:>4}")
    if archived:
        print(f"\nArchived:")
        for src, dst in archived:
            print(f"  {src}  →  {dst}")
    print(f"\nFeature dictionary: {dict_out}  ({len(feat_dict)} entries total)")
    print("Done.")


if __name__ == "__main__":
    main()

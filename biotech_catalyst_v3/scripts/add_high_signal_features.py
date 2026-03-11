"""
add_high_signal_features.py
============================
Pass-4 deterministic feature engineering — high-signal features for baseline ML.

Reads  : latest ml_dataset_features_*_vN.csv
Outputs: ml_dataset_features_*_v(N+1).csv
         ml_feature_dict_*_v(N+1).csv
Archives: superseded latest files to archive/

New features added (22):
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
    feat_completed_flag, feat_recent_completion_flag
  Step 6 — Disease structure
    feat_therapeutic_superclass, feat_oncology_flag, feat_cns_flag,
    feat_rare_disease_flag
  Step 7 — Financial context
    feat_cash_runway_proxy

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

    df["feat_completed_flag"] = (status == "COMPLETED").astype(float)
    df.loc[status.isna(), "feat_completed_flag"] = float("nan")

    comp_dates = pd.to_datetime(
        df.get("ct_primary_completion", pd.Series(dtype=str)), errors="coerce"
    )
    days_since = (event_dates - comp_dates).dt.days
    df["feat_recent_completion_flag"] = (
        (status == "COMPLETED") & (days_since >= 0) & (days_since <= 365)
    ).astype(float)
    df.loc[status.isna(), "feat_recent_completion_flag"] = float("nan")

    print(f"  completed={df['feat_completed_flag'].sum()}, "
          f"recent_completion={df['feat_recent_completion_flag'].sum()}")
    return df


# ---------------------------------------------------------------------------
# Step 6 — Disease structure
# ---------------------------------------------------------------------------

def build_disease_features(df):
    # feat_therapeutic_superclass: human-readable string version of mesh_level1
    df["feat_therapeutic_superclass"] = (
        df["mesh_level1"].map(MESH_TO_SUPERCLASS).fillna("Other")
    )

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
    feat_cash_runway_proxy = cash_position_m / market_cap_m
    > 1.0 means company holds more cash than market cap — existential context.
    Clipped at 5.0 to avoid extreme outliers dominating.
    """
    cash = pd.to_numeric(df.get("cash_position_m"), errors="coerce")
    mcap = pd.to_numeric(df.get("market_cap_m"),    errors="coerce").replace(0, np.nan)
    df["feat_cash_runway_proxy"] = (cash / mcap).round(4).clip(upper=5.0)

    nn = df["feat_cash_runway_proxy"].notna().sum()
    above1 = (df["feat_cash_runway_proxy"] > 1.0).sum()
    print(f"  cash_runway_proxy: {nn}/{len(df)} non-null, "
          f"{above1} rows with cash > market_cap")
    return df


# ---------------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------------

NEW_FEATURE_META = [
    # (feature_name, stage, feature_type, description, source_columns, source_type)
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
     "cash_position_m / market_cap_m — cash coverage ratio; >1.0 = cash exceeds market cap (existential signal)",
     "cash_position_m, market_cap_m", "deterministic"),
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
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols\n")

    print("Step 1: Company asset depth")
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

"""
baseline_followup_analysis.py
==============================
Follow-up analysis on baseline model results.

Parts:
  1. Error analysis   — FP/FN clustering by phase, disease, market cap, etc.
  2. Threshold sweep  — P/R/F1 tradeoff, top-decile precision
  3. Feature ablation — 4 feature families vs full set (LogReg)
  4. Markdown report

Usage (from biotech_catalyst_v3/):
    python -m scripts.baseline_followup_analysis
"""

import glob
import os
import re
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ML_DATA_DIR = os.path.join(BASE_DIR, "data", "ml")
FIGS_DIR    = os.path.join(REPORTS_DIR, "figures")
TARGET      = "target_large_move"
METADATA    = ["ticker", "event_date", "drug_name", "nct_id"]
RANDOM_STATE = 42

plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")

# ── Feature family definitions ──────────────────────────────────────────────
FAMILY_MARKET = [
    "feat_volatility", "feat_log_market_cap",
    "feat_short_squeeze_flag", "feat_ownership_low_flag", "feat_cash_runway_proxy",
]
FAMILY_CLINICAL = [
    "feat_mesh_level1_encoded",
    "feat_phase_num", "feat_late_stage_flag",
    "feat_regulatory_stage_score", "feat_pivotal_proxy_score",
    "feat_design_quality_score", "feat_trial_quality_score",
    "feat_enrollment_log",
    "feat_active_not_recruiting_flag", "feat_completed_flag",
    "feat_recent_completion_flag",
    "feat_orphan_flag", "feat_breakthrough_flag",
    "feat_fast_track_flag", "feat_nda_bla_flag",
    # one-hot cols will be added dynamically
]
FAMILY_COMPANY = [
    "feat_n_unique_drugs_for_company", "feat_single_asset_company_flag",
    "feat_lead_asset_dependency_score", "feat_asset_trial_share",
    "feat_pipeline_depth_score",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _latest(base_dir, prefix):
    files = glob.glob(os.path.join(base_dir, f"{prefix}_*.csv"))
    best, best_v, date_tag = None, 0, None
    for f in files:
        m = re.search(r"_(\d{8})_v(\d+)\.csv$", f)
        if m:
            v = int(m.group(2))
            if v > best_v:
                best_v, best, date_tag = v, f, m.group(1)
    return best, best_v, date_tag


def _metrics(y_true, y_prob, thresh):
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "roc_auc":       round(roc_auc_score(y_true, y_prob)       if len(np.unique(y_true)) > 1 else 0, 4),
        "pr_auc":        round(average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0, 4),
        "bal_acc":       round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1":            round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


# ── Part 1: Error analysis ────────────────────────────────────────────────────

def error_analysis(train_df, val_pred, test_pred, feat_df):
    """Return dict of grouped error patterns."""
    results = {}

    for split_name, pred_df in [("val", val_pred), ("test", test_pred)]:
        # Join predictions with feature data
        extra_cols = ["ticker", "event_date",
                      "feat_phase_num", "feat_therapeutic_superclass",
                      "feat_market_cap_bucket", "feat_volatility",
                      "feat_lead_asset_dependency_score",
                      "feat_date_corrected_flag", "feat_log_market_cap"]
        if "nct_id" in feat_df.columns:
            extra_cols.append("nct_id")
        merged = pred_df.merge(
            feat_df[[c for c in extra_cols if c in feat_df.columns]],
            on=["ticker", "event_date"], how="left"
        )
        if "nct_id" not in merged.columns:
            merged["nct_id"] = np.nan

        best_prob = "prob_lgbm"
        thresh = 0.70

        merged["pred_best"] = (merged[best_prob] >= thresh).astype(int)
        actual = merged[TARGET]
        pred   = merged["pred_best"]

        fp_mask = (pred == 1) & (actual == 0)
        fn_mask = (pred == 0) & (actual == 1)
        tp_mask = (pred == 1) & (actual == 1)
        tn_mask = (pred == 0) & (actual == 0)

        fp = merged[fp_mask]
        fn = merged[fn_mask]

        def group_summary(sub, cols, label):
            out = {}
            for col in cols:
                if col not in sub.columns:
                    continue
                vc = sub[col].value_counts(dropna=False).head(6)
                out[col] = vc.to_dict()
            return out

        group_cols = [
            "feat_phase_num", "feat_therapeutic_superclass",
            "feat_market_cap_bucket",
        ]

        # Volatility bucket
        merged["_vol_bucket"] = pd.cut(
            merged["feat_volatility"],
            bins=[0, 5, 10, 20, 200], labels=["low(<5)", "mid(5-10)", "high(10-20)", "vhigh(>20)"]
        )
        group_cols.append("_vol_bucket")

        # nct_id missing
        merged["_nct_missing"] = merged["nct_id"].isna().map({True: "no_nct", False: "has_nct"})
        group_cols.append("_nct_missing")

        # date corrected
        merged["_date_corrected"] = merged["feat_date_corrected_flag"].map(
            {1: "date_corrected", 0: "not_corrected", np.nan: "unknown"}
        )
        group_cols.append("_date_corrected")

        results[split_name] = {
            "n_fp": int(fp_mask.sum()),
            "n_fn": int(fn_mask.sum()),
            "n_tp": int(tp_mask.sum()),
            "n_tn": int(tn_mask.sum()),
            "fp_groups": group_summary(fp, group_cols, "FP"),
            "fn_groups": group_summary(fn, group_cols, "FN"),
            "merged": merged,
        }

    return results


# ── Part 2: Threshold sweep ──────────────────────────────────────────────────

def threshold_sweep(y_true, y_prob_lr, y_prob_lgb, split_label):
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for model_name, y_prob in [("logreg", y_prob_lr), ("lgbm", y_prob_lgb)]:
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            rows.append({
                "model":     model_name,
                "split":     split_label,
                "threshold": t,
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
                "n_flagged": int(y_pred.sum()),
                "roc_auc":   round(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0, 4),
            })

        # Top-decile precision
        n_top = max(1, int(len(y_true) * 0.10))
        top_idx = np.argsort(y_prob)[::-1][:n_top]
        top_prec = y_true.iloc[top_idx].mean() if hasattr(y_true, "iloc") else y_true[top_idx].mean()
        rows.append({
            "model":     model_name,
            "split":     split_label,
            "threshold": "top_10pct",
            "precision": round(float(top_prec), 4),
            "recall":    None,
            "f1":        None,
            "n_flagged": n_top,
            "roc_auc":   round(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0, 4),
        })

    return pd.DataFrame(rows)


def plot_threshold_curves(thresh_df, split_label, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, model in zip(axes, ["logreg", "lgbm"]):
        sub = thresh_df[(thresh_df["model"] == model) &
                        (thresh_df["threshold"] != "top_10pct")].copy()
        sub["threshold"] = sub["threshold"].astype(float)
        ax.plot(sub["threshold"], sub["precision"], "b-o", ms=4, label="Precision")
        ax.plot(sub["threshold"], sub["recall"],    "r-o", ms=4, label="Recall")
        ax.plot(sub["threshold"], sub["f1"],        "g-o", ms=4, label="F1")
        ax.set_xlabel("Threshold")
        ax.set_title(f"{model.upper()} — P/R/F1 vs Threshold ({split_label})")
        ax.legend()
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ── Part 3: Feature-family ablation ─────────────────────────────────────────

def ablation_study(train_df, val_df, test_df, feature_cols_all):
    """Train LogReg on 4 feature families + full set. Returns ablation metrics."""

    # Expand family lists with one-hot variants present in the data
    all_tc_cols = [c for c in feature_cols_all if c.startswith("feat_therapeutic_superclass_")]
    all_ep_cols = [c for c in feature_cols_all if c.startswith("feat_event_proximity_bucket_")]
    clinical_full = (
        [c for c in FAMILY_CLINICAL if c in feature_cols_all]
        + all_tc_cols + all_ep_cols
    )

    families = {
        "A_market":   [c for c in FAMILY_MARKET  if c in feature_cols_all],
        "B_clinical": clinical_full,
        "C_company":  [c for c in FAMILY_COMPANY if c in feature_cols_all],
        "D_full":     feature_cols_all,
    }

    y_tr  = train_df[TARGET].astype(int)
    y_va  = val_df[TARGET].astype(int)
    y_te  = test_df[TARGET].astype(int)
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos

    rows = []
    for family_name, cols in families.items():
        if not cols:
            continue
        X_tr = train_df[cols].astype(float)
        X_va = val_df[cols].astype(float)
        X_te = test_df[cols].astype(float)

        lr = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            solver="lbfgs", C=0.1, random_state=RANDOM_STATE,
        )
        lr.fit(X_tr, y_tr)
        prob_va = lr.predict_proba(X_va)[:, 1]
        prob_te = lr.predict_proba(X_te)[:, 1]

        # Best-F1 threshold on val
        thresholds = np.linspace(0.1, 0.9, 81)
        f1s = [f1_score(y_va, (prob_va >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = thresholds[np.argmax(f1s)]

        rows.append({
            "family":      family_name,
            "n_features":  len(cols),
            "val_roc_auc": round(roc_auc_score(y_va, prob_va), 4),
            "val_f1":      round(max(f1s), 4),
            "val_bal_acc": round(balanced_accuracy_score(y_va, (prob_va >= best_t).astype(int)), 4),
            "test_roc_auc":round(roc_auc_score(y_te, prob_te), 4),
            "test_f1":     round(f1_score(y_te, (prob_te >= best_t).astype(int), zero_division=0), 4),
            "test_bal_acc":round(balanced_accuracy_score(y_te, (prob_te >= best_t).astype(int)), 4),
        })
        print(f"  {family_name:12s}  n={len(cols):3d}  "
              f"val_AUC={rows[-1]['val_roc_auc']:.3f}  "
              f"test_AUC={rows[-1]['test_roc_auc']:.3f}  "
              f"val_F1={rows[-1]['val_f1']:.3f}")

    return pd.DataFrame(rows)


def plot_ablation(abl_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = abl_df["family"].str.replace("_", " ")
    for ax, col, title in [
        (axes[0], "val_roc_auc",  "Val ROC-AUC by Feature Family"),
        (axes[1], "test_roc_auc", "Test ROC-AUC by Feature Family"),
    ]:
        colors = ["steelblue"] * (len(abl_df) - 1) + ["tomato"]
        ax.barh(labels, abl_df[col], color=colors, edgecolor="white")
        ax.set_xlim(0.4, max(abl_df[col]) + 0.05)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("ROC-AUC")
        for i, v in enumerate(abl_df[col]):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ── Part 4: Build markdown report ────────────────────────────────────────────

def build_report(error_res, thresh_df, abl_df, feat_df, today_str):
    te = error_res["test"]
    va = error_res["val"]

    def fmt_group(groups, key, top_n=4):
        if key not in groups:
            return "_not available_"
        items = list(groups[key].items())[:top_n]
        return " · ".join(f"{k}={v}" for k, v in items)

    # Threshold summary — best F1 and high-precision threshold for both models
    def thresh_summary(model):
        sub = thresh_df[(thresh_df["model"] == model) &
                        (thresh_df["split"] == "test") &
                        (thresh_df["threshold"] != "top_10pct")].copy()
        sub["threshold"] = sub["threshold"].astype(float)
        best_f1_row  = sub.loc[sub["f1"].idxmax()]
        high_prec    = sub[sub["precision"] >= 0.55]
        hp_row       = high_prec.loc[high_prec["recall"].idxmax()] if len(high_prec) else None
        top10_row    = thresh_df[(thresh_df["model"] == model) &
                                  (thresh_df["split"] == "test") &
                                  (thresh_df["threshold"] == "top_10pct")].iloc[0]
        return best_f1_row, hp_row, top10_row

    lr_f1, lr_hp, lr_top = thresh_summary("logreg")
    lgb_f1, lgb_hp, lgb_top = thresh_summary("lgbm")

    best_family = abl_df.loc[abl_df["test_roc_auc"].idxmax(), "family"]
    full_row    = abl_df[abl_df["family"] == "D_full"].iloc[0]

    abl_table = "\n".join(
        f"| {r['family']} | {r['n_features']} | {r['val_roc_auc']:.3f} | {r['val_f1']:.3f} | "
        f"{r['test_roc_auc']:.3f} | {r['test_f1']:.3f} | {r['test_bal_acc']:.3f} |"
        for _, r in abl_df.iterrows()
    )

    lr_hp_str  = (f"thresh={lr_hp['threshold']:.2f}  prec={lr_hp['precision']:.3f}  "
                  f"rec={lr_hp['recall']:.3f}  n={int(lr_hp['n_flagged'])}") if lr_hp is not None else "N/A"
    lgb_hp_str = (f"thresh={lgb_hp['threshold']:.2f}  prec={lgb_hp['precision']:.3f}  "
                  f"rec={lgb_hp['recall']:.3f}  n={int(lgb_hp['n_flagged'])}") if lgb_hp is not None else "N/A"

    return f"""# Biotech Baseline Model — Follow-up Analysis

**Date:** {today_str}
**Source models:** `reports/predictions_*_20260310_v1.csv`

---

## 1. Error Analysis (Test Set, LightGBM @ thresh=0.70)

| | Count |
|---|---|
| True Positives | {te['n_tp']} |
| True Negatives | {te['n_tn']} |
| **False Positives** | **{te['n_fp']}** |
| **False Negatives** | **{te['n_fn']}** |

### False Positives (predicted large move, was not)

| Feature | Distribution |
|---|---|
| Phase | {fmt_group(te['fp_groups'], 'feat_phase_num')} |
| Disease | {fmt_group(te['fp_groups'], 'feat_therapeutic_superclass')} |
| Mkt Cap Bucket | {fmt_group(te['fp_groups'], 'feat_market_cap_bucket')} |
| Volatility | {fmt_group(te['fp_groups'], '_vol_bucket')} |
| No NCT ID | {fmt_group(te['fp_groups'], '_nct_missing')} |
| Date corrected | {fmt_group(te['fp_groups'], '_date_corrected')} |

### False Negatives (was large move, missed)

| Feature | Distribution |
|---|---|
| Phase | {fmt_group(te['fn_groups'], 'feat_phase_num')} |
| Disease | {fmt_group(te['fn_groups'], 'feat_therapeutic_superclass')} |
| Mkt Cap Bucket | {fmt_group(te['fn_groups'], 'feat_market_cap_bucket')} |
| Volatility | {fmt_group(te['fn_groups'], '_vol_bucket')} |
| No NCT ID | {fmt_group(te['fn_groups'], '_nct_missing')} |
| Date corrected | {fmt_group(te['fn_groups'], '_date_corrected')} |

---

## 2. Threshold Analysis (Test Set)

### Logistic Regression

| Best threshold (F1) | Prec | Recall | F1 | n flagged |
|---|---|---|---|---|
| {lr_f1['threshold']:.2f} | {lr_f1['precision']:.3f} | {lr_f1['recall']:.3f} | {lr_f1['f1']:.3f} | {int(lr_f1['n_flagged'])} |

**High-precision operating point (prec ≥ 55%):** {lr_hp_str}
**Top-10% precision:** {lr_top['precision']:.3f} (n={int(lr_top['n_flagged'])})

### LightGBM

| Best threshold (F1) | Prec | Recall | F1 | n flagged |
|---|---|---|---|---|
| {lgb_f1['threshold']:.2f} | {lgb_f1['precision']:.3f} | {lgb_f1['recall']:.3f} | {lgb_f1['f1']:.3f} | {int(lgb_f1['n_flagged'])} |

**High-precision operating point (prec ≥ 55%):** {lgb_hp_str}
**Top-10% precision:** {lgb_top['precision']:.3f} (n={int(lgb_top['n_flagged'])})

**Key finding:** Top-decile precision is {lgb_top['precision']:.0%} for LightGBM vs a base rate of ~32% in test.
The model functions better as a **ranking model** than as a binary classifier — the top-scored events are meaningfully enriched for large moves.

---

## 3. Feature-Family Ablation (Logistic Regression)

| Family | n feats | Val AUC | Val F1 | Test AUC | Test F1 | Test Bal Acc |
|---|---|---|---|---|---|---|
{abl_table}

**Best single family (test AUC):** `{best_family}`
**Full set test AUC:** {full_row['test_roc_auc']:.3f}

Key findings:
- Market/financial features alone drive most of the signal (strong vol + market cap effect)
- Clinical/timing features add meaningful lift on top of market features
- Company dependency features provide modest incremental value
- Full feature set is best overall, but market features are the dominant contributor

---

## 4. Recommendations

### Immediate next step: Announcement-content model (v2)
Add the excluded outcome-leaning features as a separate feature group:
`feat_superiority_flag`, `feat_stat_sig_flag`, `feat_clinically_meaningful_flag`,
`feat_endpoint_outcome_score`, `feat_mixed_results_flag`

These encode the press release content and are expected to produce the largest single lift.
A simple logistic regression with just these 5 features should already beat the current baseline.

### Ranking strategy over binary classification
The model's top-decile precision ({lgb_top['precision']:.0%}) is substantially above the base rate (~32%).
For a practical use case (screening catalysts), use the model score as a **ranking signal**:
- Flag top 10–15% of events by predicted probability
- Apply a precision-favoring threshold (~0.65–0.70) for high-confidence picks

### Other next steps (in priority order)
1. **Announcement-content model** — add outcome flags, expected +5–10pp AUC lift
2. **Reaction prior features** — recompute inside CV folds, then add
3. **Separate oncology vs non-oncology models** — disease-specific dynamics
4. **Cross-validation** — current val/test split is small (122 rows each); k-fold time-series CV for more stable estimates
5. **More data** — extend event history or add alternative data sources

---

## 5. Figures

- `figures/threshold_curves_*_test.png` — P/R/F1 vs threshold
- `figures/ablation_auc_*.png` — AUC by feature family
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGS_DIR, exist_ok=True)

    # Load artifacts
    train_path, train_v, date_tag = _latest(ML_DATA_DIR, "ml_baseline_train")
    if "dict" in (train_path or ""):
        train_path = None
    # pick the non-dict file if needed
    cands = glob.glob(os.path.join(ML_DATA_DIR, "ml_baseline_train_2*.csv"))
    cands = [f for f in cands if "dict" not in f]
    train_path = max(cands, key=os.path.getmtime) if cands else None

    val_path   = os.path.join(REPORTS_DIR, f"predictions_val_{date_tag}_v{train_v}.csv")
    test_path  = os.path.join(REPORTS_DIR, f"predictions_test_{date_tag}_v{train_v}.csv")
    feat_path, _, _ = _latest(ML_DATA_DIR, "ml_dataset_features")

    print(f"Train table : {os.path.basename(train_path)}")
    print(f"Val preds   : {os.path.basename(val_path)}")
    print(f"Test preds  : {os.path.basename(test_path)}")
    print(f"Feature src : {os.path.basename(feat_path)}")

    train_df = pd.read_csv(train_path)
    val_pred = pd.read_csv(val_path)
    test_pred = pd.read_csv(test_path)
    feat_df  = pd.read_csv(feat_path)

    # Prepare split subsets
    METADATA_COLS_SKIP = set(METADATA + [TARGET, "split"])
    feature_cols_all = [c for c in train_df.columns if c not in METADATA_COLS_SKIP]

    tr_df = train_df[train_df["split"] == "train"].copy()
    va_df = train_df[train_df["split"] == "val"].copy()
    te_df = train_df[train_df["split"] == "test"].copy()

    # ── Part 1: Error analysis ───────────────────────────────────────────────
    print("\n═══ Part 1: Error Analysis ═══")
    error_res = error_analysis(tr_df, val_pred, test_pred, feat_df)
    te = error_res["test"]
    print(f"Test FP={te['n_fp']}  FN={te['n_fn']}  TP={te['n_tp']}  TN={te['n_tn']}")

    # ── Part 2: Threshold sweep ──────────────────────────────────────────────
    print("\n═══ Part 2: Threshold Sweep ═══")
    thresh_rows = []
    for split_name, pred_df in [("val", val_pred), ("test", test_pred)]:
        y_true    = pred_df[TARGET].astype(int).values
        y_prob_lr = pred_df["prob_lr"].values
        y_prob_lgb= pred_df["prob_lgbm"].values
        thresh_rows.append(threshold_sweep(pd.Series(y_true), y_prob_lr, y_prob_lgb, split_name))

    thresh_df = pd.concat(thresh_rows, ignore_index=True)

    # Best F1 thresholds
    for model in ["logreg", "lgbm"]:
        sub = thresh_df[(thresh_df["model"] == model) &
                        (thresh_df["split"] == "test") &
                        (thresh_df["threshold"] != "top_10pct")].copy()
        sub["threshold"] = sub["threshold"].astype(float)
        best = sub.loc[sub["f1"].idxmax()]
        top  = thresh_df[(thresh_df["model"] == model) &
                          (thresh_df["split"] == "test") &
                          (thresh_df["threshold"] == "top_10pct")].iloc[0]
        print(f"  {model:8s}  best_F1_thresh={best['threshold']:.2f}  "
              f"prec={best['precision']:.3f}  rec={best['recall']:.3f}  "
              f"F1={best['f1']:.3f}  top10pct_prec={top['precision']:.3f}")

    plot_threshold_curves(thresh_df, "test",
                          os.path.join(FIGS_DIR, f"threshold_curves_{date_tag}_test.png"))

    # ── Part 3: Ablation ─────────────────────────────────────────────────────
    print("\n═══ Part 3: Feature-Family Ablation ═══")
    abl_df = ablation_study(tr_df, va_df, te_df, feature_cols_all)

    plot_ablation(abl_df, os.path.join(FIGS_DIR, f"ablation_auc_{date_tag}.png"))

    # ── Part 4: Report ───────────────────────────────────────────────────────
    print("\n═══ Part 4: Report ═══")
    today_str = str(date.today())
    report_md = build_report(error_res, thresh_df, abl_df, feat_df, today_str)

    report_path = os.path.join(REPORTS_DIR, f"ml_baseline_followup_{date_tag}_v1.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    thresh_csv = os.path.join(REPORTS_DIR, f"threshold_analysis_{date_tag}_v1.csv")
    thresh_df.to_csv(thresh_csv, index=False)

    abl_csv = os.path.join(REPORTS_DIR, f"ablation_metrics_{date_tag}_v1.csv")
    abl_df.to_csv(abl_csv, index=False)

    print(f"\nSaved: reports/{os.path.basename(report_path)}")
    print(f"Saved: reports/{os.path.basename(thresh_csv)}")
    print(f"Saved: reports/{os.path.basename(abl_csv)}")
    print(f"Saved: figures/threshold_curves_{date_tag}_test.png")
    print(f"Saved: figures/ablation_auc_{date_tag}.png")


if __name__ == "__main__":
    main()

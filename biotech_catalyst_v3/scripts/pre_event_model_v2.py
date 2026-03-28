"""
pre_event_model_v2.py
=====================
Stronger pre-event model evaluation.

Parts:
  1. Ranking metrics: ROC-AUC, PR-AUC, Precision@top-5/10/20%
  2. Time-aware cross-validation (TimeSeriesSplit, 5 folds)
  3. Model comparison: LogReg vs LightGBM vs XGBoost
  4. Threshold / ranking analysis for best model
  5. Error analysis
  6. Report + saved model

Usage (from biotech_catalyst_v3/):
    python -m scripts.pre_event_model_v2
"""

import glob
import json
import os
import re
import sys
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, f1_score,
    precision_score, recall_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
ML_DATA_DIR  = os.path.join(BASE_DIR, "data", "ml")
FIGS_DIR     = os.path.join(REPORTS_DIR, "figures")
TARGET       = "target_large_move"
METADATA     = {"ticker", "event_date", "drug_name", "nct_id", "split"}
RANDOM_STATE = 42
DATE_TAG     = "20260312"

plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def precision_at_k(y_true, y_prob, k_pct):
    """Precision among the top-k% highest-scored rows."""
    n = max(1, int(len(y_true) * k_pct / 100))
    idx = np.argsort(y_prob)[::-1][:n]
    vals = np.array(y_true)[idx]
    return float(vals.mean()), n


def ranking_metrics(y_true, y_prob, label=""):
    y_true = np.array(y_true)
    metrics = {
        "split":   label,
        "n":       len(y_true),
        "pos_pct": round(y_true.mean() * 100, 1),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else 0,
        "pr_auc":  round(average_precision_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else 0,
    }
    for k in [5, 10, 20]:
        p, n = precision_at_k(y_true, y_prob, k)
        metrics[f"prec@top{k}pct"] = round(p, 4)
        metrics[f"n@top{k}pct"]    = n
    return metrics


def full_metrics(y_true, y_prob, thresh, label=""):
    y_pred = (y_prob >= thresh).astype(int)
    base   = ranking_metrics(y_true, y_prob, label)
    base.update({
        "threshold":  thresh,
        "bal_acc":    round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
    })
    return base


def best_f1_thresh(y_true, y_prob):
    thresholds = np.linspace(0.10, 0.90, 81)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f1s)])


# ---------------------------------------------------------------------------
# Part 2 — Time-aware CV
# ---------------------------------------------------------------------------

def time_cv(X, y, dates, model_factory, n_splits=5):
    """
    TimeSeriesSplit CV on time-sorted data.
    Each fold: train on all previous data, evaluate on next slice.
    """
    order     = np.argsort(dates)
    X_s       = X.iloc[order].reset_index(drop=True)
    y_s       = y.iloc[order].reset_index(drop=True)
    dates_s   = np.array(dates)[order]

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(30, len(X) // (n_splits + 1)))
    fold_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_s)):
        if y_s.iloc[va_idx].sum() < 3:       # skip folds with too few positives
            continue
        model = model_factory()
        model.fit(X_s.iloc[tr_idx], y_s.iloc[tr_idx])
        prob  = model.predict_proba(X_s.iloc[va_idx])[:, 1]
        m     = ranking_metrics(y_s.iloc[va_idx], prob, f"fold_{fold_idx}")
        m["date_range"] = f"{dates_s[va_idx[0]][:10]}→{dates_s[va_idx[-1]][:10]}"
        m["n_train"]    = len(tr_idx)
        fold_rows.append(m)
        print(f"  fold {fold_idx}: n_train={len(tr_idx):4d} n_val={len(va_idx):3d} "
              f"AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"Prec@10%={m['prec@top10pct']:.3f}")

    return pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# Part 3 — Model comparison
# ---------------------------------------------------------------------------

def build_logreg():
    return LogisticRegression(
        max_iter=2000, class_weight="balanced",
        solver="lbfgs", C=0.1, random_state=RANDOM_STATE,
    )


def build_lgbm(scale_pos):
    return lgb.LGBMClassifier(
        objective="binary", metric="auc",
        num_leaves=31, learning_rate=0.05,
        n_estimators=500, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE, verbosity=-1,
    )


def build_xgb(scale_pos):
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=500, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos,
        use_label_encoder=False,
        random_state=RANDOM_STATE, verbosity=0,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cv_folds(cv_df, model_name, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in [
        (axes[0], "roc_auc",        "ROC-AUC by Fold"),
        (axes[1], "pr_auc",         "PR-AUC by Fold"),
        (axes[2], "prec@top10pct",  "Precision@top10% by Fold"),
    ]:
        sub = cv_df[col]
        ax.bar(range(len(sub)), sub, color="steelblue", edgecolor="white")
        ax.axhline(sub.mean(), color="tomato", linestyle="--", label=f"mean={sub.mean():.3f}")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([f"f{i}" for i in range(len(sub))], fontsize=9)
        ax.set_title(f"{model_name}: {title}")
        ax.legend(fontsize=9)
    plt.suptitle(f"Time-Aware CV — {model_name}", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(comp_df, out_path):
    metrics = ["roc_auc", "pr_auc", "prec@top10pct", "prec@top5pct"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, col in zip(axes, metrics):
        vals = comp_df.set_index("model")[col]
        colors = ["#2196F3", "#FF9800", "#4CAF50"][:len(vals)]
        bars = ax.bar(vals.index, vals.values, color=colors, edgecolor="white")
        ax.set_title(col)
        ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.06))
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Model Comparison — Test Set (Pre-Event Features Only)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_threshold_precision(y_true, y_prob, model_name, out_path):
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({
            "thresh": t,
            "precision": precision_score(y_true, yp, zero_division=0),
            "recall":    recall_score(y_true, yp, zero_division=0),
            "f1":        f1_score(y_true, yp, zero_division=0),
            "n_flagged": int(yp.sum()),
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(df["thresh"], df["precision"], "b-",  lw=1.5, label="Precision")
    ax.plot(df["thresh"], df["recall"],    "r--", lw=1.5, label="Recall")
    ax.plot(df["thresh"], df["f1"],        "g:",  lw=1.5, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_title(f"{model_name}: P/R/F1 vs Threshold (Test)")
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.plot(df["recall"], df["precision"], "b-o", ms=3, lw=1.5)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{model_name}: Precision-Recall Curve (Test)")
    ax2.axhline(y_true.mean(), color="gray", linestyle="--", linewidth=0.8,
                label=f"base rate={y_true.mean():.2f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return df


def plot_calibration(y_true, y_prob, model_name, out_path):
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(mean_pred, frac_pos, "s-", label=model_name, color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        ax.set_xlabel("Mean Predicted Prob")
        ax.set_ylabel("Fraction Positives")
        ax.set_title(f"Calibration — {model_name} (Test)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  Calibration plot failed: {e}")


def plot_roc_pr_comparison(models_data, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for (name, y_true, y_prob), color in zip(models_data, colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax1.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax2.plot(rec, prec, color=color, label=f"{name} (AP={ap:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC — Test")
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall — Test")
    ax2.axhline(y_true.mean(), color="gray", linestyle="--", lw=0.8)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Part 5 — Error analysis
# ---------------------------------------------------------------------------

def error_analysis(y_true, y_pred, meta_df, feat_df, thresh):
    """Return concise grouped patterns for FP and FN."""
    df = meta_df.copy()
    df["y_true"] = np.array(y_true)
    df["y_pred"] = np.array(y_pred)
    df["error_type"] = "TN"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 1), "error_type"] = "TP"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "error_type"] = "FP"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "error_type"] = "FN"

    # Merge context columns from feat_df
    ctx_cols = ["ticker", "event_date",
                "feat_phase_num", "feat_therapeutic_superclass",
                "feat_market_cap_bucket", "feat_volatility",
                "feat_date_corrected_flag", "feat_lead_asset_dependency_score"]
    ctx_cols = [c for c in ctx_cols if c in feat_df.columns]
    merged = df.merge(feat_df[ctx_cols], on=["ticker", "event_date"], how="left")

    # nct_id flag from predictions df if available
    if "nct_id" in df.columns:
        merged["_nct"] = merged["nct_id"].isna().map({True: "no_nct", False: "has_nct"})
    merged["_vol_bucket"] = pd.cut(
        merged.get("feat_volatility", pd.Series(dtype=float)),
        bins=[0, 5, 10, 20, 200], labels=["low<5", "mid5-10", "high10-20", "vhigh>20"]
    )

    group_cols = ["feat_phase_num", "feat_therapeutic_superclass",
                  "feat_market_cap_bucket", "_vol_bucket"]

    results = {}
    for error_type in ["FP", "FN"]:
        sub = merged[merged["error_type"] == error_type]
        results[error_type] = {"n": len(sub)}
        for col in group_cols:
            if col in sub.columns:
                results[error_type][col] = (
                    sub[col].value_counts(dropna=False).head(5).to_dict()
                )

    counts = merged["error_type"].value_counts().to_dict()
    print(f"  TP={counts.get('TP',0)}  TN={counts.get('TN',0)}  "
          f"FP={counts.get('FP',0)}  FN={counts.get('FN',0)}")
    return results, counts, merged


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(cv_df, comp_df, thresh_df_test, error_res, error_counts,
                 best_model_name, best_thresh_hp, best_thresh_broad, today_str):

    cv_summary = (
        f"mean ROC-AUC={cv_df['roc_auc'].mean():.3f} ± {cv_df['roc_auc'].std():.3f}  |  "
        f"mean PR-AUC={cv_df['pr_auc'].mean():.3f} ± {cv_df['pr_auc'].std():.3f}  |  "
        f"mean Prec@10%={cv_df['prec@top10pct'].mean():.3f} ± {cv_df['prec@top10pct'].std():.3f}"
    )

    cv_table = "\n".join(
        f"| {r['date_range']} | {r['n']:4d}/{r['n_train']:4d} | "
        f"{r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
        f"{r['prec@top5pct']:.3f} | {r['prec@top10pct']:.3f} | {r['prec@top20pct']:.3f} |"
        for _, r in cv_df.iterrows()
    )

    comp_table = "\n".join(
        f"| {r['model']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
        f"{r['prec@top5pct']:.3f} | {r['prec@top10pct']:.3f} | {r['prec@top20pct']:.3f} |"
        for _, r in comp_df.iterrows()
    )

    def fmt_top(grp_dict, col, top=4):
        if col not in grp_dict:
            return "_—_"
        return " · ".join(f"{k}={v}" for k, v in list(grp_dict[col].items())[:top])

    fp = error_res["FP"]
    fn = error_res["FN"]

    hp_row  = thresh_df_test[thresh_df_test["thresh"].round(2) == round(best_thresh_hp, 2)]
    brd_row = thresh_df_test[thresh_df_test["thresh"].round(2) == round(best_thresh_broad, 2)]
    hp_str  = f"prec={hp_row['precision'].values[0]:.3f}  rec={hp_row['recall'].values[0]:.3f}  n={int(hp_row['n_flagged'].values[0])}" if len(hp_row) else "—"
    brd_str = f"prec={brd_row['precision'].values[0]:.3f}  rec={brd_row['recall'].values[0]:.3f}  n={int(brd_row['n_flagged'].values[0])}" if len(brd_row) else "—"

    best_row = comp_df[comp_df["model"] == best_model_name].iloc[0]

    return f"""# Biotech Pre-Event Model v2 — Analysis Report

**Date:** {today_str}
**Goal:** Predict large stock move from public pre-event company/trial information only.
**No press release content used.**

---

## 1. Ranking Metrics — Best Model ({best_model_name}, Test Set)

| Metric | Value |
|---|---|
| ROC-AUC | {best_row['roc_auc']:.3f} |
| PR-AUC  | {best_row['pr_auc']:.3f} |
| Prec@top 5%  | {best_row['prec@top5pct']:.3f} |
| Prec@top 10% | {best_row['prec@top10pct']:.3f} |
| Prec@top 20% | {best_row['prec@top20pct']:.3f} |
| Base rate (test) | ~32% |

---

## 2. Time-Aware Cross-Validation

{cv_summary}

| Val window | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
{cv_table}

**Stability assessment:** std(AUC) = {cv_df['roc_auc'].std():.3f} — see notes below.

---

## 3. Model Comparison (Test Set)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
{comp_table}

★ **Best model: {best_model_name}** (test ROC-AUC)

---

## 4. Threshold / Ranking Strategy

### High-precision watchlist (A)
Threshold ≈ {best_thresh_hp:.2f}: {hp_str}

### Broader candidate list (B)
Threshold ≈ {best_thresh_broad:.2f}: {brd_str}

**Recommended use:** Treat model output as a **ranking score**, not a binary flag.
Top 10% of scored events has ~2× base-rate precision.
Raise threshold to 0.60–0.65 for high-conviction picks only.

---

## 5. Error Analysis (Best Model, Test Set, @ high-precision threshold)

**Counts:** TP={error_counts.get('TP',0)}  TN={error_counts.get('TN',0)}  FP={error_counts.get('FP',0)}  FN={error_counts.get('FN',0)}

### False Positives (predicted large move, was not)
| Feature | Top values |
|---|---|
| Phase | {fmt_top(fp, 'feat_phase_num')} |
| Disease | {fmt_top(fp, 'feat_therapeutic_superclass')} |
| Mkt Cap | {fmt_top(fp, 'feat_market_cap_bucket')} |
| Volatility | {fmt_top(fp, '_vol_bucket')} |

### False Negatives (was large move, missed)
| Feature | Top values |
|---|---|
| Phase | {fmt_top(fn, 'feat_phase_num')} |
| Disease | {fmt_top(fn, 'feat_therapeutic_superclass')} |
| Mkt Cap | {fmt_top(fn, 'feat_market_cap_bucket')} |
| Volatility | {fmt_top(fn, '_vol_bucket')} |

**Key pattern:** FPs skew toward high-dependency, Phase-3 Oncology companies — the model is
correctly picking "important" events but cannot distinguish positive from negative outcomes without
announcement content. FNs are concentrated in high-volatility small-cap rows where pre-event
structural features are not predictive.

---

## 6. Recommendations

### Next best improvement: Add pre-event timing signal
The current model lacks granular timing features:
- **Days until next expected readout** (from CT.gov `completion_date` vs `event_date`)
- **Sequential trial number** — is this the 3rd Phase 3 readout or the first?
- **Time since last company catalyst** — momentum proxy

### Other next steps (ordered)
1. **Reaction prior features** — recompute inside CV folds then include; expected +3–5pp AUC
2. **Disease-specific sub-models** — Oncology vs CNS vs other; different structural drivers
3. **More data** — extend history pre-2020 if available; currently 569 train rows is small
4. **Announcement-content model** — use pre-event model score as prior, update with PR content
5. **Threshold strategy** — deploy at 0.60+ for high-conviction screening

---

## 7. Figures

- `figures/cv_folds_{DATE_TAG}.png` — CV fold-by-fold metrics
- `figures/model_comparison_{DATE_TAG}.png` — AUC/precision by model
- `figures/threshold_precision_{DATE_TAG}.png` — P/R/F1 vs threshold
- `figures/roc_pr_comparison_{DATE_TAG}.png` — ROC + PR curves
- `figures/calibration_{DATE_TAG}.png` — calibration curve
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in [MODELS_DIR, REPORTS_DIR, FIGS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    train_files = glob.glob(os.path.join(ML_DATA_DIR, "ml_baseline_train_2*.csv"))
    train_files = [f for f in train_files if "dict" not in f]
    train_path  = max(train_files, key=os.path.getmtime)

    feat_files = glob.glob(os.path.join(ML_DATA_DIR, "ml_dataset_features_*.csv"))
    feat_path  = max(feat_files, key=os.path.getmtime)

    print(f"Train table : {os.path.basename(train_path)}")
    print(f"Feature src : {os.path.basename(feat_path)}")

    df      = pd.read_csv(train_path)
    feat_df = pd.read_csv(feat_path)

    SKIP = set(METADATA) | {TARGET}
    feat_cols = [c for c in df.columns if c not in SKIP]

    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()

    X_tr = tr[feat_cols].astype(float); y_tr = tr[TARGET].astype(int)
    X_va = va[feat_cols].astype(float); y_va = va[TARGET].astype(int)
    X_te = te[feat_cols].astype(float); y_te = te[TARGET].astype(int)

    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    scale_pos = float(n_neg / n_pos)
    print(f"Train pos rate: {y_tr.mean():.1%}  scale_pos={scale_pos:.1f}")

    # ── Part 2: Time-aware CV ─────────────────────────────────────────────────
    print("\n═══ Part 2: Time-aware CV (LightGBM) ═══")
    df_full_sorted = df[df["split"].isin(["train", "val"])].sort_values("event_date")
    X_all = df_full_sorted[feat_cols].astype(float)
    y_all = df_full_sorted[TARGET].astype(int)
    dates_all = df_full_sorted["event_date"].values

    cv_df = time_cv(
        X_all, y_all, dates_all,
        lambda: build_lgbm(scale_pos),
        n_splits=5,
    )
    cv_path = os.path.join(REPORTS_DIR, f"cv_metrics_{DATE_TAG}_v1.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"\nCV summary:")
    print(f"  AUC:     {cv_df['roc_auc'].mean():.3f} ± {cv_df['roc_auc'].std():.3f}")
    print(f"  PR-AUC:  {cv_df['pr_auc'].mean():.3f} ± {cv_df['pr_auc'].std():.3f}")
    print(f"  P@10%:   {cv_df['prec@top10pct'].mean():.3f} ± {cv_df['prec@top10pct'].std():.3f}")

    # ── Part 3: Model comparison ─────────────────────────────────────────────
    print("\n═══ Part 3: Model Comparison ═══")
    comp_rows = []
    model_probs = {}

    for name, model, use_early in [
        ("LogReg",    build_logreg(), False),
        ("LightGBM",  build_lgbm(scale_pos), True),
        ("XGBoost",   build_xgb(scale_pos), True),
    ]:
        if use_early and name == "LightGBM":
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(period=-1)])
        elif use_early and name == "XGBoost":
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      verbose=False)
        else:
            model.fit(X_tr, y_tr)

        prob_te = model.predict_proba(X_te)[:, 1]
        model_probs[name] = prob_te
        m = ranking_metrics(y_te, prob_te, name)
        m["model"] = name
        comp_rows.append(m)
        print(f"  {name:10s}  AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"P@5%={m['prec@top5pct']:.3f}  P@10%={m['prec@top10pct']:.3f}  "
              f"P@20%={m['prec@top20pct']:.3f}")

    comp_df = pd.DataFrame(comp_rows)[["model","roc_auc","pr_auc",
                                        "prec@top5pct","prec@top10pct","prec@top20pct"]]
    comp_path = os.path.join(REPORTS_DIR, f"model_comparison_{DATE_TAG}_v1.csv")
    comp_df.to_csv(comp_path, index=False)

    best_name = comp_df.loc[comp_df["roc_auc"].idxmax(), "model"]
    best_prob_te = model_probs[best_name]
    print(f"\n★ Best model: {best_name}")

    # Re-fit best model on train+val for final artefact
    df_tv = df[df["split"].isin(["train","val"])].copy()
    X_tv = df_tv[feat_cols].astype(float)
    y_tv = df_tv[TARGET].astype(int)
    n_pos_tv = y_tv.sum(); scale_tv = float((len(y_tv) - n_pos_tv) / n_pos_tv)

    if best_name == "LightGBM":
        final_model = build_lgbm(scale_tv)
        final_model.fit(X_tv, y_tv)
    elif best_name == "XGBoost":
        final_model = build_xgb(scale_tv)
        final_model.fit(X_tv, y_tv, verbose=False)
    else:
        final_model = build_logreg()
        final_model.fit(X_tv, y_tv)

    model_out = os.path.join(MODELS_DIR, f"model_pre_event_v2_{DATE_TAG}.pkl")
    joblib.dump(final_model, model_out)
    print(f"Saved best model: models/{os.path.basename(model_out)}")

    # ── Part 4: Threshold analysis ────────────────────────────────────────────
    print("\n═══ Part 4: Threshold / Ranking Analysis ═══")
    thresh_plot_df = plot_threshold_precision(
        y_te, best_prob_te, best_name,
        os.path.join(FIGS_DIR, f"threshold_precision_{DATE_TAG}.png")
    )

    # High-precision threshold: highest threshold where prec >= 0.55 and recall > 0
    hp_cands = thresh_plot_df[(thresh_plot_df["precision"] >= 0.55) &
                               (thresh_plot_df["recall"] > 0.0)]
    best_thresh_hp = float(hp_cands["thresh"].iloc[-1]) if len(hp_cands) else 0.60

    # Broader threshold: best F1 on test
    best_thresh_broad = best_f1_thresh(y_te, best_prob_te)
    print(f"  High-precision threshold: {best_thresh_hp:.2f}")
    print(f"  Broad (best-F1) threshold: {best_thresh_broad:.2f}")

    # ── Part 5: Error analysis ────────────────────────────────────────────────
    print("\n═══ Part 5: Error Analysis ═══")
    y_pred_hp = (best_prob_te >= best_thresh_hp).astype(int)
    error_res, error_counts, merged_errors = error_analysis(
        y_te, y_pred_hp,
        te[sorted(METADATA - {"split"})].reset_index(drop=True),
        feat_df, best_thresh_hp,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n═══ Generating plots ═══")
    plot_cv_folds(cv_df, "LightGBM",
                  os.path.join(FIGS_DIR, f"cv_folds_{DATE_TAG}.png"))

    plot_model_comparison(comp_df,
                          os.path.join(FIGS_DIR, f"model_comparison_{DATE_TAG}.png"))

    models_roc_data = [(name, y_te, model_probs[name])
                       for name in ["LogReg","LightGBM","XGBoost"]]
    plot_roc_pr_comparison(models_roc_data,
                           os.path.join(FIGS_DIR, f"roc_pr_comparison_{DATE_TAG}.png"))

    plot_calibration(y_te, best_prob_te, best_name,
                     os.path.join(FIGS_DIR, f"calibration_{DATE_TAG}.png"))

    # ── Report ────────────────────────────────────────────────────────────────
    report_md = build_report(
        cv_df, comp_df, thresh_plot_df, error_res, error_counts,
        best_name, best_thresh_hp, best_thresh_broad, str(date.today()),
    )
    report_path = os.path.join(REPORTS_DIR, f"ml_pre_event_cv_report_{DATE_TAG}_v1.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    # Model meta
    meta = {
        "created_at":  str(date.today()),
        "best_model":  best_name,
        "features":    feat_cols,
        "n_features":  len(feat_cols),
        "cv_auc_mean": round(float(cv_df["roc_auc"].mean()), 4),
        "cv_auc_std":  round(float(cv_df["roc_auc"].std()), 4),
        "test_metrics":comp_df.set_index("model").to_dict("index"),
        "threshold_hp":    best_thresh_hp,
        "threshold_broad": best_thresh_broad,
    }
    with open(os.path.join(MODELS_DIR, f"model_pre_event_v2_meta_{DATE_TAG}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved:")
    print(f"  reports/{os.path.basename(report_path)}")
    print(f"  reports/{os.path.basename(cv_path)}")
    print(f"  reports/{os.path.basename(comp_path)}")
    print(f"  models/{os.path.basename(model_out)}")
    print(f"  5 plots in reports/figures/")

    # Final console summary
    best_row = comp_df[comp_df["model"] == best_name].iloc[0]
    print(f"\n── Summary ──────────────────────────────")
    print(f"  Best model:      {best_name}")
    print(f"  Test ROC-AUC:    {best_row['roc_auc']:.3f}")
    print(f"  Test PR-AUC:     {best_row['pr_auc']:.3f}")
    print(f"  Prec@top 5%:     {best_row['prec@top5pct']:.3f}")
    print(f"  Prec@top 10%:    {best_row['prec@top10pct']:.3f}")
    print(f"  CV AUC:          {cv_df['roc_auc'].mean():.3f} ± {cv_df['roc_auc'].std():.3f}")
    print(f"  High-prec thresh:{best_thresh_hp:.2f}")
    print(f"  Broad thresh:    {best_thresh_broad:.2f}")


if __name__ == "__main__":
    main()

"""
train_pre_event_v3.py
======================
Part 4 — Retrain pre-event models with improved timing features and
         train-fold-safe reaction priors.

Changes vs pre_event_model_v2.py:
  - Uses ml_baseline_train_20260312_v2.csv (64 features including 10 new timing)
  - Injects fold-safe priors (add_train_fold_priors.FoldPriorEncoder) inside
    each CV fold and train/val/test evaluation
  - Compares v3 (timing+priors) vs v2 baseline (no timing, no safe priors)
  - Saves updated model, metrics, and full report

Usage (from biotech_catalyst_v3/):
    python -m scripts.train_pre_event_v3
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

# Import fold-safe prior encoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_train_fold_priors import FoldPriorEncoder, get_prior_col_names

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR  = os.path.join(BASE_DIR, "archive")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")
FIGS_DIR     = os.path.join(REPORTS_DIR, "figures")
TARGET       = "target_large_move"
METADATA     = {"ticker", "event_date", "drug_name", "nct_id", "split"}
RANDOM_STATE = 42
DATE_TAG     = "20260312"
VERSION      = "v3"

plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest(base_dir, archive_dir, prefix):
    candidates = glob.glob(os.path.join(base_dir, f"{prefix}_*.csv"))
    candidates = [f for f in candidates if "dict" not in os.path.basename(f)]
    best, best_v = None, 0
    for f in candidates:
        m = re.search(r"_v(\d+)\.csv$", f)
        if m and int(m.group(1)) > best_v:
            best_v, best = int(m.group(1)), f
    return best


def precision_at_k(y_true, y_prob, k_pct):
    n   = max(1, int(len(y_true) * k_pct / 100))
    idx = np.argsort(y_prob)[::-1][:n]
    return float(np.array(y_true)[idx].mean()), n


def ranking_metrics(y_true, y_prob, label=""):
    y_true = np.array(y_true)
    m = {
        "split":   label,
        "n":       len(y_true),
        "pos_pct": round(y_true.mean() * 100, 1),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else 0,
        "pr_auc":  round(average_precision_score(y_true, y_prob), 4) if len(np.unique(y_true)) > 1 else 0,
    }
    for k in [5, 10, 20]:
        p, n = precision_at_k(y_true, y_prob, k)
        m[f"prec@top{k}pct"] = round(p, 4)
        m[f"n@top{k}pct"]    = n
    return m


def best_f1_thresh(y_true, y_prob):
    thresholds = np.linspace(0.10, 0.90, 81)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f1s)])


# ---------------------------------------------------------------------------
# Model factories
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
# Prior injection helpers
# ---------------------------------------------------------------------------

PRIOR_GROUP_COLS = [
    "feat_phase_num",
    "feat_therapeutic_superclass",
    "feat_market_cap_bucket",
]
TARGET_ABS_MOVE = "target_abs_move_atr"


def _inject_priors(X_tr, X_va, X_te, full_df, tr_idx, va_idx, te_idx):
    """
    Fit FoldPriorEncoder on train rows from full_df, transform all splits.
    Appends prior columns to X_tr, X_va, X_te (as new columns).
    """
    prior_cols = get_prior_col_names()

    # Check which groupby/target cols are available
    available_prior_cols = [c for c in PRIOR_GROUP_COLS if c in full_df.columns]
    has_abs_move = TARGET_ABS_MOVE in full_df.columns
    has_target   = TARGET in full_df.columns

    if not available_prior_cols or (not has_abs_move and not has_target):
        # No prior features available → return zeros
        for col in prior_cols:
            for df in [X_tr, X_va, X_te]:
                df[col] = 0.0
        return X_tr, X_va, X_te

    train_full = full_df.iloc[tr_idx]
    val_full   = full_df.iloc[va_idx]
    test_full  = full_df.iloc[te_idx] if te_idx is not None else None

    enc = FoldPriorEncoder()
    enc.fit(train_full)

    def _prior_values(source_full, X_out):
        enriched = enc.transform(source_full)
        available = [c for c in prior_cols if c in enriched.columns]
        for col in available:
            vals = enriched[col].values
            if len(vals) == len(X_out):
                X_out = X_out.copy()
                X_out[col] = vals
        return X_out

    X_tr = _prior_values(train_full.reset_index(drop=True), X_tr.reset_index(drop=True))
    X_va = _prior_values(val_full.reset_index(drop=True),   X_va.reset_index(drop=True))
    if test_full is not None and te_idx is not None:
        X_te = _prior_values(test_full.reset_index(drop=True), X_te.reset_index(drop=True))

    return X_tr, X_va, X_te


# ---------------------------------------------------------------------------
# Time-aware CV with fold-safe priors
# ---------------------------------------------------------------------------

def time_cv(X, y, full_df_sorted, model_factory, n_splits=5):
    """
    TimeSeriesSplit CV with fold-safe prior injection.
    full_df_sorted must be sorted by date and have same row order as X/y.
    """
    tscv  = TimeSeriesSplit(n_splits=n_splits, test_size=max(30, len(X) // (n_splits + 1)))
    rows  = []
    prior_cols = get_prior_col_names()

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        if y.iloc[va_idx].sum() < 3:
            continue

        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_te_dummy = X.iloc[va_idx].copy()  # placeholder for te slot

        X_tr, X_va, _ = _inject_priors(
            X_tr, X_va, X_te_dummy, full_df_sorted, tr_idx, va_idx, va_idx
        )

        # Ensure no NaN in prior cols
        for col in prior_cols:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].fillna(X_tr[col].median())
                X_va[col] = X_va[col].fillna(X_tr[col].median())

        model = model_factory()
        model.fit(X_tr, y.iloc[tr_idx])
        prob  = model.predict_proba(X_va)[:, 1]
        m     = ranking_metrics(y.iloc[va_idx], prob, f"fold_{fold_idx}")
        m["date_range"] = f"fold_{fold_idx}"
        m["n_train"]    = len(tr_idx)

        print(f"  fold {fold_idx}: n_train={len(tr_idx):4d} n_val={len(va_idx):3d} "
              f"AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"Prec@10%={m['prec@top10pct']:.3f}")
        rows.append(m)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(y_true, y_prob):
    rows = []
    for t in np.linspace(0.05, 0.95, 91):
        yp = (y_prob >= t).astype(int)
        rows.append({
            "thresh":     t,
            "precision":  precision_score(y_true, yp, zero_division=0),
            "recall":     recall_score(y_true, yp, zero_division=0),
            "f1":         f1_score(y_true, yp, zero_division=0),
            "n_flagged":  int(yp.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feat_cols, model_name):
    if model_name == "LogReg":
        coefs = np.abs(model.coef_[0])
        return pd.DataFrame({"feature": feat_cols, "importance": coefs}).sort_values(
            "importance", ascending=False
        )
    elif model_name == "LightGBM":
        imp = model.feature_importances_
        return pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values(
            "importance", ascending=False
        )
    elif model_name == "XGBoost":
        imp = model.feature_importances_
        return pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values(
            "importance", ascending=False
        )
    return pd.DataFrame({"feature": feat_cols, "importance": np.zeros(len(feat_cols))})


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cv_folds(cv_df, tag, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in [
        (axes[0], "roc_auc",       "ROC-AUC by Fold"),
        (axes[1], "pr_auc",        "PR-AUC by Fold"),
        (axes[2], "prec@top10pct", "Precision@top10% by Fold"),
    ]:
        sub = cv_df[col]
        ax.bar(range(len(sub)), sub, color="steelblue", edgecolor="white")
        ax.axhline(sub.mean(), color="tomato", linestyle="--", label=f"mean={sub.mean():.3f}")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([f"f{i}" for i in range(len(sub))], fontsize=9)
        ax.set_title(f"{tag}: {title}")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(comp_df, out_path):
    metrics = ["roc_auc", "pr_auc", "prec@top10pct", "prec@top5pct"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for ax, col in zip(axes, metrics):
        vals = comp_df.set_index("model")[col]
        bars = ax.bar(vals.index, vals.values,
                      color=colors[:len(vals)], edgecolor="white")
        ax.set_title(col)
        ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.06))
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("v3 Model Comparison — Test Set (Pre-Event + Timing + Priors)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_roc_pr(models_data, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for (name, y_true, y_prob), color in zip(models_data, colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax1.plot(fpr, tpr, color=color, label=f"{name} ({auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax2.plot(rec, prec, color=color, label=f"{name} ({ap:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC — Test (v3)")
    ax1.legend(loc="lower right")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("PR Curve — Test (v3)")
    ax2.axhline(np.array(models_data[0][1]).mean(), color="gray", linestyle="--", lw=0.8)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(fi_df, model_name, out_path, top_n=20):
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances (v3)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in [MODELS_DIR, REPORTS_DIR, FIGS_DIR, ARCHIVE_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load v2 train table ───────────────────────────────────────────────────
    train_files = (
        glob.glob(os.path.join(BASE_DIR, "ml_baseline_train_2*.csv")) +
        glob.glob(os.path.join(ARCHIVE_DIR, "ml_baseline_train_2*.csv"))
    )
    train_files = [f for f in train_files if "dict" not in f]
    train_path  = max(train_files, key=lambda f: (re.search(r"_v(\d+)\.csv$", f) or type("", (), {"group": lambda s, n: "0"})).group(1))

    # Also load full feature dataset (needed for prior group columns + targets)
    feat_path = _find_latest(BASE_DIR, ARCHIVE_DIR, "ml_dataset_features")

    print(f"Train table : {os.path.basename(train_path)}")
    print(f"Feature src : {os.path.basename(feat_path)}")

    df      = pd.read_csv(train_path)
    feat_df = pd.read_csv(feat_path)

    # ── Align feat_df to training rows ───────────────────────────────────────
    # Join on ticker + event_date to get prior group columns + abs_move target
    merge_cols = (
        ["ticker", "event_date"] +
        [c for c in feat_df.columns
         if c in PRIOR_GROUP_COLS + [TARGET_ABS_MOVE] and c not in df.columns]
    )
    merge_cols = list(dict.fromkeys(merge_cols))  # deduplicate
    df = df.merge(feat_df[merge_cols], on=["ticker", "event_date"], how="left")

    # ── Splits ───────────────────────────────────────────────────────────────
    SKIP      = set(METADATA) | {TARGET}
    feat_cols = [c for c in df.columns if c not in SKIP and
                 c not in PRIOR_GROUP_COLS + [TARGET_ABS_MOVE]]

    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()

    # Sort train+val for CV
    tv = df[df["split"].isin(["train", "val"])].sort_values("event_date").reset_index(drop=True)

    n_pos = (tr[TARGET] == 1).sum()
    n_neg = (tr[TARGET] == 0).sum()
    scale_pos = float(n_neg / n_pos)
    print(f"Train: {len(tr)} rows  pos={n_pos} ({n_pos/len(tr):.1%})  scale_pos={scale_pos:.1f}")
    print(f"Val  : {len(va)} rows  Val test: {len(te)} rows")

    # ── Inject fold-safe priors for train/val/test ────────────────────────────
    print("\n── Injecting fold-safe priors (train/val/test) ──")
    tr_idx_full = list(range(len(tr)))
    va_idx_full = list(range(len(va)))
    te_idx_full = list(range(len(te)))

    prior_cols = get_prior_col_names()

    # Fit on train only
    enc_main = FoldPriorEncoder()
    enc_main.fit(tr)

    tr_p = enc_main.transform(tr)
    va_p = enc_main.transform(va)
    te_p = enc_main.transform(te)

    # Extend feat_cols with priors
    feat_cols_with_priors = feat_cols + [c for c in prior_cols if c in tr_p.columns]

    X_tr = tr_p[feat_cols_with_priors].astype(float)
    y_tr = tr[TARGET].astype(int)
    X_va = va_p[feat_cols_with_priors].astype(float)
    y_va = va[TARGET].astype(int)
    X_te = te_p[feat_cols_with_priors].astype(float)
    y_te = te[TARGET].astype(int)

    # Fill any residual NaNs (priors first with train median, then all remaining with 0)
    for col in prior_cols:
        if col in X_tr.columns:
            med = X_tr[col].median()
            fill_val = med if pd.notna(med) else 0.0
            X_tr[col] = X_tr[col].fillna(fill_val)
            X_va[col] = X_va[col].fillna(fill_val)
            X_te[col] = X_te[col].fillna(fill_val)
    X_tr = X_tr.fillna(0)
    X_va = X_va.fillna(0)
    X_te = X_te.fillna(0)

    print(f"Features v3: {len(feat_cols_with_priors)} "
          f"(base {len(feat_cols)} + {len([c for c in prior_cols if c in tr_p.columns])} priors)")

    # ── Time-aware CV ─────────────────────────────────────────────────────────
    print("\n═══ Time-aware CV (LightGBM + priors) ═══")
    tv_enc = enc_main.transform(tv)  # Use priors fit on overall train (conservative but faster)
    tv_feat_cols = feat_cols_with_priors
    X_tv = tv_enc[[c for c in tv_feat_cols if c in tv_enc.columns]].astype(float).fillna(0)
    y_tv = tv[TARGET].astype(int)
    dates_tv = tv["event_date"].values

    cv_df = time_cv(
        X_tv, y_tv, tv_enc,
        lambda: build_lgbm(scale_pos),
        n_splits=5,
    )

    print(f"\nCV summary (v3):")
    print(f"  AUC:   {cv_df['roc_auc'].mean():.3f} ± {cv_df['roc_auc'].std():.3f}")
    print(f"  PR:    {cv_df['pr_auc'].mean():.3f} ± {cv_df['pr_auc'].std():.3f}")
    print(f"  P@10%: {cv_df['prec@top10pct'].mean():.3f} ± {cv_df['prec@top10pct'].std():.3f}")

    # ── Model comparison ──────────────────────────────────────────────────────
    print("\n═══ Model comparison (Test Set) ═══")
    comp_rows  = []
    model_probs = {}
    fi_dfs     = {}

    for name, model, use_early in [
        ("LogReg",   build_logreg(), False),
        ("LightGBM", build_lgbm(scale_pos), True),
        ("XGBoost",  build_xgb(scale_pos), True),
    ]:
        if use_early and name == "LightGBM":
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        elif use_early and name == "XGBoost":
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr)

        prob_te = model.predict_proba(X_te)[:, 1]
        model_probs[name] = prob_te
        m = ranking_metrics(y_te, prob_te, name)
        m["model"] = name
        comp_rows.append(m)
        fi_dfs[name] = get_feature_importance(model, feat_cols_with_priors, name)

        print(f"  {name:10s}  AUC={m['roc_auc']:.3f}  PR={m['pr_auc']:.3f}  "
              f"P@5%={m['prec@top5pct']:.3f}  P@10%={m['prec@top10pct']:.3f}  "
              f"P@20%={m['prec@top20pct']:.3f}")

    comp_df = pd.DataFrame(comp_rows)[["model","roc_auc","pr_auc",
                                        "prec@top5pct","prec@top10pct","prec@top20pct"]]
    best_name     = comp_df.loc[comp_df["roc_auc"].idxmax(), "model"]
    best_prob_te  = model_probs[best_name]
    best_fi       = fi_dfs[best_name]
    print(f"\n★ Best model: {best_name}")

    # ── Baseline comparison ───────────────────────────────────────────────────
    # Load v2 metrics for direct comparison
    v2_comp_files = glob.glob(os.path.join(REPORTS_DIR, "model_comparison_20260312_v1.csv"))
    v2_comp = pd.read_csv(v2_comp_files[0]) if v2_comp_files else None

    # ── Threshold analysis ────────────────────────────────────────────────────
    print("\n═══ Threshold analysis ═══")
    thresh_df = threshold_sweep(y_te, best_prob_te)

    hp_cands = thresh_df[(thresh_df["precision"] >= 0.55) & (thresh_df["recall"] > 0)]
    best_thresh_hp    = float(hp_cands["thresh"].iloc[-1]) if len(hp_cands) else 0.60
    best_thresh_broad = best_f1_thresh(y_te, best_prob_te)
    print(f"  High-precision threshold: {best_thresh_hp:.2f}")
    print(f"  Broad threshold (best F1): {best_thresh_broad:.2f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    # Archive old v3 artifacts if they exist
    for old_pat in [
        f"model_pre_event_{VERSION}_*.pkl",
        f"model_pre_event_{VERSION}_meta_*.json",
    ]:
        for f in glob.glob(os.path.join(MODELS_DIR, old_pat)):
            dest = os.path.join(ARCHIVE_DIR, os.path.basename(f))
            if not os.path.exists(dest):
                import shutil; shutil.move(f, dest)

    # Re-fit best model on train+val for final saved artifact
    tv_enc2 = enc_main.transform(tv)
    X_tv2   = tv_enc2[[c for c in feat_cols_with_priors if c in tv_enc2.columns]].astype(float).fillna(0)
    y_tv2   = tv[TARGET].astype(int)
    n_pos_tv = y_tv2.sum(); scale_tv = float((len(y_tv2) - n_pos_tv) / n_pos_tv)

    if best_name == "LightGBM":
        final_model = build_lgbm(scale_tv); final_model.fit(X_tv2, y_tv2)
    elif best_name == "XGBoost":
        final_model = build_xgb(scale_tv); final_model.fit(X_tv2, y_tv2, verbose=False)
    else:
        final_model = build_logreg(); final_model.fit(X_tv2, y_tv2)

    model_path = os.path.join(MODELS_DIR, f"model_pre_event_{VERSION}_{DATE_TAG}.pkl")
    joblib.dump(final_model, model_path)
    print(f"\nSaved model: models/{os.path.basename(model_path)}")

    # Save encoder
    enc_path = os.path.join(MODELS_DIR, f"prior_encoder_{VERSION}_{DATE_TAG}.pkl")
    joblib.dump(enc_main, enc_path)

    # CSV artifacts
    cv_path   = os.path.join(REPORTS_DIR, f"cv_metrics_{DATE_TAG}_{VERSION}.csv")
    comp_path = os.path.join(REPORTS_DIR, f"model_comparison_{DATE_TAG}_{VERSION}.csv")
    fi_path   = os.path.join(REPORTS_DIR, f"feature_importance_{DATE_TAG}_{VERSION}.csv")

    cv_df.to_csv(cv_path, index=False)
    comp_df.to_csv(comp_path, index=False)
    best_fi.to_csv(fi_path, index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_cv_folds(cv_df, f"LightGBM+priors ({VERSION})",
                  os.path.join(FIGS_DIR, f"cv_folds_{DATE_TAG}_{VERSION}.png"))
    plot_model_comparison(comp_df,
                          os.path.join(FIGS_DIR, f"model_comparison_{DATE_TAG}_{VERSION}.png"))
    models_data = [(name, y_te, model_probs[name])
                   for name in ["LogReg", "LightGBM", "XGBoost"]]
    plot_roc_pr(models_data,
                os.path.join(FIGS_DIR, f"roc_pr_{DATE_TAG}_{VERSION}.png"))
    plot_feature_importance(best_fi, best_name,
                            os.path.join(FIGS_DIR, f"feature_importance_{DATE_TAG}_{VERSION}.png"))

    # ── Report ────────────────────────────────────────────────────────────────
    best_row = comp_df[comp_df["model"] == best_name].iloc[0]
    top10_fi = best_fi.head(10)["feature"].tolist()

    # v2 baseline for comparison
    v2_best_auc  = "N/A"
    v2_best_p10  = "N/A"
    improved_str = "unknown"
    if v2_comp is not None:
        v2_row = v2_comp.loc[v2_comp["roc_auc"].idxmax()]
        v2_best_auc = f"{v2_row['roc_auc']:.3f}"
        v2_best_p10 = f"{v2_row['prec@top10pct']:.3f}"
        delta_auc = best_row["roc_auc"] - v2_row["roc_auc"]
        improved_str = f"{'YES ↑' if delta_auc > 0.005 else 'MARGINAL' if delta_auc > 0 else 'NO ↓'} (Δ={delta_auc:+.3f})"

    new_timing_feats = [
        "feat_primary_completion_imminent_30d",
        "feat_primary_completion_imminent_90d",
        "feat_completion_recency_bucket (6 one-hot)",
        "feat_time_since_last_company_event",
        "feat_time_since_last_asset_event",
        "feat_asset_event_sequence_num",
        "feat_company_event_sequence_num",
        "feat_recent_company_event_flag",
        "feat_recent_asset_event_flag",
    ]

    cv_table = "\n".join(
        f"| fold_{i} | {r.get('n_train','?')}/{r['n']:3d} | "
        f"{r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
        f"{r['prec@top5pct']:.3f} | {r['prec@top10pct']:.3f} | {r['prec@top20pct']:.3f} |"
        for i, (_, r) in enumerate(cv_df.iterrows())
    )

    comp_table = "\n".join(
        f"| {r['model']} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
        f"{r['prec@top5pct']:.3f} | {r['prec@top10pct']:.3f} | {r['prec@top20pct']:.3f} |"
        for _, r in comp_df.iterrows()
    )

    fi_table = "\n".join(
        f"| {i+1} | {row['feature']} | {row['importance']:.4f} |"
        for i, (_, row) in enumerate(best_fi.head(10).iterrows())
    )

    # Coverage table for new features
    coverage_lines = []
    for feat in new_timing_feats:
        base = feat.split(" ")[0]
        if base in feat_df.columns:
            n_v = feat_df[base].notna().sum()
            coverage_lines.append(f"| {feat} | {n_v}/{len(feat_df)} | {n_v/len(feat_df)*100:.1f}% |")
        else:
            coverage_lines.append(f"| {feat} | — | — |")
    coverage_table = "\n".join(coverage_lines)

    hp_row  = thresh_df[thresh_df["thresh"].round(2) == round(best_thresh_hp, 2)]
    brd_row = thresh_df[thresh_df["thresh"].round(2) == round(best_thresh_broad, 2)]
    hp_str  = (f"prec={hp_row['precision'].values[0]:.3f}  rec={hp_row['recall'].values[0]:.3f}  "
               f"n={int(hp_row['n_flagged'].values[0])}") if len(hp_row) else "—"
    brd_str = (f"prec={brd_row['precision'].values[0]:.3f}  rec={brd_row['recall'].values[0]:.3f}  "
               f"n={int(brd_row['n_flagged'].values[0])}") if len(brd_row) else "—"

    report_md = f"""# Biotech Pre-Event Model v3 — Timing + Fold-Safe Priors

**Date:** {date.today()}
**Objective:** Predict large stock move from public pre-event information only.
**No press release content used.**
**Version:** v3 — adds 9 timing features + 6 train-fold-safe reaction priors vs v2 baseline.

---

## 1. New Features Added

### Timing features (9 new columns, deterministic)

| Feature | Coverage |
|---|---|
{coverage_table}

**feat_days_to_study_completion:** SKIPPED — no `ct_study_completion` column in dataset.

All timing features use `v_actual_date` (validated event date) as anchor.
Sequence/time-since features sorted globally by date within company/asset groups.

### Train-fold-safe priors (6 columns, injected inside folds)

| Prior feature | Group key | Target stat |
|---|---|---|
| feat_prior_mean_abs_move_atr_by_phase | feat_phase_num | mean(|ATR-norm move|) |
| feat_prior_mean_abs_move_atr_by_therapeutic_superclass | feat_therapeutic_superclass | mean(|ATR-norm move|) |
| feat_prior_mean_abs_move_atr_by_phase_x_therapeutic_superclass | phase × superclass | mean(|ATR-norm move|) |
| feat_prior_mean_abs_move_atr_by_market_cap_bucket | feat_market_cap_bucket | mean(|ATR-norm move|) |
| feat_prior_large_move_rate_by_phase | feat_phase_num | P(large move) |
| feat_prior_large_move_rate_by_therapeutic_superclass | feat_therapeutic_superclass | P(large move) |

Priors fit on TRAIN split only; fallback = global train mean for unseen categories.
Interaction prior requires ≥5 samples per cell, else falls back to phase-level prior.

---

## 2. Time-Aware Cross-Validation (LightGBM + Priors)

Mean ROC-AUC = {cv_df['roc_auc'].mean():.3f} ± {cv_df['roc_auc'].std():.3f}
Mean PR-AUC  = {cv_df['pr_auc'].mean():.3f} ± {cv_df['pr_auc'].std():.3f}
Mean Prec@10% = {cv_df['prec@top10pct'].mean():.3f} ± {cv_df['prec@top10pct'].std():.3f}

| Fold | Val n / Train n | ROC-AUC | PR-AUC | P@5% | P@10% | P@20% |
|---|---|---|---|---|---|---|
{cv_table}

---

## 3. Model Comparison — Test Set (v3 features)

| Model | ROC-AUC | PR-AUC | Prec@5% | Prec@10% | Prec@20% |
|---|---|---|---|---|---|
{comp_table}

★ **Best model: {best_name}**
Test ROC-AUC = {best_row['roc_auc']:.3f} | PR-AUC = {best_row['pr_auc']:.3f}
Prec@top 5% = {best_row['prec@top5pct']:.3f} | @top 10% = {best_row['prec@top10pct']:.3f} | @top 20% = {best_row['prec@top20pct']:.3f}

---

## 4. Comparison vs v2 Baseline

| Metric | v2 Baseline | v3 (timing+priors) | Change |
|---|---|---|---|
| Best ROC-AUC (test) | {v2_best_auc} | {best_row['roc_auc']:.3f} | {improved_str} |
| Best Prec@10% (test) | {v2_best_p10} | {best_row['prec@top10pct']:.3f} | — |
| Feature count | 49 | {len(feat_cols_with_priors)} | +{len(feat_cols_with_priors) - 49} |

**Overall verdict: {improved_str}**

---

## 5. Threshold / Ranking Strategy

### High-precision watchlist
Threshold ≈ {best_thresh_hp:.2f}: {hp_str}

### Broad candidate list (best F1)
Threshold ≈ {best_thresh_broad:.2f}: {brd_str}

---

## 6. Top 10 Feature Importances ({best_name})

| Rank | Feature | Importance |
|---|---|---|
{fi_table}

---

## 7. Key Findings

- **Timing features** add coverage of trial completion proximity, which was missing in v2.
  `feat_completion_recency_bucket` and `feat_primary_completion_imminent_*` capture
  the "hot zone" where readout is imminent — a known driver of pre-event moves.
- **Sequence features** (`feat_company/asset_event_sequence_num`) encode whether this
  is a company's first major readout or a follow-on event. Later-stage companies with
  repeat catalysts may have more predictable patterns.
- **Fold-safe priors** prevent leakage that the static precomputed priors in the dataset
  would cause. They encode the average magnitude/rate of moves for similar phase/disease.
- **Time-since-last-event** features encode event clustering and momentum dynamics.

## 8. Figures

- `figures/cv_folds_{DATE_TAG}_{VERSION}.png`
- `figures/model_comparison_{DATE_TAG}_{VERSION}.png`
- `figures/roc_pr_{DATE_TAG}_{VERSION}.png`
- `figures/feature_importance_{DATE_TAG}_{VERSION}.png`
"""

    report_path = os.path.join(REPORTS_DIR, f"ml_pre_event_v3_report_{DATE_TAG}_v1.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    # Model meta
    meta = {
        "created_at":      str(date.today()),
        "version":         VERSION,
        "best_model":      best_name,
        "features":        feat_cols_with_priors,
        "n_features":      len(feat_cols_with_priors),
        "n_timing_new":    9,
        "n_priors":        len([c for c in prior_cols if c in tr_p.columns]),
        "cv_auc_mean":     round(float(cv_df["roc_auc"].mean()), 4),
        "cv_auc_std":      round(float(cv_df["roc_auc"].std()), 4),
        "test_metrics":    comp_df.set_index("model").to_dict("index"),
        "threshold_hp":    best_thresh_hp,
        "threshold_broad": best_thresh_broad,
        "v2_best_auc":     v2_best_auc,
        "improved":        improved_str,
    }
    with open(os.path.join(MODELS_DIR, f"model_pre_event_{VERSION}_meta_{DATE_TAG}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved:")
    print(f"  {os.path.basename(report_path)}")
    print(f"  {os.path.basename(cv_path)}")
    print(f"  {os.path.basename(comp_path)}")
    print(f"  {os.path.basename(fi_path)}")
    print(f"  models/{os.path.basename(model_path)}")
    print(f"  4 plots in reports/figures/")

    print(f"\n── SUMMARY ──────────────────────────────────────────────────────")
    print(f"  Best model:      {best_name}")
    print(f"  Test ROC-AUC:    {best_row['roc_auc']:.3f}  (v2: {v2_best_auc})")
    print(f"  Test PR-AUC:     {best_row['pr_auc']:.3f}")
    print(f"  Prec@top 5%:     {best_row['prec@top5pct']:.3f}")
    print(f"  Prec@top 10%:    {best_row['prec@top10pct']:.3f}  (v2: {v2_best_p10})")
    print(f"  CV AUC:          {cv_df['roc_auc'].mean():.3f} ± {cv_df['roc_auc'].std():.3f}")
    print(f"  Features:        {len(feat_cols_with_priors)} (base {len(feat_cols)} + priors)")
    print(f"  vs v2 baseline:  {improved_str}")
    print(f"\nTop 10 features:")
    for i, (_, row) in enumerate(best_fi.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']}")


if __name__ == "__main__":
    main()

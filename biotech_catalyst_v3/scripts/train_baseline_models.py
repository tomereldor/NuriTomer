"""
train_baseline_models.py
=========================
Train first baseline models for biotech large-move prediction.

Target : target_large_move  (binary: 1 = High/Extreme ATR move)
Models :
  1. Majority-class baseline
  2. Logistic Regression
  3. LightGBM (primary tree model)

Split  : time-based (train/val/test) from baseline training table

Outputs:
  models/model_logreg_20260310_v1.pkl
  models/model_lgbm_20260310_v1.pkl
  models/model_meta_20260310_v1.json
  reports/metrics_summary_20260310_v1.csv
  reports/feature_importance_20260310_v1.csv
  reports/predictions_val_20260310_v1.csv
  reports/predictions_test_20260310_v1.csv
  reports/figures/*.png
  reports/ml_baseline_report_20260310_v1.md

Usage (from biotech_catalyst_v3/):
    python -m scripts.train_baseline_models
"""

import glob
import json
import os
import re
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGS_DIR    = os.path.join(REPORTS_DIR, "figures")

METADATA_COLS = ["ticker", "event_date", "drug_name", "nct_id"]
TARGET_COL    = "target_large_move"
RANDOM_STATE  = 42

plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")


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


def _next_v(base_dir, prefix, date_tag):
    files = glob.glob(os.path.join(base_dir, f"*{prefix}*{date_tag}*v*."))
    files += glob.glob(os.path.join(MODELS_DIR, f"*{prefix}*{date_tag}_v*.pkl"))
    files += glob.glob(os.path.join(REPORTS_DIR, f"*{prefix}*{date_tag}_v*.csv"))
    nums = [int(re.search(r"_v(\d+)", f).group(1))
            for f in files if re.search(r"_v(\d+)", f)]
    return max(nums) + 1 if nums else 1


def metrics_dict(y_true, y_pred, y_prob, label):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "split":              label,
        "n":                  int(len(y_true)),
        "n_positive":         int(y_true.sum()),
        "positive_pct":       round(y_true.mean() * 100, 1),
        "accuracy":           round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy":  round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision":          round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":             round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                 round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":            round(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0, 4),
        "pr_auc":             round(average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_class_balance(df, target_col, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df[target_col].value_counts()
    ax.bar(["Negative (0)", "Positive (1)"], [vc.get(0, 0), vc.get(1, 0)],
           color=["steelblue", "tomato"], edgecolor="white")
    ax.set_title("Target Class Balance (full training set)")
    ax.set_ylabel("Count")
    for i, v in enumerate([vc.get(0, 0), vc.get(1, 0)]):
        ax.text(i, v + 2, str(v), ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_missingness(df_raw, feature_cols, out_path):
    miss_pct = df_raw[feature_cols].isna().mean().sort_values(ascending=True) * 100
    miss_pct = miss_pct[miss_pct > 0]
    if miss_pct.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(miss_pct) * 0.35)))
    ax.barh(miss_pct.index, miss_pct.values, color="coral", edgecolor="white")
    ax.set_xlabel("Missing %")
    ax.set_title("Feature Missingness (before imputation)")
    ax.axvline(5, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(20, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(models_roc, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, fpr, tpr, auc_val in models_roc:
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr_curve(models_pr, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prec, rec, ap in models_pr:
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(importances, out_path, top_n=20):
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(5, len(top) * 0.4)))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_calibration(y_true, y_prob, model_name, out_path):
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_pred, frac_pos, "s-", label=model_name)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Plot — Test Set")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in [MODELS_DIR, REPORTS_DIR, FIGS_DIR]:
        os.makedirs(d, exist_ok=True)

    train_path, train_v, date_tag = _latest(BASE_DIR, "ml_baseline_train")
    if not train_path or "dict" in train_path:
        print("ERROR: no ml_baseline_train_*.csv found", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {os.path.basename(train_path)}")
    df = pd.read_csv(train_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    if "split" not in df.columns:
        print("ERROR: run make_time_splits.py first", file=sys.stderr)
        sys.exit(1)

    # Feature columns = everything except metadata, target, split
    drop_cols = set(METADATA_COLS) | {TARGET_COL, "split"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Feature columns: {len(feature_cols)}")

    # Data splits
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    te = df[df["split"] == "test"].copy()

    X_tr = tr[feature_cols].astype(float)
    y_tr = tr[TARGET_COL].astype(int)
    X_va = va[feature_cols].astype(float)
    y_va = va[TARGET_COL].astype(int)
    X_te = te[feature_cols].astype(float)
    y_te = te[TARGET_COL].astype(int)

    print(f"\nSplit sizes: train={len(tr)}, val={len(va)}, test={len(te)}")
    print(f"Train pos rate: {y_tr.mean():.1%} | Val: {y_va.mean():.1%} | Test: {y_te.mean():.1%}")

    # Version suffix
    ver = _next_v(BASE_DIR, "model", date_tag)
    suffix = f"{date_tag}_v{ver}"

    # ── 1. Majority-class baseline ──────────────────────────────────────────
    print("\n─ Majority-class baseline ─")
    majority = int(y_tr.mode()[0])
    mj_pred_va = np.full(len(y_va), majority)
    mj_pred_te = np.full(len(y_te), majority)
    mj_prob_va = np.full(len(y_va), y_tr.mean())
    mj_prob_te = np.full(len(y_te), y_tr.mean())
    mj_va = metrics_dict(y_va, mj_pred_va, mj_prob_va, "val")
    mj_te = metrics_dict(y_te, mj_pred_te, mj_prob_te, "test")
    mj_va["model"] = mj_te["model"] = "majority_baseline"
    print(f"  Val  balanced_acc={mj_va['balanced_accuracy']:.3f}  roc_auc={mj_va['roc_auc']:.3f}")
    print(f"  Test balanced_acc={mj_te['balanced_accuracy']:.3f}  roc_auc={mj_te['roc_auc']:.3f}")

    # ── 2. Logistic Regression ──────────────────────────────────────────────
    print("\n─ Logistic Regression ─")
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    class_weight = {0: 1.0, 1: n_neg / n_pos}  # balance manually

    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        solver="lbfgs", C=0.1, random_state=RANDOM_STATE,
    )
    lr.fit(X_tr, y_tr)
    lr_prob_va = lr.predict_proba(X_va)[:, 1]
    lr_prob_te = lr.predict_proba(X_te)[:, 1]
    # Pick threshold using val to maximise F1
    thresholds = np.linspace(0.1, 0.9, 81)
    val_f1 = [f1_score(y_va, (lr_prob_va >= t).astype(int), zero_division=0) for t in thresholds]
    best_lr_thresh = thresholds[np.argmax(val_f1)]
    lr_pred_te = (lr_prob_te >= best_lr_thresh).astype(int)
    lr_pred_va = (lr_prob_va >= best_lr_thresh).astype(int)
    lr_va = metrics_dict(y_va, lr_pred_va, lr_prob_va, "val")
    lr_te = metrics_dict(y_te, lr_pred_te, lr_prob_te, "test")
    lr_va["model"] = lr_te["model"] = "logistic_regression"
    lr_va["threshold"] = lr_te["threshold"] = round(float(best_lr_thresh), 3)
    print(f"  Best val threshold: {best_lr_thresh:.2f}  val_f1={max(val_f1):.3f}")
    print(f"  Val  balanced_acc={lr_va['balanced_accuracy']:.3f}  roc_auc={lr_va['roc_auc']:.3f}  f1={lr_va['f1']:.3f}")
    print(f"  Test balanced_acc={lr_te['balanced_accuracy']:.3f}  roc_auc={lr_te['roc_auc']:.3f}  f1={lr_te['f1']:.3f}")

    joblib.dump(lr, os.path.join(MODELS_DIR, f"model_logreg_{suffix}.pkl"))

    # ── 3. LightGBM ────────────────────────────────────────────────────────
    print("\n─ LightGBM ─")
    scale_pos = float(n_neg / n_pos)
    lgb_params = {
        "objective":        "binary",
        "metric":           "auc",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "n_estimators":     500,
        "min_child_samples": 15,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos,
        "random_state":     RANDOM_STATE,
        "verbosity":        -1,
    }
    lgbm = lgb.LGBMClassifier(**lgb_params)
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    lgb_prob_va = lgbm.predict_proba(X_va)[:, 1]
    lgb_prob_te = lgbm.predict_proba(X_te)[:, 1]
    val_f1_lgb = [f1_score(y_va, (lgb_prob_va >= t).astype(int), zero_division=0) for t in thresholds]
    best_lgb_thresh = thresholds[np.argmax(val_f1_lgb)]
    lgb_pred_va = (lgb_prob_va >= best_lgb_thresh).astype(int)
    lgb_pred_te = (lgb_prob_te >= best_lgb_thresh).astype(int)
    lgb_va = metrics_dict(y_va, lgb_pred_va, lgb_prob_va, "val")
    lgb_te = metrics_dict(y_te, lgb_pred_te, lgb_prob_te, "test")
    lgb_va["model"] = lgb_te["model"] = "lightgbm"
    lgb_va["threshold"] = lgb_te["threshold"] = round(float(best_lgb_thresh), 3)
    lgb_va["n_estimators_used"] = lgb_te["n_estimators_used"] = lgbm.best_iteration_
    print(f"  Best iteration: {lgbm.best_iteration_}  best_val_threshold: {best_lgb_thresh:.2f}")
    print(f"  Val  balanced_acc={lgb_va['balanced_accuracy']:.3f}  roc_auc={lgb_va['roc_auc']:.3f}  f1={lgb_va['f1']:.3f}")
    print(f"  Test balanced_acc={lgb_te['balanced_accuracy']:.3f}  roc_auc={lgb_te['roc_auc']:.3f}  f1={lgb_te['f1']:.3f}")

    joblib.dump(lgbm, os.path.join(MODELS_DIR, f"model_lgbm_{suffix}.pkl"))

    # ── Best model (by val ROC-AUC) ─────────────────────────────────────────
    best_name = max(
        [("logistic_regression", lr_va["roc_auc"]),
         ("lightgbm",            lgb_va["roc_auc"])],
        key=lambda x: x[1],
    )[0]
    print(f"\n★ Best model (val ROC-AUC): {best_name}")

    # ── Metrics CSV ────────────────────────────────────────────────────────
    all_metrics = [mj_va, mj_te, lr_va, lr_te, lgb_va, lgb_te]
    metrics_df  = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(REPORTS_DIR, f"metrics_summary_{suffix}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # ── Feature importance ──────────────────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": lgbm.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_path = os.path.join(REPORTS_DIR, f"feature_importance_{suffix}.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"\nTop 15 features (LightGBM gain):")
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:<50}  {row['importance']:.0f}")

    # ── Predictions ────────────────────────────────────────────────────────
    for split_name, sub_df, probs_lr, probs_lgb, preds_lr, preds_lgb in [
        ("val",  va, lr_prob_va, lgb_prob_va, lr_pred_va, lgb_pred_va),
        ("test", te, lr_prob_te, lgb_prob_te, lr_pred_te, lgb_pred_te),
    ]:
        pred_df = sub_df[METADATA_COLS + [TARGET_COL]].copy()
        pred_df["prob_lr"]     = probs_lr
        pred_df["prob_lgbm"]   = probs_lgb
        pred_df["pred_lr"]     = preds_lr
        pred_df["pred_lgbm"]   = preds_lgb
        pred_df.to_csv(os.path.join(REPORTS_DIR, f"predictions_{split_name}_{suffix}.csv"), index=False)

    # ── Plots ───────────────────────────────────────────────────────────────
    # 1. Class balance
    plot_class_balance(df, TARGET_COL,
                       os.path.join(FIGS_DIR, f"class_balance_{suffix}.png"))

    # 2. Feature missingness (from raw features before imputation)
    base_features_raw = [c for c in df.columns if c not in set(METADATA_COLS) | {TARGET_COL, "split"}]
    plot_feature_missingness(df, base_features_raw,
                             os.path.join(FIGS_DIR, f"feature_missingness_{suffix}.png"))

    # 3. Confusion matrices (test)
    for name, preds in [("LogReg", lr_pred_te), ("LightGBM", lgb_pred_te)]:
        cm = confusion_matrix(y_te, preds)
        plot_confusion_matrix(cm, ["No", "Yes"], f"Confusion Matrix — {name} (Test)",
                              os.path.join(FIGS_DIR, f"confusion_{name.lower()}_{suffix}.png"))

    # 4. ROC curves
    roc_data = []
    for name, probs in [("LogReg", lr_prob_te), ("LightGBM", lgb_prob_te)]:
        fpr, tpr, _ = roc_curve(y_te, probs)
        auc_val = roc_auc_score(y_te, probs)
        roc_data.append((name, fpr, tpr, auc_val))
    plot_roc(roc_data, os.path.join(FIGS_DIR, f"roc_curves_{suffix}.png"))

    # 5. PR curves
    pr_data = []
    for name, probs in [("LogReg", lr_prob_te), ("LightGBM", lgb_prob_te)]:
        prec, rec, _ = precision_recall_curve(y_te, probs)
        ap = average_precision_score(y_te, probs)
        pr_data.append((name, prec, rec, ap))
    plot_pr_curve(pr_data, os.path.join(FIGS_DIR, f"pr_curves_{suffix}.png"))

    # 6. Feature importance
    plot_feature_importance(imp_df, os.path.join(FIGS_DIR, f"feature_importance_{suffix}.png"))

    # 7. Calibration (best model)
    best_probs_te = lgb_prob_te if best_name == "lightgbm" else lr_prob_te
    plot_calibration(y_te, best_probs_te, best_name,
                     os.path.join(FIGS_DIR, f"calibration_{suffix}.png"))

    # ── Model metadata ──────────────────────────────────────────────────────
    meta = {
        "created_at":       datetime.now().isoformat(),
        "date_tag":         date_tag,
        "version":          ver,
        "training_table":   os.path.basename(train_path),
        "target":           TARGET_COL,
        "features":         feature_cols,
        "n_features":       len(feature_cols),
        "n_train":          int(len(tr)),
        "n_val":            int(len(va)),
        "n_test":           int(len(te)),
        "best_model":       best_name,
        "lgbm_params":      lgb_params,
        "lgbm_best_iter":   int(lgbm.best_iteration_),
        "lgbm_threshold":   float(best_lgb_thresh),
        "logreg_C":         0.1,
        "logreg_threshold": float(best_lr_thresh),
        "test_metrics": {
            "majority_baseline": {"roc_auc": mj_te["roc_auc"], "balanced_acc": mj_te["balanced_accuracy"], "f1": mj_te["f1"]},
            "logistic_regression": {"roc_auc": lr_te["roc_auc"], "balanced_acc": lr_te["balanced_accuracy"], "f1": lr_te["f1"]},
            "lightgbm": {"roc_auc": lgb_te["roc_auc"], "balanced_acc": lgb_te["balanced_accuracy"], "f1": lgb_te["f1"]},
        },
        "top_10_features": imp_df.head(10)["feature"].tolist(),
    }
    with open(os.path.join(MODELS_DIR, f"model_meta_{suffix}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Markdown report ─────────────────────────────────────────────────────
    report_path = os.path.join(REPORTS_DIR, f"ml_baseline_report_{suffix}.md")
    with open(report_path, "w") as f:
        f.write(_build_report(meta, mj_va, mj_te, lr_va, lr_te, lgb_va, lgb_te,
                               imp_df, best_name, suffix))

    print(f"\n✓ Saved models:")
    print(f"  models/model_logreg_{suffix}.pkl")
    print(f"  models/model_lgbm_{suffix}.pkl")
    print(f"  models/model_meta_{suffix}.json")
    print(f"\n✓ Reports:")
    print(f"  reports/ml_baseline_report_{suffix}.md")
    print(f"  reports/metrics_summary_{suffix}.csv")
    print(f"  reports/feature_importance_{suffix}.csv")
    print(f"  reports/predictions_val_{suffix}.csv")
    print(f"  reports/predictions_test_{suffix}.csv")
    print(f"\n✓ Figures (7 plots) in reports/figures/")


def _build_report(meta, mj_va, mj_te, lr_va, lr_te, lgb_va, lgb_te,
                   imp_df, best_name, suffix):
    top_feats = "\n".join(
        f"| {i+1:2d} | `{row['feature']}` | {row['importance']:.0f} |"
        for i, (_, row) in enumerate(imp_df.head(15).iterrows())
    )
    return f"""# Biotech Large-Move Prediction — Baseline ML Report

**Generated:** {meta['created_at']}
**Dataset:** {meta['training_table']}
**Target:** `{meta['target']}` (1 = High/Extreme ATR-normalised move)

---

## 1. Dataset Summary

| | Count |
|---|---|
| Training rows | {meta['n_train']} |
| Validation rows | {meta['n_val']} |
| Test rows | {meta['n_test']} |
| Features | {meta['n_features']} |
| Split method | Time-based: oldest 70% train / next 15% val / newest 15% test |

---

## 2. Baseline Feature Audit Notes

The following features from the proposed list were **excluded** from baseline v1 due to outcome-leaning risk
(they are derived from press release result text and encode what happened, not pre-event context):

- `feat_endpoint_outcome_score` — from `primary_endpoint_met` (yes/no = trial outcome)
- `feat_primary_endpoint_known_flag` — from `primary_endpoint_met`
- `feat_superiority_flag` — keyword-extracted from `primary_endpoint_result`, `v_summary`
- `feat_stat_sig_flag` — keyword-extracted from result text (p-values, HR)
- `feat_clinically_meaningful_flag` — keyword-extracted from result text
- `feat_mixed_results_flag` — keyword-extracted from result text

These should be used in a separate **"given announcement" model** where the press release content is available.
Historical prior features are also excluded (train-fold-only computation required to avoid leakage).

---

## 3. Model Performance

### Validation Set

| Model | Balanced Acc | ROC-AUC | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| Majority baseline | {mj_va['balanced_accuracy']:.3f} | {mj_va['roc_auc']:.3f} | {mj_va['f1']:.3f} | {mj_va['precision']:.3f} | {mj_va['recall']:.3f} | {mj_va['pr_auc']:.3f} |
| Logistic Regression | {lr_va['balanced_accuracy']:.3f} | {lr_va['roc_auc']:.3f} | {lr_va['f1']:.3f} | {lr_va['precision']:.3f} | {lr_va['recall']:.3f} | {lr_va['pr_auc']:.3f} |
| **LightGBM** | **{lgb_va['balanced_accuracy']:.3f}** | **{lgb_va['roc_auc']:.3f}** | **{lgb_va['f1']:.3f}** | **{lgb_va['precision']:.3f}** | **{lgb_va['recall']:.3f}** | **{lgb_va['pr_auc']:.3f}** |

### Test Set (final evaluation)

| Model | Balanced Acc | ROC-AUC | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|---|
| Majority baseline | {mj_te['balanced_accuracy']:.3f} | {mj_te['roc_auc']:.3f} | {mj_te['f1']:.3f} | {mj_te['precision']:.3f} | {mj_te['recall']:.3f} | {mj_te['pr_auc']:.3f} |
| Logistic Regression | {lr_te['balanced_accuracy']:.3f} | {lr_te['roc_auc']:.3f} | {lr_te['f1']:.3f} | {lr_te['precision']:.3f} | {lr_te['recall']:.3f} | {lr_te['pr_auc']:.3f} |
| **LightGBM** | **{lgb_te['balanced_accuracy']:.3f}** | **{lgb_te['roc_auc']:.3f}** | **{lgb_te['f1']:.3f}** | **{lgb_te['precision']:.3f}** | **{lgb_te['recall']:.3f}** | **{lgb_te['pr_auc']:.3f}** |

★ **Best model: {best_name}** (by validation ROC-AUC)

### LightGBM Confusion Matrix (Test Set)
TP={lgb_te['tp']}  FP={lgb_te['fp']}  FN={lgb_te['fn']}  TN={lgb_te['tn']}

---

## 4. Class Imbalance

Positive rate (large move) in training set: approx {meta['n_train']}
Both models use class weighting / `scale_pos_weight` to handle imbalance.

---

## 5. Top 15 Feature Importances (LightGBM)

| Rank | Feature | Importance |
|---|---|---|
{top_feats}

---

## 6. Model Configuration

### LightGBM
- Best iteration: {meta['lgbm_best_iter']} (early stopping on val AUC)
- Classification threshold: {meta['lgbm_threshold']:.2f} (maximises val F1)
- scale_pos_weight: auto from train class ratio
- num_leaves=31, lr=0.05, subsample=0.8, colsample=0.8

### Logistic Regression
- C=0.1, class_weight=balanced, solver=lbfgs, max_iter=2000
- Classification threshold: {meta['logreg_threshold']:.2f} (maximises val F1)

---

## 7. Plots

All plots saved in `reports/figures/`:

- `class_balance_{suffix}.png` — target class distribution
- `feature_missingness_{suffix}.png` — missing values per feature
- `confusion_logreg_{suffix}.png` — LogReg confusion matrix
- `confusion_lightgbm_{suffix}.png` — LightGBM confusion matrix
- `roc_curves_{suffix}.png` — ROC curves for both models
- `pr_curves_{suffix}.png` — Precision-Recall curves
- `feature_importance_{suffix}.png` — Top-20 LightGBM importances
- `calibration_{suffix}.png` — Calibration curve for best model

---

## 8. Saved Artifacts

| Artifact | Path |
|---|---|
| LightGBM model | `models/model_lgbm_{suffix}.pkl` |
| LogReg model | `models/model_logreg_{suffix}.pkl` |
| Model metadata | `models/model_meta_{suffix}.json` |
| Metrics CSV | `reports/metrics_summary_{suffix}.csv` |
| Feature importance CSV | `reports/feature_importance_{suffix}.csv` |
| Val predictions | `reports/predictions_val_{suffix}.csv` |
| Test predictions | `reports/predictions_test_{suffix}.csv` |

---

## 9. Next Steps

1. **Add announcement-content model**: include `feat_superiority_flag`, `feat_stat_sig_flag`, etc. as a separate feature group — expected major lift
2. **Cross-validation**: k-fold with time-aware split for more robust estimates
3. **Hyperparameter tuning**: grid/Bayesian search on LightGBM
4. **Reaction prior features**: recompute inside training folds to avoid leakage, then add as features
5. **Feature interaction**: phase × disease × company size interaction terms
"""


if __name__ == "__main__":
    main()

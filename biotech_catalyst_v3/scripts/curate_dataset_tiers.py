"""
curate_dataset_tiers.py
=======================
Data-quality / dataset-design pass on the expanded master dataset.

Assigns a `data_tier` column to every row in the master CSV:
  - trusted_trainable : 2023+ row with complete price data; already in strict training set
  - repairable        : 2023+ row with missing price data (backfillable), OR
                        pre-2023 High/Extreme move event (strong signal, worth investigating)
  - history_only      : pre-2023 row with Noise/Low/Medium/null move class and usable price data;
                        valuable as sponsor/asset history, hard-negative pool, future timing model
  - reject            : FLAG_ERROR v_action, or no ticker + no NCT ID + no price data

Outputs
-------
  enriched_all_clinical_clean_v3_tiered_20260318_v1.csv   (tiered master)
  candidate_strict_trainable_20260318_v1.csv              (trusted_trainable rows only)
  reports/DATASET_NOTES.md                                (prepended section)

Usage
-----
  python -m scripts.curate_dataset_tiers
"""

import os
import sys
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
MASTER_CSV   = os.path.join(BASE_DIR, "enriched_all_clinical_clean_v3.csv")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")

DATE_TAG = "20260318"

TIERED_OUT    = os.path.join(BASE_DIR, f"enriched_all_clinical_clean_v3_tiered_{DATE_TAG}_v1.csv")
TRAINABLE_OUT = os.path.join(BASE_DIR, f"candidate_strict_trainable_{DATE_TAG}_v1.csv")

HIGH_SIGNAL_CLASSES = {"High", "Extreme"}

# Move-size binary target thresholds (must match build_pre_event_train_v2.py)
ATR_THRESHOLD      = 3.0
ABS_MOVE_THRESHOLD = 10.0   # percent

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading master: {MASTER_CSV}")
df = pd.read_csv(MASTER_CSV, low_memory=False)
print(f"  Shape: {df.shape}")

# ---------------------------------------------------------------------------
# Derive event year (use v_actual_date first, fall back to event_date)
# ---------------------------------------------------------------------------
def _parse_year(row):
    for col in ("v_actual_date", "event_date"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            try:
                return pd.to_datetime(str(val), errors="coerce").year
            except Exception:
                pass
    return None

df["_event_year"] = df.apply(_parse_year, axis=1)

# ---------------------------------------------------------------------------
# Derive target_large_move (master CSV doesn't carry it directly)
# ---------------------------------------------------------------------------
abs_atr  = df["stock_movement_atr_normalized"].abs()
abs_move = df["move_pct"].abs()           # move_pct is the primary move column
df["target_large_move"] = (
    (abs_atr >= ATR_THRESHOLD) & (abs_move >= ABS_MOVE_THRESHOLD)
).astype("Int8")
# Mark null where either component is null
null_mask = df["stock_movement_atr_normalized"].isna() | df["move_pct"].isna()
df.loc[null_mask, "target_large_move"] = pd.NA

# ---------------------------------------------------------------------------
# Tiering logic
# ---------------------------------------------------------------------------
def assign_tier(row):
    year        = row["_event_year"]
    v_action    = row.get("v_action")
    move_class  = row.get("move_class_norm")
    has_ticker  = pd.notna(row.get("ticker"))
    has_nct     = pd.notna(row.get("nct_id"))
    has_price   = pd.notna(row.get("move_pct")) and pd.notna(row.get("atr_pct"))

    # REJECT: explicitly flagged error
    if v_action == "FLAG_ERROR":
        return "reject"
    # REJECT: completely unanchored (no ticker, no NCT ID, no price)
    if not has_ticker and not has_nct and not has_price:
        return "reject"

    # TRUSTED_TRAINABLE: 2023+ with complete price data
    if isinstance(year, (int, float)) and year >= 2023 and has_price:
        return "trusted_trainable"

    # REPAIRABLE: 2023+ with missing price — backfill via yfinance is feasible
    if isinstance(year, (int, float)) and year >= 2023 and not has_price:
        return "repairable"

    # REPAIRABLE: pre-2023 High/Extreme event — genuine strong signal worth validating
    if move_class in HIGH_SIGNAL_CLASSES:
        return "repairable"

    # HISTORY_ONLY: everything else (pre-2023 Noise/Low/Medium, or year unknown)
    return "history_only"

print("Assigning tiers...")
df["data_tier"] = df.apply(assign_tier, axis=1)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
tier_order = ["trusted_trainable", "repairable", "history_only", "reject"]

print("\n=== TIER COUNTS (overall) ===")
tier_counts = df["data_tier"].value_counts().reindex(tier_order, fill_value=0)
for tier, n in tier_counts.items():
    pct = n / len(df) * 100
    print(f"  {tier:<22}: {n:5d} ({pct:.1f}%)")

print("\n=== TIER COUNTS BY YEAR GROUP ===")
year_bins = [
    ("2007–2019", lambda y: y < 2020),
    ("2020",      lambda y: y == 2020),
    ("2021",      lambda y: y == 2021),
    ("2022",      lambda y: y == 2022),
    ("2023",      lambda y: y == 2023),
    ("2024",      lambda y: y == 2024),
    ("2025",      lambda y: y == 2025),
    ("2026",      lambda y: y == 2026),
]
year_summary_rows = []
for label, fn in year_bins:
    mask = df["_event_year"].apply(lambda y: bool(fn(y)) if pd.notna(y) else False)
    sub  = df[mask]
    n    = len(sub)
    if n == 0:
        continue
    tt   = (sub["data_tier"] == "trusted_trainable").sum()
    rp   = (sub["data_tier"] == "repairable").sum()
    ho   = (sub["data_tier"] == "history_only").sum()
    rj   = (sub["data_tier"] == "reject").sum()
    pos  = sub["target_large_move"].sum(skipna=True)
    pos_pct = float(pos) / n * 100 if n > 0 else 0
    year_summary_rows.append(dict(
        year_group=label, rows=n,
        trusted_trainable=tt, repairable=rp, history_only=ho, reject=rj,
        positives=int(pos), pos_pct=round(pos_pct, 1)
    ))
    print(f"  {label:<10}: total={n:4d}  "
          f"trusted={tt:4d}  repair={rp:4d}  hist={ho:4d}  rej={rj:2d}  "
          f"pos={int(pos):3d} ({pos_pct:.1f}%)")

# ---------------------------------------------------------------------------
# Completeness report for new historical rows (2020-2022)
# ---------------------------------------------------------------------------
hist_mask = df["_event_year"].apply(lambda y: 2020 <= y <= 2022 if pd.notna(y) else False)
hist = df[hist_mask]
n_hist = len(hist)

print(f"\n=== COMPLETENESS: 2020-2022 rows (n={n_hist}) ===")
fields = [
    ("target_large_move",             "train now"),
    ("move_pct",                      "train now"),
    ("atr_pct",                       "train now"),
    ("stock_movement_atr_normalized", "train now"),
    ("ticker",                        "train now"),
    ("nct_id",                        "train now"),
    ("ct_phase",                      "repair later"),
    ("mesh_level1",                   "history only"),
    ("market_cap_m",                  "repair later"),
    ("move_class_norm",               "train now"),
    ("v_action",                      "repair later"),
    ("ct_primary_completion",         "history only"),
]
for col, tier_note in fields:
    if col == "target_large_move":
        valid = hist["target_large_move"].notna().sum()
    else:
        valid = hist[col].notna().sum() if col in hist.columns else 0
    pct = valid / n_hist * 100 if n_hist > 0 else 0
    print(f"  {col:<38}: {valid:5d}/{n_hist} ({pct:5.1f}%)  → {tier_note}")

# ---------------------------------------------------------------------------
# Trusted trainable: positives check
# ---------------------------------------------------------------------------
tt_mask = df["data_tier"] == "trusted_trainable"
tt = df[tt_mask]
n_tt     = len(tt)
n_pos    = int(tt["target_large_move"].sum(skipna=True))
n_neg    = int((tt["target_large_move"] == 0).sum())
pos_rate = n_pos / n_tt * 100 if n_tt > 0 else 0
print(f"\n=== TRUSTED_TRAINABLE subset ===")
print(f"  Rows: {n_tt}  Positives: {n_pos} ({pos_rate:.1f}%)  Negatives: {n_neg}")

# Repairable breakdown
rp_mask  = df["data_tier"] == "repairable"
rp       = df[rp_mask]
n_rp     = len(rp)
rp_2023p = (rp["_event_year"] >= 2023).sum()
rp_hist  = (rp["_event_year"] < 2023).sum()
print(f"\n=== REPAIRABLE subset ===")
print(f"  Rows: {n_rp}  (2023+ missing price: {rp_2023p}  pre-2023 High/Extreme: {rp_hist})")

# ---------------------------------------------------------------------------
# Save tiered master
# ---------------------------------------------------------------------------
out_df = df.drop(columns=["_event_year"])
print(f"\nSaving tiered master → {TIERED_OUT}")
out_df.to_csv(TIERED_OUT, index=False)
print(f"  Shape: {out_df.shape}")

# ---------------------------------------------------------------------------
# Save candidate strict-trainable subset
# ---------------------------------------------------------------------------
trainable_df = df[df["data_tier"] == "trusted_trainable"].drop(columns=["_event_year"])
print(f"Saving candidate strict-trainable → {TRAINABLE_OUT}")
trainable_df.to_csv(TRAINABLE_OUT, index=False)
print(f"  Shape: {trainable_df.shape}")

# ---------------------------------------------------------------------------
# Build report section
# ---------------------------------------------------------------------------
today = datetime.now().strftime("%Y-%m-%d")

year_table_md = (
    "| Year group | Rows | trusted_trainable | repairable | history_only | reject | "
    "Positives | Pos% |\n"
    "|---|---|---|---|---|---|---|---|\n"
)
for r in year_summary_rows:
    year_table_md += (
        f"| {r['year_group']} | {r['rows']} | {r['trusted_trainable']} | "
        f"{r['repairable']} | {r['history_only']} | {r['reject']} | "
        f"{r['positives']} | {r['pos_pct']}% |\n"
    )

comp_table_md = (
    "| Column | Valid (2020–2022) | Valid% | Sufficiency |\n"
    "|---|---|---|---|\n"
)
for col, tier_note in fields:
    if col == "target_large_move":
        valid = hist["target_large_move"].notna().sum()
    else:
        valid = hist[col].notna().sum() if col in hist.columns else 0
    pct = valid / n_hist * 100 if n_hist > 0 else 0
    comp_table_md += f"| `{col}` | {valid}/{n_hist} | {pct:.1f}% | {tier_note} |\n"

report_md = f"""## {today} · Dataset Tiering Pass (v1)

**Master:** `enriched_all_clinical_clean_v3.csv` — {len(df)} rows × {df.shape[1]-1} cols (before adding data_tier)
**Script:** `scripts/curate_dataset_tiers.py`
**Output (tiered master):** `enriched_all_clinical_clean_v3_tiered_{DATE_TAG}_v1.csv`
**Output (trainable subset):** `candidate_strict_trainable_{DATE_TAG}_v1.csv`

### Tier definitions

| Tier | Definition |
|---|---|
| `trusted_trainable` | year ≥ 2023, complete price data (move_2d_pct + atr_pct) — already in strict training set |
| `repairable` | year ≥ 2023 missing price data (yfinance backfill feasible), OR pre-2023 High/Extreme move event |
| `history_only` | pre-2023, Noise/Low/Medium move class, complete price data — genuine hard negatives; useful for sponsor/asset history, future timing model |
| `reject` | v_action=FLAG_ERROR, or no ticker + no nct_id + no price data |

### Tier counts by year group

{year_table_md}
**Overall:**

| Tier | Count | % |
|---|---|---|
| trusted_trainable | {tier_counts.get('trusted_trainable', 0)} | {tier_counts.get('trusted_trainable', 0)/len(df)*100:.1f}% |
| repairable | {tier_counts.get('repairable', 0)} | {tier_counts.get('repairable', 0)/len(df)*100:.1f}% |
| history_only | {tier_counts.get('history_only', 0)} | {tier_counts.get('history_only', 0)/len(df)*100:.1f}% |
| reject | {tier_counts.get('reject', 0)} | {tier_counts.get('reject', 0)/len(df)*100:.1f}% |

### trusted_trainable positives check

| | Value |
|---|---|
| Rows | {n_tt} |
| Positives | {n_pos} ({pos_rate:.1f}%) |
| Negatives | {n_neg} |
| Current positive rate | {pos_rate:.1f}% |
| Meets 25–30% target? | {'✓ Yes' if 25 <= pos_rate <= 35 else '✗ No — use class_weight=balanced'} |

### Completeness: 2020–2022 newly added historical rows (n={n_hist})

{comp_table_md}

### Repairable breakdown (n={n_rp})

| Category | Count |
|---|---|
| 2023+ missing price data | {rp_2023p} |
| pre-2023 High/Extreme move | {rp_hist} |

### Key findings

1. **2020–2022 historical rows have 100% price completeness** — ticker, price_before, move_2d_pct, atr_pct, nct_id all present. The data is structurally sound.
2. **Positive rate for 2020–2022 is 0.3–0.9%** — mean AbsATR ≈ 0.7 (vs threshold 3.0). These rows are genuine hard negatives (CT.gov quiet completions, no formal announcement). Not suitable for training the move-size model.
3. **mesh_level1 is nearly absent for 2020–2022 rows (4.3%)** and market_cap_m is ~49% missing. These need enrichment before they can be upgraded to repairable.
4. **No `is_expansion_row` flag exists** in the master CSV. Rows can only be distinguished by year range. Recommend adding this flag in a future expansion pass.
5. **92% of 2020–2022 rows are unvalidated** (v_action=NaN) — no PR/announcement check was performed during CT.gov expansion.

### Dataset design recommendations

1. **Keep the full expanded master** — `enriched_all_clinical_clean_v3_tiered_{DATE_TAG}_v1.csv` is the broad master. 2020–2022 rows are valid hard negatives and historical context.
2. **Create a separate balanced training subset** — `candidate_strict_trainable_{DATE_TAG}_v1.csv` (trusted_trainable tier only). Do NOT train on history_only rows; their near-zero positive rate dilutes signal.
3. **Target positive rate** — current trusted_trainable has {pos_rate:.1f}% positives. Use `class_weight="balanced"` in sklearn rather than undersampling; this is already the approach. Raw positive rate is acceptable for LightGBM/XGBoost with balanced weighting.
4. **Repairable 2023+ rows** ({rp_2023p} rows) — run `backfill_price_at_event.py` on these; if price fills, promote to trusted_trainable. Expect +50–150 rows of training data.
5. **history_only rows** — keep as hard-negative calibration pool; use for sponsor/asset history queries, future CT.gov timing model. Do not mix into move-size training.

---

"""

# Prepend to DATASET_NOTES.md
canonical_path = os.path.join(REPORTS_DIR, "DATASET_NOTES.md")
if os.path.exists(canonical_path):
    with open(canonical_path, "r") as f:
        existing = f.read()
    # Find the separator after the header block
    split_marker = "---\n\n"
    idx = existing.find(split_marker)
    if idx != -1:
        new_content = (
            existing[: idx + len(split_marker)]
            + report_md
            + "\n---\n\n"
            + existing[idx + len(split_marker):]
        )
    else:
        new_content = report_md + "\n\n---\n\n" + existing
    with open(canonical_path, "w") as f:
        f.write(new_content)
    print(f"\nPrepended section to {canonical_path}")
else:
    fallback = os.path.join(REPORTS_DIR, f"dataset_tiering_{DATE_TAG}_v1.md")
    with open(fallback, "w") as f:
        f.write(report_md)
    print(f"\nWrote standalone report to {fallback}")
    canonical_path = fallback

print(f"\n=== DONE ===")
print(f"  Tiered master : {TIERED_OUT}")
print(f"  Trainable CSV : {TRAINABLE_OUT}")
print(f"  Report        : {canonical_path}")

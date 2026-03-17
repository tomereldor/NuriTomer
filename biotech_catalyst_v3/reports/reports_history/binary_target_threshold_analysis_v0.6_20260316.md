# Binary Target Threshold Analysis — Pre-Event Model

**Date:** 2026-03-16
**Dataset:** `enriched_all_clinical_clean_v3.csv` — 2514 rows (2500 with valid ATR + abs_move)
**Purpose:** Evaluate candidate binary target definitions before re-running the full feature pipeline on the expanded dataset.

---

## Part 1 — Current threshold (confirmed)

| | Value |
|---|---|
| **Target** | `target_large_move = 1` when `move_class_norm ∈ {High, Extreme}` |
| **ATR threshold** | ≥ 5× (High bucket starts at 5×, Extreme at 8×) |
| **Absolute floor** | None applied |
| **Why chosen** | Original dataset composition: High+Extreme were naturally ~20% of the curated v2 event set — a workable positive class without any tuning |
| **Positive rate (v0.3 training table)** | **167 / 813 = 20.5%** (ml_baseline_train_v0.3, v2-derived) |
| **Positive rate on full v3** | **189 / 2500 = 7.6%** (diluted by 1619 expanded hard-negative rows) |

**Why the rate dropped:** The dataset expansion added 1619 hard-negative rows (CT.gov completions 2020–2022). These rows are 99.8% Noise/Low class — only 4 High/Extreme rows in 1619 expanded rows. This is expected behavior: CT.gov completions without formal announcements naturally produce small moves.

---

## Part 2 — Candidate threshold analysis

### Overall counts

| Candidate | ATR floor | Abs move floor | N positive | % (full v3, 2500 rows) | % (v2-only rows, 881) |
|---|---|---|---|---|---|
| Baseline (current) | ≥ 5.0× | none | 189 | **7.6%** | 21.0% |
| E | ≥ 2.0× | ≥ 10% | 269 | 10.8% | 28.0% (estimated) |
| A | ≥ 2.5× | ≥ 10% | 253 | 10.1% | 28.0% |
| B | ≥ 2.5× | ≥ 15% | 249 | 10.0% | 27.4% |
| **C** | **≥ 3.0×** | **≥ 10%** | **242** | **9.7%** | **26.9%** |
| D | ≥ 3.0× | ≥ 15% | 239 | 9.6% | 26.6% |

### Year-by-year breakdown (n positive, % of year)

| Year | Baseline(≥5×) | A(2.5/10%) | B(2.5/15%) | C(3.0/10%) | D(3.0/15%) | E(2.0/10%) | N rows |
|---|---|---|---|---|---|---|---|
| 2012–2019 | 0–1 (0%) | 0–1 (0%) | 0–1 (0%) | 0–1 (0%) | 0–1 (0%) | 0–1 (0%) | 5–156 |
| 2020 | 1 (0%) | 3 (1%) | 3 (1%) | 3 (1%) | 3 (1%) | 6 (1%) | 464 |
| 2021 | 2 (0%) | 3 (1%) | 3 (1%) | 2 (0%) | 2 (0%) | 4 (1%) | 445 |
| 2022 | 1 (0%) | 1 (0%) | 1 (0%) | 1 (0%) | 1 (0%) | 5 (1%) | 397 |
| 2023 | 45 (22%) | 61 (30%) | 61 (30%) | 60 (30%) | 60 (30%) | 66 (33%) | 202 |
| 2024 | 49 (19%) | 68 (27%) | 65 (26%) | 62 (25%) | 59 (23%) | 70 (28%) | 253 |
| 2025 | 74 (28%) | 99 (37%) | 98 (37%) | 97 (37%) | 97 (37%) | 100 (38%) | 265 |
| 2026 | 9 (27%) | 11 (33%) | 11 (33%) | 10 (30%) | 10 (30%) | 11 (33%) | 33 |
| **TOTAL** | **189 (8%)** | **253 (10%)** | **249 (10%)** | **242 (10%)** | **239 (10%)** | **269 (11%)** | **2500** |

### Key observations

1. **The desired 25–30% positive rate is NOT achievable on the full v3 dataset.** The expansion rows (1619 rows, 2007–2022, mostly CT.gov quiet completions) have ≤0.4% positive rate regardless of threshold. This is expected and correct — they are genuine hard negatives.

2. **The 25–30% target IS achievable on the original v2-derived rows (881 rows).** At ≥3.0× ATR + ≥10% abs floor: 237/881 = 26.9%.

3. **All candidates produce 25–37% positive rates in 2023–2025** — the years that dominate the original curated set. The overall dilution is purely from the expansion.

4. **Lowering from ≥5× to ≥3× adds the Medium class** (3–5× ATR) as positives. These are real catalyst events — the Medium class has median abs_move = 32%, mean 27%. Only 23/81 Medium rows have abs_move < 10% (and those are all large-cap pharma: AZN, PFE, SNY, AMGN — where a 3× ATR move is 5–7% absolute).

5. **The absolute floor (10% vs 15%) makes almost no difference** for C vs D: C gives 242, D gives 239 — a 3-row difference. The 10% floor is marginally better as it retains a few legitimate small/mid-cap events where 10–14% is economically meaningful.

### What does Medium class add? (economic check)

| | Count | Median abs move | Abs >= 10% | Abs >= 15% |
|---|---|---|---|---|
| Extreme (≥8×) | 98 | ~70%+ | 95/98 | 94/98 |
| High (5–8×) | 92 | ~30–50% | ~90/92 | ~87/92 |
| Medium (3–5×) | 81 | 32% | 58/81 | 58/81 |

Medium-class events with abs >= 10% are genuine tradeable events. The 23 Medium rows below 10% are all large-cap pharma (ATR ~1.5%) — filtering them with the 10% floor is correct.

---

## Part 3 — Date mismatch analysis

| Mismatch (event_date vs v_actual_date) | Count | % |
|---|---|---|
| 0 days | 2,398 | **96.8%** |
| 1 day | 42 | 1.7% |
| 2–3 days | 9 | 0.4% |
| 4–7 days | 12 | 0.5% |
| 8–14 days | 6 | 0.2% |
| 15–30 days | 5 | 0.2% |
| > 30 days | 4 | 0.2% |
| Max | 60 days | — |

**What this measures:** The offset between `event_date` (source date) and `v_actual_date` (validated announcement date). The v_action="DATE_FIXED" rows (580 rows) are the ones where validation corrected the date; the ATR move computation uses `v_actual_date` where available.

**For expanded rows (1619):** `v_actual_date` is set equal to `event_date` (CT.gov primary_completion_date). No independent validation. The relevant mismatch for these rows is the CT.gov lag (documented oncology caveat) — not a date-field discrepancy.

**Recommendation:**
- **0–3 days: fully workable.** 98.9% of all rows. A 1–2 day offset in announcement date introduces negligible label noise (moves are typically concentrated at the overnight window; 1-day shifts may include an extra pre/post gap but not materially change the direction or size class).
- **4–7 days: acceptable with flag.** Only 0.5% of rows. These may include cases where pricing was delayed or dates were ambiguous. Keep, but flag.
- **> 7 days: borderline.** 0.6% of rows. These likely have structural issues (wrong CT.gov date used as proxy; announcement preceded by leak; wrong NCT matched). **Do not hard-filter at this time** — the count is tiny and removing them risks losing a few genuine High/Extreme rows from the pre-2023 period.

**No hard filter recommended.** The mismatch distribution is excellent. The main risk is the CT.gov date lag for expanded rows — handled by the oncology interaction features, not by date filtering.

**Optional:** Add `feat_is_validated_date` binary flag (1 = v_actual_date was independently set, 0 = no validation) as a model feature. This lets the model discount expanded rows' timing-based labels automatically.

---

## Part 4 — Recommendation

### Recommended new binary target definition

```
target_large_move = 1  if  stock_movement_atr_normalized >= 3.0  AND  abs(move_pct) >= 10.0
                   0  otherwise
```

**Rationale:**

| Dimension | Assessment |
|---|---|
| Positive count (full v3) | 242 / 2500 = **9.7%** — workable with class_weight="balanced" |
| Positive count (v2-only rows) | 237 / 881 = **26.9%** — hits 25–30% target for the curated event subset |
| Economic meaning | All positives have ≥10% one-day absolute move = tradeable regardless of market cap |
| Removes false positives | 23 large-cap pharma rows with 3–4× ATR but only 5–9% abs move are excluded |
| New examples added | +53 Medium-class rows vs current baseline (242 vs 189) — genuine catalyst events |
| Temporal distribution | 25–37% positive rate in 2023–2026 = model will see strong signal in training years |

### On absolute floor: 10% vs 15%

**Choose 10%.** The difference is 3 rows (242 vs 239). The 10% floor is sufficient to exclude large-cap noise events. 15% adds no benefit and would exclude a handful of legitimate small/mid-cap events in the 10–14% range.

### On 25–30% target: achievable in v2 subset

The 25–30% desired rate is correct for the original curated event set. On the full expanded v3 dataset, the realistic positive rate is ~10% due to hard-negative dilution. This is fine — use `class_weight="balanced"` in all models. The 1619 expanded hard-negative rows add genuine value to calibration; diluting the positive rate is the expected and correct outcome of adding true negatives.

### On date mismatch: no filter needed

Keep all rows. Flag `>7 day mismatch` rows optionally. The 96.8% zero-mismatch rate makes hard filtering unnecessary.

### Summary decision table

| Question | Answer |
|---|---|
| New ATR threshold | ≥ 3.0× |
| Absolute move floor | ≥ 10% |
| Positive class on full v3 | 9.7% (242/2500) |
| Positive class on v2 subset | 26.9% (237/881) |
| Class imbalance strategy | class_weight="balanced" |
| Date mismatch filter | None needed (96.8% exact) |
| Medium class included? | Yes |
| Retrain now? | **Not yet** — update threshold first, then run full feature pipeline |

---

## Part 5 — Next steps before retraining

1. **Update `build_pre_event_train_v2.py`:** Change binary target logic from `move_class_norm ∈ {High, Extreme}` to `stock_movement_atr_normalized >= 3.0 AND abs(move_pct) >= 10`.
2. **Run full feature pipeline on v3 master** (see `run_full_pre_event_pipeline.py` skeleton).
3. **Retrain** with `class_weight="balanced"` in all three models.
4. **Expected result:** LogReg baseline AUC should improve due to larger positive class and more temporal diversity. CV variance should decrease due to more even year distribution.

See pipeline orchestration script: `scripts/run_full_pre_event_pipeline.py`

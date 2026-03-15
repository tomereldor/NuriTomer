# Pre-Event Model — Follow-Up Actions & Audit Findings

**Date:** 2026-03-15
**Based on:** ML audit of 2026-03-13
**Status:** Documentation confirmed; oncology interaction features added in v0.5

---

## 1. Target definitions (confirmed)

### `target_abs_move_atr`

```
target_abs_move_atr = abs(price_after - price_before) / price_before / atr_pct
```

- `price_before` = closing price on the **last trading day strictly before** the event date
- `price_after` = closing price on the **first trading day strictly after** the event date
- Window = 1 overnight move bracketing the announcement (raw column called `move_2d_pct` — "2 days apart", not a 2-day trailing return)
- **Trading days only** — not calendar days
- `atr_pct` = Wilder's RMA `ewm(alpha=1/20, adjust=False)`, 20 trading-day lookback, computed on price data **strictly before** the event

### `target_large_move`

Binary. **1 = High or Extreme class = ATR-normalised move ≥ 5×.**

Note: prior docs said "≥ 3–5×" which was ambiguous. The actual lower bound of the positive class is 5× (the High bucket starts at 5×, not 3×). This is now corrected throughout.

### `target_move_bucket`

5-class ordinal: Noise (<1.5×) / Low (1.5–3×) / Medium (3–5×) / High (5–8×) / Extreme (≥8×).
Not used in the current model.

---

## 2. Post-event features — confirmed exclusion list

The following features exist in the feature dataset but are **permanently excluded** from the pre-event model. All derive from the current announcement text or its recorded outcome.

| Feature | Source | Reason |
|---|---|---|
| `feat_superiority_flag` | `primary_endpoint_result`, `v_pr_key_info`, `v_summary`, `catalyst_summary` | Current PR text — keyword match on result language |
| `feat_stat_sig_flag` | Same text fields | p-values, HR/OR from current result |
| `feat_clinically_meaningful_flag` | Same text fields | Clinical significance language from current result |
| `feat_mixed_results_flag` | Same text fields | Failure/miss language from current result |
| `feat_endpoint_outcome_score` | `primary_endpoint_met` (Yes/No/Unclear) | Records the outcome of the current catalyst |
| `feat_primary_endpoint_known_flag` | `primary_endpoint_met` | 1 = outcome is known, i.e. announcement already happened |

These are confirmed absent from `ml_baseline_train_v0.3_20260313.csv` and must not be added to any future pre-event training table.

**Future valid version:** keyword signals from prior-phase publications for the same drug (not yet implemented).

---

## 3. Timing features — oncology caveat

CT.gov primary completion date–based features are valid but carry a known limitation for oncology:

**Issue:** For OS/PFS/DFS-driven oncology trials, the efficacy readout occurs when a target number of events accrues — typically 6–24 months before the CT.gov primary completion date. The CT.gov date reflects the administrative study close, not the readout. As a result, imminence flags can indicate "far" even when an oncology readout is genuinely imminent.

**Decision:** Keep all timing features as-is. They carry real signal for non-oncology (~40% of dataset). For oncology the "COMPLETED + recently closed" signal remains meaningful.

**Mitigation added in v0.5:** Oncology-aware interaction features (see section 5 below) allow the model to weight CT.gov timing differently for oncology vs non-oncology rows.

**Future improvement:** Extract enrollment completion language or "on track / delayed / ahead of schedule" from CT.gov `detailed_description` or investor guidance. Not yet implemented.

---

## 4. Binary vs multiclass

- **Binary (`target_large_move`) remains the primary objective.** Aligned with investment use case.
- **Multiclass deferred** until binary AUC exceeds 0.70 reliably.
- No one-vs-rest confusion matrices needed yet — precision/recall at threshold is the right diagnostic for the current binary model.
- `target_move_bucket` (5-class) exists in dataset; no multiclass model has been trained.

---

## 5. Oncology timing interaction features (added in v0.5)

New file: `ml_dataset_features_v0.5_20260315.csv`
Script: `scripts/add_oncology_timing_interactions.py`

Three deterministic interaction features added:

| Feature | Definition |
|---|---|
| `feat_oncology_x_imminent_30d` | 1 if oncology AND primary completion imminent ≤30d |
| `feat_oncology_x_imminent_90d` | 1 if oncology AND primary completion imminent ≤90d |
| `feat_oncology_x_recent_completion` | 1 if oncology AND recent completion flag = 1 |
| `feat_oncology_x_recency_imminent` | 1 if oncology AND recency bucket = imminent_0_30 |

These allow the model to learn separate coefficients for oncology vs non-oncology timing — the main mitigation for the oncology mismatch identified in the audit.

### Quick comparison (LogReg, same v0.3 train/val/test split)

| | Test ROC-AUC | Test PR-AUC | Prec@top10% |
|---|---|---|---|
| Baseline (v0.3, 64 features) | 0.653 | 0.392 | 0.250 |
| + Oncology interactions (+4) | 0.658 | 0.394 | 0.250 |
| Val baseline | 0.685 | 0.389 | 0.333 |
| Val + interactions | 0.685 | 0.389 | 0.333 |

**Finding:** Marginal lift on test (+0.005 AUC), neutral on val. No degradation. The features are not harmful but the signal is sparse — only ~43 rows have oncology=1 AND imminent_30d=1 (5.2% of train). The interaction features will contribute more value once the next model is retrained on the full v0.5 feature set including CT.gov pipeline proxies.

---

## 6. Management / CRO quality (deferred)

PI track record is feasible via CT.gov (same API pattern as sponsor queries, ~1–2 days). CEO/CMO level requires LLM enrichment — not practical now. Classified as B-priority research feature.

# Data Quality Rules

## The pre-event hard rule (ABSOLUTE — never break)

> A feature used in the pre-event move-size model must be computable using ONLY
> information publicly available BEFORE the future event occurs.
>
> **FORBIDDEN anchors:** `v_actual_date`, `event_date`, `event_trading_date`,
> realized announcement date, realized stock move date, announcement content or outcome.
>
> **Features that use `v_actual_date` as a computation anchor** (e.g., "days to CT.gov
> completion from event date") are **INVALID** even if the underlying CT.gov data is public.
>
> **Valid:** ordinal sequence counts (how many prior events exist for this company/drug),
> CT.gov status as-of a fixed pull date, trial design flags, financial metrics as-of pull date.

### Currently excluded INVALID_FOR_PRE_EVENT features

These must never appear in a training table without first fixing their anchor to `prediction_date`:

- `feat_days_to_primary_completion` — uses `v_actual_date` as anchor
- `feat_primary_completion_imminent_30d` / `_90d` — derived from above
- `feat_completion_recency_bucket` — derived from above
- `feat_recent_completion_flag` — uses `event_date` as anchor
- `feat_time_since_last_company_event` — current event endpoint = future date
- `feat_time_since_last_asset_event` — same
- `feat_recent_company_event_flag` / `feat_recent_asset_event_flag` — derived from above

### Currently excluded POST_EVENT features (permanently banned from pre-event model)

- `feat_superiority_flag`, `feat_stat_sig_flag`, `feat_clinically_meaningful_flag`
- `feat_mixed_results_flag`, `feat_endpoint_outcome_score`, `feat_primary_endpoint_known_flag`

---

## Dataset tiering

Every row in the master dataset must carry a `data_tier` label (assigned by `scripts/curate_dataset_tiers.py`):

| Tier | Definition | Training use |
|---|---|---|
| `trusted_trainable` | year ≥ 2023, complete price data (move_pct + atr_pct) | Primary training pool |
| `repairable` | year ≥ 2023 missing price (yfinance backfillable), OR pre-2023 High/Extreme event | Train after fix |
| `history_only` | pre-2023, Noise/Low/Medium move class, valid price — genuine hard negatives | Sponsor/asset history, future timing model only |
| `reject` | FLAG_ERROR v_action, or no ticker + no nct_id + no price | Discard |

**Do NOT mix `history_only` rows into the move-size training split.** Their near-zero positive rate (~0.3%) poisons the train distribution — confirmed empirically (596-row cohort vs 2049-row naive set).

---

## Data readiness checklist (run before every training)

Before building a train table or running a model, verify:

- [ ] Target (`target_large_move`) is derivable: `abs(stock_movement_atr_normalized) >= 3.0 AND abs(move_pct) >= 10.0`
- [ ] `move_pct` is the primary move column (not `move_2d_pct` — different completeness)
- [ ] Year filter applied: `MIN_EVENT_YEAR = 2023` (2020-2022 rows excluded from training)
- [ ] All `INVALID_FOR_PRE_EVENT` features absent from feature list
- [ ] All `POST_EVENT` features absent from feature list
- [ ] Class balance in 25–35% positive range (use `class_weight="balanced"` if needed)
- [ ] Time-based split (never random) using `v_actual_date`
- [ ] `row_ready` exclusion reasons inspected — don't blindly trust the flag (check *why* rows were excluded)
- [ ] Any relaxed filters are documented with rationale

---

## Class balance strategy

- **Target:** 25–30% positive rate in the training cohort.
- **Method:** `class_weight="balanced"` in sklearn / `scale_pos_weight` in XGBoost / `is_unbalance=True` in LightGBM.
- **Do NOT undersample the negative class** — we have relatively few positives and want all signal.
- **Do NOT oversample** (SMOTE etc.) until the baseline is stable at AUC ≥ 0.75.
- The broad master dataset stays realistic (9.7% positives overall) — the training cohort is curated.

---

## Binary target definition (canonical)

```python
target_large_move = 1  if  abs(stock_movement_atr_normalized) >= 3.0
                            AND abs(move_pct) >= 10.0
                   0  otherwise
```

- `stock_movement_atr_normalized = abs(price_after - price_before) / price_before / atr_pct`
- ATR: Wilder's RMA `ewm(alpha=1/20, adjust=False)`, 20 trading-day lookback, strictly pre-event
- Window: 1 overnight move (price_before = last close before event; price_after = first close after)
- Chosen threshold: ≥3.0× ATR captures Medium + High + Extreme classes; 10% floor excludes large-cap noise

---

## DO_NOT_USE_FOR_MODEL flag

Any column sourced from post-event data or current-event explanatory text must be:
1. Named with suffix `__DO_NOT_USE_FOR_MODEL` if stored in a CSV, OR
2. Tagged `DO_NOT_USE_FOR_MODEL = True` in a metadata column

Currently flagged sources:
- Benzinga `bz_teaser` / `bz_body` — current event text
- `v_pr_key_info`, `v_summary` — PR result text (validation columns, not features)
- `primary_endpoint_result`, `primary_endpoint_met` — outcome data
- WIIM (Why Is It Moving) — by definition post-event

---

## Event-date confidence

| Mismatch (event_date vs v_actual_date) | Count | Treatment |
|---|---|---|
| 0 days | 96.8% | Use as-is |
| 1–3 days | 2.1% | Use as-is |
| 4–7 days | 0.5% | Note in report |
| > 7 days | 0.6% | Investigate; may indicate CT.gov lag or oncology timing issue |

For oncology: CT.gov primary completion date ≠ readout date (readout can occur 6–24 months earlier when event count triggers). Timing features carry this limitation. Mitigation: `feat_oncology_x_imminent_*` interaction features.

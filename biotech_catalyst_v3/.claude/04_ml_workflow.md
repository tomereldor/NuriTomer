# ML Workflow Rules

## The 8-step pipeline

All steps must be re-run after a master dataset expansion. Use the one-command orchestrator:

```bash
python -m scripts.run_full_pre_event_pipeline
# Or from step N:
python -m scripts.run_full_pre_event_pipeline --start-step N
```

| Step | Script | Output |
|---|---|---|
| 1 | `prepare_ml_dataset.py` | `ml_dataset_features_*_v{N}.csv` (with `row_ready` flag) |
| 2 | `add_high_signal_features.py` | Feature columns added |
| 3 | `refresh_ctgov_features.py` | CT.gov API features |
| 4 | `build_ctgov_pipeline_proxies.py` | Pipeline proxy features |
| 5 | `add_pre_event_timing_features.py` | Timing/sequence features |
| 6 | `add_oncology_timing_interactions.py` | Oncology × timing interactions |
| 7 | `build_pre_event_train_v2.py` | `ml_baseline_train_{DATE}_v{N}.csv` |
| 8 | `train_pre_event_v3.py` | Model `.pkl`, metrics CSVs, `MODEL_REPORTS.md` update |

**Never run step 8 without first running step 7** (models train on the train table, not the feature file directly).

---

## Train table build rules (step 7)

Key parameters in `build_pre_event_train_v2.py`:

```python
MIN_EVENT_YEAR = 2023          # exclude pre-2023 rows (near-zero positive rate)
ATR_THRESHOLD = 3.0            # abs_atr floor for target_large_move
ABS_MOVE_THRESHOLD = 10.0      # abs(move_pct)% floor
```

Row filter logic (ordered by precedence):
1. `row_ready == True` OR (`row_not_ready_reason == "missing_mesh_level1"` AND year ≥ 2023) — mesh imputes as "unknown"
2. `v_actual_date` not null
3. `_event_year >= MIN_EVENT_YEAR`

When bumping the train table version:
- Bump `VERSION` integer
- Update `DATE_TAG` to today
- The archiving logic auto-moves the previous version to `archive/`

---

## Feature validity (mandatory pre-training check)

Before every training run, verify the active feature roster against:
- `INVALID_FOR_PRE_EVENT` list (9 features — see `02_data_quality.md`)
- `POST_EVENT_EXCLUDED` list (6 features)

The feature dict CSV (`ml_baseline_train_dict_*.csv`) is the authoritative record of what was used.

**If any INVALID feature appears in the train table, stop. Fix the script. Rebuild. Retrain.**

---

## Evaluation standards

Every trained model must report all of the following:

| Metric | Why |
|---|---|
| Test ROC-AUC | Overall discrimination |
| CV AUC (5-fold time-aware) ± std | Generalization stability |
| Test PR-AUC | Useful for imbalanced classes |
| Prec@top 5% | Investment-relevant: can we trust the top picks? |
| Prec@top 10% | Primary investment metric |
| Prec@top 20% | Broader coverage |

**Prec@top10% is the primary investment metric.** AUC alone is not sufficient — a high-AUC model with poor Prec@top10% is not useful for stock selection.

### Mandatory training report fields

Every entry in `MODEL_REPORTS.md` must include:
- Train table filename
- Feature dataset filename
- Total rows / train / val / test row counts
- Class balance per split (% positive)
- Split method (always time-based on `v_actual_date`)
- Year range per split
- Feature count + list of excluded features
- Pre-event validity status (`STRICT_CLEAN` or explanation)
- All 6 metrics above
- Top 10 feature importances

---

## Model comparison discipline

**Every retrain must be compared against the previous strict-clean baseline.**

Report the delta table:

| Metric | New model | Previous baseline | Δ |
|---|---|---|---|
| Rows | N | N_prev | +/- |
| Test ROC-AUC | X | X_prev | +/- |
| CV AUC | X ± σ | X_prev ± σ_prev | +/- |
| Prec@10% | X | X_prev | +/- |

State clearly: **"The extra N rows materially improve / do not materially improve the model."**

A result is "material" if:
- AUC delta ≥ +0.01, OR
- Prec@top10% delta ≥ +0.05, OR
- CV std decreases by ≥ 20% (tighter generalization)

---

## Model selection policy

- Always train Logistic Regression, XGBoost, and LightGBM.
- Pick the best by **Test ROC-AUC** as primary, **Prec@top10%** as tiebreaker.
- Do not assume XGBoost is best — LogReg won at v5.
- Save the winning model to `models/model_pre_event_v3_20260312.pkl` (overwrite in place; the filename is the stable model path, versioning is tracked via train table version).

---

## Split rules

- **Always time-based** — sort by `v_actual_date`, split at 70/15/15 percentiles.
- **Never random split** — would leak future information across the split boundary.
- CV: `TimeSeriesSplit` with 5 folds on train+val combined.
- Fold-safe priors: fit on train fold, apply to val fold only. Never fit on the full dataset.

---

## When NOT to retrain

- Do not retrain to "see what happens" without a specific hypothesis.
- Do not retrain without first verifying the train table is clean.
- Do not retrain if the only change is a minor code refactor with no data/feature impact.
- Do retrain when: new features added, new rows added to trusted pool, filter logic changed, contamination found and fixed.

---

## Current model status (as of v5 — 2026-03-19)

| | Value |
|---|---|
| Train table | `ml_baseline_train_20260318_v5.csv` |
| Rows | 701 (2023+) |
| Features | 25 base + 6 priors = 31 |
| Best model | Logistic Regression |
| Test ROC-AUC | 0.703 |
| CV AUC (5-fold) | 0.752 ± 0.053 |
| Prec@top 10% | 0.545 |
| Validity | ✓ STRICT_CLEAN |
| Model file | `models/model_pre_event_v3_20260312.pkl` |

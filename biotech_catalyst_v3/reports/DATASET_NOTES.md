# Dataset Notes — Biotech Catalyst v3

Canonical running document for dataset expansion, target definitions, and coverage analysis.
Newest entry at top.

---

## 2026-03-16 · Binary Target Threshold Analysis (v0.6)

**Dataset:** `enriched_all_clinical_clean_v3.csv` — 2514 rows (2500 with valid ATR + abs_move)
**Purpose:** Evaluate candidate binary target definitions before re-running full feature pipeline on expanded dataset.

### Previous threshold (v0.3 and earlier)

`target_large_move = 1` when `move_class_norm ∈ {High, Extreme}` (ATR ≥ 5×, no absolute floor).
- Positive rate (v0.3 training table): 167/813 = 20.5%
- Positive rate on full v3: 189/2500 = 7.6% (diluted by 1619 hard-negative expansion rows)

### Candidate analysis

| Candidate | ATR floor | Abs move floor | N positive | % (full v3) | % (v2-only, 881 rows) |
|---|---|---|---|---|---|
| Baseline | ≥ 5.0× | none | 189 | 7.6% | 21.0% |
| E | ≥ 2.0× | ≥ 10% | 269 | 10.8% | ~28% |
| A | ≥ 2.5× | ≥ 10% | 253 | 10.1% | 28.0% |
| B | ≥ 2.5× | ≥ 15% | 249 | 10.0% | 27.4% |
| **C (chosen)** | **≥ 3.0×** | **≥ 10%** | **242** | **9.7%** | **26.9%** |
| D | ≥ 3.0× | ≥ 15% | 239 | 9.6% | 26.6% |

### Year-by-year breakdown (n positive, % of year)

| Year | Baseline | C (3.0/10%) | N rows |
|---|---|---|---|
| 2020 | 1 (0%) | 3 (1%) | 464 |
| 2021 | 2 (0%) | 2 (0%) | 445 |
| 2022 | 1 (0%) | 1 (0%) | 397 |
| 2023 | 45 (22%) | 60 (30%) | 202 |
| 2024 | 49 (19%) | 62 (25%) | 253 |
| 2025 | 74 (28%) | 97 (37%) | 265 |
| 2026 | 9 (27%) | 10 (30%) | 33 |
| **TOTAL** | **189 (8%)** | **242 (10%)** | **2500** |

### Key findings

1. **25–30% positive rate is NOT achievable on full v3** — expansion rows (1619 rows, 2020–2022, CT.gov quiet completions) have ≤0.4% positive rate regardless of threshold. Expected and correct — genuine hard negatives.
2. **25–30% IS achievable on v2-derived rows (881 rows):** Candidate C = 237/881 = 26.9%.
3. **2023–2025 rows show 25–37% positive rate** — model sees strong signal in training years.
4. **Lowering ≥5× → ≥3× adds the Medium class** (3–5× ATR) as positives. Medium-class events: median abs_move = 32%, mean 27%. Only 23 of 81 Medium rows have abs_move < 10% (all large-cap pharma: AZN/PFE/SNY/AMGN — correctly excluded by 10% floor).
5. **10% vs 15% floor: 3-row difference** (242 vs 239). 10% chosen — sufficient to exclude large-cap noise; 15% would drop a handful of legitimate small/mid-cap events.

### Recommendation and decision

```
target_large_move = 1  if  stock_movement_atr_normalized >= 3.0  AND  abs(move_pct) >= 10.0
                   0  otherwise
```

| Question | Answer |
|---|---|
| ATR threshold | ≥ 3.0× |
| Absolute move floor | ≥ 10% |
| Positive class on full v3 | 9.7% (242/2500) |
| Positive class on v2 subset | 26.9% (237/881) |
| Class imbalance strategy | class_weight="balanced" |
| Date mismatch filter | None needed (96.8% exact — see below) |
| Medium class included? | Yes |

### Date mismatch analysis (event_date vs v_actual_date)

| Mismatch | Count | % |
|---|---|---|
| 0 days | 2,398 | 96.8% |
| 1 day | 42 | 1.7% |
| 2–3 days | 9 | 0.4% |
| 4–7 days | 12 | 0.5% |
| > 7 days | 15 | 0.6% |
| Max | 60 days | — |

**Decision:** No hard filter. 0–3 days (98.9%) fully workable. >7 days (0.6%) likely have structural issues but removing risks losing genuine High/Extreme rows from pre-2023. The main risk for expanded rows is CT.gov date lag for oncology (handled by interaction features, not date filtering).

---

## 2026-03-16 · Dataset Expansion Strategy (v0.5)

**Master at time of expansion:** `enriched_all_clinical_clean_v2.csv` — 862 rows, 2007–2026

### Problem: temporal concentration

| Period | Rows | % |
|---|---|---|
| 2007–2019 | 43 | 5.0% |
| 2020–2022 | 74 | 8.6% |
| 2023–2026 | 734 | 85.2% |

85% of data from last 3 years — model has not seen multiple market regimes or rate environments.

### Expansion path decision

**Chosen: Path A — historical extension 2020–2022** (implemented as `scripts/expand_historical_events.py`).

Re-ran CT.gov-first event collection for Phase 2/3 COMPLETED trials 2020-01-01 to 2022-12-31.

| Path | Value | Difficulty | Decision |
|---|---|---|---|
| **A — historical extension** | Very high (temporal diversity, macro regimes) | Low (existing infrastructure) | **Priority 1 — implemented** |
| B — CT.gov-grounded collection | Already default approach | N/A | Same as A |
| C — hard-negative expansion | Medium | Medium | Priority 3 — A covers this naturally |
| D — more medium-move examples | Medium | High | Priority 4 — appear naturally in A |
| E — more oncology/late-stage | Low-Medium | Low | Include in A; no separate pass |

**Not chosen paths:** B (already default), C/D/E (low priority; covered partially by A).

### Expansion results

**Master expanded: 862 → 2514 rows (+1652 new rows)**

| | Value |
|---|---|
| New rows | 1652 |
| Phase breakdown | Phase 2: 695, Phase 3: 688, Phase 2/3: 55, Ph1/2: 182 |
| Year coverage | 2020=464, 2021=445, 2022=398 (was 19/21/34) |
| New row class distribution | Noise=1466 (89%), Low=133, Med=17, High=3, Extreme=2 |
| Oncology in new rows | 26.8% |
| All new rows have | nct_id + ATR computed |
| Full dataset class | Noise=81.2%, Low=7.5%, Medium=3.2%, High=3.7%, Extreme=3.9% |

**Key finding after expansion:** 2020–2022 rows have near-zero positive rate (~0.3–0.5%) regardless of threshold. This is expected — CT.gov completions without formal announcements naturally produce small moves. These rows add genuine hard-negative value for calibration. Training must be restricted to 2023+ rows to avoid poisoning the train split with near-zero label density (implemented in `build_pre_event_train_v2.py` as `MIN_EVENT_YEAR = 2023`).

### Post-expansion pipeline

All 8 steps must be re-run after master expansion — use one-command orchestration:
```bash
python -m scripts.run_full_pre_event_pipeline
```
Or from step N: `python -m scripts.run_full_pre_event_pipeline --start-step N`

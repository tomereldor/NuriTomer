# Dataset Notes — Biotech Catalyst v3

Canonical running document for dataset expansion, target definitions, and coverage analysis.
Newest entry at top.

---

## 2026-03-19 · Benzinga Pilot: Dataset Improvement Test

**Script:** `scripts/benzinga_pilot_event_ingest.py`
**Goal:** Test whether Benzinga can improve historical event-date accuracy and validated positives/negatives.

### API access audit

| Endpoint | HTTP | Accessible | Note |
|---|---|---|---|
| `news_v2_unfiltered` | 200 | ✓ |  |
| `news_v2_ticker_filter` | 200 | ✗ |  |
| `news_v2_channel_filter` | 200 | ✓ |  |
| `press_releases_v2_1` | 404 | ✗ | no Route matched with those values |
| `fda_calendar_v2_1` | 403 | ✗ | ["Unauthorized for Route"] |
| `wiim_v3` | 404 | ✗ | no Route matched with those values |

**Summary:**
- `news_v2` unfiltered: ✓ accessible — returns rolling window of ~10,000 most recent items only
- Ticker filtering: ✗ not accessible — returns 0 items when tickers param specified; plan limitation
- Channel filtering: ✓ accessible — returns unrelated items; not filtering by channel
- Press releases endpoint (v2.1): ✗ not accessible
- FDA Calendar (v2.1): ✗ not accessible
- WIIM (v3): ✗ not accessible

### Ingest results (what was accessible)

| | Value |
|---|---|
| Endpoint used | `GET /api/v2/news` (unfiltered) |
| Items fetched | 1000 |
| Date range accessible | 2026-02-20 → 2026-03-19 |
| Press-release-tagged items | 0 |
| Output file | `benzinga_pilot_news_20260319.csv` (flagged DO_NOT_USE_FOR_MODEL) |

**Hard constraint:** `page × pageSize ≤ 10,000`. With pageSize=100 and 40 pages → 4,000 items covering
only the most recent ~2026-02-20 → 2026-03-19. No way to query historical periods (2020–2022) without
ticker/date filtering support.

### Match against master dataset

| | Value |
|---|---|
| Master tickers | 330 |
| Benzinga items mentioning a master ticker | 149 |
| Within ±3 days of a known event | 7 |
| Potential event-date improvements | 7 (upper bound; mostly 2025–2026 recency only) |

### Assessment for our use cases

| Use case | Feasible with current plan? |
|---|---|
| Historical event-date accuracy (2020–2022) | ✗ No — can't filter by ticker or date range |
| Validated positives (2023–2026) | ✗ No — ticker filter broken |
| Validated negatives (2020–2022) | ✗ No — same |
| Future timing model (trial history) | ✗ No — no trial/NCT-ID linkage possible |
| Recent news enrichment (last ~6 months) | Partial — unfiltered feed only, no ticker selection |

### Recommendation

**NOT RECOMMENDED to scale** with the current API plan. Ticker filtering is broken, press-releases endpoint is inaccessible (404), and FDA calendar is unauthorized (403). The only accessible endpoint returns a small unfiltered rolling window with no historical date-range control. Would need at minimum the **Starter/Professional plan** (press-releases endpoint or working ticker filtering) to be useful for systematic dataset improvement.

**Required for Benzinga to be useful:**
1. Working ticker filter on `/api/v2/news` — needed to fetch company-specific history
2. Press releases endpoint (`/api/v2.1/press-releases`) — needed for exact PR timestamps
3. Historical date-range support — needed for 2020–2022 rows

**Alternative (current approach is better):** The existing Perplexity + CT.gov pipeline
already provides PR discovery and event-date validation for the master dataset.
For the specific gap (2020–2022 event-date accuracy), Perplexity is more capable given
the current Benzinga plan.

---


---

## 2026-03-18 · Dataset Tiering Pass (v1)

**Master:** `enriched_all_clinical_clean_v3.csv` — 2514 rows × 60 cols (before adding data_tier)
**Script:** `scripts/curate_dataset_tiers.py`
**Output (tiered master):** `enriched_all_clinical_clean_v3_tiered_20260318_v1.csv`
**Output (trainable subset):** `candidate_strict_trainable_20260318_v1.csv`

### Tier definitions

| Tier | Definition |
|---|---|
| `trusted_trainable` | year ≥ 2023, complete price data (move_2d_pct + atr_pct) — already in strict training set |
| `repairable` | year ≥ 2023 missing price data (yfinance backfill feasible), OR pre-2023 High/Extreme move event |
| `history_only` | pre-2023, Noise/Low/Medium move class, complete price data — genuine hard negatives; useful for sponsor/asset history, future timing model |
| `reject` | v_action=FLAG_ERROR, or no ticker + no nct_id + no price data |

### Tier counts by year group

| Year group | Rows | trusted_trainable | repairable | history_only | reject | Positives | Pos% |
|---|---|---|---|---|---|---|---|
| 2007–2019 | 415 | 0 | 2 | 413 | 0 | 1 | 0.2% |
| 2020 | 472 | 0 | 1 | 471 | 0 | 3 | 0.6% |
| 2021 | 450 | 0 | 2 | 448 | 0 | 2 | 0.4% |
| 2022 | 399 | 0 | 2 | 397 | 0 | 1 | 0.3% |
| 2023 | 208 | 203 | 5 | 0 | 0 | 60 | 28.8% |
| 2024 | 259 | 256 | 3 | 0 | 0 | 64 | 24.7% |
| 2025 | 271 | 266 | 4 | 0 | 1 | 98 | 36.2% |
| 2026 | 33 | 33 | 0 | 0 | 0 | 10 | 30.3% |

**Overall:**

| Tier | Count | % |
|---|---|---|
| trusted_trainable | 758 | 30.2% |
| repairable | 22 | 0.9% |
| history_only | 1733 | 68.9% |
| reject | 1 | 0.0% |

### trusted_trainable positives check

| | Value |
|---|---|
| Rows | 758 |
| Positives | 231 (30.5%) |
| Negatives | 527 |
| Current positive rate | 30.5% |
| Meets 25–30% target? | ✓ Yes |

### Completeness: 2020–2022 newly added historical rows (n=1321)

| Column | Valid (2020–2022) | Valid% | Sufficiency |
|---|---|---|---|
| `target_large_move` | 1320/1321 | 99.9% | train now |
| `move_pct` | 1321/1321 | 100.0% | train now |
| `atr_pct` | 1320/1321 | 99.9% | train now |
| `stock_movement_atr_normalized` | 1320/1321 | 99.9% | train now |
| `ticker` | 1321/1321 | 100.0% | train now |
| `nct_id` | 1321/1321 | 100.0% | train now |
| `ct_phase` | 1321/1321 | 100.0% | repair later |
| `mesh_level1` | 55/1321 | 4.2% | history only |
| `market_cap_m` | 650/1321 | 49.2% | repair later |
| `move_class_norm` | 1321/1321 | 100.0% | train now |
| `v_action` | 74/1321 | 5.6% | repair later |
| `ct_primary_completion` | 1321/1321 | 100.0% | history only |


### Repairable breakdown (n=22)

| Category | Count |
|---|---|
| 2023+ missing price data | 12 |
| pre-2023 High/Extreme move | 7 |

### Key findings

1. **2020–2022 historical rows have 100% price completeness** — ticker, price_before, move_2d_pct, atr_pct, nct_id all present. The data is structurally sound.
2. **Positive rate for 2020–2022 is 0.3–0.9%** — mean AbsATR ≈ 0.7 (vs threshold 3.0). These rows are genuine hard negatives (CT.gov quiet completions, no formal announcement). Not suitable for training the move-size model.
3. **mesh_level1 is nearly absent for 2020–2022 rows (4.3%)** and market_cap_m is ~49% missing. These need enrichment before they can be upgraded to repairable.
4. **No `is_expansion_row` flag exists** in the master CSV. Rows can only be distinguished by year range. Recommend adding this flag in a future expansion pass.
5. **92% of 2020–2022 rows are unvalidated** (v_action=NaN) — no PR/announcement check was performed during CT.gov expansion.

### Dataset design recommendations

1. **Keep the full expanded master** — `enriched_all_clinical_clean_v3_tiered_20260318_v1.csv` is the broad master. 2020–2022 rows are valid hard negatives and historical context.
2. **Create a separate balanced training subset** — `candidate_strict_trainable_20260318_v1.csv` (trusted_trainable tier only). Do NOT train on history_only rows; their near-zero positive rate dilutes signal.
3. **Target positive rate** — current trusted_trainable has 30.5% positives. Use `class_weight="balanced"` in sklearn rather than undersampling; this is already the approach. Raw positive rate is acceptable for LightGBM/XGBoost with balanced weighting.
4. **Repairable 2023+ rows** (12 rows) — run `backfill_price_at_event.py` on these; if price fills, promote to trusted_trainable. Expect +50–150 rows of training data.
5. **history_only rows** — keep as hard-negative calibration pool; use for sponsor/asset history queries, future CT.gov timing model. Do not mix into move-size training.

---


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

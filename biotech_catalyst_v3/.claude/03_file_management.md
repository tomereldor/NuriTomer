# File Management Rules

## File naming convention

```
{descriptive_name}_{YYYYMMDD}_v{N}.{ext}
```

- Date before version number
- Version is a simple integer (v1, v2, v3...), not semver
- Always use lowercase with underscores
- Examples:
  - `ml_baseline_train_20260318_v5.csv`
  - `ml_dataset_features_20260316_v2.csv`
  - `model_pre_event_v3_20260312.pkl`

**Never** create files with suffixes like `_linked`, `_mesh`, `_cleaned`, `_final`, `_v2_fixed`. Use the versioning convention instead.

---

## The master file rule

There is **one** master dataset file: `enriched_all_clinical_clean_v3.csv`

- All pipeline enrichments write to this file in place.
- No derivative copies with modified suffixes.
- When a new master version is created, the old one goes to `archive/` and the new one gets the next version number.

Exception: the `_tiered` file (`enriched_all_clinical_clean_v3_tiered_*.csv`) is an augmented view of the master with `data_tier` + `target_large_move` columns added. It is a derived artifact, not a replacement for the master.

---

## Archive policy

When a file is superseded by a newer version:
1. Move the old file to `archive/` immediately — same session, same commit.
2. The `archive/` directory is gitignored. Old files are kept locally but not tracked.
3. Delete from `archive/` when they are no longer needed for reference (e.g., >2 versions old).

Files that belong in archive:
- Superseded train tables (`ml_baseline_train_*.csv` when a newer version exists)
- Superseded feature datasets (when a newer pipeline run exists)
- Superseded model pickle files (when the model is retrained)

---

## Documentation structure

### Single source of truth: three canonical running docs

All model, feature, and dataset findings live in:

| File | Contents | Update rule |
|---|---|---|
| `reports/MODEL_REPORTS.md` | All model training results — newest section at top | `train_pre_event_v3.py` auto-prepends; fix boilerplate header each run |
| `reports/FEATURE_NOTES.md` | Feature validity audit, CT.gov notes, oncology caveats, excluded features | Prepend new dated section manually |
| `reports/DATASET_NOTES.md` | Dataset expansion, tiering analysis, target threshold analysis | Prepend new dated section manually |

**Never create a new standalone `.md` report file.** Always prepend to the relevant canonical doc.

### README is the entry point

`biotech_catalyst_v3/README.md` is for humans and collaborators:
- **Summary section** — current model state, dataset state, in plain language
- **Current state table** — source-of-truth files, current model metrics
- **Pipeline status line** — one sentence on where we are
- **Changelog** — one entry per significant change (v1.0, v1.1, etc.)

The README references the canonical reports for details. It does NOT duplicate them.

### What goes where

| Content | Location |
|---|---|
| Model training results, metrics, feature importances | `reports/MODEL_REPORTS.md` |
| Feature validity decisions, audit findings | `reports/FEATURE_NOTES.md` |
| Dataset design, tiering, target analysis | `reports/DATASET_NOTES.md` |
| Quick-start, current state, changelog | `biotech_catalyst_v3/README.md` |
| Plot artifacts | `reports/figures/` |
| CSV artifacts (cv_metrics, model_comparison, feature_importance) | `reports/` |
| Old standalone `.md` files | `reports/reports_history/` |

---

## Git hygiene

### Commit message format

```
Short description of change: key metric or row count (v{N}.{NN})

- bullet 1: specific file changed + what changed
- bullet 2: key numbers (rows, AUC, etc.)
- bullet 3: what was archived / removed

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Examples of good commit messages:
- `v5 retrain: 701-row expanded pool, LogReg AUC 0.703 — new strict-clean best (v3.34)`
- `Dataset tiering pass: tier all 2514 rows, candidate trainable subset 758 rows (v3.33)`
- `Strict-clean retrain: 9 invalid features removed → XGBoost AUC 0.692 (v3.31)`

### Push frequency

Push after every completed unit of work:
- After a retrain + docs update
- After a dataset change
- After a new script
- Never accumulate more than one logical change in a single push

### Never skip hooks or force-push to main.

---

## Environment

- Python venv: `/Users/tomer/Code/NuriTomer/.venv`
- Activate: `source .venv/bin/activate`
- Run scripts from `biotech_catalyst_v3/`: `python -m scripts.<name>`
- API keys: `.env` at repo root AND `biotech_catalyst_v3/.env` (gitignored)
  - Note: Benzinga key is `BENZIN_API_KEY` (not `BENZINGA_API_KEY`)

---

## Gitignore rules

```
biotech_catalyst_v3/archive/      # superseded CSVs
biotech_catalyst_v3/benzinga_pilot_*.csv  # DO_NOT_USE_FOR_MODEL data
*.bak.csv
biotech_catalyst_v3/data/ohlc/
.env (root)
biotech_catalyst_v3/.env
```

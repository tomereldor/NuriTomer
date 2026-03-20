# Biotech Catalyst v3 — Claude Rules & Best Practices

Project working rules, distilled from all sessions. Claude reads these at the start of each task.

---

## Files in this folder

| File | Contents |
|---|---|
| [`01_principles.md`](01_principles.md) | Core philosophy, decision-making framework, ways of working |
| [`02_data_quality.md`](02_data_quality.md) | Data readiness, tiering, pre-event validity, DO_NOT_USE flags |
| [`03_file_management.md`](03_file_management.md) | Naming conventions, archiving, docs structure, git hygiene |
| [`04_ml_workflow.md`](04_ml_workflow.md) | Training pipeline, evaluation standards, model comparison discipline |
| [`05_agents_efficiency.md`](05_agents_efficiency.md) | Agent/model selection, credits efficiency, reviewer stage |

---

## Quick-reference: the most important rules

1. **Plan before implementing.** For any non-trivial task: state the approach, implications, and file changes before writing code.
2. **Data-first.** Verify data is clean, correctly tiered, and all features are pre-event valid before any training run.
3. **Pre-event hard rule.** Never use a feature anchored to `v_actual_date`, `event_date`, or any realized post-event information in the pre-event model.
4. **One canonical doc per topic.** Never create a new standalone `.md` report file — prepend to the relevant canonical running doc.
5. **Always update README after any significant change** — changelog entry, updated pipeline status, updated current state table.
6. **Archive superseded files immediately.** Old train tables, old model files → `archive/` (gitignored). Never leave stale files in the project root.
7. **Compare every retrain against the previous strict-clean baseline.** Report AUC delta, CV AUC delta, Prec@top10% delta. State clearly: did the change materially improve the model?
8. **Push after every completed unit of work.** Never leave significant changes uncommitted.

# Principles — Ways of Working

## Core philosophy

Think like a **top-tier data scientist and ML engineer** working on a real investment-grade signal.
Every decision should maximize the probability of a genuine 10× improvement in prediction quality.
Be rigorous, data-driven, and intellectually honest — especially about what the data cannot yet support.

---

## Decision-making framework

### 1. Plan before implementing

For any non-trivial task (new feature, retrain, pipeline change, dataset modification):
- State the approach and its implications before writing a single line of code.
- Identify which files will change, what the expected outcome is, and what could go wrong.
- For architecture decisions, use the `Plan` subagent to think through trade-offs first.

**Never start coding a significant change without a clear plan.**

### 2. Be data-driven

Never make claims about model improvement without numbers.
Every assertion must be backed by:
- row counts (before and after)
- class balance (positive rate)
- metric deltas (AUC, Prec@top10%)
- explicit comparison to the previous baseline

Opinions are cheap. Show the delta.

### 3. Prioritize high-impact work

Before starting any task, ask: *is this the highest-leverage thing to do right now?*

**High-leverage:** expanding the trusted training set, fixing data leakage, adding genuinely new features with pre-event validity, improving evaluation discipline.

**Low-leverage:** refactoring for aesthetics, adding comments to unchanged code, creating new report files that duplicate existing ones, premature feature engineering before the baseline is stable.

Default to the minimum change that produces the maximum verified improvement.

### 4. Intellectual honesty about what you don't know

- If a result is surprising, investigate before reporting it.
- If the data supports two interpretations, state both.
- Never inflate metrics or hide model limitations in reports.
- If a plan upgrade (e.g. Benzinga, external API) doesn't work as expected, say so clearly and recommend the correct next step.

### 5. Do not over-engineer

- Don't add error handling, fallbacks, or abstractions for scenarios that can't happen.
- Don't design for hypothetical future requirements.
- The right amount of code is the minimum needed for the current task.
- Three similar lines of code is better than a premature abstraction.

---

## Workflow sequence for any significant task

```
1. Read relevant files first (never modify code you haven't read)
2. State the plan explicitly
3. Verify data readiness (see 02_data_quality.md)
4. Implement
5. Verify output (check counts, metrics, feature validity)
6. Update canonical docs (MODEL_REPORTS.md / FEATURE_NOTES.md / DATASET_NOTES.md)
7. Update README (changelog entry + current state)
8. Commit with descriptive message including row counts and key metrics
9. Push
```

---

## Things we learned the hard way

- **Contamination check before every retrain.** We ran a model for weeks with 9 invalid features leaking future event dates. AUC dropped from 0.730 → 0.692 when fixed. Always verify the feature list against `INVALID_FOR_PRE_EVENT` before training.
- **2020–2022 expansion rows nearly poisoned the training split.** Near-zero positive rate in those rows collapsed the signal when included in training. Always check class balance by year group before setting `MIN_EVENT_YEAR`.
- **"row_ready" excluded valid rows for the wrong reason.** 105 rows were excluded only for missing mesh_level1 — a valid imputable feature. Expanding the filter added +0.011 AUC. Check exclusion reasons before trusting quality filters blindly.
- **Benzinga API appeared accessible but wasn't.** HTTP 200 with 0 filtered results. Always do a functional test (does the filter actually work?), not just an HTTP status check.

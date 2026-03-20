# Agent Selection & Credits Efficiency

## Model selection by task type

| Task | Recommended model | Reasoning |
|---|---|---|
| Architecture decisions, complex planning | `opus` | Needs deep reasoning across trade-offs |
| Coding, implementation, debugging | `sonnet` (default) | Strong code quality, sufficient reasoning |
| File search, pattern matching, quick lookups | `haiku` / Explore agent | Fast, cheap, no reasoning needed |
| Reviewing code quality externally | `opus` | Independent high-quality review |
| Writing report sections, docs | `sonnet` | Good writing quality at lower cost |

Use the `model` parameter when spawning subagents:
```python
Agent(subagent_type="Plan", model="opus", ...)     # architecture
Agent(subagent_type="Explore", model="haiku", ...) # file search
```

---

## When to use subagents

### Use the Explore agent (fast, cheap)
- Finding files by pattern: `"src/**/*.py"`, `"*train*.csv"`
- Searching code for keywords: `"INVALID_FOR_PRE_EVENT"`, `"row_ready"`
- Quick codebase questions: "how does the feature pipeline work?"
- Do NOT use when a direct Glob/Grep would find it in 1-2 tries

### Use the Plan agent (before major work)
- Designing a new pipeline step
- Deciding between 2+ implementation approaches
- Planning a data schema change
- Any task touching >3 files or >100 lines of new code

### Use the general-purpose agent (background research)
- Deep multi-step analysis that would bloat the main context
- When you need to protect the main context window from large outputs
- Independent data audits (like the tiering analysis)
- Run in background when the result isn't needed to continue current work

### Use the Reviewer agent pattern (periodic quality checks)

After major implementations, spawn a reviewer with fresh context:
```
Agent(subagent_type="general-purpose", model="opus", prompt="""
You are reviewing the biotech_catalyst_v3 pre-event ML pipeline.
Read the following files and report any quality issues, potential bugs,
or violations of the pre-event validity rules:
- scripts/build_pre_event_train_v2.py
- scripts/train_pre_event_v3.py
- reports/MODEL_REPORTS.md (latest section only)

Check specifically:
1. Are any INVALID_FOR_PRE_EVENT features in the active feature list?
2. Is the time-based split implemented correctly?
3. Are fold-safe priors being computed correctly (train-only, not full dataset)?
4. Does the report section contain all mandatory fields?
Report findings concisely. Do not fix anything, just report.
""")
```

Run the reviewer agent:
- After every retrain
- After adding new features
- After any change to the train table build logic
- Before any external presentation of results

---

## Credits efficiency rules

### 1. Read before you act
Always `Read` the relevant files before modifying them. Never modify code you haven't read.

### 2. Search directly before spawning an agent
Use `Glob` or `Grep` directly for targeted searches. Only spawn an Explore agent if 2-3 direct searches don't find what you need.

### 3. Run parallel tool calls
When multiple independent reads/searches are needed, call them in one message:
```python
# Good: parallel
Read(file1), Read(file2), Grep(pattern)

# Bad: sequential when not dependent
Read(file1)
# ... wait ...
Read(file2)
```

### 4. Don't re-read files unnecessarily
If a file was read earlier in the conversation and hasn't changed, don't read it again. Use the content from the prior read.

### 5. Use focused scripts for data analysis
Instead of running multiple Python snippets to explore data, write a single analysis script that answers all questions in one run. The Bash tool has overhead; minimize round-trips.

### 6. Compact context proactively
Use `/compact` when the conversation has accumulated large intermediate outputs (data analysis results, long error traces) that are no longer needed for the current task.

### 7. Avoid redundant doc updates
Don't write to MODEL_REPORTS.md, FEATURE_NOTES.md, DATASET_NOTES.md, and README in separate commits. Batch all doc updates into one commit after completing the work.

---

## API access patterns (learned)

### CT.gov API
- Rate limit: be polite, add `time.sleep(0.1)` between requests
- Cache results in `cache/ctgov_details_v1.json` — never re-fetch what's already cached
- Use NCT IDs from master dataset, not raw queries

### Benzinga API (current plan: `BENZIN_API_KEY`)
- `GET /api/v2/news` unfiltered: ✓ works
- Ticker filter (`tickers=` param): ✗ returns 0 items — plan limitation
- Channel filter (`channels=` param): ✗ HTTP 200 but returns unrelated items
- Press releases `/api/v2.1/press-releases`: ✗ 404
- FDA Calendar `/api/v2.1/calendar/fda`: ✗ 403
- WIIM `/api/v3/wiim`: ✗ 404
- Max pagination: `page × pageSize ≤ 10,000`
- **Do not scale until plan is upgraded to support ticker filtering**

### Perplexity API (current validated workflow)
- Used for: PR discovery, event-date validation, noise-class row checking
- Script: `validate_catalysts.py`
- More capable than Benzinga for historical biotech PR lookup given current Benzinga plan

### yfinance
- Used for: price data backfill (`backfill_price_at_event.py`)
- Note: `move_pct` is the primary move column (vs `move_2d_pct` which has lower coverage)
- For 2023+ repairable rows: 12 rows with missing price are backfillable candidates

---

## When to NOT use an agent

- Reading a specific known file → use `Read` directly
- Searching for a specific class/function → use `Glob` directly
- Editing 1-3 lines in a file → use `Edit` directly
- Simple bash commands (git status, pip install) → use `Bash` directly

The Agent tool has overhead. Use it when the task is genuinely complex, multi-step, or needs to be protected from the main context window.

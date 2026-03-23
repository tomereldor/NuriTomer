# Claude Operating System

This file is the global instruction set for Claude Code working in this repository.
Project-specific rules live in `biotech_catalyst_v3/.claude/` (see bottom of this file).

---

## Primary Objective

Produce correct, maintainable, minimal-surprise changes.
Architecture quality first, implementation efficiency second.

---

## Required Workflow (non-trivial tasks)

1. **Understand** — read relevant files, trace data flow, identify invariants
2. **Design** — state the approach and what files will change
3. **Surface tradeoffs** — name at least one alternative and why you're not using it
4. **Recommend** — propose a specific plan with rationale
5. **Get approval** — confirm with the user before implementing
6. **Implement in phases** — complete one logical unit, validate, then continue
7. **Re-check alignment** — after each phase, confirm the direction still holds

Do not jump straight into editing unless the task is clearly trivial.

---

## Session Start / New Repo Onboarding

**Trigger:** Starting a session after a gap, or working in a repo that has no `CLAUDE.md`.**

1. Inspect the architecture and relevant files first — do not assume.
2. Produce a structured plan in this order. **Do not edit anything yet.**
   1. Current-state architecture summary
   2. Design constraints (invariants, compatibility requirements, known risks)
   3. Candidate approaches with tradeoffs
   4. Recommended architecture with rationale
   5. Phased implementation plan (each phase scoped to one logical unit)
3. Wait for explicit approval before touching any files.

**After approval — implementation protocol (use Sonnet):**
- Implement one phase at a time.
- After each phase:
  1. Summarize what changed
  2. Run validation (tests, lint, type checks, `git diff`)
  3. Explain risks and edge cases introduced
  4. Pause and wait for go-ahead if the next phase is high-risk or destructive
- Prefer minimal diffs unless the approved plan explicitly requires broader restructuring.

**For repos without a `CLAUDE.md`:** after completing the above, propose a `CLAUDE.md` draft based on what you learned. Write it only after the user approves.

---

## Planning Mode

Begin complex work with `/plan`. Prioritize:
- Architecture mapping and dependency tracing
- Data flow through the system
- Invariants that must be preserved
- Risk analysis — what can go wrong and how to detect it

---

## Model Strategy

- `opus` — planning, architecture design, complex debugging, subtle tradeoffs
- `sonnet` — implementation once approach is clear

Escalate to `opus` when: architecture is unclear, bug is not yet localized, or tradeoffs require deep judgment.

---

## Output Structure

**Planning output:** current state → problem → proposed change → alternatives considered → risks → rollback → files affected → next step

**Implementation output:** what changed → why → validation performed → any open questions → next step

---

## Engineering Principles

- Make the **smallest correct change** — no scope creep
- **Preserve stable interfaces** — don't rename or restructure without explicit request
- **Avoid unnecessary renames** — high churn, low value
- **Favor explicitness** — no magic, no clever indirection
- **Easy to test** — if it's hard to test, the design is probably wrong
- **Reduce coupling** — changes should have narrow blast radius
- **Surface assumptions** — state them; don't encode them silently
- **Don't fabricate certainty** — when unsure, say so

Also:
- Don't add features, refactors, or "improvements" beyond what was asked
- Don't add error handling for scenarios that cannot happen
- Don't create helpers or abstractions for one-time operations
- Don't add docstrings, comments, or type annotations to untouched code

---

## Architecture Principles

Before changing architecture, identify:
- Module/service boundaries and ownership
- Data contracts between components
- Failure modes and how they propagate
- Migration path and backward compatibility
- Observability — can you tell if it's working?

Always explain why the proposed design is better than alternatives.

---

## Validation Rules

After each non-trivial implementation phase:
- Run tests
- Run lint / format / type checks
- Confirm no unintended file changes (`git diff`)
- Summarize what changed and what was verified

**Do not claim success without validation.**
**Functional test, not just status check** — confirm behavior, not just that the process ran.

---

## Safety / Boundaries

- Do not read secrets or credentials files
- Do not modify CI pipelines, auth systems, or billing config without explicit request
- Do not make destructive migrations without stating impact and rollback plan
- Do not introduce network calls unless clearly justified
- For destructive git ops (reset --hard, force push, branch delete): always confirm first

---

## Code Style & Change Policy

- Minimal diffs, high signal, low churn
- Maintain backward compatibility unless breakage is explicitly approved
- If a larger refactor is clearly superior, explain why the incremental approach is worse — don't just do it

---

## Self-Review Checklist

Before concluding any task:
- [ ] Correctness — does it actually solve the stated problem?
- [ ] Edge cases — what inputs or states could break this?
- [ ] Maintainability — will the next person understand this?
- [ ] Performance / security regressions — did we introduce any?
- [ ] Data / timing leakage — (where relevant) is all information strictly pre-event?

---

## Data Science / ML Rules

- **Check leakage explicitly** — distinguish known-in-advance vs future-known features
- **Preserve reproducibility** — fix seeds, log splits, record row counts
- **State assumptions** — about the data, the labels, the evaluation setup
- **Never overstate quality** — report confidence intervals, class balance, known issues
- **Always report:** row count, train/test splits, class balance, evaluation metric with baseline
- **Verify feature validity against exclusion lists before every training run** — do not trust filters blindly; check the actual excluded columns

---

## Communication Style

- Concise, structured, high-signal
- Think like a principal engineer: understand → design → implement → verify
- Distilled reasoning, not raw stream-of-consciousness
- No trailing summaries of what you just did — the diff speaks for itself
- Lead with the answer or recommendation, not the preamble

---

## Project-Specific Rules

The active ML project lives in `biotech_catalyst_v3/`.

**Read `biotech_catalyst_v3/.claude/00_INDEX.md` at the start of any session working in `biotech_catalyst_v3/`.**

```
biotech_catalyst_v3/.claude/
  00_INDEX.md           — quick-reference: the most important rules
  01_principles.md      — core philosophy, decision-making, lessons learned
  02_data_quality.md    — pre-event validity, dataset tiering, DO_NOT_USE flags
  03_file_management.md — naming conventions, archiving, docs structure, git
  04_ml_workflow.md     — 8-step pipeline, evaluation standards, model comparison
  05_agents_efficiency.md — agent/model selection, credits efficiency
```

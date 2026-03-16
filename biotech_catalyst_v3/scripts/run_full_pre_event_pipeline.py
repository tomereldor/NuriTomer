"""
run_full_pre_event_pipeline.py
==============================
One-command orchestration for the full pre-event ML pipeline.

Runs all 8 steps in order, each as a subprocess, with per-step timing and
immediate failure on any non-zero exit code (fail-fast).

Usage (from biotech_catalyst_v3/):
    python -m scripts.run_full_pre_event_pipeline

    # Skip steps already done (resume from step N):
    python -m scripts.run_full_pre_event_pipeline --start-step 4

    # Dry run — print commands only, do not execute:
    python -m scripts.run_full_pre_event_pipeline --dry-run

    # Override master CSV (default: enriched_all_clinical_clean_v3.csv):
    python -m scripts.run_full_pre_event_pipeline --master enriched_all_clinical_clean_v3.csv

Steps:
    1  prepare_ml_dataset             initial feature table from master CSV
    2  add_high_signal_features       Pass-4: pivotal proxy, trial design, therapeutic class
    3  refresh_ctgov_features         Pass-6: CT.gov timing (11 features per NCT ID)
    4  build_ctgov_pipeline_proxies   Pass-7: CT.gov sponsor + drug aggregate features (8)
    5  add_pre_event_timing_features  Pass-5: imminence, recency, sequence features
    6  add_oncology_timing_interactions  Pass-8: oncology × timing interaction features (4)
    7  build_pre_event_train_v2       build train/val/test table from feature dataset
    8  train_pre_event_v3             train LogReg + LightGBM + XGBoost, report, plots

Notes:
    - Step 3 (refresh_ctgov_features) calls the CT.gov API and may take 20–40 min
      for ~2500 rows. Caches results in ctgov_cache/ — safe to re-run.
    - Step 4 (build_ctgov_pipeline_proxies) similarly calls CT.gov API; cached.
    - Steps 1–2, 5–6 are local transforms: fast (~30s each).
    - Step 7 (build train table) takes ~2 min.
    - Step 8 (train) takes ~5 min (3 models × 5-fold CV).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

STEPS = [
    {
        "num": 1,
        "module": "scripts.prepare_ml_dataset",
        "label": "prepare_ml_dataset",
        "description": "Build initial feature table from master CSV",
        "extra_args": [],          # will be set at runtime if --master is passed
        "slow": False,
    },
    {
        "num": 2,
        "module": "scripts.add_high_signal_features",
        "label": "add_high_signal_features",
        "description": "Pass-4: pivotal proxy, trial design, therapeutic class",
        "extra_args": [],
        "slow": False,
    },
    {
        "num": 3,
        "module": "scripts.refresh_ctgov_features",
        "label": "refresh_ctgov_features",
        "description": "Pass-6: CT.gov timing features (API, ~20–40 min for v3)",
        "extra_args": [],
        "slow": True,
    },
    {
        "num": 4,
        "module": "scripts.build_ctgov_pipeline_proxies",
        "label": "build_ctgov_pipeline_proxies",
        "description": "Pass-7: CT.gov sponsor/drug pipeline proxy features (API)",
        "extra_args": [],
        "slow": True,
    },
    {
        "num": 5,
        "module": "scripts.add_pre_event_timing_features",
        "label": "add_pre_event_timing_features",
        "description": "Pass-5: imminence flags, recency bucket, sequence features",
        "extra_args": [],
        "slow": False,
    },
    {
        "num": 6,
        "module": "scripts.add_oncology_timing_interactions",
        "label": "add_oncology_timing_interactions",
        "description": "Pass-8: oncology × timing interaction features (4)",
        "extra_args": [],
        "slow": False,
    },
    {
        "num": 7,
        "module": "scripts.build_pre_event_train_v2",
        "label": "build_pre_event_train_v2",
        "description": "Build train/val/test table from latest feature dataset",
        "extra_args": [],
        "slow": False,
    },
    {
        "num": 8,
        "module": "scripts.train_pre_event_v3",
        "label": "train_pre_event_v3",
        "description": "Train LogReg + LightGBM + XGBoost, CV, report, plots",
        "extra_args": [],
        "slow": False,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def run_pipeline(start_step: int = 1, dry_run: bool = False, master: str = None):
    base_dir = Path(__file__).parent.parent  # biotech_catalyst_v3/
    python = sys.executable

    if master:
        # Inject --input flag for step 1 only
        for step in STEPS:
            if step["num"] == 1:
                step["extra_args"] = ["--input", master]

    steps_to_run = [s for s in STEPS if s["num"] >= start_step]
    total = len(steps_to_run)

    print(f"\n{'='*60}")
    print(f"  Pre-event pipeline  ({total} steps, start={start_step})")
    if dry_run:
        print("  DRY RUN — commands printed, not executed")
    print(f"{'='*60}\n")

    pipeline_start = time.time()
    results = []

    for i, step in enumerate(steps_to_run, 1):
        cmd = [python, "-m", step["module"]] + step["extra_args"]
        cmd_str = " ".join(cmd)
        label = f"[{i}/{total}] Step {step['num']}: {step['label']}"

        print(f"\n{'-'*60}")
        print(f"  {label}")
        print(f"  {step['description']}")
        if step["slow"]:
            print(f"  ⚠ API step — may take 20–40 min (cached)")
        print(f"  cmd: python -m {step['module']}")
        if dry_run:
            results.append((step["num"], step["label"], "DRY_RUN", 0.0))
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(base_dir))
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        results.append((step["num"], step["label"], status, elapsed))
        print(f"\n  → {status}  ({_fmt_elapsed(elapsed)})")

        if result.returncode != 0:
            print(f"\n  ✗ Pipeline aborted at step {step['num']}: {step['label']}")
            print(f"    Fix the error above and re-run with --start-step {step['num']}")
            _print_summary(results, time.time() - pipeline_start)
            sys.exit(1)

    _print_summary(results, time.time() - pipeline_start)


def _print_summary(results, total_elapsed):
    print(f"\n{'='*60}")
    print(f"  Pipeline summary")
    print(f"{'='*60}")
    for num, label, status, elapsed in results:
        marker = "✓" if status in ("OK", "DRY_RUN") else "✗"
        elapsed_str = _fmt_elapsed(elapsed) if elapsed > 0 else "—"
        print(f"  {marker} Step {num}: {label:<40} {status:<12} {elapsed_str}")
    print(f"\n  Total: {_fmt_elapsed(total_elapsed)}")
    all_ok = all(s in ("OK", "DRY_RUN") for _, _, s, _ in results)
    if all_ok:
        print("  Status: COMPLETE")
    else:
        print("  Status: FAILED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full pre-event ML pipeline end-to-end."
    )
    parser.add_argument(
        "--start-step", type=int, default=1, metavar="N",
        help="Start from step N (1–8). Use to resume after a failure."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing."
    )
    parser.add_argument(
        "--master", type=str, default=None,
        help="Override master CSV filename passed to prepare_ml_dataset (step 1)."
    )
    args = parser.parse_args()

    if not 1 <= args.start_step <= len(STEPS):
        parser.error(f"--start-step must be 1–{len(STEPS)}")

    run_pipeline(
        start_step=args.start_step,
        dry_run=args.dry_run,
        master=args.master,
    )


if __name__ == "__main__":
    main()

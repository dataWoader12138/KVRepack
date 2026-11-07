#!/usr/bin/env python3
"""Plan paired Spot/MMCoQA Random-head experiments; use --execute to run them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvrepack.experiments.random_head_images import (
    DEFAULT_HEAD_SEEDS, DEFAULT_PRETRAINED, METHODS, REFERENCE_COMMIT, TASKS, build_matrix,
)
from kvrepack.experiments.runner import ExperimentPlan, run_plan


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def execute_matrix(plans: Sequence[ExperimentPlan], output_root: str | Path, *,
                   runner: Callable | None = None, reporter: Callable | None = None) -> dict:
    """Execute one isolated method/task at a time; stop at the first failure."""
    if not plans:
        raise ValueError("No experiment plans were supplied")
    output = Path(output_root).expanduser().resolve()
    expected = {str(output / plan.resolved_config["public_method"] / plan.resolved_config["task"])
                for plan in plans}
    if len(expected) != len(plans) or expected != {str(plan.output) for plan in plans}:
        raise ValueError("Plan output directories do not belong to the selected experiment group")
    # No resume or implicit reuse: this exclusive mkdir also prevents two
    # controllers from using the same experiment group concurrently.
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "controller_state.json"
    state = {
        "schema_version": 1, "status": "starting", "reference_commit": REFERENCE_COMMIT,
        "started_utc": _timestamp(), "planned_runs": len(plans), "completed_runs": 0,
        "runs": [{"method": plan.resolved_config["public_method"], "task": plan.resolved_config["task"],
                  "output": str(plan.output), "status": "pending"} for plan in plans],
    }
    _write_state(state_path, state)
    launch = runner or run_plan
    try:
        if reporter is None:
            from kvrepack.ablations.milebench_head_selection_report import summarize_run
            reporter = summarize_run
        for index, plan in enumerate(plans):
            row = state["runs"][index]
            state.update(status="generating", current_run=index, updated_utc=_timestamp())
            row["status"] = "generating"
            _write_state(state_path, state)
            code = launch(plan)
            row["returncode"] = code
            if code != 0:
                raise RuntimeError(f"Generation returned {code} for {row['method']}/{row['task']}; see {plan.output / 'run.log'}")
            state.update(status="scoring", updated_utc=_timestamp())
            row["status"] = "scoring"
            _write_state(state_path, state)
            reporter(plan.output, plan.output / "report")
            row.update(status="complete", report=str(plan.output / "report"), finished_utc=_timestamp())
            state["completed_runs"] += 1
            _write_state(state_path, state)
        state.update(status="complete", finished_utc=_timestamp())
        _write_state(state_path, state)
        return state
    except BaseException as error:
        if "current_run" in state:
            state["runs"][state["current_run"]]["status"] = "failed"
        state.update(status="failed", error_type=type(error).__name__, finished_utc=_timestamp())
        _write_state(state_path, state)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True, help="new experiment-group directory; never reused")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--hf-home", help="explicit model/data cache root")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--head-seeds", nargs="+", type=int, default=DEFAULT_HEAD_SEEDS)
    parser.add_argument("--seed", type=int, default=42, help="fixed model RNG seed, separate from selector seeds")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="original input prefix (default 200 unless --all-samples)")
    selection.add_argument("--all-samples", action="store_true", help="read every annotation row; no sample cap")
    parser.add_argument("--budget-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--execute", action="store_true", help="start model generation and offline scoring; default only prints plans")
    args = parser.parse_args(argv)
    try:
        plans = build_matrix(
            args.data_root, args.output_root, methods=args.methods, tasks=args.tasks,
            pretrained=args.pretrained, hf_home=args.hf_home, head_seeds=args.head_seeds,
            seed=args.seed, limit=args.limit, all_samples=args.all_samples, budget_ratio=args.budget_ratio,
            device=args.device, max_new_tokens=args.max_new_tokens,
        )
        if args.execute:
            state = execute_matrix(plans, args.output_root)
            print(json.dumps(state, indent=2, ensure_ascii=False))
        else:
            print(json.dumps({
                "benchmark": "milebench", "ablation": "random_head", "executed": False,
                "reference_commit": REFERENCE_COMMIT, "planned_runs": len(plans),
                "requested_samples_per_run": None if args.all_samples else (200 if args.limit is None else args.limit),
                "sample_selection": "complete_annotation" if args.all_samples else "original_prefix",
                "selected_samples_by_run": [
                    {"method": plan.resolved_config["public_method"], "task": plan.resolved_config["task"],
                     "sample_count": plan.resolved_config["limit"]} for plan in plans
                ],
                "requested_total_generations": sum(plan.resolved_config["limit"] * len(plan.resolved_config["ablation"]["conditions"]) for plan in plans),
                "experiments": [plan.to_dict() for plan in plans],
            }, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"MileBench Random-head failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

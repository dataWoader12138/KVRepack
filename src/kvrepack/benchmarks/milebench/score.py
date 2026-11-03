"""Explicit task coverage and percent-scale MileBench result aggregation.

The source score.py exported each primary metric multiplied by 100. This module
retains that scale and adds declared expected tasks and missing/partial status.
It never selects a latest run silently when multiple runs exist for one task.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


def summarize(task_results: Mapping[str, dict], expected_tasks: Sequence[str]) -> dict:
    """Summarize an explicitly selected experiment group.

    Values may be raw eval.json metric mappings or evaluate_predictions results.
    Macro averaging includes only available tasks, never zero-fills missing ones.
    ``complete`` describes task availability; ``full_dataset_complete`` is only
    true when all tasks also carry full-sample-count evidence.
    """
    expected = list(expected_tasks)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_tasks must be a nonempty unique task list")
    unexpected = sorted(set(task_results) - set(expected))
    if unexpected:
        raise ValueError(f"results contain tasks outside this experiment group: {unexpected}")
    completed = [task for task in expected if task in task_results]
    missing = [task for task in expected if task not in task_results]
    rows, values, partial, unverified = [], [], [], []
    for task in completed:
        record = task_results[task]
        metrics = record.get("eval", record)
        primary = [key for key in ("Accuracy", "Rouge-L f") if key in metrics]
        if len(primary) != 1:
            raise ValueError(f"{task}: expected exactly one primary metric (Accuracy or Rouge-L f)")
        metric = primary[0]
        raw = metrics[metric]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or not 0 <= raw <= 1:
            raise ValueError(f"{task}: primary metric must be a finite 0–1 value, got {raw!r}")
        percent = float(raw) * 100
        full_task = record.get("full_task") if "eval" in record else None
        if full_task is False:
            partial.append(task)
        elif full_task is not True:
            unverified.append(task)
        rows.append({"task": task, "metric": metric, "value": float(raw), "percent": percent,
                     "sample_count": record.get("sample_count") if "eval" in record else None,
                     "full_task": full_task})
        values.append(percent)
    full = not missing and not partial and not unverified
    average = sum(values) / len(values) if values else None
    return {"schema_version": 1, "metric_scale": "percent", "expected_tasks": expected,
            "completed_tasks": completed, "missing_tasks": missing, "partial_tasks": partial,
            "unverified_sample_count_tasks": unverified, "complete": not missing,
            "full_dataset_complete": full, "macro_average_scope": "completed_tasks_only",
            "macro_average_percent": average,
            "full_dataset_macro_average_percent": average if full else None, "tasks": rows}


def summarize_runs(run_directories: Mapping[str, str | os.PathLike], expected_tasks: Sequence[str]) -> dict:
    """Collect exactly the selected run for each task, not unrelated old results.

    Each directory may be a scoring directory or a generation run containing a
    scoring/ directory. No result means missing; more than one eval is an error.
    """
    unexpected = sorted(set(run_directories) - set(expected_tasks))
    if unexpected:
        raise ValueError(f"run directories contain tasks outside this experiment group: {unexpected}")
    records = {}
    for task, directory in run_directories.items():
        path = Path(directory)
        candidates = [candidate for candidate in (path / "eval.json", path / "scoring" / "eval.json")
                      if candidate.is_file()]
        if len(candidates) > 1:
            raise ValueError(f"{task}: multiple evaluations found; select one scoring directory explicitly")
        if not candidates:
            continue
        evaluation_path = candidates[0]
        record = {"eval": json.loads(evaluation_path.read_text(encoding="utf-8"))}
        metadata_path = evaluation_path.parent / "evaluation_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("task") != task:
                raise ValueError(f"task metadata mismatch at {metadata_path}")
            record.update({key: metadata.get(key) for key in ("sample_count", "full_task")})
        records[task] = record
    result = summarize(records, expected_tasks)
    result["selected_run_directories"] = {task: str(Path(path).expanduser().resolve())
                                          for task, path in run_directories.items()}
    return result


def summarize_directory(directory: str | os.PathLike, expected_tasks: Sequence[str]) -> dict:
    """Read a single group's <directory>/<task>/{eval.json,scoring/eval.json}."""
    root = Path(directory)
    return summarize_runs({task: root / task for task in expected_tasks}, expected_tasks)


def write_summary(result: dict, output_dir: str | os.PathLike) -> Path:
    # Reuse the same source-tree protection as offline scoring, imported lazily.
    from .evaluate import _output_path
    output = _output_path(output_dir)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    with (output / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        columns = ["task", "status", "metric", "value", "percent", "sample_count", "full_task"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        available = {row["task"]: row for row in result["tasks"]}
        for task in result["expected_tasks"]:
            row = available.get(task)
            if row is None:
                writer.writerow({"task": task, "status": "missing"})
            else:
                status = "partial" if row["full_task"] is False else "completed" if row["full_task"] is True else "sample_count_unverified"
                writer.writerow({**row, "status": status})
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize one explicitly selected MileBench experiment group")
    parser.add_argument("--tasks", nargs="+", required=True, help="expected task names, including any missing tasks")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--results-root", help="directory containing one selected run directory per task")
    sources.add_argument("--run", action="append", metavar="TASK=DIRECTORY", help="repeat to choose exact per-task generation/scoring runs")
    parser.add_argument("--output", required=True, help="new summary directory")
    args = parser.parse_args(argv)
    try:
        if args.results_root:
            result = summarize_directory(args.results_root, args.tasks)
        else:
            directories = {}
            for item in args.run:
                task, separator, path = item.partition("=")
                if not separator or not path or task in directories:
                    raise ValueError("--run must specify unique TASK=DIRECTORY pairs")
                directories[task] = path
            result = summarize_runs(directories, args.tasks)
        write_summary(result, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"milebench summary failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

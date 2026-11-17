#!/usr/bin/env python3
"""Plan or execute all full Qwen7B styled image/E2E comparisons sequentially."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kvrepack.ablations.styled_milebench_worker import ARMS, TASKS, arm_spec
from kvrepack.ablations.run_head_selection import atomic_json, timestamp
from kvrepack.benchmarks.milebench.dataset import annotation_path, load_annotation


def planned_runs(args):
    result = []
    for arm in args.arms:
        for task in args.tasks:
            annotation = load_annotation(args.data_root, task, limit=None)
            count = len(annotation["data"])
            if args.limit is not None and not 1 <= args.limit <= count:
                raise ValueError("Explicit prefix does not fit the complete annotation")
            result.append(dict(arm=arm, task=task, output=str(args.output / arm / task),
                               target_samples=count if args.limit is None else args.limit,
                               annotation_total_samples=count, all_samples=args.limit is None,
                               annotation_path=str(annotation_path(args.data_root, task)), status="pending"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "pretrained", "hf-home", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit", type=int, help="optional explicit smoke prefix; default all annotation rows")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.max_new_tokens < 1:
        parser.error("Warmup must be nonnegative and max-new-tokens positive")
    if len(set(args.arms)) != len(args.arms) or len(set(args.tasks)) != len(args.tasks):
        parser.error("Arms and tasks must be unique")
    for name in ("data_root", "pretrained", "hf_home", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    runs = planned_runs(args)
    if not args.execute:
        print(json.dumps(dict(executed=False, planned_runs=len(runs), arms=args.arms, tasks=args.tasks,
            planned_generations=sum(row["target_samples"] for row in runs),
            sample_selection="complete_annotation" if args.limit is None else "explicit_prefix", runs=runs),
            ensure_ascii=False, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, status="starting", source_commit="b9104d8", arms=args.arms,
        tasks=args.tasks, planned_runs=len(runs), completed_runs=0, runs=runs,
        warmup_per_arm=args.warmup, max_new_tokens=args.max_new_tokens,
        all_samples=args.limit is None, started_utc=timestamp(), controller_pid=os.getpid(),
        process_group=os.getpgrp(), target_generations=sum(row["target_samples"] for row in runs))
    state_path = args.output / "experiment_state.json"
    atomic_json(state_path, state)
    child = None

    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=8)
        state.update(status="interrupted", signal=signum, finished_utc=timestamp())
        atomic_json(state_path, state)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT / "vendor/lmms-eval"),
        PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1")
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    begin = time.perf_counter()
    try:
        for index, row in enumerate(runs):
            command = [sys.executable, "-B", "-m", "kvrepack.ablations.styled_milebench_worker",
                "--arm", row["arm"], "--task", row["task"], "--data-root", str(args.data_root),
                "--pretrained", str(args.pretrained), "--hf-home", str(args.hf_home),
                "--output", row["output"], "--warmup", str(args.warmup),
                "--max-new-tokens", str(args.max_new_tokens)]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            state.update(status="generating", current_run=index, command=command, updated_utc=timestamp())
            row["status"] = "running"
            with (args.output / f"{row['arm']}_{row['task']}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                state["worker_pid"] = child.pid
                atomic_json(state_path, state)
                code = child.wait()
            row["returncode"] = code
            if code:
                row["status"] = "failed"
                raise RuntimeError(f"Worker failed: {row['arm']}/{row['task']}")
            row.update(status="complete", finished_utc=timestamp())
            state["completed_runs"] += 1
            atomic_json(state_path, state)
        state.update(status="generated_all", updated_utc=timestamp())
        atomic_json(state_path, state)
        command = [sys.executable, "-B", str(ROOT / "scripts/summarize_styled_milebench.py"),
                   "--root", str(args.output), "--output", str(args.output / "report")]
        state.update(status="scoring", command=command)
        atomic_json(state_path, state)
        with (args.output / "scoring.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=""),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            if child.wait():
                raise RuntimeError("Independent image score/E2E report failed")
        state.update(status="complete", finished_utc=timestamp(), controller_seconds=time.perf_counter()-begin)
        atomic_json(state_path, state)
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

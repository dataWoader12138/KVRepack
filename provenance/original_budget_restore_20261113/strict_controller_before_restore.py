#!/usr/bin/env python3
"""Run the eight matched NExTQA merge conditions, then score them offline.

GPU processes are serialized. The four requested style conditions run first;
native/KVRepack controls then supply the missing comparable clean E2E records.
"""
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
sys.path.insert(0, str(ROOT/"src"))
from kvrepack.ablations.run_head_selection import atomic_json, timestamp

CASES = (("style", "snapkv"), ("style", "hybridkv"),
         ("baseline", "snapkv"), ("baseline", "hybridkv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pretrained", "hf-home", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(1, 200), required=True)
    parser.add_argument("--warmup", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("pretrained", "hf_home", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output/"experiment_state.json"
    if state_path.exists():
        if not args.resume:
            raise ValueError("Use a fresh output root")
        old = json.loads(state_path.read_text())
        if old.get("resume_authorized") is False or old.get("status") == "stopped_by_user":
            raise ValueError("Stopped experiments require explicit new authorization")
        if old.get("task") != "nextqa_oe_test" or old.get("docs_per_condition") != args.limit:
            raise ValueError("Task or sample count changed")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in
        ("src", "vendor/lmms-eval", "csrc")), PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix)/"lib")+os.pathsep+env.get("LD_LIBRARY_PATH", "")
    state = dict(status="starting", task="nextqa_oe_test", started_utc=timestamp(),
        controller_pid=os.getpid(), process_group=os.getpgrp(), docs_per_condition=args.limit,
        target_generations=8*args.limit, warmup_generations=8*args.warmup,
        mode="formal" if args.limit == 200 else "pilot_not_formal",
        completed_cases=[], resume_authorized=True, pretrained=str(args.pretrained),
        hf_home=str(args.hf_home), report_directory=str(args.output/"report"))
    atomic_json(state_path, state)
    started = time.perf_counter()
    child = None

    def interrupted(signum, _frame):
        # A signal sent only to the controller must also release its GPU child.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=8)
        case_state = args.output/state.get("current_case", "")/"run_state.json"
        if state.get("status") == "generating" and case_state.is_file():
            prior = json.loads(case_state.read_text())
            if prior.get("status") != "complete":
                prior.update(status="stopped_by_user", resume_authorized=False,
                             signal=signum, finished_utc=timestamp())
                atomic_json(case_state, prior)
        state.update(status="stopped_by_user", resume_authorized=False,
                     signal=signum, finished_utc=timestamp())
        atomic_json(state_path, state)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for kind, backbone in CASES:
            case = args.output/kind/backbone
            if kind == "style":
                worker, selector, value = "run_style_comparison", "--method", backbone+"_kvrepack"
            else:
                worker, selector, value = "run_baseline_completion", "--backbone", backbone
            command = [sys.executable, "-B", "-m", "kvrepack.ablations."+worker,
                selector, value, "--task", "nextqa_oe_test", "--pretrained", str(args.pretrained),
                "--hf-home", str(args.hf_home), "--output", str(case),
                "--limit", str(args.limit), "--warmup", str(args.warmup)]
            if args.resume and (case/"run_contract.json").exists():
                prior = json.loads((case/"run_state.json").read_text())
                if prior.get("status") == "complete":
                    if prior.get("completed_docs") != args.limit:
                        raise ValueError("Completed case has a different sample count")
                    state["completed_cases"].append(kind+"/"+backbone)
                    continue
                command.append("--resume")
            state.update(status="generating", current_case=kind+"/"+backbone,
                         command=command, updated_utc=timestamp())
            with (args.output/(kind+"_"+backbone+".log")).open("a" if args.resume else "x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                state["worker_pid"] = child.pid
                atomic_json(state_path, state)
                if child.wait():
                    raise RuntimeError(kind+"/"+backbone+" generation failed")
            state["completed_cases"].append(kind+"/"+backbone)
            atomic_json(state_path, state)
        command = [sys.executable, "-B", "-m", "kvrepack.ablations.nextqa_merge_report",
            "--root", str(args.output), "--expected-count", str(args.limit),
            "--hf-home", str(args.hf_home), "--output", str(args.output/"report")]
        state.update(status="scoring", command=command, updated_utc=timestamp())
        atomic_json(state_path, state)
        with (args.output/"scoring.log").open("a" if args.resume else "x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=""),
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state["worker_pid"] = child.pid
            atomic_json(state_path, state)
            if child.wait():
                raise RuntimeError("Offline NExTQA scoring failed")
        state.update(status="complete", finished_utc=timestamp(),
                     controller_seconds=time.perf_counter()-started)
        atomic_json(state_path, state)
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     finished_utc=timestamp())
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

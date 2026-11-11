#!/usr/bin/env python3
"""Run the explicitly scoped VATEX fixed-group merge/latency matrix serially."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from kvrepack.ablations.run_head_selection import atomic_json, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pretrained", "hf-home", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--limit", type=int, choices=(1, 2, 200), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("hybridkv_kvrepack", "snapkv_kvrepack"),
                        default=["snapkv_kvrepack", "hybridkv_kvrepack"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output/"experiment_state.json"
    if state_path.exists() and not args.resume:
        raise ValueError("Use a fresh output directory or explicit --resume")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in
        ("src", "vendor/lmms-eval", "csrc")), PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix)/"lib")+os.pathsep+env.get("LD_LIBRARY_PATH", "")
    state = dict(status="starting", started_utc=timestamp(), controller_pid=os.getpid(),
        process_group=os.getpgrp(), methods=args.methods, docs_per_arm=args.limit,
        target_generations=3*args.limit*len(args.methods), completed_methods=[])
    atomic_json(state_path, state)
    began = time.perf_counter()
    try:
        for method in args.methods:
            case = args.output/method
            command = [sys.executable, "-B", "-m", "kvrepack.ablations.run_merge_comparison",
                "--method", method, "--pretrained", str(args.pretrained.resolve()),
                "--hf-home", str(args.hf_home.resolve()), "--output", str(case),
                "--limit", str(args.limit), "--warmup", str(args.warmup)]
            if args.audit:
                command.append("--audit")
            if args.resume and (case/"run_contract.json").exists():
                saved = json.loads((case/"run_state.json").read_text())
                if saved.get("status") == "complete":
                    previous = json.loads((case/"run_contract.json").read_text())
                    expected = dict(method=method, target_docs=args.limit, diagnostic_audit=args.audit,
                        warmup_per_arm=args.warmup, model_snapshot=str(args.pretrained.resolve()),
                        hf_home=str(args.hf_home.resolve()))
                    if any(previous.get(k) != v for k,v in expected.items()):
                        raise ValueError("Completed run specification changed")
                    if saved.get("completed_docs") != args.limit or len(list((case/"samples").glob("*.json"))) != args.limit:
                        raise ValueError("Completed state lacks its committed samples")
                    state["completed_methods"].append(method)
                    continue
                if saved.get("resume_authorized") is False or saved.get("status") == "stopped_by_user":
                    raise ValueError("A stopped run cannot be resumed automatically")
                command.append("--resume")
            state.update(status="generating", current_method=method, command=command)
            atomic_json(state_path, state)
            with (args.output/(method+".log")).open("a" if args.resume else "x") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                state["worker_pid"] = child.pid
                atomic_json(state_path, state)
                if child.wait():
                    raise RuntimeError(f"{method} failed; see worker log")
            state["completed_methods"].append(method)
            atomic_json(state_path, state)
        state.update(status="complete", finished_utc=timestamp(),
                     controller_seconds=time.perf_counter()-began)
        atomic_json(state_path, state)
        return 0
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

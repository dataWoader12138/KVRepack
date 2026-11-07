#!/usr/bin/env python3
"""Run the 200-input HybridKV-KVRepack Random-head ablation, then score it."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kvrepack.ablations.run_head_selection import atomic_json, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_name(output.name + "_controller_state.json")
    log_path = output.with_name(output.name + "_worker.log")
    score_log = output.with_name(output.name + "_scoring.log")
    state = dict(status="starting", task="nextqa_oe_test", method="hybridkv_kvrepack", target_docs=200,
                 target_generations=800, conditions=["similarity", "random_global_seed0",
                 "random_global_seed1", "random_global_seed2"], started_utc=timestamp(),
                 controller_pid=os.getpid(), process_group=os.getpgrp(),
                 worker_log=str(log_path), score_log=str(score_log), output=str(output))
    atomic_json(state_path, state)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    runtime = str(Path(sys.prefix) / "lib")
    environment["LD_LIBRARY_PATH"] = runtime + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    worker = [sys.executable, "-B", "-m", "kvrepack.ablations.run_hybrid_head_selection",
              "--limit", "200", "--pretrained", str(args.pretrained.expanduser().resolve()),
              "--hf-home", str(args.hf_home.expanduser().resolve()), "--output", str(output)]
    if args.resume:
        worker.append("--resume")
    try:
        with log_path.open("a" if args.resume else "x") as stream:
            child = subprocess.Popen(worker, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state.update(status="generating", worker_pid=child.pid, updated_utc=timestamp())
            atomic_json(state_path, state)
            code = child.wait()
            if code:
                raise RuntimeError(f"Generation worker exited {code}; see {log_path}")
        state.update(status="scoring", updated_utc=timestamp())
        atomic_json(state_path, state)
        score = [sys.executable, "-B", "-m", "kvrepack.ablations.hybrid_head_selection_report",
                 "--run-root", str(output), "--hf-home", str(args.hf_home.expanduser().resolve()),
                 "--output", str(output / "report"), "--limit", "200"]
        with score_log.open("a" if args.resume else "x") as stream:
            subprocess.run(score, cwd=ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
        state.update(status="complete", finished_utc=timestamp())
        atomic_json(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()

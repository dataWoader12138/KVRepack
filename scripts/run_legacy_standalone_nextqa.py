#!/usr/bin/env python3
"""Run only original standalone LOOK-M and AsymKV on NExTQA, then score E2E."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from kvrepack.ablations.run_head_selection import atomic_json,timestamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("pretrained","hf-home","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--limit",type=int,choices=(1,200),default=200)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    for name in ("pretrained","hf_home","output"):setattr(args,name,getattr(args,name).expanduser().resolve())
    if args.output==ROOT or any(args.output.is_relative_to(ROOT/p) for p in ("src","vendor","csrc","provenance")):
        raise ValueError("Use an experiment output directory outside source and provenance")
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/".controller.lock").open("a+")
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=args.output/"experiment_state.json"
    spec=dict(task="nextqa_oe_test",methods=["lookm","asymkv"],limit=args.limit,
        pretrained=str(args.pretrained),hf_home=str(args.hf_home),policy="original_standalone_algorithms")
    if state_path.exists():
        prior=json.loads(state_path.read_text())
        if not args.resume or prior.get("spec")!=spec:raise ValueError("Select a fresh output or matching explicit resume")
        if prior.get("resume_authorized") is False:raise ValueError("Stopped experiment cannot resume automatically")
    state=dict(status="starting",spec=spec,controller_pid=os.getpid(),process_group=os.getpgrp(),
        target_generations=2*args.limit,docs_per_method=args.limit,warmup_generations=0,
        started_utc=timestamp(),completed_methods=[],resume_authorized=True)
    atomic_json(state_path,state);began=time.perf_counter();child=None
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ("src","vendor/lmms-eval","csrc")),
        OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib")+os.pathsep+env.get("LD_LIBRARY_PATH","")

    def stop(signum,_frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=8)
        state.update(status="stopped_by_user",resume_authorized=False,signal=signum,finished_utc=timestamp())
        atomic_json(state_path,state)
        if state.get("current_method"):
            p=args.output/state["current_method"]/"run_state.json"
            if p.exists():
                saved=json.loads(p.read_text())
                if saved.get("status")!="complete":
                    saved.update(status="stopped_by_user",resume_authorized=False,finished_utc=timestamp());atomic_json(p,saved)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for method in ("lookm","asymkv"):
            case=args.output/method
            command=[sys.executable,"-B","-m","kvrepack.benchmarks.run_legacy_nextqa_e2e",
                "--method",method,"--pretrained",str(args.pretrained),"--hf-home",str(args.hf_home),
                "--output",str(case),"--limit",str(args.limit)]
            saved=json.loads((case/"run_state.json").read_text()) if (case/"run_state.json").exists() else {}
            if saved.get("status")=="complete" and args.resume:
                if saved.get("completed_docs")!=args.limit:raise ValueError("Completed case count changed")
                state["completed_methods"].append(method);continue
            if args.resume and (case/"run_contract.json").exists():command.append("--resume")
            state.update(status="generating",current_method=method,command=command,updated_utc=timestamp())
            with (args.output/(method+".log")).open("a" if args.resume else "x") as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state["worker_pid"]=child.pid;atomic_json(state_path,state)
                if child.wait():raise RuntimeError(method+" generation failed; inspect task log")
            state["completed_methods"].append(method);atomic_json(state_path,state)
        command=[sys.executable,"-B","-m","kvrepack.benchmarks.legacy_nextqa_report",
            "--run-root",str(args.output),"--hf-home",str(args.hf_home),"--output",str(args.output/"report"),"--limit",str(args.limit)]
        state.update(status="scoring",command=command,updated_utc=timestamp());atomic_json(state_path,state)
        with (args.output/"scoring.log").open("a" if args.resume else "x") as log:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=""),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state["worker_pid"]=child.pid;atomic_json(state_path,state)
            if child.wait():raise RuntimeError("Native scoring/E2E report failed")
        state.update(status="complete",finished_utc=timestamp(),controller_seconds=time.perf_counter()-began)
        atomic_json(state_path,state);return 0
    except Exception as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}",finished_utc=timestamp());atomic_json(state_path,state);raise


if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""Run only missing strict VATEX baselines, then join verified existing merge arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from kvrepack.ablations.run_head_selection import atomic_json,timestamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("pretrained","hf-home","output","style-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--limit",type=int,choices=(1,200),required=True)
    parser.add_argument("--warmup",type=int,choices=(1,2),default=2)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    for name in ("pretrained","hf_home","output","style_root"):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    state_path=args.output/"experiment_state.json"
    if state_path.exists() and not args.resume:raise ValueError("Use a fresh output root")
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if previous.get("resume_authorized") is False or previous.get("status")=="stopped_by_user":
            raise ValueError("Stopped runs require explicit new authorization")
    for method in ("snapkv_kvrepack","hybridkv_kvrepack"):
        source=json.loads((args.style_root/method/"run_state.json").read_text())
        if source.get("status")!="complete" or source.get("completed_docs")!=200:
            raise ValueError("Require the complete existing four merge arms")
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ("src","vendor/lmms-eval","csrc")),
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUDA_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib")+os.pathsep+env.get("LD_LIBRARY_PATH","")
    state=dict(status="starting",started_utc=timestamp(),controller_pid=os.getpid(),process_group=os.getpgrp(),
        task="vatex_test",new_target_generations=4*args.limit,existing_style_generations=4*args.limit,
        final_comparison_generations=8*args.limit,docs_per_condition=args.limit,
        warmup_generations=4*args.warmup,completed_backbones=[],style_root=str(args.style_root),
        pretrained=str(args.pretrained),hf_home=str(args.hf_home),resume_authorized=True)
    atomic_json(state_path,state);started=time.perf_counter()
    try:
        for backbone in ("snapkv","hybridkv"):
            case=args.output/backbone
            command=[sys.executable,"-B","-m","kvrepack.ablations.run_baseline_completion",
                "--backbone",backbone,"--pretrained",str(args.pretrained),"--hf-home",str(args.hf_home),
                "--output",str(case),"--limit",str(args.limit),"--warmup",str(args.warmup)]
            if args.resume and (case/"run_contract.json").exists():
                prior=json.loads((case/"run_state.json").read_text())
                if prior.get("status")=="complete":
                    if prior.get("completed_docs")!=args.limit:raise ValueError("Completed prefix differs")
                    state["completed_backbones"].append(backbone);continue
                command.append("--resume")
            state.update(status="generating",current_backbone=backbone,command=command,updated_utc=timestamp())
            with (args.output/(backbone+".log")).open("a" if args.resume else "x") as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state["worker_pid"]=child.pid;atomic_json(state_path,state)
                if child.wait():raise RuntimeError(f"{backbone} baseline generation failed")
            state["completed_backbones"].append(backbone);atomic_json(state_path,state)
        command=[sys.executable,"-B","-m","kvrepack.ablations.baseline_completion_report",
            "--run-root",str(args.output),"--style-root",str(args.style_root),
            "--output",str(args.output/"report"),"--limit",str(args.limit)]
        state.update(status="scoring",command=command,updated_utc=timestamp());atomic_json(state_path,state)
        with (args.output/"scoring.log").open("a" if args.resume else "x") as log:
            subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=""),stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status="complete",finished_utc=timestamp(),controller_seconds=time.perf_counter()-started)
        atomic_json(state_path,state);return 0
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}",finished_utc=timestamp())
        atomic_json(state_path,state);raise


if __name__=="__main__":raise SystemExit(main())

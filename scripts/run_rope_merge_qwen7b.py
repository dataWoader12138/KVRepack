#!/usr/bin/env python3
"""Run both K aggregation spaces for SnapKV/HybridKV and NextQA/VATEX."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kvrepack.ablations.run_head_selection import atomic_json,timestamp
from kvrepack.ablations.run_head_selection_full import output_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained',required=True,type=Path)
    parser.add_argument('--hf-home',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--limit',required=True,type=int,choices=(1,200))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    output=args.output.expanduser().resolve()
    with output_lock(output,resume=args.resume):
        paths=['scripts/run_rope_merge_qwen7b.py','src/kvrepack/ablations/run_rope_merge.py',
            'src/kvrepack/ablations/rope_merge.py','src/kvrepack/ablations/rope_merge_report.py',
            'src/kvrepack/qwen_model_snapkv_kvrepack.py','src/kvrepack/hybridkv_utils.py',
            'src/kvrepack/modeling_qwen2_5_vl.py','src/kvrepack/baseline_utils.py',
            'src/kvrepack/merge_score.py','src/kvrepack/configs/param_config.json',
            'src/kvrepack/data/head_score/qwen2_5.json']
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
        contract=dict(schema_version=1,experiment='rope_merge',limit=args.limit,
            target_generations=8*args.limit,pretrained=str(args.pretrained.expanduser().resolve()),
            hf_home=str(args.hf_home.expanduser().resolve()),sources_sha256=sources)
        contract_path=output/'run_contract.json'
        if args.resume:
            if json.loads(contract_path.read_text())!=contract:raise ValueError('Frozen matrix contract changed')
        else:atomic_json(contract_path,contract)
        state=dict(status='starting',started_utc=timestamp(),target_generations=8*args.limit,
            target_runs=4,completed_runs=[],limit=args.limit,controller_pid=os.getpid(),process_group=os.getpgrp())
        state_path=output/'experiment_state.json';atomic_json(state_path,state)
        environment=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
        environment['LD_LIBRARY_PATH']=str(Path(sys.prefix)/'lib')+os.pathsep+environment.get('LD_LIBRARY_PATH','')
        (output/'logs').mkdir(exist_ok=True)
        try:
            for task in ['nextqa_oe_test','vatex_test']:
                for method in ['snapkv_kvrepack','hybridkv_kvrepack']:
                    for path,digest in sources.items():
                        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                            raise RuntimeError('Frozen source/resource changed: '+path)
                    run=output/method/task
                    state.update(status='generating',current_method=method,current_task=task,updated_utc=timestamp())
                    atomic_json(state_path,state)
                    command=[sys.executable,'-B','-m','kvrepack.ablations.run_rope_merge','--method',method,
                        '--task',task,'--limit',str(args.limit),'--pretrained',contract['pretrained'],
                        '--hf-home',contract['hf_home'],'--output',str(run)]
                    if args.resume and (run/'run_contract.json').exists():command.append('--resume')
                    print(json.dumps(dict(stage='generating',method=method,task=task,limit=args.limit)),flush=True)
                    with (output/'logs'/f'{method}_{task}_worker.log').open('a' if args.resume else 'x') as stream:
                        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                        state['worker_pid']=child.pid;atomic_json(state_path,state)
                        if child.wait():raise RuntimeError('Generation failed; inspect '+str(run/'run_state.json'))
                    state.update(status='scoring',updated_utc=timestamp());atomic_json(state_path,state)
                    report=run/'report'
                    if not report.exists():
                        score=[sys.executable,'-B','-m','kvrepack.ablations.rope_merge_report','--run-root',str(run),
                            '--hf-home',contract['hf_home'],'--output',str(report),'--limit',str(args.limit)]
                        with (output/'logs'/f'{method}_{task}_scoring.log').open('a' if args.resume else 'x') as stream:
                            subprocess.run(score,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
                    summary=json.loads((report/'summary.json').read_text())
                    if summary.get('status')!='complete':raise RuntimeError('Incomplete scoring metrics for '+str(run))
                    state['completed_runs'].append(dict(method=method,task=task,report=str(report)))
                    atomic_json(state_path,state)
            state.update(status='complete',finished_utc=timestamp());atomic_json(state_path,state)
        except BaseException as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=timestamp())
            atomic_json(state_path,state);raise


if __name__=='__main__':main()

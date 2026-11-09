#!/usr/bin/env python3
"""Run only native LookM on the original video benchmark prefixes, then score."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kvrepack.ablations.run_head_selection import atomic_json,timestamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pretrained','hf-home','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tasks',nargs='+',choices=('nextqa_oe_test','vatex_test'),default=['nextqa_oe_test','vatex_test'])
    parser.add_argument('--limit',type=int,choices=(1,200),default=200)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    output=args.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    if output==ROOT or any(output.is_relative_to(ROOT/p) for p in ('src','vendor','csrc','provenance')):
        raise ValueError('Output must be outside source and published evidence')
    if len(set(args.tasks))!=len(args.tasks):raise ValueError('Duplicate tasks')
    with (output/'.controller.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.resume and any(p.name!='.controller.lock' for p in output.iterdir()):raise ValueError('Use a fresh output directory or explicit --resume')
        state_path=output/'lookm_state.json'
        prior=json.loads(state_path.read_text()) if state_path.exists() else {}
        if prior.get('resume_authorized') is False or prior.get('status')=='stopped_by_user':raise ValueError('Stopped run cannot be resumed automatically')
        spec=dict(method='lookm',tasks=args.tasks,limit=args.limit,budget_ratio=.1,pretrained=str(args.pretrained.resolve()),hf_home=str(args.hf_home.resolve()))
        if prior and prior['spec']!=spec:raise ValueError('Resume specification changed')
        state=dict(status='starting',spec=spec,started_utc=prior.get('started_utc',timestamp()),completed_tasks=[],controller_pid=os.getpid(),process_group=os.getpgrp())
        atomic_json(state_path,state)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ('src','vendor/lmms-eval','csrc')),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        env['LD_LIBRARY_PATH']=str(Path(sys.prefix)/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
        try:
            for task in args.tasks:
                case=output/task;common=['--run-root',str(case),'--hf-home',str(args.hf_home.resolve()),'--output',str(case/'report'),'--limit',str(args.limit)]
                existing=json.loads((case/'run_state.json').read_text()) if (case/'run_state.json').exists() else {}
                ready=existing.get('status')=='complete' and existing.get('completed_docs')==args.limit and existing.get('completed_generations')==args.limit
                if not ready:
                    command=[sys.executable,'-B','-m','kvrepack.benchmarks.run_video_comparison','--method','lookm','--task',task,'--limit',str(args.limit),'--pretrained',str(args.pretrained.resolve()),'--hf-home',str(args.hf_home.resolve()),'--output',str(case)]
                    if args.resume and (case/'run_contract.json').exists():command.append('--resume')
                    state.update(status='generating',current_task=task,updated_utc=timestamp());atomic_json(state_path,state)
                    with (output/(task+'.log')).open('a' if args.resume else 'x') as stream:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                        state['worker_pid']=child.pid;atomic_json(state_path,state)
                        if child.wait():raise RuntimeError('LookM generation failed; see task log')
                report=case/'report'
                try:scored=json.loads((report/'summary.json').read_text()).get('status')=='complete'
                except (OSError,ValueError):scored=False
                if not scored:
                    if report.exists():report.rename(case/('report_incomplete_'+timestamp().replace(':','')))
                    state.update(status='scoring',current_task=task,updated_utc=timestamp());atomic_json(state_path,state)
                    with (output/(task+'_scoring.log')).open('a' if args.resume else 'x') as stream:
                        result=subprocess.run([sys.executable,'-B','-m','kvrepack.benchmarks.single_video_report',*common],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    if result.returncode:raise RuntimeError('LookM scoring failed; see scoring log')
                state['completed_tasks'].append(task);atomic_json(state_path,state)
            state.update(status='complete',finished_utc=timestamp());atomic_json(state_path,state);return 0
        except BaseException as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=timestamp());atomic_json(state_path,state);raise


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Serial LongBench/VATEX raw-K top200 adjacent observations, then CPU report."""
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

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from kvrepack.observations.adjacent import atomic_json,timestamp


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','text-model','video-model','hf-home','text-reference','video-reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-count',type=int,choices=(2,200),default=200)
    p.add_argument('--device',default='0')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-report',action='store_true')
    args=p.parse_args(argv)
    output=args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        p.error('Use a fresh output directory or explicit --resume')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','no_report')}
    statepath=output/'experiment_state.json'
    if statepath.exists() and json.loads(statepath.read_text())['spec']!=spec:
        p.error('Refusing mixed-protocol resume')
    state=dict(status='running',spec=spec,pid=os.getpid(),started_utc=timestamp(),
               target_inputs=2*args.sample_count,completed_groups=[],gpu_execution=True)
    atomic_json(statepath,state)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(REPO/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    child=None
    started=time.monotonic()

    def stop(signum,frame):
        if child is not None and child.poll() is None:
            child.terminate();child.wait()
        state.update(status='stopped_by_user',signal=signum,finished_utc=timestamp(),elapsed_seconds=time.monotonic()-started)
        atomic_json(statepath,state)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for group in ('text','video'):
            dest=output/group
            command=[sys.executable,'-B','-m','kvrepack.observations.top_adjacent','--group',group,
                     '--pretrained',str(getattr(args,group+'_model').resolve()),
                     '--hf-home',str(args.hf_home.resolve()),'--reference-records',str(getattr(args,group+'_reference').resolve()),
                     '--sample-count',str(args.sample_count),'--device',args.device,'--output',str(dest)]
            if group=='text':command+=['--inputs',str(args.inputs.resolve())]
            if args.resume and dest.exists():command+=['--resume']
            state.update(current_group=group,command=command,updated_utc=timestamp())
            atomic_json(statepath,state)
            with (output/(group+'.log')).open('a') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['worker_pid']=child.pid;atomic_json(statepath,state)
                code=child.wait()
            if code:raise RuntimeError(f'{group} worker failed ({code}); see its log')
            state['completed_groups'].append(group);atomic_json(statepath,state)
        if not args.no_report:
            command=[sys.executable,'-B','-m','kvrepack.observations.top_adjacent_report',
                     '--text-input',str(output/'text/records.jsonl'),'--video-input',str(output/'video/records.jsonl'),
                     '--sample-count',str(args.sample_count),'--output',str(output/'report')]
            state.update(status='reporting',command=command);atomic_json(statepath,state)
            with (output/'report.log').open('a') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                if child.wait():raise RuntimeError('CPU report failed; GPU records preserved')
        state.update(status='complete',finished_utc=timestamp(),elapsed_seconds=time.monotonic()-started)
        atomic_json(statepath,state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
        return 0
    except Exception as error:
        state.update(status='failed',error=str(error),finished_utc=timestamp(),elapsed_seconds=time.monotonic()-started)
        atomic_json(statepath,state)
        raise


if __name__=='__main__':
    raise SystemExit(main())

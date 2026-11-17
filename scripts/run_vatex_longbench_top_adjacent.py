#!/usr/bin/env python3
"""Serial LongBench/VATEX raw-K top200 adjacent observations, then CPU report."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from kvrepack.observations.adjacent import atomic_json,timestamp,sha


def validate_completed_group(destination, group, args):
    """Reuse complete frozen observations without loading their model again."""
    destination=Path(destination)
    statepath=destination/'run_state.json'
    if not statepath.exists():
        return False
    state=json.loads(statepath.read_text())
    if state.get('status')!='complete':
        return False
    contractpath=destination/'contract.json'
    contract=json.loads(contractpath.read_text())
    if (state.get('completed')!=args.sample_count or state.get('target')!=args.sample_count
            or contract.get('sample_count')!=args.sample_count or contract.get('group')!=group
            or contract.get('statistics_kind')!='raw_k_top200_sequence_adjacent_v1'
            or contract.get('top_k')!=200 or contract.get('ranking_metric')!='k_raw_before_rope'
            or contract.get('doc_ids')!=list(range(args.sample_count))
            or Path(contract['pretrained']).resolve()!=getattr(args,group+'_model').resolve()
            or contract['reference_records_sha256']!=sha(getattr(args,group+'_reference'))
            or (group=='text' and contract['inputs_sha256']!=sha(args.inputs))):
        raise ValueError('Completed group protocol changed; refusing reuse')
    if not contract.get('sources_sha256') or not contract.get('model_metadata_sha256'):
        raise ValueError('Completed group is missing frozen sources/model metadata')
    for name,expected in contract['sources_sha256'].items():
        if sha(name)!=expected:
            raise ValueError('Completed group numerical source changed')
    for name,expected in contract['model_metadata_sha256'].items():
        if sha(Path(contract['pretrained'])/name)!=expected:
            raise ValueError('Completed group model metadata changed')
    for name,expected in contract['versions'].items():
        if importlib.metadata.version(name)!=expected:
            raise ValueError('Completed group dependency version changed')
    seen=set()
    for line in (destination/'records.jsonl').open():
        row=json.loads(line)
        if (not line.endswith('\n') or row.get('status')!='completed' or row.get('group')!=group
                or row.get('statistics_kind')!=contract['statistics_kind']
                or row.get('contract_sha256')!=sha(contractpath)
                or type(row.get('doc_id')) is not int or row['doc_id'] not in range(args.sample_count)
                or row['doc_id'] in seen):
            raise ValueError('Completed group records are incomplete or incompatible')
        selection=Path(row['selected_pairs_path'])
        if selection.is_absolute() or not (destination/selection).resolve().is_relative_to(destination.resolve()):
            raise ValueError('Selection path is outside the completed group')
        if sha(destination/selection)!=row['selected_pairs_sha256']:
            raise ValueError('Completed group selected pairs changed')
        seen.add(row['doc_id'])
    if seen!=set(range(args.sample_count)):
        raise ValueError('Completed group is missing input rows')
    return True


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
    previous=json.loads(statepath.read_text()) if statepath.exists() else None
    if previous and previous['spec']!=spec:
        p.error('Refusing mixed-protocol resume')
    state=dict(status='running',spec=spec,pid=os.getpid(),started_utc=timestamp(),
               target_inputs=2*args.sample_count,completed_groups=[],gpu_execution=True)
    if previous:
        state['original_started_utc']=previous.get('original_started_utc',previous['started_utc'])
        state['previous_attempts']=previous.get('previous_attempts',[])+[
            {k:previous.get(k) for k in ('pid','started_utc','finished_utc','elapsed_seconds','status','completed_groups')}]
    atomic_json(statepath,state)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(REPO/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    child=None
    started=time.monotonic()

    def stop(signum,frame):
        state.update(status='interrupted_by_signal',signal=signum,finished_utc=timestamp(),elapsed_seconds=time.monotonic()-started)
        atomic_json(statepath,state)
        if child is not None and child.poll() is None:
            child.terminate()
        # Do not wait here: the interrupted main wait may own _waitpid_lock.
        # Unwind it first; the outer exception handler reaps the child.
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for group in ('text','video'):
            dest=output/group
            if args.resume and validate_completed_group(dest,group,args):
                state['completed_groups'].append(group)
                state.setdefault('reused_completed_groups',[]).append(group)
                atomic_json(statepath,state)
                print(json.dumps(dict(reused_completed_group=group,inputs=args.sample_count)),flush=True)
                continue
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
    except BaseException as error:
        if child is not None:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if state.get('status')!='interrupted_by_signal':
            state.update(status='failed',error=str(error))
        state.update(finished_utc=timestamp(),elapsed_seconds=time.monotonic()-started)
        atomic_json(statepath,state)
        raise


if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run resumable controlled video evaluations and fail on any mismatch."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
REPO = HERE.parent / 'KVRepack'
INVENTORY = Path('/tmp/gpu_retest_data_inventory.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def checked(command, log):
    with log.open('a') as stream:
        result = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Exit {result.returncode}: {command}; see {log}')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stages(plan, counts):
    items=[]
    for method in plan['subset_methods']:
        for task in counts:
            items.append((method, task, min(plan['subset_limit'], counts[task])))
    for method in plan['full_methods']:
        for task, count in counts.items():
            items.append((method, task, count))
    if len(set(items)) != len(items):
        raise ValueError('Duplicate matrix stages')
    return items


def output_dir(method, task, implementation):
    family=method.removesuffix('_kvrepack')
    dataset='nextqa' if task=='nextqa_oe_test' else 'vatex'
    return HERE / f'{family}_{dataset}_{implementation}'


class TraceCounter:
    def __init__(self):
        self.positions={}

    def count(self, path):
        if not path.exists():
            return 0
        offset, total=self.positions.get(path, (0,0))
        if path.stat().st_size < offset:
            raise RuntimeError(f'Trace unexpectedly truncated: {path}')
        with path.open('rb') as stream:
            stream.seek(offset)
            for block in iter(lambda: stream.read(1024*1024), b''):
                total+=block.count(b'\n')
            self.positions[path]=(stream.tell(),total)
        return total


def report(state):
    write(HERE/'matrix_state.json',state)
    lines=['# Video evaluation progress', '', f"Status: **{state['status']}**; updated UTC: {state['updated_utc']}", '',
           'This is real-model generation through the vendored evaluation adapter. Subsets and full splits are recorded separately.', '',
           '| Method | Task | Count | Scope | Status |', '|---|---|---:|---|---|']
    for job in state['stages']:
        lines.append(f"| {job['method']} | {job['task']} | {job['count']} | {'full split' if job['full_split'] else 'first-doc subset'} | {job['status']} |")
    if state.get('current'):
        lines+=['', 'Current generation progress:', '', '```json', json.dumps(state['current'],indent=2), '```']
    if state.get('error'):
        lines+=['', 'Failure: '+state['error']]
    lines+=['', 'Full logits are checked by SHA-256 of every step and vocabulary entry, with shapes/dtypes; this is hash comparison, not a stored-tensor maximum-error calculation.',
            'Each completed comparison also checks input tensors, generated tokens/text and all 36 layers of prefill/final K/V tensor hashes. Timing includes tracing overhead.']
    (HERE/'matrix_report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=HERE/'matrix_plan.json')
    args=parser.parse_args()
    lock=(HERE/'matrix.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=json.loads(INVENTORY.read_text())
    helper_hashes={str(path):file_sha(path) for path in HERE.glob('*.py')}
    counts={d['task']:int(d['rows']) for d in inventory['datasets'] if d['task'] in {'nextqa_oe_test','vatex_test'}}
    asset_path=HERE/'asset_manifest.json'
    checked([PYTHON,'-B',HERE/'freeze_assets.py','--inventory',INVENTORY,'--repo',REPO,
             '--source-root',HERE.parent/'HybridKV-artifact','--output',asset_path],HERE/'freeze_assets.log')
    spec=importlib.util.spec_from_file_location('matrix_frozen_assets',HERE/'freeze_assets.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)

    def verify_assets():
        changed=frozen.verify_manifest(asset_path)
        if changed:
            raise RuntimeError('Frozen inference/data resources changed: '+json.dumps(changed[:10]))
        for path,expected in helper_hashes.items():
            if file_sha(path)!=expected:
                raise RuntimeError(f'Validation helper changed during the matrix: {path}')

    previous=json.loads((HERE/'matrix_state.json').read_text()) if (HERE/'matrix_state.json').exists() else {}
    completed={job['key']:job for job in previous.get('stages',[]) if job['status']=='passed'}
    for old in completed.values():
        if not old.get('artifacts_sha256'):
            raise RuntimeError('Completed stage has no artifact hashes; inspect state before resuming')
        for path,expected in old['artifacts_sha256'].items():
            if file_sha(path)!=expected:
                raise RuntimeError(f'Completed stage artifact changed: {path}')
    state={'schema_version':1,'pid':os.getpid(),'started_utc':now(),'updated_utc':now(),'status':'running',
           'dataset_split_counts':counts,'asset_manifest':str(asset_path),'helper_sha256':helper_hashes,
           'stages':list(completed.values()),'current':None}
    counter=TraceCounter(); running=[]

    def stop(*_):
        raise InterruptedError('Matrix interrupted; persisted traces can resume')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        while True:
            plan=json.loads(args.plan.read_text())
            queue=stages(plan,counts)
            next_stage=next((x for x in queue if f'{x[0]}:{x[1]}:{x[2]}' not in completed),None)
            if next_stage is None:
                break
            verify_assets()
            method,task,count=next_stage
            key=f'{method}:{task}:{count}'
            job={'key':key,'method':method,'task':task,'count':count,'full_split':count==counts[task],
                 'status':'running','started_utc':now()}
            state['stages']=list(completed.values())+[job]
            state['current']={'stage':key,'implementations':{}}
            state['updated_utc']=now();report(state)
            running=[]
            for implementation in ['original','repack']:
                output=output_dir(method,task,implementation)
                log=HERE/f'{output.name}.matrix.launch.log'
                command=[str(PYTHON),'-B','-u',str(HERE/'launch_traced.py'),'--method',method,'--task',task,
                         '--limit',str(count),'--implementation',implementation,'--output',str(output)]
                with log.open('a') as stream:
                    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                running.append((implementation,child,output))
                state['current']['implementations'][implementation]={'pid':child.pid,'trace':str(output/'trace.jsonl'),
                                                                     'launcher_log':str(log),'count':0,'returncode':None}
            while True:
                verify_assets()
                for implementation,child,output in running:
                    info=state['current']['implementations'][implementation]
                    info.update(count=counter.count(output/'trace.jsonl'),returncode=child.poll())
                    if info['returncode'] not in (None,0):
                        raise RuntimeError(f'{key}/{implementation} exited {info["returncode"]}; see {info["launcher_log"]}')
                state['updated_utc']=now();report(state)
                if all(child.poll() is not None for _,child,_ in running):
                    break
                time.sleep(15)
            running=[]
            verify_assets()
            comparison=HERE/f'{method}_{task}_{count}_comparison.json'
            command=[PYTHON,'-B',HERE/'compare_full_runs.py','--reference',output_dir(method,task,'original')/'trace.jsonl',
                     '--candidate',output_dir(method,task,'repack')/'trace.jsonl','--expected-count',str(counts[task]),'--output',comparison]
            if count<counts[task]:
                command+=['--limit',str(count)]
            checked(command,HERE/'matrix_postprocess.log')
            comparison_result=json.loads(comparison.read_text())
            if comparison_result['status']!='passed':
                raise RuntimeError(f'Comparison did not pass: {comparison}')
            metric_paths={}
            for implementation in ['original','repack']:
                target=HERE/f'{method}_{task}_{count}_{implementation}_metrics.json'
                command=[PYTHON,'-B',HERE/'score_full_runs.py','--records',output_dir(method,task,implementation)/'trace.jsonl',
                         '--task',task,'--expected-count',str(count),'--output',target,'--repo-root',REPO]
                if implementation=='repack':
                    command+=['--reuse-metrics-from',metric_paths['original']]
                checked(command,HERE/'matrix_postprocess.log');metric_paths[implementation]=target
            left=json.loads(metric_paths['original'].read_text())['metrics']
            right=json.loads(metric_paths['repack'].read_text())['metrics']
            if left!=right:
                raise RuntimeError(f'Task metric mismatch for {key}: {left} / {right}')
            job.update(status='passed',finished_utc=now(),comparison=str(comparison),metrics=left,
                       metrics_files={k:str(v) for k,v in metric_paths.items()},
                       artifacts_sha256={str(path):file_sha(path) for path in [comparison,*metric_paths.values()]})
            completed[key]=job
            state.update(stages=list(completed.values()),current=None,updated_utc=now())
            report(state)
            print(json.dumps({'stage_complete':key,'metrics':left},ensure_ascii=False),flush=True)
        state.update(status='complete',current=None,finished_utc=now(),updated_utc=now())
        report(state)
    except BaseException as error:
        for _,child,_ in running:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        state.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',error=str(error),updated_utc=now())
        if state.get('stages') and state['stages'][-1]['status']=='running':
            state['stages'][-1]['status']=state['status']
        report(state)
        traceback.print_exc()
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())

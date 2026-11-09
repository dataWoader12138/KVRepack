#!/usr/bin/env python3
"""Run the two-task, 10%-budget LookM comparison in isolated processes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kvrepack.ablations.run_head_selection import atomic_json, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ours', choices=('hybridkv_kvrepack', 'lookm_kvrepack'), default='hybridkv_kvrepack')
    parser.add_argument('--limit', type=int, choices=(1, 200), default=200)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    pretrained, hf_home = args.pretrained.expanduser().resolve(), args.hf_home.expanduser().resolve()
    if not pretrained.is_dir() or not hf_home.is_dir():
        raise FileNotFoundError('Local model snapshot and dataset cache are required')
    if output == ROOT or any(output.is_relative_to(ROOT / name) for name in ('src', 'vendor', 'csrc', 'provenance')):
        raise ValueError('Output must be outside source and published evidence')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(schema_version=1, experiment='lookm_vs_kvrepack',
                    conditions=['lookm', args.ours], tasks=['nextqa_oe_test', 'vatex_test'],
                    limit=args.limit, budget_ratio=0.1, frames=64,
                    model_snapshot=str(pretrained), hf_home=str(hf_home),
                    difference='second condition minus first condition',
                    budget_policy='Native nominal 10% settings; measure actual prefill cache occupancy without altering core algorithms')
        manifest = output / 'matrix.json'
        if args.resume:
            if not manifest.is_file() or json.loads(manifest.read_text()) != spec:
                raise ValueError('Resume requires the identical comparison specification')
        else:
            if any(p.name != '.controller.lock' for p in output.iterdir()):
                raise ValueError('Select a fresh output directory')
            atomic_json(manifest, spec)
        state_path = output / 'experiment_state.json'
        old = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}
        if old.get('status') == 'stopped_by_user' or old.get('resume_authorized') is False:
            raise ValueError('Stopped experiments require a new explicit authorization')
        state = dict(status='starting', started_utc=old.get('started_utc', timestamp()),
                     target_generations=4*args.limit, completed_cases=[],
                     controller_pid=os.getpid(), process_group=os.getpgrp(), resume_authorized=True)
        atomic_json(state_path, state)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('src', 'vendor/lmms-eval', 'csrc')))
        environment['LD_LIBRARY_PATH'] = str(Path(sys.prefix) / 'lib') + os.pathsep + environment.get('LD_LIBRARY_PATH', '')
        try:
            for task in spec['tasks']:
                for method in spec['conditions']:
                    case = output / method / task
                    command = [sys.executable, '-B', '-m', 'kvrepack.benchmarks.run_video_comparison',
                               '--method', method, '--task', task, '--limit', str(args.limit),
                               '--pretrained', str(pretrained), '--hf-home', str(hf_home), '--output', str(case)]
                    if args.resume and (case/'run_contract.json').exists():
                        command.append('--resume')
                    log = output / f'{method}_{task}.log'
                    state.update(status='generating', current_method=method, current_task=task, updated_utc=timestamp())
                    atomic_json(state_path, state)
                    with log.open('a' if args.resume else 'x') as stream:
                        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                        state.update(worker_pid=child.pid); atomic_json(state_path, state)
                        code = child.wait()
                    if code:
                        raise RuntimeError(f'{method}/{task} exited {code}; see {log}')
                    state['completed_cases'].append(dict(method=method, task=task))
                    atomic_json(state_path, state)
            state.update(status='scoring', updated_utc=timestamp()); atomic_json(state_path, state)
            report = output / 'report'
            try:
                report_complete = json.loads((report/'summary.json').read_text()).get('status') == 'complete'
            except (OSError, ValueError):
                report_complete = False
            if not report_complete:
                if report.exists():
                    archived = output / ('report_incomplete_' + timestamp().replace(':', '').replace('+', '_'))
                    report.rename(archived)
                command = [sys.executable, '-B', '-m', 'kvrepack.benchmarks.video_comparison_report',
                           '--run-root', str(output), '--hf-home', str(hf_home),
                           '--output', str(report), '--limit', str(args.limit)]
                with (output/'scoring.log').open('a' if args.resume else 'x') as stream:
                    code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
                if code:
                    raise RuntimeError(f'Scoring exited {code}; see {output / "scoring.log"}')
            if json.loads((report/'summary.json').read_text()).get('status') != 'complete':
                raise ValueError('Report did not validate the full requested prefix')
            state.update(status='complete', finished_utc=timestamp()); atomic_json(state_path, state)
            return 0
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=timestamp())
            atomic_json(state_path, state)
            raise


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run both fixed Qwen2.5 sequence-adjacent observations serially, then plot."""
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from kvrepack.observations.adjacent import atomic_json, timestamp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--text-model', type=Path, required=True)
    parser.add_argument('--vision-model', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, help='Previous 200+200 all-pairs run for input identity checks')
    parser.add_argument('--sample-count', type=int, choices=(2, 200), default=200)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-report', action='store_true', help='Collect the separate timing pilot without plotting')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Use a fresh output directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key not in ('resume', 'no_report')}
    state_path = output/'experiment_state.json'
    if state_path.exists() and json.loads(state_path.read_text())['spec'] != spec:
        parser.error('Refusing to resume another experiment protocol')
    state = dict(status='running', spec=spec, pid=os.getpid(), started_utc=timestamp(),
                 target_inputs=2*args.sample_count, completed_groups=[], gpu_execution=True)
    atomic_json(state_path, state)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO/'src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    child = None
    started = time.monotonic()

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        state.update(status='stopped_by_user', signal=signum, finished_utc=timestamp(),
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(state_path, state)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for group in ('text', 'vision'):
            destination = output/group
            if args.resume and (destination/'run_state.json').exists():
                old = json.loads((destination/'run_state.json').read_text())
                # The worker still checks the complete frozen contract on resume.
            command = [sys.executable, '-B', '-m', 'kvrepack.observations.reference_adjacent',
                       '--group', group, '--sample-count', str(args.sample_count),
                       '--pretrained', str(getattr(args, group+'_model').resolve()),
                       '--hf-home', str(args.hf_home.resolve()), '--inputs', str(args.inputs.resolve()),
                       '--output', str(destination), '--device', args.device]
            if args.reference_root:
                command += ['--reference-records', str(args.reference_root.resolve()/group/'records.jsonl')]
            if args.resume and destination.exists():
                command += ['--resume']
            state.update(current_group=group, command=command, updated_utc=timestamp())
            atomic_json(state_path, state)
            with (output/(group+'.log')).open('a') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                state['worker_pid'] = child.pid
                atomic_json(state_path, state)
                code = child.wait()
            if code:
                raise RuntimeError(f'{group} worker failed ({code}); see {output/(group+".log")}')
            state['completed_groups'].append(group)
            atomic_json(state_path, state)
        if not args.no_report:
            command = [sys.executable, '-B', '-m', 'kvrepack.observations.reference_adjacent_report',
                       '--text-input', str(output/'text/records.jsonl'),
                       '--vision-input', str(output/'vision/records.jsonl'),
                       '--sample-count', str(args.sample_count), '--output', str(output/'report')]
            state.update(status='reporting', command=command)
            atomic_json(state_path, state)
            with (output/'report.log').open('a') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                if child.wait():
                    raise RuntimeError('Report failed; completed GPU records have been preserved')
        state.update(status='complete', finished_utc=timestamp(), elapsed_seconds=time.monotonic()-started)
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        state.update(status='failed', error=str(error), finished_utc=timestamp(),
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())

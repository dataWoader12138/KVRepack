"""Real subprocess regressions for controller interruption and frozen reuse.

Workers are replaced with standard-library sleeping processes or forbidden
entirely. These tests never import a model runner or call CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO/'scripts/run_vatex_longbench_top_adjacent.py'
KIND = 'raw_k_top200_sequence_adjacent_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controller_module():
    spec = importlib.util.spec_from_file_location('top_adjacent_controller_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(root):
    root = Path(root)
    for name in ('inputs.json', 'text-reference.jsonl', 'video-reference.jsonl', 'frozen-source.py'):
        (root/name).write_text('frozen-'+name+'\n')
    for name in ('text-model', 'video-model'):
        (root/name).mkdir()
        (root/name/'config.json').write_text('{"model_type": "test-only-no-weights"}\n')
    return argparse.Namespace(inputs=root/'inputs.json', text_model=root/'text-model',
                              video_model=root/'video-model', hf_home=root/'hf',
                              text_reference=root/'text-reference.jsonl', video_reference=root/'video-reference.jsonl',
                              output=root/'output', sample_count=2, device='0', resume=True, no_report=True)


def cli_args(args, *, resume):
    result = []
    for name in ('inputs', 'text_model', 'video_model', 'hf_home', 'text_reference', 'video_reference', 'output'):
        result += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    result += ['--sample-count', '2', '--no-report']
    if resume:
        result += ['--resume']
    return result


def frozen_group(args, group):
    """Build only the frozen closure checked by the no-model reuse guard."""
    destination = args.output/group
    (destination/'selected_pairs').mkdir(parents=True)
    model = getattr(args, group+'_model')
    contract = dict(sample_count=2, group=group, statistics_kind=KIND, top_k=200,
                    ranking_metric='k_raw_before_rope', doc_ids=[0, 1], pretrained=str(model.resolve()),
                    reference_records_sha256=digest(getattr(args, group+'_reference')),
                    inputs_sha256=digest(args.inputs) if group == 'text' else None,
                    sources_sha256={str(args.inputs.parent/'frozen-source.py'): digest(args.inputs.parent/'frozen-source.py')},
                    model_metadata_sha256={'config.json': digest(model/'config.json')}, versions={})
    path = destination/'contract.json'
    path.write_text(json.dumps(contract)+'\n')
    records = []
    for doc_id in (0, 1):
        selected = destination/'selected_pairs'/f'{doc_id:06d}.npz'
        # Artifact contents/shape are independently checked by the report tests;
        # controller reuse checks their frozen bytes, without loading NumPy.
        selected.write_bytes(f'frozen-selection-{group}-{doc_id}'.encode())
        records.append(dict(status='completed', group=group, statistics_kind=KIND,
                            contract_sha256=digest(path), doc_id=doc_id,
                            selected_pairs_path=str(selected.relative_to(destination)), selected_pairs_sha256=digest(selected)))
    (destination/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (destination/'run_state.json').write_text(json.dumps(dict(status='complete', completed=2, target=2))+'\n')
    return destination


WRAPPER = r'''
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

script, mode, marker_root, arguments = sys.argv[1:]
marker_root = Path(marker_root)
spec = importlib.util.spec_from_file_location('actual_top_adjacent_controller', script)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
real_popen = subprocess.Popen

def fake_popen(command, **kwargs):
    if mode == 'forbid':
        (marker_root/'unexpected-worker').write_text(json.dumps(command))
        raise AssertionError('Completed groups must not spawn or load their models')
    if 'kvrepack.observations.top_adjacent' not in command:
        raise AssertionError('This test must not invoke the report or an unrelated command')
    child_code = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    child = real_popen([sys.executable, '-B', '-c', child_code, str(marker_root/'worker-ready')],
                       cwd=kwargs.get('cwd'), env=kwargs.get('env'), stdout=kwargs.get('stdout'), stderr=kwargs.get('stderr'))
    original_wait = child.wait
    def wait(*args, **kwargs):
        (marker_root/'wait-entered').write_text(str(child.pid))
        return original_wait(*args, **kwargs)
    child.wait = wait
    return child

controller.subprocess.Popen = fake_popen
raise SystemExit(controller.main(json.loads(arguments)))
'''


def start_controller(root, args, mode, *, resume):
    wrapper = Path(root)/'controller-wrapper.py'
    wrapper.write_text(textwrap.dedent(WRAPPER))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(REPO/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return subprocess.Popen([sys.executable, '-B', str(wrapper), str(SCRIPT), mode, str(root),
                             json.dumps(cli_args(args, resume=resume))],
                            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)


@unittest.skipUnless(os.name == 'posix', 'POSIX process signals required')
class TopAdjacentControllerTests(unittest.TestCase):
    def test_sigterm_during_real_child_wait_exits_and_reaps_without_reentrant_deadlock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = arguments(root)
            process = start_controller(root, args, 'sleep', resume=False)
            child_pid = None
            try:
                deadline = time.monotonic()+5
                while not ((root/'wait-entered').exists() and (root/'worker-ready').exists()):
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        self.fail(f'Controller exited before its wait: {stdout}\n{stderr}')
                    if time.monotonic() >= deadline:
                        self.fail('Fake child or controller wait did not become ready')
                    time.sleep(.01)
                child_pid = int((root/'worker-ready').read_text())
                self.assertEqual(child_pid, int((root/'wait-entered').read_text()))
                # Both markers are present; allow original Popen.wait to acquire
                # its non-reentrant waitpid lock before delivering SIGTERM.
                time.sleep(.1)
                process.send_signal(signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self.fail('SIGTERM reentered child.wait and deadlocked the controller')
                self.assertEqual(process.returncode, 128+signal.SIGTERM, f'{stdout}\n{stderr}')
                state = json.loads((args.output/'experiment_state.json').read_text())
                self.assertEqual(state['status'], 'interrupted_by_signal')
                self.assertEqual(state['signal'], signal.SIGTERM)
                self.assertIn('finished_utc', state)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_completed_resume_validates_closure_and_spawns_no_worker_even_on_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = arguments(root)
            text = frozen_group(args, 'text'); frozen_group(args, 'video')
            controller = controller_module()
            self.assertTrue(controller.validate_completed_group(text, 'text', args))
            missing = args.output/'missing'
            self.assertFalse(controller.validate_completed_group(missing, 'text', args))
            process = start_controller(root, args, 'forbid', resume=True)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, f'{stdout}\n{stderr}')
            self.assertFalse((root/'unexpected-worker').exists())
            state = json.loads((args.output/'experiment_state.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['reused_completed_groups'], ['text', 'video'])
            self.assertEqual(state['completed_groups'], ['text', 'video'])
            # A changed selected artifact must fail closed before any model can
            # launch; the completed text group's bytes are otherwise preserved.
            selected = text/'selected_pairs/000000.npz'
            selected.write_bytes(selected.read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError, 'selected pairs changed'):
                controller.validate_completed_group(text, 'text', args)
            process = start_controller(root, args, 'forbid', resume=True)
            stdout, stderr = process.communicate(timeout=5)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('selected pairs changed', stderr)
            self.assertFalse((root/'unexpected-worker').exists())
            state = json.loads((args.output/'experiment_state.json').read_text())
            self.assertEqual(state['status'], 'failed')
            # Hash validation is also anchored in numerical sources and model
            # metadata, not merely the completed flag or a saved sample count.
            selected.write_bytes(selected.read_bytes().removesuffix(b'tampered'))
            for target, error_text in ((root/'frozen-source.py', 'numerical source'),
                                       (args.text_model/'config.json', 'model metadata'),
                                       (args.text_reference, 'protocol changed')):
                original = target.read_bytes(); target.write_bytes(original+b'changed')
                try:
                    with self.subTest(target=target), self.assertRaisesRegex(ValueError, error_text):
                        controller.validate_completed_group(text, 'text', args)
                finally:
                    target.write_bytes(original)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Run original compression function ASTs on tiny CPU tensors, never a full model.

Example (activate the pinned hybridkv environment first):
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 scripts/check_cpu_compression.py \
  --source-root ..

No GPU-dependent imports, downloads, model weights, compilation, or formulas copied
into a replacement implementation. Definitions are compiled from their source AST;
imports and unrelated module initialization are excluded. Class shells select the
original methods and use explicit test state instead of model/cache constructors.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/kvrepack'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class CPUOnly:
    """Block CUDA entry points after the CPU torch import; restore after the suite."""
    def __enter__(self):
        self.environment = patch.dict(os.environ, {
            'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        })
        self.environment.start()
        try:
            import torch
        except ImportError as exc:
            self.environment.stop()
            raise unittest.SkipTest('CPU tensor checks require torch; use the pinned hybridkv Python') from exc
        self.torch = torch
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.attempts = []
        self.saved = []
        for name, value in vars(torch.cuda).copy().items():
            if inspect.isfunction(value) or inspect.isbuiltin(value):
                def reject(*args, _name=name, **kwargs):
                    self.attempts.append('torch.cuda.' + _name)
                    raise AssertionError('CUDA is forbidden by this CPU check: ' + _name)
                self.saved.append((torch.cuda, name, value))
                setattr(torch.cuda, name, reject)
        # A tensor allocation with device='cuda' must fail even if it bypasses Python helpers.
        for name in ('_cuda_init', '_cuda_getDeviceCount'):
            if hasattr(torch._C, name):
                value = getattr(torch._C, name)
                def reject(*args, _name=name, **kwargs):
                    self.attempts.append('torch._C.' + _name)
                    raise AssertionError('CUDA is forbidden by this CPU check: ' + _name)
                self.saved.append((torch._C, name, value))
                setattr(torch._C, name, reject)
        return self

    def __exit__(self, *exc):
        for module, name, value in reversed(self.saved):
            setattr(module, name, value)
        self.torch.set_num_threads(self.previous_threads)
        self.environment.stop()


class SourceFunctions:
    """Compile unchanged definitions and their local definition dependencies only."""
    def __init__(self, torch):
        self.torch = torch
        self.definitions = {}
        self.files = {}

    def load(self, path, roots, *, externals=None, class_methods=None, transform=None):
        path = Path(path).resolve()
        original = path.read_bytes()
        text = original.decode()
        tree = ast.parse(transform(text) if transform else text, filename=str(path))
        definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                constants[node.targets[0].id] = node
        namespace = {
            '__name__': 'cpu_extracted_' + _sha(str(path).encode())[:12],
            '__file__': str(path), 'torch': self.torch,
            'F': self.torch.nn.functional, 'nn': self.torch.nn,
            'math': math, 'os': os, 'json': json, 'random': random,
        }
        namespace.update(externals or {})
        selected, selected_constants = {}, {}
        method_selections = class_methods or {}

        def visit_definition(name):
            if name in selected or name in (externals or {}):
                return
            node = copy.deepcopy(definitions[name])
            if isinstance(node, ast.ClassDef):
                methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
                wanted = set(method_selections.get(name, methods))
                while True:
                    more = {n.attr for method in wanted for n in ast.walk(methods[method])
                            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                            and n.value.id == 'self' and n.attr in methods}
                    if more.issubset(wanted):
                        break
                    wanted.update(more)
                # This is an explicit class shell, not execution of the GPU/model constructor.
                node = ast.ClassDef(name=name, bases=[], keywords=[],
                                    body=[n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
                                    decorator_list=[])
                ast.fix_missing_locations(node)
            selected[name] = node
            for use in ast.walk(node):
                if isinstance(use, ast.Name) and isinstance(use.ctx, ast.Load):
                    if use.id in definitions:
                        visit_definition(use.id)
                    elif use.id in constants:
                        selected_constants[use.id] = copy.deepcopy(constants[use.id])
        for root in roots:
            visit_definition(root)
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += list(selected_constants.values()) + list(selected.values())
        unit = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(unit, str(path), 'exec', dont_inherit=True), namespace)
        self.files[str(path)] = {'sha256': _sha(original), 'branding_normalized': transform is not None}
        for name, node in selected.items():
            functions = [n for n in node.body if isinstance(n, ast.FunctionDef)] if isinstance(node, ast.ClassDef) else [node]
            for function in functions:
                key = str(path) + ':' + (name + '.' if isinstance(node, ast.ClassDef) else '') + function.name
                self.definitions[key] = {
                    'source_start_line': function.lineno, 'source_end_line': function.end_lineno,
                    'executed_definition_ast_sha256': _sha(ast.dump(function, include_attributes=False).encode()),
                }
        return types.SimpleNamespace(**namespace)


class CPUCompressionChecks(unittest.TestCase):
    """Assertions cover CPU numerical/state invariants, not attention-kernel equivalence."""
    reference_root = None

    @classmethod
    def setUpClass(cls):
        cls.guard = CPUOnly().__enter__()
        cls.addClassCleanup(cls.guard.__exit__, None, None, None)
        cls.t = cls.guard.torch
        try:
            import numpy as np
            transformers_distribution = importlib.metadata.distribution('transformers')
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise unittest.SkipTest('CPU tensor checks also need numpy and the local transformers source; use the pinned hybridkv Python') from exc
        cls.extract = SourceFunctions(cls.t)
        cls.merge = cls.extract.load(SOURCE / 'merge_score.py', ['merge_score_from_importance', 'normalize_merge_score_mode'])
        cls.base = cls.extract.load(SOURCE / 'baseline_utils.py', ['load_head_score', 'repeat_kv'])
        transformers_source = Path(transformers_distribution.locate_file(
            'transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py'))
        rotary_dependency = cls.extract.load(transformers_source, ['repeat_kv'])
        common = {'merge_score_from_importance': cls.merge.merge_score_from_importance,
                  'normalize_merge_score_mode': cls.merge.normalize_merge_score_mode,
                  'repeat_kv': rotary_dependency.repeat_kv}
        cls.snap = cls.extract.load(SOURCE / 'qwen_model_snapkv_kvrepack.py', [
            '_history_budget', '_candidate_len', '_topk_sorted', '_topk_from_candidates_sorted',
            '_adjacent_similarity', '_paired_merge_one_sequence', '_build_prompt_record',
            '_global_merge_masks', '_finalize_prompt_cache', 'compute_snapkv_importance_observation'], externals=common)
        cls.look = cls.extract.load(SOURCE / 'qwen_model_lookm_kvrepack.py', [
            '_merge_adjacent_raw_reapply', '_history_budget', '_candidate_len', '_global_merge_masks',
            '_build_layer_record', '_update_decode_group_size', '_apply_decode_score_correction'], externals=common)
        cls.mada = cls.extract.load(SOURCE / 'qwen_model_madakv_kvrepack.py', [
            '_merge_adjacent_raw_reapply', '_current_budget_tensor', '_candidate_len', '_global_merge_masks',
            '_env_float', '_env_int'], externals=common)
        cls.sparse = cls.extract.load(SOURCE / 'qwen_model_sparsemm_kvrepack.py', [
            'init_sparsemm_kvrepack', '_topk_sparsemm_order'],
            class_methods={'SparseMMKVRepackCluster': ['__init__', 'ensure_head_capacity', 'init_metadata']},
            externals={'np': np, 'load_head_score': cls.base.load_head_score,
                       '_env_float': cls.mada._env_float, '_env_int': cls.mada._env_int})
        cls.hybrid = cls.extract.load(SOURCE / 'hybridkv_utils.py', ['HybridKVCache'],
            class_methods={'HybridKVCache': ['_hybrid_kvrepack_merge_one_sequence',
                '_hybrid_kvrepack_target_history_len', '_hybrid_kvrepack_candidate_len',
                '_hybrid_kvrepack_compress_history', '_hybrid_kvrepack_build_simhead_mask']}, externals=common)
        cls.legacy = cls.extract.load(SOURCE / 'baseline_utils.py', ['init_sparsemm'],
            class_methods={'SparseMM': ['__init__', 'update_kv']},
            externals={'np': np, 'load_head_score': cls.base.load_head_score})
        cls.findings = {}

    def assertTensor(self, actual, expected):
        self.assertEqual(actual.device.type, 'cpu')
        expected = self.t.as_tensor(expected, dtype=actual.dtype, device='cpu')
        self.t.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def sequence(self):
        t = self.t
        raw = t.arange(24, dtype=t.float32).reshape(4, 6) / 8
        value = raw + 10
        # Different positions have different phases; averaging already-rotated K is detectably different.
        angles = t.tensor([0., math.pi / 2, math.pi, 3 * math.pi / 2, 0.3, 0.7])
        cos = t.cos(angles).reshape(-1, 1).expand(-1, 6).clone()
        sin = t.sin(angles).reshape(-1, 1).expand(-1, 6).clone()
        pos = t.arange(4, dtype=t.long)
        rope = self.snap._apply_mrope_to_merged_keys(raw, self.snap._select_mrope_batch(cos, 0, pos),
                                                   self.snap._select_mrope_batch(sin, 0, pos), [1, 1, 1])
        return raw, rope, value, t.tensor([1., 2., 10., 20.]), pos, cos, sin

    def merge_call(self, module, target=2, **extra):
        raw, rope, value, importance, pos, cos, sin = self.sequence()
        args = dict(raw_k_seq=raw, rope_k_seq=rope, v_seq=value, importance=importance,
                    target_len=target, positions=pos, cos=cos, sin=sin, batch_idx=0,
                    mrope_section=[1, 1, 1])
        args.update(extra)
        if hasattr(module, '_paired_merge_one_sequence'):
            return module._paired_merge_one_sequence(**args, k_diff_alpha=0., merge_score_mode='full')
        return module._merge_adjacent_raw_reapply(**args, alpha=0.)

    def hybrid_instance(self, max_group=0, module=None):
        h = (module or self.hybrid).HybridKVCache()
        h.hybrid_kvrepack_compress_ratio = .5
        h.hybrid_kvrepack_max_group_size = max_group
        h.hybrid_kvrepack_k_diff_alpha = 0.
        h.hybrid_kvrepack_merge_score_mode = 'full'
        h.hybrid_kvrepack_k_merge_space = 'raw_reapply'
        h.hybrid_asymkv = False
        h.head_dim = 6
        h.chunk_size = 1
        h.retrieval_head_mapping = [[0, 1]]
        return h

    def snap_config(self, gate=0):
        return types.SimpleNamespace(snapkv_kvrepack_window_size=2, snapkv_kvrepack_budget_ratio=.25,
            snapkv_kvrepack_compress_ratio=.5, snapkv_kvrepack_enable=True,
            snapkv_simhead_sim_space='raw', snapkv_simhead_merge_ratio=gate,
            snapkv_kvrepack_verbose=False, snapkv_kvrepack_k_diff_alpha=0.,
            snapkv_kvrepack_merge_score_mode='full')

    def prompt(self, length=8, heads=2):
        t = self.t
        raw = t.arange(heads * length * 6, dtype=t.float32).reshape(1, heads, length, 6) / 11
        values = raw + 20
        cos, sin = t.ones((length, 6)), t.zeros((length, 6))
        # Strict scores avoid implementation-dependent top-k tie choices.
        importance = t.arange(length, dtype=t.float32).flip(0).view(1, 1, length).expand(1, heads, -1).clone()
        return raw, raw.clone(), values, importance, cos, sin

    def test_01_budget_candidate_and_selection_boundaries(self):
        config = self.snap_config()
        self.assertEqual([self.snap._history_budget(config, n, n-2) for n in (8, 16, 8)], [2, 4, 2])
        for n in (0, 1, 5, 12):
            for budget in (-1, 0, 1, 4, 20):
                for ratio in (.1, .5, 1.):
                    for module in (self.snap, self.look, self.mada):
                        candidate = module._candidate_len(budget, n, ratio)
                        self.assertGreaterEqual(candidate, min(max(budget, 0), n))
                        self.assertLessEqual(candidate, n)
        self.assertTensor(self.snap._topk_sorted(self.t.tensor([2., 9., 1., 7.]), 2), [1, 3])
        self.assertTensor(self.sparse._topk_sparsemm_order(self.t.tensor([2., 9., 1., 7.]), 2), [1, 3])
        self.assertTensor(self.snap._topk_from_candidates_sorted(self.t.tensor([8, 2, 5]), self.t.tensor([9., 8., 1.]), 2), [1, 0])
        self.assertTensor(self.mada._current_budget_tensor(None, 16, 14, .25, self.t.device('cpu')), [4])
        self.assertTensor(self.mada._current_budget_tensor(self.t.tensor([99, -1]), 16, 14, .25, self.t.device('cpu')), [14, 0])

    def test_02_global_gate_selects_highest_heads_and_zero_falls_back(self):
        records = [{'head_sim': self.t.tensor([[.1, .9], [.8, .2]])},
                   {'head_sim': self.t.tensor([[.7, .3], [.4, .6]])}]
        for module in (self.snap, self.look, self.mada):
            with self.subTest(module=module.__name__):
                masks = module._global_merge_masks(records, .5)
                self.assertTensor(self.t.stack(masks, 1), [[[False, True], [True, False]], [[True, False], [False, True]]])
                self.assertEqual(sum(int(x.sum()) for x in module._global_merge_masks(records, 0)), 0)
                self.assertEqual(sum(int(x.sum()) for x in module._global_merge_masks(records, 1)), 8)
        self.assertTensor(self.snap._adjacent_similarity(self.t.tensor([[1., 0.], [1., 0.]])), 1.)
        self.assertTensor(self.snap._adjacent_similarity(self.t.tensor([[1., 0.], [-1., 0.]])), -1.)

    def test_03_real_adjacent_merge_raw_key_rope_and_value_conservation(self):
        raw, rope, value, _, pos, cos, sin = self.sequence()
        for module in (self.snap, self.look, self.mada):
            with self.subTest(module=module.__name__):
                key, val, groups = self.merge_call(module)
                self.assertTensor(groups, [3, 1])
                # Known partition from strict importance ordering: [0,1,2], [3].
                expected_raw = self.t.stack([raw[:3].mean(0), raw[3]])
                expected_val = self.t.stack([value[:3].mean(0), value[3]])
                anchors = self.t.tensor([0, 3])
                expected_key = self.snap._apply_mrope_to_merged_keys(expected_raw,
                    self.snap._select_mrope_batch(cos, 0, anchors), self.snap._select_mrope_batch(sin, 0, anchors), [1, 1, 1])
                self.assertTensor(key, expected_key)
                self.assertTensor(val, expected_val)
                self.assertTensor((val * groups[:, None]).sum(0), value.sum(0))
                self.assertFalse(self.t.allclose(key[0], rope[:3].mean(0)))
                unchanged = self.merge_call(module, target=4)
                self.assertTensor(unchanged[0], rope)
                self.assertTensor(unchanged[2], [1, 1, 1, 1])

    def test_03b_mrope_known_quarter_turn_and_embedding_layouts(self):
        t = self.t
        raw = t.tensor([[1., 2., 3., 4., 5., 6.]])
        expected = t.tensor([[-4., -5., -6., 1., 2., 3.]])
        for module in (self.snap, self.look, self.mada):
            for shape in ((5, 6), (3, 5, 6), (3, 2, 5, 6)):
                cos, sin = t.zeros(shape), t.ones(shape)
                pos = t.tensor([3])
                c = module._select_mrope_batch(cos, 1, pos)
                s = module._select_mrope_batch(sin, 1, pos)
                self.assertEqual(c.shape, (3, 1, 6))
                self.assertTensor(module._apply_mrope_to_merged_keys(raw, c, s, [1, 1, 1]), expected)
        # Different t/h/w lanes: sections choose lane 0,1,2,0,1,2 respectively.
        cos = t.tensor([[[2.]*6], [[3.]*6], [[5.]*6]])
        sin = t.zeros_like(cos)
        self.assertTensor(self.snap._apply_mrope_to_merged_keys(t.ones((1,6)),cos,sin,[1,1,1]), [[2,3,5,2,3,5]])

    def test_04_madakv_protected_tokens_and_anchor_positions(self):
        key, value, groups, anchors = self.merge_call(self.mada, target=1,
            protected_mask=self.t.tensor([False, True, False, False]), return_anchor_positions=True)
        self.assertTensor(groups, [1, 1, 2])
        self.assertTensor(anchors, [0, 1, 2])
        self.assertEqual(int(groups.sum()), 4)
        self.assertEqual(key.shape[0], 3)  # Protection can prevent reaching the requested budget.

    def test_05_hybrid_merge_group_cap_and_mixed_gate_padding(self):
        raw, rope, value, importance, pos, cos, sin = self.sequence()
        h = self.hybrid_instance(max_group=2)
        key, val, groups = h._hybrid_kvrepack_merge_one_sequence(rope, value, importance, 1,
            positions=pos, raw_k_seq=raw, cos=cos, sin=sin, mrope_section=[1, 1, 1])
        self.assertTensor(groups, [2, 2])
        self.assertEqual(int(groups.sum()), 4)
        self.assertTensor((val * groups[:, None]).sum(0), value.sum(0))
        with self.assertRaisesRegex(ValueError, 'requires raw_key_states'):
            h._hybrid_kvrepack_merge_one_sequence(rope, value, importance, 2)
        h = self.hybrid_instance()
        raw, rope, value, imp, cos, sin = self.prompt(length=8)
        out_k, out_v, size, lengths = h._hybrid_kvrepack_compress_history(0, rope, value, rope, 6, 8,
            raw_key_states_offload=raw, cos=cos, sin=sin, mrope_section=[1, 1, 1],
            importance=imp[:, :, :6], merge_head_mask=self.t.tensor([[True, False]]))
        self.assertTensor(lengths, [[3, 6]])
        self.assertTensor(size.sum(-1), [[6, 6]])
        self.assertTensor(size[0, 0, 3:], [0, 0, 0])
        self.assertTensor(out_k[0, 1], rope[0, 1, :6])
        self.assertTensor((out_v * size[..., None]).sum(-2), value[:, :, :6].sum(-2))

    def test_06_snap_prefill_cache_gate_zero_is_exact_selection(self):
        raw, rope, value, imp, cos, sin = self.prompt()
        for gate in (0., 1.):
            config = self.snap_config(gate)
            record = self.snap._build_prompt_record(config, 0, raw, rope, value, imp, cos, sin, [1, 1, 1])
            self.assertEqual(record['candidate_history'], 4)
            self.assertEqual(record['final_history'], 2)
            self.assertTensor(record['candidate_idx'][0, 0], [0, 1, 2, 3])
            config._snapkv_kvrepack_records = [record]
            cache = types.SimpleNamespace(key_cache=[], value_cache=[], _seen_tokens=0)
            self.snap._finalize_prompt_cache(config, cache)
            self.assertEqual(cache.key_cache[0].shape, (1, 2, 4, 6))
            self.assertEqual(cache._seen_tokens, 4)
            self.assertEqual(config._snapkv_kvrepack_records, [])
            self.assertTensor(cache.key_cache[0][:, :, -2:], rope[:, :, -2:])
            self.assertTensor(cache.value_cache[0][:, :, -2:], value[:, :, -2:])
            if gate == 0:
                self.assertTensor(cache.key_cache[0][:, :, :2], rope[:, :, :2])
                self.assertTensor(cache.value_cache[0][:, :, :2], value[:, :, :2])
            else:
                expected = self.merge_call_from_record(record)
                self.assertTensor(cache.key_cache[0][0, 0, :2], expected[0])
                self.assertTensor(cache.value_cache[0][0, 0, :2], expected[1])

    def merge_call_from_record(self, record):
        return self.snap._paired_merge_one_sequence(record['candidate_raw'][0, 0], record['candidate_rope'][0, 0],
            record['candidate_value'][0, 0], record['candidate_imp'][0, 0], record['final_history'],
            record['candidate_idx'][0, 0], record['cos'], record['sin'], 0, 0., 'full', record['mrope_section'])

    def test_07_prompt_short_disabled_and_successive_lengths(self):
        config = self.snap_config()
        for n in (8, 16, 8):
            data = self.prompt(length=n)
            record = self.snap._build_prompt_record(config, 0, *data[:4], *data[4:], [1, 1, 1])
            self.assertEqual(record['final_history'], n // 4)
        data = self.prompt(length=1)
        record = self.snap._build_prompt_record(config, 0, *data[:4], *data[4:], [1, 1, 1])
        self.assertIn('direct_key', record)
        config.snapkv_kvrepack_enable = False
        data = self.prompt(length=8)
        record = self.snap._build_prompt_record(config, 0, *data[:4], *data[4:], [1, 1, 1])
        self.assertTensor(record['direct_key'], data[1])

    def test_08_decode_group_metadata_append_and_stale_shape_reset(self):
        t = self.t
        module = types.SimpleNamespace(_lookm_kvrepack_group_size=t.tensor([[[3., 1., 1.]]]))
        actual = self.look._update_decode_group_size(module, t.zeros((1, 1, 4, 6)), 1)
        self.assertTensor(actual, [[[3., 1., 1., 1.]]])
        actual = self.look._update_decode_group_size(module, t.zeros((1, 1, 6, 6)), 2)
        self.assertTensor(actual, [[[3., 1., 1., 1., 1., 1.]]])
        actual = self.look._update_decode_group_size(module, t.zeros((1, 2, 3, 6)), 1)
        self.assertTensor(actual, t.ones((1, 2, 3)))
        scores = t.zeros((1, 2, 1, 3))
        self.assertIs(self.look._apply_decode_score_correction(scores, actual, 'none'), scores)

    def sparse_attention(self, legacy=False):
        attrs = dict(num_hidden_layers=28 if legacy else 36,
            num_attention_heads=28 if legacy else 16, num_key_value_heads=4 if legacy else 2,
            model_type='llava-ov-7b' if legacy else 'qwen2_5_vl-3b',
            window_size=2, kernel_size=1, pooling='avgpool', max_capacity_prompt=64,
            base_capacity_prompt=64, budget_ratio=.25, ratio=.1, head_score='visual', gqa_func='mean',
            sparsemm_kvrepack_window_size=2, sparsemm_kvrepack_kernel_size=1,
            sparsemm_kvrepack_budget_ratio=.25, sparsemm_kvrepack_ratio=.1)
        return types.SimpleNamespace(config=types.SimpleNamespace(**attrs), layer_idx=0,
                                     _flash_attn_uses_top_left_mask=False)

    def test_09_sparsemm_rebuild_recomputes_budget_and_varlen_metadata(self):
        attn = self.sparse_attention()
        with patch.dict(os.environ, {'MODEL_TYPE': attn.config.model_type, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'}, clear=True):
            self.sparse.init_sparsemm_kvrepack(attn, rebuild_cluster=True)
            first = attn.kv_cluster
            short = first.ensure_head_capacity(8, self.t.device('cpu')).clone()
            self.sparse.init_sparsemm_kvrepack(attn, rebuild_cluster=True)
            self.assertIsNot(first, attn.kv_cluster)
            long = attn.kv_cluster.ensure_head_capacity(16, self.t.device('cpu'))
            self.assertFalse(self.t.equal(short, long))
            attn.kv_cluster.init_metadata([2, 4], self.t.device('cpu'))
            self.assertTensor(attn.kv_cluster.cu_klen, [0, 2, 6])
            self.assertTensor(attn.kv_cluster.cu_qlen, [0, 1, 2])
            self.assertEqual(attn.kv_cluster.klen_sum, 6)
            self.assertEqual(attn.kv_cluster.max_seqlen_k, 4)

    def test_10_observation_attention_cpu_gqa_and_causal_window(self):
        t = self.t
        query = t.arange(4*8*6, dtype=t.float32).reshape(1, 4, 8, 6) / 100
        key = t.arange(2*8*6, dtype=t.float32).reshape(1, 2, 8, 6) / 100
        for pooling in ('avgpool', 'maxpool'):
            for gqa in ('mean', 'max'):
                out = self.snap.compute_snapkv_importance_observation(query, key, 2, 6, 2, 1, pooling, gqa)
                self.assertEqual(out.shape, (1, 2, 8))
                self.assertTrue(bool(t.isfinite(out).all()))
                self.assertTrue(bool((out >= 0).all()))
                self.assertTensor(out[:, :, -2:], t.zeros((1, 2, 2)))
                self.assertGreater(float(out[:, :, :6].sum()), 0.)

    def test_11_original_source_same_input_differential(self):
        if self.reference_root is None:
            self.skipTest('Use --source-root to compare the original artifact on identical CPU inputs')
        spec = importlib.util.spec_from_file_location('_cpu_check_naming', ROOT/'scripts/check_sources.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        manifest = json.loads((ROOT/'provenance/core_sources.json').read_text())
        by_target = {Path(e['target_path']).name: e for e in manifest['entries']}
        merge_entry = by_target['merge_score.py']
        merge_path = Path(self.reference_root)/merge_entry['source_path']
        self.assertEqual(_sha(merge_path.read_bytes()), merge_entry['source_sha256'])
        refmerge = self.extract.load(merge_path, ['merge_score_from_importance'], transform=checker.rename_brand)
        external = {'merge_score_from_importance': refmerge.merge_score_from_importance}
        for target, module, roots in (
            ('qwen_model_snapkv_kvrepack.py', self.snap, ['_paired_merge_one_sequence','_candidate_len','_global_merge_masks']),
            ('qwen_model_lookm_kvrepack.py', self.look, ['_merge_adjacent_raw_reapply','_candidate_len','_global_merge_masks']),
            ('qwen_model_madakv_kvrepack.py', self.mada, ['_merge_adjacent_raw_reapply','_candidate_len','_global_merge_masks'])):
            entry = by_target[target]
            path = Path(self.reference_root)/entry['source_path']
            self.assertEqual(_sha(path.read_bytes()), entry['source_sha256'])
            ref = self.extract.load(path, roots, externals=external, transform=checker.rename_brand)
            for target_len in (1, 2, 3, 4):
                for got, expected in zip(self.merge_call(module, target_len), self.merge_call(ref, target_len)):
                    self.assertTensor(got, expected)
            for budget in (0,1,4,12):
                self.assertEqual(module._candidate_len(budget, 10, .7), ref._candidate_len(budget, 10, .7))
            records = [{'head_sim': self.t.tensor([[.1,.9]])}, {'head_sim': self.t.tensor([[.4,.7]])}]
            for got, expected in zip(module._global_merge_masks(records,.5),ref._global_merge_masks(records,.5)):
                self.assertTensor(got,expected)

        entry = by_target['hybridkv_utils.py']
        path = Path(self.reference_root)/entry['source_path']
        self.assertEqual(_sha(path.read_bytes()), entry['source_sha256'])
        ref = self.extract.load(path, ['HybridKVCache'],
            class_methods={'HybridKVCache': ['_hybrid_kvrepack_merge_one_sequence']},
            externals=external, transform=checker.rename_brand)
        raw, rope, value, importance, pos, cos, sin = self.sequence()
        for group_limit in (0, 2):
            current = self.hybrid_instance(group_limit)
            original = self.hybrid_instance(group_limit, module=ref)
            for target_len in (1, 2, 3, 4):
                kwargs = dict(positions=pos, raw_k_seq=raw, cos=cos, sin=sin, mrope_section=[1, 1, 1])
                got = current._hybrid_kvrepack_merge_one_sequence(rope, value, importance, target_len, **kwargs)
                expected = original._hybrid_kvrepack_merge_one_sequence(rope, value, importance, target_len, **kwargs)
                for actual_tensor, expected_tensor in zip(got, expected):
                    self.assertTensor(actual_tensor, expected_tensor)

    def test_12_reproduce_llava_sparsemm_cross_prompt_budget_finding(self):
        t = self.t
        attn = self.sparse_attention(legacy=True)
        with patch.dict(os.environ, {'MODEL_TYPE':'llava-ov-7b','METHOD':'sparsemm','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}, clear=True):
            self.legacy.init_sparsemm(attn)
            first_cluster = attn.kv_cluster
            def update(module, n):
                key = t.arange(4*n*6,dtype=t.float32).reshape(1,4,n,6)/100
                query = t.arange(28*n*6,dtype=t.float32).reshape(1,28,n,6)/100
                module.kv_cluster.update_kv(key,query,key+1)
                return module.kv_cluster.head_adaptive_capacity.clone()
            short = update(attn,8)
            self.legacy.init_sparsemm(attn)  # Actual LLaVA caller does not request rebuild.
            reused = update(attn,16)
            fresh = self.sparse_attention(legacy=True)
            self.legacy.init_sparsemm(fresh)
            fresh_capacity = update(fresh,16)
        finding = {'status':'reproduced','source':'src/kvrepack/baseline_utils.py',
            'functions':['init_sparsemm','SparseMM.update_kv'],
            'prompt_lengths':[8,16],'same_cluster_reused':attn.kv_cluster is first_cluster,
            'second_budget_equals_first':bool(t.equal(short,reused)),
            'second_budget_differs_from_fresh_same_length':bool(not t.equal(reused,fresh_capacity)),
            'layer0_reused_capacity':reused[0].tolist(),'layer0_fresh_capacity':fresh_capacity[0].tolist(),
            'boundary':'Real baseline init/update/attention-score/metadata code; tiny CPU tensors and bundled visual-head scores. No full LLaVA model or CUDA kernel.'}
        self.assertTrue(finding['same_cluster_reused'])
        self.assertTrue(finding['second_budget_equals_first'])
        self.assertTrue(finding['second_budget_differs_from_fresh_same_length'])
        self.findings['llava_sparsemm_stale_prompt_budget'] = finding

    def test_13_reproduce_asymkv_chunk_boundary_omission_finding(self):
        t = self.t
        traces=[]
        class CacheFixture:
            def __init__(self):self.length=0
            def get_seq_length(self):return self.length
        def forward(model,cache,embeddings,positions,cache_position):
            traces.append(('forward',cache_position.tolist()))
            cache.length += int(cache_position.numel())
            return t.zeros((1,2))
        def probe(*args,**kwargs):
            traces.append(('probe',args[5].tolist()))
            return None,{}
        m = self.extract.load(SOURCE/'asymkv_official/generation.py',
            ['_prefill_like_upstream','AsymKVOfficialRuntimeConfig'],
            class_methods={'AsymKVOfficialRuntimeConfig':['resolve_cache_size']},
            externals={'AsymKVOfficialCache':CacheFixture,'_forward_chunk':forward,
                       '_probe_and_compress_history':probe})
        runtime=m.AsymKVOfficialRuntimeConfig()
        runtime.absolute_cache_size=64
        runtime.budget_ratio=.1
        runtime.start_size=0  # Permits a one-token prompt to reach the actual loop.
        runtime.chunk_size=512
        model=types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(num_key_value_groups=1))])))
        cases=[]
        for n in (1,2,511,512,513,1024,1025):
            traces.clear()
            ids=t.arange(n).reshape(1,n)
            error=None
            try:
                m._prefill_like_upstream(model,ids,t.zeros((1,n,2)),t.arange(n).reshape(1,1,n).expand(3,1,n),runtime)
            except RuntimeError as exc:
                error=str(exc)
            seen=[position for name,positions in traces if name=='forward' for position in positions]
            missing=sorted(set(range(n))-set(seen))
            cases.append({'prompt_length':n,'forwarded_tokens':len(seen),'missing_positions':missing,
                'chunks':[[positions[0],positions[-1]+1] for name,positions in traces if name=='forward' and positions],
                'probe_calls':sum(name=='probe' for name,_ in traces),'error':error})
        indexed={x['prompt_length']:x for x in cases}
        self.assertEqual(indexed[513]['missing_positions'],[512])
        self.assertEqual(indexed[1025]['missing_positions'],[1024])
        self.assertEqual(indexed[512]['missing_positions'],[])
        self.assertEqual(indexed[1024]['missing_positions'],[])
        self.assertEqual(indexed[1]['forwarded_tokens'],0)
        self.assertIn('did not process',indexed[1]['error'])
        self.findings['asymkv_prefill_chunk_boundary_omission']={
            'status':'reproduced','source':'src/kvrepack/asymkv_official/generation.py',
            'function':'_prefill_like_upstream','cases':cases,
            'boundary':'Original function/control flow and runtime budget resolver on real CPU input tensors. Cache, model forwarding and gradient/compression probe are recording fixtures; no gradient/compression correctness claim.'}

    def test_99_no_cuda_entry_point_called(self):
        self.assertEqual(self.guard.attempts, [])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,help='Optional original workspace for same-input artifact comparison')
    args=parser.parse_args(argv)
    CPUCompressionChecks.reference_root=args.source_root.resolve() if args.source_root else None
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CPUCompressionChecks)
    result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(suite)
    cls=CPUCompressionChecks
    report={
        'status':'completed_with_findings' if result.wasSuccessful() and getattr(cls,'findings',{}) else ('passed' if result.wasSuccessful() else 'failed'),
        'validation_kind':'real_cpu_small_tensors_with_ast_extracted_original_definitions',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':[reason for _,reason in result.skipped],
        'torch_version':str(cls.t.__version__) if hasattr(cls,'t') else None,
        'cuda_visible_devices_during_checks':'','tensor_device':'cpu','torch_threads_during_checks':1,
        'cuda_entry_point_attempts':getattr(getattr(cls,'guard',None),'attempts',[]),
        'original_source_differential_requested':args.source_root is not None,
        'findings':getattr(cls,'findings',{}),
        'boundaries':[
            'Complete GPU-dependent core modules and full model attention forwards were not imported/executed.',
            'No replacement compression formula: unchanged source definition ASTs and their local dependencies execute on torch CPU.',
            'Selected class shells omit unrelated constructors/bases; explicit tiny fixture state is recorded in the checks.',
            'SnapKV prefill uses a key/value-list cache fixture; LookM decode checks the original group-metadata helper, not FlashAttention decode.',
            'AsymKV boundary reproduction replaces model forward/cache/probe operations with recorders and validates only real prefill control flow.',
            'Original artifact comparison, when requested, applies only the recorded naming transformation and verifies original source SHA-256.',
            'No GPU kernels, numerical full-model equivalence, memory/latency, datasets or scores are validated.',
        ],
        'source_files':getattr(getattr(cls,'extract',None),'files',{}),
        'extracted_definitions':getattr(getattr(cls,'extract',None),'definitions',{}),
    }
    print(json.dumps(report,indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())

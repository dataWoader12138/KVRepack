"""LLaVA image migration checks: real control flow and tiny CPU functions only."""

from __future__ import annotations

import ast
import copy
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/kvrepack"


def tool(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = tool("llava_replay", "scripts/check_llava_sources.py")
cpu = tool("llava_cpu_helpers", "scripts/check_cpu_compression.py")


class LlavaImageStructureTests(unittest.TestCase):
    def test_exact_original_bytes_and_declared_port_transforms_replay(self):
        result = replay.check()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["ported_modules"]), 5)

    def test_image_only_extensions_leave_video_support_matrix_unchanged(self):
        from kvrepack.experiments.registry import METHODS_BY_MODEL
        from kvrepack.experiments.llava_image_methods import LLAVA_IMAGE_UPGRADES
        self.assertEqual(sum(map(len, METHODS_BY_MODEL.values())), 19)
        self.assertFalse(set(LLAVA_IMAGE_UPGRADES) & set(METHODS_BY_MODEL["llava_ov"]))
        self.assertEqual(set(LLAVA_IMAGE_UPGRADES), {
            "hybridkv_kvrepack", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack"})
        for name, selected in LLAVA_IMAGE_UPGRADES.items():
            self.assertEqual(selected.name, selected.internal_method)
            self.assertEqual(selected.model_family, "llava_ov")
            self.assertIn(selected.alpha_dependent, selected.parameters)

    def test_four_methods_execute_real_worker_dispatch_in_fresh_processes(self):
        dispatch = tool("llava_image_dispatch", "scripts/check_llava_dispatch.py")
        fixture = tool("llava_dispatch_fixture", "tests/test_dispatch.py")
        with tempfile.TemporaryDirectory() as temp:
            hf = Path(temp) / "transformers"
            fixture.write_hf_structure_fixture(hf)
            report = dispatch.audit(hf)
        self.assertTrue(report["passed"], report)
        rows = {row["method"]: row for row in report["combinations"]}
        hybrid = rows["hybridkv_kvrepack"]
        self.assertEqual(hybrid["patch_targets"], [])
        self.assertEqual(hybrid["hybrid_cache_check"]["language_backbone"], "kvrepack.modeling_qwen2_kvrepack")
        for name, count in (("snapkv_kvrepack", 2), ("sparsemm_kvrepack", 3), ("madakv_kvrepack", 4)):
            self.assertEqual(len(rows[name]["patch_targets"]), count)
            self.assertEqual(rows[name]["algorithm_functions_executed"], 0)


class LlavaCPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = cpu.CPUOnly().__enter__()
        cls.addClassCleanup(cls.guard.__exit__, None, None, None)
        cls.t = cls.guard.torch
        try:
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest("LLaVA CPU checks require numpy and torch") from exc
        cls.extract = cpu.SourceFunctions(cls.t)
        try:
            rotary_path = Path(importlib.metadata.distribution("transformers").locate_file(
                "transformers/models/qwen2/modeling_qwen2.py"))
        except importlib.metadata.PackageNotFoundError as exc:
            raise unittest.SkipTest("LLaVA CPU checks require the installed transformers source") from exc
        cls.rotary = cls.extract.load(rotary_path, ["rotate_half", "repeat_kv"])
        cls.common = {"rotate_half": cls.rotary.rotate_half, "repeat_kv": cls.rotary.repeat_kv}
        cls.base = cls.extract.load(SOURCE / "baseline_utils.py", ["init_sparsemm"],
            class_methods={"SparseMM": ["__init__", "update_kv"]}, externals={"np": np})
        cls.snap = cls.extract.load(SOURCE / "qwen2_model_snapkv_kvrepack.py", ["_paired_merge_one_sequence", "init_snapkv_kvrepack"], externals=cls.common)
        cls.mada = cls.extract.load(SOURCE / "qwen2_model_madakv_kvrepack.py", ["_merge_adjacent_raw_reapply", "_env_float", "_env_int"], externals=cls.common)
        cls.sparse = cls.extract.load(SOURCE / "qwen2_model_sparsemm_kvrepack.py", ["init_sparsemm_kvrepack"],
            class_methods={"SparseMMKVRepackCluster": ["__init__", "ensure_head_capacity", "init_metadata"]},
            externals={"np": np, "load_head_score": cls.base.load_head_score,
                       "_env_float": cls.mada._env_float, "_env_int": cls.mada._env_int})
        cls.hybrid = cls.extract.load(SOURCE / "hybridkv_utils_llava.py", ["HybridKVCache"],
            class_methods={"HybridKVCache": ["_hybrid_kvrepack_merge_one_sequence"]}, externals=cls.common)
        cls.entries = {Path(e["target_path"]).name: e for e in json.loads(
            (ROOT / "provenance/llava_image_sources.json").read_text())["entries"]}

    def attention(self):
        return cpu.CPUCompressionChecks.sparse_attention(self, legacy=True)

    def test_baseline_sparsemm_prefill_rebuilds_budget_and_decode_keeps_metadata(self):
        """Run the exact new caller prefix, then real baseline init/update functions."""
        t = self.t
        path = SOURCE / "qwen2_model.py"
        tree = ast.parse(path.read_text())
        caller = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "qwen2_flash_attn2_forward_SparseMM")
        # Stop before projection/attention. Only the actual initialization control
        # flow executes; update_kv below remains the original numerical function.
        stop = next(i for i, node in enumerate(caller.body)
                    if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "num_attention_heads" for x in node.targets))
        prefix = ast.FunctionDef(name="initialize_for_call", args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="hidden_states")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=copy.deepcopy(caller.body[:stop]), decorator_list=[])
        namespace = {"init_sparsemm": self.base.init_sparsemm}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])), str(path), "exec"), namespace)
        initialize = namespace["initialize_for_call"]
        def prefill(attn, n):
            initialize(attn, t.zeros(1, n, 1))
            key = t.arange(4*n*6, dtype=t.float32).reshape(1, 4, n, 6)/100
            query = t.arange(28*n*6, dtype=t.float32).reshape(1, 28, n, 6)/100
            attn.kv_cluster.update_kv(key, query, key+1)
            return attn.kv_cluster.head_adaptive_capacity.clone()
        with patch.dict(os.environ, {"MODEL_TYPE": "llava-ov-7b", "METHOD": "sparsemm", "CUDA_VISIBLE_DEVICES": ""}, clear=True):
            attn = self.attention()
            short = prefill(attn, 8)
            first = attn.kv_cluster
            long = prefill(attn, 16)
            self.assertIsNot(first, attn.kv_cluster)
            self.assertFalse(t.equal(short, long))
            fresh = self.attention()
            t.testing.assert_close(long, prefill(fresh, 16), rtol=0, atol=0)
            active = attn.kv_cluster
            metadata = (active.head_lens, active.cu_klen, active.klen_sum, active.max_seqlen_k)
            initialize(attn, t.zeros(1, 1, 1))
            self.assertIs(attn.kv_cluster, active)
            self.assertIs(active.head_lens, metadata[0])
            self.assertIs(active.cu_klen, metadata[1])
            self.assertEqual((active.klen_sum, active.max_seqlen_k), metadata[2:])

    def test_sparsemm_upgrade_capacity_refresh_and_decode_initialization(self):
        t = self.t
        attn = self.attention()
        with patch.dict(os.environ, {"MODEL_TYPE": "llava-ov-7b", "METHOD": "sparsemm_kvrepack"}, clear=True):
            self.sparse.init_sparsemm_kvrepack(attn)
            cluster = attn.kv_cluster
            short = cluster.ensure_head_capacity(8, t.device("cpu")).clone()
            long = cluster.ensure_head_capacity(16, t.device("cpu")).clone()
            fresh = self.attention()
            self.sparse.init_sparsemm_kvrepack(fresh)
            t.testing.assert_close(long, fresh.kv_cluster.ensure_head_capacity(16, t.device("cpu")), rtol=0, atol=0)
            self.assertFalse(t.equal(short, long))
            cluster.init_metadata([2, 3, 4, 5], t.device("cpu"))
            metadata = cluster.head_lens
            self.sparse.init_sparsemm_kvrepack(attn)
            self.assertIs(attn.kv_cluster, cluster)
            self.assertIs(cluster.head_lens, metadata)

    def test_snapkv_prefixed_alpha_is_effective_and_global_fallback_is_preserved(self):
        with patch.dict(os.environ, {"K_DIFF_ALPHA": "2", "SNAPKV_KVREPACK_K_DIFF_ALPHA": "3"}, clear=True):
            attn = types.SimpleNamespace(config=types.SimpleNamespace(), _flash_attn_uses_top_left_mask=False)
            self.snap.init_snapkv_kvrepack(attn)
            self.assertEqual(attn.config.snapkv_kvrepack_k_diff_alpha, 3)
        with patch.dict(os.environ, {"K_DIFF_ALPHA": "2"}, clear=True):
            attn = types.SimpleNamespace(config=types.SimpleNamespace(), _flash_attn_uses_top_left_mask=False)
            self.snap.init_snapkv_kvrepack(attn)
            self.assertEqual(attn.config.snapkv_kvrepack_k_diff_alpha, 2)

    def test_original_llava_merge_functions_match_port_on_plain_qwen2_rope(self):
        t = self.t
        raw = t.arange(24, dtype=t.float32).reshape(4, 6)/10
        value = raw + 10
        positions = t.arange(4)
        angles = t.arange(24, dtype=t.float32).reshape(1, 4, 6)/9
        cos, sin = angles.cos(), angles.sin()
        rope = self.snap._apply_mrope_to_merged_keys(raw, cos[0], sin[0], None)
        importance = t.tensor([1., 2., 10., 20.])
        for filename, function, current, alpha_key in (
            ("qwen2_model_snapkv_kvrepack.py", "_paired_merge_one_sequence", self.snap, "k_diff_alpha"),
            ("qwen2_model_madakv_kvrepack.py", "_merge_adjacent_raw_reapply", self.mada, "alpha"),
        ):
            entry = self.entries[filename]
            original = self.extract.load(SOURCE / filename, [function], externals=self.common, transform=lambda text, e=entry:
                                         replay.restore_recorded_edits(text, e["port_replay"]))
            for target in (1, 2, 3, 4):
                kwargs = dict(raw_k_seq=raw, rope_k_seq=rope, v_seq=value, importance=importance,
                              target_len=target, positions=positions, cos=cos, sin=sin,
                              batch_idx=0, mrope_section=None, **{alpha_key: 2.})
                got = getattr(current, function)(**kwargs)
                expected = getattr(original, function)(**kwargs)
                for actual, reference in zip(got, expected):
                    self.assertEqual(actual.device.type, "cpu")
                    t.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertEqual(len(got[0]), target)
                self.assertEqual(float(got[2].sum()), 4)
        # Hybrid uses a cache method; compare with the original method and state.
        entry = self.entries["hybridkv_utils_llava.py"]
        original = self.extract.load(SOURCE / "hybridkv_utils_llava.py", ["HybridKVCache"],
            class_methods={"HybridKVCache": ["_hybrid_decouple_merge_one_sequence"]},
            externals=self.common,
            transform=lambda text: replay.restore_recorded_edits(text, entry["port_replay"]))
        for target in (1, 2, 3, 4):
            actual = self.hybrid.HybridKVCache()
            legacy = original.HybridKVCache()
            for instance, prefix in ((actual, "hybrid_kvrepack"), (legacy, "hybrid_decouple")):
                setattr(instance, prefix+"_k_merge_space", "raw_reapply")
                setattr(instance, prefix+"_max_group_size", 0)
                setattr(instance, prefix+"_k_diff_alpha", 2.)
                instance.head_dim = 6
            kwargs = dict(positions=positions, raw_k_seq=raw, cos=cos, sin=sin, mrope_section=None)
            got = actual._hybrid_kvrepack_merge_one_sequence(rope, value, importance, target, **kwargs)
            expected = legacy._hybrid_decouple_merge_one_sequence(rope, value, importance, target, **kwargs)
            for output, reference in zip(got, expected):
                t.testing.assert_close(output, reference, rtol=0, atol=0)
        self.assertFalse(self.guard.attempts)


if __name__ == "__main__":
    unittest.main()

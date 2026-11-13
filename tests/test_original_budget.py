"""CPU evidence that restored-budget adapters preserve native decisions."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from kvrepack.ablations.original_budget import (
    BUDGET_POLICY, NativePruningAdapter, OriginalSnapMergeAdapter,
    OriginalHybridMergeAdapter, active_prefill_original,
)
from test_strict_pruning import NativeSnap, NativeHybrid, ShapeTensor, snap_input, populate_hybrid_final_cache
from test_merge_snap import core_fixture, SOURCE
from test_rope_merge import source_module


def prompt_fixture():
    module = core_fixture()
    names = {"_candidate_len", "_topk_sorted", "_adjacent_similarity", "_build_prompt_record"}
    nodes = [n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    module.__dict__["F"] = torch.nn.functional
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    return module


class OriginalBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_real_native_snap_KV_and_state_match_original_bitwise(self):
        source = SOURCE.with_name("baseline_utils.py")
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {"repeat_kv", "SnapKVCluster"}]
        ns = dict(torch=torch, math=math, F=torch.nn.functional, nn=torch.nn)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
        owner = ns["SnapKVCluster"]
        rng = torch.Generator().manual_seed(17)
        key, query, value = [torch.randn(shape, generator=rng) for shape in
            ((1, 2, 128, 8), (1, 4, 128, 8), (1, 2, 128, 8))]
        opts = dict(window_size=8, max_capacity_prompt=128, budget_ratio=.25, num_key_value_groups=2)
        reference = owner(**opts)
        expected = reference.update_kv(key, query, value)
        actual = owner(**opts)
        original = owner.update_kv
        with NativePruningAdapter("snapkv", budget_ratio=.25,
                                  module=SimpleNamespace(SnapKVCluster=owner)) as adapter:
            observed = actual.update_kv(key, query, value)
            audit = adapter.finish_sample()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(observed, expected)))
        self.assertEqual(observed[0].shape[-2], 40)  # 32 history + 8 recent
        self.assertEqual(actual.max_capacity_prompt, reference.max_capacity_prompt)
        self.assertEqual(actual.max_capacity_prompt, 32)
        self.assertEqual(audit["prefill_calls"][0]["capacity_per_head"], 40)
        self.assertEqual(audit["actual_merge_calls"], 0)
        self.assertIs(owner.update_kv, original)

    def test_native_snap_keeps_original_capacity_state_and_changes_no_timed_tensor(self):
        original = NativeSnap.update_kv
        with NativePruningAdapter("snapkv", budget_ratio=.3,
                                  module=SimpleNamespace(SnapKVCluster=NativeSnap)) as adapter:
            cluster = NativeSnap()
            for i, n in enumerate((100, 150)):
                adapter.begin_sample("nextqa_oe_test", i)
                key, value = cluster.update_kv(*snap_input(n))
                self.assertEqual(key.shape[-2], math.floor(.3*n)+4)
                self.assertEqual(cluster.max_capacity_prompt, math.floor(.3*n))
                self.assertEqual(adapter.finish_sample()["actual_merge_calls"], 0)
        self.assertIs(NativeSnap.update_kv, original)

    def test_native_hybrid_does_not_scale_round_or_reassign_retrieval_budget(self):
        module = SimpleNamespace(HybridKVCache=NativeHybrid)
        original = NativeHybrid.allocate_budget_by_ratio
        with patch("kvrepack.ablations.merge_hybrid.shrink_global_budgets", side_effect=AssertionError("must not shrink")):
            with NativePruningAdapter("hybridkv", budget_ratio=.3, module=module) as adapter:
                adapter.begin_sample("nextqa_oe_test", 7)
                cache = NativeHybrid()
                self.assertEqual(cache.allocate_budget_by_ratio(ShapeTensor((1, 2, 100, 8))), "native allocation result")
                self.assertEqual(cache.sparse_budget_per_head, [[[30, 20]], [[30, 20]]])
                self.assertEqual(cache.retrieval_budget, 30)  # not rounded to24
                populate_hybrid_final_cache(cache)
                audit = adapter.finish_sample()
        self.assertIs(NativeHybrid.allocate_budget_by_ratio, original)
        self.assertEqual(audit["budget"]["active_prefill_positions"], 140)
        self.assertEqual(audit["budget"]["scale"], 1.)
        self.assertEqual(audit["actual_merge_calls"], 0)
        self.assertIsNone(adapter._cache)

    def test_native_requires_current_nonmerge_cache_and_restores_on_error(self):
        module = SimpleNamespace(HybridKVCache=NativeHybrid)
        original_init = NativeHybrid.__init__
        with self.assertRaisesRegex(ValueError, "without merge"):
            with NativePruningAdapter("hybridkv", module=module):
                NativeHybrid(merge=True)
        self.assertIs(NativeHybrid.__init__, original_init)
        existing = NativeHybrid()
        with NativePruningAdapter("hybridkv", module=module) as adapter:
            with self.assertRaisesRegex(ValueError, "current sample"):
                existing.allocate_budget_by_ratio(ShapeTensor((1, 2, 100, 8)))
            with self.assertRaisesRegex(RuntimeError, "No completed"):
                adapter.finish_sample()

    def test_all_snap_aggregators_leave_original_budget_candidates_and_heads_identical(self):
        module = prompt_fixture()
        rng = torch.Generator().manual_seed(9)
        raw, rope, value = [torch.randn((1, 2, 128, 6), generator=rng) for _ in range(3)]
        imp = torch.randn((1, 2, 128), generator=rng)
        cos, sin = torch.ones((3, 1, 128, 6)), torch.zeros((3, 1, 128, 6))
        cfg = SimpleNamespace(snapkv_kvrepack_budget_ratio=.25, snapkv_kvrepack_window_size=8,
            snapkv_kvrepack_enable=True, snapkv_kvrepack_compress_ratio=.7,
            snapkv_simhead_sim_space="raw", snapkv_simhead_merge_ratio=.5,
            snapkv_kvrepack_verbose=False, snapkv_kvrepack_k_diff_alpha=2.,
            snapkv_kvrepack_merge_score_mode="full")
        def record():
            return module._build_prompt_record(cfg, 0, raw, rope, value, imp, cos, sin, [1,1,1])
        expected = record()
        expected_mask = module._global_merge_masks([expected], .5)
        original_history = module._history_budget
        for variant in ("kvrepack", "lookm", "asymkv"):
            with self.subTest(variant=variant), patch("kvrepack.ablations.merge_snap.strict_history_budget", side_effect=AssertionError("must not subtract recent")):
                with OriginalSnapMergeAdapter(variant, budget_ratio=.25, module=module) as adapter:
                    self.assertIs(module._history_budget, original_history)
                    observed = record()
                    self.assertEqual(observed["final_history"], 32)
                    self.assertEqual(observed["candidate_history"], 46)
                    for name in ("candidate_idx", "candidate_raw", "candidate_rope", "candidate_value", "candidate_imp", "head_sim"):
                        self.assertTrue(torch.equal(observed[name], expected[name]), name)
                    self.assertTrue(torch.equal(module._global_merge_masks([observed], .5)[0], expected_mask[0]))
                    self.assertTrue(adapter.transform_spec["budget"]["native_budget_unchanged"])
            self.assertIs(module._history_budget, original_history)

    def test_restored_snap_kvrepack_final_cache_is_original_bitwise(self):
        module = prompt_fixture()
        rng = torch.Generator().manual_seed(27)
        raw = torch.randn((1, 2, 128, 6), generator=rng)
        value = torch.randn((1, 2, 128, 6), generator=rng)
        imp = torch.randn((1, 2, 128), generator=rng)
        cfg = SimpleNamespace(snapkv_kvrepack_budget_ratio=.25, snapkv_kvrepack_window_size=8,
            snapkv_kvrepack_enable=True, snapkv_kvrepack_compress_ratio=.7,
            snapkv_simhead_sim_space="raw", snapkv_simhead_merge_ratio=.5,
            snapkv_kvrepack_verbose=False, snapkv_kvrepack_k_diff_alpha=2.,
            snapkv_kvrepack_merge_score_mode="full")
        def prepare():
            rec = module._build_prompt_record(cfg, 0, raw, raw, value, imp,
                torch.ones((3, 1, 128, 6)), torch.zeros((3, 1, 128, 6)), [1,1,1])
            cfg._snapkv_kvrepack_records = [rec]
            return SimpleNamespace(key_cache=[], value_cache=[], _seen_tokens=0)
        reference = prepare();module._finalize_prompt_cache(cfg, reference)
        with OriginalSnapMergeAdapter("kvrepack", budget_ratio=.25, module=module) as adapter:
            actual = prepare();module._finalize_prompt_cache(cfg, actual)
            audit = adapter.finish_sample()
        self.assertTrue(torch.equal(reference.key_cache[0], actual.key_cache[0]))
        self.assertTrue(torch.equal(reference.value_cache[0], actual.value_cache[0]))
        self.assertEqual(actual.key_cache[0].shape[-2], 40)
        self.assertGreater(audit["actual_merge_calls"], 0)

    def test_all_hybrid_variants_bypass_strict_allocation_wrapper(self):
        # Isolate precisely the changed hook for all three operators, including
        # AsymKV, without building a model or modifying its attention function.
        module = SimpleNamespace(HybridKVCache=NativeHybrid)
        original = NativeHybrid.allocate_budget_by_ratio
        for variant in ("kvrepack", "lookm", "asymkv"):
            adapter = OriginalHybridMergeAdapter(variant, budget_ratio=.3, module=module)
            try:
                adapter._patch(NativeHybrid, "allocate_budget_by_ratio", lambda *a: self.fail("strict wrapper called"))
                cache = NativeHybrid()
                cache.allocate_budget_by_ratio(ShapeTensor((1, 2, 100, 8)))
                self.assertEqual(cache.sparse_budget_per_head, [[[30, 20]], [[30, 20]]])
                self.assertEqual(cache.retrieval_budget, 30)
                self.assertEqual(adapter._budget_meta["active_prefill_positions"], 140)
                self.assertTrue(adapter._budget_meta["native_budget_unchanged"])
                self.assertEqual(adapter.transform_spec["budget"]["transformed"], False)
            finally:
                adapter.__exit__(None, None, None)
            self.assertIs(NativeHybrid.allocate_budget_by_ratio, original)

    def test_original_hybrid_snapshot_is_not_an_alias_to_mutable_budgets(self):
        adapter = OriginalHybridMergeAdapter("lookm", module=SimpleNamespace(HybridKVCache=NativeHybrid))
        try:
            adapter._patch(NativeHybrid, "allocate_budget_by_ratio", lambda *a: None)
            cache = NativeHybrid();cache.allocate_budget_by_ratio(ShapeTensor((1, 2, 100, 8)))
            cache.sparse_budget_per_head[0][0][0] = 1
            self.assertEqual(adapter._budget_meta["history_budget"][0][0][0], 30)
            self.assertEqual(adapter._budget_meta["original_history_budget"], adapter._budget_meta["history_budget"])
        finally:
            adapter.__exit__(None, None, None)

    def test_snap_physical_cache_accepts_original_budget_above_strict_cap(self):
        shapes = [dict(layer=i, key_shape=[1,4,993,128], value_shape=[1,4,993,128]) for i in range(28)]
        budget = active_prefill_original(None, 2, 9607, "snapkv", shape_fn=lambda _: shapes)
        self.assertEqual(budget["prefill_tokens_total"], 111104)
        self.assertEqual(budget["capacity"], 111104)
        self.assertEqual(budget["nominal_total_cap"], 107520)
        self.assertGreater(budget["actual_retained_ratio"], .1)
        self.assertEqual(budget["budget_policy"], BUDGET_POLICY)
        shapes[0]["value_shape"][-2] = 992
        with self.assertRaisesRegex(ValueError, "physical K/V"):
            active_prefill_original(None, 2, 9607, "snapkv", shape_fn=lambda _: shapes)

    def test_hybrid_physical_cache_requires_original_allocation_not_strict_cap(self):
        # 28layers x4heads x(100 history +32 local)=14784 prefill positions.
        shapes = [dict(layer=i, key_shape=[532,128], value_shape=[532,128]) for i in range(28)]
        meta = dict(policy=BUDGET_POLICY,native_budget_unchanged=True,
            post_allocation_scaling=False,scale=1.,sequence_length=1000,layers=28,native_kv_heads=4,
            original_history_positions=11200,history_positions=11200,local_positions=3584,
            original_history_budget=[[[100]*4]]*28,history_budget=[[[100]*4]]*28,
            active_prefill_positions=14784,capacity_positions=14784)
        audit = dict(budget=meta)
        value = active_prefill_original(None,2,1000,"hybridkv",shape_fn=lambda _:shapes,audit=audit)
        self.assertEqual(value["prefill_tokens_total"], 14784)
        self.assertEqual(value["nominal_total_cap"],11200)
        for bad in (None, {"budget":dict(meta,scale=.9)}, {"budget":dict(meta,history_positions=11199)}):
            with self.assertRaises(ValueError):
                active_prefill_original(None,2,1000,"hybridkv",shape_fn=lambda _:shapes,audit=bad)

    def test_partial_layers_or_wrong_physical_length_fail_closed(self):
        for shapes in ([], [dict(key_shape=[1,4,993,128],value_shape=[1,4,993,128])],
                       [dict(key_shape=[1,4,961,128],value_shape=[1,4,961,128])]*28):
            with self.assertRaises(ValueError):
                active_prefill_original(None,2,9607,"snapkv",shape_fn=lambda _:shapes)


if __name__ == "__main__":
    unittest.main()

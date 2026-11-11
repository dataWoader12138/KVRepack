"""Independent CPU oracles for the Hybrid fixed-group insertion and LMA."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from kvrepack.ablations.merge_hybrid import (
    MergeHybridAdapter, _compile_merge, append_flat_cardinality,
    lma_flat_readout, shrink_global_budgets,
)
from kvrepack.ablations.merge_operators import reduce_fixed_groups
from test_rope_merge import source_module, sequence


class HybridFixedGroupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_budget_includes_local_preserves_head_class_and_scales_globally(self):
        original = [[[100, 200, 300, 400]], [[100, 200, 300, 400]]]
        result, meta = shrink_global_budgets(original, [[1], [1]], seq_len=1000,
                                            local_len=32, ratio=.1, chunk_size=8)
        self.assertEqual(original[0][0], [100, 200, 300, 400])
        self.assertEqual(meta["capacity_positions"], 800)
        self.assertEqual(meta["local_positions"], 256)
        self.assertLessEqual(meta["active_prefill_positions"], 800)
        self.assertEqual(result[0][0][1] % 8, 0)
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[0][0][0], int(100 * meta["scale"]))
        # Heterogeneous allocation can exceed 10% in an individual head.
        self.assertGreater(result[0][0][-1] + 32, 100)

    def test_small_budget_does_not_round_retrieval_up_outside_capacity(self):
        with self.assertRaisesRegex(ValueError, "retrieval chunk"):
            shrink_global_budgets([[[1, 99]]], [[0]], seq_len=400,
                                  local_len=32, ratio=.1, chunk_size=8)
        with self.assertRaisesRegex(ValueError, "short"):
            shrink_global_budgets([[[10]]], [[]], seq_len=100,
                                  local_len=32, ratio=.1, chunk_size=8)

    def test_count_update_matches_per_head_insertion_including_zero_length_head(self):
        counts = torch.tensor([2., 1., 3., 5., 4.])
        cumulative = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
        actual = append_flat_cardinality(counts, cumulative)
        self.assertEqual(actual.tolist(), [2., 1., 1., 1., 3., 5., 4., 1.])

    def test_batched_lma_matches_official_denominator_for_ragged_gqa(self):
        rng = torch.Generator().manual_seed(42)
        query = torch.randn((1, 6, 1, 8), generator=rng)
        keys = torch.randn((7, 8), generator=rng)
        values = torch.randn((7, 8), generator=rng)
        counts = torch.tensor([1., 2., 5., 1., 3., 1., 4.])
        cumulative = torch.tensor([0, 2, 7], dtype=torch.int32)
        lens = torch.tensor([2, 5], dtype=torch.int32)
        actual = lma_flat_readout(query, keys, values, counts, cumulative, lens, 5, 2)
        expected_parts = []
        for head, (lo, hi) in enumerate(((0, 2), (2, 7))):
            q = query[:, head*3:(head+1)*3]
            score = q @ keys[lo:hi].T / (8**.5)
            exponent = torch.exp(score - score.amax(-1, keepdim=True))
            probability = exponent / (exponent * counts[lo:hi]).sum(-1, keepdim=True)
            expected_parts.append(probability @ values[lo:hi])
        expected = torch.cat(expected_parts, dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=5e-6)
        # Ordinary softmax with Vsum is observably wrong on these counts.
        ordinary = torch.softmax(query[:, :3] @ keys[:2].T / 8**.5, -1) @ values[:2]
        self.assertGreater(float((ordinary - actual[:, :3]).abs().max()), .01)

    def test_bfloat16_batched_lma_matches_existing_official_operators_exactly(self):
        from kvrepack.asymkv_official.attention import official_lma_attention
        rng = torch.Generator().manual_seed(87)
        query = torch.randn((1, 6, 1, 8), generator=rng).bfloat16()
        keys = torch.randn((7, 8), generator=rng).bfloat16()
        values = torch.randn((7, 8), generator=rng).bfloat16()
        counts = torch.tensor([1., 2., 5., 1., 3., 1., 4.])
        cumulative = torch.tensor([0, 2, 7], dtype=torch.int32)
        lens = torch.tensor([2, 5], dtype=torch.int32)
        actual = lma_flat_readout(query, keys, values, counts, cumulative, lens, 5, 2)
        expected_parts = []
        for head, (lo, hi) in enumerate(((0, 2), (2, 7))):
            output, _ = official_lma_attention(query[:, head*3:(head+1)*3],
                keys[None, None, lo:hi], values[None, None, lo:hi],
                counts[None, None, lo:hi], current_query_length=1)
            expected_parts.append(output)
        expected = torch.cat(expected_parts, dim=1)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def _evaluate(self, variant, seq):
        module = source_module("hybridkv")
        original = module.HybridKVCache._hybrid_kvrepack_merge_one_sequence
        obj = module.HybridKVCache()
        obj.hybrid_kvrepack_k_merge_space = "raw_reapply"
        obj.hybrid_asymkv = False
        obj.hybrid_kvrepack_max_group_size = 0
        obj.hybrid_kvrepack_k_diff_alpha = 2.
        obj.hybrid_kvrepack_merge_score_mode = "full"
        obj.head_dim = 6
        observed = []
        gradient = torch.arange(1., 61.).reshape(10, 6) / 10

        def reducer(cache, raw, value, ids, sizes, start, positions, layer, head, batch):
            return reduce_fixed_groups(variant, raw, value, ids, sizes, start,
                                       fisher_gradient=gradient[positions] if variant == "asymkv" else None)

        def record(cache, rope, raw, value, importance, positions, ids, sizes, start, *identity):
            observed.append(dict(ids=ids.clone(), sizes=sizes.clone(), start=start.clone(),
                                 positions=positions.clone(), raw=raw, values=value))

        function, spec = _compile_merge(original, variant, reducer, record)
        kw = {k: v for k, v in seq.items() if k not in {"rope_k_seq", "k_diff_alpha", "merge_score_mode"}}
        kw.update(k_seq=seq["rope_k_seq"], layer_idx=0, head_idx=0)
        result = function(obj, **kw)
        baseline = original(obj, **kw)
        return result, baseline, observed[0], spec, gradient

    def test_native_arm_bitwise_unchanged_groups_match_and_pivot_formula_exact(self):
        seq = sequence(identity_rope=True)
        native, baseline, original_groups, _, _ = self._evaluate("kvrepack", seq)
        for actual, expected in zip(native, baseline):
            self.assertTrue(torch.equal(actual, expected))
        for variant in ("lookm", "asymkv"):
            result, _, groups, spec, gradient = self._evaluate(variant, seq)
            for name in ("ids", "sizes", "start", "positions"):
                self.assertTrue(torch.equal(groups[name], original_groups[name]))
            raw, values, ids = groups["raw"], groups["values"], groups["ids"]
            for group in range(3):
                members = torch.nonzero(ids == group).flatten()
                if variant == "lookm":
                    anchor = int(members[0])
                    k_sources = torch.cat((raw[anchor:anchor+1], (raw[members[1:]] + raw[anchor]) / 2))
                    v_sources = torch.cat((values[anchor:anchor+1], (values[members[1:]] + values[anchor]) / 2))
                    expected_k, expected_v = k_sources.mean(0), v_sources.mean(0)
                else:
                    fisher = gradient[groups["positions"]][members].square()
                    expected_k = (raw[members] * fisher).sum(0) / (fisher.sum(0) + 1e-21)
                    expected_v = values[members].sum(0)
                torch.testing.assert_close(result[0][group], expected_k, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(result[1][group], expected_v, atol=1e-7, rtol=1e-6)
            self.assertIn("native anchor", spec["adaptation"])

    def test_context_restores_original_methods_on_exception(self):
        module = source_module("hybridkv")
        owner = module.HybridKVCache
        owner.allocate_budget_by_ratio = lambda cache, key: None
        owner.update = lambda cache, key, value, layer, kwargs=None: (key, value)
        original = owner._hybrid_kvrepack_merge_one_sequence
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with MergeHybridAdapter("lookm", module=module) as adapter:
                self.assertIsNot(owner._hybrid_kvrepack_merge_one_sequence, original)
                adapter._cache = object()
                adapter.fisher = [torch.ones(2)]
                adapter._group_refs = [torch.ones(2)]
                raise RuntimeError("fixture")
        self.assertIs(owner._hybrid_kvrepack_merge_one_sequence, original)
        self.assertIsNone(adapter._cache)
        self.assertIsNone(adapter.fisher)
        self.assertEqual(adapter._group_refs, [])

    def test_finish_and_begin_drop_gpu_owners_keep_only_cpu_metadata(self):
        adapter = MergeHybridAdapter("asymkv", module=source_module("hybridkv"))
        adapter._cache = SimpleNamespace(
            hybrid_kvrepack_merge_mask=torch.ones((1, 1, 1), dtype=torch.bool),
            key_cache=[torch.ones((3, 2))], value_cache=[torch.ones((3, 2))])
        adapter._budget_meta = {"active_prefill_positions": 3}
        adapter.fisher = [torch.ones((1, 1, 10, 2))]
        adapter._group_refs = [dict(group_ids=torch.tensor([0, 0, 1]))]
        report = adapter.finish_sample()
        self.assertIsNone(adapter._cache)
        self.assertIsNone(adapter.fisher)
        self.assertEqual(adapter._group_refs, [])
        self.assertEqual(report["selected_head_count"], 1)
        self.assertEqual(report["merge_calls"][0]["group_ids"]["shape"], [3])
        adapter._cache, adapter.fisher = object(), [torch.ones(2)]
        adapter.begin_sample("vatex_test", 1)
        self.assertIsNone(adapter._cache)
        self.assertIsNone(adapter.fisher)


if __name__ == "__main__":
    unittest.main()

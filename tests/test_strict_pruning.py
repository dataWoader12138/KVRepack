"""CPU-only budget, source-transform, and patch-lifecycle checks."""
import ast
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest

from kvrepack.ablations.strict_pruning import StrictPruningAdapter, strict_snap_history


class ShapeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def cpu(self):
        raise AssertionError("No tensor-to-CPU work belongs in the adapter")


class NativeSnap:
    def __init__(self, ratio=.3, window=4):
        self.window_size = window
        self.budget_ratio = ratio
        self.max_capacity_prompt = 1024
        self.pyram_mode = False
        self.layer_idx = None

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        q_len = origin_key_states.shape[-2]
        self.max_capacity_prompt = int(q_len * self.budget_ratio)
        shape = list(origin_key_states.shape)
        shape[-2] = self.max_capacity_prompt+self.window_size
        return ShapeTensor(shape), ShapeTensor(shape)


class BadOutputSnap(NativeSnap):
    def update_kv(self, origin_key_states, query_states, origin_value_states):
        q_len = origin_key_states.shape[-2]
        self.max_capacity_prompt = int(q_len * self.budget_ratio)
        shape = list(origin_key_states.shape)
        shape[-2] = self.max_capacity_prompt+self.window_size+1
        return ShapeTensor(shape), ShapeTensor(shape)


class ChangedSourceSnap(NativeSnap):
    def update_kv(self, origin_key_states, query_states, origin_value_states):
        q_len = origin_key_states.shape[-2]
        self.max_capacity_prompt = math.floor(q_len * self.budget_ratio)
        return origin_key_states, origin_value_states


class NativeHybrid:
    def __init__(self, merge=False, local_len=10):
        self.hybridkv_kvrepack = merge
        self.hybrid_asymkv = False
        self.local_len = local_len
        self.chunk_size = 8
        self.retrieval_head_mapping = [[0], [0]]
        self.retrieval_budget = 30
        self.key_cache = []
        self.value_cache = []

    def allocate_budget_by_ratio(self, key_states):
        length = key_states.shape[-2]
        self.sparse_budget_per_head = [[[int(.3*length), int(.2*length)]] for _ in range(2)]
        self.retrieval_budget = int(.3*length)
        return "native allocation result"


def snap_input(length=100):
    return ShapeTensor((1, 4, length, 8)), ShapeTensor((1, 28, length, 8)), ShapeTensor((1, 4, length, 8))


def populate_hybrid_final_cache(cache):
    cache.key_cache = [ShapeTensor((sum(batch[0])+2*cache.local_len, 8)) for batch in cache.sparse_budget_per_head]
    cache.value_cache = [ShapeTensor(value.shape) for value in cache.key_cache]


class StrictPruningTests(unittest.TestCase):
    def test_total_snap_capacity_includes_recent_and_rounds_down(self):
        self.assertEqual(strict_snap_history(103, 4, .3), 26)
        self.assertEqual(strict_snap_history(103, 4, 1), 99)
        for args in ((0, 4, .3), (10, 10, .3), (20, 10, .1), (100, 0, .3), (100, 4, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                strict_snap_history(*args)

    def test_snap_changes_one_budget_expression_and_preserves_native_body(self):
        original = NativeSnap.update_kv
        adapter = StrictPruningAdapter("snapkv", budget_ratio=.3, module=SimpleNamespace(SnapKVCluster=NativeSnap))
        with adapter:
            self.assertIsNot(NativeSnap.update_kv, original)
            spec = adapter.transform_spec["update_kv"]
            self.assertEqual(spec["changed_assignments"], 1)
            old = ast.parse(textwrap.dedent(inspect.getsource(original)))
            new = ast.parse(spec["transformed_source"])
            target = lambda node: isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == "max_capacity_prompt"
            old_assignment = next(node for node in ast.walk(old) if target(node))
            new_assignment = next(node for node in ast.walk(new) if target(node))
            old_assignment.value = new_assignment.value
            self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))
            cluster = NativeSnap()
            key, value = cluster.update_kv(*snap_input(103))
            self.assertEqual(key.shape, (1, 4, 30, 8))
            self.assertEqual(key.shape, value.shape)
            self.assertEqual(cluster.max_capacity_prompt, 1024)
            audit = adapter.finish_sample()
            self.assertEqual(audit["actual_merge_calls"], 0)
            self.assertEqual(audit["prefill_calls"][0]["history_positions"], 26)
            json.dumps(audit)
        self.assertIs(NativeSnap.update_kv, original)
        # Outside the context the unmodified nominal-history behavior remains.
        self.assertEqual(NativeSnap().update_kv(*snap_input(103))[0].shape[-2], 34)

    def test_snap_recomputes_every_prompt_and_releases_audit_references(self):
        adapter = StrictPruningAdapter("snapkv", budget_ratio=.3, module=SimpleNamespace(SnapKVCluster=NativeSnap))
        with adapter:
            cluster = NativeSnap()
            for doc_id, length in enumerate((100, 150)):
                adapter.begin_sample("vatex_test", doc_id)
                cluster.update_kv(*snap_input(length))
                audit = adapter.finish_sample()
                self.assertEqual(audit["doc_id"], doc_id)
                self.assertEqual(audit["prefill_calls"][0]["capacity_per_head"], math.floor(.3*length))
                self.assertEqual(cluster.max_capacity_prompt, 1024)
                self.assertEqual(adapter._snap_rows, [])

    def test_real_native_snap_cpu_matches_history_budget_oracle(self):
        import torch
        source_path = Path(__file__).resolve().parents[1]/"src/kvrepack/baseline_utils.py"
        source_tree = ast.parse(source_path.read_text())
        selected = [node for node in source_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                    and node.name in ("repeat_kv", "SnapKVCluster")]
        self.assertEqual(len(selected), 2)
        namespace = dict(torch=torch, math=math, F=torch.nn.functional,
                         nn=torch.nn, __name__="cpu_native_snap_oracle")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
        owner = namespace["SnapKVCluster"]
        generator = torch.Generator(device="cpu").manual_seed(17)
        key = torch.randn(1, 2, 128, 8, generator=generator)
        query = torch.randn(1, 4, 128, 8, generator=generator)
        value = torch.randn(1, 2, 128, 8, generator=generator)
        # 128*.25=32 total positions, including8 recent; original native
        # implementation with24/128 exact binary ratio is an independent
        # historical-slot oracle for its unmodified scoring and value gather.
        reference = owner(window_size=8, max_capacity_prompt=128,
                          budget_ratio=24/128, num_key_value_groups=2)
        expected_key, expected_value = reference.update_kv(key, query, value)
        actual = owner(window_size=8, max_capacity_prompt=128,
                       budget_ratio=.25, num_key_value_groups=2)
        with StrictPruningAdapter("snapkv", budget_ratio=.25,
                                  module=SimpleNamespace(SnapKVCluster=owner)) as adapter:
            actual_key, actual_value = actual.update_kv(key, query, value)
            audit = adapter.finish_sample()
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))
        self.assertTrue(torch.equal(actual_key[:, :, -8:], key[:, :, -8:]))
        self.assertTrue(torch.equal(actual_value[:, :, -8:], value[:, :, -8:]))
        self.assertEqual(audit["prefill_calls"][0]["capacity_per_head"], 32)

    def test_snap_rejects_unreviewed_source_without_leaking_patches(self):
        original = ChangedSourceSnap.update_kv
        adapter = StrictPruningAdapter("snapkv", budget_ratio=.3, module=SimpleNamespace(SnapKVCluster=ChangedSourceSnap))
        with self.assertRaisesRegex(ValueError, "assignment changed"):
            with adapter:
                pass
        self.assertIs(ChangedSourceSnap.update_kv, original)
        self.assertFalse(adapter._entered)

    def test_snap_bad_outputs_restore_capacity_and_class_on_failure(self):
        original = BadOutputSnap.update_kv
        cluster = BadOutputSnap()
        with self.assertRaisesRegex(ValueError, "strict total capacity"):
            with StrictPruningAdapter("snapkv", budget_ratio=.3, module=SimpleNamespace(SnapKVCluster=BadOutputSnap)):
                cluster.update_kv(*snap_input())
        self.assertIs(BadOutputSnap.update_kv, original)
        self.assertEqual(cluster.max_capacity_prompt, 1024)

    def test_snap_rejects_pyramid_or_ratio_mismatch(self):
        module = SimpleNamespace(SnapKVCluster=NativeSnap)
        for pyramid, ratio in ((True, .3), (False, .2)):
            with self.subTest(pyramid=pyramid, ratio=ratio):
                cluster = NativeSnap(ratio=ratio)
                cluster.pyram_mode = pyramid
                with StrictPruningAdapter("snapkv", budget_ratio=.3, module=module):
                    with self.assertRaises(ValueError):
                        cluster.update_kv(*snap_input())
                    self.assertEqual(cluster.max_capacity_prompt, 1024)

    def test_hybrid_keeps_native_allocation_and_shares_global_budget_policy(self):
        original_init, original_allocate = NativeHybrid.__init__, NativeHybrid.allocate_budget_by_ratio
        module = SimpleNamespace(HybridKVCache=NativeHybrid)
        with StrictPruningAdapter("hybridkv", budget_ratio=.3, module=module) as adapter:
            adapter.begin_sample("vatex_test", 7)
            cache = NativeHybrid()
            self.assertEqual(cache.allocate_budget_by_ratio(ShapeTensor((1, 4, 100, 8))), "native allocation result")
            self.assertEqual(cache.sparse_budget_per_head, [[[24, 16]], [[24, 16]]])
            self.assertEqual(cache.retrieval_budget, 24)
            self.assertFalse(cache.hybridkv_kvrepack)
            populate_hybrid_final_cache(cache)
            audit = adapter.finish_sample()
            self.assertEqual(audit["budget"]["active_prefill_positions"], 120)
            self.assertEqual(audit["budget"]["capacity_positions"], 120)
            self.assertEqual(audit["budget"]["original_history_budget"], [[[30, 20]], [[30, 20]]])
            self.assertEqual(audit["actual_merge_calls"], 0)
            self.assertEqual(audit["doc_id"], 7)
            json.dumps(audit)
            self.assertIsNone(adapter._cache)
        self.assertIs(NativeHybrid.__init__, original_init)
        self.assertIs(NativeHybrid.allocate_budget_by_ratio, original_allocate)

    def test_hybrid_recomputes_every_prompt_and_respects_chunk_floors(self):
        module = SimpleNamespace(HybridKVCache=NativeHybrid)
        with StrictPruningAdapter("hybridkv", budget_ratio=.3, module=module) as adapter:
            for doc_id, length in enumerate((100, 200)):
                adapter.begin_sample("vatex_test", doc_id)
                cache = NativeHybrid()
                cache.allocate_budget_by_ratio(ShapeTensor((1, 4, length, 8)))
                populate_hybrid_final_cache(cache)
                audit = adapter.finish_sample()
                self.assertEqual(audit["budget"]["sequence_length"], length)
                self.assertLessEqual(audit["budget"]["active_prefill_positions"], 4*math.floor(.3*length))
                self.assertTrue(all(batch[0][0] % 8 == 0 for batch in cache.sparse_budget_per_head))

    def test_hybrid_fails_closed_on_merge_or_preexisting_cache(self):
        module = SimpleNamespace(HybridKVCache=NativeHybrid)
        original = NativeHybrid.__init__
        with self.assertRaisesRegex(ValueError, "without merge"):
            with StrictPruningAdapter("hybridkv", budget_ratio=.3, module=module):
                NativeHybrid(merge=True)
        self.assertIs(NativeHybrid.__init__, original)
        cache = NativeHybrid()
        with StrictPruningAdapter("hybridkv", budget_ratio=.3, module=module):
            with self.assertRaisesRegex(ValueError, "current context"):
                cache.allocate_budget_by_ratio(ShapeTensor((1, 4, 100, 8)))

    def test_audit_requires_complete_current_sample_and_patches_restore_on_exception(self):
        module = SimpleNamespace(SnapKVCluster=NativeSnap)
        original = NativeSnap.update_kv
        adapter = StrictPruningAdapter("snapkv", budget_ratio=.3, module=module)
        with self.assertRaises(RuntimeError):
            adapter.finish_sample()
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with adapter:
                with self.assertRaises(RuntimeError):
                    adapter.finish_sample()
                NativeSnap().update_kv(*snap_input())
                with self.assertRaisesRegex(RuntimeError, "previous"):
                    adapter.begin_sample("vatex_test", 2)
                raise RuntimeError("sentinel")
        self.assertIs(NativeSnap.update_kv, original)
        self.assertEqual(adapter._patches, [])
        self.assertEqual(adapter._snap_rows, [])

    def test_invalid_method_ratio_and_double_entry_fail_closed(self):
        for method, ratio in (("fullkv", .1), ("snapkv", 0), ("hybridkv", float("nan")), ("snapkv", 1.1)):
            with self.subTest(method=method, ratio=ratio), self.assertRaises(ValueError):
                StrictPruningAdapter(method, budget_ratio=ratio)
        adapter = StrictPruningAdapter("snapkv", budget_ratio=.3, module=SimpleNamespace(SnapKVCluster=NativeSnap))
        with adapter:
            with self.assertRaisesRegex(RuntimeError, "twice"):
                adapter.__enter__()


if __name__ == "__main__":
    unittest.main()

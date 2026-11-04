"""CPU tensors compare the linear estimator to dense, explicitly excluded diagonals."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvrepack.observations.all_pairs import AllPairsObserver, all_pairs_cosine_statistics
from test_observation_hooks import FakeModel, RotaryStub


def dense(value):
    normalized = torch.nn.functional.normalize(value.float(), dim=-1).double()
    similarities = normalized @ normalized.transpose(1, 2)
    tokens = value.shape[1]
    mask = ~torch.eye(tokens, dtype=torch.bool)
    return similarities[:, mask].mean(dim=1), similarities.mean(dim=(1, 2))


class AllPairsFormulaTests(unittest.TestCase):
    def test_dense_off_diagonal_and_inclusive_agree_in_each_head(self):
        generator = torch.Generator().manual_seed(91)
        value = torch.randn(4, 7, 13, generator=generator)
        expected, inclusive = dense(value)
        result = all_pairs_cosine_statistics(value)
        self.assertEqual(result["token_count"], 7)
        self.assertEqual(result["pair_count"], 42)
        self.assertEqual(result["status"], "supported")
        torch.testing.assert_close(torch.tensor(result["off_diagonal"], dtype=torch.float64), expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(torch.tensor(result["inclusive"], dtype=torch.float64), inclusive, atol=1e-12, rtol=0)

    def test_opposite_vectors_retain_negative_one_and_zero_inclusive(self):
        value = torch.tensor([[[1., 0.], [-1., 0.]]])
        result = all_pairs_cosine_statistics(value)
        self.assertEqual(result["off_diagonal"], [-1.])
        self.assertEqual(result["inclusive"], [0.])
        self.assertEqual(result["diagonal"], [1.])

    def test_empty_and_singleton_are_unsupported_rather_than_fabricated_zero(self):
        for tokens in (0, 1):
            result = all_pairs_cosine_statistics(torch.ones(2, tokens, 3))
            self.assertEqual(result["status"], "unsupported_t_lt_2")
            self.assertEqual(result["pair_count"], 0)
            self.assertIsNone(result["off_diagonal"])
            if tokens:
                self.assertTrue(all(abs(x - 1) < 1e-6 for x in result["inclusive"]))
            else:
                self.assertIsNone(result["inclusive"])
                self.assertIsNone(result["diagonal"])

    def test_bfloat16_normalization_matches_fp32_dense_reference(self):
        value = torch.tensor([[[1., 2., -1.], [-2., 1., 1.], [1., -3., 2.]],
                              [[3., 1., 0.], [2., -1., 1.], [-1., 2., 2.]]], dtype=torch.bfloat16)
        expected, _ = dense(value)
        result = all_pairs_cosine_statistics(value)
        torch.testing.assert_close(torch.tensor(result["off_diagonal"], dtype=torch.float64), expected, atol=1e-12, rtol=0)

    def test_zero_and_nonfinite_vectors_are_rejected(self):
        for invalid in (0., float("nan"), float("inf")):
            value = torch.ones(2, 3, 4)
            value[0, 2] = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "zero|nonfinite"):
                all_pairs_cosine_statistics(value)
        for invalid in (torch.ones(2, 3), torch.ones(2, 3, 4, dtype=torch.long)):
            with self.assertRaisesRegex(ValueError, "floating"):
                all_pairs_cosine_statistics(invalid)


class AllPairsObserverTests(unittest.TestCase):
    def setUp(self):
        self.rotary = RotaryStub()
        self.original = self.rotary.apply_multimodal_rotary_pos_emb
        self.model = FakeModel(self.rotary)
        self.value = torch.tensor([[[1., 0., 1., 0.], [-1., 0., 0., 1.],
                                    [0., 1., 1., 1.], [1., 1., -1., 1.]]])
        self.plan = {"input_length": 4, "groups": {
            "visual_spatial": {"pairs": [[0, 1]], "selected_pair_count": 1},
            "text_question": {"pairs": [[2, 3]], "selected_pair_count": 1}}}
        self.pools = {"prompt_all": [0, 1, 2, 3], "visual_all": [0, 1],
                      "text_primary": [2, 3], "natural_user": [1, 2, 3], "empty": [], "singleton": [0]}

    def observer(self, **kwargs):
        return AllPairsObserver(self.model, self.plan, self.pools, rotary_module=self.rotary, **kwargs)

    def assert_clean(self):
        self.assertIs(self.rotary.apply_multimodal_rotary_pos_emb, self.original)
        for layer in self.model.model.language_model.layers:
            attention = layer.self_attn
            self.assertEqual(len(attention._forward_pre_hooks), 0)
            self.assertEqual(len(attention.k_proj._forward_hooks), 0)
            self.assertEqual(len(attention.v_proj._forward_hooks), 0)

    def test_real_forward_identity_adjacent_preserved_native_heads_and_dense_pool(self):
        expected = self.model(self.value)
        self.rotary.calls = 0
        self.rotary.results.clear()
        with self.observer(include_pair_values=True) as observer:
            actual = self.model(self.value)
            stats = observer.finish()
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(self.rotary.calls, 2)
            self.assertEqual(stats[0]["kv_heads"], 2)
            self.assertEqual(self.model.model.language_model.layers[0].self_attn.repeated_heads, 4)
            self.assertEqual(stats[0]["groups"]["visual_spatial"]["pair_cosines"]["k_raw"][0], [-1.])
            for i, layer in enumerate(self.model.model.language_model.layers):
                self.assertIs(layer.self_attn.returned_rotary, self.rotary.results[i])
            expected_pool, _ = dense(self.value.reshape(1, 4, 2, 2)[0].transpose(0, 1))
            pool = stats[0]["all_pairs"]["prompt_all"]
            torch.testing.assert_close(torch.tensor(pool["k_raw"], dtype=torch.float64), expected_pool, atol=1e-12, rtol=0)
            self.assertEqual(pool["capture_counts"], {"k_raw": 1, "v": 1, "k_rope": 1})
            self.assertEqual(len(pool["k_raw"]), 2)
            self.assertNotEqual(pool["k_raw"], pool["k_rope"])
            self.assertEqual(stats, json.loads(json.dumps(stats, allow_nan=False)))
        self.assert_clean()
        self.assertEqual(stats, observer.finish())

    def test_unsupported_pools_keep_counts_for_every_metric(self):
        with self.observer() as observer:
            self.model(self.value)
            for layer in observer.finish():
                for name, count in (("empty", 0), ("singleton", 1)):
                    pool = layer["all_pairs"][name]
                    self.assertEqual(pool["token_count"], count)
                    self.assertEqual(pool["status"], "unsupported_t_lt_2")
                    self.assertTrue(all(pool[metric] is None for metric in ("k_raw", "v", "k_rope")))
                    self.assertEqual(pool["capture_counts"], {"k_raw": 1, "v": 1, "k_rope": 1})
        self.assert_clean()

    def test_invalid_pool_positions_and_names_fail_before_hooks(self):
        for pools in ({}, {"": [0]}, {"p": [0, 0]}, {"p": [-1]}, {"p": [4]}, {"p": [True]}, {"p": "12"}):
            with self.subTest(pools=pools), self.assertRaises(ValueError):
                AllPairsObserver(self.model, self.plan, pools, rotary_module=self.rotary)
            self.assert_clean()

    def test_missing_capture_and_external_error_restore_original_hooks(self):
        self.model.model.language_model.layers[-1].self_attn.omit_rope = True
        with self.observer() as observer:
            self.model(self.value)
            with self.assertRaises(ValueError):
                observer.finish()
        self.assert_clean()
        self.model.model.language_model.layers[-1].self_attn.omit_rope = False
        self.model.model.language_model.layers[0].self_attn.fail_after_rope = True
        with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            with self.observer():
                self.model(self.value)
        self.assert_clean()

    def test_all_pairs_failure_outside_adjacent_selected_positions_is_not_silenced(self):
        # Adjacent endpoints cover positions 0,1 only; the extra pool also checks
        # position 3, so its invalid vector must fail rather than be omitted.
        plan = {"input_length": 4, "groups": {"visual_spatial": {"pairs": [[0, 1]], "selected_pair_count": 1}}}
        value = self.value.clone()
        value[0, 3] = 0
        observer = AllPairsObserver(self.model, plan, {"prompt_all": [0, 1, 2, 3]}, rotary_module=self.rotary)
        with self.assertRaisesRegex(ValueError, "zero"):
            with observer:
                self.model(value)
        self.assert_clean()
        with self.assertRaisesRegex(RuntimeError, "Observation failed"):
            observer.finish()

    def test_bfloat16_is_unchanged_and_pool_copy_prevents_late_selection_changes(self):
        self.model.to(dtype=torch.bfloat16)
        value = self.value.to(dtype=torch.bfloat16)
        expected = self.model(value)
        observer = self.observer()
        self.pools["prompt_all"].clear()
        with observer:
            actual = self.model(value)
            stats = observer.finish()
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(stats[0]["all_pairs"]["prompt_all"]["token_count"], 4)
        self.assert_clean()

    def test_matched_draws_dense_reference_negative_values_and_equal_counts(self):
        draws = {"visual_matched": [[0, 1], [2, 3], [0, 3]],
                 "text_matched": [[1, 2], [0, 2], [1, 3]]}
        expected_output = self.model(self.value)
        self.rotary.results.clear()
        raw = self.value[0].reshape(4, 2, 2).transpose(0, 1)
        value = self.model.model.language_model.layers[0].self_attn.v_proj(self.value)[0].reshape(4, 2, 2).transpose(0, 1)
        with AllPairsObserver(self.model, self.plan, self.pools, matched_pools=draws, rotary_module=self.rotary) as observer:
            actual_output = self.model(self.value)
            stats = observer.finish()
        self.assertTrue(torch.equal(actual_output, expected_output))
        rope = self.rotary.results[0][1][0]
        for name, positions in draws.items():
            group = stats[0]["matched_all_pairs"][name]
            self.assertEqual((group["token_count"], group["draw_count"], group["pair_count_per_draw"]), (2, 3, 2))
            self.assertEqual(group["status"], "supported")
            self.assertEqual(group["capture_counts"], {"k_raw": 1, "v": 1, "k_rope": 1})
            for metric, vectors in (("k_raw", raw), ("v", value), ("k_rope", rope)):
                expected_draws = torch.stack([dense(vectors[:, draw])[0] for draw in positions])
                torch.testing.assert_close(torch.tensor(group["draw_values"][metric], dtype=torch.float64), expected_draws, atol=1e-12, rtol=0)
                torch.testing.assert_close(torch.tensor(group[metric], dtype=torch.float64), expected_draws.mean(dim=0), atol=1e-12, rtol=0)
        self.assertEqual(stats[0]["matched_all_pairs"]["visual_matched"]["draw_values"]["k_raw"][0][0], -1.)
        self.assert_clean()

    def test_matched_draws_allow_repeated_draws_and_validate_within_and_between_groups(self):
        valid = {"visual_matched": [[0, 1], [0, 1]], "text_matched": [[2, 3], [2, 3]]}
        with AllPairsObserver(self.model, self.plan, self.pools, matched_pools=valid, rotary_module=self.rotary) as observer:
            self.model(self.value)
            stats = observer.finish()
        group = stats[0]["matched_all_pairs"]["visual_matched"]
        self.assertEqual(group["draw_values"]["k_raw"][0], group["draw_values"]["k_raw"][1])
        self.assert_clean()
        invalid = [
            {"p": []}, {"p": [[1, 0]]}, {"p": [[1, 1]]}, {"p": [[0, 4]]}, {"p": [[True, 1]]},
            {"p": [[0], [0, 1]]}, {"p": [[0, 1]], "q": [[2, 3], [2, 3]]},
            {"p": [[0, 1]], "q": [[2]]}, {"": [[0, 1]]},
        ]
        for pools in invalid:
            with self.subTest(pools=pools), self.assertRaises(ValueError):
                AllPairsObserver(self.model, self.plan, self.pools, matched_pools=pools, rotary_module=self.rotary)
            self.assert_clean()

    def test_singleton_matched_draws_are_explicitly_unsupported(self):
        draws = {"visual_matched": [[0], [1]], "text_matched": [[2], [3]]}
        with AllPairsObserver(self.model, self.plan, self.pools, matched_pools=draws, rotary_module=self.rotary) as observer:
            self.model(self.value)
            stats = observer.finish()
        for group in stats[0]["matched_all_pairs"].values():
            self.assertEqual(group["status"], "unsupported_t_lt_2")
            self.assertEqual(group["pair_count_per_draw"], 0)
            self.assertIsNone(group["k_raw"])
            self.assertIsNone(group["draw_values"]["k_raw"])
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()

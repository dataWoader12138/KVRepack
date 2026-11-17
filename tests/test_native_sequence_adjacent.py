"""CPU-only oracles for every consecutive pair in the complete input sequence.

These checks distinguish adjacency from the older all-off-diagonal estimator
and exercise the real hook interfaces without model weights or CUDA calls.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kvrepack.observations.native_adjacent import (
    NativeAdjacentObserver,
    adjacent_cosine_statistics,
)
from test_observation_hooks import RotaryStub
from test_observation_native_all_pairs import NativeFakeModel


def adjacent_oracle(value):
    """Enumerate exactly (0,1), ..., (N-2,N-1), using FP32-normalized vectors."""
    normalized = torch.nn.functional.normalize(value.float(), dim=-1).double()
    cosines = torch.stack([
        (normalized[:, index] * normalized[:, index + 1]).sum(dim=-1)
        for index in range(value.shape[1] - 1)
    ], dim=-1)
    return cosines.mean(dim=-1)


class AdjacentFormulaTests(unittest.TestCase):
    def test_consecutive_pairs_have_different_mean_from_all_pairs(self):
        value = torch.tensor([[[1., 0.], [1., 0.], [-1., 0.]]])
        result = adjacent_cosine_statistics(value)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["token_count"], 3)
        self.assertEqual(result["pair_count"], 2)
        torch.testing.assert_close(torch.as_tensor(result["mean"]), torch.zeros(1, dtype=torch.float64))
        normalized = value.double()
        matrix = normalized @ normalized.transpose(1, 2)
        all_pairs = matrix[:, ~torch.eye(3, dtype=torch.bool)].mean(dim=1)
        self.assertAlmostEqual(all_pairs.item(), -1. / 3.)
        self.assertGreater(abs(float(torch.as_tensor(result["mean"])[0] - all_pairs[0])), .3)

    def test_every_head_matches_explicit_adjacent_oracle(self):
        generator = torch.Generator().manual_seed(717)
        value = torch.randn(4, 9, 13, generator=generator)
        result = adjacent_cosine_statistics(value)
        self.assertEqual(result["pair_count"], 8)
        self.assertEqual(tuple(torch.as_tensor(result["mean"]).shape), (4,))
        torch.testing.assert_close(torch.as_tensor(result["mean"], dtype=torch.float64),
                                   adjacent_oracle(value), rtol=0, atol=1e-7)

    def test_last_pair_is_included_and_sequence_boundaries_are_not_filtered(self):
        # The last pair contributes -1: silently dropping it changes the mean.
        value = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., -1.]]])
        result = adjacent_cosine_statistics(value)
        self.assertEqual(result["pair_count"], 3)
        self.assertAlmostEqual(float(torch.as_tensor(result["mean"])[0]), 0.)
        # Consecutive positions 1,2 must remain even if a caller would label
        # positions 0,1 and 2,3 as different image/text/row segments.
        value[0, 3] = torch.tensor([0., 1.])
        result = adjacent_cosine_statistics(value)
        self.assertAlmostEqual(float(torch.as_tensor(result["mean"])[0]), 2. / 3., places=7)

    def test_antipodal_pairs_retain_negative_values_in_fp32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            value = torch.tensor([[[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]]], dtype=dtype)
            with self.subTest(dtype=dtype):
                result = adjacent_cosine_statistics(value)
                self.assertEqual(float(torch.as_tensor(result["mean"])[0]), -1.)
                self.assertEqual(result["pair_count"], 3)

    def test_bfloat16_uses_fp32_vector_normalization(self):
        value = torch.tensor([[[1., 2., -1.], [-2., 1., 1.], [1., -3., 2.]],
                              [[3., 1., 0.], [2., -1., 1.], [-1., 2., 2.]]], dtype=torch.bfloat16)
        result = adjacent_cosine_statistics(value)
        torch.testing.assert_close(torch.as_tensor(result["mean"], dtype=torch.float64),
                                   adjacent_oracle(value), rtol=0, atol=1e-7)

    def test_empty_and_singleton_do_not_fabricate_adjacent_mean(self):
        for tokens in (0, 1):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                adjacent_cosine_statistics(torch.ones(2, tokens, 3))

    def test_zero_and_nonfinite_vectors_anywhere_in_sequence_are_rejected(self):
        for position in (0, 1, 3):
            for invalid in (0., float("nan"), float("inf")):
                value = torch.ones(2, 4, 3)
                value[0, position] = invalid
                with self.subTest(position=position, invalid=invalid), self.assertRaises(ValueError):
                    adjacent_cosine_statistics(value)

    def test_invalid_rank_dtype_and_empty_head_dimension_are_rejected(self):
        invalid = (torch.ones(2, 3), torch.ones(1, 2, 3, 4),
                   torch.ones(2, 3, 4, dtype=torch.long),
                   torch.ones(0, 3, 4), torch.ones(2, 3, 0))
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                adjacent_cosine_statistics(value)


class NativeAdjacentObserverTests(unittest.TestCase):
    def setUp(self):
        self.value = torch.tensor([[[1., 0., 1., 0.], [-1., 0., 0., 1.],
                                    [0., 1., 1., 1.], [1., 1., -1., 1.]]])

    def build(self, family):
        rotary = RotaryStub()
        model = NativeFakeModel(family, rotary)
        name = "apply_rotary_pos_emb" if family == "qwen2" else "apply_multimodal_rotary_pos_emb"
        original = getattr(rotary, name)
        observer = NativeAdjacentObserver(model, family=family, input_length=4, rotary_module=rotary)
        return rotary, model, name, original, observer

    def assert_clean(self, rotary, model, name, original):
        self.assertIs(getattr(rotary, name), original)
        for layer in model.layers:
            attention = layer.self_attn
            self.assertEqual(len(attention._forward_pre_hooks), 0)
            self.assertEqual(len(attention.k_proj._forward_hooks), 0)
            self.assertEqual(len(attention.v_proj._forward_hooks), 0)

    def test_both_native_paths_preserve_outputs_and_capture_before_gqa_repeat(self):
        for family in ("qwen2", "qwen2_5_vl"):
            with self.subTest(family=family):
                rotary, model, name, original, observer = self.build(family)
                expected_output = model(self.value)
                projected = {"k_raw": model.layers[0].self_attn.k_proj(self.value)[0],
                             "v": model.layers[0].self_attn.v_proj(self.value)[0]}
                rotary.calls = 0
                rotary.results.clear()
                with observer:
                    actual_output = model(self.value)
                    stats = observer.finish()
                self.assertTrue(torch.equal(actual_output, expected_output))
                self.assertEqual(rotary.calls, 2)
                self.assertEqual(len(stats), 2)
                for index, layer in enumerate(model.layers):
                    self.assertEqual(stats[index]["kv_heads"], 2)
                    self.assertEqual(stats[index]["sequence_length"], 4)
                    self.assertEqual(stats[index]["adjacent_pair_count"], 3)
                    self.assertNotIn("inclusive", stats[index])
                    self.assertEqual(layer.self_attn.repeated_heads, 4)
                    self.assertIs(layer.self_attn.returned_rotary, rotary.results[index])
                    self.assertEqual(stats[index]["capture_counts"], {"k_raw": 1, "v": 1, "k_rope": 1})
                projected["k_rope"] = rotary.results[0][1][0]
                for metric, vectors in projected.items():
                    if metric != "k_rope":
                        vectors = vectors.reshape(4, 2, 2).transpose(0, 1)
                    torch.testing.assert_close(torch.as_tensor(stats[0][metric], dtype=torch.float64),
                                               adjacent_oracle(vectors), atol=1e-7, rtol=0)
                self.assertNotEqual(stats[0]["k_raw"], stats[0]["k_rope"])
                self.assertEqual(stats, json.loads(json.dumps(stats, allow_nan=False)))
                self.assert_clean(rotary, model, name, original)

    def test_bfloat16_model_forward_is_unchanged(self):
        rotary, model, name, original, observer = self.build("qwen2")
        model.to(dtype=torch.bfloat16)
        value = self.value.to(dtype=torch.bfloat16)
        expected = model(value)
        with observer:
            actual = model(value)
            stats = observer.finish()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(stats[0]["kv_heads"], 2)
        self.assertFalse(hasattr(model.layers[0].self_attn, "num_key_value_heads"))
        self.assert_clean(rotary, model, name, original)

    def test_missing_layers_or_rope_and_repeated_projection_fail(self):
        for family in ("qwen2", "qwen2_5_vl"):
            rotary, model, name, original, observer = self.build(family)
            model.skip_last = True
            with observer:
                model(self.value)
                with self.assertRaises(ValueError):
                    observer.finish()
            self.assert_clean(rotary, model, name, original)
            rotary, model, name, original, observer = self.build(family)
            model.layers[-1].self_attn.omit_rope = True
            with observer:
                model(self.value)
                with self.assertRaises(ValueError):
                    observer.finish()
            self.assert_clean(rotary, model, name, original)
            rotary, model, name, original, observer = self.build(family)
            model.layers[0].self_attn.repeat_projection = True
            with self.assertRaises(ValueError):
                with observer:
                    model(self.value)
            self.assert_clean(rotary, model, name, original)

    def test_errors_wrong_batch_shape_and_repeated_gqa_heads_restore_hooks(self):
        for kind in ("model_exception", "wrong_batch", "wrong_length", "wrong_heads", "bad_vector"):
            rotary, model, name, original, observer = self.build("qwen2_5_vl")
            value = self.value.clone()
            if kind == "model_exception":
                model.layers[0].self_attn.fail_after_rope = True
            elif kind == "wrong_batch":
                value = value.repeat(2, 1, 1)
            elif kind == "wrong_length":
                value = value[:, :3]
            elif kind == "wrong_heads":
                model.layers[0].self_attn.bad_rope_heads = True
            else:
                value[0, -1] = 0
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                with observer:
                    model(value)
            self.assert_clean(rotary, model, name, original)
            with self.assertRaises(RuntimeError):
                observer.finish()

    def test_finish_is_independent_copy_and_requires_exactly_one_complete_forward(self):
        rotary, model, name, original, observer = self.build("qwen2")
        with self.assertRaises(RuntimeError):
            observer.finish()
        with observer:
            with self.assertRaises(ValueError):
                observer.finish()
            model(self.value)
            result = observer.finish()
            result[0]["k_raw"].clear()
            self.assertEqual(len(observer.finish()[0]["k_raw"]), 2)
            with self.assertRaises(ValueError):
                model(self.value)
            with self.assertRaises(RuntimeError):
                observer.finish()
        self.assert_clean(rotary, model, name, original)

    def test_adjacent_observer_rejects_single_token_and_invalid_family(self):
        rotary = RotaryStub()
        model = NativeFakeModel("qwen2", rotary)
        for family, length in (("qwen2", 1), ("qwen2", True), ("unavailable", 4)):
            with self.subTest(family=family, length=length), self.assertRaises(ValueError):
                NativeAdjacentObserver(model, family=family, input_length=length, rotary_module=rotary)


if __name__ == "__main__":
    unittest.main()

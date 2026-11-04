"""Real CPU torch tensors exercise hooks without loading a model or GPU."""
import copy
import json
import unittest

import torch
from torch import nn

from kvrepack.observations.hooks import AdjacentObserver


class RotaryStub:
    def __init__(self):
        self.calls = 0
        self.results = []

        def apply(q, k, *args, **kwargs):
            self.calls += 1
            # Position-dependent orthogonal rotation of each 2-D key vector.
            flipped = k.flip(-1).clone()
            flipped[..., 0] *= -1
            rotated = k.clone()
            rotated[:, :, 1::2] = flipped[:, :, 1::2]
            result = (q, rotated)
            self.results.append(result)
            return result

        self.apply_multimodal_rotary_pos_emb = apply


class FakeAttention(nn.Module):
    def __init__(self, layer, rotary):
        super().__init__()
        self.layer_idx = layer
        self.num_key_value_heads = 2
        self.num_heads = 4
        self.head_dim = 2
        self.rotary = rotary
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.k_proj.weight.copy_(torch.eye(4))
            self.v_proj.weight.copy_(torch.diag(torch.tensor([1.0, -1.0, 2.0, -2.0])))
        self.repeated_heads = None
        self.returned_rotary = None
        self.omit_rope = False
        self.repeat_projection = False
        self.bad_rope_heads = False
        self.fail_after_rope = False

    def forward(self, hidden_states):
        key = self.k_proj(hidden_states)
        if self.repeat_projection:
            self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        batch, length, _ = key.shape
        key = key.reshape(batch, length, 2, 2).transpose(1, 2)
        value = value.reshape(batch, length, 2, 2).transpose(1, 2)
        query = key.repeat_interleave(2, dim=1)
        if not self.omit_rope:
            if self.bad_rope_heads:
                key = key.repeat_interleave(2, dim=1)
            result = self.rotary.apply_multimodal_rotary_pos_emb(query, key, None, None, [1])
            self.returned_rotary = result
            _, key = result
        if self.fail_after_rope:
            raise RuntimeError("synthetic model failure")
        # Mimic later GQA expansion: the observer must have seen two KV heads.
        repeated = (key + value).repeat_interleave(2, dim=1)
        self.repeated_heads = repeated.shape[1]
        return hidden_states + repeated.reshape(batch, 2, 2, length, 2).mean(dim=2).transpose(1, 2).reshape(batch, length, 4)


class FakeModel(nn.Module):
    def __init__(self, rotary, layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList()
        for index in range(layers):
            layer = nn.Module()
            layer.self_attn = FakeAttention(index, rotary)
            self.model.language_model.layers.append(layer)
        self.skip_last = False

    def forward(self, values):
        layers = self.model.language_model.layers
        for layer in (list(layers)[:-1] if self.skip_last else layers):
            values = layer.self_attn(values)
        return values


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.rotary = RotaryStub()
        self.original = self.rotary.apply_multimodal_rotary_pos_emb
        self.model = FakeModel(self.rotary)
        self.values = torch.tensor([[[1., 0., 1., 0.], [-1., 0., 0., 1.],
                                     [0., 1., 1., 1.], [1., 1., -1., 1.]]])
        self.plan = {"input_length": 4, "groups": {
            "visual_spatial": {"pairs": [[0, 1], [2, 3]], "selected_pair_count": 2},
            "text_question": {"pairs": [[1, 2]], "selected_pair_count": 1},
            "text_instruction": {"pairs": [], "selected_pair_count": 0}}}

    def observer(self, **kwargs):
        return AdjacentObserver(self.model, self.plan, rotary_module=self.rotary, **kwargs)

    def assert_clean(self):
        self.assertIs(self.rotary.apply_multimodal_rotary_pos_emb, self.original)
        for layer in self.model.model.language_model.layers:
            attention = layer.self_attn
            self.assertEqual(len(attention._forward_pre_hooks), 0)
            self.assertEqual(len(attention.k_proj._forward_hooks), 0)
            self.assertEqual(len(attention.v_proj._forward_hooks), 0)

    def test_exact_forward_identity_and_original_tuple_returned_once(self):
        expected = self.model(self.values)
        self.rotary.calls = 0
        self.rotary.results.clear()
        with self.observer(include_pair_values=True) as observer:
            actual = self.model(self.values)
            stats = observer.finish()
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(self.rotary.calls, 2)
            for index, layer in enumerate(self.model.model.language_model.layers):
                self.assertIs(layer.self_attn.returned_rotary, self.rotary.results[index])
                self.assertEqual(layer.self_attn.repeated_heads, 4)
                self.assertEqual(stats[index]["kv_heads"], 2)
                self.assertEqual(stats[index]["capture_counts"], {"k_raw": 1, "v": 1, "k_rope": 1})
            self.assertEqual(stats, json.loads(json.dumps(stats, allow_nan=False)))
        self.assert_clean()
        self.assertEqual(observer.finish(), stats)

    def test_known_per_head_cosines_include_negative_values(self):
        with self.observer(include_pair_values=True) as observer:
            self.model(self.values)
            group = observer.finish()[0]["groups"]["visual_spatial"]
        self.assertAlmostEqual(group["pair_cosines"]["k_raw"][0][0], -1.0, places=6)
        self.assertAlmostEqual(group["pair_cosines"]["k_raw"][1][0], 0.0, places=6)
        self.assertAlmostEqual(group["k_raw"][0], (-1 + 2 ** -0.5) / 2, places=6)
        self.assertNotEqual(group["k_raw"], group["k_rope"])
        self.assert_clean()

    def test_bfloat16_forward_is_unchanged_and_cosines_use_float32(self):
        self.model.to(dtype=torch.bfloat16)
        values = self.values.to(dtype=torch.bfloat16)
        expected = self.model(values)
        with self.observer() as observer:
            actual = self.model(values)
            stats = observer.finish()
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(stats[0]["dtype"], "torch.bfloat16")
        cosine = stats[0]["groups"]["visual_spatial"]["k_raw"][0]
        self.assertAlmostEqual(cosine, (-1 + 2 ** -0.5) / 2, places=6)
        self.assert_clean()

    def test_zero_pair_group_omitted_but_every_layer_observed(self):
        with self.observer() as observer:
            self.model(self.values)
            stats = observer.finish()
        self.assertNotIn("text_instruction", stats[0]["groups"])
        self.assertEqual(len(stats), 2)
        self.assertNotIn("pair_cosines", stats[0]["groups"]["visual_spatial"])

    def test_all_zero_groups_are_valid_after_complete_forward(self):
        for group in self.plan["groups"].values():
            group.update(pairs=[], selected_pair_count=0)
        with self.observer() as observer:
            self.model(self.values)
            self.assertEqual([layer["groups"] for layer in observer.finish()], [{}, {}])
        self.assert_clean()

    def test_missing_layer_or_metric_cannot_finish(self):
        self.model.skip_last = True
        with self.observer() as observer:
            self.model(self.values)
            with self.assertRaisesRegex(ValueError, "missing actual"):
                observer.finish()
        self.assert_clean()
        self.model.skip_last = False
        self.model.model.language_model.layers[-1].self_attn.omit_rope = True
        with self.observer() as observer:
            self.model(self.values)
            with self.assertRaisesRegex(ValueError, "metric capture"):
                observer.finish()
        self.assert_clean()

    def test_duplicate_projection_capture_fails_and_cleanup_is_complete(self):
        self.model.model.language_model.layers[0].self_attn.repeat_projection = True
        with self.assertRaisesRegex(ValueError, "repeated k_raw"):
            with self.observer() as observer:
                self.model(self.values)
        self.assert_clean()
        with self.assertRaisesRegex(RuntimeError, "Observation failed"):
            observer.finish()

    def test_second_model_forward_is_rejected(self):
        with self.observer() as observer:
            self.model(self.values)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                self.model(self.values)
            with self.assertRaisesRegex(RuntimeError, "Observation failed"):
                observer.finish()
        self.assert_clean()

    def test_batch_decode_and_wrong_sequence_length_are_rejected(self):
        for values in (self.values.repeat(2, 1, 1), self.values[:, :1], self.values[:, :3]):
            with self.subTest(shape=list(values.shape)), self.assertRaisesRegex(ValueError, "full-prefill"):
                with self.observer():
                    self.model(values)
            self.assert_clean()

    def test_post_gqa_rope_keys_are_rejected(self):
        self.model.model.language_model.layers[0].self_attn.bad_rope_heads = True
        with self.assertRaisesRegex(ValueError, "before GQA"):
            with self.observer():
                self.model(self.values)
        self.assert_clean()

    def test_wrong_projection_shape_and_missing_rope_between_layers_fail(self):
        attention = self.model.model.language_model.layers[0].self_attn
        handle = attention.k_proj.register_forward_hook(lambda module, args, output: output[..., :-1])
        try:
            with self.assertRaisesRegex(ValueError, "expected.*before GQA"):
                with self.observer():
                    self.model(self.values)
        finally:
            handle.remove()
        self.assert_clean()
        attention.omit_rope = True
        with self.assertRaisesRegex(ValueError, "Previous language attention"):
            with self.observer():
                self.model(self.values)
        self.assert_clean()

    def test_nan_inf_and_zero_selected_vector_are_rejected(self):
        for value in (float("nan"), float("inf"), 0.0):
            values = self.values.clone()
            values[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite|zero"):
                with self.observer():
                    self.model(values)
            self.assert_clean()

    def test_nonfloating_projection_and_dtype_mismatch_are_rejected(self):
        attention = self.model.model.language_model.layers[0].self_attn
        for conversion in (lambda output: output.long(), lambda output: output.double()):
            handle = attention.v_proj.register_forward_hook(lambda module, args, output: conversion(output))
            try:
                with self.assertRaisesRegex(ValueError, "floating|dtype mismatch"):
                    with self.observer():
                        self.model(self.values)
            finally:
                handle.remove()
            self.assert_clean()

    def test_external_model_failure_restores_state(self):
        self.model.model.language_model.layers[0].self_attn.fail_after_rope = True
        with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            with self.observer():
                self.model(self.values)
        self.assert_clean()

    def test_partial_hook_install_failure_restores_prior_hooks_and_rotary(self):
        projection = self.model.model.language_model.layers[0].self_attn.v_proj
        original_registration = projection.register_forward_hook
        projection.register_forward_hook = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("install failure"))
        try:
            with self.assertRaisesRegex(RuntimeError, "install failure"):
                with self.observer():
                    pass
        finally:
            projection.register_forward_hook = original_registration
        self.assert_clean()

    def test_nested_observer_rejected_without_disturbing_outer(self):
        with self.observer() as outer:
            wrapper = self.rotary.apply_multimodal_rotary_pos_emb
            with self.assertRaisesRegex(RuntimeError, "already has"):
                with self.observer():
                    pass
            self.assertIs(self.rotary.apply_multimodal_rotary_pos_emb, wrapper)
            self.model(self.values)
            self.assertEqual(len(outer.finish()), 2)
        self.assert_clean()

    def test_invalid_pairs_and_noncontiguous_layers_fail_before_install(self):
        for pair in ([-1, 1], [1, 1], [1, 4], [2, 1], [True, 2], [1, 2, 3]):
            plan = copy.deepcopy(self.plan)
            plan["groups"]["visual_spatial"].update(pairs=[pair], selected_pair_count=1)
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                AdjacentObserver(self.model, plan, rotary_module=self.rotary)
        self.model.model.language_model.layers[1].self_attn.layer_idx = 5
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.observer()
        self.assert_clean()

    def test_finish_requires_enter_and_result_is_independent_copy(self):
        observer = self.observer()
        with self.assertRaisesRegex(RuntimeError, "not been entered"):
            observer.finish()
        with observer:
            with self.assertRaises(ValueError):
                observer.finish()
            self.model(self.values)
            result = observer.finish()
            result[0]["groups"].clear()
            self.assertTrue(observer.finish()[0]["groups"])
        with self.assertRaisesRegex(RuntimeError, "cannot be entered twice"):
            with observer:
                pass
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()

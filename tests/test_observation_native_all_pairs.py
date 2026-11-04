"""CPU fake native-model interfaces; no checkpoint construction or GPU calls."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvrepack.observations.native_all_pairs import NativeAllPairsObserver
from test_observation_hooks import FakeAttention, RotaryStub


class NativeFakeModel(nn.Module):
    def __init__(self, family, rotary, layers=2):
        super().__init__()
        self.model = nn.Module()
        if family == "qwen2_5_vl":
            self.model.language_model = nn.Module()
            decoder = self.model.language_model
        else:
            decoder = self.model
        decoder.layers = nn.ModuleList()
        self.family = family
        if family == "qwen2":
            rotary.apply_rotary_pos_emb = rotary.apply_multimodal_rotary_pos_emb
            # FakeAttention uses the VL spelling; this proxy routes its call
            # to the real selected Qwen2 rotary symbol observed by the class.
            selected_rotary = types.SimpleNamespace(
                apply_multimodal_rotary_pos_emb=lambda *args, **kwargs: rotary.apply_rotary_pos_emb(*args, **kwargs))
        else:
            selected_rotary = rotary
        for index in range(layers):
            layer = nn.Module()
            attention = FakeAttention(index, selected_rotary)
            attention.config = types.SimpleNamespace(num_key_value_heads=2, num_attention_heads=4, hidden_size=8)
            if family == "qwen2":
                # This is the important current-HF interface difference.
                del attention.num_key_value_heads
                del attention.num_heads
            layer.self_attn = attention
            decoder.layers.append(layer)
        self.skip_last = False

    @property
    def layers(self):
        return self.model.layers if self.family == "qwen2" else self.model.language_model.layers

    def forward(self, values):
        for layer in list(self.layers)[:-1] if self.skip_last else self.layers:
            values = layer.self_attn(values)
        return values


def dense(value):
    normalized = torch.nn.functional.normalize(value.float(), dim=-1).double()
    similarities = normalized @ normalized.transpose(1, 2)
    tokens = value.shape[1]
    off = similarities[:, ~torch.eye(tokens, dtype=torch.bool)].mean(dim=1)
    return off, similarities.mean(dim=(1, 2))


class NativeAllPairsTests(unittest.TestCase):
    def setUp(self):
        self.value = torch.tensor([[[1., 0., 1., 0.], [-1., 0., 0., 1.],
                                    [0., 1., 1., 1.], [1., 1., -1., 1.]]])

    def build(self, family):
        rotary = RotaryStub()
        model = NativeFakeModel(family, rotary)
        name = "apply_rotary_pos_emb" if family == "qwen2" else "apply_multimodal_rotary_pos_emb"
        original = getattr(rotary, name)
        observer = NativeAllPairsObserver(model, family=family, input_length=4, rotary_module=rotary)
        return rotary, model, name, original, observer

    def assert_clean(self, rotary, model, name, original):
        self.assertIs(getattr(rotary, name), original)
        for layer in model.layers:
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)
            self.assertEqual(len(layer.self_attn.k_proj._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn.v_proj._forward_hooks), 0)

    def test_both_native_paths_exact_forward_tuple_identity_heads_and_dense_metrics(self):
        for family in ("qwen2", "qwen2_5_vl"):
            with self.subTest(family=family):
                rotary, model, name, original, observer = self.build(family)
                expected_output = model(self.value)
                projected = {"k_raw": model.layers[0].self_attn.k_proj(self.value)[0],
                             "v": model.layers[0].self_attn.v_proj(self.value)[0]}
                rotary.calls = 0
                rotary.results.clear()
                with observer:
                    actual = model(self.value)
                    result = observer.finish()
                self.assertTrue(torch.equal(actual, expected_output))
                self.assertEqual(rotary.calls, 2)
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0]["kv_heads"], 2)
                self.assertEqual(model.layers[0].self_attn.repeated_heads, 4)
                for index, layer in enumerate(model.layers):
                    self.assertIs(layer.self_attn.returned_rotary, rotary.results[index])
                projected["k_rope"] = rotary.results[0][1][0]
                for metric, values in projected.items():
                    if metric != "k_rope":
                        values = values.reshape(4, 2, 2).transpose(0, 1)
                    off, inclusive = dense(values)
                    torch.testing.assert_close(torch.tensor(result[0][metric], dtype=torch.float64), off, atol=1e-12, rtol=0)
                    torch.testing.assert_close(torch.tensor(result[0]["inclusive"][metric], dtype=torch.float64), inclusive, atol=1e-12, rtol=0)
                self.assertEqual(result[0]["capture_counts"], {"k_raw": 1, "v": 1, "k_rope": 1})
                self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))
                self.assert_clean(rotary, model, name, original)

    def test_negative_off_diagonal_is_not_clipped_to_zero(self):
        rotary = RotaryStub()
        model = NativeFakeModel("qwen2", rotary, layers=1)
        value = torch.tensor([[[1., 0., 1., 0.], [-1., 0., -1., 0.]]])
        with NativeAllPairsObserver(model, family="qwen2", input_length=2, rotary_module=rotary) as observer:
            model(value)
            result = observer.finish()
        self.assertEqual(result[0]["k_raw"], [-1., -1.])
        self.assertEqual(result[0]["v"], [-1., -1.])

    def test_bfloat16_forward_is_exact_and_config_head_fallback_is_used(self):
        rotary, model, name, original, observer = self.build("qwen2")
        model.to(dtype=torch.bfloat16)
        values = self.value.to(dtype=torch.bfloat16)
        expected = model(values)
        with observer:
            actual = model(values)
            result = observer.finish()
        self.assertFalse(hasattr(model.layers[0].self_attn, "num_key_value_heads"))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(result[0]["kv_heads"], 2)
        self.assert_clean(rotary, model, name, original)

    def test_full_batch_and_one_forward_guards_restore_every_hook(self):
        for values in (self.value.repeat(2, 1, 1), self.value[:, :3]):
            rotary, model, name, original, observer = self.build("qwen2")
            with self.assertRaisesRegex(ValueError, "batch-one"):
                with observer:
                    model(values)
            self.assert_clean(rotary, model, name, original)
        rotary, model, name, original, observer = self.build("qwen2_5_vl")
        with observer:
            model(self.value)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                model(self.value)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                observer.finish()
        self.assert_clean(rotary, model, name, original)

    def test_missing_layer_and_duplicate_projection_are_not_silently_accepted(self):
        rotary, model, name, original, observer = self.build("qwen2")
        model.skip_last = True
        with observer:
            model(self.value)
            with self.assertRaisesRegex(ValueError, "missing language"):
                observer.finish()
        self.assert_clean(rotary, model, name, original)
        rotary, model, name, original, observer = self.build("qwen2_5_vl")
        model.layers[0].self_attn.repeat_projection = True
        with self.assertRaisesRegex(ValueError, "repeated native"):
            with observer:
                model(self.value)
        self.assert_clean(rotary, model, name, original)

    def test_model_exception_and_invalid_all_token_vector_restore_global_rotary(self):
        for family in ("qwen2", "qwen2_5_vl"):
            rotary, model, name, original, observer = self.build(family)
            model.layers[0].self_attn.fail_after_rope = True
            with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
                with observer:
                    model(self.value)
            self.assert_clean(rotary, model, name, original)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                observer.finish()
            rotary, model, name, original, observer = self.build(family)
            value = self.value.clone()
            value[0, 3] = 0
            with self.assertRaisesRegex(ValueError, "zero"):
                with observer:
                    model(value)
            self.assert_clean(rotary, model, name, original)

    def test_missing_rope_wrong_gqa_heads_and_partial_install_cleanup(self):
        rotary, model, name, original, observer = self.build("qwen2")
        model.layers[-1].self_attn.omit_rope = True
        with observer:
            model(self.value)
            with self.assertRaisesRegex(ValueError, "metric capture"):
                observer.finish()
        self.assert_clean(rotary, model, name, original)
        rotary, model, name, original, observer = self.build("qwen2_5_vl")
        model.layers[0].self_attn.bad_rope_heads = True
        with self.assertRaisesRegex(ValueError, "before GQA"):
            with observer:
                model(self.value)
        self.assert_clean(rotary, model, name, original)
        rotary, model, name, original, observer = self.build("qwen2")
        projection = model.layers[0].self_attn.v_proj
        registration = projection.register_forward_hook
        projection.register_forward_hook = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("install failure"))
        try:
            with self.assertRaisesRegex(RuntimeError, "install failure"):
                with observer:
                    pass
        finally:
            projection.register_forward_hook = registration
        self.assert_clean(rotary, model, name, original)

    def test_nested_observer_and_invalid_metadata_fail_without_disturbing_active_one(self):
        rotary, model, name, original, outer = self.build("qwen2")
        with outer:
            wrapper = getattr(rotary, name)
            with self.assertRaisesRegex(RuntimeError, "active observer"):
                with NativeAllPairsObserver(model, family="qwen2", input_length=4, rotary_module=rotary):
                    pass
            self.assertIs(getattr(rotary, name), wrapper)
            model(self.value)
            self.assertEqual(len(outer.finish()), 2)
        self.assert_clean(rotary, model, name, original)
        for family, length in (("other", 4), ("qwen2", 1), ("qwen2", True)):
            with self.assertRaises(ValueError):
                NativeAllPairsObserver(model, family=family, input_length=length, rotary_module=rotary)
        with self.assertRaisesRegex(ValueError, "decoder path"):
            NativeAllPairsObserver(model, family="qwen2_5_vl", input_length=4, rotary_module=rotary)

    def test_finish_requires_capture_returns_independent_copy_and_disallows_reentry(self):
        rotary, model, name, original, observer = self.build("qwen2_5_vl")
        with self.assertRaisesRegex(RuntimeError, "not been entered"):
            observer.finish()
        with observer:
            with self.assertRaises(ValueError):
                observer.finish()
            model(self.value)
            result = observer.finish()
            result[0]["k_raw"].clear()
            self.assertEqual(len(observer.finish()[0]["k_raw"]), 2)
        self.assert_clean(rotary, model, name, original)
        with self.assertRaisesRegex(RuntimeError, "entered twice"):
            with observer:
                pass


if __name__ == "__main__":
    unittest.main()

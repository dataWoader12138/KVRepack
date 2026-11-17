"""CPU geometric oracles and unchanged-forward checks for raw-K diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvrepack.observations.native_adjacent import adjacent_cosine_statistics
from kvrepack.observations.native_key_diagnostics import (
    DIAGNOSTIC_METRICS, NativeKeyDiagnosticsObserver, key_diagnostic_statistics,
)
from test_observation_hooks import RotaryStub
from test_observation_native_all_pairs import NativeFakeModel


def adjacent_oracle(value):
    normalized = torch.nn.functional.normalize(value.float(), dim=-1).double()
    return torch.stack([(normalized[:, i]*normalized[:, i+1]).sum(dim=-1).clamp(-1, 1)
                        for i in range(value.shape[1]-1)], dim=-1).mean(dim=-1)


def all_pairs_oracle(value):
    normalized = torch.nn.functional.normalize(value.float(), dim=-1).double()
    dense = normalized@normalized.transpose(-1, -2)
    mask = ~torch.eye(value.shape[1], dtype=torch.bool)
    return dense[:, mask].mean(dim=-1)


class KeyDiagnosticFormulaTests(unittest.TestCase):
    def test_explicit_dense_and_centered_adjacent_oracles_for_every_head(self):
        generator = torch.Generator().manual_seed(4117)
        for dtype in (torch.float32, torch.bfloat16):
            value = torch.randn(4, 7, 13, generator=generator).to(dtype)
            bias = torch.randn(4, 13, generator=generator).to(dtype)
            result = key_diagnostic_statistics(value, bias)
            fp32 = value.float()
            expected = dict(k_raw_all_pairs=all_pairs_oracle(fp32),
                            k_raw_centered_adjacent=adjacent_oracle(fp32-fp32.mean(dim=1, keepdim=True)),
                            k_raw_without_bias_adjacent=adjacent_oracle(fp32-bias.float()[:, None, :]))
            for name, oracle in expected.items():
                with self.subTest(dtype=dtype, metric=name):
                    torch.testing.assert_close(torch.tensor(result[name], dtype=torch.float64), oracle, rtol=0, atol=1e-12)
            self.assertTrue(result['k_proj_bias_present'])
            torch.testing.assert_close(torch.tensor(result['k_proj_bias_norm']), torch.linalg.vector_norm(bias.float(), dim=-1))

    def test_shared_bias_can_raise_raw_cosine_while_subtraction_lowers_it(self):
        residual = torch.tensor([[[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]])
        bias = torch.tensor([[0., 20.]])
        value = residual+bias[:, None, :]
        raw_adjacent = adjacent_cosine_statistics(value)['mean'][0].item()
        result = key_diagnostic_statistics(value, bias)
        self.assertGreater(raw_adjacent, .99)
        self.assertLess(result['k_raw_without_bias_adjacent'][0], -.66)
        self.assertLess(result['k_raw_centered_adjacent'][0], -.66)
        self.assertGreater(result['k_raw_all_pairs'][0], .99)
        self.assertEqual(result['k_proj_bias_norm'], [20.])

    def test_no_projection_bias_equals_original_raw_adjacency(self):
        value = torch.tensor([[[1., 2.], [-1., 2.], [2., -1.], [-2., -1.]]])
        result = key_diagnostic_statistics(value)
        self.assertFalse(result['k_proj_bias_present'])
        self.assertEqual(result['k_proj_bias_norm'], [0.])
        self.assertEqual(result['k_raw_without_bias_adjacent'], adjacent_cosine_statistics(value)['mean'].tolist())

    def test_undefined_centered_or_unbiased_norm_is_an_error_not_zero(self):
        with self.assertRaisesRegex(ValueError, 'Zero or nonfinite'):
            key_diagnostic_statistics(torch.ones(2, 4, 3))
        value = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
        with self.assertRaisesRegex(ValueError, 'Zero or nonfinite'):
            key_diagnostic_statistics(value, torch.tensor([[1., 0.]]))

    def test_invalid_inputs_bias_and_nonfinite_raw_are_rejected(self):
        value = torch.tensor([[[1., 2.], [-1., 2.], [2., -1.]]])
        for bad in (value[:, :1], value.long(), value[0], value.repeat(1, 1, 1).fill_(float('nan'))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                key_diagnostic_statistics(bad)
        for bias in (torch.ones(2), torch.ones(2, 2), torch.ones(1, 2, dtype=torch.int64), torch.full((1, 2), float('inf'))):
            with self.subTest(bias=bias), self.assertRaises(ValueError):
                key_diagnostic_statistics(value, bias)


class NativeKeyDiagnosticObserverTests(unittest.TestCase):
    def setUp(self):
        self.value = torch.tensor([[[1., 0., 1., 0.], [-1., 0., 0., 1.],
                                    [0., 1., 1., 1.], [1., 1., -1., 1.]]])

    def build(self, family, *, bias=False, dtype=torch.float32):
        rotary = RotaryStub(); model = NativeFakeModel(family, rotary)
        if bias:
            for layer in model.layers:
                layer.self_attn.k_proj.bias = nn.Parameter(torch.tensor([3., -2., 4., 1.]))
        model.to(dtype=dtype)
        name = 'apply_rotary_pos_emb' if family == 'qwen2' else 'apply_multimodal_rotary_pos_emb'
        original = getattr(rotary, name)
        observer = NativeKeyDiagnosticsObserver(model, family=family, input_length=4, rotary_module=rotary)
        return rotary, model, name, original, observer

    def assert_clean(self, rotary, model, name, original):
        self.assertIs(getattr(rotary, name), original)
        for layer in model.layers:
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)
            self.assertEqual(len(layer.self_attn.k_proj._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn.v_proj._forward_hooks), 0)

    def test_both_native_paths_bias_cases_preserve_forward_and_match_oracles(self):
        for family in ('qwen2', 'qwen2_5_vl'):
            for bias in (False, True):
                rotary, model, name, original, observer = self.build(family, bias=bias)
                expected = model(self.value)
                projected = model.layers[0].self_attn.k_proj(self.value)[0].reshape(4, 2, 2).transpose(0, 1)
                projection_bias = model.layers[0].self_attn.k_proj.bias
                if projection_bias is not None:
                    projection_bias = projection_bias.reshape(2, 2)
                oracle = key_diagnostic_statistics(projected, projection_bias)
                rotary.calls = 0; rotary.results.clear()
                with observer:
                    actual = model(self.value)
                    result = observer.finish()
                with self.subTest(family=family, bias=bias):
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(rotary.calls, 2)
                    for layer_id, layer in enumerate(model.layers):
                        row = result[layer_id]
                        self.assertEqual(row['kv_heads'], 2)
                        self.assertEqual(layer.self_attn.repeated_heads, 4)
                        self.assertIs(layer.self_attn.returned_rotary, rotary.results[layer_id])
                        self.assertEqual(row['capture_counts'], {'k_raw': 1, 'v': 1, 'k_rope': 1})
                        self.assertEqual(row['key_diagnostic_capture_count'], 1)
                        self.assertEqual(row['k_proj_bias_present'], bias)
                    for metric in DIAGNOSTIC_METRICS:
                        torch.testing.assert_close(torch.tensor(result[0][metric], dtype=torch.float64),
                                                   torch.tensor(oracle[metric], dtype=torch.float64), rtol=0, atol=1e-12)
                    if not bias:
                        self.assertEqual(result[0]['k_raw_without_bias_adjacent'], result[0]['k_raw'])
                    self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))
                    self.assert_clean(rotary, model, name, original)

    def test_bfloat16_forward_and_parameter_bytes_are_unchanged(self):
        rotary, model, name, original, observer = self.build('qwen2', bias=True, dtype=torch.bfloat16)
        values = self.value.bfloat16()
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        expected = model(values)
        with observer:
            actual = model(values); result = observer.finish()
        self.assertTrue(torch.equal(actual, expected))
        for name, value in model.named_parameters(): self.assertTrue(torch.equal(value, parameters[name]))
        self.assertEqual(result[0]['kv_heads'], 2)
        # Captured BF16 outputs are promoted first; bias subtraction is a
        # post-hoc probe rather than a second, bias-disabled model forward.
        self.assertNotEqual(result[0]['k_raw_without_bias_adjacent'], result[0]['k_raw'])
        self.assert_clean(rotary, model, name='apply_rotary_pos_emb', original=original)

    def test_capture_failure_and_undefined_geometry_restore_every_hook(self):
        for failure in ('repeated_projection', 'model_failure', 'constant_centered', 'missing_rope'):
            rotary, model, name, original, observer = self.build('qwen2_5_vl')
            values = self.value.clone()
            if failure == 'repeated_projection': model.layers[0].self_attn.repeat_projection = True
            elif failure == 'model_failure': model.layers[0].self_attn.fail_after_rope = True
            elif failure == 'constant_centered': values.fill_(1.)
            else: model.layers[-1].self_attn.omit_rope = True
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                with observer:
                    model(values); observer.finish()
            self.assert_clean(rotary, model, name, original)

    def test_finish_checks_extra_captures_and_returns_independent_copy(self):
        rotary, model, name, original, observer = self.build('qwen2')
        with self.assertRaises(RuntimeError): observer.finish()
        with observer:
            model(self.value)
            result = observer.finish(); result[0]['k_raw_all_pairs'].clear()
            self.assertEqual(len(observer.finish()[0]['k_raw_all_pairs']), 2)
            observer._stats[0]['key_diagnostic_capture_count'] = 0
            with self.assertRaisesRegex(ValueError, 'diagnostic capture'): observer.finish()
            observer._stats[0]['key_diagnostic_capture_count'] = 1
            observer._stats[0]['k_raw_centered_adjacent'] = [float('nan')]*2
            with self.assertRaisesRegex(ValueError, 'diagnostic cosine'): observer.finish()
        self.assert_clean(rotary, model, name, original)


if __name__ == '__main__':
    unittest.main()

"""CPU oracles for exact per-input raw-K top adjacent pairs."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kvrepack.observations.native_top_adjacent import RawKTopAdjacentObserver, top_adjacent_cosines
from kvrepack.observations.native_adjacent import adjacent_cosine_statistics
from test_observation_hooks import RotaryStub
from test_observation_native_all_pairs import NativeFakeModel


def explicit_top_oracle(value, count):
    normalized = torch.nn.functional.normalize(value.float(), dim=-1).double()
    scores, starts = [], []
    for head in range(value.shape[0]):
        pairs = [((normalized[head, index] * normalized[head, index+1]).sum().clamp(-1, 1).item(), index)
                 for index in range(value.shape[1]-1)]
        chosen = sorted(pairs, key=lambda pair: (-pair[0], pair[1]))[:count]
        scores.append([score for score, _ in chosen])
        starts.append([index for _, index in chosen])
    return torch.tensor(scores, dtype=torch.float64), torch.tensor(starts, dtype=torch.long)


class TopAdjacentFormulaTests(unittest.TestCase):
    def test_exact_explicit_order_and_means_for_each_head(self):
        value = torch.randn(4, 23, 11, generator=torch.Generator().manual_seed(192))
        expected_scores, expected_starts = explicit_top_oracle(value, 7)
        result = top_adjacent_cosines(value, 7)
        self.assertEqual(result['pair_count'], 22)
        self.assertEqual(result['token_count'], 23)
        self.assertEqual(result['top_k'], 7)
        self.assertEqual(result['status'], 'supported')
        self.assertTrue(torch.equal(result['pair_starts'], expected_starts))
        torch.testing.assert_close(result['scores'], expected_scores, rtol=0, atol=1e-7)
        torch.testing.assert_close(result['mean'], expected_scores.mean(-1), rtol=0, atol=1e-7)
        torch.testing.assert_close(result['minimum'], expected_scores[:, -1], rtol=0, atol=1e-7)
        self.assertEqual(result['scores'].dtype, torch.float64)
        self.assertGreaterEqual(float((result['mean']-adjacent_cosine_statistics(value)['mean']).min()), 0.)

    def test_default_count_is_exactly_200_distinct_pairs_per_head(self):
        value = torch.randn(3, 402, 7, generator=torch.Generator().manual_seed(11))
        result = top_adjacent_cosines(value)
        self.assertEqual(result['pair_starts'].shape, (3, 200))
        for row in result['pair_starts'].tolist():
            self.assertEqual(len(set(row)), 200)
        expected_scores, expected_starts = explicit_top_oracle(value, 200)
        self.assertTrue(torch.equal(result['pair_starts'], expected_starts))
        torch.testing.assert_close(result['scores'], expected_scores, rtol=0, atol=1e-7)

    def test_ties_keep_ascending_original_pair_start(self):
        value = torch.tensor([[[1., 0.], [1., 0.], [-1., 0.], [-1., 0.], [-1., 0.], [1., 0.]]])
        result = top_adjacent_cosines(value, 4)
        self.assertEqual(result['pair_starts'].tolist(), [[0, 2, 3, 1]])
        self.assertEqual(result['scores'].tolist(), [[1., 1., 1., -1.]])
        self.assertEqual(result['mean'].item(), .5)
        self.assertEqual(result['minimum'].item(), -1.)

    def test_last_pair_and_all_sequence_boundaries_are_eligible(self):
        value = torch.tensor([[[1., 0.], [-1., 0.], [0., 1.], [0., 1.]]])
        result = top_adjacent_cosines(value, 2)
        self.assertEqual(result['pair_starts'].tolist(), [[2, 1]])
        self.assertEqual(result['scores'].tolist(), [[1., 0.]])
        self.assertEqual(result['pair_count'], 3)

    def test_all_pairs_as_top_count_matches_full_adjacent_mean(self):
        value = torch.randn(2, 13, 6, generator=torch.Generator().manual_seed(99))
        result = top_adjacent_cosines(value, 12)
        torch.testing.assert_close(result['mean'], adjacent_cosine_statistics(value)['mean'], rtol=0, atol=1e-12)

    def test_bfloat16_retains_signed_scores_with_fp32_normalization(self):
        value = torch.tensor([[[1., 2.], [2., -1.], [-1., -2.], [-2., 1.], [1., 2.]],
                              [[1., 0.], [-1., 0.], [1., 0.], [-1., 0.], [1., 0.]]], dtype=torch.bfloat16)
        result = top_adjacent_cosines(value, 3)
        expected_scores, expected_starts = explicit_top_oracle(value, 3)
        self.assertTrue(torch.equal(result['pair_starts'], expected_starts))
        torch.testing.assert_close(result['scores'], expected_scores, rtol=0, atol=1e-7)
        self.assertEqual(result['mean'][1].item(), -1.)

    def test_short_input_is_rejected_without_replication_or_min_fallback(self):
        for length in (0, 1, 2, 200):
            with self.subTest(length=length), self.assertRaises(ValueError):
                top_adjacent_cosines(torch.ones(1, length, 2))
        # Exactly 201 tokens are sufficient for exactly200 adjacent positions.
        result = top_adjacent_cosines(torch.ones(1, 201, 2))
        self.assertEqual(result['pair_starts'].tolist(), [list(range(200))])

    def test_invalid_count_rank_type_and_vector_values_are_rejected(self):
        valid = torch.ones(2, 4, 3)
        for count in (0, -1, True, 1.5, '2', 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                top_adjacent_cosines(valid, count)
        for value in (torch.ones(2, 4), torch.ones(1, 2, 4, 3), torch.ones(0, 4, 3),
                      torch.ones(2, 4, 0), torch.ones(2, 4, 3, dtype=torch.long)):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                top_adjacent_cosines(value, 2)
        for invalid in (0., float('nan'), float('inf')):
            value = valid.clone(); value[0, -1] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                top_adjacent_cosines(value, 2)


class TopAdjacentObserverTests(unittest.TestCase):
    def setUp(self):
        self.value = torch.tensor([[[1., 0., 1., 0.], [-1., 0., 0., 1.],
                                    [0., 1., 1., 1.], [1., 1., -1., 1.]]])

    def build(self, family='qwen2', count=2):
        rotary = RotaryStub()
        model = NativeFakeModel(family, rotary)
        name = 'apply_rotary_pos_emb' if family == 'qwen2' else 'apply_multimodal_rotary_pos_emb'
        original = getattr(rotary, name)
        observer = RawKTopAdjacentObserver(model, family=family, input_length=4, rotary_module=rotary, top_k=count)
        return rotary, model, name, original, observer

    def assert_clean(self, rotary, model, name, original):
        self.assertIs(getattr(rotary, name), original)
        for layer in model.layers:
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)
            self.assertEqual(len(layer.self_attn.k_proj._forward_hooks), 0)
            self.assertEqual(len(layer.self_attn.v_proj._forward_hooks), 0)

    def test_native_paths_use_raw_projection_before_rope_gqa_and_keep_outputs(self):
        for family in ('qwen2', 'qwen2_5_vl'):
            rotary, model, name, original, observer = self.build(family)
            expected_output = model(self.value)
            projected = model.layers[0].self_attn.k_proj(self.value)[0].reshape(4, 2, 2).transpose(0, 1)
            expected_scores, expected_starts = explicit_top_oracle(projected, 2)
            rotary.results.clear(); rotary.calls = 0
            with observer:
                output = model(self.value)
                stats = observer.finish()
            with self.subTest(family=family):
                self.assertTrue(torch.equal(output, expected_output))
                self.assertEqual(len(stats), 2)
                self.assertEqual(stats[0]['kv_heads'], 2)
                self.assertEqual(model.layers[0].self_attn.repeated_heads, 4)
                self.assertEqual(stats[0]['top_k'], 2)
                self.assertEqual(stats[0]['adjacent_pair_count'], 3)
                self.assertEqual(stats[0]['capture_counts'], {'k_raw': 1, 'k_rope': 1, 'v': 1})
                self.assertEqual(stats[0]['raw_k_topk_pair_starts'], expected_starts.tolist())
                torch.testing.assert_close(torch.tensor(stats[0]['raw_k_topk_scores'], dtype=torch.float64), expected_scores, rtol=0, atol=1e-7)
                torch.testing.assert_close(torch.tensor(stats[0]['raw_k_topk_mean'], dtype=torch.float64), expected_scores.mean(-1), rtol=0, atol=1e-7)
                for key in ('k_raw', 'k_rope', 'v'):
                    self.assertEqual(len(stats[0][key]), 2)
                self.assertNotEqual(stats[0]['k_raw'], stats[0]['k_rope'])
                self.assert_clean(rotary, model, name, original)

    def test_bfloat16_forward_and_top_arrays_remain_original_head_count(self):
        rotary, model, name, original, observer = self.build('qwen2_5_vl')
        model.to(dtype=torch.bfloat16)
        value = self.value.to(dtype=torch.bfloat16)
        expected = model(value)
        with observer:
            actual = model(value)
            stats = observer.finish()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(torch.tensor(stats[0]['raw_k_topk_pair_starts']).shape), (2, 2))
        self.assert_clean(rotary, model, name, original)

    def test_invalid_top_count_or_short_sequence_fail_before_hooks(self):
        for count in (0, True, 1.5, 4, 200):
            rotary = RotaryStub(); model = NativeFakeModel('qwen2', rotary)
            original = rotary.apply_rotary_pos_emb
            with self.subTest(count=count), self.assertRaises(ValueError):
                RawKTopAdjacentObserver(model, family='qwen2', input_length=4, rotary_module=rotary, top_k=count)
            self.assert_clean(rotary, model, 'apply_rotary_pos_emb', original)

    def test_incomplete_or_model_failure_restores_hooks(self):
        for failure in ('missing_rope', 'missing_layer', 'repeated_projection', 'exception'):
            rotary, model, name, original, observer = self.build()
            if failure == 'missing_rope': model.layers[-1].self_attn.omit_rope = True
            elif failure == 'missing_layer': model.skip_last = True
            elif failure == 'repeated_projection': model.layers[0].self_attn.repeat_projection = True
            else: model.layers[0].self_attn.fail_after_rope = True
            with self.subTest(failure=failure):
                if failure in ('missing_rope', 'missing_layer'):
                    with observer:
                        model(self.value)
                        with self.assertRaises(ValueError): observer.finish()
                else:
                    with self.assertRaises((ValueError, RuntimeError)):
                        with observer: model(self.value)
                self.assert_clean(rotary, model, name, original)

    def test_finish_returns_copy_and_rejects_tampered_selection(self):
        rotary, model, name, original, observer = self.build()
        with observer:
            model(self.value)
            first = observer.finish()
            first[0]['raw_k_topk_scores'][0].clear()
            self.assertEqual(len(observer.finish()[0]['raw_k_topk_scores'][0]), 2)
            baseline = copy.deepcopy(observer._stats)
            corruptions = [lambda r: r['raw_k_topk_pair_starts'][0].__setitem__(1, r['raw_k_topk_pair_starts'][0][0]),
                           lambda r: r['raw_k_topk_scores'][0].__setitem__(0, float('nan')),
                           lambda r: r['raw_k_topk_mean'].__setitem__(0, 0.),
                           lambda r: r.__setitem__('top_k', 1),
                           lambda r: r.__setitem__('raw_k_topk_min', [])]
            for mutate in corruptions:
                observer._stats = copy.deepcopy(baseline)
                mutate(observer._stats[0])
                with self.assertRaises(ValueError): observer.finish()
            observer._stats = baseline
            observer._topk_capture_counts[0] = 0
            with self.assertRaises(ValueError): observer.finish()
        self.assert_clean(rotary, model, name, original)


if __name__ == '__main__':
    unittest.main()

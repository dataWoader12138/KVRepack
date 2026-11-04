"""CPU checks for matched sampling and context-control alignment."""
import copy
import unittest

from kvrepack.observations.mixkv_style import compare_reference, make_pools, matched_token_draws, natural_user_positions


class MixKVStyleInputsTests(unittest.TestCase):
    def test_matched_draws_equal_tokens_and_map_same_text_to_control(self):
        visual, primary, control = list(range(20, 300)), [2, 3, 4, 5, 6], [12, 13, 14, 15, 16]
        multi, text = matched_token_draws(visual, primary, control, seed=99, cap=4, draws=32)
        self.assertEqual((multi, text), matched_token_draws(visual, primary, control, seed=99, cap=4, draws=32))
        self.assertNotEqual(multi, matched_token_draws(visual, primary, control, seed=100, cap=4, draws=32)[0])
        for vp, tp, cp in zip(multi['visual_matched'], multi['text_matched'], text['text_matched']):
            self.assertEqual(len(vp), 4)
            self.assertEqual(len(tp), 4)
            self.assertEqual(vp, sorted(set(vp)))
            self.assertEqual(tp, sorted(set(tp)))
            self.assertTrue(set(vp) <= set(visual))
            self.assertTrue(set(tp) <= set(primary))
            self.assertEqual(cp, [p+10 for p in tp])

    def test_short_text_is_retained_without_duplication_within_draw(self):
        multi, _ = matched_token_draws(list(range(40)), [100, 101, 102], [0, 1, 2], seed=1)
        self.assertTrue(all(x == [100, 101, 102] for x in multi['text_matched']))
        self.assertTrue(all(len(x) == 3 for x in multi['visual_matched']))

    def test_invalid_matching_is_rejected(self):
        for kwargs in ({'cap': 0}, {'draws': 0}, {'cap': 1}, {'draws': True}):
            with self.assertRaises(ValueError):
                matched_token_draws([1, 2], [3, 4], [5, 6], seed=1, **kwargs)
        for visual, text, control in (([1, 1], [2, 3], [4, 5]), ([1, 2], [3], [4]),
                                      ([1, 2], [3, 4], [5]), ([1, 2], [4, 3], [5, 6])):
            with self.assertRaises(ValueError):
                matched_token_draws(visual, text, control, seed=1)

    def test_pools_keep_shell_separate_and_reject_overlap_padding_or_missing_text(self):
        ids, mask = [99, 88, 88, 10, 11, 12], [1, 1, 1, 1, 1, 0]
        groups = [{'name': 'question', 'positions': [3, 4]}]
        pools = make_pools(ids, mask, groups, [3, 4], 'question', 88)
        self.assertEqual(pools, {'prompt_all': [0, 1, 2, 3, 4], 'visual_all': [1, 2],
                                 'text_primary': [3, 4], 'natural_user': [3, 4]})
        for positions in ([3, 5], [1, 3, 4], [3], [4, 3]):
            with self.assertRaises(ValueError):
                make_pools(ids, mask, groups, positions, 'question', 88)

    def test_user_span_does_not_include_shell_and_special_tokens(self):
        positions = natural_user_positions([99, 2, 3, 4, 5], [1]*5,
                                            [[0, 2], [2, 3], [3, 4], [4, 5], [5, 7]],
                                            'xxabczz', 'abc', {3})
        self.assertEqual(positions, [1, 3])

    def test_reference_requires_exact_input_bytes_and_reports_numerical_difference(self):
        meta = {'original_row_sha256': 'a', 'expanded_text_sha256': 'b',
                'frame_sampling': {'indices': [0, 0, 1]}, 'video_grid_thw': [2, 4, 4]}
        stats = [{'layer': 0, 'kv_heads': 2, 'groups': {'text_question': {
            'k_raw': [.3, -.2], 'v': [-.1, .9], 'k_rope': [.2, .1]}}}]
        ref = {**meta, 'input_hashes': {'input_ids': 'c'}, 'layer_stats': copy.deepcopy(stats)}
        result = compare_reference(meta, {'input_ids': 'c'}, stats, ref)
        self.assertTrue(result['input_tensor_bytes_identical'])
        self.assertEqual(result['adjacent_max_absolute_difference'], 0)
        with self.assertRaisesRegex(ValueError, 'tensor bytes'):
            compare_reference(meta, {'input_ids': 'd'}, stats, ref)
        changed = copy.deepcopy(stats)
        changed[0]['groups']['text_question']['v'][0] += .01
        with self.assertRaisesRegex(ValueError, 'beyond tolerance'):
            compare_reference(meta, {'input_ids': 'c'}, changed, ref)


if __name__ == '__main__':
    unittest.main()

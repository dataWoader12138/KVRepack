"""CPU validation of fixed-count pre-RoPE adjacent upper-tail reports."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.observations.top_adjacent_report import (
    CAPTURES, GROUPS, METRICS, PRIMARY_METRIC, STATISTICS_KIND, TOP_K,
    build_summary, expected_ids, main, verify_selected_pairs, write_report,
)


def record(group, doc_id, *, top=.9, full=.4, minimum=.85, length=1000):
    spec = GROUPS[group]
    return dict(schema_version=1, status='completed', statistics_kind=STATISTICS_KIND,
                group=group, doc_id=doc_id, task=spec['task'], model_id=spec['model_id'], model_revision=spec['revision'],
                input_tokens=length, adjacent_pair_count=length-1, top_k=TOP_K,
                source_metadata={spec['cluster_key']: f'source-{doc_id}'}, input_hashes={'input_ids': '0'*64},
                selected_pairs_path=f'selected_pairs/{doc_id:06d}.npz', selected_pairs_sha256='0'*64,
                layer_stats=[dict(layer=layer, kv_heads=4, head_dim=128, sequence_length=length,
                                  adjacent_pair_count=length-1, top_k=TOP_K,
                                  capture_counts=dict.fromkeys(CAPTURES, 1),
                                  k_raw=[full]*4, k_rope=[full-.01]*4, v=[full-.1]*4,
                                  raw_k_topk_mean=[top]*4, raw_k_topk_min=[minimum]*4)
                             for layer in range(28)])


def records(sample_count=2, text=.9, video=.95):
    return [record(group, doc_id, top=text if group == 'text' else video,
                   minimum=min(.85, (text if group == 'text' else video)-.05))
            for group, ids in expected_ids(sample_count).items() for doc_id in ids]


def save_selection(root, row, *, scores=None, starts=None):
    import numpy as np

    if scores is None:
        scores = np.full((28, 4, TOP_K), row['layer_stats'][0]['raw_k_topk_mean'][0], dtype=np.float64)
    if starts is None:
        starts = np.broadcast_to(np.arange(TOP_K, dtype=np.int32), (28, 4, TOP_K)).copy()
    path = Path(root)/row['selected_pairs_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pair_starts=starts, scores=scores,
                        left_token_ids=np.zeros((28, 4, TOP_K), dtype=np.int32))
    row['selected_pairs_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for layer in range(28):
        row['layer_stats'][layer]['raw_k_topk_mean'] = scores[layer].mean(axis=-1).tolist()
        row['layer_stats'][layer]['raw_k_topk_min'] = scores[layer, :, -1].tolist()
    return path


class TopAdjacentReportTests(unittest.TestCase):
    def test_exact_counts_tasks_and_primary_scope(self):
        self.assertEqual(expected_ids(2), {'text': (0, 1), 'video': (0, 1)})
        self.assertEqual(expected_ids(200)['video'], tuple(range(200)))
        for count in (False, 1, 3, 201, 2.):
            with self.subTest(count=count), self.assertRaises(ValueError):
                expected_ids(count)
        summary = build_summary(records(), sample_count=2)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['primary_metric'], PRIMARY_METRIC)
        self.assertFalse(summary['bootstrap'])
        for group in GROUPS:
            self.assertEqual(summary['groups'][group]['n_inputs'], 2)
            self.assertEqual(summary['groups'][group]['selected_pairs_total_per_head_layer'], 400)
        self.assertIn('neither post-RoPE K nor V', summary['supplemental_metrics'])

    def test_equal_input_not_candidate_count_weighting(self):
        rows = records()
        rows[0] = record('text', 0, top=.6, full=.2, minimum=.5, length=501)
        rows[1] = record('text', 1, top=1., full=.8, minimum=.9, length=22064)
        summary = build_summary(rows, sample_count=2)
        self.assertAlmostEqual(summary['comparisons'][PRIMARY_METRIC]['text'], .8)
        self.assertAlmostEqual(summary['comparisons']['k_raw_all_adjacent']['text'], .5)
        self.assertEqual(summary['groups']['text']['candidate_adjacent_pairs_total'], 500+22063)
        for row in summary['head_table']:
            if row['group'] == 'text' and row['metric'] == PRIMARY_METRIC:
                self.assertAlmostEqual(row['mean'], .8)
                self.assertEqual(row['n_inputs'], 2)

    def test_native_layer_head_values_preserved(self):
        rows = records(text=.8)
        for r in rows[:2]:
            r['layer_stats'][0]['raw_k_topk_mean'] = [.5, .6, .7, 1.]
            r['layer_stats'][0]['raw_k_topk_min'] = [.5]*4
        summary = build_summary(rows, sample_count=2)
        expected = ((.5+.6+.7+1.)/4 + 27*.8)/28
        self.assertAlmostEqual(summary['comparisons'][PRIMARY_METRIC]['text'], expected)
        for group in GROUPS:
            self.assertEqual(sum(r['group'] == group and r['metric'] == PRIMARY_METRIC for r in summary['head_table']), 112)

    def test_rejects_other_statistic_short_sequences_or_changed_count(self):
        changes = [lambda r: r.update(statistics_kind='all_sequence_adjacent_v1'),
                   lambda r: r.update(statistics_kind='all_pairs'),
                   lambda r: r.update(input_tokens=200, adjacent_pair_count=199),
                   lambda r: r.update(top_k=199), lambda r: r.update(top_k=True),
                   lambda r: r.update(adjacent_pair_count=998),
                   lambda r: r['layer_stats'][0].update(top_k=199),
                   lambda r: r['layer_stats'][0].update(adjacent_pair_count=16)]
        for mutate in changes:
            rows = records(); mutate(rows[0])
            with self.subTest(mutation=mutate):
                s = build_summary(rows, sample_count=2)
                self.assertEqual(s['status'], 'incomplete')
                self.assertEqual(s['groups']['text']['n_inputs'], 1)

    def test_upper_tail_mean_constraints_and_signed_values(self):
        rows = records()
        for r in rows:
            for layer in r['layer_stats']:
                layer.update(raw_k_topk_mean=[-.2]*4, raw_k_topk_min=[-.4]*4,
                             k_raw=[-.6]*4, k_rope=[-.7]*4, v=[-.8]*4)
        s = build_summary(rows, sample_count=2)
        self.assertEqual(s['status'], 'complete')
        self.assertAlmostEqual(s['comparisons'][PRIMARY_METRIC]['text'], -.2)
        for field in ('raw_k_topk_mean', 'raw_k_topk_min', 'k_raw'):
            rows = records()
            rows[0]['layer_stats'][0][field] = [.99 if field != 'raw_k_topk_mean' else .1]*4
            self.assertEqual(build_summary(rows, sample_count=2)['status'], 'incomplete')

    def test_invalid_record_never_partially_enters_means(self):
        mutations = [lambda r: r['layer_stats'].pop(), lambda r: r['layer_stats'][0].update(layer=1),
                     lambda r: r['layer_stats'][0].update(kv_heads=28), lambda r: r['layer_stats'][0].update(head_dim=128.),
                     lambda r: r['layer_stats'][0]['capture_counts'].update(k_raw=True),
                     lambda r: r['layer_stats'][0].update(raw_k_topk_mean=[float('nan')]*4),
                     lambda r: r['layer_stats'][0].update(raw_k_topk_min=[1.1]*4),
                     lambda r: r.update(task='OCRBench'), lambda r: r.update(model_revision='other'),
                     lambda r: r.update(input_hashes={}), lambda r: r.update(source_metadata={}),
                     lambda r: r.update(selected_pairs_path='../escape.npz'), lambda r: r.update(selected_pairs_sha256='bad')]
        for mutate in mutations:
            rows = records(); mutate(rows[0])
            with self.subTest(mutation=mutate):
                s = build_summary(rows, sample_count=2)
                self.assertEqual(s['status'], 'incomplete')
                self.assertEqual({r['n_inputs'] for r in s['head_table'] if r['group'] == 'text'}, {1})

    def test_formal200_requires_complete_original_prefix(self):
        summary = build_summary(records(sample_count=200), sample_count=200)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['expected_total_inputs'], 400)
        self.assertEqual(summary['groups']['video']['selected_pairs_total_per_head_layer'], 40000)
        for kind in ('missing', 'duplicate', 'wrong_id'):
            rows = records()
            if kind == 'missing': rows.pop()
            elif kind == 'duplicate': rows.append(deepcopy(rows[0]))
            else: rows[-1]['doc_id'] = 200
            self.assertEqual(build_summary(rows, sample_count=2)['status'], 'incomplete')

    def test_saved_selection_exact_shape_order_scores_and_sha(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); row = record('text', 0)
            scores = np.broadcast_to(np.linspace(.99, .85, TOP_K), (28, 4, TOP_K)).copy()
            path = save_selection(root, row, scores=scores)
            report = verify_selected_pairs(row, root/'records.jsonl')
            self.assertEqual(report['shape'], [28, 4, 200])
            self.assertTrue(report['scores_verified'])
            bad = deepcopy(row); bad['selected_pairs_sha256'] = '0'*64
            with self.assertRaises(ValueError): verify_selected_pairs(bad, root/'records.jsonl')
            bad = deepcopy(row); bad['layer_stats'][0]['raw_k_topk_mean'][0] -= .1
            with self.assertRaises(ValueError): verify_selected_pairs(bad, root/'records.jsonl')
            for mode in ('shape', 'dtype', 'position', 'duplicate', 'sorting', 'tie', 'nan'):
                row = record('text', 0)
                starts = np.broadcast_to(np.arange(TOP_K, dtype=np.int32), (28, 4, TOP_K)).copy()
                scores = np.full((28, 4, TOP_K), .9, dtype=np.float64)
                if mode == 'shape': starts = starts[:, :, :199]
                elif mode == 'dtype': starts = starts.astype(np.int64)
                elif mode == 'position': starts[0, 0, -1] = 999
                elif mode == 'duplicate': starts[0, 0, -1] = 0
                elif mode == 'sorting': scores[0, 0, -1] = .99
                elif mode == 'tie': starts[0, 0, :2] = [1, 0]
                else: scores[0, 0, 0] = np.nan
                save_selection(root, row, starts=starts, scores=scores)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    verify_selected_pairs(row, root/'records.jsonl')

    def test_cli_verifies_npz_and_never_accepts_wrong_group_or_corrupt_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = records()
            inputs = {}
            for group in GROUPS:
                group_dir = root/group; group_dir.mkdir()
                selected = [r for r in rows if r['group'] == group]
                for row in selected: save_selection(group_dir, row)
                inputs[group] = group_dir/'records.jsonl'
                inputs[group].write_text(''.join(json.dumps(r)+'\n' for r in selected))
            args = ['--text-input', str(inputs['text']), '--video-input', str(inputs['video']),
                    '--sample-count', '2', '--output', str(root/'report'), '--no-plots']
            self.assertEqual(main(args), 0)
            result = json.loads((root/'report/summary.json').read_text())
            self.assertEqual(len(result['provenance']['selected_artifacts']), 4)
            with (root/'report/record_summary.csv').open() as f: saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), 16)
            self.assertEqual({int(r['selected_pair_count']) for r in saved}, {200})
            inputs['video'].write_text(json.dumps(rows[0])+'\n{bad\n')
            self.assertEqual(main(args), 2)
            self.assertEqual(json.loads((root/'report/summary.json').read_text())['artifacts']['plot'], 'disabled')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'optional matplotlib unavailable')
    def test_continuous_plot_uses112_top200_head_means_and_clear_legend(self):
        summary = build_summary(records(), sample_count=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root/'adjacent_combined.png'; sentinel.write_bytes(b'old-plot')
            compact = write_report(summary, root)
            for extension in ('png', 'pdf', 'svg'):
                self.assertGreater((root/f'raw_k_top200.{extension}').stat().st_size, 20)
            self.assertEqual(sentinel.read_bytes(), b'old-plot')
            metadata = compact['artifacts']['plot']
            self.assertEqual(metadata['metric'], PRIMARY_METRIC)
            self.assertEqual({r['n_head_means'] for r in metadata['curves']}, {112})
            self.assertEqual({r['color'] for r in metadata['curves']}, {'#B98A2F', '#2F4373'})
            self.assertGreater(metadata['legend']['bounds_in_axes_coordinates'][1], 1)
            for curve in metadata['curves']: self.assertAlmostEqual(curve['density_area'], 1)
            svg = (root/'raw_k_top200.svg').read_text()
            for text in ('Before RoPE', 'Cosine Similarity', 'VATEX', 'LongBench'):
                self.assertIn(text, svg)
            self.assertNotIn(b'\r\n', (root/'comparison_summary.csv').read_bytes())


if __name__ == '__main__':
    unittest.main()

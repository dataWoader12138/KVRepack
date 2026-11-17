"""CPU tests for complete-text adjacent-key subset reports."""
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.observations.longbench_controls_report import (
    CAPTURES, METRICS, MODEL_ID, MODEL_REVISION, PRIMARY_METRIC, STATISTICS_KIND,
    _plot, build_summary, expected_ids, main, write_report,
)


SUBSETS = ('hotpotqa', 'triviaqa', 'lcc')


def record(subset, doc_id, *, value=.8, length=1000):
    return dict(schema_version=1, status='completed', statistics_kind=STATISTICS_KIND,
                contract_sha256='0'*64, subset=subset, doc_id=doc_id, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                input_tokens=length, image_tokens=0, adjacent_pair_count=length-1, chat_sha256='1'*64,
                source_metadata=dict(subset=subset, row_index=doc_id, context_sha256=f'{doc_id:064x}',
                                     original_row_bytes_sha256='2'*64),
                input_hashes={'input_ids': '3'*64, 'attention_mask': '4'*64},
                layer_stats=[dict(layer=layer, kv_heads=4, head_dim=128, sequence_length=length,
                                  adjacent_pair_count=length-1, capture_counts=dict.fromkeys(CAPTURES, 1),
                                  key_diagnostic_capture_count=1, k_proj_bias_present=True,
                                  k_proj_bias_norm=[1.]*4, k_raw_sequence_mean_norm=[2.]*4,
                                  **{metric: [value]*4 for metric in METRICS}) for layer in range(28)])


def records(count=2):
    return [record(subset, doc_id, value=.7+index*.05)
            for index, subset in enumerate(SUBSETS) for doc_id in range(count)]


class LongBenchControlsReportTests(unittest.TestCase):
    def test_complete_native_scope_and_six_metrics(self):
        summary = build_summary(records(), sample_count=2, subsets=SUBSETS)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['expected_total_inputs'], 6)
        self.assertEqual(summary['primary_metric'], 'k_raw')
        self.assertFalse(summary['bootstrap'])
        self.assertFalse(summary['token_or_pair_sampling'])
        self.assertIn('post-hoc', summary['interpretation'])
        self.assertIn('not a forward pass with the bias disabled', summary['interpretation'])
        for subset in SUBSETS:
            self.assertEqual(summary['groups'][subset]['n_inputs'], 2)
            self.assertEqual(set(summary['groups'][subset]['metrics']), set(METRICS))
            self.assertEqual(summary['groups'][subset]['adjacent_pairs_total_per_head_layer'], 1998)
        all_pairs = next(r for r in summary['record_table'] if r['metric'] == 'k_raw_all_pairs')
        self.assertEqual(all_pairs['observed_pair_count'], 1000*999)
        self.assertEqual(all_pairs['pair_kind'], 'all_distinct_token_pairs')
        adjacent = next(r for r in summary['record_table'] if r['metric'] == 'k_raw')
        self.assertEqual(adjacent['observed_pair_count'], 999)

    def test_prescribed_prefix_and_subset_validation(self):
        self.assertEqual(expected_ids(2, SUBSETS)['lcc'], (0, 1))
        self.assertEqual(expected_ids(200, SUBSETS)['hotpotqa'], tuple(range(200)))
        for count in (True, 1, 3, 201, 2.):
            with self.subTest(count=count), self.assertRaises(ValueError): expected_ids(count, SUBSETS)
        for subsets in ([], ('hotpotqa', 'hotpotqa'), ('qasper',), 'hotpotqa'):
            with self.subTest(subsets=subsets), self.assertRaises(ValueError): expected_ids(2, subsets)

    def test_equal_inputs_not_length_or_pair_count_weights(self):
        rows = records()
        rows[0] = record('hotpotqa', 0, value=.2, length=2)
        rows[1] = record('hotpotqa', 1, value=.8, length=22064)
        summary = build_summary(rows, sample_count=2, subsets=SUBSETS)
        self.assertEqual(summary['status'], 'complete')
        for metric in METRICS:
            self.assertAlmostEqual(summary['groups']['hotpotqa']['metrics'][metric]['mean'], .5)
        self.assertEqual(summary['groups']['hotpotqa']['adjacent_pairs_total_per_head_layer'], 22064)
        for row in summary['head_table']:
            if row['subset'] == 'hotpotqa': self.assertAlmostEqual(row['mean'], .5)

    def test_all_native_head_and_layer_values_remain_distinct(self):
        rows = records()
        for row in rows[:2]:
            for metric in METRICS: row['layer_stats'][0][metric] = [-1., -.5, .5, 1.]
        summary = build_summary(rows, sample_count=2, subsets=SUBSETS)
        self.assertEqual(summary['status'], 'complete')
        self.assertAlmostEqual(summary['groups']['hotpotqa']['metrics']['k_raw']['mean'], 27*.7/28)
        selected = [r for r in summary['head_table'] if r['subset'] == 'hotpotqa' and r['metric'] == 'k_raw']
        self.assertEqual(len(selected), 112)
        self.assertEqual([r['mean'] for r in selected if r['layer'] == 0], [-1., -.5, .5, 1.])
        self.assertTrue(all(r['n_inputs'] == 2 for r in selected))

    def test_rejects_old_kind_and_incomplete_adjacent_counts(self):
        changes = [lambda r: r.update(statistics_kind='all_sequence_adjacent_v1'),
                   lambda r: r.update(statistics_kind='raw_k_top200_sequence_adjacent_v1'),
                   lambda r: r.update(adjacent_pair_count=998),
                   lambda r: r.update(input_tokens=1, adjacent_pair_count=0),
                   lambda r: r['layer_stats'][0].update(adjacent_pair_count=16),
                   lambda r: r['layer_stats'][0].update(sequence_length=999)]
        for mutate in changes:
            rows = records(); mutate(rows[0])
            with self.subTest(mutate=mutate):
                s = build_summary(rows, sample_count=2, subsets=SUBSETS)
                self.assertEqual(s['status'], 'incomplete')
                self.assertEqual(s['groups']['hotpotqa']['n_inputs'], 1)

    def test_rejects_invalid_records_without_partial_aggregate(self):
        changes = [lambda r: r['layer_stats'].pop(),
                   lambda r: r['layer_stats'][0].update(layer=1),
                   lambda r: r['layer_stats'][0].update(kv_heads=28),
                   lambda r: r['layer_stats'][0].update(head_dim=128.),
                   lambda r: r['layer_stats'][0]['capture_counts'].update(k_raw=True),
                   lambda r: r['layer_stats'][0].update(key_diagnostic_capture_count=0),
                   lambda r: r['layer_stats'][0].update(k_raw_centered_adjacent=[float('nan')]*4),
                   lambda r: r['layer_stats'][0].update(k_raw_without_bias_adjacent=[1.001]*4),
                   lambda r: r['layer_stats'][0].update(k_proj_bias_norm=[-1.]*4),
                   lambda r: r['layer_stats'][0].update(k_proj_bias_present=1),
                   lambda r: r.update(model_revision='different'),
                   lambda r: r.update(model_id='Qwen/Qwen2.5-VL-7B-Instruct'),
                   lambda r: r.update(image_tokens=1), lambda r: r.update(input_hashes={}),
                   lambda r: r['source_metadata'].update(row_index=1),
                   lambda r: r['source_metadata'].update(context_sha256='bad')]
        for mutate in changes:
            rows = records(); mutate(rows[0])
            with self.subTest(mutate=mutate):
                s = build_summary(rows, sample_count=2, subsets=SUBSETS)
                self.assertEqual(s['status'], 'incomplete')
                self.assertEqual(s['groups']['hotpotqa']['n_inputs'], 1)
                self.assertEqual({r['n_inputs'] for r in s['head_table'] if r['subset'] == 'hotpotqa'}, {1})

    def test_negative_geometry_and_no_bias_diagnostic_are_preserved(self):
        rows = records()
        for row in rows:
            for layer in row['layer_stats']:
                layer.update(k_raw_centered_adjacent=[-.4]*4, k_raw_without_bias_adjacent=[-.2]*4,
                             k_proj_bias_present=False, k_proj_bias_norm=[0.]*4)
        summary = build_summary(rows, sample_count=2, subsets=SUBSETS)
        self.assertEqual(summary['status'], 'complete')
        self.assertAlmostEqual(summary['groups']['lcc']['metrics']['k_raw_centered_adjacent']['mean'], -.4)
        self.assertTrue(all(not r['k_proj_bias_present'] and r['k_proj_bias_norm'] == 0 for r in summary['head_table']))

    def test_complete200_requires_all_original_rows_and_rejects_duplicates(self):
        self.assertEqual(build_summary(records(200), sample_count=200, subsets=SUBSETS)['status'], 'complete')
        for kind in ('missing', 'duplicate', 'wrong_id'):
            rows = records()
            if kind == 'missing': rows.pop()
            elif kind == 'duplicate': rows.append(deepcopy(rows[0]))
            else: rows[-1]['doc_id'] = 2
            with self.subTest(kind=kind): self.assertEqual(build_summary(rows, sample_count=2, subsets=SUBSETS)['status'], 'incomplete')

    def test_cli_csvs_and_malformed_input_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'records.jsonl'; rows = records()
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            args = ['--input', str(source), '--output', str(root/'report'), '--sample-count', '2', '--subsets', *SUBSETS, '--no-plots']
            self.assertEqual(main(args), 0)
            summary = json.loads((root/'report/summary.json').read_text())
            self.assertEqual(summary['status'], 'complete')
            with (root/'report/record_summary.csv').open() as f: saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), 6*len(METRICS))
            self.assertNotIn(b'\r\n', (root/'report/group_summary.csv').read_bytes())
            (root/'contract.json').write_text('{"different": true}\n')
            self.assertEqual(main(args), 2)
            self.assertTrue(any('contract' in r['error'] for r in json.loads((root/'report/summary.json').read_text())['validation_issues']))
            (root/'contract.json').unlink()
            source.write_text(source.read_text()+'{bad\n')
            self.assertEqual(main(args), 2)
            self.assertEqual(json.loads((root/'report/summary.json').read_text())['artifacts']['plot'], 'disabled')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib unavailable')
    def test_plot_uses112_raw_key_head_means_per_subset_and_clear_legend(self):
        summary = build_summary(records(), sample_count=2, subsets=SUBSETS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compact = write_report(summary, root)
            for extension in ('png', 'pdf', 'svg'): self.assertGreater((root/f'raw_k_adjacent_controls.{extension}').stat().st_size, 50)
            plot = compact['artifacts']['plot']
            self.assertEqual(plot['metric'], PRIMARY_METRIC)
            self.assertEqual({row['n_head_means'] for row in plot['curves']}, {112})
            self.assertEqual({row['subset'] for row in plot['curves']}, set(SUBSETS))
            self.assertGreater(plot['legend']['bounds_in_axes_coordinates'][1], 1)
            for row in plot['curves']: self.assertAlmostEqual(row['density_area'], 1)
            svg = (root/'raw_k_adjacent_controls.svg').read_text()
            for label in ('HotpotQA', 'TriviaQA', 'LCC', 'Cosine Similarity', 'Before RoPE'): self.assertIn(label, svg)
            bad = deepcopy(summary); bad['head_table'].pop(0)
            with self.assertRaises(ValueError): _plot(bad, root)


if __name__ == '__main__':
    unittest.main()

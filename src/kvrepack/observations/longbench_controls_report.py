"""Report complete adjacent-K comparisons across fixed LongBench text subsets.

The raw-key adjacency statistic remains the primary observation. Centering and
projection-bias subtraction are post-hoc probes on captured vectors, not model
modifications or generation/merge-quality experiments.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .report import read_jsonl
from .reference_report import HEAD_DIM, HEADS, LAYERS, _density_curve, _integer, _vector


STATISTICS_KIND = 'longbench_full_adjacent_key_controls_v1'
MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'
MODEL_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
PRIMARY_METRIC = 'k_raw'
CAPTURES = ('k_raw', 'k_rope', 'v')
METRICS = (*CAPTURES, 'k_raw_all_pairs', 'k_raw_centered_adjacent', 'k_raw_without_bias_adjacent')
SUBSETS = {
    'hotpotqa': dict(label='HotpotQA', color='#B98A2F'),
    'triviaqa': dict(label='TriviaQA', color='#805823'),
    'lcc': dict(label='LCC', color='#D0A353'),
    '2wikimqa': dict(label='2WikiMQA', color='#A66A25'),
    'musique': dict(label='MuSiQue', color='#604125'),
    'qmsum': dict(label='QMSum', color='#C19757'),
}


def expected_ids(sample_count, subsets):
    if type(sample_count) is not int or sample_count not in (2, 200):
        raise ValueError('Use the explicit pilot2 or formal200 original inputs per subset')
    if (not isinstance(subsets, (list, tuple)) or not subsets
            or any(name not in SUBSETS for name in subsets) or len(set(subsets)) != len(subsets)):
        raise ValueError('Require unique supported LongBench subset names')
    return {name: tuple(range(sample_count)) for name in subsets}


def _digest(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(label+' must be a SHA256 string')
    return value


def _norms(value, label):
    if (not isinstance(value, list) or len(value) != HEADS
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in value)):
        raise ValueError(label+' requires four finite nonnegative native-head norms')
    return value


def _collapse(record):
    if record.get('statistics_kind') != STATISTICS_KIND:
        raise ValueError('Require complete LongBench adjacent-key controls; old all-pair/top200 records cannot be used')
    length = _integer(record.get('input_tokens'), 'input_tokens', 2)
    count = _integer(record.get('adjacent_pair_count'), 'adjacent_pair_count', 1)
    if count != length-1:
        raise ValueError('Complete immediate adjacency requires exactly N-1 pairs')
    if type(record.get('image_tokens')) is not int or record['image_tokens'] != 0:
        raise ValueError('These controls require complete pure-text inputs')
    _digest(record.get('chat_sha256'), 'chat_sha256')
    hashes = record.get('input_hashes')
    if not isinstance(hashes, dict) or set(hashes) != {'input_ids', 'attention_mask'}:
        raise ValueError('Require both original pure-text processor tensor fingerprints')
    for name, digest in hashes.items():
        _digest(digest, 'input_hashes.'+name)
    layers = record.get('layer_stats')
    if not isinstance(layers, list) or len(layers) != LAYERS:
        raise ValueError('Require exactly28 decoder layers')
    if sorted(_integer(layer.get('layer'), 'layer') for layer in layers) != list(range(LAYERS)):
        raise ValueError('Layer IDs must cover0..27 exactly once')
    values, details = defaultdict(list), []
    for layer in layers:
        if (type(layer.get('kv_heads')) is not int or layer['kv_heads'] != HEADS
                or type(layer.get('head_dim')) is not int or layer['head_dim'] != HEAD_DIM):
            raise ValueError('Require four native KV heads of dimension128, before GQA repetition')
        if (type(layer.get('sequence_length')) is not int or layer['sequence_length'] != length
                or type(layer.get('adjacent_pair_count')) is not int or layer['adjacent_pair_count'] != count):
            raise ValueError('Every layer must cover the same complete N-1 adjacent pairs')
        captures = layer.get('capture_counts', {})
        if set(captures) != set(CAPTURES) or any(type(captures[m]) is not int or captures[m] != 1 for m in CAPTURES):
            raise ValueError('Raw K, post-RoPE K and V must each be captured exactly once')
        if type(layer.get('key_diagnostic_capture_count')) is not int or layer['key_diagnostic_capture_count'] != 1:
            raise ValueError('Raw-K geometry diagnostics must be captured exactly once')
        if type(layer.get('k_proj_bias_present')) is not bool:
            raise ValueError('Projection-bias presence must be explicit')
        bias_norm = _norms(layer.get('k_proj_bias_norm'), 'k_proj_bias_norm')
        common_norm = _norms(layer.get('k_raw_sequence_mean_norm'), 'k_raw_sequence_mean_norm')
        for metric in METRICS:
            heads = _vector(layer.get(metric), metric)
            values[metric].append(statistics.fmean(heads))
            details.extend(dict(metric=metric, layer=layer['layer'], kv_head=head, mean=value,
                                k_proj_bias_present=layer['k_proj_bias_present'],
                                k_proj_bias_norm=bias_norm[head], k_raw_sequence_mean_norm=common_norm[head])
                           for head, value in enumerate(heads))
    return {metric: statistics.fmean(values[metric]) for metric in METRICS}, details


def build_summary(records, *, sample_count=200, subsets=('hotpotqa', 'triviaqa', 'lcc'), read_errors=None):
    prescribed = expected_ids(sample_count, subsets)
    issues, seen, valid = list(read_errors or []), defaultdict(set), defaultdict(set)
    record_rows, source_rows, head_values, context_ids = [], [], defaultdict(list), defaultdict(set)
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict) or type(record.get('schema_version')) is not int or record['schema_version'] != 1:
                raise ValueError('Require schema_version1 record objects')
            if record.get('status') != 'completed':
                raise ValueError('Require completed records')
            subset = record.get('subset')
            if subset not in prescribed:
                raise ValueError('Unexpected LongBench subset')
            doc_id = _integer(record.get('doc_id'), 'doc_id')
            if doc_id not in prescribed[subset] or doc_id in seen[subset]:
                raise ValueError('Out-of-range or duplicate original subset/doc_id')
            seen[subset].add(doc_id)
            if record.get('model_id') != MODEL_ID or record.get('model_revision') != MODEL_REVISION:
                raise ValueError('Require the fixed Qwen2.5-7B-Instruct model and revision')
            source = record.get('source_metadata')
            if (not isinstance(source, dict) or source.get('subset') != subset
                    or type(source.get('row_index')) is not int or source['row_index'] != doc_id):
                raise ValueError('Source metadata must preserve the original subset and row index')
            context = _digest(source.get('context_sha256'), 'context_sha256')
            _digest(source.get('original_row_bytes_sha256'), 'original_row_bytes_sha256')
            _digest(record.get('contract_sha256'), 'contract_sha256')
            means, details = _collapse(record)
            # Only a fully validated record may contribute any metric/head.
            common = dict(subset=subset, doc_id=doc_id, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                          input_tokens=record['input_tokens'], adjacent_pair_count=record['adjacent_pair_count'],
                          source_context_sha256=context)
            for metric in METRICS:
                pair_kind = 'all_distinct_token_pairs' if metric == 'k_raw_all_pairs' else 'immediate_sequence_adjacent'
                pair_count = (record['input_tokens']*(record['input_tokens']-1)
                              if metric == 'k_raw_all_pairs' else record['adjacent_pair_count'])
                record_rows.append(dict(**common, metric=metric, mean=means[metric],
                                        pair_kind=pair_kind, observed_pair_count=pair_count))
            for detail in details:
                head_values[subset, detail['metric'], detail['layer'], detail['kv_head']].append(detail)
            source_rows.append(dict(**common, chat_sha256=record['chat_sha256'],
                                    original_row_bytes_sha256=source['original_row_bytes_sha256'],
                                    input_hashes_json=json.dumps(record['input_hashes'], sort_keys=True)))
            context_ids[subset].add(context)
            valid[subset].add(doc_id)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            issues.append(dict(record_index=index, subset=record.get('subset') if isinstance(record, dict) else None,
                               doc_id=record.get('doc_id') if isinstance(record, dict) else None, error=str(error)))
    heads = []
    for (subset, metric, layer, head), values in sorted(head_values.items()):
        if len({r['k_proj_bias_present'] for r in values}) != 1:
            issues.append(dict(subset=subset, layer=layer, kv_head=head, error='Projection-bias presence changed across inputs'))
        heads.append(dict(subset=subset, metric=metric, layer=layer, kv_head=head, n_inputs=len(values),
                          mean=statistics.fmean(r['mean'] for r in values),
                          k_proj_bias_present=values[0]['k_proj_bias_present'],
                          k_proj_bias_norm=statistics.fmean(r['k_proj_bias_norm'] for r in values),
                          k_raw_sequence_mean_norm=statistics.fmean(r['k_raw_sequence_mean_norm'] for r in values)))
    groups = {}
    for subset in subsets:
        missing = sorted(set(prescribed[subset])-valid[subset])
        if missing:
            issues.append(dict(subset=subset, error='Missing valid completed original prefix rows', doc_ids=missing))
        selected = [r for r in source_rows if r['subset'] == subset]
        groups[subset] = dict(model_id=MODEL_ID, model_revision=MODEL_REVISION, n_inputs=len(valid[subset]),
                              n_source_contexts=len(context_ids[subset]), valid_doc_ids=sorted(valid[subset]), missing_doc_ids=missing,
                              input_tokens=dict(min=min(r['input_tokens'] for r in selected),
                                                max=max(r['input_tokens'] for r in selected),
                                                mean=statistics.fmean(r['input_tokens'] for r in selected)) if selected else None,
                              adjacent_pairs_total_per_head_layer=sum(r['adjacent_pair_count'] for r in selected),
                              metrics={metric: dict(mean=statistics.fmean(r['mean'] for r in record_rows
                                                                         if r['subset'] == subset and r['metric'] == metric))
                                       for metric in METRICS} if selected else {})
    return dict(schema_version=1, statistics_kind=STATISTICS_KIND, status='incomplete' if issues else 'complete',
                sample_count=sample_count, subsets=list(subsets), expected_total_inputs=sample_count*len(subsets),
                primary_metric=PRIMARY_METRIC, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                dimensions=dict(layers=LAYERS, native_kv_heads=HEADS, head_dim=HEAD_DIM), groups=groups,
                statistic='Primary: cosine over every original sequence-adjacent pair (i,i+1), exactly N-1 per complete input/head/layer, before current-layer RoPE',
                aggregation='Equal adjacent-pair weights within each input/head/layer, then equal native-head/layer and prescribed-input weights; no input-length weighting',
                distribution_unit='112 decoder-layer/native-KV-head means over the prescribed inputs per subset, not individual pair cosines or independent observations',
                diagnostics='Raw non-self all-pairs; sequence-mean-subtracted raw-K adjacency; projection-bias-subtracted raw-K adjacency. Transformations operate only on captured FP32 copies of native BF16 outputs and never change the model forward.',
                interpretation='Dataset controls use the same fixed model and prescribed original inputs. Centering is a post-hoc geometry probe; subtracting projection bias from rounded native outputs is not a forward pass with the bias disabled. These measurements do not prove merge safety or isolate a unique causal mechanism.',
                bootstrap=False, token_or_pair_sampling=False, validation_issues=issues,
                record_table=record_rows, head_table=heads, source_table=source_rows)


def _csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.linspace(-1, 1, 2049)
    data = {subset: [r['mean'] for r in summary['head_table'] if r['subset'] == subset and r['metric'] == PRIMARY_METRIC]
            for subset in summary['subsets']}
    if any(len(v) != LAYERS*HEADS for v in data.values()):
        raise ValueError('Require all112 native layer/head means per complete subset')
    left = max(-1., math.floor((min(min(v) for v in data.values())-.03)*100)/100)
    curves = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': '#30343B',
                         'axes.labelcolor': '#30343B', 'axes.edgecolor': '#C8CBD0'}):
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        fig.subplots_adjust(left=.1, right=.985, bottom=.15, top=.76)
        for index, (subset, values) in enumerate(data.items()):
            spec = SUBSETS[subset]
            density, bandwidth = _density_curve(values, grid)
            axis.fill_between(grid, 0, density, color=spec['color'], alpha=.09, linewidth=0, zorder=1)
            axis.plot(grid, density, color=spec['color'], linewidth=2,
                      linestyle=('-', '--', '-.', ':')[index % 4], label=spec['label'], zorder=3)
            curves.append(dict(subset=subset, metric=PRIMARY_METRIC, n_head_means=len(values),
                               observed_min=min(values), observed_max=max(values), bandwidth=bandwidth,
                               color=spec['color'], density_area=float(np.trapz(density, grid))))
        axis.set(xlabel='Cosine Similarity', ylabel='Density', xlim=(left, 1), ylim=(0, None))
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(color='#E2E4E7', linewidth=.65, linestyle=(0, (2, 2)))
        axis.set_axisbelow(True)
        legend = axis.legend(loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=min(3, len(data)), frameon=False, fontsize=8)
        title = fig.suptitle('Key Similarity Distribution Comparison (Before RoPE)', y=.98, fontsize=10)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = legend.get_window_extent(renderer).transformed(axis.transAxes.inverted()).bounds
        if bounds[1] <= 1 or title.get_window_extent(renderer).y0 <= legend.get_window_extent(renderer).y1:
            raise ValueError('Legend must lie above the data axes and below the title')
        for extension in ('png', 'pdf', 'svg'):
            path = Path(output)/f'raw_k_adjacent_controls.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    return dict(metric=PRIMARY_METRIC, representation='boundary-reflected Gaussian KDE',
                bandwidth_rule='Scott; minimum0.01', grid_points=len(grid), display_range=[left, 1],
                curves=curves, legend=dict(location='outside_above_data_axes', bounds_in_axes_coordinates=list(bounds)))


def write_report(summary, output, *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output/'record_summary.csv', summary['record_table'],
         ['subset', 'doc_id', 'model_id', 'model_revision', 'input_tokens', 'adjacent_pair_count', 'source_context_sha256',
          'metric', 'pair_kind', 'observed_pair_count', 'mean'])
    _csv(output/'head_summary.csv', summary['head_table'],
         ['subset', 'metric', 'layer', 'kv_head', 'n_inputs', 'mean', 'k_proj_bias_present', 'k_proj_bias_norm', 'k_raw_sequence_mean_norm'])
    _csv(output/'source_inputs.csv', summary['source_table'],
         ['subset', 'doc_id', 'input_tokens', 'adjacent_pair_count', 'source_context_sha256', 'chat_sha256',
          'original_row_bytes_sha256', 'input_hashes_json'])
    _csv(output/'group_summary.csv',
         [dict(subset=subset, model_id=MODEL_ID, n_inputs=data['n_inputs'], n_source_contexts=data['n_source_contexts'],
               metric=metric, mean=value['mean'], adjacent_pairs_total_per_head_layer=data['adjacent_pairs_total_per_head_layer'],
               **{'input_tokens_'+key: val for key, val in (data['input_tokens'] or {}).items()})
          for subset, data in summary['groups'].items() for metric, value in data['metrics'].items()],
         ['subset', 'model_id', 'n_inputs', 'n_source_contexts', 'metric', 'mean', 'input_tokens_min', 'input_tokens_max',
          'input_tokens_mean', 'adjacent_pairs_total_per_head_layer'])
    compact = {key: value for key, value in summary.items() if not key.endswith('_table')}
    compact['artifacts'] = dict(tables=['record_summary.csv', 'head_summary.csv', 'source_inputs.csv', 'group_summary.csv'], plot='disabled')
    if plots and summary['status'] == 'complete':
        compact['artifacts']['plot'] = _plot(summary, output)
    lines = ['# Complete adjacent-key controls on LongBench', '', f"Status: **{summary['status']}**.", '',
             summary['statistic'], '', summary['aggregation'], '',
             '| Subset | N | Raw K adjacency | Raw K all-pairs | Centered K adjacency | Bias-subtracted K adjacency |',
             '|---|---:|---:|---:|---:|---:|']
    for subset, data in summary['groups'].items():
        if data['metrics']:
            values = data['metrics']
            lines.append(f"| {subset} | {data['n_inputs']} | {values['k_raw']['mean']:.9f} | {values['k_raw_all_pairs']['mean']:.9f} | {values['k_raw_centered_adjacent']['mean']:.9f} | {values['k_raw_without_bias_adjacent']['mean']:.9f} |")
    lines += ['', summary['diagnostics'], '', summary['distribution_unit'], '', summary['interpretation'], '',
              'Post-RoPE K and V are supplemental all-adjacent values in the CSV files. No token/pair sampling, truncation, bootstrap or significance testing was performed.']
    if summary['validation_issues']:
        lines += ['', 'Incomplete/invalid coverage; aggregates are provisional:', '', '```json', json.dumps(summary['validation_issues'], indent=2), '```']
    (output/'summary.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, choices=(2, 200), default=200)
    parser.add_argument('--subsets', nargs='+', choices=tuple(SUBSETS), default=['hotpotqa', 'triviaqa', 'lcc'])
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    expected_ids(args.sample_count, args.subsets)
    records, errors = read_jsonl(args.input)
    contract_path = args.input.parent/'contract.json'
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest() if contract_path.is_file() else None
    if contract_sha is not None:
        for index, row in enumerate(records):
            if not isinstance(row, dict) or row.get('contract_sha256') != contract_sha:
                errors.append(dict(record_index=index, error='Record does not match the adjacent frozen contract'))
    summary = build_summary(records, sample_count=args.sample_count, subsets=args.subsets, read_errors=errors)
    summary['provenance'] = dict(input_file=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                                 contract_file=str(contract_path.resolve()) if contract_sha else None, contract_sha256=contract_sha)
    write_report(summary, args.output, plots=not args.no_plots)
    print(json.dumps(dict(status=summary['status'], output=str(args.output), validation_issues=len(summary['validation_issues']))))
    return 0 if summary['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())

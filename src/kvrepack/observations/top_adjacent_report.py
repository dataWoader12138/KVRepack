"""Top-200 pre-RoPE K adjacency: LongBench text versus VATEX video.

Each input, layer and native KV head contributes the mean of exactly its 200
highest raw-K cosines over the N-1 immediate sequence pairs. This is a selected
upper tail, not an unbiased full-sequence similarity or a merge-quality test.
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


STATISTICS_KIND = 'raw_k_top200_sequence_adjacent_v1'
TOP_K = 200
PRIMARY_METRIC = 'raw_k_top200'
FIELDS = {PRIMARY_METRIC: 'raw_k_topk_mean', 'k_raw_all_adjacent': 'k_raw',
          'k_rope_all_adjacent': 'k_rope', 'v_all_adjacent': 'v'}
METRICS = tuple(FIELDS)
CAPTURES = ('k_raw', 'k_rope', 'v')
GROUPS = {
    'text': dict(model_id='Qwen/Qwen2.5-7B-Instruct', task='qasper',
                 revision='a09a35458c702b33eeacc393d103063234e8bc28',
                 label='Qwen2.5-7B · LongBench', color='#B98A2F', cluster_key='context_sha256'),
    'video': dict(model_id='Qwen/Qwen2.5-VL-7B-Instruct', task='vatex_test',
                  revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',
                  label='Qwen2.5-VL-7B · VATEX', color='#2F4373', cluster_key='source_video_id'),
}


def expected_ids(sample_count):
    if type(sample_count) is not int or sample_count not in (2, 200):
        raise ValueError('Use exactly the pilot2 or formal200 original inputs per group')
    return {group: tuple(range(sample_count)) for group in GROUPS}


def _selection_declaration(record):
    name, digest = record.get('selected_pairs_path'), record.get('selected_pairs_sha256')
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or '..' in Path(name).parts or not name.endswith('.npz')):
        raise ValueError('Selected-pair NPZ path must be relative to its worker output directory')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Selected-pair artifact must declare its SHA256')
    return name, digest


def _collapse(record):
    if record.get('statistics_kind') != STATISTICS_KIND:
        raise ValueError('Require raw_k_top200_sequence_adjacent_v1; old all-pairs/adjacent records cannot be used')
    length = _integer(record.get('input_tokens'), 'input_tokens', TOP_K+1)
    pairs = _integer(record.get('adjacent_pair_count'), 'adjacent_pair_count', TOP_K)
    if pairs != length-1:
        raise ValueError('Candidates must be exactly the complete N-1 immediate sequence pairs')
    if type(record.get('top_k')) is not int or record['top_k'] != TOP_K:
        raise ValueError('Exactly200 pairs must be selected per input/layer/native head')
    _selection_declaration(record)
    layers = record.get('layer_stats')
    if not isinstance(layers, list) or len(layers) != LAYERS:
        raise ValueError(f'Require exactly{LAYERS} decoder layers')
    if sorted(_integer(layer.get('layer'), 'layer') for layer in layers) != list(range(LAYERS)):
        raise ValueError('Layer IDs must cover0..27 exactly once')
    values, details = defaultdict(list), []
    for layer in layers:
        if (type(layer.get('kv_heads')) is not int or layer['kv_heads'] != HEADS
                or type(layer.get('head_dim')) is not int or layer['head_dim'] != HEAD_DIM):
            raise ValueError('Require4 native KV heads of dimension128 before GQA repetition')
        if type(layer.get('sequence_length')) is not int or layer['sequence_length'] != length:
            raise ValueError('Layer sequence length differs from the full input')
        if type(layer.get('adjacent_pair_count')) is not int or layer['adjacent_pair_count'] != pairs:
            raise ValueError('Layer must observe every N-1 immediate sequence pair')
        if type(layer.get('top_k')) is not int or layer['top_k'] != TOP_K:
            raise ValueError('Layer selected count must be exactly200')
        captures = layer.get('capture_counts', {})
        if set(captures) != set(CAPTURES) or any(type(captures[m]) is not int or captures[m] != 1 for m in CAPTURES):
            raise ValueError('Raw K, post-RoPE K and V must each be captured once')
        vectors = {metric: _vector(layer.get(field), field) for metric, field in FIELDS.items()}
        minima = _vector(layer.get('raw_k_topk_min'), 'raw_k_topk_min')
        for head in range(HEADS):
            top = vectors[PRIMARY_METRIC][head]
            if top+1e-12 < vectors['k_raw_all_adjacent'][head] or top+1e-12 < minima[head]:
                raise ValueError('Top200 mean must be at least its minimum and the full-adjacent raw K mean')
        for metric, heads in vectors.items():
            values[metric].append(statistics.fmean(heads))
            details.extend(dict(metric=metric, layer=layer['layer'], kv_head=head, mean=value,
                                selected_minimum=minima[head] if metric == PRIMARY_METRIC else None)
                           for head, value in enumerate(heads))
    return {metric: statistics.fmean(v) for metric, v in values.items()}, details


def verify_selected_pairs(record, records_path):
    """Validate the saved exact selection; never reselect, reorder or run a model."""
    import numpy as np

    relative, digest = _selection_declaration(record)
    root = Path(records_path).resolve().parent
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Selected-pair artifact is missing or outside the worker directory')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Selected-pair NPZ SHA256 mismatch')
    with np.load(path, allow_pickle=False) as archive:
        if not {'pair_starts', 'scores'} <= set(archive.files):
            raise ValueError('Selected-pair NPZ requires pair_starts and scores')
        starts, scores = archive['pair_starts'], archive['scores']
    shape = (LAYERS, HEADS, TOP_K)
    if (starts.shape != shape or scores.shape != shape or starts.dtype.kind not in 'iu'
            or starts.dtype.itemsize != 4 or scores.dtype.kind != 'f' or scores.dtype.itemsize != 8):
        raise ValueError('Selected-pair NPZ requires [28,4,200] int32 positions and float64 scores')
    if (not np.isfinite(scores).all() or (scores < -1).any() or (scores > 1).any()
            or (starts < 0).any() or (starts >= record['adjacent_pair_count']).any()):
        raise ValueError('Selected-pair score or sequence position is invalid')
    if (np.diff(scores, axis=-1) > 0).any():
        raise ValueError('Selected scores must be descending')
    if (np.diff(np.sort(starts, axis=-1), axis=-1) == 0).any():
        raise ValueError('Selected adjacent-pair positions must be unique within each layer/head')
    if ((np.diff(scores, axis=-1) == 0) & (np.diff(starts, axis=-1) <= 0)).any():
        raise ValueError('Tied scores must retain ascending original pair position')
    by_layer = {layer['layer']: layer for layer in record['layer_stats']}
    for layer in range(LAYERS):
        if (not np.allclose(scores[layer].mean(axis=-1), by_layer[layer]['raw_k_topk_mean'], rtol=0, atol=1e-12)
                or not np.allclose(scores[layer, :, -1], by_layer[layer]['raw_k_topk_min'], rtol=0, atol=1e-12)):
            raise ValueError('Selected-pair scores do not reproduce the declared mean/minimum')
    return dict(path=str(path), sha256=digest, shape=list(shape), scores_verified=True)


def build_summary(records, *, sample_count=200, read_errors=None):
    prescribed = expected_ids(sample_count)
    issues, seen, valid = list(read_errors or []), defaultdict(set), defaultdict(set)
    record_rows, head_values, head_minima, sources = [], defaultdict(list), defaultdict(list), defaultdict(set)
    metadata, input_meta = {}, defaultdict(list)
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict) or record.get('status') != 'completed' or type(record.get('schema_version')) is not int or record['schema_version'] != 1:
                raise ValueError('Require a completed schema1 record')
            group = record.get('group')
            if group not in GROUPS:
                raise ValueError('Require group=text or video')
            doc_id = _integer(record.get('doc_id'), 'doc_id')
            if doc_id not in prescribed[group] or doc_id in seen[group]:
                raise ValueError('Duplicate input or row outside the prescribed original prefix')
            seen[group].add(doc_id)
            spec = GROUPS[group]
            if record.get('model_id') != spec['model_id'] or record.get('model_revision') != spec['revision'] or record.get('task') != spec['task']:
                raise ValueError('Model, revision or dataset differs from the prescribed comparison')
            source = record.get('source_metadata')
            if not isinstance(source, dict) or not isinstance(source.get(spec['cluster_key']), str) or not source[spec['cluster_key']]:
                raise ValueError(f'Missing original source identity {spec["cluster_key"]}')
            if not isinstance(record.get('input_hashes'), dict) or not record['input_hashes']:
                raise ValueError('Require original processor input tensor hashes')
            means, details = _collapse(record)
            common = dict(group=group, doc_id=doc_id, model_id=spec['model_id'], task=spec['task'],
                          input_tokens=record['input_tokens'], adjacent_pair_count=record['adjacent_pair_count'],
                          selected_pair_count=TOP_K, source_id=source[spec['cluster_key']])
            # No partially validated input may contribute to any statistic.
            record_rows.extend(dict(**common, metric=m, mean=v) for m, v in means.items())
            for detail in details:
                key = (group, detail['metric'], detail['layer'], detail['kv_head'])
                head_values[key].append(detail['mean'])
                if detail['selected_minimum'] is not None:
                    head_minima[key].append(detail['selected_minimum'])
            sources[group].add(source[spec['cluster_key']])
            metadata[group] = dict(model_id=spec['model_id'], model_revision=spec['revision'], task=spec['task'])
            input_meta[group].append(dict(doc_id=doc_id, input_tokens=record['input_tokens'],
                                          adjacent_pair_count=record['adjacent_pair_count'], selected_pair_count=TOP_K,
                                          source_id=source[spec['cluster_key']],
                                          selected_pairs_path=record['selected_pairs_path'], selected_pairs_sha256=record['selected_pairs_sha256']))
            valid[group].add(doc_id)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            issues.append(dict(record_index=index, group=record.get('group') if isinstance(record, dict) else None,
                               doc_id=record.get('doc_id') if isinstance(record, dict) else None, error=str(error)))
    heads = [dict(group=k[0], metric=k[1], layer=k[2], kv_head=k[3], n_inputs=len(v), mean=statistics.fmean(v),
                  selected_minimum_mean=statistics.fmean(head_minima[k]) if head_minima[k] else None)
             for k, v in sorted(head_values.items())]
    groups = {}
    for group, spec in GROUPS.items():
        missing = sorted(set(prescribed[group])-valid[group])
        if missing:
            issues.append(dict(group=group, error='Missing valid completed prescribed inputs', doc_ids=missing))
        rows = input_meta[group]
        groups[group] = dict(**metadata.get(group, dict(model_id=spec['model_id'], model_revision=spec['revision'], task=spec['task'])),
                             n_inputs=len(rows), n_source_clusters=len(sources[group]), valid_doc_ids=sorted(valid[group]), missing_doc_ids=missing,
                             input_tokens=dict(min=min(r['input_tokens'] for r in rows), max=max(r['input_tokens'] for r in rows),
                                               mean=statistics.fmean(r['input_tokens'] for r in rows)) if rows else None,
                             candidate_adjacent_pairs_total=sum(r['adjacent_pair_count'] for r in rows),
                             selected_pairs_total_per_head_layer=TOP_K*len(rows),
                             metrics={m: dict(mean=statistics.fmean(r['mean'] for r in record_rows if r['group'] == group and r['metric'] == m))
                                      for m in METRICS} if rows else {})
    comparisons = {m: dict(text=groups['text']['metrics'][m]['mean'], video=groups['video']['metrics'][m]['mean'],
                           video_minus_text=groups['video']['metrics'][m]['mean']-groups['text']['metrics'][m]['mean'])
                   for m in METRICS if all(m in groups[g]['metrics'] for g in GROUPS)}
    return dict(schema_version=1, status='incomplete' if issues else 'complete', statistics_kind=STATISTICS_KIND,
                sample_count=sample_count, expected_total_inputs=2*sample_count, top_k=TOP_K, primary_metric=PRIMARY_METRIC,
                dimensions=dict(layers=LAYERS, native_kv_heads=HEADS, head_dim=HEAD_DIM), groups=groups, comparisons=comparisons,
                statistic='Per input/layer/native KV head: rank all N-1 immediate sequence-pair raw-K cosines before RoPE; take exactly the highest200 and average them',
                selection_scope='All consecutive positions in each original complete prefill sequence, including text/video/structural and sequence-boundary pairs; no pairing across inputs',
                selection_tie_rule='Descending cosine, ascending original pair-start position for exact ties',
                aggregation=f'Equal weights for the200 selected pairs within each input/head/layer, then equal head/layer and {sample_count} input weights; no length weighting',
                distribution_unit='112 layer/native-KV-head means over the prescribed inputs per model, not individual pair cosines or independent observations',
                supplemental_metrics='k_raw_all_adjacent, k_rope_all_adjacent and v_all_adjacent cover all N-1 pairs; neither post-RoPE K nor V is measured on the raw-K-selected200 positions',
                interpretation='Selected upper-tail raw-K similarity, not the average over all token pairs. Equal selected counts do not equalize candidate-pool sizes. Models, datasets, lengths and token roles differ; no isolated modality or merge-safety claim.',
                bootstrap=False, validation_issues=issues, record_table=record_rows, head_table=heads,
                source_table=[dict(group=g, **r) for g, rows in input_meta.items() for r in rows])


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
    selected = {g: [r['mean'] for r in summary['head_table'] if r['group'] == g and r['metric'] == PRIMARY_METRIC] for g in GROUPS}
    if any(len(v) != LAYERS*HEADS for v in selected.values()):
        raise ValueError('Complete plot requires112 layer/native-head means per group')
    display_left = max(-1., math.floor((min(min(v) for v in selected.values())-.03)*100)/100)
    curves = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': '#30343B',
                         'axes.labelcolor': '#30343B', 'axes.edgecolor': '#C8CBD0'}):
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        fig.subplots_adjust(left=.1, right=.985, bottom=.15, top=.76)
        for group, spec in GROUPS.items():
            density, bandwidth = _density_curve(selected[group], grid)
            axis.fill_between(grid, 0, density, color=spec['color'], alpha=.12, linewidth=0, zorder=1)
            axis.plot(grid, density, color=spec['color'], linewidth=2, label=spec['label'], zorder=3)
            curves.append(dict(group=group, metric=PRIMARY_METRIC, n_head_means=len(selected[group]),
                               observed_min=min(selected[group]), observed_max=max(selected[group]),
                               bandwidth=bandwidth, color=spec['color'],
                               density_area=float(np.trapz(density, grid))))
        axis.set(xlabel='Cosine Similarity', ylabel='Density', xlim=(display_left, 1), ylim=(0, None))
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(color='#E2E4E7', linewidth=.65, linestyle=(0, (2, 2)))
        axis.set_axisbelow(True)
        legend = axis.legend(loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=2, frameon=False, fontsize=8)
        title = fig.suptitle('Key Similarity Distribution Comparison (Before RoPE)', y=.98, fontsize=10)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = legend.get_window_extent(renderer).transformed(axis.transAxes.inverted()).bounds
        if bounds[1] <= 1 or title.get_window_extent(renderer).y0 <= legend.get_window_extent(renderer).y1:
            raise ValueError('Legend must lie above the data axes and below the title')
        for extension in ('png', 'pdf', 'svg'):
            path = Path(output)/f'raw_k_top200.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    return dict(statistics_kind=STATISTICS_KIND, metric=PRIMARY_METRIC, representation='boundary-reflected Gaussian KDE',
                bandwidth_rule='Scott; minimum0.01', grid_points=len(grid), full_density_domain=[-1, 1], display_range=[display_left, 1],
                curves=curves, legend=dict(location='outside_above_data_axes', bounds_in_axes_coordinates=list(bounds), below_title=True))


def write_report(summary, output, *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output/'record_summary.csv', summary['record_table'],
         ['group', 'doc_id', 'model_id', 'task', 'input_tokens', 'adjacent_pair_count', 'selected_pair_count', 'source_id', 'metric', 'mean'])
    _csv(output/'head_summary.csv', summary['head_table'],
         ['group', 'metric', 'layer', 'kv_head', 'n_inputs', 'mean', 'selected_minimum_mean'])
    _csv(output/'source_inputs.csv', summary['source_table'],
         ['group', 'doc_id', 'input_tokens', 'adjacent_pair_count', 'selected_pair_count', 'source_id', 'selected_pairs_path', 'selected_pairs_sha256'])
    group_rows = [dict(group=g, model_id=d['model_id'], model_revision=d['model_revision'], task=d['task'], n_inputs=d['n_inputs'],
                       n_source_clusters=d['n_source_clusters'], metric=m, mean=v['mean'],
                       candidate_adjacent_pairs_total=d['candidate_adjacent_pairs_total'], selected_pairs_total_per_head_layer=d['selected_pairs_total_per_head_layer'],
                       **{'input_tokens_'+k: value for k, value in (d['input_tokens'] or {}).items()})
                  for g, d in summary['groups'].items() for m, v in d['metrics'].items()]
    _csv(output/'group_summary.csv', group_rows,
         ['group', 'model_id', 'model_revision', 'task', 'n_inputs', 'n_source_clusters', 'metric', 'mean', 'input_tokens_min', 'input_tokens_max', 'input_tokens_mean',
          'candidate_adjacent_pairs_total', 'selected_pairs_total_per_head_layer'])
    _csv(output/'comparison_summary.csv', [dict(metric=m, **v) for m, v in summary['comparisons'].items()],
         ['metric', 'text', 'video', 'video_minus_text'])
    compact = {k: v for k, v in summary.items() if not k.endswith('_table')}
    compact['artifacts'] = dict(tables=['record_summary.csv', 'head_summary.csv', 'source_inputs.csv', 'group_summary.csv', 'comparison_summary.csv'], plot='disabled')
    if plots and summary['status'] == 'complete':
        compact['artifacts']['plot'] = _plot(summary, output)
    lines = ['# Raw-K top200 adjacent-token comparison: VATEX versus LongBench', '', f"Status: **{summary['status']}**.", '',
             summary['statistic'], '', summary['selection_scope'], '', summary['aggregation'], '',
             '| Metric | Qwen2.5-7B / LongBench | Qwen2.5-VL-7B / VATEX | Video minus text |', '|---|---:|---:|---:|']
    for metric, value in summary['comparisons'].items():
        lines.append(f"| {metric} | {value['text']:.9f} | {value['video']:.9f} | {value['video_minus_text']:+.9f} |")
    lines += ['', summary['supplemental_metrics'], '', summary['distribution_unit'], '', summary['interpretation'], '',
              'No bootstrap or significance test was performed. Input rows are the original prescribed prefix; repeated sources remain.']
    if summary['validation_issues']:
        lines += ['', 'Incomplete coverage or invalid records; these aggregates are provisional:', '', '```json', json.dumps(summary['validation_issues'], indent=2), '```']
    (output/'summary.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-input', type=Path, required=True)
    parser.add_argument('--video-input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, choices=(2, 200), default=200)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    records, errors, inputs, selections = [], [], {}, []
    for group, path in (('text', args.text_input), ('video', args.video_input)):
        loaded, issues = read_jsonl(path)
        errors.extend(dict(input_group=group, **issue) for issue in issues)
        inputs[group] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for row in loaded:
            try:
                if not isinstance(row, dict) or row.get('group') != group:
                    raise ValueError('Input file contains the wrong model group')
                _collapse(row)
                selected = verify_selected_pairs(row, path)
                selections.append(dict(group=group, doc_id=row['doc_id'], **selected))
                records.append(row)
            except (ValueError, TypeError, KeyError, OSError) as error:
                errors.append(dict(input_group=group, doc_id=row.get('doc_id') if isinstance(row, dict) else None, error=str(error)))
    summary = build_summary(records, sample_count=args.sample_count, read_errors=errors)
    summary['provenance'] = dict(input_files=inputs, selected_artifacts=selections, selection_artifact_verification='SHA, shape, signed bounds, unique positions, stable descending order, mean and minimum')
    write_report(summary, args.output, plots=not args.no_plots)
    print(json.dumps(dict(status=summary['status'], output=str(args.output), validation_issues=len(summary['validation_issues']))))
    return 0 if summary['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())

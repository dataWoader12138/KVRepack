#!/usr/bin/env python3
"""Strict CPU comparison of already completed GPU worker artifacts.

Worker JSON: a list, or {"records": [...]}; each record supplies id, config,
input_hashes, generated_ids, text, prefill_cache_shapes and cache_shapes. Cache
rows may also supply key_sha256/value_sha256. Scores are a torch.save dict
mapping those ids to CPU tensors stacked by generation step (or {"scores": dict}).

Examples:
  python compare_gpu_runs.py --reference-json original.json --candidate-json new.json \
    --reference-scores original.pt --candidate-scores new.pt --output-dir comparison

Exit codes: 0 passed; 1 failed; 2 incomplete (for example scores unavailable).
Inputs/config mismatches always fail even when generated tokens are identical.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def outcome(status: str, **details: Any) -> dict[str, Any]:
    return {'status': status, **details}


def equal_check(left: Any, right: Any) -> dict[str, Any]:
    return outcome('passed' if left == right else 'failed', reference=left, candidate=right)


def merged_status(checks) -> str:
    statuses = [check['status'] for check in checks]
    if 'failed' in statuses:
        return 'failed'
    if any(status in {'not_checked', 'incomplete'} for status in statuses):
        return 'incomplete'
    return 'passed'


def read_records(path: Path):
    def reject_constant(value):
        raise ValueError(f'{path}: non-finite JSON constant {value} is not permitted')
    data = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    records = data.get('records') if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError(f'{path}: expected a records list or an object with records:list')
    indexed = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not record['id']:
            raise ValueError(f'{path}: record {index} needs a nonempty string id')
        if record['id'] in indexed:
            raise ValueError(f'{path}: duplicate record id {record["id"]!r}')
        indexed[record['id']] = record
    if not indexed:
        raise ValueError(f'{path}: records list is empty')
    return indexed


def load_cpu_scores(path: Path | None):
    if path is None:
        return None, 'score artifact was not supplied'
    # This process performs comparison only. It cannot contend with GPU workers.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)

    def cuda_forbidden(*args, **kwargs):
        raise RuntimeError('CUDA initialization is forbidden in the artifact comparator')
    torch.cuda._lazy_init = cuda_forbidden
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict) and isinstance(data.get('scores'), dict):
        data = data['scores']
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise ValueError(f'{path}: score artifact must map string record ids to tensors')
    if not data:
        return {}, 'score artifact has no record scores'
    return data, None


def valid_tokens(value):
    return isinstance(value, list) and all(
        valid_tokens(item) if isinstance(item, list) else isinstance(item, int) and not isinstance(item, bool)
        for item in value
    )


def nonempty_tokens(value):
    return any(nonempty_tokens(item) if isinstance(item, list) else True for item in value)


def normalize_cache(value):
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        raise ValueError('cache_shapes must be a list')
    result = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get('layer'), int) or isinstance(row.get('layer'), bool):
            raise ValueError('each cache row needs an integer layer')
        layer = row['layer']
        if layer in result:
            raise ValueError(f'duplicate cache layer {layer}')
        for key in ('key', 'value'):
            shape = row.get(key)
            if not isinstance(shape, list) or not shape or not all(
                isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in shape
            ):
                raise ValueError(f'cache layer {layer} needs {key}:list[int] shape')
        result[layer] = row
    return {str(layer): result[layer] for layer in sorted(result)}


def cache_lengths(normalized):
    if normalized is None:
        return None
    result = {}
    for layer, row in normalized.items():
        result[layer] = {}
        for key in ('key', 'value'):
            shape = row[key]
            result[layer][key] = {
                'shape': shape,
                'stored_sequence_axis_length': shape[-2] if len(shape) >= 2 else None,
                'layout_note': 'flattened rows; per-head logical lengths require additional worker metadata'
                               if len(shape) == 2 else 'axis -2 of the reported tensor shape',
            }
    return result



def compare_cache_snapshot(left_value, right_value, *, required):
    """Compare shapes/dtypes plus separately validated per-layer tensor hashes."""
    try:
        left, right = normalize_cache(left_value), normalize_cache(right_value)
    except ValueError as exc:
        return (outcome('failed', reason=str(exc)),
                outcome('not_checked', compared=False, reason='cache metadata is malformed'))
    if left is None or right is None:
        details = {'reason': 'cache snapshots missing or empty',
                   'reference_present': left is not None, 'candidate_present': right is not None}
        return (outcome('failed' if required else 'not_checked', **details),
                outcome('not_checked', compared=False, reason='cache snapshots missing or empty'))
    hash_fields = ('key_sha256', 'value_sha256')
    def without_hashes(rows):
        return {layer: {key: value for key, value in row.items() if key not in hash_fields}
                for layer, row in rows.items()}
    shapes = equal_check(without_hashes(left), without_hashes(right))
    shapes.update({'reference_layer_count': len(left), 'candidate_layer_count': len(right),
                   'reference_lengths': cache_lengths(left), 'candidate_lengths': cache_lengths(right)})
    any_hash = any(field in row for rows in (left, right) for row in rows.values() for field in hash_fields)
    if not any_hash:
        return shapes, outcome('not_checked', compared=False,
                               reason='per-layer K/V tensor SHA-256 values were not exported')
    invalid = []
    for side, rows in (('reference', left), ('candidate', right)):
        for layer, row in rows.items():
            for field in hash_fields:
                value = row.get(field)
                if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
                    invalid.append({'side': side, 'layer': layer, 'field': field})
    if invalid:
        return shapes, outcome('failed', compared=False, reason='cache tensor hashes are incomplete or invalid',
                               invalid_hash_fields=invalid)
    left_hashes = {layer: {field: row[field] for field in hash_fields} for layer, row in left.items()}
    right_hashes = {layer: {field: row[field] for field in hash_fields} for layer, row in right.items()}
    hashes = equal_check(left_hashes, right_hashes)
    hashes.update({'compared': True, 'hash_algorithm': 'SHA-256',
                   'reference_layer_count': len(left), 'candidate_layer_count': len(right),
                   'meaning': 'Equality of exported complete per-layer K and V tensor content hashes'})
    return shapes, hashes


def number(value):
    value = float(value)
    return value if math.isfinite(value) else 'inf' if value > 0 else '-inf' if value < 0 else 'nan'


def compare_scores(left, right, *, atol: float, rtol: float):
    if left is None or right is None:
        return outcome('not_checked', reason='scores missing for this record',
                       reference_present=left is not None, candidate_present=right is not None)
    import torch
    # A tuple/list of original per-step output_scores is also accepted; no squeeze,
    # trimming, vocabulary sampling or token-only score comparison is performed.
    def tensor(value):
        if isinstance(value, (tuple, list)) and value and all(isinstance(x, torch.Tensor) for x in value):
            return torch.stack(list(value))
        return value
    left, right = tensor(left), tensor(right)
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return outcome('failed', reason='score values must be tensors or nonempty sequences of tensors')
    details = {
        'reference_shape': list(left.shape), 'candidate_shape': list(right.shape),
        'reference_dtype': str(left.dtype), 'candidate_dtype': str(right.dtype),
        'reference_device': str(left.device), 'candidate_device': str(right.device),
        'atol': atol, 'rtol': rtol,
    }
    if left.device.type != 'cpu' or right.device.type != 'cpu':
        return outcome('failed', reason='loaded score tensors must be on CPU', **details)
    if tuple(left.shape) != tuple(right.shape):
        return outcome('failed', reason='score tensor shape mismatch', **details)
    if left.ndim < 2:
        return outcome('failed', reason='expected [generation_steps, ..., vocabulary] score tensor', **details)
    if left.numel() == 0:
        return outcome('not_checked', reason='score tensor is empty', **details)
    if not left.is_floating_point() or not right.is_floating_point():
        return outcome('failed', reason='expected floating point full-vocabulary scores', **details)
    a, b = left.to(torch.float64), right.to(torch.float64)
    finite_pair = torch.isfinite(a) & torch.isfinite(b)
    matching_infinity = (torch.isposinf(a) & torch.isposinf(b)) | (torch.isneginf(a) & torch.isneginf(b))
    invalid = ~(finite_pair | matching_infinity)  # NaN always fails, including NaN on both sides.
    error = torch.zeros_like(a)
    error[finite_pair] = torch.abs(a[finite_pair] - b[finite_pair])
    error[invalid] = float('inf')
    flat_error = error.reshape(error.shape[0], -1)
    close = torch.zeros_like(finite_pair)
    close[finite_pair] = error[finite_pair] <= atol + rtol * torch.abs(a[finite_pair])
    close[matching_infinity] = True
    exact = bool(torch.equal(left, right))
    dtype_equal = left.dtype == right.dtype
    value_close = bool(close.all().item())
    details.update({
        'generation_steps': int(left.shape[0]), 'vocabulary_size': int(left.shape[-1]),
        'compared_values': int(left.numel()), 'dtype_equal': dtype_equal,
        'exact_values_equal': exact, 'all_values_within_tolerance': value_close,
        'max_abs_error': number(error.max().item()),
        'per_step_max_abs_error': [number(x) for x in flat_error.max(dim=1).values.tolist()],
        'values_outside_tolerance': int((~close).sum().item()),
        'reference_nan_count': int(torch.isnan(a).sum().item()),
        'candidate_nan_count': int(torch.isnan(b).sum().item()),
        'matching_infinity_count': int(matching_infinity.sum().item()),
        'mismatching_or_invalid_nonfinite_count': int(invalid.sum().item()),
    })
    return outcome('passed' if dtype_equal and value_close else 'failed', **details)


def compare_record(reference, candidate, reference_scores, candidate_scores, *, atol, rtol):
    checks = {}
    for field in ('config', 'input_hashes'):
        left, right = reference.get(field), candidate.get(field)
        if not isinstance(left, dict) or not left or not isinstance(right, dict) or not right:
            checks[field] = outcome('failed', reason=f'both records require a nonempty {field} object')
        else:
            checks[field] = equal_check(left, right)
            if field == 'input_hashes' and not all(
                isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value)
                for values in (left, right) for value in values.values()
            ):
                checks[field] = outcome('failed', reason='input hashes must be SHA-256 hex strings',
                                        reference=left, candidate=right)
    left_ids, right_ids = reference.get('generated_ids'), candidate.get('generated_ids')
    if not valid_tokens(left_ids) or not valid_tokens(right_ids):
        checks['generated_ids'] = outcome('failed', reason='both records require generated_ids:list[int] (nested batches permitted)')
    elif not nonempty_tokens(left_ids) or not nonempty_tokens(right_ids):
        checks['generated_ids'] = outcome('not_checked', reason='generated token sequence is empty')
    else:
        checks['generated_ids'] = equal_check(left_ids, right_ids)
    if 'text' in reference and 'text' in candidate:
        checks['decoded_text'] = equal_check(reference['text'], candidate['text'])
    else:
        checks['decoded_text'] = outcome('not_checked', reason='decoded text was not supplied on both sides')
    for field, required in (('prefill_cache_shapes', True), ('cache_shapes', False)):
        shapes, hashes = compare_cache_snapshot(reference.get(field), candidate.get(field), required=required)
        checks[field] = shapes
        hash_check = 'prefill_cache_tensor_hashes' if field == 'prefill_cache_shapes' else 'cache_tensor_hashes'
        checks[hash_check] = hashes
    # These are compared when provided; neither is invented from token equality.
    optional = {}
    for field in ('task', 'doc_id', 'input_shapes', 'input_dtypes', 'output_dtype', 'cache_dtypes',
                  'cache_lengths', 'runtime_parameters'):
        if field in reference or field in candidate:
            if field not in reference or field not in candidate:
                checks[field] = outcome('failed', reason='metadata is present on only one side')
            else:
                checks[field] = equal_check(reference[field], candidate[field])
        else:
            optional[field] = 'not supplied; not checked separately'
    scores = compare_scores(reference_scores, candidate_scores, atol=atol, rtol=rtol)
    checks['full_vocabulary_scores'] = scores
    return {'id': reference['id'], 'status': merged_status(checks.values()),
            'checks': checks, 'optional_metadata_not_checked': optional}


def markdown(report):
    lines = [
        '# GPU worker artifact comparison', '',
        f"Overall status: **{report['status']}**. Comparison runs on CPU only.", '',
        f"Score tolerance: atol={report['score_tolerance']['atol']}, rtol={report['score_tolerance']['rtol']}. "
        'Matching signed infinities count as equal; any NaN fails.', '',
        '| Record | Config | Input hashes | Tokens | Text | Prefill shapes | Prefill K/V SHA | Final shapes | Final K/V SHA | Score dtype | Full-vocab max abs error | Scores | Status |',
        '|---|---|---|---|---|---|---|---|---|---|---:|---|---|',
    ]
    def escape(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    for row in report.get('records', []):
        checks = row['checks']
        score = checks.get('full_vocabulary_scores', {})
        dtype = ('equal' if score.get('dtype_equal') else 'different') if 'dtype_equal' in score else 'not checked'
        fields = [row['id'], *[checks.get(key, {}).get('status', 'not checked') for key in
            ('config', 'input_hashes', 'generated_ids', 'decoded_text', 'prefill_cache_shapes',
             'prefill_cache_tensor_hashes', 'cache_shapes', 'cache_tensor_hashes')],
            dtype, score.get('max_abs_error', 'not checked'), score.get('status', 'not checked'), row['status']]
        lines.append('| ' + ' | '.join(escape(value) for value in fields) + ' |')
    lines += ['', 'The JSON report includes every generation step’s maximum absolute error, complete tensor shapes, '
              'record/config differences, source artifact hashes and unchecked metadata.', '',
              'Prefill cache snapshots are mandatory: missing or empty snapshots fail. Both prefill and final '
              'cache shapes/stored lengths are compared. Per-layer K/V tensor content hashes are compared only '
              'when all hashes are present and valid SHA-256 values; malformed or partial hashes fail, '
              'and absent hashes remain explicitly not checked.', '',
              'A missing/empty score artifact is explicitly not checked and makes the result incomplete. '
              'Equal generated tokens alone cannot make the full comparison pass.']
    if report.get('errors'):
        lines += ['', '## Errors', ''] + ['- ' + escape(value) for value in report['errors']]
    for name, result in report.get('global_checks', {}).items():
        if result['status'] != 'passed':
            lines += ['', f'- {name}: {escape(json.dumps(result, ensure_ascii=False))}']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-json', type=Path, required=True)
    parser.add_argument('--candidate-json', type=Path, required=True)
    parser.add_argument('--reference-scores', type=Path)
    parser.add_argument('--candidate-scores', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=0., help='Explicit absolute score tolerance; default exact')
    parser.add_argument('--rtol', type=float, default=0., help='Explicit relative score tolerance; default exact')
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error('atol and rtol must be finite and nonnegative')
    input_paths = {path.resolve() for path in (args.reference_json, args.candidate_json,
                   args.reference_scores, args.candidate_scores) if path is not None}
    for name in ('summary.json', 'comparison.md'):
        if (args.output_dir / name).resolve() in input_paths:
            parser.error(f'output {name} would overwrite an input artifact; choose another --output-dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'schema_version': 2, 'status': 'failed', 'comparison_device': 'cpu',
        'comparison_threads': 1, 'full_model_or_gpu_execution_performed_by_comparator': False,
        'score_tolerance': {'atol': args.atol, 'rtol': args.rtol,
                            'mode': 'exact' if args.atol == args.rtol == 0 else 'explicit_tolerance'},
        'artifacts': {}, 'global_checks': {}, 'records': [], 'errors': [],
    }
    try:
        for name in ('reference_json', 'candidate_json', 'reference_scores', 'candidate_scores'):
            path = getattr(args, name)
            report['artifacts'][name] = {'path': str(path.resolve()), 'sha256': sha256_file(path)} if path else None
        reference, candidate = read_records(args.reference_json), read_records(args.candidate_json)
        report['global_checks']['record_ids'] = equal_check(sorted(reference), sorted(candidate))
        ref_scores, ref_reason = load_cpu_scores(args.reference_scores)
        new_scores, new_reason = load_cpu_scores(args.candidate_scores)
        if args.reference_scores or args.candidate_scores:
            import torch
            report['torch_version'] = str(torch.__version__)
        for name, scores, expected, reason in (
            ('reference_score_records', ref_scores, reference, ref_reason),
            ('candidate_score_records', new_scores, candidate, new_reason),
        ):
            if scores is None or not scores:
                report['global_checks'][name] = outcome('not_checked', reason=reason)
            elif set(scores) - set(expected):
                report['global_checks'][name] = outcome('failed', reason='unexpected score record ids', extra=sorted(set(scores)-set(expected)))
            elif set(expected) - set(scores):
                report['global_checks'][name] = outcome('not_checked', reason='some records have no scores', missing=sorted(set(expected)-set(scores)))
            else:
                report['global_checks'][name] = outcome('passed', record_count=len(scores))
        for record_id in sorted(set(reference) & set(candidate)):
            report['records'].append(compare_record(reference[record_id], candidate[record_id],
                ref_scores.get(record_id) if ref_scores else None,
                new_scores.get(record_id) if new_scores else None, atol=args.atol, rtol=args.rtol))
        report['status'] = merged_status([*report['global_checks'].values(),
            *[{'status': row['status']} for row in report['records']]])
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        report['status'] = 'failed'
    report['counts'] = dict(Counter(row['status'] for row in report['records']))
    summary_path = args.output_dir / 'summary.json'
    markdown_path = args.output_dir / 'comparison.md'
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    markdown_path.write_text(markdown(report), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'record_counts': report['counts'],
                      'summary': str(summary_path), 'markdown': str(markdown_path)}, indent=2))
    return {'passed': 0, 'failed': 1, 'incomplete': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare full-benchmark worker JSONL artifacts using bounded-memory disk indexing.

  python compare_full_runs.py --reference original.jsonl --candidate repack.jsonl \
      --expected-count 9178 --output comparison.json

Order is irrelevant. Default final mode returns 0 only for complete exact matches;
1 means failed integrity/equality and 2 means incomplete. --limit N compares
only doc_id 0..N-1 of a growing full-run file; final mode validates the entire
declared doc_id range 0..expected_count-1. --progress permits exit 0
for an incomplete snapshot, but the JSON status remains 'incomplete'. No torch,
models, GPU operations, raw logits or raw cache tensors are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time

SHA256 = re.compile(r'[0-9a-fA-F]{64}\Z')
CACHE_FIELDS = ('prefill_cache_shapes', 'cache_shapes')
REQUIRED_FIELDS = ('doc_id', 'task', 'input_hashes', 'generated_ids', 'text',
                   'logits_sha256', 'logits_shape', 'logits_dtype', 'config')
OPTIONAL_FIELDS = ('input_shapes', 'input_dtypes', 'prompt_length', 'input_length',
                   'runtime_parameters', 'generation_kwargs')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_record(raw):
    def reject(value):
        raise ValueError('Non-finite JSON constant is not permitted: ' + value)
    return json.loads(raw, parse_constant=reject)


def valid_hash(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def token_list(value):
    return isinstance(value, list) and bool(value) and all(
        token_list(item) if isinstance(item, list)
        else isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in value
    )


def shape(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value
    )


def cache_map(value, expected_layers):
    if not isinstance(value, list) or not value:
        raise ValueError('missing or empty cache snapshots')
    result = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError('cache layer snapshot is not an object')
        layer = row.get('layer')
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError('cache layer must be a nonnegative integer')
        if layer in result:
            raise ValueError(f'duplicate cache layer {layer}')
        if not all(shape(row.get(key)) for key in ('key', 'value')):
            raise ValueError(f'cache layer {layer} has missing/invalid key/value shapes')
        if not all(valid_hash(row.get(key)) for key in ('key_sha256', 'value_sha256')):
            raise ValueError(f'cache layer {layer} has missing/invalid key/value SHA-256 hashes')
        result[layer] = row
    if set(result) != set(range(expected_layers)):
        raise ValueError(f'expected complete layers 0..{expected_layers-1}; got {sorted(result)}')
    return result


def prompt_length(record):
    lengths = []
    for key in ('prompt_length', 'input_length'):
        if key in record:
            value = record[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(key + ' must be a positive integer')
            lengths.append(value)
    inputs = record.get('input_shapes')
    if isinstance(inputs, dict) and 'input_ids' in inputs:
        if not shape(inputs['input_ids']):
            raise ValueError('input_shapes.input_ids is not a valid shape')
        lengths.append(inputs['input_ids'][-1])
    if lengths and len(set(lengths)) != 1:
        raise ValueError('exported input/prompt lengths disagree')
    return lengths[0] if lengths else None


def validate_record(record, expected_layers, expected_count):
    errors = []
    if not isinstance(record, dict):
        return ['record is not an object']
    if not isinstance(record.get('id'), str) or not record.get('id'):
        errors.append('id must be a nonempty string')
    status = str(record.get('status', '')).lower()
    if status in {'error', 'failed', 'failure', 'incomplete', 'skipped', 'pending'} or record.get('error'):
        errors.append('worker reported an error/non-success record: ' + str(record.get('error', status))[:500])
    missing = [key for key in REQUIRED_FIELDS if key not in record]
    if missing:
        errors.append('missing required fields: ' + ','.join(missing))
    if not isinstance(record.get('task'), str) or not record.get('task'):
        errors.append('task must be a nonempty string')
    doc_id = record.get('doc_id')
    if not isinstance(doc_id, int) or isinstance(doc_id, bool):
        errors.append('doc_id must be an integer')
    elif not 0 <= doc_id < expected_count:
        errors.append(f'doc_id must cover the declared range 0..{expected_count-1}')
    if isinstance(record.get('task'), str) and isinstance(doc_id, int) and record.get('id') != f"{record['task']}:{doc_id}":
        errors.append('id must equal task:doc_id')
    hashes = record.get('input_hashes')
    if not isinstance(hashes, dict) or not hashes or not all(
        isinstance(key, str) and bool(key) and valid_hash(value) for key, value in hashes.items()
    ):
        errors.append('input_hashes must be a nonempty mapping of tensor names to SHA-256 strings')
    if not isinstance(record.get('config'), dict) or not record.get('config'):
        errors.append('config must contain actual nonempty experiment parameters')
    if not token_list(record.get('generated_ids')):
        errors.append('generated_ids must be a nonempty integer token list (nested batches allowed)')
    if not isinstance(record.get('text'), str):
        errors.append('text must be a string')
    if not valid_hash(record.get('logits_sha256')):
        errors.append('logits_sha256 must be a SHA-256 string')
    logits_shape = record.get('logits_shape')
    if not shape(logits_shape) or len(logits_shape) < 2:
        errors.append('logits_shape must have nonempty generation-step and full-vocabulary axes')
    if not isinstance(record.get('logits_dtype'), str) or not record.get('logits_dtype'):
        errors.append('logits_dtype must be a nonempty string')
    for field in CACHE_FIELDS:
        try:
            cache_map(record.get(field), expected_layers)
        except ValueError as exc:
            errors.append(field + ': ' + str(exc))
    try:
        prompt_length(record)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def brief(value):
    if isinstance(value, str) and len(value) > 240:
        return {'length': len(value), 'sha256_utf8': digest(value.encode()), 'prefix': value[:100]}
    if isinstance(value, list) and len(value) > 12:
        return {'length': len(value), 'sha256_canonical_json': digest(canonical(value).encode()), 'prefix': value[:8]}
    if isinstance(value, dict) and len(canonical(value)) > 1000:
        return {'keys': list(value), 'sha256_canonical_json': digest(canonical(value).encode())}
    return value


def field_difference(left, right, limit):
    if isinstance(left, dict) and isinstance(right, dict):
        changed = [key for key in sorted(set(left) | set(right))
                   if key not in left or key not in right or canonical(left[key]) != canonical(right[key])]
        return {'different_key_count': len(changed), 'different_keys': changed,
                'examples': [{'key': key, 'reference': brief(left.get(key)), 'candidate': brief(right.get(key)),
                              'reference_present': key in left, 'candidate_present': key in right}
                             for key in changed[:limit]]}
    if isinstance(left, list) and isinstance(right, list):
        examples = [{'index': i, 'reference': brief(a), 'candidate': brief(b)}
                    for i, (a, b) in enumerate(zip(left, right)) if canonical(a) != canonical(b)]
        return {'reference_length': len(left), 'candidate_length': len(right),
                'different_overlap_entries': len(examples), 'examples': examples[:limit],
                'reference_sha256_canonical_json': digest(canonical(left).encode()),
                'candidate_sha256_canonical_json': digest(canonical(right).encode())}
    return {'reference': brief(left), 'candidate': brief(right)}


def compare_records(left, right, expected_layers, limit):
    differences = {}
    for field in REQUIRED_FIELDS + ('id',) + OPTIONAL_FIELDS:
        if field not in left and field not in right:
            continue
        if field not in left or field not in right or canonical(left.get(field)) != canonical(right.get(field)):
            differences[field] = field_difference(left.get(field), right.get(field), limit)
    for field in CACHE_FIELDS:
        a, b = cache_map(left[field], expected_layers), cache_map(right[field], expected_layers)
        changed = [layer for layer in a if canonical(a[layer]) != canonical(b[layer])]
        if changed:
            differences[field] = {
                'different_layer_count': len(changed), 'different_layers': changed,
                'examples': [{'layer': layer, **field_difference(a[layer], b[layer], limit)} for layer in changed[:limit]],
            }
    return differences


def compression_evidence(record, expected_layers):
    cache = cache_map(record['prefill_cache_shapes'], expected_layers)
    length = prompt_length(record)
    if length is None:
        return {'measured': False, 'reason': 'worker did not export prompt_length/input_length/input_shapes.input_ids'}
    # Flat [sum(head_lengths), dim] tensors cannot establish per-head compression.
    if any(len(row['key']) < 3 or len(row['value']) < 3 for row in cache.values()):
        return {'measured': False, 'reason': 'flattened cache needs exported per-head lengths to prove reduction'}
    shorter = [layer for layer, row in cache.items() if row['key'][-2] < length and row['value'][-2] < length]
    return {'measured': True, 'prompt_length': length, 'compressed_layers': len(shorter),
            'all_layers_compressed': len(shorter) == expected_layers,
            'prefill_key_length_min': min(row['key'][-2] for row in cache.values()),
            'prefill_key_length_max': max(row['key'][-2] for row in cache.values())}


def add_issue(db, side, line, record_id, kind, details):
    db.execute('INSERT INTO issues(side,line,id,kind,details) VALUES(?,?,?,?,?)',
               (side, line, record_id, kind, canonical(details)))


def scan_file(path, side, db, *, progress, expected_layers, expected_count, selection_limit=None, on_record=None):
    stats = {'path': str(path.resolve()), 'physical_lines': 0, 'json_records': 0,
             'unique_ids': 0, 'valid_unique_records': 0, 'duplicate_ids': 0,
             'error_or_invalid_records': 0, 'malformed_lines': 0, 'pending_partial_tail': 0,
             'records_outside_selected_prefix': 0,
             'missing_file': False, 'file_changed_while_scanning': False}
    if not path.is_file():
        stats['missing_file'] = True
        add_issue(db, side, None, None, 'missing_file', {'path': str(path), 'pending': progress})
        return stats
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        boundary = before.st_size
        stats['snapshot_size_bytes'] = boundary
        while handle.tell() < boundary:
            offset = handle.tell()
            raw = handle.readline(boundary - offset)
            if not raw:
                break
            hasher.update(raw)
            stats['physical_lines'] += 1
            line = stats['physical_lines']
            if not raw.strip():
                continue
            try:
                record = parse_record(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                pending = bool((progress or selection_limit is not None) and handle.tell() == boundary and not raw.endswith(b'\n'))
                stats['pending_partial_tail' if pending else 'malformed_lines'] += 1
                add_issue(db, side, line, None, 'pending_partial_tail' if pending else 'malformed_json', {'message': str(exc)[:500]})
                continue
            stats['json_records'] += 1
            if (selection_limit is not None and isinstance(record, dict)
                    and isinstance(record.get('doc_id'), int) and not isinstance(record.get('doc_id'), bool)
                    and record['doc_id'] >= selection_limit and isinstance(record.get('task'), str)
                    and record.get('id') == f"{record['task']}:{record['doc_id']}"):
                stats['records_outside_selected_prefix'] += 1
                continue
            errors = validate_record(record, expected_layers, expected_count)
            record_id = record.get('id') if isinstance(record, dict) else None
            if errors:
                stats['error_or_invalid_records'] += 1
                add_issue(db, side, line, record_id, 'invalid_record', {'errors': errors})
            if not isinstance(record_id, str) or not record_id:
                continue
            task = record.get('task') if isinstance(record.get('task'), str) else None
            doc = str(record.get('doc_id')) if record.get('doc_id') is not None else None
            try:
                db.execute('INSERT INTO records(side,id,offset,length,line,valid,task,doc,record_sha) VALUES(?,?,?,?,?,?,?,?,?)',
                           (side, record_id, offset, len(raw), line, int(not errors), task, doc, digest(raw)))
            except sqlite3.IntegrityError:
                stats['duplicate_ids'] += 1
                add_issue(db, side, line, record_id, 'duplicate_id', {'id': record_id})
                continue
            stats['unique_ids'] += 1
            if not errors:
                stats['valid_unique_records'] += 1
                duplicate_doc = db.execute('SELECT id FROM records WHERE side=? AND task=? AND doc=? AND id<>? LIMIT 1',
                                           (side, task, doc, record_id)).fetchone()
                if duplicate_doc:
                    add_issue(db, side, line, record_id, 'duplicate_document_identity', {'task': task, 'doc_id': doc, 'other_id': duplicate_doc[0]})
                if on_record:
                    on_record(record, line)
            if stats['json_records'] % 1000 == 0:
                db.commit()
        after = os.fstat(handle.fileno())
        stats['scanned_prefix_sha256'] = hasher.hexdigest()
        stats['size_after_scan_bytes'] = after.st_size
        stats['file_changed_while_scanning'] = before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
    db.commit()
    return stats


def stream_json_report(output, summary, array_queries, db):
    """Atomic JSON output with streamed detail arrays, avoiding an all-errors RAM list."""
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                         prefix='.' + output.name + '.', suffix='.tmp', delete=False)
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write('{\n')
            first = True
            for key, value in summary.items():
                if not first:
                    handle.write(',\n')
                handle.write('  ' + json.dumps(key) + ': ' + json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
                first = False
            for key, (query, encode_row) in array_queries.items():
                handle.write(',\n  ' + json.dumps(key) + ': [')
                first_item = True
                for row in db.execute(query):
                    if not first_item:
                        handle.write(',')
                    handle.write('\n    ' + encode_row(row))
                    first_item = False
                handle.write('\n  ]')
            handle.write('\n}\n')
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Original worker JSONL')
    parser.add_argument('--candidate', type=Path, required=True, help='Repacked worker JSONL')
    parser.add_argument('--expected-count', type=int, required=True, help='Declared full-run document count per side')
    parser.add_argument('--output', type=Path, required=True, help='JSON summary file; every mismatched id is included')
    parser.add_argument('--expected-cache-layers', type=int, default=36)
    parser.add_argument('--limit', type=int, help='Compare only doc_id 0..limit-1 from full/growing files; not a full-benchmark result')
    parser.add_argument('--max-examples-per-field', type=int, default=8, help='Limit details per mismatch, not the number of mismatched documents')
    parser.add_argument('--progress', action='store_true', help='Allow exit0 for incomplete snapshots, with status still incomplete')
    args = parser.parse_args(argv)
    if min(args.expected_count, args.expected_cache_layers, args.max_examples_per_field) <= 0:
        parser.error('expected count, cache layers, and detail limit must be positive')
    if args.limit is not None and not 0 < args.limit <= args.expected_count:
        parser.error('--limit must be positive and no greater than --expected-count')
    target_count = args.limit if args.limit is not None else args.expected_count
    if args.output.resolve() in {args.reference.resolve(), args.candidate.resolve()}:
        parser.error('--output would overwrite an input JSONL')
    if args.output.is_dir():
        parser.error('--output must be a JSON file, not a directory')
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='kvrepack-compare-index-') as directory:
        db = sqlite3.connect(Path(directory) / 'index.sqlite')
        db.execute('PRAGMA cache_size=-8192')
        db.execute('CREATE TABLE records(side TEXT,id TEXT,offset INTEGER,length INTEGER,line INTEGER,valid INTEGER,task TEXT,doc TEXT,record_sha TEXT,PRIMARY KEY(side,id))')
        db.execute('CREATE INDEX doc_identity ON records(side,task,doc)')
        db.execute('CREATE TABLE issues(side TEXT,line INTEGER,id TEXT,kind TEXT,details TEXT)')
        db.execute('CREATE TABLE mismatches(id TEXT PRIMARY KEY,details TEXT)')
        db.execute('CREATE TABLE expected_docs(doc INTEGER PRIMARY KEY)')
        db.executemany('INSERT INTO expected_docs(doc) VALUES(?)', ((i,) for i in range(target_count)))
        counts = Counter()
        field_counts = Counter()
        proof = Counter()
        reasons = Counter()
        reference_stats = scan_file(args.reference, 'reference', db, progress=args.progress,
                                    expected_layers=args.expected_cache_layers, expected_count=target_count, selection_limit=args.limit)
        ref_handle = args.reference.open('rb') if args.reference.is_file() else None
        def on_candidate(record, candidate_line):
            location = db.execute('SELECT offset,length,line,valid,record_sha FROM records WHERE side=? AND id=?',
                                  ('reference', record['id'])).fetchone()
            if location is None or not location[3] or ref_handle is None:
                return
            offset, length, ref_line, _, original_sha = location
            ref_handle.seek(offset)
            raw = ref_handle.read(length)
            if digest(raw) != original_sha:
                add_issue(db, 'reference', ref_line, record['id'], 'reference_record_changed_during_comparison', {})
                return
            reference = parse_record(raw)
            differences = compare_records(reference, record, args.expected_cache_layers, args.max_examples_per_field)
            counts['compared_valid_records'] += 1
            for cache_field, proof_key in [('prefill_cache_shapes','prefill_cache_tensor_hashes_equal'),
                                            ('cache_shapes','final_cache_tensor_hashes_equal')]:
                a, b = cache_map(reference[cache_field], args.expected_cache_layers), cache_map(record[cache_field], args.expected_cache_layers)
                if all(a[layer]['key_sha256'] == b[layer]['key_sha256'] and a[layer]['value_sha256'] == b[layer]['value_sha256'] for layer in a):
                    proof[proof_key] += 1
            if all(field not in differences for field in CACHE_FIELDS):
                proof['prefill_and_final_complete_cache_snapshots_equal'] += 1
            left_compression = compression_evidence(reference, args.expected_cache_layers)
            right_compression = compression_evidence(record, args.expected_cache_layers)
            if left_compression['measured'] and right_compression['measured']:
                proof['records_with_compression_amount_measured'] += 1
                if left_compression['compressed_layers'] > 0 and right_compression['compressed_layers'] > 0:
                    proof['records_with_some_prefill_layers_compressed_on_both_sides'] += 1
                if left_compression['all_layers_compressed'] and right_compression['all_layers_compressed']:
                    proof['records_with_all_prefill_layers_compressed_on_both_sides'] += 1
            else:
                reasons[left_compression.get('reason', right_compression.get('reason', 'unmeasured'))] += 1
            if differences:
                counts['mismatched_records'] += 1
                field_counts.update(differences.keys())
                detail = {'id': record['id'], 'task': record['task'], 'doc_id': record['doc_id'],
                          'reference_line': ref_line, 'candidate_line': candidate_line,
                          'different_fields': list(differences), 'differences': differences,
                          'compression_evidence': {'reference': left_compression, 'candidate': right_compression}}
                db.execute('INSERT INTO mismatches(id,details) VALUES(?,?)', (record['id'], canonical(detail)))
            else:
                counts['exactly_matching_records'] += 1
        try:
            candidate_stats = scan_file(args.candidate, 'candidate', db, progress=args.progress,
                                        expected_layers=args.expected_cache_layers, expected_count=target_count,
                                        selection_limit=args.limit, on_record=on_candidate)
        finally:
            if ref_handle is not None:
                ref_handle.close()
        task_sets = {}
        coverage = {}
        for side in ('reference', 'candidate'):
            tasks = [row[0] for row in db.execute('SELECT DISTINCT task FROM records WHERE side=? AND valid=1 ORDER BY task', (side,))]
            task_sets[side] = tasks
            if len(tasks) > 1:
                add_issue(db, side, None, None, 'multiple_tasks_in_single_task_file', {'tasks': tasks})
            low, high, distinct_count = db.execute('SELECT MIN(CAST(doc AS INTEGER)),MAX(CAST(doc AS INTEGER)),COUNT(DISTINCT doc) FROM records WHERE side=? AND valid=1', (side,)).fetchone()
            missing_count = db.execute('SELECT COUNT(*) FROM expected_docs e WHERE NOT EXISTS(SELECT 1 FROM records r WHERE r.side=? AND r.valid=1 AND CAST(r.doc AS INTEGER)=e.doc)', (side,)).fetchone()[0]
            coverage[side] = {'expected_min': 0, 'expected_max': target_count-1, 'observed_min': low, 'observed_max': high,
                              'distinct_valid_doc_ids': distinct_count, 'missing_expected_doc_ids': missing_count}
        if task_sets['reference'] and task_sets['candidate'] and task_sets['reference'] != task_sets['candidate']:
            add_issue(db, 'both', None, None, 'task_mismatch', task_sets)
        issue_counts = dict(db.execute('SELECT kind,COUNT(*) FROM issues GROUP BY kind').fetchall())
        matched_ids = db.execute("SELECT COUNT(*) FROM records a JOIN records b ON a.id=b.id WHERE a.side='reference' AND b.side='candidate'").fetchone()[0]
        missing_candidate = db.execute("SELECT COUNT(*) FROM records a LEFT JOIN records b ON a.id=b.id AND b.side='candidate' WHERE a.side='reference' AND b.id IS NULL").fetchone()[0]
        missing_reference = db.execute("SELECT COUNT(*) FROM records a LEFT JOIN records b ON a.id=b.id AND b.side='reference' WHERE a.side='candidate' AND b.id IS NULL").fetchone()[0]
        hard_kinds = {key: value for key, value in issue_counts.items() if key not in {'pending_partial_tail', 'missing_file'}}
        hard_failure = bool(hard_kinds or counts['mismatched_records'] or
                            max(reference_stats['unique_ids'], candidate_stats['unique_ids']) > target_count or
                            (not args.progress and issue_counts.get('missing_file')))
        complete = all(stats['valid_unique_records'] == target_count and
                       (args.limit is not None or (not stats['pending_partial_tail'] and not stats['file_changed_while_scanning']))
                       for stats in (reference_stats, candidate_stats)) and counts['compared_valid_records'] == target_count and not missing_candidate and not missing_reference and all(c['missing_expected_doc_ids'] == 0 for c in coverage.values())
        status = 'failed' if hard_failure else 'passed' if complete else 'incomplete'
        per_task = {side: {task: {'unique_ids': total, 'valid_records': valid} for task, total, valid in
                          db.execute('SELECT task,COUNT(*),SUM(valid) FROM records WHERE side=? GROUP BY task', (side,))}
                    for side in ('reference', 'candidate')}
        for key in ('compared_valid_records', 'exactly_matching_records', 'mismatched_records'):
            counts.setdefault(key, 0)
        for key in ('prefill_cache_tensor_hashes_equal', 'final_cache_tensor_hashes_equal',
                    'prefill_and_final_complete_cache_snapshots_equal', 'records_with_compression_amount_measured',
                    'records_with_some_prefill_layers_compressed_on_both_sides',
                    'records_with_all_prefill_layers_compressed_on_both_sides'):
            proof.setdefault(key, 0)
        summary = {
            'schema_version': 1, 'status': status, 'mode': 'progress' if args.progress else 'final',
            'complete_expected_run': bool(complete and not hard_failure), 'expected_count_per_side': target_count,
            'declared_full_count_per_side': args.expected_count, 'selected_prefix_limit': args.limit,
            'full_benchmark_validated': bool(complete and not hard_failure and args.limit is None),
            'scope': 'full declared doc_id range' if args.limit is None else f'prefix doc_id 0..{target_count-1} only',
            'doc_id_coverage': coverage, 'task_sets': task_sets,
            'expected_cache_layers': args.expected_cache_layers, 'reference': reference_stats, 'candidate': candidate_stats,
            'counts': {**dict(counts), 'matched_ids': matched_ids, 'missing_candidate_ids': missing_candidate,
                       'missing_reference_ids': missing_reference}, 'per_task_counts': per_task,
            'mismatch_field_counts': dict(field_counts), 'issue_counts': issue_counts,
            'compressed_cache_proof': {**dict(proof), 'compression_amount_unchecked_reasons': dict(reasons),
                'all_expected_prefill_and_final_hashes_equal': proof['prefill_and_final_complete_cache_snapshots_equal'] == target_count and bool(complete and not hard_kinds),
                'compression_amount_checked_for_all_expected_records': proof['records_with_compression_amount_measured'] == target_count and bool(complete and not hard_kinds),
                'some_prefill_compression_observed': proof['records_with_some_prefill_layers_compressed_on_both_sides'] > 0},
            'checks': ['matched unique record ids and declared full counts', 'task and doc_id', 'actual nonempty config',
                       'actual input tensor SHA-256 mappings', 'exact generated token ids and decoded text',
                       'exact all-step full-vocabulary logits SHA-256, shape and dtype',
                       'complete prefill and final cache layer shapes/metadata and K/V tensor SHA-256',
                       'optional exported input shapes/dtypes, prompt lengths and runtime parameters'],
            'limitations': ['Tensor hash equality is checked from worker exports; raw logits/cache tensors are not loaded and max_abs_error cannot be recovered from a mismatching hash.',
                           'Declared expected counts are supplied by the caller; this tool does not download a dataset or independently derive its benchmark population.',
                           'Cache size reduction is measured only when prompt_length/input_length/input_shapes.input_ids and dense cache shapes are exported.',
                           'Progress scans compare finite file-size snapshots; incomplete selected doc_id ranges never receive passed status.',
                           'With --limit, later doc_ids and an unfinished trailing line outside the completed selected prefix are not benchmark validation; full_benchmark_validated remains false.'],
            'elapsed_seconds': round(time.time()-started, 3), 'gpu_or_torch_used': False,
        }
        queries = {
            'mismatches': ('SELECT details FROM mismatches ORDER BY id', lambda row: row[0]),
            'issues': ('SELECT side,line,id,kind,details FROM issues ORDER BY side,line',
                       lambda row: canonical(dict(zip(('side','line','id','kind','details'), (*row[:4], json.loads(row[4])))))),
            'missing_in_candidate': ("SELECT a.id FROM records a LEFT JOIN records b ON a.id=b.id AND b.side='candidate' WHERE a.side='reference' AND b.id IS NULL ORDER BY a.id", lambda row: canonical(row[0])),
            'missing_in_reference': ("SELECT a.id FROM records a LEFT JOIN records b ON a.id=b.id AND b.side='reference' WHERE a.side='candidate' AND b.id IS NULL ORDER BY a.id", lambda row: canonical(row[0])),
            'reference_missing_expected_doc_ids': ("SELECT e.doc FROM expected_docs e WHERE NOT EXISTS(SELECT 1 FROM records r WHERE r.side='reference' AND r.valid=1 AND CAST(r.doc AS INTEGER)=e.doc) ORDER BY e.doc", lambda row: canonical(row[0])),
            'candidate_missing_expected_doc_ids': ("SELECT e.doc FROM expected_docs e WHERE NOT EXISTS(SELECT 1 FROM records r WHERE r.side='candidate' AND r.valid=1 AND CAST(r.doc AS INTEGER)=e.doc) ORDER BY e.doc", lambda row: canonical(row[0])),
        }
        stream_json_report(args.output, summary, queries, db)
        db.close()
    print(json.dumps({'status': status, 'complete_expected_run': summary['complete_expected_run'],
                      'expected_count': target_count, 'counts': summary['counts'], 'output': str(args.output.resolve())}, indent=2))
    if status == 'failed':
        return 1
    if status == 'incomplete' and not args.progress:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

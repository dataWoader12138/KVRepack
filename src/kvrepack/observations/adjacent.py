"""Observe spatial visual and natural-text adjacent K/V pairs in cached videos.

Example: python -m kvrepack.observations.adjacent --pretrained MODEL_PATH \
  --hf-home CACHE_PATH --output outputs/observations/qwen7b_first200

This uses the native uncompressed multimodal backbone without an LM head,
generation, compression monkeypatch, or external scoring service.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

TASKS = ('nextqa_oe_test', 'vatex_test')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def read_completed(path, tasks, limit):
    records = {}
    if not Path(path).exists():
        return records
    with Path(path).open(encoding='utf-8') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.endswith('\n') or not line.strip():
                raise ValueError(f'Partial/empty result at line {lineno}; inspect before resuming')
            row = json.loads(line)
            key = (row.get('task'), row.get('doc_id'))
            if key[0] not in tasks or type(key[1]) is not int or not 0 <= key[1] < limit:
                raise ValueError(f'Result outside declared task/first-row range: {key}')
            if row.get('status') != 'completed' or key in records:
                raise ValueError(f'Incomplete or duplicate result: {key}')
            records[key] = row
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--hf-home', type=Path, default=Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser())
    parser.add_argument('--task', choices=('both', *TASKS), default='both')
    parser.add_argument('--limit', type=int, default=200, help='First N original Arrow rows, without video deduplication')
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--max-pairs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-pixels', type=int, default=256 * 28 * 28)
    parser.add_argument('--max-pixels', type=int, default=1605632)
    parser.add_argument('--device', default='0', help='One CUDA device index or UUID')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--nextqa-arrow', type=Path)
    parser.add_argument('--vatex-arrow', type=Path)
    parser.add_argument('--nextqa-video-root', type=Path)
    parser.add_argument('--vatex-video-root', type=Path)
    parser.add_argument('--store-pair-values', action='store_true')
    parser.add_argument('--summarize', action='store_true', help='Generate the strict two-task first-200 report after all rows complete')
    args = parser.parse_args(argv)
    if min(args.limit, args.frames, args.max_pairs, args.min_pixels, args.max_pixels) < 1:
        parser.error('Counts and pixel bounds must be positive')
    if args.min_pixels > args.max_pixels:
        parser.error('min-pixels exceeds max-pixels')
    if args.summarize and (args.task != 'both' or args.limit != 200):
        parser.error('--summarize requires --task both --limit 200')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output is not empty; use a fresh directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.device), HF_HOME=str(args.hf_home.expanduser().resolve()),
                      PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    if args.offline:
        os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as hf_modeling
    from .inputs import load_task, prepare_sample
    from .pairs import build_pair_plan
    from .hooks import AdjacentObserver

    if not torch.cuda.is_available():
        raise RuntimeError('This runner requires a CUDA GPU; use pairs/report tests for CPU verification')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hf_modeling.Qwen2_5_VLFlashAttention2.forward.__module__ != hf_modeling.__name__:
        raise RuntimeError('Attention is already monkeypatched; run observation in a fresh native-model process')
    tasks = list(TASKS) if args.task == 'both' else [args.task]
    data = {}
    for task in tasks:
        prefix = 'nextqa' if task == 'nextqa_oe_test' else 'vatex'
        data[task] = load_task(args.hf_home, task, getattr(args, prefix + '_arrow'),
                               getattr(args, prefix + '_video_root'))
        if args.limit > len(data[task]['dataset']):
            raise ValueError(f'{task} has fewer than {args.limit} rows')
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'flash-attn', 'numpy', 'datasets', 'qwen-vl-utils')}
    sources = {str(path): sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
    sources[str(Path(hf_modeling.__file__))] = sha(hf_modeling.__file__)
    model_location = Path(args.pretrained).expanduser()
    model_metadata = {}
    if model_location.is_dir():
        for path in sorted(model_location.iterdir()):
            if path.is_file() and path.suffix in {'.json', '.txt'}:
                model_metadata[path.name] = sha(path)
    contract = {
        'schema_version': 1, 'experiment': 'adjacent_visual_text_cosine',
        'pretrained': str(model_location.resolve()) if model_location.is_dir() else args.pretrained,
        'model_metadata_sha256': model_metadata, 'tasks': tasks, 'limit': args.limit,
        'row_policy': 'First N original test rows; repeated videos retained',
        'frames': args.frames, 'max_pairs': args.max_pairs, 'seed': args.seed,
        'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels,
        'dtype': 'bfloat16', 'statistics_dtype': 'float32', 'attention': 'flash_attention_2',
        'compression': False, 'backbone_only': True, 'use_cache': False, 'generation': False,
        'pair_selection': 'Outcome-independent nearest-neighbor subsampling; same locations for every layer/KV head',
        'primary_text_groups': {'nextqa_oe_test': 'text_question', 'vatex_test': 'text_instruction'},
        'store_pair_values': args.store_pair_values, 'versions': versions, 'sources_sha256': sources,
        'datasets': {task: {key: value for key, value in d.items()
                            if key.endswith('_sha256') or key in {'arrow_path', 'video_root', 'task_kwargs', 'row_policy'}}
                     for task, d in data.items()},
    }
    contract_path = output / 'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Code, inputs, settings or dependencies changed; refusing mixed-protocol resume')
    elif args.resume and (output / 'records.jsonl').exists():
        raise ValueError('Existing records have no contract')
    else:
        atomic_json(contract_path, contract)
    completed = read_completed(output / 'records.jsonl', tasks, args.limit)
    state = {'schema_version': 1, 'status': 'loading', 'pid': os.getpid(), 'started_utc': timestamp(),
             'updated_utc': timestamp(), 'completed': len(completed), 'target': len(tasks) * args.limit,
             'gpu': torch.cuda.get_device_name(0), 'versions': versions}
    atomic_json(output / 'run_state.json', state)
    start = time.monotonic()
    try:
        processor = AutoProcessor.from_pretrained(args.pretrained, min_pixels=args.min_pixels,
                                                  max_pixels=args.max_pixels, local_files_only=args.offline)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.pretrained, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
            device_map='cuda:0', local_files_only=args.offline).eval()
        layers = model.model.language_model.layers
        heads = int(layers[0].self_attn.num_key_value_heads)
        state.update(status='running', model_load_seconds=time.monotonic()-start,
                     expected_layers=len(layers), expected_kv_heads=heads,
                     model_hidden_size=int(model.model.language_model.config.hidden_size))
        atomic_json(output / 'run_state.json', state)
        print(json.dumps({'loaded': args.pretrained, 'layers': len(layers), 'kv_heads': heads,
                          'seconds': state['model_load_seconds']}, ensure_ascii=False), flush=True)
        for task in tasks:
            for doc_id in range(args.limit):
                key = (task, doc_id)
                if key in completed:
                    continue
                state.update(current_task=task, current_doc_id=doc_id, updated_utc=timestamp())
                atomic_json(output / 'run_state.json', state)
                before = time.monotonic()
                batch, metadata = prepare_sample(data[task], doc_id, processor, args.frames,
                                                   args.min_pixels, args.max_pixels)
                primary = contract['primary_text_groups'][task]
                ids = batch['input_ids'][0].tolist()
                row_seed = args.seed + doc_id + (0 if task == 'nextqa_oe_test' else 1000000)
                plan = build_pair_plan(ids, metadata['video_grid_thw'],
                                       int(model.config.vision_config.spatial_merge_size),
                                       int(model.config.video_token_id), metadata['text_segments'],
                                       row_seed, args.max_pairs, primary_text_group=primary)
                plan['input_length'] = len(ids)
                preprocess_seconds = time.monotonic()-before
                input_hashes = {name: hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                for name, value in batch.items() if torch.is_tensor(value)}
                gpu_batch = {name: value.to('cuda:0') if torch.is_tensor(value) else value for name, value in batch.items()}
                torch.cuda.reset_peak_memory_stats()
                forward_start = time.monotonic()
                with torch.inference_mode(), AdjacentObserver(model, plan, include_pair_values=args.store_pair_values) as observer:
                    result = model.model(**gpu_batch, use_cache=False, return_dict=True,
                                         output_attentions=False, output_hidden_states=False)
                    torch.cuda.synchronize()
                    layer_stats = observer.finish()
                forward_seconds = time.monotonic()-forward_start
                row = {'schema_version': 1, 'status': 'completed', **metadata,
                       'pair_counts': {group: group_data['selected_pair_count'] for group, group_data in plan['groups'].items()},
                       'pair_plan': plan, 'eligibility': plan['coverage'], 'input_hashes': input_hashes,
                       'layer_stats': layer_stats, 'timing_seconds': {'preprocess': preprocess_seconds, 'forward_with_observation': forward_seconds,
                                                                  'total': time.monotonic()-before},
                       'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30, 'completed_utc': timestamp()}
                append_json(output / 'records.jsonl', row)
                completed[key] = row
                state.update(completed=len(completed), counts_by_task=dict(Counter(k[0] for k in completed)), updated_utc=timestamp(),
                             elapsed_seconds=time.monotonic()-start, last_seconds=row['timing_seconds'])
                atomic_json(output / 'run_state.json', state)
                print(json.dumps({'completed': len(completed), 'target': state['target'], 'task': task, 'doc_id': doc_id,
                                  'pairs': plan['matched_pair_count'], 'tokens': len(ids),
                                  'seconds': round(row['timing_seconds']['total'], 3), 'peak_gib': round(row['peak_allocated_gib'], 3)},
                                 ensure_ascii=False), flush=True)
                del result, gpu_batch, batch
                torch.cuda.empty_cache()
        if len(completed) != state['target']:
            raise RuntimeError('Completed record count differs from required first-row coverage')
        del model
        torch.cuda.empty_cache()
        if args.summarize:
            state.update(status='summarizing', updated_utc=timestamp())
            atomic_json(output / 'run_state.json', state)
            from .report import main as report_main
            code = report_main(['--input', str(output/'records.jsonl'), '--output', str(output/'report'),
                                '--expected-layers', str(state['expected_layers']), '--expected-kv-heads', str(heads)])
            if code:
                raise RuntimeError(f'Report validation returned {code}')
        state.update(status='complete', updated_utc=timestamp(), finished_utc=timestamp(), elapsed_seconds=time.monotonic()-start)
        atomic_json(output / 'run_state.json', state)
        return 0
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', updated_utc=timestamp(),
                     elapsed_seconds=time.monotonic()-start)
        atomic_json(output / 'run_state.json', state)
        append_json(output / 'errors.jsonl', {'time_utc': timestamp(), 'error': state['error'],
                                             'task': state.get('current_task'), 'doc_id': state.get('current_doc_id')})
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

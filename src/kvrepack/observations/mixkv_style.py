"""MixKV-style all-token cosine distributions and video-context controls.

An adaptation to the existing Qwen2.5-VL/video experiment, not a replication of
MixKV's Qwen2/LongBench versus Qwen2-VL/OCRBench model comparison. No compression,
generation, token replacement, or result-dependent pool selection is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback

from .adjacent import TASKS, append_json, atomic_json, read_completed, sha, timestamp
from .inputs import _align_segments, _expand_video_text, _natural_spans, load_task, prepare_sample


PRIMARY = {'nextqa_oe_test': 'question', 'vatex_test': 'instruction'}


def matched_token_draws(visual_positions, text_positions, text_only_positions, *, seed, cap=16, draws=32):
    """Prespecify equal-sized pools; sample positions without reading activations.

    Sampling is without replacement *within* a draw. Independent draws may reuse
    positions; they are Monte Carlo measurements of one input, not independent
    benchmark observations. The text-only condition reuses the identical tokens.
    """
    if any(type(x) is not int or x <= 0 for x in (cap, draws)):
        raise ValueError('Matched token cap and draw count must be positive integers')
    for positions in (visual_positions, text_positions, text_only_positions):
        if positions != sorted(set(positions)) or any(type(p) is not int or p < 0 for p in positions):
            raise ValueError('Matched source positions must be sorted, unique, nonnegative integers')
    if len(text_positions) != len(text_only_positions):
        raise ValueError('Context-control text token counts differ')
    size = min(cap, len(visual_positions), len(text_positions))
    if size < 2:
        raise ValueError('Matched all-pair pools require at least two tokens per modality')
    rng = random.Random(seed)
    visual, text = [], []
    mapping = dict(zip(text_positions, text_only_positions))
    for _ in range(draws):
        visual.append(sorted(rng.sample(visual_positions, size)))
        text.append(sorted(rng.sample(text_positions, size)))
    control = [[mapping[p] for p in positions] for positions in text]
    return ({'visual_matched': visual, 'text_matched': text}, {'text_matched': control})


def natural_user_positions(ids, attention, offsets, text, prompt, special_ids):
    """Select the actual complete user-text span, including VATEX text examples."""
    spans = [{'name': 'user', 'text': prompt, 'char_span': [0, len(prompt)]}]
    groups = _align_segments(ids, offsets, attention, text, prompt, spans, special_ids)
    return groups[0]['positions']


def make_pools(ids, attention, segments, user_positions, primary, video_token_id=None):
    if len(ids) != len(attention) or any(value not in (0, 1) for value in attention):
        raise ValueError('Input IDs and binary attention mask must have equal length')
    pools = {'prompt_all': [i for i, valid in enumerate(attention) if valid],
             'text_primary': [p for group in segments if group['name'] == primary for p in group['positions']],
             'natural_user': list(user_positions)}
    if video_token_id is not None:
        pools['visual_all'] = [i for i, token in enumerate(ids) if attention[i] and token == video_token_id]
    for name, positions in pools.items():
        if len(positions) < 2 or positions != sorted(set(positions)):
            raise ValueError(f'{name} must have at least two unique, sorted positions')
        if any(type(i) is not int or not 0 <= i < len(ids) or not attention[i] for i in positions):
            raise ValueError(f'{name} contains invalid/padded positions')
    if not set(pools['text_primary']).issubset(pools['natural_user']):
        raise ValueError('Primary text is not inside the complete user-text pool')
    if set(pools.get('visual_all', [])) & set(pools['natural_user']):
        raise ValueError('Visual and user-text positions overlap')
    return pools


def prepare_text_only(task_data, doc_id, processor, prompt):
    """Remove the video message only; retain original system and full task text."""
    doc = task_data['dataset'][doc_id]
    expected = task_data['_prompt_function'](doc, task_data['task_kwargs'])
    if prompt != expected:
        raise ValueError('Text-only control must preserve the exact original task prompt')
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = dict(processor.tokenizer([text], padding=True, return_tensors='pt', return_offsets_mapping=True))
    offsets = encoded.pop('offset_mapping')[0].tolist()
    ids = encoded['input_ids'][0].tolist()
    attention = encoded['attention_mask'][0].tolist()
    special_ids = set(processor.tokenizer.all_special_ids)
    spans = _natural_spans(task_data['task'], doc, prompt, task_data['task_kwargs'])
    segments = _align_segments(ids, offsets, attention, text, prompt, spans, special_ids)
    user = natural_user_positions(ids, attention, offsets, text, prompt, special_ids)
    pools = make_pools(ids, attention, segments, user, PRIMARY[task_data['task']])
    return encoded, segments, pools


def multimodal_pools(batch, metadata, processor, video_token_id):
    ids = batch['input_ids'][0].tolist()
    attention = batch['attention_mask'][0].tolist()
    # Reconstruct the same chat shell solely for native offsets, independently
    # checking every ID against the actual video processor output.
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': [{'type': 'video', 'video': metadata['video_path']},
                                           {'type': 'text', 'text': metadata['prompt']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expanded = _expand_video_text(text, processor, metadata['video_grid_thw'])
    encoded = processor.tokenizer([expanded], padding=True, return_tensors='pt', return_offsets_mapping=True)
    if encoded['input_ids'][0].tolist() != ids or encoded['attention_mask'][0].tolist() != attention:
        raise ValueError('All-pair pool re-tokenization differs from original benchmark input')
    user = natural_user_positions(ids, attention, encoded['offset_mapping'][0].tolist(), expanded,
                                  metadata['prompt'], set(processor.tokenizer.all_special_ids))
    return make_pools(ids, attention, metadata['text_segments'], user, PRIMARY[metadata['task']], video_token_id)


def compare_reference(metadata, hashes, stats, reference, tolerance=1e-6):
    """Check exact original inputs and old adjacent statistics in this new forward."""
    for key in ('original_row_sha256', 'expanded_text_sha256', 'frame_sampling', 'video_grid_thw'):
        if metadata[key] != reference[key]:
            raise ValueError(f'Original adjacent input metadata changed: {key}')
    if hashes != reference['input_hashes']:
        raise ValueError('Actual input tensor bytes differ from original adjacent experiment')
    if len(stats) != len(reference['layer_stats']):
        raise ValueError('Layer count differs from adjacent reference')
    largest = 0.0
    for current, previous in zip(stats, reference['layer_stats']):
        if current['layer'] != previous['layer'] or current['kv_heads'] != previous['kv_heads']:
            raise ValueError('Layer/head identity differs from adjacent reference')
        if set(current['groups']) != set(previous['groups']):
            raise ValueError('Adjacent group coverage changed')
        for group, metrics in previous['groups'].items():
            for metric in ('k_raw', 'v', 'k_rope'):
                left, right = current['groups'][group][metric], metrics[metric]
                if len(left) != len(right):
                    raise ValueError('Adjacent reference KV-head count differs')
                largest = max(largest, *(abs(a-b) for a, b in zip(left, right)))
    if largest > tolerance:
        raise ValueError(f'Adjacent statistics changed beyond tolerance: {largest} > {tolerance}')
    return {'input_tensor_bytes_identical': True, 'adjacent_max_absolute_difference': largest,
            'adjacent_tolerance': tolerance}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--min-pixels', type=int, default=200704)
    parser.add_argument('--max-pixels', type=int, default=1605632)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--matched-token-cap', type=int, default=16)
    parser.add_argument('--matched-draws', type=int, default=32)
    parser.add_argument('--device', default='0')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--reference-adjacent', type=Path,
                        help='Original records.jsonl; checks exact inputs and adjacent statistics')
    args = parser.parse_args(argv)
    if min(args.limit, args.frames, args.min_pixels, args.max_pixels, args.matched_draws) < 1 or args.matched_token_cap < 2 or args.min_pixels > args.max_pixels:
        parser.error('Counts/pixel bounds must be positive and min-pixels <= max-pixels')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Use a fresh output directory or explicit --resume')
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
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as native
    from .all_pairs import AllPairsObserver
    from .pairs import build_pair_plan

    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires a CUDA GPU')
    if native.Qwen2_5_VLFlashAttention2.forward.__module__ != native.__name__:
        raise RuntimeError('Run in a fresh process without compression monkeypatches')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = {task: load_task(args.hf_home, task) for task in TASKS}
    if any(args.limit > len(d['dataset']) for d in data.values()):
        raise ValueError('Requested original row range exceeds dataset')
    reference = {}
    if args.reference_adjacent:
        reference = read_completed(args.reference_adjacent, TASKS, 200)
        if any((task, i) not in reference for task in TASKS for i in range(args.limit)):
            raise ValueError('Adjacent reference is missing a requested original row')
    sources = {str(path): sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
    sources[str(Path(native.__file__))] = sha(native.__file__)
    model_path = Path(args.pretrained).expanduser()
    contract = {'schema_version': 1, 'experiment': 'mixkv_style_video_all_pairs',
                'literature': {'paper': 'https://arxiv.org/abs/2510.20707', 'figure': '1(b)',
                               'adaptation': 'Same Qwen2.5-VL model; video datasets; short original task text. '
                                             'Not the original Qwen2/LongBench vs Qwen2-VL/OCRBench comparison.'},
                'pretrained': str(model_path.resolve()) if model_path.is_dir() else args.pretrained,
                'model_metadata_sha256': {p.name: sha(p) for p in sorted(model_path.glob('*'))
                                          if p.is_file() and p.suffix in {'.json', '.txt'}},
                'tasks': list(TASKS), 'limit': args.limit, 'frames': args.frames, 'seed': args.seed,
                'primary_comparison': 'Equal token counts in uniform within-input draws: visual versus primary text',
                'matched_token_cap': args.matched_token_cap, 'matched_draws': args.matched_draws,
                'matched_selection': 'T=min(cap,visual count,primary text count); draw uniformly without replacement '
                                     'within each modality, repeated R times; draw positions may recur across repeats. '
                                     'Same primary text token IDs/draws in the text-only condition.',
                'length_scope': 'Measured token counts and pair counts are matched. Full causal input context '
                                'lengths, semantic roles, and spatial/linguistic distances are not matched.',
                'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels,
                'row_policy': 'First original rows, repeated videos retained; equal row weighting',
                'conditions': ['multimodal', 'text_only'], 'compression': False, 'use_cache': False,
                'generation': False, 'dtype': 'bfloat16', 'attention': 'flash_attention_2',
                'statistics': 'All off-diagonal cosine pairs within each predefined token pool; '
                              'FP32 normalization and float64 accumulation; no dense T*T matrix',
                'distribution_unit': 'One value per layer and native KV head, averaged over original rows',
                'sources_sha256': sources,
                'versions': {name: importlib.metadata.version(name) for name in
                             ('torch', 'transformers', 'flash-attn', 'numpy', 'datasets', 'qwen-vl-utils')},
                'datasets': {task: {k: v for k, v in d.items()
                                    if k.endswith('_sha256') or k in {'arrow_path', 'video_root', 'task_kwargs'}}
                             for task, d in data.items()},
                'reference_adjacent_sha256': sha(args.reference_adjacent) if args.reference_adjacent else None}
    contract_path = output / 'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Protocol/code/dependencies changed; refusing mixed resume')
    elif args.resume and (output / 'records.jsonl').exists():
        raise ValueError('Records without contract cannot be resumed')
    else:
        atomic_json(contract_path, contract)
    completed = read_completed(output / 'records.jsonl', TASKS, args.limit)
    started = time.monotonic()
    state = {'schema_version': 1, 'status': 'loading', 'pid': os.getpid(), 'started_utc': timestamp(),
             'updated_utc': timestamp(), 'completed': len(completed), 'target': 2*args.limit,
             'gpu': torch.cuda.get_device_name(0)}
    atomic_json(output / 'run_state.json', state)
    try:
        processor = AutoProcessor.from_pretrained(args.pretrained, min_pixels=args.min_pixels,
                                                  max_pixels=args.max_pixels, local_files_only=args.offline)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.pretrained, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
            device_map='cuda:0', local_files_only=args.offline).eval()
        layers = model.model.language_model.layers
        state.update(status='running', expected_layers=len(layers),
                     expected_kv_heads=layers[0].self_attn.num_key_value_heads,
                     model_load_seconds=time.monotonic()-started)
        atomic_json(output / 'run_state.json', state)
        print(json.dumps({'loaded': args.pretrained, 'seconds': state['model_load_seconds']}), flush=True)
        for task in TASKS:
            for doc_id in range(args.limit):
                if (task, doc_id) in completed:
                    continue
                state.update(current_task=task, current_doc_id=doc_id, updated_utc=timestamp())
                atomic_json(output / 'run_state.json', state)
                before = time.monotonic()
                batch, metadata = prepare_sample(data[task], doc_id, processor, args.frames,
                                                   args.min_pixels, args.max_pixels)
                ids = batch['input_ids'][0].tolist()
                multimodal = multimodal_pools(batch, metadata, processor, int(model.config.video_token_id))
                row_seed = args.seed + doc_id + (0 if task == TASKS[0] else 1000000)
                plan = build_pair_plan(ids, metadata['video_grid_thw'],
                                       int(model.config.vision_config.spatial_merge_size),
                                       int(model.config.video_token_id), metadata['text_segments'],
                                       row_seed, 16, primary_text_group='text_' + PRIMARY[task])
                plan['input_length'] = len(ids)
                text_batch, text_segments, text_pools = prepare_text_only(data[task], doc_id, processor, metadata['prompt'])
                text_ids = text_batch['input_ids'][0].tolist()
                if [ids[i] for i in multimodal['text_primary']] != [text_ids[i] for i in text_pools['text_primary']]:
                    raise ValueError('Primary text tokens differ between contexts; paired control is not aligned')
                if [ids[i] for i in multimodal['natural_user']] != [text_ids[i] for i in text_pools['natural_user']]:
                    raise ValueError('Complete user text tokens differ between contexts')
                matched_multi, matched_text = matched_token_draws(
                    multimodal['visual_all'], multimodal['text_primary'], text_pools['text_primary'],
                    seed=row_seed + 2000000, cap=args.matched_token_cap, draws=args.matched_draws)
                preprocess_seconds = time.monotonic()-before
                conditions = {}
                torch.cuda.reset_peak_memory_stats()
                for condition, cpu_batch, pools, matched, pair_plan, segments in (
                    ('multimodal', batch, multimodal, matched_multi, plan, metadata['text_segments']),
                    ('text_only', text_batch, text_pools, matched_text, {'input_length': len(text_ids), 'groups': {}}, text_segments)):
                    hashes = {name: hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                              for name, value in cpu_batch.items() if torch.is_tensor(value)}
                    gpu_batch = {name: value.to('cuda:0') if torch.is_tensor(value) else value
                                 for name, value in cpu_batch.items()}
                    forward_started = time.monotonic()
                    with torch.inference_mode(), AllPairsObserver(model, pair_plan, pools, matched_pools=matched) as observer:
                        result = model.model(**gpu_batch, use_cache=False, return_dict=True,
                                             output_attentions=False, output_hidden_states=False)
                        torch.cuda.synchronize()
                        stats = observer.finish()
                    conditions[condition] = {'input_tokens': len(cpu_batch['input_ids'][0]), 'pools': pools, 'matched_pools': matched,
                                             'text_segments': segments, 'input_hashes': hashes, 'layer_stats': stats,
                                             'forward_seconds': time.monotonic()-forward_started}
                    if condition == 'multimodal' and reference:
                        conditions[condition]['adjacent_reference'] = compare_reference(
                            metadata, hashes, stats, reference[(task, doc_id)])
                    del result, gpu_batch
                    torch.cuda.empty_cache()
                row = {'schema_version': 1, 'status': 'completed', **metadata, 'conditions': conditions,
                       'pair_plan': plan, 'timing_seconds': {'preprocess': preprocess_seconds,
                                                            'total': time.monotonic()-before},
                       'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30, 'completed_utc': timestamp()}
                append_json(output / 'records.jsonl', row)
                completed[(task, doc_id)] = row
                state.update(completed=len(completed), counts_by_task=dict(Counter(key[0] for key in completed)),
                             elapsed_seconds=time.monotonic()-started, updated_utc=timestamp(),
                             last_seconds=row['timing_seconds'])
                atomic_json(output / 'run_state.json', state)
                print(json.dumps({'completed': len(completed), 'target': state['target'], 'task': task,
                                  'doc_id': doc_id, 'tokens': {k: v['input_tokens'] for k, v in conditions.items()},
                                  'seconds': round(row['timing_seconds']['total'], 3)}, ensure_ascii=False), flush=True)
                del batch, text_batch
        if len(completed) != state['target']:
            raise RuntimeError('Incomplete original-row coverage')
        del model
        torch.cuda.empty_cache()
        if args.summarize:
            state.update(status='summarizing', updated_utc=timestamp())
            atomic_json(output / 'run_state.json', state)
            from .all_pairs_report import main as report_main
            result = report_main(['--input', str(output / 'records.jsonl'), '--output', str(output / 'report'),
                                  '--expected-limit', str(args.limit), '--expected-layers', str(state['expected_layers']),
                                  '--expected-kv-heads', str(state['expected_kv_heads']),
                                  '--matched-draws', str(args.matched_draws),
                                  '--matched-token-cap', str(args.matched_token_cap)])
            if result:
                raise RuntimeError(f'All-pair report returned {result}')
        state.update(status='complete', finished_utc=timestamp(), updated_utc=timestamp(),
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(output / 'run_state.json', state)
        return 0
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', updated_utc=timestamp(),
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(output / 'run_state.json', state)
        append_json(output / 'errors.jsonl', {'time_utc': timestamp(), 'error': state['error'],
                                             'task': state.get('current_task'), 'doc_id': state.get('current_doc_id')})
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Five-input all-token pilot inspired by MixKV Figure 1(b), using Qwen2.5.

Each invocation runs one group of exactly five complete inputs. There is no
token sampling, generation, compression, truncation, or automatic scale-up.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from .adjacent import append_json, atomic_json, sha, timestamp

MODELS = {
    'text': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28', 'qwen2'),
    'vision': ('Qwen/Qwen2.5-VL-7B-Instruct', 'cc594898137f460bfe9f0759e9844b3ce807cfb5', 'qwen2_5_vl'),
}
IDS = {'text': [0, 1, 2, 3, 4], 'vision': [0, 200, 400, 600, 800]}
SYSTEM = 'You are a helpful assistant.'


def prepare_input(sample, tokenizer, *, processor=None, min_pixels=25088, max_pixels=12845056):
    """Use complete original benchmark prompts/images with native chat framing."""
    if sample['group'] == 'text':
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': sample['prompt']}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = dict(tokenizer([chat], return_tensors='pt', padding=False))
        info = {'chat_sha256': hashlib.sha256(chat.encode()).hexdigest(), 'image_tokens': 0,
                'image_grid_thw': None, 'original_image_dimensions': None}
    else:
        if processor is None:
            raise ValueError('Vision inputs require the native processor')
        from qwen_vl_utils import process_vision_info
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [
            {'type': 'image', 'image': sample['image_path'], 'min_pixels': min_pixels, 'max_pixels': max_pixels},
            {'type': 'text', 'text': sample['prompt']}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        if not images or len(images) != 1 or videos:
            raise ValueError('Require exactly one complete image and no video')
        batch = dict(processor(text=[chat], images=images, return_tensors='pt', padding=False))
        image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
        image_tokens = int((batch['input_ids'] == image_token_id).sum().item())
        grid = batch['image_grid_thw'].tolist()
        merge = int(processor.image_processor.merge_size)
        if len(grid) != 1 or image_tokens != grid[0][0]*grid[0][1]*grid[0][2]//(merge*merge):
            raise ValueError('Expanded image token count does not match the native grid')
        info = {'chat_sha256': hashlib.sha256(chat.encode()).hexdigest(), 'image_tokens': image_tokens,
                'image_grid_thw': grid, 'processed_image_dimensions': list(images[0].size)}
    if tuple(batch['input_ids'].shape[:1]) != (1,) or not bool((batch['attention_mask'] == 1).all().item()):
        raise ValueError('Require one complete unpadded prompt')
    info['input_tokens'] = int(batch['input_ids'].shape[1])
    if info['input_tokens'] < 2:
        raise ValueError('Non-self all-pair cosine requires at least two tokens')
    return batch, info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=tuple(MODELS))
    parser.add_argument('--pretrained', required=True, type=Path, help='Pinned local snapshot of the selected model')
    parser.add_argument('--hf-home', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path, help='reference_samples.json prepared once, before model runs')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='0')
    parser.add_argument('--min-pixels', type=int, default=25088)
    parser.add_argument('--max-pixels', type=int, default=12845056)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.min_pixels <= args.max_pixels:
        parser.error('Require positive, ordered pixel bounds')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Use a fresh output directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.device, HF_HOME=str(args.hf_home.expanduser().resolve()),
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoTokenizer, AutoProcessor, Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration
    from .native_all_pairs import NativeAllPairsObserver

    if not torch.cuda.is_available():
        raise RuntimeError('Use a CUDA GPU for the model pilot; CPU tests do not load model weights')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(args.inputs.read_text())
    all_samples = manifest['samples'] if isinstance(manifest, dict) else manifest
    samples = [row for row in all_samples if row['group'] == args.group]
    if [row['doc_id'] for row in samples] != IDS[args.group]:
        raise ValueError('Require exactly the five prespecified complete inputs in original order')
    model_id, revision, family = MODELS[args.group]
    pretrained = args.pretrained.expanduser().resolve()
    if pretrained.name != revision:
        raise ValueError(f'Use the pinned local snapshot directory named {revision}')
    config = json.loads((pretrained / 'config.json').read_text())
    if (config.get('model_type') != family or config.get('num_hidden_layers') != 28
            or config.get('num_key_value_heads') != 4 or config.get('hidden_size') != 3584):
        raise ValueError('Model configuration differs from the selected Qwen2.5 7B architecture')
    model_index = json.loads((pretrained / 'model.safetensors.index.json').read_text())
    if any(not (pretrained/name).is_file() for name in set(model_index['weight_map'].values())):
        raise FileNotFoundError('Model download is incomplete; do not run with missing shards')
    contract = {'schema_version': 1, 'experiment': 'mixkv_reference_qwen25_five_per_group',
                'group': args.group, 'model_id': model_id, 'model_revision': revision,
                'pretrained': str(pretrained), 'doc_ids': IDS[args.group], 'sample_count': 5,
                'inputs_sha256': sha(args.inputs), 'system_prompt': SYSTEM,
                'all_prefill_tokens': True, 'token_sampling': False, 'truncation': False,
                'compression': False, 'generation': False, 'use_cache': False,
                'dtype': 'bfloat16', 'statistics': 'FP32 normalization, FP64 all-off-diagonal reduction',
                'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels,
                'pixel_source_scope': 'MixKV public analysis scripts, not disclosed Figure 1 settings',
                'auto_expand': False, 'input_lengths_matched': False,
                'model_metadata_sha256': {p.name: sha(p) for p in sorted(pretrained.iterdir())
                                          if p.is_file() and p.suffix in {'.json', '.txt'}},
                'sources_sha256': {str(p): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                'versions': {name: importlib.metadata.version(name) for name in
                             ('torch', 'transformers', 'flash-attn', 'numpy', 'qwen-vl-utils')}}
    path = output / 'contract.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Protocol/code/inputs changed; refusing mixed resume')
    else:
        atomic_json(path, contract)
    completed = set()
    if (output / 'records.jsonl').exists():
        for line in (output / 'records.jsonl').open():
            row = json.loads(line)
            if (not line.endswith('\n') or row['status'] != 'completed' or row['group'] != args.group
                    or row['doc_id'] not in IDS[args.group] or row['doc_id'] in completed):
                raise ValueError('Existing results are partial, duplicated or outside the five-input pilot')
            completed.add(row['doc_id'])
    started = time.monotonic()
    state = {'schema_version': 1, 'status': 'loading', 'pid': os.getpid(), 'group': args.group,
             'started_utc': timestamp(), 'updated_utc': timestamp(), 'completed': len(completed), 'target': 5,
             'auto_expand': False, 'gpu': torch.cuda.get_device_name(0)}
    atomic_json(output / 'run_state.json', state)
    try:
        processor = None
        if args.group == 'text':
            tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
            model_class = Qwen2ForCausalLM
        else:
            processor = AutoProcessor.from_pretrained(pretrained, local_files_only=True,
                                                      min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            tokenizer = processor.tokenizer
            model_class = Qwen2_5_VLForConditionalGeneration
        model = model_class.from_pretrained(pretrained, local_files_only=True, torch_dtype=torch.bfloat16,
                                           attn_implementation='flash_attention_2', device_map='cuda:0').eval()
        state.update(status='running', model_load_seconds=time.monotonic()-started)
        atomic_json(output / 'run_state.json', state)
        for sample in samples:
            if sample['doc_id'] in completed:
                continue
            before = time.monotonic()
            state.update(current_doc_id=sample['doc_id'], updated_utc=timestamp())
            atomic_json(output / 'run_state.json', state)
            cpu, info = prepare_input(sample, tokenizer, processor=processor,
                                      min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            if info['input_tokens'] > int(config['max_position_embeddings']):
                raise ValueError('Complete prompt exceeds native context window; no silent truncation/substitution')
            hashes = {name: hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                      for name, tensor in cpu.items() if torch.is_tensor(tensor)}
            batch = {name: value.to('cuda:0') if torch.is_tensor(value) else value for name, value in cpu.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), NativeAllPairsObserver(model, family=family, input_length=info['input_tokens']) as observer:
                result = model.model(**batch, use_cache=False, output_attentions=False, output_hidden_states=False, return_dict=True)
                torch.cuda.synchronize()
                stats = observer.finish()
            row = {'schema_version': 1, 'status': 'completed', 'group': args.group, 'doc_id': sample['doc_id'],
                   'model_id': model_id, 'model_revision': revision, **info,
                   'source_metadata': sample['source_metadata'], 'input_hashes': hashes, 'layer_stats': stats,
                   'seconds': time.monotonic()-before, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'completed_utc': timestamp()}
            append_json(output / 'records.jsonl', row)
            completed.add(sample['doc_id'])
            state.update(completed=len(completed), updated_utc=timestamp(), elapsed_seconds=time.monotonic()-started)
            atomic_json(output / 'run_state.json', state)
            print(json.dumps({'group': args.group, 'doc_id': sample['doc_id'], 'completed': len(completed),
                              'target': 5, 'tokens': info['input_tokens'], 'image_tokens': info['image_tokens'],
                              'seconds': row['seconds']}, ensure_ascii=False), flush=True)
            del result, batch, cpu
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        state.update(status='complete', finished_utc=timestamp(), updated_utc=timestamp(), elapsed_seconds=time.monotonic()-started)
        atomic_json(output / 'run_state.json', state)
        return 0
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}', updated_utc=timestamp(), elapsed_seconds=time.monotonic()-started)
        atomic_json(output / 'run_state.json', state)
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())

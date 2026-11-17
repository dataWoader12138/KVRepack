"""Full-sequence adjacent-token comparison on the fixed Qwen2.5 reference inputs."""
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
from .native_adjacent import NativeAdjacentObserver, STATISTICS_KIND
from .reference import MODELS, SYSTEM, prepare_input


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=tuple(MODELS))
    parser.add_argument('--sample-count', type=int, choices=(2, 200), default=200)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference-records', type=Path,
                        help='Prior all-pairs records: require identical full processor inputs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--min-pixels', type=int, default=25088)
    parser.add_argument('--max-pixels', type=int, default=12845056)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.min_pixels <= args.max_pixels:
        parser.error('Require positive ordered pixel bounds')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Use a fresh output directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.device, HF_HOME=str(args.hf_home.resolve()),
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    import torch
    from transformers import AutoTokenizer, AutoProcessor, Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the model observation')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(args.inputs.read_text())
    samples = [row for row in manifest['samples'] if row['group'] == args.group]
    all_ids = list(range(200)) if args.group == 'text' else list(range(0, 1000, 5))
    if [row['doc_id'] for row in samples] != all_ids or manifest['sample_count'] != 400:
        raise ValueError('Reuse exactly the original prescribed 200+200 input manifest')
    samples = samples[:args.sample_count]
    doc_ids = [row['doc_id'] for row in samples]
    reference = {}
    if args.reference_records:
        for line in args.reference_records.open():
            row = json.loads(line)
            if row['group'] != args.group or row['doc_id'] in reference or row['status'] != 'completed':
                raise ValueError('Invalid original reference corpus')
            reference[row['doc_id']] = row
        if set(reference) != set(all_ids):
            raise ValueError('Original all-pairs reference must contain exactly the prescribed 200 rows')
    model_id, revision, family = MODELS[args.group]
    pretrained = args.pretrained.expanduser().resolve()
    config = json.loads((pretrained/'config.json').read_text())
    if (pretrained.name != revision or config.get('model_type') != family
            or config.get('num_hidden_layers') != 28 or config.get('num_key_value_heads') != 4
            or config.get('hidden_size') != 3584):
        raise ValueError('Require the original pinned Qwen2.5 7B snapshot')
    index = json.loads((pretrained/'model.safetensors.index.json').read_text())
    if any(not (pretrained/name).is_file() for name in set(index['weight_map'].values())):
        raise FileNotFoundError('Model download is incomplete')
    source_names = ('reference_adjacent.py', 'native_adjacent.py', 'native_all_pairs.py',
                    'reference.py', 'reference_inputs.py', 'adjacent.py')
    contract = dict(schema_version=1, statistics_kind=STATISTICS_KIND,
                    group=args.group, model_id=model_id, model_revision=revision,
                    pretrained=str(pretrained), doc_ids=doc_ids, sample_count=args.sample_count,
                    inputs_sha256=sha(args.inputs), system_prompt=SYSTEM,
                    all_prefill_tokens=True, token_sampling=False, truncation=False,
                    generation=False, compression=False, use_cache=False, dtype='bfloat16',
                    statistics='Every sequence pair (i,i+1), FP32 normalization, FP64 products and reductions',
                    includes_structural_and_modality_boundary_pairs=True,
                    aggregation='Equal inputs, equal layers and native KV heads; no token-length weighting',
                    min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                    reference_records_sha256=sha(args.reference_records) if args.reference_records else None,
                    model_metadata_sha256={p.name: sha(p) for p in sorted(pretrained.iterdir())
                                           if p.is_file() and p.suffix in {'.json', '.txt'}},
                    sources_sha256={str(Path(__file__).parent/name): sha(Path(__file__).parent/name)
                                    for name in source_names},
                    versions={name: importlib.metadata.version(name) for name in
                              ('torch', 'transformers', 'flash-attn', 'numpy', 'qwen-vl-utils')},
                    auto_expand=False)
    path = output/'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Protocol/inputs/source changed; refusing mixed resume')
    atomic_json(path, contract)
    completed = set()
    if (output/'records.jsonl').exists():
        for line in (output/'records.jsonl').open():
            row = json.loads(line)
            if (not line.endswith('\n') or row['status'] != 'completed' or row['group'] != args.group
                    or row['statistics_kind'] != STATISTICS_KIND or row['doc_id'] not in doc_ids
                    or row['doc_id'] in completed or row['contract_sha256'] != sha(path)):
                raise ValueError('Partial, duplicate or incompatible existing records')
            completed.add(row['doc_id'])
    started = time.monotonic()
    state = dict(status='loading', pid=os.getpid(), group=args.group, started_utc=timestamp(),
                 completed=len(completed), target=args.sample_count, gpu=torch.cuda.get_device_name(0),
                 statistics_kind=STATISTICS_KIND, auto_expand=False)
    atomic_json(output/'run_state.json', state)
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
        atomic_json(output/'run_state.json', state)
        for sample in samples:
            if sample['doc_id'] in completed:
                continue
            before = time.monotonic()
            cpu, info = prepare_input(sample, tokenizer, processor=processor,
                                      min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            if info['input_tokens'] > int(config['max_position_embeddings']):
                raise ValueError('Native context exceeded; no truncation or input substitution')
            hashes = {name: hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                      for name, tensor in cpu.items() if torch.is_tensor(tensor)}
            if reference:
                old = reference[sample['doc_id']]
                if (hashes != old['input_hashes'] or info['input_tokens'] != old['input_tokens']
                        or info['chat_sha256'] != old['chat_sha256']
                        or sample['source_metadata'] != old['source_metadata']):
                    raise ValueError('Input differs from the original full-sequence all-pairs experiment')
            batch = {name: value.to('cuda:0') if torch.is_tensor(value) else value for name, value in cpu.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), NativeAdjacentObserver(model, family=family, input_length=info['input_tokens']) as observer:
                result = model.model(**batch, use_cache=False, output_attentions=False,
                                     output_hidden_states=False, return_dict=True)
                torch.cuda.synchronize()
                stats = observer.finish()
            row = dict(schema_version=1, status='completed', statistics_kind=STATISTICS_KIND,
                       contract_sha256=sha(path), group=args.group, doc_id=sample['doc_id'],
                       model_id=model_id, model_revision=revision, **info,
                       adjacent_pair_count=info['input_tokens']-1,
                       source_metadata=sample['source_metadata'], input_hashes=hashes,
                       original_input_match=bool(reference), layer_stats=stats,
                       seconds=time.monotonic()-before,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, completed_utc=timestamp())
            append_json(output/'records.jsonl', row)
            completed.add(sample['doc_id'])
            state.update(completed=len(completed), current_doc_id=sample['doc_id'],
                         updated_utc=timestamp(), elapsed_seconds=time.monotonic()-started)
            atomic_json(output/'run_state.json', state)
            print(json.dumps(dict(group=args.group, doc_id=sample['doc_id'], completed=len(completed),
                                  target=args.sample_count, tokens=info['input_tokens'],
                                  adjacent_pairs=row['adjacent_pair_count'], seconds=row['seconds'])), flush=True)
            del result, batch, cpu
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        state.update(status='complete', finished_utc=timestamp(), elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json', state)
        return 0
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json', state)
        traceback.print_exc()
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())

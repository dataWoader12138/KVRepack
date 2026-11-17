"""Compare per-input top200 raw-K adjacent pairs: LongBench and VATEX."""
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
from .reference import MODELS, prepare_input

KIND = 'raw_k_top200_sequence_adjacent_v1'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=('text', 'video'), required=True)
    parser.add_argument('--sample-count', type=int, choices=(2, 200), default=200)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, help='Original LongBench/OCRBench400 manifest; use only text rows')
    parser.add_argument('--reference-records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Use a fresh output directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.device, HF_HOME=str(args.hf_home.resolve()),
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoProcessor, Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration
    from .native_top_adjacent import RawKTopAdjacentObserver
    from .inputs import load_task, prepare_sample

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for model observation')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = {}
    for line in args.reference_records.open():
        row = json.loads(line)
        if args.group == 'video' and row.get('task') != 'vatex_test':
            continue
        if args.group == 'text' and row.get('group') != 'text':
            continue
        if row['status'] != 'completed' or row['doc_id'] in reference:
            raise ValueError('Invalid or duplicate original reference record')
        reference[row['doc_id']] = row
    if set(reference) != set(range(200)):
        raise ValueError('Require exactly200 original reference rows for the selected dataset')
    data = None
    samples = None
    if args.group == 'text':
        if args.inputs is None:
            parser.error('--inputs required for LongBench text')
        manifest = json.loads(args.inputs.read_text())
        samples = [r for r in manifest['samples'] if r['group'] == 'text']
        if [r['doc_id'] for r in samples] != list(range(200)):
            raise ValueError('Require original LongBench qasper first200 inputs')
    else:
        data = load_task(args.hf_home, 'vatex_test')
    model_id, revision, family = MODELS['text' if args.group == 'text' else 'vision']
    pretrained = args.pretrained.expanduser().resolve()
    config = json.loads((pretrained/'config.json').read_text())
    if (pretrained.name != revision or config.get('model_type') != family
            or config.get('num_hidden_layers') != 28 or config.get('num_key_value_heads') != 4
            or config.get('hidden_size') != 3584):
        raise ValueError('Use the pinned Qwen2.5 7B snapshot')
    names = ('top_adjacent.py', 'native_top_adjacent.py', 'native_adjacent.py',
             'native_all_pairs.py', 'reference.py', 'inputs.py', 'adjacent.py')
    contract = dict(schema_version=1, statistics_kind=KIND, group=args.group,
                    task='qasper' if args.group == 'text' else 'vatex_test',
                    model_id=model_id, model_revision=revision, pretrained=str(pretrained),
                    sample_count=args.sample_count, doc_ids=list(range(args.sample_count)),
                    top_k=200, ranking_metric='k_raw_before_rope',
                    selection='For each input/layer/native KV head, top200 among every (i,i+1); exact descending cosine then increasing pair index',
                    pair_pool='Complete unpadded input including text, video and structural tokens and boundaries',
                    aggregation='Equal inputs, layers and native KV heads; no length weighting',
                    dtype='bfloat16', statistics='FP32 normalization; FP64 adjacent products/reductions',
                    frames=64 if args.group == 'video' else None,
                    min_pixels=200704 if args.group == 'video' else None,
                    max_pixels=1605632 if args.group == 'video' else None,
                    generation=False, compression=False, use_cache=False, seed=42,
                    reference_records_sha256=sha(args.reference_records),
                    inputs_sha256=sha(args.inputs) if args.group == 'text' else None,
                    model_metadata_sha256={p.name: sha(p) for p in sorted(pretrained.iterdir())
                                           if p.is_file() and p.suffix in {'.json', '.txt'}},
                    sources_sha256={str(Path(__file__).parent/name): sha(Path(__file__).parent/name) for name in names},
                    dataset_metadata={k:v for k,v in (data or {}).items()
                                      if k.endswith('_sha256') or k in ('arrow_path','video_root','task_kwargs')},
                    versions={name: importlib.metadata.version(name) for name in
                              ('torch','transformers','flash-attn','numpy','qwen-vl-utils')})
    path = output/'contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Protocol/source/input changed; refusing mixed resume')
    atomic_json(path, contract)
    contract_sha = sha(path)
    completed = set()
    if (output/'records.jsonl').exists():
        for line in (output/'records.jsonl').open():
            row = json.loads(line)
            if (not line.endswith('\n') or row.get('statistics_kind') != KIND
                    or row.get('contract_sha256') != contract_sha or row.get('status') != 'completed'
                    or row['group'] != args.group or row['doc_id'] not in range(args.sample_count)
                    or row['doc_id'] in completed):
                raise ValueError('Incompatible/duplicate/partial existing record')
            if sha(output/row['selected_pairs_path']) != row['selected_pairs_sha256']:
                raise ValueError('Selected-pair file changed')
            completed.add(row['doc_id'])
    (output/'selected_pairs').mkdir(exist_ok=True)
    started = time.monotonic()
    state = dict(status='loading',pid=os.getpid(),group=args.group,started_utc=timestamp(),
                 target=args.sample_count,completed=len(completed),gpu=torch.cuda.get_device_name(0),
                 statistics_kind=KIND)
    atomic_json(output/'run_state.json',state)
    try:
        processor = None
        if args.group == 'text':
            tokenizer = AutoTokenizer.from_pretrained(pretrained,local_files_only=True)
            model_class = Qwen2ForCausalLM
        else:
            processor = AutoProcessor.from_pretrained(pretrained,local_files_only=True,
                                                      min_pixels=200704,max_pixels=1605632)
            tokenizer = processor.tokenizer
            model_class = Qwen2_5_VLForConditionalGeneration
        model = model_class.from_pretrained(pretrained,local_files_only=True,torch_dtype=torch.bfloat16,
                                            attn_implementation='flash_attention_2',device_map='cuda:0').eval()
        state.update(status='running',model_load_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json',state)
        for doc_id in range(args.sample_count):
            if doc_id in completed:
                continue
            before = time.monotonic()
            old = reference[doc_id]
            if args.group == 'text':
                sample=samples[doc_id]
                cpu,info=prepare_input(sample,tokenizer)
                source_metadata=sample['source_metadata']
                original_match=(info['chat_sha256']==old['chat_sha256']
                                and source_metadata==old['source_metadata'])
            else:
                cpu,info=prepare_sample(data,doc_id,processor,64,200704,1605632)
                keys=('task','video_id','source_video_id','original_row_sha256','expanded_text_sha256',
                      'video_grid_thw','frame_sampling','arrow_sha256','task_yaml_sha256','task_utils_sha256')
                source_metadata={key:info[key] for key in keys}
                original_match=all(info[key]==old[key] for key in keys)
            length=info['input_tokens']
            if length-1 < 200 or length>int(config['max_position_embeddings']):
                raise ValueError('Require at least200 original adjacent pairs within native context; no substitution')
            hashes={name:hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for name,t in cpu.items() if torch.is_tensor(t)}
            if not original_match or hashes!=old['input_hashes'] or length!=old['input_tokens']:
                raise ValueError('Input differs from the corresponding original VATEX/LongBench observation')
            ids=cpu['input_ids'][0].numpy()
            batch={name:v.to('cuda:0') if torch.is_tensor(v) else v for name,v in cpu.items()}
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), RawKTopAdjacentObserver(model,family=family,input_length=length,top_k=200) as observer:
                result=model.model(**batch,use_cache=False,output_attentions=False,output_hidden_states=False,return_dict=True)
                torch.cuda.synchronize()
                stats=observer.finish()
            starts=np.asarray([layer.pop('raw_k_topk_pair_starts') for layer in stats],dtype=np.int32)
            scores=np.asarray([layer.pop('raw_k_topk_scores') for layer in stats],dtype=np.float64)
            if starts.shape!=(28,4,200) or scores.shape!=starts.shape:
                raise ValueError('Expected200 selected adjacent pairs in every layer/native head')
            destination=output/'selected_pairs'/f'{doc_id:06d}.npz'
            temporary=destination.with_suffix('.npz.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f,pair_starts=starts,scores=scores,
                                    left_token_ids=ids[starts].astype(np.int32),
                                    right_token_ids=ids[starts+1].astype(np.int32))
            temporary.replace(destination)
            row=dict(schema_version=1,status='completed',statistics_kind=KIND,contract_sha256=contract_sha,
                     group=args.group,task=contract['task'],doc_id=doc_id,model_id=model_id,model_revision=revision,
                     input_tokens=length,adjacent_pair_count=length-1,top_k=200,
                     source_metadata=source_metadata,input_hashes=hashes,original_input_match=True,
                     layer_stats=stats,selected_pairs_path=str(destination.relative_to(output)),
                     selected_pairs_sha256=sha(destination),seconds=time.monotonic()-before,
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,completed_utc=timestamp())
            append_json(output/'records.jsonl',row)
            completed.add(doc_id)
            state.update(completed=len(completed),current_doc_id=doc_id,updated_utc=timestamp(),
                         elapsed_seconds=time.monotonic()-started)
            atomic_json(output/'run_state.json',state)
            print(json.dumps(dict(group=args.group,doc_id=doc_id,completed=len(completed),target=args.sample_count,
                                  tokens=length,top_k=200,seconds=row['seconds'])),flush=True)
            del result,batch,cpu
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        state.update(status='complete',finished_utc=timestamp(),elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json',state)
        return 0
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'run_state.json',state)
        traceback.print_exc()
        return 130 if isinstance(error,KeyboardInterrupt) else 1


if __name__=='__main__':
    raise SystemExit(main())

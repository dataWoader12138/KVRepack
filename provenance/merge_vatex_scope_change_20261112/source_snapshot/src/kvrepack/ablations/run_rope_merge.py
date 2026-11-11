"""Paired K aggregation before/after RoPE, using one input for both conditions."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

from .run_head_selection import (FRAMES, MIN_PIXELS, MAX_PIXELS, MODEL_SEED,
    atomic_json, json_sha, sha256, tensor_fingerprints, timestamp, cache_shapes)
from .run_head_selection_full import output_lock, _fresh_prompt

METHODS = ('snapkv_kvrepack', 'hybridkv_kvrepack')
TASKS = ('nextqa_oe_test', 'vatex_test')
VARIANTS = ('raw_reapply', 'post_rope')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--limit', required=True, type=int, choices=(1, 200))
    parser.add_argument('--pretrained', required=True, type=Path)
    parser.add_argument('--hf-home', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def configure(args):
    from kvrepack.experiments.runner import build_plan
    if args.method == 'hybridkv_kvrepack':
        overrides = ('HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO=.50',
            'HYBRID_KVREPACK_COMPRESS_RATIO=.50', 'K_DIFF_ALPHA=2',
            'HYBRID_KVREPACK_SIMHEAD_SIM_SPACE=raw', 'HYBRID_KVREPACK_K_MERGE_SPACE=raw_reapply')
    else:
        simtop, alpha = ('.30', '2') if args.task == 'nextqa_oe_test' else ('.15', '.5')
        overrides = (f'SNAPKV_SIMHEAD_MERGE_RATIO={simtop}',
            'SNAPKV_KVREPACK_COMPRESS_RATIO=.70', f'K_DIFF_ALPHA={alpha}')
    plan = build_plan(args.method, task=args.task, pretrained=str(args.pretrained),
        model_type='qwen2_5_vl-7b', frames=FRAMES, limit=args.limit, budget_ratio=.1,
        output=args.output, overrides=overrides)
    for name in list(os.environ):
        if name.startswith(('SNAPKV_', 'MADAKV_', 'HYBRID_', 'SPARSEMM_', 'LOOKM_')) or name in {
            'METHOD', 'MODEL_TYPE', 'BUDGET', 'BUDGET_RATIO', 'RATIO', 'K_DIFF_ALPHA',
            'HEAD_CLASSIFY', 'BUDGET_METHOD', 'K_MERGE_SPACE', 'KVREPACK_MERGE_SCORE_MODE'}:
            os.environ.pop(name)
    os.environ.update(plan.env)
    os.environ.update(HF_HOME=str(args.hf_home), HF_HUB_CACHE=str(args.hf_home/'hub'),
        HUGGINGFACE_HUB_CACHE=str(args.hf_home/'hub'), HF_DATASETS_CACHE=str(args.hf_home/'datasets'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return plan


def contract_for(args, data, plan):
    names = ['kvrepack.ablations.rope_merge', 'kvrepack.ablations.run_head_selection',
        'kvrepack.ablations.run_head_selection_full', 'kvrepack.observations.inputs',
        'kvrepack.experiments.runner', 'kvrepack.experiments.registry']
    names += (['kvrepack.hybridkv_utils', 'kvrepack.modeling_qwen2_5_vl', 'kvrepack.hybridkv_cpu']
        if args.method == 'hybridkv_kvrepack' else ['kvrepack.qwen_model_snapkv_kvrepack', 'kvrepack.monkeypatch'])
    sources = {str(Path(__file__).resolve()): sha256(__file__)}
    for name in names:
        module = importlib.import_module(name)
        sources[str(Path(module.__file__).resolve())] = sha256(module.__file__)
    result = dict(schema_version=1, experiment='rope_merge', method=args.method, task=args.task,
        target_docs=args.limit, target_generations=2*args.limit, dataset_rows=len(data['dataset']),
        conditions=list(VARIANTS), model_snapshot=str(args.pretrained),
        model_metadata={p.name:sha256(p) for p in sorted(args.pretrained.iterdir()) if p.is_file() and p.suffix in {'.json','.txt'}},
        arrow_path=data['arrow_path'], arrow_sha256=data['arrow_sha256'], video_root=data['video_root'],
        task_yaml_sha256=data['task_yaml_sha256'], task_utils_sha256=data['task_utils_sha256'],
        included_yaml_sha256=data.get('included_yaml_sha256'), frames=FRAMES,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, budget_ratio=.1,
        algorithm_env=plan.resolved_config['algorithm_env'], model_seed=MODEL_SEED,
        dtype='bfloat16', attention='flash_attention_2', do_sample=False, use_cache=True,
        max_new_tokens=128 if args.task == 'nextqa_oe_test' else 64,
        sources_sha256=sources, versions={n:importlib.metadata.version(n) for n in
            ('torch','transformers','flash-attn','qwen-vl-utils','datasets','decord','numpy')},
        intervention='Only merged K representation: rotate(mean(raw K)) versus mean(rotated K); keep head/edge selection and V aggregation unchanged',
        input_policy='One original input prepared once per row; same CPU/GPU tensors used by both arms; no persistent tensor cache')
    if args.method == 'hybridkv_kvrepack':
        import kvrepack_kernel
        result['kernel'] = dict(path=str(Path(kvrepack_kernel.__file__).resolve()), sha256=sha256(kvrepack_kernel.__file__))
    return result


def validate_commit(bundle, identity, doc_id, method, task):
    if (bundle.get('status') != 'completed' or bundle.get('doc_id') != doc_id
        or bundle.get('contract_sha256') != identity or bundle.get('method') != method
        or bundle.get('task') != task or set(bundle.get('records', {})) != set(VARIANTS)
        or bundle.get('fairness', {}).get('status') != 'passed'):
        raise ValueError('Invalid completed paired bundle')
    meta, fingerprints = bundle['reference_metadata'], bundle['input_fingerprints']
    reference = bundle['records']['raw_reapply']
    for variant, row in bundle['records'].items():
        if (row.get('status') != 'completed' or row.get('doc_id') != doc_id
            or row.get('method') != method or row.get('task') != task
            or row.get('reference_metadata') != meta or row.get('config', {}).get('variant') != variant
            or any(row.get(k) != v for k,v in fingerprints.items())
            or row.get('text') != row.get('prediction') or not row.get('generated_ids')
            or row.get('generated_tokens') != len(row['generated_ids'])):
            raise ValueError('Paired row/source/configuration mismatch')
        paired = lambda config:{k:v for k,v in config.items() if k != 'variant'}
        if paired(row['config']) != paired(reference['config']):
            raise ValueError('Non-intervention configuration changed')


def run_locked(args):
    plan = configure(args)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from kvrepack.observations import inputs
    from .rope_merge import RopeMergeAdapter, compare_merge_arms
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    torch.set_num_threads(4)
    data = inputs.load_task(args.hf_home, args.task)
    if args.limit > len(data['dataset']): raise ValueError('Requested prefix exceeds test split')
    contract = contract_for(args, data, plan)
    contract_path = args.output/'run_contract.json'
    if args.resume:
        if json.loads(contract_path.read_text()) != contract: raise ValueError('Frozen run contract changed')
    else: atomic_json(contract_path, contract)
    identity = json_sha(contract)
    paths = sorted((args.output/'samples').glob('*.json'))
    if len(paths)>args.limit or [p.name for p in paths] != [f'{i:06d}.json' for i in range(len(paths))]:
        raise ValueError('Resume requires a contiguous complete original prefix')
    for doc_id, path in enumerate(paths): validate_commit(json.loads(path.read_text()), identity, doc_id, args.method, args.task)
    initial = len(paths)
    state_path = args.output/'run_state.json'
    previous = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}
    state = dict(status='loading', task=args.task, method=args.method, target_docs=args.limit,
        target_generations=2*args.limit, completed_docs=initial, completed_generations=2*initial,
        contract_sha256=identity, started_utc=previous.get('started_utc', timestamp()),
        updated_utc=timestamp(), conditions=list(VARIANTS), estimated_remaining_seconds=None)
    atomic_json(state_path, state)
    started = time.monotonic()
    try:
        if initial == args.limit:
            state.update(status='complete',finished_utc=timestamp()); atomic_json(state_path,state); return 0
        processor = AutoProcessor.from_pretrained(str(args.pretrained),local_files_only=True,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
        if args.method == 'hybridkv_kvrepack':
            from kvrepack.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        else:
            from kvrepack.monkeypatch import replace_qwen
            replace_qwen('snapkv_kvrepack')
            from transformers import Qwen2_5_VLForConditionalGeneration
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained),torch_dtype=torch.bfloat16,
            device_map=0,attn_implementation='flash_attention_2',local_files_only=True).eval()
        kwargs=dict(eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False,temperature=None,top_p=None,num_beams=1,max_new_tokens=contract['max_new_tokens'],
            use_cache=True,return_dict_in_generate=True,output_scores=False)
        config={k:contract[k] for k in ['method','task','model_snapshot','frames','budget_ratio','algorithm_env',
            'min_pixels','max_pixels','dtype','attention','model_seed','max_new_tokens','do_sample','use_cache']}
        config.update(stream_contract_sha256=identity,source_files=contract['sources_sha256'],
            effective_generation_config=model.generation_config.to_dict(),generation_overrides=kwargs)
        atomic_json(args.output/'run_config.json',config)
        for doc_id in range(initial,args.limit):
            state.update(status='preparing',current_doc_id=doc_id,current_variant=None,updated_utc=timestamp());atomic_json(state_path,state)
            random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED)
            batch,metadata=inputs.prepare_sample(data,doc_id,processor,FRAMES,MIN_PIXELS,MAX_PIXELS)
            fingerprints=tensor_fingerprints(batch); gpu={k:v.to('cuda:0') for k,v in batch.items()}
            rows={}; adapters=[]
            for variant in VARIANTS:
                _fresh_prompt(model)
                if args.method == 'hybridkv_kvrepack':
                    for layer in model.model.language_model.layers:
                        if hasattr(layer.self_attn,'kv_cluster'): delattr(layer.self_attn,'kv_cluster')
                random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED)
                state.update(status='running',current_variant=variant,updated_utc=timestamp());atomic_json(state_path,state)
                adapter=RopeMergeAdapter(method=args.method,variant=variant)
                if doc_id == initial and variant == 'raw_reapply':
                    spec_path=args.output/'intervention.json'
                    if spec_path.exists() and json.loads(spec_path.read_text()) != adapter.transform_spec:
                        raise ValueError('Intervention source transformation changed')
                    if not spec_path.exists(): atomic_json(spec_path,adapter.transform_spec)
                with adapter:
                    adapter.set_sample(args.task,doc_id)
                    torch.cuda.reset_peak_memory_stats();began=time.monotonic()
                    with torch.inference_mode(): result=model.generate(**gpu,**kwargs)
                    torch.cuda.synchronize();elapsed=time.monotonic()-began;audit=adapter.finish_sample()
                adapters.append(adapter)
                ids=result.sequences[0,batch['input_ids'].shape[-1]:].tolist()
                text=processor.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                rows[variant]=dict(status='completed',task=args.task,method=args.method,doc_id=doc_id,
                    source_video_id=metadata['source_video_id'],video_id=metadata['video_id'],
                    text=text,prediction=text,generated_ids=ids,generated_tokens=len(ids),**fingerprints,
                    reference_metadata=metadata,config=dict(config,variant=variant),audit=audit,
                    input_tokens=metadata['input_tokens'],max_new_tokens=contract['max_new_tokens'],
                    eos_token_id=processor.tokenizer.eos_token_id,eos_reached=bool(ids and ids[-1]==processor.tokenizer.eos_token_id),
                    final_cache_shapes=cache_shapes(result.past_key_values),
                    runtime=dict(generation_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),completed_utc=timestamp())
                del result
                print(json.dumps(dict(stage='condition_completed',task=args.task,method=args.method,doc_id=doc_id,variant=variant,generated_tokens=len(ids))),flush=True)
            fairness=compare_merge_arms(adapters[0],adapters[1])
            if fairness.get('status')!='passed': raise ValueError('Pairing did not pass')
            if tensor_fingerprints(batch)!=fingerprints: raise ValueError('Shared CPU input mutated')
            bundle=dict(schema_version=1,status='completed',task=args.task,method=args.method,doc_id=doc_id,
                contract_sha256=identity,input_fingerprints=fingerprints,reference_metadata=metadata,
                records=rows,fairness=fairness,completed_utc=timestamp())
            validate_commit(bundle,identity,doc_id,args.method,args.task)
            path=args.output/'samples'/f'{doc_id:06d}.json'
            if path.exists():raise ValueError('Refusing to overwrite a committed sample')
            atomic_json(path,bundle)
            completed=doc_id+1;elapsed_run=time.monotonic()-started
            state.update(completed_docs=completed,completed_generations=2*completed,updated_utc=timestamp(),
                elapsed_current_session_seconds=elapsed_run,estimated_remaining_seconds=elapsed_run/(completed-initial)*(args.limit-completed))
            atomic_json(state_path,state)
            print(json.dumps(dict(stage='sample_committed',task=args.task,method=args.method,doc_id=doc_id,completed_docs=completed,target_docs=args.limit)),flush=True)
            del rows,bundle,adapters,adapter,gpu,batch
        state.update(status='complete',finished_utc=timestamp(),estimated_remaining_seconds=0);atomic_json(state_path,state)
        return 0
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=timestamp());atomic_json(state_path,state);raise


def main(argv=None):
    args=parse_args(argv)
    for name in ['pretrained','hf_home','output']:setattr(args,name,getattr(args,name).expanduser().resolve())
    if not args.pretrained.is_dir():raise FileNotFoundError(args.pretrained)
    with output_lock(args.output,resume=args.resume):return run_locked(args)


if __name__=='__main__':raise SystemExit(main())

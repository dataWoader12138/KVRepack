"""HybridKV-KVRepack global Random-head versus similarity, NextQA prefix only."""
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
    atomic_json, cache_shapes, json_sha, sha256, tensor_fingerprints, timestamp)
from .run_head_selection_full import output_lock, _fresh_prompt

TASK='nextqa_oe_test'
METHOD='hybridkv_kvrepack'
CONDITIONS=(('similarity','similarity',0),('random_global_seed0','random_global',0),
    ('random_global_seed1','random_global',1),('random_global_seed2','random_global',2))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit',required=True,type=int,choices=(1,200))
    p.add_argument('--pretrained',required=True,type=Path)
    p.add_argument('--hf-home',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    return p.parse_args(argv)


def configure(args):
    from kvrepack.experiments.runner import build_plan
    plan=build_plan(METHOD,task=TASK,pretrained=str(args.pretrained),model_type='qwen2_5_vl-7b',
        frames=FRAMES,limit=args.limit,budget_ratio=.1,output=args.output,overrides=(
        'HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO=.50','HYBRID_KVREPACK_COMPRESS_RATIO=.50',
        'K_DIFF_ALPHA=2','HYBRID_KVREPACK_SIMHEAD_SIM_SPACE=raw','HYBRID_KVREPACK_K_MERGE_SPACE=raw_reapply'))
    for name in list(os.environ):
        if name.startswith(('HYBRID_','SNAPKV_','MADAKV_','SPARSEMM_','LOOKM_')) or name in {
            'METHOD','MODEL_TYPE','BUDGET','BUDGET_RATIO','RATIO','K_DIFF_ALPHA','HEAD_CLASSIFY',
            'BUDGET_METHOD','K_MERGE_SPACE','KVREPACK_MERGE_SCORE_MODE'}:os.environ.pop(name)
    os.environ.update(plan.env)
    os.environ.update(HF_HOME=str(args.hf_home),HF_HUB_CACHE=str(args.hf_home/'hub'),
        HUGGINGFACE_HUB_CACHE=str(args.hf_home/'hub'),HF_DATASETS_CACHE=str(args.hf_home/'datasets'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return plan


def contract_for(args,data,plan):
    names=['kvrepack.ablations.hybrid_head_selection','kvrepack.ablations.hybrid_head_selection_report',
        'kvrepack.ablations.run_head_selection','kvrepack.ablations.run_head_selection_full',
        'kvrepack.ablations.head_selection','kvrepack.observations.inputs','kvrepack.experiments.runner',
        'kvrepack.experiments.registry','kvrepack.hybridkv_utils','kvrepack.modeling_qwen2_5_vl',
        'kvrepack.hybridkv_cpu','kvrepack.baseline_utils','kvrepack.merge_score']
    files={str(Path(__file__).resolve()):sha256(__file__)}
    for name in names:
        m=importlib.import_module(name);files[str(Path(m.__file__).resolve())]=sha256(m.__file__)
    package=Path(__file__).resolve().parents[1]
    for rel in ['configs/param_config.json','data/head_score/qwen2_5.json']:
        path=package/rel;files[str(path)]=sha256(path)
    import kvrepack_kernel
    return dict(schema_version=1,task=TASK,method=METHOD,target_docs=args.limit,target_generations=4*args.limit,
        conditions=[list(c) for c in CONDITIONS],dataset_rows=len(data['dataset']),
        model_snapshot=str(args.pretrained),model_metadata={p.name:sha256(p) for p in sorted(args.pretrained.iterdir())
            if p.is_file() and p.suffix in {'.json','.txt'}},arrow_path=data['arrow_path'],arrow_sha256=data['arrow_sha256'],
        video_root=data['video_root'],task_yaml_sha256=data['task_yaml_sha256'],task_utils_sha256=data['task_utils_sha256'],
        included_yaml_sha256=data.get('included_yaml_sha256'),frames=FRAMES,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS,
        budget_ratio=.1,algorithm_env=plan.resolved_config['algorithm_env'],dtype='bfloat16',attention='flash_attention_2',
        max_new_tokens=128,do_sample=False,use_cache=True,model_seed=MODEL_SEED,selected_heads=56,
        source_files=files,sources_sha256=files,kernel=dict(path=str(Path(kvrepack_kernel.__file__).resolve()),
            sha256=sha256(kvrepack_kernel.__file__)),versions={name:importlib.metadata.version(name) for name in
            ('torch','transformers','flash-attn','qwen-vl-utils','datasets','decord','numpy')},
        intervention='Only global head identities change. Native classification and every per-head retained budget stay fixed. Total candidate expansion/removal may change with the identities of heterogeneous-budget heads.',
        input_policy='Original first-prefix order; one CPU preparation, identical tensors for four conditions; no persistent tensor cache')


def validate_bundle(bundle,*,doc_id,contract_sha256):
    if (bundle.get('schema_version')!=1 or bundle.get('status')!='completed' or bundle.get('task')!=TASK
        or bundle.get('method')!=METHOD or bundle.get('doc_id')!=doc_id
        or bundle.get('contract_sha256')!=contract_sha256
        or set(bundle.get('records',{}))!={c[0] for c in CONDITIONS}
        or bundle.get('fairness',{}).get('status')!='passed'):raise ValueError('Incomplete Hybrid paired sample')
    from .hybrid_head_selection_report import _audit
    fingerprints,meta=bundle['input_fingerprints'],bundle['reference_metadata']
    baseline=bundle['records']['similarity']
    for label,strategy,seed in CONDITIONS:
        row=bundle['records'][label];config=row.get('config',{});audit=row.get('audit',{})
        if (row.get('status')!='completed' or row.get('method')!=METHOD or row.get('task')!=TASK
            or row.get('doc_id')!=doc_id or row.get('reference_metadata')!=meta
            or any(row.get(k)!=v for k,v in fingerprints.items())
            or config.get('strategy')!=strategy or config.get('mask_seed')!=seed
            or config.get('stream_contract_sha256')!=contract_sha256
            or audit.get('strategy')!=strategy or audit.get('seed')!=seed
            or row.get('text')!=row.get('prediction') or not row.get('generated_ids')
            or row.get('generated_tokens')!=len(row['generated_ids'])):raise ValueError('Hybrid row/source/config mismatch')
        comparable=lambda c:{k:v for k,v in c.items() if k not in ('strategy','mask_seed')}
        if comparable(config)!=comparable(baseline['config']):raise ValueError('Nonselection settings differ')
        if (audit.get('pre_selection')!=baseline['audit'].get('pre_selection')
            or audit.get('baseline_masks')!=baseline['audit'].get('baseline_masks')):
            raise ValueError('Native budgets/classification/candidates or baseline ranking changed')
        _audit(row,label)
    return bundle


def run_locked(args):
    plan=configure(args)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from kvrepack.observations import inputs
    from .hybrid_head_selection import HybridHeadSelectionAdapter,compare_hybrid_arms
    from kvrepack.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED);torch.set_num_threads(4)
    data=inputs.load_task(args.hf_home,TASK)
    if args.limit>len(data['dataset']):raise ValueError('Prefix exceeds source test split')
    contract=contract_for(args,data,plan);path=args.output/'run_contract.json'
    if args.resume:
        if json.loads(path.read_text())!=contract:raise ValueError('Frozen Hybrid run contract changed')
    else:atomic_json(path,contract)
    identity=json_sha(contract);files=sorted((args.output/'samples').glob('*.json'))
    if len(files)>args.limit or [p.name for p in files]!=[f'{i:06}.json' for i in range(len(files))]:
        raise ValueError('Committed Hybrid samples must be a contiguous complete prefix')
    for doc,path in enumerate(files):validate_bundle(json.loads(path.read_text()),doc_id=doc,contract_sha256=identity)
    state_path=args.output/'run_state.json';old=json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}
    initial=len(files);state=dict(status='loading',task=TASK,method=METHOD,target_docs=args.limit,
        target_generations=4*args.limit,completed_docs=initial,completed_generations=4*initial,
        contract_sha256=identity,started_utc=old.get('started_utc',timestamp()),updated_utc=timestamp(),
        conditions=[c[0] for c in CONDITIONS],estimated_remaining_seconds=None)
    atomic_json(state_path,state);started=time.monotonic()
    try:
        if initial==args.limit:
            state.update(status='complete',finished_utc=timestamp());atomic_json(state_path,state);return 0
        processor=AutoProcessor.from_pretrained(str(args.pretrained),local_files_only=True,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained),torch_dtype=torch.bfloat16,
            device_map=0,attn_implementation='flash_attention_2',local_files_only=True).eval()
        generation=dict(eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False,temperature=None,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True,
            return_dict_in_generate=True,output_scores=False)
        config={k:contract[k] for k in ['task','method','model_snapshot','frames','budget_ratio','algorithm_env',
            'min_pixels','max_pixels','dtype','attention','max_new_tokens','do_sample','use_cache','model_seed']}
        config.update(stream_contract_sha256=identity,source_files=contract['source_files'],
            effective_generation_config=model.generation_config.to_dict(),generation_overrides=generation)
        atomic_json(args.output/'run_config.json',config)
        for doc in range(initial,args.limit):
            state.update(status='preparing',current_doc_id=doc,current_strategy=None,updated_utc=timestamp());atomic_json(state_path,state)
            random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED)
            batch,meta=inputs.prepare_sample(data,doc,processor,FRAMES,MIN_PIXELS,MAX_PIXELS)
            fingerprints=tensor_fingerprints(batch);gpu={k:v.to('cuda:0') for k,v in batch.items()};rows={};adapters=[]
            for label,strategy,seed in CONDITIONS:
                _fresh_prompt(model)
                for layer in model.model.language_model.layers:
                    if hasattr(layer.self_attn,'kv_cluster'):delattr(layer.self_attn,'kv_cluster')
                random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED)
                state.update(status='running',current_strategy=label,updated_utc=timestamp());atomic_json(state_path,state)
                adapter=HybridHeadSelectionAdapter(strategy=strategy,seed=seed)
                with adapter:
                    adapter.set_sample(TASK,doc);torch.cuda.reset_peak_memory_stats();began=time.monotonic()
                    with torch.inference_mode():result=model.generate(**gpu,**generation)
                    torch.cuda.synchronize();elapsed=time.monotonic()-began;audit=adapter.finish_sample()
                adapters.append(adapter);ids=result.sequences[0,batch['input_ids'].shape[-1]:].tolist()
                text=processor.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                rows[label]=dict(task=TASK,method=METHOD,doc_id=doc,status='completed',source_video_id=meta['source_video_id'],
                    video_id=meta['video_id'],text=text,prediction=text,generated_ids=ids,generated_tokens=len(ids),
                    input_tokens=meta['input_tokens'],**fingerprints,input_fingerprint_sha256=json_sha(fingerprints),
                    reference_metadata=meta,config=dict(config,strategy=strategy,mask_seed=seed),audit=audit,
                    final_cache_shapes=cache_shapes(result.past_key_values),max_new_tokens=128,
                    eos_token_id=processor.tokenizer.eos_token_id,eos_reached=bool(ids and ids[-1]==processor.tokenizer.eos_token_id),
                    runtime=dict(generation_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),
                    completed_utc=timestamp())
                del result
                print(json.dumps(dict(stage='condition_completed',doc_id=doc,condition=label,
                    selected=audit.get('selected_count'),actual_merges=audit.get('actual_merge_calls'),generated_tokens=len(ids))),flush=True)
            fairness=compare_hybrid_arms(adapters)
            if fairness.get('status')!='passed':raise ValueError('Hybrid selection fairness gate failed')
            if tensor_fingerprints(batch)!=fingerprints:raise ValueError('Shared CPU input mutated')
            bundle=dict(schema_version=1,status='completed',task=TASK,method=METHOD,doc_id=doc,contract_sha256=identity,
                input_fingerprints=fingerprints,reference_metadata=meta,records=rows,fairness=fairness,completed_utc=timestamp())
            validate_bundle(bundle,doc_id=doc,contract_sha256=identity)
            sample=args.output/'samples'/f'{doc:06d}.json'
            if sample.exists():raise ValueError('Refusing to overwrite a committed sample')
            atomic_json(sample,bundle);completed=doc+1;elapsed_run=time.monotonic()-started
            state.update(completed_docs=completed,completed_generations=4*completed,updated_utc=timestamp(),
                elapsed_current_session_seconds=elapsed_run,estimated_remaining_seconds=elapsed_run/(completed-initial)*(args.limit-completed))
            atomic_json(state_path,state)
            print(json.dumps(dict(stage='sample_committed',completed_docs=completed,target_docs=args.limit)),flush=True)
            del batch,gpu,rows,bundle,adapters,adapter
        state.update(status='complete',finished_utc=timestamp(),estimated_remaining_seconds=0);atomic_json(state_path,state);return 0
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_utc=timestamp());atomic_json(state_path,state);raise


def main(argv=None):
    args=parse_args(argv)
    for key in ['pretrained','hf_home','output']:setattr(args,key,getattr(args,key).expanduser().resolve())
    if not args.pretrained.is_dir():raise FileNotFoundError(args.pretrained)
    with output_lock(args.output,resume=args.resume):return run_locked(args)


if __name__=='__main__':raise SystemExit(main())

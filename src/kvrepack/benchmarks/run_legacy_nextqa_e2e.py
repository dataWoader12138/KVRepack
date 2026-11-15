"""Original standalone LOOK-M/AsymKV on the NExTQA prefix, timed video to text.

Original algorithm files stay frozen. LOOK-M's two pre-fix functions are
restored in a scoped context. AsymKV runs the complete released generation
driver, including all Fisher probes and its native chunk/sink semantics.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

from ..ablations.run_head_selection import (FRAMES, MIN_PIXELS, MAX_PIXELS,
    MODEL_SEED, atomic_json, cache_shapes, json_sha, sha256, tensor_fingerprints, timestamp)
from ..ablations.run_head_selection_full import output_lock, _fresh_prompt
from ..ablations.run_baseline_completion import apply_plan
from ..ablations.video_e2e_inputs import prepare_input

POLICIES = dict(lookm="original_lookm_history_plus_recent_v1", asymkv="original_asymkv_official_v1")


def sources():
    root = Path(__file__).resolve().parents[3]
    names = ["benchmarks/run_legacy_nextqa_e2e.py", "benchmarks/legacy_lookm.py",
        "ablations/video_e2e_inputs.py", "ablations/run_head_selection.py",
        "ablations/run_head_selection_full.py", "ablations/run_baseline_completion.py",
        "observations/inputs.py", "experiments/runner.py", "experiments/registry.py",
        "baseline_utils.py", "monkeypatch.py", "qwen_model.py", "qwen_model_croatt.py",
        "_asymkv_helpers.py", "asymkv_official/integration.py", "asymkv_official/generation.py",
        "asymkv_official/compression.py", "asymkv_official/attention.py", "asymkv_official/cache.py"]
    result = {"src/kvrepack/"+name: sha256(root/"src/kvrepack"/name) for name in names}
    for name in ("provenance/core_sources.json", "provenance/patches/lookm_strict_budget_v2.patch"):
        result[name] = sha256(root/name)
    return result


def legacy_input(data, doc_id, processor):
    """Preserve the extra first-frame video open in the original lmms adapter."""
    import decord
    from ..observations.inputs import _video_path
    path = _video_path(Path(data["video_root"]), str(data["dataset"][doc_id]["video"]))
    reader = decord.VideoReader(str(path))
    first_frame = reader[0].asnumpy()
    _height, _width = first_frame.shape[:2]
    del first_frame, reader
    return prepare_input(data, doc_id, processor)


def execution_order(data, processor, limit):
    """Same descending prompt-token length and lexical tie-break as lmms."""
    return sorted(range(limit), key=lambda i: (
        -len(processor.tokenizer.encode(data["_prompt_function"](data["dataset"][i], data["task_kwargs"]))),
        data["_prompt_function"](data["dataset"][i], data["task_kwargs"])))


def run(args):
    from ..experiments.runner import build_plan
    plan = build_plan(args.method, task="nextqa_oe_test", model_type="qwen2_5_vl-7b",
        pretrained=str(args.pretrained), frames=64, limit=args.limit, budget_ratio=.1, output=args.output)
    apply_plan(plan, args.hf_home)
    os.environ.pop("ASYMKV_OFFICIAL_CACHE_SIZE", None)
    os.environ.update(ASYMKV_START_SIZE="32", ASYMKV_CHUNK_SIZE="512")
    # Official audit payloads are captured in memory; disk I/O is after the timer.
    os.environ.pop("ASYMKV_AUDIT_PATH", None)
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from ..monkeypatch import replace_qwen
    from ..observations.inputs import load_task
    from .legacy_lookm import LegacyLookMContext, source_identity
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    data = load_task(args.hf_home, "nextqa_oe_test")
    frozen = sources()
    contract = dict(schema_version=1, experiment="legacy_standalone_response_e2e",
        method=args.method, task="nextqa_oe_test", target_docs=args.limit, target_generations=args.limit,
        model_snapshot=str(args.pretrained), hf_home=str(args.hf_home), frames=64,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, budget_ratio=.1, budget_policy=POLICIES[args.method],
        model_seed=42, dtype="bfloat16", do_sample=False, max_new_tokens=128,
        repetition_penalty=1.05 if args.method=="lookm" else None,
        decoding_policy="hf_greedy_repetition_penalty_1p05" if args.method=="lookm" else "official_argmax_no_repetition_penalty",
        algorithm_env=plan.resolved_config["algorithm_env"], source_sha256=frozen,
        model_metadata_sha256={p.name:sha256(p) for p in sorted(args.pretrained.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt"}},
        arrow_sha256=data["arrow_sha256"], dataset_rows=len(data["dataset"]),
        task_yaml_sha256=data["task_yaml_sha256"], task_utils_sha256=data["task_utils_sha256"],
        versions={name:importlib.metadata.version(name) for name in
            ("torch", "transformers", "flash-attn", "qwen-vl-utils", "datasets", "decord", "numpy")},
        warmup_per_arm=0, diagnostic_audit=False,
        timing_scope="Original video first-frame read+decode/processor+H2D+complete native generate (including AsymKV Fisher)+text decode; excludes loading/scoring/audit hashing/writes",
        order_policy="Original firstN IDs, executed in lmms descending-prompt-length order",
        legacy_lookm_identity=source_identity() if args.method=="lookm" else None,
        asymkv_runtime=dict(start_size=32, chunk_size=512, extra_recent_added=False) if args.method=="asymkv" else None)
    identity=json_sha(contract);contract_path=args.output/"run_contract.json"
    if args.resume:
        if json.loads(contract_path.read_text())!=contract:raise ValueError("Frozen contract changed")
        prior=json.loads((args.output/"run_state.json").read_text())
        if prior.get("resume_authorized") is False:raise ValueError("Stopped run may not resume")
    else:atomic_json(contract_path,contract)
    existing={int(p.stem):p for p in (args.output/"samples").glob("*.json")}
    if any(not 0<=i<args.limit for i in existing):raise ValueError("Unexpected completed row")
    for i,p in existing.items():
        row=json.loads(p.read_text())
        if row.get("doc_id")!=i or row.get("contract_sha256")!=identity:raise ValueError("Saved identity differs")
    previous_loop=float(prior.get("response_loop_seconds",0.)) if args.resume else 0.
    state=dict(status="loading",method=args.method,task="nextqa_oe_test",contract_sha256=identity,
        target_docs=args.limit,target_generations=args.limit,completed_docs=len(existing),
        completed_generations=len(existing),started_utc=timestamp(),pid=os.getpid(),
        warmup_per_arm=0,response_loop_seconds=0.,timed_e2e_seconds_sum=0.)
    state_path=args.output/"run_state.json";atomic_json(state_path,state)
    try:
        processor=AutoProcessor.from_pretrained(str(args.pretrained),local_files_only=True,
            min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
        replace_qwen("lookm" if args.method=="lookm" else "asymkv_official")
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained),
            torch_dtype=torch.bfloat16,device_map=0,attn_implementation="flash_attention_2",local_files_only=True).eval()
        kwargs=dict(eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False,temperature=None,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True,
            return_dict_in_generate=True,output_scores=False)
        captures=[];audit_module=None;original_append=None
        if args.method=="asymkv":
            from ..asymkv_official import generation as audit_module
            original_append=audit_module._append_audit
            audit_module._append_audit=lambda path,record:captures.append(record)
        atomic_json(args.output/"environment.json",dict(gpu=torch.cuda.get_device_name(0),
            generation_kwargs=kwargs,model_generation_config=model.generation_config.to_dict()))
        order=execution_order(data,processor,args.limit)
        state["execution_order"]=order
        context=LegacyLookMContext() if args.method=="lookm" else nullcontext()
        timer_sum=sum(json.loads(p.read_text())["runtime"]["end_to_end_seconds"] for p in existing.values())
        loop_begin=time.perf_counter()
        with context:
            if args.method=="lookm":atomic_json(args.output/"legacy_lookm_transform.json",context.transform_spec)
            for doc_id in order:
                if doc_id in existing:continue
                _fresh_prompt(model);captures.clear()
                for layer in model.model.language_model.layers:
                    if hasattr(layer.self_attn,"kv_cluster"):delattr(layer.self_attn,"kv_cluster")
                state.update(status="running",current_doc_id=doc_id,updated_utc=timestamp());atomic_json(state_path,state)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                batch,metadata=legacy_input(data,doc_id,processor);prepared=time.perf_counter()
                gpu={k:v.to("cuda:0") for k,v in batch.items()};torch.cuda.synchronize();transferred=time.perf_counter()
                grad_context=torch.inference_mode() if args.method=="lookm" else nullcontext()
                with grad_context:result=model.generate(**gpu,**kwargs)
                torch.cuda.synchronize();generated=time.perf_counter()
                sequence=result.sequences if hasattr(result,"sequences") else result
                ids=sequence[0,batch["input_ids"].shape[-1]:].tolist()
                text=processor.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                torch.cuda.synchronize();end=time.perf_counter()
                if args.method=="lookm":
                    shapes=cache_shapes(result.past_key_values)
                    physical=sum(__import__("math").prod(z["key_shape"][:-1]) for z in shapes)-112*(len(ids)-1)
                    expected=112*(int(.1*metadata["input_tokens"])+32)
                    if physical!=expected:raise ValueError("Original LOOK-M budget changed")
                    audit=dict(physical_prefill_positions=physical,retained_ratio=physical/(112*metadata["input_tokens"]),final_cache_shapes=shapes)
                else:
                    if len(captures)!=1:raise ValueError("Missing official AsymKV audit")
                    audit=captures.pop()
                    if audit["prompt_length"]!=metadata["input_tokens"]:raise ValueError("AsymKV input differs")
                row=dict(status="completed",method=args.method,task="nextqa_oe_test",doc_id=doc_id,
                    contract_sha256=identity,text=text,prediction=text,generated_ids=ids,generated_tokens=len(ids),
                    input_tokens=metadata["input_tokens"],reference_metadata=metadata,input_fingerprints=tensor_fingerprints(batch),
                    eos_reached=bool(ids and ids[-1]==processor.tokenizer.eos_token_id),
                    capped_without_eos=len(ids)==128 and ids[-1]!=processor.tokenizer.eos_token_id,
                    warmup=False,audit=audit,runtime=dict(preprocessing_seconds=prepared-begin,h2d_seconds=transferred-prepared,
                        generation_seconds=generated-transferred,text_decode_seconds=end-generated,
                        end_to_end_seconds=end-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),completed_utc=timestamp())
                if sources()!=frozen:raise ValueError("Source changed during experiment")
                atomic_json(args.output/"samples"/f"{doc_id:06d}.json",row)
                existing[doc_id]=args.output/"samples"/f"{doc_id:06d}.json";timer_sum+=end-begin
                state.update(completed_docs=len(existing),completed_generations=len(existing),
                    response_loop_seconds=previous_loop+time.perf_counter()-loop_begin,timed_e2e_seconds_sum=timer_sum,
                    updated_utc=timestamp());atomic_json(state_path,state)
                print(json.dumps(dict(event="sample_complete",method=args.method,doc_id=doc_id,
                    text=text,runtime=row["runtime"])),flush=True)
                del result,sequence,gpu,batch
        if audit_module is not None:audit_module._append_audit=original_append
        state.update(status="complete",finished_utc=timestamp(),response_loop_seconds=previous_loop+time.perf_counter()-loop_begin)
        atomic_json(state_path,state);return 0
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}",finished_utc=timestamp());atomic_json(state_path,state);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=tuple(POLICIES),required=True)
    for name in ("pretrained","hf-home","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--limit",type=int,choices=(1,200),required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    for name in ("pretrained","hf_home","output"):setattr(args,name,getattr(args,name).expanduser().resolve())
    with output_lock(args.output,resume=args.resume):return run(args)


if __name__=="__main__":raise SystemExit(main())

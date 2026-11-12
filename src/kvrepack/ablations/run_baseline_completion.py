"""Paired native/KVRepack video baselines with the merge experiment's E2E scope.

This is a separate worker: completed and stopped historical runs stay frozen.
Native compression uses its own original implementation, with only the shared
total-active-prefill budget policy applied by StrictPruningAdapter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from .run_head_selection import (FRAMES, MIN_PIXELS, MAX_PIXELS, MODEL_SEED,
    atomic_json, json_sha, sha256, tensor_fingerprints, timestamp)
from .run_head_selection_full import output_lock, _fresh_prompt
from .run_merge_comparison import active_prefill, sources as shared_sources
from .video_e2e_inputs import prepare_input

VARIANTS = ("native", "kvrepack")


def sources():
    result = shared_sources()
    root = Path(__file__).resolve().parents[3]
    for name in ("ablations/run_baseline_completion.py", "ablations/strict_pruning.py",
                 "ablations/video_e2e_inputs.py", "baseline_utils.py"):
        relative = "src/kvrepack/"+name
        result[relative] = sha256(root/relative)
    return result


def apply_plan(plan, hf_home):
    for name in list(os.environ):
        if name.startswith(("SNAPKV_", "MADAKV_", "HYBRID_", "SPARSEMM_", "LOOKM_")) or name in {
            "METHOD", "MODEL_TYPE", "BUDGET", "BUDGET_RATIO", "RATIO", "K_DIFF_ALPHA",
            "HEAD_CLASSIFY", "BUDGET_METHOD", "K_MERGE_SPACE", "KVREPACK_MERGE_SCORE_MODE"}:
            os.environ.pop(name)
    os.environ.update(plan.env)
    os.environ.update(HF_HOME=str(hf_home), HF_HUB_CACHE=str(hf_home/"hub"),
        HUGGINGFACE_HUB_CACHE=str(hf_home/"hub"), HF_DATASETS_CACHE=str(hf_home/"datasets"),
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def plans_for(args):
    from ..experiments.runner import build_plan
    from .run_rope_merge import configure
    native = build_plan(args.backbone, task=args.task, pretrained=str(args.pretrained),
        model_type="qwen2_5_vl-7b", frames=FRAMES, limit=args.limit, budget_ratio=.1,
        output=args.output, overrides=())
    improved = configure(SimpleNamespace(method=args.backbone+"_kvrepack", task=args.task,
        pretrained=args.pretrained, hf_home=args.hf_home, limit=args.limit, output=args.output))
    return {"native": native, "kvrepack": improved}


@contextmanager
def snap_forward(variant):
    """Switch only attention; install the shared generation wrapper once."""
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as hf
    from ..qwen_model import qwen_flash_attn2_forward_SnapKV
    from ..qwen_model_snapkv_kvrepack import qwen_flash_attn2_forward_SnapKVKVRepack
    owner = hf.Qwen2_5_VLFlashAttention2
    original = owner.forward
    owner.forward = (qwen_flash_attn2_forward_SnapKV if variant == "native"
                     else qwen_flash_attn2_forward_SnapKVKVRepack)
    try:
        yield
    finally:
        owner.forward = original


def validate_bundle(bundle, identity, backbone, doc_id):
    if (bundle.get("status") != "completed" or bundle.get("contract_sha256") != identity
            or bundle.get("doc_id") != doc_id or bundle.get("backbone") != backbone
            or set(bundle.get("records", {})) != set(VARIANTS)
            or bundle.get("fairness", {}).get("status") != "passed"):
        raise ValueError("Invalid baseline bundle")
    reference = bundle["records"]["native"]
    for variant, row in bundle["records"].items():
        expected_method = backbone if variant == "native" else backbone+"_kvrepack"
        if (row.get("status") != "completed" or row.get("method") != expected_method
                or row.get("doc_id") != doc_id or row.get("variant") != variant
                or row.get("task") != reference.get("task")
                or row.get("text") != row.get("prediction") or not row.get("generated_ids")
                or row.get("generated_tokens") != len(row["generated_ids"])
                or row.get("warmup") is not False
                or row.get("input_fingerprints") != reference.get("input_fingerprints")
                or row.get("reference_metadata") != reference.get("reference_metadata")):
            raise ValueError("Baseline row or actual model inputs differ")
        budget = row.get("budget", {})
        if not 0 < budget.get("prefill_tokens_total", 0) <= budget.get("capacity", 0):
            raise ValueError("Measured active prefill cache exceeds its capacity")
        if budget.get("capacity") != reference.get("budget", {}).get("capacity"):
            raise ValueError("Declared native/KVRepack capacity differs")
        if backbone == "snapkv" and budget.get("prefill_tokens_total") != reference["budget"]["prefill_tokens_total"]:
            raise ValueError("Actual SnapKV native/KVRepack prefill budgets differ")


def run(args):
    max_new_tokens = 128 if args.task == "nextqa_oe_test" else 64
    plans = plans_for(args)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from ..observations.inputs import load_task
    from .merge_snap import SnapMergeAdapter
    from .merge_hybrid import MergeHybridAdapter
    from .strict_pruning import StrictPruningAdapter
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    apply_plan(plans["kvrepack"], args.hf_home)
    data = load_task(args.hf_home, args.task)
    frozen = sources()
    contract = dict(schema_version=1, experiment="strict_baseline_completion", backbone=args.backbone,
        method=args.backbone, variants=list(VARIANTS), task=args.task, target_docs=args.limit,
        target_generations=2*args.limit, frames=FRAMES, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        model_snapshot=str(args.pretrained), hf_home=str(args.hf_home), budget_ratio=.1,
        budget_policy="total_active_prefill_including_local_global_cap_v1", diagnostic_audit=False,
        max_new_tokens=max_new_tokens, repetition_penalty=1.05, do_sample=False, dtype="bfloat16", model_seed=MODEL_SEED,
        arrow_sha256=data["arrow_sha256"], dataset_rows=len(data["dataset"]),
        task_yaml_sha256=data["task_yaml_sha256"], task_utils_sha256=data["task_utils_sha256"],
        source_sha256=frozen, warmup_per_arm=args.warmup,
        algorithm_env={v:p.resolved_config["algorithm_env"] for v,p in plans.items()},
        model_metadata_sha256={p.name:sha256(p) for p in sorted(args.pretrained.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt"}},
        versions={name:importlib.metadata.version(name) for name in
            ("torch", "transformers", "flash-attn", "qwen-vl-utils", "datasets", "decord", "numpy")},
        timing_scope="video read/decode+processor+H2D+Fisher(if AsymKV)+generate+text decode; synchronized GPU; excludes model load/warmup/audit hashing/writes/scoring",
        order_policy="native/KVRepack rotate by doc_id; each arm independently reads and prepares the same input")
    identity = json_sha(contract)
    contract_path = args.output/"run_contract.json"
    if args.resume:
        prior = json.loads((args.output/"run_state.json").read_text())
        if prior.get("resume_authorized") is False or prior.get("status") == "stopped_by_user":
            raise ValueError("Stopped baselines require new explicit authorization")
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Frozen baseline contract changed")
    else:
        atomic_json(contract_path, contract)
    paths = sorted((args.output/"samples").glob("*.json"))
    if [p.name for p in paths] != [f"{i:06d}.json" for i in range(len(paths))] or len(paths)>args.limit:
        raise ValueError("Only complete contiguous prefix can be resumed")
    for i,path in enumerate(paths): validate_bundle(json.loads(path.read_text()), identity, args.backbone, i)
    initial = len(paths)
    state = dict(status="loading", backbone=args.backbone, method=args.backbone, task=args.task,
        target_docs=args.limit, target_generations=2*args.limit, completed_docs=initial,
        completed_generations=2*initial, contract_sha256=identity, pid=os.getpid(), started_utc=timestamp())
    state_path=args.output/"run_state.json"; atomic_json(state_path,state)
    started=time.perf_counter()
    try:
        processor=AutoProcessor.from_pretrained(str(args.pretrained),local_files_only=True,
            min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
        if args.backbone=="snapkv":
            from ..monkeypatch import replace_qwen
            replace_qwen("snapkv_kvrepack")  # Exactly one shared generation wrapper.
            from transformers import Qwen2_5_VLForConditionalGeneration
            Adapter=SnapMergeAdapter
        else:
            from ..modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            Adapter=MergeHybridAdapter
        load_started=time.perf_counter()
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained),
            torch_dtype=torch.bfloat16,device_map=0,attn_implementation="flash_attention_2",local_files_only=True).eval()
        model.requires_grad_(False);torch.cuda.synchronize()
        state["model_load_seconds"]=time.perf_counter()-load_started
        adapters=dict(native=StrictPruningAdapter(args.backbone,budget_ratio=.1),
                      kvrepack=Adapter("kvrepack",budget_ratio=.1,audit=False))
        kwargs=dict(eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            do_sample=False,temperature=None,top_p=None,num_beams=1,max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,use_cache=True,return_dict_in_generate=True,output_scores=False)
        atomic_json(args.output/"environment.json",dict(gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,transformers=importlib.metadata.version("transformers"),generation_kwargs=kwargs))

        def one(doc_id,variant,warmup=False):
            from contextlib import nullcontext
            apply_plan(plans[variant],args.hf_home)
            _fresh_prompt(model)
            for layer in model.model.language_model.layers:
                if hasattr(layer.self_attn,"kv_cluster"):delattr(layer.self_attn,"kv_cluster")
            random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED)
            adapter=adapters[variant]
            attention=snap_forward(variant) if args.backbone=="snapkv" else nullcontext()
            with attention,adapter:
                adapter.begin_sample(args.task,doc_id)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                begin=time.perf_counter()
                batch,metadata=prepare_input(data,doc_id,processor)
                prepared=time.perf_counter()
                gpu={k:v.to("cuda:0") for k,v in batch.items()}
                torch.cuda.synchronize();transferred=time.perf_counter()
                torch.cuda.synchronize();probed=time.perf_counter()
                with torch.inference_mode():result=model.generate(**gpu,**kwargs)
                torch.cuda.synchronize();generated=time.perf_counter()
                ids=result.sequences[0,batch["input_ids"].shape[-1]:].tolist()
                text=processor.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                torch.cuda.synchronize();end=time.perf_counter()
                peak=torch.cuda.max_memory_allocated()/2**30
                audit=adapter.finish_sample()
                budget=active_prefill(result.past_key_values,len(ids),metadata["input_tokens"])
                fingerprints=tensor_fingerprints(batch)
                row=dict(status="completed",task=args.task,method=args.backbone if variant=="native" else args.backbone+"_kvrepack",
                    variant=variant,doc_id=doc_id,text=text,prediction=text,generated_ids=ids,generated_tokens=len(ids),
                    input_tokens=metadata["input_tokens"],eos_reached=bool(ids and ids[-1]==processor.tokenizer.eos_token_id),
                    capped_without_eos=len(ids)==max_new_tokens and ids[-1]!=processor.tokenizer.eos_token_id,
                    reference_metadata=metadata,source_video_id=metadata["source_video_id"],
                    input_fingerprints=fingerprints,audit=audit,budget=budget,fisher=None,
                    runtime=dict(preprocessing_seconds=prepared-begin,h2d_seconds=transferred-prepared,
                        fisher_seconds=0.,generation_seconds=generated-probed,text_decode_seconds=end-generated,
                        model_end_to_end_seconds=generated-transferred,end_to_end_seconds=end-begin,peak_allocated_gib=peak),
                    warmup=warmup,completed_utc=timestamp())
                del result,gpu
                path=args.output/("transform_"+variant+".json")
                if not path.exists():atomic_json(path,adapter.transform_spec)
                return row

        state.update(status="warming_up",warmup_per_arm=args.warmup);atomic_json(state_path,state)
        warm_started=time.perf_counter()
        for doc_id in range(args.warmup):
            for variant in VARIANTS:one(doc_id,variant,warmup=True)
        state["warmup_seconds"]=time.perf_counter()-warm_started
        formal=time.perf_counter()
        for doc_id in range(initial,args.limit):
            order=VARIANTS[doc_id%2:]+VARIANTS[:doc_id%2];records={}
            for variant in order:
                state.update(status="running",current_doc_id=doc_id,current_variant=variant,updated_utc=timestamp())
                atomic_json(state_path,state);records[variant]=one(doc_id,variant)
                print(json.dumps(dict(event="arm_complete",backbone=args.backbone,doc_id=doc_id,variant=variant,runtime=records[variant]["runtime"])),flush=True)
            if sources()!=frozen:raise ValueError("Source changed during frozen baseline experiment")
            bundle=dict(status="completed",doc_id=doc_id,backbone=args.backbone,contract_sha256=identity,
                order=list(order),records=records,fairness=dict(status="passed",inputs="identical",budget="strict10_total"))
            validate_bundle(bundle,identity,args.backbone,doc_id)
            atomic_json(args.output/"samples"/f"{doc_id:06d}.json",bundle)
            n=doc_id+1;elapsed=time.perf_counter()-formal
            state.update(completed_docs=n,completed_generations=2*n,elapsed_formal_seconds=elapsed,
                estimated_remaining_seconds=elapsed/(n-initial)*(args.limit-n),updated_utc=timestamp())
            atomic_json(state_path,state)
        state.update(status="complete",finished_utc=timestamp(),estimated_remaining_seconds=0,process_seconds=time.perf_counter()-started)
        atomic_json(state_path,state)
        return 0
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}",finished_utc=timestamp());atomic_json(state_path,state)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone",choices=("snapkv","hybridkv"),required=True)
    parser.add_argument("--task",choices=("vatex_test","nextqa_oe_test"),default="vatex_test")
    for name in ("pretrained","hf-home","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--limit",type=int,choices=(1,2,200),required=True)
    parser.add_argument("--warmup",type=int,choices=(0,1,2),default=2)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    for name in ("pretrained","hf_home","output"):setattr(args,name,getattr(args,name).expanduser().resolve())
    with output_lock(args.output,resume=args.resume):return run(args)


if __name__=="__main__":raise SystemExit(main())

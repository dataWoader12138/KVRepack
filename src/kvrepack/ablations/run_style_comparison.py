"""Matched native video merge aggregation and end-to-end latency experiment.

One process owns one backbone. Each original input is independently prepared
for both aggregation arms, whose execution order rotates by row. Source compilation,
model loading, warmup, audit hashing, JSON writes and scoring are untimed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

from .run_head_selection import (FRAMES, MIN_PIXELS, MAX_PIXELS, MODEL_SEED,
    atomic_json, json_sha, sha256, tensor_fingerprints, timestamp)
from .run_head_selection_full import output_lock, _fresh_prompt

VARIANTS = ("lookm", "asymkv")
ALL_VARIANTS = ("kvrepack", "lookm", "asymkv")
METHODS = ("snapkv_kvrepack", "hybridkv_kvrepack")


from .video_e2e_inputs import prepare_input


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--task", choices=("nextqa_oe_test", "vatex_test"), required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(1, 2, 200), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--audit", action="store_true", help="Additional diagnostic pilot; not a formal timing result")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=list(VARIANTS))
    args = parser.parse_args(argv)
    if not 0 <= args.warmup <= 2:
        parser.error("warmup must be 0, 1 or 2")
    if len(set(args.variants)) != len(args.variants) or len(args.variants) < 2:
        parser.error("Choose at least two distinct variants")
    for name in ("pretrained", "hf_home", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def sources():
    from .run_merge_comparison import sources as shared_sources
    result = shared_sources()
    root = Path(__file__).resolve().parents[3]
    for name in ("ablations/run_style_comparison.py", "ablations/video_e2e_inputs.py"):
        relative = "src/kvrepack/"+name
        result[relative] = sha256(root/relative)
    return result


from .run_merge_comparison import active_prefill, pairing_signature


def run(args):
    from .run_rope_merge import configure
    plan = configure(args)
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from ..observations.inputs import load_task
    from .merge_operators import collect_raw_key_fisher
    from .merge_snap import SnapMergeAdapter
    from .merge_hybrid import MergeHybridAdapter

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    data = load_task(args.hf_home, args.task)
    max_new_tokens = 128 if args.task == "nextqa_oe_test" else 64
    frozen_sources = sources()
    contract = dict(schema_version=1, experiment="merge_aggregation_latency",
        method=args.method, task=args.task, target_docs=args.limit,
        target_generations=args.limit*len(args.variants), variants=list(args.variants), frames=FRAMES,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, model_snapshot=str(args.pretrained),
        hf_home=str(args.hf_home), budget_ratio=.1, k_space="raw_reapply",
        budget_policy="total_active_prefill_including_local_global_cap_v1",
        algorithm_env=plan.resolved_config["algorithm_env"], max_new_tokens=max_new_tokens,
        repetition_penalty=1.05, do_sample=False, dtype="bfloat16", model_seed=MODEL_SEED,
        arrow_sha256=data["arrow_sha256"], dataset_rows=len(data["dataset"]),
        task_yaml_sha256=data["task_yaml_sha256"], task_utils_sha256=data["task_utils_sha256"],
        source_sha256=frozen_sources, warmup_per_arm=args.warmup,
        diagnostic_audit=args.audit, fisher_prompt_tail_tokens=32,
        fisher_probe_policy="native_no_grad_prefix_cache_raw_leaves_text_suffix_backward_v2",
        model_metadata_sha256={p.name:sha256(p) for p in sorted(args.pretrained.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt"}},
        versions={name:importlib.metadata.version(name) for name in
            ("torch", "transformers", "flash-attn", "qwen-vl-utils", "datasets", "decord", "numpy")},
        timing_scope="video read/decode+processor+H2D+Fisher(if AsymKV)+generate+text decode; synchronized GPU; excludes model load/warmup/audit hashing/writes/scoring",
        order_policy="rotate requested variants by original doc_id modulo arm count; no historical reuse")
    identity = json_sha(contract)
    contract_path = args.output/"run_contract.json"
    if args.resume:
        previous = json.loads((args.output/"run_state.json").read_text())
        if previous.get("resume_authorized") is False or previous.get("status") == "stopped_by_user":
            raise ValueError("Stopped runs require new explicit authorization")
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Frozen run contract changed")
    else:
        atomic_json(contract_path, contract)
    paths = sorted((args.output/"samples").glob("*.json"))
    if len(paths) > args.limit or [p.name for p in paths] != [f"{i:06d}.json" for i in range(len(paths))]:
        raise ValueError("Resume requires a complete contiguous prefix")
    for i, path in enumerate(paths):
        item = json.loads(path.read_text())
        if item.get("contract_sha256") != identity or item.get("doc_id") != i:
            raise ValueError("Committed sample identity differs")
    initial = len(paths)
    state = dict(status="loading", method=args.method, target_docs=args.limit,
        completed_docs=initial, target_generations=len(args.variants)*args.limit,
        completed_generations=len(args.variants)*initial,
        reused_docs=0, task=args.task,
        started_utc=timestamp(), pid=os.getpid(), contract_sha256=identity)
    state_path = args.output/"run_state.json"
    atomic_json(state_path, state)
    started = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(str(args.pretrained), local_files_only=True,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        if args.method == "hybridkv_kvrepack":
            from ..modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            Adapter = MergeHybridAdapter
        else:
            from ..monkeypatch import replace_qwen
            replace_qwen("snapkv_kvrepack")
            from transformers import Qwen2_5_VLForConditionalGeneration
            Adapter = SnapMergeAdapter
        load_started = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained),
            torch_dtype=torch.bfloat16, device_map=0, attn_implementation="flash_attention_2",
            local_files_only=True).eval()
        model.requires_grad_(False)
        torch.cuda.synchronize()
        state["model_load_seconds"] = time.perf_counter()-load_started
        kwargs = dict(eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id, do_sample=False,
            temperature=None, top_p=None, num_beams=1, max_new_tokens=max_new_tokens,
            repetition_penalty=1.05, use_cache=True,
            return_dict_in_generate=True, output_scores=False)
        adapters = {v: Adapter(v, budget_ratio=.1, audit=args.audit) for v in args.variants}
        atomic_json(args.output/"environment.json", dict(
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            transformers=importlib.metadata.version("transformers"),
            flash_attn=importlib.metadata.version("flash-attn"), generation_kwargs=kwargs))

        def one(doc_id, variant, warmup=False):
            _fresh_prompt(model)
            for layer in model.model.language_model.layers:
                if hasattr(layer.self_attn, "kv_cluster"):
                    delattr(layer.self_attn, "kv_cluster")
            random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
            adapter = adapters[variant]
            with adapter:
                adapter.begin_sample(args.task, doc_id)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                batch, metadata = prepare_input(data, doc_id, processor)
                prepared = time.perf_counter()
                gpu = {k:v.to("cuda:0") for k,v in batch.items()}
                torch.cuda.synchronize(); transferred = time.perf_counter()
                fisher_stats = None
                if variant == "asymkv":
                    gradients, fisher_stats = collect_raw_key_fisher(
                        model, gpu, loss_tail_tokens=32, parameters_already_frozen=True)
                    adapter.set_fisher(gradients)
                torch.cuda.synchronize(); probed = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(**gpu, **kwargs)
                torch.cuda.synchronize(); generated = time.perf_counter()
                ids = result.sequences[0, batch["input_ids"].shape[-1]:].tolist()
                text = processor.batch_decode([ids], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]
                torch.cuda.synchronize(); end = time.perf_counter()
                peak = torch.cuda.max_memory_allocated()/2**30
                # All tensor-to-CPU hashes and diagnostics start after `end`.
                audit = adapter.finish_sample()
                budget = active_prefill(result.past_key_values, len(ids), metadata["input_tokens"])
                fingerprints = tensor_fingerprints(batch)
                row = dict(status="completed", task=args.task, method=args.method,
                    variant=variant, doc_id=doc_id, text=text, prediction=text,
                    generated_ids=ids, generated_tokens=len(ids), input_tokens=metadata["input_tokens"],
                    eos_reached=bool(ids and ids[-1] == processor.tokenizer.eos_token_id),
                    capped_without_eos=len(ids)==max_new_tokens and ids[-1]!=processor.tokenizer.eos_token_id,
                    reference_metadata=metadata, source_video_id=metadata["source_video_id"],
                    input_fingerprints=fingerprints, audit=audit, budget=budget,
                    fisher=fisher_stats, runtime=dict(
                        preprocessing_seconds=prepared-begin, h2d_seconds=transferred-prepared,
                        fisher_seconds=probed-transferred if variant=="asymkv" else 0.,
                        generation_seconds=generated-probed, text_decode_seconds=end-generated,
                        model_end_to_end_seconds=generated-transferred,
                        end_to_end_seconds=end-begin, peak_allocated_gib=peak),
                    warmup=warmup, completed_utc=timestamp())
                del result, gpu
                if variant == "asymkv":
                    adapter.set_fisher(None)
                    del gradients
                spec_path = args.output/("transform_"+variant+".json")
                if not spec_path.exists():
                    atomic_json(spec_path, adapter.transform_spec)
                return row

        state.update(status="warming_up", warmup_per_arm=args.warmup)
        atomic_json(state_path, state)
        warm_started = time.perf_counter()
        for doc_id in range(args.warmup):
            for variant in args.variants:
                one(doc_id, variant, warmup=True)
        state["warmup_seconds"] = time.perf_counter()-warm_started
        formal_started = time.perf_counter()
        for doc_id in range(initial, args.limit):
            offset = doc_id % len(args.variants)
            order = args.variants[offset:]+args.variants[:offset]
            records = {}
            for variant in order:
                state.update(status="running", current_doc_id=doc_id,
                    current_variant=variant, updated_utc=timestamp())
                atomic_json(state_path, state)
                records[variant] = one(doc_id, variant)
                print(json.dumps(dict(event="arm_complete", method=args.method, doc_id=doc_id,
                    variant=variant, runtime=records[variant]["runtime"])), flush=True)
            reference = records[args.variants[0]]
            reference_signature = pairing_signature(reference["audit"], args.method)
            for row in records.values():
                if (row["reference_metadata"] != reference["reference_metadata"] or
                    row["input_fingerprints"] != reference["input_fingerprints"] or
                    row["budget"]["prefill_tokens_total"] != reference["budget"]["prefill_tokens_total"]):
                    raise ValueError("Shared inputs/budgets differ across merge aggregators")
                if pairing_signature(row["audit"], args.method) != reference_signature:
                    raise ValueError("Actual candidates/selected heads/groups/anchors differ across aggregators")
            if sources() != frozen_sources:
                raise ValueError("Source changed during the frozen experiment")
            bundle = dict(status="completed", doc_id=doc_id, method=args.method,
                task=args.task, contract_sha256=identity, order=list(order), records=records,
                fairness=dict(inputs="identical", final_budget="identical", status="passed",
                    groups_heads_candidates="identical", signature_sha256=json_sha(reference_signature)))
            atomic_json(args.output/"samples"/f"{doc_id:06d}.json", bundle)
            n = doc_id+1
            elapsed = time.perf_counter()-formal_started
            state.update(completed_docs=n, completed_generations=n*len(args.variants),
                elapsed_formal_seconds=elapsed,
                estimated_remaining_seconds=elapsed/(n-initial)*(args.limit-n))
            atomic_json(state_path, state)
        state.update(status="complete", finished_utc=timestamp(),
            process_seconds=time.perf_counter()-started, estimated_remaining_seconds=0)
        atomic_json(state_path, state)
        return 0
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
        atomic_json(state_path, state)
        raise


def main(argv=None):
    args = parse_args(argv)
    with output_lock(args.output, resume=args.resume):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

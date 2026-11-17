"""Full-annotation Qwen7B image experiments with image-read-to-text timing.

Uses b9104d8's original-budget aggregators and original standalone algorithms.
No algorithm file is modified. Audit hashing, persistence and scoring are untimed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

from .run_head_selection import atomic_json, cache_shapes, json_sha, sha256, tensor_fingerprints, timestamp

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("hybridkv_asymkv", "hybridkv_lookm", "hybridkv_kvrepack",
        "snapkv_asymkv", "snapkv_lookm", "snapkv_kvrepack", "asymkv", "lookm")
TASKS = ("Spot-the-Diff", "MMCoQA")


def arm_spec(arm):
    if arm not in ARMS:
        raise ValueError("Unknown styled image experiment arm")
    if arm in ("asymkv", "lookm"):
        return {"arm": arm, "method": arm, "backbone": None, "variant": None,
                "standalone": True}
    backbone, variant = arm.split("_", 1)
    return {"arm": arm, "method": backbone + "_kvrepack", "backbone": backbone,
            "variant": variant, "standalone": False}


def generation_policy(arm, maximum=512):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_new_tokens must be positive")
    result = dict(max_new_tokens=maximum, do_sample=False, temperature=None,
                  top_p=None, num_beams=1, use_cache=True,
                  return_dict_in_generate=True, output_scores=False)
    if arm != "asymkv":
        result.update(min_new_tokens=1, repetition_penalty=1.05)
    return result


def source_files():
    paths = sorted((ROOT / "src/kvrepack").rglob("*.py"))
    paths += sorted((ROOT / "src/kvrepack").rglob("*.json"))
    paths += [ROOT / "scripts/run_styled_milebench.py", ROOT / "provenance/core_sources.json",
              ROOT / "provenance/patches/lookm_strict_budget_v2.patch"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def configure(args):
    from ..experiments.runner import build_plan
    from .run_baseline_completion import apply_plan
    spec = arm_spec(args.arm)
    overrides = ()
    if spec["backbone"] == "hybridkv":
        overrides = ("HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO=.50", "HYBRID_KVREPACK_COMPRESS_RATIO=.50",
                     "K_DIFF_ALPHA=2", "HYBRID_KVREPACK_SIMHEAD_SIM_SPACE=raw",
                     "HYBRID_KVREPACK_K_MERGE_SPACE=raw_reapply")
    elif spec["backbone"] == "snapkv":
        overrides = ("SNAPKV_SIMHEAD_MERGE_RATIO=.30", "SNAPKV_KVREPACK_COMPRESS_RATIO=.70", "K_DIFF_ALPHA=2")
    # The existing planner supplies algorithm flags only. The image task and
    # image protocol are frozen separately; no video inputs are prepared.
    plan = build_plan(spec["method"], task="nextqa_oe_test", pretrained=str(args.pretrained),
                      model_type="qwen2_5_vl-7b", frames=64, limit=1, budget_ratio=.1,
                      output=args.output, overrides=overrides)
    apply_plan(plan, args.hf_home)
    os.environ.update(BUDGET="512", BUDGET_RATIO="0.1", RATIO="0.1")
    for key in ("ASYMKV_OFFICIAL_CACHE_SIZE", "ASYMKV_AUDIT_PATH"):
        os.environ.pop(key, None)
    os.environ.update(ASYMKV_START_SIZE="32", ASYMKV_CHUNK_SIZE="512")
    algorithm = dict(plan.resolved_config["algorithm_env"], BUDGET="512", BUDGET_RATIO="0.1", RATIO="0.1")
    if args.arm == "asymkv":
        algorithm.update(ASYMKV_START_SIZE="32", ASYMKV_CHUNK_SIZE="512")
    return spec, algorithm


def first_eos_index(ids, eos):
    return next((index for index, value in enumerate(ids) if value == eos), None)


def select_annotation(data_root, task, limit=None):
    from ..benchmarks.milebench.dataset import annotation_path, load_annotation, validate_data
    complete = load_annotation(data_root, task, limit=None)
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(complete["data"])):
        raise ValueError("Explicit prefix must fit the complete annotation")
    annotation = load_annotation(data_root, task, limit=limit)
    preflight = validate_data(data_root, task, limit=limit, verify_images=True)
    if not preflight["valid"]:
        raise ValueError("Image preflight failed: " + "; ".join(preflight["errors"]))
    path = annotation_path(data_root, task)
    if sha256(path) != preflight["annotation_sha256"]:
        raise ValueError("Annotation changed during preflight")
    if [row["sample_id"] for row in annotation["data"]] != preflight["sample_ids"]:
        raise ValueError("Selected image IDs changed during preflight")
    # Preserve the source grouping by number of images, after ID selection.
    groups = {}
    for row in annotation["data"]:
        groups.setdefault(len(row["task_instance"]["images_path"]), []).append(row)
    ordered = [row for group in groups.values() for row in group]
    return annotation, ordered, preflight


def fresh_prompt(model):
    from .run_head_selection_full import _fresh_prompt
    _fresh_prompt(model)
    for layer in model.model.language_model.layers:
        if hasattr(layer.self_attn, "kv_cluster"):
            delattr(layer.self_attn, "kv_cluster")


def run(args):
    annotation, ordered, preflight = select_annotation(args.data_root, args.task, args.limit)
    spec, algorithm = configure(args)
    from ..benchmarks.milebench.dataset import MileBenchDataset
    from ..benchmarks.milebench.protocol import DEFAULT_IMAGE_CONFIG, qwen_messages
    from ..benchmarks.legacy_lookm import LegacyLookMContext, source_identity
    from .original_budget import (OriginalHybridMergeAdapter, OriginalSnapMergeAdapter,
                                  active_prefill_original)
    from .merge_operators import collect_raw_key_fisher
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    frozen = source_files()
    image = dict(DEFAULT_IMAGE_CONFIG)
    effective_generation = generation_policy(args.arm, args.max_new_tokens)
    contract = dict(schema_version=1, experiment="styled_milebench_image_response_e2e",
        source_commit="b9104d8", **spec, task=args.task, data_root=str(args.data_root),
        all_samples=args.limit is None, limit=len(ordered), annotation_total_samples=preflight["total_samples"],
        selected_sample_ids=preflight["sample_ids"], generation_sample_ids=[v["sample_id"] for v in ordered],
        annotation_sha256=preflight["annotation_sha256"], pretrained=str(args.pretrained), hf_home=str(args.hf_home),
        model_type="qwen2_5_vl-7b", head_geometry={"layers": 28, "native_kv_heads": 4},
        model_seed=42, dtype="bfloat16", attention="flash_attention_2", image=image,
        generation=effective_generation, algorithm_env=algorithm, budget_ratio=.1,
        budget_policy=("original_asymkv_official_v1" if args.arm == "asymkv" else
                       "original_lookm_history_plus_recent_v1" if args.arm == "lookm" else
                       "original_history_plus_local_v1"),
        decoding_policy="official_argmax_no_repetition_penalty" if args.arm == "asymkv" else "hf_greedy_repetition_penalty_1p05",
        warmup_per_arm=args.warmup, source_sha256=frozen,
        model_metadata_sha256={p.name: sha256(p) for p in sorted(args.pretrained.iterdir())
                               if p.is_file() and p.suffix in {".json", ".txt"}},
        legacy_lookm_identity=source_identity() if args.arm == "lookm" else None,
        versions={name: importlib.metadata.version(name) for name in
                  ("torch", "transformers", "flash-attn", "qwen-vl-utils", "numpy", "Pillow", "rouge")},
        timing_scope="dataset request/image reads+RGB JPEG+processor+H2D+Fisher if applicable+native generate+text decode; synchronized GPU; excludes model loading/warmup/audit hashes/writes/scoring",
        stage_semantics={"model_end_to_end_seconds": "Fisher plus generate; excludes preprocessing/H2D/text decode",
                         "model_load_seconds": "processor plus model setup/loading, outside response timing",
                         "external_fisher_seconds": "separately timed prerequisite probe; zero for standalone AsymKV whose original probes are inside generate",
                         "fisher_seconds": "null for original standalone AsymKV because probes are inseparable from generation_seconds; external probe duration for styled AsymKV"},
        asymkv_boundary_policy="Preserve released chunk loop, record any missing final prompt token; no padding/skips/algorithm edits" if args.arm == "asymkv" else None)
    if spec["backbone"] == "hybridkv":
        import kvrepack_kernel
        binary = Path(kvrepack_kernel.__file__).resolve()
        contract["kernel_binary"] = dict(path=str(binary), sha256=sha256(binary))
    identity = json_sha(contract)
    state = dict(status="loading", arm=args.arm, task=args.task, pid=os.getpid(),
                 target_samples=len(ordered), completed_samples=0, completed_generations=0,
                 target_generations=len(ordered), contract_sha256=identity, started_utc=timestamp())
    state_path = args.output / "run_state.json"
    atomic_json(args.output / "run_contract.json", contract)
    atomic_json(state_path, state)
    started = time.perf_counter()
    capture_module = original_append = None
    try:
        load_begin = time.perf_counter()
        processor = AutoProcessor.from_pretrained(str(args.pretrained), local_files_only=True,
                       use_fast=False, min_pixels=image["min_pixels"], max_pixels=image["max_pixels"])
        if spec["backbone"] == "hybridkv":
            from ..modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        else:
            from ..monkeypatch import replace_qwen
            replace_qwen("asymkv_official" if args.arm == "asymkv" else
                         "lookm" if args.arm == "lookm" else "snapkv_kvrepack")
            from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained),
                    torch_dtype=torch.bfloat16, device_map=0, attn_implementation="flash_attention_2",
                    local_files_only=True).eval()
        model.requires_grad_(False)
        if model.dtype != torch.bfloat16:
            raise ValueError("Actual model dtype is not bfloat16")
        if (model.config.text_config.num_hidden_layers, model.config.text_config.num_key_value_heads) != (28, 4):
            raise ValueError("Expected Qwen7B 28 layers / 4 native KV heads")
        torch.cuda.synchronize()
        state["model_load_seconds"] = time.perf_counter() - load_begin
        kwargs = dict(effective_generation, eos_token_id=processor.tokenizer.eos_token_id,
                      pad_token_id=processor.tokenizer.pad_token_id)
        atomic_json(args.output / "environment.json", dict(gpu=torch.cuda.get_device_name(0),
            dtype=str(model.dtype), model_generation_config=model.generation_config.to_dict(),
            effective_generation_kwargs=kwargs, resolved_revision=getattr(model.config, "_commit_hash", None)))
        dataset = MileBenchDataset(ordered, annotation["meta_data"]["task_instruction"],
                    str(args.data_root / args.task / "images"), image["max_context_len"],
                    image["n_tokens_per_image"], processor.tokenizer, args.task)
        adapter = None
        if not spec["standalone"]:
            adapter_class = OriginalHybridMergeAdapter if spec["backbone"] == "hybridkv" else OriginalSnapMergeAdapter
            adapter = adapter_class(spec["variant"], budget_ratio=.1, audit=False)
        captures = []
        if args.arm == "asymkv":
            from ..asymkv_official import generation as capture_module
            original_append = capture_module._append_audit
            capture_module._append_audit = lambda path, record: captures.append(record)
        legacy = LegacyLookMContext() if args.arm == "lookm" else nullcontext()

        def one(index, warmup=False):
            fresh_prompt(model)
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            captures.clear()
            context = adapter if adapter is not None else nullcontext()
            with context:
                if adapter is not None:
                    adapter.begin_sample(args.task, ordered[index]["sample_id"])
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                sample = dataset[index]
                if not sample["raw_img_list"]:
                    raise ValueError("Image truncation removed all images")
                messages = qwen_messages(sample["context"], sample["raw_img_list"], image)
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = process_vision_info([messages])
                try:
                    encoded = processor(text=[prompt], images=images, videos=videos,
                                        padding=True, return_tensors="pt")
                finally:
                    for visual in images or ():
                        if hasattr(visual, "close"):
                            visual.close()
                os.environ["QUESTION"] = sample["context"].replace("<ImageHere>", "").strip()
                batch = dict(encoded)
                prepared = time.perf_counter()
                gpu = {key: value.to("cuda:0") for key, value in batch.items()}
                torch.cuda.synchronize(); transferred = time.perf_counter()
                gradients = fisher = None
                probed = transferred
                if spec["variant"] == "asymkv":
                    gradients, fisher = collect_raw_key_fisher(model, gpu,
                                    loss_tail_tokens=32, parameters_already_frozen=True)
                    adapter.set_fisher(gradients)
                    torch.cuda.synchronize(); probed = time.perf_counter()
                with (nullcontext() if args.arm == "asymkv" else torch.inference_mode()):
                    result = model.generate(**gpu, **kwargs)
                torch.cuda.synchronize(); generated = time.perf_counter()
                sequence = result.sequences if hasattr(result, "sequences") else result
                input_tokens = int(batch["input_ids"].shape[-1])
                ids = sequence[0, input_tokens:].tolist()
                answer = processor.batch_decode([ids], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0].strip()
                torch.cuda.synchronize(); end = time.perf_counter()
                # Everything below is after image-read-to-text timing stops.
                audit = adapter.finish_sample() if adapter is not None else None
                budget = None
                if adapter is not None:
                    budget = active_prefill_original(result.past_key_values, len(ids), input_tokens,
                                                    spec["backbone"], audit=audit)
                    transform = args.output / "adapter_transform.json"
                    if not transform.exists():
                        atomic_json(transform, adapter.transform_spec)
                elif args.arm == "asymkv":
                    if len(captures) != 1:
                        raise ValueError("Original AsymKV did not emit exactly one audit")
                    audit = captures.pop()
                    if audit["prompt_length"] != input_tokens:
                        raise ValueError("Original AsymKV audit input length differs")
                    physical = sum(audit["prefill_physical_lengths"]) * 4
                    budget = dict(prefill_tokens_total=physical,
                                  actual_retained_ratio=physical/(112*input_tokens),
                                  policy="original_asymkv_official_v1")
                else:
                    shapes = cache_shapes(result.past_key_values)
                    physical = sum(__import__("math").prod(row["key_shape"][:-1]) for row in shapes) - 112*(len(ids)-1)
                    expected = 112*(int(.1*input_tokens)+32)
                    if physical != expected:
                        raise ValueError("Original LOOK-M history-plus-recent budget changed")
                    audit = dict(final_cache_shapes=shapes, physical_prefill_positions=physical)
                    budget = dict(prefill_tokens_total=physical, actual_retained_ratio=physical/(112*input_tokens),
                                  policy="original_lookm_history_plus_recent_v1")
                eos = processor.tokenizer.eos_token_id
                row = dict(schema_version=1, status="completed", arm=args.arm, **{key:spec[key] for key in ("method", "backbone", "variant", "standalone")},
                    task=args.task, sample_index=index, sample_id=sample["sample_id"], doc_id=sample["sample_id"],
                    contract_sha256=identity, warmup=warmup,
                    prediction=dict(sample_id=sample["sample_id"], gt_response=sample["response"],
                        pred_response=answer, image=sample["raw_img_list"], question=sample["context"],
                        gen_model_id="qwen2.5-vl-7b", gen_kwargs=effective_generation),
                    input_tokens=input_tokens, input_fingerprints=tensor_fingerprints(batch),
                    generated_ids=ids, generated_tokens=len(ids), eos_token_id=eos,
                    eos_reached=bool(ids and ids[-1] == eos), first_eos_index=first_eos_index(ids, eos),
                    hit_max_new_tokens=len(ids)>=args.max_new_tokens,
                    capped_without_eos=len(ids)>=args.max_new_tokens and not bool(ids and ids[-1]==eos),
                    audit=audit, budget=budget, fisher=fisher,
                    generation_includes_fisher=args.arm == "asymkv",
                    separately_timed_fisher=spec["variant"] == "asymkv",
                    runtime=dict(preprocessing_seconds=prepared-begin, h2d_seconds=transferred-prepared,
                        external_fisher_seconds=probed-transferred,
                        fisher_seconds=None if args.arm == "asymkv" else probed-transferred,
                        generation_seconds=generated-probed,
                        text_decode_seconds=end-generated, model_end_to_end_seconds=generated-transferred,
                        end_to_end_seconds=end-begin, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),
                    completed_utc=timestamp())
                if args.arm == "asymkv":
                    covered = int(audit["prefill_logical_tokens"])
                    row.update(asymkv_prompt_tokens_covered=covered,
                        asymkv_prompt_tokens_missing=input_tokens-covered,
                        asymkv_boundary_affected=(covered != input_tokens))
                if gradients is not None:
                    adapter.set_fisher(None)
                del result, gpu, gradients
                return row

        with legacy:
            if args.arm == "lookm":
                atomic_json(args.output / "legacy_lookm_transform.json", legacy.transform_spec)
            state.update(status="warming_up", warmup_per_arm=args.warmup)
            atomic_json(state_path, state)
            for index in range(args.warmup):
                warm = one(index % len(dataset), warmup=True)
                atomic_json(args.output / "warmup" / f"{index:06d}.json", warm)
            formal_begin = time.perf_counter()
            for index in range(len(dataset)):
                state.update(status="running", current_sample_id=ordered[index]["sample_id"],
                             current_sample_index=index, updated_utc=timestamp())
                atomic_json(state_path, state)
                row = one(index)
                if source_files() != frozen:
                    raise ValueError("Bound runtime source changed during the experiment")
                atomic_json(args.output / "samples" / f"{index:06d}.json", row)
                state.update(completed_samples=index+1, completed_generations=index+1,
                             formal_loop_seconds=time.perf_counter()-formal_begin)
                atomic_json(state_path, state)
                print(json.dumps(dict(event="sample_complete", arm=args.arm, task=args.task,
                                     sample_id=row["sample_id"], e2e=row["runtime"]["end_to_end_seconds"])), flush=True)
        state.update(status="complete", finished_utc=timestamp(), process_seconds=time.perf_counter()-started)
        atomic_json(state_path, state)
        return 0
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
        atomic_json(state_path, state)
        raise
    finally:
        if capture_module is not None:
            capture_module._append_audit = original_append


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    for name in ("data-root", "pretrained", "hf-home", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--limit", type=int, help="explicit smoke prefix; omitted reads the entire annotation")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    for name in ("data_root", "pretrained", "hf_home", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    from .run_head_selection_full import output_lock
    with output_lock(args.output):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

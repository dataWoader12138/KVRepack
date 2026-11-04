"""Stream an explicitly sized NExTQA prefix through four matched selectors.

One CPU input and one loaded model serve all four conditions. Only a complete,
validated four-condition sample is atomically committed. No tensor cache grows
with dataset size; explicit resume requires the identical frozen run contract.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from .run_head_selection import (FRAMES, MIN_PIXELS, MAX_PIXELS, MODEL_SEED,
    atomic_json, cache_shapes, json_sha, sha256, tensor_fingerprints, timestamp)

TASK = "nextqa_oe_test"
CONDITIONS = (("similarity", "similarity", 0),
              ("random_global_seed0", "random_global", 0),
              ("random_global_seed1", "random_global", 1),
              ("random_global_seed2", "random_global", 2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True,
                        help="Explicit original-row prefix, e.g. 200; never inferred or enlarged")
    parser.add_argument("--resume", action="store_true",
                        help="Validate the frozen contract and continue complete sample bundles")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    return args


@contextmanager
def output_lock(directory, *, resume=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this output directory") from error
        try:
            entries = [p for p in directory.iterdir() if p.name != ".run.lock"]
            if not resume and entries:
                raise ValueError("Use a fresh output directory or explicitly --resume")
            if resume and not (directory / "run_contract.json").is_file():
                raise ValueError("Resume requires an existing run_contract.json")
            stream.seek(0); stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "locked_utc": timestamp()}) + "\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _flatten(mask):
    if not isinstance(mask, list) or len(mask) != 1 or len(mask[0]) != 28:
        raise ValueError("Expected batch1, 28 layers and four native KV heads")
    if any(not isinstance(layer, list) or len(layer) != 4 or any(type(x) is not bool for x in layer) for layer in mask[0]):
        raise ValueError("Invalid native Boolean head mask")
    return [x for layer in mask[0] for x in layer]


def validate_bundle(bundle, *, doc_id, contract_sha256):
    """Validate both new and resumed commits without importing model packages."""
    if (bundle.get("schema_version") != 1 or bundle.get("status") != "completed"
            or bundle.get("task") != TASK or bundle.get("doc_id") != doc_id
            or bundle.get("contract_sha256") != contract_sha256):
        raise ValueError("Sample identity/status/contract mismatch")
    fingerprints, metadata, rows = bundle.get("input_fingerprints"), bundle.get("reference_metadata"), bundle.get("records")
    if (not isinstance(fingerprints, dict) or not fingerprints.get("input_hashes")
            or not isinstance(metadata, dict) or metadata.get("task") != TASK or metadata.get("doc_id") != doc_id
            or not isinstance(rows, dict) or set(rows) != {c[0] for c in CONDITIONS}):
        raise ValueError("Missing paired input metadata or one of four conditions")
    reference = rows["similarity"]
    paired = lambda config: {k: v for k, v in config.items() if k not in ("strategy", "mask_seed")}
    from .head_selection import _ranking
    for label, strategy, seed in CONDITIONS:
        row = rows[label]
        if (row.get("status") != "completed" or row.get("task") != TASK or row.get("doc_id") != doc_id
                or row.get("reference_metadata") != metadata or row.get("source_video_id") != metadata.get("source_video_id")
                or any(row.get(k) != v for k, v in fingerprints.items())
                or row.get("input_fingerprint_sha256") != json_sha(fingerprints)):
            raise ValueError("Condition input/source identity mismatch")
        ids = row.get("generated_ids")
        if (not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids)
                or row.get("generated_tokens") != len(ids) or not isinstance(row.get("text"), str)
                or row.get("text") != row.get("prediction")
                or row.get("eos_reached") != (ids[-1] == row.get("eos_token_id"))):
            raise ValueError("Invalid generated prediction/token record")
        config, audit = row.get("config", {}), row.get("audit", {})
        if (config.get("strategy") != strategy or config.get("mask_seed") != seed
                or config.get("stream_contract_sha256") != contract_sha256
                or paired(config) != paired(reference.get("config", {}))
                or audit.get("strategy") != strategy or audit.get("seed") != seed
                or audit.get("status") != "completed" or audit.get("task") != TASK or audit.get("doc_id") != doc_id
                or audit.get("layers") != 28 or audit.get("native_kv_heads") != 4 or audit.get("batch_size") != 1):
            raise ValueError("Condition configuration or selector audit mismatch")
        selected, baseline = _flatten(audit.get("selected_masks")), _flatten(audit.get("baseline_masks"))
        if (sum(selected) != 34 or sum(baseline) != 34 or audit.get("selected_count") != [34]
                or audit.get("baseline_selected_count") != [34]
                or audit.get("actual_merge_calls") != 34 or audit.get("expected_effective_merge_count") != 34):
            raise ValueError("Condition changed the selected/effectively merged head count")
        if strategy == "similarity" and selected != baseline:
            raise ValueError("Similarity mask changed")
        if strategy == "random_global":
            ranking = _ranking(seed, "global", 28, 4, 0).sample(range(112), 112)
            chosen = set(ranking[:34])
            if selected != [i in chosen for i in range(112)]:
                raise ValueError("Random mask differs from its fixed seed/model ranking")
        calls = audit.get("merge_calls", [])
        if (len(calls) != 34 or len({(c["layer"], c["batch"], c["head"]) for c in calls}) != 34
                or any(c["batch"] != 0 or not selected[c["layer"]*4+c["head"]] or c["before"] <= c["after"] for c in calls)
                or sum(c["before"]-c["after"] for c in calls) != audit.get("actual_removed_tokens")
                or audit.get("actual_removed_tokens") != audit.get("expected_removed_tokens")):
            raise ValueError("Actual merge work disagrees with the selected head set")
        for field in ("prefill_cache_shapes", "candidate_hashes"):
            values = audit.get(field, [])
            if len(values) != 28 or [v.get("layer") for v in values] != list(range(28)):
                raise ValueError("Incomplete layer cache/candidate audit")
        for field in ("baseline_masks", "baseline_selected_count", "candidate_hashes", "prefill_cache_shapes",
                      "expected_effective_merge_count", "expected_removed_tokens", "actual_merge_calls", "actual_removed_tokens"):
            if audit[field] != reference["audit"][field]:
                raise ValueError(f"Four-condition pairing failed: {field}")
        if len(row.get("final_cache_shapes", [])) != 28:
            raise ValueError("Incomplete final cache layer coverage")
    if bundle.get("fairness", {}).get("status") != "passed":
        raise ValueError("Sample lacks its completed fairness gate")
    return bundle


def load_resume_prefix(directory, *, target, contract_sha256):
    samples = Path(directory) / "samples"
    files = sorted(samples.glob("*.json"))
    if len(files) > target or [p.name for p in files] != [f"{i:06d}.json" for i in range(len(files))]:
        raise ValueError("Resume sample coverage is not a contiguous original-row prefix")
    for doc_id, path in enumerate(files):
        bundle = json.loads(path.read_text())
        validate_bundle(bundle, doc_id=doc_id, contract_sha256=contract_sha256)
    return len(files)


def _configure_environment(args):
    from kvrepack.experiments.runner import build_plan
    plan = build_plan("snapkv_kvrepack", task=TASK, pretrained=str(args.pretrained), model_type="qwen2_5_vl-7b",
                      frames=FRAMES, limit=args.limit, budget_ratio=.1, output=args.output,
                      overrides=("SNAPKV_SIMHEAD_MERGE_RATIO=.30", "SNAPKV_KVREPACK_COMPRESS_RATIO=.7", "K_DIFF_ALPHA=2"))
    for name in list(os.environ):
        if name.startswith(("SNAPKV_", "MADAKV_", "HYBRID_", "SPARSEMM_", "LOOKM_")) or name in {
                "METHOD", "MODEL_TYPE", "BUDGET", "BUDGET_RATIO", "RATIO", "K_DIFF_ALPHA"}:
            os.environ.pop(name)
    os.environ.update(plan.env)
    os.environ.update(HF_HOME=str(args.hf_home), HF_HUB_CACHE=str(args.hf_home / "hub"),
                      HUGGINGFACE_HUB_CACHE=str(args.hf_home / "hub"), HF_DATASETS_CACHE=str(args.hf_home / "datasets"),
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return plan


def _contract(args, data, plan):
    modules = ("kvrepack.ablations.run_head_selection", "kvrepack.ablations.head_selection",
               "kvrepack.observations.inputs", "kvrepack.monkeypatch", "kvrepack.experiments.runner",
               "kvrepack.experiments.registry", "kvrepack.qwen_model_snapkv_kvrepack")
    sources = {str(Path(__file__).resolve()): sha256(__file__)}
    for name in modules:
        path = Path(importlib.import_module(name).__file__).resolve()
        sources[str(path)] = sha256(path)
    model_metadata = {p.name: sha256(p) for p in sorted(args.pretrained.iterdir())
                      if p.is_file() and p.suffix in (".json", ".txt")}
    return dict(schema_version=1, task=TASK, target_docs=args.limit, target_generations=args.limit*4,
                dataset_rows=len(data["dataset"]), model_snapshot=str(args.pretrained), model_metadata=model_metadata,
                arrow_path=data["arrow_path"], arrow_sha256=data["arrow_sha256"], video_root=data["video_root"],
                task_yaml_sha256=data["task_yaml_sha256"], task_utils_sha256=data["task_utils_sha256"],
                included_yaml_sha256=data.get("included_yaml_sha256"), conditions=[list(c) for c in CONDITIONS],
                frames=FRAMES, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, model_seed=MODEL_SEED,
                budget_ratio=.1, algorithm_env=plan.resolved_config["algorithm_env"], dtype="bfloat16",
                attention="flash_attention_2", max_new_tokens=128, do_sample=False, use_cache=True,
                sources_sha256=sources, versions={name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "flash-attn", "qwen-vl-utils", "datasets", "decord", "numpy")},
                input_policy="Original prefix, one CPU preparation per row, same tensors for all four conditions; no persistent tensor cache",
                commit_policy="Atomic complete four-condition bundle; no partial sample is a committed result")


def _fresh_prompt(model):
    # The core also resets records at native layer0. These generation-local
    # fields are cleared explicitly so no previous condition retains its cache.
    for obj in (model, getattr(model, "model", None)):
        if obj is not None and hasattr(obj, "rope_deltas"):
            obj.rope_deltas = None
    for layer in model.model.language_model.layers:
        layer.self_attn.kv_seq_len = 0
        if hasattr(layer.self_attn.config, "_snapkv_kvrepack_records"):
            layer.self_attn.config._snapkv_kvrepack_records = []


def _run_locked(args):
    plan = _configure_environment(args)
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from kvrepack.observations import inputs
    from kvrepack.monkeypatch import replace_qwen
    from .head_selection import SnapHeadSelectionAdapter
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    torch.set_num_threads(4)
    data = inputs.load_task(args.hf_home, TASK)
    if args.limit > len(data["dataset"]):
        raise ValueError("Requested prefix exceeds the actual NExTQA test split")
    contract = _contract(args, data, plan)
    contract_path = args.output / "run_contract.json"
    if args.resume:
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Run contract changed; refusing to resume")
    else:
        atomic_json(contract_path, contract)
    identity = json_sha(contract)
    completed = load_resume_prefix(args.output, target=args.limit, contract_sha256=identity)
    state_path = args.output / "run_state.json"
    previous = json.loads(state_path.read_text()) if args.resume and state_path.is_file() else {}
    state = dict(status="loading", task=TASK, target_docs=args.limit, target_generations=args.limit*4,
                 completed_docs=completed, completed_generations=completed*4, contract_sha256=identity,
                 started_utc=previous.get("started_utc", timestamp()), updated_utc=timestamp(),
                 model_seed=MODEL_SEED, conditions=[c[0] for c in CONDITIONS],
                 resumed=bool(args.resume), estimated_remaining_seconds=None)
    atomic_json(state_path, state)
    started = time.monotonic(); initial_completed = completed
    try:
        if completed == args.limit:
            state.update(status="complete", finished_utc=timestamp()); atomic_json(state_path, state)
            return 0
        processor = AutoProcessor.from_pretrained(str(args.pretrained), local_files_only=True,
                                                  min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        replace_qwen("snapkv_kvrepack")
        module = importlib.import_module("kvrepack.qwen_model_snapkv_kvrepack")
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained), torch_dtype=torch.bfloat16,
                    device_map=0, attn_implementation="flash_attention_2", local_files_only=True).eval()
        generation_kwargs = dict(eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id, do_sample=False, temperature=None,
            top_p=None, num_beams=1, max_new_tokens=128, use_cache=True,
            return_dict_in_generate=True, output_scores=False)
        config = {k: contract[k] for k in ("task", "model_snapshot", "frames", "budget_ratio", "algorithm_env",
                   "min_pixels", "max_pixels", "dtype", "attention", "max_new_tokens", "use_cache", "do_sample", "model_seed")}
        config.update(model_type="qwen2_5_vl-7b", stream_contract_sha256=identity,
                      source_files=contract["sources_sha256"],
                      effective_generation_config=model.generation_config.to_dict(),
                      generation_overrides={k:v for k,v in generation_kwargs.items()})
        atomic_json(args.output / "run_config.json", config)
        for doc_id in range(completed, args.limit):
            state.update(status="preparing", current_doc_id=doc_id, current_strategy=None, updated_utc=timestamp())
            atomic_json(state_path, state)
            # Per-row preprocessing is deterministic; seed it independently of
            # condition execution/resumption so a resumed input has identical bytes.
            random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
            batch, metadata = inputs.prepare_sample(data, doc_id, processor, FRAMES, MIN_PIXELS, MAX_PIXELS)
            fingerprints = tensor_fingerprints(batch)
            fingerprint_sha = json_sha(fingerprints)
            gpu = {key: value.to("cuda:0") for key, value in batch.items()}
            rows = {}
            for label, strategy, seed in CONDITIONS:
                state.update(status="running", current_strategy=label, updated_utc=timestamp()); atomic_json(state_path, state)
                _fresh_prompt(model)
                random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
                with SnapHeadSelectionAdapter(module, strategy=strategy, seed=seed) as adapter:
                    adapter.set_sample(TASK, doc_id)
                    torch.cuda.reset_peak_memory_stats(); began = time.monotonic()
                    with torch.inference_mode():
                        result = model.generate(**gpu, **generation_kwargs)
                    torch.cuda.synchronize()
                    elapsed, audit = time.monotonic()-began, adapter.finish_sample()
                ids = result.sequences[0, batch["input_ids"].shape[-1]:].tolist()
                text = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                rows[label] = dict(task=TASK, doc_id=doc_id, id=f"{TASK}:{doc_id}", status="completed",
                    source_video_id=metadata["source_video_id"], video_id=metadata["video_id"], text=text, prediction=text,
                    generated_ids=ids, **fingerprints, input_fingerprint_sha256=fingerprint_sha,
                    reference_metadata=metadata, config=dict(config, strategy=strategy, mask_seed=seed), audit=audit,
                    final_cache_shapes=cache_shapes(result.past_key_values), eos_token_id=processor.tokenizer.eos_token_id,
                    eos_reached=bool(ids and ids[-1] == processor.tokenizer.eos_token_id), input_tokens=metadata["input_tokens"],
                    generated_tokens=len(ids), runtime=dict(generation_seconds=elapsed,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30), completed_utc=timestamp())
                del result
                print(json.dumps(dict(stage="condition_completed", doc_id=doc_id, condition=label,
                    actual_merge_calls=audit["actual_merge_calls"], generated_tokens=len(ids))), flush=True)
            if tensor_fingerprints(batch) != fingerprints:
                raise ValueError("Shared CPU input tensors changed during condition execution")
            bundle = dict(schema_version=1, status="completed", task=TASK, doc_id=doc_id, contract_sha256=identity,
                          input_fingerprints=fingerprints, reference_metadata=metadata, records=rows,
                          fairness=dict(status="passed", selected_heads=34, conditions=4,
                              checks="identical input/candidate hashes, parameters, actual merge count/work, prefill shapes"),
                          completed_utc=timestamp())
            validate_bundle(bundle, doc_id=doc_id, contract_sha256=identity)
            path = args.output / "samples" / f"{doc_id:06d}.json"
            if path.exists():
                raise ValueError("Refusing to overwrite a committed sample")
            atomic_json(path, bundle)
            completed += 1; elapsed_run = time.monotonic()-started
            state.update(completed_docs=completed, completed_generations=completed*4, updated_utc=timestamp(),
                elapsed_current_session_seconds=elapsed_run,
                estimated_remaining_seconds=elapsed_run/(completed-initial_completed)*(args.limit-completed))
            atomic_json(state_path, state)
            print(json.dumps(dict(stage="sample_committed", doc_id=doc_id, completed_docs=completed,
                                  target_docs=args.limit, estimated_remaining_seconds=state["estimated_remaining_seconds"])), flush=True)
            del rows, bundle, gpu, batch
        state.update(status="complete", finished_utc=timestamp(), estimated_remaining_seconds=0)
        atomic_json(state_path, state)
        return 0
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
        atomic_json(state_path, state)
        raise


def main(argv=None):
    args = parse_args(argv)
    for name in ("pretrained", "hf_home", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.pretrained.is_dir():
        raise FileNotFoundError(args.pretrained)
    with output_lock(args.output, resume=args.resume):
        return _run_locked(args)


if __name__ == "__main__":
    raise SystemExit(main())

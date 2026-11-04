"""One explicitly limited SnapKV-KVRepack head-selection experiment.

CPU preprocessing is cached once per task and reused byte-for-byte. This worker
does not launch other strategies, score predictions, download assets, or resume
partial generation runs. Model/GPU imports happen only in the selected run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile
import time


TASKS = ("nextqa_oe_test", "vatex_test")
STRATEGIES = ("similarity", "random_global", "random_layer_matched")
FRAMES, MIN_PIXELS, MAX_PIXELS, MODEL_SEED = 64, 200704, 1605632, 42


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".head-run-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_contract(path, expected):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Cache contract changed; select a fresh cache directory: {path}")
    else:
        atomic_json(path, expected)


def tensor_fingerprints(batch):
    import torch
    if not isinstance(batch, dict) or not batch:
        raise ValueError("CPU input cache requires a nonempty tensor dictionary")
    hashes, shapes, dtypes = {}, {}, {}
    for name, tensor in sorted(batch.items()):
        if not isinstance(name, str) or not torch.is_tensor(tensor) or tensor.device.type != "cpu":
            raise ValueError("Only named CPU tensors may enter the input cache")
        raw = tensor.detach().contiguous().view(torch.uint8).numpy()
        hashes[name] = hashlib.sha256(memoryview(raw)).hexdigest()
        shapes[name], dtypes[name] = list(tensor.shape), str(tensor.dtype)
    return {"input_hashes": hashes, "input_shapes": shapes, "input_dtypes": dtypes}


def load_cached_input(task_cache, doc_id, contract_sha):
    import torch
    task_cache = Path(task_cache)
    tensor_path, info_path = task_cache / f"{doc_id:06d}.pt", task_cache / f"{doc_id:06d}.json"
    if not tensor_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(f"Incomplete cached input for doc_id={doc_id}")
    info = json.loads(info_path.read_text())
    if info.get("contract_sha256") != contract_sha or info.get("doc_id") != doc_id:
        raise ValueError("Cached input contract/doc_id mismatch")
    if json_sha(info.get("metadata")) != info.get("metadata_sha256"):
        raise ValueError("Cached input source metadata SHA256 mismatch")
    if sha256(tensor_path) != info.get("pt_sha256"):
        raise ValueError("Cached input .pt SHA256 mismatch")
    batch = torch.load(tensor_path, map_location="cpu", weights_only=True)
    actual = tensor_fingerprints(batch)
    if any(actual[k] != info.get(k) for k in actual):
        raise ValueError("Cached input tensor bytes/shape/dtype mismatch")
    return batch, info["metadata"], info


def save_cached_input(task_cache, doc_id, batch, metadata, contract_sha):
    import torch
    task_cache = Path(task_cache)
    task_cache.mkdir(parents=True, exist_ok=True)
    tensor_path, info_path = task_cache / f"{doc_id:06d}.pt", task_cache / f"{doc_id:06d}.json"
    fingerprints = tensor_fingerprints(batch)
    if tensor_path.exists() or info_path.exists():
        _, previous_metadata, info = load_cached_input(task_cache, doc_id, contract_sha)
        if previous_metadata != metadata or any(info[k] != v for k, v in fingerprints.items()):
            raise ValueError("Refusing to replace different cached input bytes/metadata")
        return info
    fd, temporary = tempfile.mkstemp(prefix=".input-", suffix=".pt", dir=task_cache)
    os.close(fd)
    try:
        torch.save(batch, temporary)
        os.replace(temporary, tensor_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    info = dict(schema_version=1, doc_id=doc_id, contract_sha256=contract_sha,
                pt_sha256=sha256(tensor_path), metadata=metadata,
                metadata_sha256=json_sha(metadata), **fingerprints)
    atomic_json(info_path, info)
    return info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Head-mask seed only; model RNG remains42")
    parser.add_argument("--limit", type=int, default=20, help="Explicit prefix count; no automatic expansion")
    parser.add_argument("--pretrained", type=Path, required=True, help="Existing local Qwen2.5-VL-7B snapshot")
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Shared CPU preprocessing cache root")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output for this one task/strategy/seed")
    parser.add_argument("--prepare-only", action="store_true", help="CPU cache preparation only; GPUs are hidden")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.seed < 0:
        parser.error("--limit must be positive and --seed nonnegative")
    return args


def prepare_cache(args, processor):
    from kvrepack.observations import inputs
    data = inputs.load_task(args.hf_home, args.task)
    if args.limit > len(data["dataset"]):
        raise ValueError("Requested prefix exceeds the actual test split")
    source = Path(inputs.__file__).resolve()
    model_files = {p.name: sha256(p) for p in sorted(args.pretrained.iterdir())
                   if p.is_file() and p.suffix in {".json", ".txt"}}
    contract = dict(schema_version=1, task=args.task, model_snapshot=str(args.pretrained), model_metadata=model_files,
                    frames=FRAMES, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, preprocessing_seed=MODEL_SEED,
                    arrow_path=data["arrow_path"], arrow_sha256=data["arrow_sha256"], video_root=data["video_root"],
                    task_yaml_sha256=data["task_yaml_sha256"], task_utils_sha256=data["task_utils_sha256"],
                    included_yaml_sha256=data.get("included_yaml_sha256"),
                    sources_sha256={str(source): sha256(source), str(Path(__file__).resolve()): sha256(__file__)},
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "qwen-vl-utils", "datasets", "decord", "numpy")})
    directory = args.cache / args.task
    ensure_contract(directory / "contract.json", contract)
    identity = json_sha(contract)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "contract_sha256": identity, "records": {}}
    if manifest.get("contract_sha256") != identity:
        raise ValueError("Shared preprocessing manifest contract mismatch")
    for doc_id in range(args.limit):
        tensor_path, info_path = directory / f"{doc_id:06d}.pt", directory / f"{doc_id:06d}.json"
        if tensor_path.exists() or info_path.exists():
            batch, metadata, info = load_cached_input(directory, doc_id, identity)
        else:
            batch, metadata = inputs.prepare_sample(data, doc_id, processor, FRAMES, MIN_PIXELS, MAX_PIXELS)
            info = save_cached_input(directory, doc_id, batch, metadata, identity)
        if metadata.get("task") != args.task or metadata.get("doc_id") != doc_id:
            raise ValueError("Preprocessed source metadata task/doc_id mismatch")
        manifest["records"][str(doc_id)] = info
        manifest["prepared_count"] = len(manifest["records"])
        atomic_json(manifest_path, manifest)
        print(json.dumps({"stage": "cpu_input_ready", "task": args.task, "doc_id": doc_id,
                          "input_tokens": metadata["input_tokens"]}), flush=True)
        del batch
    return directory, identity, manifest_path


def cache_shapes(cache):
    if cache is None or not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
        raise ValueError("Generation did not return a supported final KV cache")
    return [dict(layer=i, key_shape=list(k.shape), value_shape=list(v.shape),
                 key_dtype=str(k.dtype), value_dtype=str(v.dtype))
            for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache))]


def append_record(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    args = parse_args(argv)
    for key in ("pretrained", "hf_home", "cache", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.pretrained.is_dir():
        raise FileNotFoundError(args.pretrained)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Select a fresh output directory; partial generation is never silently resumed")
    from kvrepack.experiments.runner import build_plan
    simtop, alpha = (".30", "2") if args.task == "nextqa_oe_test" else (".15", ".5")
    plan = build_plan("snapkv_kvrepack", task=args.task, pretrained=str(args.pretrained),
                      model_type="qwen2_5_vl-7b", frames=FRAMES, limit=args.limit, budget_ratio=.1,
                      output=args.output, overrides=(f"SNAPKV_SIMHEAD_MERGE_RATIO={simtop}",
                                                     "SNAPKV_KVREPACK_COMPRESS_RATIO=.7", f"K_DIFF_ALPHA={alpha}"))
    # Apply explicit algorithm/default policies before ML/preprocessor imports.
    for name in list(os.environ):
        if name.startswith(("SNAPKV_", "MADAKV_", "HYBRID_", "SPARSEMM_", "LOOKM_")) or name in {
                "METHOD", "MODEL_TYPE", "BUDGET", "BUDGET_RATIO", "RATIO", "K_DIFF_ALPHA"}:
            os.environ.pop(name)
    os.environ.update(plan.env)
    os.environ.update(HF_HOME=str(args.hf_home), HF_HUB_CACHE=str(args.hf_home / "hub"),
                      HUGGINGFACE_HUB_CACHE=str(args.hf_home / "hub"), HF_DATASETS_CACHE=str(args.hf_home / "datasets"),
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.prepare_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.output.mkdir(parents=True, exist_ok=True)
    state = dict(status="preparing", task=args.task, strategy=args.strategy, mask_seed=args.seed,
                 model_seed=MODEL_SEED, target=args.limit, completed=0, started_utc=timestamp())
    atomic_json(args.output / "run_state.json", state)
    try:
        import numpy as np
        import torch
        from transformers import AutoProcessor
        random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(str(args.pretrained), local_files_only=True,
                                                  min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        directory, identity, manifest_path = prepare_cache(args, processor)
        state["cache_manifest"] = str(manifest_path)
        if args.prepare_only:
            state.update(status="prepared", prepared=args.limit, finished_utc=timestamp())
            atomic_json(args.output / "run_state.json", state)
            return 0
        from kvrepack.monkeypatch import replace_qwen
        replace_qwen("snapkv_kvrepack")
        module = importlib.import_module("kvrepack.qwen_model_snapkv_kvrepack")
        from kvrepack.ablations.head_selection import SnapHeadSelectionAdapter
        from transformers import Qwen2_5_VLForConditionalGeneration
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state["status"] = "loading"
        atomic_json(args.output / "run_state.json", state)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.pretrained), torch_dtype=torch.bfloat16,
                    device_map=0, attn_implementation="flash_attention_2", local_files_only=True).eval()
        config = dict(task=args.task, model_snapshot=str(args.pretrained), model_type="qwen2_5_vl-7b",
                      frames=FRAMES, budget_ratio=.1, algorithm_env=plan.resolved_config["algorithm_env"],
                      min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, dtype="bfloat16", attention="flash_attention_2",
                      max_new_tokens=128, use_cache=True, do_sample=False, model_seed=MODEL_SEED,
                      strategy=args.strategy, mask_seed=args.seed, cache_contract_sha256=identity)
        config["source_files"] = {str(Path(m.__file__).resolve()): sha256(m.__file__) for m in (
            module, importlib.import_module("kvrepack.monkeypatch"),
            importlib.import_module("kvrepack.ablations.head_selection"))}
        atomic_json(args.output / "run_config.json", config)
        with SnapHeadSelectionAdapter(module, strategy=args.strategy, seed=args.seed,
                                      audit_path=args.output / "head_audit.jsonl") as adapter:
            for doc_id in range(args.limit):
                state.update(status="running", current_doc_id=doc_id, updated_utc=timestamp())
                atomic_json(args.output / "run_state.json", state)
                batch, metadata, info = load_cached_input(directory, doc_id, identity)
                if metadata["task"] != args.task or metadata["doc_id"] != doc_id:
                    raise ValueError("Generation input source task/doc_id mismatch")
                adapter.set_sample(args.task, doc_id)
                gpu = {key: value.to("cuda:0") for key, value in batch.items()}
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                with torch.inference_mode():
                    result = model.generate(**gpu, eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id, do_sample=False, temperature=None,
                        top_p=None, num_beams=1, max_new_tokens=128, use_cache=True,
                        return_dict_in_generate=True, output_scores=False)
                torch.cuda.synchronize()
                elapsed, audit = time.monotonic() - started, adapter.finish_sample()
                if audit.get("task") != args.task or audit.get("doc_id") != doc_id:
                    raise ValueError("Head-selection audit task/doc_id mismatch")
                ids = result.sequences[0, batch["input_ids"].shape[-1]:].tolist()
                text = processor.batch_decode([ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                row = dict(task=args.task, doc_id=doc_id, id=f"{args.task}:{doc_id}", status="completed",
                           source_video_id=metadata["source_video_id"], video_id=metadata["video_id"],
                           text=text, prediction=text, generated_ids=ids, input_hashes=info["input_hashes"],
                           input_shapes=info["input_shapes"], input_dtypes=info["input_dtypes"],
                           input_cache_sha256=info["pt_sha256"], reference_metadata=metadata, config=config,
                           audit=audit, final_cache_shapes=cache_shapes(result.past_key_values),
                           eos_token_id=processor.tokenizer.eos_token_id,
                           eos_reached=bool(ids and ids[-1] == processor.tokenizer.eos_token_id),
                           input_tokens=metadata["input_tokens"], generated_tokens=len(ids),
                           runtime={"generation_seconds": elapsed, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30},
                           completed_utc=timestamp())
                append_record(args.output / "records.jsonl", row)
                state["completed"] += 1
                atomic_json(args.output / "run_state.json", state)
                print(json.dumps({"stage": "completed", "task": args.task, "doc_id": doc_id,
                                  "strategy": args.strategy, "seed": args.seed, "generated_tokens": len(ids)}), flush=True)
                del result, gpu, batch
        state.update(status="complete", finished_utc=timestamp())
        atomic_json(args.output / "run_state.json", state)
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
        atomic_json(args.output / "run_state.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

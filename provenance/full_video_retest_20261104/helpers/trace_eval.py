#!/usr/bin/env python3
"""External instrumentation for the real lmms_eval adapter; no replacement algorithm.

Records full-vocabulary logit hashes and all-layer KV hashes, with synchronous
per-sample JSONL persistence. These added transfers invalidate timing benchmarks.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_records(path, runtime):
    """Never silently drop partial/corrupt records or duplicate IDs on resume."""
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"Incomplete/empty trace row {line_no}; inspect before resuming")
            row = json.loads(line)
            key = row["id"]
            if key in rows:
                raise ValueError(f"Duplicate trace ID {key!r}")
            if (row["config"] != runtime["config"] or row["config_sha256"] != runtime["config_sha256"]
                    or row["implementation"] != runtime["implementation"]):
                raise ValueError(f"Incompatible existing trace row {key!r}")
            rows[key] = row
    return rows


def request_id(args):
    if len(args) != 6:
        raise ValueError("Expected real Qwen adapter's six-field generation request")
    return f"{args[4]}:{int(args[3])}"


def install(runtime):
    # Imported only in the launched evaluation process. The launcher remains stdlib-only.
    import torch
    repo = Path(runtime["repo_root"]).resolve()
    source_root = Path(runtime["source_root"]).resolve()
    trace_path = Path(runtime["trace_path"])
    completed = read_records(trace_path, runtime)
    state = {"chunk": None, "active": False, "adapter": None, "sources": None}
    if runtime["implementation"] == "original":
        import kvrepack.monkeypatch as new_patch
        new_origin = Path(new_patch.__file__).resolve()
        if not new_origin.is_relative_to(repo / "src"):
            raise ValueError("Current dispatcher escaped this checkout")
        sys.path.insert(0, str(source_root))
        try:
            old_patch = importlib.import_module("hybridkv.monkeypatch")
        finally:
            sys.path.remove(str(source_root))
        if not Path(old_patch.__file__).resolve().is_relative_to(source_root):
            raise ValueError("Original dispatcher escaped explicitly selected artifact tree")

        def translate_replace(method):
            expected = runtime["config"]["method"]
            if method != expected or method not in {"snapkv_kvrepack", "madakv_kvrepack"}:
                raise ValueError(f"Unexpected original-control method {method!r}")
            for key, value in runtime["config"]["algorithm_env"].items():
                os.environ[key.replace("KVREPACK", "DECOUPLE")] = value
            old_method = method.replace("_kvrepack", "_decouplekv")
            os.environ["METHOD"] = old_method
            old_patch.replace_qwen(old_method)

        new_patch.replace_qwen = translate_replace

    from lmms_eval import utils
    from lmms_eval.models.qwen2_5_vl import Qwen2_5_VL
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as hf
    if not Path(inspect.getfile(Qwen2_5_VL)).resolve().is_relative_to(repo / "vendor"):
        raise ValueError("Adapter is not this checkout's real vendored adapter")
    original_batched = utils.Collator.get_batched
    original_generate_until = Qwen2_5_VL.generate_until
    original_generate = hf.Qwen2_5_VLForConditionalGeneration.generate

    def digest(tensor):
        value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
        return hashlib.sha256(value.tobytes()).hexdigest()

    def cache_snapshot(cache):
        if cache is None or not hasattr(cache, "key_cache"):
            raise ValueError("Expected the real compression path's populated DynamicCache")
        rows = []
        if len(cache.key_cache) != len(cache.value_cache):
            raise ValueError("Mismatched key/value cache layer counts")
        for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
            rows.append({"layer": layer, "key": list(key.shape), "value": list(value.shape),
                         "key_sha256": digest(key), "value_sha256": digest(value)})
        return rows

    def current_sources():
        sources = {}
        prefix = "hybridkv" if runtime["implementation"] == "original" else "kvrepack"
        for name, module in list(sys.modules.items()):
            if name == prefix or name.startswith(prefix + "."):
                path = getattr(module, "__file__", None)
                if path and str(path).endswith(".py"):
                    path = Path(path).resolve()
                    if not path.is_relative_to(source_root):
                        raise ValueError(f"Algorithm module {name} escaped source root: {path}")
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for obj in (Qwen2_5_VL, hf.Qwen2_5_VLForConditionalGeneration,
                    hf.Qwen2_5_VLTextModel.forward, hf.Qwen2_5_VLFlashAttention2.forward):
            path = Path(inspect.getsourcefile(obj)).resolve()
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [{"path": path, "sha256": value} for path, value in sorted(sources.items())]

    def tracked_batched(collator, n=1, batch_fn=None):
        for chunk in original_batched(collator, n=n, batch_fn=batch_fn):
            if state["active"]:
                if n != 1 or len(chunk) != 1:
                    raise ValueError("Trace requires the real adapter to use batch_size=1")
                request_id(chunk[0])
                state["chunk"] = chunk[0]
            yield chunk

    def traced_generate(model, *args, **kwargs):
        if not state["active"] or state["chunk"] is None:
            raise ValueError("Generation outside tracked real-adapter batch")
        if args or kwargs.get("num_beams", 1) != 1 or kwargs.get("do_sample", False) or not kwargs.get("use_cache", True):
            raise ValueError("Trace only supports explicit batch-one greedy cached adapter calls")
        chunk = state["chunk"]
        key = request_id(chunk)
        if key in completed:
            raise ValueError("Resume filter failed: completed request reached model generation")
        input_tensors = {name: value for name, value in kwargs.items() if torch.is_tensor(value)}
        if input_tensors["input_ids"].shape[0] != 1:
            raise ValueError("Trace requires exactly one input sequence")
        input_hashes = {name: digest(value) for name, value in input_tensors.items()}
        input_shapes = {name: list(value.shape) for name, value in input_tensors.items()}
        generation = {name: value for name, value in kwargs.items() if name not in input_tensors}
        if int(generation["max_new_tokens"]) != runtime["config"]["max_new_tokens"]:
            raise ValueError("Adapter max_new_tokens differs from recorded profile")
        sources = current_sources()
        if state["sources"] is None:
            state["sources"] = sources
            for old in completed.values():
                if old["source_files"] != sources:
                    raise ValueError("Algorithm/adapter/dependency source files changed since existing trace")
        elif state["sources"] != sources:
            raise ValueError("Source module set changed during generation")
        prefill = []

        def prefill_hook(module, call_args, call_kwargs, result):
            if result.last_hidden_state.shape[1] > 1:
                prefill.append(cache_snapshot(result.past_key_values))

        hook = model.model.language_model.register_forward_hook(prefill_hook, with_kwargs=True)
        try:
            result = original_generate(model, **kwargs, return_dict_in_generate=True, output_logits=True, output_scores=False)
        finally:
            hook.remove()
        expected_layers = model.config.text_config.num_hidden_layers
        if len(prefill) != 1 or len(prefill[0]) != expected_layers:
            raise ValueError(f"Expected one complete {expected_layers}-layer prefill cache")
        final_cache = cache_snapshot(result.past_key_values)
        if len(final_cache) != expected_layers:
            raise ValueError("Incomplete final KV cache")
        logits_hash = hashlib.sha256()
        logits = result.logits
        if not logits or any(value.shape[0] != 1 for value in logits):
            raise ValueError("Expected nonempty batch-one all-vocabulary generation logits")
        for value in logits:
            logits_hash.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        length = input_shapes["input_ids"][1]
        trimmed = result.sequences[:, length:]
        ids = trimmed[0].tolist()
        adapter = state["adapter"]
        decoded = adapter.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        until = chunk[1].get("until", [adapter.tokenizer.decode(adapter.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        text = decoded
        for term in until:
            if term and term != "\n\n":
                text = text.split(term)[0]
        row = {"schema_version": 1, "id": key, "task": chunk[4], "split": chunk[5], "doc_id": int(chunk[3]),
               "implementation": runtime["implementation"], "config": runtime["config"], "config_sha256": runtime["config_sha256"],
               "source_files": sources, "input_hashes": input_hashes, "input_shapes": input_shapes,
               "prefill_cache_shapes": prefill[0], "cache_shapes": final_cache,
               "logits_sha256": logits_hash.hexdigest(), "logits_shape": [len(logits), logits[0].shape[-1]],
               "logits_dtype": str(logits[0].dtype), "logits_kind": "unprocessed_generation_logits",
               "generated_ids": ids, "text": text, "decoded_text": decoded, "generation_kwargs": generation}
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        completed[key] = row
        print(json.dumps({"traced": key, "implementation": runtime["implementation"], "tokens": len(ids),
                          "trace_count": len(completed)}, ensure_ascii=False), flush=True)
        return result.sequences

    def resumed_generate_until(adapter, requests):
        requests = list(requests)
        keys = [request_id(request.args) for request in requests]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate task/doc requests; trace ID would be ambiguous")
        if int(adapter.batch_size) != 1:
            raise ValueError("Adapter batch size must be one")
        pending = [request for request, key in zip(requests, keys) if key not in completed]
        # Source validation also applies to a resume that needs zero model generations.
        sources = current_sources()
        for key in keys:
            if key in completed and completed[key]["source_files"] != sources:
                raise ValueError("Existing trace source files differ from current model/adapter")
        state.update(active=True, adapter=adapter, chunk=None)
        try:
            answers = original_generate_until(adapter, pending) if pending else []
        finally:
            state.update(active=False, adapter=None, chunk=None)
        if len(answers) != len(pending):
            raise ValueError("Adapter returned wrong pending-answer count")
        fresh = {request_id(request.args): answer for request, answer in zip(pending, answers)}
        for key, answer in fresh.items():
            if key not in completed or completed[key]["text"] != answer:
                raise ValueError("Trace text differs from actual adapter answer")
        return [fresh[key] if key in fresh else completed[key]["text"] for key in keys]

    utils.Collator.get_batched = tracked_batched
    Qwen2_5_VL.generate_until = resumed_generate_until
    hf.Qwen2_5_VLForConditionalGeneration.generate = traced_generate


def main():
    runtime = json.loads(Path(os.environ["KVREPACK_TRACE_CONFIG"]).read_text())
    if json_sha(runtime["config"]) != runtime["config_sha256"]:
        raise ValueError("Trace configuration digest mismatch")
    install(runtime)
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()

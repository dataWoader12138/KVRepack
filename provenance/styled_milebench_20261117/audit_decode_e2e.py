#!/usr/bin/env python3
"""Independent CPU-only audit of saved token IDs, timing sums and contract hashes.

No repository modules, model weights, image data or CUDA are loaded. By default
only controller-complete runs at startup are audited; --require-full requires
the complete 16 x 200 official matrix. Warmups are checked separately.
"""
from __future__ import annotations

import os
for _key, _value in {
    "CUDA_VISIBLE_DEVICES": "", "USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[_key] = _value

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tempfile

ARMS = ("hybridkv_asymkv", "hybridkv_lookm", "hybridkv_kvrepack",
        "snapkv_asymkv", "snapkv_lookm", "snapkv_kvrepack", "asymkv", "lookm")
TASKS = ("Spot-the-Diff", "MMCoQA")
PHASES = ("preprocessing_seconds", "h2d_seconds", "external_fisher_seconds",
          "generation_seconds", "text_decode_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def checked_json(path, snapshot):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing/symlinked file: {path}")
    data = path.read_bytes()
    snapshot[path] = sha(data)
    return json.loads(data)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid {name}")
    return value


def close(actual, expected, name):
    number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8),
            f"{name}: {actual!r} differs from recomputed {expected!r}")
    return abs(actual - expected)


def audit_run(root, entry, tokenizer_cache, snapshot):
    arm, task = entry["arm"], entry["task"]
    folder = root / arm / task
    require(Path(entry["output"]).resolve() == folder, "Controller run path differs")
    contract = checked_json(folder / "run_contract.json", snapshot)
    state = checked_json(folder / "run_state.json", snapshot)
    identity = json_sha(contract)
    require(contract.get("experiment") == "styled_milebench_image_response_e2e"
            and contract.get("source_commit") == "b9104d8" and contract.get("arm") == arm
            and contract.get("task") == task and contract.get("all_samples") is True
            and contract.get("limit") == contract.get("annotation_total_samples") == 200,
            f"Unexpected complete-run contract: {arm}/{task}")
    require(state.get("status") == "complete" and state.get("contract_sha256") == identity
            and all(state.get(key) == 200 for key in ("target_samples", "completed_samples",
                    "target_generations", "completed_generations")), f"Incomplete run: {arm}/{task}")
    selected, ordered = contract["selected_sample_ids"], contract["generation_sample_ids"]
    require(len(selected) == len(ordered) == len(set(selected)) == 200 and set(selected) == set(ordered),
            "Contract formal ID coverage differs")
    model_path = Path(contract["pretrained"]).resolve()
    for name, expected in contract["model_metadata_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid metadata relative path")
        path = model_path / relative
        data = path.read_bytes()
        require(sha(data) == expected, f"Model/tokenizer metadata changed: {name}")
        snapshot[path] = expected
    key = str(model_path)
    if key not in tokenizer_cache:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                  trust_remote_code=False, use_fast=False)
        require(not tokenizer.is_fast, "Require worker's slow tokenizer decode semantics")
        tokenizer_cache[key] = tokenizer
    tokenizer = tokenizer_cache[key]
    require("torch" not in sys.modules, "CPU token audit unexpectedly imported torch")
    maximum_e2e_error = maximum_model_error = 0.0
    sums = {phase: 0.0 for phase in PHASES}
    formal_tokens = formal_e2e = 0
    eos_seen = early_eos = 0
    warmup_count = contract["warmup_per_arm"]
    require(type(warmup_count) is int and warmup_count >= 0, "Invalid warmup count")
    for directory, count in (("samples", 200), ("warmup", warmup_count)):
        path = folder / directory
        files = sorted(path.iterdir()) if path.exists() else []
        require([file.name for file in files] == [f"{index:06d}.json" for index in range(count)],
                f"File coverage differs: {arm}/{task}/{directory}")
        for index, file in enumerate(files):
            row = checked_json(file, snapshot)
            label = f"{arm}/{task}/{directory}/{index}"
            require(row.get("schema_version") == 1 and row.get("status") == "completed"
                    and row.get("arm") == arm and row.get("task") == task
                    and row.get("sample_index") == index and row.get("contract_sha256") == identity
                    and row.get("warmup") is (directory == "warmup"), f"Row contract/identity differs: {label}")
            if directory == "samples":
                require(row.get("sample_id") == row.get("doc_id") == ordered[index], f"Formal ID differs: {label}")
            prediction, ids = row["prediction"], row["generated_ids"]
            require(prediction.get("sample_id") == row.get("sample_id")
                    and prediction.get("gen_kwargs") == contract["generation"], f"Prediction contract differs: {label}")
            require(isinstance(ids, list) and 0 < len(ids) <= contract["generation"]["max_new_tokens"]
                    and all(type(token) is int and token >= 0 for token in ids)
                    and row.get("generated_tokens") == len(ids), f"Token count differs: {label}")
            decoded = tokenizer.batch_decode([ids], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0].strip()
            require(isinstance(prediction.get("pred_response"), str)
                    and decoded == prediction["pred_response"].strip(), f"Independent decode differs: {label}")
            eos = tokenizer.eos_token_id
            first = next((i for i, token in enumerate(ids) if token == eos), None)
            hit, reached = len(ids) == contract["generation"]["max_new_tokens"], ids[-1] == eos
            require(row.get("eos_token_id") == eos and row.get("first_eos_index") == first
                    and row.get("eos_reached") is reached and row.get("hit_max_new_tokens") is hit
                    and row.get("capped_without_eos") is (hit and not reached), f"EOS accounting differs: {label}")
            runtime = row["runtime"]
            values = {name: number(runtime[name], f"{label}:{name}") for name in PHASES}
            e2e_error = close(runtime["end_to_end_seconds"], sum(values.values()), f"{label}:E2E five-stage sum")
            model_error = close(runtime["model_end_to_end_seconds"], values["external_fisher_seconds"]
                                + values["generation_seconds"], f"{label}:model E2E sum")
            if arm == "asymkv":
                require(runtime["fisher_seconds"] is None and values["external_fisher_seconds"] == 0
                        and row.get("generation_includes_fisher") is True,
                        f"Original AsymKV internal Fisher semantics differ: {label}")
            else:
                close(runtime["fisher_seconds"], values["external_fisher_seconds"], f"{label}:external Fisher")
            maximum_e2e_error = max(maximum_e2e_error, e2e_error)
            maximum_model_error = max(maximum_model_error, model_error)
            if directory == "samples":
                formal_tokens += len(ids); formal_e2e += runtime["end_to_end_seconds"]
                eos_seen += first is not None; early_eos += first is not None and first < len(ids)-1
                for name, value in values.items():
                    sums[name] += value
    return {"arm": arm, "task": task, "formal_samples_checked": 200,
            "warmups_checked_and_excluded": warmup_count, "contract_sha256": identity,
            "tokenizer_class": type(tokenizer).__name__, "generated_tokens": formal_tokens,
            "request_e2e_sum_seconds": formal_e2e, "stage_sums_seconds": sums,
            "maximum_absolute_e2e_sum_error_seconds": maximum_e2e_error,
            "maximum_absolute_model_sum_error_seconds": maximum_model_error,
            "first_eos_seen_count": eos_seen, "early_eos_count": early_eos, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists(), "Output already exists")
    require(not output.is_relative_to(root), "Audit output must be outside raw experiment")
    data = (root / "experiment_state.json").read_bytes(); controller = json.loads(data)
    all_entries = controller["runs"]
    require(controller.get("arms") == list(ARMS) and controller.get("tasks") == list(TASKS)
            and controller.get("planned_runs") == 16 and controller.get("target_generations") == 3200
            and controller.get("all_samples") is True, "Not a complete planned official matrix")
    require(len(all_entries) == 16 and {(entry["arm"], entry["task"]) for entry in all_entries}
            == {(arm, task) for arm in ARMS for task in TASKS}, "Planned matrix differs")
    completed = [entry for entry in all_entries if entry.get("status") == "complete"]
    require(completed, "No controller-complete runs available")
    if args.require_full:
        require(controller.get("status") == "complete" and controller.get("completed_runs") == 16
                and len(completed) == 16, "--require-full requires completed 16 x 200 matrix")
    snapshot, cache = {}, {}
    results = [audit_run(root, entry, cache, snapshot) for entry in completed]
    for path, identity in snapshot.items():
        require(path.is_file() and sha(path.read_bytes()) == identity, f"Audited source changed: {path}")
    require("torch" not in sys.modules, "Torch imported during independent CPU audit")
    summary = {"schema_version": 1, "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "root": str(root), "full_3200_audit": len(results) == 16,
               "controller_status_at_start": controller.get("status"), "controller_sha256_at_start": sha(data),
               "completed_runs_checked": len(results), "formal_samples_checked": 200*len(results),
               "warmups_checked_and_excluded": sum(row["warmups_checked_and_excluded"] for row in results),
               "transformers_version": importlib.metadata.version("transformers"), "gpu_access": False,
               "torch_imported": False, "decode_parameters": {"use_fast": False, "skip_special_tokens": True,
                  "clean_up_tokenization_spaces": False, "strip": True},
               "scope": "independent saved-ID text decode, five-stage E2E/model sums, contract hash, IDs, EOS and source stability",
               "results": results, "audited_file_sha256": {str(path): identity for path, identity in sorted(snapshot.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".styled-audit-", dir=output.parent,
                                     delete=False) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
        temporary = Path(stream.name)
    try:
        require(not output.exists(), "Output appeared during audit")
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({key: summary[key] for key in ("status", "full_3200_audit", "completed_runs_checked",
                     "formal_samples_checked", "warmups_checked_and_excluded", "torch_imported")}, ensure_ascii=False))
    print(f"Audit JSON: {output}")


if __name__ == "__main__":
    main()

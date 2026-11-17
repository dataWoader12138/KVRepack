#!/usr/bin/env python3
"""Independent standard-library audit of saved styled-image pairing evidence.

Compares full formal records by sample_id, never relying on report summaries.
Anchors are determined by positions[start_mask]; matching both saved tensor
fingerprints proves the same recorded anchor decision without storing tensors.
No model, tensor framework or GPU is imported or accessed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
VARIANTS = ("asymkv", "lookm", "kvrepack")
TASKS = ("Spot-the-Diff", "MMCoQA")
FIELDS = ("positions", "group_ids", "group_size", "start_mask")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def js(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read(path, snapshot):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing/symlinked file: {path}")
    data = path.read_bytes(); snapshot[path] = digest(data)
    return json.loads(data)


def fingerprint(value, shape=None, dtype=None):
    require(isinstance(value, dict) and set(value) == {"shape", "dtype", "sha256"}
            and isinstance(value["shape"], list) and all(type(v) is int and v >= 0 for v in value["shape"])
            and isinstance(value["dtype"], str)
            and re.fullmatch("[0-9a-f]{64}", value.get("sha256", "")), "Malformed tensor fingerprint")
    if shape is not None:
        require(value["shape"] == shape, f"Fingerprint shape differs: {value['shape']} vs {shape}")
    if dtype is not None:
        require(value["dtype"] == dtype, "Fingerprint dtype differs")
    return value


def validate_inputs(row):
    fp = row.get("input_fingerprints", {})
    hashes, shapes, dtypes = (fp.get(key, {}) for key in ("input_hashes", "input_shapes", "input_dtypes"))
    require(set(hashes) == set(shapes) == set(dtypes)
            and {"input_ids", "pixel_values", "image_grid_thw", "attention_mask"} <= set(hashes),
            "Missing input fingerprint coverage")
    for key in hashes:
        fingerprint({"shape": shapes[key], "dtype": dtypes[key], "sha256": hashes[key]})
    require(shapes["input_ids"] == shapes["attention_mask"] == [1, row["input_tokens"]], "Input shape mismatch")


def physical_per_layer(row, backbone):
    shapes = row["budget"].get("final_shapes")
    require(isinstance(shapes, list) and len(shapes) == 28, "Missing 28-layer physical cache")
    generated = row["generated_tokens"]
    require(type(generated) is int and generated == len(row["generated_ids"]) and generated > 0, "Invalid generated token count")
    prefill = []
    for layer, item in enumerate(shapes):
        key, value = item.get("key_shape"), item.get("value_shape")
        require(item.get("layer") == layer and key == value and isinstance(key, list)
                and all(type(n) is int and n > 0 for n in key) and key[-1] == 128
                and ((backbone == "hybridkv" and len(key) == 2)
                     or (backbone == "snapkv" and len(key) == 4 and key[:2] == [1, 4])),
                "Invalid native cache shape")
        active = math.prod(key[:-1]) - 4*(generated-1)
        require(active > 0, "Nonpositive reconstructed prefill length")
        prefill.append(active)
    budget, audit, n = row["budget"], row["audit"], row["input_tokens"]
    physical = sum(prefill)
    require(physical == budget.get("prefill_tokens_total") == budget.get("capacity")
            and budget.get("native_budget_unchanged") is True
            and budget.get("budget_policy") == "original_history_plus_local_v1",
            "Physical prefill/nominal stored capacity differs")
    require(math.isclose(budget["actual_retained_ratio"], physical/(112*n), rel_tol=1e-12, abs_tol=1e-12),
            "Physical retained ratio differs")
    if backbone == "hybridkv":
        native = audit["budget"]
        budgets = native.get("history_budget")
        require(native.get("policy") == "original_history_plus_local_v1"
                and native.get("native_budget_unchanged") is True and native.get("post_allocation_scaling") is False
                and native.get("scale") == 1.0 and native.get("sequence_length") == n
                and native.get("layers") == 28 and native.get("native_kv_heads") == 4
                and native.get("budget_ratio") == .1 and native.get("original_history_budget") == budgets,
                "Original heterogeneous budget changed")
        require(isinstance(budgets, list) and len(budgets) == 28 and all(isinstance(b, list) and len(b) == 1
                and isinstance(b[0], list) and len(b[0]) == 4 and all(type(x) is int and x >= 0 for x in b[0]) for b in budgets),
                "Invalid per-head native history budgets")
        local = native["local_len"]
        require(type(local) is int and 0 <= local <= n, "Invalid native local window")
        expected = [sum(batch[0]) + 4*local for batch in budgets]
        history = sum(sum(batch[0]) for batch in budgets)
        require(prefill == expected == native.get("per_layer_active_positions")
                and native.get("original_history_positions") == native.get("history_positions") == history
                and native.get("local_positions") == 112*local
                and native.get("active_prefill_positions") == native.get("capacity_positions") == physical,
                "Native per-head history/local capacity differs from reconstructed physical cache")
        observed = audit.get("cache_shapes")
        require(isinstance(observed, list) and len(observed) == 28
                and all({name: item[name] for name in ("layer", "key_shape", "value_shape")} == other
                        for item, other in zip(shapes, observed)), "Actual Hybrid cache shape audits differ")
    else:
        per_head = min(n-32, math.floor(.1*n)) + 32
        require(prefill == [4*per_head]*28 and audit.get("prefill_cache_shapes") == [[1, 4, per_head, 128]]*28,
                "Original SnapKV native history/recent capacity differs")
    return prefill


def decisions(row, backbone):
    audit = row["audit"]
    require(audit.get("status") == "complete" and audit.get("variant") == row["variant"], "Incomplete/mislabeled actual audit")
    calls = audit.get("merge_calls")
    require(isinstance(calls, list) and calls, "Missing successful merge calls")
    identities, groups, anchors = [], [], []
    for call in calls:
        before = call["positions"]["shape"][0]
        after = call["group_size"]["shape"][0]
        require(0 < after < before, "Expected successful reducing merge call")
        for name, shape, dtype in (("positions", [before], "torch.int64"), ("group_ids", [before], "torch.int64"),
                                   ("group_size", [after], "torch.float32"), ("start_mask", [before], "torch.bool")):
            fingerprint(call.get(name), shape, dtype)
        groups.append({name: call[name] for name in FIELDS})
        anchors.append({"positions": call["positions"], "start_mask": call["start_mask"], "anchor_count": after})
        if backbone == "hybridkv":
            ident = {key: call[key] for key in ("layer_idx", "head_idx", "batch_idx", "before", "after")}
            require(type(ident["layer_idx"]) is int and 0 <= ident["layer_idx"] < 28
                    and type(ident["head_idx"]) is int and 0 <= ident["head_idx"] < 4
                    and ident["batch_idx"] == 0 and ident["before"] == before and ident["after"] == after,
                    "Invalid actual layer/head merge identity")
            identities.append(ident)
    if backbone == "hybridkv":
        require(len({(x["layer_idx"], x["head_idx"], x["batch_idx"]) for x in identities}) == len(identities),
                "Duplicate actual head merge call")
        fingerprint(audit.get("merge_mask"), [28, 1, 4], "torch.bool")
        selected = audit.get("selected_head_count")
        require(type(selected) is int and len(calls) <= selected <= 112 and audit.get("raw_reapply") is True,
                "Invalid native selected head audit")
        head = {"merge_mask": audit["merge_mask"], "selected_head_count": selected}
        signature = {"groups": groups, "identities": identities, "head": head, "budget": audit["budget"]}
    else:
        masks, layers = audit.get("selected_masks"), audit.get("layers")
        require(isinstance(masks, list) and len(masks) == 28 and isinstance(layers, list) and len(layers) == 28,
                "Missing 28-layer SnapKV masks/candidates")
        for mask in masks:
            fingerprint(mask, [1, 4], "torch.bool")
        candidate_records = []
        for index, layer in enumerate(layers):
            require(layer.get("layer") == index and type(layer.get("candidate_history")) is int
                    and type(layer.get("final_history")) is int
                    and 0 < layer["final_history"] < layer["candidate_history"], "Invalid native SnapKV candidate layer")
            fingerprint(layer.get("candidate_idx"), [1, 4, layer["candidate_history"]], "torch.int64")
            candidate_records.append({key: layer[key] for key in ("layer", "candidate_idx", "candidate_history", "final_history")})
        require(audit.get("actual_merge_calls") == len(calls), "Actual SnapKV merge call count differs")
        head = {"selected_masks": masks, "layers": candidate_records}
        signature = {"groups": groups, "head": head, "prefill_shapes": audit["prefill_cache_shapes"]}
    return signature, {"head_selection_sha256": js(head), "candidates_and_grouping_sha256": js(groups),
                       "derived_anchor_decisions_sha256": js(anchors), "full_pairing_sha256": js(signature),
                       "actual_merge_calls": len(calls), "total_anchor_count": sum(x["anchor_count"] for x in anchors)}


def load_run(root, backbone, variant, task, snapshot):
    arm = backbone + "_" + variant; folder = root / arm / task
    contract = read(folder / "run_contract.json", snapshot); state = read(folder / "run_state.json", snapshot)
    identity = js(contract)
    require(state.get("status") == "complete" and state.get("contract_sha256") == identity
            and all(state.get(key) == 200 for key in ("target_samples", "completed_samples", "target_generations", "completed_generations")),
            f"Run incomplete: {arm}/{task}")
    require(contract.get("arm") == arm and contract.get("backbone") == backbone and contract.get("variant") == variant
            and contract.get("standalone") is False and contract.get("method") == backbone + "_kvrepack"
            and contract.get("task") == task and contract.get("source_commit") == "b9104d8"
            and contract.get("all_samples") is True and contract.get("limit") == contract.get("annotation_total_samples") == 200,
            f"Invalid complete-run contract: {arm}/{task}")
    ordered, selected = contract["generation_sample_ids"], contract["selected_sample_ids"]
    require(len(ordered) == len(selected) == len(set(selected)) == 200 and set(ordered) == set(selected), "Contract ID coverage differs")
    files = sorted((folder / "samples").iterdir())
    require([file.name for file in files] == [f"{index:06d}.json" for index in range(200)], "Formal file coverage differs")
    result = {}
    for index, file in enumerate(files):
        row = read(file, snapshot); sample_id = ordered[index]
        require(row.get("status") == "completed" and row.get("warmup") is False and row.get("contract_sha256") == identity
                and row.get("arm") == arm and row.get("task") == task and row.get("backbone") == backbone
                and row.get("variant") == variant and row.get("standalone") is False
                and row.get("sample_index") == index and row.get("sample_id") == row.get("doc_id") == sample_id,
                f"Raw identity differs: {file}")
        require(type(row.get("input_tokens")) is int and row["input_tokens"] > 0
                and row["prediction"].get("sample_id") == sample_id, "Invalid raw token/prediction identity")
        validate_inputs(row)
        capacity = physical_per_layer(row, backbone)
        signature, details = decisions(row, backbone)
        result[sample_id] = (row, signature, capacity, details)
    return contract, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbones", choices=("hybridkv", "snapkv"), nargs="+", default=["hybridkv", "snapkv"])
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists() and not output.is_relative_to(root), "Output must be new and outside raw data")
    require(len(set(args.backbones)) == len(args.backbones), "Backbones must be unique")
    controller_bytes = (root / "experiment_state.json").read_bytes(); controller = json.loads(controller_bytes)
    entries = {(row["arm"], row["task"]): row for row in controller["runs"]}
    snapshot, results = {}, []
    for backbone in args.backbones:
        for task in TASKS:
            loaded = {}
            for variant in VARIANTS:
                require(entries.get((backbone + "_" + variant, task), {}).get("status") == "complete",
                        f"Controller arm incomplete: {backbone}_{variant}/{task}")
                loaded[variant] = load_run(root, backbone, variant, task, snapshot)
            baseline_contract, baseline = loaded["asymkv"]
            common = ("selected_sample_ids", "generation_sample_ids", "annotation_sha256", "image", "generation", "algorithm_env",
                      "source_sha256", "model_metadata_sha256", "kernel_binary", "dtype", "attention", "model_seed", "head_geometry")
            for variant in ("lookm", "kvrepack"):
                contract, records = loaded[variant]
                require({key: baseline_contract.get(key) for key in common} == {key: contract.get(key) for key in common},
                        f"Common contracts differ: {backbone}/{task}/{variant}")
                require(set(records) == set(baseline), "Sample ID set differs")
                for sample_id, (ref, signature, capacity, details) in baseline.items():
                    row, observed, physical, actual = records[sample_id]
                    label = f"{backbone}/{task}/sample_id={sample_id}/{variant}"
                    require(row["input_tokens"] == ref["input_tokens"] and row["input_fingerprints"] == ref["input_fingerprints"],
                            f"Input tensor fingerprints differ: {label}")
                    require({key: row["prediction"].get(key) for key in ("image", "question", "gt_response", "gen_kwargs", "gen_model_id")}
                            == {key: ref["prediction"].get(key) for key in ("image", "question", "gt_response", "gen_kwargs", "gen_model_id")},
                            f"Image/question/GT/generation identity differs: {label}")
                    require(observed == signature and actual == details, f"Head/candidate/group/anchor decisions differ: {label}")
                    require(physical == capacity, f"Actual per-layer prefill cache capacity differs: {label}")
            per_input = []
            for sample_id, (ref, signature, capacity, details) in baseline.items():
                per_input.append({"sample_id": sample_id, "input_tokens": ref["input_tokens"],
                                  "input_fingerprint_sha256": js(ref["input_fingerprints"]), **details,
                                  "actual_prefill_total": sum(capacity), "actual_prefill_per_layer": capacity,
                                  "actual_retained_ratio": sum(capacity)/(112*ref["input_tokens"])})
            results.append({"backbone": backbone, "task": task, "sample_ids_checked": 200,
                            "pairwise_variant_sample_comparisons": 400, "status": "passed", "per_input": per_input})
    for path, identity in snapshot.items():
        require(path.is_file() and digest(path.read_bytes()) == identity, f"Audited raw file changed: {path}")
    require("torch" not in sys.modules, "Unexpected torch import")
    summary = {"schema_version": 1, "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "root": str(root), "backbones": args.backbones, "controller_sha256_at_start": digest(controller_bytes),
               "completed_runs_checked": 6*len(args.backbones), "formal_records_checked": 1200*len(args.backbones),
               "pairwise_variant_sample_comparisons": 800*len(args.backbones), "gpu_access": False,
               "scope": "actual saved raw records; inputs, selected heads, candidates, groups, anchors and reconstructed per-layer physical prefill",
               "anchor_evidence": "anchor positions are determined by positions[start_mask]; both saved tensor hashes/shapes/dtypes are compared exactly",
               "limits": ["Recorded tensor fingerprints are compared; original input/decision tensors are not regenerated.",
                          "Final cache shapes include decode growth; 4*(generated_tokens-1) positions per layer are removed before comparisons.",
                          "Selected Hybrid heads may require no merge when target >= candidate length; actual merge calls need not equal selected head count."],
               "results": results, "audited_file_sha256": {str(path): identity for path, identity in sorted(snapshot.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".styled-pair-audit-", dir=output.parent, delete=False) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n"); temporary = Path(stream.name)
    try:
        require(not output.exists(), "Output appeared during audit"); temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({key: summary[key] for key in ("status", "backbones", "completed_runs_checked", "formal_records_checked",
                                                  "pairwise_variant_sample_comparisons")}, ensure_ascii=False))
    print(f"Audit JSON: {output}")


if __name__ == "__main__":
    main()

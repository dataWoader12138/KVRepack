#!/usr/bin/env python3
"""CPU-only coverage, source, pairing, quality and E2E audit for image runs.

A report is emitted only after every requested child run is complete. Selected
smoke matrices can be scored, but are explicitly not the full 16-run experiment.
The source experiment's ROUGE-L/Accuracy functions are used without rounding.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvrepack.ablations.run_head_selection import json_sha
from kvrepack.ablations.run_merge_comparison import pairing_signature

ARMS = ("hybridkv_asymkv", "hybridkv_lookm", "hybridkv_kvrepack",
        "snapkv_asymkv", "snapkv_lookm", "snapkv_kvrepack", "asymkv", "lookm")
TASKS = ("Spot-the-Diff", "MMCoQA")
PHASES = ("preprocessing_seconds", "h2d_seconds", "fisher_seconds",
          "generation_seconds", "text_decode_seconds")
SUMMARY_COLUMNS = ["arm", "task", "metric", "score_percent", "sample_count", "full_task",
    "e2e_mean_seconds", "e2e_median_seconds", "e2e_p95_seconds", "e2e_total_seconds",
    "model_e2e_mean_seconds", "preprocessing_mean_seconds", "h2d_mean_seconds",
    "fisher_mean_seconds", "generation_mean_seconds", "text_decode_mean_seconds", "fisher_scope",
    "generated_tokens_mean", "generated_tokens_total", "eos_count", "hit_max_new_tokens_count",
    "capped_without_eos_count", "first_eos_seen_count", "early_eos_count", "post_first_eos_tokens_total",
    "prefill_retained_ratio_mean", "prefill_retained_ratio_min",
    "prefill_retained_ratio_max", "budget_policy", "asymkv_boundary_affected_count",
    "asymkv_missing_prompt_tokens_total", "asymkv_compression_event_count", "model_load_seconds",
    "formal_loop_seconds", "process_seconds"]
PER_INPUT_COLUMNS = ["arm", "task", "sample_id", "generation_index", "score_percent", "input_tokens",
    "generated_tokens", "eos_reached", "hit_max_new_tokens", "capped_without_eos",
    "first_eos_index", "early_eos", "tokens_after_first_eos",
    "end_to_end_seconds", "model_end_to_end_seconds", *PHASES, "external_fisher_seconds", "fisher_scope", "peak_allocated_gib",
    "actual_prefill_retained_ratio", "budget_policy", "asymkv_prompt_tokens_covered",
    "asymkv_prompt_tokens_missing", "asymkv_boundary_affected", "asymkv_compression_events"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, name, *, positive=False):
    require(type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"Invalid {name}: {value!r}")
    return float(value)


def equal_number(a, b, name):
    require(math.isclose(a, b, rel_tol=1e-7, abs_tol=2e-5), f"Inconsistent {name}: {a} versus {b}")


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_sources(mapping):
    require(isinstance(mapping, dict) and mapping, "Missing frozen runtime source map")
    for relative, expected in mapping.items():
        path = (ROOT / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(ROOT) and path.is_file()
                and valid_sha(expected) and file_sha(path) == expected, f"Changed bound runtime source: {relative}")


def validate_fingerprints(fingerprints, input_tokens):
    require(isinstance(fingerprints, dict), "Missing input tensor fingerprints")
    hashes = fingerprints.get("input_hashes")
    shapes = fingerprints.get("input_shapes")
    dtypes = fingerprints.get("input_dtypes")
    require(isinstance(hashes, dict) and isinstance(shapes, dict) and isinstance(dtypes, dict)
            and set(hashes) == set(shapes) == set(dtypes)
            and {"input_ids", "pixel_values", "image_grid_thw"} <= set(hashes),
            "Incomplete image input fingerprint coverage")
    require(all(valid_sha(value) for value in hashes.values()) and shapes["input_ids"] == [1, input_tokens]
            and all(isinstance(value, str) and value for value in dtypes.values()),
            "Malformed input tensor fingerprints")


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def physical_from_shapes(shapes, generated_tokens):
    require(isinstance(shapes, list) and len(shapes) == 28, "Missing 28-layer physical cache shapes")
    final = 0
    for index, item in enumerate(shapes):
        key, value = item.get("key_shape"), item.get("value_shape")
        require(item.get("layer") == index and key == value and isinstance(key, list)
                and len(key) in (2, 4) and all(type(v) is int and v > 0 for v in key)
                and key[-1] == 128 and (len(key) == 2 or key[:2] == [1, 4]),
                "Invalid native Qwen7B physical K/V shapes")
        final += math.prod(key[:-1])
    return final - 112 * (generated_tokens - 1)


def observed_budget(row, arm):
    """Observe physical prefill, without imposing a new nominal 10% cap."""
    n, budget, audit = row["input_tokens"], row.get("budget"), row.get("audit")
    require(isinstance(budget, dict) and isinstance(audit, dict), "Missing actual budget/audit record")
    physical = budget.get("prefill_tokens_total")
    require(type(physical) is int and physical > 0, "Missing positive physical prefill length")
    ratio = number(budget.get("actual_retained_ratio"), "actual prefill retained ratio", positive=True)
    equal_number(ratio, physical / (112 * n), "physical prefill retained ratio")
    policy = budget.get("budget_policy", budget.get("policy"))
    require(isinstance(policy, str) and policy, "Missing original budget policy")
    covered = missing = boundary = events = None
    if arm == "asymkv":
        lengths = audit.get("prefill_physical_lengths")
        require(isinstance(lengths, list) and len(lengths) == 28
                and all(type(value) is int and value > 0 for value in lengths)
                and sum(lengths) * 4 == physical, "Original AsymKV physical cache record differs")
        require(audit.get("prompt_length") == n, "Original AsymKV prompt length differs")
        covered = audit.get("prefill_logical_tokens")
        expected_covered = n - 1 if n % 512 == 1 else n
        require(type(covered) is int and 0 < covered == expected_covered, "Invalid original AsymKV prompt coverage")
        settings = audit.get("runtime", {})
        require(audit.get("method") == "AsymKV-official" and settings.get("budget_ratio") == .1
                and settings.get("absolute_cache_size") is None and settings.get("start_size") == 32
                and settings.get("chunk_size") == 512 and audit.get("configured_cache_size") == math.floor(.1*n),
                "Original AsymKV runtime differs from its released nominal budget")
        missing, boundary = n - covered, covered != n
        # Preserve released-driver omissions. Report them; do not drop samples.
        require(row.get("asymkv_prompt_tokens_covered") == covered
                and row.get("asymkv_prompt_tokens_missing") == missing
                and row.get("asymkv_boundary_affected") is boundary,
                "Original AsymKV coverage flags disagree with actual audit")
        compression = audit.get("compression_events")
        require(isinstance(compression, list), "Original AsymKV compression events are absent")
        events = len(compression)
    else:
        shapes = audit.get("final_cache_shapes") if arm == "lookm" else budget.get("final_shapes")
        require(physical_from_shapes(shapes, row["generated_tokens"]) == physical,
                "Physical prefill total differs from actual final cache shapes")
        if arm == "lookm":
            require(audit.get("physical_prefill_positions") == physical
                    and physical == 112 * (math.floor(.1*n) + 32), "Original LOOK-M history-plus-recent budget differs")
        elif arm.startswith("snapkv_"):
            recent = min(32, n)
            require(physical == 112*(min(n-recent, math.floor(.1*n))+recent), "Original SnapKV budget differs")
            prefill = audit.get("prefill_cache_shapes")
            require(isinstance(prefill, list) and len(prefill) == 28, "Missing native SnapKV prefill shapes")
            for shape, final in zip(prefill, shapes):
                key = final["key_shape"]
                require(shape == [key[0], key[1], key[2] - row["generated_tokens"] + 1, key[3]],
                        "SnapKV recorded prefill/cache shapes differ")
        else:
            observed = audit.get("cache_shapes")
            require(isinstance(observed, list) and len(observed) == 28, "Missing native Hybrid cache shapes")
            require(all({name: item[name] for name in ("layer", "key_shape", "value_shape")} == other
                        for item, other in zip(shapes, observed)), "Hybrid final cache shapes differ across audit/budget")
            native = audit.get("budget", {})
            require(native.get("native_budget_unchanged") is True and native.get("post_allocation_scaling") is False
                    and native.get("scale") == 1.0
                    and native.get("original_history_budget") == native.get("history_budget")
                    and native.get("original_history_positions") == native.get("history_positions")
                    and native.get("history_positions", -1) + native.get("local_positions", -1) == physical
                    and native.get("active_prefill_positions") == native.get("capacity_positions") == physical,
                    "Hybrid original heterogeneous budget changed")
    return ratio, policy, covered, missing, boundary, events


def validate_sample(row, contract, index, expected_id, *, warmup=False):
    arm, task = contract["arm"], contract["task"]
    standalone = arm in ("asymkv", "lookm")
    backbone, variant = (None, None) if standalone else arm.split("_", 1)
    spec = {"method": arm if standalone else backbone + "_kvrepack", "backbone": backbone,
            "variant": variant, "standalone": standalone}
    require(all(contract.get(key) == value and row.get(key) == value for key, value in spec.items()),
            "Arm/backbone/variant labels disagree")
    require(row.get("schema_version") == 1 and row.get("status") == "completed"
            and row.get("arm") == arm and row.get("task") == task
            and type(row.get("sample_index")) is int and row["sample_index"] == index
            and row.get("sample_id") == expected_id and row.get("doc_id") == expected_id
            and row.get("contract_sha256") == json_sha(contract)
            and row.get("warmup") is warmup, "Incorrect/incomplete sample or warmup mixed into results")
    prediction = row.get("prediction")
    require(isinstance(prediction, dict) and prediction.get("sample_id") == expected_id
            and isinstance(prediction.get("pred_response"), str)
            and prediction.get("gen_kwargs") == contract["generation"], "Prediction identity/generation config differs")
    n = row.get("input_tokens")
    require(type(n) is int and n > 0, "Missing positive input token count")
    validate_fingerprints(row.get("input_fingerprints"), n)
    ids, maximum, eos = row.get("generated_ids"), contract["generation"]["max_new_tokens"], row.get("eos_token_id")
    require(isinstance(ids, list) and ids and all(type(v) is int and v >= 0 for v in ids)
            and type(row.get("generated_tokens")) is int and row["generated_tokens"] == len(ids)
            and type(maximum) is int and 0 < len(ids) <= maximum and type(eos) is int,
            "Generated token accounting differs")
    reached, hit = ids[-1] == eos, len(ids) == maximum
    first_eos = next((i for i, token in enumerate(ids) if token == eos), None)
    require(row.get("eos_reached") is reached and row.get("hit_max_new_tokens") is hit
            and row.get("capped_without_eos") is (hit and not reached)
            and row.get("first_eos_index") == first_eos
            and (arm == "asymkv" or first_eos is None or first_eos == len(ids) - 1),
            "EOS/maximum/first-EOS flags differ from token IDs")
    early_eos = first_eos is not None and first_eos < len(ids) - 1
    runtime = row.get("runtime")
    require(isinstance(runtime, dict), "Missing runtime stages")
    for name in (*(name for name in PHASES if name != "fisher_seconds"), "model_end_to_end_seconds", "peak_allocated_gib"):
        number(runtime.get(name), name)
    external_fisher = number(runtime.get("external_fisher_seconds", runtime.get("fisher_seconds")), "external Fisher stage")
    if arm != "asymkv":
        number(runtime.get("fisher_seconds"), "separate Fisher time")
        equal_number(runtime["fisher_seconds"], external_fisher, "external/separate Fisher accounting")
    total = number(runtime.get("end_to_end_seconds"), "response E2E", positive=True)
    require(runtime["generation_seconds"] > 0, "Generation time must be positive")
    equal_number(total, sum(runtime[name] for name in PHASES if name != "fisher_seconds") + external_fisher, "E2E stage sum")
    # Worker contract explicitly defines model-E2E as Fisher plus generate;
    # text decode belongs to response E2E and is reported separately.
    equal_number(runtime["model_end_to_end_seconds"], external_fisher + runtime["generation_seconds"],
                 "model E2E stage sum")
    if arm == "asymkv":
        require(row.get("generation_includes_fisher") is True and row.get("separately_timed_fisher") is False
                and runtime.get("fisher_seconds") is None and external_fisher == 0,
                "Original AsymKV Fisher must stay inside native generation timing, with separate time unmeasured")
        scope = "included_in_official_generation_not_separately_measured"
    elif arm.endswith("_asymkv"):
        require(row.get("generation_includes_fisher") is False and row.get("separately_timed_fisher") is True
                and runtime["fisher_seconds"] > 0 and isinstance(row.get("fisher"), dict),
                "Styled AsymKV requires its actual separately measured Fisher probe")
        scope = "separately_measured_raw_key_probe"
    else:
        require(row.get("generation_includes_fisher") is False and row.get("separately_timed_fisher") is False
                and runtime["fisher_seconds"] == 0, "Unexpected independent Fisher stage")
        scope = "not_applicable"
    ratio, policy, covered, missing, boundary, events = observed_budget(row, arm)
    if not standalone:
        audit = row["audit"]
        require(audit.get("variant") == variant, "Actual merge variant differs from requested arm")
        pairing_signature(audit, backbone + "_kvrepack")
    require(policy == contract["budget_policy"], "Observed budget policy differs from contract")
    return {"arm": arm, "task": task, "sample_id": expected_id, "generation_index": index,
            "input_tokens": n, "generated_tokens": len(ids), "eos_reached": reached,
            "hit_max_new_tokens": hit, "capped_without_eos": hit and not reached,
            "first_eos_index": first_eos, "early_eos": early_eos,
            "tokens_after_first_eos": len(ids) - first_eos - 1 if first_eos is not None else 0,
            "end_to_end_seconds": total, "model_end_to_end_seconds": runtime["model_end_to_end_seconds"],
            **{name: (None if arm == "asymkv" and name == "fisher_seconds" else runtime[name]) for name in PHASES},
            "external_fisher_seconds": external_fisher,
            "fisher_scope": scope, "peak_allocated_gib": runtime["peak_allocated_gib"],
            "actual_prefill_retained_ratio": ratio, "budget_policy": policy,
            "asymkv_prompt_tokens_covered": covered, "asymkv_prompt_tokens_missing": missing,
            "asymkv_boundary_affected": boundary, "asymkv_compression_events": events}


def read_run(path, expected_arm, expected_task):
    from kvrepack.benchmarks.milebench.dataset import annotation_path, load_annotation
    from kvrepack.benchmarks.milebench.evaluate import evaluate_predictions
    path = Path(path)
    contract, state = read_json(path / "run_contract.json"), read_json(path / "run_state.json")
    require(contract.get("experiment") == "styled_milebench_image_response_e2e"
            and contract.get("arm") == expected_arm and contract.get("task") == expected_task
            and contract.get("source_commit") == "b9104d8"
            and contract.get("model_type") == "qwen2_5_vl-7b"
            and contract.get("head_geometry") == {"layers": 28, "native_kv_heads": 4}, "Unexpected experiment contract")
    require(type(contract.get("all_samples")) is bool, "Contract all_samples must be explicit")
    count = contract.get("limit")
    require(type(count) is int and count > 0 and state.get("status") == "complete"
            and state.get("contract_sha256") == json_sha(contract)
            and all(state.get(name) == count for name in ("target_samples", "completed_samples",
                    "target_generations", "completed_generations")), "Child run is incomplete")
    for name in ("model_load_seconds", "formal_loop_seconds", "process_seconds"):
        number(state.get(name), name, positive=True)
    verify_sources(contract.get("source_sha256"))
    if expected_arm.startswith("hybridkv_"):
        kernel = contract.get("kernel_binary", {})
        require(isinstance(kernel.get("path"), str) and valid_sha(kernel.get("sha256"))
                and Path(kernel["path"]).is_file() and file_sha(kernel["path"]) == kernel["sha256"],
                "Executed Hybrid CUDA binary changed")
    annotation_file = annotation_path(contract["data_root"], expected_task)
    require(file_sha(annotation_file) == contract.get("annotation_sha256"), "Complete annotation changed")
    complete = load_annotation(contract["data_root"], expected_task, limit=None)
    require(contract.get("annotation_total_samples") == len(complete["data"]), "Full annotation total differs")
    selected = complete if contract["all_samples"] else load_annotation(contract["data_root"], expected_task, limit=count)
    ids = [row["sample_id"] for row in selected["data"]]
    require(len(ids) == count and ids == contract.get("selected_sample_ids"), "Selected annotation ID coverage differs")
    groups = {}
    for sample in selected["data"]:
        groups.setdefault(len(sample["task_instance"]["images_path"]), []).append(sample["sample_id"])
    ordered = [sample_id for group in groups.values() for sample_id in group]
    require(ordered == contract.get("generation_sample_ids"), "Source image-count execution order changed")
    files = sorted((path / "samples").iterdir())
    require([p.name for p in files] == [f"{i:06d}.json" for i in range(count)]
            and all(p.is_file() and not p.is_symlink() for p in files), "Sample file coverage differs")
    warmup_count = contract.get("warmup_per_arm")
    require(type(warmup_count) is int and warmup_count >= 0, "Missing declared warmup count")
    warmup_files = sorted((path / "warmup").glob("*.json"))
    require([p.name for p in warmup_files] == [f"{i:06d}.json" for i in range(warmup_count)], "Warmup file coverage differs")
    for index, file in enumerate(warmup_files):
        source_index = index % count
        validate_sample(read_json(file), contract, source_index, ordered[source_index], warmup=True)
    predictions, latency, rows, evidence = [], [], {}, []
    metric = "Accuracy" if expected_task == "MMCoQA" else "Rouge-L f"
    references = {row["sample_id"]: row for row in selected["data"]}
    for index, file in enumerate(files):
        row = read_json(file)
        derived = validate_sample(row, contract, index, ordered[index])
        singleton = {"meta_data": deepcopy(selected["meta_data"]), "data": [deepcopy(references[row["sample_id"]])]}
        derived["score_percent"] = float(evaluate_predictions([row["prediction"]], singleton, expected_task)["eval"][metric]) * 100
        latency.append(derived); predictions.append(row["prediction"]); rows[row["sample_id"]] = row
        evidence.append({"sample_id": row["sample_id"], "sample_file": file.name, "sha256": file_sha(file),
                         "input_fingerprints_sha256": json_sha(row["input_fingerprints"])})
    evaluation = evaluate_predictions(predictions, selected, expected_task)
    score = float(evaluation["eval"][metric]) * 100
    equal_number(score, statistics.fmean(v["score_percent"] for v in latency), "aggregate quality")
    times = [row["end_to_end_seconds"] for row in latency]
    require(state["formal_loop_seconds"] + 2e-5 >= sum(times)
            and state["process_seconds"] + 2e-5 >= state["formal_loop_seconds"] + state["model_load_seconds"],
            "Whole-loop/process timers are smaller than their included phases")
    ratios = [row["actual_prefill_retained_ratio"] for row in latency]
    summary = dict(arm=expected_arm, task=expected_task, metric=metric, score_percent=score,
        sample_count=count, full_task=len(ids) == len(complete["data"]),
        e2e_mean_seconds=statistics.fmean(times), e2e_median_seconds=statistics.median(times),
        e2e_p95_seconds=percentile(times, .95), e2e_total_seconds=sum(times),
        model_e2e_mean_seconds=statistics.fmean(v["model_end_to_end_seconds"] for v in latency),
        fisher_scope=latency[0]["fisher_scope"],
        generated_tokens_mean=statistics.fmean(v["generated_tokens"] for v in latency),
        generated_tokens_total=sum(v["generated_tokens"] for v in latency), eos_count=sum(v["eos_reached"] for v in latency),
        hit_max_new_tokens_count=sum(v["hit_max_new_tokens"] for v in latency),
        capped_without_eos_count=sum(v["capped_without_eos"] for v in latency),
        first_eos_seen_count=sum(v["first_eos_index"] is not None for v in latency),
        early_eos_count=sum(v["early_eos"] for v in latency),
        post_first_eos_tokens_total=sum(v["tokens_after_first_eos"] for v in latency),
        prefill_retained_ratio_mean=statistics.fmean(ratios), prefill_retained_ratio_min=min(ratios),
        prefill_retained_ratio_max=max(ratios), budget_policy=latency[0]["budget_policy"],
        asymkv_boundary_affected_count=sum(bool(v["asymkv_boundary_affected"]) for v in latency),
        asymkv_missing_prompt_tokens_total=sum(v["asymkv_prompt_tokens_missing"] or 0 for v in latency),
        asymkv_compression_event_count=(sum(v["asymkv_compression_events"] or 0 for v in latency)
                                        if expected_arm == "asymkv" else None),
        model_load_seconds=state.get("model_load_seconds"), formal_loop_seconds=state.get("formal_loop_seconds"),
        process_seconds=state.get("process_seconds"))
    for name in PHASES:
        summary[name.replace("_seconds", "_mean_seconds")] = (None if expected_arm == "asymkv" and name == "fisher_seconds"
            else statistics.fmean(v[name] for v in latency))
    # Keep a stable, explicit CSV field order.
    summary = {key: summary[key] for key in SUMMARY_COLUMNS}
    audit = {"arm": expected_arm, "task": expected_task, "contract_sha256": json_sha(contract),
             "contract_file_sha256": file_sha(path / "run_contract.json"), "state_sha256": file_sha(path / "run_state.json"),
             "annotation_sha256": file_sha(annotation_file), "annotation_total_samples": len(complete["data"]),
             "all_samples": contract["all_samples"], "source_sha256": contract["source_sha256"], "sample_files": evidence,
             "excluded_warmup_files": [{"name": p.name, "sha256": file_sha(p)} for p in warmup_files]}
    return contract, rows, summary, latency, evaluation, audit


def validate_pairing(runs):
    evidence = []
    for task in TASKS:
        group = [(arm, result) for (arm, selected_task), result in runs.items() if selected_task == task]
        if len(group) < 2:
            continue
        reference_arm, reference = group[0]
        contract, rows = reference[0], reference[1]
        for arm, record in group[1:]:
            for key in ("selected_sample_ids", "annotation_sha256", "image", "model_metadata_sha256", "dtype", "attention", "model_seed"):
                require(record[0].get(key) == contract.get(key), f"Cross-method image input config changed: {key}")
            require(set(rows) == set(record[1]), "Cross-method image input IDs differ")
            for sample_id, sample in record[1].items():
                require(sample["input_fingerprints"] == rows[sample_id]["input_fingerprints"], "Cross-method processor inputs differ")
        evidence.append({"task": task, "scope": "all_requested_methods_input_identity", "arms": [arm for arm, _ in group],
                         "validated_sample_pairs": len(rows) * (len(group) - 1), "reference_arm": reference_arm, "status": "passed"})
    for backbone in ("hybridkv", "snapkv"):
        for task in TASKS:
            group = [(arm, record) for (arm, selected_task), record in runs.items()
                     if selected_task == task and arm.startswith(backbone + "_")]
            if len(group) < 2:
                continue
            reference_arm, reference = group[0]
            reference_contract, reference_rows = reference[0], reference[1]
            for arm, record in group[1:]:
                contract, samples = record[0], record[1]
                for key in ("selected_sample_ids", "generation_sample_ids", "annotation_sha256", "algorithm_env",
                            "image", "generation", "model_metadata_sha256", "dtype", "attention", "source_sha256", "budget_policy", "kernel_binary"):
                    require(contract.get(key) == reference_contract.get(key), f"Cross-arm fixed configuration changed: {key}")
                require(set(samples) == set(reference_rows), "Cross-arm input IDs differ")
                for sample_id, row in samples.items():
                    baseline = reference_rows[sample_id]
                    require(row["input_fingerprints"] == baseline["input_fingerprints"], "Cross-arm image inputs differ")
                    for key in ("image", "question", "gt_response", "gen_kwargs", "gen_model_id"):
                        require(row["prediction"].get(key) == baseline["prediction"].get(key), f"Cross-arm prediction inputs differ: {key}")
                    require(pairing_signature(row["audit"], backbone + "_kvrepack")
                            == pairing_signature(baseline["audit"], backbone + "_kvrepack"),
                            "Cross-arm native candidates/heads/groups/anchors changed")
                    require(row["budget"]["prefill_tokens_total"] == baseline["budget"]["prefill_tokens_total"],
                            "Cross-arm physical prefill work differs")
            evidence.append({"task": task, "backbone": backbone, "arms": [arm for arm, _ in group],
                             "validated_sample_pairs": len(reference_rows) * (len(group) - 1),
                             "reference_arm": reference_arm, "status": "passed"})
    return evidence


def csv_write(path, rows, columns):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def summarize(root, output):
    from kvrepack.benchmarks.milebench.evaluate import _output_path, write_evaluation
    root, output = Path(root).expanduser().resolve(), _output_path(output)
    require(not output.exists() and output != root and not root.is_relative_to(output), "Use a new report directory")
    controller_path = root / "experiment_state.json"
    controller = read_json(controller_path)
    require(controller.get("status") in ("generated_all", "scoring", "complete"), "Controller has not finished generation")
    matrix = controller.get("runs")
    require(isinstance(matrix, list) and matrix, "Missing requested experiment matrix")
    pairs = [(row.get("arm"), row.get("task")) for row in matrix]
    require(len(set(pairs)) == len(pairs) and all(arm in ARMS and task in TASKS for arm, task in pairs), "Duplicate/unknown arm-task runs")
    arms = controller.get("arms", sorted({arm for arm, _ in pairs}))
    tasks = controller.get("tasks", sorted({task for _, task in pairs}))
    require(set(pairs) == {(arm, task) for arm in arms for task in tasks}, "Requested matrix has missing runs")
    require(controller.get("completed_runs") == len(matrix) and all(row.get("status") == "complete" for row in matrix),
            "Controller has incomplete child runs")
    before = file_sha(controller_path)
    runs, summaries, latency, evaluations, evidence = {}, [], [], {}, []
    for row in matrix:
        run = root / row["arm"] / row["task"]
        require(Path(row["output"]).resolve() == run, "Controller run path escapes the matrix")
        require(not output.is_relative_to(Path(read_json(run / "run_contract.json")["data_root"]).resolve()), "Report must be outside benchmark data")
        result = read_run(run, row["arm"], row["task"])
        runs[row["arm"], row["task"]] = result
        summaries.append(result[2]); latency.extend(result[3]); evaluations[row["arm"], row["task"]] = result[4]; evidence.append(result[5])
    pairing = validate_pairing(runs)
    require(before == file_sha(controller_path), "Controller changed while reporting")
    for contract, *_ in runs.values():
        verify_sources(contract["source_sha256"])
    full = (set(pairs) == {(arm, task) for arm in ARMS for task in TASKS}
            and all(item["all_samples"] for item in evidence) and all(row["full_task"] for row in summaries))
    result = {"schema_version": 1, "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_root": str(root), "full_experiment": full,
              "scope": "full_16_run_annotation_matrix" if full else "selected_or_smoke_matrix_not_full_experiment",
              "completed_runs": len(matrix), "formal_generations": len(latency),
              "warmup_generations_excluded": sum(len(item["excluded_warmup_files"]) for item in evidence),
              "controller_generation_sha256": before, "results": summaries, "pairing": pairing,
              "source_and_sample_evidence": evidence,
              "timing_scope": "image request/read + RGB/JPEG + processor + H2D + applicable Fisher + native generation + text decode; model load/warmup/audit/write/scoring excluded",
              "model_e2e_scope": "Fisher + generate only, excludes text decode/H2D/preprocessing",
              "fisher_scope": "Standalone AsymKV Fisher is included in original generate and not separately measurable; empty CSV cells mean unmeasured, not zero cost.",
              "warmup_policy": "Only samples/ with warmup=false are scored; separate warmup/ records excluded.",
              "eos_policy": "Original standalone AsymKV may continue after its initial EOS. Full tokens/text are retained; first EOS and post-EOS counts are reported, while eos_reached means terminal EOS.",
              "budget_scope": "Original native policies retained; report physical prefill ratios without imposing strict 10%.",
              "limits": ["AsymKV released chunk-loop omissions, if any, are reported without dropping affected inputs.",
                         "End-to-end latency includes output-length differences and is not pure merge-operator time."]}
    output.mkdir(parents=True, exist_ok=False)
    for (arm, task), evaluation in evaluations.items():
        write_evaluation(evaluation, output / "conditions" / arm / task)
    csv_write(output / "comparison.csv", summaries, SUMMARY_COLUMNS)
    csv_write(output / "per_input_latency.csv", latency, PER_INPUT_COLUMNS)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# Qwen7B styled image experiments", "", f"Scope: {result['scope']}; {len(matrix)} runs; {len(latency)} formal generations.", "",
             "Spot: original ROUGE-L F; MMCoQA: original substring Accuracy. Scores are on a 0–100 scale.", "",
             "| Arm | Task | N | Score | Mean E2E (s) | P95 E2E (s) | Mean physical prefill ratio |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['arm']} | {row['task']} | {row['sample_count']} | {row['score_percent']:.6f} | {row['e2e_mean_seconds']:.6f} | {row['e2e_p95_seconds']:.6f} | {row['prefill_retained_ratio_mean']:.6f} |")
    lines += ["", "Primary E2E includes image I/O, JPEG conversion, preprocessing, H2D, Fisher where applicable, generation, and text decode. Model loading, warmup, hashes, writes and scoring are excluded.",
              "Standalone AsymKV Fisher is inside original generation; its separate Fisher time is unmeasured (empty CSV), not zero.",
              "Original AsymKV chunk-boundary omissions and original budget/cache differences are retained and reported, not repaired or filtered.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = summarize(args.root, args.output)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f"Styled MileBench report failed: {error}\n")
    print(json.dumps({"status": result["status"], "full_experiment": result["full_experiment"],
                      "completed_runs": result["completed_runs"], "formal_generations": result["formal_generations"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

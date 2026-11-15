"""Offline WUPS and response E2E for original standalone LOOK-M/AsymKV.

Each method owns one fresh single-row-per-input run. LOOK-M retains the old
history-plus-recent policy; AsymKV retains its own complete original algorithm.
The report never relabels either method as a fixed-group merge transplant.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from ..ablations.run_head_selection import atomic_json, json_sha, sha256
from ..ablations.nextqa_merge_report import _reference, _verify_model, score_rows

TASK = "nextqa_oe_test"
METHODS = ("lookm", "asymkv")
POLICIES = {"lookm": "original_lookm_history_plus_recent_v1",
            "asymkv": "original_asymkv_official_v1"}
DECODING_POLICIES = {"lookm": "hf_greedy_repetition_penalty_1p05",
                     "asymkv": "official_argmax_no_repetition_penalty"}
SHARED_FIELDS = ("task", "model_snapshot", "model_metadata_sha256", "frames",
    "min_pixels", "max_pixels", "dtype", "max_new_tokens", "do_sample",
    "budget_ratio", "arrow_sha256", "dataset_rows",
    "task_yaml_sha256", "task_utils_sha256")
PHASES = ("preprocessing_seconds", "h2d_seconds", "generation_seconds", "text_decode_seconds")
REQUIRED_SOURCES = ("src/kvrepack/ablations/video_e2e_inputs.py",)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Invalid {name}")
    return value


def _sources(contract, repo_root):
    sources = contract.get("source_sha256")
    if not isinstance(sources, dict) or not set(REQUIRED_SOURCES).issubset(sources):
        raise ValueError("Contract lacks original generation and input source identities")
    for relative, expected in sources.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Generation sources must be repository-relative paths")
        path = (repo_root/relative).resolve()
        if not path.is_relative_to(repo_root.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Generation source is missing or changed: {relative}")


def _load_case(case, method, count, repo_root):
    contract, state = _read(case/"run_contract.json"), _read(case/"run_state.json")
    if (contract.get("method") != method or contract.get("task") != TASK
            or type(contract.get("target_docs")) is not int or contract["target_docs"] != count
            or contract.get("target_generations") != count
            or contract.get("budget_policy") != POLICIES[method]
            or contract.get("budget_ratio") != .1 or contract.get("frames") != 64
            or contract.get("min_pixels") != 200704 or contract.get("max_pixels") != 1605632
            or contract.get("dtype") != "bfloat16" or contract.get("max_new_tokens") != 128
            or contract.get("do_sample") is not False
            or "repetition_penalty" not in contract
            or contract["repetition_penalty"] != (1.05 if method == "lookm" else None)
            or contract.get("decoding_policy") != DECODING_POLICIES[method]
            or contract.get("diagnostic_audit", False) is not False):
        raise ValueError("Not the requested original standalone NExTQA/7B/64-frame protocol")
    if (state.get("status") != "complete" or state.get("completed_docs") != count
            or state.get("completed_generations") != count
            or state.get("contract_sha256") != json_sha(contract)):
        raise ValueError("A report requires a complete method with its frozen contract")
    paths = sorted((case/"samples").glob("*.json"))
    if [p.name for p in paths] != [f"{i:06d}.json" for i in range(count)]:
        raise ValueError("Missing, duplicated or out-of-prefix original input records")
    _sources(contract, repo_root)
    return contract, state, paths


def _row(row, contract, method, doc_id, original):
    ids = row.get("generated_ids")
    if (row.get("status") != "completed" or row.get("method") != method
            or row.get("task") != TASK or type(row.get("doc_id")) is not int or row["doc_id"] != doc_id
            or row.get("warmup", False) is not False
            or not isinstance(row.get("text"), str) or row.get("prediction") != row["text"]
            or not isinstance(ids, list) or not ids or len(ids) > 128
            or any(type(token) is not int or token < 0 for token in ids)
            or row.get("generated_tokens") != len(ids)
            or type(row.get("eos_reached")) is not bool or type(row.get("capped_without_eos")) is not bool):
        raise ValueError("Wrong standalone method, original row, prediction or generation accounting")
    if row.get("contract_sha256", json_sha(contract)) != json_sha(contract):
        raise ValueError("Prediction row uses a different generation contract")
    if row["capped_without_eos"] != (len(ids) == 128 and not row["eos_reached"]):
        raise ValueError("EOS and generation-cap flags disagree")
    if "eos_token_id" in row and row["eos_reached"] != (ids[-1] == row["eos_token_id"]):
        raise ValueError("Actual final token and EOS flag disagree")
    n, meta = row.get("input_tokens"), row.get("reference_metadata", {})
    video_id = str(original["video"])
    if (type(n) is not int or n < 1 or meta.get("task") != TASK or meta.get("doc_id") != doc_id
            or meta.get("video_id") != video_id or meta.get("source_video_id") != video_id
            or meta.get("input_tokens") != n or meta.get("arrow_sha256") != contract["arrow_sha256"]
            or meta.get("original_row_sha256") != json_sha(original)):
        raise ValueError("Prediction does not identify the original question/video/reference row")
    indices = meta.get("frame_indices")
    decoded = meta.get("decoded_frames")
    if (not isinstance(indices, list) or len(indices) != 64
            or type(decoded) is not int or decoded <= 0
            or any(type(index) is not int or not 0 <= index < decoded for index in indices)):
        raise ValueError("Prediction lacks the actual original 64-frame sampling evidence")
    fingerprints = row.get("input_fingerprints", {})
    hashes, shapes, dtypes = [fingerprints.get(key, {}) for key in ("input_hashes", "input_shapes", "input_dtypes")]
    if (not isinstance(hashes, dict) or not hashes or set(hashes) != set(shapes) or set(hashes) != set(dtypes)
            or shapes.get("input_ids") != [1, n]
            or not hashes.get("input_ids") or not hashes.get("pixel_values_videos")
            or any(not isinstance(value, str) or not value for value in hashes.values())):
        raise ValueError("Prediction lacks actual processor input tensor fingerprints")
    runtime = row.get("runtime", {})
    total = _number(runtime.get("end_to_end_seconds"), "full response E2E", positive=True)
    for name in PHASES:
        _number(runtime.get(name), name, positive=name == "generation_seconds")
    if abs(total-sum(runtime[name] for name in PHASES)) > 1e-5:
        raise ValueError("Full E2E is inconsistent with its input/generation/text phases")
    if "peak_allocated_gib" in runtime:
        _number(runtime["peak_allocated_gib"], "peak GPU allocation")
    # The complete official AsymKV generation function owns its Fisher pass;
    # generation_seconds includes it. Never add a second invented Fisher cost.
    return row


def _percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered)-1)*fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)


def _timings(rows, state):
    totals = [row["runtime"]["end_to_end_seconds"] for row in rows]
    measured_sum = sum(totals)
    timed_sum = _number(state.get("timed_e2e_seconds_sum"), "recorded timed E2E sum", positive=True)
    loop = _number(state.get("response_loop_seconds"), "formal response-loop time", positive=True)
    if abs(measured_sum-timed_sum) > 1e-5:
        raise ValueError("Formal timed E2E sum differs from the actual sample records")
    if loop+1e-5 < measured_sum:
        raise ValueError("Formal response loop is shorter than its contained E2E measurements")
    result = dict(n=len(rows), end_to_end_seconds_mean=statistics.mean(totals),
        end_to_end_seconds_total=measured_sum, e2e_p50_seconds=statistics.median(totals),
        e2e_p95_seconds=_percentile(totals, .95), response_loop_seconds=loop,
        response_loop_seconds_per_input=loop/len(rows),
        untimed_loop_overhead_seconds=loop-measured_sum,
        generated_tokens_mean=statistics.mean(row["generated_tokens"] for row in rows),
        eos_count=sum(row["eos_reached"] for row in rows),
        capped_without_eos=sum(row["capped_without_eos"] for row in rows))
    for name in PHASES:
        values = [row["runtime"][name] for row in rows]
        result[name+"_mean"], result[name+"_total"] = statistics.mean(values), sum(values)
    peaks = [row["runtime"]["peak_allocated_gib"] for row in rows if "peak_allocated_gib" in row["runtime"]]
    if peaks:
        result["peak_allocated_gib_max"] = max(peaks)
    return result


def _csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(dict.fromkeys(name for row in rows for name in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report(run_root, hf_home, output, *, limit=200, repo_root=None, methods=METHODS):
    """Validate complete original inputs, then independently score each corpus.

    ``methods`` may contain one method for a pilot or an intermediate CPU
    report; the controller's final comparison uses the default two methods.
    """
    if type(limit) is not int or limit not in (1, 200):
        raise ValueError("Only explicit first1 pilot or first200 formal reports are supported")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError("Choose unique original standalone methods")
    run_root, hf_home, output = map(Path, (run_root, hf_home, output))
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    cases, raw_hashes = {}, {}
    for method in methods:
        case = run_root/method
        contract, state, paths = _load_case(case, method, limit, repo_root)
        cases[method] = dict(contract=contract, state=state, paths=paths)
        for path in (case/"run_contract.json", case/"run_state.json", *paths):
            raw_hashes[str(path)] = sha256(path)
    reference = cases[methods[0]]["contract"]
    for case in cases.values():
        contract = case["contract"]
        if any(name not in contract or name not in reference or contract[name] != reference[name] for name in SHARED_FIELDS):
            raise ValueError("The standalone methods use different model/input/generation protocols")
        for name in ("model_seed", "versions"):
            if contract.get(name) != reference.get(name):
                raise ValueError(f"Standalone method {name} differs")
    _verify_model(reference)
    arrow, originals = _reference(hf_home, reference, limit, repo_root)
    records = {method: [] for method in methods}
    for doc_id, original in enumerate(originals):
        paired = []
        for method in methods:
            case = cases[method]
            row = _row(_read(case["paths"][doc_id]), case["contract"], method, doc_id, original)
            records[method].append(row)
            paired.append(row)
        if any(row[key] != paired[0][key] for row in paired for key in
               ("reference_metadata", "input_fingerprints", "input_tokens")):
            raise ValueError("Standalone methods did not receive the same original processor input")
    timings = {method: _timings(records[method], cases[method]["state"]) for method in methods}
    # Do not create or score partial/invalid corpora.
    output.mkdir(parents=True, exist_ok=False)
    comparisons, per_input, scoring = [], [], {}
    for method in methods:
        contract = cases[method]["contract"]
        wups, proof = score_rows(records[method], contract, arrow, output/"scoring"/method, repo_root)
        scoring[method] = proof
        comparisons.append(dict(method=method, WUPS=wups, budget_policy=POLICIES[method],
            decoding_policy=contract["decoding_policy"], repetition_penalty=contract["repetition_penalty"],
            **timings[method]))
        for row in records[method]:
            per_input.append(dict(method=method, doc_id=row["doc_id"], generated_tokens=row["generated_tokens"],
                eos_reached=row["eos_reached"], capped_without_eos=row["capped_without_eos"], **row["runtime"]))
    for path, expected in raw_hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError("Recorded generation evidence changed during scoring")
    for case in cases.values():
        _sources(case["contract"], repo_root)
    if sha256(arrow) != reference["arrow_sha256"]:
        raise ValueError("Original scoring reference changed during scoring")
    _csv(output/"comparison.csv", comparisons)
    _csv(output/"per_input_latency.csv", per_input)
    summary = dict(schema_version=1, status="complete", task=TASK,
        scope="formal_first200_original_standalone" if limit == 200 else "pilot_first1_original_standalone",
        methods=list(methods), docs_per_method=limit, total_generations=limit*len(methods),
        gpu_execution=False, comparisons=comparisons, scoring=scoring,
        contracts={method: data["contract"] for method, data in cases.items()},
        contract_identities={method: json_sha(data["contract"]) for method, data in cases.items()},
        source_files_sha256=raw_hashes, reference_arrow=str(arrow), reference_arrow_sha256=reference["arrow_sha256"],
        input_pairing=("Exact row/video/frame metadata and all processor tensor fingerprints identical across methods"
            if len(methods) > 1 else "Single method; no cross-method input pairing claimed"),
        timing_scope="Per-input full E2E: video reading/decoding, processor, H2D, complete original method generation (including AsymKV Fisher where required), and text decoding. Model load and after-timer fingerprints/audits/writes/scoring are excluded.",
        response_loop_scope="Formal loop wall time includes the measured input E2E intervals plus intervening record/audit/log/write overhead. Report this separately from timed E2E.",
        limitations=["Original standalone methods, not fixed-group merge transplants.",
            "LOOK-M uses original historical budget plus 32 recent; AsymKV uses its own original budget and algorithm, without an invented +32 extension.",
            "Original generation behavior is method-specific: LOOK-M uses HF greedy decoding with repetition penalty1.05; official AsymKV uses its native argmax loop without that HF penalty.",
            "Single latency observation per input; no repeated steady-state timing or significance claim.",
            "Historical response-loop averages and these two reported timing scopes must not be silently interchanged."])
    atomic_json(output/"summary.json", summary)
    lines = ["# Original standalone LOOK-M / AsymKV: NExTQA", "",
        f"Qwen2.5-VL-7B, original first {limit} rows, 64 frames. Original method budgets and complete algorithms.", "",
        "| Method | N | WUPS | Timed response E2E mean (s) | P50 (s) | P95 (s) | Formal loop / input (s) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        values = [row["method"], str(row["n"])]+[f"{row[key]:.6f}" for key in
            ("WUPS", "end_to_end_seconds_mean", "e2e_p50_seconds", "e2e_p95_seconds", "response_loop_seconds_per_input")]
        lines.append("| "+" | ".join(values)+" |")
    lines += ["", summary["timing_scope"], "", summary["response_loop_scope"], "",
              *["- "+line for line in summary["limitations"]]]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "hf-home", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--limit", type=int, choices=(1, 200), default=200)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    summary = build_report(args.run_root.expanduser().resolve(), args.hf_home.expanduser().resolve(),
        args.output.expanduser().resolve(), limit=args.limit, methods=args.methods)
    print(json.dumps(dict(status=summary["status"], task=TASK, scope=summary["scope"],
                         methods=summary["methods"], output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

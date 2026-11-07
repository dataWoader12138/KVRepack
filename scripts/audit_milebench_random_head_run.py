#!/usr/bin/env python3
"""Read-only CPU audit of the full Qwen7B seed-0/1 image experiment matrix.

An active controller produces a progress snapshot, never a partial score.
Only a complete four-run controller can pass the scoring audit. Output must
be a new directory outside the experiment and annotation directories.
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

from kvrepack.ablations.milebench_head_selection_report import json_sha, merge_work_stats, validate_bundle

METHODS = ("hybridkv_kvrepack", "snapkv_kvrepack")
TASKS = ("Spot-the-Diff", "MMCoQA")
LABELS = ("similarity", "random_global_seed0", "random_global_seed1")
REFERENCE = "6c65bf801ab1e5d0cd85f1d24991e3d5399bebcf"
BOUND_SOURCES = tuple("src/kvrepack/ablations/" + name for name in (
    "head_selection.py", "hybrid_head_selection.py", "run_milebench_head_selection.py",
    "milebench_paired_worker.py", "milebench_head_selection_report.py")) + tuple(
    "src/kvrepack/benchmarks/milebench/" + name for name in ("dataset.py", "evaluate.py", "workers.py")) + tuple(
    "src/kvrepack/" + name for name in ("hybridkv_utils.py", "qwen_model_snapkv_kvrepack.py",
                                          "modeling_qwen2_5_vl.py", "configs/param_config.json"))
RESULT_COLUMNS = ["method", "task", "metric", "sample_count", "similarity_percent", "seed0_percent",
                  "seed1_percent", "random_mean_percent", "random_sample_sd_percent",
                  "similarity_minus_seed0_points", "similarity_minus_seed1_points",
                  "similarity_minus_random_mean_points"]


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, location):
    """Compare serialized evidence, with tolerance only for real metric values."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, f"{location}: value mismatch")
    elif isinstance(expected, (int, float)):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
                f"{location}: numeric mismatch ({actual!r} versus {expected!r})")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{location}: keys mismatch")
        for key, value in expected.items():
            compare(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{location}: list coverage mismatch")
        for index, (value, wanted) in enumerate(zip(actual, expected)):
            compare(value, wanted, f"{location}[{index}]")
    else:
        raise TypeError(f"Unsupported audit value at {location}: {type(expected)}")


def compare_csv(path, expected):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        require(bool(expected) and reader.fieldnames == list(expected[0]), f"{path}: CSV columns mismatch")
    require(len(rows) == len(expected), f"{path}: CSV row coverage mismatch")
    for index, (actual, wanted) in enumerate(zip(rows, expected)):
        for key, value in wanted.items():
            if type(value) in (float, int):
                try:
                    number = float(actual[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{index}:{key}: invalid number") from exc
                compare(number, value, f"{path}:{index}:{key}")
            else:
                require(actual[key] == str(value), f"{path}:{index}:{key}: value mismatch")


def runtime_stats(rows, maximum):
    lengths, seconds, inputs = [], [], []
    eos_count = hit_count = truncated_count = 0
    for runtime in rows:
        require(isinstance(runtime, dict), "Missing generation runtime evidence")
        ids, eos = runtime.get("generated_ids"), runtime.get("eos_token_id")
        require(isinstance(ids, list) and ids and all(type(token) is int and token >= 0 for token in ids),
                "Invalid generated token IDs")
        require(type(eos) is int and eos >= 0, "Invalid EOS token ID")
        require(type(runtime.get("generated_tokens")) is int and runtime["generated_tokens"] == len(ids)
                and len(ids) <= maximum, "Generated token count disagrees with IDs or maximum")
        reached, hit = ids[-1] == eos, len(ids) == maximum
        for field, expected in (("eos_reached", reached), ("hit_max_new_tokens", hit),
                                ("truncated_without_eos", hit and not reached)):
            require(type(runtime.get(field)) is bool and runtime[field] == expected,
                    f"Generation runtime {field} disagrees with token IDs")
        require(eos not in ids[:-1], "Generation continued after its recorded EOS")
        require(type(runtime.get("input_tokens")) is int and runtime["input_tokens"] > 0,
                "Invalid input token count")
        duration = runtime.get("generation_seconds")
        require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                "Invalid generation duration")
        lengths.append(len(ids)); inputs.append(runtime["input_tokens"]); seconds.append(duration)
        eos_count += reached; hit_count += hit; truncated_count += hit and not reached
    return {"generation_count": len(rows), "eos_count": eos_count, "hit_max_new_tokens_count": hit_count,
            "truncated_without_eos_count": truncated_count,
            "stopped_without_eos_below_limit_count": len(rows) - eos_count - truncated_count,
            "generated_tokens_total": sum(lengths), "generated_tokens_min": min(lengths),
            "generated_tokens_max": max(lengths), "generated_tokens_mean": statistics.mean(lengths),
            "generated_tokens_median": statistics.median(lengths),
            "generated_tokens_sample_sd": statistics.stdev(lengths) if len(lengths) > 1 else None,
            "input_tokens_min": min(inputs), "input_tokens_max": max(inputs),
            "generation_seconds_total": sum(seconds)}


def verify_bound_sources(bound, source_archives=()):
    """Verify exact executed bytes, allowing only explicitly supplied archives.

    Completed runs retain their original contracts after a reporting fix. An
    archive supplies those historical bytes; it never changes the contract.
    The current validator and scorer still recompute all recorded evidence.
    """
    require(set(bound) == set(BOUND_SOURCES), "Bound source file coverage mismatch")
    roots = [ROOT, *(Path(path).expanduser().resolve() for path in source_archives)]
    resolved = {}
    for relative, expected in bound.items():
        for root in roots:
            path = (root / relative).resolve()
            if path.is_relative_to(root) and path.is_file() and file_sha(path) == expected:
                resolved[relative] = {"path": str(path), "sha256": expected}
                break
        else:
            raise ValueError(f"Bound experiment source changed or absent from explicit archives: {relative}")
    return resolved


def audit_one(run, method, task, source_archives=()):
    # No report reader is used here: complete annotation and output coverage are
    # checked independently before the original metric implementation is called.
    from kvrepack.benchmarks.milebench.dataset import annotation_path, load_annotation
    from kvrepack.benchmarks.milebench.evaluate import evaluate_predictions

    contract = read_json(run / "run_contract.json")
    require(contract.get("method") == method and contract.get("task") == task, "Contract method/task mismatch")
    require(contract.get("reference_commit") == REFERENCE, "Contract reference commit mismatch")
    require(contract.get("all_samples") is True, "Contract must explicitly select all_samples")
    require(contract.get("conditions") == list(LABELS), "Require exactly similarity, random seed 0 and seed 1")
    compare(contract.get("condition_settings"), [
        {"label": label, "strategy": "similarity" if index == 0 else "random_global", "mask_seed": max(0, index - 1)}
        for index, label in enumerate(LABELS)], "contract.condition_settings")
    require(contract.get("head_geometry") == {"layers": 28, "native_kv_heads": 4}, "Require Qwen7B 28 x 4 native KV heads")
    require(contract.get("model_type") == "qwen2_5_vl-7b", "Require Qwen2.5-VL-7B model type")
    ratio_key = "HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO" if method == METHODS[0] else "SNAPKV_SIMHEAD_MERGE_RATIO"
    ratio, quota = (0.5, 56) if method == METHODS[0] else (0.3, 34)
    require(float(contract["algorithm_env"][ratio_key]) == ratio, "Unexpected native selected-head ratio")
    bound = contract.get("source_files_sha256", {})
    verified_sources = verify_bound_sources(bound, source_archives)
    annotation_file = annotation_path(contract["data_root"], task)
    require(file_sha(annotation_file) == contract.get("annotation_sha256"), "Complete annotation SHA mismatch")
    annotation = load_annotation(contract["data_root"], task)  # Intentionally no prefix limit.
    count = len(annotation["data"])
    require(type(contract.get("limit")) is int and contract["limit"] == count
            and contract.get("annotation_total_samples") == count
            and annotation["meta_data"].get("num_sample") == count, "Full annotation count mismatch")
    ids = [sample["sample_id"] for sample in annotation["data"]]
    require(contract.get("selected_sample_ids") == ids, "Full annotation ID coverage/order mismatch")
    generation_ids = contract.get("generation_sample_ids")
    require(isinstance(generation_ids, list) and len(generation_ids) == count and set(generation_ids) == set(ids),
            "Generation IDs do not cover the complete annotation")
    identity = json_sha(contract)
    state = read_json(run / "ablation_state.json")
    require(state.get("status") == "complete" and state.get("contract_sha256") == identity,
            "Run state is incomplete or bound to another contract")
    for field, expected in (("target_samples", count), ("completed_samples", count),
                            ("target_generations", count * 3), ("completed_generations", count * 3)):
        require(type(state.get(field)) is int and state[field] == expected, f"Run state {field} mismatch")
    paths = sorted((run / "samples").iterdir())
    require([path.name for path in paths] == [f"{i:06d}.json" for i in range(count)]
            and all(path.is_file() and not path.is_symlink() for path in paths), "Sample filename coverage mismatch")
    predictions = {label: [] for label in LABELS}
    runtimes = {label: [] for label in LABELS}
    sample_evidence, selector_rows, fixed_masks = [], [], {}
    work = {label: [] for label in LABELS}
    for index, path in enumerate(paths):
        bundle = read_json(path)
        require(bundle.get("sample_index") == index, "Bundle index disagrees with filename")
        audited = validate_bundle(bundle, contract, identity)
        sample_evidence.append({"sample_id": bundle["sample_id"], "sample_file": path.name,
                                "sample_sha256": file_sha(path),
                                "input_fingerprints_sha256": json_sha(bundle["input_fingerprints"])})
        for label in LABELS:
            record = bundle["records"][label]
            require(audited[label]["selected_count"] == quota, "Selected merge-head quota mismatch")
            effective = audited[label]["effective_merge_count"]
            require(0 < effective <= quota and (method == METHODS[0] or effective == quota),
                    "Actual effective merge count disagrees with the reference method")
            work[label].append(merge_work_stats(record["audit"]))
            if label != "similarity":
                mask = record["audit"]["selected_masks"]
                require(fixed_masks.setdefault(label, mask) == mask, "Random mask changed across input samples")
            predictions[label].append(record["prediction"])
            runtimes[label].append(record.get("runtime"))
            selector_rows.append({"sample_id": bundle["sample_id"], "condition": label, **audited[label]})
    maximum = contract.get("generation", {}).get("max_new_tokens")
    require(type(maximum) is int and maximum > 0, "Invalid frozen generation maximum")
    generation = [{"method": method, "task": task, "condition": label, **runtime_stats(runtimes[label], maximum)}
                  for label in LABELS]
    work_stats = []
    for label in LABELS:
        stats = {"method": method, "task": task, "condition": label, "sample_count": count}
        for key in work[label][0]:
            values = [row[key] for row in work[label]]
            stats.update({key + "_min": min(values), key + "_max": max(values), key + "_total": sum(values)})
        work_stats.append(stats)
    report = run / "report"  # Always derived from --run-root, never from a report pointer.
    require({p.name for p in report.iterdir()} == {"conditions", "summary.json", "scores.csv", "paired_scores.csv",
                                                "selector_audit.csv", "RESULTS.md"}, "Report tree coverage mismatch")
    require({p.name for p in (report / "conditions").iterdir()} == set(LABELS), "Report condition coverage mismatch")
    metric = "Accuracy" if task == "MMCoQA" else "Rouge-L f"
    by_id = {sample["sample_id"]: sample for sample in annotation["data"]}
    scores, arms, score_rows = {}, {}, []
    for label in LABELS:
        evaluation = evaluate_predictions(predictions[label], annotation, task)
        folder = report / "conditions" / label
        expected_files = {"eval.json", "eval_score.json", "evaluation_metadata.json"}
        require({p.name for p in folder.iterdir()} == expected_files, f"{label}: report evaluation files mismatch")
        compare(read_json(folder / "eval.json"), evaluation["eval"], f"{label}.eval")
        compare(read_json(folder / "eval_score.json"), evaluation["eval_score"], f"{label}.eval_score")
        metadata = {key: evaluation[key] for key in ("task", "sample_ids", "sample_count", "annotation_total_samples", "full_task")}
        compare(read_json(folder / "evaluation_metadata.json"), metadata, f"{label}.evaluation_metadata")
        scores[label] = {}
        for prediction in predictions[label]:
            sample_id = prediction["sample_id"]
            singleton = {"meta_data": deepcopy(annotation["meta_data"]), "data": [deepcopy(by_id[sample_id])]}
            value = float(evaluate_predictions([prediction], singleton, task)["eval"][metric])
            scores[label][sample_id] = value * 100
            score_rows.append({"sample_id": sample_id, "condition": label, "metric": metric,
                               "score": value, "score_percent": value * 100})
        arms[label] = statistics.mean(scores[label].values())
        compare(float(evaluation["eval"][metric]) * 100, arms[label], f"{label}.aggregate")
    paired_rows = []
    for sample_id in ids:
        native = scores["similarity"][sample_id]
        random = [scores[label][sample_id] for label in LABELS[1:]]
        for label, value in [(label, scores[label][sample_id]) for label in LABELS[1:]] + [("random_seed_mean", statistics.mean(random))]:
            paired_rows.append({"sample_id": sample_id, "random_condition": label, "similarity_percent": native,
                                "random_percent": value, "similarity_minus_random_points": native - value})
    compare_csv(report / "scores.csv", score_rows)
    compare_csv(report / "paired_scores.csv", paired_rows)
    compare_csv(report / "selector_audit.csv", selector_rows)
    random = [arms[label] for label in LABELS[1:]]
    mean, sd = statistics.mean(random), statistics.stdev(random)
    expected_summary = {"schema_version": 1, "status": "complete", "task": task, "method": method,
                        "contract_sha256": identity, "conditions": list(LABELS), "sample_count": count,
                        "generation_count": count * 3, "metric": metric, "metric_scale": "percent",
                        "full_task": True, "annotation_total_samples": count, "arm_scores_percent": arms,
                        "random_mean_percent": mean, "random_sample_sd_percent": sd, "random_seed_count": 2,
                        "similarity_minus_random_mean_points": arms["similarity"] - mean,
                        "per_seed_paired_difference_points": {label: arms["similarity"] - arms[label] for label in LABELS[1:]},
                        "selected_head_quota": quota, "pairing_validation": "recomputed_passed", "sample_files": sample_evidence}
    summary = read_json(report / "summary.json")
    for key, expected in expected_summary.items():
        compare(summary.get(key), expected, "summary." + key)
    require((report / "RESULTS.md").is_file() and (report / "RESULTS.md").stat().st_size > 0,
            "Human-readable report missing")
    row = {"method": method, "task": task, "metric": metric, "sample_count": count,
           "similarity_percent": arms["similarity"], "seed0_percent": random[0], "seed1_percent": random[1],
           "random_mean_percent": mean, "random_sample_sd_percent": sd,
           "similarity_minus_seed0_points": arms["similarity"] - random[0],
           "similarity_minus_seed1_points": arms["similarity"] - random[1],
           "similarity_minus_random_mean_points": arms["similarity"] - mean}
    evidence = {"method": method, "task": task, "status": "passed", "run": str(run), "report": str(report),
                "contract_sha256": identity, "annotation_path": str(annotation_file),
                "annotation_sha256": file_sha(annotation_file), "annotation_total_samples": count,
                "selected_head_quota": quota, "validated_bundles": count, "validated_generations": count * 3,
                "random_mask_sha256": {label: json_sha(mask) for label, mask in fixed_masks.items()},
                "bound_source_files": bound, "verified_source_locations": verified_sources,
                "sample_files": sample_evidence, "scores": row, "merge_work_stats": work_stats,
                "report_files_sha256": {path.relative_to(report).as_posix(): file_sha(path)
                                        for path in sorted(report.rglob("*")) if path.is_file()},
                "generation_stats": generation, "report_summary_sha256": file_sha(report / "summary.json")}
    return evidence, row, generation


def audit_run(run_root, source_archives=()):
    root = Path(run_root).expanduser().resolve()
    controller_path = root / "controller_state.json"
    controller = read_json(controller_path)
    expected = [(method, task) for method in METHODS for task in TASKS]
    runs = controller.get("runs")
    require(controller.get("schema_version") == 1 and controller.get("reference_commit") == REFERENCE,
            "Controller schema/reference mismatch")
    require(controller.get("planned_runs") == 4 and isinstance(runs, list) and len(runs) == 4
            and [(row.get("method"), row.get("task")) for row in runs] == expected, "Controller matrix coverage mismatch")
    for row in runs:
        require(Path(row["output"]).resolve() == root / row["method"] / row["task"], "Controller run output escapes matrix")
    result = {"schema_version": 1, "audited_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(root),
              "controller_status": controller.get("status"), "controller_sha256": file_sha(controller_path),
              "status": "progress", "scoring_performed": False, "runs": [],
              "explicit_source_archives": [str(Path(path).expanduser().resolve()) for path in source_archives]}
    if controller.get("status") != "complete":
        for row in runs:
            run = root / row["method"] / row["task"]
            state_path = run / "ablation_state.json"
            state = read_json(state_path) if state_path.is_file() else {}
            result["runs"].append({"method": row["method"], "task": row["task"], "controller_run_status": row.get("status"),
                                   "generation_status": state.get("status", "not_started"),
                                   "committed_sample_files": len(list((run / "samples").glob("*.json"))),
                                   **{key: state.get(key) for key in ("completed_samples", "target_samples", "completed_generations", "target_generations")}})
        return result
    require(controller.get("completed_runs") == 4 and all(row.get("status") == "complete" and row.get("returncode") == 0 for row in runs),
            "Complete controller has incomplete/failed child runs")
    contract_paths = {path.resolve() for path in root.rglob("run_contract.json")}
    require(contract_paths == {root / method / task / "run_contract.json" for method, task in expected},
            "Output tree has missing or extra method/task runs")
    results, generation = [], []
    for method, task in expected:
        row = next(row for row in runs if row["method"] == method and row["task"] == task)
        run = root / method / task
        require(Path(row.get("report", "")).resolve() == run / "report", "Controller report pointer differs from derived path")
        evidence, scores, stats = audit_one(run, method, task, source_archives)
        result["runs"].append(evidence); results.append(scores); generation.extend(stats)
    require(file_sha(controller_path) == result["controller_sha256"], "Controller changed during final audit")
    result.update(status="passed", scoring_performed=True, results=results, generation_stats=generation,
                  merge_work_stats=[row for run in result["runs"] for row in run["merge_work_stats"]],
                  statistical_scope="SD is the sample SD across two fixed model-level head seeds; it is not input uncertainty.")
    return result


def write_audit(run_root, output, source_archives=()):
    from kvrepack.benchmarks.milebench.evaluate import _output_path
    root, output = Path(run_root).expanduser().resolve(), _output_path(output)
    require(not output.exists() and not output.is_relative_to(root) and not root.is_relative_to(output),
            "Audit output must be a new directory outside the experiment tree")
    for contract_path in root.glob("*/*/run_contract.json"):
        data = Path(read_json(contract_path)["data_root"]).expanduser().resolve()
        require(not output.is_relative_to(data), "Audit output must be outside annotation data")
    try:
        result = audit_run(root, source_archives)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        result = {"schema_version": 1, "status": "failed", "run_root": str(root), "scoring_performed": False,
                  "error_type": type(error).__name__, "error": str(error)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if result["status"] == "passed":
        for name, rows, columns in (("results.csv", result["results"], RESULT_COLUMNS),
                                     ("generation_stats.csv", result["generation_stats"], list(result["generation_stats"][0])),
                                     ("merge_work_stats.csv", result["merge_work_stats"], list(result["merge_work_stats"][0]))):
            with (output / name).open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, action="append", default=[],
                        help="explicit archive of unchanged executed source bytes for a completed historical run")
    args = parser.parse_args(argv)
    try:
        result = write_audit(args.run_root, args.output, args.source_archive)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f"Image Random-head audit failed: {error}\n")
    print(json.dumps({"status": result["status"], "scoring_performed": result["scoring_performed"],
                      "output": str(args.output.resolve()), "error": result.get("error")}, ensure_ascii=False))
    return 2 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

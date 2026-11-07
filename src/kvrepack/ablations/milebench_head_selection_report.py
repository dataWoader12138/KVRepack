"""Offline scoring and pairing checks for MileBench Random-head experiments.

Only the original MileBench scoring protocol is used. No inference package is
imported, no image/model is loaded, and no download or GPU call is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


METHODS = ("hybridkv_kvrepack", "snapkv_kvrepack")
TASKS = ("Spot-the-Diff", "MMCoQA")
ROOT = Path(__file__).resolve().parents[3]


def json_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def conditions(contract: dict) -> list[tuple[str, str, int]]:
    labels = contract.get("conditions")
    if (not isinstance(labels, list) or len(labels) < 2 or labels[0] != "similarity"
            or len(labels) != len(set(labels))):
        raise ValueError("Require similarity followed by unique fixed random-head seed labels")
    result = [("similarity", "similarity", 0)]
    for label in labels[1:]:
        match = re.fullmatch(r"random_global_seed([0-9]+)", label) if isinstance(label, str) else None
        if match is None:
            raise ValueError("Only fixed random_global_seedN conditions are supported")
        result.append((label, "random_global", int(match.group(1))))
    if len({seed for _, _, seed in result[1:]}) != len(result) - 1:
        raise ValueError("Random head seed labels must encode different seeds")
    return result


def _geometry(contract: dict) -> tuple[int, int, float, int]:
    geometry = contract.get("head_geometry", {})
    layers, heads = geometry.get("layers"), geometry.get("native_kv_heads")
    if any(type(value) is not int or value < 1 for value in (layers, heads)):
        raise ValueError("Contract requires positive native head geometry")
    key = ("HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO" if contract.get("method") == "hybridkv_kvrepack"
           else "SNAPKV_SIMHEAD_MERGE_RATIO")
    try:
        ratio = float(contract["algorithm_env"][key])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Contract requires its exact selected-head ratio") from exc
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("Random-head experiment requires a positive finite head-selection ratio")
    return layers, heads, ratio, min(layers * heads, math.ceil(layers * heads * ratio))


def _mask(mask, layers: int, heads: int, hybrid: bool) -> list[bool]:
    if hybrid:
        if not isinstance(mask, list) or len(mask) != layers:
            raise ValueError("Hybrid mask must have native [layers,batch=1,KV-heads] layout")
        rows = []
        for layer in mask:
            if not isinstance(layer, list) or len(layer) != 1:
                raise ValueError("Hybrid mask must have batch size one")
            rows.append(layer[0])
    else:
        if not isinstance(mask, list) or len(mask) != 1 or not isinstance(mask[0], list) or len(mask[0]) != layers:
            raise ValueError("SnapKV mask must have native [batch=1,layers,KV-heads] layout")
        rows = mask[0]
    if any(not isinstance(row, list) or len(row) != heads or any(type(bit) is not bool for bit in row) for row in rows):
        raise ValueError("Native mask dimensions or Boolean values are invalid")
    return [bit for row in rows for bit in row]


def _has_fingerprint(value) -> bool:
    if isinstance(value, dict):
        if "sha256" in value:
            if (not isinstance(value.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
                    or not isinstance(value.get("shape"), list)
                    or any(type(dim) is not int or dim < 0 for dim in value["shape"])
                    or not isinstance(value.get("dtype"), str) or not value["dtype"]):
                raise ValueError("Malformed recorded tensor fingerprint")
            return True
        return any([_has_fingerprint(child) for child in value.values()])
    if isinstance(value, list):
        return any([_has_fingerprint(child) for child in value])
    return False


def _audit(audit: dict, contract: dict, *, sample_id: int, strategy: str, seed: int) -> dict:
    layers, heads, ratio, quota = _geometry(contract)
    hybrid = contract["method"] == "hybridkv_kvrepack"
    if (not isinstance(audit, dict) or audit.get("status") != "completed"
            or audit.get("task") != contract["task"] or audit.get("doc_id") != sample_id
            or audit.get("strategy") != strategy or audit.get("seed") != seed
            or audit.get("layers") != layers or audit.get("native_kv_heads") != heads
            or audit.get("batch_size") != 1 or audit.get("merge_ratio") != ratio):
        raise ValueError("Selector audit identity, geometry or selection settings disagree with the contract")
    selected = _mask(audit.get("selected_masks"), layers, heads, hybrid)
    baseline = _mask(audit.get("baseline_masks"), layers, heads, hybrid)
    if (sum(selected) != quota or sum(baseline) != quota or audit.get("selected_count") != [quota]
            or audit.get("baseline_selected_count") != [quota]):
        raise ValueError("Selected native head quota changed")
    if strategy == "similarity" and selected != baseline:
        raise ValueError("Similarity condition changed its original selection")
    if strategy == "random_global":
        from .head_selection import _ranking
        chosen = set(_ranking(seed, "global", layers, heads, 0).sample(range(layers * heads), layers * heads)[:quota])
        if selected != [index in chosen for index in range(layers * heads)]:
            raise ValueError("Random mask differs from its fixed seed/model ranking")
    calls = audit.get("merge_calls")
    if not isinstance(calls, list) or len(calls) != quota or audit.get("actual_merge_calls") != quota:
        raise ValueError("Every selected head must have an effective original merge")
    seen, removed = set(), 0
    for call in calls:
        if not isinstance(call, dict) or any(type(call.get(name)) is not int for name in ("layer", "batch", "head", "before", "after")):
            raise ValueError("Incomplete actual merge-call record")
        layer, batch, head, before, after = (call[name] for name in ("layer", "batch", "head", "before", "after"))
        identity = (layer, batch, head)
        if (not 0 <= layer < layers or batch != 0 or not 0 <= head < heads or identity in seen
                or not selected[layer * heads + head] or not 0 < after < before):
            raise ValueError("Merge calls contain an ineffective, duplicated or unselected native head")
        seen.add(identity)
        removed += before - after
        if hybrid:
            try:
                budget = audit["pre_selection"]["sparse_budget_per_head"][layer][0][head]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("Hybrid audit lacks the original per-head retained budgets") from exc
            if call.get("retained_head_budget") != budget:
                raise ValueError("A Hybrid merge changed its native retained head budget")
    if audit.get("actual_removed_tokens") != removed:
        raise ValueError("Actual removed-token counter differs from effective merge calls")
    if hybrid:
        if audit.get("method") != "hybridkv_kvrepack" or not isinstance(audit.get("pre_selection"), dict):
            raise ValueError("Hybrid classification, budgets and candidates must be recorded")
        if audit["pre_selection"].get("calibrated_head_category") is None:
            raise ValueError("Hybrid calibrated head classification snapshot is absent")
        if (not _has_fingerprint(audit["pre_selection"].get("head_scores"))
                or not _has_fingerprint(audit["pre_selection"]["calibrated_head_category"])):
            raise ValueError("Hybrid native score/classification tensor fingerprints are absent")
        candidates = audit["pre_selection"].get("candidate_hashes")
    else:
        if audit.get("expected_effective_merge_count") != quota or audit.get("expected_removed_tokens") != removed:
            raise ValueError("SnapKV actual work differs from the original eligible matched work")
        for field in ("prefill_cache_shapes",):
            values = audit.get(field)
            if not isinstance(values, list) or len(values) != layers or [row.get("layer") for row in values] != list(range(layers)):
                raise ValueError("Incomplete SnapKV prefill cache coverage")
        candidates = audit.get("candidate_hashes")
    if not isinstance(candidates, list) or len(candidates) != layers or [row.get("layer") for row in candidates] != list(range(layers)):
        raise ValueError("Incomplete native candidate fingerprint coverage")
    if any(not _has_fingerprint(row) for row in candidates):
        raise ValueError("Native candidate bytes were not fingerprinted for every layer")
    union = sum(a or b for a, b in zip(selected, baseline))
    return {"selected_count": quota, "effective_merge_count": len(seen), "removed_tokens": removed,
            "mask_hamming": sum(a != b for a, b in zip(selected, baseline)),
            "mask_jaccard": sum(a and b for a, b in zip(selected, baseline)) / union if union else 1.0}


def validate_bundle(bundle: dict, contract: dict, contract_sha256: str | None = None) -> dict:
    """Recompute identity, fixed ranking, effective work and pairing gates.

    Accepts actual adapter audit dictionaries; does not import torch or trust a
    stored ``fairness.status`` alone. The runtime calls this before committing.
    """
    identity = json_sha(contract)
    if contract_sha256 is not None and contract_sha256 != identity:
        raise ValueError("Declared contract hash differs from its canonical contents")
    if contract.get("method") not in METHODS or contract.get("task") not in TASKS:
        raise ValueError("Unsupported image Random-head method or task")
    selected_ids = contract.get("selected_sample_ids")
    generation_ids = contract.get("generation_sample_ids", selected_ids)
    limit = contract.get("limit")
    if (type(limit) is not int or limit < 1 or not isinstance(selected_ids, list) or len(selected_ids) != limit
            or any(type(value) is not int for value in selected_ids) or len(set(selected_ids)) != limit
            or not isinstance(generation_ids, list) or len(generation_ids) != limit
            or len(set(generation_ids)) != limit or set(generation_ids) != set(selected_ids)):
        raise ValueError("Contract must freeze the exact unique selected and generation ID sets")
    index = bundle.get("sample_index")
    if (bundle.get("schema_version") != 1 or bundle.get("status") != "completed"
            or bundle.get("task") != contract["task"] or bundle.get("method") != contract["method"]
            or type(index) is not int or not 0 <= index < limit
            or bundle.get("sample_id") != generation_ids[index] or bundle.get("contract_sha256") != identity):
        raise ValueError("Paired image sample identity, status or contract mismatch")
    rows, fingerprints = bundle.get("records"), bundle.get("input_fingerprints")
    choices = conditions(contract)
    if not isinstance(rows, dict) or set(rows) != {label for label, _, _ in choices}:
        raise ValueError("Missing or extra Random-head paired conditions")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("Shared prepared-input fingerprints are required")
    reference = rows["similarity"]
    baseline = reference.get("audit", {})
    summaries, audits = {}, []
    for label, strategy, seed in choices:
        row = rows[label]
        if (row.get("input_fingerprints") != fingerprints
                or row.get("condition") != {"strategy": strategy, "mask_seed": seed}):
            raise ValueError("Paired condition input fingerprints or local head seed changed")
        prediction = row.get("prediction")
        if (not isinstance(prediction, dict) or prediction.get("sample_id") != bundle["sample_id"]
                or not isinstance(prediction.get("pred_response"), str)):
            raise ValueError("Missing or mismatched paired prediction")
        for field in ("image", "question", "gt_response", "gen_model_id", "gen_kwargs"):
            if prediction.get(field) != reference["prediction"].get(field):
                raise ValueError("Paired conditions changed a nonselection input/configuration field: " + field)
        if "generation" in contract and prediction.get("gen_kwargs") != contract["generation"]:
            raise ValueError("Paired generation settings differ from the frozen contract")
        audit = row.get("audit", {})
        summaries[label] = _audit(audit, contract, sample_id=bundle["sample_id"], strategy=strategy, seed=seed)
        for field in ("baseline_masks", "baseline_selected_count"):
            if audit.get(field) != baseline.get(field):
                raise ValueError("Paired original native selection changed: " + field)
        if contract["method"] == "hybridkv_kvrepack":
            if audit.get("pre_selection") != baseline.get("pre_selection"):
                raise ValueError("Hybrid pairing changed original classification, budgets or candidates")
        else:
            for field in ("candidate_hashes", "prefill_cache_shapes", "expected_effective_merge_count",
                          "expected_removed_tokens", "actual_merge_calls", "actual_removed_tokens"):
                if audit.get(field) != baseline.get(field):
                    raise ValueError("SnapKV pairing changed its original candidates/cache/work: " + field)
        audits.append(audit)
    if contract["method"] == "hybridkv_kvrepack":
        from .hybrid_head_selection import compare_hybrid_arms
        compare_hybrid_arms(audits)
    if bundle.get("fairness", {}).get("status") != "passed":
        raise ValueError("Paired bundle was not committed through its successful runtime gate")
    return summaries


def _read_run(run_root: Path, data_root=None):
    contract = json.loads((run_root / "run_contract.json").read_text(encoding="utf-8"))
    state = json.loads((run_root / "ablation_state.json").read_text(encoding="utf-8"))
    choices = conditions(contract)
    limit = contract.get("limit")
    if type(limit) is not int or limit < 1:
        raise ValueError("Invalid frozen image sample limit")
    targets = {"target_samples": limit, "target_generations": limit * len(choices),
               "completed_samples": limit, "completed_generations": limit * len(choices)}
    if state.get("status") != "complete" or any(state.get(key) != value for key, value in targets.items()):
        raise ValueError("Generation is incomplete; no partial experiment will be scored")
    paths = sorted((run_root / "samples").glob("*.json"))
    if [path.name for path in paths] != [f"{index:06d}.json" for index in range(limit)]:
        raise ValueError("Committed sample filenames do not exactly cover the frozen prefix")
    for relative, expected in contract.get("source_files_sha256", {}).items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Bound experiment source changed: " + relative)
    from ..benchmarks.milebench.dataset import annotation_path, load_annotation
    source = data_root if data_root is not None else contract["data_root"]
    path = annotation_path(source, contract["task"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract.get("annotation_sha256"):
        raise ValueError("Scoring annotation differs from the frozen generation source")
    annotation = load_annotation(source, contract["task"], limit=limit)
    if [sample["sample_id"] for sample in annotation["data"]] != contract.get("selected_sample_ids"):
        raise ValueError("Scoring sample prefix differs from generation")
    return contract, annotation, paths


def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(run_root, output, data_root=None) -> dict:
    """Validate a complete method/task run and score each of its fixed arms."""
    from ..benchmarks.milebench.evaluate import _output_path, evaluate_predictions, write_evaluation
    run_root, output = Path(run_root).expanduser().resolve(), _output_path(output)
    if output.exists() or output.is_relative_to(run_root / "samples"):
        raise ValueError("Report output must be a fresh directory outside sample commits")
    contract, annotation, paths = _read_run(run_root, data_root)
    source = Path(data_root if data_root is not None else contract["data_root"]).expanduser().resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Report output must be outside the image data directory")
    labels = [label for label, _, _ in conditions(contract)]
    predictions = {label: [] for label in labels}
    audit_rows, fingerprints = [], []
    identity = json_sha(contract)
    for index, path in enumerate(paths):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle.get("sample_index") != index:
            raise ValueError("Sample content index disagrees with its committed filename")
        audited = validate_bundle(bundle, contract, identity)
        fingerprints.append({"sample_id": bundle["sample_id"], "sample_file": path.name,
                             "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "input_fingerprints_sha256": json_sha(bundle["input_fingerprints"])})
        for label in labels:
            predictions[label].append(bundle["records"][label]["prediction"])
            audit_rows.append({"sample_id": bundle["sample_id"], "condition": label, **audited[label]})
    evaluations = {label: evaluate_predictions(predictions[label], annotation, contract["task"]) for label in labels}
    metric = "Accuracy" if contract["task"] == "MMCoQA" else "Rouge-L f"
    if any(metric not in result["eval"] for result in evaluations.values()):
        raise ValueError("Image task metric differs from its original protocol")
    scores = {label: {} for label in labels}
    by_id = {row["sample_id"]: row for row in annotation["data"]}
    score_rows, paired_rows = [], []
    # Source eval_score stores only rounded strings; recover the true original
    # per-input metric instead of silently rounding paired comparisons to 3dp.
    for label in labels:
        for prediction in predictions[label]:
            sample_id = prediction["sample_id"]
            singleton = {"meta_data": deepcopy(annotation["meta_data"]), "data": [deepcopy(by_id[sample_id])]}
            value = float(evaluate_predictions([prediction], singleton, contract["task"])["eval"][metric])
            scores[label][sample_id] = value * 100
            score_rows.append({"sample_id": sample_id, "condition": label, "metric": metric,
                               "score": value, "score_percent": value * 100})
        average = statistics.mean(scores[label].values()) / 100
        if not math.isclose(average, float(evaluations[label]["eval"][metric]), abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Original per-input scores do not reproduce the task aggregate")
    arm_scores = {label: float(result["eval"][metric]) * 100 for label, result in evaluations.items()}
    for sample_id in contract["selected_sample_ids"]:
        native = scores["similarity"][sample_id]
        random_values = [scores[label][sample_id] for label in labels[1:]]
        for label in labels[1:]:
            paired_rows.append({"sample_id": sample_id, "random_condition": label,
                                "similarity_percent": native, "random_percent": scores[label][sample_id],
                                "similarity_minus_random_points": native - scores[label][sample_id]})
        paired_rows.append({"sample_id": sample_id, "random_condition": "random_seed_mean",
                            "similarity_percent": native, "random_percent": statistics.mean(random_values),
                            "similarity_minus_random_points": native - statistics.mean(random_values)})
    random_scores = [arm_scores[label] for label in labels[1:]]
    summary = {"schema_version": 1, "status": "complete", "task": contract["task"], "method": contract["method"],
               "contract_sha256": identity, "conditions": labels, "sample_count": contract["limit"],
               "generation_count": contract["limit"] * len(labels), "metric": metric, "metric_scale": "percent",
               "full_task": evaluations["similarity"]["full_task"],
               "annotation_total_samples": evaluations["similarity"]["annotation_total_samples"],
               "arm_scores_percent": arm_scores, "random_mean_percent": statistics.mean(random_scores),
               "random_sample_sd_percent": statistics.stdev(random_scores) if len(random_scores) > 1 else None,
               "random_seed_count": len(random_scores),
               "similarity_minus_random_mean_points": arm_scores["similarity"] - statistics.mean(random_scores),
               "per_seed_paired_difference_points": {label: arm_scores["similarity"] - arm_scores[label] for label in labels[1:]},
               "selected_head_quota": _geometry(contract)[3], "pairing_validation": "recomputed_passed",
               "work_semantics": "Hybrid removed-token totals may differ across head sets; SnapKV eligible work remains matched.",
               "statistical_scope": "Fixed model-level random head seeds; seed SD is not input uncertainty. No image clustering or confidence interval is assumed; repeated seeds do not create independent inputs.",
               "sample_files": fingerprints}
    # Validation and all scoring finish before any report directory is created.
    output.mkdir(parents=True, exist_ok=False)
    for label, result in evaluations.items():
        write_evaluation(result, output / "conditions" / label)
    _csv(output / "scores.csv", score_rows, ["sample_id", "condition", "metric", "score", "score_percent"])
    _csv(output / "paired_scores.csv", paired_rows, ["sample_id", "random_condition", "similarity_percent", "random_percent", "similarity_minus_random_points"])
    _csv(output / "selector_audit.csv", audit_rows, ["sample_id", "condition", "selected_count", "effective_merge_count", "removed_tokens", "mask_hamming", "mask_jaccard"])
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    sd = summary["random_sample_sd_percent"]
    text = (f"# {contract['method']} / {contract['task']} Random-head\n\n"
            f"Scored {contract['limit']} paired inputs and {summary['generation_count']} generations with the original {metric} protocol. Values are on a 0–100 scale.\n\n"
            f"Similarity: {arm_scores['similarity']:.6f}; random seed mean: {summary['random_mean_percent']:.6f}; "
            f"sample SD: {sd if sd is not None else 'undefined for one seed'}. "
            f"Similarity minus random mean: {summary['similarity_minus_random_mean_points']:+.6f} percentage points.\n\n"
            f"{summary['statistical_scope']}\n\n{summary['work_semantics']}\n")
    with (output / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write(text)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize_run(args.run_root, args.output, args.data_root)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.exit(2, f"MileBench Random-head scoring failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

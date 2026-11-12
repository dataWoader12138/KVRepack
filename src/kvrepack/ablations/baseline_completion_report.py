"""Offline VATEX report combining measured native/KVRepack baselines and merge arms.

Generation never runs here. New baseline predictions are scored independently;
completed full-corpus style scores are reused only after checking their exact
prediction corpus, reference identity, contract and available frozen evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .merge_comparison_report import (QUALITY_METRICS, latency_summary,
    score_native_rows, validate_bundle)
from .run_head_selection import atomic_json, json_sha, sha256

BACKBONES = ("snapkv", "hybridkv")
BASELINE_VARIANTS = ("native", "kvrepack")
STYLE_VARIANTS = ("lookm", "asymkv")
PROTOCOL_FIELDS = ("task", "model_snapshot", "frames", "min_pixels", "max_pixels",
    "budget_ratio", "budget_policy", "max_new_tokens", "repetition_penalty",
    "do_sample", "dtype", "model_seed", "arrow_sha256", "dataset_rows",
    "task_yaml_sha256", "task_utils_sha256")


def _read(path):
    return json.loads(Path(path).read_text())


def _canonical_hash(value):
    # This is the native scorer's Unicode-preserving corpus identity.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive(value, label, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {label}")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"Invalid {label}")
    return value


def _verify_sources(contract, repo_root):
    sources = contract.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Contract lacks its numerical source identities")
    for relative, expected in sources.items():
        path = repo_root/relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Generation source is unavailable or changed: {relative}")


def _load_case(case, limit, repo_root):
    contract_path, state_path = case/"run_contract.json", case/"run_state.json"
    contract, state = _read(contract_path), _read(state_path)
    n = contract.get("target_docs")
    if type(n) is not int or n < limit:
        raise ValueError("Run does not cover the requested original prefix")
    if state.get("status") != "complete" or state.get("completed_docs") != n:
        raise ValueError("Only completed generation runs can be reported")
    if state.get("completed_generations", 2*n) != 2*n:
        raise ValueError("Completed run does not contain both arms for every input")
    if contract.get("task") != "vatex_test" or contract.get("diagnostic_audit") is not False:
        raise ValueError("Require VATEX without intrusive diagnostic timing")
    if limit == 200 and contract.get("warmup_per_arm", 0) < 1:
        raise ValueError("Formal first200 latency requires warmed measurements")
    paths = sorted((case/"samples").glob("*.json"))
    if [p.name for p in paths] != [f"{i:06d}.json" for i in range(n)]:
        raise ValueError("Run lacks its exact declared consecutive input coverage")
    _verify_sources(contract, repo_root)
    return contract, state, paths[:limit]


def validate_baseline_bundle(bundle, contract, backbone, doc_id):
    if (bundle.get("status") != "completed" or bundle.get("doc_id") != doc_id
            or bundle.get("backbone") != backbone
            or bundle.get("contract_sha256") != json_sha(contract)
            or bundle.get("fairness", {}).get("status") != "passed"
            or set(bundle.get("records", {})) != set(BASELINE_VARIANTS)):
        raise ValueError("Baseline bundle failed identity, coverage or fairness validation")
    reference = bundle["records"]["native"]
    for variant in BASELINE_VARIANTS:
        row = bundle["records"][variant]
        method = backbone if variant == "native" else backbone+"_kvrepack"
        ids = row.get("generated_ids", [])
        if (row.get("status") != "completed" or row.get("warmup", False)
                or row.get("doc_id") != doc_id or row.get("task") != "vatex_test"
                or row.get("variant") != variant or row.get("method") != method
                or not isinstance(row.get("text"), str) or row.get("text") != row.get("prediction")
                or not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids)
                or row.get("generated_tokens") != len(ids) or len(ids) > contract["max_new_tokens"]
                or type(row.get("eos_reached")) is not bool
                or type(row.get("capped_without_eos")) is not bool):
            raise ValueError("Baseline row contains a wrong arm, input or prediction")
        if row["capped_without_eos"] != (len(ids) == contract["max_new_tokens"] and not row["eos_reached"]):
            raise ValueError("EOS/cap accounting differs from the actual output")
        if (not row.get("input_fingerprints") or not row.get("reference_metadata")
                or row["input_fingerprints"] != reference.get("input_fingerprints")
                or row["reference_metadata"] != reference.get("reference_metadata")
                or row.get("input_tokens") != reference.get("input_tokens")
                or type(row.get("input_tokens")) is not int or row["input_tokens"] <= 0):
            raise ValueError("Baseline arms do not share the exact original input")
        ratio = _positive(row["budget"]["actual_retained_ratio"], "active prefill ratio")
        if ratio > contract["budget_ratio"]+1e-12:
            raise ValueError("Baseline exceeds the declared total active-cache budget")
        budget = row["budget"]
        if "prefill_tokens_total" in budget and "capacity" in budget:
            if not 0 < budget["prefill_tokens_total"] <= budget["capacity"]:
                raise ValueError("Recorded physical prefill budget is invalid")
        timing = row["runtime"]
        for key in ("end_to_end_seconds", "model_end_to_end_seconds", "generation_seconds",
                    "preprocessing_seconds", "h2d_seconds", "fisher_seconds", "text_decode_seconds",
                    "peak_allocated_gib"):
            _positive(timing[key], key, zero=key not in ("end_to_end_seconds", "generation_seconds"))
        if timing["fisher_seconds"] != 0:
            raise ValueError("Native and KVRepack baselines must not contain a Fisher probe")
        total = sum(timing[k] for k in ("preprocessing_seconds", "h2d_seconds",
                                      "model_end_to_end_seconds", "text_decode_seconds"))
        if (abs(total-timing["end_to_end_seconds"]) > 1e-5
                or timing["model_end_to_end_seconds"]+1e-5 < timing["generation_seconds"]):
            raise ValueError("Full E2E/model E2E/generation timing boundaries are inconsistent")
    return bundle["records"]


def _match_protocol(baseline, style):
    for key in PROTOCOL_FIELDS:
        if key not in baseline or key not in style or baseline[key] != style[key]:
            raise ValueError(f"Baseline/style protocol differs: {key}")
    if (baseline["frames"] != 64 or baseline["max_new_tokens"] != 64
            or baseline["repetition_penalty"] != 1.05 or baseline["budget_ratio"] != .1):
        raise ValueError("Not the prescribed Qwen7B/VATEX64-frame strict10 experiment")
    environments = baseline.get("algorithm_env", {})
    kv_environment = environments.get("kvrepack", environments.get(style["method"]))
    if not isinstance(kv_environment, dict) or any(kv_environment.get(k) != v
            for k, v in style.get("algorithm_env", {}).items()):
        raise ValueError("KVRepack baseline and style arms use different algorithm parameters")


def _match_input(rows, backbone, doc_id):
    reference = rows[0]
    if any(row["input_fingerprints"] != reference["input_fingerprints"]
           or row["reference_metadata"] != reference["reference_metadata"]
           or row["input_tokens"] != reference["input_tokens"] for row in rows[1:]):
        raise ValueError(f"Baseline/style input mismatch: {backbone}/{doc_id}")
    ratios = [row["budget"]["actual_retained_ratio"] for row in rows]
    counts = [row["budget"].get("prefill_tokens_total") for row in rows]
    if backbone == "snapkv":
        if max(ratios)-min(ratios) > 1e-12 or (all(x is not None for x in counts) and len(set(counts)) != 1):
            raise ValueError("SnapKV arms do not have identical final physical cache capacity")
    return dict(backbone=backbone, doc_id=doc_id,
        prefill_tokens_total=dict(zip(("native", "kvrepack", "lookm", "asymkv"), counts)),
        actual_retained_ratio=dict(zip(("native", "kvrepack", "lookm", "asymkv"), ratios)),
        identical_prefill_counts=len(set(counts)) == 1 if all(x is not None for x in counts) else None,
        identical_retained_ratios=max(ratios)-min(ratios) <= 1e-12)


def _cached_style_metrics(case, rows, contract, repo_root):
    directory = case.parent/"report"/contract["method"]
    variant = rows[0]["variant"]
    metrics_path, predictions_path = directory/variant/"metrics.json", directory/variant/"predictions.jsonl"
    metrics = _read(metrics_path)
    expected = [dict(task="vatex_test", doc_id=row["doc_id"], text=row["text"]) for row in rows]
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    if predictions != expected:
        raise ValueError("Cached style predictions differ from the validated raw bundles")
    identity = metrics.get("scoring_identity", {})
    if (metrics.get("status") != "complete" or metrics.get("n") != len(rows)
            or metrics.get("expected_count") != len(rows) or metrics.get("task") != "vatex_test"
            or identity.get("reference_arrow_sha256") != contract["arrow_sha256"]
            or identity.get("prediction_corpus_sha256") != _canonical_hash(expected)
            or metrics.get("scoring_identity_sha256") != _canonical_hash(identity)
            or metrics.get("records", {}).get("file_sha256") != sha256(predictions_path)):
        raise ValueError("Cached style scoring identity does not match this exact corpus")
    saved_report = _read(case.parent/"report/summary.json")
    saved = saved_report["methods"][contract["method"]]
    if saved.get("contract") != contract:
        raise ValueError("Cached report uses a different style generation contract")
    reference = next(row for row in saved_report["comparisons"]
                     if row["method"] == contract["method"] and row["variant"] == variant)
    if reference["n"] != len(rows):
        raise ValueError("Cached quality uses a different reference corpus size")
    quality = {name: metrics["metrics"][name] for name in QUALITY_METRICS}
    if any(not math.isfinite(v) or abs(v-reference[k]) > 1e-12 for k, v in quality.items()):
        raise ValueError("Cached report and scorer quality results disagree")
    # If these reports are part of the published immutable evidence, verify
    # their hashes against that index in addition to the corpus checks above.
    frozen = repo_root/"provenance/merge_aggregation_vatex_200_20261112/raw_evidence_index.json"
    indexed = {}
    if frozen.is_file():
        indexed = {item["path"]: item["sha256"] for item in _read(frozen)["files"]}
    paths = (metrics_path, predictions_path, case.parent/"report/summary.json")
    proof = dict(status="verified_cached_full_corpus", files_sha256={}, frozen_files_checked=[])
    for path in paths:
        digest = sha256(path)
        proof["files_sha256"][str(path)] = digest
        try:
            relative = str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            relative = None
        if relative in indexed:
            if digest != indexed[relative]:
                raise ValueError("Published cached style scoring evidence changed")
            proof["frozen_files_checked"].append(relative)
    return quality, proof


def _csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def build_report(run_root, style_root, output, *, limit=200, repo_root=None):
    if limit not in (1, 200):
        raise ValueError("Only explicit first1 pilot or first200 formal reports are supported")
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    run_root, style_root, output = map(Path, (run_root, style_root, output))
    cases, budgets, contracts = {}, [], {}
    for backbone in BACKBONES:
        case, style_case = run_root/backbone, style_root/(backbone+"_kvrepack")
        contract, state, paths = _load_case(case, limit, repo_root)
        style_contract, style_state, style_paths = _load_case(style_case, limit, repo_root)
        _match_protocol(contract, style_contract)
        if set(style_contract.get("variants", ())) != set(STYLE_VARIANTS):
            raise ValueError("Style contract must contain exactly LOOK-M and AsymKV merge arms")
        arms = {v: [] for v in (*BASELINE_VARIANTS, *STYLE_VARIANTS)}
        raw_hashes = {}
        for doc_id, (path, style_path) in enumerate(zip(paths, style_paths)):
            baseline = validate_baseline_bundle(_read(path), contract, backbone, doc_id)
            style = validate_bundle(_read(style_path), style_contract, doc_id)
            rows = [baseline[v] for v in BASELINE_VARIANTS]+[style[v] for v in STYLE_VARIANTS]
            budgets.append(_match_input(rows, backbone, doc_id))
            for variant, row in zip(arms, rows):
                arms[variant].append(row)
            raw_hashes.update({str(path): sha256(path), str(style_path): sha256(style_path)})
        cases[backbone] = dict(case=case, style_case=style_case, arms=arms,
            contract=contract, style_contract=style_contract, state=state, style_state=style_state,
            raw_files_sha256=raw_hashes)
        contracts[backbone] = dict(baseline_sha256=json_sha(contract), style_sha256=json_sha(style_contract))
    # Input construction should also agree across both backbones.
    for doc_id in range(limit):
        left, right = [cases[b]["arms"]["native"][doc_id] for b in BACKBONES]
        if left["input_fingerprints"] != right["input_fingerprints"] or left["reference_metadata"] != right["reference_metadata"]:
            raise ValueError("The two backbones evaluated different original inputs")
    output.mkdir(parents=True, exist_ok=False)
    table, per_input, score_proofs = [], [], {}
    for backbone, data in cases.items():
        for variant, rows in data["arms"].items():
            contract = data["contract"] if variant in BASELINE_VARIANTS else data["style_contract"]
            if variant in STYLE_VARIANTS and len(rows) == contract["target_docs"]:
                metrics, proof = _cached_style_metrics(data["style_case"], rows, contract, repo_root)
            else:
                metrics = score_native_rows(rows, contract, output/"scoring"/backbone/variant, repo_root)
                proof = dict(status="independently_scored_current_corpus", n=len(rows))
            if set(metrics) != set(QUALITY_METRICS) or any(not math.isfinite(x) for x in metrics.values()):
                raise ValueError("Native scorer returned invalid VATEX quality")
            entry = dict(backbone=backbone, method=backbone if variant == "native" else backbone+"_kvrepack",
                variant=variant, source_scope="new_baseline" if variant in BASELINE_VARIANTS else "completed_style_run",
                **latency_summary(rows), **metrics, VATEX_Avg=statistics.fmean(metrics.values()))
            table.append(entry); score_proofs[backbone+"/"+variant] = proof
            for row in rows:
                per_input.append(dict(backbone=backbone, variant=variant, doc_id=row["doc_id"],
                    generated_tokens=row["generated_tokens"], **row["runtime"]))
    flat_budgets = [dict(backbone=row["backbone"], doc_id=row["doc_id"],
        identical_prefill_counts=row["identical_prefill_counts"], identical_retained_ratios=row["identical_retained_ratios"],
        **{variant+"_prefill_tokens_total": value for variant, value in row["prefill_tokens_total"].items()},
        **{variant+"_actual_retained_ratio": value for variant, value in row["actual_retained_ratio"].items()}) for row in budgets]
    _csv(output/"comparison.csv", table)
    _csv(output/"per_input_latency.csv", per_input)
    _csv(output/"budget_comparison.csv", flat_budgets)
    summary = dict(status="complete", scope="formal_first200" if limit == 200 else "pilot_first1",
        gpu_execution=False, docs_per_arm=limit, arms=8, total_generations=8*limit,
        new_baseline_generations=4*limit, reused_style_generations=4*limit,
        contract_identities=contracts, comparisons=table, scoring=score_proofs,
        protocols={b:dict(baseline=d["contract"], style=d["style_contract"]) for b,d in cases.items()},
        source_files_sha256={p:digest for data in cases.values() for p,digest in data["raw_files_sha256"].items()},
        budget_validation=dict(snapkv_exact_final_capacity=True,
            hybridkv_identical_prefill_counts=all(row["identical_prefill_counts"] is True for row in budgets if row["backbone"] == "hybridkv"),
            per_input_csv="budget_comparison.csv"),
        timing_scope="Full E2E includes native video processing, H2D, Fisher where required, generation and text decoding. Model E2E starts with GPU-resident input. Load/warmup/audit hashing/writes/scoring are excluded.",
        limitations=["One measurement per original input/arm, not repeated steady-state latency.",
            "Baseline arms are newly measured; style arms retain their original generation times and measurement session.",
            "Native and KVRepack rows are distinct implementations; style rows transplant only aggregation rules at the KVRepack insertion point.",
            "HybridKV can have different actual global slot counts due to heterogeneous chunk allocation; all differences are reported, not silently declared identical.",
            "Active KV position budget is distinct from total VRAM or HybridKV backing storage."])
    atomic_json(output/"summary.json", summary)
    lines = ["# VATEX native/KVRepack and merge-only comparison", "",
        f"Original first {limit} inputs per arm; Qwen2.5-VL-7B, 64 frames, strict total-active-prefill 10% budget.", "",
        "| Backbone | Variant | N | VATEX Avg. | Full E2E (s) | Model E2E (s) | P50 (s) | P95 (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| "+" | ".join([row["backbone"], row["variant"], str(row["n"])]
            +[f"{row[k]:.6f}" for k in ("VATEX_Avg", "end_to_end_seconds_mean", "model_end_to_end_seconds_mean", "e2e_p50_seconds", "e2e_p95_seconds")])+" |")
    lines += ["", summary["timing_scope"], "", *["- "+text for text in summary["limitations"]]]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--style-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, choices=(1, 200), default=200)
    args = parser.parse_args(argv)
    summary = build_report(args.run_root.expanduser().resolve(), args.style_root.expanduser().resolve(),
                           args.output.expanduser().resolve(), limit=args.limit)
    print(json.dumps(dict(status=summary["status"], scope=summary["scope"], arms=8,
                         docs_per_arm=args.limit, output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

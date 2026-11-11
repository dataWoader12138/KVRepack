#!/usr/bin/env python3
"""Independent CPU audit of four first200 corpora and three parent-prefix corpora.

No production report/worker aggregation function is imported.  Latency and
length statistics come directly from raw sample bundles.  Predictions are
rebuilt from those bundles and scored against the contract-pinned Arrow using
the unchanged native offline scoring helper, without metrics reuse.  The
production predictions.jsonl files are checked, never used as input data.

By default this script only prints its plan.  --execute requires both completed
200-input cases and a complete production report before loading any tokenizer,
Arrow, scoring package, or Java process.  It never resumes an experiment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


METHODS = ("hybridkv_kvrepack", "snapkv_kvrepack")
VARIANTS = ("lookm", "asymkv")
METRICS = ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr")
PHASES = ("end_to_end_seconds", "preprocessing_seconds", "h2d_seconds",
          "fisher_seconds", "generation_seconds", "text_decode_seconds")


def load(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    # Match the run contract's documented JSON identity format (ASCII escapes,
    # sorted keys and compact separators), independently of worker imports.
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def percentile(values, q):
    """Linear interpolated sample quantile, independently using stdlib only."""
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stats(rows):
    result = {"n": len(rows)}
    for name in PHASES:
        values = [row["runtime"][name] for row in rows]
        if not values or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("Invalid raw latency: " + name)
        result[name + "_mean"] = sum(values) / len(values)
        result[name + "_total"] = sum(values)
        result[name + "_p50"] = percentile(values, .5)
        result[name + "_p95"] = percentile(values, .95)
    result["e2e_p50_seconds"] = result["end_to_end_seconds_p50"]
    result["e2e_p95_seconds"] = result["end_to_end_seconds_p95"]
    model = [row["runtime"]["model_end_to_end_seconds"] for row in rows]
    result["model_end_to_end_seconds_mean"] = sum(model) / len(model)
    result["model_end_to_end_seconds_total"] = sum(model)
    result["model_end_to_end_seconds_p50"] = percentile(model, .5)
    result["model_end_to_end_seconds_p95"] = percentile(model, .95)
    lengths = [row["generated_tokens"] for row in rows]
    result["generated_tokens_mean"] = sum(lengths) / len(lengths)
    result["generated_tokens_total"] = sum(lengths)
    result["generated_tokens_min"], result["generated_tokens_max"] = min(lengths), max(lengths)
    result["generated_tokens_p50"], result["generated_tokens_p95"] = percentile(lengths, .5), percentile(lengths, .95)
    result["tokens_per_generation_second"] = sum(lengths) / result["generation_seconds_total"]
    result["eos_count"] = sum(row["eos_reached"] for row in rows)
    result["capped_without_eos"] = sum(row["capped_without_eos"] for row in rows)
    ratios = [row["budget"]["actual_retained_ratio"] for row in rows]
    result["actual_retained_ratio_mean"] = sum(ratios) / len(ratios)
    result["actual_retained_ratio_max"] = max(ratios)
    result["peak_allocated_gib_max"] = max(row["runtime"]["peak_allocated_gib"] for row in rows)
    result["peak_allocated_gib_mean"] = statistics.fmean(row["runtime"]["peak_allocated_gib"] for row in rows)
    return result


def index_table(rows):
    index = {(row["method"], row["variant"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate method/variant report row")
    return index


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_numbers(expected, reported, label, checks):
    for key, want in expected.items():
        if key not in reported or not isinstance(want, (int, float)):
            continue
        got = float(reported[key])
        error = abs(got - want)
        checks.append({"target": label, "field": key, "expected": want,
                       "reported": got, "absolute_error": error})
        if not math.isfinite(got) or not math.isclose(got, want, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Independent numerical mismatch {label}/{key}: {want} != {got}")


def read_case(case, expected_n, variants, parent=False):
    contract = load(case / "run_contract.json")
    if set(contract["variants"]) != set(variants):
        raise ValueError("Unexpected raw variants in " + str(case))
    if not parent:
        state = load(case / "run_state.json")
        if state.get("status") != "complete" or state.get("completed_docs") != expected_n:
            raise ValueError("GPU case is incomplete; CPU scoring is not allowed yet")
    if contract.get("diagnostic_audit") is not False or contract.get("warmup_per_arm", 0) < 1:
        raise ValueError("Expected warmed, unaudited measurements")
    files = sorted((case / "samples").glob("*.json"))
    if [path.name for path in files] != [f"{i:06d}.json" for i in range(expected_n)]:
        raise ValueError("Raw sample coverage differs")
    arms = {variant: [] for variant in variants}
    hashes, reused = {}, 0
    for doc_id, path in enumerate(files):
        hashes[str(path)] = file_sha(path)
        bundle = load(path)
        if (bundle["doc_id"] != doc_id or bundle["contract_sha256"] != object_sha(contract)
                or bundle.get("fairness", {}).get("status") != "passed"
                or set(bundle["records"]) != set(variants)):
            raise ValueError("Raw bundle identity/fairness mismatch")
        reuse = bundle.get("reused_from_parent")
        if reuse:
            reused += 1
            parent_bundle = Path(reuse["parent_bundle_path"])
            parent_contract = Path(reuse["parent_contract_path"])
            if (file_sha(parent_bundle) != reuse["parent_bundle_sha256"]
                    or file_sha(parent_contract) != reuse["parent_contract_file_sha256"]
                    or object_sha(load(parent_contract)) != reuse["parent_contract_sha256"]):
                raise ValueError("Reused parent provenance changed")
            original = load(parent_bundle)
            if any(row != original["records"][variant] for variant, row in bundle["records"].items()):
                raise ValueError("Reused raw record was altered")
        reference = next(iter(bundle["records"].values()))
        for variant, row in bundle["records"].items():
            ids = row["generated_ids"]
            if (row["doc_id"] != doc_id or row["variant"] != variant or row.get("warmup")
                    or row["generated_tokens"] != len(ids) or not ids
                    or any(type(token) is not int for token in ids)
                    or row["input_fingerprints"] != reference["input_fingerprints"]
                    or row["reference_metadata"] != reference["reference_metadata"]):
                raise ValueError("Raw prediction/input identity differs")
            budget = row["budget"]
            total = sum(math.prod(layer["key_shape"][:-1]) for layer in budget["final_shapes"])
            actual = total - 112 * (len(ids) - 1)
            capacity = 112 * math.floor(.1 * row["input_tokens"])
            ratio = actual / (112 * row["input_tokens"])
            if (not 0 < actual <= capacity or actual != budget["prefill_tokens_total"]
                    or capacity != budget["capacity"] or not math.isclose(ratio, budget["actual_retained_ratio"], rel_tol=1e-13)):
                raise ValueError("Independent physical budget reconstruction differs")
            arms[variant].append(row)
    return contract, arms, hashes, reused


def score_from_raw(rows, contract, production_corpus, output, repo_root):
    output.mkdir(parents=True)
    predictions = [{"task": "vatex_test", "doc_id": row["doc_id"], "text": row["text"]} for row in rows]
    raw_jsonl = output / "predictions_from_raw.jsonl"
    raw_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    published = [json.loads(line) for line in (production_corpus / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if published != predictions:
        raise ValueError("Production predictions.jsonl differs from raw generated records")
    arrows = list((Path(contract["hf_home"]) / "datasets/lmms-lab___vatex/vatex_test").glob("**/vatex-test.arrow"))
    if len(arrows) != 1 or file_sha(arrows[0]) != contract["arrow_sha256"]:
        raise ValueError("Native Arrow reference differs from generation contract")
    helper = repo_root / "provenance/full_video_retest_20261104/helpers/score_full_runs.py"
    command = [sys.executable, "-B", str(helper), "--records", str(raw_jsonl), "--task", "vatex_test",
               "--expected-count", str(len(rows)), "--reference-arrow", str(arrows[0]),
               "--repo-root", str(repo_root), "--output", str(output / "native_metrics.json")]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       LD_LIBRARY_PATH=str(Path(sys.prefix) / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    with (output / "native_scoring.log").open("w") as stream:
        subprocess.run(command, env=environment, cwd=repo_root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    scored = load(output / "native_metrics.json")
    if scored["n"] != len(rows) or scored["reuse"]["reused"] or scored["status"] != "complete":
        raise ValueError("Independent native scoring did not freshly score this corpus")
    metrics = {name: scored["metrics"][name] for name in METRICS}
    metrics["VATEX_Avg"] = sum(metrics.values()) / 4
    return metrics, {"n": len(rows), "predictions_sha256": file_sha(raw_jsonl),
                     "arrow_sha256": file_sha(arrows[0]), "scoring_helper_sha256": file_sha(helper),
                     "scoring_identity": scored["scoring_identity"], "metrics_path": str(output / "native_metrics.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "run-root", "parent-case", "report-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually run CPU audit; requires all GPU cases and report complete")
    args = parser.parse_args()
    for name in ("repo_root", "run_root", "parent_case", "report_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.execute:
        print(json.dumps({"status": "prepared_only", "gpu_execution": False,
                          "expected_main": "2 backbones x2 variants x200", "expected_parent": "3 variants x114",
                          "scoring_corpora": 7, "action": "Run the same command with --execute after report completion."}))
        return 0
    report = load(args.report_root / "summary.json")
    if report.get("status") != "complete":
        raise ValueError("Production report must complete before independent CPU verification")
    # Check complete GPU states before importing Transformers, Arrow, or scorers.
    for method in METHODS:
        state = load(args.run_root / method / "run_state.json")
        if state.get("status") != "complete" or state.get("completed_docs") != 200:
            raise ValueError("A GPU case is still incomplete; no scoring started")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    began = time.monotonic()
    report_files = sorted(path for path in args.report_root.rglob("*") if path.is_file())
    report_hashes = {str(path): file_sha(path) for path in report_files}
    checks, results, corpus_details, input_hashes = [], [], [], {}
    main_json = index_table(report["comparisons"])
    main_csv = index_table(read_csv(args.report_root / "comparison.csv"))
    if set(main_json) != {(method, variant) for method in METHODS for variant in VARIANTS}:
        raise ValueError("Expected exactly four main report rows, without a KVRepack200 baseline")
    datasets = []
    total_reused = 0
    for method in METHODS:
        contract, arms, hashes, reused = read_case(args.run_root / method, 200, VARIANTS)
        total_reused += reused
        input_hashes.update(hashes)
        for variant in VARIANTS:
            datasets.append(("main200", method, variant, arms[variant], contract,
                             args.report_root / method / variant, main_json[(method, variant)], main_csv[(method, variant)]))
    parent_json_doc = load(args.report_root / "parent_matched_prefix_comparison.json")
    parent_csv = index_table(read_csv(args.report_root / "parent_matched_prefix_comparison.csv"))
    parent_json = index_table(parent_json_doc["comparisons"])
    parent_latency = index_table(load(args.report_root / "parent_matched_prefix_latency.json")["comparisons"])
    n_parent = parent_json_doc["n"]
    if n_parent != 114:
        raise ValueError("This audit expects the explicitly stopped114-input parent")
    parent_contract, parent_arms, hashes, _ = read_case(args.parent_case, n_parent, ("kvrepack", *VARIANTS), parent=True)
    input_hashes.update(hashes)
    for variant in ("kvrepack", *VARIANTS):
        key = (parent_contract["method"], variant)
        datasets.append(("parent114", key[0], variant, parent_arms[variant], parent_contract,
                         args.report_root / "parent_matched_prefix_scoring" / variant, parent_json[key], parent_csv[key]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(datasets[0][4]["model_snapshot"], local_files_only=True)
    decoded = set()
    for scope, method, variant, rows, contract, production_corpus, json_row, csv_row in datasets:
        label = scope + "/" + method + "/" + variant
        for row in rows:
            identity = (method, variant, row["doc_id"], tuple(row["generated_ids"]))
            if identity in decoded:
                continue
            text = tokenizer.decode(row["generated_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if text != row["text"] or text != row["prediction"]:
                raise ValueError("Generated token/text mismatch: " + label)
            eos = row["generated_ids"][-1] == tokenizer.eos_token_id
            if eos != row["eos_reached"] or row["capped_without_eos"] != (len(row["generated_ids"]) >= contract["max_new_tokens"] and not eos):
                raise ValueError("EOS/cap mismatch: " + label)
            decoded.add(identity)
        calculated = stats(rows)
        compare_numbers(calculated, json_row, label + "/json", checks)
        compare_numbers(calculated, csv_row, label + "/csv", checks)
        if scope == "parent114":
            compare_numbers(calculated, parent_latency[(method, variant)], label + "/latency_json", checks)
        independent_dir = args.output / scope / method / variant
        quality, provenance = score_from_raw(rows, contract, production_corpus, independent_dir, args.repo_root)
        compare_numbers(quality, json_row, label + "/quality_json", checks)
        compare_numbers(quality, csv_row, label + "/quality_csv", checks)
        compare_numbers(quality, load(production_corpus / "metrics.json")["metrics"] | {
            "VATEX_Avg": load(production_corpus / "metrics.json")["metrics"]["mean4"]}, label + "/metric_file", checks)
        results.append(dict(scope=scope, method=method, variant=variant, **calculated, **quality))
        corpus_details.append(dict(label=label, **provenance))
        dump(args.output / "progress.json", dict(status="auditing", completed_corpora=len(results), total_corpora=7))
        print(json.dumps(dict(stage="corpus_verified", label=label, n=len(rows))), flush=True)
    if len(decoded) != 914:
        raise ValueError("Expected800 main plus114 unique parent KVRepack generations")
    for path, expected in {**input_hashes, **report_hashes}.items():
        if file_sha(path) != expected:
            raise ValueError("Raw/report artifact changed during independent audit: " + path)
    summary = dict(status="passed", gpu_execution=False, created_utc=datetime.now(timezone.utc).isoformat(),
                   independent_script_sha256=file_sha(__file__), source_modified=False,
                   numerical_tolerance=dict(relative=1e-10, absolute=1e-10), numeric_comparisons=len(checks),
                   max_absolute_error=max((item["absolute_error"] for item in checks), default=0),
                   completed_main_corpora=4, completed_parent_corpora=3, decoded_unique_generations=len(decoded),
                   reused_main_docs=total_reused, reused_main_generations=total_reused * 2,
                   immutable_raw_files=len(input_hashes), immutable_report_files=len(report_hashes),
                   report_files_unchanged=True, raw_files_unchanged=True,
                   scoring="Fresh native task scoring from independently rebuilt raw-generated predictions and pinned Arrow; no metrics reuse; CIDEr DF recomputed separately for each200/114 corpus",
                   results=results, corpora=corpus_details, checks=checks, seconds=time.monotonic() - began)
    dump(args.output / "independent_results_check.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "numeric_comparisons", "max_absolute_error", "decoded_unique_generations", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

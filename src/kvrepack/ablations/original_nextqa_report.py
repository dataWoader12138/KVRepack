"""Validate and score NExTQA with the original historical/local budgets.

Expected layout: ROOT/{baseline,style}/{snapkv,hybridkv}/. All arms cover
exactly the same original first 1 (pilot) or 200 (formal) reference rows.
The original nominal historical budget is retained, including its additional
recent/local cache positions. It must never be labelled a strict total10 run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .merge_comparison_report import latency_summary
from .run_head_selection import atomic_json, json_sha, sha256
from .run_merge_comparison import pairing_signature

TASK = "nextqa_oe_test"
BACKBONES = ("snapkv", "hybridkv")
VARIANTS = ("native", "kvrepack", "lookm", "asymkv")
FAMILIES = {"baseline": VARIANTS[:2], "style": VARIANTS[2:]}
BUDGET_POLICY = "original_history_plus_local_v1"
SHARED_SOURCES = tuple("src/kvrepack/"+name for name in (
    "ablations/run_merge_comparison.py", "ablations/merge_operators.py",
    "ablations/merge_snap.py", "ablations/merge_hybrid.py",
    "qwen_model_snapkv_kvrepack.py", "hybridkv_utils.py", "modeling_qwen2_5_vl.py",
    "hybrid_asymkv.py", "qwen_model.py", "merge_score.py", "monkeypatch.py",
    "asymkv_official/attention.py", "ablations/run_head_selection.py",
    "ablations/run_head_selection_full.py", "ablations/run_rope_merge.py",
    "observations/inputs.py", "experiments/registry.py", "experiments/runner.py",
    "ablations/video_e2e_inputs.py", "ablations/original_budget.py"))
PROTOCOL_FIELDS = ("task", "model_snapshot", "model_metadata_sha256", "versions",
    "frames", "min_pixels", "max_pixels", "budget_ratio", "budget_policy",
    "max_new_tokens", "repetition_penalty", "do_sample", "dtype", "model_seed",
    "arrow_sha256", "dataset_rows", "task_yaml_sha256", "task_utils_sha256")
RUNTIME_FIELDS = ("end_to_end_seconds", "model_end_to_end_seconds",
    "generation_seconds", "preprocessing_seconds", "h2d_seconds", "fisher_seconds",
    "text_decode_seconds", "peak_allocated_gib")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Invalid {label}")
    return value


def _verify_sources(contract, repo_root):
    sources = contract.get("source_sha256")
    if not isinstance(sources, dict) or not set(SHARED_SOURCES).issubset(sources):
        raise ValueError("Contract lacks the shared numerical sources and input helper")
    for relative, digest in sources.items():
        path = (repo_root/relative).resolve()
        if (not path.is_relative_to(repo_root.resolve()) or not path.is_file()
                or sha256(path) != digest):
            raise ValueError(f"Generation source is unavailable or changed: {relative}")


def _verify_model(contract):
    snapshot = Path(contract["model_snapshot"])
    expected = contract.get("model_metadata_sha256")
    actual = {p.name: sha256(p) for p in sorted(snapshot.iterdir())
              if p.is_file() and p.suffix in {".json", ".txt"}}
    if not expected or actual != expected:
        raise ValueError("Model metadata differs from the generation snapshot")
    config = _read(snapshot/"config.json")
    text = config.get("text_config", config)
    if (config.get("model_type") != "qwen2_5_vl" or text.get("num_hidden_layers") != 28
            or text.get("num_key_value_heads") != 4 or text.get("hidden_size") != 3584):
        raise ValueError("Expected the prescribed Qwen2.5-VL-7B architecture")


def _load_case(case, family, backbone, count, repo_root):
    contract, state = _read(case/"run_contract.json"), _read(case/"run_state.json")
    if (type(contract.get("target_docs")) is not int or contract["target_docs"] != count
            or contract.get("target_generations") != 2*count
            or tuple(contract.get("variants", ())) != FAMILIES[family]):
        raise ValueError("Case does not declare exactly the requested prefix and two arms")
    if (state.get("status") != "complete" or state.get("completed_docs") != count
            or state.get("completed_generations") != 2*count
            or state.get("contract_sha256") != json_sha(contract)):
        raise ValueError("Report requires complete runs with matching contract identities")
    if (contract.get("task") != TASK or contract.get("diagnostic_audit") is not False
            or contract.get("frames") != 64 or contract.get("max_new_tokens") != 128
            or contract.get("min_pixels") != 200704 or contract.get("max_pixels") != 1605632
            or contract.get("budget_ratio") != .1 or contract.get("budget_policy") != BUDGET_POLICY
            or contract.get("repetition_penalty") != 1.05 or contract.get("do_sample") is not False
            or contract.get("dtype") != "bfloat16" or contract.get("model_seed") != 42):
        raise ValueError("Not the prescribed unaudited NExTQA/Qwen7B/original-budget protocol")
    if count == 200 and contract.get("warmup_per_arm", 0) < 1:
        raise ValueError("Formal latency measurements require warmup")
    method = backbone if family == "baseline" else backbone+"_kvrepack"
    if contract.get("method") != method or (family == "baseline" and contract.get("backbone") != backbone):
        raise ValueError("Case uses the wrong backbone or method")
    paths = sorted((case/"samples").glob("*.json"))
    if [p.name for p in paths] != [f"{i:06d}.json" for i in range(count)]:
        raise ValueError("Case lacks exact consecutive original-prefix coverage")
    _verify_sources(contract, repo_root)
    return contract, state, paths


def _match_protocol(cases):
    reference = next(iter(cases.values()))["contract"]
    for case in cases.values():
        contract = case["contract"]
        for key in PROTOCOL_FIELDS:
            if key not in contract or key not in reference or contract[key] != reference[key]:
                raise ValueError(f"Cases use different protocols: {key}")
        for path in SHARED_SOURCES:
            if contract["source_sha256"][path] != reference["source_sha256"][path]:
                raise ValueError(f"Cases use different numerical sources: {path}")
    _verify_model(reference)
    for backbone in BACKBONES:
        baseline, style = [cases[(family, backbone)]["contract"] for family in FAMILIES]
        environment = baseline.get("algorithm_env", {}).get("kvrepack")
        if not environment or environment != style.get("algorithm_env"):
            raise ValueError("KVRepack and style arms use different algorithm parameters")
        prefix = "SNAPKV_" if backbone == "snapkv" else "HYBRID_KVREPACK_"
        gate = "SNAPKV_SIMHEAD_MERGE_RATIO" if backbone == "snapkv" else prefix+"SIMHEAD_MERGE_RATIO"
        compress = "SNAPKV_KVREPACK_COMPRESS_RATIO" if backbone == "snapkv" else prefix+"COMPRESS_RATIO"
        expected = {gate: .30 if backbone == "snapkv" else .50,
                    compress: .70 if backbone == "snapkv" else .50, "K_DIFF_ALPHA": 2.}
        try:
            if any(float(environment[key]) != value for key, value in expected.items()):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("Not the prescribed NExTQA gate/compression/alpha parameters") from None
    return reference


def _reference(hf_home, contract, count, repo_root):
    paths = sorted((hf_home/"datasets/lmms-lab___n_ex_tqa/OE").glob("**/n_ex_tqa-test.arrow"))
    if len(paths) != 1 or sha256(paths[0]) != contract["arrow_sha256"]:
        raise ValueError("Offline NExTQA scoring reference differs from generation")
    for name, filename in (("task_yaml_sha256", "nextqa_oe_test.yaml"), ("task_utils_sha256", "utils.py")):
        path = repo_root/"vendor/lmms-eval/lmms_eval/tasks/nextqa"/filename
        if not path.is_file() or sha256(path) != contract[name]:
            raise ValueError(f"Native NExTQA task source changed: {filename}")
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(paths[0]), "r") as stream:
        table = ipc.open_stream(stream).read_all()
    if table.num_rows != contract["dataset_rows"] or table.num_rows < count:
        raise ValueError("Reference split size differs from generation")
    return paths[0], table.slice(0, count).to_pylist()


def _original_hybrid_capacity(meta, n):
    """Prove the actual native per-head history budgets were not rescaled."""
    if (meta.get("policy") != BUDGET_POLICY or meta.get("native_budget_unchanged") is not True
            or meta.get("post_allocation_scaling") is not False or meta.get("scale") != 1.
            or meta.get("sequence_length") != n or meta.get("budget_ratio") != .1
            or meta.get("layers") != 28 or meta.get("native_kv_heads") != 4):
        raise ValueError("HybridKV allocation was changed or lacks original-budget evidence")
    original, current = meta.get("original_history_budget"), meta.get("history_budget")
    local = meta.get("local_len")
    if (not isinstance(original, list) or len(original) != 28 or original != current
            or type(local) is not int or local <= 0):
        raise ValueError("HybridKV historical budgets or local window changed")
    per_layer = []
    for batch in original:
        if (not isinstance(batch, list) or len(batch) != 1 or not isinstance(batch[0], list)
                or len(batch[0]) != 4 or any(type(value) is not int or value < 0 for value in batch[0])):
            raise ValueError("Malformed original HybridKV layer/head allocation")
        per_layer.append(sum(batch[0])+4*local)
    history = sum(per_layer)-112*local
    active = sum(per_layer)
    if (history <= 0 or meta.get("original_history_positions") != history
            or meta.get("history_positions") != history or meta.get("local_positions") != 112*local
            or meta.get("active_prefill_positions") != active
            or meta.get("capacity_positions") != active
            or meta.get("per_layer_active_positions") != per_layer):
        raise ValueError("HybridKV original allocation accounting is inconsistent")
    return active


def _validate_row(row, contract, backbone, variant, doc_id, original_doc):
    method = backbone if variant == "native" else backbone+"_kvrepack"
    ids = row.get("generated_ids")
    if (row.get("status") != "completed" or row.get("warmup") is not False
            or type(row.get("doc_id")) is not int or row["doc_id"] != doc_id
            or row.get("task") != TASK or row.get("variant") != variant or row.get("method") != method
            or not isinstance(row.get("text"), str) or row.get("text") != row.get("prediction")
            or not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids)
            or row.get("generated_tokens") != len(ids) or len(ids) > 128
            or type(row.get("eos_reached")) is not bool or type(row.get("capped_without_eos")) is not bool):
        raise ValueError("Row contains a wrong method, input, warmup or prediction")
    if row["capped_without_eos"] != (len(ids) == 128 and not row["eos_reached"]):
        raise ValueError("EOS/cap accounting differs from the actual output")
    n = row.get("input_tokens")
    meta, fingerprints = row.get("reference_metadata", {}), row.get("input_fingerprints", {})
    if (type(n) is not int or n <= 0 or not fingerprints or not meta
            or meta.get("doc_id") != doc_id or meta.get("task") != TASK
            or meta.get("input_tokens") != n or meta.get("arrow_sha256") != contract["arrow_sha256"]
            or meta.get("original_row_sha256") != json_sha(original_doc)
            or fingerprints.get("input_shapes", {}).get("input_ids") != [1, n]
            or not fingerprints.get("input_hashes", {}).get("input_ids")
            or not fingerprints.get("input_hashes", {}).get("pixel_values_videos")):
        raise ValueError("Row does not identify the exact original reference/video input")
    budget = row.get("budget", {})
    total = budget.get("prefill_tokens_total")
    nominal = 112*math.floor(.1*n)
    if (budget.get("budget_policy") != BUDGET_POLICY
            or budget.get("nominal_total_cap") != nominal
            or type(total) is not int or total <= 0
            or abs(_number(budget.get("actual_retained_ratio"), "prefill ratio", positive=True)-total/(112*n)) > 1e-12):
        raise ValueError("Physical cache lacks the unchanged original budget identity")
    if backbone == "snapkv":
        capacity = 112*(min(n-32, math.floor(.1*n))+32)
        if n <= 32 or total != capacity or budget.get("capacity") != capacity:
            raise ValueError("SnapKV does not retain its original history plus recent budget")
    else:
        capacity = _original_hybrid_capacity(row.get("audit", {}).get("budget", {}), n)
        if total != capacity or budget.get("capacity") != capacity:
            raise ValueError("HybridKV physical cache differs from its original unscaled allocation")
    shapes = budget.get("final_shapes", [])
    if len(shapes) != 28 or [layer.get("layer") for layer in shapes] != list(range(28)):
        raise ValueError("Budget lacks all 28 actual cache shapes")
    final = 0
    for layer in shapes:
        key, value = layer.get("key_shape"), layer.get("value_shape")
        if (not isinstance(key, list) or len(key) != (4 if backbone == "snapkv" else 2) or key != value
                or any(type(x) is not int or x <= 0 for x in key) or key[-1] != 128):
            raise ValueError("Invalid actual K/V cache shape")
        final += math.prod(key[:-1])
    if final-112*(len(ids)-1) != total:
        raise ValueError("Recorded physical cache count differs from actual cache shapes")
    runtime = row.get("runtime", {})
    for name in RUNTIME_FIELDS:
        _number(runtime.get(name), name, positive=name in ("end_to_end_seconds", "generation_seconds"))
    if variant == "asymkv":
        if runtime["fisher_seconds"] <= 0 or not isinstance(row.get("fisher"), dict):
            raise ValueError("AsymKV-style E2E must include its Fisher probe")
    elif runtime["fisher_seconds"] != 0:
        raise ValueError("Only AsymKV-style may include a Fisher probe")
    components = sum(runtime[k] for k in ("preprocessing_seconds", "h2d_seconds",
                                        "model_end_to_end_seconds", "text_decode_seconds"))
    if (abs(components-runtime["end_to_end_seconds"]) > 1e-5
            or runtime["model_end_to_end_seconds"]+1e-5 < runtime["generation_seconds"]+runtime["fisher_seconds"]):
        raise ValueError("Full E2E/model E2E/generation timing boundaries are inconsistent")
    return row


def _validate_bundle(bundle, contract, backbone, variants, doc_id, original_doc):
    if (bundle.get("status") != "completed" or type(bundle.get("doc_id")) is not int
            or bundle["doc_id"] != doc_id or bundle.get("contract_sha256") != json_sha(contract)
            or bundle.get("fairness", {}).get("status") != "passed"
            or set(bundle.get("records", {})) != set(variants)):
        raise ValueError("Bundle failed identity, coverage or fairness validation")
    if "reused_from_parent" in bundle:
        raise ValueError("This experiment requires fresh measurements, not historical reused bundles")
    return {variant: _validate_row(bundle["records"][variant], contract, backbone,
                                  variant, doc_id, original_doc) for variant in variants}


def _match_input(rows, backbone, doc_id):
    reference = rows["native"]
    fields = ("input_fingerprints", "reference_metadata", "input_tokens")
    if any(row[key] != reference[key] for row in rows.values() for key in fields):
        raise ValueError(f"Baseline/style input mismatch: {backbone}/{doc_id}")
    counts = {variant: row["budget"]["prefill_tokens_total"] for variant, row in rows.items()}
    if backbone == "snapkv" and len(set(counts.values())) != 1:
        raise ValueError("SnapKV arms do not share identical final physical cache capacity")
    signature = pairing_signature(rows["kvrepack"]["audit"], backbone+"_kvrepack")
    for variant in ("lookm", "asymkv"):
        if (pairing_signature(rows[variant]["audit"], backbone+"_kvrepack") != signature
                or counts[variant] != counts["kvrepack"]):
            raise ValueError("Merge candidates/heads/groups/anchors or physical budgets differ")
    return dict(backbone=backbone, doc_id=doc_id, identical_prefill_counts=len(set(counts.values())) == 1,
                **{variant+"_prefill_tokens_total": n for variant, n in counts.items()},
                **{variant+"_actual_retained_ratio": row["budget"]["actual_retained_ratio"] for variant, row in rows.items()})


def score_rows(rows, contract, reference_arrow, output, repo_root):
    """Invoke the native offline WUPS scorer and check its exact corpus identity."""
    output.mkdir(parents=True, exist_ok=False)
    corpus = [dict(task=TASK, doc_id=row["doc_id"], text=row["text"]) for row in rows]
    predictions = output/"predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in corpus), encoding="utf-8")
    helper = repo_root/"provenance/full_video_retest_20261104/helpers/score_full_runs.py"
    command = [sys.executable, "-B", str(helper), "--records", str(predictions), "--task", TASK,
        "--expected-count", str(len(rows)), "--output", str(output/"metrics.json"),
        "--repo-root", str(repo_root), "--reference-arrow", str(reference_arrow)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        LD_LIBRARY_PATH=str(Path(sys.prefix)/"lib")+os.pathsep+os.environ.get("LD_LIBRARY_PATH", ""))
    with (output/"scoring.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=repo_root, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    scored = _read(output/"metrics.json")
    identity = scored.get("scoring_identity", {})
    if (scored.get("status") != "complete" or scored.get("task") != TASK
            or scored.get("n") != len(rows) or scored.get("expected_count") != len(rows)
            or identity.get("reference_arrow_sha256") != contract["arrow_sha256"]
            or identity.get("prediction_corpus_sha256") != _canonical_hash(corpus)
            or scored.get("scoring_identity_sha256") != _canonical_hash(identity)
            or scored.get("records", {}).get("file_sha256") != sha256(predictions)):
        raise ValueError("Native WUPS scorer returned a different reference/prediction corpus")
    wups = _number(scored.get("metrics", {}).get("WUPS"), "native WUPS")
    if wups > 100:
        raise ValueError("Native WUPS must be on the original 0–100 scale")
    return wups, dict(status="independently_scored_current_corpus",
        metrics_sha256=sha256(output/"metrics.json"), scorer_sha256=sha256(helper),
        scoring_identity=identity, scoring_identity_sha256=scored["scoring_identity_sha256"])


def _csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def build_report(root, hf_home, output, *, expected_count=200, repo_root=None):
    if type(expected_count) is not int or expected_count not in (1, 200):
        raise ValueError("Only explicit first1 pilot or first200 formal reports are supported")
    root, hf_home, output = map(Path, (root, hf_home, output))
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    cases, raw_hashes = {}, {}
    for family in FAMILIES:
        for backbone in BACKBONES:
            case = root/family/backbone
            contract, state, paths = _load_case(case, family, backbone, expected_count, repo_root)
            cases[(family, backbone)] = dict(contract=contract, state=state, paths=paths)
            for path in (case/"run_contract.json", case/"run_state.json", *paths):
                raw_hashes[str(path)] = sha256(path)
    common = _match_protocol(cases)
    arrow, original_docs = _reference(hf_home, common, expected_count, repo_root)
    arms = {(b, v): [] for b in BACKBONES for v in VARIANTS}
    budgets = []
    for doc_id, original_doc in enumerate(original_docs):
        inputs = []
        for backbone in BACKBONES:
            rows = {}
            for family, variants in FAMILIES.items():
                data = cases[(family, backbone)]
                rows.update(_validate_bundle(_read(data["paths"][doc_id]), data["contract"],
                    backbone, variants, doc_id, original_doc))
            budgets.append(_match_input(rows, backbone, doc_id))
            inputs.append(rows["native"])
            for variant, row in rows.items():
                arms[(backbone, variant)].append(row)
        if any(inputs[0][key] != inputs[1][key] for key in ("input_fingerprints", "reference_metadata", "input_tokens")):
            raise ValueError("The two backbones evaluated different original inputs")
    # Validation is complete before scoring or creating a report directory.
    output.mkdir(parents=True, exist_ok=False)
    table, per_input, proofs = [], [], {}
    for (backbone, variant), rows in arms.items():
        contract = cases[("baseline" if variant in FAMILIES["baseline"] else "style", backbone)]["contract"]
        wups, proof = score_rows(rows, contract, arrow, output/"scoring"/backbone/variant, repo_root)
        proofs[backbone+"/"+variant] = proof
        table.append(dict(backbone=backbone, method=backbone if variant == "native" else backbone+"_kvrepack",
            variant=variant, WUPS=wups, **latency_summary(rows)))
        for row in rows:
            per_input.append(dict(backbone=backbone, variant=variant, doc_id=row["doc_id"],
                generated_tokens=row["generated_tokens"], eos_reached=row["eos_reached"],
                capped_without_eos=row["capped_without_eos"], **row["runtime"]))
    for backbone in BACKBONES:
        native = next(row for row in table if row["backbone"] == backbone and row["variant"] == "native")
        for row in table:
            if row["backbone"] == backbone:
                row["e2e_change_from_native_percent"] = 100*(row["end_to_end_seconds_mean"]/native["end_to_end_seconds_mean"]-1)
    for path, digest in raw_hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Generation evidence changed while this report was being built")
    for data in cases.values():
        _verify_sources(data["contract"], repo_root)
    if sha256(arrow) != common["arrow_sha256"]:
        raise ValueError("Scoring reference changed while this report was being built")
    _csv(output/"comparison.csv", table)
    _csv(output/"per_input_latency.csv", per_input)
    _csv(output/"budget_comparison.csv", budgets)
    summary = dict(schema_version=1, status="complete", task=TASK,
        scope="formal_first200_original_budget" if expected_count == 200 else "pilot_first1_original_budget",
        budget_policy=BUDGET_POLICY,
        gpu_execution=False, docs_per_arm=expected_count, arms=8, total_generations=8*expected_count,
        comparisons=table, scoring=proofs, source_files_sha256=raw_hashes,
        protocols={family: {b: cases[(family,b)]["contract"] for b in BACKBONES} for family in FAMILIES},
        contract_identities={family: {b: json_sha(cases[(family,b)]["contract"]) for b in BACKBONES} for family in FAMILIES},
        reference_arrow=str(arrow), reference_arrow_sha256=common["arrow_sha256"],
        budget_validation=dict(original_budget_preserved=True, strict_total10=False,
            snapkv_exact_history_plus_recent=True,
            hybridkv_identical_prefill_counts=all(row["identical_prefill_counts"] for row in budgets if row["backbone"] == "hybridkv"),
            merge_candidates_heads_groups_anchors_identical=True, per_input_csv="budget_comparison.csv"),
        timing_scope="Full E2E includes video read/decode, processor, H2D, Fisher where required, generation and text decoding, with synchronized GPU boundaries. Model E2E includes Fisher and generation. Model load, warmup, diagnostic hashing, writes and scoring are excluded.",
        limitations=["One measurement per original input/arm; execution order rotates within each paired worker.",
            "Four workers are measured serially; this is not repeated steady-state timing.",
            "LOOK-M/AsymKV rows transplant only merging at the KVRepack insertion point; they are not standalone methods.",
            "HybridKV native and merge arms can have different original allocations; actual per-head budgets are preserved and differences are reported.",
            "The nominal historical budget is 10%; original recent/local positions are additional. This is not a strict total-active-cache 10% experiment.",
            "The active-cache position budget is distinct from total GPU memory and HybridKV backing storage."])
    atomic_json(output/"summary.json", summary)
    lines = ["# NExTQA merge comparison", "",
        f"Original first {expected_count} inputs per arm; Qwen2.5-VL-7B, 64 frames; original nominal 10% historical budget plus its unchanged recent/local positions.", "",
        "| Backbone | Variant | N | WUPS | Full E2E (s) | Model E2E (s) | P50 (s) | P95 (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| "+" | ".join([row["backbone"], row["variant"], str(row["n"])]+
            [f"{row[key]:.6f}" for key in ("WUPS", "end_to_end_seconds_mean",
             "model_end_to_end_seconds_mean", "e2e_p50_seconds", "e2e_p95_seconds")])+" |")
    lines += ["", summary["timing_scope"], "", *["- "+item for item in summary["limitations"]]]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, choices=(1, 200), default=200)
    args = parser.parse_args(argv)
    summary = build_report(args.root.expanduser().resolve(), args.hf_home.expanduser().resolve(),
        args.output.expanduser().resolve(), expected_count=args.expected_count)
    print(json.dumps(dict(status=summary["status"], scope=summary["scope"], arms=8,
        docs_per_arm=args.expected_count, output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

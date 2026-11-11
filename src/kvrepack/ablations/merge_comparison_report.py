"""Validate and score completed fixed-group VATEX merge/latency experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .run_head_selection import atomic_json, json_sha, sha256
from .run_merge_comparison import pairing_signature

KNOWN_VARIANTS = ("kvrepack", "lookm", "asymkv")
QUALITY_METRICS = ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr")


def contract_variants(contract):
    variants = tuple(contract.get("variants", ()))
    if not variants or len(set(variants)) != len(variants) or any(v not in KNOWN_VARIANTS for v in variants):
        raise ValueError("Contract must declare unique known aggregation arms")
    return variants


def _reuse_provenance(bundle):
    """Check immutable reused rows against their original parent bundle."""
    reuse = bundle.get("reused_from_parent")
    if not reuse:
        return None
    info = reuse if isinstance(reuse, dict) else bundle.get("parent_provenance", bundle)
    def field(*names):
        return next((info[name] for name in names if info.get(name) is not None), None)
    parent_path = field("parent_bundle_path", "bundle_path", "parent_sample_path")
    parent_contract_path = field("parent_contract_path", "contract_path")
    expected_contract = field("parent_contract_sha256", "contract_sha256")
    if not parent_path or not parent_contract_path or not expected_contract:
        raise ValueError("Reused bundle lacks explicit parent bundle/contract identity")
    parent_path, parent_contract_path = Path(parent_path), Path(parent_contract_path)
    parent_contract = json.loads(parent_contract_path.read_text())
    if json_sha(parent_contract) != expected_contract:
        raise ValueError("Reused parent contract SHA differs")
    expected_contract_file = field("parent_contract_file_sha256")
    if expected_contract_file and sha256(parent_contract_path) != expected_contract_file:
        raise ValueError("Reused parent contract file SHA differs")
    expected_bundle = field("parent_bundle_sha256", "bundle_sha256")
    if expected_bundle and sha256(parent_path) != expected_bundle:
        raise ValueError("Reused parent bundle SHA differs")
    parent = json.loads(parent_path.read_text())
    if parent.get("contract_sha256") != expected_contract or parent.get("doc_id") != bundle["doc_id"]:
        raise ValueError("Reused parent bundle identity differs")
    for variant, row in bundle["records"].items():
        if parent["records"].get(variant) != row:
            raise ValueError("Reused prediction/runtime/audit metadata changed")
    snapshot = field("source_snapshot_path", "parent_source_snapshot", "source_snapshot")
    if isinstance(snapshot, str) and not Path(snapshot).exists():
        raise ValueError("Parent source snapshot is unavailable")
    expected_snapshot = field("source_snapshot_sha256")
    if expected_snapshot and (not isinstance(snapshot, str) or sha256(Path(snapshot)) != expected_snapshot):
        raise ValueError("Parent source snapshot SHA differs")
    return info


def validate_bundle(bundle, contract, doc_id, *, verify_reuse=True):
    variants = contract_variants(contract)
    if (bundle.get("doc_id") != doc_id or bundle.get("contract_sha256") != json_sha(contract)
            or bundle.get("fairness", {}).get("status") != "passed"
            or set(bundle.get("records", {})) != set(variants)):
        raise ValueError("Failed contract, declared-arm coverage, or fairness gate")
    if verify_reuse:
        _reuse_provenance(bundle)
    reference = bundle["records"][variants[0]]
    signature = pairing_signature(reference["audit"], contract["method"])
    for variant in variants:
        row = bundle["records"][variant]
        if row["doc_id"] != doc_id or row["warmup"] or row["variant"] != variant:
            raise ValueError("Warmup or incorrect row mixed into result")
        if (row["input_fingerprints"] != reference["input_fingerprints"] or
            row["reference_metadata"] != reference["reference_metadata"] or
            pairing_signature(row["audit"], contract["method"]) != signature or
            row["budget"]["prefill_tokens_total"] != reference["budget"]["prefill_tokens_total"] or
            row["budget"]["prefill_tokens_total"] > row["budget"]["capacity"]):
            raise ValueError("Independent actual-group/input/budget recheck failed")
    return {v: bundle["records"][v] for v in variants}


def parent_prefix_latency(parent_root):
    """Separate matched three-arm latency table; never complete missing rows."""
    path = Path(parent_root)
    case = path if (path/"run_contract.json").is_file() else path/"snapkv_kvrepack"
    contract = json.loads((case/"run_contract.json").read_text())
    variants = contract_variants(contract)
    if set(variants) != set(KNOWN_VARIANTS):
        raise ValueError("Parent matched latency requires its original three arms")
    if contract.get("diagnostic_audit") is not False or contract.get("warmup_per_arm", 0) < 1:
        raise ValueError("Parent prefix must be a warmed unaudited measurement")
    files = sorted((case/"samples").glob("*.json"))
    if not files or [p.name for p in files] != [f"{i:06d}.json" for i in range(len(files))]:
        raise ValueError("Parent samples must be an unchanged consecutive complete prefix")
    arms = {v: [] for v in variants}
    for doc_id, sample in enumerate(files):
        records = validate_bundle(json.loads(sample.read_text()), contract, doc_id)
        for variant, row in records.items():
            arms[variant].append(row)
    table = [dict(method=contract["method"], variant=variant, scope="parent_matched_prefix_only",
                  **latency_summary(rows)) for variant, rows in arms.items()]
    base = next(row for row in table if row["variant"] == "kvrepack")
    for row in table:
        row["e2e_ratio_to_kvrepack"] = row["end_to_end_seconds_mean"] / base["end_to_end_seconds_mean"]
        row["model_e2e_ratio_to_kvrepack"] = row["model_end_to_end_seconds_mean"] / base["model_end_to_end_seconds_mean"]
    return dict(scope="parent_matched_prefix_only", n=len(files),
                parent_contract_sha256=json_sha(contract), parent_case=str(case.resolve()),
                comparisons=table, quality_scored=False,
                limitation="This paired prefix is a separate sample set, not a 200-row KVRepack baseline; no cross-corpus CIDEr comparison.")


def score_native_rows(rows, contract, output, repo_root):
    """Score exactly this original prefix using the unchanged native helper."""
    count = len(rows)
    if count <= 0 or [row["doc_id"] for row in rows] != list(range(count)):
        raise ValueError("Scoring requires exactly the declared original prefix")
    arrows = list((Path(contract["hf_home"])/"datasets/lmms-lab___vatex/vatex_test").glob("**/vatex-test.arrow"))
    if len(arrows) != 1 or sha256(arrows[0]) != contract["arrow_sha256"]:
        raise ValueError("Scoring reference differs from generation")
    output, repo_root = Path(output), Path(repo_root)
    output.mkdir(parents=True, exist_ok=False)
    predictions = output/"predictions.jsonl"
    predictions.write_text("".join(json.dumps(dict(task="vatex_test", doc_id=row["doc_id"], text=row["text"]), ensure_ascii=False)+"\n" for row in rows))
    helper = repo_root/"provenance/full_video_retest_20261104/helpers/score_full_runs.py"
    command = [sys.executable, "-B", str(helper), "--records", str(predictions),
               "--task", "vatex_test", "--expected-count", str(count),
               "--output", str(output/"metrics.json"), "--repo-root", str(repo_root),
               "--reference-arrow", str(arrows[0])]
    with (output/"scoring.log").open("w") as log:
        subprocess.run(command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=str(Path(sys.prefix)/"lib")+os.pathsep+os.environ.get("LD_LIBRARY_PATH", "")))
    scored = json.loads((output/"metrics.json").read_text())
    if scored.get("n") != count or scored.get("expected_count") != count:
        raise ValueError("Native scorer returned a different corpus size")
    return {name: scored["metrics"][name] for name in QUALITY_METRICS}


def score_parent_prefix(parent_latency, output, repo_root):
    """Separate three-arm quality on the parent corpus; no main-table mutation."""
    case = Path(parent_latency["parent_case"])
    contract = json.loads((case/"run_contract.json").read_text())
    if json_sha(contract) != parent_latency["parent_contract_sha256"]:
        raise ValueError("Parent contract changed before prefix scoring")
    count = parent_latency["n"]
    arms = {variant: [] for variant in contract_variants(contract)}
    for doc_id in range(count):
        data = json.loads((case/"samples"/f"{doc_id:06d}.json").read_text())
        records = validate_bundle(data, contract, doc_id)
        for variant, row in records.items():
            arms[variant].append(row)
    comparisons = []
    for latency in parent_latency["comparisons"]:
        variant = latency["variant"]
        metrics = score_native_rows(arms[variant], contract, Path(output)/variant, repo_root)
        comparisons.append(dict(latency, **metrics, VATEX_Avg=statistics.fmean(metrics.values()),
                                quality_corpus_scope=f"parent_original_first_{count}_only"))
    return dict(parent_latency, comparisons=comparisons, quality_scored=True,
                quality_reference_docs=count,
                quality_corpus_scope=f"parent_original_first_{count}_only",
                limitation=f"All three arms use the same original first {count} VATEX references. CIDEr document frequencies are recomputed on these {count} references and must not be compared directly with the separate first200 table.")


def latency_summary(rows):
    import numpy as np
    result = dict(n=len(rows))
    for name in ("end_to_end_seconds", "preprocessing_seconds", "h2d_seconds",
                 "fisher_seconds", "generation_seconds", "text_decode_seconds"):
        values = [row["runtime"][name] for row in rows]
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("Invalid recorded latency")
        result[name+"_mean"] = statistics.fmean(values)
        result[name+"_total"] = sum(values)
        if name == "end_to_end_seconds":
            result["e2e_p50_seconds"], result["e2e_p95_seconds"] = map(float, np.quantile(values, [.5, .95]))
    result["model_end_to_end_seconds_mean"] = statistics.fmean(
        row["runtime"].get("model_end_to_end_seconds", row["runtime"]["fisher_seconds"]+row["runtime"]["generation_seconds"])
        for row in rows)
    result["generated_tokens_mean"] = statistics.fmean(row["generated_tokens"] for row in rows)
    result["tokens_per_generation_second"] = sum(row["generated_tokens"] for row in rows)/sum(row["runtime"]["generation_seconds"] for row in rows)
    result["eos_count"] = sum(row["eos_reached"] for row in rows)
    result["capped_without_eos"] = sum(row["capped_without_eos"] for row in rows)
    result["actual_retained_ratio_mean"] = statistics.fmean(row["budget"]["actual_retained_ratio"] for row in rows)
    result["actual_retained_ratio_max"] = max(row["budget"]["actual_retained_ratio"] for row in rows)
    result["peak_allocated_gib_max"] = max(row["runtime"]["peak_allocated_gib"] for row in rows)
    return result


def plot_latencies(table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [("HybridKV" if row["method"].startswith("hybrid") else "SnapKV")
              + " + " + {"kvrepack":"KVRepack", "lookm":"LookM agg.", "asymkv":"AsymKV agg."}[row["variant"]]
              for row in table]
    colors = [{"kvrepack":"#2878B5", "lookm":"#F28E2B", "asymkv":"#59A14F"}[row["variant"]] for row in table]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    y = np.arange(len(table))
    for ax, key, title in zip(axes, ("end_to_end_seconds_mean", "model_end_to_end_seconds_mean"),
                              ("Video-to-text end-to-end", "Model end-to-end (includes Fisher)")):
        values = [row[key] for row in table]
        ax.barh(y, values, color=colors, height=.65)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Mean seconds per video")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(values)*1.17)
        for index, value in enumerate(values):
            ax.text(value+max(values)*.015, index, f"{value:.3f}", va="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("VATEX first200 · Qwen2.5-VL-7B · 10% active prefill KV slots", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/("latency_comparison."+suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-run-root", type=Path,
                        help="Optional stopped three-arm SnapKV run; report its complete paired prefix separately")
    parser.add_argument("--score-parent-prefix", action="store_true",
                        help="Also score the separate matched three-arm parent prefix on its own reference corpus")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=False)
    table, detail, input_rows = [], {}, []
    parent_root = args.parent_run_root
    for case in sorted(args.run_root.iterdir()):
        if not (case/"run_contract.json").is_file():
            continue
        contract = json.loads((case/"run_contract.json").read_text())
        variants = contract_variants(contract)
        if parent_root is None and contract.get("parent_prefix"):
            parent_root = contract["parent_prefix"].get("run_root")
        state = json.loads((case/"run_state.json").read_text())
        n = contract["target_docs"]
        if n != 200 or contract.get("diagnostic_audit") is not False or contract.get("warmup_per_arm", 0) < 1:
            raise ValueError("Formal report requires warmed, unaudited first200 runs; pilot diagnostics are separate")
        if state.get("status") != "complete" or state.get("completed_docs") != n:
            raise ValueError("Report requires complete, explicitly scoped runs")
        files = sorted((case/"samples").glob("*.json"))
        if [p.name for p in files] != [f"{i:06d}.json" for i in range(n)]:
            raise ValueError("Original-prefix coverage differs")
        arms = {v:[] for v in variants}
        reuse_count = 0
        for doc_id, path in enumerate(files):
            bundle = json.loads(path.read_text())
            records = validate_bundle(bundle, contract, doc_id)
            reuse_count += bool(bundle.get("reused_from_parent"))
            for variant, row in records.items():
                arms[variant].append(row)
                input_rows.append(dict(method=contract["method"], variant=variant, doc_id=doc_id,
                    generated_tokens=row["generated_tokens"], **row["runtime"]))
        detail[contract["method"]] = dict(contract=contract, state=state, variants={},
                                         reused_docs=reuse_count, new_docs=n-reuse_count)
        for variant, rows in arms.items():
            arm_dir = args.output/contract["method"]/variant
            metrics = score_native_rows(rows, contract, arm_dir, root)
            stats = latency_summary(rows)
            entry = dict(method=contract["method"], variant=variant, **stats,
                **{k:metrics[k] for k in ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr")})
            entry["VATEX_Avg"] = statistics.fmean(entry[k] for k in ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr"))
            table.append(entry)
            detail[contract["method"]]["variants"][variant] = entry
    if not table:
        raise ValueError("No complete runs")
    for method in detail:
        reference_variant = next(iter(detail[method]["variants"]))
        base = detail[method]["variants"][reference_variant]["end_to_end_seconds_mean"]
        for row in table:
            if row["method"] == method:
                row["latency_reference_variant"] = reference_variant
                row["e2e_ratio_to_reference"] = row["end_to_end_seconds_mean"]/base
                if "kvrepack" in detail[method]["variants"]:
                    row["e2e_ratio_to_kvrepack"] = row["end_to_end_seconds_mean"] / detail[method]["variants"]["kvrepack"]["end_to_end_seconds_mean"]
    parent = parent_prefix_latency(parent_root) if parent_root else None
    if parent is not None:
        atomic_json(args.output/"parent_matched_prefix_latency.json", parent)
        with (args.output/"parent_matched_prefix_latency.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(parent["comparisons"][0]))
            writer.writeheader(); writer.writerows(parent["comparisons"])
    if args.score_parent_prefix and parent is None:
        raise ValueError("--score-parent-prefix requires an explicitly identified parent run")
    parent_comparison = score_parent_prefix(parent, args.output/"parent_matched_prefix_scoring", root) if args.score_parent_prefix else None
    if parent_comparison is not None:
        atomic_json(args.output/"parent_matched_prefix_comparison.json", parent_comparison)
        with (args.output/"parent_matched_prefix_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(parent_comparison["comparisons"][0]))
            writer.writeheader(); writer.writerows(parent_comparison["comparisons"])
    for filename, rows in (("comparison.csv", table), ("per_input_latency.csv", input_rows)):
        with (args.output/filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader(); writer.writerows(rows)
    atomic_json(args.output/"summary.json", dict(status="complete", methods=detail,
        comparisons=table, parent_matched_prefix_latency=parent,
        parent_matched_prefix_comparison=parent_comparison,
        timing_scope=next(iter(detail.values()))["contract"]["timing_scope"],
        limitations=["Fixed first200, single measured generation per input/arm, order rotated; not repeated steady-state microbenchmark.",
            "Only arms explicitly declared by each contract are scored; stopped parent measurements retain their original rows, order and timing metadata.",
            "The optional three-arm parent-prefix latency table is separate and supplies no missing 200-input KVRepack baseline.",
            "Merger formulas transplanted onto identical host groups/anchors/raw-K space; not complete LookM or AsymKV pipelines.",
            "HybridKV uses a global active-slot budget with heterogeneous head allocation; SnapKV uniform head budgets.",
            "Reported VRAM also includes HybridKV backing/offload storage and AsymKV probe; KV-slot budget is not a total-memory budget."]))
    plot_latencies(table, args.output)
    lines = ["# VATEX fixed-group merge comparison", "", "Qwen2.5-VL-7B, 64 frames, original first200, strict 10% total active prefill KV positions.", "",
        "| Backbone | Aggregation | BLEU-4 | CIDEr | METEOR | R-L | VATEX Avg. | E2E mean (s) | P50 (s) | P95 (s) | Model E2E (s) | Fisher mean (s) | Generation mean (s) | Generated tokens |", 
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| "+" | ".join([row["method"], row["variant"]]+[f"{row[k]:.6f}" for k in ("Bleu_4", "CIDEr", "METEOR", "ROUGE_L", "VATEX_Avg", "end_to_end_seconds_mean", "e2e_p50_seconds", "e2e_p95_seconds", "model_end_to_end_seconds_mean", "fisher_seconds_mean", "generation_seconds_mean", "generated_tokens_mean")])+" |")
    if parent is not None:
        lines += ["", f"Separate matched parent prefix: original first {parent['n']} inputs, three arms measured before the stop. These rows are not a 200-input KVRepack baseline and no prefix CIDEr is mixed with first200 scores.", "",
                  "| Backbone | Aggregation | N | E2E mean (s) | Model E2E (s) | E2E / KVRepack | Model / KVRepack |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in parent["comparisons"]:
            lines.append("| "+" | ".join([row["method"], row["variant"], str(row["n"])]+[f"{row[k]:.6f}" for k in ("end_to_end_seconds_mean", "model_end_to_end_seconds_mean", "e2e_ratio_to_kvrepack", "model_e2e_ratio_to_kvrepack")])+" |")
    if parent_comparison is not None:
        parent_lines = [f"# Matched parent first{parent_comparison['n']} quality and latency", "",
            parent_comparison["limitation"], "",
            "| Backbone | Aggregation | N | BLEU-4 | CIDEr | METEOR | R-L | VATEX Avg. | E2E mean (s) | Model E2E (s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in parent_comparison["comparisons"]:
            parent_lines.append("| "+" | ".join([row["method"], row["variant"], str(row["n"])]+[f"{row[k]:.6f}" for k in ("Bleu_4", "CIDEr", "METEOR", "ROUGE_L", "VATEX_Avg", "end_to_end_seconds_mean", "model_end_to_end_seconds_mean")])+" |")
        (args.output/"parent_matched_prefix_comparison.md").write_text("\n".join(parent_lines)+"\n")
        lines += [""] + parent_lines
    lines += ["", next(iter(detail.values()))["contract"]["timing_scope"], "", "Model E2E is Fisher(if needed)+prefill+decode from GPU-resident input to generated token IDs. Full E2E additionally includes native video decode/preprocessing, H2D and text decoding. Model loading, warmup, audit hashing, result writes and CPU scoring are excluded.", "", "LookM/AsymKV labels denote fixed-group aggregation transplants. All arms share KVRepack groups, heads, anchors and raw-K/reapplied-mRoPE position policy. AsymKV retains the released matmul/exp/cardinality-denominator/Vsum attention arithmetic, batched across heads; full-prompt tail32 Fisher is an adaptation of the released chunked probe."]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

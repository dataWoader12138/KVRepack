"""Offline HybridKV-KVRepack similarity versus fixed Random-head on NextQA.

Each committed sample holds all four conditions. Only one bundle is retained in
memory; original task WUPS functions are memoized on exact prediction/reference
arguments. This scorer never loads a model, uses a GPU, downloads, or generates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

from .head_selection_report import (
    VARIANTS, _canonical, _helper, _nextqa_namespace, _sha,
    offline_resources, paired_wups_intervals,
)


TASK = "nextqa_oe_test"
METHOD = "hybridkv_kvrepack"
SELECTED_HEADS = 56


def _json_sha(value):
    # Match the worker's default ensure_ascii=True contract/fingerprint encoding.
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _run_contract(root, limit, *, complete=True):
    root = Path(root)
    contract = json.loads((root/"run_contract.json").read_text())
    state = json.loads((root/"run_state.json").read_text())
    if (contract.get("schema_version") != 1 or contract.get("task") != TASK or contract.get("method") != METHOD
            or contract.get("target_docs") != limit or contract.get("target_generations") != 4*limit):
        raise ValueError("Declared source/target differs from the explicit NextQA prefix")
    if (state.get("target_docs") != limit or state.get("target_generations") != 4*limit
            or (complete and (state.get("status") != "complete" or state.get("completed_docs") != limit
                             or state.get("completed_generations") != 4*limit))):
        raise ValueError("Generation is incomplete; --partial reports progress without scoring")
    paths = sorted((root/"samples").glob("*.json"))
    expected = {f"{doc:06d}.json" for doc in range(limit)}
    if any(path.name not in expected for path in paths) or (complete and {p.name for p in paths} != expected):
        raise ValueError("Committed sample filenames differ from the prescribed first-prefix IDs")
    return contract, state, paths


def progress(root, limit):
    contract, state, paths = _run_contract(root, limit, complete=False)
    return dict(status=state.get("status"), task=TASK, expected_inputs=limit,
                committed_sample_files=len(paths), expected_generations=4*limit,
                reported_completed_generations=state.get("completed_generations"),
                progress_only=True, scoring_performed=False, metrics=None,
                note="File counts are progress only; source/content validation is performed before completed-run scoring")


def _audit(row, variant):
    """Record heterogeneous merge work; the worker checks per-head budgets."""
    audit = row["audit"]
    def flatten(mask):
        if (not isinstance(mask, list) or len(mask) != 28 or any(not isinstance(layer, list) or len(layer) != 1
                or not isinstance(layer[0], list) or len(layer[0]) != 4 or any(type(bit) is not bool for bit in layer[0]) for layer in mask)):
            raise ValueError("Hybrid native masks require [layers=28,batch=1,KVheads=4]")
        return [bit for layer in mask for bit in layer[0]]
    selected, baseline = flatten(audit.get("selected_masks")), flatten(audit.get("baseline_masks"))
    if (audit.get("status") != "completed" or audit.get("layers") != 28 or audit.get("native_kv_heads") != 4
            or audit.get("batch_size") != 1 or sum(selected) != SELECTED_HEADS or sum(baseline) != SELECTED_HEADS
            or audit.get("selected_count") != [56] or audit.get("baseline_selected_count") != [56]):
        raise ValueError("Hybrid requires 56 selected native heads out of 112")
    if variant == "similarity" and selected != baseline:
        raise ValueError("Similarity strategy changed its native mask")
    removed = audit.get("actual_removed_tokens")
    if type(removed) is not int or removed < 0 or audit.get("actual_merge_calls") != SELECTED_HEADS:
        raise ValueError("Missing actual Hybrid merge work or 56 effective merges")
    calls = audit.get("merge_calls")
    if (not isinstance(calls, list) or len(calls) != 56
            or any(any(type(call.get(key)) is not int for key in ("layer", "batch", "head", "before", "after"))
                   or not 0 <= call["layer"] < 28 or call["batch"] != 0 or not 0 <= call["head"] < 4
                   or not selected[4*call["layer"]+call["head"]] or not 0 <= call["after"] < call["before"] for call in calls)
            or len({(call["layer"], call["head"]) for call in calls}) != 56
            or sum(call["before"]-call["after"] for call in calls) != removed):
        raise ValueError("Actual merge calls/removed work disagree with selected Hybrid heads")
    union = sum(a or b for a, b in zip(selected, baseline))
    return dict(selected_count=SELECTED_HEADS, effective_merge_count=SELECTED_HEADS,
                removed_tokens=removed, mask_hamming=sum(a != b for a, b in zip(selected, baseline)),
                mask_jaccard=sum(a and b for a, b in zip(selected, baseline))/union,
                selected_mask=selected)


def score_stream(root, documents, reference, *, scorer, aggregate, limit=200,
                 bootstrap_samples=5000, seed=20261104, validator=None):
    """Validate and score one four-condition bundle at a time; no partial scores."""
    root = Path(root)
    if validator is None:
        from .run_hybrid_head_selection import validate_bundle
        validator = validate_bundle
    contract, state, paths = _run_contract(root, limit)
    identity = _json_sha(contract)
    if len(documents) != limit or contract.get("arrow_sha256") != reference.get("sha256"):
        raise ValueError("Scoring references differ from the generation Arrow/prefix")
    if contract.get("dataset_rows") != reference.get("split_rows"):
        raise ValueError("Reference split length differs from the recorded source")
    values = {variant: [] for variant in VARIANTS}
    task_results = {variant: [] for variant in VARIANTS}
    score_rows, audit_rows, source_files, source_rows = [], [], [], []
    frequencies = {variant: [0]*112 for variant in VARIANTS}
    fixed_random_masks, cache = {}, {}
    hits, evaluated, reference_config = 0, 0, None
    generation = {variant: dict(eos_count=0, reached_length_limit=0, capped_without_eos=0, tokens=[]) for variant in VARIANTS}
    fairness = []
    for doc_id, (path, doc) in enumerate(zip(paths, documents)):
        payload = path.read_bytes()
        bundle = json.loads(payload)
        validator(bundle, doc_id=doc_id, contract_sha256=identity)
        metadata = bundle["reference_metadata"]
        digest = hashlib.sha256(_canonical(doc).encode()).hexdigest()
        if (metadata.get("arrow_sha256") != reference["sha256"] or metadata.get("original_row_sha256") != digest
                or metadata.get("source_video_id") != str(doc["video"]) or metadata.get("video_id") != str(doc["video"])):
            raise ValueError(f"Input/scoring original row or source video mismatch: doc_id={doc_id}")
        rows = bundle["records"]
        baseline_pre = rows["similarity"]["audit"].get("pre_selection")
        # Native allocation releases head_category; the observer captures its
        # calibrated values and tensor fingerprint before that release.
        calibrated = baseline_pre.get("calibrated_head_category") if isinstance(baseline_pre, dict) else None
        if (not isinstance(baseline_pre, dict) or not isinstance(calibrated, dict)
                or not calibrated.get("values") or not calibrated.get("tensor")
                or not baseline_pre.get("sparse_budget_per_head") or not baseline_pre.get("candidate_hashes")
                or any(rows[variant]["audit"].get("pre_selection") != baseline_pre
                       or rows[variant]["audit"].get("baseline_masks") != rows["similarity"]["audit"].get("baseline_masks") for variant in VARIANTS)):
            raise ValueError("Native pre-selection categories/budgets/candidates differ across conditions")
        paired_config = {key: value for key, value in rows["similarity"]["config"].items() if key not in ("strategy", "mask_seed")}
        if reference_config is None:
            reference_config = paired_config
        if paired_config != reference_config:
            raise ValueError("Nonselection method/model/generation configuration changed across inputs")
        source_rows.append(dict(doc_id=doc_id, source_video_id=metadata["source_video_id"]))
        for variant in VARIANTS:
            row = rows[variant]
            audit = _audit(row, variant)
            selected = audit.pop("selected_mask")
            audit_rows.append(dict(doc_id=doc_id, variant=variant, **audit))
            ids, cap = row["generated_ids"], row["config"]["max_new_tokens"]
            eos = ids[-1] in row["eos_token_id"] if isinstance(row["eos_token_id"], list) else ids[-1] == row["eos_token_id"]
            if cap != 128 or len(ids) > cap or row["generated_tokens"] != len(ids) or row["eos_reached"] is not eos:
                raise ValueError("Generation token cap/count/EOS metadata mismatch")
            generation[variant]["eos_count"] += eos
            generation[variant]["reached_length_limit"] += len(ids) == cap
            generation[variant]["capped_without_eos"] += len(ids) == cap and not eos
            generation[variant]["tokens"].append(len(ids))
            for index, bit in enumerate(selected):
                frequencies[variant][index] += bit
            if variant != "similarity":
                if fixed_random_masks.setdefault(variant, selected) != selected:
                    raise ValueError("Random-global ranking changed across inputs")
            # Scoring depends on exact prediction and the original answer/type
            # fields, not on input position or a different question's prediction.
            key = (row["text"], _canonical({name: doc[name] for name in ("answer", "type", "additional_ref_answer")}))
            if key not in cache:
                result = scorer(doc, [row["text"]])["WUPS"]
                if not isinstance(result, dict) or any(type(result.get(name)) not in (int, float)
                        or not math.isfinite(result[name]) or not 0 <= result[name] <= 1 for name in ("0", "0.9")):
                    raise ValueError("Original task scorer returned invalid WUPS")
                cache[key] = result
                evaluated += 1
            else:
                hits += 1
            result = cache[key]
            task_results[variant].append(result)
            score = 100.*result["0"]
            values[variant].append(score)
            score_rows.append(dict(doc_id=doc_id, source_video_id=metadata["source_video_id"], variant=variant,
                                   WUPS=score, WUPS_0_9=100.*result["0.9"],
                                   prediction_sha256=hashlib.sha256(row["text"].encode()).hexdigest()))
        if path.read_bytes() != payload:
            raise ValueError("Committed sample changed while scoring")
        source_files.append(dict(doc_id=doc_id, path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
        fairness.append(dict(doc_id=doc_id, fairness=bundle["fairness"]))
    scores = {variant: float(aggregate(task_results[variant])) for variant in VARIANTS}
    if any(abs(scores[variant]-statistics.fmean(values[variant])) > 1e-10 for variant in VARIANTS):
        raise ValueError("Input score mean differs from the unchanged official task aggregate")
    interval = paired_wups_intervals({TASK: {"similarity": source_rows}}, {TASK: values}, samples=bootstrap_samples, seed=seed)
    random = [scores[variant] for variant in VARIANTS[1:]]
    paired_rows = [dict(doc_id=doc_id, source_video_id=row["source_video_id"], variant=variant,
                        similarity=values["similarity"][doc_id], random=values[variant][doc_id],
                        difference=values["similarity"][doc_id]-values[variant][doc_id])
                   for doc_id, row in enumerate(source_rows) for variant in VARIANTS[1:]]
    merge_work = {}
    for variant in VARIANTS:
        work = [row["removed_tokens"] for row in audit_rows if row["variant"] == variant]
        baseline = [row["removed_tokens"] for row in audit_rows if row["variant"] == "similarity"]
        merge_work[variant] = dict(min=min(work), max=max(work), mean=statistics.fmean(work),
                                   total=sum(work), inputs_differing_from_similarity=sum(a != b for a,b in zip(work,baseline)),
                                   mean_minus_similarity=statistics.fmean(a-b for a,b in zip(work,baseline)))
        tokens = generation[variant].pop("tokens")
        generation[variant].update(n_inputs=limit, generated_tokens_min=min(tokens), generated_tokens_max=max(tokens), generated_tokens_mean=statistics.fmean(tokens))
    return dict(schema_version=1, status="complete", method=METHOD, task=TASK, limit=limit,
                scope=f"Exactly original NextQA rows0..{limit-1}; four conditions; no VATEX or automatic expansion",
                n_generations=4*limit, n_source_videos=len({row["source_video_id"] for row in source_rows}),
                scores=scores, random_mean=statistics.fmean(random), random_sample_std=statistics.stdev(random),
                similarity_minus_random_mean=scores["similarity"]-statistics.fmean(random), paired_WUPS=interval,
                bootstrap=dict(samples=bootstrap_samples, seed=seed, unit="source_video_id", pairing="Same cluster draws for all conditions, conditional on fixed random seeds 0,1,2"),
                merge_work=merge_work, generation=generation,
                audit=dict(status="passed", n_inputs=limit, n_conditions=4,
                           checks="Original Arrow/row/video identity, same input/candidate bytes, native masks and per-head category/retained budgets, 56 selected and 56 effective merges, fixed seed ranking and nonselection configuration; actual merge work is recorded per strategy and may differ"),
                metric="Unchanged native NextQA WUPS@0 aggregate in 0-100 points; negative paired differences are retained",
                interpretation="Hybrid Global Random-head changes head identities and layer allocation under the same 56-of-112 global quota and unchanged native per-head retained budgets. Heterogeneous head budgets mean total removed-token work may differ as a consequence of selected identities; work is reported, not forced equal. Three seeds are not additional independent inputs; seed sample SD and input-cluster CI describe different uncertainty. This prefix alone does not establish general superiority.",
                scorer_cache=dict(unique_prediction_reference_evaluations=evaluated, exact_cache_hits=hits),
                provenance=dict(run_contract_path=str((root/"run_contract.json").resolve()), run_contract_file_sha256=_sha(root/"run_contract.json"),
                                run_contract_sha256=identity, generation_contract=contract, reference=reference,
                                scoring_source_sha256=_sha(__file__), sample_files=source_files, fairness=fairness),
                scores_table=score_rows, paired_table=paired_rows, audit_table=audit_rows,
                selection_frequency={variant: [count/limit for count in counts] for variant, counts in frequencies.items()})


def _csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.5))
        axes[0].plot(range(4), [summary["scores"][variant] for variant in VARIANTS], "o", color="#377EB8")
        axes[0].errorbar(4, summary["random_mean"], yerr=summary["random_sample_std"], fmt="s", capsize=3, color="#E68A2E")
        axes[0].set(xticks=range(5), xticklabels=["Similarity", "Seed 0", "Seed 1", "Seed 2", "Mean ± SD"], ylabel="WUPS@0 (0–100)", title="Task score; SD across three fixed rankings")
        for index, value in enumerate([summary["paired_WUPS"]["variants"][variant] for variant in VARIANTS[1:]]+[summary["paired_WUPS"]["random_mean"]]):
            axes[1].plot(index, value["difference"], "o", color="#377EB8")
            if value["ci95"] is not None:
                axes[1].vlines(index, *value["ci95"], color="#377EB8")
        axes[1].axhline(0, color=".6", linestyle="--", linewidth=.8)
        axes[1].set(xticks=range(4), xticklabels=["Seed 0", "Seed 1", "Seed 2", "Seed mean"], ylabel="Similarity − random (points)", title="Paired 95% source-video cluster intervals")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", labelrotation=25)
        fig.suptitle(f"HybridKV-KVRepack / NextQA: {summary['limit']} inputs, {summary['n_source_videos']} source videos", fontsize=10)
        fig.text(.5,.015,"56/112 native heads; unchanged heterogeneous per-head budgets; removed work may differ.\nSD across three fixed rankings and paired input-cluster confidence intervals are separate quantities.",ha="center",fontsize=7.2)
        fig.subplots_adjust(left=.08,right=.985,top=.81,bottom=.29,wspace=.32)
        for extension in ("png","pdf","svg"):
            fig.savefig(output/f"nextqa_scores.{extension}",dpi=300,bbox_inches="tight")
        plt.close(fig)


def write_report(summary, output, *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _csv(output/"scores.csv", summary["scores_table"], ["doc_id","source_video_id","variant","WUPS","WUPS_0_9","prediction_sha256"])
    _csv(output/"paired_scores.csv", summary["paired_table"], ["doc_id","source_video_id","variant","similarity","random","difference"])
    _csv(output/"selector_audit.csv", summary["audit_table"], ["doc_id","variant","selected_count","effective_merge_count","removed_tokens","mask_hamming","mask_jaccard"])
    _csv(output/"merge_work.csv", [dict(variant=variant, **values) for variant,values in summary["merge_work"].items()],
         ["variant","min","max","mean","total","inputs_differing_from_similarity","mean_minus_similarity"])
    _csv(output/"generation_summary.csv", [dict(variant=variant, **values) for variant,values in summary["generation"].items()],
         ["variant","n_inputs","eos_count","reached_length_limit","capped_without_eos","generated_tokens_min","generated_tokens_mean","generated_tokens_max"])
    _csv(output/"sample_files.csv", summary["provenance"]["sample_files"], ["doc_id","path","sha256","bytes"])
    run_rows = []
    for variant in VARIANTS:
        values = summary["paired_WUPS"]["variants"].get(variant, {})
        run_rows.append(dict(variant=variant, WUPS=summary["scores"][variant], similarity_minus_random=values.get("difference"),
                             ci95_low=(values.get("ci95") or [None,None])[0], ci95_high=(values.get("ci95") or [None,None])[1]))
    _csv(output/"run_scores.csv", run_rows, ["variant","WUPS","similarity_minus_random","ci95_low","ci95_high"])
    compact = {key:value for key,value in summary.items() if not key.endswith("_table")}
    lines = ["# HybridKV-KVRepack NextQA Random-head evaluation", "", f"Status: **{summary['status']}**. {summary['scope']}.", "", summary["metric"], "", summary["interpretation"], "",
             "| Condition | WUPS | Similarity − random (95% paired CI) |", "|---|---:|---|"]
    for row in run_rows:
        difference = "—" if row["similarity_minus_random"] is None else f"{row['similarity_minus_random']:.6f}"
        if row["ci95_low"] is not None:difference+=f" [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"
        lines.append(f"| {row['variant']} | {row['WUPS']:.6f} | {difference} |")
    interval=summary["paired_WUPS"]["random_mean"]["ci95"]
    lines += ["", f"Random mean ± sample SD (ddof=1): **{summary['random_mean']:.6f} ± {summary['random_sample_std']:.6f}**.",
              f"Similarity − random mean: **{summary['similarity_minus_random_mean']:.6f}**; 95% paired interval: {interval if interval is not None else 'unavailable (fewer than two source videos)' }.",
              f"Each condition has {summary['limit']} inputs. Bootstrap resamples {summary['n_source_videos']} source videos with all member questions, using common draws across conditions and the fixed three seeds.", ""]
    lines += ["| Condition | Mean removed tokens | Total removed tokens | Inputs differing from similarity |", "|---|---:|---:|---:|"]
    for variant,data in summary["merge_work"].items():
        lines.append(f"| {variant} | {data['mean']:.3f} | {data['total']} | {data['inputs_differing_from_similarity']} |")
    lines += ["", "Removed-token totals may differ because the fixed native per-head retained budgets are heterogeneous. This is an identity-selection consequence, not an extra budget intervention.", ""]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if plots:_plot(summary,output)


def main(argv=None):
    runtime=str(Path(sys.prefix)/"lib")
    if argv is None and (Path(runtime)/"libstdc++.so.6").is_file() and os.environ.get("LD_LIBRARY_PATH","").split(os.pathsep)[0]!=runtime:
        env=dict(os.environ,LD_LIBRARY_PATH=runtime+os.pathsep+os.environ.get("LD_LIBRARY_PATH",""))
        os.execve(sys.executable,[sys.executable,"-m","kvrepack.ablations.hybrid_head_selection_report",*sys.argv[1:]],env)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--hf-home",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--limit",type=int,choices=(1,200),default=200)
    parser.add_argument("--bootstrap-samples",type=int,default=5000)
    parser.add_argument("--seed",type=int,default=20261104)
    parser.add_argument("--nltk-data",type=Path,default=Path(os.environ.get("NLTK_DATA",str(Path.home()/"nltk_data"))))
    parser.add_argument("--partial",action="store_true",help="Report only progress; never score incomplete runs")
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args(argv)
    if args.partial:
        print(json.dumps(progress(args.run_root,args.limit)));return 0
    output=args.output or args.run_root/"report"
    if output.exists():raise ValueError("Report output already exists; refusing overwrite or duplicate scoring")
    contract,_,_=_run_contract(args.run_root,args.limit)
    repo=Path(__file__).resolve().parents[3]
    helper,helper_path=_helper(repo)
    began=time.monotonic()
    from ..observations.inputs import TASKS as DATA_TASKS
    _,_,cache_name,arrow_name=DATA_TASKS[TASK]
    matches=list((args.hf_home/"datasets"/cache_name).glob(f"**/{arrow_name}"))
    if len(matches)!=1:raise ValueError("Require exactly one cached original NextQA reference Arrow")
    with offline_resources(nltk_data=args.nltk_data,java_home=None):
        records=[dict(task=TASK,doc_id=doc,text="") for doc in range(args.limit)]
        documents,reference=helper.reference_rows(SimpleNamespace(reference_arrow=matches[0],task=TASK,expected_count=args.limit),records)
        namespace,source=_nextqa_namespace(helper,repo)
        summary=score_stream(args.run_root,documents,reference,scorer=namespace["nextqa_process_results"],aggregate=namespace["nextqa_aggregate_results"],
                             limit=args.limit,bootstrap_samples=args.bootstrap_samples,seed=args.seed)
    summary["provenance"].update(task_scorer=source,helper_path=str(helper_path),helper_sha256=_sha(helper_path),
                                 dependency_versions={name:importlib.metadata.version(name) for name in ("nltk","pywsd","wn","pandas","pyarrow")})
    summary["scoring_seconds"]=time.monotonic()-began
    write_report(summary,output,plots=not args.no_plots)
    print(json.dumps(dict(status=summary["status"],limit=args.limit,scores=summary["scores"],random_mean=summary["random_mean"],
                          random_sample_std=summary["random_sample_std"],difference=summary["similarity_minus_random_mean"],
                          ci95=summary["paired_WUPS"]["random_mean"]["ci95"],output=str(output))))
    return 0


if __name__=="__main__":raise SystemExit(main())

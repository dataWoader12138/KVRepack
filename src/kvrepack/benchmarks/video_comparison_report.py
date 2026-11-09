"""Score a completed two-method NextQA/VATEX matrix using native offline metrics.

The ordered matrix conditions define every difference as second minus first.
No model/GPU is loaded, no inference/download is performed, and old reports
are never overwritten. CIDEr uses the same selected reference corpus per method.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

from ..ablations.head_selection_report import (
    VATEX_METRICS, _canonical, _helper, _nextqa_namespace, _sha, offline_resources,
)

TASKS = ("nextqa_oe_test", "vatex_test")
COMMON_FIELDS = ("model_snapshot", "model_metadata", "frames", "min_pixels", "max_pixels",
                 "budget_ratio", "dtype", "attention", "model_seed", "max_new_tokens", "do_sample", "use_cache", "repetition_penalty")


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _cache_audit(audit, *, method, task, doc_id):
    if (not isinstance(audit, dict) or audit.get("status") != "completed" or audit.get("method") != method
            or audit.get("task") != task or audit.get("doc_id") != doc_id or audit.get("native_kv_heads") != 4
            or audit.get("layers_count") != 28 or audit.get("nominal_budget_ratio") != .1):
        raise ValueError("Incomplete native cache audit or wrong method/task/head coverage")
    layers = audit.get("layers")
    if not isinstance(layers,list) or [layer.get("layer") for layer in layers] != list(range(28)):
        raise ValueError("Native prefill cache audit must cover all28 layers")
    for layer in layers:
        lengths, before = layer.get("head_lengths"), layer.get("before_tokens_per_head")
        if (not isinstance(lengths,list) or len(lengths)!=4 or any(type(n) is not int or n<1 for n in lengths)
                or type(before) is not int or before<1 or layer.get("after_tokens_total") != sum(lengths)):
            raise ValueError("Cache token counts disagree with the native per-head lengths")
    actual = sum(layer["after_tokens_total"] for layer in layers)
    before = sum(4*layer["before_tokens_per_head"] for layer in layers)
    ratio = audit.get("actual_retained_ratio")
    if (audit.get("prefill_tokens_total") != actual or audit.get("uncompressed_tokens_total") != before
            or type(ratio) not in (int,float) or not math.isfinite(ratio) or abs(ratio-actual/before)>1e-12):
        raise ValueError("Native cache retained ratio/totals disagree")
    return dict(prefill_tokens_total=actual,uncompressed_tokens_total=before,actual_retained_ratio=ratio)


def load_matrix(root, *, limit=200, validator=None):
    """Read one sample at a time and retain only compact score inputs/audit hashes."""
    root = Path(root)
    if validator is None:
        from .run_video_comparison import validate_sample
        validator = validate_sample
    matrix_path = root/"matrix.json"
    matrix = json.loads(matrix_path.read_text())
    methods = matrix.get("conditions")
    if (matrix.get("schema_version") != 1 or matrix.get("experiment") != "lookm_vs_kvrepack"
            or not isinstance(methods, list) or len(methods) != 2 or len(set(methods)) != 2
            or any(not isinstance(method, str) or not re.fullmatch(r"[a-z0-9_]+", method) for method in methods)
            or matrix.get("tasks") != list(TASKS) or matrix.get("limit") != limit
            or matrix.get("budget_ratio") != .1 or matrix.get("frames") != 64):
        raise ValueError("Require the ordered two-method, two-task declared 10%/64-frame prefix matrix")
    runs = {task: {} for task in TASKS}
    provenance = dict(matrix=matrix, matrix_path=str(matrix_path.resolve()), matrix_sha256=_sha(matrix_path), runs={})
    for task in TASKS:
        provenance["runs"][task] = {}
        for method in methods:
            directory = root/method/task
            contract_path, state_path = directory/"run_contract.json", directory/"run_state.json"
            contract = json.loads(contract_path.read_text()); state = json.loads(state_path.read_text())
            if (contract.get("schema_version") != 1 or contract.get("method") != method or contract.get("task") != task
                    or contract.get("target_docs") != limit or contract.get("target_generations") != limit
                    or contract.get("budget_ratio") != .1 or contract.get("frames") != 64
                    or state.get("status") != "complete" or state.get("target_docs") != limit
                    or state.get("target_generations") != limit or state.get("completed_docs") != limit
                    or state.get("completed_generations") != limit):
                raise ValueError(f"Require all four complete runs with exactly {limit} original inputs: {method}/{task}")
            identity = _json_sha(contract)
            paths = sorted((directory/"samples").glob("*.json"))
            if [path.name for path in paths] != [f"{doc:06d}.json" for doc in range(limit)]:
                raise ValueError("Missing, duplicate or out-of-prefix committed sample files")
            rows, files, common_config = [], [], None
            for doc_id, path in enumerate(paths):
                payload = path.read_bytes(); row = json.loads(payload)
                validator(row,doc_id=doc_id,contract_sha256=identity,method=method,task=task)
                meta, fingerprints, config = row.get("reference_metadata"), row.get("input_fingerprints"), row.get("config")
                if (row.get("schema_version") != 1 or row.get("status") != "completed" or row.get("method") != method
                        or row.get("task") != task or row.get("doc_id") != doc_id or row.get("contract_sha256") != identity
                        or not isinstance(meta, dict) or meta.get("task") != task or meta.get("doc_id") != doc_id
                        or not isinstance(fingerprints, dict) or not fingerprints.get("input_hashes")
                        or any(row.get(key) != value for key,value in fingerprints.items())
                        or not isinstance(config, dict) or not config or not isinstance(row.get("text"), str)
                        or row.get("prediction") != row["text"] or row.get("source_video_id") != meta.get("source_video_id")):
                    raise ValueError("Sample identity, original inputs, configuration or exact prediction is missing/mismatched")
                if common_config is None: common_config = config
                if config != common_config:
                    raise ValueError("Method/model/generation configuration changed across inputs in one run")
                audit = row.get("cache_audit", row.get("audit"))
                cache_counts = _cache_audit(audit,method=method,task=task,doc_id=doc_id)
                ids, cap, eos_id = row.get("generated_ids"), row.get("max_new_tokens", config.get("max_new_tokens")), row.get("eos_token_id")
                if (not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids)
                        or type(cap) is not int or cap < 1 or cap != contract.get("max_new_tokens")
                        or len(ids) > cap or row.get("generated_tokens") != len(ids)):
                    raise ValueError("Invalid generated tokens or generation cap")
                eos = ids[-1] in eos_id if isinstance(eos_id, list) else ids[-1] == eos_id
                if row.get("eos_reached") is not eos:
                    raise ValueError("Actual generated tokens disagree with EOS metadata")
                rows.append(dict(task=task, method=method, doc_id=doc_id, text=row["text"],
                                 source_video_id=meta["source_video_id"], reference_metadata=meta,
                                 input_fingerprints=fingerprints, generated_tokens=len(ids), max_new_tokens=cap,
                                 eos_reached=eos, reached_length_limit=len(ids)==cap,
                                 capped_without_eos=len(ids)==cap and not eos,
                                 **cache_counts,
                                 cache_audit_sha256=hashlib.sha256(_canonical(audit).encode()).hexdigest()))
                if path.read_bytes() != payload: raise ValueError("Committed sample changed while reading")
                files.append(dict(doc_id=doc_id, path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest()))
            runs[task][method] = rows
            provenance["runs"][task][method] = dict(contract=contract, contract_sha256=identity,
                contract_file_sha256=_sha(contract_path), state_file_sha256=_sha(state_path), config=common_config, files=files)
        first_contract, second_contract = [provenance["runs"][task][method]["contract"] for method in methods]
        if any(first_contract.get(key) != second_contract.get(key) for key in COMMON_FIELDS):
            raise ValueError("Models, budgets, frames, processing or generation settings differ across methods")
        for a,b in zip(*(runs[task][method] for method in methods)):
            if (a["input_fingerprints"] != b["input_fingerprints"] or a["reference_metadata"] != b["reference_metadata"]
                    or a["source_video_id"] != b["source_video_id"]):
                raise ValueError("Paired methods used different original inputs or references")
    if _sha(matrix_path) != provenance["matrix_sha256"]: raise ValueError("Matrix changed while reading")
    return runs, provenance


def paired_interval(source_rows, first, second, *, samples=5000, seed=20261104):
    """Source-video cluster bootstrap of second minus first, retaining row weights."""
    import numpy as np
    if type(samples) is not int or samples < 1: raise ValueError("Positive bootstrap samples required")
    delta = np.asarray(second, dtype=np.float64)-np.asarray(first, dtype=np.float64)
    if (delta.ndim != 1 or len(delta) != len(source_rows) or not len(delta) or not np.isfinite(delta).all()):
        raise ValueError("Require paired finite per-input scores")
    groups = defaultdict(list)
    for index,row in enumerate(source_rows): groups[row["source_video_id"]].append(index)
    result = dict(difference=float(delta.mean()), ci95=None, n_inputs=len(delta), n_source_videos=len(groups))
    if len(groups) < 2: return result
    counts = np.asarray([len(indices) for _,indices in sorted(groups.items())])
    totals = np.asarray([delta[indices].sum() for _,indices in sorted(groups.items())])
    rng = np.random.default_rng(seed); draws = []
    for start in range(0,samples,256):
        selected = rng.integers(0,len(groups),size=(min(256,samples-start),len(groups)))
        draws.extend((totals[selected].sum(axis=1)/counts[selected].sum(axis=1)).tolist())
    result["ci95"] = np.quantile(draws,[.025,.975]).tolist()
    return result


def score_runs(runs, provenance, *, hf_home, repo_root, nltk_data, java_home):
    """Native task AST and scorers; every task uses identical references for both methods."""
    from ..observations.inputs import TASKS as DATA_TASKS
    helper, helper_path = _helper(repo_root)
    methods = provenance["matrix"]["conditions"]; limit = provenance["matrix"]["limit"]
    metrics, per_input, references, task_sources = {}, {}, {}, {}
    with offline_resources(nltk_data=nltk_data,java_home=java_home), tempfile.TemporaryDirectory(prefix="video-comparison-score-") as temporary:
        for task in TASKS:
            _,_,cache,filename = DATA_TASKS[task]
            matches = list((Path(hf_home)/"datasets"/cache).glob("**/"+filename))
            if len(matches) != 1: raise ValueError("Require exactly one local original reference Arrow")
            docs, reference = helper.reference_rows(SimpleNamespace(reference_arrow=matches[0],task=task,expected_count=limit),runs[task][methods[0]])
            references[task] = reference
            for method in methods:
                contract = provenance["runs"][task][method]["contract"]
                if contract.get("arrow_sha256") != reference["sha256"] or contract.get("dataset_rows") != reference["split_rows"]:
                    raise ValueError("Scoring Arrow differs from the actual generation source")
                for doc,row in zip(docs,runs[task][method]):
                    meta = row["reference_metadata"]; video = str(doc["video"] if task==TASKS[0] else doc["videoID"])
                    source_video = video if task==TASKS[0] else video.rsplit("_",2)[0]
                    if (meta.get("arrow_sha256") != reference["sha256"] or meta.get("original_row_sha256") != hashlib.sha256(_canonical(doc).encode()).hexdigest()
                            or meta.get("video_id") != video or meta.get("source_video_id") != source_video):
                        raise ValueError("Scoring reference row/video does not match the actual model input")
            metrics[task] = {}; per_input[task] = {}
            if task==TASKS[0]:
                namespace, source = _nextqa_namespace(helper,repo_root); cache_results = {}
                for method in methods:
                    results = []
                    for row,doc in zip(runs[task][method],docs):
                        key = (row["text"],_canonical({field:doc[field] for field in ("answer","type","additional_ref_answer")}))
                        if key not in cache_results: cache_results[key] = namespace["nextqa_process_results"](doc,[row["text"]])["WUPS"]
                        results.append(cache_results[key])
                    metrics[task][method] = {"WUPS":float(namespace["nextqa_aggregate_results"](results))}
                    per_input[task][method] = [dict(WUPS=100*result["0"],WUPS_0_9=100*result["0.9"]) for result in results]
            else:
                path = Path(repo_root)/"vendor/lmms-eval/lmms_eval/tasks/vatex/utils.py"
                nodes, source = helper.function_spec(path,helper.VATEX_FUNCTIONS)
                for method in methods:
                    result, _ = helper.vatex_score(path,nodes,runs[task][method],docs,Path(temporary))
                    metrics[task][method] = {name:float(result[name]) for name in VATEX_METRICS}
            task_sources[task] = source
    provenance.update(references=references,task_sources=task_sources,helper_path=str(helper_path),helper_sha256=_sha(helper_path),
                      scoring_source_sha256=_sha(__file__),versions={name:importlib.metadata.version(name) for name in ("nltk","pywsd","wn","pandas","pyarrow","pycocoevalcap","pycocotools")})
    return metrics,per_input


def build_summary(runs, metrics, per_input, *, provenance, bootstrap_samples=5000, seed=20261104):
    methods = provenance["matrix"]["conditions"]; limit = provenance["matrix"]["limit"]
    comparisons, generation, score_rows, paired_rows, cache_rows, cache_summary = [], [], [], [], [], []
    for task in TASKS:
        for method in methods:
            rows = runs[task][method]
            if len(rows)!=limit: raise ValueError("Each task/method requires the complete declared prefix")
            generation.append(dict(task=task,method=method,n_inputs=limit,eos_count=sum(row["eos_reached"] for row in rows),
                reached_length_limit=sum(row["reached_length_limit"] for row in rows),capped_without_eos=sum(row["capped_without_eos"] for row in rows),
                generated_tokens_min=min(row["generated_tokens"] for row in rows),generated_tokens_max=max(row["generated_tokens"] for row in rows),
                generated_tokens_mean=statistics.fmean(row["generated_tokens"] for row in rows)))
            cache_rows.extend(dict(task=task,method=method,doc_id=row["doc_id"],cache_audit_sha256=row["cache_audit_sha256"],
                **{key:row[key] for key in ("prefill_tokens_total","uncompressed_tokens_total","actual_retained_ratio")}) for row in rows)
            cache_summary.append(dict(task=task,method=method,configured_budget_ratio=.1,
                actual_retained_ratio_min=min(row["actual_retained_ratio"] for row in rows),
                actual_retained_ratio_mean=statistics.fmean(row["actual_retained_ratio"] for row in rows),
                actual_retained_ratio_max=max(row["actual_retained_ratio"] for row in rows)))
            for metric in (("WUPS",) if task==TASKS[0] else VATEX_METRICS):
                value = metrics[task][method].get(metric)
                if type(value) not in (int,float) or not math.isfinite(value):
                    raise ValueError("Require every original task metric, including local METEOR")
            if task==TASKS[0]:
                values = per_input[task][method]
                if (len(values)!=limit or any(type(row.get(key)) not in (int,float) or not math.isfinite(row[key]) or not 0<=row[key]<=100 for row in values for key in ("WUPS","WUPS_0_9"))
                        or abs(statistics.fmean(row["WUPS"] for row in values)-metrics[task][method]["WUPS"])>1e-10):
                    raise ValueError("Per-input WUPS must reproduce the native aggregate in 0–100 points")
                score_rows.extend(dict(task=task,method=method,doc_id=row["doc_id"],source_video_id=row["source_video_id"],**score,
                    prediction_sha256=hashlib.sha256(row["text"].encode()).hexdigest()) for row,score in zip(rows,values))
        for metric in (("WUPS",) if task==TASKS[0] else VATEX_METRICS):
            first,second = [metrics[task][method][metric] for method in methods]
            interval = paired_interval(runs[task][methods[0]],*[ [row["WUPS"] for row in per_input[task][method]] for method in methods],samples=bootstrap_samples,seed=seed) if task==TASKS[0] else None
            comparisons.append(dict(task=task,metric=metric,reference_method=methods[0],candidate_method=methods[1],reference=first,candidate=second,
                difference=second-first,ci95=interval["ci95"] if interval else None,n_inputs=limit,n_source_videos=interval["n_source_videos"] if interval else None))
        if task==TASKS[0]:
            paired_rows.extend(dict(doc_id=i,source_video_id=runs[task][methods[0]][i]["source_video_id"],reference=per_input[task][methods[0]][i]["WUPS"],
                candidate=per_input[task][methods[1]][i]["WUPS"],difference=per_input[task][methods[1]][i]["WUPS"]-per_input[task][methods[0]][i]["WUPS"]) for i in range(limit))
    return dict(schema_version=1,status="complete",conditions=methods,tasks=list(TASKS),limit=limit,n_generations=4*limit,comparisons=comparisons,
        metrics=metrics,generation=generation,cache_budget_summary=cache_summary,bootstrap=dict(samples=bootstrap_samples,seed=seed,unit="source_video_id",task=TASKS[0]),provenance=provenance,
        interpretation="Differences are second matrix condition minus first; signed differences are retained. Configured budgets and actual paired input bytes are checked. Each method retains its native cache/compression semantics; cache audits are recorded without assuming identical actual cache lengths across algorithms.",
        metric_scope="Native NextQA WUPS@0 in 0–100 points; VATEX Bleu_4/METEOR/ROUGE_L/CIDEr in original raw corpus units with the same selected reference corpus/DF. No composite, per-sentence CIDEr substitute or VATEX bootstrap interval.",
        EOS_scope="At-cap includes EOS exactly at the cap; capped_without_eos counts only capped outputs lacking EOS.",scores_table=score_rows,paired_table=paired_rows,cache_table=cache_rows)


def write_report(summary, output, *, plots=True):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".video-report-",dir=output.parent) as temporary:
        target=Path(temporary)
        def table(name,rows,fields):
            with (target/name).open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
        comparison=[dict(row,ci95_low=(row["ci95"] or [None,None])[0],ci95_high=(row["ci95"] or [None,None])[1]) for row in summary["comparisons"]]
        table("comparison.csv",comparison,["task","metric","reference_method","candidate_method","reference","candidate","difference","ci95_low","ci95_high","n_inputs","n_source_videos"])
        table("nextqa_scores.csv",summary["scores_table"],["task","method","doc_id","source_video_id","WUPS","WUPS_0_9","prediction_sha256"])
        table("nextqa_paired_scores.csv",summary["paired_table"],["doc_id","source_video_id","reference","candidate","difference"])
        table("generation_summary.csv",summary["generation"],["task","method","n_inputs","eos_count","reached_length_limit","capped_without_eos","generated_tokens_min","generated_tokens_mean","generated_tokens_max"])
        table("cache_audit.csv",summary["cache_table"],["task","method","doc_id","cache_audit_sha256","prefill_tokens_total","uncompressed_tokens_total","actual_retained_ratio"])
        table("cache_budget_summary.csv",summary["cache_budget_summary"],["task","method","configured_budget_ratio","actual_retained_ratio_min","actual_retained_ratio_mean","actual_retained_ratio_max"])
        files=[dict(task=task,method=method,**row) for task,methods in summary["provenance"]["runs"].items() for method,run in methods.items() for row in run["files"]]
        table("sample_files.csv",files,["task","method","doc_id","path","sha256"])
        compact={key:value for key,value in summary.items() if not key.endswith("_table")}
        (target/"summary.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        lines=["# Video benchmark comparison","",f"Complete: {summary['limit']} inputs per task/method, {summary['n_generations']} generations.","",summary["interpretation"],"",summary["metric_scope"],"",
            "| Task | Metric | First condition | Second condition | Second − first | Paired 95% CI |","|---|---|---:|---:|---:|---|"]
        for row in comparison:
            interval="—" if row["ci95"] is None else f"[{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}]"
            lines.append(f"| {row['task']} | {row['metric']} | {row['reference']:.6f} | {row['candidate']:.6f} | {row['difference']:.6f} | {interval} |")
        lines += ["",f"First: `{summary['conditions'][0]}`. Second: `{summary['conditions'][1]}`.","",summary["EOS_scope"],"", "NextQA intervals resample source videos with all member questions; these fixed-prefix comparisons do not establish general superiority."]
        (target/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        if plots:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,len(comparison),figsize=(15,3.7),squeeze=False)
            labels=[method.replace("_kvrepack","\n+KVRepack") for method in summary["conditions"]]
            for axis,row in zip(axes[0],comparison):
                axis.scatter([0,1],[row["reference"],row["candidate"]],color=["#377EB8","#E68A2E"])
                axis.set(xticks=[0,1],xticklabels=labels,title=("NExTQA" if row["task"]==TASKS[0] else "VATEX")+"\n"+row["metric"],ylabel="Original task units")
                axis.spines[["top","right"]].set_visible(False)
            fig.suptitle(f"Qwen2.5-VL-7B: first {summary['limit']} inputs per task; configured KV budget10%",fontsize=11)
            fig.text(.5,.015,"Raw task metric scales; no composite. NextQA paired source-video confidence intervals are in comparison.csv.",ha="center",fontsize=8)
            fig.subplots_adjust(left=.05,right=.985,top=.75,bottom=.25,wspace=.5)
            for suffix in ("png","pdf","svg"):fig.savefig(target/f"comparison.{suffix}",dpi=300,bbox_inches="tight")
            plt.close(fig)
        target.rename(output)


def main(argv=None):
    runtime=str(Path(sys.prefix)/"lib")
    if argv is None and (Path(runtime)/"libstdc++.so.6").is_file() and os.environ.get("LD_LIBRARY_PATH","").split(os.pathsep)[0]!=runtime:
        os.execve(sys.executable,[sys.executable,"-m","kvrepack.benchmarks.video_comparison_report",*sys.argv[1:]],dict(os.environ,LD_LIBRARY_PATH=runtime+os.pathsep+os.environ.get("LD_LIBRARY_PATH","")))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",required=True,type=Path);parser.add_argument("--hf-home",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--limit",type=int,choices=(1,200),default=200);parser.add_argument("--bootstrap-samples",type=int,default=5000);parser.add_argument("--seed",type=int,default=20261104)
    parser.add_argument("--nltk-data",type=Path,default=Path(os.environ.get("NLTK_DATA",str(Path.home()/"nltk_data"))))
    java=Path(sys.prefix)/"lib/jvm";parser.add_argument("--java-home",type=Path,default=java if (java/"bin/java").is_file() else None)
    parser.add_argument("--no-plots",action="store_true");args=parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError("Refusing report overwrite or duplicate scoring")
    began=time.monotonic();runs,provenance=load_matrix(args.run_root,limit=args.limit)
    metrics,per_input=score_runs(runs,provenance,hf_home=args.hf_home,repo_root=Path(__file__).resolve().parents[3],nltk_data=args.nltk_data,java_home=args.java_home)
    summary=build_summary(runs,metrics,per_input,provenance=provenance,bootstrap_samples=args.bootstrap_samples,seed=args.seed)
    summary["scoring_seconds"]=time.monotonic()-began;write_report(summary,args.output,plots=not args.no_plots)
    print(json.dumps(dict(status=summary["status"],conditions=summary["conditions"],limit=args.limit,comparisons=summary["comparisons"],output=str(args.output))))
    return 0


if __name__=="__main__":raise SystemExit(main())

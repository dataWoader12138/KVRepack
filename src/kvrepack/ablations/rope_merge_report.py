"""Offline task scores for paired raw-reapply versus post-RoPE key merging.

One invocation scores one method/task, exactly its completed first1 or first200
inputs. Original task functions/scorers are reused. There is no model, GPU,
generation, network request, parameter selection, or historical-report rewrite.
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
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

from .head_selection_report import (
    VATEX_METRICS, _canonical, _helper, _nextqa_namespace, _sha,
    _vatex_score, offline_resources,
)


ARMS = ("raw_reapply", "post_rope")
TASKS = ("nextqa_oe_test", "vatex_test")
METHODS = ("snapkv_kvrepack", "hybridkv_kvrepack")


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_run(root, *, limit=200):
    """Check common inputs/references/config; retain the adapter's merge audit.

    The adapter owns the detailed head/group/budget and V-tolerance check. The
    full downstream cache need not have identical V or retrieval indices after
    different K choices; this scorer does not impose that invalid constraint.
    """
    if type(limit) is not int or limit not in (1, 200):
        raise ValueError("Require the explicit smoke1 or original-prefix200 protocol")
    root = Path(root)
    contract = json.loads((root/"run_contract.json").read_text())
    state = json.loads((root/"run_state.json").read_text())
    task, method = contract.get("task"), contract.get("method")
    if (task not in TASKS or method not in METHODS or contract.get("target_docs") != limit
            or contract.get("target_generations") != 2*limit or state.get("status") != "complete"
            or state.get("target_docs") != limit or state.get("target_generations") != 2*limit
            or state.get("completed_docs") != limit or state.get("completed_generations") != 2*limit):
        raise ValueError("Require a completed single-method/task run of the declared prefix")
    paths = sorted((root/"samples").glob("*.json"))
    if [path.name for path in paths] != [f"{doc:06d}.json" for doc in range(limit)]:
        raise ValueError("Missing, duplicate or out-of-prefix sample commits")
    identity = _json_sha(contract)
    rows, files, audits, common_config = {arm: [] for arm in ARMS}, [], [], None
    for doc_id, path in enumerate(paths):
        payload = path.read_bytes(); bundle = json.loads(payload)
        if (bundle.get("schema_version") != 1 or bundle.get("status") != "completed"
                or bundle.get("task") != task or bundle.get("method") != method or bundle.get("doc_id") != doc_id
                or bundle.get("contract_sha256") != identity or bundle.get("fairness", {}).get("status") != "passed"
                or not isinstance(bundle.get("records"), dict) or set(bundle["records"]) != set(ARMS)):
            raise ValueError(f"Incomplete/mismatched sample or failed merge fairness: {doc_id}")
        metadata, fingerprints = bundle.get("reference_metadata"), bundle.get("input_fingerprints")
        if (not isinstance(metadata, dict) or metadata.get("task") != task or metadata.get("doc_id") != doc_id
                or not isinstance(fingerprints, dict) or not fingerprints.get("input_hashes")):
            raise ValueError("Missing original reference metadata or actual input fingerprints")
        for arm in ARMS:
            row = bundle["records"][arm]
            config = row.get("config")
            if (row.get("status") != "completed" or row.get("task") != task or row.get("method") != method
                    or row.get("doc_id") != doc_id or row.get("reference_metadata") != metadata
                    or not isinstance(config, dict) or config.get("variant") != arm
                    or any(row.get(key) != value for key, value in fingerprints.items())):
                raise ValueError("Two merge arms differ in sample identity/input/config labels")
            clean_config = {key: value for key, value in config.items() if key != "variant"}
            if common_config is None: common_config = clean_config
            if clean_config != common_config:
                raise ValueError("Parameters/model/generation changed beyond config.variant")
            text = row.get("text")
            if not isinstance(text, str) or row.get("prediction") != text:
                raise ValueError("Missing or ambiguous original generated prediction")
            ids, cap, eos_id = row.get("generated_ids"), row.get("max_new_tokens", config.get("max_new_tokens")), row.get("eos_token_id")
            if (not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids)
                    or type(cap) is not int or cap < 1 or len(ids) > cap or row.get("generated_tokens") != len(ids)
                    or row.get("source_video_id") != metadata.get("source_video_id")):
                raise ValueError("Generated token length/cap/source identity is invalid")
            eos = ids[-1] in eos_id if isinstance(eos_id, list) else ids[-1] == eos_id
            if row.get("eos_reached") is not eos:
                raise ValueError("EOS metadata disagrees with actual generated tokens")
            rows[arm].append(dict(task=task, method=method, doc_id=doc_id, text=text,
                                  source_video_id=row["source_video_id"], reference_metadata=metadata,
                                  generated_tokens=len(ids), max_new_tokens=cap, eos_reached=eos,
                                  reached_length_limit=len(ids) == cap, capped_without_eos=len(ids) == cap and not eos))
        if path.read_bytes() != payload: raise ValueError("Committed result changed while reading")
        files.append(dict(doc_id=doc_id, path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest()))
        audits.append(dict(doc_id=doc_id, fairness=bundle["fairness"]))
    return rows, dict(contract=contract, contract_sha256=identity, files=files, fairness=audits, config=common_config,
                      limit=limit, task=task, method=method)


def paired_interval(rows, values, *, samples=5000, seed=20261104):
    import numpy as np
    if type(samples) is not int or samples < 1: raise ValueError("bootstrap samples must be positive")
    difference = np.asarray(values[ARMS[0]], dtype=np.float64)-np.asarray(values[ARMS[1]], dtype=np.float64)
    sources = defaultdict(list)
    for index, row in enumerate(rows[ARMS[0]]): sources[row["source_video_id"]].append(index)
    result = dict(difference=float(difference.mean()), ci95=None, n_inputs=len(difference), n_source_videos=len(sources))
    if len(sources) < 2: return result
    counts = np.asarray([len(indices) for _, indices in sorted(sources.items())])
    totals = np.asarray([difference[indices].sum() for _, indices in sorted(sources.items())])
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, samples, 256):
        indices = rng.integers(0, len(sources), size=(min(256, samples-start), len(sources)))
        draws.extend((totals[indices].sum(axis=1)/counts[indices].sum(axis=1)).tolist())
    result["ci95"] = np.quantile(draws, [.025, .975]).tolist()
    return result


def build_summary(rows, metrics, *, provenance, per_input=None, bootstrap_samples=5000, seed=20261104):
    task, method, limit = provenance["task"], provenance["method"], provenance["limit"]
    wanted = ("WUPS",) if task == "nextqa_oe_test" else VATEX_METRICS
    comparison, generation = {}, {}
    for metric in wanted:
        scores = {arm: metrics[arm]["metrics"].get(metric) for arm in ARMS}
        for arm, value in scores.items():
            if value is None and metric not in metrics[arm].get("missing_metrics", {}):
                raise ValueError("Metric missing without an explicit local-scorer reason")
            if value is None and metric != "METEOR": raise ValueError("Required task metric unavailable")
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                raise ValueError("Nonfinite task score")
        comparison[metric] = dict(**scores, raw_minus_post=scores[ARMS[0]]-scores[ARMS[1]] if all(value is not None for value in scores.values()) else None, ci95=None)
    if task == "nextqa_oe_test":
        if per_input is None: raise ValueError("NextQA requires original per-input WUPS")
        for arm in ARMS:
            if (len(per_input[arm]) != limit or any(not math.isfinite(value) or not 0 <= value <= 100 for value in per_input[arm])
                    or abs(statistics.fmean(per_input[arm])-metrics[arm]["metrics"]["WUPS"]) > 1e-10):
                raise ValueError("Per-input WUPS must reproduce the original aggregate")
        interval = paired_interval(rows, per_input, samples=bootstrap_samples, seed=seed)
        comparison["WUPS"]["ci95"] = interval["ci95"]
    else: interval = None
    for arm in ARMS:
        group = rows[arm]
        generation[arm] = dict(n_inputs=len(group), eos_count=sum(row["eos_reached"] for row in group),
                               reached_length_limit=sum(row["reached_length_limit"] for row in group),
                               capped_without_eos=sum(row["capped_without_eos"] for row in group),
                               generated_tokens_min=min(row["generated_tokens"] for row in group),
                               generated_tokens_max=max(row["generated_tokens"] for row in group),
                               generated_tokens_mean=statistics.fmean(row["generated_tokens"] for row in group))
    return dict(schema_version=1, status="complete" if all(data["raw_minus_post"] is not None for data in comparison.values()) else "partial_metrics",
                method=method, task=task, limit=limit, n_generations=2*limit, comparisons=comparison,
                generation=generation, paired_WUPS=interval, metrics=metrics, provenance=provenance,
                bootstrap=dict(samples=bootstrap_samples, seed=seed, unit="source_video_id", pairing="same source-video draws for both arms") if interval else None,
                interpretation="Paired key merge-space ablation; raw-reapply minus direct post-RoPE. Inputs, selected heads/groups, parameters and merge budgets are controlled by the recorded adapter fairness gate. Downstream V/cache/retrieval can change after different K choices.",
                metric_scope="NextQA original WUPS@0 uses 0-100 points. VATEX uses original raw corpus units and the same selected reference corpus/DF; no per-row CIDEr substitute, CIDEr CI or composite score.",
                EOS_scope="reached_length_limit counts any output at max_new_tokens, including EOS exactly at the cap; capped_without_eos counts only capped outputs without EOS",
                per_input_WUPS=per_input)


def write_report(summary, output, *, plots=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    def table(name, rows, fields):
        with (output/name).open("w", newline="", encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    rows=[]
    for metric,data in summary["comparisons"].items():
        rows.append(dict(metric=metric,**data,ci95_low=(data["ci95"] or [None,None])[0],ci95_high=(data["ci95"] or [None,None])[1]))
    table("task_scores.csv",rows,["metric",*ARMS,"raw_minus_post","ci95_low","ci95_high"])
    table("generation_summary.csv",[dict(variant=arm,**data) for arm,data in summary["generation"].items()],
          ["variant","n_inputs","eos_count","reached_length_limit","capped_without_eos","generated_tokens_min","generated_tokens_mean","generated_tokens_max"])
    if summary["per_input_WUPS"] is not None:
        table("paired_WUPS.csv",[dict(doc_id=i,raw_reapply=summary["per_input_WUPS"][ARMS[0]][i],post_rope=summary["per_input_WUPS"][ARMS[1]][i],
                                      raw_minus_post=summary["per_input_WUPS"][ARMS[0]][i]-summary["per_input_WUPS"][ARMS[1]][i]) for i in range(summary["limit"])],
              ["doc_id",*ARMS,"raw_minus_post"])
    lines=["# Key merge-space ablation","",f"Status: **{summary['status']}**; {summary['method']} / {summary['task']}; first {summary['limit']} inputs.","",summary["interpretation"],"",summary["metric_scope"],"",
           "| Metric | Raw + reapply RoPE | Direct post-RoPE | Raw − post (95% paired CI for WUPS) |","|---|---:|---:|---|"]
    fmt=lambda value:"missing" if value is None else f"{value:.6f}"
    for metric,data in summary["comparisons"].items():
        interval=f" [{data['ci95'][0]:.6f}, {data['ci95'][1]:.6f}]" if data["ci95"] is not None else ""
        lines.append(f"| {metric} | {fmt(data[ARMS[0]])} | {fmt(data[ARMS[1]])} | {fmt(data['raw_minus_post'])}{interval} |")
    lines += ["",summary["EOS_scope"],"","| Arm | EOS | At cap | Capped without EOS | Mean generated tokens |","|---|---:|---:|---:|---:|"]
    for arm,data in summary["generation"].items():lines.append(f"| {arm} | {data['eos_count']} | {data['reached_length_limit']} | {data['capped_without_eos']} | {data['generated_tokens_mean']:.2f} |")
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(rows),figsize=(max(4,3*len(rows)),3.4),squeeze=False)
        for axis,row in zip(axes[0],rows):
            for i,arm in enumerate(ARMS):
                if row[arm] is not None:axis.plot(i,row[arm],"o",color=("#377EB8","#E68A2E")[i])
            axis.set(xticks=[0,1],xticklabels=["Raw + RoPE","Post-RoPE"],title=row["metric"],ylabel="Original task score")
            axis.spines[["top","right"]].set_visible(False)
        fig.suptitle(f"{summary['method']} / {summary['task']}: n={summary['limit']}",fontsize=10)
        fig.text(.5,.02,"Paired inputs/groups/budgets; source-video WUPS intervals in the table. VATEX scores use raw corpus units.",ha="center",fontsize=7)
        fig.subplots_adjust(bottom=.23,top=.8,wspace=.5)
        for extension in ("png","pdf","svg"):fig.savefig(output/f"task_scores.{extension}",dpi=300,bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    runtime=str(Path(sys.prefix)/"lib")
    if argv is None and (Path(runtime)/"libstdc++.so.6").is_file() and os.environ.get("LD_LIBRARY_PATH","").split(os.pathsep)[0]!=runtime:
        os.execve(sys.executable,[sys.executable,"-m","kvrepack.ablations.rope_merge_report",*sys.argv[1:]],dict(os.environ,LD_LIBRARY_PATH=runtime+os.pathsep+os.environ.get("LD_LIBRARY_PATH","")))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True);parser.add_argument("--hf-home",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--limit",type=int,choices=(1,200),default=200)
    parser.add_argument("--bootstrap-samples",type=int,default=5000);parser.add_argument("--seed",type=int,default=20261104)
    parser.add_argument("--nltk-data",type=Path,default=Path(os.environ.get("NLTK_DATA",str(Path.home()/"nltk_data"))))
    java=Path(sys.prefix)/"lib/jvm";parser.add_argument("--java-home",type=Path,default=java if (java/"bin/java").is_file() else None)
    parser.add_argument("--no-plots",action="store_true");args=parser.parse_args(argv)
    if args.output.exists():raise ValueError("Output already exists; refusing report overwrite")
    began=time.monotonic();rows,provenance=load_run(args.run_root,limit=args.limit)
    repo=Path(__file__).resolve().parents[3];helper,helper_path=_helper(repo)
    task=provenance["task"]
    from ..observations.inputs import TASKS as DATA_TASKS
    _,_,cache,filename=DATA_TASKS[task];matches=list((args.hf_home/"datasets"/cache).glob("**/"+filename))
    if len(matches)!=1:raise ValueError("Require one cached original task Arrow")
    with offline_resources(nltk_data=args.nltk_data,java_home=args.java_home),tempfile.TemporaryDirectory(prefix="rope-score-") as temporary:
        documents,reference=helper.reference_rows(SimpleNamespace(reference_arrow=matches[0],task=task,expected_count=args.limit),rows[ARMS[0]])
        if provenance["contract"].get("arrow_sha256")!=reference["sha256"]:raise ValueError("Scoring Arrow differs from the frozen generation source")
        for i,doc in enumerate(documents):
            digest=hashlib.sha256(_canonical(doc).encode()).hexdigest()
            for arm in ARMS:
                meta=rows[arm][i]["reference_metadata"]
                video=str(doc["video"] if task=="nextqa_oe_test" else doc["videoID"])
                if meta.get("arrow_sha256")!=reference["sha256"] or meta.get("original_row_sha256")!=digest or meta.get("video_id")!=video:
                    raise ValueError("Scoring original row/video differs from the actual model input")
        metrics,individual={},None
        if task=="nextqa_oe_test":
            namespace,source=_nextqa_namespace(helper,repo);cache_results={};individual={}
            for arm in ARMS:
                results=[]
                for row,doc in zip(rows[arm],documents):
                    key=(row["text"],_canonical({name:doc[name] for name in ("answer","type","additional_ref_answer")}))
                    if key not in cache_results:cache_results[key]=namespace["nextqa_process_results"](doc,[row["text"]])["WUPS"]
                    results.append(cache_results[key])
                metrics[arm]=dict(metrics={"WUPS":float(namespace["nextqa_aggregate_results"](results))},missing_metrics={})
                individual[arm]=[100.*result["0"] for result in results]
        else:
            path=repo/"vendor/lmms-eval/lmms_eval/tasks/vatex/utils.py";nodes,source=helper.function_spec(path,helper.VATEX_FUNCTIONS)
            for arm in ARMS:metrics[arm]=_vatex_score(helper,path,nodes,rows[arm],documents,temporary,repo)
    provenance.update(reference=reference,task_scorer=source,helper_path=str(helper_path),helper_sha256=_sha(helper_path),scoring_source_sha256=_sha(__file__))
    summary=build_summary(rows,metrics,provenance=provenance,per_input=individual,bootstrap_samples=args.bootstrap_samples,seed=args.seed)
    summary["scoring_seconds"]=time.monotonic()-began;write_report(summary,args.output,plots=not args.no_plots)
    print(json.dumps(dict(status=summary["status"],method=summary["method"],task=task,limit=args.limit,comparisons=summary["comparisons"],output=str(args.output))))
    return 0


if __name__=="__main__":raise SystemExit(main())

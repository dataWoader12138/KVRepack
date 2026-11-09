"""Offline native task scores for one completed original video prefix.

This standalone report requires one method/task only. It never resumes an
experiment, loads a model/GPU, downloads, or constructs comparison conditions.
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
import tempfile
import time
from types import SimpleNamespace

from ..ablations.head_selection_report import (
    VATEX_METRICS, _canonical, _helper, _nextqa_namespace, _sha, offline_resources,
)
from .run_video_comparison import (TASKS, STRICT_LOOKM_POLICY, LEGACY_LOOKM_POLICY,
    json_sha, normalize_budget_policy, validate_budget_audit, validate_sample)


def load_case(root, *, limit=200):
    root=Path(root)
    contract_path=root/"run_contract.json";state_path=root/"run_state.json"
    contract=json.loads(contract_path.read_text());state=json.loads(state_path.read_text())
    method,task=contract.get("method"),contract.get("task")
    if (type(limit) is not int or limit not in (1,200) or contract.get("schema_version")!=1 or task not in TASKS
            or contract.get("target_docs")!=limit or contract.get("target_generations")!=limit
            or state.get("status")!="complete" or state.get("target_docs")!=limit
            or state.get("completed_docs")!=limit or state.get("target_generations")!=limit
            or state.get("completed_generations")!=limit):
        raise ValueError("Require one complete original-prefix run: explicit smoke1 or first200")
    policy=normalize_budget_policy(method,contract.get("budget_policy"))
    identity=json_sha(contract);paths=sorted((root/"samples").glob("*.json"))
    if [path.name for path in paths]!=[f"{doc:06d}.json" for doc in range(limit)]:
        raise ValueError("Missing or out-of-prefix committed samples")
    rows,files,config=[],[],None
    for doc_id,path in enumerate(paths):
        payload=path.read_bytes();row=json.loads(payload)
        validate_sample(row,doc_id=doc_id,contract_sha256=identity,method=method,task=task,budget_policy=policy)
        if config is None:config=row["config"]
        if row["config"]!=config:raise ValueError("Method/model/generation configuration changed across inputs")
        if row["config"]["max_new_tokens"]!=contract.get("max_new_tokens"):
            raise ValueError("Actual generation cap differs from frozen contract")
        cache=row["cache_audit"];length=len(row["generated_ids"]);cap=contract["max_new_tokens"]
        policy_status=validate_budget_audit(cache,method=method,budget_policy=policy)
        rows.append(dict(task=task,method=method,doc_id=doc_id,text=row["text"],source_video_id=row["source_video_id"],
            reference_metadata=row["reference_metadata"],generated_tokens=length,eos_reached=row["eos_reached"],
            reached_length_limit=length==cap,capped_without_eos=length==cap and not row["eos_reached"],
            prefill_tokens_total=cache["prefill_tokens_total"],uncompressed_tokens_total=cache["uncompressed_tokens_total"],
            actual_retained_ratio=cache["actual_retained_ratio"],cache_audit_sha256=hashlib.sha256(_canonical(cache).encode()).hexdigest()))
        rows[-1].update(policy_status)
        if path.read_bytes()!=payload:raise ValueError("Committed sample changed while reading")
        files.append(dict(doc_id=doc_id,path=str(path.resolve()),sha256=hashlib.sha256(payload).hexdigest()))
    return rows,dict(method=method,task=task,limit=limit,budget_policy=policy,generation_contract=contract,config=config,contract_sha256=identity,
        contract_file_sha256=_sha(contract_path),state_file_sha256=_sha(state_path),sample_files=files,scoring_source_sha256=_sha(__file__))


def validate_references(rows,docs,reference,contract):
    if (len(docs)!=len(rows) or contract.get("arrow_sha256")!=reference["sha256"]
            or contract.get("dataset_rows")!=reference["split_rows"] or reference["doc_ids"]!=list(range(len(rows)))):
        raise ValueError("Original reference Arrow or prefix differs from generation")
    for row,doc in zip(rows,docs):
        meta=row["reference_metadata"];task=row["task"]
        video=str(doc["video"] if task=="nextqa_oe_test" else doc["videoID"])
        source=video if task=="nextqa_oe_test" else video.rsplit("_",2)[0]
        if (meta.get("arrow_sha256")!=reference["sha256"] or meta.get("original_row_sha256")!=hashlib.sha256(_canonical(doc).encode()).hexdigest()
                or meta.get("video_id")!=video or meta.get("source_video_id")!=source or row["source_video_id"]!=source):
            raise ValueError("Original scoring row/video/hash differs from the actual model input")


def build_summary(rows,metrics,*,provenance,per_input=None):
    method,task=provenance["method"],provenance["task"]
    count=len(rows)
    if count not in (1,200):raise ValueError("Only the explicit smoke1 or original first200 report is supported")
    policy=normalize_budget_policy(method,provenance["generation_contract"].get("budget_policy"))
    if any(row.get("budget_policy")!=policy for row in rows):raise ValueError("Mixed legacy/strict budget policies in one report")
    strict=policy==STRICT_LOOKM_POLICY
    if strict and any(row.get("strict_budget_passed") is not True for row in rows):raise ValueError("Strict report lacks validated total-budget evidence")
    expected=("WUPS",) if task=="nextqa_oe_test" else VATEX_METRICS
    if set(metrics)!=set(expected) or any(type(value) not in (int,float) or not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Require all native task metrics, including local METEOR for VATEX")
    if task=="nextqa_oe_test":
        if (per_input is None or len(per_input)!=len(rows)
                or any(type(score.get(key)) not in (float,int) or not math.isfinite(score[key]) or not 0<=score[key]<=100 for score in per_input for key in ("WUPS","WUPS_0_9"))
                or abs(statistics.fmean(score["WUPS"] for score in per_input)-metrics["WUPS"])>1e-10):
            raise ValueError("Every per-input WUPS must reproduce the original native aggregate")
    generation=dict(n_inputs=len(rows),eos_count=sum(row["eos_reached"] for row in rows),
        reached_length_limit=sum(row["reached_length_limit"] for row in rows),capped_without_eos=sum(row["capped_without_eos"] for row in rows),
        generated_tokens_min=min(row["generated_tokens"] for row in rows),generated_tokens_max=max(row["generated_tokens"] for row in rows),
        generated_tokens_mean=statistics.fmean(row["generated_tokens"] for row in rows))
    cache=dict(budget_policy=policy,strict_budget_passed=True if strict else False if method=="lookm" else None,legacy_budget_result=policy==LEGACY_LOOKM_POLICY,
        configured_budget_ratio=provenance["generation_contract"]["budget_ratio"],
        actual_retained_ratio_min=min(row["actual_retained_ratio"] for row in rows),actual_retained_ratio_mean=statistics.fmean(row["actual_retained_ratio"] for row in rows),
        actual_retained_ratio_max=max(row["actual_retained_ratio"] for row in rows),
        scope="Logical active prefill/decode KV slots; excludes backing history/offload stores and is not total VRAM")
    inputs=[dict(doc_id=row["doc_id"],source_video_id=row["source_video_id"],prediction_sha256=hashlib.sha256(row["text"].encode()).hexdigest(),**score)
            for row,score in zip(rows,per_input)] if per_input is not None else []
    return dict(schema_version=1,status="complete",method=method,task=task,limit=len(rows),n_generations=len(rows),metrics=metrics,
        sample_scope="smoke_first1" if count==1 else "original_first200",formal_result=count==200,
        budget_policy=policy,strict_budget_passed=cache["strict_budget_passed"],legacy_budget_result=cache["legacy_budget_result"],
        budget_label="Strict total prefill including recent positions (v2)" if strict else "Legacy historical budget result; not a strict-v2 result" if method=="lookm" else "Native method-specific budget result",
        generation=generation,cache_budget=cache,provenance=provenance,scores_table=inputs,
        cache_table=[{key:row[key] for key in ("doc_id","source_video_id","prefill_tokens_total","uncompressed_tokens_total","actual_retained_ratio","cache_audit_sha256","budget_policy","strict_budget_passed","legacy_budget_result")} for row in rows],
        scope=f"Single completed method/task, exact original first{count} inputs; no matrix requirement, contrast, delta or inferred comparison result. "+("One-input smoke verifies the pipeline only; it is not a formal200 result or a basis for quality/bootstrap conclusions." if count==1 else "Fixed200-input prefix result."),
        metric_scope=f"Native NextQA WUPS@0 in 0–100 points; VATEX Bleu_4/METEOR/ROUGE_L/CIDEr in original corpus units using only the same selected{count} reference videos for CIDEr DF; no composite or per-sentence CIDEr substitute",
        EOS_scope="At-cap includes EOS exactly at max_new_tokens; capped_without_eos counts capped outputs lacking EOS")


def write_report(summary,output):
    output=Path(output)
    if output.exists():raise FileExistsError("Refusing report overwrite or duplicate scoring")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".single-video-report-",dir=output.parent) as temporary:
        target=Path(temporary)
        def table(name,rows,fields):
            with (target/name).open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
        table("task_scores.csv",[dict(metric=key,score=value) for key,value in summary["metrics"].items()],["metric","score"])
        if summary["scores_table"]:table("scores.csv",summary["scores_table"],["doc_id","source_video_id","WUPS","WUPS_0_9","prediction_sha256"])
        table("generation_summary.csv",[summary["generation"]],["n_inputs","eos_count","reached_length_limit","capped_without_eos","generated_tokens_min","generated_tokens_mean","generated_tokens_max"])
        table("cache_audit.csv",summary["cache_table"],["doc_id","source_video_id","prefill_tokens_total","uncompressed_tokens_total","actual_retained_ratio","cache_audit_sha256","budget_policy","strict_budget_passed","legacy_budget_result"])
        table("sample_files.csv",summary["provenance"]["sample_files"],["doc_id","path","sha256"])
        compact={key:value for key,value in summary.items() if not key.endswith("_table")}
        (target/"summary.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        lines=["# Single video benchmark result","",f"Complete: `{summary['method']}` / `{summary['task']}`, original first {summary['limit']} inputs.","",f"**{summary['budget_label']}**; policy `{summary['budget_policy']}`; strict_budget_passed={summary['strict_budget_passed']}.","",summary["scope"],"",summary["metric_scope"],"","| Metric | Native score |","|---|---:|"]
        for metric,value in summary["metrics"].items():lines.append(f"| {metric} | {value:.8f} |")
        gen=summary["generation"];cache=summary["cache_budget"]
        lines += ["",f"EOS: {gen['eos_count']}/{gen['n_inputs']}; at cap: {gen['reached_length_limit']}; capped without EOS: {gen['capped_without_eos']}.","",summary["EOS_scope"],"",cache["scope"],"",
            f"Configured budget ratio: {cache['configured_budget_ratio']}; actual active-KV retained fraction min/mean/max: {cache['actual_retained_ratio_min']:.8f}/{cache['actual_retained_ratio_mean']:.8f}/{cache['actual_retained_ratio_max']:.8f}.","","Scores describe this fixed prefix only."]
        (target/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        target.rename(output)


def main(argv=None):
    runtime=str(Path(sys.prefix)/"lib")
    if argv is None and (Path(runtime)/"libstdc++.so.6").is_file() and os.environ.get("LD_LIBRARY_PATH","").split(os.pathsep)[0]!=runtime:
        os.execve(sys.executable,[sys.executable,"-B","-m","kvrepack.benchmarks.single_video_report",*sys.argv[1:]],dict(os.environ,LD_LIBRARY_PATH=runtime+os.pathsep+os.environ.get("LD_LIBRARY_PATH","")))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",required=True,type=Path);parser.add_argument("--hf-home",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--limit",type=int,choices=(1,200),default=200)
    parser.add_argument("--nltk-data",type=Path,default=Path(os.environ.get("NLTK_DATA",str(Path.home()/"nltk_data"))))
    java=Path(sys.prefix)/"lib/jvm";parser.add_argument("--java-home",type=Path,default=java if (java/"bin/java").is_file() else None)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError("Report already exists; refusing overwrite")
    began=time.monotonic();rows,provenance=load_case(args.run_root,limit=args.limit)
    repo=Path(__file__).resolve().parents[3];helper,helper_path=_helper(repo);task=provenance["task"]
    from ..observations.inputs import TASKS as DATA_TASKS
    _,_,cache,filename=DATA_TASKS[task];matches=list((args.hf_home/"datasets"/cache).glob("**/"+filename))
    if len(matches)!=1:raise ValueError("Require one cached original task Arrow")
    per_input=None
    with offline_resources(nltk_data=args.nltk_data,java_home=args.java_home),tempfile.TemporaryDirectory(prefix="single-video-score-") as temporary:
        docs,reference=helper.reference_rows(SimpleNamespace(reference_arrow=matches[0],task=task,expected_count=args.limit),rows)
        validate_references(rows,docs,reference,provenance["generation_contract"])
        if task=="nextqa_oe_test":
            namespace,source=_nextqa_namespace(helper,repo)
            namespace["remove_stop"]=getattr(namespace["remove_stop"],"__wrapped__",namespace["remove_stop"])
            results=[namespace["nextqa_process_results"](doc,[row["text"]])["WUPS"] for row,doc in zip(rows,docs)]
            metrics={"WUPS":float(namespace["nextqa_aggregate_results"](results))}
            per_input=[dict(WUPS=100*value["0"],WUPS_0_9=100*value["0.9"]) for value in results]
            execution="Original six task-function ASTs, one uncached native WUPS call per input; no prediction result reuse"
        else:
            path=repo/"vendor/lmms-eval/lmms_eval/tasks/vatex/utils.py";nodes,source=helper.function_spec(path,helper.VATEX_FUNCTIONS)
            result,_=helper.vatex_score(path,nodes,rows,docs,Path(temporary));metrics={key:float(result[key]) for key in VATEX_METRICS}
            execution="Original task aggregation AST and all four real local scorers; same selected200-reference corpus/DF"
    packages=("nltk","pywsd","wn","pandas","pyarrow") if task=="nextqa_oe_test" else ("pycocoevalcap","pycocotools","pyarrow")
    provenance.update(reference=reference,task_scorer=source,helper_path=str(helper_path),helper_sha256=_sha(helper_path),execution=execution,
        dependency_versions={name:importlib.metadata.version(name) for name in packages})
    if task=="nextqa_oe_test":provenance["stopwords_sha256"]=_sha(repo/"vendor/lmms-eval/lmms_eval/tasks/nextqa/stopwords.csv")
    else:
        distribution=importlib.metadata.distribution("pycocoevalcap")
        provenance["local_caption_resources_sha256"]={name:_sha(distribution.locate_file(name)) for name in ("pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar","pycocoevalcap/meteor/meteor-1.5.jar","pycocoevalcap/meteor/data/paraphrase-en.gz")}
    summary=build_summary(rows,metrics,provenance=provenance,per_input=per_input);summary["scoring_seconds"]=time.monotonic()-began
    write_report(summary,args.output)
    print(json.dumps(dict(status=summary["status"],method=summary["method"],task=task,limit=args.limit,metrics=metrics,
        sample_scope=summary["sample_scope"],formal_result=summary["formal_result"],
        budget_policy=summary["budget_policy"],strict_budget_passed=summary["strict_budget_passed"],legacy_budget_result=summary["legacy_budget_result"],
        scoring_seconds=summary["scoring_seconds"],output=str(args.output))))
    return 0


if __name__=="__main__":raise SystemExit(main())

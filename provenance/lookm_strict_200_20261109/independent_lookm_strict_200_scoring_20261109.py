#!/usr/bin/env python3
"""Independently rescore ONLY strict LookM, NextQA/VATEX first200 each.

Run after BOTH single-method reports complete, never on a partial prefix:
  python independent_lookm_strict_200_scoring_20261109.py --run-root STRICT_ROOT \
      --hf-home /path/to/huggingface_cache

No comparison matrix, second method, inference/GPU, cache reuse, downloads,
source edits or old-result rewriting. Only a new independent audit JSON is written.
"""

from __future__ import annotations
import argparse
import ast
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import types
import urllib.request

TASKS = ("nextqa_oe_test", "vatex_test")
VATEX_METRICS = ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr")
NEXTQA_FUNCTIONS = ("remove_stop", "wup", "wups", "get_wups", "nextqa_process_results", "nextqa_aggregate_results")
STRICT_POLICY = "total_prefill_including_recent_v2"
TOLERANCE = 1e-10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def contract_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def snapshot(directory):
    return {str(path.relative_to(directory)): sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def functions(path, names):
    nodes = [node for node in ast.parse(path.read_text(), filename=str(path)).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names) or {node.name for node in nodes} != set(names):
        raise ValueError("Missing/duplicate original task functions")
    return nodes, dict(path=str(path.resolve()), source_sha256=sha(path),
        function_ast_sha256={node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() for node in nodes})


def execute(nodes, path, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def deny(*args, **kwargs): raise RuntimeError("Independent audit is offline; networking/downloads disabled")


def block_network(nltk_data, java_home):
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NLTK_DATA=str(nltk_data))
    if java_home:
        os.environ.update(JAVA_HOME=str(java_home), PATH=str(java_home/"bin")+os.pathsep+os.environ.get("PATH", ""))
    socket.socket.connect = deny; socket.socket.connect_ex = deny; socket.create_connection = deny
    urllib.request.urlopen = deny; urllib.request.urlretrieve = deny
    import nltk
    nltk.download = deny; nltk.downloader.Downloader.download = deny


def nextqa_namespace(repo):
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from pywsd.utils import lemmatize_sentence
    import pandas as pd
    path = repo/"vendor/lmms-eval/lmms_eval/tasks/nextqa/utils.py"
    nodes, source = functions(path, NEXTQA_FUNCTIONS)
    namespace = dict(wordnet=wordnet, word_tokenize=word_tokenize, lemmatize_sentence=lemmatize_sentence,
                     stopwords=set(pd.read_csv(path.parent/"stopwords.csv").squeeze()))
    execute(nodes, path, namespace)
    # Deliberately do not memoize remove_stop or per-prediction WUPS results.
    source["stopwords_sha256"] = sha(path.parent/"stopwords.csv")
    source["prediction_result_cache"] = False
    return namespace, source


def vatex_score(repo, records, docs, temporary):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb
    path = repo/"vendor/lmms-eval/lmms_eval/tasks/vatex/utils.py"
    nodes, source = functions(path, ("vatex_process_result", "vatex_aggregation_result"))
    original = ptb.PTBTokenizer.tokenize
    scope = dict(original.__globals__)
    scope["__file__"] = str(temporary/"ptbtokenizer.py")
    scope["STANFORD_CORENLP_3_4_1_JAR"] = str(Path(ptb.__file__).parent/ptb.STANFORD_CORENLP_3_4_1_JAR)
    relocated = types.FunctionType(original.__code__, scope, original.__name__, original.__defaults__)
    tokens = []
    class LocalPTB:
        def tokenize(self, captions):
            result = relocated(self, captions)
            tokens.append(dict(input_sha256=value_sha(captions), output_sha256=value_sha(result)))
            return result
    class QuietLogger:
        def info(self, *args, **kwargs): pass
    meteor_instances = []
    def real_meteor():
        if not meteor_instances: meteor_instances.append(Meteor())
        return meteor_instances[0]
    namespace = dict(Bleu=Bleu, Cider=Cider, Rouge=Rouge, Meteor=real_meteor, COCO=COCO, COCOEvalCap=COCOEvalCap,
        PTBTokenizer=LocalPTB, VATEX_METRICS=["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"],
        eval_logger=QuietLogger(), json=json, generate_submission_file=lambda name,args: str(temporary/name))
    execute(nodes, path, namespace)
    processed = [namespace["vatex_process_result"](doc, [row["text"]])["vatex_Bleu_4"] for row,doc in zip(records,docs)]
    if len({row["video_id"] for row in processed}) != len(processed): raise ValueError("Duplicate VATEX videos would overwrite corpus references")
    metrics = {}
    try:
        for metric in VATEX_METRICS:
            with contextlib.redirect_stdout(io.StringIO()): metrics[metric] = float(namespace["vatex_aggregation_result"](processed, metric))
            print(json.dumps(dict(stage="independent_vatex_metric", metric=metric, value=metrics[metric])), flush=True)
        source["meteor_command"] = meteor_instances[0].meteor_cmd
    finally:
        namespace.clear()
        if meteor_instances:
            scorer = meteor_instances.pop(); del scorer
    # Native CiderScorer builds DF only from this same selected reference set.
    # Compute its reference DF independently as a provenance fingerprint, too.
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    annotations = {}
    for doc in docs: annotations[doc["videoID"]] = [dict(caption=text) for text in doc["enCap"]]
    reference_tokens = LocalPTB().tokenize(annotations)
    df = CiderScorer()
    for video in reference_tokens: df += ("", reference_tokens[video])
    df.compute_doc_freq()
    source["CIDEr_reference_document_frequency"] = dict(n_videos=len(df.crefs), unique_reference_ngrams=len(df.document_frequency),
        sha256=value_sha([[list(ngram),frequency] for ngram,frequency in sorted(df.document_frequency.items())]))
    source["PTB_calls"] = tokens
    source["runtime_changes"] = "Original AST/scorer compute functions; only temporary submission/tokenizer file location and reuse of the real Meteor process across constructor calls. No WUPS, metric or tokenization result cache is reused from the report."
    return metrics, source


def table(path, key):
    with path.open(newline="") as stream: rows = list(csv.DictReader(stream))
    indexed = {key(row):row for row in rows}
    if len(indexed) != len(rows): raise ValueError("Duplicate CSV identity: "+str(path))
    return indexed


def strict_sample(row):
    """Independent integer accounting, including recent positions and decode."""
    cache=row['cache_audit'];config=row['config'];meta=row['reference_metadata']
    assert row['method']=='lookm'
    assert config['budget_policy']==cache['budget_policy']==row['audit']['budget_policy']==STRICT_POLICY
    assert cache['strict_budget_passed'] is True and cache['nominal_budget_ratio']==config['budget_ratio']==.1
    tokens=cache['input_tokens'];assert type(tokens) is int and tokens>=10
    assert tokens==row['input_tokens']==meta['input_tokens']
    assert row['input_fingerprints']['input_shapes']['input_ids']==[1,tokens]
    capacity=math.floor(tokens*.1);window=min(32,capacity)
    assert cache['total_prefill_capacity']==capacity
    layers=cache['layers'];finals=row['final_cache_shapes']
    assert cache['layers_count']==len(layers)==len(finals)==28 and cache['native_kv_heads']==4
    ids=row['generated_ids'];length=len(ids);cap=config['max_new_tokens']
    assert ids and all(type(token) is int and token>=0 for token in ids) and length==row['generated_tokens'] and length<=cap
    eos=ids[-1] in row['eos_token_id'] if isinstance(row['eos_token_id'],list) else ids[-1]==row['eos_token_id']
    assert row['eos_reached'] is eos and row['max_new_tokens']==cap
    for index,(layer,final) in enumerate(zip(layers,finals)):
        assert layer['layer']==final['layer']==index and layer['before_tokens_per_head']==tokens
        assert layer['head_lengths']==[capacity]*4 and layer['after_tokens_total']==4*capacity
        assert layer['total_prefill_capacity']==capacity and layer['configured_recent_window']==32
        assert layer['recent_window']==window and layer['selected_history_tokens']==capacity-window
        assert layer['key_shape']==layer['value_shape']==[1,4,capacity,128]
        assert final['key_shape']==final['value_shape']==[1,4,capacity+length-1,128]
    assert cache['prefill_tokens_total']==112*capacity and cache['uncompressed_tokens_total']==112*tokens
    assert cache['actual_retained_ratio']==capacity/tokens<=.1
    return dict(tokens=tokens,capacity=capacity,window=window,actual_retained_ratio=capacity/tokens,
                generated_tokens=length,eos=eos,at_cap=length==cap,capped_without_eos=length==cap and not eos)


def require_ready(root):
    """Both complete reports and all200 commits are required before any scoring."""
    controller=json.loads((root/'lookm_state.json').read_text())
    spec=controller['spec']
    if (controller['status']!='complete' or controller['completed_tasks']!=list(TASKS)
            or spec['method']!='lookm' or spec['tasks']!=list(TASKS) or spec['limit']!=200
            or spec['budget_policy']!=STRICT_POLICY or spec['budget_ratio']!=.1):
        raise ValueError('Require the completed ONLY LookM strict200 two-task controller')
    reports={}
    for task in TASKS:
        case=root/task
        state=json.loads((case/'run_state.json').read_text())
        summary=json.loads((case/'report/summary.json').read_text())
        if (state['status']!='complete' or state['completed_docs']!=200 or state['target_docs']!=200
                or state['completed_generations']!=200 or state['target_generations']!=200):
            raise ValueError('Require complete strict200 generation for '+task)
        if (summary['status']!='complete' or summary['method']!='lookm' or summary['task']!=task
                or summary['limit']!=200 or summary['n_generations']!=200 or summary.get('formal_result') is not True
                or summary.get('sample_scope')!='original_first200' or summary.get('budget_policy')!=STRICT_POLICY
                or summary.get('strict_budget_passed') is not True or summary.get('legacy_budget_result') is not False):
            raise ValueError('Reject smoke, legacy or incomplete single-case report: '+task)
        if [p.name for p in sorted((case/'samples').glob('*.json'))]!=[f'{i:06d}.json' for i in range(200)]:
            raise ValueError('Missing/out-of-prefix strict sample commits')
        reports[task]=summary
    return reports


def audit(args,result):
    reports=require_ready(args.run_root)
    result['checks']['both_formal_strict200_single_reports_complete']=True
    # Only now load local reference/scoring libraries and execute original tasks.
    block_network(args.nltk_data,args.java_home)
    import pyarrow as pa
    import pyarrow.ipc as ipc
    def check(name,value):
        result['checks'][name]=bool(value)
        if not value:raise ValueError('Independent check failed: '+name)
    def equal(name,actual,expected):
        error=abs(float(actual)-float(expected))
        if not math.isfinite(error):raise ValueError('Nonfinite verification difference: '+name)
        result['maximum_absolute_errors'][name]=max(error,result['maximum_absolute_errors'].get(name,0.))
        if error>TOLERANCE:raise ValueError(f'{name}: {actual} != {expected}')
    metrics={};tasks={};source_files={}
    next_namespace,next_source=nextqa_namespace(args.repo_root)
    for task in TASKS:
        case=args.run_root/task;summary=reports[task];provenance=summary['provenance']
        contract_path=case/'run_contract.json';contract=json.loads(contract_path.read_text());identity=contract_sha(contract)
        check('frozen_contract_'+task,contract==provenance['generation_contract'] and identity==provenance['contract_sha256']
              and sha(contract_path)==provenance['contract_file_sha256'] and contract['budget_policy']==STRICT_POLICY
              and contract['method']=='lookm' and contract['task']==task and contract['target_docs']==contract['target_generations']==200)
        check('frozen_state_'+task,sha(case/'run_state.json')==provenance['state_file_sha256'])
        for path,digest in contract['sources_sha256'].items():
            if sha(path)!=digest:raise ValueError('Frozen generation source changed: '+path)
            source_files[path]=digest
        check('single_report_source_SHA_'+task,sha(args.repo_root/'src/kvrepack/benchmarks/single_video_report.py')==provenance['scoring_source_sha256'])
        reference=provenance['reference'];arrow=Path(reference['path'])
        if not arrow.is_file():
            filename='n_ex_tqa-test.arrow' if task==TASKS[0] else 'vatex-test.arrow'
            matches=list((args.hf_home/'datasets').glob('**/'+filename))
            if len(matches)!=1:raise ValueError('Require one cached original Arrow')
            arrow=matches[0]
        arrow_sha=sha(arrow)
        with pa.memory_map(str(arrow),'r') as memory:raw=ipc.open_stream(memory).read_all()
        docs=raw.slice(0,200).to_pylist()
        check('original_Arrow_prefix_'+task,reference['sha256']==contract['arrow_sha256']==arrow_sha
              and reference['split_rows']==contract['dataset_rows']==raw.num_rows and reference['selected_rows']==200
              and reference['doc_ids']==list(range(200)) and reference['selected_reference_rows_sha256']==value_sha(docs))
        task_csv=table(case/'report/task_scores.csv',lambda row:row['metric'])
        cache_csv=table(case/'report/cache_audit.csv',lambda row:int(row['doc_id']))
        sample_csv=table(case/'report/sample_files.csv',lambda row:int(row['doc_id']))
        check('complete_cache_sample_CSV_'+task,set(cache_csv)==set(sample_csv)==set(range(200)))
        official_files={item['doc_id']:item['sha256'] for item in provenance['sample_files']}
        rows=[];stats=[];file_hashes={};native=[];input_scores=[]
        scores_csv=table(case/'report/scores.csv',lambda row:int(row['doc_id'])) if task==TASKS[0] else None
        if scores_csv is not None:check('NextQA_CSV200_complete',set(scores_csv)==set(range(200)))
        for i,doc in enumerate(docs):
            path=case/'samples'/f'{i:06d}.json';payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest();row=json.loads(payload)
            assert digest==official_files[i]==sample_csv[i]['sha256'];file_hashes[path.name]=digest
            assert row['schema_version']==1 and row['status']=='completed' and row['task']==task and row['doc_id']==i and row['contract_sha256']==identity
            meta=row['reference_metadata'];video=str(doc['video'] if task==TASKS[0] else doc['videoID']);source=video if task==TASKS[0] else video.rsplit('_',2)[0]
            assert meta['arrow_sha256']==arrow_sha and meta['original_row_sha256']==value_sha(doc)
            assert row['source_video_id']==meta['source_video_id']==source and row['video_id']==meta['video_id']==video
            assert row['text']==row['prediction'] and all(row[k]==v for k,v in row['input_fingerprints'].items())
            stat=strict_sample(row);stats.append(stat);cache=row['cache_audit'];csv_row=cache_csv[i]
            assert csv_row['cache_audit_sha256']==value_sha(cache) and csv_row['budget_policy']==STRICT_POLICY
            assert csv_row['strict_budget_passed']=='True' and csv_row['legacy_budget_result']=='False'
            for field in ('prefill_tokens_total','uncompressed_tokens_total','actual_retained_ratio'):equal('cache_CSV_'+field,cache[field],csv_row[field])
            rows.append(dict(task=task,doc_id=i,text=row['text'],source_video_id=source))
            if task==TASKS[0]:
                score=next_namespace['nextqa_process_results'](doc,[row['text']])['WUPS']
                native.append(score);input_scores.append(dict(doc_id=i,WUPS=100*score['0'],WUPS_0_9=100*score['0.9']))
                for field,key in (('WUPS','0'),('WUPS_0_9','0.9')):equal('NextQA_per_input_'+field,100*score[key],scores_csv[i][field])
                assert scores_csv[i]['prediction_sha256']==hashlib.sha256(row['text'].encode()).hexdigest() and scores_csv[i]['source_video_id']==source
            assert path.read_bytes()==payload
        if task==TASKS[0]:
            metric={'WUPS':float(next_namespace['nextqa_aggregate_results'](native))};native_source=next_source
            equal('NextQA_native_aggregate_vs_input_mean',metric['WUPS'],statistics.fmean(score['WUPS'] for score in input_scores))
            check('200_uncached_original_WUPS_calls',len(native)==200)
        else:
            with tempfile.TemporaryDirectory(prefix='independent-strict-lookm-vatex-') as temporary:
                metric,native_source=vatex_score(args.repo_root,rows,docs,Path(temporary))
            check('VATEX_same200_reference_DF',native_source['CIDEr_reference_document_frequency']['n_videos']==200)
        expected_source=provenance['task_scorer']
        check('native_task_source_AST_'+task,native_source['source_sha256']==expected_source['source_sha256']
              and native_source['function_ast_sha256']==expected_source['function_ast_sha256'])
        check('exact_native_metric_set_'+task,set(metric)==set(summary['metrics'])==set(task_csv))
        verification={}
        for name,value in metric.items():
            equal('native_metric_JSON_'+name,value,summary['metrics'][name]);equal('native_metric_CSV_'+name,value,task_csv[name]['score'])
            verification[name]=dict(independently_recomputed=value,report_JSON=summary['metrics'][name],report_CSV=float(task_csv[name]['score']),
                                    recomputed_minus_report=value-summary['metrics'][name])
        generation=dict(n_inputs=200,eos_count=sum(s['eos'] for s in stats),reached_length_limit=sum(s['at_cap'] for s in stats),capped_without_eos=sum(s['capped_without_eos'] for s in stats),
            generated_tokens_min=min(s['generated_tokens'] for s in stats),generated_tokens_max=max(s['generated_tokens'] for s in stats),generated_tokens_mean=statistics.fmean(s['generated_tokens'] for s in stats))
        generation_csv=next(iter(table(case/'report/generation_summary.csv',lambda row:row['n_inputs']).values()))
        for key,value in generation.items():equal('generation_summary_'+key,value,summary['generation'][key]);equal('generation_CSV_'+key,value,generation_csv[key])
        ratios=[s['actual_retained_ratio'] for s in stats]
        actual=dict(actual_retained_ratio_min=min(ratios),actual_retained_ratio_mean=statistics.fmean(ratios),actual_retained_ratio_max=max(ratios))
        for key,value in actual.items():equal('strict_cache_summary_'+key,value,summary['cache_budget'][key])
        check('strict_formal200_meta_and_ratio_'+task,summary['cache_budget']['budget_policy']==STRICT_POLICY
              and summary['cache_budget']['strict_budget_passed'] is True and summary['cache_budget']['legacy_budget_result'] is False and max(ratios)<=.1)
        check('dependency_versions_match_'+task,all(importlib.metadata.version(name)==version for name,version in provenance['dependency_versions'].items()))
        helper_path=Path(provenance['helper_path']);_,helper_source=functions(helper_path,('function_spec','execute_functions','reference_rows','nextqa_score','vatex_score'))
        check('versioned_helper_SHA_'+task,sha(helper_path)==provenance['helper_sha256'])
        metrics[task]=metric;tasks[task]=dict(metric_verification=verification,native_task_source=native_source,versioned_helper_source=helper_source,
            reference=dict(path=str(arrow),sha256=arrow_sha,selected_reference_rows_sha256=value_sha(docs),split_rows=raw.num_rows,selected_original_doc_ids=list(range(200))),
            sample_file_sha256=file_hashes,prediction_corpus_sha256=value_sha(rows),formal_result=summary['formal_result'],strict_budget_passed=summary['strict_budget_passed'],
            strict_budget_samples=len(stats),strict_budget_layers=28*len(stats),strict_budget_heads=112*len(stats),
            generation=generation,active_cache_ratio=actual,input_WUPS=input_scores if input_scores else None)
    nltk_files={str(p.relative_to(args.nltk_data)):sha(p) for p in sorted(args.nltk_data.rglob('*')) if p.is_file()}
    cap=importlib.metadata.distribution('pycocoevalcap')
    resources={name:sha(cap.locate_file(name)) for name in ('pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar','pycocoevalcap/meteor/meteor-1.5.jar','pycocoevalcap/meteor/data/paraphrase-en.gz',
        'pycocoevalcap/bleu/bleu.py','pycocoevalcap/cider/cider.py','pycocoevalcap/cider/cider_scorer.py','pycocoevalcap/rouge/rouge.py','pycocoevalcap/meteor/meteor.py')}
    java=args.java_home/'bin/java' if args.java_home else Path(__import__('shutil').which('java') or '')
    version=subprocess.run([str(java),'-version'],capture_output=True,text=True,check=True)
    check('CPU_only_no_torch_or_model','torch' not in sys.modules)
    result.update(metrics=metrics,tasks=tasks,method='lookm',n_generations=400,n_samples_per_task=200,formal_result=True,budget_policy=STRICT_POLICY,strict_budget_passed=True,
        formal_result_reports=sum(summary.get('formal_result') is True for summary in reports.values()),
        strict_budget_reports=sum(summary.get('strict_budget_passed') is True for summary in reports.values()),
        strict_budget_samples=sum(task['strict_budget_samples'] for task in tasks.values()),
        strict_budget_layers=sum(task['strict_budget_layers'] for task in tasks.values()),
        strict_budget_heads=sum(task['strict_budget_heads'] for task in tasks.values()),generation_source_files_sha256=source_files,
        local_resources=dict(nltk_data_files_sha256=nltk_files,nltk_data_tree_sha256=value_sha(nltk_files),caption_assets_and_sources_sha256=resources,
            java=dict(path=str(java),sha256=sha(java),version_output=(version.stdout+version.stderr).strip())),
        dependency_versions={name:importlib.metadata.version(name) for name in ('nltk','pywsd','wn','pandas','pyarrow','pycocoevalcap','pycocotools')},
        scope='ONLY strict LookM, NextQA/VATEX original first200 each. No second method, comparison matrix, bootstrap CI, composite metric, prediction-result reuse, inference or GPU.',
        limitations=['CIDEr DF belongs to these200VATEX reference videos, not single captions or another corpus.',
                     'The strict10% cap covers all prefill slots including recent/text positions; decoding appends subsequent tokens and is checked separately.',
                     'Active logical KV counts are not a total-VRAM measurement. Earlier legacy400results are neither reused nor relabelled by this audit.'])


def main(argv=None):
    runtime=str(Path(sys.prefix)/'lib')
    if argv is None and (Path(runtime)/'libstdc++.so.6').is_file() and os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep)[0]!=runtime:
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]],dict(os.environ,LD_LIBRARY_PATH=runtime+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True,type=Path);parser.add_argument('--hf-home',required=True,type=Path)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--output',type=Path)
    parser.add_argument('--nltk-data',type=Path,default=Path(os.environ.get('NLTK_DATA',str(Path.home()/'nltk_data'))))
    java=Path(sys.prefix)/'lib/jvm';parser.add_argument('--java-home',type=Path,default=java if (java/'bin/java').is_file() else None)
    args=parser.parse_args(argv)
    args.run_root=args.run_root.resolve();args.repo_root=args.repo_root.resolve();args.hf_home=args.hf_home.resolve()
    output=args.output or args.run_root/'independent_scoring_check.json'
    if output.exists():raise FileExistsError('Refusing to overwrite an existing independent audit')
    began=time.monotonic()
    before={task:snapshot(args.run_root/task/'report') for task in TASKS}
    result=dict(schema_version=1,status='failed',created_utc=datetime.now(timezone.utc).isoformat(),checks={},maximum_absolute_errors={},original_report_file_sha256=before)
    try:
        audit(args,result);result['status']='passed'
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        after={task:snapshot(args.run_root/task/'report') for task in TASKS}
        unchanged=before==after and all(before.values())
        result['checks']['original_single_report_files_unchanged']=unchanged
        if not unchanged:result['status']='failed'
        result['elapsed_seconds']=time.monotonic()-began
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],output=str(output),elapsed_seconds=result['elapsed_seconds'],metrics=result.get('metrics'),maximum_absolute_errors=result['maximum_absolute_errors'],error=result.get('error'))))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

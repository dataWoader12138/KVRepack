"""Small CPU report/schema fixtures; no models, downloads or task score reuse."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.benchmarks.video_comparison_report import (
    TASKS, VATEX_METRICS, _canonical, _json_sha, build_summary, load_matrix,
    paired_interval, write_report,
)
from kvrepack.benchmarks.run_video_comparison import LEGACY_LOOKM_POLICY,NATIVE_BUDGET_POLICY,STRICT_LOOKM_POLICY


def fixture(root, methods=("lookm","hybridkv_kvrepack")):
    root=Path(root);matrix=dict(schema_version=1,experiment="lookm_vs_kvrepack",conditions=list(methods),tasks=list(TASKS),limit=1,budget_ratio=.1,frames=64)
    (root/"matrix.json").write_text(json.dumps(matrix))
    for task in TASKS:
        for method in methods:
            policy=LEGACY_LOOKM_POLICY if method=="lookm" else NATIVE_BUDGET_POLICY
            directory=root/method/task;(directory/"samples").mkdir(parents=True)
            contract=dict(schema_version=1,method=method,task=task,target_docs=1,target_generations=1,frames=64,budget_ratio=.1,max_new_tokens=128 if task==TASKS[0] else 64,model_snapshot="qwen7b",repetition_penalty=1.05,budget_policy=policy)
            (directory/"run_contract.json").write_text(json.dumps(contract))
            (directory/"run_state.json").write_text(json.dumps(dict(status="complete",target_docs=1,target_generations=1,completed_docs=1,completed_generations=1)))
            meta=dict(task=task,doc_id=0,source_video_id="video",video_id="video",arrow_sha256="arrow",original_row_sha256="row")
            inputs=dict(input_hashes={"input_ids":"actual-bytes"},input_shapes={"input_ids":[1,100]},input_dtypes={"input_ids":"torch.int64"})
            lengths=10 if method==methods[0] else 12
            layers=[dict(layer=i,before_tokens_per_head=100,head_lengths=[lengths]*4,after_tokens_total=4*lengths) for i in range(28)]
            audit=dict(status="completed",method=method,task=task,doc_id=0,native_kv_heads=4,layers_count=28,layers=layers,nominal_budget_ratio=.1,prefill_tokens_total=112*lengths,uncompressed_tokens_total=11200,actual_retained_ratio=lengths/100,budget_policy=policy,strict_budget_passed=False if method=="lookm" else None)
            row=dict(schema_version=1,status="completed",method=method,task=task,doc_id=0,contract_sha256=_json_sha(contract),text="answer",prediction="answer",reference_metadata=meta,input_fingerprints=inputs,**inputs,source_video_id="video",config=dict(method=method,max_new_tokens=contract["max_new_tokens"],budget_policy=policy),generated_ids=[7,2],generated_tokens=2,eos_token_id=2,eos_reached=True,max_new_tokens=contract["max_new_tokens"],cache_audit=audit)
            (directory/"samples/000000.json").write_text(json.dumps(row))
    return matrix


def load(root):
    return load_matrix(root,limit=1,validator=lambda *args,**kwargs:None)


def summary(runs,provenance):
    first,second=provenance["matrix"]["conditions"]
    scores={TASKS[0]:{first:{"WUPS":60.},second:{"WUPS":40.}},TASKS[1]:{first:dict(zip(VATEX_METRICS,[.2,.3,.4,.5])),second:dict(zip(VATEX_METRICS,[.1,.3,.42,.41]))}}
    per={TASKS[0]:{first:[dict(WUPS=60.,WUPS_0_9=50.)],second:[dict(WUPS=40.,WUPS_0_9=10.)]},TASKS[1]:{}}
    return build_summary(runs,scores,per,provenance=provenance,bootstrap_samples=20)


class VideoComparisonReportTests(unittest.TestCase):
    def test_ordered_methods_negative_deltas_native_scales_and_different_actual_cache(self):
        for methods in (("lookm","hybridkv_kvrepack"),("lookm","lookm_kvrepack")):
            with self.subTest(methods=methods),tempfile.TemporaryDirectory() as directory:
                fixture(directory,methods);runs,provenance=load(directory);s=summary(runs,provenance)
                self.assertEqual(s["conditions"],list(methods));self.assertEqual(s["n_generations"],4)
                self.assertEqual(s["comparisons"][0]["difference"],-20.)
                self.assertIsNone(s["comparisons"][0]["ci95"])
                self.assertTrue(all(row["ci95"] is None for row in s["comparisons"][1:]))
                self.assertNotEqual(s["cache_budget_summary"][0]["actual_retained_ratio_mean"],s["cache_budget_summary"][1]["actual_retained_ratio_mean"])
                self.assertTrue(s["cache_budget_summary"][0]["legacy_budget_result"])
                self.assertFalse(s["cache_budget_summary"][0]["strict_budget_passed"])
                output=Path(directory)/"report";write_report(s,output,plots=False)
                for name in ("comparison.csv","nextqa_scores.csv","nextqa_paired_scores.csv","generation_summary.csv","cache_audit.csv","cache_budget_summary.csv","sample_files.csv","summary.json","RESULTS.md"):
                    self.assertGreater((output/name).stat().st_size,20)
                self.assertFalse(any("mean4" in key for values in s["metrics"][TASKS[1]].values() for key in values))
                with self.assertRaises(FileExistsError):write_report(s,output,plots=False)

    def test_source_cluster_bootstrap_retains_all_questions_and_signed_delta(self):
        rows=[dict(source_video_id="a"),dict(source_video_id="a"),dict(source_video_id="b")]
        result=paired_interval(rows,[20.,20.,20.],[10.,10.,30.],samples=5000)
        self.assertAlmostEqual(result["difference"],-10/3)
        self.assertEqual(result["n_source_videos"],2)
        self.assertEqual(result["ci95"],[-10.,10.])
        self.assertEqual(result,paired_interval(rows,[20.,20.,20.],[10.,10.,30.],samples=5000))

    def test_public_worker_validator_and_report_share_native_single_row_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory)
            for path in Path(directory).glob("*/*/samples/*.json"):
                row=json.loads(path.read_text());row["video_id"]=row["reference_metadata"]["video_id"]
                row["config"].update(task=row["task"],budget_ratio=.1,stream_contract_sha256=row["contract_sha256"])
                row["cache_audit"]["input_tokens"]=100
                row["final_cache_shapes"]=[]
                for layer in row["cache_audit"]["layers"]:
                    length=layer["head_lengths"][0];shape=[1,4,length,128]
                    layer.update(key_shape=shape,value_shape=shape)
                    final=[1,4,length+1,128]
                    row["final_cache_shapes"].append(dict(layer=layer["layer"],key_shape=final,value_shape=final))
                row["audit"]=deepcopy(row["cache_audit"])
                path.write_text(json.dumps(row))
            runs,provenance=load_matrix(directory,limit=1)
            self.assertEqual(sum(len(group) for methods in runs.values() for group in methods.values()),4)
            self.assertEqual(summary(runs,provenance)["status"],"complete")

    def test_incomplete_or_wrong_prefix_never_scores(self):
        for mode in ("state","missing","extra","contract","matrix"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                fixture(directory);root=Path(directory);case=root/"lookm"/TASKS[0]
                if mode=="state":
                    p=case/"run_state.json";v=json.loads(p.read_text());v["status"]="running";p.write_text(json.dumps(v))
                elif mode=="missing":(case/"samples/000000.json").unlink()
                elif mode=="extra":(case/"samples/000001.json").write_text('{}')
                elif mode=="contract":
                    p=case/"run_contract.json";v=json.loads(p.read_text());v["target_docs"]=200;p.write_text(json.dumps(v))
                else:
                    p=root/"matrix.json";v=json.loads(p.read_text());v["limit"]=200;p.write_text(json.dumps(v))
                with self.assertRaises(ValueError):load(root)

    def test_pairing_generation_and_native_cache_tamper_rejected(self):
        for mode in ("input","reference","EOS","cachecoverage","cachework","auditstatus","generation"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                fixture(directory);p=Path(directory)/"hybridkv_kvrepack"/TASKS[0]/"samples/000000.json";r=json.loads(p.read_text())
                if mode=="input":r["input_fingerprints"]["input_hashes"]["input_ids"]="changed";r["input_hashes"]["input_ids"]="changed"
                elif mode=="reference":r["reference_metadata"]["original_row_sha256"]="changed"
                elif mode=="EOS":r["eos_reached"]=False
                elif mode=="cachecoverage":r["cache_audit"]["layers"].pop()
                elif mode=="cachework":r["cache_audit"]["prefill_tokens_total"]+=1
                elif mode=="auditstatus":r["cache_audit"]["status"]="failed"
                else:r["generated_tokens"]+=1
                p.write_text(json.dumps(r))
                with self.assertRaises(ValueError):load(directory)

    def test_native_metric_completeness_and_no_per_sentence_CIDEr_substitute(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory);runs,provenance=load(directory);s=summary(runs,provenance);first,second=s["conditions"]
            for mode in ("missingMETEOR","nonfinite","WUPSmean"):
                metrics=deepcopy(s["metrics"]);per={TASKS[0]:{first:[dict(WUPS=60.,WUPS_0_9=50.)],second:[dict(WUPS=40.,WUPS_0_9=10.)]}}
                if mode=="missingMETEOR":metrics[TASKS[1]][second].pop("METEOR")
                elif mode=="nonfinite":metrics[TASKS[1]][second]["CIDEr"]=float("nan")
                else:per[TASKS[0]][second][0]["WUPS"]=41.
                with self.subTest(mode=mode),self.assertRaises(ValueError):build_summary(runs,metrics,per,provenance=provenance)


if __name__=="__main__":unittest.main()

"""CPU-only single-case completeness, reference and score checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.benchmarks.single_video_report import (
    VATEX_METRICS, _canonical, build_summary, json_sha, load_case,
    validate_references, write_report,STRICT_LOOKM_POLICY,LEGACY_LOOKM_POLICY,main,
)


def fixture(directory,*,strict=False,count=200):
    root=Path(directory);(root/"samples").mkdir()
    policy=STRICT_LOOKM_POLICY if strict else LEGACY_LOOKM_POLICY
    contract=dict(schema_version=1,method="lookm",task="nextqa_oe_test",target_docs=count,target_generations=count,max_new_tokens=128,budget_ratio=.1,arrow_sha256="arrow",dataset_rows=9178,budget_policy=policy)
    identity=json_sha(contract)
    (root/"run_contract.json").write_text(json.dumps(contract))
    (root/"run_state.json").write_text(json.dumps(dict(status="complete",target_docs=count,completed_docs=count,target_generations=count,completed_generations=count)))
    docs=[]
    for i in range(count):
        doc=dict(video=f"video{i//2}",answer="yes",type="CW",additional_ref_answer="",question=f"q{i}");docs.append(doc)
        meta=dict(task="nextqa_oe_test",doc_id=i,video_id=doc["video"],source_video_id=doc["video"],arrow_sha256="arrow",original_row_sha256=hashlib.sha256(_canonical(doc).encode()).hexdigest())
        fp=dict(input_hashes={"input_ids":f"hash{i}"},input_shapes={"input_ids":[1,100]},input_dtypes={"input_ids":"torch.int64"})
        shape=[1,4,10,128];final=[1,4,11,128]
        cache=dict(status="completed",method="lookm",task="nextqa_oe_test",doc_id=i,input_tokens=100,native_kv_heads=4,layers_count=28,
            layers=[dict(layer=l,before_tokens_per_head=100,head_lengths=[10]*4,after_tokens_total=40,key_shape=shape,value_shape=shape) for l in range(28)],
            nominal_budget_ratio=.1,prefill_tokens_total=1120,uncompressed_tokens_total=11200,actual_retained_ratio=.1,budget_policy=policy,strict_budget_passed=strict)
        if strict:
            meta["input_tokens"]=100;cache["total_prefill_capacity"]=10
            for layer in cache["layers"]:layer.update(total_prefill_capacity=10,configured_recent_window=32,recent_window=10,selected_history_tokens=0)
        row=dict(schema_version=1,status="completed",method="lookm",task="nextqa_oe_test",doc_id=i,contract_sha256=identity,video_id=doc["video"],source_video_id=doc["video"],reference_metadata=meta,
            text="yes",prediction="yes",generated_ids=[7,2],generated_tokens=2,eos_token_id=2,eos_reached=True,max_new_tokens=128,
            input_fingerprints=fp,**fp,input_tokens=100,config=dict(method="lookm",task="nextqa_oe_test",stream_contract_sha256=identity,max_new_tokens=128,budget_ratio=.1,budget_policy=policy),audit=cache,cache_audit=cache,
            final_cache_shapes=[dict(layer=l,key_shape=final,value_shape=final) for l in range(28)])
        (root/"samples"/f"{i:06d}.json").write_text(json.dumps(row))
    return docs,dict(sha256="arrow",split_rows=9178,doc_ids=list(range(count)))


class SingleVideoReportTests(unittest.TestCase):
    def test_actual_one_row_strict_smoke_is_not_a_formal200_result(self):
        with tempfile.TemporaryDirectory() as directory:
            docs,reference=fixture(directory,strict=True,count=1)
            rows,provenance=load_case(directory,limit=1)
            validate_references(rows,docs,reference,provenance["generation_contract"])
            summary=build_summary(rows,{"WUPS":100.},provenance=provenance,per_input=[dict(WUPS=100.,WUPS_0_9=100.)])
            self.assertEqual(summary["limit"],1);self.assertEqual(summary["sample_scope"],"smoke_first1")
            self.assertFalse(summary["formal_result"]);self.assertTrue(summary["strict_budget_passed"])
            self.assertIn("selected1 reference",summary["metric_scope"])
            self.assertIn("not a formal200",summary["scope"])
            output=Path(directory)/"report";write_report(summary,output)
            self.assertEqual(len((output/"scores.csv").read_text().splitlines()),2)
            with self.assertRaises(ValueError):load_case(directory,limit=200)

    def test_main_really_parses_and_dispatches_limit1(self):
        class Dispatched(Exception):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch("kvrepack.benchmarks.single_video_report.load_case",side_effect=Dispatched) as loader:
                with self.assertRaises(Dispatched):
                    main(["--run-root",str(root),"--hf-home",str(root/"cache"),"--output",str(root/"report"),"--limit","1"])
                loader.assert_called_once_with(root,limit=1)

    def test_single_case_needs_no_matrix_and_preserves_native_scores_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            docs,reference=fixture(directory);rows,provenance=load_case(directory)
            validate_references(rows,docs,reference,provenance["generation_contract"])
            s=build_summary(rows,{"WUPS":100.},provenance=provenance,per_input=[dict(WUPS=100.,WUPS_0_9=100.)]*200)
            self.assertEqual(s["n_generations"],200);self.assertEqual(s["generation"]["eos_count"],200)
            self.assertEqual(s["cache_budget"]["actual_retained_ratio_mean"],.1)
            self.assertTrue(s["legacy_budget_result"]);self.assertFalse(s["strict_budget_passed"])
            self.assertEqual(s["budget_policy"],LEGACY_LOOKM_POLICY)
            output=Path(directory)/"report";write_report(s,output)
            for name in ("task_scores.csv","scores.csv","generation_summary.csv","cache_audit.csv","sample_files.csv","summary.json","RESULTS.md"):
                self.assertGreater((output/name).stat().st_size,20)
            self.assertNotIn("comparisons",json.loads((output/"summary.json").read_text()))
            with self.assertRaises(FileExistsError):write_report(s,output)

    def test_v2_report_requires_explicit_validated_total_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory,strict=True);rows,provenance=load_case(directory)
            summary=build_summary(rows,{"WUPS":100.},provenance=provenance,per_input=[dict(WUPS=100.,WUPS_0_9=100.)]*200)
            self.assertEqual(summary["budget_policy"],STRICT_LOOKM_POLICY);self.assertTrue(summary["strict_budget_passed"])
            self.assertFalse(summary["legacy_budget_result"])
            rows[0]["strict_budget_passed"]=False
            with self.assertRaises(ValueError):build_summary(rows,{"WUPS":100.},provenance=provenance,per_input=[dict(WUPS=100.,WUPS_0_9=100.)]*200)

    def test_partial_prefix_bad_generation_and_reference_tampering_rejected(self):
        for mode in ("incomplete","missing","generation","reference"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                docs,reference=fixture(directory);root=Path(directory)
                if mode=="incomplete":
                    p=root/"run_state.json";value=json.loads(p.read_text());value["status"]="running";p.write_text(json.dumps(value))
                elif mode=="missing":(root/"samples/000199.json").unlink()
                elif mode=="generation":
                    p=root/"samples/000000.json";value=json.loads(p.read_text());value["generated_tokens"]=3;p.write_text(json.dumps(value))
                else:
                    rows,provenance=load_case(root);docs[0]["answer"]="changed"
                    with self.assertRaises(ValueError):validate_references(rows,docs,reference,provenance["generation_contract"])
                    continue
                with self.assertRaises(ValueError):load_case(root)

    def test_missing_METEOR_nonfinite_or_mismatched_input_WUPS_never_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory);rows,provenance=load_case(directory)
            with self.assertRaises(ValueError):build_summary(rows,{"WUPS":100.},provenance=provenance,per_input=[dict(WUPS=0.,WUPS_0_9=0.)]*200)
            vatex=deepcopy(provenance);vatex["task"]="vatex_test"
            with self.assertRaises(ValueError):build_summary(rows,{k:.2 for k in VATEX_METRICS if k!="METEOR"},provenance=vatex)
            with self.assertRaises(ValueError):build_summary(rows,{k:float("nan") if k=="CIDEr" else .2 for k in VATEX_METRICS},provenance=vatex)


if __name__=="__main__":unittest.main()

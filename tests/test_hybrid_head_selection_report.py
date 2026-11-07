"""CPU report fixtures; scorer injection does not claim GPU/official WUPS coverage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from kvrepack.ablations.hybrid_head_selection_report import (
    METHOD, TASK, VARIANTS, _canonical, _json_sha, progress, score_stream, write_report,
)


def validate_fixture(bundle, *, doc_id, contract_sha256):
    if (bundle.get("doc_id") != doc_id or bundle.get("contract_sha256") != contract_sha256
            or bundle.get("fairness", {}).get("status") != "passed"):
        raise ValueError("Incomplete fixture fairness/identity")
    rows = bundle["records"]
    baseline = {k:v for k,v in rows["similarity"]["config"].items() if k not in ("strategy","mask_seed")}
    if any({k:v for k,v in row["config"].items() if k not in ("strategy","mask_seed")} != baseline for row in rows.values()):
        raise ValueError("Fixture nonselection configuration changed")


def fixture(directory, *, complete=True):
    root=Path(directory);(root/"samples").mkdir()
    docs=[dict(video=f"video-{i//2}",answer="yes",type="CW",additional_ref_answer="maybe" if i==2 else "",question=f"q{i}") for i in range(3)]
    reference=dict(sha256="arrow",split_rows=9178,doc_ids=list(range(3)))
    contract=dict(schema_version=1,method=METHOD,task=TASK,target_docs=3,target_generations=12,arrow_sha256="arrow",dataset_rows=9178)
    identity=_json_sha(contract)
    (root/"run_contract.json").write_text(json.dumps(contract))
    (root/"run_state.json").write_text(json.dumps(dict(status="complete" if complete else "running",target_docs=3,target_generations=12,completed_docs=3,completed_generations=12)))
    predictions=(("yes","no","yes","yes"),("no","yes","yes","no"),("yes","yes","no","no"))
    baseline=[[[4*l+h<56 for h in range(4)]] for l in range(28)]
    pre=dict(head_category=None,calibrated_head_category=dict(values=[[[h%2==0 for h in range(4)]] for l in range(28)],tensor=dict(shape=[28,1,4],dtype="torch.bool",sha256="native-category")),sparse_budget_per_head=[[[20+(4*l+h)%17 for h in range(4)]] for l in range(28)],candidate_hashes=[dict(layer=l,sha256=f"candidate-{l}") for l in range(28)])
    for doc_id,doc in enumerate(docs):
        meta=dict(task=TASK,doc_id=doc_id,video_id=doc["video"],source_video_id=doc["video"],arrow_sha256="arrow",original_row_sha256=hashlib.sha256(_canonical(doc).encode()).hexdigest())
        rows={}
        for index,variant in enumerate(VARIANTS):
            offset=0 if variant=="similarity" else 28+7*int(variant[-1])
            chosen={(offset+i)%112 for i in range(56)}
            mask=[[[4*l+h in chosen for h in range(4)]] for l in range(28)]
            calls=[dict(layer=l,batch=0,head=h,before=100,requested_target=pre["sparse_budget_per_head"][l][0][h],after=pre["sparse_budget_per_head"][l][0][h]) for l in range(28) for h in range(4) if mask[l][0][h]]
            audit=dict(status="completed",layers=28,native_kv_heads=4,batch_size=1,baseline_masks=deepcopy(baseline),selected_masks=mask,baseline_selected_count=[56],selected_count=[56],actual_merge_calls=56,actual_removed_tokens=sum(c["before"]-c["after"] for c in calls),merge_calls=calls,pre_selection=deepcopy(pre))
            config=dict(method=METHOD,strategy="similarity" if index==0 else "random_global",mask_seed=0 if index==0 else index-1,max_new_tokens=128,budget_ratio=.1,frames=64)
            rows[variant]=dict(text=predictions[doc_id][index],prediction=predictions[doc_id][index],config=config,audit=audit,generated_ids=[1,2],generated_tokens=2,eos_token_id=2,eos_reached=True)
        bundle=dict(schema_version=1,status="completed",method=METHOD,task=TASK,doc_id=doc_id,contract_sha256=identity,reference_metadata=meta,records=rows,fairness=dict(status="passed"))
        (root/"samples"/f"{doc_id:06d}.json").write_text(json.dumps(bundle))
    return docs,reference


def scorer(doc,predictions):
    value=float(predictions[0]==doc["answer"])
    return {"WUPS":{"0":value,"0.9":value,"qtype":"CW"}}


def aggregate(values):return 100*sum(v["0"] for v in values)/len(values)


def score(root,docs,ref,**kwargs):
    return score_stream(root,docs,ref,scorer=scorer,aggregate=aggregate,limit=3,bootstrap_samples=25,validator=validate_fixture,**kwargs)


class HybridHeadSelectionReportTests(unittest.TestCase):
    def test_heterogeneous_work_is_retained_without_forcing_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            docs,ref=fixture(directory);s=score(directory,docs,ref)
            self.assertEqual(s["method"],METHOD)
            self.assertEqual(s["n_generations"],12)
            self.assertEqual(s["n_source_videos"],2)
            self.assertEqual(s["scores"]["similarity"],200/3)
            self.assertAlmostEqual(s["random_mean"],500/9)
            self.assertAlmostEqual(s["random_sample_std"],statistics.stdev([200/3,200/3,100/3]))
            self.assertTrue(any(x["difference"]<0 for x in s["paired_table"]))
            self.assertTrue(any(s["merge_work"][v]["total"]!=s["merge_work"]["similarity"]["total"] for v in VARIANTS[1:]))
            self.assertEqual(s["scorer_cache"],dict(unique_prediction_reference_evaluations=4,exact_cache_hits=8))
            for v in VARIANTS:self.assertEqual(s["generation"][v]["eos_count"],3)
            output=Path(directory)/"report";write_report(s,output,plots=False)
            for name in ("scores.csv","paired_scores.csv","run_scores.csv","selector_audit.csv","merge_work.csv","generation_summary.csv","sample_files.csv","summary.json","RESULTS.md"):
                self.assertGreater((output/name).stat().st_size,20)
            self.assertIn("heterogeneous",(output/"RESULTS.md").read_text())
            with self.assertRaises(FileExistsError):write_report(s,output,plots=False)

    def test_native_layer_batch_layout_invalid_count_calls_and_budget_tamper_rejected(self):
        for mode in ("transpose","count","work","calls","budget","candidate","calibrated","config","fairness"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                docs,ref=fixture(directory);path=Path(directory)/"samples/000000.json";b=json.loads(path.read_text());r=b["records"][VARIANTS[1]]
                if mode=="transpose":r["audit"]["selected_masks"]=[[layer[0] for layer in r["audit"]["selected_masks"]]]
                elif mode=="count":r["audit"]["selected_count"]=[55]
                elif mode=="work":r["audit"]["actual_removed_tokens"]+=1
                elif mode=="calls":r["audit"]["merge_calls"][0]["after"]=100
                elif mode=="budget":r["audit"]["pre_selection"]["sparse_budget_per_head"][0][0][0]+=1
                elif mode=="candidate":r["audit"]["pre_selection"]["candidate_hashes"][0]["sha256"]="changed"
                elif mode=="calibrated":
                    for row in b["records"].values():row["audit"]["pre_selection"].pop("calibrated_head_category")
                elif mode=="config":r["config"]["frames"]=32
                else:b["fairness"]["status"]="failed"
                path.write_text(json.dumps(b))
                with self.assertRaises(ValueError):score(directory,docs,ref)

    def test_random_mask_must_remain_fixed_across_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            docs,ref=fixture(directory);path=Path(directory)/"samples/000001.json";b=json.loads(path.read_text())
            b["records"][VARIANTS[1]]["audit"]=deepcopy(b["records"][VARIANTS[2]]["audit"])
            path.write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,"ranking changed"):score(directory,docs,ref)

    def test_reference_coverage_and_completion_are_required(self):
        for mode in ("row","arrow","missing","method","running"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                docs,ref=fixture(directory,complete=mode!="running")
                if mode=="row":docs[0]["answer"]="changed"
                elif mode=="arrow":ref["sha256"]="changed"
                elif mode=="missing":(Path(directory)/"samples/000001.json").unlink()
                elif mode=="method":
                    p=Path(directory)/"run_contract.json";c=json.loads(p.read_text());c["method"]="snapkv_kvrepack";p.write_text(json.dumps(c))
                else:self.assertFalse(progress(directory,3)["scoring_performed"])
                with self.assertRaises(ValueError):score(directory,docs,ref)

    def test_nonfinite_scores_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            docs,ref=fixture(directory)
            bad=lambda doc,pred:{"WUPS":{"0":float("nan"),"0.9":1.}}
            with self.assertRaises(ValueError):score_stream(directory,docs,ref,scorer=bad,aggregate=aggregate,limit=3,validator=validate_fixture)


if __name__=="__main__":unittest.main()

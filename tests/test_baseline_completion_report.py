"""CPU checks for matching full/model latency and reusing the correct corpus."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.ablations import baseline_completion_report as report
from kvrepack.ablations.run_head_selection import atomic_json, json_sha, sha256


def fixture(root):
    source = root/"src/source.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen numerical fixture\n")
    common = dict(task="vatex_test", model_snapshot="Qwen2.5-VL-7B-snapshot", frames=64,
        min_pixels=200704, max_pixels=1605632, budget_ratio=.1,
        budget_policy="total_active_prefill_including_local_global_cap_v1",
        max_new_tokens=64, repetition_penalty=1.05, do_sample=False, dtype="bfloat16",
        model_seed=42, arrow_sha256="fixture-arrow", dataset_rows=4478,
        task_yaml_sha256="fixture-yaml", task_utils_sha256="fixture-utils",
        hf_home=str(root/"hf"), diagnostic_audit=False, warmup_per_arm=2,
        source_sha256={"src/source.py":sha256(source)})
    for backbone in report.BACKBONES:
        baseline = dict(common, backbone=backbone, target_docs=1,
            algorithm_env={"native":{}, "kvrepack":{"MERGE_RATIO":".5"}},
            variants=["native","kvrepack"])
        style = dict(common, method=backbone+"_kvrepack", target_docs=200,
            algorithm_env={"MERGE_RATIO":".5"}, variants=["lookm","asymkv"])
        for base, contract in ((root/"new"/backbone, baseline),
                               (root/"styles"/(backbone+"_kvrepack"), style)):
            base.mkdir(parents=True)
            atomic_json(base/"run_contract.json",contract)
            atomic_json(base/"run_state.json",dict(status="complete",
                completed_docs=contract["target_docs"], completed_generations=2*contract["target_docs"]))
        rows={variant:make_row(backbone,variant,0) for variant in report.BASELINE_VARIANTS}
        atomic_json(root/"new"/backbone/"samples/000000.json",dict(status="completed",doc_id=0,
            backbone=backbone,contract_sha256=json_sha(baseline),records=rows,fairness={"status":"passed"}))
        for doc_id in range(200):
            rows={variant:make_row(backbone,variant,doc_id) for variant in report.STYLE_VARIANTS}
            atomic_json(root/"styles"/(backbone+"_kvrepack")/"samples"/f"{doc_id:06d}.json",
                dict(status="completed",doc_id=doc_id,contract_sha256=json_sha(style),
                     records=rows,fairness={"status":"passed"}))
    return common


def make_row(backbone, variant, doc_id):
    ratio = .09 if backbone=="hybridkv" and variant=="native" else .1
    group = dict(positions=[0,1],group_ids=[0,0],group_size=[2],start_mask=[True,False],
                 layer_idx=0,head_idx=0,batch_idx=0,before=2,after=1)
    audit=dict(status="complete",merge_calls=[group],layers=[dict(layer=0,candidate_idx=[0,1],
        final_history=1,candidate_history=2)],selected_masks=[True],prefill_cache_shapes=[[1]],
        merge_mask=[True],budget={"cap":11200})
    return dict(status="completed",task="vatex_test",method=backbone if variant=="native" else backbone+"_kvrepack",
        variant=variant,doc_id=doc_id,text=f"caption {doc_id}",prediction=f"caption {doc_id}",
        generated_ids=[5,6],generated_tokens=2,input_tokens=1000,eos_reached=True,
        capped_without_eos=False,warmup=False,reference_metadata={"doc_id":doc_id,"video_id":str(doc_id)},
        input_fingerprints={"input_ids":str(doc_id)},audit=audit,
        budget=dict(actual_retained_ratio=ratio,prefill_tokens_total=int(ratio*112000),capacity=11200),
        runtime=dict(end_to_end_seconds=3.12,model_end_to_end_seconds=2.01,
            generation_seconds=2.,preprocessing_seconds=1.,h2d_seconds=.1,
            text_decode_seconds=.01,fisher_seconds=0.,peak_allocated_gib=20.))


def fake_score(rows, contract, output, repo_root):
    return dict(zip(report.QUALITY_METRICS,(.2,.3,.4,.5)))


class BaselineCompletionReportTests(unittest.TestCase):
    def test_pilot_scores_its_own_eight_corpora_and_keeps_hybrid_budget_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            with patch.object(report,"score_native_rows",side_effect=fake_score) as scorer:
                summary=report.build_report(root/"new",root/"styles",root/"report",limit=1,repo_root=root)
            self.assertEqual(scorer.call_count,8)
            self.assertEqual(summary["scope"],"pilot_first1")
            self.assertEqual(len(summary["comparisons"]),8)
            self.assertTrue(all(row["n"]==1 for row in summary["comparisons"]))
            self.assertFalse(summary["budget_validation"]["hybridkv_identical_prefill_counts"])
            self.assertAlmostEqual(summary["comparisons"][0]["end_to_end_seconds_mean"],3.12)
            self.assertAlmostEqual(summary["comparisons"][0]["model_end_to_end_seconds_mean"],2.01)
            self.assertTrue((root/"report/budget_comparison.csv").is_file())

    def test_different_style_input_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            p=root/"new/snapkv/samples/000000.json";bundle=json.loads(p.read_text())
            for row in bundle["records"].values():row["input_fingerprints"]={"input_ids":"wrong-video"}
            atomic_json(p,bundle)
            with patch.object(report,"score_native_rows") as scorer:
                with self.assertRaisesRegex(ValueError,"input mismatch"):
                    report.build_report(root/"new",root/"styles",root/"report",limit=1,repo_root=root)
            scorer.assert_not_called()
            self.assertFalse((root/"report").exists())

    def test_native_snap_cache_capacity_must_match_other_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            p=root/"new/snapkv/samples/000000.json";bundle=json.loads(p.read_text())
            bundle["records"]["native"]["budget"].update(actual_retained_ratio=.08,prefill_tokens_total=8960)
            atomic_json(p,bundle)
            with self.assertRaisesRegex(ValueError,"identical final physical"):
                report.build_report(root/"new",root/"styles",root/"report",limit=1,repo_root=root)

    def test_generation_time_cannot_be_relabeled_full_e2e(self):
        contract=dict(max_new_tokens=64,budget_ratio=.1)
        bundle=dict(status="completed",doc_id=0,backbone="snapkv",contract_sha256=json_sha(contract),
            fairness={"status":"passed"},records={v:make_row("snapkv",v,0) for v in report.BASELINE_VARIANTS})
        bundle["records"]["native"]["runtime"]["end_to_end_seconds"]=2.
        with self.assertRaisesRegex(ValueError,"timing boundaries"):
            report.validate_baseline_bundle(bundle,contract,"snapkv",0)

    def test_changed_generation_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            (root/"src/source.py").write_text("# a different algorithm\n")
            with self.assertRaisesRegex(ValueError,"source is unavailable or changed"):
                report.build_report(root/"new",root/"styles",root/"report",limit=1,repo_root=root)

    def test_cached_quality_requires_the_exact_prediction_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            case=root/"styles/snapkv_kvrepack";contract=json.loads((case/"run_contract.json").read_text())
            rows=[make_row("snapkv","lookm",i) for i in range(200)]
            directory=root/"styles/report/snapkv_kvrepack/lookm";directory.mkdir(parents=True)
            predictions=[dict(task="vatex_test",doc_id=i,text=r["text"]) for i,r in enumerate(rows)]
            p=directory/"predictions.jsonl";p.write_text("".join(json.dumps(r)+"\n" for r in predictions))
            identity=dict(reference_arrow_sha256=contract["arrow_sha256"],prediction_corpus_sha256=report._canonical_hash(predictions))
            quality=fake_score(None,None,None,None)
            metrics=dict(status="complete",task="vatex_test",n=200,expected_count=200,metrics=quality,
                scoring_identity=identity,scoring_identity_sha256=report._canonical_hash(identity),records={"file_sha256":sha256(p)})
            atomic_json(directory/"metrics.json",metrics)
            atomic_json(root/"styles/report/summary.json",dict(methods={"snapkv_kvrepack":{"contract":contract}},
                comparisons=[dict(method="snapkv_kvrepack",variant="lookm",n=200,**quality)]))
            recovered,proof=report._cached_style_metrics(case,rows,contract,root)
            self.assertEqual(recovered,quality)
            p.write_text(p.read_text().replace('caption 0','wrong image caption',1))
            with self.assertRaisesRegex(ValueError,"predictions differ"):
                report._cached_style_metrics(case,rows,contract,root)


if __name__=="__main__":
    unittest.main()

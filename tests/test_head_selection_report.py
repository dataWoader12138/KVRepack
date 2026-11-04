"""CPU fairness/coverage and paired-seed statistics; no model or GPU execution."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.ablations.head_selection_report import (
    COUNTS, TASKS, VARIANTS, build_summary, load_runs, offline_resources,
    paired_wups_intervals, write_report,
)


def masks(indices):
    return [[[4*layer+head in indices for head in range(4)] for layer in range(28)]]


def row(task, variant, doc=0):
    count = COUNTS[task]
    seed = 0 if variant == "similarity" else int(variant[-1])
    baseline = masks({(index+doc)%112 for index in range(count)})
    selected = baseline if variant == "similarity" else masks({(index+9*seed)%112 for index in range(count)})
    audit = dict(status="completed", strategy="similarity" if variant == "similarity" else "random_global",
                 seed=seed, native_kv_heads=4, layers=28, batch_size=1,
                 baseline_masks=baseline, selected_masks=selected, baseline_selected_count=[count], selected_count=[count],
                 baseline_layer_counts=[[sum(layer) for layer in baseline[0]]], selected_layer_counts=[[sum(layer) for layer in selected[0]]],
                 expected_effective_merge_count=count, actual_merge_calls=count, expected_removed_tokens=count*3, actual_removed_tokens=count*3,
                 prefill_cache_shapes=[dict(layer=l, key_shape=[1,4,10,128], value_shape=[1,4,10,128], key_dtype="bfloat16", value_dtype="bfloat16") for l in range(28)],
                 candidate_hashes=[dict(layer=l, candidate_raw={"sha256": "a"*64}, candidate_value={"sha256": "b"*64}) for l in range(28)])
    return dict(task=task, doc_id=doc, source_video_id=f"video-{doc//2}", status="completed", prediction="answer", generated_ids=[1,2],
                input_hashes={"input_ids": f"hash-{doc}"}, config=dict(model="7b", frames=64, max_new_tokens=128, budget_ratio=.1,
                     mask_strategy=audit["strategy"], mask_seed=seed), audit=audit)


def write_runs(root, limit):
    for task in TASKS:
        for variant in VARIANTS:
            path=Path(root)/task/variant/"records.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(row(task,variant,doc))+"\n" for doc in range(limit)))


def score_fixture(limit=20):
    runs={task:{variant:[row(task,variant,i) for i in range(limit)] for variant in VARIANTS} for task in TASKS}
    scores={task:{} for task in TASKS}; per_input={"nextqa_oe_test": {}, "vatex_test": {}}
    for index,variant in enumerate(VARIANTS):
        wups=100.-10*index
        per_input["nextqa_oe_test"][variant]=[wups]*limit
        scores["nextqa_oe_test"][variant]=dict(metrics={"WUPS":wups},missing_metrics={})
        scores["vatex_test"][variant]=dict(metrics={metric:1.-.1*index for metric in ("Bleu_4","METEOR","ROUGE_L","CIDEr")},missing_metrics={})
    audit=dict(limit=limit,audit_rows=[])
    return runs,scores,per_input,audit


class HeadSelectionReportTests(unittest.TestCase):
    def test_complete_160_prediction_coverage_and_global_not_layer_matched(self):
        with tempfile.TemporaryDirectory() as root:
            write_runs(root,20)
            runs,audit=load_runs(root,limit=20)
            self.assertEqual(sum(len(values) for task in runs.values() for values in task.values()),160)
            self.assertEqual(len(audit["audit_rows"]),160)
            original=runs[TASKS[0]]["similarity"][0]["audit"]["selected_layer_counts"]
            random=runs[TASKS[0]]["random_global_seed1"][0]["audit"]["selected_layer_counts"]
            self.assertNotEqual(original,random)

    def test_smoke1_is_allowed_and_wrong_coverage_task_or_partial_JSONL_fails(self):
        for mode in ("valid", "missing", "duplicate", "wrong_task", "partial"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                write_runs(root,1)
                path=Path(root)/TASKS[0]/VARIANTS[0]/"records.jsonl"
                if mode=="missing":path.write_text("")
                if mode=="duplicate":path.write_text(path.read_text()*2)
                if mode=="wrong_task":
                    r=json.loads(path.read_text());r["task"]=TASKS[1];path.write_text(json.dumps(r)+"\n")
                if mode=="partial":path.write_text(path.read_text().rstrip())
                if mode=="valid":self.assertEqual(load_runs(root,limit=1)[1]["limit"],1)
                else:
                    with self.assertRaises(ValueError):load_runs(root,limit=1)

    def test_changed_inputs_candidates_work_or_configuration_rejected(self):
        mutations=(lambda r:r["input_hashes"].update(input_ids="changed"),
                   lambda r:r["config"].update(max_new_tokens=64),
                   lambda r:r["audit"]["candidate_hashes"][0]["candidate_raw"].update(sha256="changed"),
                   lambda r:r["audit"].update(actual_merge_calls=33),
                   lambda r:r["audit"].update(actual_removed_tokens=0),
                   lambda r:r["audit"].update(selected_count=[33]))
        for mutate in mutations:
            with self.subTest(mutation=mutate),tempfile.TemporaryDirectory() as root:
                write_runs(root,1)
                path=Path(root)/TASKS[0]/VARIANTS[1]/"records.jsonl"
                value=json.loads(path.read_text());mutate(value);path.write_text(json.dumps(value)+"\n")
                with self.assertRaises(ValueError):load_runs(root,limit=1)

    def test_seed_mean_and_sample_SD_are_run_statistics_not_60_independent_inputs(self):
        runs,scores,individual,audit=score_fixture()
        summary=build_summary(runs,scores,individual,audit=audit,bootstrap_samples=100)
        value=next(r for r in summary["seed_summary"] if r["metric"]=="WUPS")
        self.assertEqual(value["random_mean"],80.)
        self.assertEqual(value["random_sample_std"],10.)
        self.assertEqual(value["similarity_minus_random_mean"],20.)
        self.assertEqual(value["ci95"],[20.,20.])
        self.assertEqual(summary["paired_WUPS"]["n_inputs"],20)
        self.assertEqual(summary["paired_WUPS"]["n_source_videos"],10)
        self.assertEqual(len(summary["paired_input_WUPS"]),60)
        self.assertTrue(all(r["ci95"] is None for r in summary["seed_summary"] if r["task"]=="vatex_test"))

    def test_repeated_source_members_stay_together_and_all_seeds_are_paired(self):
        runs,scores,individual,audit=score_fixture(21)
        for variant in VARIANTS:
            for i,record in enumerate(runs[TASKS[0]][variant]):record["source_video_id"]="a" if i<20 else "b"
        individual[TASKS[0]]["similarity"]=[100.]*20+[0.]
        for variant in VARIANTS[1:]:individual[TASKS[0]][variant]=[0.]*20+[100.]
        result=paired_wups_intervals(runs,individual,samples=1000)
        self.assertAlmostEqual(result["random_mean"]["difference"],1900/21)
        self.assertEqual(result["random_mean"]["ci95"],[-100.,100.])
        self.assertEqual(result["n_source_videos"],2)

    def test_single_source_has_no_CI_and_missing_METEOR_is_explicit_without_composite(self):
        runs,scores,individual,audit=score_fixture(1)
        scores["vatex_test"][VARIANTS[1]]["metrics"].pop("METEOR")
        scores["vatex_test"][VARIANTS[1]]["missing_metrics"]["METEOR"]="local resource unavailable"
        summary=build_summary(runs,scores,individual,audit=audit)
        self.assertEqual(summary["status"],"partial_metrics")
        self.assertIsNone(summary["paired_WUPS"]["random_mean"]["ci95"])
        meteor=next(r for r in summary["seed_summary"] if r["metric"]=="METEOR")
        self.assertIsNone(meteor["random_mean"])
        self.assertNotIn("mean4",json.dumps(summary))

    def test_nonfinite_or_inconsistent_WUPS_rejected(self):
        for number in (float("nan"),-1.,101.,50.):
            with self.subTest(number=number):
                runs,scores,individual,audit=score_fixture(1)
                individual[TASKS[0]][VARIANTS[0]]=[number]
                with self.assertRaises(ValueError):build_summary(runs,scores,individual,audit=audit)

    def test_report_preserves_seed_scores_and_WUPS_pair_table(self):
        runs,scores,individual,audit=score_fixture(1)
        summary=build_summary(runs,scores,individual,audit=audit)
        with tempfile.TemporaryDirectory() as directory:
            write_report(summary,directory,plots=False)
            for name in ("summary.json","RESULTS.md","run_metrics.csv","seed_summary.csv","selector_audit.csv","paired_input_WUPS.csv"):
                self.assertGreater((Path(directory)/name).stat().st_size,20)

    def test_worker_strategy_config_field_and_standalone_plots(self):
        with tempfile.TemporaryDirectory() as root:
            write_runs(root,1)
            for path in Path(root).glob('*/*/records.jsonl'):
                record=json.loads(path.read_text())
                record['config']['strategy']=record['config'].pop('mask_strategy')
                path.write_text(json.dumps(record)+'\n')
            runs,audit=load_runs(root,limit=1)
            _,scores,individual,_=score_fixture(1)
            summary=build_summary(runs,scores,individual,audit=audit)
            try:import matplotlib
            except ImportError:self.skipTest('optional matplotlib unavailable')
            output=Path(root)/'report'
            write_report(summary,output)
            for stem in ('task_scores','head_selection'):
                for extension in ('png','pdf','svg'):
                    self.assertGreater((output/(stem+'.'+extension)).stat().st_size,100)

    def test_offline_download_guard_restores_state_after_error(self):
        import urllib.request
        original=urllib.request.urlopen
        with self.assertRaisesRegex(RuntimeError,"offline"):
            with offline_resources(nltk_data=Path('/root/nltk_data'),java_home=None):
                urllib.request.urlopen('https://offline-test.invalid')
        self.assertIs(urllib.request.urlopen,original)


if __name__=="__main__":unittest.main()

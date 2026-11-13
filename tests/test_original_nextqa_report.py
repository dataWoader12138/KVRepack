"""Original-budget reports must preserve native allocations and reject strict runs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.ablations import original_nextqa_report as report
from kvrepack.ablations.run_head_selection import atomic_json, json_sha, sha256


def fixture(root):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    for relative in report.SHARED_SOURCES:
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen source fixture\n")
    task = root/"vendor/lmms-eval/lmms_eval/tasks/nextqa"
    task.mkdir(parents=True)
    (task/"nextqa_oe_test.yaml").write_text("task: nextqa_oe_test\n")
    (task/"utils.py").write_text("# frozen native scorer fixture\n")
    snapshot = root/"model"
    snapshot.mkdir()
    atomic_json(snapshot/"config.json", dict(model_type="qwen2_5_vl", hidden_size=3584,
        num_hidden_layers=28, num_key_value_heads=4))
    hf_home = root/"hf"
    arrow = hf_home/"datasets/lmms-lab___n_ex_tqa/OE/fixture/n_ex_tqa-test.arrow"
    arrow.parent.mkdir(parents=True)
    doc = dict(video="0", question="Who is walking?", answer="person", type="CW", additional_ref_answer="")
    table = pa.Table.from_pylist([doc])
    with pa.OSFile(str(arrow), "wb") as stream:
        with ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    common = dict(task=report.TASK, model_snapshot=str(snapshot), frames=64,
        min_pixels=200704, max_pixels=1605632, budget_ratio=.1, budget_policy=report.BUDGET_POLICY,
        max_new_tokens=128, repetition_penalty=1.05, do_sample=False, dtype="bfloat16",
        model_seed=42, arrow_sha256=sha256(arrow), dataset_rows=1,
        task_yaml_sha256=sha256(task/"nextqa_oe_test.yaml"), task_utils_sha256=sha256(task/"utils.py"),
        hf_home=str(hf_home), diagnostic_audit=False, warmup_per_arm=2,
        model_metadata_sha256={"config.json": sha256(snapshot/"config.json")}, versions={"fixture":"1"},
        source_sha256={relative: sha256(root/relative) for relative in report.SHARED_SOURCES})
    for backbone in report.BACKBONES:
        environment = ({"SNAPKV_SIMHEAD_MERGE_RATIO":".30", "SNAPKV_KVREPACK_COMPRESS_RATIO":".70", "K_DIFF_ALPHA":"2"}
            if backbone == "snapkv" else {"HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO":".50",
                "HYBRID_KVREPACK_COMPRESS_RATIO":".50", "K_DIFF_ALPHA":"2"})
        for family, variants in report.FAMILIES.items():
            contract = dict(common, method=backbone if family == "baseline" else backbone+"_kvrepack",
                target_docs=1, target_generations=2, variants=list(variants),
                algorithm_env={"native":{},"kvrepack":environment} if family == "baseline" else environment)
            if family == "baseline":
                contract["backbone"] = backbone
            case = root/"run"/family/backbone
            atomic_json(case/"run_contract.json", contract)
            atomic_json(case/"run_state.json", dict(status="complete", completed_docs=1,
                completed_generations=2, contract_sha256=json_sha(contract)))
            rows = {variant: make_row(backbone, variant, common, doc) for variant in variants}
            atomic_json(case/"samples/000000.json", dict(status="completed", doc_id=0,
                contract_sha256=json_sha(contract), fairness={"status":"passed"}, records=rows))
    return hf_home, arrow, doc


def make_row(backbone, variant, common, doc):
    history = 100 if backbone == "snapkv" or variant == "native" else 104
    count = 112*(history+32)
    shape = [1,4,133,128] if backbone == "snapkv" else [count//28+4,128]
    group = dict(positions=[0,1], group_ids=[0,0], group_size=[2], start_mask=[True,False],
        layer_idx=0, head_idx=0, batch_idx=0, before=2, after=1)
    audit = dict(status="complete", merge_calls=[group], layers=[dict(layer=0,
        candidate_idx=[0,1], final_history=1, candidate_history=2)], selected_masks=[True],
        prefill_cache_shapes=[[1]], merge_mask=[True], budget=dict(
            policy=report.BUDGET_POLICY, native_budget_unchanged=True,
            post_allocation_scaling=False, scale=1., sequence_length=1000,
            layers=28, native_kv_heads=4, budget_ratio=.1,
            original_history_budget=[[[history]*4] for _ in range(28)],
            history_budget=[[[history]*4] for _ in range(28)],
            original_history_positions=112*history, history_positions=112*history,
            local_len=32, local_positions=112*32, active_prefill_positions=count,
            capacity_positions=count, per_layer_active_positions=[4*(history+32)]*28))
    fisher = .5 if variant == "asymkv" else 0.
    return dict(status="completed", task=report.TASK, method=backbone if variant == "native" else backbone+"_kvrepack",
        variant=variant, doc_id=0, text="person", prediction="person", generated_ids=[5,6],
        generated_tokens=2, input_tokens=1000, eos_reached=True, capped_without_eos=False, warmup=False,
        reference_metadata=dict(task=report.TASK, doc_id=0, input_tokens=1000, video_id="0",
            arrow_sha256=common["arrow_sha256"], original_row_sha256=json_sha(doc)),
        input_fingerprints={"input_shapes":{"input_ids":[1,1000]},
            "input_hashes":{"input_ids":"same-question", "pixel_values_videos":"same-video"}},
        audit=audit, fisher={"loss":1.} if variant == "asymkv" else None,
        budget=dict(prefill_tokens_total=count, capacity=count, actual_retained_ratio=count/112000,
            budget_policy=report.BUDGET_POLICY, nominal_total_cap=11200,
            final_shapes=[dict(layer=i,key_shape=shape,value_shape=shape) for i in range(28)]),
        runtime=dict(end_to_end_seconds=3.11+fisher, model_end_to_end_seconds=2.+fisher,
            generation_seconds=2., preprocessing_seconds=1., h2d_seconds=.1,
            text_decode_seconds=.01, fisher_seconds=fisher, peak_allocated_gib=20.))


def fake_score(rows, contract, arrow, output, repo_root):
    return 33., {"status":"test_scoring_stub", "n":len(rows)}


class OriginalNextQAMergeReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hf, self.arrow, self.doc = fixture(self.root)

    def build(self):
        return report.build_report(self.root/"run", self.hf, self.root/"report",
            expected_count=1, repo_root=self.root)

    def mutate(self, relative, operation):
        path = self.root/"run"/relative
        data = json.loads(path.read_text())
        operation(data)
        atomic_json(path, data)

    def reject(self, match):
        with patch.object(report, "score_rows") as scorer:
            with self.assertRaisesRegex(ValueError, match):
                self.build()
        scorer.assert_not_called()
        self.assertFalse((self.root/"report").exists())

    def test_valid_eight_arm_pilot_reports_full_pipeline_and_budget_differences(self):
        with patch.object(report, "score_rows", side_effect=fake_score) as scorer:
            result = self.build()
        self.assertEqual(scorer.call_count, 8)
        self.assertEqual(result["task"], "nextqa_oe_test")
        self.assertEqual(result["scope"], "pilot_first1_original_budget")
        self.assertEqual(result["budget_policy"], report.BUDGET_POLICY)
        self.assertFalse(result["budget_validation"]["strict_total10"])
        self.assertTrue(result["comparisons"][0]["actual_retained_ratio_mean"] > .1)
        self.assertEqual(result["total_generations"], 8)
        self.assertEqual(len(result["comparisons"]), 8)
        self.assertFalse(result["budget_validation"]["hybridkv_identical_prefill_counts"])
        native, asym = result["comparisons"][0], result["comparisons"][3]
        self.assertAlmostEqual(native["end_to_end_seconds_mean"], 3.11)
        self.assertAlmostEqual(asym["end_to_end_seconds_mean"], 3.61)
        self.assertAlmostEqual(asym["model_end_to_end_seconds_mean"], 2.5)
        self.assertTrue((self.root/"report/RESULTS.md").is_file())
        self.assertIn("WUPS", (self.root/"report/comparison.csv").read_text())
        self.assertNotIn("VATEX", (self.root/"report/RESULTS.md").read_text())

    def test_style_only_video_fingerprint_change_is_rejected(self):
        def change(bundle):
            for row in bundle["records"].values():
                row["input_fingerprints"]["input_hashes"]["pixel_values_videos"] = "different-video"
        self.mutate("style/snapkv/samples/000000.json", change)
        self.reject("input mismatch")

    def test_same_wrong_reference_in_every_arm_is_rejected_against_arrow(self):
        for family in report.FAMILIES:
            for backbone in report.BACKBONES:
                def change(bundle):
                    for row in bundle["records"].values():
                        row["reference_metadata"]["original_row_sha256"] = "wrong-row"
                self.mutate(f"{family}/{backbone}/samples/000000.json", change)
        self.reject("exact original reference")

    def test_strict_total_cache_cannot_be_substituted_for_original_snap_capacity(self):
        def change(bundle):
            row = bundle["records"]["lookm"]
            row["budget"].update(prefill_tokens_total=11200, capacity=11200,
                actual_retained_ratio=.1)
            row["budget"]["final_shapes"] = [dict(layer=i,key_shape=[1,4,101,128],
                value_shape=[1,4,101,128]) for i in range(28)]
        self.mutate("style/snapkv/samples/000000.json", change)
        self.reject("original history plus recent budget")

    def test_original_hybrid_allocation_cannot_be_rescaled(self):
        def change(bundle):
            row = bundle["records"]["kvrepack"]
            row["audit"]["budget"].update(scale=.95, post_allocation_scaling=True)
        self.mutate("baseline/hybridkv/samples/000000.json", change)
        self.reject("allocation was changed")

    def test_hybrid_history_arrays_must_match_the_original_allocation(self):
        def change(bundle):
            bundle["records"]["lookm"]["audit"]["budget"]["history_budget"][0][0][0] -= 8
        self.mutate("style/hybridkv/samples/000000.json", change)
        self.reject("historical budgets or local window changed")

    def test_hybrid_capacity_is_checked_against_unscaled_head_budgets(self):
        def change(bundle):
            bundle["records"]["lookm"]["budget"]["prefill_tokens_total"] -= 112
            row=bundle["records"]["lookm"]
            row["budget"]["actual_retained_ratio"] = row["budget"]["prefill_tokens_total"] / 112000
        self.mutate("style/hybridkv/samples/000000.json", change)
        self.reject("physical cache differs from its original unscaled allocation")

    def test_strict_contract_is_rejected_even_if_rows_are_labelled_original(self):
        self.mutate("style/hybridkv/run_contract.json", lambda d:d.update(
            budget_policy="total_active_prefill_including_local_global_cap_v1"))
        contract=json.loads((self.root/"run/style/hybridkv/run_contract.json").read_text())
        self.mutate("style/hybridkv/run_state.json", lambda d:d.update(contract_sha256=json_sha(contract)))
        self.reject("original-budget protocol")

    def test_physical_cache_shapes_are_checked_independently(self):
        def change(bundle):
            row = bundle["records"]["native"]
            row["budget"]["final_shapes"][0]["key_shape"] = [1,4,134,128]
            row["budget"]["final_shapes"][0]["value_shape"] = [1,4,134,128]
        self.mutate("baseline/snapkv/samples/000000.json", change)
        self.reject("actual cache shapes")

    def test_different_merge_routes_are_rejected_even_with_equal_budget(self):
        def change(bundle):
            bundle["records"]["lookm"]["audit"]["merge_calls"][0]["positions"] = [0,2]
        self.mutate("style/hybridkv/samples/000000.json", change)
        self.reject("candidates/heads/groups/anchors")

    def test_generation_only_time_cannot_be_relabelled_full_e2e(self):
        self.mutate("baseline/snapkv/samples/000000.json",
            lambda b: b["records"]["native"]["runtime"].update(end_to_end_seconds=2.))
        self.reject("timing boundaries")

    def test_asymkv_fisher_cost_cannot_be_excluded(self):
        self.mutate("style/snapkv/samples/000000.json",
            lambda b: b["records"]["asymkv"]["runtime"].update(fisher_seconds=0.))
        self.reject("include its Fisher probe")

    def test_changed_numerical_source_is_rejected(self):
        (self.root/report.SHARED_SOURCES[0]).write_text("# changed implementation\n")
        self.reject("source is unavailable or changed")

    def test_changed_model_metadata_is_rejected(self):
        atomic_json(self.root/"model/config.json", {"model_type":"another_model"})
        self.reject("Model metadata differs")

    def test_incomplete_case_is_rejected(self):
        self.mutate("style/hybridkv/run_state.json", lambda d:d.update(status="running"))
        self.reject("complete runs")

    def test_native_scorer_receives_nextqa_task_exact_arrow_and_cpu_only_environment(self):
        case = self.root/"run/baseline/snapkv"
        contract = json.loads((case/"run_contract.json").read_text())
        row = json.loads((case/"samples/000000.json").read_text())["records"]["native"]
        helper = self.root/"provenance/full_video_retest_20261104/helpers/score_full_runs.py"
        helper.parent.mkdir(parents=True)
        helper.write_text("# external scoring fixture\n")
        def invoke(command, **kwargs):
            self.assertEqual(command[command.index("--task")+1], "nextqa_oe_test")
            self.assertEqual(command[command.index("--reference-arrow")+1], str(self.arrow))
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(kwargs["env"]["HF_DATASETS_OFFLINE"], "1")
            predictions = Path(command[command.index("--records")+1])
            corpus = [json.loads(line) for line in predictions.read_text().splitlines()]
            identity = dict(reference_arrow_sha256=contract["arrow_sha256"],
                prediction_corpus_sha256=report._canonical_hash(corpus))
            atomic_json(Path(command[command.index("--output")+1]), dict(status="complete",
                task=report.TASK,n=1,expected_count=1,metrics={"WUPS":100.},
                scoring_identity=identity,scoring_identity_sha256=report._canonical_hash(identity),
                records={"file_sha256":sha256(predictions)}))
        with patch.object(report.subprocess, "run", side_effect=invoke):
            wups, proof = report.score_rows([row], contract, self.arrow, self.root/"scored", self.root)
        self.assertEqual(wups, 100.)
        self.assertEqual(proof["status"], "independently_scored_current_corpus")


if __name__ == "__main__":
    unittest.main()

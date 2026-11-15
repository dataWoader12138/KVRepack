"""Validate original standalone methods and preserve distinct E2E scopes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.benchmarks import legacy_nextqa_report as report
from kvrepack.ablations.run_head_selection import atomic_json, json_sha, sha256


def fixture(root):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    source = root/report.REQUIRED_SOURCES[0]
    source.parent.mkdir(parents=True)
    source.write_text("# frozen input preparation\n")
    task = root/"vendor/lmms-eval/lmms_eval/tasks/nextqa"
    task.mkdir(parents=True)
    (task/"nextqa_oe_test.yaml").write_text("task: nextqa_oe_test\n")
    (task/"utils.py").write_text("# original native scorer fixture\n")
    model = root/"model"
    model.mkdir()
    atomic_json(model/"config.json", dict(model_type="qwen2_5_vl", hidden_size=3584,
        num_hidden_layers=28, num_key_value_heads=4))
    hf = root/"hf"
    arrow = hf/"datasets/lmms-lab___n_ex_tqa/OE/test/n_ex_tqa-test.arrow"
    arrow.parent.mkdir(parents=True)
    original = dict(video="video-0", question="Who is walking?", answer="person",
                    type="CW", additional_ref_answer="")
    table = pa.Table.from_pylist([original])
    with pa.OSFile(str(arrow), "wb") as stream:
        with ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    common = dict(task=report.TASK, target_docs=1, target_generations=1,
        model_snapshot=str(model), model_metadata_sha256={"config.json":sha256(model/"config.json")},
        frames=64, min_pixels=200704, max_pixels=1605632, dtype="bfloat16", max_new_tokens=128,
        do_sample=False, budget_ratio=.1, model_seed=42,
        versions={"fixture":"1"}, arrow_sha256=sha256(arrow), dataset_rows=1,
        task_yaml_sha256=sha256(task/"nextqa_oe_test.yaml"), task_utils_sha256=sha256(task/"utils.py"),
        source_sha256={report.REQUIRED_SOURCES[0]:sha256(source)}, hf_home=str(hf))
    for method in report.METHODS:
        case = root/"run"/method
        contract = dict(common, method=method, budget_policy=report.POLICIES[method],
            repetition_penalty=1.05 if method == "lookm" else None,
            decoding_policy=report.DECODING_POLICIES[method])
        atomic_json(case/"run_contract.json", contract)
        generated = 3. if method == "asymkv" else 2.
        runtime = dict(preprocessing_seconds=1., h2d_seconds=.1,
            generation_seconds=generated, text_decode_seconds=.01,
            end_to_end_seconds=1.11+generated, peak_allocated_gib=20.)
        atomic_json(case/"run_state.json", dict(status="complete", completed_docs=1,
            completed_generations=1, contract_sha256=json_sha(contract),
            timed_e2e_seconds_sum=runtime["end_to_end_seconds"],
            response_loop_seconds=runtime["end_to_end_seconds"]+.2))
        row = dict(status="completed", task=report.TASK, method=method, doc_id=0, warmup=False,
            contract_sha256=json_sha(contract), text="person", prediction="person",
            generated_ids=[5,151645], generated_tokens=2, input_tokens=1000,
            eos_token_id=151645, eos_reached=True, capped_without_eos=False,
            reference_metadata=dict(task=report.TASK, doc_id=0, video_id="video-0",
                source_video_id="video-0", input_tokens=1000, frame_indices=[0]*64, decoded_frames=1,
                arrow_sha256=common["arrow_sha256"], original_row_sha256=json_sha(original)),
            input_fingerprints=dict(input_shapes={"input_ids":[1,1000],"pixel_values_videos":[8,3]},
                input_hashes={"input_ids":"question", "pixel_values_videos":"video"},
                input_dtypes={"input_ids":"torch.int64", "pixel_values_videos":"torch.float32"}),
            runtime=runtime)
        atomic_json(case/"samples/000000.json", row)
    return hf, arrow


def fake_score(rows, contract, arrow, output, repo_root):
    return 33.5, dict(status="test_scoring_stub", n=len(rows))


class LegacyNextQAReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hf, self.arrow = fixture(self.root)

    def mutate(self, relative, change):
        path = self.root/"run"/relative
        value = json.loads(path.read_text())
        change(value)
        atomic_json(path, value)

    def build(self, **kwargs):
        return report.build_report(self.root/"run", self.hf, self.root/"report",
                                   limit=1, repo_root=self.root, **kwargs)

    def reject(self, expression):
        with patch.object(report, "score_rows") as scorer:
            with self.assertRaisesRegex(ValueError, expression):
                self.build()
        scorer.assert_not_called()
        self.assertFalse((self.root/"report").exists())

    def test_original_standalone_report_preserves_timed_and_loop_scopes(self):
        with patch.object(report, "score_rows", side_effect=fake_score) as scorer:
            summary = self.build()
        self.assertEqual(scorer.call_count, 2)
        self.assertEqual(summary["total_generations"], 2)
        self.assertEqual(summary["scope"], "pilot_first1_original_standalone")
        self.assertFalse(summary["gpu_execution"])
        lookm, asym = summary["comparisons"]
        self.assertAlmostEqual(lookm["end_to_end_seconds_mean"], 3.11)
        self.assertAlmostEqual(lookm["response_loop_seconds_per_input"], 3.31)
        self.assertAlmostEqual(lookm["untimed_loop_overhead_seconds"], .2)
        # AsymKV's own full generation includes Fisher; do not double-count it.
        self.assertAlmostEqual(asym["generation_seconds_mean"], 3.)
        self.assertAlmostEqual(asym["end_to_end_seconds_mean"], 4.11)
        self.assertEqual(lookm["budget_policy"], "original_lookm_history_plus_recent_v1")
        self.assertEqual(asym["budget_policy"], "original_asymkv_official_v1")
        self.assertEqual(lookm["repetition_penalty"], 1.05)
        self.assertIsNone(asym["repetition_penalty"])
        self.assertTrue((self.root/"report/per_input_latency.csv").is_file())
        text = (self.root/"report/RESULTS.md").read_text()
        self.assertIn("Formal loop / input", text)
        self.assertNotIn("VATEX", text)

    def test_a_single_completed_method_can_be_scored_as_an_explicit_pilot(self):
        with patch.object(report, "score_rows", side_effect=fake_score) as scorer:
            summary = self.build(methods=("asymkv",))
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(summary["methods"], ["asymkv"])
        self.assertEqual(summary["total_generations"], 1)

    def test_strict_lookm_result_is_not_accepted_as_original(self):
        self.mutate("lookm/run_contract.json", lambda row: row.update(budget_policy="total_prefill_including_recent_v2"))
        self.reject("original standalone")

    def test_style_transplant_cannot_be_called_standalone_asymkv(self):
        self.mutate("asymkv/run_contract.json", lambda row: row.update(method="hybridkv_kvrepack"))
        self.reject("original standalone")

    def test_original_asymkv_decode_cannot_silently_add_hf_repetition_penalty(self):
        self.mutate("asymkv/run_contract.json", lambda row: row.update(repetition_penalty=1.05))
        self.reject("original standalone")

    def test_incomplete_generation_is_rejected(self):
        self.mutate("asymkv/run_state.json", lambda row: row.update(completed_docs=0))
        self.reject("complete method")

    def test_out_of_scope_or_extra_sample_is_rejected(self):
        path = self.root/"run/lookm/samples/000000.json"
        (path.parent/"000001.json").write_bytes(path.read_bytes())
        self.reject("original input records")

    def test_worker_source_change_is_rejected(self):
        (self.root/report.REQUIRED_SOURCES[0]).write_text("# changed generation pipeline\n")
        self.reject("source is missing or changed")

    def test_model_metadata_change_is_rejected(self):
        path = self.root/"model/config.json"
        value = json.loads(path.read_text()); value["num_hidden_layers"] = 32
        atomic_json(path, value)
        self.reject("Model metadata differs")

    def test_scoring_arrow_change_is_rejected(self):
        with self.arrow.open("ab") as stream:
            stream.write(b"changed")
        self.reject("reference differs")

    def test_wrong_original_video_is_rejected(self):
        self.mutate("lookm/samples/000000.json", lambda row: row["reference_metadata"].update(video_id="other"))
        self.reject("original question/video")

    def test_changed_model_input_between_methods_is_rejected(self):
        def change(row):
            row["input_fingerprints"]["input_hashes"]["pixel_values_videos"] = "different pixels"
        self.mutate("asymkv/samples/000000.json", change)
        self.reject("same original processor input")

    def test_wrong_frame_selection_is_rejected(self):
        self.mutate("lookm/samples/000000.json", lambda row: row["reference_metadata"].update(frame_indices=[1]*64))
        self.reject("64-frame sampling")

    def test_fisher_or_generation_phase_cannot_be_removed_from_total(self):
        self.mutate("asymkv/samples/000000.json", lambda row: row["runtime"].update(generation_seconds=1.))
        self.reject("Full E2E is inconsistent")

    def test_response_loop_cannot_be_shorter_than_contained_e2e(self):
        self.mutate("lookm/run_state.json", lambda row: row.update(response_loop_seconds=1.))
        self.reject("response loop is shorter")

    def test_recorded_formal_timed_sum_must_match_samples(self):
        self.mutate("lookm/run_state.json", lambda row: row.update(timed_e2e_seconds_sum=4.))
        self.reject("E2E sum differs")

    def test_warmup_cannot_be_scored_as_formal_input(self):
        self.mutate("lookm/samples/000000.json", lambda row: row.update(warmup=True))
        self.reject("original row")

    def test_actual_eos_token_is_checked(self):
        self.mutate("lookm/samples/000000.json", lambda row: row.update(generated_ids=[5,6]))
        self.reject("final token and EOS")

    def test_output_length_flag_is_checked(self):
        self.mutate("lookm/samples/000000.json", lambda row: row.update(capped_without_eos=True))
        self.reject("generation-cap flags")

    def test_prediction_evidence_mutation_during_scoring_is_rejected(self):
        def mutate_score(rows, contract, arrow, output, repo_root):
            path = self.root/"run/lookm/samples/000000.json"
            value = json.loads(path.read_text()); value["text"] = value["prediction"] = "changed"
            atomic_json(path, value)
            return fake_score(rows, contract, arrow, output, repo_root)
        with patch.object(report, "score_rows", side_effect=mutate_score):
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                self.build()
        self.assertFalse((self.root/"report/summary.json").exists())


if __name__ == "__main__":
    unittest.main()

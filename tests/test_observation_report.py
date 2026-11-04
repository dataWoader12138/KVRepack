"""CPU inference checks: video clustering, matched estimands, and strict coverage."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.observations.report import (
    build_summary, cluster_statistics, main, read_jsonl, write_report,
)


def record(doc_id=0, task="nextqa_oe_test", video="a", visual=0.6, text=0.2, pairs=1):
    primary = "text_question" if task == "nextqa_oe_test" else "text_instruction"
    return dict(task=task, doc_id=doc_id, video_id=video, source_video_id=video,
                status="completed", input_tokens=100, pair_counts={"visual_spatial": pairs, primary: pairs},
                layer_stats=[dict(layer=layer, kv_heads=2, groups={
                    "visual_spatial": {"k_raw": [visual] * 2, "v": [visual] * 2},
                    primary: {"k_raw": [text] * 2, "v": [text] * 2}}) for layer in range(2)])


class ObservationReportTests(unittest.TestCase):
    def summarize(self, records, count, **kwargs):
        return build_summary(records, expected_count=count, expected_tasks=("nextqa_oe_test",),
                             bootstrap_samples=600, **kwargs)

    def test_cluster_bootstrap_keeps_repeated_questions_together_and_row_weights(self):
        # Twenty questions about A and one about B: row mean differs from video mean.
        rows = [dict(source_video_id="a", visual=1., text=0., difference=1.) for _ in range(20)]
        rows += [dict(source_video_id="b", visual=-1., text=0., difference=-1.)]
        result = cluster_statistics(rows, bootstrap_samples=1000, seed=13)
        self.assertEqual(result["n_records"], 21)
        self.assertEqual(result["n_source_videos"], 2)
        self.assertAlmostEqual(result["difference"], 19 / 21)
        self.assertEqual(result["ci95"]["difference"], [-1., 1.])
        self.assertEqual(result["positive_source_video_count"], 1)
        self.assertEqual(result["positive_source_video_fraction"], .5)

    def test_paired_resampling_does_not_break_visual_text_covariance(self):
        rows = [dict(source_video_id=str(i), visual=v, text=v, difference=0.)
                for i, v in enumerate((-.8, -.1, .4, .9))]
        result = cluster_statistics(rows, bootstrap_samples=1000)
        self.assertEqual(result["ci95"]["difference"], [0., 0.])
        self.assertEqual(result["ci95"]["visual"], result["ci95"]["text"])

    def test_negative_observation_is_retained_without_positive_clipping(self):
        result = self.summarize([record(0, video="a", visual=-.7, text=.5),
                                 record(1, video="b", visual=-.7, text=.5)], 2)
        stats = result["tasks"]["nextqa_oe_test"]["metrics"]["k_raw"]
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(stats["visual"], -.7)
        self.assertAlmostEqual(stats["difference"], -1.2)
        self.assertEqual(stats["positive_source_video_count"], 0)
        self.assertLess(stats["ci95"]["difference"][1], 0)

    def test_document_weights_do_not_follow_pair_count(self):
        result = self.summarize([record(0, video="a", visual=1, text=0, pairs=100),
                                 record(1, video="b", visual=-1, text=0, pairs=1)], 2)
        self.assertEqual(result["tasks"]["nextqa_oe_test"]["metrics"]["v"]["difference"], 0.)

    def test_average_heads_then_layers_before_documents(self):
        row = record()
        row["layer_stats"][0]["groups"]["visual_spatial"]["k_raw"] = [1., -1.]
        row["layer_stats"][1]["groups"]["visual_spatial"]["k_raw"] = [.4, .4]
        result = self.summarize([row], 1)
        self.assertAlmostEqual(result["tasks"]["nextqa_oe_test"]["metrics"]["k_raw"]["visual"], .2)
        self.assertIsNone(result["tasks"]["nextqa_oe_test"]["metrics"]["k_raw"]["ci95"])

    def test_M0_completed_coverage_is_not_an_eligible_pair(self):
        zero = record(1, pairs=0)
        zero["layer_stats"] = []
        result = self.summarize([record(), zero], 2)
        task = result["tasks"]["nextqa_oe_test"]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(task["valid_completed_records"], 2)
        self.assertEqual(task["paired_records"], 1)
        self.assertEqual(len(task["excluded_primary_M0"]), 1)

    def test_first_200_requires_ID_range_completed_and_no_duplicates(self):
        rows = [record(i, video=str(i)) for i in range(200)]
        ok = build_summary(rows + [record(i, "vatex_test", str(i)) for i in range(200)], bootstrap_samples=20)
        self.assertEqual(ok["status"], "complete")
        self.assertEqual(ok["expected_total_records"], 400)
        for broken in (rows[:-1], rows + [deepcopy(rows[0])], rows[:-1] + [record(200)]):
            self.assertEqual(self.summarize(broken, 200)["status"], "incomplete")
        failed = deepcopy(rows)
        failed[3].update(status="failed", error="decode failed")
        summary = self.summarize(failed, 200)
        self.assertEqual(summary["tasks"]["nextqa_oe_test"]["missing_completed_doc_ids"], [3])
        self.assertTrue(any("decode failed" in issue.get("error", "") for issue in summary["validation_issues"]))

    def test_shape_and_pair_contract_rejects_invalid_vectors(self):
        mutations = [
            lambda r: r["pair_counts"].update(text_question=2),
            lambda r: r["layer_stats"].pop(),
            lambda r: r["layer_stats"][0].update(kv_heads=1),
            lambda r: r["layer_stats"][0]["groups"]["text_question"].update(k_raw=[float("nan"), .1]),
            lambda r: r["layer_stats"][0]["groups"]["visual_spatial"].update(k_rope=[.2, .3]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                row = record(1, video="b")
                mutate(row)
                self.assertEqual(self.summarize([record(), row], 2)["status"], "incomplete")

    def test_inferred_dimensions_must_match_every_record_and_explicit_flags(self):
        result = self.summarize([record()], 1, expected_layers=3, expected_kv_heads=2)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["tasks"]["nextqa_oe_test"]["valid_completed_records"], 0)

    def test_optional_rope_is_secondary_and_full_coverage_is_visible(self):
        row = record()
        for layer in row["layer_stats"]:
            for group in layer["groups"].values():
                group["k_rope"] = list(group["k_raw"])
        result = self.summarize([row, record(1, video="b")], 2)
        self.assertEqual(result["tasks"]["nextqa_oe_test"]["metrics"]["k_rope"]["n_records"], 1)
        self.assertEqual(result["tasks"]["nextqa_oe_test"]["metrics"]["k_raw"]["n_records"], 2)

    def test_JSONL_corruption_and_no_plot_CLI_report_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.jsonl"
            source.write_text(json.dumps(record()) + "\n{bad\n")
            rows, errors = read_jsonl(source)
            self.assertEqual(len(rows), 1)
            self.assertEqual(errors[0]["line"], 2)
            self.assertEqual(main(["--input", str(source), "--output", str(root / "report"),
                                   "--bootstrap-samples", "20", "--no-plots"]), 2)
            saved = json.loads((root / "report/summary.json").read_text())
            self.assertEqual(saved["status"], "incomplete")
            self.assertEqual(saved["provenance"]["record_metadata_variants"], [])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib is unavailable")
    def test_plot_and_csv_artifacts_are_standalone(self):
        summary = self.summarize([record(0, video="a"), record(1, video="b")], 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(summary, root)
            for name in ("summary.json", "RESULTS.md", "task_summary.csv", "record_summary.csv", "head_summary.csv",
                         "layer_summary.csv", "nextqa_oe_test.png", "nextqa_oe_test.pdf", "nextqa_oe_test.svg"):
                self.assertGreater((root / name).stat().st_size, 20)


if __name__ == "__main__":
    unittest.main()

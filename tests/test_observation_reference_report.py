"""CPU checks for full-input 5+5 descriptive means and fixed source coverage."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.observations.reference_report import GROUPS, METRICS, build_summary, main, write_report


def record(group, doc_id, mean=.2, length=100):
    inclusive = ((length-1)*mean+1)/length
    return dict(schema_version=1, status="completed", group=group, doc_id=doc_id,
                model_id=GROUPS[group]["model_id"], model_revision="revision-"+group,
                input_tokens=length, image_tokens=0 if group == "text" else 1,
                source_metadata=dict(dataset_id="LongBench-qasper" if group == "text" else "OCRBench",
                                     dataset_revision="fixed-test-revision", doc_id=doc_id, content_sha256="test-content"),
                layer_stats=[dict(layer=layer, kv_heads=4, head_dim=128, sequence_length=length,
                                  capture_counts=dict.fromkeys(METRICS, 1),
                                  inclusive={metric: [inclusive]*4 for metric in METRICS},
                                  **{metric: [mean]*4 for metric in METRICS}) for layer in range(28)])


def records(text=.2, vision=.6, length=100):
    return [record(group, doc_id, text if group == "text" else vision, length)
            for group, data in GROUPS.items() for doc_id in data["doc_ids"]]


class ReferenceReportTests(unittest.TestCase):
    def test_complete_fixed_groups_models_ids_and_no_inference_from_heads(self):
        summary = build_summary(records())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["expected_total_inputs"], 10)
        for group, data in GROUPS.items():
            self.assertEqual(summary["groups"][group]["valid_doc_ids"], list(data["doc_ids"]))
            self.assertEqual(summary["groups"][group]["n_inputs"], 5)
            heads = [row for row in summary["head_table"] if row["group"] == group and row["metric"] == "k_rope"]
            self.assertEqual(len(heads), 112)
            self.assertEqual({row["n_inputs"] for row in heads}, {5})
        self.assertNotIn("ci95", json.dumps(summary))
        self.assertIn("no confidence intervals", summary["uncertainty"])

    def test_negative_cosines_and_opposite_direction_are_retained(self):
        summary = build_summary(records(text=.5, vision=-.7, length=2))
        self.assertEqual(summary["status"], "complete")
        for metric in METRICS:
            self.assertAlmostEqual(summary["comparisons"][metric]["vision"], -.7)
            self.assertAlmostEqual(summary["comparisons"][metric]["vision_minus_text"], -1.2)
        self.assertTrue(any(row["mean"] == -.7 for row in summary["head_table"]))

    def test_inputs_have_equal_weight_despite_different_token_lengths(self):
        rows = records(text=0, vision=0, length=10000)
        rows[0] = record("text", 0, mean=1, length=2)
        summary = build_summary(rows)
        self.assertAlmostEqual(summary["groups"]["text"]["metrics"]["k_rope"]["mean"], .2)
        self.assertAlmostEqual(summary["comparisons"]["k_rope"]["vision_minus_text"], -.2)
        self.assertEqual(summary["groups"]["text"]["input_tokens"], dict(min=2, max=10000, mean=8000.4))
        self.assertEqual(next(row for row in summary["record_table"] if row["group"] == "text" and row["doc_id"] == 0)["ordered_nonself_pairs"], 2)

    def test_heads_then_layers_then_inputs_and_layer_summary(self):
        rows = records(text=0, vision=0)
        for row in rows[:5]:
            row["layer_stats"][0]["k_raw"] = [1., -1., .4, .4]
        summary = build_summary(rows)
        self.assertAlmostEqual(summary["groups"]["text"]["metrics"]["k_raw"]["mean"], .2/28)
        layer = next(row for row in summary["layer_table"] if row["group"] == "text" and row["metric"] == "k_raw" and row["layer"] == 0)
        self.assertAlmostEqual(layer["mean"], .2)
        self.assertEqual(layer["kv_heads"], 4)
        self.assertEqual(layer["n_inputs"], 5)

    def test_missing_duplicate_wrong_group_or_outside_prescribed_ids_fails(self):
        complete = records()
        bad_id = deepcopy(complete)
        bad_id[0]["doc_id"] = 5
        wrong_group = deepcopy(complete)
        wrong_group[0]["group"] = "other"
        for rows in (complete[:-1], complete+[deepcopy(complete[0])], bad_id, wrong_group):
            with self.subTest(length=len(rows)):
                self.assertEqual(build_summary(rows)["status"], "incomplete")

    def test_invalid_dimensions_full_length_capture_or_metadata_do_not_partially_enter(self):
        mutations = [
            lambda row: row["layer_stats"].pop(),
            lambda row: row["layer_stats"][0].update(layer=1),
            lambda row: row["layer_stats"][0].update(kv_heads=28),
            lambda row: row["layer_stats"][0].update(head_dim=64),
            lambda row: row["layer_stats"][0].update(sequence_length=99),
            lambda row: row["layer_stats"][0]["capture_counts"].update(k_raw=0),
            lambda row: row["layer_stats"][0]["capture_counts"].update(k_rope=True),
            lambda row: row["layer_stats"][0].update(v=[.2]*28),
            lambda row: row["layer_stats"][0].update(k_rope=[float("nan")]*4),
            lambda row: row["layer_stats"][0].update(k_raw=[1.1]*4),
            lambda row: row["layer_stats"][0]["inclusive"].pop("v"),
            lambda row: row.update(image_tokens=1),
            lambda row: row.update(model_id=GROUPS["vision"]["model_id"]),
            lambda row: row.update(model_revision="different-revision"),
            lambda row: row.update(source_metadata={}),
            lambda row: row.update(status="failed"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                rows = records()
                mutate(rows[1])
                summary = build_summary(rows)
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["groups"]["text"]["n_inputs"], 4)
                self.assertEqual({row["n_inputs"] for row in summary["head_table"] if row["group"] == "text"}, {4})
                self.assertNotIn(1, [row["doc_id"] for row in summary["record_table"] if row["group"] == "text"])

    def test_source_revisions_metadata_and_token_lengths_are_preserved(self):
        rows = records()
        summary = build_summary(rows)
        self.assertEqual(summary["groups"]["vision"]["model_revision"], "revision-vision")
        self.assertEqual(summary["source_inputs"]["vision"][2]["doc_id"], 400)
        self.assertEqual(summary["source_inputs"]["text"][0]["metadata"], rows[0]["source_metadata"])
        saved = next(row for row in summary["record_table"] if row["group"] == "text")
        self.assertEqual(json.loads(saved["source_metadata_json"]), rows[0]["source_metadata"])

    def test_corrupt_JSONL_incomplete_report_returns_nonzero_and_does_not_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"records.jsonl"
            source.write_text(json.dumps(records()[0])+"\n{bad\n")
            self.assertEqual(main(["--input", str(source), "--output", str(root/"report"), "--no-plots"]), 2)
            summary = json.loads((root/"report/summary.json").read_text())
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["artifacts"]["plots"], "disabled")
            self.assertTrue(any(issue.get("line") == 2 for issue in summary["validation_issues"]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib unavailable")
    def test_standalone_main_raw_figures_tables_and_descriptive_metadata(self):
        summary = build_summary(records())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compact = write_report(summary, root)
            for filename in ("summary.json", "RESULTS.md", "group_summary.csv", "comparison_summary.csv",
                             "record_summary.csv", "head_summary.csv", "layer_summary.csv",
                             "reference_main.png", "reference_main.pdf", "reference_main.svg",
                             "reference_raw.png", "reference_raw.pdf", "reference_raw.svg"):
                self.assertGreater((root/filename).stat().st_size, 20)
            for figure in compact["artifacts"]["plots"].values():
                self.assertEqual(figure["range"], [-1, 1])
                for curve in figure["curves"]:
                    self.assertEqual(curve["n_head_means"], 112)
                    self.assertEqual(sum(curve["bin_counts"]), 112)
            with (root/"comparison_summary.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertNotIn("head_table", compact)


if __name__ == "__main__":
    unittest.main()

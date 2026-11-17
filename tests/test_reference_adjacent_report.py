"""CPU validation of adjacent/reference separation, coverage and aggregation."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.observations.reference_adjacent_report import (
    GROUPS, METRICS, STATISTICS_KIND, build_summary, expected_ids, main, write_report,
)


def record(group, doc_id, mean=.2, length=100):
    return dict(schema_version=1, status="completed", statistics_kind=STATISTICS_KIND,
                group=group, doc_id=doc_id, model_id=GROUPS[group]["model_id"],
                model_revision="revision-"+group, input_tokens=length,
                adjacent_pair_count=length-1, image_tokens=0 if group == "text" else 1,
                source_metadata=dict(dataset_id="qasper" if group == "text" else "OCRBench",
                                     context_sha256=f"context-{doc_id}", original_image_sha256=f"image-{doc_id}"),
                layer_stats=[dict(layer=layer, kv_heads=4, head_dim=128, sequence_length=length,
                                  adjacent_pair_count=length-1,
                                  capture_counts=dict.fromkeys(METRICS, 1),
                                  **{metric: [mean]*4 for metric in METRICS}) for layer in range(28)])


def records(sample_count=2, text=.2, vision=.6, length=100):
    return [record(group, doc_id, text if group == "text" else vision, length)
            for group, ids in expected_ids(sample_count).items() for doc_id in ids]


class ReferenceAdjacentReportTests(unittest.TestCase):
    def test_same_formal_source_prefixes_and_all_adjacent_pair_counts(self):
        self.assertEqual(expected_ids(2), {"text": (0, 1), "vision": (0, 5)})
        self.assertEqual(expected_ids(200)["vision"], tuple(range(0, 1000, 5)))
        for count in (False, 0, 201, 2.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                expected_ids(count)
        summary = build_summary(records(), sample_count=2)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["statistics_kind"], STATISTICS_KIND)
        self.assertFalse(summary["token_sampling"])
        self.assertEqual(summary["groups"]["text"]["adjacent_pairs"]["total"], 198)
        self.assertEqual({row["adjacent_pair_count"] for row in summary["record_table"]}, {99})
        self.assertFalse(summary["bootstrap"]["enabled"])

    def test_explicitly_rejects_all_pair_or_earlier_sampled_neighbor_statistics(self):
        for kind in (None, "all_pairs", "sampled_adjacent", "all_sequence_adjacent_v2"):
            with self.subTest(kind=kind):
                rows = records()
                rows[0]["statistics_kind"] = kind
                summary = build_summary(rows, sample_count=2)
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["groups"]["text"]["n_inputs"], 1)
        for pair_count in (0, 16, 98, 100, 9900, True):
            with self.subTest(pair_count=pair_count):
                rows = records()
                rows[0]["adjacent_pair_count"] = pair_count
                self.assertEqual(build_summary(rows, sample_count=2)["status"], "incomplete")

    def test_equal_input_weighting_not_weighted_by_sequence_length_or_pair_count(self):
        rows = records(text=0, vision=0, length=10000)
        rows[0] = record("text", 0, mean=1, length=2)
        summary = build_summary(rows, sample_count=2)
        self.assertAlmostEqual(summary["groups"]["text"]["metrics"]["k_rope"]["mean"], .5)
        self.assertAlmostEqual(summary["comparisons"]["k_rope"]["vision_minus_text"], -.5)
        self.assertEqual(summary["groups"]["text"]["adjacent_pairs"]["total"], 10000)
        self.assertEqual(summary["groups"]["text"]["input_tokens"]["mean"], 5001)
        for head in (r for r in summary["head_table"] if r["group"] == "text"):
            self.assertAlmostEqual(head["mean"], .5)

    def test_native_heads_and_layers_are_preserved_and_averaged_equally(self):
        rows = records(text=0, vision=0)
        for row in rows[:2]:
            row["layer_stats"][0]["k_raw"] = [1., -1., .4, .4]
        summary = build_summary(rows, sample_count=2)
        self.assertAlmostEqual(summary["groups"]["text"]["metrics"]["k_raw"]["mean"], .2/28)
        for group in GROUPS:
            heads = [r for r in summary["head_table"] if r["group"] == group and r["metric"] == "k_raw"]
            self.assertEqual(len(heads), 112)
            self.assertEqual({r["n_inputs"] for r in heads}, {2})
        layer = next(r for r in summary["layer_table"] if r["group"] == "text" and r["layer"] == 0 and r["metric"] == "k_raw")
        self.assertAlmostEqual(layer["mean"], .2)
        self.assertEqual(layer["kv_heads"], 4)

    def test_negative_values_and_unfavorable_direction_are_retained(self):
        summary = build_summary(records(text=.5, vision=-.7, length=2), sample_count=2)
        self.assertEqual(summary["status"], "complete")
        for metric in METRICS:
            self.assertAlmostEqual(summary["comparisons"][metric]["vision"], -.7)
            self.assertAlmostEqual(summary["comparisons"][metric]["vision_minus_text"], -1.2)

    def test_invalid_complete_record_never_partially_enters_an_aggregate(self):
        mutations = [
            lambda r: r["layer_stats"].pop(),
            lambda r: r["layer_stats"][0].update(layer=1),
            lambda r: r["layer_stats"][0].update(kv_heads=28),
            lambda r: r["layer_stats"][0].update(head_dim=64),
            lambda r: r["layer_stats"][0].update(head_dim=128.0),
            lambda r: r["layer_stats"][0].update(sequence_length=99),
            lambda r: r["layer_stats"][0].update(adjacent_pair_count=16),
            lambda r: r["layer_stats"][0].pop("adjacent_pair_count"),
            lambda r: r["layer_stats"][0]["capture_counts"].update(k_raw=0),
            lambda r: r["layer_stats"][0]["capture_counts"].update(k_rope=True),
            lambda r: r["layer_stats"][0].update(v=[.2]*28),
            lambda r: r["layer_stats"][0].update(k_rope=[float("nan")]*4),
            lambda r: r["layer_stats"][0].update(k_raw=[1.1]*4),
            lambda r: r.update(image_tokens=1),
            lambda r: r.update(model_id=GROUPS["vision"]["model_id"]),
            lambda r: r.update(model_revision="other-revision"),
            lambda r: r.update(source_metadata={}),
            lambda r: r.update(status="failed"),
        ]
        for mutate in mutations:
            rows = records()
            mutate(rows[1])
            with self.subTest(mutation=mutate):
                summary = build_summary(rows, sample_count=2)
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["groups"]["text"]["n_inputs"], 1)
                self.assertEqual({r["n_inputs"] for r in summary["head_table"] if r["group"] == "text"}, {1})

    def test_wrong_ids_duplicate_missing_rows_and_group_fail(self):
        correct = records()
        for mutation in ("wrong_id", "wrong_group", "duplicate", "missing"):
            rows = deepcopy(correct)
            if mutation == "wrong_id":
                rows[-1]["doc_id"] = 200
            elif mutation == "wrong_group":
                rows[-1]["group"] = "other"
            elif mutation == "duplicate":
                rows.append(deepcopy(rows[0]))
            else:
                rows.pop()
            with self.subTest(mutation=mutation):
                self.assertEqual(build_summary(rows, sample_count=2)["status"], "incomplete")

    def test_formal_200_coverage_bootstraps_sources_and_preserves_pairs(self):
        summary = build_summary(records(sample_count=200), sample_count=200, bootstrap_samples=20)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["expected_total_inputs"], 400)
        self.assertTrue(summary["bootstrap"]["enabled"])
        for group in GROUPS:
            self.assertEqual(summary["groups"][group]["n_inputs"], 200)
            self.assertEqual(summary["groups"][group]["n_source_clusters"], 200)
            self.assertEqual(summary["groups"][group]["adjacent_pairs"]["total"], 19800)
        for metric in METRICS:
            self.assertAlmostEqual(summary["comparisons"][metric]["ci95"][0], .4)
        rows = records(sample_count=200)
        rows[0]["source_metadata"].pop("context_sha256")
        self.assertEqual(build_summary(rows, sample_count=200, bootstrap_samples=10)["status"], "incomplete")

    def test_no_head_pseudoreplication_when_sources_are_not_independent(self):
        rows = records(sample_count=200)
        for row in rows[:200]:
            row["source_metadata"]["context_sha256"] = "one-shared-context"
        summary = build_summary(rows, sample_count=200, bootstrap_samples=10)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["groups"]["text"]["n_source_clusters"], 1)
        for metric in METRICS:
            self.assertIsNone(summary["groups"]["text"]["metrics"][metric]["ci95"])
            self.assertIsNone(summary["comparisons"][metric]["ci95"])

    def test_cli_validates_both_input_file_groups_and_corrupt_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text, vision = root/"text.jsonl", root/"vision.jsonl"
            rows = records()
            text.write_text("".join(json.dumps(r)+"\n" for r in rows[:2]))
            vision.write_text("".join(json.dumps(r)+"\n" for r in rows[2:]))
            args = ["--text-input", str(text), "--vision-input", str(vision), "--sample-count", "2", "--output", str(root/"report"), "--no-plots"]
            self.assertEqual(main(args), 0)
            summary = json.loads((root/"report/summary.json").read_text())
            self.assertEqual(set(summary["provenance"]["input_files"]), {"text", "vision"})
            with (root/"report/record_summary.csv").open() as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), 12)
            self.assertEqual({int(r["adjacent_pair_count"]) for r in saved}, {99})
            self.assertIn("2+2", (root/"report/RESULTS.md").read_text())
            vision.write_text(json.dumps(rows[0])+"\n{bad\n")
            self.assertEqual(main(args), 2)
            summary = json.loads((root/"report/summary.json").read_text())
            self.assertEqual(summary["artifacts"]["plots"], "disabled")
            self.assertTrue(any(r.get("line") == 2 for r in summary["validation_issues"]))
            self.assertTrue(any("wrong model group" in r["error"] for r in summary["validation_issues"]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib unavailable")
    def test_figures_reuse_kde_and_keep_adjacent_assets_separate(self):
        summary = build_summary(records(), sample_count=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A caller may have a reference file nearby; no old asset is overwritten.
            sentinel = root/"reference_main.png"
            sentinel.write_bytes(b"old-reference-must-survive")
            compact = write_report(summary, root)
            for stem in ("main", "raw", "key", "value", "combined"):
                for extension in ("png", "pdf", "svg"):
                    self.assertGreater((root/f"adjacent_{stem}.{extension}").stat().st_size, 20)
            self.assertEqual(sentinel.read_bytes(), b"old-reference-must-survive")
            for metadata in compact["artifacts"]["plots"].values():
                self.assertEqual(metadata["statistics_kind"], STATISTICS_KIND)
                for curve in metadata["curves"]:
                    self.assertEqual(curve["n_head_means"], 112)
                    self.assertAlmostEqual(curve["density_area"], 1)
            combined = compact["artifacts"]["plots"]["combined"]
            self.assertEqual(combined["legend"]["location"], "outside_above_data_axes")
            self.assertGreater(combined["legend"]["bounds_in_axes_coordinates"][1], 1.)
            self.assertTrue(combined["legend"]["below_title"])
            # The independent layout must retain exactly the KDE values used
            # in the existing key/value views for the same 112 points.
            for metric, view in (("k_rope", "key"), ("v", "value")):
                original = compact["artifacts"]["plots"][view]["curves"]
                actual = [curve for curve in combined["curves"] if curve["metric"] == metric]
                self.assertEqual(actual, original)
            svg = (root/"adjacent_main.svg").read_text()
            for label in ("Key Similarity Distribution Comparison", "Value Similarity Distribution Comparison", "Qwen2.5-7B", "Qwen2.5-VL-7B", "Cosine Similarity"):
                self.assertIn(label, svg)
            self.assertFalse(any(p.name.startswith(".adjacent-plots-") for p in root.iterdir()))


if __name__ == "__main__":
    unittest.main()

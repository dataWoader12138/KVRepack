"""CPU checks for the all-pairs estimand, head distributions and strict coverage."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kvrepack.observations.all_pairs_report import (
    build_summary, main, METRICS, POOLS, write_report,
)


def record(doc_id=0, task="nextqa_oe_test", source="a", mm=.3, text=.6):
    conditions = {}
    for condition in POOLS:
        pools = ({"prompt_all": list(range(6)), "visual_all": [0, 1, 2],
                  "text_primary": [3, 4], "natural_user": [3, 4, 5]}
                 if condition == "multimodal" else
                 {"prompt_all": list(range(4)), "text_primary": [1, 2], "natural_user": [1, 2, 3]})
        layers = []
        matched_pools = ({"visual_matched": [[0, 1], [1, 2]]*16, "text_matched": [[3, 4]]*32}
                         if condition == "multimodal" else {"text_matched": [[1, 2]]*32})
        for index in range(2):
            all_pairs = {}
            for pool, positions in pools.items():
                number = mm if condition == "multimodal" else text
                count = len(positions)
                all_pairs[pool] = dict(token_count=count, pair_count=count*(count-1), status="supported",
                                       capture_counts=dict.fromkeys(METRICS, 1),
                                       **{metric: [number]*2 for metric in METRICS})
            layers.append(dict(layer=index, kv_heads=2, sequence_length=len(pools["prompt_all"]),
                               capture_counts=dict.fromkeys(METRICS, 1), all_pairs=all_pairs,
                               matched_all_pairs={name: dict(token_count=2, draw_count=32, pair_count_per_draw=2,
                                                            status="supported",
                                                            capture_counts=dict.fromkeys(METRICS, 1),
                                                            draw_values={metric: [[number]*2 for _ in range(32)] for metric in METRICS},
                                                            **{metric: [number]*2 for metric in METRICS}) for name in matched_pools}))
        conditions[condition] = dict(input_tokens=len(pools["prompt_all"]), pools=pools,
                                     matched_pools=matched_pools, layer_stats=layers)
    return dict(schema_version=1, status="completed", task=task, doc_id=doc_id,
                source_video_id=source, model="synthetic-test-model", frames=64, conditions=conditions)


class AllPairsReportTests(unittest.TestCase):
    def summary(self, rows, count, **kwargs):
        return build_summary(rows, expected_limit=count, expected_tasks=("nextqa_oe_test",),
                             bootstrap_samples=300, **kwargs)

    def test_negative_means_and_differences_remain_negative(self):
        result = self.summary([record(0, source="a", mm=-.1, text=.6),
                               record(1, source="b", mm=-.1, text=.6)], 2)
        self.assertEqual(result["status"], "complete")
        stats = result["tasks"]["nextqa_oe_test"]["comparisons"]["full_prompt"]["k_rope"]
        self.assertAlmostEqual(stats["left"], -.1)
        self.assertAlmostEqual(stats["difference"], -.7)
        self.assertLess(stats["ci95"]["difference"][1], 0)
        self.assertTrue(any(row["mean"] < 0 for row in result["head_table"]))

    def test_head_distribution_has_one_point_per_layer_and_native_head(self):
        rows = [record(i, source=str(i), mm=.1*i) for i in range(4)]
        result = self.summary(rows, 4)
        points = [row for row in result["head_table"] if row["condition"] == "multimodal"
                  and row["pool"] == "visual_all" and row["metric"] == "k_raw"]
        self.assertEqual(len(points), 2*2)
        self.assertEqual({row["n_records"] for row in points}, {4})
        self.assertAlmostEqual(points[0]["mean"], .15)
        self.assertEqual(result["dimensions"], {"layers": 2, "native_kv_heads": 2})

    def test_rows_are_equal_weight_and_repeated_video_rows_resample_together(self):
        rows = [record(i, source="a", mm=1, text=0) for i in range(20)]
        rows.append(record(20, source="b", mm=-.2, text=0))
        result = self.summary(rows, 21)
        stats = result["tasks"]["nextqa_oe_test"]["comparisons"]["full_prompt"]["v"]
        self.assertEqual(stats["n_source_videos"], 2)
        self.assertAlmostEqual(stats["difference"], 19.8/21)
        self.assertEqual(stats["ci95"]["difference"], [-.2, 1.])
        self.assertEqual(stats["positive_source_video_count"], 1)

    def test_paired_draws_preserve_covariance_for_all_comparisons(self):
        rows = [record(i, source=str(i), mm=x, text=x) for i, x in enumerate((-.1, .2, .7))]
        result = self.summary(rows, 3)
        for metrics in result["tasks"]["nextqa_oe_test"]["comparisons"].values():
            for stats in metrics.values():
                self.assertEqual(stats["difference"], 0)
                self.assertEqual(stats["ci95"]["difference"], [0, 0])
                self.assertEqual(stats["ci95"]["left"], stats["ci95"]["right"])

    def test_single_source_has_no_spurious_bootstrap_interval(self):
        result = self.summary([record(0), record(1)], 2)
        for metrics in result["tasks"]["nextqa_oe_test"]["comparisons"].values():
            self.assertTrue(all(stats["ci95"] is None for stats in metrics.values()))

    def test_heads_then_layers_then_rows_average_without_token_count_weighting(self):
        row = record(mm=0, text=0)
        layers = row["conditions"]["multimodal"]["layer_stats"]
        layers[0]["all_pairs"]["visual_all"]["k_raw"] = [1, -1]
        layers[1]["all_pairs"]["visual_all"]["k_raw"] = [.4, .4]
        result = self.summary([row], 1)
        stats = result["tasks"]["nextqa_oe_test"]["comparisons"]["within_multimodal"]["k_raw"]
        self.assertAlmostEqual(stats["left"], .2)
        self.assertEqual(stats["right"], 0)

    def test_exact_first_200_per_task_and_no_duplicates(self):
        rows = [record(i, source=str(i)) for i in range(200)]
        complete = build_summary(rows+[record(i, task="vatex_test", source=str(i)) for i in range(200)], bootstrap_samples=20)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["expected_total_records"], 400)
        for broken in (rows[:-1], rows+[deepcopy(rows[0])], rows[:-1]+[record(200)]):
            with self.subTest(length=len(broken)):
                result = self.summary(broken, 200)
                self.assertEqual(result["status"], "incomplete")

    def test_invalid_shape_value_pool_or_capture_cannot_partially_contribute(self):
        def group(row):
            return row["conditions"]["multimodal"]["layer_stats"][0]["all_pairs"]["visual_all"]
        mutations = [
            lambda row: row["conditions"].pop("text_only"),
            lambda row: row["conditions"]["multimodal"]["layer_stats"].pop(),
            lambda row: row["conditions"]["text_only"]["layer_stats"][0].update(kv_heads=4),
            lambda row: group(row).update(k_raw=[float("nan"), .1]),
            lambda row: group(row).update(v=[float("inf"), .1]),
            lambda row: group(row).update(k_rope=[1.01, .1]),
            lambda row: group(row).update(k_raw=[.1]),
            lambda row: group(row).update(status="unsupported_t_lt_2"),
            lambda row: group(row).update(token_count=1),
            lambda row: group(row).update(pair_count=3),
            lambda row: group(row)["capture_counts"].update(k_raw=2),
            lambda row: row["conditions"]["multimodal"]["pools"].update(visual_all=[0, 0, 1]),
            lambda row: row["conditions"]["multimodal"]["pools"].update(visual_all=[0, 1, 6]),
            lambda row: row["conditions"]["text_only"]["pools"].update(text_primary=[1]),
            lambda row: row["conditions"]["text_only"]["pools"].update(prompt_all=[0, 1, 2]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad = record(1, source="b")
                mutate(bad)
                result = self.summary([record(0), bad], 2, expected_layers=2, expected_kv_heads=2)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["tasks"]["nextqa_oe_test"]["valid_completed_records"], 1)
                self.assertEqual({row["n_records"] for row in result["head_table"]}, {1})
                self.assertEqual({row["doc_id"] for row in result["record_table"]}, {0})

    def test_context_comparison_requires_matched_text_token_counts(self):
        row = record()
        condition = row["conditions"]["text_only"]
        condition["pools"]["text_primary"] = [1, 2, 3]
        for layer in condition["layer_stats"]:
            layer["all_pairs"]["text_primary"].update(token_count=3, pair_count=6)
        result = self.summary([row], 1)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("token count" in issue["error"] for issue in result["validation_issues"]))

    def test_matched_draw_integrity_and_mean_checks(self):
        def group(row):
            return row["conditions"]["multimodal"]["layer_stats"][0]["matched_all_pairs"]["visual_matched"]
        mutations = [
            lambda row: group(row).update(draw_count=31),
            lambda row: group(row).update(token_count=3),
            lambda row: group(row).update(pair_count_per_draw=6),
            lambda row: group(row).update(k_raw=[.8, .8]),
            lambda row: group(row)["draw_values"]["v"].pop(),
            lambda row: group(row)["draw_values"]["k_rope"][0].append(.2),
            lambda row: row["conditions"]["multimodal"]["matched_pools"]["visual_matched"][0].__setitem__(1, 4),
            lambda row: row["conditions"]["multimodal"]["matched_pools"]["text_matched"][0].__setitem__(1, 3),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                row = record()
                mutate(row)
                self.assertEqual(self.summary([row], 1)["status"], "incomplete")

    def test_matched_draws_average_within_head_before_distributions_and_bootstrap(self):
        row = record(mm=.2, text=.4)
        for layer in row["conditions"]["multimodal"]["layer_stats"]:
            group = layer["matched_all_pairs"]["visual_matched"]
            group["k_rope"] = [0., .5]
            group["draw_values"]["k_rope"] = [[-.5, .2], [.5, .8]]*16
        result = self.summary([row], 1)
        stats = result["tasks"]["nextqa_oe_test"]["comparisons"]["matched_visual_vs_text"]["k_rope"]
        self.assertAlmostEqual(stats["left"], .25)
        self.assertAlmostEqual(stats["right"], .2)
        self.assertAlmostEqual(stats["difference"], .05)
        points = [point for point in result["head_table"] if point["pool"] == "visual_matched" and point["metric"] == "k_rope"]
        self.assertEqual(len(points), 4)
        self.assertTrue(all(point["n_records"] == 1 for point in points))

    def test_context_matched_draws_preserve_identical_relative_text_indices(self):
        row = record()
        for condition, positions in (("multimodal", [3, 4, 5]), ("text_only", [0, 1, 2])):
            value = row["conditions"][condition]
            value["pools"]["text_primary"] = positions
            value["pools"]["natural_user"] = positions
            value["matched_pools"]["text_matched"] = [positions[:2] for _ in range(32)]
            for layer in value["layer_stats"]:
                for pool in ("text_primary", "natural_user"):
                    layer["all_pairs"][pool].update(token_count=3, pair_count=6)
        self.assertEqual(self.summary([row], 1, matched_token_cap=2)["status"], "complete")
        row["conditions"]["text_only"]["matched_pools"]["text_matched"][0] = [1, 2]
        result = self.summary([row], 1, matched_token_cap=2)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("primary-token indices" in issue["error"] for issue in result["validation_issues"]))

    def test_matching_token_count_is_also_limited_by_visual_pool(self):
        row = record()
        row["conditions"]["multimodal"]["pools"]["visual_all"] = [0, 1]
        row["conditions"]["multimodal"]["matched_pools"]["visual_matched"] = [[0, 1] for _ in range(32)]
        for condition, positions in (("multimodal", [3, 4, 5]), ("text_only", [1, 2, 3])):
            value = row["conditions"][condition]
            value["pools"]["text_primary"] = positions
            for layer in value["layer_stats"]:
                layer["all_pairs"]["text_primary"].update(token_count=3, pair_count=6)
                if condition == "multimodal":
                    layer["all_pairs"]["visual_all"].update(token_count=2, pair_count=2)
        self.assertEqual(self.summary([row], 1)["status"], "complete")

    def test_JSONL_corruption_and_incomplete_CLI_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"records.jsonl"
            source.write_text(json.dumps(record())+"\n{bad\n")
            self.assertEqual(main(["--input", str(source), "--output", str(root/"report"),
                                   "--expected-limit", "1", "--bootstrap-samples", "20", "--no-plots"]), 2)
            saved = json.loads((root/"report/summary.json").read_text())
            self.assertEqual(saved["status"], "incomplete")
            self.assertEqual(saved["provenance"]["input_sha256"], __import__("hashlib").sha256(source.read_bytes()).hexdigest())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib unavailable")
    def test_standalone_plot_csv_and_metadata_outputs(self):
        summary = self.summary([record(0), record(1, source="b")], 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(summary, root)
            for filename in ("summary.json", "RESULTS.md", "record_summary.csv", "record_comparisons.csv",
                             "head_summary.csv", "layer_summary.csv", "task_summary.csv",
                             "nextqa_oe_test.png", "nextqa_oe_test.pdf", "nextqa_oe_test.svg",
                             "nextqa_oe_test_raw.png", "nextqa_oe_test_supp_full_prompt.pdf", "nextqa_oe_test_supp_text_context.svg"):
                self.assertGreater((root/filename).stat().st_size, 20)
            saved = json.loads((root/"summary.json").read_text())
            self.assertNotIn("head_table", saved)
            self.assertEqual(saved["artifacts"]["histogram_bins"]["minimum"], -1)
            self.assertIn("not an exact paper reproduction", saved["interpretation"])


if __name__ == "__main__":
    unittest.main()

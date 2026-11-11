"""Contract-driven two-arm reporting and immutable parent-prefix evidence."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.ablations import merge_comparison_report as report
from kvrepack.ablations.run_head_selection import json_sha, sha256


def contract(variants=("lookm", "asymkv")):
    return dict(method="snapkv_kvrepack", variants=list(variants), target_docs=200,
                diagnostic_audit=False, warmup_per_arm=2, timing_scope="fixture timing scope")


def row(variant, doc_id):
    return dict(doc_id=doc_id, variant=variant, warmup=False, text=f"caption {doc_id}",
                input_fingerprints={"tokens": "same"}, reference_metadata={"row": doc_id},
                audit={"sig": {"groups": [0, 0, 1], "heads": [1]}},
                budget={"prefill_tokens_total": 10, "capacity": 10, "actual_retained_ratio": .1},
                generated_tokens=5, eos_reached=True, capped_without_eos=False,
                runtime={"end_to_end_seconds": 3., "preprocessing_seconds": 1.,
                         "h2d_seconds": .1, "fisher_seconds": .5 if variant == "asymkv" else 0.,
                         "generation_seconds": 1., "text_decode_seconds": .01,
                         "model_end_to_end_seconds": 1.5 if variant == "asymkv" else 1.,
                         "peak_allocated_gib": 20.})


def bundle(spec, doc_id=0):
    return dict(doc_id=doc_id, contract_sha256=json_sha(spec), fairness={"status": "passed"},
                records={variant: row(variant, doc_id) for variant in spec["variants"]})


class MergeOnlyReportTests(unittest.TestCase):
    def setUp(self):
        self.pairing = patch.object(report, "pairing_signature", side_effect=lambda audit, method: audit["sig"])
        self.pairing.start()

    def tearDown(self):
        self.pairing.stop()

    def test_two_arms_use_first_declared_arm_without_a_baseline(self):
        spec = contract()
        rows = report.validate_bundle(bundle(spec), spec, 0)
        self.assertEqual(tuple(rows), ("lookm", "asymkv"))
        self.assertNotIn("kvrepack", rows)
        self.assertEqual(report.contract_variants(contract(("asymkv", "lookm"))), ("asymkv", "lookm"))

    def test_actual_pairing_is_rechecked_even_when_saved_fairness_passed(self):
        spec = contract()
        data = bundle(spec)
        data["records"]["asymkv"]["audit"]["sig"]["groups"] = [0, 1, 1]
        with self.assertRaisesRegex(ValueError, "actual-group"):
            report.validate_bundle(data, spec, 0)
        data = bundle(spec)
        data["records"]["kvrepack"] = row("kvrepack", 0)
        with self.assertRaisesRegex(ValueError, "declared-arm"):
            report.validate_bundle(data, spec, 0)

    def test_parent_reuse_requires_unchanged_rows_and_verified_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = contract(("kvrepack", "lookm", "asymkv"))
            parent = bundle(original)
            contract_path, sample_path, source_path = root/"contract.json", root/"sample.json", root/"source_manifest.json"
            contract_path.write_text(json.dumps(original))
            sample_path.write_text(json.dumps(parent))
            source_path.write_text('{"original_frozen_source": "sha"}')
            spec = contract()
            reused = bundle(spec)
            reused["records"] = {v: copy.deepcopy(parent["records"][v]) for v in spec["variants"]}
            reused["reused_from_parent"] = dict(
                parent_bundle_path=str(sample_path), parent_bundle_sha256=sha256(sample_path),
                parent_contract_path=str(contract_path), parent_contract_sha256=json_sha(original),
                parent_contract_file_sha256=sha256(contract_path),
                source_snapshot_path=str(source_path), source_snapshot_sha256=sha256(source_path))
            report.validate_bundle(reused, spec, 0)
            mutated = copy.deepcopy(reused)
            mutated["records"]["lookm"]["runtime"]["end_to_end_seconds"] += .1
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                report.validate_bundle(mutated, spec, 0)
            source_path.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "snapshot SHA"):
                report.validate_bundle(reused, spec, 0)

    def test_stopped_parent_prefix_is_separate_three_arm_latency_not_200_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root/"snapkv_kvrepack"
            (case/"samples").mkdir(parents=True)
            original = contract(("kvrepack", "lookm", "asymkv"))
            (case/"run_contract.json").write_text(json.dumps(original))
            for doc_id in range(2):
                (case/"samples"/f"{doc_id:06d}.json").write_text(json.dumps(bundle(original, doc_id)))
            data = report.parent_prefix_latency(root)
            self.assertEqual(data["n"], 2)
            self.assertFalse(data["quality_scored"])
            self.assertEqual(len(data["comparisons"]), 3)
            self.assertEqual({r["n"] for r in data["comparisons"]}, {2})
            self.assertTrue(all(r["scope"] == "parent_matched_prefix_only" for r in data["comparisons"]))
            (case/"samples"/"000001.json").rename(case/"samples"/"000002.json")
            with self.assertRaisesRegex(ValueError, "consecutive"):
                report.parent_prefix_latency(root)

    def test_parent_quality_scores_only_its_three_matched_arms_and_own_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root/"snapkv_kvrepack"
            (case/"samples").mkdir(parents=True)
            original = contract(("kvrepack", "lookm", "asymkv"))
            (case/"run_contract.json").write_text(json.dumps(original))
            for doc_id in range(2):
                (case/"samples"/f"{doc_id:06d}.json").write_text(json.dumps(bundle(original, doc_id)))
            parent_latency = report.parent_prefix_latency(root)
            immutable_latency = copy.deepcopy(parent_latency)
            observed = []
            def scorer(rows, spec, output, repo_root):
                observed.append((Path(output).name, [r["doc_id"] for r in rows]))
                self.assertEqual(json_sha(spec), json_sha(original))
                return {"Bleu_4": .2, "METEOR": .3, "ROUGE_L": .4, "CIDEr": .5}
            with patch.object(report, "score_native_rows", side_effect=scorer):
                scored = report.score_parent_prefix(parent_latency, root/"scores", root)
            self.assertEqual(observed, [(v, [0, 1]) for v in original["variants"]])
            self.assertEqual(parent_latency, immutable_latency)
            self.assertTrue(scored["quality_scored"])
            self.assertEqual(scored["quality_reference_docs"], 2)
            self.assertEqual({r["n"] for r in scored["comparisons"]}, {2})
            self.assertEqual({r["quality_corpus_scope"] for r in scored["comparisons"]}, {"parent_original_first_2_only"})
            self.assertTrue(all(abs(r["VATEX_Avg"] - .35) < 1e-12 for r in scored["comparisons"]))

    def test_complete_report_keeps_main_four_200_rows_and_parent_three_rows_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_root, parent_root = root/"new", root/"parent"
            def save_case(run_root, method, variants, count):
                case = run_root/method
                (case/"samples").mkdir(parents=True)
                spec = dict(contract(variants), method=method)
                (case/"run_contract.json").write_text(json.dumps(spec))
                (case/"run_state.json").write_text(json.dumps(dict(status="complete", completed_docs=count)))
                for doc_id in range(count):
                    (case/"samples"/f"{doc_id:06d}.json").write_text(json.dumps(bundle(spec, doc_id)))
            for method in ("hybridkv_kvrepack", "snapkv_kvrepack"):
                save_case(main_root, method, ("lookm", "asymkv"), 200)
            save_case(parent_root, "snapkv_kvrepack", ("kvrepack", "lookm", "asymkv"), 2)
            scored_counts = []
            def scorer(rows, spec, output, repo_root):
                scored_counts.append((len(rows), Path(output).name))
                return {"Bleu_4": .2, "METEOR": .3, "ROUGE_L": .4, "CIDEr": .5}
            with patch.object(report, "score_native_rows", side_effect=scorer), patch.object(report, "plot_latencies") as plot:
                report.main(["--run-root", str(main_root), "--output", str(root/"report"),
                             "--parent-run-root", str(parent_root), "--score-parent-prefix"])
            with (root/"report/comparison.csv").open(encoding="utf-8-sig") as stream:
                main_rows = list(csv.DictReader(stream))
            with (root/"report/parent_matched_prefix_comparison.csv").open(encoding="utf-8-sig") as stream:
                parent_rows = list(csv.DictReader(stream))
            self.assertEqual(len(main_rows), 4)
            self.assertEqual({r["n"] for r in main_rows}, {"200"})
            self.assertEqual({r["variant"] for r in main_rows}, {"lookm", "asymkv"})
            self.assertEqual(len(parent_rows), 3)
            self.assertEqual({r["n"] for r in parent_rows}, {"2"})
            self.assertEqual({r["variant"] for r in parent_rows}, {"kvrepack", "lookm", "asymkv"})
            self.assertEqual([n for n, _ in scored_counts], [200]*4 + [2]*3)
            self.assertEqual(len(plot.call_args.args[0]), 4)
            summary = json.loads((root/"report/summary.json").read_text())
            self.assertEqual(len(summary["comparisons"]), 4)
            self.assertEqual(summary["parent_matched_prefix_comparison"]["quality_reference_docs"], 2)

    def test_native_scoring_command_uses_exact_prefix_and_disables_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrow = root/"cache/datasets/lmms-lab___vatex/vatex_test/revision/vatex-test.arrow"
            arrow.parent.mkdir(parents=True)
            arrow.write_bytes(b"reference fixture")
            spec = dict(contract(), hf_home=str(root/"cache"), arrow_sha256=sha256(arrow))
            observed = []
            def execute(command, **kwargs):
                observed.append(command)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                count = int(command[command.index("--expected-count")+1])
                metrics_path = Path(command[command.index("--output")+1])
                predictions = Path(command[command.index("--records")+1])
                self.assertEqual([json.loads(s)["doc_id"] for s in predictions.read_text().splitlines()], [0, 1])
                metrics_path.write_text(json.dumps(dict(n=count, expected_count=count,
                    metrics={"Bleu_4": .2, "METEOR": .3, "ROUGE_L": .4, "CIDEr": .5})))
            with patch.object(report.subprocess, "run", side_effect=execute):
                metrics = report.score_native_rows([row("kvrepack", i) for i in range(2)], spec, root/"scores", root)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][observed[0].index("--expected-count")+1], "2")
            self.assertEqual(metrics["CIDEr"], .5)


if __name__ == "__main__":
    unittest.main()

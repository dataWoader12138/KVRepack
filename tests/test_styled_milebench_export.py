"""Small synthetic archives exercise export integrity, never model performance."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = import_file("styled_export_test_target", REPO / "scripts/export_styled_milebench.py")
fixtures = import_file("styled_export_test_fixtures", REPO / "tests/test_styled_milebench_report.py")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class StyledImageExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = fixtures.fixture(self.root / "fixture", warmup=1)
        model = self.root / "fixture_model"
        save(model / "config.json", {"model_type": "synthetic_only"})
        for arm in exporter.ARMS:
            for task in exporter.TASKS:
                run = self.raw / arm / task
                contract = exporter.read_json(run / "run_contract.json")
                contract.update(pretrained=str(model), model_metadata_sha256={"config.json": exporter.file_sha(model / "config.json")})
                save(run / "run_contract.json", contract)
                for folder in ("samples", "warmup"):
                    for file in (run / folder).glob("*.json"):
                        row = exporter.read_json(file); row["contract_sha256"] = exporter.json_sha(contract); save(file, row)
                state = exporter.read_json(run / "run_state.json")
                state["contract_sha256"] = exporter.json_sha(contract); save(run / "run_state.json", state)
                save(run / "environment.json", {"synthetic": True})
                if not contract["standalone"]:
                    save(run / "adapter_transform.json", {"synthetic": True, "variant": contract["variant"]})
                elif arm == "lookm":
                    save(run / "legacy_lookm_transform.json", {"synthetic": True})
        self.report = self.root / "report"
        self.report_summary = fixtures.report.summarize(self.raw, self.report)
        controller = exporter.read_json(self.raw / "experiment_state.json")
        controller.update(status="complete", finished_utc="synthetic_finished", source_commit="b9104d8",
                          all_samples=True, planned_runs=16, target_generations=32)
        for row in controller["runs"]:
            row.update(target_samples=2, annotation_total_samples=2, all_samples=True)
        save(self.raw / "experiment_state.json", controller)
        self.output = self.root / "release"
        # Two records per run keep the fixtures small. Production CLI has no
        # count override and always requires the real 200 x 16 matrix.
        self.count_patch = patch.object(exporter, "EXPECTED_SAMPLES", 2)
        self.count_patch.start()

    def tearDown(self):
        self.count_patch.stop()
        self.temporary.cleanup()

    def export(self):
        return exporter.export_experiment(self.raw, self.report, self.output)

    def alter(self, path, update):
        value = exporter.read_json(path); update(value); save(path, value)

    def extract(self):
        unpacked = self.root / "unpacked"; unpacked.mkdir()
        with tarfile.open(self.output / "recorded_runs.tar.gz", "r:gz") as archive:
            for member in archive.getmembers():
                path = unpacked / exporter.safe_relative(member.name)
                self.assertTrue(member.isfile())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
        return unpacked

    def test_complete_archive_preserves_every_raw_file_and_report_without_source_writes(self):
        (self.raw / "weights.safetensors").write_bytes(b"excluded synthetic weights")
        (self.raw / "id_ed25519").write_bytes(b"excluded synthetic private key")
        (self.raw / "image.jpg").write_bytes(b"excluded synthetic image")
        before = exporter.record_sources([p for p in self.raw.rglob("*") if p.is_file()])
        manifest = self.export()
        exporter.assert_unchanged(before)
        self.assertEqual(manifest["condition_count"], 16)
        self.assertEqual(manifest["formal_generations"], 32)
        self.assertEqual(manifest["warmup_generations_excluded"], 16)
        self.assertTrue(manifest["round_trip_every_archived_file_sha256_verified"])
        self.assertLess(manifest["archive_bytes"], 90_000_000)
        self.assertEqual(manifest["archive_sha256"], exporter.file_sha(self.output / "recorded_runs.tar.gz"))
        index = exporter.read_json(self.output / "recorded_files_sha256.json")
        self.assertEqual(len(index["files"]), 111)
        self.assertTrue(all(entry["path"].endswith(".json") for entry in index["files"]))
        self.assertTrue((self.output / "configuration/model/config.json").is_file())
        self.assertTrue((self.output / "cpu_rescore/scripts/export_styled_milebench.py").is_file())
        self.assertEqual((self.output / "report/comparison.csv").read_bytes(), (self.report / "comparison.csv").read_bytes())
        check = exporter.read_json(self.output / "verification.json")
        self.assertNotEqual(check["controller_at_scoring_sha256"], check["final_controller_sha256"])
        exporter.verify_extracted(self.extract())

    def test_provenance_allowlist_can_be_copied_to_new_child_release_directory(self):
        provenance = self.root / "provenance"
        save(provenance / "formal_launch.json", {"command": ["synthetic"]})
        (provenance / "private_key.txt").write_text("excluded synthetic private key")
        self.output = provenance / "final"
        exporter.export_experiment(self.raw, self.report, self.output, provenance=provenance)
        self.assertTrue((self.output / "provenance/formal_launch.json").is_file())
        self.assertFalse((self.output / "provenance/private_key.txt").exists())

    def test_production_default_rejects_two_sample_synthetic_full_report(self):
        with patch.object(exporter, "EXPECTED_SAMPLES", 200), self.assertRaisesRegex(ValueError, "partial|official 200|protocol/counts"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_controller_must_be_complete_and_exactly_sixteen_conditions(self):
        path = self.raw / "experiment_state.json"
        original = exporter.read_json(path)
        for update in (lambda row: row.update(status="scoring"), lambda row: row["runs"].pop(),
                       lambda row: row["arms"].pop()):
            value = json.loads(json.dumps(original)); update(value); save(path, value)
            with self.assertRaises(ValueError): self.export()
            self.assertFalse(self.output.exists())
        save(path, original)

    def test_partial_summary_or_missing_condition_cannot_be_exported(self):
        path = self.report / "summary.json"; original = exporter.read_json(path)
        for update in (lambda row: row.update(full_experiment=False), lambda row: row["results"].pop(),
                       lambda row: row.update(formal_generations=1)):
            value = json.loads(json.dumps(original)); update(value); save(path, value)
            with self.assertRaises(ValueError): self.export()
        self.assertFalse(self.output.exists())

    def test_raw_changes_are_rejected_by_original_report_hash_evidence(self):
        sample = self.raw / "hybridkv_asymkv/Spot-the-Diff/samples/000000.json"
        self.alter(sample, lambda row: row.update(extra_after_report="changed"))
        with self.assertRaisesRegex(ValueError, "SHA evidence"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_each_csv_score_and_timing_value_is_recomputed(self):
        for filename, column in (("comparison.csv", "score_percent"), ("per_input_latency.csv", "end_to_end_seconds")):
            path = self.report / filename; original = path.read_bytes()
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream); columns = reader.fieldnames; rows = list(reader)
            rows[0][column] = str(float(rows[0][column]) + 1)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "CSV values"):
                self.export()
            path.write_bytes(original)
        self.assertFalse(self.output.exists())

    def test_evaluation_json_must_match_offline_original_metric(self):
        path = self.report / "conditions/hybridkv_asymkv/Spot-the-Diff/eval_score.json"
        self.alter(path, lambda rows: rows[0].update(score="0.0001"))
        with self.assertRaisesRegex(ValueError, "eval_score.json"):
            self.export()

    def test_archive_size_failure_is_explicit_atomic_and_does_not_drop_records(self):
        before = exporter.record_sources([p for p in self.raw.rglob("*") if p.is_file()])
        with patch.object(exporter, "MAX_ARCHIVE_BYTES", 1), self.assertRaisesRegex(ValueError, "no samples were removed"):
            self.export()
        exporter.assert_unchanged(before)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".styled-export-*")))

    def test_source_mutation_during_archive_fails_final_check_before_publish(self):
        original = exporter.archive_files
        environment = self.raw / "asymkv/MMCoQA/environment.json"

        def mutate(*args, **kwargs):
            original(*args, **kwargs)
            self.alter(environment, lambda row: row.update(changed_during_export=True))

        with patch.object(exporter, "archive_files", mutate), self.assertRaisesRegex(ValueError, "Source changed"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_symlinked_record_are_rejected(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.export()
        self.output.rmdir()
        environment = self.raw / "asymkv/MMCoQA/environment.json"
        target = self.root / "outside.json"; environment.rename(target); environment.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            self.export()

    def test_portable_rescore_works_with_relocated_annotations_without_original_model_or_kernel(self):
        self.export(); unpacked = self.extract()
        relocated = self.root / "relocated_annotations"
        (self.root / "fixture/data").rename(relocated)
        shutil.rmtree(self.root / "fixture_model")
        (self.root / "fixture/fixture_kernel.so").unlink()
        output = self.root / "cpu_scores"
        bundle = self.output / "cpu_rescore/scripts/export_styled_milebench.py"
        # A clean interpreter imports only the bundled CPU package; none of the
        # original annotation/model/kernel paths exist by this point.
        code = ("import importlib.util, sys; "
                f"spec=importlib.util.spec_from_file_location('bundle', {str(bundle)!r}); "
                "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                "module.EXPECTED_SAMPLES=2; "
                f"module.rescore_extracted({str(unpacked)!r}, {str(relocated)!r}, {str(output)!r}); "
                "assert 'torch' not in sys.modules")
        process = subprocess.run([sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        result = exporter.read_json(output / "summary.json")
        self.assertTrue(result["full_quality_rescore"])
        self.assertFalse(result["gpu_execution"])
        self.assertFalse(result["latency_remeasured"])
        self.assertEqual(result["formal_generations"], 32)
        expected = {(row["arm"], row["task"]): row["score_percent"] for row in self.report_summary["results"]}
        self.assertEqual({(row["arm"], row["task"]): row["score_percent"] for row in result["results"]}, expected)

    def test_portable_rescore_rejects_changed_raw_or_annotation(self):
        self.export(); unpacked = self.extract()
        annotation = self.root / "fixture/data/Spot-the-Diff/Spot-the-Diff.json"
        original = annotation.read_bytes()
        annotation.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "annotation bytes/IDs"):
            exporter.rescore_extracted(unpacked, annotation.parents[1], self.root / "bad_annotation_scores")
        annotation.write_bytes(original)
        raw = unpacked / "asymkv/MMCoQA/samples/000000.json"
        self.alter(raw, lambda row: row["prediction"].update(pred_response="tampered"))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            exporter.rescore_extracted(unpacked, annotation.parents[1], self.root / "bad_raw_scores")
        self.assertFalse((self.root / "bad_annotation_scores").exists())
        self.assertFalse((self.root / "bad_raw_scores").exists())


if __name__ == "__main__":
    unittest.main()

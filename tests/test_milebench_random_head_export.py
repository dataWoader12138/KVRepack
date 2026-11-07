"""CPU-only exporter tests using synthetic completed evidence, never GPU data."""
from __future__ import annotations

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("milebench_result_export", ROOT / "scripts/export_milebench_random_head_results.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixture(base):
    """Use the real on-disk schema, with two synthetic inputs in each run."""
    root, audit_root, data = base / "matrix", base / "audit", base / "data"
    data.mkdir()
    source = root / "source_archives" / "old" / "src" / "synthetic.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic archived source\n", encoding="utf-8")
    source_map = {"src/synthetic.py": export.file_sha(source)}
    runs, evidence = [], []
    for method in export.METHODS:
        for task in export.TASKS:
            run = root / method / task
            contract = {"schema_version": 1, "reference_commit": export.REFERENCE, "method": method, "task": task,
                        "all_samples": True, "limit": 2, "annotation_total_samples": 2, "data_root": str(data),
                        "annotation_sha256": "a" * 64, "selected_sample_ids": [10, 11], "generation_sample_ids": [11, 10],
                        "model_type": "qwen2_5_vl-7b", "head_geometry": {"layers": 28, "native_kv_heads": 4},
                        "conditions": list(export.LABELS), "condition_settings": [
                            {"label": label, "strategy": "similarity" if i == 0 else "random_global",
                             "mask_seed": max(0, i - 1)} for i, label in enumerate(export.LABELS)],
                        "generation": {"max_new_tokens": 4, "do_sample": False},
                        "source_files_sha256": deepcopy(source_map)}
            identity = export.json_sha(contract)
            save(run / "run_contract.json", contract)
            for name in ("resolved_config.json", "provenance.json", "model_runtime.json"):
                save(run / name, {"synthetic_evidence": True, "method": method, "task": task})
            save(run / "run_state.json", {"returncode": 0, "finished_at": "fixture"})
            save(run / "ablation_state.json", {"status": "complete", "contract_sha256": identity,
                                                "target_samples": 2, "completed_samples": 2,
                                                "target_generations": 6, "completed_generations": 6})
            sample_files = []
            for index, sample_id in enumerate(contract["generation_sample_ids"]):
                inputs = {"input_ids": {"shape": [1, 8], "dtype": "int64", "sha256": "c" * 64}}
                records = {}
                for i, label in enumerate(export.LABELS):
                    tokens = [3, 3, 3, 3] if i == 2 else [3, 2]
                    records[label] = {
                        "condition": {k: contract["condition_settings"][i][k] for k in ("strategy", "mask_seed")},
                        "prediction": {"sample_id": sample_id, "pred_response": f"答 {sample_id} {label}",
                                       "gt_response": "GROUND_TRUTH_SENTINEL", "question": "QUESTION_SENTINEL",
                                       "image": ["PRIVATE_IMAGE_SENTINEL.png"]},
                        "input_fingerprints": deepcopy(inputs),
                        "audit": {"large_head_array": ["HEAD_ARRAY_SENTINEL"] * 128},
                        "runtime": {"generated_ids": tokens, "generated_tokens": len(tokens), "eos_token_id": 2,
                                    "eos_reached": i != 2, "hit_max_new_tokens": i == 2,
                                    "truncated_without_eos": i == 2, "input_tokens": 8,
                                    "generation_seconds": 0.25, "cache": {"head_array": ["CACHE_SENTINEL"] * 128}}}
                path = run / "samples" / f"{index:06d}.json"
                save(path, {"schema_version": 1, "status": "completed", "method": method, "task": task,
                            "sample_index": index, "sample_id": sample_id, "contract_sha256": identity,
                            "input_fingerprints": inputs, "records": records, "fairness": {"status": "passed"}})
                sample_files.append({"sample_id": sample_id, "sample_file": path.name,
                                     "sample_sha256": export.file_sha(path),
                                     "input_fingerprints_sha256": export.json_sha(inputs)})
            report = run / "report"
            save(report / "summary.json", {"status": "complete", "contract_sha256": identity,
                                            "sample_count": 2, "generation_count": 6, "full_task": True,
                                            "sample_files": sample_files})
            for name in ("scores.csv", "paired_scores.csv", "selector_audit.csv", "RESULTS.md"):
                (report / name).write_text("synthetic fixture\n", encoding="utf-8")
            for label in export.LABELS:
                for name in ("eval.json", "eval_score.json", "evaluation_metadata.json"):
                    save(report / "conditions" / label / name, {"synthetic": True})
            runs.append({"method": method, "task": task, "output": str(run), "report": str(report),
                         "status": "complete", "returncode": 0})
            evidence.append({"method": method, "task": task, "run": str(run), "report": str(report),
                             "status": "passed", "contract_sha256": identity, "annotation_sha256": "a" * 64,
                             "annotation_total_samples": 2, "validated_bundles": 2, "validated_generations": 6,
                             "bound_source_files": deepcopy(contract["source_files_sha256"]),
                             "sample_files": sample_files, "report_summary_sha256": export.file_sha(report / "summary.json"),
                             "report_files_sha256": {p.relative_to(report).as_posix(): export.file_sha(p)
                                                     for p in report.rglob("*") if p.is_file()}})
    save(root / "controller_state.json", {"schema_version": 1, "reference_commit": export.REFERENCE, "status": "complete",
                                          "planned_runs": 4, "completed_runs": 4, "runs": runs})
    driver = base / "continuation_driver.py"
    driver.write_text("# Synthetic controller driver\n", encoding="utf-8")
    save(base / "matrix_launch.json", {"output_root": str(root), "command": ["python", str(driver)],
                                       "continuation_driver_sha256": export.file_sha(driver)})
    save(root / "reuse_manifest.json", {"status": "copied", "synthetic": True})
    save(root / "source_archives" / "old" / "SNAPSHOT.json", {"source_files_sha256": source_map})
    tables = {"results": [{"method": r["method"], "task": r["task"], "sample_count": 2, "similarity_percent": 75.0}
                          for r in evidence],
              "generation_stats": [{"method": r["method"], "task": r["task"], "condition": label,
                                    "generation_count": 2, "generated_tokens_sample_sd": None}
                                   for r in evidence for label in export.LABELS],
              "merge_work_stats": [{"method": r["method"], "task": r["task"], "condition": label,
                                    "effective_merge_count_total": 112}
                                   for r in evidence for label in export.LABELS]}
    save(audit_root / "audit.json", {"schema_version": 1, "status": "passed", "scoring_performed": True,
                                    "controller_status": "complete", "run_root": str(root),
                                    "controller_sha256": export.file_sha(root / "controller_state.json"),
                                    "runs": evidence, **tables})
    for name, key in export.AUDIT_TABLES.items():
        with (audit_root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tables[key][0]))
            writer.writeheader()
            writer.writerows(tables[key])
    return root, audit_root


class CompactExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.audit = fixture(self.base)
        self.output = self.base / "compact"

    def tearDown(self):
        self.temp.cleanup()

    def run_export(self):
        return export.export_results(self.root, self.audit, self.output)

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.run_export()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.base.glob(".compact.export-*")))

    def test_success_preserves_sources_and_records_all_hashes(self):
        before = {str(p): export.file_sha(p) for p in self.base.rglob("*") if p.is_file()}
        result = self.run_export()
        self.assertEqual((result["bundle_count"], result["prediction_count"]), (8, 24))
        self.assertEqual(result, read(self.output / "EXPORT.json"))
        self.assertFalse(result["merge_audit_reproducible_from_export"])
        self.assertIn("raw samples/*.json", result["limitations"][0])
        self.assertEqual(result["source_archive_directories"], ["source_archives"])
        for path, sha in before.items():
            self.assertEqual(export.file_sha(path), sha)
        files = {row["path"]: row for row in result["files"]}
        actual = {p.relative_to(self.output).as_posix() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(set(files), actual - {"EXPORT.json"})
        for name, row in files.items():
            self.assertEqual(export.file_sha(self.output / name), row["sha256"])
            self.assertEqual((self.output / name).stat().st_size, row["bytes"])
        self.assertEqual({p.name for p in (self.output / "audit").iterdir()}, {"audit.json", *export.AUDIT_TABLES})
        self.assertTrue((self.output / "reuse_manifest.json").is_file())
        self.assertTrue((self.output / "continuation_driver.py").is_file())
        self.assertEqual((self.output / "controller_state.json").read_bytes(), (self.root / "controller_state.json").read_bytes())
        projection = (self.output / "predictions.jsonl").read_text(encoding="utf-8")
        for sentinel in ("GROUND_TRUTH_SENTINEL", "QUESTION_SENTINEL", "PRIVATE_IMAGE_SENTINEL",
                         "HEAD_ARRAY_SENTINEL", "CACHE_SENTINEL"):
            self.assertNotIn(sentinel, projection)
        rows = [json.loads(line) for line in projection.splitlines()]
        self.assertEqual(len(rows), 24)
        self.assertEqual([r["sample_id"] for r in rows[:6]], [11, 11, 11, 10, 10, 10])
        self.assertEqual(rows[2]["generated_ids"], [3] * 4)
        self.assertTrue(rows[2]["truncated_without_eos"])
        self.assertTrue(rows[0]["eos_reached"])
        manifest = [json.loads(line) for line in (self.output / "bundle_manifest.jsonl").read_text().splitlines()]
        self.assertEqual(len(manifest), 8)
        for row in manifest:
            self.assertEqual(export.file_sha(self.root / row["sample_file"]), row["sha256"])
        self.assertFalse(list(self.output.rglob("samples")))

    def test_refuses_nonfinal_controller_and_audit(self):
        controller_path = self.root / "controller_state.json"
        original_controller, original_audit = read(controller_path), read(self.audit / "audit.json")
        for status in ("running", "failed"):
            with self.subTest(controller=status):
                controller = deepcopy(original_controller)
                controller["status"] = status
                save(controller_path, controller)
                self.assert_rejected("Controller must be complete")
        save(controller_path, original_controller)
        for patch in ({"status": "progress"}, {"status": "failed"}, {"scoring_performed": False}, {"controller_status": "running"}):
            with self.subTest(audit=patch):
                save(self.audit / "audit.json", dict(original_audit, **patch))
                self.assert_rejected("Independent final audit must have passed")

    def test_rejects_stale_controller_audit(self):
        path = self.root / "controller_state.json"
        save(path, dict(read(path), finished_utc="changed"))
        self.assert_rejected("stale controller SHA")

    def test_accepts_original_launch_schema_and_rejects_mismatched_driver(self):
        path = self.base / "matrix_launch.json"
        original = read(path)
        driver = self.base / "continuation_driver.py"
        driver.write_text("# changed\n")
        self.assert_rejected("SHA-256 mismatch")
        save(path, {"output": str(self.root), "command": ["synthetic", "--all-samples"]})
        result = self.run_export()
        self.assertEqual(result["status"], "complete")
        self.assertFalse((self.output / "continuation_driver.py").exists())

    def test_rejects_raw_or_report_mutation_after_audit(self):
        bundle = self.root / export.METHODS[0] / export.TASKS[0] / "samples" / "000000.json"
        original = bundle.read_bytes()
        save(bundle, dict(read(bundle), changed=True))
        self.assert_rejected("SHA-256 mismatch")
        bundle.write_bytes(original)
        summary = bundle.parent.parent / "report" / "summary.json"
        save(summary, dict(read(summary), changed=True))
        self.assert_rejected("SHA-256 mismatch")

    def test_rejects_runtime_inconsistency_even_with_refreshed_audit_hash(self):
        run = self.root / export.METHODS[0] / export.TASKS[0]
        path = run / "samples" / "000000.json"
        bundle = read(path)
        bundle["records"][export.LABELS[0]]["runtime"]["eos_reached"] = False
        save(path, bundle)
        audit = read(self.audit / "audit.json")
        audit["runs"][0]["sample_files"][0]["sample_sha256"] = export.file_sha(path)
        summary = read(run / "report" / "summary.json")
        summary["sample_files"] = audit["runs"][0]["sample_files"]
        save(run / "report" / "summary.json", summary)
        audit["runs"][0]["report_summary_sha256"] = export.file_sha(run / "report" / "summary.json")
        audit["runs"][0]["report_files_sha256"]["summary.json"] = audit["runs"][0]["report_summary_sha256"]
        save(self.audit / "audit.json", audit)
        self.assert_rejected("eos_reached disagrees")

    def test_rejects_extra_report_data_and_incomplete_runner_state(self):
        run = self.root / export.METHODS[0] / export.TASKS[0]
        extra = run / "report" / "raw_bundles.json"
        save(extra, {"unexpected": "raw data must not be copied"})
        self.assert_rejected("Report tree has missing or unexpected files")
        extra.unlink()
        save(run / "run_state.json", {"returncode": 1, "finished_at": "fixture"})
        self.assert_rejected("Runner final state")

    def test_refuses_overwrite_and_protected_destinations(self):
        self.output.mkdir()
        keep = self.output / "keep.txt"
        keep.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_export()
        self.assertEqual(keep.read_text(), "keep")
        for path in (self.root / "export", self.audit / "export", self.base / "data" / "export"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                export.export_results(self.root, self.audit, path)
            self.assertFalse(path.exists())

    def test_rejects_source_symlinks_and_large_or_nontext_audit_files(self):
        extra = self.audit / "results.csv"
        original = extra.read_bytes()
        extra.unlink()
        extra.symlink_to(self.audit / "audit.json")
        self.assert_rejected("Symlink")
        extra.unlink()
        extra.write_bytes(original)
        binary = self.audit / "weights.safetensors"
        binary.write_bytes(b"synthetic")
        self.assert_rejected("must contain exactly")
        binary.unlink()
        extra.write_bytes(b"x" * (export.SMALL_FILE_LIMIT + 1))
        self.assert_rejected("small-file limit")

    def test_rejects_tampered_report_or_audit_csvs_and_missing_hashes(self):
        run = self.root / export.METHODS[0] / export.TASKS[0]
        report_file = run / "report" / "scores.csv"
        original = report_file.read_bytes()
        report_file.write_text("changed scores\n", encoding="utf-8")
        self.assert_rejected("SHA-256 mismatch")
        report_file.write_bytes(original)
        table = self.audit / "merge_work_stats.csv"
        original = table.read_bytes()
        table.write_bytes(original.replace(b"112", b"113", 1))
        self.assert_rejected("value mismatch")
        table.write_bytes(original)
        audit = read(self.audit / "audit.json")
        del audit["runs"][0]["report_summary_sha256"]
        save(self.audit / "audit.json", audit)
        self.assert_rejected("Missing or invalid SHA-256")

    def test_rejects_source_archive_mutation_or_extra_raw_data(self):
        archive = self.root / "source_archives" / "old"
        source = archive / "src" / "synthetic.py"
        original = source.read_bytes()
        source.write_text("# changed\n")
        self.assert_rejected("SHA-256 mismatch")
        source.write_bytes(original)
        save(archive / "raw.json", {"raw_head_arrays": [0] * 100})
        self.assert_rejected("Unexpected file in source snapshot")


if __name__ == "__main__":
    unittest.main()

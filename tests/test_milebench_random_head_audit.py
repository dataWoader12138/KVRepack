"""CPU-only synthetic evidence tests; these fixtures are not experiment results."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.ablations.head_selection import _ranking
from kvrepack.ablations.milebench_head_selection_report import json_sha, summarize_run

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("image_random_head_audit", REPO / "scripts/audit_milebench_random_head_run.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fingerprint(tag):
    return {"shape": [1, 4], "dtype": "torch.bfloat16", "sha256": hashlib.sha256(tag.encode()).hexdigest()}


def make_bundle(contract, index):
    hybrid = contract["method"] == audit.METHODS[0]
    quota = 56 if hybrid else 34
    sample_id = contract["generation_sample_ids"][index]
    native = [i < quota for i in range(112)]
    layout = lambda flat: [[flat[i:i + 4]] for i in range(0, 112, 4)] if hybrid else [[flat[i:i + 4] for i in range(0, 112, 4)]]
    inputs = {"input_ids": fingerprint("same-ids"), "pixel_values": fingerprint("same-image")}
    budgets = [[[3] * 4] for _ in range(28)]
    pre = {"head_scores": fingerprint("scores"), "head_score_values": [[[1] * 4] for _ in range(28)],
           "head_category": None, "calibrated_head_category": {"values": [[[0] * 4] for _ in range(28)], "tensor": fingerprint("categories")},
           "sparse_budget_per_head": budgets, "candidate_hashes": [
               {"layer": layer, "pruned": {"candidate": fingerprint(str(layer))}, "retrieval": None} for layer in range(28)]}
    records = {}
    for arm, label in enumerate(audit.LABELS):
        seed, strategy = max(0, arm - 1), "similarity" if arm == 0 else "random_global"
        chosen = set(range(quota)) if arm == 0 else set(_ranking(seed, "global", 28, 4, 0).sample(range(112), 112)[:quota])
        calls = [{"layer": value // 4, "batch": 0, "head": value % 4, "before": 5 if hybrid else 8,
                  "after": 3 if hybrid else 4, "retained_head_budget": 3} for value in sorted(chosen)]
        selector = {"status": "completed", "task": contract["task"], "doc_id": sample_id, "strategy": strategy,
                    "seed": seed, "layers": 28, "native_kv_heads": 4, "batch_size": 1,
                    "merge_ratio": 0.5 if hybrid else 0.3, "baseline_masks": layout(native),
                    "selected_masks": layout([i in chosen for i in range(112)]),
                    "baseline_selected_count": [quota], "selected_count": [quota], "merge_calls": calls,
                    "actual_merge_calls": quota, "actual_removed_tokens": quota * (2 if hybrid else 4)}
        if hybrid:
            selector.update(method=contract["method"], mask_layout="layers,batch,kv_heads", pre_selection=deepcopy(pre))
        else:
            selector.update(expected_effective_merge_count=quota, expected_removed_tokens=quota * 4,
                            candidate_hashes=[{"layer": layer, "candidate_raw": fingerprint(str(layer))} for layer in range(28)],
                            prefill_cache_shapes=[{"layer": layer, "key_shape": [1, 4, 4, 4], "value_shape": [1, 4, 4, 4],
                                                   "key_dtype": "torch.bfloat16", "value_dtype": "torch.bfloat16"} for layer in range(28)])
        answer = "red" if index == 0 else "blue"
        prediction = {"sample_id": sample_id, "gt_response": answer, "pred_response": "wrong" if arm == 1 else answer,
                      "image": ["one.png", "two.png"], "question": "Compare", "gen_model_id": "fixture",
                      "gen_kwargs": deepcopy(contract["generation"])}
        runtime = {"input_tokens": 15, "generated_ids": [4] * 4 if arm == 2 else [1, 2],
                   "generated_tokens": 4 if arm == 2 else 2, "eos_token_id": 2, "eos_reached": arm != 2,
                   "hit_max_new_tokens": arm == 2, "truncated_without_eos": arm == 2, "generation_seconds": 0.2}
        records[label] = {"prediction": prediction, "audit": selector, "condition": {"strategy": strategy, "mask_seed": seed},
                          "input_fingerprints": deepcopy(inputs), "runtime": runtime}
    return {"schema_version": 1, "status": "completed", "task": contract["task"], "method": contract["method"],
            "sample_index": index, "sample_id": sample_id, "contract_sha256": json_sha(contract),
            "input_fingerprints": inputs, "records": records, "fairness": {"status": "passed"}}


def make_matrix(root):
    data, run_root = root / "data", root / "matrix"
    runs = []
    for method in audit.METHODS:
        for task in audit.TASKS:
            annotation = {"meta_data": {"num_sample": 2, "question_type": "open-ended", "task_instruction": ["Compare"]},
                          "data": [{"sample_id": 10 + i, "task_instruction_id": 0,
                                    "task_instance": {"context": "Compare", "images_path": ["one.png", "two.png"]},
                                    "response": "red" if i == 0 else "blue", "image_quantity_level": "Few"} for i in range(2)]}
            source = data / task / f"{task}.json"
            save(source, annotation)
            hybrid = method == audit.METHODS[0]
            ratio_key = "HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO" if hybrid else "SNAPKV_SIMHEAD_MERGE_RATIO"
            contract = {"schema_version": 1, "reference_commit": audit.REFERENCE, "method": method, "task": task,
                        "limit": 2, "all_samples": True, "annotation_total_samples": 2, "data_root": str(data),
                        "annotation_sha256": audit.file_sha(source), "selected_sample_ids": [10, 11], "generation_sample_ids": [10, 11],
                        "conditions": list(audit.LABELS), "condition_settings": [
                            {"label": label, "strategy": "similarity" if i == 0 else "random_global", "mask_seed": max(0, i - 1)}
                            for i, label in enumerate(audit.LABELS)], "head_geometry": {"layers": 28, "native_kv_heads": 4},
                        "model_type": "qwen2_5_vl-7b", "algorithm_env": {ratio_key: "0.5" if hybrid else "0.3"},
                        "generation": {"max_new_tokens": 4, "do_sample": False},
                        "source_files_sha256": {name: audit.file_sha(REPO / name) for name in audit.BOUND_SOURCES}}
            run = run_root / method / task
            save(run / "run_contract.json", contract)
            save(run / "ablation_state.json", {"status": "complete", "contract_sha256": json_sha(contract), "target_samples": 2,
                                               "completed_samples": 2, "target_generations": 6, "completed_generations": 6})
            for i in range(2):
                save(run / "samples" / f"{i:06d}.json", make_bundle(contract, i))
            summarize_run(run, run / "report")
            runs.append({"method": method, "task": task, "output": str(run), "report": str(run / "report"),
                         "status": "complete", "returncode": 0})
    save(run_root / "controller_state.json", {"schema_version": 1, "status": "complete", "reference_commit": audit.REFERENCE,
                                              "planned_runs": 4, "completed_runs": 4, "runs": runs})
    return run_root


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("rouge"), "Original metrics need NumPy and rouge")
class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = make_matrix(self.root)
        self.first = self.run / audit.METHODS[0] / audit.TASKS[0]

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, path, change):
        value = audit.read_json(path); change(value); save(path, value)

    def test_full_audit_recomputes_four_rows_and_generation_statistics_without_writing_inputs(self):
        before = {str(path): audit.file_sha(path) for path in self.run.rglob("*") if path.is_file()}
        result = audit.write_audit(self.run, self.root / "audit")
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["scoring_performed"])
        self.assertEqual(len(result["results"]), 4)
        self.assertEqual([row["selected_head_quota"] for row in result["runs"]], [56, 56, 34, 34])
        row = result["results"][1]
        self.assertAlmostEqual(row["similarity_percent"], 100)
        self.assertAlmostEqual(row["random_mean_percent"], 50)
        self.assertAlmostEqual(row["random_sample_sd_percent"], 100 / 2**0.5)
        self.assertAlmostEqual(row["similarity_minus_random_mean_points"], 50)
        self.assertEqual(result["generation_stats"][0]["eos_count"], 2)
        self.assertEqual(result["generation_stats"][2]["truncated_without_eos_count"], 2)
        self.assertEqual(result["generation_stats"][2]["generated_tokens_mean"], 4)
        self.assertEqual(before, {str(path): audit.file_sha(path) for path in self.run.rglob("*") if path.is_file()})
        with (self.root / "audit/results.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_active_controller_emits_only_progress_even_when_completed_run_has_bad_scores(self):
        self.mutate(self.run / "controller_state.json", lambda value: value.update(status="generating", completed_runs=1))
        (self.first / "report/summary.json").write_text("invalid")
        with patch("kvrepack.benchmarks.milebench.evaluate.evaluate_predictions", side_effect=AssertionError("partial scoring")):
            result = audit.write_audit(self.run, self.root / "progress")
        self.assertEqual(result["status"], "progress")
        self.assertFalse(result["scoring_performed"])
        self.assertEqual({p.name for p in (self.root / "progress").iterdir()}, {"audit.json"})

    def test_historical_source_requires_explicit_archive_with_exact_hashes(self):
        bound = {relative: audit.file_sha(REPO / relative) for relative in audit.BOUND_SOURCES}
        relative = audit.BOUND_SOURCES[0]
        archive = self.root / "archive"
        archived = archive / relative
        archived.parent.mkdir(parents=True)
        archived.write_bytes(b"# historical frozen source\n" + (REPO / relative).read_bytes())
        bound[relative] = audit.file_sha(archived)
        with self.assertRaisesRegex(ValueError, "absent from explicit archives"):
            audit.verify_bound_sources(bound)
        verified = audit.verify_bound_sources(bound, [archive])
        self.assertEqual(verified[relative]["path"], str(archived))
        archived.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "absent from explicit archives"):
            audit.verify_bound_sources(bound, [archive])

    def test_hybrid_native_skips_and_noops_are_reported_separately_from_quota(self):
        sample = self.first / "samples/000000.json"
        bundle = audit.read_json(sample)
        for label in audit.LABELS[1:]:
            row = bundle["records"][label]["audit"]
            if label == audit.LABELS[1]:
                row["merge_calls"].pop()
            else:
                row["merge_calls"][0]["after"] = row["merge_calls"][0]["before"]
            row["actual_merge_calls"] = sum(call["after"] < call["before"] for call in row["merge_calls"])
            row["actual_removed_tokens"] = sum(call["before"] - call["after"] for call in row["merge_calls"])
        save(sample, bundle)
        shutil.rmtree(self.first / "report")
        summarize_run(self.first, self.first / "report")
        result = audit.write_audit(self.run, self.root / "audit")
        self.assertEqual(result["status"], "passed", result.get("error"))
        work = result["runs"][0]["merge_work_stats"]
        self.assertEqual(work[1]["selected_count_min"], 56)
        self.assertEqual(work[1]["effective_merge_count_min"], 55)
        self.assertEqual(work[1]["selected_without_invocation_count_total"], 1)
        self.assertEqual(work[2]["no_op_call_count_total"], 1)
        self.assertTrue((self.root / "audit/merge_work_stats.csv").is_file())

    def test_summary_tampering_and_underlying_eval_tampering_are_both_rejected(self):
        for relative, change in (("summary.json", lambda value: value.update(random_mean_percent=-1)),
                                 ("conditions/random_global_seed0/eval.json", lambda value: value.update(**{"Rouge-L f": 1}))):
            path = self.first / "report" / relative
            original = path.read_bytes()
            self.mutate(path, change)
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, "numeric mismatch"):
                audit.audit_run(self.run)
            path.write_bytes(original)

    def test_csv_scores_are_checked_independently(self):
        path = self.first / "report/paired_scores.csv"
        path.write_text(path.read_text().replace("random_global_seed0", "random_global_seed2", 1))
        with self.assertRaisesRegex(ValueError, "value mismatch"):
            audit.audit_run(self.run)

    def test_all_samples_exact_conditions_and_frozen_sources_are_required(self):
        path = self.first / "run_contract.json"
        original = path.read_bytes()
        for change, message in ((lambda value: value.update(all_samples=False), "all_samples"),
                                (lambda value: value["conditions"].append("random_global_seed2"), "exactly"),
                                (lambda value: value.update(annotation_total_samples=1), "count"),
                                (lambda value: value["source_files_sha256"].clear(), "source file coverage")):
            self.mutate(path, change)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                audit.audit_run(self.run)
            path.write_bytes(original)

    def test_complete_annotation_sha_is_verified(self):
        annotation = self.root / "data" / audit.TASKS[0] / f"{audit.TASKS[0]}.json"
        self.mutate(annotation, lambda value: value["data"].append(deepcopy(value["data"][0])))
        with self.assertRaisesRegex(ValueError, "annotation SHA"):
            audit.audit_run(self.run)

    def test_missing_samples_and_extra_runs_cannot_pass(self):
        path = self.first / "samples/000001.json"
        original = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, "filename coverage"):
            audit.audit_run(self.run)
        path.write_bytes(original)
        save(self.run / "extra/run_contract.json", {})
        with self.assertRaisesRegex(ValueError, "extra method/task"):
            audit.audit_run(self.run)

    def test_actual_merges_fixed_masks_and_runtime_flags_are_recomputed(self):
        path = self.first / "samples/000001.json"
        original = path.read_bytes()
        mutations = [(lambda value: value["records"][audit.LABELS[1]]["audit"].update(actual_merge_calls=55), "effective merge-call counter"),
                     (lambda value: value["records"][audit.LABELS[1]]["audit"].update(selected_masks=value["records"][audit.LABELS[0]]["audit"]["selected_masks"]), "fixed seed/model"),
                     (lambda value: value["records"][audit.LABELS[0]]["runtime"].update(eos_reached=False), "eos_reached"),
                     (lambda value: value["records"][audit.LABELS[0]]["runtime"].update(generated_tokens=3), "token count")]
        for change, message in mutations:
            self.mutate(path, change)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                audit.audit_run(self.run)
            path.write_bytes(original)

    def test_failed_audit_writes_failure_evidence_without_results(self):
        self.mutate(self.first / "report/summary.json", lambda value: value.update(random_seed_count=3))
        result = audit.write_audit(self.run, self.root / "failed")
        self.assertEqual(result["status"], "failed")
        self.assertIn("random_seed_count", result["error"])
        self.assertEqual({p.name for p in (self.root / "failed").iterdir()}, {"audit.json"})

    def test_audit_output_cannot_reuse_or_touch_experiment_or_dataset(self):
        for path in (self.run, self.first / "audit", self.root / "data/audit", self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.write_audit(self.run, path)

    def test_cli_exit_code_and_import_do_not_load_model_dependencies(self):
        code = (f"import runpy,sys; d=runpy.run_path({str(REPO / 'scripts/audit_milebench_random_head_run.py')!r}); "
                "assert all(name not in sys.modules for name in ('torch','transformers','accelerate')); "
                f"assert d['audit_run']({str(self.run)!r})['status']=='passed'; "
                "assert all(name not in sys.modules for name in ('torch','transformers','accelerate'))")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

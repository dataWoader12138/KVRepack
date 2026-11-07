"""Offline image Random-head scoring and tamper rejection on fixed fake runs.

Fixtures model real audit shapes and ranking conventions, not model behavior.
Their synthetic predictions/metrics are never reported as dataset experiments.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kvrepack.ablations.head_selection import _ranking
from kvrepack.ablations.milebench_head_selection_report import json_sha, merge_work_stats, summarize_run, validate_bundle


LABELS = ["similarity", "random_global_seed0", "random_global_seed1", "random_global_seed2"]


def fingerprint(tag, shape=(1, 2)):
    return {"shape": list(shape), "dtype": "torch.float32", "sha256": hashlib.sha256(tag.encode()).hexdigest()}


def make_contract(data_root, task="MMCoQA", method="hybridkv_kvrepack", limit=2):
    root = Path(data_root)
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    annotation = {"meta_data": {"task_instruction": ["Compare the images."], "question_type": "open-ended", "num_sample": limit},
                  "data": [{"sample_id": 10 + index, "task_instruction_id": 0,
                            "task_instance": {"context": "{image#1} versus {image#2}", "images_path": ["one.png", "two.png"]},
                            "response": "red" if index % 2 == 0 else "blue", "image_quantity_level": "Few"}
                           for index in range(limit)]}
    raw = json.dumps(annotation).encode()
    (folder / f"{task}.json").write_bytes(raw)
    ratio_key = "HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO" if method == "hybridkv_kvrepack" else "SNAPKV_SIMHEAD_MERGE_RATIO"
    return {"schema_version": 1, "method": method, "task": task, "limit": limit,
            "conditions": list(LABELS), "data_root": str(root), "annotation_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_sample_ids": [10 + index for index in range(limit)], "head_geometry": {"layers": 2, "native_kv_heads": 2},
            "algorithm_env": {ratio_key: "0.5"}, "source_files_sha256": {},
            "generation": {"do_sample": False, "max_new_tokens": 512}}


def make_bundle(contract, index, answers=None):
    sample_id = contract.get("generation_sample_ids", contract["selected_sample_ids"])[index]
    hybrid = contract["method"] == "hybridkv_kvrepack"
    rows = {}
    native = [True, True, False, False]
    budgets = [[[3, 5]], [[7, 9]]]
    pre = {"head_scores": fingerprint("scores"), "head_score_values": [[[1, 1]], [[0, 0]]],
           "head_category": None, "calibrated_head_category": {"values": [[[0, 1]], [[0, 1]]], "tensor": fingerprint("categories")},
           "sparse_budget_per_head": budgets, "retrieval_head_mapping": [[1], [1]], "pruned_head_mapping": [[0], [0]],
           "parameters": {"sparse_budget_ratio": 0.1},
           "candidate_hashes": [{"layer": layer, "pruned": {"candidate": fingerprint(f"hybrid-candidate-{layer}")}, "retrieval": None}
                                for layer in range(2)]}
    inputs = {"input_ids": fingerprint("same prepared IDs", (1, 12)), "pixel_values": fingerprint("same pixels", (2, 3, 4))}
    answer = "red" if sample_id % 2 == 0 else "blue"
    for label in LABELS:
        seed = 0 if label == "similarity" else int(label.removeprefix("random_global_seed"))
        strategy = "similarity" if label == "similarity" else "random_global"
        chosen = set(range(2)) if strategy == "similarity" else set(_ranking(seed, "global", 2, 2, 0).sample(range(4), 4)[:2])
        flat = [head in chosen for head in range(4)]
        layout = lambda values: [[[values[0], values[1]]], [[values[2], values[3]]]] if hybrid else [[[values[0], values[1]], [values[2], values[3]]]]
        calls = []
        for value in sorted(chosen):
            layer, head = divmod(value, 2)
            budget = budgets[layer][0][head]
            calls.append({"layer": layer, "batch": 0, "head": head,
                          "before": budget + layer + head + 3 if hybrid else 8,
                          "after": budget if hybrid else 4, "retained_head_budget": budget})
        audit = {"status": "completed", "task": contract["task"], "doc_id": sample_id,
                 "strategy": strategy, "seed": seed, "layers": 2, "native_kv_heads": 2, "batch_size": 1,
                 "merge_ratio": 0.5, "baseline_masks": layout(native), "selected_masks": layout(flat),
                 "baseline_selected_count": [2], "selected_count": [2], "merge_calls": calls,
                 "actual_merge_calls": 2, "actual_removed_tokens": sum(call["before"] - call["after"] for call in calls)}
        if hybrid:
            audit.update(method=contract["method"], mask_layout="layers,batch,kv_heads", pre_selection=deepcopy(pre))
        else:
            audit.update(expected_effective_merge_count=2, expected_removed_tokens=8,
                         candidate_hashes=[{"layer": layer, "candidate_raw": fingerprint(f"snap-candidate-{layer}")}
                                           for layer in range(2)],
                         prefill_cache_shapes=[{"layer": layer, "key_shape": [1, 2, 4, 4], "value_shape": [1, 2, 4, 4],
                                                "key_dtype": "torch.float32", "value_dtype": "torch.float32"}
                                               for layer in range(2)])
        prediction = {"sample_id": sample_id, "gt_response": answer, "pred_response": (answers or {}).get(label, answer),
                      "image": ["one.png", "two.png"], "question": "Compare", "gen_model_id": "fixture",
                      "gen_kwargs": {"do_sample": False, "max_new_tokens": 512}}
        rows[label] = {"prediction": prediction, "audit": audit, "input_fingerprints": deepcopy(inputs),
                       "condition": {"strategy": strategy, "mask_seed": seed}}
    return {"schema_version": 1, "status": "completed", "task": contract["task"], "method": contract["method"],
            "sample_index": index, "sample_id": sample_id, "contract_sha256": json_sha(contract),
            "input_fingerprints": inputs, "records": rows, "fairness": {"status": "passed"}}


def write_run(root, contract, bundles=None):
    root = Path(root)
    (root / "samples").mkdir(parents=True)
    (root / "run_contract.json").write_text(json.dumps(contract))
    n = contract["limit"]
    (root / "ablation_state.json").write_text(json.dumps({"status": "complete", "target_samples": n,
                                                       "completed_samples": n, "target_generations": n * 4, "completed_generations": n * 4}))
    for index in range(n):
        bundle = bundles[index] if bundles is not None else make_bundle(contract, index)
        (root / "samples" / f"{index:06d}.json").write_text(json.dumps(bundle))


def recount_hybrid(audit):
    """Model the reference adapter's observed counters after native branching."""
    calls = audit["merge_calls"]
    audit["actual_merge_calls"] = sum(call["after"] < call["before"] for call in calls)
    audit["actual_removed_tokens"] = sum(call["before"] - call["after"] for call in calls)
    audit["merged_heads"] = [[call["layer"], call["batch"], call["head"]]
                             for call in calls if call["after"] < call["before"]]


class PairingValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_both_native_layouts_and_fixed_random_rankings_validate(self):
        for method in ("hybridkv_kvrepack", "snapkv_kvrepack"):
            contract = make_contract(self.root, method=method)
            result = validate_bundle(make_bundle(contract, 0), contract, json_sha(contract))
            self.assertEqual(set(result), set(LABELS))
            self.assertTrue(all(row["selected_count"] == row["effective_merge_count"] == 2 for row in result.values()))

    def test_hybrid_heterogeneous_work_is_kept_but_budget_change_is_rejected(self):
        contract = make_contract(self.root)
        bundle = make_bundle(contract, 0)
        result = validate_bundle(bundle, contract)
        self.assertGreater(len({row["removed_tokens"] for row in result.values()}), 1)
        bundle["records"]["random_global_seed0"]["audit"]["pre_selection"]["sparse_budget_per_head"][0][0][0] += 1
        with self.assertRaisesRegex(ValueError, "budget|pairing"):
            validate_bundle(bundle, contract)

    def test_hybrid_native_non_invocation_keeps_selected_quota_and_actual_effective_count(self):
        contract = make_contract(self.root)
        bundle = make_bundle(contract, 0)
        audit = bundle["records"]["random_global_seed0"]["audit"]
        audit["merge_calls"].pop()
        recount_hybrid(audit)
        result = validate_bundle(bundle, contract)
        self.assertEqual(result["random_global_seed0"]["selected_count"], 2)
        self.assertEqual(result["random_global_seed0"]["effective_merge_count"], 1)
        self.assertEqual(set(result["random_global_seed0"]),
                         {"selected_count", "effective_merge_count", "removed_tokens", "mask_hamming", "mask_jaccard"})
        self.assertEqual(merge_work_stats(audit), {
            "selected_count": 2, "merge_invocation_count": 1, "effective_merge_count": 1,
            "no_op_call_count": 0, "selected_without_invocation_count": 1,
            "removed_tokens": audit["actual_removed_tokens"]})

    def test_hybrid_native_noop_is_observed_but_not_counted_as_an_effective_merge(self):
        contract = make_contract(self.root)
        bundle = make_bundle(contract, 0)
        audit = bundle["records"]["random_global_seed1"]["audit"]
        audit["merge_calls"][1]["after"] = audit["merge_calls"][1]["before"]
        recount_hybrid(audit)
        result = validate_bundle(bundle, contract)
        self.assertEqual(result["random_global_seed1"]["effective_merge_count"], 1)
        stats = merge_work_stats(audit)
        self.assertEqual(stats["merge_invocation_count"], 2)
        self.assertEqual(stats["no_op_call_count"], 1)
        self.assertEqual(stats["selected_without_invocation_count"], 0)

    def test_hybrid_relaxed_native_work_still_rejects_forged_counters_and_invalid_calls(self):
        contract = make_contract(self.root)
        original = make_bundle(contract, 0)
        audit = original["records"]["random_global_seed0"]["audit"]
        audit["merge_calls"].pop()
        recount_hybrid(audit)
        validate_bundle(original, contract)
        mutations = [
            lambda row: row.update(actual_merge_calls=2),
            lambda row: row.update(actual_merge_calls=True),
            lambda row: row.update(actual_removed_tokens=row["actual_removed_tokens"] + 1),
            lambda row: row["merged_heads"].append([9, 0, 0]),
            lambda row: row["merge_calls"].append(deepcopy(row["merge_calls"][0])),
            lambda row: row["merge_calls"][0].update(after=0),
            lambda row: row["merge_calls"][0].update(after=row["merge_calls"][0]["before"] + 1),
        ]
        selected = audit["selected_masks"]
        unselected = next((layer, head) for layer in range(2) for head in range(2) if not selected[layer][0][head])
        mutations.append(lambda row: row["merge_calls"][0].update(layer=unselected[0], head=unselected[1]))
        for mutation in mutations:
            bundle = deepcopy(original)
            mutation(bundle["records"]["random_global_seed0"]["audit"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_bundle(bundle, contract)

    def test_hybrid_zero_effective_work_is_rejected_even_with_consistent_counters(self):
        contract = make_contract(self.root)
        for no_calls in (False, True):
            bundle = make_bundle(contract, 0)
            audit = bundle["records"]["similarity"]["audit"]
            if no_calls:
                audit["merge_calls"].clear()
            else:
                for call in audit["merge_calls"]:
                    call["after"] = call["before"]
            recount_hybrid(audit)
            with self.subTest(no_calls=no_calls), self.assertRaisesRegex(ValueError, "never exercised"):
                validate_bundle(bundle, contract)

    def test_snap_still_requires_the_original_strict_effective_merge_quota(self):
        contract = make_contract(self.root, method="snapkv_kvrepack")
        for no_call in (False, True):
            bundle = make_bundle(contract, 0)
            audit = bundle["records"]["random_global_seed0"]["audit"]
            if no_call:
                audit["merge_calls"].pop()
            else:
                audit["merge_calls"][1]["after"] = audit["merge_calls"][1]["before"]
            recount_hybrid(audit)
            with self.subTest(no_call=no_call), self.assertRaises(ValueError):
                validate_bundle(bundle, contract)

    def test_saved_pass_flag_does_not_override_input_mask_or_actual_work_tampering(self):
        contract = make_contract(self.root)
        original = make_bundle(contract, 0)
        mutations = [
            lambda row: row["records"].pop("random_global_seed2"),
            lambda row: row["records"]["random_global_seed0"]["input_fingerprints"].clear(),
            lambda row: row["records"]["random_global_seed0"]["audit"].update(actual_merge_calls=1),
            lambda row: row["records"]["random_global_seed0"]["audit"]["merge_calls"].append({}),
            lambda row: row["records"]["random_global_seed0"]["audit"].update(selected_masks=[[[True, True]], [[False, False]]]),
            lambda row: row["records"]["random_global_seed0"]["prediction"].update(question="different input"),
            lambda row: row.update(contract_sha256="0" * 64),
        ]
        for mutation in mutations:
            bundle = deepcopy(original)
            mutation(bundle)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_bundle(bundle, contract)

    def test_snap_cache_or_candidate_changes_cannot_pass(self):
        contract = make_contract(self.root, method="snapkv_kvrepack")
        bundle = make_bundle(contract, 0)
        bundle["records"]["random_global_seed1"]["audit"]["candidate_hashes"][0]["candidate_raw"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "candidates/cache/work"):
            validate_bundle(bundle, contract)

    def test_all_arms_cannot_share_the_wrong_generation_configuration(self):
        contract = make_contract(self.root)
        bundle = make_bundle(contract, 0)
        for row in bundle["records"].values():
            row["prediction"]["gen_kwargs"]["max_new_tokens"] = 128
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            validate_bundle(bundle, contract)

    def test_import_and_validation_do_not_load_inference_dependencies(self):
        script = "from kvrepack.ablations.milebench_head_selection_report import validate_bundle; import sys; assert all(m not in sys.modules for m in ('torch','transformers','accelerate'))"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("rouge"), "Original metrics require NumPy and rouge")
class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_both_tasks_keep_original_metric_and_negative_paired_differences(self):
        for task in ("Spot-the-Diff", "MMCoQA"):
            contract = make_contract(self.root / "data", task=task)
            bundles = [make_bundle(contract, index, answers={"similarity": "wrong"}) for index in range(2)]
            run = self.root / task
            write_run(run, contract, bundles)
            output = self.root / f"{task}-report"
            result = summarize_run(run, output)
            self.assertEqual(result["metric"], "Rouge-L f" if task == "Spot-the-Diff" else "Accuracy")
            self.assertEqual(result["arm_scores_percent"]["similarity"], 0)
            self.assertAlmostEqual(result["random_mean_percent"], 100, delta=1e-6)
            self.assertEqual(result["random_sample_sd_percent"], 0)
            self.assertAlmostEqual(result["similarity_minus_random_mean_points"], -100, delta=1e-6)
            self.assertTrue(result["full_task"])
            self.assertEqual(result["sample_count"], 2)
            with (output / "paired_scores.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(abs(float(row["similarity_minus_random_points"]) + 100) < 1e-6 for row in rows))
            with self.assertRaisesRegex(ValueError, "fresh"):
                summarize_run(run, output)

    def test_hybrid_report_preserves_csv_columns_and_real_effective_counts_with_native_skips(self):
        contract = make_contract(self.root / "data")
        bundles = [make_bundle(contract, index) for index in range(2)]
        for bundle in bundles:
            skipped = bundle["records"]["random_global_seed0"]["audit"]
            skipped["merge_calls"].pop()
            recount_hybrid(skipped)
            noop = bundle["records"]["random_global_seed1"]["audit"]
            noop["merge_calls"][1]["after"] = noop["merge_calls"][1]["before"]
            recount_hybrid(noop)
        run, output = self.root / "run", self.root / "report"
        write_run(run, contract, bundles)
        summary = summarize_run(run, output)
        self.assertEqual(summary["selected_head_quota"], 2)
        self.assertAlmostEqual(summary["random_mean_percent"], 100)
        with (output / "selector_audit.csv").open() as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, ["sample_id", "condition", "selected_count", "effective_merge_count",
                                                 "removed_tokens", "mask_hamming", "mask_jaccard"])
        reduced = [row for row in rows if row["condition"] in ("random_global_seed0", "random_global_seed1")]
        self.assertEqual(len(reduced), 4)
        self.assertTrue(all(row["selected_count"] == "2" and row["effective_merge_count"] == "1" for row in reduced))

    def test_incomplete_or_changed_annotation_is_rejected_before_output_creation(self):
        contract = make_contract(self.root / "data")
        run, output = self.root / "run", self.root / "report"
        write_run(run, contract)
        state_path = run / "ablation_state.json"
        state = json.loads(state_path.read_text())
        state["completed_generations"] = 7
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize_run(run, output)
        self.assertFalse(output.exists())
        state["completed_generations"] = 8
        state_path.write_text(json.dumps(state))
        source = self.root / "data/MMCoQA/MMCoQA.json"
        source.write_text(source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "annotation"):
            summarize_run(run, output)
        self.assertFalse(output.exists())

    def test_unknown_sample_file_and_changed_bound_source_are_rejected(self):
        contract = make_contract(self.root / "data")
        run, output = self.root / "run", self.root / "report"
        write_run(run, contract)
        (run / "samples/000002.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "filenames"):
            summarize_run(run, output)
        (run / "samples/000002.json").unlink()
        contract["source_files_sha256"] = {"src/kvrepack/ablations/milebench_head_selection_report.py": "0" * 64}
        (run / "run_contract.json").write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "source changed"):
            summarize_run(run, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

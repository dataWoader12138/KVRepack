"""Synthetic CPU fixtures verify report accounting, never model performance."""
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

from kvrepack.ablations.run_head_selection import json_sha

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("styled_image_report", REPO / "scripts/summarize_styled_milebench.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fingerprint(value):
    return {"shape": [2], "dtype": "torch.int64", "sha256": hashlib.sha256(value.encode()).hexdigest()}


def fixture(root, *, all_samples=True, arms=report.ARMS, tasks=report.TASKS, warmup=0):
    root.mkdir(parents=True, exist_ok=True)
    data, matrix = root / "data", root / "matrix"
    source = "src/kvrepack/benchmarks/milebench/protocol.py"
    source_map = {source: report.file_sha(REPO / source)}
    kernel = root / "fixture_kernel.so"
    kernel.write_bytes(b"synthetic non-executable fixture; no CUDA code")
    runs = []
    for task in tasks:
        annotation = {"meta_data": {"question_type": "open-ended", "task_instruction": ["Compare."], "num_sample": 2},
                      "data": [{"sample_id": value, "task_instruction_id": 0, "response": "red", "image_quantity_level": "Few",
                                "task_instance": {"context": "{image#1} Change?", "images_path": ["image.jpg"]}}
                               for value in (9, 3)]}
        ann = data / task / (task + ".json")
        save(ann, annotation)
        for arm in arms:
            run = matrix / arm / task
            count = 2 if all_samples else 1
            standalone = arm in ("asymkv", "lookm")
            backbone, variant = (None, None) if standalone else arm.split("_", 1)
            method = arm if standalone else backbone + "_kvrepack"
            policy = ("original_asymkv_official_v1" if arm == "asymkv" else
                      "original_lookm_history_plus_recent_v1" if arm == "lookm" else "original_history_plus_local_v1")
            generation = {"max_new_tokens": 4, "do_sample": False, "num_beams": 1}
            contract = {"schema_version": 1, "experiment": "styled_milebench_image_response_e2e", "source_commit": "b9104d8",
                        "arm": arm, "task": task, "method": method, "backbone": backbone, "variant": variant, "standalone": standalone,
                        "model_type": "qwen2_5_vl-7b", "head_geometry": {"layers": 28, "native_kv_heads": 4},
                        "data_root": str(data), "all_samples": all_samples, "limit": count, "annotation_total_samples": 2,
                        "selected_sample_ids": [9, 3][:count], "generation_sample_ids": [9, 3][:count],
                        "annotation_sha256": report.file_sha(ann), "source_sha256": source_map, "warmup_per_arm": warmup,
                        "algorithm_env": {"BUDGET_RATIO": "0.1"}, "generation": generation, "image": {},
                        "model_metadata_sha256": {"config.json": "3" * 64}, "dtype": "bfloat16", "attention": "flash_attention_2",
                        "model_seed": 42, "budget_policy": policy}
            if backbone == "hybridkv":
                contract["kernel_binary"] = {"path": str(kernel), "sha256": report.file_sha(kernel)}
            save(run / "run_contract.json", contract)
            timed_sum = 0
            for index, sample_id in enumerate(contract["generation_sample_ids"]):
                n = 1025 if index == 0 else 600
                ids = ([99, 10] if arm == "asymkv" else [10, 99]) if index == 0 else [10, 11, 12, 13]
                history = int(.1*n) + 32
                shapes = [{"layer": i, "key_shape": [1, 4, history + len(ids) - 1, 128],
                           "value_shape": [1, 4, history + len(ids) - 1, 128]} for i in range(28)]
                physical = 112 * history
                if arm == "asymkv":
                    covered = n-1 if n % 512 == 1 else n
                    physical = 112 * covered
                    audit = {"method": "AsymKV-official", "prompt_length": n, "prefill_logical_tokens": covered,
                             "prefill_physical_lengths": [covered] * 28, "compression_events": [],
                             "configured_cache_size": int(.1*n),
                             "runtime": {"budget_ratio": .1, "absolute_cache_size": None, "start_size": 32, "chunk_size": 512}}
                elif arm == "lookm":
                    audit = {"final_cache_shapes": shapes, "physical_prefill_positions": physical}
                else:
                    call = {key: fingerprint(key) for key in ("positions", "group_ids", "group_size", "start_mask")}
                    audit = {"status": "complete", "variant": variant, "merge_calls": [call]}
                    if backbone == "snapkv":
                        audit.update(selected_masks=[fingerprint("mask")] * 28,
                                     layers=[{"layer": i, "candidate_idx": fingerprint("candidate"), "final_history": history-32,
                                              "candidate_history": history} for i in range(28)],
                                     prefill_cache_shapes=[[1, 4, history, 128]] * 28)
                    else:
                        call.update(layer_idx=0, batch_idx=0, head_idx=0, before=20, after=10)
                        native = {"native_budget_unchanged": True, "post_allocation_scaling": False, "scale": 1.,
                                  "original_history_budget": [history-32], "history_budget": [history-32],
                                  "original_history_positions": 112*(history-32), "history_positions": 112*(history-32),
                                  "local_positions": 112*32, "active_prefill_positions": physical, "capacity_positions": physical}
                        audit.update(merge_mask=fingerprint("mask"), budget=native, cache_shapes=shapes)
                timed_fisher = .2 if variant == "asymkv" else 0.
                generation_seconds = .5 + index
                total = .1 + .1 + timed_fisher + generation_seconds + .1
                timed_sum += total
                row = {"schema_version": 1, "status": "completed", "arm": arm, "method": method,
                       "backbone": backbone, "variant": variant, "standalone": standalone, "task": task,
                       "sample_index": index, "sample_id": sample_id, "doc_id": sample_id, "warmup": False,
                       "contract_sha256": json_sha(contract),
                       "prediction": {"sample_id": sample_id, "pred_response": "red" if index == 0 else "blue",
                                      "gt_response": "red", "image": ["image.jpg"], "question": "Compare. Change?",
                                      "gen_kwargs": generation, "gen_model_id": "qwen2.5-vl-7b"},
                       "input_tokens": n, "input_fingerprints": {
                           "input_hashes": {key: hashlib.sha256(key.encode()).hexdigest() for key in ("input_ids", "pixel_values", "image_grid_thw")},
                           "input_shapes": {"input_ids": [1, n], "pixel_values": [1, 3], "image_grid_thw": [1, 3]},
                           "input_dtypes": {"input_ids": "torch.int64", "pixel_values": "torch.float32", "image_grid_thw": "torch.int64"}},
                       "generated_ids": ids, "generated_tokens": len(ids), "eos_token_id": 99,
                       "eos_reached": ids[-1] == 99, "first_eos_index": ids.index(99) if 99 in ids else None,
                       "hit_max_new_tokens": len(ids) == 4, "capped_without_eos": len(ids) == 4 and ids[-1] != 99,
                       "audit": audit, "budget": {"prefill_tokens_total": physical, "actual_retained_ratio": physical/(112*n),
                                                  "policy": policy, "final_shapes": shapes},
                       "generation_includes_fisher": arm == "asymkv", "separately_timed_fisher": variant == "asymkv",
                       "fisher": {"synthetic": True} if variant == "asymkv" else None,
                       "runtime": {"preprocessing_seconds": .1, "h2d_seconds": .1, "fisher_seconds": None if arm == "asymkv" else timed_fisher,
                                   "external_fisher_seconds": timed_fisher, "generation_seconds": generation_seconds,
                                   "text_decode_seconds": .1, "model_end_to_end_seconds": timed_fisher+generation_seconds,
                                   "end_to_end_seconds": total, "peak_allocated_gib": 1.}}
                if arm == "asymkv":
                    row.update(asymkv_prompt_tokens_covered=covered, asymkv_prompt_tokens_missing=n-covered,
                               asymkv_boundary_affected=covered != n)
                save(run / "samples" / f"{index:06d}.json", row)
                if index == 0:
                    for j in range(warmup):
                        warm = deepcopy(row); warm["warmup"] = True
                        save(run / "warmup" / f"{j:06d}.json", warm)
            save(run / "run_state.json", {"status": "complete", "contract_sha256": json_sha(contract),
                "target_samples": count, "completed_samples": count, "target_generations": count,
                "completed_generations": count, "model_load_seconds": .3,
                "formal_loop_seconds": timed_sum+.7, "process_seconds": timed_sum+2.})
            runs.append({"arm": arm, "task": task, "output": str(run), "status": "complete"})
    save(matrix / "experiment_state.json", {"schema_version": 1, "status": "scoring", "arms": list(arms),
        "tasks": list(tasks), "completed_runs": len(runs), "runs": runs})
    return matrix


class StyledImageReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.matrix = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def alter(self, path, change):
        value = report.read_json(path); change(value); save(path, value)

    def first(self, arm="hybridkv_asymkv"):
        return self.matrix / arm / "Spot-the-Diff" / "samples/000000.json"

    def test_complete_matrix_scores_exact_ids_and_exposes_original_asymkv_semantics(self):
        before = {p: report.file_sha(p) for p in self.matrix.rglob("*") if p.is_file()}
        result = report.summarize(self.matrix, self.root / "report")
        self.assertTrue(result["full_experiment"])
        self.assertEqual(result["formal_generations"], 32)
        self.assertEqual(len(result["results"]), 16)
        self.assertTrue(all(abs(row["score_percent"]-50)<1e-5 for row in result["results"]))
        native = next(row for row in result["results"] if row["arm"] == "asymkv" and row["task"] == "Spot-the-Diff")
        self.assertIsNone(native["fisher_mean_seconds"])
        self.assertEqual(native["asymkv_boundary_affected_count"], 1)
        self.assertEqual(native["asymkv_missing_prompt_tokens_total"], 1)
        self.assertEqual(native["early_eos_count"], 1)
        self.assertEqual(native["post_first_eos_tokens_total"], 1)
        self.assertEqual(native["prefill_retained_ratio_max"], 1.)
        self.assertEqual(before, {p: report.file_sha(p) for p in self.matrix.rglob("*") if p.is_file()})
        with (self.root / "report/per_input_latency.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 32)
        self.assertEqual(next(row for row in rows if row["arm"]=="asymkv")["fisher_seconds"], "")

    def test_smoke_subset_is_explicitly_not_full_and_warmup_is_excluded(self):
        smoke = fixture(self.root / "small", all_samples=False, arms=("asymkv",), tasks=("MMCoQA",), warmup=1)
        result = report.summarize(smoke, self.root / "smoke_report")
        self.assertFalse(result["full_experiment"])
        self.assertEqual(result["formal_generations"], 1)
        self.assertEqual(result["warmup_generations_excluded"], 1)
        self.assertFalse(result["results"][0]["full_task"])

    def test_incomplete_child_missing_file_or_warmup_in_samples_cannot_score(self):
        for mutation in (lambda: self.first().unlink(),
                         lambda: self.alter(self.first(), lambda r:r.update(warmup=True)),
                         lambda: self.alter(self.matrix/"experiment_state.json", lambda r:r.update(status="generating"))):
            with tempfile.TemporaryDirectory() as work:
                self.matrix = fixture(Path(work))
                mutation()
                with self.assertRaises(ValueError): report.summarize(self.matrix, Path(work)/"report")

    def test_generation_and_e2e_tampering_is_rejected(self):
        mutations = [lambda r:r.update(generated_tokens=9), lambda r:r.update(eos_reached=False),
                     lambda r:r["runtime"].update(end_to_end_seconds=100),
                     lambda r:r["runtime"].update(model_end_to_end_seconds=100),
                     lambda r:r.update(variant="lookm")]
        for mutation in mutations:
            row = report.read_json(self.first())
            contract = report.read_json(self.first().parents[1]/"run_contract.json")
            mutation(row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_sample(row, contract, 0, 9)

    def test_hf_internal_eos_is_rejected_but_official_asymkv_is_preserved(self):
        path = self.first("lookm"); row = report.read_json(path)
        contract = report.read_json(path.parents[1]/"run_contract.json")
        row.update(generated_ids=[99,10], eos_reached=False, first_eos_index=0)
        with self.assertRaisesRegex(ValueError,"EOS"):
            report.validate_sample(row, contract, 0, 9)
        path = self.first("asymkv"); row = report.read_json(path)
        derived = report.validate_sample(row, report.read_json(path.parents[1]/"run_contract.json"), 0, 9)
        self.assertTrue(derived["early_eos"])
        self.assertEqual(row["generated_ids"], [99,10])

    def test_pairing_rechecks_candidates_and_input_hashes_not_only_flags(self):
        records = {}
        for arm in ("hybridkv_asymkv", "hybridkv_lookm", "hybridkv_kvrepack"):
            records[arm,"Spot-the-Diff"] = report.read_run(self.matrix/arm/"Spot-the-Diff",arm,"Spot-the-Diff")
        records["hybridkv_lookm","Spot-the-Diff"][1][9]["audit"]["merge_calls"][0]["positions"] = fingerprint("changed")
        with self.assertRaisesRegex(ValueError,"groups/anchors"):
            report.validate_pairing(records)

    def test_physical_cache_and_original_budget_counters_are_independently_recomputed(self):
        row = report.read_json(self.first("lookm"))
        row["audit"]["final_cache_shapes"][0]["key_shape"][2] += 1
        with self.assertRaisesRegex(ValueError,"K/V shapes"):
            report.observed_budget(row,"lookm")
        row = report.read_json(self.first("asymkv"))
        row["audit"]["prefill_logical_tokens"] -= 1
        with self.assertRaisesRegex(ValueError,"prompt coverage"):
            report.observed_budget(row,"asymkv")

    def test_changed_bound_sources_kernel_or_annotation_are_rejected(self):
        path = self.matrix/"hybridkv_asymkv/Spot-the-Diff/run_contract.json"
        original = report.read_json(path)
        for change in (lambda c:c["source_sha256"].update({next(iter(c["source_sha256"])):"0"*64}),
                       lambda c:c["kernel_binary"].update(sha256="0"*64),
                       lambda c:c.update(annotation_sha256="0"*64)):
            updated = deepcopy(original); change(updated)
            save(path, updated)
            state = report.read_json(path.with_name("run_state.json"));state["contract_sha256"]=json_sha(updated);save(path.with_name("run_state.json"),state)
            with self.assertRaises(ValueError):
                report.read_run(path.parent,"hybridkv_asymkv","Spot-the-Diff")

    def test_report_output_cannot_overwrite_source_or_existing_directory(self):
        with self.assertRaises(ValueError): report.summarize(self.matrix,self.matrix)
        with self.assertRaises(ValueError): report.summarize(self.matrix,REPO/"src/forbidden")
        report.summarize(self.matrix,self.root/"report")
        with self.assertRaises(ValueError): report.summarize(self.matrix,self.root/"report")

    def test_import_and_help_do_not_load_inference_dependencies(self):
        code = f"""
import importlib.util,sys
s=importlib.util.spec_from_file_location('r',{str(REPO/'scripts/summarize_styled_milebench.py')!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
for name in ('torch','transformers','PIL','numpy','rouge'):
    assert name not in sys.modules,name
"""
        subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True)


if __name__ == "__main__":
    unittest.main()

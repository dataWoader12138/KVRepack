"""CPU-only checks for the new full-data matrix and original decoder policies."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kvrepack.ablations import styled_milebench_worker as worker

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("styled_milebench_controller", ROOT / "scripts/run_styled_milebench.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class Policies(unittest.TestCase):
    def test_exact_eight_arm_matrix_distinguishes_originals_and_aggregators(self):
        self.assertEqual(len(worker.ARMS), 8)
        specs = [worker.arm_spec(arm) for arm in worker.ARMS]
        self.assertEqual(sum(row["standalone"] for row in specs), 2)
        self.assertEqual({(row["backbone"], row["variant"]) for row in specs if not row["standalone"]},
                         {(base, variant) for base in ("hybridkv", "snapkv") for variant in ("asymkv", "lookm", "kvrepack")})
        with self.assertRaises(ValueError): worker.arm_spec("lookm_kvrepack")

    def test_original_asym_decoder_does_not_receive_rejected_hf_controls(self):
        for arm in worker.ARMS:
            kwargs = worker.generation_policy(arm)
            self.assertEqual(kwargs["max_new_tokens"], 512)
            self.assertFalse(kwargs["do_sample"])
            self.assertEqual(kwargs["num_beams"], 1)
            if arm == "asymkv":
                self.assertNotIn("min_new_tokens", kwargs)
                self.assertNotIn("repetition_penalty", kwargs)
            else:
                self.assertEqual(kwargs["min_new_tokens"], 1)
                self.assertEqual(kwargs["repetition_penalty"], 1.05)

    def test_all_algorithm_environments_keep_image_budget_and_method_specific_gates(self):
        with patch.dict(os.environ, {**os.environ, "ASYMKV_OFFICIAL_CACHE_SIZE": "123", "ASYMKV_AUDIT_PATH": "should_not_write"}, clear=True):
            for arm in worker.ARMS:
                args = SimpleNamespace(arm=arm, pretrained=Path("/tmp/Qwen2.5-VL-7B-Instruct"),
                    hf_home=Path("/tmp/cache"), output=Path("/tmp/fake-only-plan"))
                spec, env = worker.configure(args)
                self.assertEqual(env["BUDGET"], "512")
                self.assertEqual(float(env["BUDGET_RATIO"]), .1)
                self.assertEqual(float(env["RATIO"]), .1)
                self.assertEqual(os.environ["MODEL_TYPE"], "qwen2_5_vl-7b")
                self.assertNotIn("ASYMKV_AUDIT_PATH", os.environ)
                self.assertNotIn("ASYMKV_OFFICIAL_CACHE_SIZE", os.environ)
                if spec["backbone"] == "hybridkv":
                    self.assertEqual(float(env["HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO"]), .5)
                    self.assertEqual(float(env["HYBRID_KVREPACK_COMPRESS_RATIO"]), .5)
                if spec["backbone"] == "snapkv":
                    self.assertEqual(float(env["SNAPKV_SIMHEAD_MERGE_RATIO"]), .3)
                    self.assertEqual(float(env["SNAPKV_KVREPACK_COMPRESS_RATIO"]), .7)

    def test_first_eos_is_preserved_even_if_original_asym_continues(self):
        self.assertEqual(worker.first_eos_index([9, 4, 9], 9), 0)
        self.assertEqual(worker.first_eos_index([4, 9], 9), 1)
        self.assertIsNone(worker.first_eos_index([4, 8], 9))
        self.assertIsNone(worker.first_eos_index([], 9))

    def test_resource_hash_freeze_covers_head_calibration(self):
        sources = worker.source_files()
        paths = list((ROOT / "src/kvrepack/data/head_score").rglob("*.json"))
        self.assertTrue(paths)
        self.assertTrue(all(str(path.relative_to(ROOT)) in sources for path in paths))


class FullData(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for task in worker.TASKS:
            p = self.root/task
            (p/"images").mkdir(parents=True)
            Image.new("RGB", (3, 3)).save(p/"images/one.jpg")
            rows = [{"sample_id": i, "task_instruction_id": 0,
                     "task_instance": {"context": "{image#1}Difference:", "images_path": ["one.jpg"]},
                     "response": "fixture"} for i in range(301)]
            (p/(task+".json")).write_text(json.dumps({"meta_data": {"num_sample":301,
                "question_type":"open-ended", "task_instruction":["Compare."]}, "data":rows}))

    def tearDown(self):
        self.temp.cleanup()

    def test_full_annotations_above_200_have_no_implicit_cap(self):
        ann, ordered, preflight = worker.select_annotation(self.root, worker.TASKS[0])
        self.assertEqual(len(ann["data"]), 301)
        self.assertEqual(len(ordered), 301)
        self.assertEqual(preflight["sample_ids"], list(range(301)))
        args = SimpleNamespace(arms=worker.ARMS, tasks=worker.TASKS, data_root=self.root,
                               output=self.root/"unused", limit=None)
        plans = controller.planned_runs(args)
        self.assertEqual(len(plans), 16)
        self.assertEqual(sum(row["target_samples"] for row in plans), 16*301)
        self.assertTrue(all(row["all_samples"] for row in plans))
        self.assertFalse((self.root/"unused").exists())

    def test_explicit_smoke_and_invalid_prefix_are_not_silent_full_runs(self):
        ann, ordered, info = worker.select_annotation(self.root, worker.TASKS[0], 1)
        self.assertEqual(len(ordered), 1)
        self.assertEqual(info["total_samples"], 301)
        for limit in (0, -1, 302):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                worker.select_annotation(self.root, worker.TASKS[0], limit)


if __name__ == "__main__":
    unittest.main()

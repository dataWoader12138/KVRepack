"""CPU-only contracts for Random-head image planning and serial control flow."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.experiments.random_head_images import (
    DEFAULT_PRETRAINED, METHODS, REFERENCE_COMMIT, TASKS,
    build_matrix, build_random_head_plan, conditions,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_milebench_random_heads.py"
spec = importlib.util.spec_from_file_location("milebench_random_head_controller", SCRIPT)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class RandomHeadPlanTests(unittest.TestCase):
    def matrix(self, output, **kwargs):
        return build_matrix("/external/milebench", output, environ={}, **kwargs)

    def test_default_matrix_is_four_paired_7b_runs_with_reference_merge_recipes(self):
        plans = self.matrix("/external/new-group")
        self.assertEqual(len(plans), 4)
        self.assertEqual([(p.resolved_config["public_method"], p.resolved_config["task"]) for p in plans],
                         [(method, task) for method in METHODS for task in TASKS])
        for plan in plans:
            config = plan.resolved_config
            method = config["public_method"]
            self.assertEqual(config["model"], "qwen2_5_vl")
            self.assertEqual(config["model_type"], "qwen2_5_vl-7b")
            self.assertEqual(config["pretrained"], DEFAULT_PRETRAINED)
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["generation"]["max_new_tokens"], 512)
            self.assertEqual(config["generation"]["repetition_penalty"], 1.05)
            self.assertEqual(config["image"]["torch_dtype"], "bfloat16")
            self.assertEqual(config["ablation"]["expected_geometry"], {"layers": 28, "native_kv_heads": 4})
            self.assertEqual(config["ablation"]["reference_commit"], REFERENCE_COMMIT)
            self.assertEqual([arm["label"] for arm in config["ablation"]["conditions"]],
                             ["similarity", "random_global_seed0", "random_global_seed1", "random_global_seed2"])
            self.assertEqual(plan.command[2], "kvrepack.ablations.run_milebench_head_selection")
            self.assertEqual(plan.command[-1], str(plan.output / "resolved_config.json"))
            self.assertEqual(config["algorithm_env"]["BUDGET"], "512")
            self.assertEqual(config["algorithm_env"]["RATIO"], "0.1")
            self.assertEqual(float(config["algorithm_env"]["BUDGET_RATIO"]), 0.1)
            self.assertEqual(config["algorithm_env"]["K_DIFF_ALPHA"], "2")
            if method == "hybridkv_kvrepack":
                self.assertEqual(config["algorithm_env"]["HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO"], "0.5")
                self.assertEqual(config["algorithm_env"]["HYBRID_KVREPACK_COMPRESS_RATIO"], "0.5")
                self.assertEqual(config["algorithm_env"]["HEAD_CLASSIFY"], "head_ada")
                self.assertEqual(config["algorithm_env"]["BUDGET_METHOD"], "head_ada")
            else:
                self.assertEqual(config["algorithm_env"]["SNAPKV_SIMHEAD_MERGE_RATIO"], "0.30")
                self.assertEqual(config["algorithm_env"]["SNAPKV_KVREPACK_COMPRESS_RATIO"], "0.70")

    def test_selector_seeds_do_not_change_model_seed_and_explicit_cache_is_consistent(self):
        plan = build_random_head_plan(
            "snapkv_kvrepack", "MMCoQA", "/data", output="/new/run", head_seeds=(7, 9),
            seed=123, hf_home="/cache-root", budget_ratio=0.2, max_new_tokens=23,
            environ={"HF_HOME": "/stale", "HF_DATASETS_CACHE": "/other-stale", "BUDGET": "9", "HF_TOKEN": "private-test-value"},
        )
        self.assertEqual(plan.resolved_config["seed"], 123)
        self.assertEqual([arm["mask_seed"] for arm in plan.resolved_config["ablation"]["conditions"]], [0, 7, 9])
        self.assertEqual(plan.env["HF_HOME"], "/cache-root")
        self.assertEqual(plan.env["HF_DATASETS_CACHE"], "/cache-root/datasets")
        self.assertEqual(plan.env["HF_HUB_CACHE"], "/cache-root/hub")
        self.assertEqual(plan.env["BUDGET"], "512")
        self.assertEqual(plan.env["RATIO"], "0.1")
        self.assertEqual(plan.resolved_config["generation"]["max_new_tokens"], 23)
        self.assertNotIn("private-test-value", json.dumps(plan.to_dict()))

    def test_invalid_scopes_seeds_and_protocol_sizes_are_rejected(self):
        for seeds in ((), (0, 0), (-1,), (True,), (0.1,)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                conditions(seeds)
        for methods, tasks in (((), TASKS), (("snapkv",), TASKS), (("snapkv_kvrepack",) * 2, TASKS),
                               (METHODS, ()), (METHODS, ("DocVQA",)), (METHODS, ("MMCoQA",) * 2)):
            with self.assertRaises(ValueError):
                self.matrix("/external/new-group", methods=methods, tasks=tasks)
        for limit in (0, 201, True):
            with self.assertRaises(ValueError):
                self.matrix("/external/new-group", limit=limit)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.matrix("/external/new-group", pretrained="Qwen/Qwen2.5-VL-3B-Instruct")

    def test_cli_default_is_no_write_no_launch_no_inference_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "not-created"
            script = """
import runpy, sys
sys.argv = [sys.argv[1], '--data-root', sys.argv[2], '--output-root', sys.argv[3]]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not any(name in sys.modules for name in ('torch', 'transformers', 'accelerate', 'PIL', 'numpy'))
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(SCRIPT), str(root / "missing-data"), str(output)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(result.stdout)
            self.assertFalse(payload["executed"])
            self.assertEqual(payload["planned_runs"], 4)
            self.assertEqual(payload["requested_total_generations"], 3200)
            self.assertFalse(output.exists())
            self.assertFalse((root / "missing-data").exists())
            with patch.object(controller, "execute_matrix") as launch, redirect_stdout(io.StringIO()):
                self.assertEqual(controller.main(["--data-root", "/missing-data", "--output-root", str(output)]), 0)
            launch.assert_not_called()


class RandomHeadControllerTests(unittest.TestCase):
    def plans(self, root):
        return build_matrix("/external/milebench", root, environ={}, limit=1)

    def test_execution_is_serial_generation_then_scoring_for_each_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "group"
            plans = self.plans(output)
            calls = []
            def run(plan):
                calls.append(("generate", str(plan.output)))
                return 0
            def report(run_root, output_root):
                self.assertEqual(output_root, run_root / "report")
                calls.append(("score", str(run_root)))
            state = controller.execute_matrix(plans, output, runner=run, reporter=report)
            self.assertEqual(calls, [entry for plan in plans for entry in (("generate", str(plan.output)), ("score", str(plan.output)))])
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed_runs"], 4)
            self.assertTrue(all(row["status"] == "complete" for row in state["runs"]))
            self.assertEqual(json.loads((output / "controller_state.json").read_text())["status"], "complete")

    def test_generation_failure_stops_remaining_tasks_and_records_nonzero_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "group"
            plans = self.plans(output)
            calls = []
            def fail(plan):
                calls.append(plan.output)
                return 7
            reporter = unittest.mock.Mock()
            with self.assertRaisesRegex(RuntimeError, "Generation returned 7"):
                controller.execute_matrix(plans, output, runner=fail, reporter=reporter)
            self.assertEqual(calls, [plans[0].output])
            reporter.assert_not_called()
            state = json.loads((output / "controller_state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["completed_runs"], 0)
            self.assertEqual(state["runs"][0]["returncode"], 7)
            self.assertTrue(all(row["status"] == "pending" for row in state["runs"][1:]))

    def test_scoring_failure_stops_next_run_without_claiming_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "group"
            plans = self.plans(output)
            runner = unittest.mock.Mock(return_value=0)
            reporter = unittest.mock.Mock(side_effect=ValueError("incomplete sample bundle"))
            with self.assertRaisesRegex(ValueError, "incomplete sample bundle"):
                controller.execute_matrix(plans, output, runner=runner, reporter=reporter)
            self.assertEqual(runner.call_count, 1)
            state = json.loads((output / "controller_state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["completed_runs"], 0)
            self.assertEqual(state["runs"][0]["status"], "failed")

    def test_existing_experiment_group_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "group"
            output.mkdir()
            marker = output / "controller_state.json"
            marker.write_text("old state")
            runner = unittest.mock.Mock(return_value=0)
            with self.assertRaises(FileExistsError):
                controller.execute_matrix(self.plans(output), output, runner=runner, reporter=lambda *args: None)
            runner.assert_not_called()
            self.assertEqual(marker.read_text(), "old state")


if __name__ == "__main__":
    unittest.main()

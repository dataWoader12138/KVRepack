"""Pure launch planning and source isolation for the image backend."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.experiments.milebench import IMAGE_METHODS, load_profile
from kvrepack.experiments.runner import build_plan, child_environment, verify_milebench_sources
from kvrepack.run import main


class ImagePlanTests(unittest.TestCase):
    def plan(self, method="hybridkv_kvrepack", **kwargs):
        return build_plan(method, benchmark="milebench", data_root="/external/milebench", environ={}, **kwargs)

    def test_complete_candidate_matrix_uses_one_core_and_no_video_adapter(self):
        for profile in ("image8", "main11"):
            for model in ("qwen2_5_vl", "llava_ov"):
                for method in IMAGE_METHODS:
                    with self.subTest(profile=profile, model=model, method=method):
                        plan = self.plan(method, profile=profile, model=model)
                        self.assertIn("kvrepack.benchmarks.milebench.generate", plan.command)
                        self.assertNotIn("lmms_eval", plan.command)
                        self.assertNotIn("frames", plan.resolved_config)
                        self.assertNotIn("frames", plan.provenance["selected_method"]["defaults"])
                        self.assertEqual(plan.env["METHOD"], method)
                        self.assertEqual(plan.resolved_config["generation"]["max_new_tokens"], 512)
                        self.assertEqual(plan.resolved_config["batch_size"], 1)
                        self.assertEqual(plan.resolved_config["model_type"], "qwen2_5_vl-3b" if model == "qwen2_5_vl" else "llava-ov-7b")
                        self.assertIn("GPU", plan.provenance["validation_status"])
                        self.assertTrue(all(not module.startswith("hybridkv") for module in plan.provenance["selected_method"]["implementation"]))

    def test_source_budgets_and_independent_ratios(self):
        qwen = self.plan(budget_ratio=0.25)
        self.assertEqual(qwen.env["BUDGET"], "512")
        self.assertEqual(qwen.env["BUDGET_RATIO"], "0.25")
        self.assertEqual(qwen.env["RATIO"], "0.1")
        self.assertEqual(qwen.env["HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO"], "0.50")
        sparse = self.plan("sparsemm_kvrepack", budget_ratio=0.25)
        self.assertEqual(sparse.env["BUDGET"], "1024")
        self.assertEqual(sparse.env["SPARSEMM_KVREPACK_BUDGET_RATIO"], "0.25")
        self.assertEqual(sparse.env["SPARSEMM_KVREPACK_RATIO"], "0.10")
        llava = self.plan(model="llava_ov")
        self.assertEqual(llava.env["BUDGET"], "1024")
        self.assertEqual(llava.env["HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO"], "0.30")
        video = build_plan("hybridkv_kvrepack", environ={}, budget_ratio=0.25)
        self.assertEqual(video.env["RATIO"], "0.25")
        self.assertEqual(video.env["HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO"], "0.30")

    def test_preprocessing_values_are_explicit_and_overrides_are_intentional(self):
        plan = self.plan()
        self.assertEqual(plan.resolved_config["image"]["min_pixels"], 7168)
        self.assertEqual(plan.resolved_config["image"]["message_min_pixels"], 200704)
        override = self.plan(min_pixels=1000, max_pixels=4000, seed=12, max_new_tokens=9)
        self.assertEqual(override.resolved_config["image"]["message_min_pixels"], 1000)
        self.assertEqual(override.resolved_config["generation"]["max_new_tokens"], 9)
        for args in ({"min_pixels": 2000, "max_pixels": 1000}, {"seed": -1}, {"max_new_tokens": 0}, {"frames": 3}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(**args)
        with self.assertRaisesRegex(ValueError, "Qwen image options"):
            self.plan(model="llava_ov", min_pixels=1000)

    def test_invalid_combinations_are_rejected_before_model_or_data_access(self):
        for method in ("randomkv", "asymkv", "lookm_kvrepack"):
            with self.assertRaisesRegex(ValueError, "not included"):
                self.plan(method)
        with self.assertRaises(ValueError):
            self.plan(task="nextqa_oe_test")
        with self.assertRaises(ValueError):
            self.plan(overrides=["HYBRID_KVREPACK_MERGE_SCORE_MODE=difference_only"])
        with self.assertRaises(ValueError):
            build_plan("fullkv", benchmark="milebench")
        with self.assertRaises(ValueError):
            build_plan("hybridkv_kvrepack", model="llava_ov")

    def test_dry_run_is_no_write_no_launch_and_does_not_require_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "never-created"
            stdout = io.StringIO()
            with patch("subprocess.run") as launch, redirect_stdout(stdout):
                result = main(["--benchmark", "milebench", "--method", "snapkv_kvrepack", "--task", "CLEVR-Change",
                               "--data-root", str(root / "missing-data"), "--output", str(output), "--dry-run"])
            self.assertEqual(result, 0)
            launch.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(stdout.getvalue())["resolved_config"]["benchmark"], "milebench")
        root = Path(__file__).resolve().parents[1]
        script = "import sys; from kvrepack.run import main; main(['--benchmark','milebench','--method','fullkv','--data-root','/missing','--dry-run']); assert not any(x in sys.modules for x in ('torch','transformers','accelerate','PIL','numpy'))"
        environment = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ambient_algorithm_flags_and_secrets_are_not_serialized(self):
        ambient = {"METHOD": "evil", "RATIO": "1", "BUDGET": "1", "PYTHONPATH": "/old-source",
                   "HF_TOKEN": "fixture-secret-value", "HF_HOME": "/cache"}
        plan = build_plan("snapkv_kvrepack", benchmark="milebench", data_root="/data", environ=ambient)
        self.assertEqual(plan.env["BUDGET"], "512")
        self.assertEqual(plan.env["SNAPKV_SIMHEAD_MERGE_RATIO"], "0.50")
        self.assertNotIn("fixture-secret-value", json.dumps(plan.to_dict()))
        self.assertEqual(child_environment(plan, ambient)["HF_TOKEN"], "fixture-secret-value")
        self.assertNotIn("/old-source", plan.env["PYTHONPATH"])

    def test_profile_lists_and_protected_outputs(self):
        self.assertEqual(len(load_profile("image8")["tasks"]), 8)
        self.assertEqual(len(load_profile("main11")["tasks"]), 11)
        root = Path(__file__).resolve().parents[1]
        for path in (root, root / "src/x", root / "provenance/x", root / "docs/x", root.parent):
            with self.assertRaises(ValueError):
                self.plan(output=path)

    def test_milebench_manifest_detects_mutated_backend(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = root / "src/kvrepack/benchmarks/milebench/dataset.py"
            backend.parent.mkdir(parents=True)
            backend.write_text("original")
            (root / "provenance").mkdir()
            payload = {"schema_version": 1, "targets": [{"path": str(backend.relative_to(root)), "sha256": hashlib.sha256(backend.read_bytes()).hexdigest()}]}
            (root / "provenance/milebench_sources.json").write_text(json.dumps(payload))
            self.assertEqual(len(verify_milebench_sources(root)["verified_targets"]), 1)
            backend.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_milebench_sources(root)


if __name__ == "__main__":
    unittest.main()

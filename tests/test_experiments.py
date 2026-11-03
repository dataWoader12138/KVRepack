"""CPU-only checks for configuration, source isolation, and experiment provenance."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kvrepack.experiments.registry import METHODS, METHODS_BY_MODEL, resolve_experiment
from kvrepack.experiments.runner import (
    build_plan, child_environment, run_plan, verify_import_sources, verify_manifest, verify_vendor_sources,
)
from kvrepack.run import main


class RegistryTests(unittest.TestCase):
    def test_exactly_one_implementation_per_public_method_and_baseline_pair(self):
        self.assertEqual(len(METHODS), 13)
        for name in ("hybridkv", "snapkv", "sparsemm", "madakv", "lookm"):
            upgraded = METHODS[name + "_kvrepack"]
            self.assertEqual(upgraded.baseline, name)
            self.assertTrue(upgraded.source_script.startswith("HybridKV-artifact/scripts/eval/"))
            self.assertEqual(upgraded.internal_method, upgraded.name)
        self.assertEqual(METHODS["asymkv"].internal_method, "asymkv_official")
        self.assertNotIn("asymkv_paper", METHODS)
        self.assertNotIn("lookm_headsplit_kvrepack", METHODS)

    def test_new_method_names_preserve_historical_source_paths_and_reject_old_names(self):
        sources = {
            "hybridkv_kvrepack": "qwen_ratio_hybrid_decouplekv_3b.sh",
            "snapkv_kvrepack": "qwen_ratio_snapkv_decouplekv_simhead_3b.sh",
            "sparsemm_kvrepack": "qwen_ratio_sparsemm_decouplekv_simhead_3b.sh",
            "madakv_kvrepack": "qwen_ratio_madakv_decouplekv_simhead_3b.sh",
            "lookm_kvrepack": "qwen_ratio_lookm_decouplekv_3b.sh",
        }
        for method, source in sources.items():
            with self.subTest(method=method):
                spec, env, _ = resolve_experiment(method)
                self.assertEqual(spec.source_script, "HybridKV-artifact/scripts/eval/" + source)
                self.assertFalse(any("DECOUPLE" in key for key in env))
                for key, parameter in spec.parameters.items():
                    if "KVREPACK" in key:
                        with self.assertRaisesRegex(ValueError, "not allowed"):
                            resolve_experiment(method, [key.replace("KVREPACK", "DECOUPLE") + "=" + parameter.default])
        for method in ("hybrid_decouplekv", "snapkv_decouplekv", "sparsemm_decouplekv",
                       "madakv_decouplekv", "lookm_decouplekv"):
            with self.subTest(legacy_method=method), self.assertRaisesRegex(ValueError, "Unknown method"):
                resolve_experiment(method)

    def test_adapter_method_guards_match_registry_without_importing_inference_packages(self):
        root = Path(__file__).resolve().parents[1]
        for model, adapter in (("qwen2_5_vl", "qwen2_5_vl"), ("llava_ov", "llava_hf")):
            path = root / "vendor/lmms-eval/lmms_eval/models" / (adapter + ".py")
            guards = [node.value for node in ast.walk(ast.parse(path.read_text()))
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "supported_methods"
                              for target in node.targets)]
            self.assertEqual(len(guards), 1)
            self.assertEqual(ast.literal_eval(guards[0]),
                             {spec.internal_method for spec in METHODS_BY_MODEL[model].values()})

    def test_selected_script_defaults_and_unset_budget_are_preserved(self):
        _, snap, _ = resolve_experiment("snapkv_kvrepack")
        self.assertEqual(snap["SNAPKV_SIMHEAD_MERGE_RATIO"], "0")
        self.assertEqual(snap["SNAPKV_KVREPACK_COMPRESS_RATIO"], "0.90")
        self.assertNotIn("BUDGET", snap)
        self.assertNotIn("BUDGET", resolve_experiment("randomkv")[1])
        self.assertEqual(resolve_experiment("hybridkv_kvrepack")[1]["HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO"], "0.30")
        self.assertEqual(resolve_experiment("sparsemm_kvrepack")[1]["K_DIFF_ALPHA"], "0")
        self.assertEqual(METHODS["madakv_kvrepack"].default_limit, 20000)
        for method in ("snapkv", "sparsemm", "madakv", "lookm"):
            _, baseline, _ = resolve_experiment(method)
            self.assertEqual(baseline, {"BUDGET": "1024", "BUDGET_RATIO": "0.1", "RATIO": "0.1"})

    def test_unknown_identity_cross_method_and_nonfinite_overrides_are_rejected(self):
        for value in ("METHOD=fullkv", "PYTHONPATH=/outside", "MADAKV_SIMHEAD_MERGE_RATIO=0.5",
                      "SNAPKV_SIMHEAD_MERGE_RATIO=nan", "SNAPKV_SIMHEAD_MERGE_RATIO=1.01",
                      "SNAPKV_KVREPACK_COMPRESS_RATIO=0", "SNAPKV_KVREPACK_WINDOW_SIZE=-1",
                      "SNAPKV_KVREPACK_KERNEL_SIZE=2",
                      "SNAPKV_KVREPACK_SCORE_CORRECTION=log_count", "K_DIFF_ALPHA=inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_experiment("snapkv_kvrepack", [value])
        with self.assertRaises(ValueError):
            resolve_experiment("sparsemm_kvrepack", ["SPARSEMM_KVREPACK_KERNEL_SIZE=4"])
        with self.assertRaises(ValueError):
            resolve_experiment("hybridkv_kvrepack", ["HYBRID_KVREPACK_CHUNK_SIZE=16"])
        with self.assertRaises(ValueError):
            resolve_experiment("snapkv_kvrepack", ["K_DIFF_ALPHA=1", "K_DIFF_ALPHA=2"])

    def test_explicit_dependencies_resolve_without_stale_aliases(self):
        _, env, _ = resolve_experiment("sparsemm_kvrepack", ["K_DIFF_ALPHA=3", "RATIO=0.2"], budget_ratio=0.25)
        self.assertEqual(env["SPARSEMM_KVREPACK_BUDGET_RATIO"], "0.25")
        self.assertEqual(env["SPARSEMM_KVREPACK_RATIO"], "0.2")
        self.assertEqual(env["SPARSEMM_KVREPACK_K_DIFF_ALPHA"], "3")
        _, env, _ = resolve_experiment("madakv_kvrepack", ["K_DIFF_ALPHA=3", "MADAKV_KVREPACK_K_DIFF_ALPHA=4"])
        self.assertEqual(env["MADAKV_KVREPACK_K_DIFF_ALPHA"], "4")
        with self.assertRaises(ValueError):
            resolve_experiment("snapkv", ["BUDGET_RATIO=0.2"], budget_ratio=0.2)


class PlanTests(unittest.TestCase):
    def test_ambient_algorithm_flags_and_import_paths_do_not_change_plan(self):
        ambient = {"PATH": "/bin", "HOME": "/tmp/home", "METHOD": "asymkv_paper",
                   "BUDGET": "99999", "HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO": "1",
                   "HYBRID_DECOUPLE_SIMHEAD_MERGE_RATIO": "1", "SNAPKV_DECOUPLE_COMPRESS_RATIO": "0.1",
                   "SNAPKV_SIMHEAD_MERGE_RATIO": "1", "K_DIFF_ALPHA": "9", "THRESHOLD": "0",
                   "PYTHONPATH": "/outside/HybridKV-artifact", "LMMS_EVAL_PLUGINS": "evil_plugin",
                   "OPENAI_API_KEY": "private-test-openai", "HF_TOKEN": "private-test-hf"}
        plan = build_plan("snapkv_kvrepack", task="nextqa_oe_test", environ=ambient)
        self.assertEqual(plan.env["METHOD"], "snapkv_kvrepack")
        self.assertEqual(plan.env["SNAPKV_SIMHEAD_MERGE_RATIO"], "0")
        child = child_environment(plan, ambient)
        self.assertNotIn("BUDGET", child)
        self.assertNotIn("HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO", child)
        self.assertNotIn("HYBRID_DECOUPLE_SIMHEAD_MERGE_RATIO", child)
        self.assertNotIn("SNAPKV_DECOUPLE_COMPRESS_RATIO", child)
        self.assertNotIn("LMMS_EVAL_PLUGINS", child)
        self.assertNotIn("THRESHOLD", child)
        self.assertNotIn("/outside", child["PYTHONPATH"])
        self.assertEqual(child["OPENAI_API_KEY"], "private-test-openai")
        self.assertEqual(child["HF_TOKEN"], "private-test-hf")
        serialized = json.dumps(plan.to_dict())
        self.assertNotIn("private-test-openai", serialized)
        self.assertNotIn("private-test-hf", serialized)
        self.assertNotIn("OPENAI_API_KEY", plan.env)
        self.assertNotIn("HF_TOKEN", plan.env)

    def test_launch_uses_current_interpreter_repo_vendor_and_internal_alias(self):
        plan = build_plan("hybridkv_kvrepack", task="nextqa_oe_test", pretrained="Qwen/Qwen2.5-VL-7B-Instruct",
                          frames=16, limit=2, budget_ratio=0.2, device="cuda:2", environ={})
        self.assertEqual(plan.command[:3], (sys.executable, "-m", "accelerate.commands.launch"))
        self.assertIn("lmms_eval", plan.command)
        self.assertIn("method=hybridkv_kvrepack", plan.command[plan.command.index("--model_args") + 1])
        self.assertEqual(plan.env["MODEL_TYPE"], "qwen2_5_vl-7b")
        self.assertEqual(plan.env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(plan.env["RATIO"], "0.2")
        self.assertEqual(plan.env["PYTHONPATH"].split(os.pathsep),
                         [str(plan.repo_root / "src"), str(plan.repo_root / "vendor/lmms-eval"), str(plan.repo_root / "csrc")])
        self.assertNotIn("HF_HUB_OFFLINE", plan.env)

    def test_invalid_model_input_task_size_limit_or_device_fails_before_launch(self):
        for args in ({"pretrained": "Qwen/Qwen2.5-VL-3B-Instruct", "model_type": "qwen2_5_vl-7b"},
                     {"pretrained": "/models/custom"}, {"pretrained": "foo,method=fullkv"},
                     {"pretrained": "llava-7b"}, {"task": "arbitrary"}, {"limit": 0},
                     {"frames": -1}, {"device": "cpu"}, {"device": "0,1"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                build_plan("snapkv", **args)
        plan = build_plan("snapkv", pretrained="/models/custom", model_type="qwen2_5_vl-3b")
        self.assertEqual(plan.env["MODEL_TYPE"], "qwen2_5_vl-3b")

    def test_cli_dry_run_does_not_write_or_launch_or_create_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, cache = root / "out", root / "never-created-cache"
            stdout = io.StringIO()
            with patch.dict(os.environ, {"HF_HOME": str(cache)}), patch("subprocess.run") as launch, redirect_stdout(stdout):
                code = main(["--method", "snapkv_kvrepack", "--task", "nextqa_oe_test", "--limit", "1",
                             "--output", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["resolved_config"]["limit"], 1)
            launch.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(cache.exists())

    def test_list_methods_is_cpu_only(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), patch("subprocess.run") as launch:
            self.assertEqual(main(["--list-methods"]), 0)
        self.assertEqual(set(json.loads(stdout.getvalue())), set(METHODS))
        launch.assert_not_called()

    def test_llava_baselines_use_their_selected_adapter_and_source_defaults(self):
        self.assertEqual(set(METHODS_BY_MODEL["llava_ov"]), {"fullkv", "hybridkv", "snapkv", "sparsemm", "madakv", "lookm"})
        for method in METHODS_BY_MODEL["llava_ov"]:
            with self.subTest(method=method):
                plan = build_plan(method, model="llava_ov", task="nextqa_oe_test", frames=8, limit=1, environ={})
                model_args = plan.command[plan.command.index("--model_args") + 1]
                self.assertIn("max_frames_num=8", model_args)
                self.assertNotIn("max_num_frames", model_args)
                self.assertNotIn("method=", model_args)
                self.assertIn("dtype=bfloat16", model_args)
                self.assertIn("device_map=auto", model_args)
                self.assertEqual(plan.command[plan.command.index("--model") + 1], "llava_hf")
                self.assertEqual(plan.env["MODEL_TYPE"], "llava-ov-7b")
                self.assertEqual(plan.env["METHOD"], method)
                self.assertEqual(plan.resolved_config["pretrained"], "llava-hf/llava-onevision-qwen2-7b-ov-hf")
                self.assertEqual(plan.provenance["selected_method"]["kernel"], None if method == "fullkv" else "kvrepack_kernel")
        source = build_plan("hybridkv", model="llava_ov").provenance["selected_method"]["source_script"]
        self.assertTrue(source.endswith("llava_ov_hybridkv.sh"))

    def test_unselected_llava_upgrades_and_qwen_model_type_conflicts_are_rejected(self):
        for method in ("randomkv", "asymkv", "hybridkv_kvrepack", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack", "lookm_kvrepack"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "not included"):
                build_plan(method, model="llava_ov")
        with self.assertRaises(ValueError):
            build_plan("snapkv", pretrained="/local/model", model_type="llava-ov-7b")
        with self.assertRaises(ValueError):
            build_plan("snapkv", model="llava_ov", model_type="qwen2_5_vl-7b")

    def test_cli_llava_dry_run_reports_the_model_dimension(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), patch("subprocess.run") as launch:
            self.assertEqual(main(["--model", "llava_ov", "--method", "snapkv", "--dry-run"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["resolved_config"]["model"], "llava_ov")
        launch.assert_not_called()

    def test_invalid_cli_override_does_not_launch(self):
        with redirect_stderr(io.StringIO()), patch("subprocess.run") as launch:
            self.assertEqual(main(["--method", "hybridkv", "--set", "METHOD=fullkv", "--dry-run"]), 2)
        launch.assert_not_called()


class ProvenanceTests(unittest.TestCase):
    def make_checkout(self, root: Path) -> None:
        for package in (root / "src/kvrepack", root / "vendor/lmms-eval/lmms_eval"):
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("# CPU import source fixture\n")
        source = root / "src/kvrepack/core.py"
        source.write_text("VALUE = 1\n")
        manifest = {"schema_version": 1, "entries": [{
            "source_path": "nonexistent-original-tree/core.py", "source_sha256": "unused-original-hash",
            "target_path": "src/kvrepack/core.py", "target_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }]}
        (root / "provenance").mkdir()
        (root / "provenance/core_sources.json").write_text(json.dumps(manifest))
        hashes = {}
        for relative in ("lmms_eval/__init__.py", "lmms_eval/__main__.py", "lmms_eval/evaluator.py",
                         "lmms_eval/models/__init__.py", "lmms_eval/models/qwen2_5_vl.py"):
            target = root / "vendor/lmms-eval" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# CPU vendor source fixture\n")
            hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        vendor_manifest = {"schema_version": 1, "components": {"lmms_eval": {
            "destination": "vendor/lmms-eval", "vendored_files_sha256": hashes,
        }}}
        (root / "provenance/vendor_sources.json").write_text(json.dumps(vendor_manifest))

    def test_verification_requires_only_this_checkout_and_detects_mutated_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            self.assertEqual(len(verify_manifest(root)["verified_targets"]), 1)
            (root / "src/kvrepack/core.py").write_text("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_manifest(root)

    def test_manifest_cannot_resolve_a_target_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            path = root / "provenance/core_sources.json"
            manifest = json.loads(path.read_text())
            manifest["entries"][0]["target_path"] = "../outside.py"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "outside this checkout"):
                verify_manifest(root)

    def test_vendor_adapter_and_frontend_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            self.assertEqual(len(verify_vendor_sources(root)["verified_targets"]), 5)
            (root / "vendor/lmms-eval/lmms_eval/models/qwen2_5_vl.py").write_text("# changed adapter\n")
            with self.assertRaisesRegex(ValueError, "frontend hash mismatch"):
                verify_vendor_sources(root)

    def test_actual_child_import_specs_resolve_only_local_fixture_without_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            plan = build_plan("fullkv", repo_root=root, environ={})
            info = verify_import_sources(plan, child_environment(plan))
            self.assertTrue(info["resolved_modules"]["lmms_eval"][0].startswith(str(root)))
            self.assertTrue(info["resolved_modules"]["kvrepack"][0].startswith(str(root)))
            self.assertFalse((root / "src/kvrepack/__pycache__").exists())

    def fake_kernel_distribution(self, root: Path, source: Path) -> Path:
        # find_spec recognizes the suffix without importing this intentionally
        # non-executable extension. Metadata verification is entirely CPU-only.
        origin = root / "src" / ("kvrepack_kernel" + EXTENSION_SUFFIXES[0])
        origin.write_bytes(b"source-ownership-test-fixture")
        metadata = root / "src/kvrepack_kernels-0.1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: kvrepack-kernels\nVersion: 0.1\n")
        (metadata / "direct_url.json").write_text(json.dumps({"url": source.as_uri(), "dir_info": {}}))
        (metadata / "RECORD").write_text(f"{origin.name},,\n")
        return origin

    def test_installed_kernel_requires_owning_distribution_and_this_checkout_build_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            origin = self.fake_kernel_distribution(root, root / "csrc")
            plan = build_plan("hybridkv", repo_root=root, environ={})
            info = verify_import_sources(plan, child_environment(plan))
            self.assertEqual(info["kernel_distribution"]["origin"], str(origin))
            self.assertEqual(info["kernel_distribution"]["extension_sha256"], hashlib.sha256(origin.read_bytes()).hexdigest())

    def test_build_source_egg_info_does_not_mask_the_installed_owning_distribution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            self.fake_kernel_distribution(root, root / "csrc")
            source_metadata = root / "csrc/kvrepack_kernels.egg-info"
            source_metadata.mkdir(parents=True)
            (source_metadata / "PKG-INFO").write_text("Name: kvrepack-kernels\nVersion: 0.1\n")
            (source_metadata / "SOURCES.txt").write_text("setup.py\n")
            plan = build_plan("hybridkv", repo_root=root, environ={})
            info = verify_import_sources(plan, child_environment(plan))
            self.assertEqual(info["kernel_distribution"]["name"], "kvrepack-kernels")

    def test_installed_kernel_from_another_checkout_is_rejected_without_importing_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            self.fake_kernel_distribution(root, root / "old-project/csrc")
            plan = build_plan("hybridkv", repo_root=root, environ={})
            with self.assertRaisesRegex(ValueError, "another checkout"):
                verify_import_sources(plan, child_environment(plan))

    def test_execution_saves_resolved_inputs_and_hashes_but_no_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_checkout(root)
            plan = build_plan("fullkv", repo_root=root, output=root / "output", environ={})
            with patch("kvrepack.experiments.runner.verify_import_sources", return_value={"dependency_versions": {}}), \
                 patch("kvrepack.experiments.runner._git_commit", return_value=None), \
                 patch("kvrepack.experiments.runner.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as launch, \
                 patch.dict(os.environ, {"HF_TOKEN": "test-secret-token", "METHOD": "evil"}):
                self.assertEqual(run_plan(plan), 0)
            self.assertEqual(launch.call_args.kwargs["env"]["METHOD"], "fullkv")
            self.assertEqual(launch.call_args.kwargs["env"]["HF_TOKEN"], "test-secret-token")
            self.assertEqual(json.loads((plan.output / "resolved_config.json").read_text())["internal_method"], "fullkv")
            provenance = (plan.output / "provenance.json").read_text()
            self.assertIn("verified_targets", provenance)
            self.assertNotIn("test-secret-token", provenance)
            self.assertEqual(json.loads((plan.output / "run_state.json").read_text())["returncode"], 0)
            with patch("kvrepack.experiments.runner.verify_import_sources", return_value={}):
                with self.assertRaisesRegex(ValueError, "not empty"):
                    run_plan(plan)


if __name__ == "__main__":
    unittest.main()

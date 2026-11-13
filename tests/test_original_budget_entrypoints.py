"""Original-budget defaults and protection of completed strict experiments."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("restored_nextqa_controller",
    ROOT/"scripts/run_nextqa_merge_comparison.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class OriginalEntrypointTests(unittest.TestCase):
    def arguments(self, output):
        return ["controller", "--pretrained", "/model", "--hf-home", "/cache",
                "--output", str(output), "--limit", "1", "--warmup", "1"]

    def test_default_dispatches_only_original_budget_workers_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"new"
            child = SimpleNamespace(pid=999999, wait=lambda: 0, poll=lambda: 0)
            with patch.object(controller.sys, "argv", self.arguments(output)), \
                 patch.object(controller.subprocess, "Popen", return_value=child) as launch, \
                 patch.object(controller.signal, "signal"):
                self.assertEqual(controller.main(), 0)
            modules = [call.args[0][3] for call in launch.call_args_list]
            self.assertEqual(modules, ["kvrepack.ablations.run_original_merge_comparison"]*2+
                ["kvrepack.ablations.run_original_baselines"]*2+
                ["kvrepack.ablations.original_nextqa_report"])
            state = json.loads((output/"experiment_state.json").read_text())
            self.assertEqual(state["budget_policy"], "original_history_plus_local_v1")

    def test_original_default_refuses_to_resume_strict_results_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            state = dict(task="nextqa_oe_test", docs_per_condition=1, status="complete")
            original = json.dumps(state).encode()
            path = output/"experiment_state.json"
            path.write_bytes(original)
            with patch.object(controller.sys, "argv", self.arguments(output)+["--resume"]), \
                 patch.object(controller.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "Budget policy changed"):
                    controller.main()
            launch.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_timed_generation_path_is_unchanged_except_post_timer_budget_check(self):
        def one(filename):
            tree = ast.parse((ROOT/"src/kvrepack/ablations"/filename).read_text())
            return deepcopy(next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "one"))

        class NormalizeBudgetCheck(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "active_prefill_original":
                    node.func.id = "active_prefill"
                    node.args = node.args[:3]
                    node.keywords = []
                return self.generic_visit(node)

        for old, new in (("run_baseline_completion.py", "run_original_baselines.py"),
                         ("run_style_comparison.py", "run_original_merge_comparison.py")):
            self.assertEqual(ast.dump(one(old), include_attributes=False),
                ast.dump(NormalizeBudgetCheck().visit(one(new)), include_attributes=False))


if __name__ == "__main__":
    unittest.main()

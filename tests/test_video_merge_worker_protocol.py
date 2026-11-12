"""Keep the task migration from changing the established merge/timing path."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

from kvrepack.ablations import run_merge_comparison as frozen
from kvrepack.ablations import run_style_comparison as generic


class VideoMergeProtocolTests(unittest.TestCase):
    def test_style_task_is_required_and_not_overridden_to_vatex(self):
        argv = ["--method", "snapkv_kvrepack", "--pretrained", "/model",
                "--hf-home", "/cache", "--output", "/tmp/pilot", "--limit", "1"]
        for task in ("nextqa_oe_test", "vatex_test"):
            args = generic.parse_args(argv+["--task", task])
            self.assertEqual(args.task, task)
            self.assertEqual(args.variants, ["lookm", "asymkv"])

    def test_generation_and_fisher_timing_path_matches_frozen_worker(self):
        def one(module):
            tree = ast.parse(Path(module.__file__).read_text())
            return deepcopy(next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "one"))

        class NormalizeTaskCap(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "max_new_tokens":
                    return ast.copy_location(ast.Constant(64), node)
                return node

        self.assertEqual(ast.dump(one(frozen), include_attributes=False),
            ast.dump(NormalizeTaskCap().visit(one(generic)), include_attributes=False))

    def test_old_algorithm_dependencies_and_group_checks_are_shared(self):
        previous, current = frozen.sources(), generic.sources()
        for path, digest in previous.items():
            self.assertEqual(current[path], digest)
        self.assertIs(generic.active_prefill, frozen.active_prefill)
        self.assertIs(generic.pairing_signature, frozen.pairing_signature)
        self.assertIn("src/kvrepack/ablations/video_e2e_inputs.py", current)


if __name__ == "__main__":
    unittest.main()

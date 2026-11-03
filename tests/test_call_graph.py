"""Check that the call auditor diagnoses argument errors without descriptor false positives."""

import ast
from pathlib import Path
import runpy
import tempfile
import unittest


CHECKER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_call_graph.py"))


class CallGraphTests(unittest.TestCase):
    def test_required_and_unknown_keywords_are_detected(self):
        function = ast.parse("def f(first, second=2, *, required): pass").body[0]
        errors, status = CHECKER["binding_errors"](function, ast.parse("f(1, extra=3)").body[0].value)
        self.assertEqual(status, "checked")
        self.assertIn("unexpected_keywords:extra", errors)
        self.assertIn("missing_required_arguments:required", errors)
        self.assertEqual(CHECKER["binding_errors"](function, ast.parse("f(1, required=3)").body[0].value), ([], "checked"))

    def test_dynamic_kwargs_are_reported_as_unresolved(self):
        function = ast.parse("def f(required): pass").body[0]
        self.assertEqual(CHECKER["binding_errors"](function, ast.parse("f(**options)").body[0].value),
                         ([], "unresolved_star_kwargs"))

    def test_static_class_and_instance_descriptors_and_real_argument_error(self):
        source = """
class Example:
    def __init__(self, value=0): pass
    @staticmethod
    def static(first, second): pass
    @classmethod
    def factory(cls, value): return cls(value)
    def instance(self, value): pass
    def run(self):
        self.static(1, 2)
        Example.factory(1)
        self.instance(1)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src/kvrepack"
            package.mkdir(parents=True)
            target = package / "__init__.py"
            target.write_text(source)
            adapters = root / "vendor/lmms-eval/lmms_eval/models"
            adapters.mkdir(parents=True)
            for name in ("qwen2_5_vl", "llava_hf"):
                (adapters / (name + ".py")).write_text("")
            bindings = root / "csrc/csrc/python_extension.cu"
            bindings.parent.mkdir(parents=True)
            bindings.write_text("")
            self.assertTrue(CHECKER["audit"](root)["passed"])
            target.write_text(source.replace("self.instance(1)", "self.instance(wrong_name=1)"))
            report = CHECKER["audit"](root)
            self.assertFalse(report["passed"])
            self.assertEqual({e["error"] for e in report["errors"]},
                             {"unexpected_keywords:wrong_name", "missing_required_arguments:value"})


if __name__ == "__main__":
    unittest.main()

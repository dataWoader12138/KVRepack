"""Dispatch controls only: no algorithm/tensor doubles are ever called."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_dispatch.py"
spec = importlib.util.spec_from_file_location("kvrepack_dispatch_audit", SCRIPT)
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def write_hf_structure_fixture(root):
    """A small HF namespace fixture; installed-HF evidence comes from the audit CLI."""
    sources = {
        "__init__.py": '__version__ = "4.52.3"\n',
        "generation/utils.py": "class GenerationMixin:\n    def generate(self): pass\n",
        "models/auto/modeling_auto.py": 'MODEL_MAPPING_NAMES = OrderedDict([("qwen2", "Qwen2Model")])\n',
        "models/qwen2_5_vl/modeling_qwen2_5_vl.py": """
class Qwen2_5_VLAttention:
    def forward(self): pass
class Qwen2_5_VLFlashAttention2(Qwen2_5_VLAttention): pass
class Qwen2_5_VLSdpaAttention(Qwen2_5_VLAttention): pass
QWEN2_5_VL_ATTENTION_CLASSES = {"flash_attention_2": Qwen2_5_VLFlashAttention2}
class Qwen2_5_VLDecoderLayer:
    def forward(self): pass
class Qwen2_5_VLTextModel:
    def forward(self): pass
class Qwen2_5_VLModel:
    def __init__(self): self.language_model = Qwen2_5_VLTextModel._from_config(None)
    def forward(self): pass
class Qwen2_5_VLForConditionalGeneration(GenerationMixin):
    def __init__(self): self.model = Qwen2_5_VLModel(None)
    def prepare_inputs_for_generation(self): pass
""",
        "models/qwen2/modeling_qwen2.py": """
class Qwen2Attention:
    def forward(self): pass
class Qwen2DecoderLayer:
    def __init__(self): self.self_attn = Qwen2Attention(None)
    def forward(self): pass
class Qwen2Model:
    def __init__(self): self.layers = nn.ModuleList([])
    def forward(self): pass
""",
        "models/llava_onevision/modeling_llava_onevision.py": """
class LlavaOnevisionModel:
    def __init__(self): self.language_model = AutoModel.from_config(None)
    def forward(self): pass
class LlavaOnevisionForConditionalGeneration(GenerationMixin):
    def __init__(self): self.model = LlavaOnevisionModel(None)
""",
    }
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


class DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.hf = Path(cls.temp.name) / "transformers"
        write_hf_structure_fixture(cls.hf)
        inference_names = {"torch", "transformers", "accelerate", "flash_attn", "decord"}
        before_audit = set(sys.modules) & inference_names
        cls.report = dispatch.audit(cls.hf)
        cls.inference_modules_added = (set(sys.modules) & inference_names) - before_audit
        cls.rows = {(r["model"], r["public_method"]): r for r in cls.report["combinations"]}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_registered_combinations_execute_real_dispatch_in_distinct_workers(self):
        self.assertTrue(self.report["passed"], self.report)
        registry = dispatch._registry()
        expected = {(model, method) for model, methods in registry.METHODS_BY_MODEL.items() for method in methods}
        self.assertEqual(set(self.rows), expected)
        self.assertEqual(len(expected), 19)
        self.assertTrue(self.report["separate_process_per_combination"])
        for row in self.rows.values():
            self.assertTrue(row["passed"], row)
            self.assertEqual(row["initial_patch_count"], 0)
            self.assertEqual(row["algorithm_functions_executed"], 0)
            self.assertNotEqual(row["worker_pid"], os.getpid())

    def test_fullkv_is_unpatched_and_hybrid_class_and_cache_gates_follow_actual_sources(self):
        for family in ("qwen2_5_vl", "llava_ov"):
            self.assertEqual(self.rows[family, "fullkv"]["patch_targets"], [])
            self.assertTrue(self.rows[family, "fullkv"]["adapter_class"].startswith("transformers."))
            hybrid = self.rows[family, "hybridkv"]
            self.assertTrue(hybrid["adapter_class"].startswith("kvrepack."))
            self.assertEqual(hybrid["patch_targets"], [])
            self.assertEqual(hybrid["hybrid_cache_check"]["cache_class"], "kvrepack.hybridkv_utils.HybridKVCache")
        self.assertFalse(self.rows["qwen2_5_vl", "hybridkv"]["hybrid_cache_check"]["kvrepack_enabled"])
        upgrade = self.rows["qwen2_5_vl", "hybridkv_kvrepack"]["hybrid_cache_check"]
        self.assertTrue(upgrade["kvrepack_enabled"])
        self.assertFalse(upgrade["asymkv_enabled"])

    def test_actual_prepare_wrapper_forwards_all_values_and_only_resets_empty_cache(self):
        checks = [r["prepare_inputs_check"] for r in self.rows.values() if r["prepare_inputs_check"]["applicable"]]
        self.assertEqual(len(checks), 9)
        for check in checks:
            cases = {case["cache_case"]: case for case in check["cases"]}
            self.assertEqual(set(cases), {"none", "legacy_tuple", "empty", "nonempty"})
            self.assertEqual(cases["empty"]["kv_seq_len_after"], [0, 0])
            for case in cases.values():
                self.assertTrue(case["all_arguments_forwarded_by_identity"])
                self.assertTrue(case["empty_cache_reset_correct"])

    def test_llava_targets_the_actual_qwen2_attention_class_and_multimodal_forward(self):
        for method in ("snapkv", "sparsemm", "madakv", "lookm"):
            targets = {p["attribute"] for p in self.rows["llava_ov", method]["patch_targets"]}
            self.assertIn("transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward", targets)
            self.assertFalse(any("Qwen2FlashAttention2" in target for target in targets))
            if method in {"madakv", "lookm"}:
                self.assertIn("transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionModel.forward", targets)

    def test_patch_records_name_real_source_definitions_and_real_assignment_lines(self):
        for row in self.rows.values():
            for patch in row["patch_targets"]:
                source = (ROOT / patch["source_path"]).read_text().splitlines()
                assignment = (ROOT / patch["patch_source_path"]).read_text().splitlines()
                self.assertTrue(source[patch["line"] - 1].lstrip().startswith("def "), patch)
                self.assertIn("=", assignment[patch["patch_line"] - 1], patch)
        targets = self.rows["qwen2_5_vl", "asymkv"]["patch_targets"]
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(p["patch_source_path"].endswith("asymkv_official/integration.py") for p in targets))

    def test_missing_transformers_target_fails_instead_of_creating_a_permissive_double(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "transformers"
            write_hf_structure_fixture(root)
            path = root / "models/qwen2/modeling_qwen2.py"
            path.write_text(path.read_text().replace("class Qwen2Attention:", "class MissingAttention:"))
            command = [sys.executable, "-I", "-B", str(SCRIPT), "--worker", "llava_ov", "snapkv",
                       "--transformers-root", str(root)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0)
            row = json.loads(result.stdout)
            self.assertFalse(row["passed"])
            self.assertIn("Qwen2Attention", row["errors"][0])

    def test_report_explicitly_limits_its_claims(self):
        self.assertIn("NOT executed", self.report["evidence_boundary"])
        self.assertEqual(self.report["transformers_source"]["version"], "4.52.3")
        self.assertTrue(all(c["passed"] for c in self.report["transformers_source"]["structure_checks"]))
        self.assertFalse(self.inference_modules_added)


class ImportGuardTests(unittest.TestCase):
    def test_forbidden_import_guard_intercepts_import_before_dependency_loading(self):
        program = """
import importlib, importlib.util, sys
spec = importlib.util.spec_from_file_location('dispatch_check', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.meta_path.insert(0, module._NoInferenceImports())
try:
    importlib.import_module('torch')
except RuntimeError as exc:
    assert 'forbids inference dependency import' in str(exc)
else:
    raise AssertionError('Guard allowed inference import')
assert 'torch' not in sys.modules
print('guard intercepted before dependency loading')
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(SCRIPT)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("guard intercepted", result.stdout)


if __name__ == "__main__":
    unittest.main()

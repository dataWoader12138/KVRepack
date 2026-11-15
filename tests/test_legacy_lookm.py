"""CPU verification of the exact archived LOOK-M replay and restoration."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from kvrepack.benchmarks.legacy_lookm import LegacyLookMContext, source_identity

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"src/kvrepack/baseline_utils.py"


def manifest_entry():
    return next(x for x in json.loads((ROOT/"provenance/core_sources.json").read_text())["entries"]
                if x["target_path"] == "src/kvrepack/baseline_utils.py")


def source_module(legacy=False):
    text = SOURCE.read_text()
    if legacy:
        entry = manifest_entry()
        for change in reversed(entry["behavior_patch"]["reversible_edits"]):
            text = text[:change["target_start"]]+change["before"]+text[change["target_end"]:]
        if hashlib.sha256(text.encode()).hexdigest() != "a7219df197e13cd91e6aee0369bc1e0d8264a2f0fc65c3f863576d83df82722e":
            raise AssertionError("Independent oracle no longer matches the archived run source")
    tree = ast.parse(text)
    wanted = {"repeat_kv", "merge_kv", "LookMCluster"}
    nodes = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in wanted]
    scope = dict(torch=torch, nn=torch.nn, F=F, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(**scope)


def clone_cluster(module, n, layer):
    return module.LookMCluster(layer_idx=layer, num_hidden_layers=28,
        num_key_value_groups=2, budget_ratio=.1, gqa_func="mean",
        is_text_token_mask=torch.arange(n).remainder(17).eq(0).view(1, -1))


class LegacyLookMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_source_identity_is_stdlib_only_and_matches_archived_run(self):
        identity = source_identity()
        self.assertEqual(identity["archived_module_sha256"],
            "a7219df197e13cd91e6aee0369bc1e0d8264a2f0fc65c3f863576d83df82722e")
        self.assertEqual(set(identity["functions"]), {"merge_kv", "LookMCluster.update_kv"})
        script = "from kvrepack.benchmarks.legacy_lookm import source_identity; import sys; source_identity(); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_real_old_update_matches_independent_oracle_in_prune_and_merge_layers(self):
        for layer in (0, 2, 27):
            with self.subTest(layer=layer):
                current, oracle = source_module(), source_module(legacy=True)
                n = 400
                generator = torch.Generator().manual_seed(41+layer)
                key = torch.randn((1, 2, n, 128), generator=generator)
                value = torch.randn(key.shape, generator=generator)
                query = torch.randn((1, 4, n, 128), generator=generator)
                actual_cluster, expected_cluster = clone_cluster(current, n, layer), clone_cluster(oracle, n, layer)
                expected = expected_cluster.update_kv(key.clone(), query.clone(), value.clone())
                original = current.LookMCluster.update_kv
                with LegacyLookMContext(module=current) as context:
                    actual = actual_cluster.update_kv(key.clone(), query.clone(), value.clone())
                    self.assertEqual(actual[0].shape, (1, 2, 72, 128))
                    self.assertTrue(torch.equal(actual[0], expected[0]))
                    self.assertTrue(torch.equal(actual[1], expected[1]))
                    self.assertEqual(actual_cluster.select_num, expected_cluster.select_num)
                    self.assertIs(actual_cluster.update_kv.__func__.__globals__, current.__dict__)
                    self.assertEqual(context.transform_spec["archived_module_sha256"],
                        manifest_entry()["behavior_patch"]["previous_target_sha256"])
                self.assertIs(current.LookMCluster.update_kv, original)

    def test_first_real_prompt_budget_is_history960_plus_recent32(self):
        current = source_module()
        n = 9607
        generator = torch.Generator().manual_seed(43)
        key = torch.randn((1, 1, n, 128), generator=generator)
        value = torch.randn(key.shape, generator=generator)
        query = torch.randn((1, 2, n, 128), generator=generator)
        cluster = clone_cluster(current, n, 0)
        with LegacyLookMContext(module=current):
            out_key, out_value = cluster.update_kv(key, query, value)
            self.assertEqual(cluster.select_num, 960)
            self.assertEqual(out_key.shape, (1, 1, 992, 128))
            self.assertTrue(torch.equal(out_key[..., -32:, :], key[..., -32:, :]))
            self.assertTrue(torch.equal(out_value[..., -32:, :], value[..., -32:, :]))

    def test_original_cross_head_complement_behavior_is_replayed_not_corrected(self):
        current, oracle = source_module(), source_module(legacy=True)
        key = torch.ones((1, 2, 12, 128))
        value = torch.arange(24.).view(1, 2, 12, 1).expand(1, 2, 12, 128)
        selected = torch.tensor([[[0, 2], [1, 3]]]).unsqueeze(-1).expand(1, 2, 2, 128)
        expected = oracle.merge_kv(key, value, selected, 2, "pivot")
        corrected = current.merge_kv(key, value, selected, 2, "pivot")
        self.assertFalse(torch.equal(expected[1], corrected[1]))
        with LegacyLookMContext(module=current):
            actual = current.merge_kv(key, value, selected, 2, "pivot")
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_exception_and_double_entry_restore_exact_function_objects(self):
        module = source_module()
        originals = module.merge_kv, module.LookMCluster.update_kv
        context = LegacyLookMContext(module=module)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with context:
                with self.assertRaisesRegex(RuntimeError, "twice"):
                    context.__enter__()
                raise RuntimeError("deliberate")
        self.assertIs(module.merge_kv, originals[0])
        self.assertIs(module.LookMCluster.update_kv, originals[1])

    def test_nested_context_is_rejected_without_disrupting_outer_patch(self):
        module = source_module()
        original = module.merge_kv
        with LegacyLookMContext(module=module):
            active = module.merge_kv
            with self.assertRaises(ValueError):
                LegacyLookMContext(module=module).__enter__()
            self.assertIs(module.merge_kv, active)
        self.assertIs(module.merge_kv, original)

    def test_modified_source_and_invalid_reverse_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = manifest_entry()
            for relative in ("provenance/core_sources.json", "src/kvrepack/baseline_utils.py",
                             entry["behavior_patch"]["patch_path"]):
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/relative, target)
            target = root/"src/kvrepack/baseline_utils.py"
            target.write_text(target.read_text()+"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "Current LOOK-M source"):
                source_identity(root)
            shutil.copyfile(SOURCE, target)
            path = root/"provenance/core_sources.json"
            manifest = json.loads(path.read_text())
            record = next(x for x in manifest["entries"] if x["target_path"] == entry["target_path"])
            record["behavior_patch"]["reversible_edits"][0]["after"] += "wrong"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "reversible edit"):
                source_identity(root)


if __name__ == "__main__":
    unittest.main()

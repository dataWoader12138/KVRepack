"""CPU cache and CLI guards for the head-selection experiment runner.

These tests exercise real tensor serialization; they do not load a checkpoint,
decode media, execute compression, or establish GPU inference correctness.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kvrepack.ablations import run_head_selection as runner

HAS_TORCH = importlib.util.find_spec("torch") is not None
REJECTED = (ValueError, RuntimeError, TypeError, FileNotFoundError, KeyError)


class HeadSelectionRunCLITests(unittest.TestCase):
    def arguments(self):
        return ["--task", "nextqa_oe_test", "--strategy", "similarity",
                "--pretrained", "/local/pinned-7b", "--hf-home", "/local/cache",
                "--cache", "/local/prepared", "--output", "/local/results"]

    def test_default_limit_and_explicit_random_strategies(self):
        args = runner.parse_args(self.arguments())
        self.assertEqual(args.limit, 20)
        self.assertEqual(args.seed, 0)
        self.assertFalse(args.prepare_only)
        for strategy in ("random_global", "random_layer_matched"):
            values = self.arguments()
            values[values.index("similarity")] = strategy
            args = runner.parse_args(values + ["--seed", "2", "--limit", "3", "--prepare-only"])
            self.assertEqual((args.strategy, args.seed, args.limit, args.prepare_only),
                             (strategy, 2, 3, True))

    def test_unknown_task_strategy_and_nonpositive_limit_are_rejected(self):
        invalid = [self.arguments() + ["--limit", "0"],
                   self.arguments() + ["--limit", "-1"],
                   self.arguments() + ["--unknown-option"],
                   self.arguments() + ["--task", "not_a_task"],
                   self.arguments() + ["--strategy", "all_heads"]]
        for values in invalid:
            with self.subTest(arguments=values), patch("sys.stderr"), self.assertRaises(SystemExit):
                runner.parse_args(values)

    def test_import_has_no_model_or_tensor_dependency(self):
        code = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class BlockModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'datasets', 'qwen_vl_utils'}:
            raise RuntimeError('Heavy dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockModelImports())
from kvrepack.ablations import run_head_selection
assert callable(run_head_selection.main)
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "src")],
                                env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_contract_is_reusable_but_drift_never_overwrites_it(self):
        expected = {"schema_version": 1, "task": "nextqa_oe_test", "frames": 64,
                    "selection": {"strategy": "similarity", "seed": 0}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            runner.ensure_contract(path, expected)
            self.assertEqual(json.loads(path.read_text()), expected)
            before = path.read_bytes()
            runner.ensure_contract(path, expected)
            self.assertEqual(path.read_bytes(), before)
            changed = dict(expected, frames=32)
            with self.assertRaises(REJECTED):
                runner.ensure_contract(path, changed)
            self.assertEqual(path.read_bytes(), before)


@unittest.skipUnless(HAS_TORCH, "CPU torch is unavailable")
class HeadSelectionInputCacheTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        self.contract = hashlib.sha256(b"fixed-input-profile").hexdigest()
        self.metadata = {"task": "nextqa_oe_test", "doc_id": 7,
                         "source_video_id": "video_7", "prompt": "Why did the person run?",
                         "frame_sampling": {"selected_indices": [0, 3, 6]}}
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 17, 4]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 4), dtype=torch.int64),
            "pixel_values_videos": torch.tensor([[1., -2., .25], [0., 3., -1.]], dtype=torch.bfloat16),
            "video_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
        }
        # Any unexpected allocation on CUDA must fail before an experiment starts.
        self.cuda_guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden in CPU tests"))
        self.cuda_guard.start()
        self.addCleanup(self.cuda_guard.stop)

    def paths(self):
        return self.cache / "000007.pt", self.cache / "000007.json"

    def save(self):
        return runner.save_cached_input(self.cache, 7, self.batch, self.metadata, self.contract)

    def read_json(self):
        return json.loads(self.paths()[1].read_text())

    def write_json(self, value):
        self.paths()[1].write_text(json.dumps(value) + "\n")

    def test_real_cpu_tensor_roundtrip_has_independent_hashes_shapes_and_metadata(self):
        info = self.save()
        batch, metadata, loaded_info = runner.load_cached_input(self.cache, 7, self.contract)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(info, loaded_info)
        self.assertEqual(info["doc_id"], 7)
        self.assertEqual(info["contract_sha256"], self.contract)
        self.assertEqual(info["metadata"], self.metadata)
        self.assertEqual(info["pt_sha256"], hashlib.sha256(self.paths()[0].read_bytes()).hexdigest())
        self.assertEqual(set(batch), set(self.batch))
        for name, expected in self.batch.items():
            with self.subTest(tensor=name):
                self.assertEqual(batch[name].device.type, "cpu")
                self.assertTrue(self.torch.equal(batch[name], expected))
                self.assertEqual(batch[name].dtype, expected.dtype)
                self.assertEqual(info["input_shapes"][name], list(expected.shape))
                raw = expected.contiguous().view(self.torch.uint8).numpy().tobytes()
                self.assertEqual(info["input_hashes"][name], hashlib.sha256(raw).hexdigest())

    def test_all_selection_strategies_reuse_identical_tensor_bytes(self):
        self.save()
        file_bytes = self.paths()[0].read_bytes()
        fingerprints = runner.tensor_fingerprints(self.batch)
        for strategy in ("similarity", "random_global", "random_layer_matched"):
            with self.subTest(strategy=strategy):
                batch, metadata, info = runner.load_cached_input(self.cache, 7, self.contract)
                self.assertEqual(runner.tensor_fingerprints(batch), fingerprints)
                self.assertEqual(metadata, self.metadata)
                self.assertEqual(info["input_hashes"], self.read_json()["input_hashes"])
                self.assertEqual(self.paths()[0].read_bytes(), file_bytes)

    def test_loader_explicitly_uses_safe_cpu_tensor_deserialization(self):
        self.save()
        with patch.object(self.torch, "load", wraps=self.torch.load) as load:
            runner.load_cached_input(self.cache, 7, self.contract)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.kwargs.get("map_location"), "cpu")
        self.assertIs(load.call_args.kwargs.get("weights_only"), True)

    def test_serialized_bytes_tamper_is_rejected_before_deserialization(self):
        self.save()
        path = self.paths()[0]
        payload = bytearray(path.read_bytes())
        payload[len(payload) // 2] ^= 1
        path.write_bytes(payload)
        with patch.object(self.torch, "load", side_effect=AssertionError("Do not deserialize corrupt bytes")):
            with self.assertRaises(REJECTED):
                runner.load_cached_input(self.cache, 7, self.contract)

    def test_tensor_tamper_is_rejected_even_when_container_hash_is_updated(self):
        self.save()
        path, _ = self.paths()
        stored = self.torch.load(path, map_location="cpu", weights_only=True)
        # The cache payload is the tensor dict, without executable Python objects.
        self.assertIsInstance(stored, dict)
        if "input_ids" in stored:
            stored["input_ids"][0, 0] += 1
        elif "batch" in stored:
            stored["batch"]["input_ids"][0, 0] += 1
        else:
            self.fail("Cache tensor payload is not accessible under the declared schema")
        self.torch.save(stored, path)
        info = self.read_json()
        info["pt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_json(info)
        with self.assertRaises(REJECTED):
            runner.load_cached_input(self.cache, 7, self.contract)

    def test_cache_contract_document_id_shape_and_dtype_drift_are_rejected(self):
        self.save()
        for field, replacement in (("contract_sha256", "0" * 64), ("doc_id", 8),
                                   ("input_shapes", {"input_ids": [1, 99]}),
                                   ("input_dtypes", {"input_ids": "torch.float64"})):
            with self.subTest(field=field):
                pristine = self.paths()[1].read_bytes()
                info = self.read_json()
                info[field] = replacement
                self.write_json(info)
                with self.assertRaises(REJECTED):
                    runner.load_cached_input(self.cache, 7, self.contract)
                self.paths()[1].write_bytes(pristine)
        with self.assertRaises(REJECTED):
            runner.load_cached_input(self.cache, 7, "f" * 64)

    def test_missing_companion_file_does_not_silently_regenerate_inputs(self):
        self.save()
        self.paths()[1].unlink()
        with self.assertRaises(REJECTED):
            runner.load_cached_input(self.cache, 7, self.contract)

    def test_non_tensor_cache_input_is_rejected(self):
        bad = dict(self.batch, raw_path="/not/a/tensor")
        with self.assertRaises(REJECTED):
            runner.save_cached_input(self.cache, 7, bad, self.metadata, self.contract)


if __name__ == "__main__":
    unittest.main()

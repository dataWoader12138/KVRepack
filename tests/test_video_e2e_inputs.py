"""CPU checks for native video inputs and the task-specific paired baselines."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kvrepack.ablations import run_baseline_completion as baseline
from kvrepack.ablations import run_merge_comparison as historical
from kvrepack.ablations import video_e2e_inputs as native
from kvrepack.ablations.run_head_selection import json_sha
from kvrepack.experiments.runner import build_plan
from kvrepack.observations import inputs

ROOT = Path(__file__).resolve().parents[1]


class NativeProcessor:
    def __init__(self):
        self.messages = None
        self.calls = []
        # Direct tokenization is observation overhead and must not occur here.
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True)
        self.messages = messages
        return "native chat:" + messages[1]["content"][1]["text"]

    def __call__(self, **kwargs):
        if "videos" not in kwargs:
            raise AssertionError("Native input preparation must not re-tokenize")
        if "return_offsets_mapping" in kwargs:
            raise AssertionError("Observation offsets must not be timed")
        self.calls.append(kwargs)
        return dict(input_ids=np.asarray([[11, 12, 13]]),
                    attention_mask=np.ones((1, 3), dtype=int),
                    video_grid_thw=np.asarray([[32, 2, 2]]))


def fixture(root, task):
    if task == "nextqa_oe_test":
        docs = [{"video": 410, "question": "Skip this row?"},
                {"video": 420, "question": "  Why does the person move?  "}]
        family, function = "nextqa", "nextqa_doc_to_text"
        kwargs = dict(pre_prompt="", post_prompt="\nAnswer a question using a short phrase or sentence.")
    else:
        docs = [{"videoID": "skip_0_9"}, {"videoID": "fixture_video_10_20"}]
        family, function = "vatex", "vatex_test_doc_to_text"
        kwargs = dict(prompt="Describe the video.")
    for doc in docs:
        (root / (str(doc.get("video", doc.get("videoID"))) + ".mp4")).write_bytes(b"video fixture")
    prompt = inputs._task_prompt_function(ROOT / "vendor/lmms-eval/lmms_eval/tasks" / family / "utils.py", function)
    return dict(task=task, dataset=docs, video_root=str(root), arrow_sha256="original-arrow",
                _prompt_function=prompt, task_kwargs=kwargs)


def vision_module(frame_count=67, images=None):
    module = ModuleType("qwen_vl_utils")
    # A fresh video is returned for each invocation; helpers replace videos[0].
    module.process_vision_info = lambda messages: (
        images, [np.arange(frame_count * 12).reshape(frame_count, 3, 2, 2)])
    return module


class NativeVideoInputTests(unittest.TestCase):
    def test_nextqa_preserves_original_row_prompt_and_native_processor_call(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), "nextqa_oe_test")
            processor = NativeProcessor()
            with patch.dict(sys.modules, qwen_vl_utils=vision_module()):
                batch, metadata = native.prepare_input(data, 1, processor)
            self.assertEqual(len(processor.calls), 1)
            self.assertEqual(processor.calls[0]["return_tensors"], "pt")
            content = processor.messages[1]["content"]
            self.assertEqual(content[1]["text"], "Why does the person move?\nAnswer a question using a short phrase or sentence.")
            self.assertEqual(content[0]["min_pixels"], 200704)
            self.assertEqual(content[0]["max_pixels"], 1605632)
            self.assertEqual(metadata["task"], "nextqa_oe_test")
            self.assertEqual(metadata["doc_id"], 1)
            self.assertEqual(metadata["video_id"], "420")
            self.assertEqual(metadata["source_video_id"], "420")
            self.assertEqual(metadata["original_row_sha256"], json_sha(data["dataset"][1]))
            self.assertEqual(metadata["arrow_sha256"], "original-arrow")
            self.assertEqual(metadata["input_tokens"], batch["input_ids"].shape[-1])
            self.assertEqual(metadata["frame_indices"], np.linspace(0, 66, 64, dtype=int).tolist())
            self.assertEqual(processor.calls[0]["videos"][0].shape[0], 64)

    def test_vatex_matches_frozen_native_pipeline_and_metadata_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), "vatex_test")
            processors = [NativeProcessor(), NativeProcessor()]
            with patch.dict(sys.modules, qwen_vl_utils=vision_module()):
                actual, actual_metadata = native.prepare_input(data, 1, processors[0])
                expected, expected_metadata = historical.prepare_input(data, 1, processors[1])
            self.assertEqual(actual_metadata, expected_metadata)
            self.assertEqual(actual_metadata["source_video_id"], "fixture_video")
            self.assertEqual(processors[0].messages, processors[1].messages)
            self.assertIn("[video4] output:", processors[0].messages[1]["content"][1]["text"])
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(processors[0].calls[0]["videos"][0], processors[1].calls[0]["videos"][0])

    def test_short_video_keeps_original_repeated_frame_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), "nextqa_oe_test")
            processor = NativeProcessor()
            with patch.dict(sys.modules, qwen_vl_utils=vision_module(frame_count=3)):
                _, metadata = native.prepare_input(data, 0, processor)
            self.assertEqual(metadata["frame_indices"], np.linspace(0, 2, 64, dtype=int).tolist())
            self.assertEqual(metadata["decoded_frames"], 3)
            self.assertEqual(processor.calls[0]["videos"][0].shape[0], 64)

    def test_invalid_task_or_original_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), "nextqa_oe_test")
            with patch.dict(sys.modules, qwen_vl_utils=vision_module()):
                for doc_id in (-1, 2, True, 1.0):
                    with self.subTest(doc_id=doc_id), self.assertRaises(ValueError):
                        native.prepare_input(data, doc_id, NativeProcessor())
                with self.assertRaises(ValueError):
                    native.prepare_input(dict(data, task="wrong-task"), 0, NativeProcessor())

    def test_unexpected_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), "nextqa_oe_test")
            with patch.dict(sys.modules, qwen_vl_utils=vision_module(images=[object()])):
                with self.assertRaises(ValueError):
                    native.prepare_input(data, 0, NativeProcessor())


class VideoBaselineTaskTests(unittest.TestCase):
    def test_task_specific_plans_keep_native_pruning_and_use_selected_merge_parameters(self):
        with patch.dict(os.environ):
            for task in ("nextqa_oe_test", "vatex_test"):
                for backbone in ("snapkv", "hybridkv"):
                    with self.subTest(task=task, backbone=backbone):
                        args = SimpleNamespace(task=task, backbone=backbone,
                            pretrained=Path("/tmp/Qwen2.5-VL-7B-Instruct"), hf_home=Path("/tmp/kvrepack-test-hf"),
                            output=Path("/tmp/kvrepack-test-plans"), limit=200)
                        plans = baseline.plans_for(args)
                        original = build_plan(backbone, task=task, pretrained=str(args.pretrained),
                            model_type="qwen2_5_vl-7b", frames=64, limit=200, budget_ratio=.1,
                            output=args.output, overrides=())
                        self.assertEqual(plans["native"].resolved_config["algorithm_env"], original.resolved_config["algorithm_env"])
                        for plan in plans.values():
                            self.assertEqual(plan.resolved_config["task"], task)
                        env = plans["kvrepack"].resolved_config["algorithm_env"]
                        if backbone == "snapkv":
                            self.assertEqual(float(env["SNAPKV_SIMHEAD_MERGE_RATIO"]), .30 if task == "nextqa_oe_test" else .15)
                            self.assertEqual(float(env["SNAPKV_KVREPACK_COMPRESS_RATIO"]), .70)
                            self.assertEqual(float(env["K_DIFF_ALPHA"]), 2 if task == "nextqa_oe_test" else .5)
                        else:
                            self.assertEqual(float(env["HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO"]), .50)
                            self.assertEqual(float(env["HYBRID_KVREPACK_COMPRESS_RATIO"]), .50)
                            self.assertEqual(float(env["K_DIFF_ALPHA"]), 2)

    def test_real_contract_selects_nextqa_128_and_vatex_64_before_any_model_load(self):
        import torch
        from transformers import AutoProcessor

        class BeforeModelLoad(RuntimeError):
            pass

        for task, cap in (("nextqa_oe_test", 128), ("vatex_test", 64)):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = root / "model"; model.mkdir()
                (model / "config.json").write_text("{}")
                args = SimpleNamespace(task=task, backbone="snapkv", pretrained=model,
                    hf_home=root/"hf", output=root/"output", limit=200, warmup=2, resume=False)
                plan = SimpleNamespace(env={}, resolved_config={"algorithm_env": {}})
                data = dict(arrow_sha256="original-arrow", dataset=[{}]*200,
                    task_yaml_sha256="original-yaml", task_utils_sha256="original-utils")
                with patch.object(baseline, "plans_for", return_value={"native":plan,"kvrepack":plan}), \
                     patch.object(baseline, "apply_plan"), patch.object(baseline, "sources", return_value={}), \
                     patch.object(inputs, "load_task", return_value=data) as load, \
                     patch.object(torch, "manual_seed"), \
                     patch.object(AutoProcessor, "from_pretrained", side_effect=BeforeModelLoad), \
                     patch.object(baseline.importlib.metadata, "version", return_value="cpu-fixture"):
                    with self.assertRaises(BeforeModelLoad):
                        baseline.run(args)
                load.assert_called_once_with(args.hf_home, task)
                contract = json.loads((args.output / "run_contract.json").read_text())
                self.assertEqual(contract["task"], task)
                self.assertEqual(contract["max_new_tokens"], cap)
                self.assertEqual(contract["target_generations"], 400)

    def test_cli_defaults_to_vatex_and_accepts_explicit_nextqa(self):
        common = ["--backbone", "snapkv", "--pretrained", "/tmp/model", "--hf-home", "/tmp/hf",
                  "--output", "/tmp/kvrepack-test-output", "--limit", "200"]
        for task in (None, "nextqa_oe_test"):
            with patch.object(baseline, "output_lock", return_value=nullcontext()), \
                 patch.object(baseline, "run", return_value=0) as run:
                baseline.main(common + (["--task", task] if task else []))
                self.assertEqual(run.call_args.args[0].task, task or "vatex_test")


if __name__ == "__main__":
    unittest.main()

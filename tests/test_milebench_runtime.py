"""CPU integration of MileBench generation, artifacts, and offline scoring.

Only the model is replaced: annotation loading, image verification, dataset
truncation, prediction records, scoring, and artifact collision checks are real.
"""

from __future__ import annotations

import base64
from contextlib import nullcontext
import copy
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from kvrepack.benchmarks.milebench.generate import run
from kvrepack.benchmarks.milebench import workers


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(map(ord, text)))

    def decode(self, ids):
        return "".join(map(chr, ids))


class FakeWorker:
    tokenizer = CharacterTokenizer()

    def __init__(self, config, *, fail_id=None):
        self.config = config
        self.seen = []
        self.fail_id = fail_id

    def generate_one(self, sample):
        self.seen.append(copy.deepcopy(sample))
        if sample["sample_id"] == self.fail_id:
            raise RuntimeError("intentional model failure")
        # Answering from the fixture response tests plumbing, not model quality.
        return {"answer": sample["response"], "runtime": {
            "input_tokens": len(sample["context"]), "generated_tokens": 1,
            "cache": {"available": False, "reason": "CPU fake model"},
        }}


class Matrix:
    """A CPU list-backed ID matrix for the real worker's slicing boundary."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, indices):
        rows, columns = indices
        return Matrix([row[columns] for row in self.rows[rows]])

    def __eq__(self, value):
        count = sum(token == value for row in self.rows for token in row)
        return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: count))

    def tolist(self):
        return self.rows


class ProcessorInputs(dict):
    def to(self, device):
        return self


class MileBenchRuntimeTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.task = "CLEVR-Change"
        self.images = self.data / self.task / "images"
        self.images.mkdir(parents=True)
        with Image.new("RGBA", (5, 4), (230, 20, 10, 100)) as image:
            image.save(self.images / "first.png")
        with Image.new("RGB", (4, 5), (10, 20, 230)) as image:
            image.save(self.images / "second.png")
        self.annotation = {
            "meta_data": {"question_type": "multi-choice", "task_instruction": ["Choose the matching answer."], "num_sample": 4},
            "data": [
                {"sample_id": 41, "task_instruction_id": 0, "task_instance": {
                    "context": "{image#1} What color?", "images_path": ["first.png"], "choice_list": ["red", "blue"]}, "response": "red"},
                {"sample_id": 9, "task_instruction_id": 0, "task_instance": {
                    "context": "{image#1} Compare {image#2}. Which changed?", "images_path": ["first.png", "second.png"], "choice_list": ["red", "blue"]}, "response": "blue"},
                {"sample_id": 27, "task_instruction_id": 0, "task_instance": {
                    "context": "{image#1} What color?", "images_path": ["second.png"], "choice_list": ["red", "blue"]}, "response": "blue"},
                # This unselected image is intentionally missing. limit=3 must
                # select original IDs before checking paths or grouping samples.
                {"sample_id": 88, "task_instruction_id": 0, "task_instance": {
                    "context": "{image#1} What color?", "images_path": ["missing.png"], "choice_list": ["red", "blue"]}, "response": "red"},
            ],
        }
        self.annotation_file = self.data / self.task / f"{self.task}.json"
        self.annotation_file.write_text(json.dumps(self.annotation))
        self.config = {
            "benchmark": "milebench", "model": "qwen2_5_vl", "profile": "image8",
            "public_method": "fullkv", "internal_method": "fullkv",
            "pretrained": "fixture/Qwen2.5-VL-3B", "model_type": "qwen2_5_vl-3b",
            "task": self.task, "data_root": str(self.data), "limit": 3,
            "output": str(self.root / "output"), "batch_size": 1, "seed": 42,
            "generation": {"max_new_tokens": 512, "min_new_tokens": 1, "do_sample": False,
                           "temperature": 0, "num_beams": 1},
            "image": {"max_context_len": 32768, "n_tokens_per_image": 256,
                      "max_pixels": 1605632, "min_pixels": 7168, "message_min_pixels": 200704,
                      "system_prompt": "You are a helpful assistant."},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_fake_model_end_to_end_preserves_selection_prompt_order_and_scores(self):
        worker = FakeWorker(self.config)
        summary = run(self.config, worker_factory=lambda config: worker)
        output = Path(self.config["output"])
        selected = json.loads((output / "selected_samples.json").read_text())
        predictions = json.loads((output / "pred.json").read_text())
        runtime = [json.loads(line) for line in (output / "runtime.jsonl").read_text().splitlines()]
        self.assertEqual(selected["sample_ids"], [41, 9, 27])
        self.assertEqual(selected["generation_sample_ids"], [41, 27, 9])
        self.assertEqual([sample["sample_id"] for sample in worker.seen], [41, 27, 9])
        self.assertEqual([sample["sample_id"] for sample in predictions], [41, 27, 9])
        self.assertEqual(len(selected["annotation_sha256"]), 64)
        self.assertEqual(len(selected["sample_ids_sha256"]), 64)
        self.assertEqual(len(selected["selected_annotations_sha256"]), 64)
        self.assertEqual(selected["preflight"]["images_checked"], 2)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["metrics"]["Accuracy"], 1.0)
        self.assertEqual(json.loads((output / "scoring" / "evaluation_metadata.json").read_text())["sample_ids"], [41, 9, 27])
        self.assertTrue((output / "scoring" / "pred_with_extracted.json").is_file())
        self.assertEqual([row["sample_id"] for row in runtime], [41, 27, 9])
        self.assertIn("Choose the matching answer.", predictions[0]["question"])
        self.assertIn("Choice list:", predictions[0]["question"])
        self.assertEqual(predictions[0]["gen_model_id"], "qwen2.5-vl")
        self.assertEqual(predictions[0]["gen_kwargs"], self.config["generation"])
        self.assertFalse(runtime[0]["observations"]["cache"]["available"])

    def test_all_selected_images_are_checked_before_model_creation(self):
        self.config["limit"] = 4
        with patch("kvrepack.benchmarks.milebench.workers.create_worker") as factory:
            with self.assertRaisesRegex(ValueError, "dataset preflight failed.*missing"):
                run(self.config)
        factory.assert_not_called()
        self.assertFalse(Path(self.config["output"]).exists())

    def test_model_failure_leaves_only_diagnostics_and_propagates(self):
        worker = FakeWorker(self.config, fail_id=27)
        with self.assertRaisesRegex(RuntimeError, "intentional model failure"):
            run(self.config, worker_factory=lambda config: worker)
        output = Path(self.config["output"])
        self.assertTrue((output / "selected_samples.json").is_file())
        self.assertEqual(len((output / "runtime.jsonl").read_text().splitlines()), 1)
        self.assertFalse((output / "pred.json").exists())
        self.assertFalse((output / "scoring").exists())
        self.assertFalse((output / "generation_summary.json").exists())

    def test_completed_output_is_never_overwritten(self):
        run(self.config, worker_factory=FakeWorker)
        path = Path(self.config["output"]) / "pred.json"
        original = path.read_bytes()
        with patch("kvrepack.benchmarks.milebench.workers.create_worker") as factory:
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                run(self.config)
        factory.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_invalid_or_out_of_scope_configs_fail_before_model_creation(self):
        cases = []
        for method in ("asymkv", "randomkv", "lookm_kvrepack"):
            config = copy.deepcopy(self.config)
            config.update(public_method=method, internal_method=method)
            cases.append(config)
        config = copy.deepcopy(self.config)
        config["generation"]["do_sample"] = True
        cases.append(config)
        config = copy.deepcopy(self.config)
        config["output"] = str(ROOT / "src" / "unsafe-runtime-output")
        cases.append(config)
        with patch("kvrepack.benchmarks.milebench.workers.create_worker") as factory:
            for config in cases:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    run(config)
        factory.assert_not_called()

    def test_qwen_source_jpeg_policy_and_distinct_pixel_floors(self):
        from PIL import Image
        messages = workers.qwen_messages("Look <ImageHere> now.", [str(self.images / "first.png")], self.config["image"])
        self.assertEqual(messages[0], {"role": "system", "content": "You are a helpful assistant."})
        content = messages[1]["content"]
        self.assertEqual([entry["type"] for entry in content], ["image", "text"])
        self.assertEqual(content[1]["text"], "Look  now.")
        self.assertEqual(content[0]["min_pixels"], 200704)
        self.assertEqual(self.config["image"]["min_pixels"], 7168)
        encoded = content[0]["image"].split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual((image.format, image.mode, image.size), ("JPEG", "RGB", (5, 4)))

    def test_llava_source_user_turn_and_zero_temperature_policy(self):
        message = workers.llava_messages("<ImageHere> Compare <ImageHere>.", 2)
        self.assertEqual([row["role"] for row in message], ["user"])
        self.assertEqual([part["type"] for part in message[0]["content"]], ["image", "image", "text"])
        qwen = workers.generation_kwargs(self.config, "qwen2_5_vl")
        llava = workers.generation_kwargs(self.config, "llava_ov")
        self.assertIsNone(qwen["temperature"])
        self.assertNotIn("temperature", llava)
        self.assertFalse(qwen["do_sample"])
        self.assertEqual(qwen["max_new_tokens"], 512)

    def test_import_and_offline_runtime_do_not_import_inference_dependencies(self):
        program = """
import importlib.abc, json, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'accelerate', 'deepspeed', 'flash_attn', 'decord'}:
            raise RuntimeError('forbidden inference import: ' + fullname)
sys.meta_path.insert(0, Guard())
from kvrepack.benchmarks.milebench.generate import run
from kvrepack.benchmarks.milebench.workers import cache_statistics
from types import SimpleNamespace
class Tokenizer:
    def __call__(self, text, **kwargs): return SimpleNamespace(input_ids=list(map(ord,text)))
    def decode(self, ids): return ''.join(map(chr,ids))
class Worker:
    tokenizer=Tokenizer()
    def __init__(self, config): pass
    def generate_one(self, sample): return {'answer':sample['response'],'runtime':{'cache':cache_statistics(None)}}
config=json.loads(sys.argv[1])
assert run(config,worker_factory=Worker)['status']=='completed'
"""
        config = copy.deepcopy(self.config)
        config["output"] = str(self.root / "guarded-output")
        process = subprocess.run([sys.executable, "-B", "-c", program, json.dumps(config)],
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""},
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_real_worker_trims_prompt_and_records_observed_ids_and_grid_with_model_double(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "formatted-prompt"

            def __call__(self, **kwargs):
                self.preprocessor_args = kwargs
                return ProcessorInputs(input_ids=Matrix([[11, 12, 13, 14]]), image_grid_thw=Matrix([[1, 4, 4]]))

            def batch_decode(self, ids, **kwargs):
                self.decoded_ids = ids.tolist()
                self.decode_kwargs = kwargs
                return ["  decoded answer  "]

        processor = Processor()
        torch_double = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=Mock()), bfloat16="bf16",
            inference_mode=nullcontext, autocast=lambda **kwargs: nullcontext(),
        )
        generate = Mock(return_value=SimpleNamespace(
            sequences=Matrix([[11, 12, 13, 14, 21, 22]]),
            past_key_values=SimpleNamespace(get_seq_length=lambda: 5),
        ))
        worker = workers.MileBenchWorker.__new__(workers.MileBenchWorker)
        worker.config, worker.family = self.config, "qwen2_5_vl"
        worker.torch, worker.device = torch_double, SimpleNamespace(type="cuda")
        worker.processor, worker.tokenizer = processor, CharacterTokenizer()
        worker.model = SimpleNamespace(generate=generate, config=SimpleNamespace(image_token_id=11))
        worker.gen_kwargs = workers.generation_kwargs(self.config, worker.family)
        real_import = workers.import_module

        def imported(name):
            if name == "qwen_vl_utils":
                return SimpleNamespace(process_vision_info=lambda messages: (["processed-image"], None))
            return real_import(name)

        sample = {"context": "Question <ImageHere>.", "raw_img_list": [str(self.images / "first.png")]}
        with patch.object(workers, "import_module", side_effect=imported), patch.dict(os.environ):
            result = worker.generate_one(sample)
        self.assertEqual(result["answer"], "decoded answer")
        self.assertEqual(processor.decoded_ids, [[21, 22]])
        self.assertEqual(result["runtime"]["input_tokens"], 4)
        self.assertEqual(result["runtime"]["generated_tokens"], 2)
        self.assertEqual(result["runtime"]["visual_input_tokens"], {"image": 1})
        self.assertEqual(result["runtime"]["visual_grids"], {"image_grid_thw": [[1, 4, 4]]})
        self.assertEqual(len(result["runtime"]["input_ids_sha256"]), 64)
        self.assertEqual(result["runtime"]["cache"]["reported_seq_length"], 5)
        self.assertEqual(torch_double.cuda.synchronize.call_count, 2)
        self.assertTrue(generate.call_args.kwargs["use_cache"])
        self.assertTrue(generate.call_args.kwargs["return_dict_in_generate"])
        self.assertIsNone(generate.call_args.kwargs["temperature"])
        self.assertEqual(processor.decode_kwargs, {"skip_special_tokens": True, "clean_up_tokenization_spaces": False})

    def test_real_worker_rejects_stale_algorithm_environment_before_heavy_imports(self):
        config = copy.deepcopy(self.config)
        config["algorithm_env"] = {"BUDGET_RATIO": "0.1"}
        with patch.dict(os.environ, {"METHOD": "fullkv", "MODEL_TYPE": config["model_type"], "BUDGET_RATIO": "0.9"}), \
             patch.object(workers, "import_module") as heavy:
            with self.assertRaisesRegex(ValueError, "Algorithm environment BUDGET_RATIO disagrees"):
                workers.MileBenchWorker(config)
        heavy.assert_not_called()


class CacheObservationsTests(unittest.TestCase):
    def test_tensor_shapes_are_observed_without_claiming_compression_ratio(self):
        class Tensor:
            shape = (11, 128)
            dtype = "fixture.bfloat16"

            def numel(self):
                return 11 * 128

            def element_size(self):
                return 2

        cache = SimpleNamespace(key_cache=[Tensor()], value_cache=[Tensor()], offload_k_cache=[Tensor()])
        result = workers.cache_statistics(cache)
        self.assertEqual(result["tensor_groups"]["key_cache"][0]["shape"], [11, 128])
        self.assertEqual(result["tensor_groups"]["key_cache"][0]["tensor_bytes"], 2816)
        self.assertIn("offload_k_cache", result["tensor_groups"])
        self.assertNotIn("retention_ratio", result)
        self.assertNotIn("reported_seq_length", result)
        self.assertFalse(workers.cache_statistics(None)["available"])


if __name__ == "__main__":
    unittest.main()

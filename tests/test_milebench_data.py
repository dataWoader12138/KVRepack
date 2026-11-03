"""CPU-only protocol, coverage, image preflight and metric regression checks."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from kvrepack.benchmarks.milebench.dataset import MileBenchDataset, load_annotation, validate_data
from kvrepack.benchmarks.milebench.evaluate import evaluate_predictions, main as evaluate_main, write_evaluation
from kvrepack.benchmarks.milebench.score import summarize, summarize_runs, write_summary


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_ROUGE = importlib.util.find_spec("rouge") is not None
HAS_PILLOW = importlib.util.find_spec("PIL") is not None
SOURCE_ROOT = os.environ.get("KVREPACK_MILEBENCH_SOURCE")


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def decode(self, values):
        return "".join(chr(value) for value in values)


def annotation(*, multiple_choice=False, ids=(9, 3)):
    return {"meta_data": {"task_instruction": ["Inspect the images."],
                          "question_type": "multi-choice" if multiple_choice else "open-ended",
                          "num_sample": len(ids)},
            "data": [{"sample_id": sample_id, "task_instruction_id": 0, "response": "red",
                      "image_quantity_level": "Few",
                      "task_instance": {"context": "{image#1} Before {table#2} What changed?",
                                        "images_path": ["first.png", "second.png"],
                                        **({"choice_list": ["red", "blue", "green"]} if multiple_choice else {})}}
                     for sample_id in ids]}


def write_fixture(root, task="Fixture", *, document=None):
    directory = Path(root) / task
    directory.mkdir()
    images = directory / "images"
    images.mkdir()
    (images / "first.png").write_bytes(b"first image placeholder")
    (images / "second.png").write_bytes(b"second image placeholder")
    (directory / f"{task}.json").write_text(json.dumps(document or annotation()), encoding="utf-8")
    return directory


def make_dataset(document, image_root="/external/images", max_context_len=1000, dataset_name="Fixture"):
    return MileBenchDataset(annotation=document["data"],
                            task_instructions=document["meta_data"]["task_instruction"],
                            img_dir=str(image_root), max_context_len=max_context_len,
                            n_tokens_per_image=10, tokenizer=CharacterTokenizer(), dataset_name=dataset_name)


def source_class(filename, classname, globals_):
    source = Path(SOURCE_ROOT) / "hybridkv" / filename
    if not source.is_file():
        raise unittest.SkipTest(f"original source file unavailable: {source}")
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    klass = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == classname)
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[klass], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[classname]


class DataProtocolTests(unittest.TestCase):
    def test_imports_do_not_load_inference_or_metric_dependencies(self):
        script = """
import sys
from kvrepack.benchmarks.milebench import dataset, evaluate, score
for name in ('torch', 'transformers', 'accelerate', 'PIL', 'numpy', 'rouge'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_limit_keeps_original_id_order_before_any_grouping(self):
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root, document=annotation(ids=(9, 3, 8)))
            selected = load_annotation(root, "Fixture", limit=2)
            self.assertEqual([sample["sample_id"] for sample in selected["data"]], [9, 3])
            self.assertEqual(selected["meta_data"]["num_sample"], 3)
            for limit in (0, -1, True, 0.5):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    load_annotation(root, "Fixture", limit)

    def test_duplicate_annotation_ids_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root, document=annotation(ids=(1, "1")))
            with self.assertRaisesRegex(ValueError, "duplicate annotation"):
                load_annotation(root, "Fixture")
            with self.assertRaises(ValueError):
                load_annotation(root, "../Fixture")

    def test_all_referenced_images_are_checked_not_just_the_first(self):
        with tempfile.TemporaryDirectory() as root:
            directory = write_fixture(root)
            good = validate_data(root, "Fixture", verify_images=False)
            self.assertTrue(good["valid"], good)
            self.assertEqual(good["sample_ids"], [9, 3])
            self.assertEqual(good["image_references"], 4)
            self.assertEqual(good["images_checked"], 2)
            self.assertEqual(len(good["annotation_sha256"]), 64)
            (directory / "images" / "second.png").unlink()
            missing = validate_data(root, "Fixture", verify_images=False)
            self.assertFalse(missing["valid"])
            self.assertIn("second.png", " ".join(missing["errors"]))

    def test_annotations_cannot_reference_images_outside_their_image_directory(self):
        document = annotation(ids=(1,))
        document["data"][0]["task_instance"]["images_path"][1] = "../../outside.png"
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root, document=document)
            report = validate_data(root, "Fixture", verify_images=False)
            self.assertFalse(report["valid"])
            self.assertIn("escapes", " ".join(report["errors"]))

    @unittest.skipUnless(HAS_PILLOW, "image decoder validation requires Pillow")
    def test_corrupt_image_is_rejected_and_valid_images_decode(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            directory = write_fixture(root)
            self.assertFalse(validate_data(root, "Fixture")["valid"])
            for filename in ("first.png", "second.png"):
                Image.new("RGB", (2, 3), color="red").save(directory / "images" / filename)
            self.assertTrue(validate_data(root, "Fixture")["valid"])

    def test_original_prompt_options_image_order_and_collate_fields(self):
        data = make_dataset(annotation(multiple_choice=True))
        item = data[0]
        self.assertEqual(item["context"],
                         "Inspect the images.\n<ImageHere> Before <ImageHere> What changed?"
                         "\nChoice list: \nA. red\nB. blue\nC. green\nYour answer is: ")
        self.assertEqual(item["raw_img_list"], ["/external/images/first.png", "/external/images/second.png"])
        self.assertEqual(data.collate_fn([item]), {"id": [9], "question": [item["context"]],
                                                  "image_path": [item["raw_img_list"]], "gt_response": ["red"]})
        gpr = make_dataset(annotation(multiple_choice=True), dataset_name="GPR1200")[0]
        self.assertIn("\nred\nblue\ngreen\n", gpr["context"])
        self.assertNotIn("A. red", gpr["context"])

    @unittest.skipUnless(SOURCE_ROOT, "original prompt differential requires KVREPACK_MILEBENCH_SOURCE")
    def test_prompt_and_truncation_match_original_class(self):
        original = source_class("utils.py", "MileBenchDataset", {"os": os, "Dataset": object})
        for multiple_choice in (False, True):
            document = annotation(multiple_choice=multiple_choice)
            for length in (35, 60, 1000):
                with self.subTest(multiple_choice=multiple_choice, max_context_len=length):
                    kwargs = dict(annotation=document["data"], task_instructions=document["meta_data"]["task_instruction"],
                                  img_dir="/external/images", max_context_len=length, n_tokens_per_image=10,
                                  tokenizer=CharacterTokenizer(), dataset_name="Fixture")
                    new, old = MileBenchDataset(**kwargs), original(**kwargs)
                    self.assertEqual(new[0], old[0])
                    self.assertEqual(new.collate_fn([new[0]]), old.collate_fn([old[0]]))


class CoverageTests(unittest.TestCase):
    def test_missing_extra_and_duplicate_prediction_ids_fail_before_metric_import(self):
        document = annotation()
        cases = ([{"sample_id": 9, "pred_response": "red"}],
                 [{"sample_id": 9, "pred_response": "red"}, {"sample_id": "9", "pred_response": "red"}],
                 [{"sample_id": 9, "pred_response": "red"}, {"sample_id": 8, "pred_response": "red"}])
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaisesRegex(ValueError, "sample_id"):
                evaluate_predictions(predictions, document, "Fixture")

    def test_annotation_ground_truth_is_authoritative(self):
        with self.assertRaisesRegex(ValueError, "ground truth"):
            evaluate_predictions([{"sample_id": 1, "gt_response": "incorrect", "pred_response": "red"}],
                                 annotation(ids=(1,)), "Fixture")

    @unittest.skipUnless(HAS_NUMPY, "source Accuracy metrics require numpy")
    def test_multichoice_reorders_ids_without_mutating_inputs(self):
        document = annotation(multiple_choice=True)
        predictions = [{"sample_id": 3, "pred_response": "B"}, {"sample_id": 9, "pred_response": "A"}]
        saved = deepcopy((document, predictions))
        result = evaluate_predictions(predictions, document, "Fixture")
        self.assertEqual(result["sample_ids"], [9, 3])
        self.assertEqual(result["eval"]["Accuracy"], 0.5)
        self.assertEqual([item["sample_id"] for item in result["pred_with_extracted"]], [9, 3])
        self.assertEqual((document, predictions), saved)
        self.assertTrue(result["full_task"])

    @unittest.skipUnless(HAS_NUMPY, "source Accuracy metrics require numpy")
    def test_mmcoqa_keeps_substring_metric_and_partial_sample_metadata(self):
        document = annotation(ids=(1, 2))
        document["data"] = document["data"][:1]
        result = evaluate_predictions([{"sample_id": 1, "pred_response": "It is RED."}], document, "MMCoQA")
        self.assertEqual(result["eval"]["Accuracy"], 1.0)
        self.assertFalse(result["full_task"])
        self.assertEqual(result["annotation_total_samples"], 2)

    @unittest.skipUnless(HAS_NUMPY and HAS_ROUGE, "source Rouge-L metric requires numpy and rouge")
    def test_open_ended_metric_preserves_empty_prediction_zero(self):
        predictions = [{"sample_id": 9, "pred_response": "RED!"}, {"sample_id": 3, "pred_response": ""}]
        result = evaluate_predictions(predictions, annotation(), "Fixture")
        self.assertAlmostEqual(result["eval"]["Rouge-L f"], 0.5)

    @unittest.skipUnless(SOURCE_ROOT and HAS_NUMPY and HAS_ROUGE,
                         "original metric differential requires KVREPACK_MILEBENCH_SOURCE, numpy and rouge")
    def test_all_three_scoring_routes_match_original_fixed_predictions(self):
        import numpy as np
        from rouge import Rouge
        original = source_class("evaluate.py", "Eval", {"re": re, "np": np, "Rouge": Rouge})
        for multiple_choice, task, answers in ((False, "Fixture", ["The red car", ""]),
                                                (True, "Fixture", ["A", ""]),
                                                (False, "MMCoQA", ["It is RED.", "blue"])):
            with self.subTest(task=task, multiple_choice=multiple_choice):
                document = annotation(multiple_choice=multiple_choice)
                predictions = [{"sample_id": sid, "gt_response": "red", "pred_response": answer}
                               for sid, answer in zip((9, 3), answers)]
                result = evaluate_predictions(predictions, document, task)
                scorer = original()
                if task == "MMCoQA":
                    metrics, scores = scorer.evaluate_needle(deepcopy(predictions), deepcopy(document), needle=False)
                elif multiple_choice:
                    extracted, metrics, scores = scorer.evaluate_multichoice(deepcopy(predictions), deepcopy(document))
                    self.assertEqual(result["pred_with_extracted"], extracted)
                else:
                    metrics, scores = scorer.evaluate_rouge(deepcopy(predictions), deepcopy(document))
                self.assertEqual(result["eval"], metrics)
                self.assertEqual(result["eval_score"], scores)


class ArtifactAndSummaryTests(unittest.TestCase):
    def test_scoring_and_summary_refuse_overwrite_and_source_output(self):
        result = {"task": "Fixture", "sample_ids": [1], "sample_count": 1,
                  "annotation_total_samples": 1, "full_task": True,
                  "eval": {"Accuracy": 1.0}, "eval_score": [{"id": "1", "score": "1"}]}
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "score"
            write_evaluation(result, out)
            with self.assertRaises(FileExistsError):
                write_evaluation(result, out)
            summary = summarize_runs({"Fixture": out}, ["Fixture", "Missing"])
            self.assertEqual(summary["missing_tasks"], ["Missing"])
            self.assertEqual(summary["macro_average_percent"], 100)
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["full_dataset_macro_average_percent"])
            write_summary(summary, Path(root) / "summary")
            self.assertIn("Missing,missing", (Path(root) / "summary" / "summary.csv").read_text())
        source = Path(__file__).resolve().parents[1] / "src" / "should-not-be-created"
        with self.assertRaisesRegex(ValueError, "source|outputs/"):
            write_evaluation(result, source)
        self.assertFalse(source.exists())

    def test_macro_average_never_double_converts_or_zero_fills(self):
        result = summarize({"A": {"Accuracy": 0.4}, "B": {"Rouge-L f": 0.6}}, ["A", "B", "C"])
        self.assertEqual(result["macro_average_percent"], 50)
        self.assertEqual([row["percent"] for row in result["tasks"]], [40, 60])
        self.assertEqual(result["unverified_sample_count_tasks"], ["A", "B"])
        self.assertFalse(result["full_dataset_complete"])
        for value in (40, -0.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize({"A": {"Accuracy": value}}, ["A"])

    def test_summary_rejects_wrong_groups_and_ambiguous_runs(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "outside"):
                summarize_runs({"Typo": root}, ["A"])
            run = Path(root)
            (run / "scoring").mkdir()
            for path in (run / "eval.json", run / "scoring" / "eval.json"):
                path.write_text(json.dumps({"Accuracy": 0.5}))
            with self.assertRaisesRegex(ValueError, "multiple"):
                summarize_runs({"A": root}, ["A"])

    @unittest.skipUnless(HAS_NUMPY, "offline multichoice CLI requires numpy")
    def test_offline_cli_scores_selected_ids_without_loading_model(self):
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root, document=annotation(multiple_choice=True))
            pred = Path(root) / "pred.json"
            pred.write_text(json.dumps([{"sample_id": 9, "pred_response": "A"}]))
            out = Path(root) / "scoring"
            self.assertEqual(evaluate_main(["--data-root", root, "--task", "Fixture", "--limit", "1",
                                            "--predictions", str(pred), "--output", str(out)]), 0)
            self.assertEqual(json.loads((out / "eval.json").read_text())["Accuracy"], 1.0)
            metadata = json.loads((out / "evaluation_metadata.json").read_text())
            self.assertEqual(metadata["sample_ids"], [9])
            self.assertFalse(metadata["full_task"])


if __name__ == "__main__":
    unittest.main()

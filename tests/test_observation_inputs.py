"""CPU-only input alignment checks; no model construction or video decoding."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kvrepack.observations import inputs


class _Video:
    def __init__(self, frames):
        self.shape = (frames, 3, 8, 8)

    def __getitem__(self, indices):
        return _Video(len(indices))


class _Tokenizer:
    is_fast = True
    all_special_ids = [999, 998]

    def __init__(self):
        self.calls = 0
        self.corrupt_reencoding = False

    def __call__(self, texts, **kwargs):
        self.calls += 1
        text = texts[0]
        ids, offsets, position = [], [], 0
        while position < len(text):
            if text.startswith("<v>", position):
                ids.append(999)
                offsets.append([position, position + 3])
                position += 3
            else:
                ids.append(ord(text[position]))
                offsets.append([position, position + 1])
                position += 1
        if self.corrupt_reencoding and self.calls == 2:
            ids[0] += 1
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)], "offset_mapping": [offsets]}


class _Processor:
    video_token = "<v>"
    video_token_id = 999
    image_token_id = 998
    video_processor = types.SimpleNamespace(merge_size=2)

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.expanded = ""

    def apply_chat_template(self, messages, **kwargs):
        return "SYSTEM\n<v>\nUSER:" + messages[1]["content"][1]["text"] + "\nASSISTANT:"

    def __call__(self, text, images, videos, **kwargs):
        assert kwargs["return_offsets_mapping"]
        # The fixture represents two final visual tokens, independently of the
        # production expansion helper. No real media/model operations occur.
        self.expanded = text[0].replace("<v>", "<v><v>")
        return {**self.tokenizer([self.expanded]), "video_grid_thw": [[2, 2, 2]]}


class ObservationInputTests(unittest.TestCase):
    def test_actual_task_prompt_ast_and_vatex_examples_remain_but_are_not_primary_text(self):
        source = ROOT / "vendor/lmms-eval/lmms_eval/tasks/vatex/utils.py"
        fn = inputs._task_prompt_function(source, "vatex_test_doc_to_text")
        prompt = fn({}, {"prompt": "Describe the video."})
        self.assertIn("[video1] output: A man picks up", prompt)
        self.assertIn("[video4] output:", prompt)
        spans = inputs._natural_spans("vatex_test", {}, prompt, {"prompt": "Describe the video."})
        self.assertEqual(spans, [{"name": "instruction", "text": "Describe the video.", "char_span": [0, 19]}])

    def test_nextqa_question_and_instruction_are_separate(self):
        source = ROOT / "vendor/lmms-eval/lmms_eval/tasks/nextqa/utils.py"
        fn = inputs._task_prompt_function(source, "nextqa_doc_to_text")
        doc = {"question": "  Why run?  "}
        kwargs = {"pre_prompt": "", "post_prompt": "\nAnswer briefly."}
        prompt = fn(doc, kwargs)
        self.assertEqual(prompt, "Why run?\nAnswer briefly.")
        spans = inputs._natural_spans("nextqa_oe_test", doc, prompt, kwargs)
        self.assertEqual([(row["name"], row["char_span"]) for row in spans],
                         [("question", [0, 8]), ("instruction", [8, 24])])

    def test_alignment_excludes_special_padding_cross_boundary_and_zero_length_tokens(self):
        # The last token spans both question and instruction and must enter neither.
        text = "shell Q?\nDo it. tail"
        prompt = "Q?\nDo it."
        ids = [99, 1, 2, 3, 4, 5, 6, 7]
        offsets = [[0, 5], [6, 7], [7, 8], [9, 11], [12, 15], [0, 0], [12, 15], [7, 10]]
        groups = inputs._align_segments(ids, offsets, [1, 1, 1, 1, 1, 1, 0, 1], text, prompt,
                                        [{"name": "question", "text": "Q?", "char_span": [0, 2]},
                                         {"name": "instruction", "text": "\nDo it.", "char_span": [2, 9]}], {99, 2})
        self.assertEqual(groups[0]["positions"], [1])
        self.assertEqual(groups[1]["positions"], [3, 4])

    def test_ambiguous_prompt_span_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            inputs._align_segments([1], [[0, 1]], [1], "same same", "same", [], set())

    def test_video_extension_resolution_rejects_missing_ambiguous_and_empty_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                inputs._video_path(root, "clip")
            (root / "clip.MP4").write_bytes(b"video")
            self.assertEqual(inputs._video_path(root, "clip"), root / "clip.MP4")
            (root / "clip.mp4").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                inputs._video_path(root, "clip")
            (root / "empty.mkv").touch()
            with self.assertRaisesRegex(ValueError, "Empty"):
                inputs._video_path(root, "empty")

    def test_expanded_video_count_uses_grid_and_spatial_merge(self):
        processor = types.SimpleNamespace(video_token="<video>", video_processor=types.SimpleNamespace(merge_size=2))
        self.assertEqual(inputs._expand_video_text("a<video>b", processor, [2, 4, 4]), "a" + "<video>" * 8 + "b")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            inputs._expand_video_text("a<video><video>b", processor, [2, 4, 4])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for the native adapter frame policy")
    def test_short_video_repeats_frames_and_keeps_adapter_endpoint_policy(self):
        self.assertEqual(inputs._frame_indices(3, 5), [0, 0, 1, 1, 2])
        self.assertEqual(inputs._frame_indices(4, 1), [0, 3])
        self.assertEqual(inputs._frame_indices(1, 64), [0] * 64)

    def test_load_task_preserves_duplicate_video_rows_and_reads_no_task_imports(self):
        rows = [{"video": "same", "question": "one"}, {"video": "same", "question": "two"}]
        seen = []
        fake_dataset = types.SimpleNamespace(from_file=lambda path: seen.append(path) or rows)
        fake_yaml = types.SimpleNamespace(safe_load=lambda text: {
            "include": "_default_template_yaml", "lmms_eval_specific_kwargs": {
                "default": {"pre_prompt": "", "post_prompt": "\nAnswer."}}})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrow = root / "test.arrow"
            arrow.write_bytes(b"local fixture")
            with patch.dict(sys.modules, {"datasets": types.SimpleNamespace(Dataset=fake_dataset), "yaml": fake_yaml}):
                result = inputs.load_task(root, "nextqa_oe_test", arrow_path=arrow, video_root=root, repo_root=ROOT)
        self.assertIs(result["dataset"], rows)
        self.assertEqual(len(result["dataset"]), 2)
        self.assertEqual(len(seen), 1)
        self.assertEqual(result["_prompt_function"](rows[1], result["task_kwargs"]), "two\nAnswer.")
        self.assertEqual(len(result["arrow_sha256"]), 64)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Native frame sampling uses NumPy")
    def test_prepare_sample_aligns_real_task_functions_and_preserves_vatex_source_id(self):
        # Preload before patch.dict snapshots sys.modules; restoring the stub
        # must not remove a newly imported real NumPy module from the process.
        __import__("numpy")
        fixtures = [
            ("nextqa_oe_test", {"video": "123", "question": " Why run? "},
             {"pre_prompt": "", "post_prompt": "\nAnswer briefly."}, "nextqa_doc_to_text"),
            ("vatex_test", {"videoID": "a_b-c_defgh_000001_000010", "enCap": ["unused reference"]},
             {"prompt": "Describe the video."}, "vatex_test_doc_to_text"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task, doc, kwargs, function in fixtures:
                video = str(doc.get("video", doc.get("videoID")))
                (root / (video + ".mp4")).write_bytes(b"fixture only; decoder is stubbed")
                family = "nextqa" if task.startswith("nextqa") else "vatex"
                fn = inputs._task_prompt_function(ROOT / f"vendor/lmms-eval/lmms_eval/tasks/{family}/utils.py", function)
                task_data = {"task": task, "dataset": [doc], "video_root": str(root), "_prompt_function": fn,
                             "task_kwargs": kwargs, "arrow_path": "fixture.arrow", "arrow_sha256": "a" * 64,
                             "task_yaml_sha256": "b" * 64, "task_utils_sha256": "c" * 64}
                processor = _Processor()
                with patch.dict(sys.modules, {"qwen_vl_utils": types.SimpleNamespace(
                        process_vision_info=lambda messages: (None, [_Video(3)]))}):
                    batch, metadata = inputs.prepare_sample(task_data, 0, processor)
                self.assertNotIn("offset_mapping", batch)
                self.assertEqual(metadata["frame_sampling"]["selected_frames"], 64)
                self.assertEqual(metadata["frame_sampling"]["unique_frames"], 3)
                self.assertEqual(metadata["frame_sampling"]["repeated_frames"], 61)
                self.assertEqual(metadata["video_grid_thw"], [2, 2, 2])
                self.assertEqual(len(metadata["original_row_sha256"]), 64)
                if family == "nextqa":
                    self.assertEqual([x["name"] for x in metadata["text_segments"]], ["question", "instruction"])
                    self.assertEqual(metadata["source_video_id"], "123")
                else:
                    self.assertEqual(metadata["source_video_id"], "a_b-c_defgh")
                    self.assertEqual([x["name"] for x in metadata["text_segments"]], ["instruction"])
                    self.assertIn("[video4] output:", metadata["prompt"])
                    example_start = processor.expanded.index("[video1]")
                    offsets = processor.tokenizer([processor.expanded])["offset_mapping"][0]
                    self.assertTrue(all(offsets[p][1] <= example_start
                                        for p in metadata["text_segments"][0]["positions"]))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Native frame sampling uses NumPy")
    def test_prepare_sample_fails_if_expansion_reencoding_does_not_match(self):
        __import__("numpy")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "123.mp4").write_bytes(b"fixture")
            task_data = {"task": "nextqa_oe_test", "dataset": [{"video": "123", "question": "Why?"}],
                         "video_root": str(root), "_prompt_function": lambda doc, kwargs: "Why?",
                         "task_kwargs": {"pre_prompt": "", "post_prompt": ""}}
            processor = _Processor()
            processor.tokenizer.corrupt_reencoding = True
            with patch.dict(sys.modules, {"qwen_vl_utils": types.SimpleNamespace(
                    process_vision_info=lambda messages: (None, [_Video(3)]))}):
                with self.assertRaisesRegex(ValueError, "re-tokenization differs"):
                    inputs.prepare_sample(task_data, 0, processor)


if __name__ == "__main__":
    unittest.main()

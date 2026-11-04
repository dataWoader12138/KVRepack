"""CPU-only fixed-row and original-byte checks; no model/download execution."""
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from kvrepack.observations import reference_inputs as reference


def zip_fixture(path, *, count=7, member="data/qasper.jsonl", missing_context=False):
    rows = [dict(_id=f"original-{i}", context=f"Full article {i}: 中 {{literal braces}}\n\nThe end. ",
                 input=f" Original question {i}? ", dataset="qasper", length=300 + i) for i in range(count)]
    if rows and missing_context:
        rows[0].pop("context")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        archive.writestr("../../unexpected.txt", "This member must never be extracted")
    return rows


def prompt_fixture(path):
    template = "Article: {context}\nQuestion: {input}\nAnswer:"
    path.write_text(json.dumps({"qasper": template}))
    return template


class ReferenceTextInputTests(unittest.TestCase):
    def test_count_only_accepts_five_or_200_before_any_resource_access(self):
        with patch.object(reference, "_dataset_file") as loader:
            for count in (0, 1, 10, 199, 201, True, 200.0, "200", None):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    reference.prepare_reference_samples("unused", "unused", offline=True, sample_count=count)
            loader.assert_not_called()

    def test_zip_preserves_first_five_full_contexts_and_original_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = zip_fixture(root / "data.zip")
            template = prompt_fixture(root / "prompts.json")
            samples = reference._text_samples(root / "data.zip", template, {"revision": "fixed"})
            self.assertEqual([s["doc_id"] for s in samples], [0, 1, 2, 3, 4])
            for sample, original in zip(samples, rows):
                self.assertEqual(sample["context"], original["context"])
                self.assertEqual(sample["prompt"], template.format(context=original["context"], input=original["input"]))
                self.assertEqual(sample["source_metadata"]["original_row_id"], original["_id"])
                self.assertIn("{literal braces}", sample["prompt"])
            self.assertFalse((root.parent / "unexpected.txt").exists())

    def test_zip_missing_short_wrong_schema_and_ambiguous_member_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for args in ({"count": 4}, {"member": "../../qasper.jsonl"}, {"missing_context": True}):
                with self.subTest(args=args):
                    zip_fixture(root / "data.zip", **args)
                    with self.assertRaises(ValueError):
                        reference._text_samples(root / "data.zip", "{context}{input}", {})
            zip_fixture(root / "data.zip")
            with zipfile.ZipFile(root / "data.zip", "a") as archive:
                archive.writestr("qasper.jsonl", "{}\n")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                reference._text_samples(root / "data.zip", "{context}{input}", {})

    def test_prompt_is_JSON_data_and_requires_only_known_format_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.json"
            expected = prompt_fixture(path)
            self.assertEqual(reference._qasper_template(path), expected)
            for value in ({}, {"qasper": "{context}{input}{extra}"}, {"qasper": "{context!r}{input}"}, {"qasper": 3}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    reference._qasper_template(path)

    def test_text_200_keeps_long_final_row_and_does_not_replace_bad_selected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = zip_fixture(root / "data.zip", count=200)
            rows[-1]["context"] = "Full original scientific article. " * 4000
            def save():
                with zipfile.ZipFile(root / "data.zip", "w") as archive:
                    archive.writestr("data/qasper.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
            save()
            samples = reference._text_samples(root / "data.zip", "{context}\n{input}", {}, sample_count=200)
            self.assertEqual([r["doc_id"] for r in samples], list(range(200)))
            self.assertEqual(samples[-1]["context"], rows[-1]["context"])
            self.assertGreater(len(samples[-1]["context"]), 32768)
            rows[105]["context"] = ""
            save()
            with self.assertRaisesRegex(ValueError, "row 105"):
                reference._text_samples(root / "data.zip", "{context}{input}", {}, sample_count=200)
            zip_fixture(root / "data.zip", count=199)
            with self.assertRaisesRegex(ValueError, "200 required"):
                reference._text_samples(root / "data.zip", "{context}{input}", {}, sample_count=200)

    def test_offline_missing_prompt_and_tampered_cached_prompt_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                reference._prompt_file(directory, offline=True)
            path = Path(directory) / "hub/kvrepack-reference-prompts" / reference.PROMPT_COMMIT / "dataset2prompt.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                reference._prompt_file(directory, offline=True)

    def test_existing_different_inputs_are_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.bin"
            reference._write_once(path, b"original")
            before = path.stat()
            reference._write_once(path, b"original")
            after = path.stat()
            self.assertEqual((after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                             (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns))
            with self.assertRaises(ValueError):
                reference._write_once(path, b"different")
            self.assertEqual(path.read_bytes(), b"original")


HAS_IMAGES = bool(importlib.util.find_spec("PIL") and importlib.util.find_spec("pyarrow"))


@unittest.skipUnless(HAS_IMAGES, "CPU image fixtures require optional PIL and pyarrow")
class ReferenceVisionInputTests(unittest.TestCase):
    def parquet_fixture(self, path, count=1000, missing_bytes=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        stream = io.BytesIO()
        Image.new("RGB", (7, 3), (50, 90, 130)).save(stream, format="PNG")
        payload = stream.getvalue()
        rows = [dict(question=f" Original image question {i}? ", dataset=f"source-{i}",
                     question_type=f"type-{i}", image={"bytes": None if missing_bytes else payload,
                                                        "path": f"../../do_not_use_{i}.png"}) for i in range(count)]
        pq.write_table(pa.Table.from_pylist(rows), path)
        return payload, rows

    def test_fixed_rows_preserve_original_image_bytes_size_question_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, originals = self.parquet_fixture(root / "images.parquet")
            samples = reference._vision_samples(root / "images.parquet", root / "out", {"revision": "fixed"})
            self.assertEqual([s["doc_id"] for s in samples], [0, 200, 400, 600, 800])
            for sample in samples:
                i, metadata = sample["doc_id"], sample["source_metadata"]
                self.assertEqual(Path(sample["image_path"]).read_bytes(), payload)
                self.assertEqual(sample["prompt"], originals[i]["question"])
                self.assertEqual((metadata["original_image_width"], metadata["original_image_height"]), (7, 3))
                self.assertEqual(metadata["original_image_sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(metadata["source"], f"source-{i}")
                self.assertEqual(metadata["question_type"], f"type-{i}")
                self.assertTrue(Path(sample["image_path"]).is_relative_to(root / "out/input_images"))

    def test_missing_fixed_row_and_missing_image_bytes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kwargs in ({"count": 800}, {"count": 1001}, {"missing_bytes": True}):
                with self.subTest(kwargs=kwargs):
                    self.parquet_fixture(root / "images.parquet", **kwargs)
                    with self.assertRaises(ValueError):
                        reference._vision_samples(root / "images.parquet", root / "out", {})

    def test_200_vision_rows_use_predeclared_step_five_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, originals = self.parquet_fixture(root / "images.parquet")
            samples = reference._vision_samples(root / "images.parquet", root / "out", {}, sample_count=200)
            self.assertEqual([s["doc_id"] for s in samples], list(range(0, 1000, 5)))
            for sample in samples:
                self.assertEqual(sample["prompt"], originals[sample["doc_id"]]["question"])
                self.assertEqual(Path(sample["image_path"]).read_bytes(), payload)

    def test_prepare_200_records_count_and_complete_fixed_docids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zip_fixture(root / "data.zip", count=200)
            self.parquet_fixture(root / "images.parquet")
            prompt_fixture(root / "prompt.json")
            with patch.object(reference, "_dataset_file", side_effect=[root / "data.zip", root / "images.parquet"]), \
                    patch.object(reference, "_prompt_file", return_value=root / "prompt.json"):
                samples = reference.prepare_reference_samples(root / "cache", root / "out", offline=True, sample_count=200)
            manifest = json.loads((root / "out/reference_samples.json").read_text())
            self.assertEqual(len(samples), 400)
            self.assertEqual(manifest["sample_count"], 400)
            self.assertEqual(manifest["sample_count_per_group"], 200)
            self.assertEqual(manifest["selection"]["text"], list(range(200)))
            self.assertEqual(manifest["selection"]["vision"], list(range(0, 1000, 5)))
            self.assertTrue(all(s["source_metadata"]["row_index"] == s["doc_id"] for s in samples))

    def test_prepare_returns_exactly_ten_and_persists_only_fixed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zip_fixture(root / "data.zip")
            self.parquet_fixture(root / "images.parquet")
            prompt_fixture(root / "prompt.json")
            with ExitStack() as stack:
                loader = stack.enter_context(patch.object(reference, "_dataset_file", side_effect=[root / "data.zip", root / "images.parquet"]))
                stack.enter_context(patch.object(reference, "_prompt_file", return_value=root / "prompt.json"))
                samples = reference.prepare_reference_samples(root / "cache", root / "out", offline=True)
            self.assertEqual(loader.call_args_list[0].args, (reference.LONG_BENCH, root / "cache", True))
            self.assertEqual(loader.call_args_list[1].args, (reference.OCR_BENCH, root / "cache", True))
            self.assertEqual([s["group"] for s in samples], ["text"] * 5 + ["vision"] * 5)
            manifest = json.loads((root / "out/reference_samples.json").read_text())
            self.assertEqual(manifest["sample_count"], 10)
            self.assertEqual(manifest["selection"]["vision"], [0, 200, 400, 600, 800])
            self.assertEqual(manifest["samples"], samples)


if __name__ == "__main__":
    unittest.main()

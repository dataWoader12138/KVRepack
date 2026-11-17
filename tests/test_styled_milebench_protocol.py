"""CPU checks for image protocol helpers used by the styled image runs."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from kvrepack.benchmarks.milebench.protocol import (
    DEFAULT_IMAGE_CONFIG, cache_statistics, generation_kwargs, qwen_messages,
)


class StyledMileBenchProtocolTests(unittest.TestCase):
    def test_import_does_not_initialize_model_or_image_dependencies(self):
        code = """
import sys
from kvrepack.benchmarks.milebench import protocol
for name in ('torch', 'transformers', 'qwen_vl_utils', 'PIL', 'numpy', 'rouge'):
    assert name not in sys.modules, name
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_rgb_jpeg_image_order_and_pixel_floors_match_source_policy(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "first.png", root / "second.png"]
            Image.new("RGBA", (6, 5), (240, 10, 20, 70)).save(paths[0])
            Image.new("RGB", (5, 6), (10, 20, 240)).save(paths[1])
            original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            image = deepcopy(DEFAULT_IMAGE_CONFIG)
            messages = qwen_messages("<ImageHere> Compare. <ImageHere>", list(map(str, paths)), image)
            self.assertEqual(messages[0], {"role": "system", "content": "You are a helpful assistant."})
            content = messages[1]["content"]
            self.assertEqual([item["type"] for item in content], ["image", "image", "text"])
            self.assertEqual(content[-1]["text"], "Compare.")
            self.assertEqual(image["min_pixels"], 7168)
            for index, item in enumerate(content[:-1]):
                self.assertEqual(item["min_pixels"], 200704)
                self.assertEqual(item["max_pixels"], 1605632)
                prefix, encoded = item["image"].split(",", 1)
                self.assertEqual(prefix, "data:image/jpeg;base64")
                with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
                    self.assertEqual(decoded.mode, "RGB")
                    self.assertEqual(decoded.format, "JPEG")
                    self.assertEqual(decoded.size, (6, 5) if index == 0 else (5, 6))
                    red, _, blue = decoded.getpixel((2, 2))
                    self.assertTrue(red > blue if index == 0 else blue > red)
            self.assertEqual(original, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})

    def test_declared_greedy_settings_remain_explicit(self):
        config = {"generation": {"max_new_tokens": 512, "min_new_tokens": 1,
                                 "repetition_penalty": 1.05}}
        before = deepcopy(config)
        kwargs = generation_kwargs(config, "qwen2_5_vl")
        self.assertEqual(kwargs["max_new_tokens"], 512)
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["num_beams"], 1)
        self.assertIsNone(kwargs["temperature"])
        self.assertIsNone(kwargs["top_p"])
        self.assertEqual(kwargs["repetition_penalty"], 1.05)
        self.assertEqual(config, before)
        for changed in ({"do_sample": True}, {"num_beams": 2}, {"temperature": 0.2}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                generation_kwargs({"generation": changed}, "qwen2_5_vl")

    def test_cache_statistics_do_not_treat_packed_dimensions_as_token_counts(self):
        class TensorStub:
            shape = (1, 2, 3, 4)
            dtype = "torch.bfloat16"

            def numel(self):
                return 24

            def element_size(self):
                return 2

        cache = SimpleNamespace(key_cache=[TensorStub()], value_cache=[TensorStub()])
        observed = cache_statistics(cache)
        self.assertEqual(observed["tensor_groups"]["key_cache"][0]["tensor_bytes"], 48)
        self.assertNotIn("token_count", observed)
        self.assertIn("logical sizes", observed["tensor_bytes_semantics"])
        self.assertFalse(cache_statistics(None)["available"])


if __name__ == "__main__":
    unittest.main()

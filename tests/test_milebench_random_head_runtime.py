"""Paired image streaming with a CPU model double and real audit/scoring gates."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
    from PIL import Image
except ImportError as error:
    raise unittest.SkipTest(f"CPU runtime fixtures need Torch and Pillow: {error}")

from kvrepack.ablations.run_milebench_head_selection import run
from kvrepack.ablations.milebench_head_selection_report import summarize_run
from kvrepack.experiments.random_head_images import build_random_head_plan

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("image_head_report_fixtures", ROOT / "tests/test_milebench_head_selection_report.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(map(ord, text)))

    def decode(self, ids):
        return "".join(map(chr, ids))


class FakeWorker:
    tokenizer = CharacterTokenizer()
    generation_metadata = {"kind": "CPU fake model; not a GPU experiment"}

    def __init__(self, config, *, fail=False, mutate=False, bad_merge=False):
        self.geometry = config["ablation"]["expected_geometry"]
        self.contract = json.loads((Path(config["output"]) / "run_contract.json").read_text())
        self.prepare_calls, self.calls, self.resets = [], [], 0
        self.fail, self.mutate, self.bad_merge = fail, mutate, bad_merge

    def prepare_one(self, sample):
        self.prepare_calls.append(sample["sample_id"])
        return {"input_ids": torch.tensor([[sample["sample_id"], 2]]), "pixel_values": torch.ones(2, 3)}

    def fresh_prompt(self):
        self.resets += 1

    def selector(self, condition):
        worker = self
        class Adapter:
            def __enter__(self):
                worker.active = condition
                return self

            def set_sample(self, task, sample_id):
                worker.index = worker.contract["generation_sample_ids"].index(sample_id)

            def finish_sample(self):
                audit = deepcopy(fixtures.make_bundle(worker.contract, worker.index)["records"][condition["label"]]["audit"])
                if worker.bad_merge:
                    audit["actual_merge_calls"] = 0
                return audit

            def __exit__(self, *args):
                pass
        return Adapter()

    def generate_prepared(self, sample, prepared):
        self.calls.append((sample["sample_id"], self.active["label"], id(prepared)))
        if self.fail and self.active["label"] == "random_global_seed0":
            raise RuntimeError("intentional CPU model failure")
        if self.mutate:
            prepared["input_ids"][0, 0] += 1
        return {"answer": sample["response"], "runtime": {"generated_ids": [3, 2], "generated_tokens": 2,
                "eos_token_id": 2, "eos_reached": True, "input_tokens": 2}}


class PairedImageRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"

    def tearDown(self):
        self.temp.cleanup()

    def config(self, task="MMCoQA", method="hybridkv_kvrepack"):
        fixture = fixtures.make_contract(self.data, task=task, method=method)
        folder = self.data / task / "images"
        folder.mkdir()
        for filename in ("one.png", "two.png"):
            with Image.new("RGB", (4, 4), "red") as image:
                image.save(folder / filename)
        config = build_random_head_plan(method, task, self.data, output=self.root / "run", limit=2, environ={}).resolved_config
        # Tiny native geometry exercises real pairing gates, not a pretend 7B.
        config["ablation"]["expected_geometry"] = fixture["head_geometry"]
        config["algorithm_env"].update(fixture["algorithm_env"])
        return config

    @unittest.skipUnless(importlib.util.find_spec("rouge"), "Original Rouge scorer required")
    def test_both_methods_prepare_once_reuse_inputs_and_pass_real_report(self):
        for method in ("hybridkv_kvrepack", "snapkv_kvrepack"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                self.root = Path(temporary)
                self.data = self.root / "data"
                config = self.config(method=method)
                workers = []
                def factory(cfg):
                    workers.append(FakeWorker(cfg))
                    return workers[-1]
                state = run(config, worker_factory=factory)
                worker = workers[0]
                self.assertEqual(state["status"], "complete")
                self.assertEqual(state["completed_samples"], 2)
                self.assertEqual(state["completed_generations"], 8)
                self.assertEqual(len(worker.prepare_calls), 2)
                self.assertEqual(worker.resets, 8)
                for sample_id in worker.prepare_calls:
                    self.assertEqual(len({pointer for sid, _, pointer in worker.calls if sid == sample_id}), 1)
                summary = summarize_run(config["output"], self.root / "report")
                self.assertEqual(summary["arm_scores_percent"]["similarity"], 100)
                self.assertEqual(summary["sample_count"], 2)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    run(config, worker_factory=factory)

    def test_model_failure_input_mutation_or_ineffective_merge_never_commits_sample(self):
        for flags, message in (({"fail": True}, "intentional"), ({"mutate": True}, "mutated"),
                               ({"bad_merge": True}, "effective")):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary:
                self.root = Path(temporary)
                self.data = self.root / "data"
                config = self.config()
                with self.assertRaisesRegex((RuntimeError, ValueError), message):
                    run(config, worker_factory=lambda cfg: FakeWorker(cfg, **flags))
                state = json.loads((Path(config["output"]) / "ablation_state.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["completed_generations"], 0)
                self.assertFalse(list((Path(config["output"]) / "samples").glob("*.json")))

    def test_missing_image_rejected_before_any_model_or_run_creation(self):
        config = self.config()
        (self.data / "MMCoQA/images/one.png").unlink()
        from unittest.mock import Mock
        factory = Mock(side_effect=AssertionError("model must not load"))
        with self.assertRaisesRegex(ValueError, "preflight"):
            run(config, worker_factory=factory)
        factory.assert_not_called()
        self.assertFalse(Path(config["output"]).exists())


if __name__ == "__main__":
    unittest.main()

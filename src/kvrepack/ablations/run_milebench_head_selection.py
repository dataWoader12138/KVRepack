"""Paired four-condition random merge-head experiment for MileBench images.

The CLI is launched only by an explicit --execute on the controller. CPU tests
can inject a worker; normal execution constructs a real CUDA worker after data
preflight. No model packages are imported when this module is imported.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

REFERENCE = "6c65bf801ab1e5d0cd85f1d24991e3d5399bebcf"
METHODS = ("hybridkv_kvrepack", "snapkv_kvrepack")
TASKS = ("Spot-the-Diff", "MMCoQA")


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload, *, replace=False):
    path = Path(path)
    raw = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if replace:
            os.replace(temporary, path)
        else:
            # Atomic exclusive commit, including when another process races us.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _conditions(config):
    ablation = config.get("ablation", {})
    if ablation.get("kind") != "random_head" or ablation.get("reference_commit") != REFERENCE:
        raise ValueError("Require the reference random-head experiment configuration")
    conditions = ablation.get("conditions", [])
    if (len(conditions) < 2 or conditions[0] != {"label": "similarity", "strategy": "similarity", "mask_seed": 0}
            or len({c.get("label") for c in conditions}) != len(conditions)):
        raise ValueError("Require one original similarity arm followed by distinct random seeds")
    seeds = []
    for c in conditions[1:]:
        seed = c.get("mask_seed")
        if type(seed) is not int or seed < 0 or c.get("strategy") != "random_global" or c.get("label") != f"random_global_seed{seed}":
            raise ValueError("Invalid random-head condition")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate random head seed")
    return conditions


@contextmanager
def _locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".random_head.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((output / name).exists() for name in ("samples", "run_contract.json", "ablation_state.json", "report")):
            raise ValueError("Use a fresh random-head output; resume is not implemented")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(config, *, worker_factory=None):
    from ..benchmarks.milebench.generate import _output_path, _validate_config
    from ..benchmarks.milebench.dataset import MileBenchDataset, load_annotation, validate_data
    from .milebench_head_selection_report import validate_bundle

    _validate_config(config)
    conditions = _conditions(config)
    if config["public_method"] not in METHODS or config["task"] not in TASKS or config["model"] != "qwen2_5_vl":
        raise ValueError("This reference experiment supports Qwen HybridKV/SnapKV and Spot-the-Diff/MMCoQA only")
    output = _output_path(config)
    data = validate_data(config["data_root"], config["task"], limit=config["limit"], verify_images=True)
    if not data["valid"]:
        raise ValueError("Image data preflight failed: " + "; ".join(data["errors"]))
    annotation = load_annotation(config["data_root"], config["task"], config["limit"])
    if len(annotation["data"]) != config["limit"]:
        raise ValueError("Requested sample prefix exceeds the annotation; refusing silent truncation")
    ids = [sample["sample_id"] for sample in annotation["data"]]
    if ids != data["sample_ids"] or hashlib.sha256(Path(data["annotation_path"]).read_bytes()).hexdigest() != data["annotation_sha256"]:
        raise ValueError("Annotation changed during preflight")
    # Keep the source image-count grouping after original-order prefix selection.
    groups = {}
    for sample in annotation["data"]:
        groups.setdefault(len(sample["task_instance"]["images_path"]), []).append(sample)
    ordered = [sample for group in groups.values() for sample in group]
    root = Path(__file__).resolve().parents[3]
    paths = ["src/kvrepack/ablations/" + name for name in (
        "head_selection.py", "hybrid_head_selection.py", "run_milebench_head_selection.py",
        "milebench_paired_worker.py", "milebench_head_selection_report.py")]
    paths += ["src/kvrepack/benchmarks/milebench/" + name for name in ("dataset.py", "evaluate.py", "workers.py")]
    paths += ["src/kvrepack/" + name for name in ("hybridkv_utils.py", "qwen_model_snapkv_kvrepack.py", "modeling_qwen2_5_vl.py", "configs/param_config.json")]
    contract = dict(schema_version=1, reference_commit=REFERENCE, method=config["public_method"], task=config["task"],
                    limit=config["limit"], conditions=[c["label"] for c in conditions], condition_settings=conditions,
                    data_root=config["data_root"],
                    annotation_sha256=data["annotation_sha256"], selected_sample_ids=ids,
                    generation_sample_ids=[s["sample_id"] for s in ordered],
                    head_geometry=config["ablation"]["expected_geometry"], algorithm_env=config["algorithm_env"],
                    generation=config["generation"], image=config["image"], seed=config["seed"],
                    pretrained=config["pretrained"], model_type=config["model_type"],
                    source_files_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths})
    identity = json_sha(contract)
    with _locked(output):
        atomic_json(output / "run_contract.json", contract)
        (output / "samples").mkdir()
        state = dict(status="loading", task=config["task"], method=config["public_method"],
                     target_samples=len(ids), target_generations=len(ids) * len(conditions),
                     completed_samples=0, completed_generations=0, contract_sha256=identity, started_utc=timestamp())
        atomic_json(output / "ablation_state.json", state)
        try:
            if worker_factory is None:
                from .milebench_paired_worker import PairedMileBenchWorker
                worker_factory = PairedMileBenchWorker
            worker = worker_factory(config)
            if worker.geometry != contract["head_geometry"]:
                raise ValueError("Loaded model head geometry differs from the frozen experiment")
            atomic_json(output / "model_runtime.json", worker.generation_metadata)
            image = config["image"]
            dataset = MileBenchDataset(ordered, annotation["meta_data"]["task_instruction"],
                str(Path(config["data_root"]) / config["task"] / "images"), image["max_context_len"],
                image["n_tokens_per_image"], worker.tokenizer, config["task"])
            from .milebench_paired_worker import tensor_fingerprints
            for index in range(len(dataset)):
                sample = dataset[index]
                if not sample["raw_img_list"]:
                    raise ValueError("Truncation removed every image; cannot run a paired multimodal sample")
                prepared = worker.prepare_one(sample)
                fingerprints = tensor_fingerprints(prepared)
                records = {}
                for condition in conditions:
                    state.update(status="running", current_sample_id=sample["sample_id"], current_condition=condition["label"])
                    atomic_json(output / "ablation_state.json", state, replace=True)
                    worker.fresh_prompt()
                    if tensor_fingerprints(prepared) != fingerprints:
                        raise ValueError("Shared image input mutated before a condition")
                    with worker.selector(condition) as adapter:
                        adapter.set_sample(config["task"], sample["sample_id"])
                        result = worker.generate_prepared(sample, prepared)
                        audit = adapter.finish_sample()
                    if tensor_fingerprints(prepared) != fingerprints:
                        raise ValueError("A condition mutated the shared image input")
                    prediction = dict(sample_id=sample["sample_id"], gt_response=sample["response"],
                                      pred_response=result["answer"], image=sample["raw_img_list"],
                                      question=sample["context"], gen_model_id="qwen2.5-vl", gen_kwargs=config["generation"])
                    records[condition["label"]] = dict(prediction=prediction, audit=audit,
                        condition={k: condition[k] for k in ("strategy", "mask_seed")},
                        input_fingerprints=fingerprints, runtime=result["runtime"])
                bundle = dict(schema_version=1, status="completed", task=config["task"], method=config["public_method"],
                    sample_index=index, sample_id=sample["sample_id"], contract_sha256=identity,
                    input_fingerprints=fingerprints, records=records, fairness={"status": "passed"})
                # Independently recompute the quota, actual merge coverage and
                # immutable-budget/candidate pairing before atomic commit.
                validate_bundle(bundle, contract, identity)
                atomic_json(output / "samples" / f"{index:06d}.json", bundle)
                state.update(completed_samples=index + 1, completed_generations=(index + 1) * len(conditions), updated_utc=timestamp())
                atomic_json(output / "ablation_state.json", state, replace=True)
            state.update(status="complete", finished_utc=timestamp())
            atomic_json(output / "ablation_state.json", state, replace=True)
            return state
        except BaseException as error:
            state.update(status="failed", error_type=type(error).__name__, error=str(error), finished_utc=timestamp())
            atomic_json(output / "ablation_state.json", state, replace=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(json.loads(args.config.read_text())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

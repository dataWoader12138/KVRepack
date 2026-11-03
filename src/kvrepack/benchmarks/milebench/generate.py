"""Generate and score one validated MileBench subset in an isolated run.

``run(config, worker_factory=...)`` accepts a CPU fake worker for integration
tests. A worker exposes ``tokenizer`` and ``generate_one(sample)`` returning
``{"answer": str, "runtime": dict}``. No model packages are imported until the
real worker is constructed after all selected images pass preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping


def _dump_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("benchmark") != "milebench":
        raise ValueError("This generation entry requires benchmark=milebench")
    if config.get("model") not in {"qwen2_5_vl", "llava_ov"}:
        raise ValueError("MileBench requires model=qwen2_5_vl or llava_ov")
    if config.get("batch_size", 1) != 1:
        raise ValueError("MileBench requires batch_size=1")
    for name in ("internal_method", "public_method", "pretrained", "model_type", "task", "data_root", "output"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"Resolved configuration requires {name}")
    from ...experiments.milebench import method_spec
    spec = method_spec(config["public_method"], config["model"], config.get("profile", "main11"))
    if spec.internal_method != config["internal_method"]:
        raise ValueError("internal_method disagrees with the selected MileBench method")
    if config["task"] not in spec.default_tasks:
        raise ValueError("task is not included in the selected MileBench profile")
    image = config.get("image")
    if not isinstance(image, dict):
        raise ValueError("Resolved configuration requires image preprocessing settings")
    for name in ("max_context_len", "n_tokens_per_image", "max_pixels", "min_pixels", "message_min_pixels"):
        value = image.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"image.{name} must be a positive integer")
    if not isinstance(image.get("system_prompt"), str):
        raise ValueError("image.system_prompt must be a string")
    if image["min_pixels"] > image["max_pixels"] or image["message_min_pixels"] > image["max_pixels"]:
        raise ValueError("Image pixel floors cannot exceed max_pixels")
    limit = config.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or null")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    from .workers import generation_kwargs
    generation_kwargs(config, config["model"])


def _output_path(config: Mapping[str, Any]) -> Path:
    root = Path(__file__).resolve().parents[4]
    output = Path(config["output"]).expanduser().resolve()
    protected = ("src", "vendor", "csrc", "provenance", "tests", "scripts", "docs", "requirements", ".git", ".github")
    if output == root or output in root.parents or any(output.is_relative_to(root / part) for part in protected):
        raise ValueError("output must be outside protected repository directories and repository ancestors")
    if output.is_relative_to(root) and not output.is_relative_to(root / "outputs"):
        raise ValueError("output within the repository must be under outputs/")
    data_root = Path(config["data_root"]).expanduser().resolve()
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("output must be outside the MileBench data directory")
    for name in ("selected_samples.json", "runtime.jsonl", "pred.json", "generation_summary.json", "scoring"):
        if (output / name).exists():
            raise ValueError(f"Refusing to overwrite existing MileBench artifact: {output / name}")
    return output


def run(config: Mapping[str, Any], *, worker_factory: Callable[[Mapping[str, Any]], Any] | None = None) -> dict:
    """Run complete generation then offline scoring; every failure propagates."""
    _validate_config(config)
    output = _output_path(config)
    from .dataset import MileBenchDataset, load_annotation, validate_data
    report = validate_data(config["data_root"], config["task"], limit=config.get("limit"), verify_images=True)
    if not report["valid"]:
        raise ValueError("MileBench dataset preflight failed: " + "; ".join(map(str, report.get("errors", []))))
    annotation = load_annotation(config["data_root"], config["task"], limit=config.get("limit"))
    samples = annotation["data"]
    if not samples:
        raise ValueError("MileBench selection is empty")
    sample_ids = [sample["sample_id"] for sample in samples]
    if sample_ids != report["sample_ids"]:
        raise ValueError("Selected annotations changed between preflight and loading")
    if hashlib.sha256(Path(report["annotation_path"]).read_bytes()).hexdigest() != report["annotation_sha256"]:
        raise ValueError("Annotation file changed during dataset preflight")
    # The source groups by image count before iterating. Preserve this processing
    # order because several baselines keep state between prompts. Limit selection
    # still happens first in original annotation order.
    image_count_groups: dict[int, list[dict]] = {}
    for sample in samples:
        image_count_groups.setdefault(len(sample["task_instance"]["images_path"]), []).append(sample)
    generation_samples = [sample for group in image_count_groups.values() for sample in group]
    generation_ids = [sample["sample_id"] for sample in generation_samples]
    random.seed(config.get("seed", 42))
    output.mkdir(parents=True, exist_ok=True)
    selected = {
        "schema_version": 1, "benchmark": "milebench", "task": config["task"],
        "annotation_path": report["annotation_path"], "annotation_sha256": report["annotation_sha256"],
        "sample_ids": sample_ids, "sample_ids_sha256": _hash_json(sample_ids),
        "generation_sample_ids": generation_ids, "generation_order": "source image-count groups in first-occurrence order",
        "selected_annotations_sha256": _hash_json(samples), "requested_limit": config.get("limit"),
        "selected_count": len(samples), "preflight": report,
    }
    _dump_exclusive(output / "selected_samples.json", selected)
    if worker_factory is None:
        from .workers import create_worker
        worker_factory = create_worker
    worker = worker_factory(config)
    image = config["image"]
    dataset = MileBenchDataset(
        annotation=generation_samples, task_instructions=annotation["meta_data"]["task_instruction"],
        img_dir=str(Path(config["data_root"]).expanduser().resolve() / config["task"] / "images"),
        max_context_len=image["max_context_len"], n_tokens_per_image=image["n_tokens_per_image"],
        tokenizer=worker.tokenizer, dataset_name=config["task"], combine_image=None,
    )
    predictions = []
    with (output / "runtime.jsonl").open("x", encoding="utf-8") as runtime_file:
        for index in range(len(dataset)):
            sample = dataset[index]
            if not sample["raw_img_list"]:
                raise ValueError(f"sample_id={sample['sample_id']} has no retained images after truncation")
            result = worker.generate_one(sample)
            if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
                raise ValueError(f"Worker returned no text answer for sample_id={sample['sample_id']}")
            runtime = result.get("runtime", {})
            if not isinstance(runtime, dict):
                raise ValueError("Worker runtime statistics must be a mapping")
            record = {
                "sample_id": sample["sample_id"], "image": list(sample["raw_img_list"]),
                "question": sample["context"], "gt_response": sample["response"],
                "gen_model_id": "qwen2.5-vl" if config["model"] == "qwen2_5_vl" else "llava-ov",
                "pred_response": result["answer"], "gen_kwargs": dict(config.get("generation", {})),
            }
            predictions.append(record)
            runtime_record = {
                "sample_id": sample["sample_id"], "retained_image_paths": list(sample["raw_img_list"]),
                "prompt_context_sha256": hashlib.sha256(sample["context"].encode("utf-8")).hexdigest(),
                "observations": runtime,
            }
            runtime_file.write(json.dumps(runtime_record, ensure_ascii=False, allow_nan=False) + "\n")
            runtime_file.flush()
    if [item["sample_id"] for item in predictions] != generation_ids:
        raise ValueError("Generation did not cover the exact selected sample IDs")
    _dump_exclusive(output / "pred.json", predictions)
    from .evaluate import evaluate_predictions, write_evaluation
    evaluation = evaluate_predictions(predictions, annotation, config["task"])
    write_evaluation(evaluation, output / "scoring")
    summary = {
        "schema_version": 1, "status": "completed", "benchmark": "milebench",
        "task": config["task"], "public_method": config["public_method"],
        "sample_count": len(predictions), "sample_ids_sha256": selected["sample_ids_sha256"],
        "full_task": evaluation["full_task"], "annotation_total_samples": evaluation["annotation_total_samples"],
        "prediction_path": str(output / "pred.json"), "evaluation_path": str(output / "scoring" / "eval.json"),
        "runtime_path": str(output / "runtime.jsonl"), "metrics": evaluation["eval"],
    }
    _dump_exclusive(output / "generation_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

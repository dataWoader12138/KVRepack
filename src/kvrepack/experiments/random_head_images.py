"""Pure, source-pinned plans for paired MileBench Random-head experiments.

Each plan loads one method for one task and compares all selector conditions on
the same prepared input. Selection seeds are independent of the model seed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Mapping, Sequence
import sys

from .runner import ExperimentPlan, build_plan

REFERENCE_COMMIT = "6c65bf801ab1e5d0cd85f1d24991e3d5399bebcf"
METHODS = ("hybridkv_kvrepack", "snapkv_kvrepack")
TASKS = ("Spot-the-Diff", "MMCoQA")
DEFAULT_PRETRAINED = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_HEAD_SEEDS = (0, 1, 2)


def conditions(head_seeds: Sequence[int] = DEFAULT_HEAD_SEEDS) -> list[dict]:
    seeds = tuple(head_seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("head_seeds must be a nonempty list of nonnegative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("head_seeds must not contain duplicates")
    return [{"label": "similarity", "strategy": "similarity", "mask_seed": 0}] + [
        {"label": f"random_global_seed{seed}", "strategy": "random_global", "mask_seed": seed}
        for seed in seeds
    ]


def build_random_head_plan(method: str, task: str, data_root: str | Path, *,
                           pretrained: str = DEFAULT_PRETRAINED,
                           output: str | Path | None = None,
                           head_seeds: Sequence[int] = DEFAULT_HEAD_SEEDS,
                           seed: int = 42, limit: int | None = None, all_samples: bool = False,
                           device: str = "0",
                           budget_ratio: float = 0.1, max_new_tokens: int = 512,
                           hf_home: str | Path | None = None,
                           environ: Mapping[str, str] | None = None,
                           repo_root: str | Path | None = None) -> ExperimentPlan:
    """Build a no-write/no-network plan; all_samples explicitly reads annotations.

    Ordinary planning retains the original 200-input default without touching
    data. Full-data planning freezes the complete annotation's count and hash.
    """
    if method not in METHODS:
        raise ValueError(f"Random-head image method must be one of {', '.join(METHODS)}")
    if task not in TASKS:
        raise ValueError(f"Random-head image task must be one of {', '.join(TASKS)}")
    if type(all_samples) is not bool:
        raise ValueError("all_samples must be Boolean")
    annotation_sha256 = None
    if all_samples:
        if limit is not None:
            raise ValueError("all_samples and an explicit limit are mutually exclusive")
        from ..benchmarks.milebench.dataset import annotation_path, load_annotation
        source = annotation_path(data_root, task)
        original = source.read_bytes()
        annotation = load_annotation(data_root, task, limit=None)
        if source.read_bytes() != original:
            raise ValueError("Annotation changed during full-data planning")
        limit = len(annotation["data"])
        annotation_sha256 = hashlib.sha256(original).hexdigest()
    else:
        limit = 200 if limit is None else limit
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be an integer from 1 to 200 for this experiment")
    arms = conditions(head_seeds)
    # Keep the source head-selection and merge recipe separate from the image
    # preprocessing/generation recipe. No random token, budget or head-category
    # mode is enabled through the environment.
    if method == "hybridkv_kvrepack":
        overrides = (
            "HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO=0.5",
            "HYBRID_KVREPACK_COMPRESS_RATIO=0.5", "K_DIFF_ALPHA=2",
            "HEAD_CLASSIFY=head_ada", "BUDGET_METHOD=head_ada",
        )
    else:
        overrides = (
            "SNAPKV_SIMHEAD_MERGE_RATIO=0.30",
            "SNAPKV_KVREPACK_COMPRESS_RATIO=0.70", "K_DIFF_ALPHA=2",
        )
    ambient = dict(environ) if environ is not None else None
    if hf_home is not None:
        import os
        ambient = dict(os.environ if ambient is None else ambient)
        cache = Path(hf_home).expanduser().resolve()
        # An explicit cache root determines all cache aliases, rather than
        # combining it with stale dataset/hub paths from another environment.
        ambient.update(HF_HOME=str(cache), HF_DATASETS_CACHE=str(cache / "datasets"),
                       HUGGINGFACE_HUB_CACHE=str(cache / "hub"), HF_HUB_CACHE=str(cache / "hub"))
    plan = build_plan(
        method, model="qwen2_5_vl", task=task, pretrained=str(pretrained),
        model_type="qwen2_5_vl-7b", benchmark="milebench", profile="main11",
        data_root=data_root, output=output, seed=seed, limit=limit, device=device,
        budget_ratio=budget_ratio, max_new_tokens=max_new_tokens,
        overrides=overrides, environ=ambient, repo_root=repo_root,
    )
    config = deepcopy(plan.resolved_config)
    config["generation"]["repetition_penalty"] = 1.05
    config["image"]["torch_dtype"] = "bfloat16"
    config["ablation"] = {
        "kind": "random_head", "reference_commit": REFERENCE_COMMIT,
        "conditions": arms,
        "selection_policy": "fixed_model_level_uniform_without_replacement_global_count_matched",
        "intervention": "merge_head_mask_only",
        "pairing": "same_prepared_input_tensor_and_native_pre_selection_state",
        "expected_geometry": {"layers": 28, "native_kv_heads": 4},
        "all_samples": all_samples,
        "annotation_sha256": annotation_sha256,
        "selected_sample_count": limit,
    }
    provenance = deepcopy(plan.provenance)
    provenance["random_head_reference_commit"] = REFERENCE_COMMIT
    provenance["random_head_protocol"] = {
        "model_seed": seed, "head_seeds": list(head_seeds),
        "conditions": deepcopy(arms),
        "image_adaptation": "MileBench image protocol and 512-token default; no video frame or WUPS scoring settings",
        "gpu_validation": "pending",
        "sample_selection": "complete_annotation" if all_samples else "original_prefix",
        "selected_sample_count": limit,
        "annotation_sha256": annotation_sha256,
    }
    tracked = provenance["configuration_sources_sha256"]
    for relative in (
        "src/kvrepack/experiments/random_head_images.py",
        "scripts/run_milebench_random_heads.py",
        "src/kvrepack/ablations/run_milebench_head_selection.py",
        "src/kvrepack/ablations/milebench_head_selection_report.py",
    ):
        path = plan.repo_root / relative
        if path.is_file():
            tracked[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    command = (sys.executable, "-m", "kvrepack.ablations.run_milebench_head_selection",
               "--config", str(plan.output / "resolved_config.json"))
    return replace(plan, command=command, resolved_config=config, provenance=provenance)


def build_matrix(data_root: str | Path, output_root: str | Path, *,
                 methods: Sequence[str] = METHODS, tasks: Sequence[str] = TASKS,
                 **kwargs) -> tuple[ExperimentPlan, ...]:
    """Plan one paired run per method/task, never one model load per condition."""
    methods, tasks = tuple(methods), tuple(tasks)
    if not methods or len(methods) != len(set(methods)) or any(method not in METHODS for method in methods):
        raise ValueError("methods must be a nonempty unique subset of the two Random-head methods")
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in TASKS for task in tasks):
        raise ValueError("tasks must be a nonempty unique subset of Spot-the-Diff and MMCoQA")
    output = Path(output_root).expanduser().resolve()
    return tuple(build_random_head_plan(method, task, data_root,
                                        output=output / method / task, **kwargs)
                 for method in methods for task in tasks)

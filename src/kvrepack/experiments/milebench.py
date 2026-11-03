"""Pure MileBench profiles over the repository's one implementation per method."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Sequence

from .registry import (
    METHODS_BY_MODEL, MethodSpec, Parameter, _fraction, _integer, resolve_spec,
)

PROFILES = ("image8", "main11")
IMAGE_METHODS = (
    "fullkv", "hybridkv", "snapkv", "sparsemm", "madakv", "lookm",
    "hybridkv_kvrepack", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack",
)


def profile_path(name: str = "main11") -> Path:
    if name not in PROFILES:
        raise ValueError(f"Unknown MileBench profile {name!r}; choose {', '.join(PROFILES)}")
    return Path(__file__).resolve().parents[1] / "configs" / "milebench" / f"{name}.json"


def load_profile(name: str = "main11") -> dict:
    profile = json.loads(profile_path(name).read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("name") != name:
        raise ValueError("Unsupported MileBench profile schema")
    return profile


def method_spec(method: str, model: str = "qwen2_5_vl", profile: str = "main11") -> MethodSpec:
    selected = load_profile(profile)
    if model not in METHODS_BY_MODEL:
        raise ValueError(f"Unknown model {model!r}")
    if method not in IMAGE_METHODS:
        raise ValueError(f"{method} is not included in the MileBench image experiment")
    if method in METHODS_BY_MODEL[model]:
        base = METHODS_BY_MODEL[model][method]
    else:
        from .llava_image_methods import LLAVA_IMAGE_UPGRADES
        base = LLAVA_IMAGE_UPGRADES[method]
    recipe = selected["methods"][model][method]
    params = dict(base.parameters) if method != "fullkv" else {}
    if method != "fullkv":
        params.setdefault("BUDGET", _integer("512"))
        params.setdefault("BUDGET_RATIO", _fraction("0.1", positive=True))
        params.setdefault("RATIO", _fraction("0.1", positive=True))
    # The image experiment deliberately exposes the primary score, not ablations
    # whose old names do not have equivalent meanings in the current core.
    for key in tuple(params):
        if key.endswith("MERGE_SCORE_MODE"):
            params[key] = Parameter("full", kind="enum", choices=("full",))
    for key, value in recipe["env"].items():
        if key not in params:
            raise ValueError(f"Image recipe contains an unregistered parameter: {key}")
        params[key].validate(key, value)
        params[key] = replace(params[key], default=value)
    dependents = tuple(key for key in params if key.endswith("_BUDGET_RATIO"))
    return replace(
        base, parameters=MappingProxyType(params), source_script=recipe["source_script"],
        default_pretrained=selected["default_pretrained"][model],
        source_lines="See provenance/milebench_sources.json for the exact source snapshot",
        default_tasks=tuple(selected["tasks"]), default_limit=selected["default_limit"],
        budget_ratio_dependents=dependents,
        notes=("Canonical target core; image recipe selection alone does not prove historical numerical equivalence.",) + tuple(selected["notes"]),
    )


def resolve_image_experiment(method: str, overrides: Sequence[str] = (), *,
                             model: str = "qwen2_5_vl", profile: str = "main11",
                             budget_ratio: float | None = None):
    return resolve_spec(method_spec(method, model, profile), overrides, budget_ratio=budget_ratio)


def describe_spec(spec: MethodSpec, profile: str = "main11") -> dict:
    description = spec.describe()
    description["defaults"].pop("frames", None)
    description.update(benchmark="milebench", profile=profile, supported_models=list(METHODS_BY_MODEL))
    description["defaults"]["generation"] = load_profile(profile)["generation"]
    return description


def describe_methods(profile: str = "main11") -> dict:
    return {
        method: {
            "public_method": method, "benchmark": "milebench", "profile": profile,
            "validation": "CPU integration only; GPU inference and quality are unverified",
            "model_variants": {model: describe_spec(method_spec(method, model, profile), profile)
                               for model in METHODS_BY_MODEL},
        } for method in IMAGE_METHODS
    }

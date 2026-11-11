"""One selected implementation and explicit defaults for each public method.

Source references are provenance labels, never paths used to import or execute code.
The values reproduce the selected scripts' current defaults, not historical best scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Mapping, Sequence


TASKS = (
    "nextqa_oe_test", "vatex_test", "videochatgpt_gen", "videochatgpt_temporal",
    "videochatgpt_consistency", "videochatgpt",
)
MODELS = ("qwen2_5_vl", "llava_ov")
MODEL_TYPES = ("qwen2_5_vl-3b", "qwen2_5_vl-7b", "llava-ov-7b")
SOURCE_ROOT = "HybridKV-artifact/scripts/eval/"
MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_LLAVA_OV = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


@dataclass(frozen=True)
class Parameter:
    default: str
    kind: str = "float"
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    exclusive_minimum: bool = False

    def validate(self, key: str, value: str) -> str:
        if not value or any(c in value for c in "\r\n\x00"):
            raise ValueError(f"{key} must have a nonempty single-line value")
        if self.choices:
            if value not in self.choices:
                raise ValueError(f"{key} must be one of {', '.join(self.choices)}")
            return value
        if self.kind in {"int", "odd_int"}:
            if not re.fullmatch(r"[0-9]+", value):
                raise ValueError(f"{key} must be an integer")
            number = int(value)
            if self.kind == "odd_int" and number % 2 == 0:
                raise ValueError(f"{key} must be a positive odd integer")
        else:
            try:
                number = float(value)
            except ValueError as exc:
                raise ValueError(f"{key} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{key} must be finite")
        if self.minimum is not None and (
            number < self.minimum or (self.exclusive_minimum and number == self.minimum)
        ):
            operator = ">" if self.exclusive_minimum else ">="
            raise ValueError(f"{key} must be {operator} {self.minimum}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{key} must be <= {self.maximum}")
        return value


def _fraction(value: str, *, positive: bool = False) -> Parameter:
    return Parameter(value, minimum=0, maximum=1, exclusive_minimum=positive)


def _integer(value: str, *, minimum: int = 1) -> Parameter:
    return Parameter(value, kind="int", minimum=minimum)


def _odd_integer(value: str) -> Parameter:
    return Parameter(value, kind="odd_int", minimum=1)


def _enum(value: str, *choices: str) -> Parameter:
    return Parameter(value, kind="enum", choices=choices or (value,))


def _flag(value: str) -> Parameter:
    return _enum(value, "0", "1")


def _pruning(ratio: str = "0.10") -> dict[str, Parameter]:
    return {"BUDGET": _integer("1024"), "BUDGET_RATIO": _fraction(ratio, positive=True),
            "RATIO": _fraction(ratio, positive=True)}


def _hybrid(ratio: str = "0.10") -> dict[str, Parameter]:
    return {**_pruning(ratio), "HEAD_CLASSIFY": _enum("head_ada"),
            "BUDGET_METHOD": _enum("head_ada")}


def _merge(prefix: str, *, gate: str, compression: str, alpha: str) -> dict[str, Parameter]:
    return {
        f"{prefix}_SIMHEAD_MERGE_RATIO": _fraction(gate),
        f"{prefix}_SIMHEAD_SIM_SPACE": _enum("raw", "raw", "rope"),
        f"{prefix}_KVREPACK_COMPRESS_RATIO": _fraction(compression, positive=True),
        "K_DIFF_ALPHA": Parameter(alpha, minimum=0),
        f"{prefix}_KVREPACK_SCORE_CORRECTION": _enum("none"),
    }


@dataclass(frozen=True)
class MethodSpec:
    name: str
    internal_method: str
    baseline: str | None
    source_script: str
    source_lines: str
    implementation: tuple[str, ...]
    model_path: str
    kernel: str | None
    default_pretrained: str
    default_tasks: tuple[str, ...]
    default_limit: int
    parameters: Mapping[str, Parameter]
    runtime_defaults: Mapping[str, str]
    budget_ratio_dependents: tuple[str, ...] = ()
    alpha_dependent: str | None = None
    method_name: str | None = None
    notes: tuple[str, ...] = ()
    model_family: str = "qwen2_5_vl"

    def describe(self) -> dict:
        return {
            "public_method": self.name, "internal_method": self.internal_method,
            "model": self.model_family,
            "supported_models": [model for model, methods in METHODS_BY_MODEL.items() if self.name in methods],
            "baseline": self.baseline, "source_script": self.source_script,
            "source_lines": self.source_lines, "implementation": list(self.implementation),
            "model_path": self.model_path, "kernel": self.kernel,
            "defaults": {
                "pretrained": self.default_pretrained, "tasks": list(self.default_tasks),
                "frames": 64, "limit": self.default_limit,
                "env": {key: param.default for key, param in self.parameters.items()},
                "normalized_runtime_env": dict(self.runtime_defaults),
            },
            "allowed_overrides": {
                key: {"kind": param.kind, "minimum": param.minimum,
                      "maximum": param.maximum, "choices": list(param.choices),
                      "exclusive_minimum": param.exclusive_minimum}
                for key, param in self.parameters.items()
            },
            "notes": list(self.notes),
        }


def _spec(name: str, internal: str, source: str, lines: str, params: dict,
          *, baseline: str | None = None, custom: bool = False, kernel: bool = False,
          model: str = MODEL_3B, tasks: tuple[str, ...] = ("nextqa_oe_test",),
          limit: int = 200, blocking: str = "0", dependents: tuple[str, ...] = (),
          alpha_dependent: str | None = None, method_name: str | None = None,
          notes: tuple[str, ...] = (), family: str = "qwen2_5_vl") -> MethodSpec:
    if family == "llava_ov":
        modules = (("kvrepack.modeling_llava_onevision", "kvrepack.modeling_qwen2") if custom
                   else ("transformers.models.llava_onevision.modeling_llava_onevision",) if name == "fullkv"
                   else ("kvrepack.qwen2_model",))
    elif name == "fullkv":
        modules = ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",)
    elif name == "asymkv":
        modules = ("kvrepack.asymkv_official.integration",)
    elif custom:
        modules = ("kvrepack.modeling_qwen2_5_vl", "kvrepack.hybridkv_utils")
    elif name.endswith("_kvrepack"):
        modules = (f"kvrepack.qwen_model_{internal}",)
    else:
        modules = ("kvrepack.qwen_model_randomkv",) if name == "randomkv" else ("kvrepack.qwen_model",)
    return MethodSpec(
        name, internal, baseline, SOURCE_ROOT + source, lines, modules,
        ("custom_llava_onevision" if custom else "transformers_llava_onevision") if family == "llava_ov"
        else "custom_qwen2_5_vl" if custom else "transformers_qwen2_5_vl",
        "kvrepack_kernel" if kernel else None, model, tasks, limit,
        MappingProxyType(params),
        MappingProxyType({"CUDA_LAUNCH_BLOCKING": blocking, "OMP_NUM_THREADS": "4",
                          "DECORD_EOF_RETRY_MAX": "20480", "TOKENIZERS_PARALLELISM": "false"}),
        dependents, alpha_dependent, method_name, notes, family,
    )


_video_tasks = ("videochatgpt_gen", "videochatgpt_temporal", "videochatgpt_consistency",
                "vatex_test", "nextqa_oe_test")
_hybrid_params = {
    **_hybrid(),
    "HYBRID_KVREPACK_CHUNK_SIZE": _enum("8"),
    "HYBRID_KVREPACK_COMPRESS_RATIO": _fraction("0.50", positive=True),
    "K_DIFF_ALPHA": Parameter("2", minimum=0),
    "HYBRID_KVREPACK_K_DIFF_ALPHA": Parameter("2", minimum=0),
    "HYBRID_KVREPACK_MAX_GROUP_SIZE": _integer("0", minimum=0),
    "HYBRID_KVREPACK_BLOCK_META_WEIGHT": _enum("uniform"),
    "HYBRID_KVREPACK_STATIC_MERGE_SCOPE": _enum("visual", "visual", "all"),
    "HYBRID_KVREPACK_K_MERGE_SPACE": _enum("raw_reapply", "raw_reapply", "rope"),
    "HYBRID_KVREPACK_SIMHEAD_ENABLE": _flag("1"),
    "HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO": _fraction("0.30"),
    "HYBRID_KVREPACK_SIMHEAD_SIM_SPACE": _enum("raw", "raw", "rope"),
    "HYBRID_KVREPACK_VERBOSE": _flag("0"), "HYBRIDKV_VERBOSE": _flag("0"),
    "HYBRID_KVREPACK_MERGE_SCORE_MODE": _enum("full", "full", "importance_only", "difference_only"),
}
_snap_params = {
    "BUDGET_RATIO": _fraction("0.10", positive=True),
    **_merge("SNAPKV", gate="0", compression="0.90", alpha="1"),
    "SNAPKV_KVREPACK_ENABLE": _flag("1"), "SNAPKV_KVREPACK_WINDOW_SIZE": _integer("32"),
    "SNAPKV_KVREPACK_KERNEL_SIZE": _odd_integer("5"),
    "SNAPKV_KVREPACK_POOLING": _enum("avgpool", "avgpool", "maxpool"),
    "SNAPKV_KVREPACK_GQA_FUNC": _enum("mean", "mean", "max"),
    "SNAPKV_KVREPACK_MERGE_SCORE_MODE": _enum("full", "full", "importance_only", "difference_only"),
}
_sparse_params = {
    **_pruning(), **_merge("SPARSEMM", gate="0.5", compression="0.8", alpha="0"),
    "SPARSEMM_KVREPACK_BUDGET_RATIO": _fraction("0.10", positive=True),
    "SPARSEMM_KVREPACK_RATIO": _fraction("0.10", positive=True),
    "SPARSEMM_KVREPACK_HEAD_SCORE": _enum("visual"),
    "SPARSEMM_KVREPACK_WINDOW_SIZE": _integer("32"),
    "SPARSEMM_KVREPACK_KERNEL_SIZE": _odd_integer("5"),
    "SPARSEMM_KVREPACK_POOLING": _enum("avgpool", "avgpool", "maxpool"),
    "SPARSEMM_KVREPACK_GQA_FUNC": _enum("mean", "mean", "max"),
    "SPARSEMM_KVREPACK_K_DIFF_ALPHA": Parameter("0", minimum=0),
}
_mada_params = {
    "BUDGET_RATIO": _fraction("0.10", positive=True),
    **_merge("MADAKV", gate="0.1", compression="0.8", alpha="1"),
    "MADAKV_KVREPACK_BUDGET_RATIO": _fraction("0.10", positive=True),
    "MADAKV_KVREPACK_THRESHOLD": _fraction("0.8"),
    "MADAKV_KVREPACK_WINDOW_SIZE": _integer("32"),
    "MADAKV_KVREPACK_GQA_FUNC": _enum("mean", "mean", "max"),
    "MADAKV_KVREPACK_K_DIFF_ALPHA": Parameter("1", minimum=0),
    "MADAKV_KVREPACK_ENABLE": _flag("1"),
}
_lookm_params = {
    **_pruning(), **_merge("LOOKM", gate="0.50", compression="0.80", alpha="0"),
    "LOOKM_KVREPACK_BUDGET_RATIO": _fraction("0.10", positive=True),
    "LOOKM_KVREPACK_WINDOW_SIZE": _integer("32"),
    "LOOKM_KVREPACK_TEXT_PRIORITY": _flag("1"),
    "LOOKM_KVREPACK_GQA_FUNC": _enum("mean", "mean", "max"),
    "LOOKM_KVREPACK_K_DIFF_ALPHA": Parameter("0", minimum=0),
    "LOOKM_KVREPACK_ENABLE": _flag("1"),
}

_specs = [
    _spec("fullkv", "fullkv", "run_qwen7b_fullkv_vatex_nextqa_limit200.sh", "7,82-96", {},
          model=MODEL_7B, tasks=("vatex_test", "nextqa_oe_test"),
          notes=("The selected script explicitly used the opensource FullKV adapter; no compression patch is installed.",)),
    _spec("hybridkv", "hybridkv", "qwen_ratio_hybridkv.sh", "1-36", _hybrid("0.1"),
          custom=True, kernel=True, model=MODEL_7B, blocking="1",
          tasks=("vatex_test", "videochatgpt_temporal", "videochatgpt_gen", "videochatgpt_consistency", "nextqa_oe_test")),
    *[_spec(name, name, "qwen_ratio.sh", "1-31", _pruning("0.1"), model=MODEL_7B,
            tasks=_video_tasks, kernel=name == "sparsemm")
      for name in ("snapkv", "sparsemm", "madakv", "lookm")],
    _spec("randomkv", "randomkv", "qwen_ratio_randomkv_3b.sh", "3-45",
          {"BUDGET_RATIO": _fraction("0.1", positive=True)}, limit=20000,
          tasks=("nextqa_oe_test", "vatex_test"), blocking="1",
          notes=("BUDGET is unset: the history budget follows BUDGET_RATIO.",)),
    _spec("asymkv", "asymkv_official", "qwen_asymkv_official_lmms.sh", "6-19,43-56",
          {**_pruning(), "MASK_RATIO": _fraction("0.10", positive=True),
           "ASYMKV_START_SIZE": _integer("32", minimum=0), "ASYMKV_CHUNK_SIZE": _integer("512")},
          tasks=("vatex_test", "nextqa_oe_test"),
          notes=("Only the official integration is exposed; paper and matched Hybrid variants are excluded.",)),
    _spec("hybridkv_kvrepack", "hybridkv_kvrepack", "qwen_ratio_hybrid_decouplekv_3b.sh",
          "20-68,96-105", _hybrid_params, baseline="hybridkv", custom=True, kernel=True,
          limit=10000, dependents=("RATIO",), alpha_dependent="HYBRID_KVREPACK_K_DIFF_ALPHA"),
    _spec("snapkv_kvrepack", "snapkv_kvrepack", "qwen_ratio_snapkv_decouplekv_simhead_3b.sh",
          "19-65,80-89", _snap_params, baseline="snapkv", method_name="snapkv_kvrepack_simhead",
          notes=("The selected script defaults to merge ratio 0; enable merge explicitly with --set SNAPKV_SIMHEAD_MERGE_RATIO=... .",)),
    _spec("sparsemm_kvrepack", "sparsemm_kvrepack", "qwen_ratio_sparsemm_decouplekv_simhead_3b.sh",
          "18-63,80-89", _sparse_params, baseline="sparsemm", kernel=True,
          dependents=("SPARSEMM_KVREPACK_BUDGET_RATIO",), alpha_dependent="SPARSEMM_KVREPACK_K_DIFF_ALPHA",
          method_name="sparsemm_kvrepack_simhead"),
    _spec("madakv_kvrepack", "madakv_kvrepack", "qwen_ratio_madakv_decouplekv_simhead_3b.sh",
          "19-59,75-84", _mada_params, baseline="madakv", limit=20000,
          dependents=("MADAKV_KVREPACK_BUDGET_RATIO",), alpha_dependent="MADAKV_KVREPACK_K_DIFF_ALPHA",
          method_name="madakv_kvrepack_simhead"),
    _spec("lookm_kvrepack", "lookm_kvrepack", "qwen_ratio_lookm_decouplekv_3b.sh",
          "18-58,72-81", _lookm_params, baseline="lookm", limit=9000, tasks=("vatex_test",),
          dependents=("RATIO", "LOOKM_KVREPACK_BUDGET_RATIO"), alpha_dependent="LOOKM_KVREPACK_K_DIFF_ALPHA",
          notes=("Selects the global similarity gate implementation, not the separate headsplit experiment.",)),
]
METHODS: Mapping[str, MethodSpec] = MappingProxyType({spec.name: spec for spec in _specs})
_llava_specs = [
    _spec(name, name, "llava_ov_ratio.sh", "1-35", _pruning("0.1"),
          model=MODEL_LLAVA_OV, tasks=("videochatgpt_consistency",), blocking="1",
          kernel=name != "fullkv", family="llava_ov",
          notes=(("The source script's current loop lists FullKV/LookM/SnapKV; this additional baseline is supported by its LLaVA adapter and selected explicitly via METHOD.",)
                 if name in {"sparsemm", "madakv"} else ()))
    for name in ("fullkv", "snapkv", "sparsemm", "madakv", "lookm")
] + [
    _spec("hybridkv", "hybridkv", "llava_ov_hybridkv.sh", "1-38", _hybrid("0.1"),
          model=MODEL_LLAVA_OV, tasks=("videochatgpt_temporal",), blocking="1", custom=True,
          kernel=True, family="llava_ov"),
]
LLAVA_METHODS: Mapping[str, MethodSpec] = MappingProxyType({spec.name: spec for spec in _llava_specs})
METHODS_BY_MODEL: Mapping[str, Mapping[str, MethodSpec]] = MappingProxyType({
    "qwen2_5_vl": METHODS, "llava_ov": LLAVA_METHODS,
})


def resolve_experiment(method: str, overrides: Sequence[str] = (), *,
                       budget_ratio: float | None = None,
                       model: str = "qwen2_5_vl") -> tuple[MethodSpec, dict[str, str], dict[str, str]]:
    """Validate explicit overrides; inherited environment is never a configuration input."""
    if model not in METHODS_BY_MODEL:
        raise ValueError(f"Unknown model {model!r}; expected one of {', '.join(MODELS)}")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; use --list-methods")
    if method not in METHODS_BY_MODEL[model]:
        raise ValueError(f"{method} for {model} is not included: no selected target experiment implements this combination")
    spec = METHODS_BY_MODEL[model][method]
    env = {key: param.default for key, param in spec.parameters.items()}
    explicit: dict[str, str] = {}
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or key not in spec.parameters:
            raise ValueError(f"Override {key!r} is not allowed for {method}; use --list-methods for the whitelist")
        if key in explicit:
            raise ValueError(f"Duplicate override for {key}")
        explicit[key] = spec.parameters[key].validate(key, value)
    if budget_ratio is not None:
        if "BUDGET_RATIO" not in spec.parameters:
            raise ValueError(f"{method} has no compression budget ratio")
        if "BUDGET_RATIO" in explicit:
            raise ValueError("Use either --budget-ratio or --set BUDGET_RATIO, not both")
        explicit["BUDGET_RATIO"] = spec.parameters["BUDGET_RATIO"].validate("BUDGET_RATIO", str(budget_ratio))
    env.update(explicit)
    if "BUDGET_RATIO" in explicit:
        for key in spec.budget_ratio_dependents:
            if key not in explicit:
                env[key] = env["BUDGET_RATIO"]
    if "RATIO" in explicit and "SPARSEMM_KVREPACK_RATIO" in env and "SPARSEMM_KVREPACK_RATIO" not in explicit:
        env["SPARSEMM_KVREPACK_RATIO"] = env["RATIO"]
    if "K_DIFF_ALPHA" in explicit and spec.alpha_dependent and spec.alpha_dependent not in explicit:
        env[spec.alpha_dependent] = env["K_DIFF_ALPHA"]
    return spec, env, explicit

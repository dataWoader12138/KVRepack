"""Image-only LLaVA upgrades; the published video matrix stays unchanged.

The current opensource LLaVA port is a distinct Qwen2 architecture adaptation.
Method names remain canonical and select exactly one implementation. This module
contains no inference imports, filesystem writes, model loads or GPU detection.
"""

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from .registry import METHODS, MODEL_LLAVA_OV, MethodSpec, Parameter


def _image_upgrade(name: str) -> MethodSpec:
    original = METHODS[name]
    params = {key: value for key, value in original.parameters.items()
              if not key.endswith("_MERGE_SCORE_MODE")}
    params.update({"BUDGET": Parameter("1024", kind="int", minimum=1),
                   "RATIO": Parameter("0.10", minimum=0, maximum=1, exclusive_minimum=True)})
    prefix = "HYBRID" if name == "hybridkv_kvrepack" else name.split("_")[0].upper()
    alpha_key = f"{prefix}_KVREPACK_K_DIFF_ALPHA"
    params[alpha_key] = Parameter(params["K_DIFF_ALPHA"].default, minimum=0)
    if prefix == "SPARSEMM":
        # The selected source supports weighted variants too; this profile keeps
        # the published starting point on ordinary, uniform no-log merging.
        params["SPARSEMM_KVREPACK_MERGE_WEIGHT_MODE"] = Parameter("uniform", kind="enum", choices=("uniform",))
    custom = prefix == "HYBRID"
    implementation = (("kvrepack.modeling_llava_onevision", "kvrepack.modeling_qwen2_kvrepack",
                       "kvrepack.hybridkv_utils_llava") if custom
                      else (f"kvrepack.qwen2_model_{name}",))
    return replace(
        original, parameters=MappingProxyType(params), model_family="llava_ov",
        default_pretrained=MODEL_LLAVA_OV, default_tasks=("CLEVR-Change",),
        source_script="HybridKV-opensource/scripts/run_llavaov7b_four_repack_methods_qwen7b_best.sh",
        source_lines="34-82,208-266", implementation=implementation,
        model_path="custom_llava_onevision" if custom else "transformers_llava_onevision",
        kernel="kvrepack_kernel" if prefix in {"HYBRID", "SPARSEMM", "MADAKV"} else None,
        alpha_dependent=alpha_key,
        notes=("Image-only LLaVA adaptation from the current opensource source snapshot; CPU-checked, GPU unverified.",),
    )


LLAVA_IMAGE_UPGRADES: Mapping[str, MethodSpec] = MappingProxyType({
    name: _image_upgrade(name) for name in (
        "hybridkv_kvrepack", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack",
    )
})

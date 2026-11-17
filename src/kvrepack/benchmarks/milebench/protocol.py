"""Registry-independent MileBench Qwen input and output metadata helpers.

Functions are copied unchanged from the validated image branch workers. Heavy
Pillow/metric/inference dependencies remain lazy. This module does not patch or
select any algorithm; model loading belongs to the styled image worker.
"""
from __future__ import annotations

import base64
from importlib import import_module
from io import BytesIO
from typing import Any, Mapping


DEFAULT_IMAGE_CONFIG = {
    "max_context_len": 32768,
    "n_tokens_per_image": 256,
    "max_pixels": 1605632,
    "min_pixels": 7168,
    "message_min_pixels": 200704,
    "system_prompt": "You are a helpful assistant.",
}

def qwen_messages(question: str, image_paths: list[str], image: Mapping[str, Any]) -> list[dict]:
    """Keep the source's JPEG conversion and images-before-text prompt order."""
    Image = import_module("PIL.Image")
    visuals = []
    for path in image_paths:
        with Image.open(path) as original:
            rgb = original.convert("RGB")
        try:
            with BytesIO() as buffer:
                rgb.save(buffer, format="JPEG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        finally:
            rgb.close()
        visuals.append({
            "type": "image", "image": f"data:image/jpeg;base64,{encoded}",
            "max_pixels": image["max_pixels"],
            # The original message floor differs from AutoProcessor.min_pixels.
            "min_pixels": image["message_min_pixels"],
        })
    return [
        {"role": "system", "content": image["system_prompt"]},
        {"role": "user", "content": visuals + [
            {"type": "text", "text": question.replace("<ImageHere>", "").strip()},
        ]},
    ]


def generation_kwargs(config: Mapping[str, Any], model: str) -> dict:
    """Translate the recorded zero-temperature policy into source adapter kwargs."""
    kwargs = {"max_new_tokens": 512, "min_new_tokens": 1, "do_sample": False,
              "temperature": 0.0, "num_beams": 1, **config.get("generation", {})}
    if kwargs.get("do_sample") is not False or kwargs.get("num_beams") != 1:
        raise ValueError("MileBench supports greedy generation with num_beams=1 only")
    if kwargs.get("temperature") not in (0, 0.0, None):
        raise ValueError("MileBench greedy generation requires temperature=0")
    kwargs["top_p"] = None
    if model == "qwen2_5_vl":
        kwargs["temperature"] = None
    elif model == "llava_ov":
        kwargs.pop("temperature", None)
    else:
        raise ValueError(f"Unsupported MileBench model: {model}")
    return kwargs


def _tensor_description(value: Any) -> dict | None:
    if not all(hasattr(value, name) for name in ("shape", "numel", "element_size", "dtype")):
        return None
    return {"shape": [int(dim) for dim in value.shape], "dtype": str(value.dtype),
            "elements": int(value.numel()), "tensor_bytes": int(value.numel() * value.element_size())}


def cache_statistics(cache: Any) -> dict:
    """Report observed cache shapes, without treating packed dimensions as tokens.

    Tensor byte counts are logical tensor sizes, not allocator memory or unique
    storage. Auxiliary/offloaded caches are listed separately when exposed.
    """
    if cache is None:
        return {"available": False, "reason": "generate did not expose past_key_values"}
    result: dict = {"available": True, "class": f"{type(cache).__module__}.{type(cache).__name__}"}
    get_length = getattr(cache, "get_seq_length", None)
    if callable(get_length):
        try:
            result["reported_seq_length"] = int(get_length())
        except (AttributeError, IndexError, NotImplementedError, TypeError, ValueError):
            pass
    groups = {}
    names = ("key_cache", "value_cache", "offload_k_cache", "offload_v_cache",
             "pruned_k_cache", "pruned_v_cache")

    def tensors(value: Any, location: list[int]) -> list[dict]:
        description = _tensor_description(value)
        if description is not None:
            return [{"index": location, **description}]
        if isinstance(value, (list, tuple)):
            return [entry for i, child in enumerate(value) for entry in tensors(child, location + [i])]
        return []

    for name in names:
        entries = tensors(getattr(cache, name, None), [])
        if entries:
            groups[name] = entries
    if not groups and isinstance(cache, (tuple, list)):
        groups["legacy_layers"] = tensors(cache, [])
    result["tensor_groups"] = groups
    result["tensor_bytes_semantics"] = "logical sizes of exposed tensors; may share storage; excludes model weights and allocator overhead"
    # The custom Hybrid cache carries actual per-head packed lengths here.
    head_lengths = []
    for layer in getattr(cache, "layers", ()):
        cluster = getattr(getattr(layer, "self_attn", None), "kv_cluster", None)
        lengths = getattr(cluster, "head_lens", None)
        if lengths is not None and hasattr(lengths, "tolist"):
            head_lengths.append(lengths.tolist())
    if head_lengths:
        result["observed_head_lengths"] = head_lengths
    return result

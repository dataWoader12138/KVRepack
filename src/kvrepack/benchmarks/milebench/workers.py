"""Lazy, single-sample MileBench model adapters.

The prompt and image conversion policy follows HybridKV-opensource's
``hybridkv/workers/model_workers.py``. Model dispatch uses only this package's
selected implementations. Importing this module never imports inference tools.
"""

from __future__ import annotations

import base64
import hashlib
from importlib import import_module
from io import BytesIO
import json
import os
import time
from typing import Any, Mapping


QWEN_METHODS = frozenset({
    "fullkv", "hybridkv", "hybridkv_kvrepack", "snapkv", "sparsemm",
    "madakv", "lookm", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack",
})
LLAVA_METHODS = frozenset({
    "fullkv", "hybridkv", "snapkv", "sparsemm", "madakv", "lookm",
    "hybridkv_kvrepack", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack",
})


def select_model_class(model: str, method: str, *, profile: str = "main11") -> Any:
    """Apply one method's patches before returning its unique model class."""
    from ...experiments.milebench import method_spec
    selected = method_spec(method, model, profile)
    if selected.internal_method != method:
        raise ValueError("MileBench dispatch requires the profile's canonical internal method")
    if model == "qwen2_5_vl":
        if method not in QWEN_METHODS:
            raise ValueError(f"Unsupported Qwen MileBench method: {method}")
        if method in {"hybridkv", "hybridkv_kvrepack"}:
            module = import_module("kvrepack.modeling_qwen2_5_vl")
        else:
            if method != "fullkv":
                import_module("kvrepack.monkeypatch").replace_qwen(method)
            module = import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
        return module.Qwen2_5_VLForConditionalGeneration
    if model == "llava_ov":
        if method not in LLAVA_METHODS:
            raise ValueError(f"Unsupported LLaVA MileBench method: {method}")
        if method in {"hybridkv", "hybridkv_kvrepack"}:
            module = import_module("kvrepack.modeling_llava_onevision")
        else:
            if method != "fullkv":
                import_module("kvrepack.monkeypatch").replace_llava_onevision(method)
            module = import_module("transformers.models.llava_onevision.modeling_llava_onevision")
        return module.LlavaOnevisionForConditionalGeneration
    raise ValueError(f"Unsupported MileBench model: {model}")


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


def llava_messages(question: str, image_count: int) -> list[dict]:
    """The source LLaVA prompt has a user turn and no system turn."""
    return [{"role": "user", "content": [{"type": "image"} for _ in range(image_count)] + [
        {"type": "text", "text": question.replace("<ImageHere>", "").strip()},
    ]}]


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


class MileBenchWorker:
    """Real CUDA worker. All heavy imports occur only after dataset preflight."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.family = config["model"]
        self.method = config["internal_method"]
        from ...experiments.milebench import method_spec
        selected = method_spec(config["public_method"], self.family, config.get("profile", "main11"))
        if selected.internal_method != self.method:
            raise ValueError("internal_method disagrees with the selected MileBench method")
        if os.environ.get("METHOD") != self.method:
            raise ValueError("METHOD environment disagrees with the resolved configuration; use the experiment runner")
        if os.environ.get("MODEL_TYPE") != config["model_type"]:
            raise ValueError("MODEL_TYPE environment disagrees with the resolved configuration; use the experiment runner")
        for name, expected in config.get("algorithm_env", {}).items():
            if os.environ.get(name) != str(expected):
                raise ValueError(f"Algorithm environment {name} disagrees with the resolved configuration; use the experiment runner")
        self.torch = import_module("torch")
        self.accelerator = import_module("accelerate").Accelerator()
        if self.accelerator.num_processes != 1:
            raise ValueError("MileBench requires one process and batch_size=1")
        self.device = self.accelerator.device
        if self.device.type != "cuda":
            raise RuntimeError("MileBench model generation requires a CUDA device; offline validation and scoring do not")
        seed = int(config.get("seed", 42))
        self.torch.manual_seed(seed)
        import_module("numpy").random.seed(seed)
        transformers = import_module("transformers")
        model_class = select_model_class(self.family, self.method, profile=config.get("profile", "main11"))
        # MileBench's source worker uses auto dtype for both baseline and upgrade.
        self.model = model_class.from_pretrained(
            config["pretrained"], torch_dtype="auto", device_map=str(self.device),
            attn_implementation=config.get("attention", "flash_attention_2"),
        ).eval()
        if self.family == "qwen2_5_vl":
            self.processor = transformers.AutoProcessor.from_pretrained(
                config["pretrained"], max_pixels=config["image"]["max_pixels"],
                min_pixels=config["image"]["min_pixels"],
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(config["pretrained"])
        else:
            self.processor = transformers.AutoProcessor.from_pretrained(config["pretrained"])
            self.tokenizer = self.processor.tokenizer
            self.tokenizer.padding_side = "left"
        self.gen_kwargs = generation_kwargs(config, self.family)

    def generate_one(self, sample: Mapping[str, Any]) -> dict:
        paths = sample["raw_img_list"]
        if not paths:
            raise ValueError("MileBench generation requires a retained visual input")
        visuals = []
        try:
            if self.family == "qwen2_5_vl":
                messages = qwen_messages(sample["context"], paths, self.config["image"])
                batched = [messages]
                texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
                images, videos = import_module("qwen_vl_utils").process_vision_info(batched)
                inputs = self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
                # Selected Hybrid source utilities may consult the current prompt.
                os.environ["QUESTION"] = sample["context"].replace("<ImageHere>", "").strip()
            else:
                Image = import_module("PIL.Image")
                for path in paths:
                    with Image.open(path) as original:
                        visuals.append(original.convert("RGB"))
                prompt = self.processor.apply_chat_template(
                    llava_messages(sample["context"], len(paths)), add_generation_prompt=True,
                )
                inputs = self.processor(text=prompt, images=visuals, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            input_ids = inputs["input_ids"]
            input_len = int(input_ids.shape[-1])
            ids_list = input_ids.tolist()
            input_hash = hashlib.sha256(json.dumps(ids_list, separators=(",", ":")).encode("utf-8")).hexdigest()
            grids = {}
            for name in ("image_grid_thw", "video_grid_thw"):
                if inputs.get(name) is not None and hasattr(inputs[name], "tolist"):
                    grids[name] = inputs[name].tolist()
            self.torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with self.torch.inference_mode(), self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16):
                generated = self.model.generate(
                    **inputs, eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id, **self.gen_kwargs,
                    use_cache=True, return_dict_in_generate=True,
                )
            self.torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            sequences = getattr(generated, "sequences", generated)
            new_ids = sequences[:, input_len:]
            decode_kwargs = {"skip_special_tokens": True}
            if self.family == "qwen2_5_vl":
                decode_kwargs["clean_up_tokenization_spaces"] = False
            answer = self.processor.batch_decode(new_ids, **decode_kwargs)[0].strip()
            runtime = {
                "input_tokens": input_len, "generated_tokens": int(new_ids.shape[-1]),
                "input_ids_sha256": input_hash,
                "retained_images": len(paths), "generation_seconds": elapsed,
                "cache": cache_statistics(getattr(generated, "past_key_values", None)),
            }
            if grids:
                runtime["visual_grids"] = grids
            token_counts = {}
            for kind in ("image", "video"):
                token_id = getattr(self.model.config, f"{kind}_token_id", None)
                if token_id is not None:
                    token_counts[kind] = int((input_ids == token_id).sum().item())
            if token_counts:
                runtime["visual_input_tokens"] = token_counts
            return {"answer": answer, "runtime": runtime}
        finally:
            for visual in visuals:
                visual.close()


def create_worker(config: Mapping[str, Any]) -> MileBenchWorker:
    return MileBenchWorker(config)

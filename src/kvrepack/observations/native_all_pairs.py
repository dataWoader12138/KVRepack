"""Independent native Qwen2/Qwen2.5-VL all-token prefill instrumentation.

Only projection outputs and the original rotary function's returned key are
observed. No cache, model output, attention implementation or previous observer
is replaced. Module import uses no torch/Transformers. Reductions reuse the
frozen linear-memory all-pairs helper without token or modality sampling.
"""
from __future__ import annotations

import copy
import functools
import importlib
import math

from .all_pairs import all_pairs_cosine_statistics

_METRICS = ("k_raw", "v", "k_rope")
_FAMILIES = {
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "apply_rotary_pos_emb"),
    "qwen2_5_vl": ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "apply_multimodal_rotary_pos_emb"),
}


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


class NativeAllPairsObserver:
    """Observe one batch-one full-prefill across every native KV head/layer.

    Use the intact outer model for metadata/hooks but invoke its native
    ``model.model(**inputs, use_cache=False, ...)`` backbone in this context.
    ``input_length`` covers every actual input token, including structural
    tokens; the caller must supply an unpadded single input sequence.
    """

    def __init__(self, model, *, family, input_length, rotary_module=None):
        if family not in _FAMILIES:
            raise ValueError(f"Unsupported native family: {family!r}")
        self.family = family
        self.input_length = _positive(input_length, "input_length")
        if input_length < 2:
            raise ValueError("All-off-diagonal input_length must be at least two; singleton statistic is unsupported")
        self._rotary_module = rotary_module
        self._rotary_name = _FAMILIES[family][1]
        self._original = None
        self._handles = []
        self._entered = False
        self._active = False
        self._failure = None
        self._current_layer = None
        self._order = []
        self._torch = None
        self._dtypes = {}
        self._devices = {}
        try:
            decoder = model.model if family == "qwen2" else model.model.language_model
            layers = list(decoder.layers)
        except AttributeError as error:
            raise ValueError(f"Native {family} decoder path does not match the selected family") from error
        if not layers:
            raise ValueError("Native decoder must have at least one layer")
        self._attentions = []
        self._stats = []
        for index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None or getattr(attention, "layer_idx", None) != index:
                raise ValueError("Native attention layer_idx values must be contiguous from zero")
            config = getattr(attention, "config", None)
            heads = getattr(attention, "num_key_value_heads", None)
            if heads is None:
                heads = getattr(config, "num_key_value_heads", None)
            heads = _positive(heads, f"Layer {index} native KV head count")
            configured = getattr(config, "num_key_value_heads", heads)
            if configured != heads:
                raise ValueError(f"Layer {index}: attention/config KV head counts disagree")
            dimension = getattr(attention, "head_dim", None)
            if dimension is None:
                dimension = getattr(config, "head_dim", None)
            if dimension is None:
                hidden = _positive(getattr(config, "hidden_size", None), "config.hidden_size")
                queries = _positive(getattr(config, "num_attention_heads", None), "config.num_attention_heads")
                if hidden % queries:
                    raise ValueError("Hidden size is not divisible by query heads")
                dimension = hidden // queries
            dimension = _positive(dimension, f"Layer {index} head dimension")
            for name in ("k_proj", "v_proj"):
                projection = getattr(attention, name, None)
                if projection is None or not callable(getattr(projection, "register_forward_hook", None)):
                    raise ValueError(f"Layer {index}: missing native {name}")
                width = getattr(projection, "out_features", heads * dimension)
                if width != heads * dimension:
                    raise ValueError(f"Layer {index}: {name} output width is not native KV heads * head_dim")
            self._attentions.append(attention)
            self._stats.append({
                "layer": index, "kv_heads": heads, "head_dim": dimension,
                "sequence_length": input_length, "capture_counts": {metric: 0 for metric in _METRICS},
                **{metric: None for metric in _METRICS},
                "inclusive": {metric: None for metric in _METRICS},
            })

    def _guard(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            self._failure = self._failure or str(error)
            raise

    def _pre_attention(self, index, args, kwargs):
        if index != len(self._order):
            raise ValueError("Expected exactly one forward with native layers in order")
        if self._current_layer is not None:
            if self._stats[self._current_layer]["capture_counts"] != {metric: 1 for metric in _METRICS}:
                raise ValueError("Previous native layer did not capture every projection/RoPE metric")
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if (not self._torch.is_tensor(hidden) or hidden.ndim != 3
                or tuple(hidden.shape[:2]) != (1, self.input_length)):
            raise ValueError("Expected batch-one complete prefill matching input_length")
        self._current_layer = index
        self._order.append(index)

    def _capture(self, index, metric, value):
        torch = self._torch
        if not self._active or index != self._current_layer:
            raise ValueError("Native capture occurred outside its attention layer")
        row = self._stats[index]
        if row["capture_counts"][metric]:
            raise ValueError(f"Layer {index}: repeated native {metric} capture")
        heads, dimension = row["kv_heads"], row["head_dim"]
        expected = ((1, heads, self.input_length, dimension) if metric == "k_rope"
                    else (1, self.input_length, heads * dimension))
        if not torch.is_tensor(value) or tuple(value.shape) != expected or not value.is_floating_point():
            raise ValueError(f"Layer {index}/{metric}: expected floating {expected}, before GQA repeat")
        if index in self._dtypes and value.dtype != self._dtypes[index]:
            raise ValueError("Native K/V/RoPE capture dtype mismatch")
        if index in self._devices and value.device != self._devices[index]:
            raise ValueError("Native K/V/RoPE capture device mismatch")
        self._dtypes[index], self._devices[index] = value.dtype, value.device
        vectors = value.detach()[0]
        if metric != "k_rope":
            vectors = vectors.reshape(self.input_length, heads, dimension).transpose(0, 1)
        result = all_pairs_cosine_statistics(vectors)
        if result["status"] != "supported" or result["token_count"] != self.input_length:
            raise ValueError("All native input tokens must participate in supported off-diagonal statistics")
        row[metric] = result["off_diagonal"]
        row["inclusive"][metric] = result["inclusive"]
        row["capture_counts"][metric] += 1

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Native observer instances cannot be entered twice")
        self._entered = True
        import torch
        self._torch = torch
        try:
            if self._rotary_module is None:
                self._rotary_module = importlib.import_module(_FAMILIES[self.family][0])
            original = getattr(self._rotary_module, self._rotary_name, None)
            if not callable(original):
                raise ValueError(f"Missing native rotary function {self._rotary_name}")
            if (getattr(original, "_kvrepack_native_all_pairs_owner", None) is not None
                    or getattr(original, "_kvrepack_observer_owner", None) is not None):
                raise RuntimeError("Native rotary function already has an active observer")
            self._original = original

            @functools.wraps(original)
            def wrapped_rotary(*args, **kwargs):
                def observe():
                    result = original(*args, **kwargs)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise ValueError("Native rotary function must return its original (query,key) tuple")
                    if self._current_layer is None:
                        raise ValueError("Native rotary call is outside language attention")
                    self._capture(self._current_layer, "k_rope", result[1])
                    return result
                return self._guard(observe)

            wrapped_rotary._kvrepack_native_all_pairs_owner = self
            setattr(self._rotary_module, self._rotary_name, wrapped_rotary)
            self._active = True
            for index, attention in enumerate(self._attentions):
                def pre_hook(module, args, kwargs, layer=index):
                    self._guard(self._pre_attention, layer, args, kwargs)
                    return None
                self._handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
                for name, metric in (("k_proj", "k_raw"), ("v_proj", "v")):
                    def projection_hook(module, args, output, layer=index, kind=metric):
                        self._guard(self._capture, layer, kind, output)
                        return None
                    self._handles.append(getattr(attention, name).register_forward_hook(projection_hook))
        except BaseException:
            self._cleanup(suppress_errors=True)
            raise
        return self

    def _cleanup(self, suppress_errors=False):
        failures = []
        try:
            for handle in self._handles:
                try:
                    handle.remove()
                except Exception as error:
                    failures.append(error)
        finally:
            self._handles.clear()
            if self._original is not None:
                setattr(self._rotary_module, self._rotary_name, self._original)
            self._active = False
            self._current_layer = None
        if failures:
            self._failure = self._failure or f"Native hook cleanup failed: {failures[0]}"
            if not suppress_errors:
                raise RuntimeError(self._failure) from failures[0]

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self._failure = self._failure or str(exc)
        self._cleanup(suppress_errors=exc is not None)
        return False

    def finish(self):
        if not self._entered:
            raise RuntimeError("Native observer has not been entered")
        if self._failure is not None:
            raise RuntimeError(f"Native observation failed: {self._failure}")
        if self._order != list(range(len(self._stats))):
            raise ValueError("Native observation is missing language layers")
        for row in self._stats:
            if row["capture_counts"] != {metric: 1 for metric in _METRICS}:
                raise ValueError("Native observation is missing or repeated a metric capture")
            for metric in _METRICS:
                for values in (row[metric], row["inclusive"][metric]):
                    if (not isinstance(values, list) or len(values) != row["kv_heads"]
                            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1 for value in values)):
                        raise ValueError("Invalid native per-head all-token cosine statistics")
        return copy.deepcopy(self._stats)

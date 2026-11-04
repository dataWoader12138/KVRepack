"""Temporary, read-only Qwen2.5-VL prefill observers for selected token pairs.

Importing this module does not import torch or Transformers. The context observes
one complete batch-one prefill, saves only small Python statistics, and restores
all hooks and the original multimodal RoPE function on every exit path. A global
RoPE wrapper is necessarily process-local: overlapping observers are rejected.
"""
from __future__ import annotations

import copy
import functools
import importlib
from collections.abc import Mapping, Sequence


_METRICS = ("k_raw", "v", "k_rope")
_GROUPS = {"visual_spatial", "text_question", "text_instruction"}


class AdjacentObserver:
    def __init__(self, model, pair_plan, *, rotary_module=None, include_pair_values=False):
        self._model = model
        self._rotary_module = rotary_module
        self._include_pair_values = include_pair_values
        self._original_rotary = None
        self._wrapper = None
        self._handles = []
        self._entered = False
        self._active = False
        self._failure = None
        self._current_layer = None
        self._layer_order = []
        self._dtypes = {}
        self._devices = {}
        self._torch = None
        if not isinstance(include_pair_values, bool):
            raise ValueError("include_pair_values must be bool")
        if not isinstance(pair_plan, Mapping) or type(pair_plan.get("input_length")) is not int or pair_plan["input_length"] <= 0:
            raise ValueError("pair_plan requires a positive integer input_length")
        self._sequence_length = pair_plan["input_length"]
        groups = pair_plan.get("groups")
        if not isinstance(groups, Mapping) or set(groups) - _GROUPS:
            raise ValueError("pair_plan groups must be the named observation-A groups")
        self._pairs = {}
        for group, data in groups.items():
            if not isinstance(data, Mapping) or not isinstance(data.get("pairs"), Sequence):
                raise ValueError(f"{group} requires a pairs sequence")
            pairs = []
            for pair in data["pairs"]:
                if not isinstance(pair, Sequence) or len(pair) != 2 or any(type(position) is not int for position in pair):
                    raise ValueError(f"{group}: expected integer [left, right] pair")
                left, right = pair
                if not 0 <= left < right < self._sequence_length:
                    raise ValueError(f"{group}: invalid native sequence pair {pair}")
                if group.startswith("text_") and right != left + 1:
                    raise ValueError(f"{group}: text pair is not natively adjacent")
                pairs.append((left, right))
            if len(set(pairs)) != len(pairs):
                raise ValueError(f"{group}: duplicate pairs")
            if "selected_pair_count" in data and data["selected_pair_count"] != len(pairs):
                raise ValueError(f"{group}: selected_pair_count mismatch")
            if pairs:
                self._pairs[group] = pairs
        try:
            self._layers = list(model.model.language_model.layers)
        except AttributeError as error:
            raise ValueError("Expected native Qwen model.model.language_model.layers") from error
        if not self._layers:
            raise ValueError("Language model must contain at least one layer")
        self._attentions = []
        self._stats = []
        for index, layer in enumerate(self._layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None or getattr(attention, "layer_idx", None) != index:
                raise ValueError("Language attention layer_idx values must be contiguous from zero")
            heads = getattr(attention, "num_key_value_heads", None)
            dimension = getattr(attention, "head_dim", None)
            if type(heads) is not int or heads <= 0 or type(dimension) is not int or dimension <= 0:
                raise ValueError(f"Layer {index}: missing positive KV-head metadata")
            for name in ("k_proj", "v_proj"):
                projection = getattr(attention, name, None)
                if projection is None or not callable(getattr(projection, "register_forward_hook", None)):
                    raise ValueError(f"Layer {index}: missing {name} projection")
            self._attentions.append(attention)
            self._stats.append({"layer": index, "kv_heads": heads, "head_dim": dimension,
                                "sequence_length": self._sequence_length,
                                "capture_counts": {metric: 0 for metric in _METRICS},
                                "groups": {group: {} for group in self._pairs}})

    def _guard(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            self._failure = self._failure or str(error)
            raise

    def _pre_attention(self, layer, args, kwargs):
        if layer != len(self._layer_order):
            raise ValueError("Expected one prefill with each language attention layer exactly once in order")
        if self._current_layer is not None:
            previous = self._stats[self._current_layer]["capture_counts"]
            if any(value != 1 for value in previous.values()):
                raise ValueError("Previous language attention did not produce all three observations")
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if not self._torch.is_tensor(hidden) or hidden.ndim != 3 or tuple(hidden.shape[:2]) != (1, self._sequence_length):
            raise ValueError("Expected batch-one full-prefill hidden states matching plan input_length")
        self._current_layer = layer
        self._layer_order.append(layer)

    def _capture(self, layer, metric, value):
        torch = self._torch
        if not self._active or self._current_layer != layer:
            raise ValueError("Projection/RoPE observation is outside its language attention")
        stats = self._stats[layer]
        if stats["capture_counts"][metric]:
            raise ValueError(f"Layer {layer}: repeated {metric} capture")
        heads, dimension = stats["kv_heads"], stats["head_dim"]
        expected = ((1, heads, self._sequence_length, dimension) if metric == "k_rope"
                    else (1, self._sequence_length, heads * dimension))
        if not torch.is_tensor(value) or tuple(value.shape) != expected:
            raise ValueError(f"Layer {layer}/{metric}: expected {expected}, before GQA repeat")
        if not value.is_floating_point():
            raise ValueError(f"Layer {layer}/{metric}: expected floating-point tensor")
        if layer in self._dtypes and value.dtype != self._dtypes[layer]:
            raise ValueError(f"Layer {layer}: K/V/RoPE dtype mismatch")
        if layer in self._devices and value.device != self._devices[layer]:
            raise ValueError(f"Layer {layer}: K/V/RoPE device mismatch")
        self._dtypes[layer], self._devices[layer] = value.dtype, value.device
        stats["dtype"] = str(value.dtype)
        with torch.no_grad():
            for group, pairs in self._pairs.items():
                count = len(pairs)
                endpoints = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
                indices = torch.tensor(endpoints, dtype=torch.long, device=value.device)
                # Select before reshape/cast/transfer: never copy the full cache.
                if metric == "k_rope":
                    selected = value.detach()[0].index_select(1, indices).to(dtype=torch.float32)
                else:
                    selected = value.detach()[0].index_select(0, indices).reshape(2 * count, heads, dimension).transpose(0, 1).to(dtype=torch.float32)
                if not bool(torch.isfinite(selected).all().item()):
                    raise ValueError(f"Layer {layer}/{group}/{metric}: nonfinite selected vectors")
                norms = torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
                if not bool(torch.isfinite(norms).all().item()) or bool((norms == 0).any().item()):
                    raise ValueError(f"Layer {layer}/{group}/{metric}: zero or nonfinite norm; cosine undefined")
                normalized = selected / norms
                cosines = (normalized[:, :count] * normalized[:, count:]).sum(dim=-1)
                if not bool(torch.isfinite(cosines).all().item()):
                    raise ValueError(f"Layer {layer}/{group}/{metric}: nonfinite cosine")
                cosines = cosines.clamp(-1.0, 1.0)
                stats["groups"][group][metric] = cosines.mean(dim=1).cpu().tolist()
                if self._include_pair_values:
                    stats["groups"][group].setdefault("pair_cosines", {})[metric] = cosines.cpu().tolist()
        stats["capture_counts"][metric] += 1

    def __enter__(self):
        if self._entered:
            raise RuntimeError("AdjacentObserver instances cannot be entered twice")
        self._entered = True
        import torch
        self._torch = torch
        try:
            if self._rotary_module is None:
                self._rotary_module = importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
            original = getattr(self._rotary_module, "apply_multimodal_rotary_pos_emb", None)
            if not callable(original):
                raise ValueError("Missing apply_multimodal_rotary_pos_emb")
            if getattr(original, "_kvrepack_observer_owner", None) is not None:
                raise RuntimeError("The rotary module already has an active AdjacentObserver")
            self._original_rotary = original

            @functools.wraps(original)
            def wrapped_rotary(*args, **kwargs):
                def observe():
                    result = original(*args, **kwargs)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise ValueError("mRoPE must return its original (query, key) tuple")
                    if self._current_layer is None:
                        raise ValueError("mRoPE invoked outside observed language attention")
                    self._capture(self._current_layer, "k_rope", result[1])
                    return result  # Preserve the exact original tuple/tensors.
                return self._guard(observe)

            wrapped_rotary._kvrepack_observer_owner = self
            self._wrapper = wrapped_rotary
            self._rotary_module.apply_multimodal_rotary_pos_emb = wrapped_rotary
            self._active = True
            for index, attention in enumerate(self._attentions):
                def pre_hook(module, args, kwargs, layer=index):
                    self._guard(self._pre_attention, layer, args, kwargs)
                    return None

                self._handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
                for projection_name, metric in (("k_proj", "k_raw"), ("v_proj", "v")):
                    def projection_hook(module, args, output, layer=index, kind=metric):
                        self._guard(self._capture, layer, kind, output)
                        return None

                    self._handles.append(getattr(attention, projection_name).register_forward_hook(projection_hook))
        except BaseException:
            self._cleanup(suppress_errors=True)
            raise
        return self

    def _cleanup(self, suppress_errors=False):
        errors = []
        try:
            for handle in self._handles:
                try:
                    handle.remove()
                except Exception as error:
                    errors.append(error)
        finally:
            self._handles.clear()
            if self._original_rotary is not None:
                self._rotary_module.apply_multimodal_rotary_pos_emb = self._original_rotary
            self._active = False
            self._current_layer = None
            self._wrapper = None
        if errors:
            self._failure = self._failure or f"Hook cleanup failed: {errors[0]}"
            if not suppress_errors:
                raise RuntimeError(self._failure) from errors[0]

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self._failure = self._failure or str(exc)
        self._cleanup(suppress_errors=exc is not None)
        return False

    def finish(self):
        """Return JSON-safe per-layer means after one complete observed prefill."""
        if not self._entered:
            raise RuntimeError("Observer has not been entered")
        if self._failure is not None:
            raise RuntimeError(f"Observation failed: {self._failure}")
        if self._layer_order != list(range(len(self._layers))):
            raise ValueError("Observation is missing actual language attention layers")
        for layer in self._stats:
            if any(layer["capture_counts"][metric] != 1 for metric in _METRICS):
                raise ValueError(f"Layer {layer['layer']}: missing or repeated metric capture")
            for group in self._pairs:
                if not all(metric in layer["groups"][group] for metric in _METRICS):
                    raise ValueError(f"Layer {layer['layer']}: incomplete active group {group}")
        return copy.deepcopy(self._stats)

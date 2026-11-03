from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class AsymKVOfficialCache(Cache):
    """Qwen cache carrying the released AsymKV cardinality tensor."""

    is_compileable = False

    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self.cardinality_cache: List[Optional[torch.Tensor]] = []
        self._capture_gradients = False
        self.original_seen_tokens = 0

    def __len__(self) -> int:
        return len(self.key_cache)

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.cardinality_cache.append(None)

    def begin_step(self, capture_gradients: bool) -> None:
        self._capture_gradients = bool(capture_gradients)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self._ensure_layer(layer_idx)

        batch_size, kv_heads, current_length, _ = key_states.shape
        cardinality = torch.ones(
            (batch_size, kv_heads, current_length),
            dtype=key_states.dtype,
            device=key_states.device,
        )

        old_key = self.key_cache[layer_idx]
        if self._capture_gradients:
            current_key = key_states.detach().requires_grad_(True)
            if old_key is None:
                combined_key = current_key
            else:
                history_key = old_key.detach().requires_grad_(True)
                combined_key = torch.cat((history_key, current_key), dim=-2)
            combined_key.retain_grad()
        else:
            current_key = key_states.detach()
            combined_key = (
                current_key
                if old_key is None
                else torch.cat((old_key.detach(), current_key), dim=-2)
            )

        old_value = self.value_cache[layer_idx]
        combined_value = (
            value_states.detach()
            if old_value is None
            else torch.cat((old_value.detach(), value_states.detach()), dim=-2)
        )
        old_cardinality = self.cardinality_cache[layer_idx]
        combined_cardinality = (
            cardinality
            if old_cardinality is None
            else torch.cat((old_cardinality.detach(), cardinality), dim=-1)
        )

        self.key_cache[layer_idx] = combined_key
        self.value_cache[layer_idx] = combined_value
        self.cardinality_cache[layer_idx] = combined_cardinality
        if layer_idx == 0:
            self.original_seen_tokens += current_length
        return combined_key, combined_value

    def get_cardinality(self, layer_idx: int) -> torch.Tensor:
        cardinality = self.cardinality_cache[layer_idx]
        if cardinality is None:
            raise RuntimeError(f"AsymKV cardinality is missing for layer {layer_idx}")
        return cardinality

    def history_gradient(self, layer_idx: int, history_length: int) -> Optional[torch.Tensor]:
        key_states = self.key_cache[layer_idx]
        if key_states is None or key_states.grad is None:
            return None
        return key_states.grad[:, :, :history_length, :].detach()

    def set_layer(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cardinality: torch.Tensor,
    ) -> None:
        self._ensure_layer(layer_idx)
        self.key_cache[layer_idx] = key_states.detach()
        self.value_cache[layer_idx] = value_states.detach()
        self.cardinality_cache[layer_idx] = cardinality.detach()

    def clone_for_probe(self) -> "AsymKVOfficialCache":
        clone = AsymKVOfficialCache()
        for layer_idx, key_states in enumerate(self.key_cache):
            if key_states is None:
                continue
            value_states = self.value_cache[layer_idx]
            cardinality = self.cardinality_cache[layer_idx]
            if value_states is None or cardinality is None:
                raise RuntimeError(f"Incomplete AsymKV cache at layer {layer_idx}")
            clone.set_layer(layer_idx, key_states, value_states, cardinality)
        clone.original_seen_tokens = self.original_seen_tokens
        clone.begin_step(capture_gradients=True)
        return clone

    def detach_(self) -> "AsymKVOfficialCache":
        for layer_idx, key_states in enumerate(self.key_cache):
            if key_states is None:
                continue
            self.key_cache[layer_idx] = key_states.detach()
            self.value_cache[layer_idx] = self.value_cache[layer_idx].detach()
            self.cardinality_cache[layer_idx] = self.cardinality_cache[layer_idx].detach()
        self._capture_gradients = False
        return self

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return int(self.key_cache[layer_idx].shape[-2])

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        for layer_idx, cardinality in enumerate(self.cardinality_cache):
            if cardinality is not None:
                self.cardinality_cache[layer_idx] = cardinality.index_select(
                    0, beam_idx.to(cardinality.device)
                )

    def physical_lengths(self) -> List[int]:
        return [
            0 if key_states is None else int(key_states.shape[-2])
            for key_states in self.key_cache
        ]

"""Internal reducer retained for the unchanged HybridKV cache dependency."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

def reduce_groups_asymkv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    fisher_gradient: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
    group_size: torch.Tensor,
    *,
    epsilon: float = 1e-21,
    cardinality_clip: float = 5.0,
    reduce_batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply AsymKV's Fisher-K and sum-V rules to fixed merge groups."""
    if key_states.shape != value_states.shape:
        raise ValueError(
            "Hybrid-AsymKV requires matching K/V shapes, got "
            f"{tuple(key_states.shape)} and {tuple(value_states.shape)}."
        )
    if fisher_gradient.shape != key_states.shape:
        raise ValueError(
            "Hybrid-AsymKV Fisher shape mismatch: expected "
            f"{tuple(key_states.shape)}, got {tuple(fisher_gradient.shape)}."
        )
    if group_ids.shape != (key_states.shape[0],):
        raise ValueError("Hybrid-AsymKV group_ids must contain one id per entry.")
    if reduce_batch_size is None:
        reduce_batch_size = int(
            os.getenv("HYBRID_ASYMKV_REDUCE_BATCH_SIZE", "512")
        )
    if reduce_batch_size <= 0:
        raise ValueError("Hybrid-AsymKV reduce batch size must be positive.")

    fisher_diagonal = torch.nan_to_num(
        fisher_gradient.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).square()
    weighted_key = torch.zeros(
        (group_count, key_states.shape[-1]),
        dtype=torch.float32,
        device=key_states.device,
    )
    fisher_sum = torch.zeros_like(weighted_key)
    singleton_key = torch.zeros_like(weighted_key)
    merged_value = torch.zeros(
        (group_count, value_states.shape[-1]),
        dtype=torch.float32,
        device=value_states.device,
    )

    # AsymKV compresses long contexts in chunks. Batch only the independent
    # reductions here so HybridKV's selected groups remain exactly unchanged.
    for start in range(0, key_states.shape[0], reduce_batch_size):
        end = min(start + reduce_batch_size, key_states.shape[0])
        batch_groups = group_ids[start:end]
        batch_keys = key_states[start:end].float()
        batch_fisher = fisher_diagonal[start:end]
        weighted_key.index_add_(
            0, batch_groups, batch_keys * batch_fisher
        )
        fisher_sum.index_add_(0, batch_groups, batch_fisher)
        singleton_key.index_add_(0, batch_groups, batch_keys)
        merged_value.index_add_(
            0, batch_groups, value_states[start:end].float()
        )

    merged_key = weighted_key / (fisher_sum + epsilon)

    # AsymKV leaves entries outside selected pairs untouched. Preserve that
    # behavior for singleton groups even when their measured gradient is zero.
    singleton_key = singleton_key / group_size.clamp_min(1.0).unsqueeze(-1)
    merged_key = torch.where(
        (group_size <= 1).unsqueeze(-1), singleton_key, merged_key
    ).to(key_states.dtype)

    merged_value = merged_value.to(value_states.dtype)
    cardinality = group_size.clamp(max=float(cardinality_clip)).to(torch.float32)
    return merged_key, merged_value, cardinality

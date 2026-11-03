from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .attention import repeat_cardinality
from .cache import AsymKVOfficialCache


def _compress_layer_like_upstream(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cardinality: torch.Tensor,
    attention: torch.Tensor,
    raw_hessian: Optional[torch.Tensor],
    *,
    sequence_length: int,
    cache_size: int,
    start_size: int,
    num_key_value_groups: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Dict[str, int],
]:
    middle_key = key_states[:, :, start_size:sequence_length, :].clone()
    middle_value = value_states[:, :, start_size:sequence_length, :].clone()
    middle_cardinality = cardinality[:, :, start_size:sequence_length].clone()

    attention = attention[:, :, :, start_size:]
    attention = attention.sum(dim=-2)
    repeated_cardinality = repeat_cardinality(
        middle_cardinality, num_key_value_groups
    )
    attention = attention * repeated_cardinality[:, :, : attention.shape[2]]
    attention = attention[:, :, :-1] + attention[:, :, 1:]
    pair_weight = attention.sum(dim=1).squeeze(0)

    pair_cardinality = middle_cardinality.sum(dim=-2)
    pair_cardinality = pair_cardinality[:, :-1] + pair_cardinality[:, 1:]
    pair_cardinality = pair_cardinality[0]
    gamma = 4096.0 / math.log(512.0)
    positions = torch.arange(
        1,
        len(pair_weight) + 1,
        dtype=torch.float32,
        device=pair_weight.device,
    )
    pair_weight = pair_weight / torch.exp(positions / gamma) * pair_cardinality

    requested = sequence_length - cache_size
    merge_indices = pair_weight.topk(
        requested, dim=-1, largest=False
    ).indices

    if raw_hessian is None:
        merged_key = (middle_key[:, :, :-1, :] + middle_key[:, :, 1:, :]) / 2
        merged_value = (
            middle_value[:, :, :-1, :] + middle_value[:, :, 1:, :]
        ) / 2
        merged_cardinality = (
            middle_cardinality[:, :, :-1] + middle_cardinality[:, :, 1:]
        )
        compressed_hessian = None
    else:
        middle_hessian_diagonal = raw_hessian[:, :, start_size:sequence_length, :].pow(2)
        left_hessian = middle_hessian_diagonal[:, :, :-1, :]
        right_hessian = middle_hessian_diagonal[:, :, 1:, :]
        epsilon = 1e-21
        merged_key = 1 / (left_hessian + right_hessian + epsilon) * (
            middle_key[:, :, :-1, :] * left_hessian
            + middle_key[:, :, 1:, :] * right_hessian
        )
        merged_value = middle_value[:, :, :-1, :] + middle_value[:, :, 1:, :]
        merged_cardinality = (
            middle_cardinality[:, :, :-1] + middle_cardinality[:, :, 1:]
        )

        middle_raw_hessian = raw_hessian[:, :, start_size:sequence_length, :].clone()
        merged_raw_hessian = (
            middle_raw_hessian[:, :, :-1, :]
            + middle_raw_hessian[:, :, 1:, :]
        )
        middle_raw_hessian[:, :, merge_indices, :] = merged_raw_hessian[
            :, :, merge_indices, :
        ]
        compressed_hessian = middle_raw_hessian

    keep_mask = torch.ones(
        middle_key.shape[2], dtype=torch.bool, device=middle_key.device
    )
    keep_mask[merge_indices + 1] = False
    middle_key[:, :, merge_indices, :] = merged_key[:, :, merge_indices, :]
    middle_value[:, :, merge_indices, :] = merged_value[:, :, merge_indices, :]
    middle_cardinality[:, :, merge_indices] = merged_cardinality[
        :, :, merge_indices
    ]

    new_key = middle_key[:, :, keep_mask, :]
    new_value = middle_value[:, :, keep_mask, :]
    new_cardinality = torch.clip(middle_cardinality[:, :, keep_mask], max=5)
    if compressed_hessian is not None:
        compressed_hessian = compressed_hessian[:, :, keep_mask, :]
        compressed_hessian = torch.cat(
            (raw_hessian[:, :, :start_size, :], compressed_hessian), dim=2
        )

    output_key = torch.cat((key_states[:, :, :start_size, :], new_key), dim=2)
    output_value = torch.cat((value_states[:, :, :start_size, :], new_value), dim=2)
    output_cardinality = torch.cat(
        (cardinality[:, :, :start_size], new_cardinality), dim=2
    )
    removed = int((~keep_mask).sum().item())
    sorted_indices = torch.sort(merge_indices).values
    overlapping_pairs = int(
        (sorted_indices[1:] - sorted_indices[:-1] == 1).sum().item()
    )
    return (
        output_key,
        output_value,
        output_cardinality,
        compressed_hessian,
        {
            "requested_merges": requested,
            "unique_removed": removed,
            "overlapping_pair_boundaries": overlapping_pairs,
            "physical_after": int(output_key.shape[-2]),
        },
    )


def compress_cache_like_upstream(
    cache: AsymKVOfficialCache,
    attentions: Sequence[torch.Tensor],
    num_key_value_groups: int,
    raw_hessian_per_layer: Optional[Sequence[torch.Tensor]],
    *,
    cache_size: int,
    start_size: int = 32,
) -> Tuple[Optional[List[torch.Tensor]], Dict[str, object]]:
    """Port ``AsymKVCache.__call__`` without correcting released-code edges."""
    sequence_length = cache.get_seq_length(0)
    before = cache.physical_lengths()
    if sequence_length <= cache_size:
        return (
            None if raw_hessian_per_layer is None else list(raw_hessian_per_layer),
            {"before": before, "after": before, "layers": [], "compressed": False},
        )
    if len(attentions) != len(cache.key_cache):
        raise RuntimeError(
            f"Official AsymKV expected {len(cache.key_cache)} attention tensors, "
            f"got {len(attentions)}"
        )

    output_hessian: List[torch.Tensor] = []
    layer_reports: List[Dict[str, int]] = []
    use_hessian = raw_hessian_per_layer is not None
    for layer_idx, key_states in enumerate(cache.key_cache):
        if key_states is None:
            continue
        value_states = cache.value_cache[layer_idx]
        cardinality = cache.cardinality_cache[layer_idx]
        if value_states is None or cardinality is None:
            raise RuntimeError(f"Incomplete official AsymKV cache at layer {layer_idx}")
        raw_hessian = (
            raw_hessian_per_layer[layer_idx] if use_hessian else None
        )
        (
            output_key,
            output_value,
            output_cardinality,
            compressed_hessian,
            layer_report,
        ) = _compress_layer_like_upstream(
            key_states,
            value_states,
            cardinality,
            attentions[layer_idx],
            raw_hessian,
            sequence_length=sequence_length,
            cache_size=cache_size,
            start_size=start_size,
            num_key_value_groups=num_key_value_groups,
        )
        cache.set_layer(
            layer_idx, output_key, output_value, output_cardinality
        )
        layer_reports.append(layer_report)
        if compressed_hessian is not None:
            output_hessian.append(compressed_hessian)

    cache.detach_()
    return (
        output_hessian if use_hessian else None,
        {
            "before": before,
            "after": cache.physical_lengths(),
            "layers": layer_reports,
            "compressed": True,
        },
    )

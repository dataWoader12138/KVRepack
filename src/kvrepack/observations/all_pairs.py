"""Linear-memory pool cosine observations alongside the unchanged adjacent probe.

For a pool of T normalized vectors, the ordered off-diagonal mean is
``(||sum_i u_i||**2 - T) / (T * (T - 1))``. This module never constructs
a T-by-T similarity matrix. Numerically we subtract the actual self-dot sum
instead of T, removing FP32 normalization roundoff from the excluded diagonal.
Metrics describe each native KV head separately.
Torch and Transformers are not imported until tensor observations execute.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .hooks import AdjacentObserver

_METRICS = ("k_raw", "v", "k_rope")


def _reduce_cosines(value):
    """Reduce [H,T,D] or [H,R,T,D], retaining only its leading axes."""
    import torch

    if not torch.is_tensor(value) or value.ndim not in (3, 4) or not value.is_floating_point():
        raise ValueError("All-pairs values must be a floating [heads,(draws),tokens,dimension] tensor")
    heads, tokens, dimension = value.shape[0], value.shape[-2], value.shape[-1]
    if heads <= 0 or dimension <= 0:
        raise ValueError("All-pairs requires positive native head count and head dimension")
    result = {
        "token_count": int(tokens), "pair_count": int(tokens * (tokens - 1)),
        "status": "supported" if tokens >= 2 else "unsupported_t_lt_2",
        "off_diagonal": None, "inclusive": None, "diagonal": None,
    }
    if tokens == 0:
        return result
    with torch.no_grad():
        selected = value.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(selected).all().item()):
            raise ValueError("All-pairs pool contains nonfinite selected vectors")
        norms = torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
        if not bool(torch.isfinite(norms).all().item()) or bool((norms == 0).any().item()):
            raise ValueError("All-pairs pool contains zero or nonfinite norms; cosine undefined")
        normalized = selected / norms
        # Only [H,D] and [H] reductions survive: no quadratic pair materialization.
        summed = normalized.sum(dim=-2, dtype=torch.float64)
        squared_sum = summed.square().sum(dim=-1, dtype=torch.float64)
        # Actual self-dot mean records small FP32 normalization roundoff.
        diagonal_sum = torch.linalg.vector_norm(normalized, dim=-1, dtype=torch.float64).square().sum(dim=-1, dtype=torch.float64)
        result["diagonal"] = (diagonal_sum / tokens).clamp(-1.0, 1.0)
        result["inclusive"] = (squared_sum / (tokens * tokens)).clamp(-1.0, 1.0)
        if tokens >= 2:
            result["off_diagonal"] = ((squared_sum - diagonal_sum) / (tokens * (tokens - 1))).clamp(-1.0, 1.0)
        if not all(bool(torch.isfinite(tensor).all().item()) for tensor in (summed, squared_sum, diagonal_sum)):
            raise ValueError("All-pairs reduction produced nonfinite values")
    return result


def all_pairs_cosine_statistics(value):
    """Summarize [H,T,D] using FP32 normalization and FP64 sums/norms.

    Off-diagonal excludes self-pairs and includes both ordered directions.
    ``inclusive`` and ``diagonal`` are explicit supplementary statistics.
    Empty/singleton pools return unsupported off-diagonal data as None, never 0.
    FP32 normalization has rounding error; output cosine bounds are clipped
    only at -1/+1. Negative cosine values otherwise remain unchanged.
    """
    if getattr(value, "ndim", None) != 3:
        raise ValueError("All-pairs values must be a floating [heads,tokens,dimension] tensor")
    result = _reduce_cosines(value)
    for field in ("off_diagonal", "inclusive", "diagonal"):
        if result[field] is not None:
            result[field] = result[field].cpu().tolist()
    return result


class AllPairsObserver(AdjacentObserver):
    """Observe predefined pools without modifying adjacent statistics/lifecycle.

    ``pools`` maps any nonempty pool name to unique native sequence positions.
    Pools may overlap, arrive unsorted, and be empty. Selection must be supplied
    before observing activations; this class never selects based on outcomes.

    Each returned layer has ``all_pairs[pool]`` with token_count, pair_count,
    status, per-head k_raw/v/k_rope off-diagonal arrays (or None), supplementary
    inclusive/diagonal dictionaries, and independent metric capture counts.
    """

    def __init__(self, model, pair_plan, pools, *, matched_pools=None, rotary_module=None, include_pair_values=False):
        super().__init__(model, pair_plan, rotary_module=rotary_module, include_pair_values=include_pair_values)
        if not isinstance(pools, Mapping) or not pools:
            raise ValueError("pools must be a nonempty mapping of names to native token positions")
        self._all_pools = {}
        for name, positions in pools.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Pool names must be nonempty strings")
            if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
                raise ValueError(f"Pool {name!r} requires a sequence of integer positions")
            copied = tuple(positions)
            if any(type(position) is not int or not 0 <= position < self._sequence_length for position in copied):
                raise ValueError(f"Pool {name!r} contains a noninteger or out-of-range native position")
            if len(copied) != len(set(copied)):
                raise ValueError(f"Pool {name!r} contains duplicate positions")
            self._all_pools[name] = copied
        self._matched_pools = {}
        shape = None
        if matched_pools is not None:
            if not isinstance(matched_pools, Mapping):
                raise ValueError("matched_pools must map names to equal-count draws")
            for name, draws in matched_pools.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Matched pool names must be nonempty strings")
                if not isinstance(draws, Sequence) or isinstance(draws, (str, bytes)) or not draws:
                    raise ValueError(f"Matched pool {name!r} requires at least one draw")
                copied_draws = []
                for positions in draws:
                    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
                        raise ValueError(f"Matched pool {name!r} requires position sequences")
                    copied = tuple(positions)
                    if any(type(position) is not int or not 0 <= position < self._sequence_length for position in copied):
                        raise ValueError(f"Matched pool {name!r} contains a noninteger or out-of-range native position")
                    if copied != tuple(sorted(set(copied))):
                        raise ValueError(f"Matched pool {name!r} draws must have sorted unique positions")
                    copied_draws.append(copied)
                current_shape = (len(copied_draws), len(copied_draws[0]))
                if any(len(draw) != current_shape[1] for draw in copied_draws) or (shape is not None and shape != current_shape):
                    raise ValueError("All matched groups/draws must have identical draw_count and token_count")
                shape = current_shape
                self._matched_pools[name] = tuple(copied_draws)
        for layer in self._stats:
            layer["all_pairs"] = {
                name: {"token_count": len(positions), "pair_count": len(positions) * (len(positions) - 1),
                       "status": "supported" if len(positions) >= 2 else "unsupported_t_lt_2",
                       **{metric: None for metric in _METRICS},
                       "inclusive": {metric: None for metric in _METRICS},
                       "diagonal": {metric: None for metric in _METRICS},
                       "capture_counts": {metric: 0 for metric in _METRICS}}
                for name, positions in self._all_pools.items()
            }
            if self._matched_pools:
                layer["matched_all_pairs"] = {
                    name: {"token_count": len(draws[0]), "draw_count": len(draws),
                           "pair_count_per_draw": len(draws[0]) * (len(draws[0]) - 1),
                           "status": "supported" if len(draws[0]) >= 2 else "unsupported_t_lt_2",
                           **{metric: None for metric in _METRICS},
                           "draw_values": {metric: None for metric in _METRICS},
                           "capture_counts": {metric: 0 for metric in _METRICS}}
                    for name, draws in self._matched_pools.items()
                }

    def _capture(self, layer, metric, value):
        # Keep the real adjacent observer's shape/order/dtype checks and values.
        super()._capture(layer, metric, value)
        torch = self._torch
        stats = self._stats[layer]
        heads, dimension = stats["kv_heads"], stats["head_dim"]
        for name, positions in self._all_pools.items():
            pool = stats["all_pairs"][name]
            if pool["capture_counts"][metric]:
                raise ValueError(f"Layer {layer}/{name}: repeated all-pairs {metric} capture")
            indices = torch.tensor(positions, dtype=torch.long, device=value.device)
            if metric == "k_rope":
                selected = value.detach()[0].index_select(1, indices)
            else:
                selected = value.detach()[0].index_select(0, indices).reshape(len(positions), heads, dimension).transpose(0, 1)
            result = all_pairs_cosine_statistics(selected)
            pool[metric] = result["off_diagonal"]
            pool["inclusive"][metric] = result["inclusive"]
            pool["diagonal"][metric] = result["diagonal"]
            pool["capture_counts"][metric] += 1
        for name, draws in self._matched_pools.items():
            pool = stats["matched_all_pairs"][name]
            if pool["capture_counts"][metric]:
                raise ValueError(f"Layer {layer}/{name}: repeated matched all-pairs {metric} capture")
            count, tokens = len(draws), len(draws[0])
            indices = torch.tensor([position for draw in draws for position in draw], dtype=torch.long, device=value.device)
            if metric == "k_rope":
                selected = value.detach()[0].index_select(1, indices).reshape(heads, count, tokens, dimension)
            else:
                selected = value.detach()[0].index_select(0, indices).reshape(count, tokens, heads, dimension).permute(2, 0, 1, 3)
            result = _reduce_cosines(selected)
            if result["off_diagonal"] is not None:
                # One small R-by-H transfer, not one synchronization per draw.
                values = result["off_diagonal"].transpose(0, 1).cpu().tolist()
                pool["draw_values"][metric] = values
                pool[metric] = [math.fsum(draw[head] for draw in values) / count for head in range(heads)]
            pool["capture_counts"][metric] += 1

    def finish(self):
        result = super().finish()
        for layer in result:
            heads = layer["kv_heads"]
            for name, positions in self._all_pools.items():
                pool = layer["all_pairs"][name]
                if pool["token_count"] != len(positions) or pool["pair_count"] != len(positions) * (len(positions) - 1):
                    raise ValueError(f"Layer {layer['layer']}/{name}: pool count mismatch")
                if pool["capture_counts"] != {metric: 1 for metric in _METRICS}:
                    raise ValueError(f"Layer {layer['layer']}/{name}: missing all-pairs metric capture")
                for metric in _METRICS:
                    value = pool[metric]
                    if (len(positions) < 2 and value is not None) or (len(positions) >= 2 and (not isinstance(value, list) or len(value) != heads)):
                        raise ValueError(f"Layer {layer['layer']}/{name}: invalid off-diagonal head statistics")
            for name, draws in self._matched_pools.items():
                pool = layer["matched_all_pairs"][name]
                count, tokens = len(draws), len(draws[0])
                if pool["token_count"] != tokens or pool["draw_count"] != count or pool["pair_count_per_draw"] != tokens * (tokens - 1):
                    raise ValueError(f"Layer {layer['layer']}/{name}: matched pool count mismatch")
                if pool["capture_counts"] != {metric: 1 for metric in _METRICS}:
                    raise ValueError(f"Layer {layer['layer']}/{name}: missing matched all-pairs metric capture")
                for metric in _METRICS:
                    values, means = pool["draw_values"][metric], pool[metric]
                    if tokens < 2:
                        if values is not None or means is not None:
                            raise ValueError("Unsupported matched pools cannot report fabricated means")
                    elif (not isinstance(means, list) or len(means) != heads or not isinstance(values, list)
                          or len(values) != count or any(not isinstance(draw, list) or len(draw) != heads for draw in values)):
                        raise ValueError(f"Layer {layer['layer']}/{name}: invalid matched draw/head statistics")
        return result

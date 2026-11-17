"""Observe every consecutive sequence pair at native Qwen K/V capture points."""
from __future__ import annotations

import copy
import math

from .native_all_pairs import NativeAllPairsObserver

STATISTICS_KIND = 'all_sequence_adjacent_v1'
METRICS = ('k_raw', 'v', 'k_rope')


def adjacent_cosine_statistics(value):
    """Mean cos(x[i], x[i+1]) for every i, independently for each native head.

    There is no token sampling, modality filtering, or row-boundary removal.
    Input is [H,T,D]; FP32 normalization and FP64 products/reductions retain
    signed similarities. The result has exactly T-1 pairs per head.
    """
    import torch

    if (not torch.is_tensor(value) or value.ndim != 3 or not value.is_floating_point()
            or min(value.shape) <= 0 or value.shape[1] < 2):
        raise ValueError('Require floating [heads,tokens,dimension] with at least two tokens')
    with torch.no_grad():
        vectors = value.detach().float()
        if not bool(torch.isfinite(vectors).all().item()):
            raise ValueError('Nonfinite vector makes cosine undefined')
        norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        if bool((norms == 0).any().item()) or not bool(torch.isfinite(norms).all().item()):
            raise ValueError('Zero or nonfinite vector norm makes cosine undefined')
        unit = (vectors / norms).double()
        cosines = (unit[:, :-1, :] * unit[:, 1:, :]).sum(dim=-1).clamp(-1, 1)
        return dict(status='supported', token_count=int(value.shape[1]),
                    pair_count=int(value.shape[1]-1), mean=cosines.mean(dim=-1))


class NativeAdjacentObserver(NativeAllPairsObserver):
    """Reuse native projection/RoPE hooks, replacing only their scalar reduction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for row in self._stats:
            del row['inclusive']
            row['adjacent_pair_count'] = self.input_length - 1

    def _capture(self, index, metric, value):
        torch = self._torch
        if not self._active or index != self._current_layer:
            raise ValueError('Native capture occurred outside its attention layer')
        row = self._stats[index]
        if metric not in METRICS or row['capture_counts'][metric]:
            raise ValueError('Unknown or repeated native metric capture')
        heads, dimension = row['kv_heads'], row['head_dim']
        expected = ((1, heads, self.input_length, dimension) if metric == 'k_rope'
                    else (1, self.input_length, heads * dimension))
        if not torch.is_tensor(value) or tuple(value.shape) != expected or not value.is_floating_point():
            raise ValueError(f'Expected floating {expected}, before GQA repetition')
        if index in self._dtypes and value.dtype != self._dtypes[index]:
            raise ValueError('Native K/V/RoPE capture dtype mismatch')
        if index in self._devices and value.device != self._devices[index]:
            raise ValueError('Native K/V/RoPE capture device mismatch')
        self._dtypes[index], self._devices[index] = value.dtype, value.device
        vectors = value.detach()[0]
        if metric != 'k_rope':
            vectors = vectors.reshape(self.input_length, heads, dimension).transpose(0, 1)
        result = adjacent_cosine_statistics(vectors)
        if result['pair_count'] != self.input_length - 1:
            raise ValueError('Every consecutive sequence pair must be observed')
        row[metric] = result['mean'].tolist()
        row['capture_counts'][metric] += 1

    def finish(self):
        if not self._entered or self._failure is not None:
            raise RuntimeError(f'Incomplete native adjacent observer: {self._failure}')
        if self._order != list(range(len(self._stats))):
            raise ValueError('Native observation is missing language layers')
        for row in self._stats:
            if row['capture_counts'] != {metric: 1 for metric in METRICS}:
                raise ValueError('Missing or repeated native metric capture')
            if row['adjacent_pair_count'] != self.input_length - 1:
                raise ValueError('Wrong sequence-adjacent pair count')
            for metric in METRICS:
                values = row[metric]
                if (not isinstance(values, list) or len(values) != row['kv_heads']
                        or any(not isinstance(x, (int, float)) or not math.isfinite(x) or not -1 <= x <= 1
                               for x in values)):
                    raise ValueError('Invalid native per-head adjacent cosine statistic')
        return copy.deepcopy(self._stats)

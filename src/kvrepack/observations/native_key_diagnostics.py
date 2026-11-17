"""Read-only raw-K geometry diagnostics at the native projection capture point.

Centering and bias subtraction act on captured FP32 copies only. They do not
change the model, its attention/cache, or the original projection output. In
particular, subtracting bias from a rounded BF16 output is a geometric probe,
not a new model forward with projection bias disabled.
"""
from __future__ import annotations

import math

from .all_pairs import all_pairs_cosine_statistics
from .native_adjacent import NativeAdjacentObserver, adjacent_cosine_statistics


DIAGNOSTIC_METRICS = ('k_raw_all_pairs', 'k_raw_centered_adjacent', 'k_raw_without_bias_adjacent')


def key_diagnostic_statistics(value, bias=None):
    """Diagnose native [H,T,D] keys; optional projection bias is [H,D].

    The complete current sequence supplies the per-head FP32 mean. Every
    transformed sequence is normalized independently in FP32, then reduced in
    FP64 using the existing exact adjacency/non-self all-pairs definitions.
    Undefined cosines (zero or nonfinite norm) are errors, never fabricated0.
    """
    import torch

    if (not torch.is_tensor(value) or value.ndim != 3 or not value.is_floating_point()
            or min(value.shape) <= 0 or value.shape[1] < 2):
        raise ValueError('Key diagnostics require floating [heads,tokens,dimension] and at least two tokens')
    with torch.no_grad():
        vectors = value.detach().float()
        pairs = all_pairs_cosine_statistics(vectors)
        if pairs['status'] != 'supported':
            raise ValueError('Raw-K all-pairs diagnostics require a complete non-singleton sequence')
        sequence_mean = vectors.mean(dim=1, keepdim=True, dtype=torch.float32)
        centered = vectors-sequence_mean
        centered_adjacent = adjacent_cosine_statistics(centered)['mean'].tolist()
        if bias is None:
            without_bias = adjacent_cosine_statistics(vectors)['mean'].tolist()
            bias_norm = [0.]*vectors.shape[0]
        else:
            if (not torch.is_tensor(bias) or not bias.is_floating_point()
                    or tuple(bias.shape) != (vectors.shape[0], vectors.shape[2])):
                raise ValueError('Key projection bias must be a floating [heads,dimension] tensor')
            projection_bias = bias.detach().to(device=vectors.device, dtype=torch.float32)
            if not bool(torch.isfinite(projection_bias).all().item()):
                raise ValueError('Nonfinite key projection bias makes diagnostic cosine undefined')
            without_bias = adjacent_cosine_statistics(vectors-projection_bias[:, None, :])['mean'].tolist()
            bias_norm = torch.linalg.vector_norm(projection_bias, dim=-1).tolist()
        return dict(k_raw_all_pairs=pairs['off_diagonal'], k_raw_centered_adjacent=centered_adjacent,
                    k_raw_without_bias_adjacent=without_bias, k_proj_bias_present=bias is not None,
                    k_proj_bias_norm=bias_norm,
                    k_raw_sequence_mean_norm=torch.linalg.vector_norm(sequence_mean[:, 0, :], dim=-1).tolist())


class NativeKeyDiagnosticsObserver(NativeAdjacentObserver):
    """Keep original raw-K/RoPE-K/V adjacency; add post-hoc raw-K diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for row in self._stats:
            row.update(key_diagnostic_capture_count=0, k_proj_bias_present=None,
                       k_proj_bias_norm=None, k_raw_sequence_mean_norm=None,
                       **{name: None for name in DIAGNOSTIC_METRICS})

    def _capture(self, index, metric, value):
        super()._capture(index, metric, value)
        if metric != 'k_raw':
            return
        row = self._stats[index]
        if row['key_diagnostic_capture_count'] != 0:
            raise ValueError('Raw-K diagnostics must be captured exactly once per layer')
        heads, dimension = row['kv_heads'], row['head_dim']
        vectors = value.detach()[0].reshape(self.input_length, heads, dimension).transpose(0, 1)
        bias = getattr(self._attentions[index].k_proj, 'bias', None)
        if bias is not None:
            if tuple(bias.shape) != (heads*dimension,):
                raise ValueError('Native key projection bias width differs from native KV heads')
            bias = bias.reshape(heads, dimension)
        diagnostics = key_diagnostic_statistics(vectors, bias=bias)
        if bias is None:
            # No-bias equality is exact, using the already captured native mean.
            diagnostics['k_raw_without_bias_adjacent'] = list(row['k_raw'])
        row.update(diagnostics)
        row['key_diagnostic_capture_count'] += 1

    def finish(self):
        result = super().finish()
        for row in result:
            if type(row['key_diagnostic_capture_count']) is not int or row['key_diagnostic_capture_count'] != 1:
                raise ValueError('Missing or repeated raw-K diagnostic capture')
            for name in DIAGNOSTIC_METRICS:
                values = row[name]
                if (not isinstance(values, list) or len(values) != row['kv_heads']
                        or any(type(x) not in (int, float) or not math.isfinite(x) or not -1 <= x <= 1 for x in values)):
                    raise ValueError('Invalid per-native-head raw-K diagnostic cosine')
            if type(row['k_proj_bias_present']) is not bool:
                raise ValueError('Projection bias presence must be explicit')
            for name in ('k_proj_bias_norm', 'k_raw_sequence_mean_norm'):
                values = row[name]
                if (not isinstance(values, list) or len(values) != row['kv_heads']
                        or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values)):
                    raise ValueError('Invalid per-native-head diagnostic vector norm')
        return result

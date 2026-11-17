"""Exact highest-cosine raw-K adjacent pairs at native Qwen capture points.

Selection is local to one input, layer and original KV head. Every original
sequence pair (i,i+1) is eligible. No pair is copied to fill short sequences.
"""
from __future__ import annotations

import math

from .native_adjacent import NativeAdjacentObserver

STATISTICS_KIND = 'raw_k_top_sequence_adjacent_v1'


def _positive_top_k(top_k):
    if type(top_k) is not int or top_k <= 0:
        raise ValueError('top_k must be a positive integer')
    return top_k


def top_adjacent_cosines(value, top_k=200):
    """Rank all N-1 adjacent raw-K cosines in each [H,N,D] input head.

    FP32 normalization and FP64 products/reductions match the full-adjacency
    observer. Scores are sorted descending; exact ties retain the ascending
    original pair-start index. There must be at least top_k distinct pairs.
    """
    import torch

    top_k = _positive_top_k(top_k)
    if (not torch.is_tensor(value) or value.ndim != 3 or not value.is_floating_point()
            or min(value.shape) <= 0 or value.shape[1] < 2):
        raise ValueError('Require floating [heads,tokens,dimension] with at least two tokens')
    tokens = int(value.shape[1])
    if tokens - 1 < top_k:
        raise ValueError(f'{tokens-1} adjacent pairs are insufficient for exactly {top_k}')
    with torch.no_grad():
        vectors = value.detach().float()
        if not bool(torch.isfinite(vectors).all().item()):
            raise ValueError('Nonfinite vector makes cosine undefined')
        norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        if bool((norms == 0).any().item()) or not bool(torch.isfinite(norms).all().item()):
            raise ValueError('Zero or nonfinite vector norm makes cosine undefined')
        unit = (vectors / norms).double()
        cosines = (unit[:, :-1, :] * unit[:, 1:, :]).sum(dim=-1).clamp(-1, 1)
        # Stable sort defines the tie policy independently of topk kernel order.
        starts = torch.argsort(cosines, dim=-1, descending=True, stable=True)[:, :top_k]
        scores = torch.gather(cosines, dim=-1, index=starts)
        return dict(status='supported', token_count=tokens, pair_count=tokens-1, top_k=top_k,
                    mean=scores.mean(dim=-1), minimum=scores[:, -1],
                    pair_starts=starts, scores=scores)


class RawKTopAdjacentObserver(NativeAdjacentObserver):
    """Keep full-adjacency K/V statistics and additionally rank pre-RoPE K.

    The unchanged inherited hooks observe k_proj before RoPE, the native rotary
    function's returned K, and v_proj before GQA repetition. Top-k selection is
    applied only to raw K; full-adjacency K/V values retain their own semantics.
    """

    def __init__(self, *args, top_k=200, **kwargs):
        self.top_k = _positive_top_k(top_k)
        super().__init__(*args, **kwargs)
        if self.input_length - 1 < self.top_k:
            raise ValueError('Input has fewer adjacent pairs than the required exact top_k')
        self._topk_capture_counts = [0] * len(self._stats)
        for row in self._stats:
            row.update(top_k=self.top_k, raw_k_topk_mean=None, raw_k_topk_min=None,
                       raw_k_topk_pair_starts=None, raw_k_topk_scores=None)

    def _capture(self, index, metric, value):
        # Parent enforces native tensor shape, dtype/device and capture counts,
        # and records the unchanged full-sequence adjacent K/V metrics.
        super()._capture(index, metric, value)
        if metric != 'k_raw':
            return
        if self._topk_capture_counts[index]:
            raise ValueError('Repeated raw-K top-k capture')
        row = self._stats[index]
        vectors = value.detach()[0].reshape(self.input_length, row['kv_heads'], row['head_dim']).transpose(0, 1)
        result = top_adjacent_cosines(vectors, self.top_k)
        if result['pair_count'] != row['adjacent_pair_count']:
            raise ValueError('Top-k candidate count differs from full-sequence adjacency')
        row.update(raw_k_topk_mean=result['mean'].tolist(), raw_k_topk_min=result['minimum'].tolist(),
                   raw_k_topk_pair_starts=result['pair_starts'].tolist(),
                   raw_k_topk_scores=result['scores'].tolist())
        self._topk_capture_counts[index] += 1

    def finish(self):
        stats = super().finish()
        for index, row in enumerate(stats):
            if self._topk_capture_counts[index] != 1 or row['top_k'] != self.top_k:
                raise ValueError('Missing raw-K top-k capture or changed top_k')
            heads, pairs = row['kv_heads'], row['adjacent_pair_count']
            for name in ('raw_k_topk_mean', 'raw_k_topk_min', 'raw_k_topk_pair_starts', 'raw_k_topk_scores'):
                if not isinstance(row[name], list) or len(row[name]) != heads:
                    raise ValueError('Raw-K top-k result must cover every original KV head')
            for head in range(heads):
                mean, minimum = row['raw_k_topk_mean'][head], row['raw_k_topk_min'][head]
                starts, scores = row['raw_k_topk_pair_starts'][head], row['raw_k_topk_scores'][head]
                if (not isinstance(starts, list) or not isinstance(scores, list)
                        or len(starts) != self.top_k or len(scores) != self.top_k
                        or any(type(p) is not int or not 0 <= p < pairs for p in starts)
                        or len(set(starts)) != self.top_k
                        or any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1
                               for v in [mean, minimum, *scores])):
                    raise ValueError('Invalid exact top-k positions or signed cosine scores')
                if any(scores[i] < scores[i+1] or (scores[i] == scores[i+1] and starts[i] > starts[i+1])
                       for i in range(self.top_k-1)):
                    raise ValueError('Top-k order must be descending score, then ascending pair index')
                if (minimum != scores[-1] or not math.isclose(mean, math.fsum(scores)/self.top_k,
                                                            rel_tol=0, abs_tol=1e-12)
                        or mean < row['k_raw'][head]-1e-12):
                    raise ValueError('Top-k mean/minimum disagree with selected adjacent-pair scores')
        return stats

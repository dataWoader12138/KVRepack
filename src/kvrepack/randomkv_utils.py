from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


def repeat_kv_len(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(batch, num_key_value_heads, n_rep, slen)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen)


def repeat_kv_map(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, tk, tv = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, tk, tv)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, tk, tv)


class PromptCompressionStats:
    def __init__(self) -> None:
        self.total_before = 0
        self.total_after = 0

    def reset(self):
        self.total_before = 0
        self.total_after = 0

    def update(self, seq_len: int, tk_after: int, tv_after: int):
        self.total_before += 2 * int(seq_len)
        self.total_after += int(tk_after) + int(tv_after)

    def summary_ratio(self) -> float:
        if self.total_before == 0:
            return 1.0
        return float(self.total_after) / float(self.total_before)


class RandomKVCache(DynamicCache):
    """
    key_cache[layer]:    [B, H_kv, Tk, D]
    value_cache[layer]:  [B, H_kv, Tv, D]

    k_len_cache[layer]:  [B, H_kv, Tk]
    v_len_cache[layer]:  [B, H_kv, Tv]

    k_to_v_map[layer]:   [B, H_kv, Tk, Tv]
    k_anchor_pos[layer]: [B, H_kv, Tk]
    """

    def __init__(self) -> None:
        super().__init__()
        self.k_len_cache: List[Optional[torch.Tensor]] = []
        self.v_len_cache: List[Optional[torch.Tensor]] = []
        self.k_to_v_map: List[Optional[torch.Tensor]] = []
        self.k_anchor_pos: List[Optional[torch.Tensor]] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def set_layer_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        k_len_states: torch.Tensor,
        v_len_states: torch.Tensor,
        k_to_v_map: torch.Tensor,
        k_anchor_pos: torch.Tensor,
    ) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.k_len_cache.append(None)
            self.v_len_cache.append(None)
            self.k_to_v_map.append(None)
            self.k_anchor_pos.append(None)

        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        self.k_len_cache[layer_idx] = k_len_states
        self.v_len_cache[layer_idx] = v_len_states
        self.k_to_v_map[layer_idx] = k_to_v_map
        self.k_anchor_pos[layer_idx] = k_anchor_pos

    def update_singletons(
        self,
        key_states: torch.Tensor,     # [B, H_kv, 1, D]
        value_states: torch.Tensor,   # [B, H_kv, 1, D]
        layer_idx: int,
        cache_position: torch.Tensor,
    ):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.k_len_cache.append(None)
            self.v_len_cache.append(None)
            self.k_to_v_map.append(None)
            self.k_anchor_pos.append(None)

        bsz, h_kv, _, _ = key_states.shape
        one_k = torch.ones((bsz, h_kv, 1), device=key_states.device, dtype=key_states.dtype)
        one_v = torch.ones((bsz, h_kv, 1), device=value_states.device, dtype=value_states.dtype)

        if cache_position.ndim == 1:
            anchor = cache_position.view(1, 1, 1).expand(bsz, h_kv, 1).to(torch.long)
        else:
            anchor = cache_position.view(bsz, 1, 1).expand(bsz, h_kv, 1).to(torch.long)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.k_len_cache[layer_idx] = one_k
            self.v_len_cache[layer_idx] = one_v
            self.k_to_v_map[layer_idx] = torch.ones((bsz, h_kv, 1, 1), device=key_states.device, dtype=key_states.dtype)
            self.k_anchor_pos[layer_idx] = anchor
        else:
            old_k = self.key_cache[layer_idx]
            old_v = self.value_cache[layer_idx]
            old_k_len = self.k_len_cache[layer_idx]
            old_v_len = self.v_len_cache[layer_idx]
            old_map = self.k_to_v_map[layer_idx]
            old_anchor = self.k_anchor_pos[layer_idx]

            self.key_cache[layer_idx] = torch.cat([old_k, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([old_v, value_states], dim=-2)
            self.k_len_cache[layer_idx] = torch.cat([old_k_len, one_k], dim=-1)
            self.v_len_cache[layer_idx] = torch.cat([old_v_len, one_v], dim=-1)
            self.k_anchor_pos[layer_idx] = torch.cat([old_anchor, anchor], dim=-1)

            b, h, tk_old, tv_old = old_map.shape
            new_map = torch.zeros((b, h, tk_old + 1, tv_old + 1), device=old_map.device, dtype=old_map.dtype)
            new_map[:, :, :tk_old, :tv_old] = old_map
            new_map[:, :, tk_old, tv_old] = 1.0
            self.k_to_v_map[layer_idx] = new_map

        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            self.k_len_cache[layer_idx],
            self.v_len_cache[layer_idx],
            self.k_to_v_map[layer_idx],
        )

    @classmethod
    def from_legacy_cache(cls, past_key_values=None) -> "RandomKVCache":
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                layer = past_key_values[layer_idx]
                if len(layer) >= 6:
                    k, v, k_len, v_len, kv_map, k_anchor = layer[:6]
                    cache.set_layer_cache(layer_idx, k, v, k_len, v_len, kv_map, k_anchor)
                else:
                    k, v = layer[:2]
                    bsz, h, t, _ = k.shape
                    ones = torch.ones((bsz, h, t), device=k.device, dtype=k.dtype)
                    eye = torch.eye(t, device=k.device, dtype=k.dtype).view(1, 1, t, t).expand(bsz, h, t, t)
                    anchor = torch.arange(t, device=k.device).view(1, 1, t).expand(bsz, h, t)
                    cache.set_layer_cache(layer_idx, k, v, ones, ones, eye, anchor)
        return cache

    def to_legacy_cache(self):
        legacy = ()
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is None:
                continue
            legacy += (
                (
                    self.key_cache[layer_idx],
                    self.value_cache[layer_idx],
                    self.k_len_cache[layer_idx],
                    self.v_len_cache[layer_idx],
                    self.k_to_v_map[layer_idx],
                    self.k_anchor_pos[layer_idx],
                ),
            )
        return legacy


def sample_keep_indices(
    seq_len: int,
    keep_total: int,
    prefix_len: int,
    suffix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    随机保留索引：
    - prefix / suffix 必保留
    - 中间随机采样
    """
    keep_total = max(1, min(keep_total, seq_len))

    prefix = torch.arange(prefix_len, device=device, dtype=torch.long)
    suffix_start = max(seq_len - suffix_len, prefix_len)
    suffix = torch.arange(suffix_start, seq_len, device=device, dtype=torch.long)

    protected = torch.unique(torch.cat([prefix, suffix], dim=0), sorted=True)
    protected_count = protected.numel()

    if keep_total <= protected_count:
        return protected[:keep_total]

    mid_start = prefix_len
    mid_end = max(seq_len - suffix_len, prefix_len)

    if mid_end <= mid_start:
        return protected[:keep_total]

    mid_indices = torch.arange(mid_start, mid_end, device=device, dtype=torch.long)
    need_mid = min(keep_total - protected_count, mid_indices.numel())

    if need_mid == mid_indices.numel():
        chosen_mid = mid_indices
    else:
        perm = torch.randperm(mid_indices.numel(), device=device)
        chosen_mid = mid_indices[perm[:need_mid]]
        chosen_mid, _ = torch.sort(chosen_mid)

    keep_idx = torch.cat([protected, chosen_mid], dim=0)
    keep_idx = torch.unique(keep_idx, sorted=True)

    if keep_idx.numel() > keep_total:
        keep_idx = keep_idx[:keep_total]

    return keep_idx


def build_random_k_to_v_map(
    keep_k_idx: torch.Tensor,   # [Tk]
    keep_v_idx: torch.Tensor,   # [Tv]
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    高效 one-hot map：
    每个 kept K 映射到“最近的不大于它的 kept V”，若不存在则映射到最左边的 kept V。
    这样：
    - 完全 decoupled
    - 不需要额外复杂计算
    - map 极稀疏
    """
    tk = keep_k_idx.numel()
    tv = keep_v_idx.numel()

    if tv == 0:
        raise ValueError("keep_v_idx cannot be empty")

    pos = torch.bucketize(keep_k_idx, keep_v_idx, right=True) - 1
    pos = pos.clamp(min=0, max=tv - 1)

    m = torch.zeros((tk, tv), device=keep_k_idx.device, dtype=dtype)
    m[torch.arange(tk, device=keep_k_idx.device), pos] = 1.0
    return m


class RandomDecoupledKV:
    """
    纯随机、解耦、极简高效版本

    - K 独立随机保留
    - V 独立随机保留
    - 不做任何相似度计算
    - 不做 importance 计算
    - 每个 prompt 只输出 overall_kv_ratio
    """

    def __init__(
        self,
        start_size: int = 4,
        recent_size: int = 64,
        cache_size: int = 512,
        enabled: bool = True,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = cache_size
        self.enabled = enabled
        self.prompt_stats = PromptCompressionStats()

    def set_cache_size(self, cache_size: int) -> None:
        self.cache_size = int(cache_size)

    def begin_prompt(self):
        self.prompt_stats.reset()

    def end_prompt_ratio(self) -> float:
        return self.prompt_stats.summary_ratio()

    def _protected(self, seq_len: int) -> Tuple[int, int]:
        prefix_len = min(self.start_size, seq_len)
        suffix_len = min(self.recent_size, max(seq_len - prefix_len, 0))
        return prefix_len, suffix_len

    def compress(
        self,
        key_states: torch.Tensor,     # [B, H_kv, T, D]
        value_states: torch.Tensor,   # [B, H_kv, T, D]
    ):
        if not self.enabled:
            bsz, h, t, _ = key_states.shape
            ones = torch.ones((bsz, h, t), device=key_states.device, dtype=key_states.dtype)
            eye = torch.eye(t, device=key_states.device, dtype=key_states.dtype).view(1, 1, t, t).expand(bsz, h, t, t)
            anchor = torch.arange(t, device=key_states.device).view(1, 1, t).expand(bsz, h, t)
            return key_states, value_states, ones, ones, eye, anchor

        bsz, h_kv, seq_len, _ = key_states.shape
        prefix_len, suffix_len = self._protected(seq_len)

        keep_total = max(self.start_size + self.recent_size + 1, min(self.cache_size, seq_len))

        all_k = []
        all_v = []
        all_k_len = []
        all_v_len = []
        all_map = []
        all_anchor = []

        for b in range(bsz):
            k_heads = []
            v_heads = []
            klen_heads = []
            vlen_heads = []
            map_heads = []
            anchor_heads = []

            for h in range(h_kv):
                keep_k_idx = sample_keep_indices(
                    seq_len=seq_len,
                    keep_total=keep_total,
                    prefix_len=prefix_len,
                    suffix_len=suffix_len,
                    device=key_states.device,
                )
                keep_v_idx = sample_keep_indices(
                    seq_len=seq_len,
                    keep_total=keep_total,
                    prefix_len=prefix_len,
                    suffix_len=suffix_len,
                    device=value_states.device,
                )

                k_sel = key_states[b, h, keep_k_idx, :]
                v_sel = value_states[b, h, keep_v_idx, :]

                tk_after = k_sel.shape[0]
                tv_after = v_sel.shape[0]
                self.prompt_stats.update(seq_len, tk_after, tv_after)

                klen = torch.ones((tk_after,), device=key_states.device, dtype=key_states.dtype)
                vlen = torch.ones((tv_after,), device=value_states.device, dtype=value_states.dtype)
                kv_map = build_random_k_to_v_map(keep_k_idx, keep_v_idx, dtype=key_states.dtype)
                anchor = keep_k_idx.to(torch.long)

                k_heads.append(k_sel)
                v_heads.append(v_sel)
                klen_heads.append(klen)
                vlen_heads.append(vlen)
                map_heads.append(kv_map)
                anchor_heads.append(anchor)

            tk_max = max(x.shape[0] for x in k_heads)
            tv_max = max(x.shape[0] for x in v_heads)
            dim = k_heads[0].shape[-1]

            k_pad_heads = []
            v_pad_heads = []
            klen_pad_heads = []
            vlen_pad_heads = []
            map_pad_heads = []
            anchor_pad_heads = []

            for h in range(h_kv):
                k_h = k_heads[h]
                v_h = v_heads[h]
                klen_h = klen_heads[h]
                vlen_h = vlen_heads[h]
                map_h = map_heads[h]
                anchor_h = anchor_heads[h]

                tk = k_h.shape[0]
                tv = v_h.shape[0]

                if tk < tk_max:
                    pad = torch.zeros((tk_max - tk, dim), device=k_h.device, dtype=k_h.dtype)
                    k_h = torch.cat([k_h, pad], dim=0)
                    klen_h = torch.cat([klen_h, torch.zeros((tk_max - tk,), device=klen_h.device, dtype=klen_h.dtype)], dim=0)
                    anchor_h = torch.cat([anchor_h, torch.full((tk_max - tk,), -1, device=anchor_h.device, dtype=anchor_h.dtype)], dim=0)

                if tv < tv_max:
                    pad = torch.zeros((tv_max - tv, dim), device=v_h.device, dtype=v_h.dtype)
                    v_h = torch.cat([v_h, pad], dim=0)
                    vlen_h = torch.cat([vlen_h, torch.zeros((tv_max - tv,), device=vlen_h.device, dtype=vlen_h.dtype)], dim=0)

                map_pad = torch.zeros((tk_max, tv_max), device=map_h.device, dtype=map_h.dtype)
                map_pad[:map_h.shape[0], :map_h.shape[1]] = map_h

                k_pad_heads.append(k_h)
                v_pad_heads.append(v_h)
                klen_pad_heads.append(klen_h)
                vlen_pad_heads.append(vlen_h)
                map_pad_heads.append(map_pad)
                anchor_pad_heads.append(anchor_h)

            all_k.append(torch.stack(k_pad_heads, dim=0))
            all_v.append(torch.stack(v_pad_heads, dim=0))
            all_k_len.append(torch.stack(klen_pad_heads, dim=0))
            all_v_len.append(torch.stack(vlen_pad_heads, dim=0))
            all_map.append(torch.stack(map_pad_heads, dim=0))
            all_anchor.append(torch.stack(anchor_pad_heads, dim=0))

        out_k = torch.stack(all_k, dim=0)
        out_v = torch.stack(all_v, dim=0)
        out_k_len = torch.stack(all_k_len, dim=0)
        out_v_len = torch.stack(all_v_len, dim=0)
        out_map = torch.stack(all_map, dim=0)
        out_anchor = torch.stack(all_anchor, dim=0)

        return out_k, out_v, out_k_len, out_v_len, out_map, out_anchor
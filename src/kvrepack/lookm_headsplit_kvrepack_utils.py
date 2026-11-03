import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def repeat_kv_len(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(batch, num_key_value_heads, n_rep, slen)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen)


class LookMKVRepackPromptStats:
    def __init__(self) -> None:
        self.total_before = 0
        self.total_after = 0

    def reset(self) -> None:
        self.total_before = 0
        self.total_after = 0

    def update(self, seq_len: int, kept_len: int) -> None:
        self.total_before += 2 * int(seq_len)
        self.total_after += 2 * int(kept_len)

    def summary_ratio(self) -> float:
        if self.total_before == 0:
            return 1.0
        return float(self.total_after) / float(self.total_before)


@torch.no_grad()
def compute_lookm_importance(
    query_states_rope: torch.Tensor,
    key_states_rope: torch.Tensor,
    text_token_mask: Optional[torch.Tensor],
    num_key_value_groups: int,
    head_dim: int,
    window_size: int = 32,
    text_priority: bool = True,
    gqa_func: str = "mean",
) -> torch.Tensor:
    """Match the project LookM score: recent-window attention sum + text boost."""
    bsz, _, q_len, _ = query_states_rope.shape
    num_kv_heads = key_states_rope.shape[1]
    window_size = min(max(1, int(window_size)), q_len)
    history_len = max(0, q_len - window_size)
    if history_len == 0:
        return torch.zeros((bsz, num_kv_heads, 0), device=key_states_rope.device, dtype=query_states_rope.dtype)

    key_rep = repeat_kv(key_states_rope, num_key_value_groups)
    scores = torch.matmul(query_states_rope[..., -window_size:, :], key_rep.transpose(2, 3)) / math.sqrt(head_dim)

    # Original LookM masks only the causal part inside the recent window.
    recent_mask = torch.full(
        (window_size, window_size),
        torch.finfo(scores.dtype).min,
        device=scores.device,
    )
    mask_cond = torch.arange(recent_mask.size(-1), device=scores.device)
    recent_mask.masked_fill_(mask_cond < (mask_cond + 1).view(recent_mask.size(-1), 1), 0)
    scores[:, :, -window_size:, -window_size:] += recent_mask[None, None, :, :]

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states_rope.dtype)
    importance = probs[:, :, -window_size:, :history_len].sum(dim=-2)
    importance = importance.view(bsz, num_kv_heads, num_key_value_groups, history_len)
    gqa_func = (gqa_func or "mean").lower()
    if gqa_func == "max":
        importance = importance.max(dim=2).values
    elif gqa_func == "mean":
        importance = importance.mean(dim=2)
    else:
        raise ValueError(f"Unsupported LOOKM_KVREPACK_GQA_FUNC={gqa_func}")

    if text_priority and text_token_mask is not None:
        if text_token_mask.ndim >= 2 and text_token_mask.shape[-1] >= q_len:
            text_index = text_token_mask[:, :history_len].to(device=importance.device, dtype=torch.bool)
            text_index = text_index.unsqueeze(1).expand_as(importance)
            max_score = importance.max(dim=-1).values.unsqueeze(-1).expand_as(importance)
            importance = torch.where(text_index, importance + max_score, importance)

    return torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def classify_heads_hybridkv_style(
    query_states_rope: torch.Tensor,
    key_states_rope: torch.Tensor,
    vision_token_mask: Optional[torch.Tensor],
    num_key_value_groups: int,
    head_dim: int,
    layer_idx: int = -1,
) -> torch.Tensor:
    """HybridKV-style query-aware/static head split, with LOOKM_KVREPACK env names."""
    bsz, num_kv_heads, seq_len, _ = key_states_rope.shape
    device = key_states_rope.device
    dynamic = torch.zeros((bsz, num_kv_heads), device=device, dtype=torch.bool)

    if vision_token_mask is None or vision_token_mask.ndim < 2 or vision_token_mask.shape[-1] < seq_len:
        if _env_int("LOOKM_KVREPACK_DIAG_PRINT", 0):
            mask_shape = None if vision_token_mask is None else tuple(vision_token_mask.shape)
            print(
                "[LookM-HeadSplit-KVRepack-Diag] "
                f"layer={layer_idx} reason=invalid_vision_mask seq_len={seq_len} "
                f"vision_mask_shape={mask_shape}"
            )
        return dynamic

    vision_mask = vision_token_mask[:, :seq_len].to(device=device, dtype=torch.bool)
    score_threshold = _env_float("LOOKM_KVREPACK_SCORE_THRESHOLD", _env_float("HEADSPLIT_SCORE_THRESHOLD", 0.9))
    sparsity_ratio = _env_float("LOOKM_KVREPACK_SPARSITY_RATIO", _env_float("HEADSPLIT_SPARSITY_RATIO", 0.95))
    max_text_len = _env_int("LOOKM_KVREPACK_MAX_TEXT_LEN", _env_int("HEADSPLIT_MAX_TEXT_LEN", 512))
    key_rep = repeat_kv(key_states_rope, num_key_value_groups)

    for bidx in range(bsz):
        vm = vision_mask[bidx]
        if not bool(vm.any()):
            continue
        vision_start = int(torch.argmax(vm.long()).item())
        vision_end = int(seq_len - torch.argmax(vm.long().flip(dims=[0])).item())
        vision_len = max(0, vision_end - vision_start)
        current_text_len = min(max(0, seq_len - vision_end), max_text_len)
        topk = int(vision_len * (1.0 - sparsity_ratio))
        if current_text_len <= 0 or topk <= 0:
            continue

        text_query = query_states_rope[bidx : bidx + 1, :, vision_end : vision_end + current_text_len, :]
        text_scores = torch.matmul(text_query, key_rep[bidx : bidx + 1].transpose(2, 3)) / math.sqrt(head_dim)
        attn_mask = torch.triu(
            torch.ones(current_text_len, current_text_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        text_scores[:, :, :, vision_end : vision_end + current_text_len] = text_scores[
            :, :, :, vision_end : vision_end + current_text_len
        ].masked_fill(attn_mask, float("-inf"))
        text_probs = torch.softmax(text_scores, dim=-1, dtype=torch.float32).to(query_states_rope.dtype)
        topk_indices = torch.topk(text_probs, k=topk, dim=-1).indices
        topk_score = torch.gather(text_probs, dim=-1, index=topk_indices)
        topk_score_avg = topk_score.sum(-1).mean(-1)

        query_aware = topk_score_avg < score_threshold
        query_aware_groups = query_aware.view(1, num_kv_heads, num_key_value_groups)
        dynamic[bidx] = query_aware_groups.any(dim=-1)[0]

        if _env_int("LOOKM_KVREPACK_DIAG_PRINT", 0):
            print(
                "[LookM-HeadSplit-KVRepack-Diag] "
                f"layer={layer_idx} batch={bidx} dynamic_heads={int(dynamic[bidx].sum().item())}/{num_kv_heads} "
                f"vision_start={vision_start} vision_end={vision_end} text_len={current_text_len} "
                f"topk={topk} threshold={score_threshold:.4f} sparsity={sparsity_ratio:.4f}"
            )

    return dynamic


class LookMHeadSplitKVRepack:
    def __init__(
        self,
        window_size: int = 32,
        budget_ratio: float = 0.10,
        dynamic_kvrepack_enable: bool = True,
        static_kvrepack_enable: bool = False,
        dynamic_compress_ratio: float = 0.80,
        static_compress_ratio: float = 1.00,
        alpha: float = 2.0,
        merge_layer_ratio: float = 0.05,
        respect_layer_gate: bool = True,
        drop_merge: bool = True,
        pivot_chunk: int = 1024,
        enabled: bool = True,
    ) -> None:
        self.window_size = int(window_size)
        self.budget_ratio = float(budget_ratio)
        self.dynamic_kvrepack_enable = bool(dynamic_kvrepack_enable)
        self.static_kvrepack_enable = bool(static_kvrepack_enable)
        self.dynamic_compress_ratio = float(dynamic_compress_ratio)
        self.static_compress_ratio = float(static_compress_ratio)
        self.alpha = float(alpha)
        self.merge_layer_ratio = float(merge_layer_ratio)
        self.respect_layer_gate = bool(respect_layer_gate)
        self.drop_merge = bool(drop_merge)
        self.pivot_chunk = int(pivot_chunk)
        self.enabled = bool(enabled)
        self.prompt_stats = LookMKVRepackPromptStats()

    def begin_prompt(self) -> None:
        self.prompt_stats.reset()

    def end_prompt_ratio(self) -> float:
        return self.prompt_stats.summary_ratio()

    def _history_budget(self, seq_len: int, history_len: int) -> int:
        budget = min(max(int(seq_len * self.budget_ratio), 0), int(history_len))
        if history_len > 0 and self.budget_ratio > 0 and budget == 0:
            budget = 1
        return budget

    @staticmethod
    def _candidate_len(final_len: int, history_len: int, compress_ratio: float) -> int:
        final_len = min(max(int(final_len), 0), int(history_len))
        if final_len <= 0 or history_len <= 0:
            return 0
        ratio = float(compress_ratio)
        if ratio <= 0:
            raise ValueError("LOOKM_KVREPACK_*_COMPRESS_RATIO must be > 0")
        ratio = min(ratio, 1.0)
        return min(int(history_len), max(final_len, int(math.ceil(final_len / ratio - 1e-6))))

    @staticmethod
    def _topk_sorted(score: torch.Tensor, keep: int) -> torch.Tensor:
        total = int(score.shape[0])
        keep = min(max(int(keep), 0), total)
        if keep <= 0:
            return torch.empty((0,), device=score.device, dtype=torch.long)
        if keep >= total:
            return torch.arange(total, device=score.device, dtype=torch.long)
        idx = torch.topk(score, k=keep, largest=True, sorted=False).indices
        return torch.sort(idx).values

    @staticmethod
    def _invert_selection(select_idx: torch.Tensor, total: int) -> torch.Tensor:
        if total <= 0:
            return torch.empty((0,), device=select_idx.device, dtype=torch.long)
        mask = torch.ones((total,), device=select_idx.device, dtype=torch.bool)
        if select_idx.numel() > 0:
            mask[select_idx] = False
        return torch.arange(total, device=select_idx.device, dtype=torch.long)[mask]

    def _merge_adjacent(
        self,
        k_mid: torch.Tensor,
        v_mid: torch.Tensor,
        imp_mid: torch.Tensor,
        target_mid: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mid_len = int(k_mid.shape[0])
        target_mid = min(max(int(target_mid), 0), mid_len)
        if mid_len == 0:
            empty_g = torch.empty((0,), device=k_mid.device, dtype=torch.float32)
            return k_mid, v_mid, empty_g
        if target_mid >= mid_len:
            group = torch.ones((mid_len,), device=k_mid.device, dtype=torch.float32)
            return k_mid.contiguous(), v_mid.contiguous(), group
        if target_mid <= 0:
            target_mid = 1

        imp = torch.nan_to_num(imp_mid.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pair_sum = imp[:-1] + imp[1:]
        pair_diff = torch.abs(imp[1:] - imp[:-1]) / (pair_sum + 1e-6)
        merge_score = pair_sum * (1.0 + self.alpha * pair_diff)
        n_to_merge = min(mid_len - target_mid, mid_len - 1)
        merge_idx = torch.topk(merge_score, k=n_to_merge, largest=False, sorted=False).indices

        delete_mask = torch.zeros((mid_len,), device=k_mid.device, dtype=torch.bool)
        delete_mask[merge_idx + 1] = True
        start_mask = ~delete_mask
        start_mask[0] = True
        group_ids = torch.cumsum(start_mask.to(torch.long), dim=0) - 1
        group_count = int(group_ids[-1].item()) + 1

        ones = torch.ones((mid_len,), device=k_mid.device, dtype=torch.float32)
        group_size = torch.zeros((group_count,), device=k_mid.device, dtype=torch.float32)
        group_size.index_add_(0, group_ids, ones)

        k_sum = torch.zeros((group_count, k_mid.shape[-1]), device=k_mid.device, dtype=torch.float32)
        v_sum = torch.zeros((group_count, v_mid.shape[-1]), device=v_mid.device, dtype=torch.float32)
        k_sum.index_add_(0, group_ids, k_mid.float())
        v_sum.index_add_(0, group_ids, v_mid.float())
        denom = group_size.clamp_min(1.0).unsqueeze(-1)
        return (k_sum / denom).to(k_mid.dtype), (v_sum / denom).to(v_mid.dtype), group_size

    def _pivot_merge_dropped(
        self,
        anchor_k: torch.Tensor,
        anchor_v: torch.Tensor,
        group_size: torch.Tensor,
        dropped_k: torch.Tensor,
        dropped_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dropped_k.numel() == 0 or anchor_k.shape[0] == 0:
            return anchor_k, anchor_v, group_size

        anchor_ref = F.normalize(anchor_k.float(), dim=-1, eps=1e-6)
        pivot_indices: List[torch.Tensor] = []
        chunk = max(1, int(self.pivot_chunk))
        for start in range(0, int(dropped_k.shape[0]), chunk):
            drop_ref = F.normalize(dropped_k[start : start + chunk].float(), dim=-1, eps=1e-6)
            sim = torch.matmul(drop_ref, anchor_ref.transpose(0, 1))
            pivot_indices.append(sim.argmax(dim=-1))
        pivot_idx = torch.cat(pivot_indices, dim=0)

        k_sum = anchor_k.float() * group_size.clamp_min(1.0).unsqueeze(-1)
        v_sum = anchor_v.float() * group_size.clamp_min(1.0).unsqueeze(-1)
        new_group = group_size.float().clone()
        ones = torch.ones((dropped_k.shape[0],), device=anchor_k.device, dtype=torch.float32)
        k_sum.index_add_(0, pivot_idx, dropped_k.float())
        v_sum.index_add_(0, pivot_idx, dropped_v.float())
        new_group.index_add_(0, pivot_idx, ones)
        denom = new_group.clamp_min(1.0).unsqueeze(-1)
        return (k_sum / denom).to(anchor_k.dtype), (v_sum / denom).to(anchor_v.dtype), new_group

    def _compress_head(
        self,
        key_1d: torch.Tensor,
        value_1d: torch.Tensor,
        importance_1d: torch.Tensor,
        is_dynamic: bool,
        layer_merge_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = int(key_1d.shape[0])
        window = min(max(0, self.window_size), seq_len)
        history_len = max(0, seq_len - window)
        if not self.enabled or history_len == 0:
            group = torch.ones((seq_len,), device=key_1d.device, dtype=torch.float32)
            return key_1d.contiguous(), value_1d.contiguous(), group

        final_history = self._history_budget(seq_len, history_len)
        if final_history <= 0:
            group = torch.ones((seq_len,), device=key_1d.device, dtype=torch.float32)
            return key_1d.contiguous(), value_1d.contiguous(), group

        kvrepack_enabled = layer_merge_enabled and (
            (is_dynamic and self.dynamic_kvrepack_enable) or ((not is_dynamic) and self.static_kvrepack_enable)
        )
        compress_ratio = self.dynamic_compress_ratio if is_dynamic else self.static_compress_ratio
        candidate_history = (
            self._candidate_len(final_history, history_len, compress_ratio)
            if kvrepack_enabled
            else final_history
        )

        history_k = key_1d[:history_len]
        history_v = value_1d[:history_len]
        recent_k = key_1d[history_len:]
        recent_v = value_1d[history_len:]
        imp = importance_1d[:history_len]

        candidate_idx = self._topk_sorted(imp, candidate_history)
        dropped_idx = self._invert_selection(candidate_idx, history_len)
        cand_k = history_k.index_select(0, candidate_idx)
        cand_v = history_v.index_select(0, candidate_idx)
        cand_imp = imp.index_select(0, candidate_idx)

        if kvrepack_enabled:
            anchor_k, anchor_v, group = self._merge_adjacent(cand_k, cand_v, cand_imp, final_history)
        else:
            anchor_k = cand_k.contiguous()
            anchor_v = cand_v.contiguous()
            group = torch.ones((anchor_k.shape[0],), device=key_1d.device, dtype=torch.float32)

        if window > 0:
            anchor_k = torch.cat([anchor_k, recent_k], dim=0)
            anchor_v = torch.cat([anchor_v, recent_v], dim=0)
            group = torch.cat([
                group,
                torch.ones((recent_k.shape[0],), device=key_1d.device, dtype=torch.float32),
            ])

        if layer_merge_enabled and self.drop_merge and dropped_idx.numel() > 0:
            dropped_k = history_k.index_select(0, dropped_idx)
            dropped_v = history_v.index_select(0, dropped_idx)
            anchor_k, anchor_v, group = self._pivot_merge_dropped(anchor_k, anchor_v, group, dropped_k, dropped_v)

        return anchor_k, anchor_v, group

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        importance_history: torch.Tensor,
        dynamic_heads: torch.Tensor,
        layer_idx: int,
        num_hidden_layers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            bsz, h_kv, seq_len, _ = key_states.shape
            group = torch.ones((bsz, h_kv, seq_len), device=key_states.device, dtype=torch.float32)
            return key_states, value_states, group

        layer_threshold = float(num_hidden_layers) * self.merge_layer_ratio
        layer_merge_enabled = True
        if self.respect_layer_gate:
            layer_merge_enabled = float(layer_idx) >= layer_threshold

        bsz, h_kv, seq_len, head_dim = key_states.shape
        keys: List[List[torch.Tensor]] = []
        values: List[List[torch.Tensor]] = []
        groups: List[List[torch.Tensor]] = []
        max_len = 0

        for bidx in range(bsz):
            key_row, value_row, group_row = [], [], []
            for hidx in range(h_kv):
                k_h, v_h, g_h = self._compress_head(
                    key_states[bidx, hidx],
                    value_states[bidx, hidx],
                    importance_history[bidx, hidx],
                    bool(dynamic_heads[bidx, hidx].item()),
                    layer_merge_enabled,
                )
                self.prompt_stats.update(seq_len, int(k_h.shape[0]))
                key_row.append(k_h)
                value_row.append(v_h)
                group_row.append(g_h)
                max_len = max(max_len, int(k_h.shape[0]))
            keys.append(key_row)
            values.append(value_row)
            groups.append(group_row)

        out_k = torch.zeros((bsz, h_kv, max_len, head_dim), device=key_states.device, dtype=key_states.dtype)
        out_v = torch.zeros((bsz, h_kv, max_len, head_dim), device=value_states.device, dtype=value_states.dtype)
        group_size = torch.ones((bsz, h_kv, max_len), device=key_states.device, dtype=torch.float32)

        for bidx in range(bsz):
            for hidx in range(h_kv):
                cur_len = int(keys[bidx][hidx].shape[0])
                out_k[bidx, hidx, :cur_len].copy_(keys[bidx][hidx])
                out_v[bidx, hidx, :cur_len].copy_(values[bidx][hidx])
                group_size[bidx, hidx, :cur_len].copy_(groups[bidx][hidx])

        return out_k, out_v, group_size

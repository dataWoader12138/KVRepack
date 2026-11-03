import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, apply_rotary_pos_emb, repeat_kv, rotate_half
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
    flash_attn_supports_top_left_mask,
)


ORIGINAL_QWEN2_FORWARD = Qwen2Model.forward


def _select_mrope_batch(pos_emb: torch.Tensor, batch_idx: int, positions: torch.Tensor) -> torch.Tensor:
    if pos_emb.ndim == 4:
        return pos_emb[0, batch_idx, positions, :]
    if pos_emb.ndim == 3:
        return pos_emb[batch_idx, positions, :]
    if pos_emb.ndim == 2:
        return pos_emb[positions, :]
    raise ValueError(f"Unsupported RoPE embedding shape: {tuple(pos_emb.shape)}")


def _apply_mrope_to_merged_keys(
    raw_k: torch.Tensor,
    cos_sel: torch.Tensor,
    sin_sel: torch.Tensor,
    mrope_section,
) -> torch.Tensor:
    del mrope_section
    return raw_k * cos_sel + rotate_half(raw_k) * sin_sel


def _safe_get_env_float(name: str) -> Optional[float]:
    value = os.getenv(name, None)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_get_env_int(name: str) -> Optional[int]:
    value = os.getenv(name, None)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    value = _safe_get_env_float(name)
    return default if value is None else value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, None)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}



class PairedKVRepackCache(DynamicCache):
    """Cache for paired K/V compression; group sizes are retained only for legacy compatibility."""

    def __init__(self) -> None:
        super().__init__()
        self.k_len_cache: List[Optional[torch.Tensor]] = []

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
    ) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.k_len_cache.append(None)
        if layer_idx == 0:
            self._seen_tokens = key_states.shape[-2]
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        self.k_len_cache[layer_idx] = k_len_states

    def update_singletons(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: Optional[torch.Tensor] = None,
    ):
        del cache_position
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.k_len_cache.append(None)

        one = torch.ones(key_states.size()[:-1], device=key_states.device, dtype=key_states.dtype)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.k_len_cache[layer_idx] = one
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.k_len_cache[layer_idx] = torch.cat([self.k_len_cache[layer_idx], one], dim=-1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.k_len_cache[layer_idx]

    @classmethod
    def from_legacy_cache(cls, past_key_values=None) -> "PairedKVRepackCache":
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                layer = past_key_values[layer_idx]
                key_states, value_states = layer[:2]
                if len(layer) >= 3 and layer[2].ndim == 3:
                    k_len_states = layer[2]
                else:
                    k_len_states = torch.ones(
                        key_states.size()[:-1],
                        device=key_states.device,
                        dtype=key_states.dtype,
                    )
                cache.set_layer_cache(layer_idx, key_states, value_states, k_len_states)
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
                ),
            )
        return legacy


def ensure_kvrepack_cache(past_key_values):
    if past_key_values is None:
        return PairedKVRepackCache()
    if isinstance(past_key_values, PairedKVRepackCache):
        return past_key_values
    if isinstance(past_key_values, tuple):
        return PairedKVRepackCache.from_legacy_cache(past_key_values)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        if hasattr(past_key_values, "k_len_cache"):
            cache = PairedKVRepackCache()
            for layer_idx, key_states in enumerate(past_key_values.key_cache):
                if key_states is None:
                    continue
                value_states = past_key_values.value_cache[layer_idx]
                k_len_states = past_key_values.k_len_cache[layer_idx]
                cache.set_layer_cache(layer_idx, key_states, value_states, k_len_states)
            return cache
        return PairedKVRepackCache.from_legacy_cache(past_key_values.to_legacy_cache())
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def snapkv_kvrepack_qwen2_forward(self, *args, **kwargs):
    past_key_values = kwargs.get("past_key_values", None)
    use_cache = kwargs.get("use_cache", self.config.use_cache)
    if use_cache and not self.training:
        kwargs["past_key_values"] = ensure_kvrepack_cache(past_key_values)
    return ORIGINAL_QWEN2_FORWARD(self, *args, **kwargs)


def init_snapkv_kvrepack(self):
    if not hasattr(self.config, "snapkv_kvrepack_window_size"):
        self.config.snapkv_kvrepack_window_size = (
            _safe_get_env_int("SNAPKV_KVREPACK_WINDOW_SIZE")
            or int(getattr(self.config, "window_size", 32))
        )

    if not hasattr(self.config, "snapkv_kvrepack_kernel_size"):
        self.config.snapkv_kvrepack_kernel_size = (
            _safe_get_env_int("SNAPKV_KVREPACK_KERNEL_SIZE")
            or int(getattr(self.config, "kernel_size", 5))
        )

    if not hasattr(self.config, "snapkv_kvrepack_pooling"):
        self.config.snapkv_kvrepack_pooling = os.getenv(
            "SNAPKV_KVREPACK_POOLING", getattr(self.config, "pooling", "avgpool")
        )

    if not hasattr(self.config, "snapkv_kvrepack_gqa_func"):
        self.config.snapkv_kvrepack_gqa_func = os.getenv(
            "SNAPKV_KVREPACK_GQA_FUNC", getattr(self.config, "gqa_func", "mean")
        )

    if not hasattr(self.config, "snapkv_kvrepack_budget_ratio"):
        env_ratio = _safe_get_env_float("BUDGET_RATIO")
        self.config.snapkv_kvrepack_budget_ratio = (
            env_ratio if env_ratio is not None else getattr(self.config, "budget_ratio", None)
        )

    if not hasattr(self.config, "snapkv_kvrepack_cache_size"):
        env_budget = _safe_get_env_int("BUDGET")
        self.config.snapkv_kvrepack_cache_size = env_budget or int(getattr(self.config, "max_capacity_prompt", 512))

    if not hasattr(self.config, "snapkv_kvrepack_compress_ratio"):
        ratio = _safe_get_env_float("SNAPKV_KVREPACK_COMPRESS_RATIO")
        if ratio is None:
            ratio = _safe_get_env_float("KVREPACK_COMPRESS_RATIO")
        self.config.snapkv_kvrepack_compress_ratio = 0.8 if ratio is None else float(ratio)
        if not (0.0 < self.config.snapkv_kvrepack_compress_ratio <= 1.0):
            raise ValueError(
                "SNAPKV_KVREPACK_COMPRESS_RATIO must be in (0, 1], got "
                f"{self.config.snapkv_kvrepack_compress_ratio}."
            )

    if not hasattr(self.config, "snapkv_kvrepack_k_diff_alpha"):
        alpha = _safe_get_env_float("SNAPKV_KVREPACK_K_DIFF_ALPHA")
        if alpha is None:
            alpha = _safe_get_env_float("K_DIFF_ALPHA")
        self.config.snapkv_kvrepack_k_diff_alpha = 2.0 if alpha is None else alpha

    if not hasattr(self.config, "snapkv_simhead_merge_ratio"):
        self.config.snapkv_simhead_merge_ratio = _env_float("SNAPKV_SIMHEAD_MERGE_RATIO", 0.30)

    if not hasattr(self.config, "snapkv_simhead_sim_space"):
        self.config.snapkv_simhead_sim_space = os.getenv("SNAPKV_SIMHEAD_SIM_SPACE", "raw").lower()

    # Clean no-log variant: merged K/V are ordinary cache slots and decode stays on FlashAttention.
    self.config.snapkv_kvrepack_score_correction = "none"

    if not hasattr(self.config, "snapkv_kvrepack_enable"):
        self.config.snapkv_kvrepack_enable = _env_bool("SNAPKV_KVREPACK_ENABLE", True)

    if not hasattr(self, "_flash_attn_uses_top_left_mask"):
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()



def _history_budget(config, seq_len: int, history_len: int) -> int:
    if history_len <= 0:
        return 0
    ratio = getattr(config, "snapkv_kvrepack_budget_ratio", None)
    if ratio is not None:
        # Match original SnapKV: retained history is int(T * ratio), then the recent window is appended.
        budget = int(seq_len * float(ratio))
    else:
        budget = int(getattr(config, "snapkv_kvrepack_cache_size", 512))
    budget = min(max(budget, 0), int(history_len))
    if budget == 0 and ratio is not None and float(ratio) > 0:
        budget = 1
    return budget


def _candidate_len(final_len: int, history_len: int, compress_ratio: float) -> int:
    final_len = min(max(int(final_len), 0), int(history_len))
    if final_len <= 0 or history_len <= 0:
        return 0
    ratio = min(max(float(compress_ratio), 1e-6), 1.0)
    return min(int(history_len), max(final_len, int(math.ceil(final_len / ratio - 1e-6))))


def _topk_sorted(score: torch.Tensor, keep: int) -> torch.Tensor:
    total = int(score.shape[0])
    keep = min(max(int(keep), 0), total)
    if keep <= 0:
        return torch.empty((0,), device=score.device, dtype=torch.long)
    if keep >= total:
        return torch.arange(total, device=score.device, dtype=torch.long)
    idx = torch.topk(score.float(), k=keep, largest=True, sorted=False).indices
    return torch.sort(idx).values


def _topk_from_candidates_sorted(candidate_idx: torch.Tensor, candidate_score: torch.Tensor, keep: int) -> torch.Tensor:
    keep = min(max(int(keep), 0), int(candidate_idx.numel()))
    if keep <= 0:
        return torch.empty((0,), device=candidate_idx.device, dtype=torch.long)
    if keep >= int(candidate_idx.numel()):
        return torch.arange(candidate_idx.numel(), device=candidate_idx.device, dtype=torch.long)
    local = torch.topk(candidate_score.float(), k=keep, largest=True, sorted=False).indices
    order = torch.argsort(candidate_idx.index_select(0, local))
    return local.index_select(0, order)


def _adjacent_similarity(k_seq: torch.Tensor) -> torch.Tensor:
    if int(k_seq.shape[0]) < 2:
        return torch.zeros((), device=k_seq.device, dtype=torch.float32)
    left = F.normalize(k_seq[:-1].float(), dim=-1, eps=1e-6)
    right = F.normalize(k_seq[1:].float(), dim=-1, eps=1e-6)
    return (left * right).sum(dim=-1).mean()


@torch.no_grad()
def compute_snapkv_importance_observation(
    query_states_rope: torch.Tensor,
    key_states_rope: torch.Tensor,
    num_key_value_groups: int,
    head_dim: int,
    window_size: int = 32,
    kernel_size: int = 5,
    pooling: str = "avgpool",
    gqa_func: str = "mean",
) -> torch.Tensor:
    bsz, num_heads, q_len, _ = query_states_rope.shape
    num_kv_heads = key_states_rope.shape[1]
    key_rep = repeat_kv(key_states_rope, num_key_value_groups)

    window_size = max(1, min(int(window_size), q_len))
    past_len = max(0, q_len - window_size)
    importance = torch.zeros(
        (bsz, num_kv_heads, q_len),
        device=query_states_rope.device,
        dtype=torch.float32,
    )
    if past_len <= 0:
        return importance.to(query_states_rope.dtype)

    scores = torch.matmul(
        query_states_rope[..., -window_size:, :],
        key_rep.transpose(2, 3),
    ) / math.sqrt(float(head_dim))

    local_mask = torch.full(
        (window_size, window_size),
        torch.finfo(scores.dtype).min,
        device=scores.device,
    )
    mask_cond = torch.arange(local_mask.size(-1), device=scores.device)
    local_mask.masked_fill_(mask_cond < (mask_cond + 1).view(local_mask.size(-1), 1), 0)
    scores[:, :, -window_size:, -window_size:] += local_mask[None, None, :, :]

    scores = torch.nan_to_num(
        scores,
        nan=0.0,
        posinf=torch.finfo(scores.dtype).max,
        neginf=torch.finfo(scores.dtype).min,
    )
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states_rope.dtype)
    attn_mean = probs[:, :, -window_size:, :past_len].mean(dim=-2)
    attn_mean = attn_mean.view(bsz, num_kv_heads, num_key_value_groups, past_len)
    if gqa_func == "max":
        attn_mean = attn_mean.max(dim=2).values
    elif gqa_func == "mean":
        attn_mean = attn_mean.mean(dim=2)
    else:
        raise ValueError(f"Unsupported SNAPKV_KVREPACK_GQA_FUNC={gqa_func}")

    kernel_size = max(1, int(kernel_size))
    if pooling == "avgpool":
        pooled = torch.nn.functional.avg_pool1d(
            attn_mean.float(), kernel_size=kernel_size, padding=kernel_size // 2, stride=1
        )
    elif pooling == "maxpool":
        pooled = torch.nn.functional.max_pool1d(
            attn_mean.float(), kernel_size=kernel_size, padding=kernel_size // 2, stride=1
        )
    else:
        raise ValueError(f"Unsupported SNAPKV_KVREPACK_POOLING={pooling}")

    importance[:, :, :past_len] = pooled[:, :, :past_len]
    return importance.to(query_states_rope.dtype)


def _paired_merge_one_sequence(
    raw_k_seq: torch.Tensor,
    rope_k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    importance: torch.Tensor,
    target_len: int,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_idx: int,
    k_diff_alpha: float,
    mrope_section,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = int(rope_k_seq.shape[0])
    target_len = min(seq_len, max(1, int(target_len)))
    if target_len >= seq_len:
        group_size = torch.ones((seq_len,), device=rope_k_seq.device, dtype=torch.float32)
        return rope_k_seq.contiguous(), v_seq.contiguous(), group_size

    importance = torch.nan_to_num(importance.float(), nan=0.0, posinf=0.0, neginf=0.0)
    left_imp = importance[:-1]
    right_imp = importance[1:]
    pair_sum = left_imp + right_imp
    pair_diff = torch.abs(right_imp - left_imp) / (pair_sum + 1e-6)
    merge_score = pair_sum * (1.0 + float(k_diff_alpha) * pair_diff)
    positions = positions.to(device=rope_k_seq.device, dtype=torch.long)

    n_to_merge = min(max(seq_len - target_len, 0), seq_len - 1)
    if n_to_merge <= 0:
        merge_idx = torch.empty((0,), device=rope_k_seq.device, dtype=torch.long)
    else:
        # Adjacent means adjacent in the SnapKV-selected candidate sequence.
        merge_idx = torch.topk(merge_score, k=n_to_merge, largest=False, sorted=False).indices

    delete_mask = torch.zeros((seq_len,), device=rope_k_seq.device, dtype=torch.bool)
    delete_mask[merge_idx + 1] = True
    start_mask = ~delete_mask
    start_mask[0] = True
    group_ids = torch.cumsum(start_mask.to(torch.long), dim=0) - 1
    group_count = int(group_ids[-1].item()) + 1

    ones = torch.ones((seq_len,), device=rope_k_seq.device, dtype=torch.float32)
    group_size = torch.zeros((group_count,), device=rope_k_seq.device, dtype=torch.float32)
    group_size.index_add_(0, group_ids, ones)
    denom = group_size.clamp_min(1.0).unsqueeze(-1)

    raw_sum = torch.zeros((group_count, raw_k_seq.shape[-1]), device=raw_k_seq.device, dtype=torch.float32)
    raw_sum.index_add_(0, group_ids, raw_k_seq.float())
    raw_comp = (raw_sum / denom).to(raw_k_seq.dtype)

    v_sum = torch.zeros((group_count, v_seq.shape[-1]), device=v_seq.device, dtype=torch.float32)
    v_sum.index_add_(0, group_ids, v_seq.float())
    v_comp = (v_sum / denom).to(v_seq.dtype)

    anchor_pos = positions[start_mask].to(device=cos.device, dtype=torch.long)
    cos_sel = _select_mrope_batch(cos, int(batch_idx), anchor_pos).to(raw_comp.device)
    sin_sel = _select_mrope_batch(sin, int(batch_idx), anchor_pos).to(raw_comp.device)
    k_comp = _apply_mrope_to_merged_keys(raw_comp, cos_sel, sin_sel, mrope_section).to(rope_k_seq.dtype)
    return k_comp, v_comp, group_size


@torch.no_grad()
def _build_prompt_record(
    config,
    layer_idx: int,
    raw_key_states: torch.Tensor,
    rope_key_states: torch.Tensor,
    value_states: torch.Tensor,
    importance_kv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section,
) -> Dict:
    bsz, h_kv, seq_len, head_dim = rope_key_states.shape
    recent_len = min(max(int(config.snapkv_kvrepack_window_size), 0), seq_len)
    history_len = max(0, seq_len - recent_len)
    final_history = _history_budget(config=config, seq_len=seq_len, history_len=history_len)
    candidate_history = _candidate_len(
        final_len=final_history,
        history_len=history_len,
        compress_ratio=float(config.snapkv_kvrepack_compress_ratio),
    )
    cos = cos.to(device=rope_key_states.device)
    sin = sin.to(device=rope_key_states.device)

    if history_len == 0 or final_history <= 0 or not bool(config.snapkv_kvrepack_enable):
        group = torch.ones((bsz, h_kv, seq_len), device=rope_key_states.device, dtype=rope_key_states.dtype)
        return {
            "layer_idx": int(layer_idx),
            "direct_key": rope_key_states.detach(),
            "direct_value": value_states.detach(),
            "direct_group": group.detach(),
            "head_sim": torch.zeros((bsz, h_kv), device=rope_key_states.device, dtype=torch.float32),
        }

    hist_raw = raw_key_states[:, :, :history_len, :]
    hist_rope = rope_key_states[:, :, :history_len, :]
    hist_value = value_states[:, :, :history_len, :]
    hist_imp = importance_kv[:, :, :history_len]

    cand_raw = torch.zeros((bsz, h_kv, candidate_history, head_dim), device=raw_key_states.device, dtype=raw_key_states.dtype)
    cand_rope = torch.zeros((bsz, h_kv, candidate_history, head_dim), device=rope_key_states.device, dtype=rope_key_states.dtype)
    cand_value = torch.zeros((bsz, h_kv, candidate_history, head_dim), device=value_states.device, dtype=value_states.dtype)
    cand_imp = torch.zeros((bsz, h_kv, candidate_history), device=importance_kv.device, dtype=importance_kv.dtype)
    cand_idx = torch.zeros((bsz, h_kv, candidate_history), device=importance_kv.device, dtype=torch.long)
    head_sim = torch.zeros((bsz, h_kv), device=rope_key_states.device, dtype=torch.float32)

    for bidx in range(bsz):
        for hidx in range(h_kv):
            idx = _topk_sorted(hist_imp[bidx, hidx], candidate_history)
            cur_len = int(idx.numel())
            if cur_len == 0:
                continue
            cand_idx[bidx, hidx, :cur_len].copy_(idx)
            cand_raw[bidx, hidx, :cur_len].copy_(hist_raw[bidx, hidx].index_select(0, idx))
            cand_rope[bidx, hidx, :cur_len].copy_(hist_rope[bidx, hidx].index_select(0, idx))
            cand_value[bidx, hidx, :cur_len].copy_(hist_value[bidx, hidx].index_select(0, idx))
            cand_imp[bidx, hidx, :cur_len].copy_(hist_imp[bidx, hidx].index_select(0, idx))
            sim_source = cand_raw if config.snapkv_simhead_sim_space == "raw" else cand_rope
            head_sim[bidx, hidx] = _adjacent_similarity(sim_source[bidx, hidx, :cur_len])

    recent_key = rope_key_states[:, :, history_len:, :].detach()
    recent_value = value_states[:, :, history_len:, :].detach()

    return {
        "layer_idx": int(layer_idx),
        "candidate_raw": cand_raw.detach(),
        "candidate_rope": cand_rope.detach(),
        "candidate_value": cand_value.detach(),
        "candidate_imp": cand_imp.detach(),
        "candidate_idx": cand_idx.detach(),
        "candidate_history": int(candidate_history),
        "final_history": int(final_history),
        "history_len": int(history_len),
        "recent_key": recent_key,
        "recent_value": recent_value,
        "head_sim": head_sim.detach(),
        "cos": cos.detach(),
        "sin": sin.detach(),
        "mrope_section": mrope_section,
    }


def _global_merge_masks(records: List[Dict], merge_ratio: float) -> List[torch.Tensor]:
    if not records:
        return []
    sims = torch.stack([rec["head_sim"].float() for rec in records], dim=1)  # [B, L, H]
    bsz, num_layers, h_kv = sims.shape
    masks = [torch.zeros_like(rec["head_sim"], dtype=torch.bool) for rec in records]
    total = num_layers * h_kv
    ratio = min(max(float(merge_ratio), 0.0), 1.0)
    merge_count = int(math.ceil(total * ratio)) if ratio > 0 else 0
    merge_count = min(max(merge_count, 0), total)
    if merge_count == 0:
        return masks

    flat = sims.reshape(bsz, total)
    top = torch.topk(flat, k=merge_count, largest=True, sorted=False).indices
    for bidx in range(bsz):
        for flat_idx in top[bidx].tolist():
            layer_pos = flat_idx // h_kv
            head_idx = flat_idx % h_kv
            masks[layer_pos][bidx, head_idx] = True
    return masks


@torch.no_grad()
def _finalize_prompt_cache(config, past_key_value: PairedKVRepackCache) -> None:
    records: List[Dict] = getattr(config, "_snapkv_kvrepack_records", [])
    if not records or past_key_value is None:
        return

    masks = _global_merge_masks(records, float(config.snapkv_simhead_merge_ratio))
    total_heads = 0
    merge_heads = 0
    layer_counts: List[int] = []
    sim_values = []

    for rec, merge_mask in zip(records, masks):
        layer_idx = int(rec["layer_idx"])
        if "direct_key" in rec:
            past_key_value.set_layer_cache(layer_idx, rec["direct_key"], rec["direct_value"], rec["direct_group"])
            layer_counts.append(0)
            continue

        cand_raw = rec["candidate_raw"]
        cand_rope = rec["candidate_rope"]
        cand_value = rec["candidate_value"]
        cand_imp = rec["candidate_imp"]
        cand_idx = rec["candidate_idx"]
        recent_key = rec["recent_key"]
        recent_value = rec["recent_value"]
        final_history = int(rec["final_history"])
        candidate_history = int(rec["candidate_history"])
        bsz, h_kv, _, head_dim = cand_rope.shape
        recent_len = int(recent_key.shape[-2])
        out_len = final_history + recent_len

        out_k = torch.zeros((bsz, h_kv, out_len, head_dim), device=cand_rope.device, dtype=cand_rope.dtype)
        out_v = torch.zeros((bsz, h_kv, out_len, head_dim), device=cand_value.device, dtype=cand_value.dtype)
        group = torch.ones((bsz, h_kv, out_len), device=cand_rope.device, dtype=cand_rope.dtype)

        layer_merge = int(merge_mask.sum().item())
        layer_counts.append(layer_merge)
        total_heads += int(merge_mask.numel())
        merge_heads += layer_merge
        sim_values.append(rec["head_sim"].reshape(-1).float())

        for bidx in range(bsz):
            for hidx in range(h_kv):
                if bool(merge_mask[bidx, hidx].item()) and candidate_history > final_history:
                    hist_k, hist_v, hist_g = _paired_merge_one_sequence(
                        raw_k_seq=cand_raw[bidx, hidx, :candidate_history],
                        rope_k_seq=cand_rope[bidx, hidx, :candidate_history],
                        v_seq=cand_value[bidx, hidx, :candidate_history],
                        importance=cand_imp[bidx, hidx, :candidate_history],
                        target_len=final_history,
                        positions=cand_idx[bidx, hidx, :candidate_history],
                        cos=rec["cos"],
                        sin=rec["sin"],
                        batch_idx=bidx,
                        k_diff_alpha=float(config.snapkv_kvrepack_k_diff_alpha),
                        mrope_section=rec["mrope_section"],
                    )
                else:
                    local = _topk_from_candidates_sorted(
                        cand_idx[bidx, hidx, :candidate_history],
                        cand_imp[bidx, hidx, :candidate_history],
                        final_history,
                    )
                    hist_k = cand_rope[bidx, hidx].index_select(0, local).contiguous()
                    hist_v = cand_value[bidx, hidx].index_select(0, local).contiguous()
                    hist_g = torch.ones((hist_k.shape[0],), device=cand_rope.device, dtype=torch.float32)

                if final_history > 0:
                    out_k[bidx, hidx, :final_history].copy_(hist_k[:final_history])
                    out_v[bidx, hidx, :final_history].copy_(hist_v[:final_history])
                    group[bidx, hidx, :final_history].copy_(hist_g[:final_history].to(group.dtype))
                if recent_len > 0:
                    out_k[bidx, hidx, final_history:].copy_(recent_key[bidx, hidx])
                    out_v[bidx, hidx, final_history:].copy_(recent_value[bidx, hidx])
                    group[bidx, hidx, final_history:].fill_(1.0)

        expected_len = final_history + recent_len
        if out_k.shape[-2] != expected_len or out_v.shape[-2] != expected_len:
            raise AssertionError(
                "SnapKV-KVRepack budget mismatch: "
                f"cache_len={out_k.shape[-2]} expected={expected_len}"
            )
        past_key_value.set_layer_cache(layer_idx, out_k, out_v, group)

    if total_heads > 0:
        sims = torch.cat(sim_values) if sim_values else torch.zeros((1,))
        ratio = float(merge_heads) / float(total_heads)
        first = records[0]
        print(
            "[SnapKV-KVRepack] adjacent-sim global gate: "
            f"merge_heads={merge_heads}/{total_heads} merge_ratio={ratio:.4f} "
            f"target_ratio={float(config.snapkv_simhead_merge_ratio):.4f} "
            f"budget_ratio={float(config.snapkv_kvrepack_budget_ratio or 0):.4f} "
            f"final_history={int(first.get('final_history', 0))} "
            f"candidate_history={int(first.get('candidate_history', 0))} "
            f"compress_ratio={float(config.snapkv_kvrepack_compress_ratio):.4f} "
            f"alpha={float(config.snapkv_kvrepack_k_diff_alpha):.4f} "
            f"sim_space={config.snapkv_simhead_sim_space} "
            f"score_correction=none attention=flash "
            f"sim_min/mean/max={float(sims.min().item()):.4f}/{float(sims.mean().item()):.4f}/{float(sims.max().item()):.4f} "
            f"merge_heads_per_layer={layer_counts}"
        )

    config._snapkv_kvrepack_records = []


def qwen_flash_attn2_forward_SnapKVKVRepack(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[PairedKVRepackCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    init_snapkv_kvrepack(self)
    past_key_value = ensure_kvrepack_cache(past_key_value)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    raw_key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    num_attention_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    num_key_value_groups = self.num_key_value_groups
    hidden_size = self.config.hidden_size

    query_states = query_states.view(bsz, q_len, num_attention_heads, self.head_dim).transpose(1, 2)
    raw_key_states = raw_key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        raise ValueError("Qwen2 SnapKV-KVRepack requires shared position_embeddings from Qwen2Model.")
    else:
        cos, sin = position_embeddings
    query_states_rope, key_states_rope = apply_rotary_pos_emb(query_states, raw_key_states, cos, sin)

    is_prefill = q_len != 1
    if is_prefill:
        if self.layer_idx == 0:
            self.config._snapkv_kvrepack_records = []

        importance_kv = compute_snapkv_importance_observation(
            query_states_rope=query_states_rope,
            key_states_rope=key_states_rope,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            window_size=self.config.snapkv_kvrepack_window_size,
            kernel_size=self.config.snapkv_kvrepack_kernel_size,
            pooling=self.config.snapkv_kvrepack_pooling,
            gqa_func=self.config.snapkv_kvrepack_gqa_func,
        )

        if use_cache:
            record = _build_prompt_record(
                config=self.config,
                layer_idx=self.layer_idx,
                raw_key_states=raw_key_states,
                rope_key_states=key_states_rope,
                value_states=value_states,
                importance_kv=importance_kv,
                cos=cos,
                sin=sin,
                mrope_section=None,
            )
            self.config._snapkv_kvrepack_records.append(record)
            if self.layer_idx == self.config.num_hidden_layers - 1:
                _finalize_prompt_cache(self.config, past_key_value)

        key_for_attn = repeat_kv(key_states_rope, num_key_value_groups)
        value_for_attn = repeat_kv(value_states, num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states_rope.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states_rope = query_states_rope.to(target_dtype)
            key_for_attn = key_for_attn.to(target_dtype)
            value_for_attn = value_for_attn.to(target_dtype)

        q_fa = query_states_rope.transpose(1, 2)
        k_fa = key_for_attn.transpose(1, 2)
        v_fa = value_for_attn.transpose(1, 2)
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None
        attn_output = _flash_attention_forward(
            q_fa,
            k_fa,
            v_fa,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        attn_output = attn_output.reshape(bsz, q_len, hidden_size).contiguous()
        attn_weights = None
    else:
        if use_cache:
            key_cache, value_cache, _ = past_key_value.update_singletons(
                key_states=key_states_rope,
                value_states=value_states,
                layer_idx=self.layer_idx,
                cache_position=cache_position,
            )
        else:
            key_cache = key_states_rope
            value_cache = value_states

        key_rep = repeat_kv(key_cache, num_key_value_groups)
        value_rep = repeat_kv(value_cache, num_key_value_groups)

        input_dtype = query_states_rope.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states_rope = query_states_rope.to(target_dtype)
            key_rep = key_rep.to(target_dtype)
            value_rep = value_rep.to(target_dtype)

        q_fa = query_states_rope.transpose(1, 2)
        k_fa = key_rep.transpose(1, 2)
        v_fa = value_rep.transpose(1, 2)
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            q_fa,
            k_fa,
            v_fa,
            attention_mask,
            q_len,
            dropout=0.0 if not self.training else self.attention_dropout,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        attn_output = attn_output.reshape(bsz, q_len, hidden_size).contiguous()
        attn_weights = None

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

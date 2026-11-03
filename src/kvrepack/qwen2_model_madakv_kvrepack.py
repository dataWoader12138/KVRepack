import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
    flash_attn_supports_top_left_mask,
)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, rotate_half

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


def _env_int(name: str, default: int) -> int:
    value = _safe_get_env_int(name)
    return default if value is None else value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, None)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


def init_madakv_kvrepack(self) -> None:
    if not hasattr(self.config, "madakv_kvrepack_window_size"):
        self.config.madakv_kvrepack_window_size = _env_int("MADAKV_KVREPACK_WINDOW_SIZE", 32)
    if not hasattr(self.config, "madakv_kvrepack_budget_ratio"):
        self.config.madakv_kvrepack_budget_ratio = _env_float(
            "MADAKV_KVREPACK_BUDGET_RATIO",
            _env_float("BUDGET_RATIO", 0.10),
        )
    if not hasattr(self.config, "madakv_kvrepack_threshold"):
        self.config.madakv_kvrepack_threshold = _env_float(
            "MADAKV_KVREPACK_THRESHOLD",
            _env_float("THRESHOLD", 0.8),
        )
    if not hasattr(self.config, "madakv_kvrepack_gqa_func"):
        self.config.madakv_kvrepack_gqa_func = os.getenv("MADAKV_KVREPACK_GQA_FUNC", "mean")
    if not hasattr(self.config, "madakv_kvrepack_compress_ratio"):
        self.config.madakv_kvrepack_compress_ratio = _env_float("MADAKV_KVREPACK_COMPRESS_RATIO", 0.80)
        if not (0.0 < float(self.config.madakv_kvrepack_compress_ratio) <= 1.0):
            raise ValueError(
                "MADAKV_KVREPACK_COMPRESS_RATIO must be in (0, 1], got "
                f"{self.config.madakv_kvrepack_compress_ratio}."
            )
    if not hasattr(self.config, "madakv_kvrepack_alpha"):
        self.config.madakv_kvrepack_alpha = _env_float(
            "MADAKV_KVREPACK_K_DIFF_ALPHA",
            _env_float("K_DIFF_ALPHA", 1.0),
        )
    if not hasattr(self.config, "madakv_simhead_merge_ratio"):
        self.config.madakv_simhead_merge_ratio = _env_float("MADAKV_SIMHEAD_MERGE_RATIO", 0.50)
    if not hasattr(self.config, "madakv_simhead_sim_space"):
        self.config.madakv_simhead_sim_space = os.getenv("MADAKV_SIMHEAD_SIM_SPACE", "raw").lower()
    # Clean no-log variant: merged K/V are ordinary cache slots and decode stays on FlashAttention.
    self.config.madakv_kvrepack_score_correction = "none"
    if not hasattr(self.config, "madakv_kvrepack_enable"):
        self.config.madakv_kvrepack_enable = _env_bool("MADAKV_KVREPACK_ENABLE", True)

    if not hasattr(self, "_flash_attn_uses_top_left_mask"):
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()


def _current_budget_tensor(
    current_budget: Optional[torch.Tensor],
    seq_len: int,
    history_len: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if current_budget is None:
        budget = int(seq_len * float(ratio))
        budget = min(max(budget, 0), int(history_len))
        return torch.tensor([budget], dtype=torch.int32, device=device)
    budget = current_budget.to(device=device, dtype=torch.int32).reshape(-1)
    budget = torch.clamp(budget, min=0, max=int(history_len))
    return budget


def _candidate_len(final_len: int, history_len: int, compress_ratio: float) -> int:
    final_len = min(max(int(final_len), 0), int(history_len))
    if final_len <= 0 or history_len <= 0:
        return 0
    ratio = min(max(float(compress_ratio), 1e-6), 1.0)
    return min(int(history_len), max(final_len, int(math.ceil(final_len / ratio - 1e-6))))


def _topk_original_order(score: torch.Tensor, keep: int) -> torch.Tensor:
    total = int(score.shape[0])
    keep = min(max(int(keep), 0), total)
    if keep <= 0:
        return torch.empty((0,), device=score.device, dtype=torch.long)
    # Match MadaKVCluster.update_kv exactly: original code applies topk on a
    # [bsz=1, history] tensor with sorted=True implicit.  Keeping the 2D shape
    # also preserves PyTorch tie-breaking behavior more closely.
    return torch.topk(score.unsqueeze(0), k=keep, dim=-1).indices.squeeze(0)


def _topk_sorted(score: torch.Tensor, keep: int) -> torch.Tensor:
    total = int(score.shape[0])
    keep = min(max(int(keep), 0), total)
    if keep <= 0:
        return torch.empty((0,), device=score.device, dtype=torch.long)
    if keep >= total:
        return torch.arange(total, device=score.device, dtype=torch.long)
    idx = torch.topk(score.float(), k=keep, largest=True, sorted=False).indices
    return torch.sort(idx).values



def _adjacent_similarity(k_seq: torch.Tensor) -> torch.Tensor:
    if int(k_seq.shape[0]) < 2:
        return torch.zeros((), device=k_seq.device, dtype=torch.float32)
    left = F.normalize(k_seq[:-1].float(), dim=-1, eps=1e-6)
    right = F.normalize(k_seq[1:].float(), dim=-1, eps=1e-6)
    return (left * right).sum(dim=-1).mean()


@torch.no_grad()
def _compute_madakv_importance(
    query_states_rope: torch.Tensor,
    key_states_rope: torch.Tensor,
    num_key_value_groups: int,
    head_dim: int,
    window_size: int = 32,
    gqa_func: str = "mean",
) -> torch.Tensor:
    bsz, num_heads, q_len, _ = query_states_rope.shape
    num_kv_heads = key_states_rope.shape[1]
    window_size = min(max(1, int(window_size)), q_len)
    history_len = max(0, q_len - window_size)
    if history_len == 0:
        return torch.zeros((bsz, num_kv_heads, 0), device=key_states_rope.device, dtype=query_states_rope.dtype)

    key_rep = repeat_kv(key_states_rope, num_key_value_groups)
    scores = torch.matmul(query_states_rope[..., -window_size:, :], key_rep.transpose(2, 3)) / math.sqrt(head_dim)

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
        raise ValueError(f"Unsupported MADAKV_KVREPACK_GQA_FUNC={gqa_func}")
    return importance


def _merge_adjacent_raw_reapply(
    raw_k_seq: torch.Tensor,
    rope_k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    importance: torch.Tensor,
    target_len: int,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_idx: int,
    alpha: float,
    mrope_section,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = int(raw_k_seq.shape[0])
    target_len = min(max(int(target_len), 0), seq_len)
    if seq_len == 0 or target_len == 0:
        empty_k = raw_k_seq.new_empty((0, raw_k_seq.shape[-1]))
        empty_v = v_seq.new_empty((0, v_seq.shape[-1]))
        empty_g = torch.empty((0,), device=raw_k_seq.device, dtype=torch.float32)
        return empty_k.to(rope_k_seq.dtype), empty_v, empty_g
    if target_len >= seq_len:
        group = torch.ones((seq_len,), device=raw_k_seq.device, dtype=torch.float32)
        return rope_k_seq.contiguous(), v_seq.contiguous(), group

    imp = torch.nan_to_num(importance.float(), nan=0.0, posinf=0.0, neginf=0.0)
    pair_sum = imp[:-1] + imp[1:]
    pair_diff = torch.abs(imp[1:] - imp[:-1]) / (pair_sum + 1e-6)
    merge_score = pair_sum * (1.0 + float(alpha) * pair_diff)
    n_to_merge = min(seq_len - target_len, seq_len - 1)
    merge_idx = torch.topk(merge_score, k=n_to_merge, largest=False, sorted=False).indices

    delete_mask = torch.zeros((seq_len,), device=raw_k_seq.device, dtype=torch.bool)
    delete_mask[merge_idx + 1] = True
    start_mask = ~delete_mask
    start_mask[0] = True
    group_ids = torch.cumsum(start_mask.to(torch.long), dim=0) - 1
    group_count = int(group_ids[-1].item()) + 1

    ones = torch.ones((seq_len,), device=raw_k_seq.device, dtype=torch.float32)
    group_size = torch.zeros((group_count,), device=raw_k_seq.device, dtype=torch.float32)
    group_size.index_add_(0, group_ids, ones)
    denom = group_size.clamp_min(1.0).unsqueeze(-1)

    raw_sum = torch.zeros((group_count, raw_k_seq.shape[-1]), device=raw_k_seq.device, dtype=torch.float32)
    value_sum = torch.zeros((group_count, v_seq.shape[-1]), device=v_seq.device, dtype=torch.float32)
    raw_sum.index_add_(0, group_ids, raw_k_seq.float())
    value_sum.index_add_(0, group_ids, v_seq.float())
    raw_comp = (raw_sum / denom).to(raw_k_seq.dtype)
    value_comp = (value_sum / denom).to(v_seq.dtype)

    positions = positions.to(device=raw_k_seq.device, dtype=torch.long)
    anchor_pos = positions[start_mask].to(device=cos.device, dtype=torch.long)
    cos_sel = _select_mrope_batch(cos, int(batch_idx), anchor_pos).to(raw_comp.device)
    sin_sel = _select_mrope_batch(sin, int(batch_idx), anchor_pos).to(raw_comp.device)
    key_comp = _apply_mrope_to_merged_keys(raw_comp, cos_sel, sin_sel, mrope_section).to(rope_k_seq.dtype)
    return key_comp, value_comp, group_size


def _empty_candidate(device: torch.device, dtype: torch.dtype, head_dim: int) -> Dict:
    empty_idx = torch.empty((0,), device=device, dtype=torch.long)
    empty_vec = torch.empty((0, head_dim), device=device, dtype=dtype)
    return {
        "idx": empty_idx,
        "raw": empty_vec,
        "rope": empty_vec,
        "value": empty_vec,
        "imp": torch.empty((0,), device=device, dtype=dtype),
        "direct_idx": empty_idx,
        "direct_rope": empty_vec,
        "direct_value": empty_vec,
        "final": 0,
    }


def _select_candidates(
    score: torch.Tensor,
    raw_hist: torch.Tensor,
    rope_hist: torch.Tensor,
    value_hist: torch.Tensor,
    imp_hist: torch.Tensor,
    final_len: int,
    compress_ratio: float,
) -> Dict:
    head_dim = int(raw_hist.shape[-1])
    final_len = min(max(int(final_len), 0), int(score.shape[0]))
    if final_len <= 0:
        return _empty_candidate(raw_hist.device, raw_hist.dtype, head_dim)
    direct_idx = _topk_original_order(score, final_len)
    cand_len = _candidate_len(final_len, int(score.shape[0]), compress_ratio)
    idx = _topk_sorted(score, cand_len)
    if idx.numel() == 0:
        return _empty_candidate(raw_hist.device, raw_hist.dtype, head_dim)
    return {
        "idx": idx,
        "raw": raw_hist.index_select(0, idx),
        "rope": rope_hist.index_select(0, idx),
        "value": value_hist.index_select(0, idx),
        "imp": imp_hist.index_select(0, idx),
        "direct_idx": direct_idx,
        "direct_rope": rope_hist.index_select(0, direct_idx),
        "direct_value": value_hist.index_select(0, direct_idx),
        "final": final_len,
    }


def _compress_candidate(
    cand: Dict,
    enable_merge: bool,
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_idx: int,
    alpha: float,
    mrope_section,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    final_len = int(cand["final"])
    raw = cand["raw"]
    rope = cand["rope"]
    value = cand["value"]
    imp = cand["imp"]
    idx = cand["idx"]
    if final_len <= 0 or int(idx.numel()) == 0:
        empty_k = rope.new_empty((0, rope.shape[-1]))
        empty_v = value.new_empty((0, value.shape[-1]))
        empty_g = torch.empty((0,), device=rope.device, dtype=torch.float32)
        return empty_k, empty_v, empty_g
    if enable_merge and int(idx.numel()) > final_len:
        return _merge_adjacent_raw_reapply(
            raw_k_seq=raw,
            rope_k_seq=rope,
            v_seq=value,
            importance=imp,
            target_len=final_len,
            positions=idx,
            cos=cos,
            sin=sin,
            batch_idx=batch_idx,
            alpha=alpha,
            mrope_section=mrope_section,
        )
    hist_k = cand["direct_rope"].contiguous()
    hist_v = cand["direct_value"].contiguous()
    hist_g = torch.ones((hist_k.shape[0],), device=rope.device, dtype=torch.float32)
    return hist_k, hist_v, hist_g


def _safe_masks(
    is_vision_token_mask: Optional[torch.Tensor],
    is_text_token_mask: Optional[torch.Tensor],
    bsz: int,
    history_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_vision_token_mask is not None and is_vision_token_mask.ndim >= 2 and is_vision_token_mask.shape[-1] >= history_len:
        vision = is_vision_token_mask[:, :history_len].to(device=device, dtype=torch.bool)
    else:
        vision = torch.zeros((bsz, history_len), device=device, dtype=torch.bool)
    if is_text_token_mask is not None and is_text_token_mask.ndim >= 2 and is_text_token_mask.shape[-1] >= history_len:
        text = is_text_token_mask[:, :history_len].to(device=device, dtype=torch.bool)
    else:
        text = ~vision
    return vision, text


def _next_layer_budget_like_madakv(
    current_budget: torch.Tensor,
    q_len: int,
    history_len: int,
    budget_ratio: float,
    threshold: float,
    layer_idx: int,
    num_hidden_layers: int,
    vision_attn_score: torch.Tensor,
    text_attn_score: torch.Tensor,
    vision_prefer_score: torch.Tensor,
    text_prefer_score: torch.Tensor,
) -> torch.Tensor:
    cur = current_budget.reshape(-1)[0]
    compensate_k = 0
    num_kv_heads = int(vision_attn_score.shape[1])
    for head_idx in range(num_kv_heads):
        vision_score_tmp = vision_attn_score[0, head_idx, :].sort(dim=0, descending=True).values
        cal_sum, vi = 0, 0
        while cal_sum < threshold * vision_prefer_score[0][head_idx].item() and vi < vision_score_tmp.shape[0]:
            cal_sum += vision_score_tmp[vi].item()
            vi = vi + 1

        text_score_tmp = text_attn_score[0, head_idx, :].sort(dim=0, descending=True).values
        cal_sum, ti = 0, 0
        # Preserve the current MadaKV allocation behavior in baseline_utils.py,
        # including its batch-dimension loop bound for text.
        while cal_sum < threshold * text_prefer_score[0][head_idx].item() and ti < text_attn_score.shape[0]:
            cal_sum += text_score_tmp[ti]
            ti = ti + 1
        compensate_k += vi + ti - cur.item()

    min_budget = int(q_len * budget_ratio)
    max_budget = int(history_len)
    return torch.clamp(
        torch.tensor(cur.item() - compensate_k / (num_hidden_layers - layer_idx)).int(),
        min=min_budget,
        max=max_budget,
    ).int()


@torch.no_grad()
def _build_layer_record(
    attn_module,
    raw_key_states: torch.Tensor,
    rope_key_states: torch.Tensor,
    value_states: torch.Tensor,
    importance_history: torch.Tensor,
    is_vision_token_mask: Optional[torch.Tensor],
    is_text_token_mask: Optional[torch.Tensor],
    current_budget: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_kwargs: Dict,
):
    config = attn_module.config
    bsz, h_kv, seq_len, head_dim = rope_key_states.shape
    if bsz != 1:
        raise AssertionError("MadaKV-KVRepack currently follows MadaKV and supports batch size 1 only.")

    window = min(max(0, int(config.madakv_kvrepack_window_size)), seq_len)
    history_len = max(0, seq_len - window)
    current_budget_t = _current_budget_tensor(
        current_budget=current_budget,
        seq_len=seq_len,
        history_len=history_len,
        ratio=float(config.madakv_kvrepack_budget_ratio),
        device=importance_history.device,
    )

    if history_len == 0 or int(current_budget_t.reshape(-1)[0].item()) <= 0 or not bool(config.madakv_kvrepack_enable):
        group = torch.ones((bsz, h_kv, seq_len), device=rope_key_states.device, dtype=torch.float32)
        return {
            "layer_idx": attn_module.layer_idx,
            "module": attn_module,
            "direct_key": rope_key_states.detach(),
            "direct_value": value_states.detach(),
            "direct_group": group,
            "head_sim": torch.zeros((bsz, h_kv), device=rope_key_states.device, dtype=torch.float32),
            "cache_kwargs": cache_kwargs,
        }, current_budget_t

    vision_hist_mask, text_hist_mask = _safe_masks(
        is_vision_token_mask=is_vision_token_mask,
        is_text_token_mask=is_text_token_mask,
        bsz=bsz,
        history_len=history_len,
        device=importance_history.device,
    )
    vision_index = vision_hist_mask.unsqueeze(1).expand(bsz, h_kv, history_len)
    text_index = text_hist_mask.unsqueeze(1).expand(bsz, h_kv, history_len)

    zero = torch.zeros((), device=importance_history.device, dtype=importance_history.dtype)
    vision_prefer_score = torch.where(vision_index, importance_history, zero).sum(dim=-1)
    text_prefer_score = torch.where(text_index, importance_history, zero).sum(dim=-1)
    total_prefer_score = vision_prefer_score + text_prefer_score
    vision_budget = (current_budget_t.reshape(-1, 1).float() * (vision_prefer_score / total_prefer_score)).int()
    text_budget = (current_budget_t.reshape(-1, 1).float() - vision_budget).int()

    min_value = torch.finfo(importance_history.dtype).min
    vision_attn_score = torch.where(vision_index, importance_history, torch.full_like(importance_history, min_value))
    text_attn_score = torch.where(text_index, importance_history, torch.full_like(importance_history, min_value))

    next_budget = _next_layer_budget_like_madakv(
        current_budget=current_budget_t,
        q_len=seq_len,
        history_len=history_len,
        budget_ratio=float(config.madakv_kvrepack_budget_ratio),
        threshold=float(config.madakv_kvrepack_threshold),
        layer_idx=int(attn_module.layer_idx),
        num_hidden_layers=int(config.num_hidden_layers),
        vision_attn_score=vision_attn_score,
        text_attn_score=text_attn_score,
        vision_prefer_score=vision_prefer_score,
        text_prefer_score=text_prefer_score,
    )

    history_raw = raw_key_states[:, :, :history_len, :]
    history_rope = rope_key_states[:, :, :history_len, :]
    history_value = value_states[:, :, :history_len, :]
    recent_key = rope_key_states[:, :, history_len:, :].detach()
    recent_value = value_states[:, :, history_len:, :].detach()

    heads: List[List[Dict]] = []
    head_sim = torch.zeros((bsz, h_kv), device=importance_history.device, dtype=torch.float32)
    compress_ratio = float(config.madakv_kvrepack_compress_ratio)

    for bidx in range(bsz):
        batch_heads: List[Dict] = []
        for hidx in range(h_kv):
            v_final = int(vision_budget[bidx, hidx].item())
            t_final = int(text_budget[bidx, hidx].item())
            vision_cand = _select_candidates(
                score=vision_attn_score[bidx, hidx],
                raw_hist=history_raw[bidx, hidx],
                rope_hist=history_rope[bidx, hidx],
                value_hist=history_value[bidx, hidx],
                imp_hist=importance_history[bidx, hidx],
                final_len=v_final,
                compress_ratio=compress_ratio,
            )
            text_cand = _select_candidates(
                score=text_attn_score[bidx, hidx],
                raw_hist=history_raw[bidx, hidx],
                rope_hist=history_rope[bidx, hidx],
                value_hist=history_value[bidx, hidx],
                imp_hist=importance_history[bidx, hidx],
                final_len=t_final,
                compress_ratio=compress_ratio,
            )

            sim_terms = []
            sim_weights = []
            sim_space = (config.madakv_simhead_sim_space or "raw").lower()
            for cand in (vision_cand, text_cand):
                source = cand["raw"] if sim_space == "raw" else cand["rope"]
                n = int(source.shape[0])
                if n >= 2:
                    sim_terms.append(_adjacent_similarity(source) * float(n - 1))
                    sim_weights.append(float(n - 1))
            if sim_terms:
                head_sim[bidx, hidx] = torch.stack(sim_terms).sum() / max(sum(sim_weights), 1.0)

            batch_heads.append({"vision": vision_cand, "text": text_cand})
        heads.append(batch_heads)

    return {
        "layer_idx": attn_module.layer_idx,
        "module": attn_module,
        "heads": heads,
        "head_sim": head_sim.detach(),
        "final_history": int(current_budget_t.reshape(-1)[0].item()),
        "history_len": history_len,
        "seq_len": seq_len,
        "recent_key": recent_key,
        "recent_value": recent_value,
        "cos": cos.detach(),
        "sin": sin.detach(),
        "mrope_section": None,
        "cache_kwargs": cache_kwargs,
    }, next_budget


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
def _finalize_prompt_cache(config, past_key_value: Cache) -> None:
    records: List[Dict] = getattr(config, "_madakv_kvrepack_records", [])
    if not records or past_key_value is None:
        return

    masks = _global_merge_masks(records, float(config.madakv_simhead_merge_ratio))
    total_heads = 0
    merge_heads = 0
    layer_counts: List[int] = []
    sim_values = []

    for rec, merge_mask in zip(records, masks):
        module = rec["module"]
        layer_idx = int(rec["layer_idx"])
        if "direct_key" in rec:
            module._madakv_kvrepack_group_size = rec["direct_group"]
            if rec["direct_key"].shape[-2] != rec["direct_group"].shape[-1]:
                raise AssertionError(
                    "MadaKV-KVRepack direct-cache budget mismatch: "
                    f"cache_len={rec['direct_key'].shape[-2]} group_len={rec['direct_group'].shape[-1]}"
                )
            past_key_value.update(rec["direct_key"], rec["direct_value"], layer_idx, rec["cache_kwargs"])
            layer_counts.append(0)
            continue

        heads = rec["heads"]
        recent_key = rec["recent_key"]
        recent_value = rec["recent_value"]
        final_history = int(rec["final_history"])
        bsz, h_kv, recent_len, head_dim = recent_key.shape
        out_len = final_history + recent_len

        out_k = torch.zeros((bsz, h_kv, out_len, head_dim), device=recent_key.device, dtype=recent_key.dtype)
        out_v = torch.zeros((bsz, h_kv, out_len, head_dim), device=recent_value.device, dtype=recent_value.dtype)
        group = torch.zeros((bsz, h_kv, out_len), device=recent_key.device, dtype=torch.float32)

        layer_merge = int(merge_mask.sum().item())
        layer_counts.append(layer_merge)
        total_heads += int(merge_mask.numel())
        merge_heads += layer_merge
        sim_values.append(rec["head_sim"].reshape(-1).float())

        for bidx in range(bsz):
            for hidx in range(h_kv):
                enable_merge = bool(merge_mask[bidx, hidx].item())
                head_rec = heads[bidx][hidx]
                parts_k = []
                parts_v = []
                parts_g = []
                for name in ("vision", "text"):
                    hist_k, hist_v, hist_g = _compress_candidate(
                        cand=head_rec[name],
                        enable_merge=enable_merge,
                        cos=rec["cos"],
                        sin=rec["sin"],
                        batch_idx=bidx,
                        alpha=float(config.madakv_kvrepack_alpha),
                        mrope_section=rec["mrope_section"],
                    )
                    parts_k.append(hist_k)
                    parts_v.append(hist_v)
                    parts_g.append(hist_g)

                if parts_k:
                    hist_k = torch.cat(parts_k, dim=0)
                    hist_v = torch.cat(parts_v, dim=0)
                    hist_g = torch.cat(parts_g, dim=0)
                else:
                    hist_k = out_k.new_empty((0, head_dim))
                    hist_v = out_v.new_empty((0, head_dim))
                    hist_g = group.new_empty((0,))

                hist_len = min(int(hist_k.shape[0]), final_history)
                if hist_len > 0:
                    out_k[bidx, hidx, :hist_len].copy_(hist_k[:hist_len])
                    out_v[bidx, hidx, :hist_len].copy_(hist_v[:hist_len])
                    group[bidx, hidx, :hist_len].copy_(hist_g[:hist_len])
                if hist_len < final_history:
                    group[bidx, hidx, hist_len:final_history].fill_(0.0)
                if recent_len > 0:
                    out_k[bidx, hidx, final_history:].copy_(recent_key[bidx, hidx])
                    out_v[bidx, hidx, final_history:].copy_(recent_value[bidx, hidx])
                    group[bidx, hidx, final_history:].fill_(1.0)

        expected_len = final_history + recent_len
        if out_k.shape[-2] != expected_len or out_v.shape[-2] != expected_len:
            raise AssertionError(
                "MadaKV-KVRepack budget mismatch: "
                f"cache_len={out_k.shape[-2]} expected={expected_len}"
            )
        module._madakv_kvrepack_group_size = group
        past_key_value.update(out_k, out_v, layer_idx, rec["cache_kwargs"])

    if total_heads > 0:
        sims = torch.cat(sim_values) if sim_values else torch.zeros((1,))
        ratio = float(merge_heads) / float(total_heads)
        first = records[0]
        print(
            "[MadaKV-KVRepack] modality-local adjacent-sim gate: "
            f"merge_heads={merge_heads}/{total_heads} merge_ratio={ratio:.4f} "
            f"target_ratio={float(config.madakv_simhead_merge_ratio):.4f} "
            f"budget_ratio={float(config.madakv_kvrepack_budget_ratio):.4f} "
            f"compress_ratio={float(config.madakv_kvrepack_compress_ratio):.4f} "
            f"final_history={int(first.get('final_history', 0))} "
            f"alpha={float(config.madakv_kvrepack_alpha):.4f} "
            f"sim_space={config.madakv_simhead_sim_space} "
            f"score_correction=none attention=flash "
            f"sim_min/mean/max={float(sims.min().item()):.4f}/{float(sims.mean().item()):.4f}/{float(sims.max().item()):.4f} "
            f"merge_heads_per_layer={layer_counts}"
        )

    config._madakv_kvrepack_records = []

def _update_decode_group_size(self, key_states: torch.Tensor, q_len: int) -> torch.Tensor:
    bsz, h_kv, cache_len, _ = key_states.shape
    old_group = getattr(self, "_madakv_kvrepack_group_size", None)
    append_group = torch.ones((bsz, h_kv, q_len), device=key_states.device, dtype=torch.float32)
    if old_group is not None and old_group.shape[:2] == (bsz, h_kv) and old_group.shape[-1] == cache_len - q_len:
        group_size = torch.cat([old_group.to(device=key_states.device), append_group], dim=-1)
    else:
        group_size = torch.ones((bsz, h_kv, cache_len), device=key_states.device, dtype=torch.float32)
    self._madakv_kvrepack_group_size = group_size
    return group_size


def qwen_flash_attn2_forward_MadaKVKVRepack(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    is_vision_token_mask: Optional[torch.Tensor] = None,
    is_text_token_mask: Optional[torch.Tensor] = None,
    current_budget: Optional[torch.Tensor] = None,
    **kwargs,
):
    init_madakv_kvrepack(self)
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
        raise ValueError("Qwen2 MadaKV-KVRepack requires shared position_embeddings from Qwen2Model.")
    else:
        cos, sin = position_embeddings

    query_states_rope, key_states_rope = apply_rotary_pos_emb(query_states, raw_key_states, cos, sin)

    is_prefill = q_len != 1
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    if is_prefill:
        if self.layer_idx == 0:
            self.config._madakv_kvrepack_records = []

        importance_history = _compute_madakv_importance(
            query_states_rope=query_states_rope,
            key_states_rope=key_states_rope,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            window_size=self.config.madakv_kvrepack_window_size,
            gqa_func=self.config.madakv_kvrepack_gqa_func,
        )

        if use_cache and past_key_value is not None:
            record, current_budget = _build_layer_record(
                attn_module=self,
                raw_key_states=raw_key_states,
                rope_key_states=key_states_rope,
                value_states=value_states,
                importance_history=importance_history,
                is_vision_token_mask=is_vision_token_mask,
                is_text_token_mask=is_text_token_mask,
                current_budget=current_budget,
                cos=cos,
                sin=sin,
                cache_kwargs=cache_kwargs,
            )
            self.config._madakv_kvrepack_records.append(record)
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

        query_fa = query_states_rope.transpose(1, 2)
        key_fa = key_for_attn.transpose(1, 2)
        value_fa = value_for_attn.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_fa,
            key_fa,
            value_fa,
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
        if past_key_value is not None and use_cache:
            key_cache, value_cache = past_key_value.update(key_states_rope, value_states, self.layer_idx, cache_kwargs)
            _update_decode_group_size(self, key_cache, q_len)
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

        query_fa = query_states_rope.transpose(1, 2)
        key_fa = key_rep.transpose(1, 2)
        value_fa = value_rep.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_fa,
            key_fa,
            value_fa,
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
    return attn_output, attn_weights, current_budget

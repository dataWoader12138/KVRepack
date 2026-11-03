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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from .lookm_headsplit_kvrepack_utils import compute_lookm_importance, repeat_kv_len


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
        return pos_emb[:, batch_idx, positions, :]
    if pos_emb.ndim == 3 and pos_emb.shape[0] == 3:
        return pos_emb[:, positions, :]
    if pos_emb.ndim == 3:
        return pos_emb[batch_idx, positions, :].unsqueeze(0).expand(3, -1, -1)
    if pos_emb.ndim == 2:
        return pos_emb[positions, :].unsqueeze(0).expand(3, -1, -1)
    raise ValueError(f"Unsupported mRoPE embedding shape: {tuple(pos_emb.shape)}")


def _apply_mrope_to_merged_keys(
    raw_k: torch.Tensor,
    cos_sel: torch.Tensor,
    sin_sel: torch.Tensor,
    mrope_section,
) -> torch.Tensor:
    mrope_section = list(mrope_section) * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos_sel.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin_sel.split(mrope_section, dim=-1))], dim=-1)
    return raw_k * cos + _rotate_half(raw_k) * sin


def init_lookm_kvrepack(self) -> None:
    if not hasattr(self.config, "lookm_kvrepack_window_size"):
        self.config.lookm_kvrepack_window_size = _env_int("LOOKM_KVREPACK_WINDOW_SIZE", 32)
    if not hasattr(self.config, "lookm_kvrepack_budget_ratio"):
        self.config.lookm_kvrepack_budget_ratio = _env_float(
            "LOOKM_KVREPACK_BUDGET_RATIO",
            _env_float("BUDGET_RATIO", 0.10),
        )
    if not hasattr(self.config, "lookm_kvrepack_text_priority"):
        self.config.lookm_kvrepack_text_priority = _env_bool("LOOKM_KVREPACK_TEXT_PRIORITY", True)
    if not hasattr(self.config, "lookm_kvrepack_gqa_func"):
        self.config.lookm_kvrepack_gqa_func = os.getenv("LOOKM_KVREPACK_GQA_FUNC", "mean")
    if not hasattr(self.config, "lookm_kvrepack_compress_ratio"):
        self.config.lookm_kvrepack_compress_ratio = _env_float(
            "LOOKM_KVREPACK_COMPRESS_RATIO",
            _env_float("LOOKM_KVREPACK_DYNAMIC_COMPRESS_RATIO", 0.80),
        )
    if not hasattr(self.config, "lookm_kvrepack_alpha"):
        self.config.lookm_kvrepack_alpha = _env_float(
            "LOOKM_KVREPACK_K_DIFF_ALPHA",
            _env_float("K_DIFF_ALPHA", 2.0),
        )
    if not hasattr(self.config, "lookm_simhead_merge_ratio"):
        self.config.lookm_simhead_merge_ratio = _env_float("LOOKM_SIMHEAD_MERGE_RATIO", 0.30)
    if not hasattr(self.config, "lookm_simhead_sim_space"):
        self.config.lookm_simhead_sim_space = os.getenv("LOOKM_SIMHEAD_SIM_SPACE", "raw").lower()
    if not hasattr(self.config, "lookm_kvrepack_score_correction"):
        self.config.lookm_kvrepack_score_correction = os.getenv("LOOKM_KVREPACK_SCORE_CORRECTION", "log_count")
    if not hasattr(self.config, "lookm_kvrepack_enable"):
        self.config.lookm_kvrepack_enable = _env_bool("LOOKM_KVREPACK_ENABLE", True)

    if not hasattr(self, "_flash_attn_uses_top_left_mask"):
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()


def _history_budget(seq_len: int, history_len: int, ratio: float) -> int:
    budget = min(max(int(seq_len * float(ratio)), 0), int(history_len))
    if history_len > 0 and ratio > 0 and budget == 0:
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
    if seq_len == 0:
        empty_g = torch.empty((0,), device=raw_k_seq.device, dtype=torch.float32)
        return rope_k_seq, v_seq, empty_g
    if target_len >= seq_len:
        group = torch.ones((seq_len,), device=raw_k_seq.device, dtype=torch.float32)
        return rope_k_seq.contiguous(), v_seq.contiguous(), group
    if target_len <= 0:
        target_len = 1

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


@torch.no_grad()
def _build_layer_record(
    attn_module,
    raw_key_states: torch.Tensor,
    rope_key_states: torch.Tensor,
    value_states: torch.Tensor,
    importance_history: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_kwargs: Dict,
):
    config = attn_module.config
    bsz, h_kv, seq_len, head_dim = rope_key_states.shape
    window = min(max(0, int(config.lookm_kvrepack_window_size)), seq_len)
    history_len = max(0, seq_len - window)
    final_history = _history_budget(seq_len, history_len, float(config.lookm_kvrepack_budget_ratio))
    candidate_history = _candidate_len(final_history, history_len, float(config.lookm_kvrepack_compress_ratio))

    if history_len == 0 or final_history <= 0 or not bool(config.lookm_kvrepack_enable):
        group = torch.ones((bsz, h_kv, seq_len), device=rope_key_states.device, dtype=torch.float32)
        return {
            "layer_idx": attn_module.layer_idx,
            "module": attn_module,
            "direct_key": rope_key_states.detach(),
            "direct_value": value_states.detach(),
            "direct_group": group,
            "head_sim": torch.zeros((bsz, h_kv), device=rope_key_states.device, dtype=torch.float32),
            "cache_kwargs": cache_kwargs,
        }

    cand_raw = torch.zeros((bsz, h_kv, candidate_history, head_dim), device=raw_key_states.device, dtype=raw_key_states.dtype)
    cand_rope = torch.zeros_like(cand_raw, dtype=rope_key_states.dtype)
    cand_value = torch.zeros((bsz, h_kv, candidate_history, head_dim), device=value_states.device, dtype=value_states.dtype)
    cand_imp = torch.zeros((bsz, h_kv, candidate_history), device=importance_history.device, dtype=importance_history.dtype)
    cand_idx = torch.zeros((bsz, h_kv, candidate_history), device=importance_history.device, dtype=torch.long)
    head_sim = torch.zeros((bsz, h_kv), device=importance_history.device, dtype=torch.float32)

    history_raw = raw_key_states[:, :, :history_len, :]
    history_rope = rope_key_states[:, :, :history_len, :]
    history_value = value_states[:, :, :history_len, :]

    for bidx in range(bsz):
        for hidx in range(h_kv):
            idx = _topk_sorted(importance_history[bidx, hidx], candidate_history)
            cur_len = int(idx.numel())
            if cur_len == 0:
                continue
            cand_idx[bidx, hidx, :cur_len].copy_(idx)
            cand_raw[bidx, hidx, :cur_len].copy_(history_raw[bidx, hidx].index_select(0, idx))
            cand_rope[bidx, hidx, :cur_len].copy_(history_rope[bidx, hidx].index_select(0, idx))
            cand_value[bidx, hidx, :cur_len].copy_(history_value[bidx, hidx].index_select(0, idx))
            cand_imp[bidx, hidx, :cur_len].copy_(importance_history[bidx, hidx].index_select(0, idx))
            sim_source = cand_raw if config.lookm_simhead_sim_space == "raw" else cand_rope
            head_sim[bidx, hidx] = _adjacent_similarity(sim_source[bidx, hidx, :cur_len])

    recent_key = rope_key_states[:, :, history_len:, :].detach()
    recent_value = value_states[:, :, history_len:, :].detach()

    return {
        "layer_idx": attn_module.layer_idx,
        "module": attn_module,
        "candidate_raw": cand_raw.detach(),
        "candidate_rope": cand_rope.detach(),
        "candidate_value": cand_value.detach(),
        "candidate_imp": cand_imp.detach(),
        "candidate_idx": cand_idx.detach(),
        "candidate_history": candidate_history,
        "final_history": final_history,
        "history_len": history_len,
        "seq_len": seq_len,
        "recent_key": recent_key,
        "recent_value": recent_value,
        "head_sim": head_sim.detach(),
        "cos": cos.detach(),
        "sin": sin.detach(),
        "mrope_section": attn_module.rope_scaling["mrope_section"],
        "cache_kwargs": cache_kwargs,
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
def _finalize_prompt_cache(config, past_key_value: Cache) -> None:
    records: List[Dict] = getattr(config, "_lookm_kvrepack_records", [])
    if not records or past_key_value is None:
        return

    masks = _global_merge_masks(records, float(config.lookm_simhead_merge_ratio))
    total_heads = 0
    merge_heads = 0
    layer_counts: List[int] = []
    sim_values = []

    for rec, merge_mask in zip(records, masks):
        module = rec["module"]
        layer_idx = int(rec["layer_idx"])
        if "direct_key" in rec:
            module._lookm_kvrepack_group_size = rec["direct_group"]
            past_key_value.update(rec["direct_key"], rec["direct_value"], layer_idx, rec["cache_kwargs"])
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
        group = torch.zeros((bsz, h_kv, out_len), device=cand_rope.device, dtype=torch.float32)

        layer_merge = int(merge_mask.sum().item())
        layer_counts.append(layer_merge)
        total_heads += int(merge_mask.numel())
        merge_heads += layer_merge
        sim_values.append(rec["head_sim"].reshape(-1).float())

        for bidx in range(bsz):
            for hidx in range(h_kv):
                if bool(merge_mask[bidx, hidx].item()):
                    hist_k, hist_v, hist_g = _merge_adjacent_raw_reapply(
                        raw_k_seq=cand_raw[bidx, hidx, :candidate_history],
                        rope_k_seq=cand_rope[bidx, hidx, :candidate_history],
                        v_seq=cand_value[bidx, hidx, :candidate_history],
                        importance=cand_imp[bidx, hidx, :candidate_history],
                        target_len=final_history,
                        positions=cand_idx[bidx, hidx, :candidate_history],
                        cos=rec["cos"],
                        sin=rec["sin"],
                        batch_idx=bidx,
                        alpha=float(config.lookm_kvrepack_alpha),
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
                    group[bidx, hidx, :final_history].copy_(hist_g[:final_history])
                if recent_len > 0:
                    out_k[bidx, hidx, final_history:].copy_(recent_key[bidx, hidx])
                    out_v[bidx, hidx, final_history:].copy_(recent_value[bidx, hidx])
                    group[bidx, hidx, final_history:].fill_(1.0)

        module._lookm_kvrepack_group_size = group
        past_key_value.update(out_k, out_v, layer_idx, rec["cache_kwargs"])

    if total_heads > 0:
        sims = torch.cat(sim_values) if sim_values else torch.zeros((1,))
        ratio = float(merge_heads) / float(total_heads)
        first = records[0]
        print(
            "[LookM-KVRepack] adjacent-sim global gate: "
            f"merge_heads={merge_heads}/{total_heads} merge_ratio={ratio:.4f} "
            f"target_ratio={float(config.lookm_simhead_merge_ratio):.4f} "
            f"budget_ratio={float(config.lookm_kvrepack_budget_ratio):.4f} "
            f"compress_ratio={float(config.lookm_kvrepack_compress_ratio):.4f} "
            f"final_history={int(first.get('final_history', 0))} "
            f"candidate_history={int(first.get('candidate_history', 0))} "
            f"alpha={float(config.lookm_kvrepack_alpha):.4f} "
            f"sim_min/mean/max={float(sims.min().item()):.4f}/{float(sims.mean().item()):.4f}/{float(sims.max().item()):.4f} "
            f"merge_heads_per_layer={layer_counts}"
        )

    config._lookm_kvrepack_records = []


def _apply_decode_score_correction(scores: torch.Tensor, group_size: torch.Tensor, mode: str) -> torch.Tensor:
    mode = (mode or "none").lower()
    if mode in {"none", "off", "0"}:
        return scores
    if mode != "log_count":
        raise ValueError(f"Unsupported LOOKM_KVREPACK_SCORE_CORRECTION={mode}")
    return scores + torch.log(group_size.clamp_min(1.0)).unsqueeze(-2).to(dtype=scores.dtype)


def _update_decode_group_size(self, key_states: torch.Tensor, q_len: int) -> torch.Tensor:
    bsz, h_kv, cache_len, _ = key_states.shape
    old_group = getattr(self, "_lookm_kvrepack_group_size", None)
    append_group = torch.ones((bsz, h_kv, q_len), device=key_states.device, dtype=torch.float32)
    if old_group is not None and old_group.shape[:2] == (bsz, h_kv) and old_group.shape[-1] == cache_len - q_len:
        group_size = torch.cat([old_group.to(device=key_states.device), append_group], dim=-1)
    else:
        group_size = torch.ones((bsz, h_kv, cache_len), device=key_states.device, dtype=torch.float32)
    self._lookm_kvrepack_group_size = group_size
    return group_size


def qwen_flash_attn2_forward_LookMKVRepack(
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
):
    del is_vision_token_mask
    init_lookm_kvrepack(self)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    raw_key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    raw_key_states = raw_key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states_rope, key_states_rope = apply_multimodal_rotary_pos_emb(
        query_states,
        raw_key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )

    is_prefill = q_len != 1
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    if is_prefill:
        if self.layer_idx == 0:
            self.config._lookm_kvrepack_records = []

        importance_history = compute_lookm_importance(
            query_states_rope=query_states_rope,
            key_states_rope=key_states_rope,
            text_token_mask=is_text_token_mask,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            window_size=self.config.lookm_kvrepack_window_size,
            text_priority=self.config.lookm_kvrepack_text_priority,
            gqa_func=self.config.lookm_kvrepack_gqa_func,
        )

        if use_cache and past_key_value is not None:
            record = _build_layer_record(
                attn_module=self,
                raw_key_states=raw_key_states,
                rope_key_states=key_states_rope,
                value_states=value_states,
                importance_history=importance_history,
                cos=cos,
                sin=sin,
                cache_kwargs=cache_kwargs,
            )
            self.config._lookm_kvrepack_records.append(record)
            if self.layer_idx == self.config.num_hidden_layers - 1:
                _finalize_prompt_cache(self.config, past_key_value)

        key_for_attn = repeat_kv(key_states_rope, self.num_key_value_groups)
        value_for_attn = repeat_kv(value_states, self.num_key_value_groups)
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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_weights = None

    else:
        if past_key_value is not None and use_cache:
            key_cache, value_cache = past_key_value.update(key_states_rope, value_states, self.layer_idx, cache_kwargs)
            group_size = _update_decode_group_size(self, key_cache, q_len)
        else:
            key_cache = key_states_rope
            value_cache = value_states
            group_size = torch.ones(key_cache.size()[:-1], device=key_cache.device, dtype=torch.float32)

        key_rep = repeat_kv(key_cache, self.num_key_value_groups)
        value_rep = repeat_kv(value_cache, self.num_key_value_groups)
        group_rep = repeat_kv_len(group_size, self.num_key_value_groups).to(device=key_rep.device, dtype=torch.float32)

        scores = torch.matmul(query_states_rope, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = _apply_decode_score_correction(scores, group_rep, self.config.lookm_kvrepack_score_correction)
        scores = scores.masked_fill(group_rep.unsqueeze(-2) <= 0, torch.finfo(scores.dtype).min)
        if attention_mask is not None and attention_mask.ndim == 4:
            scores = scores + attention_mask[:, :, :, : key_rep.shape[-2]]
        scores = torch.nan_to_num(
            scores,
            nan=0.0,
            posinf=torch.finfo(scores.dtype).max,
            neginf=torch.finfo(scores.dtype).min,
        )
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states_rope.dtype)
        attn_output = torch.matmul(probs, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_weights = None if not output_attentions else probs

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value

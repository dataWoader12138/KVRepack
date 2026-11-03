import math
import os
from typing import Optional, Tuple

import torch

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextModel,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
    flash_attn_supports_top_left_mask,
)

from .randomkv_utils import (
    RandomKVCache,
    RandomDecoupledKV,
    repeat_kv_len,
    repeat_kv_map,
)

ORIGINAL_QWEN_TEXT_FORWARD = Qwen2_5_VLTextModel.forward


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


def ensure_randomkv_cache(past_key_values):
    if past_key_values is None:
        return RandomKVCache()

    if isinstance(past_key_values, RandomKVCache):
        return past_key_values

    if isinstance(past_key_values, tuple):
        return RandomKVCache.from_legacy_cache(past_key_values)

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return RandomKVCache.from_legacy_cache(past_key_values.to_legacy_cache())

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def randomkv_qwen_forward(self, *args, **kwargs):
    past_key_values = kwargs.get("past_key_values", None)
    use_cache = kwargs.get("use_cache", self.config.use_cache)

    if use_cache and not self.training:
        kwargs["past_key_values"] = ensure_randomkv_cache(past_key_values)

    return ORIGINAL_QWEN_TEXT_FORWARD(self, *args, **kwargs)


def init_randomkv(self):
    if not hasattr(self.config, "randomkv_start_size"):
        self.config.randomkv_start_size = int(getattr(self.config, "start_size", 4))

    if not hasattr(self.config, "randomkv_recent_size"):
        self.config.randomkv_recent_size = int(getattr(self.config, "recent_size", 64))

    if not hasattr(self.config, "randomkv_enable_compression"):
        self.config.randomkv_enable_compression = True

    env_budget = _safe_get_env_int("BUDGET")
    default_abs_budget = int(getattr(self.config, "max_capacity_prompt", 512))
    if not hasattr(self.config, "randomkv_cache_size"):
        self.config.randomkv_cache_size = env_budget if env_budget is not None else default_abs_budget

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = RandomDecoupledKV(
            start_size=self.config.randomkv_start_size,
            recent_size=self.config.randomkv_recent_size,
            cache_size=self.config.randomkv_cache_size,
            enabled=self.config.randomkv_enable_compression,
        )

    if not hasattr(self, "_flash_attn_uses_top_left_mask"):
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()


def resolve_budget(self, full_prompt_len: int) -> int:
    # 优先 BUDGET，其次 BUDGET_RATIO，否则 fallback 到 config
    env_budget = _safe_get_env_int("BUDGET")
    env_ratio = _safe_get_env_float("BUDGET_RATIO")

    min_budget = self.config.randomkv_start_size + self.config.randomkv_recent_size + 1

    if env_budget is not None:
        return max(min_budget, min(env_budget, full_prompt_len))

    if env_ratio is not None:
        return max(min_budget, min(int(full_prompt_len * env_ratio), full_prompt_len))

    abs_budget = int(getattr(self.config, "randomkv_cache_size", 512))
    return max(min_budget, min(abs_budget, full_prompt_len))


def qwen_flash_attn2_forward_RandomKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[RandomKVCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    init_randomkv(self)
    past_key_value = ensure_randomkv_cache(past_key_value)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states_rope, key_states_rope = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    is_prefill = q_len != 1

    if is_prefill:
        if self.layer_idx == 0:
            self.kv_cluster.begin_prompt()

        budget = resolve_budget(self, q_len)
        self.kv_cluster.set_cache_size(budget)

        key_comp, value_comp, k_len_comp, v_len_comp, k_to_v_map, k_anchor_pos = self.kv_cluster.compress(
            key_states=key_states_rope,
            value_states=value_states,
        )

        if self.layer_idx == self.config.num_hidden_layers - 1:
            ratio = self.kv_cluster.end_prompt_ratio()
            print(f"[RandomKV] overall_kv_ratio={ratio:.6f}")

        if use_cache:
            past_key_value.set_layer_cache(
                self.layer_idx,
                key_comp,
                value_comp,
                k_len_comp,
                v_len_comp,
                k_to_v_map,
                k_anchor_pos,
            )

        # prefill 当前步输出仍走 full KV
        q_fa = query_states_rope.transpose(1, 2)
        k_fa = key_states_rope.transpose(1, 2)
        v_fa = value_states.transpose(1, 2)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

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

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_weights = None

    else:
        if use_cache:
            key_cache, value_cache, k_len_cache, v_len_cache, k_to_v_map = past_key_value.update_singletons(
                key_states=key_states_rope,
                value_states=value_states,
                layer_idx=self.layer_idx,
                cache_position=cache_position,
            )
        else:
            key_cache = key_states_rope
            value_cache = value_states
            k_len_cache = torch.ones(key_cache.size()[:-1], device=key_cache.device, dtype=key_cache.dtype)
            v_len_cache = torch.ones(value_cache.size()[:-1], device=value_cache.device, dtype=value_cache.dtype)
            tk = key_cache.shape[-2]
            tv = value_cache.shape[-2]
            k_to_v_map = torch.eye(tk, device=key_cache.device, dtype=key_cache.dtype).view(1, 1, tk, tv).expand(bsz, self.num_key_value_heads, tk, tv)

        key_rep = repeat_kv(key_cache, self.num_key_value_groups)
        value_rep = repeat_kv(value_cache, self.num_key_value_groups)
        map_rep = repeat_kv_map(k_to_v_map, self.num_key_value_groups)
        v_len_rep = repeat_kv_len(v_len_cache, self.num_key_value_groups).to(torch.float32)

        scores = torch.matmul(query_states_rope, key_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_rep.shape[-2]]
            scores = scores + causal_mask

        if query_states_rope.dtype == torch.float16:
            scores = torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)

        probs_k = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states_rope.dtype)

        probs_v = torch.einsum("bhqk,bhkv->bhqv", probs_k.to(torch.float32), map_rep.to(torch.float32))

        # 因为 V 是 singleton 采样，v_len 基本是 1，这里保留统一写法
        denom = (probs_v * v_len_rep.unsqueeze(-2)).sum(dim=-1, keepdim=True).clamp_min(1e-20)
        probs_v = (probs_v / denom).to(query_states_rope.dtype)

        attn_output = torch.matmul(probs_v, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

        attn_weights = None if not output_attentions else probs_v

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value
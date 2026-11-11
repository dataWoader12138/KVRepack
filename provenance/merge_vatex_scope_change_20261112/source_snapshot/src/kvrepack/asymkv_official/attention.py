from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from .cache import AsymKVOfficialCache


def repeat_cardinality(cardinality: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return cardinality
    batch, kv_heads, sequence_length = cardinality.shape
    expanded = cardinality[:, :, None, :].expand(
        batch, kv_heads, groups, sequence_length
    )
    return expanded.reshape(batch, kv_heads * groups, sequence_length)


def official_lma_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cardinality: torch.Tensor,
    current_query_length: int,
    dropout: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Released AsymKV locally merged attention, including max_score - 30."""
    attention_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attention_weights = attention_weights / math.sqrt(query_states.shape[-1])

    key_length = key_states.shape[-2]
    history_length = key_length - current_query_length
    if current_query_length > 1:
        query_index = torch.arange(current_query_length, device=attention_weights.device)
        key_index = torch.arange(key_length, device=attention_weights.device)
        causal_mask = key_index.unsqueeze(0) > (
            history_length + query_index.unsqueeze(1)
        )
        attention_weights = attention_weights.masked_fill(
            causal_mask.view(1, 1, current_query_length, key_length),
            torch.finfo(attention_weights.dtype).min,
        )

    max_score = attention_weights.max(dim=-1, keepdim=True).values - 30
    attention_weights = torch.exp(
        (attention_weights - max_score).to(torch.float32)
    )
    denominator = (
        attention_weights * cardinality.unsqueeze(-2)
    ).sum(dim=-1, keepdim=True)
    attention_weights = attention_weights / denominator
    attention_weights = attention_weights.to(query_states.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout, training=training)
    output = torch.matmul(attention_weights, value_states)
    return output, attention_weights


def qwen2_5_vl_asymkv_official_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[AsymKVOfficialCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    if not isinstance(past_key_value, AsymKVOfficialCache):
        original_forward = getattr(
            self.__class__, "_asymkv_official_original_forward", None
        )
        if original_forward is None:
            raise TypeError("Official AsymKV attention requires AsymKVOfficialCache")
        return original_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    if position_embeddings is None:
        raise ValueError("Qwen2.5-VL official AsymKV requires MRoPE embeddings")

    batch_size, query_length, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        batch_size, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )

    if use_cache:
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx
        )
        cardinality = past_key_value.get_cardinality(self.layer_idx)
    else:
        cardinality = torch.ones(
            key_states.shape[:-1], dtype=key_states.dtype, device=key_states.device
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    cardinality = repeat_cardinality(cardinality, self.num_key_value_groups)

    attention_output, attention_weights = official_lma_attention(
        query_states,
        key_states,
        value_states,
        cardinality,
        current_query_length=query_length,
        dropout=self.attention_dropout,
        training=self.training,
    )
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(
        batch_size, query_length, self.hidden_size
    )
    attention_output = self.o_proj(attention_output)
    return (
        attention_output,
        attention_weights if output_attentions else None,
        past_key_value,
    )

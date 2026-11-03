import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import os
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import (
    logging,
    is_torchdynamo_compiling,
)
import math
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from kvrepack.baseline_utils import DynamicCacheSplitHeadFlatten
import torch.nn.functional as F
import einops as ein

logger = logging.get_logger(__name__)

# Qwen2.5_VL_Model General
def qwen_vl_forward_cross_attn(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
        The tensors corresponding to the input videos. Pixel values can be obtained using
        [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
        [`Qwen2_5_VLImageProcessor`] for processing videos.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #[cross-attn]
    is_text_token_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
    is_vision_token_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum()
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id #[bsz, seq_len]
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            # [LOOK_M]
            # 更新视觉 token 掩码
            is_vision_token_mask = is_vision_token_mask | mask

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds) #[bsz, seq_len, hidden_size]

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum()
            n_video_features = video_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)
            # [cross-attn]
            # 更新我们的全局视觉 token 掩码
            is_vision_token_mask = is_vision_token_mask | mask
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
            is_not_padding_mask = (attention_mask == 1) # 不是padding token
            is_not_vision_mask = ~is_vision_token_mask
            # text token同时满足：不是padding且不是vision
            is_text_token_mask = is_not_vision_mask & is_not_padding_mask

    if position_ids is None:
        attention_mask_2d = attention_mask
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask_2d = torch.diagonal(attention_mask_2d[:, 0], dim1=1, dim2=2)
            attention_mask_2d = attention_mask_2d / torch.finfo(attention_mask_2d.dtype).min
            attention_mask_2d = (1.0 - attention_mask_2d).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask_2d,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        #[cross-attn]
        is_vision_token_mask=is_vision_token_mask,
        is_text_token_mask=is_text_token_mask
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()

# TextModel Unflatten General
def cross_attn_qwen_forward_unflatten(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        #[cross-attn]
        is_vision_token_mask: Optional[torch.Tensor] = None,
        is_text_token_mask: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                #[cross-attn]
                is_vision_token_mask=is_vision_token_mask,
                is_text_token_mask=is_text_token_mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

# TextModel Flatten General
def cross_attn_qwen_forward_flatten(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None, # [adakv]
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    #[cross-attn]
    is_vision_token_mask: Optional[torch.Tensor] = None,
    is_text_token_mask: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    
    return_legacy_cache = False
    if (
        use_cache and not (type(past_key_values) == DynamicCacheSplitHeadFlatten) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        # return_legacy_cache = True  #! For 4.41 version.
        past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds
    # import pdb; pdb.set_trace()

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                is_vision_token_mask=is_vision_token_mask,
                is_text_token_mask=is_text_token_mask
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

# Decoderlayers General
def cross_attn_qwen_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    #[cross-attn]
    is_vision_token_mask: Optional[torch.Tensor] = None,
    is_text_token_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        #[cross-attn]
        is_vision_token_mask=is_vision_token_mask,
        is_text_token_mask=is_text_token_mask
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

# MadaKV: TextModel with layer budget
def cross_attn_qwen_forward_unflatten_MadaKV(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        #[cross-attn]
        is_vision_token_mask: Optional[torch.Tensor] = None,
        is_text_token_mask: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    current_budget = None
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs, current_budget = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                #[cross-attn]
                is_vision_token_mask=is_vision_token_mask,
                is_text_token_mask=is_text_token_mask,
                current_budget=current_budget
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

# MadaKV: Decoderlayer with layer budget
def cross_attn_qwen_decoder_forward_MadaKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    #[cross-attn]
    is_vision_token_mask: Optional[torch.Tensor] = None,
    is_text_token_mask: Optional[torch.Tensor] = None,
    current_budget: Optional[torch.Tensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value, current_budget = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        #[cross-attn]
        is_vision_token_mask=is_vision_token_mask,
        is_text_token_mask=is_text_token_mask,
        current_budget=current_budget # None or [bsz]
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs, current_budget

def index_points(points, idx):
    """
    Sample features following the index.
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def cluster_and_merge(x, cluster_num):
    """
    Cluster tokens based on density peaks and merge them.
    """
    B, N, C = x.shape
    if N == 0 or cluster_num <= 0:
        return torch.empty(B, 0, C, device=x.device)
    if cluster_num >= N:
        return x

    distance = torch.cdist(x, x, p=2)
    dist_matrix = distance / (C ** 0.5)
    
    dist_nearest, _ = torch.topk(dist_matrix, k=min(cluster_num, N), dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)

    dist_matrix_indexed = index_points(dist_matrix, index_down)
    idx_cluster = dist_matrix_indexed.argmin(dim=1)

    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    token_weight = x.new_ones(B, N, 1)
    idx = idx_cluster + torch.arange(B, device=x.device)[:, None] * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)
    
    return x_merged

# def attention_aware_prune_and_merge(self, original_vision_embeds, input_ids, attention_mask, placeholder_token_id):
#     """
#     实现了基于V-T注意力的剪枝与合并算法。
#     返回:
#         - final_vision_embeds (torch.Tensor): 最终的、经过压缩和合并的视觉特征。
#         - final_keep_indices (torch.Tensor): 在原始视觉特征中，被直接保留的token的索引。
#         - num_merged_tokens (int): 新创建的合并token的数量。
#     """
#     active_mask = (attention_mask[0] == 1)
#     original_active_ids = input_ids[0][active_mask]
#     vision_token_indices_in_active = torch.where(original_active_ids == placeholder_token_id)[0]
#     vision_start_idx = vision_token_indices_in_active[0].item()
#     vision_end_idx = vision_token_indices_in_active[-1].item() + 1
    
#     text_ids_in_active = torch.cat([original_active_ids[:vision_start_idx], original_active_ids[vision_end_idx:]])
#     text_embeds = self.get_input_embeddings()(text_ids_in_active)
#     sim_matrix_v_t = torch.matmul(original_vision_embeds, text_embeds.t())
#     m_v_t = sim_matrix_v_t.softmax(dim=1).mean(dim=0)
#     important_text_indices = torch.where(m_v_t > m_v_t.mean())[0]
#     if len(important_text_indices) == 0:
#         important_text_indices = torch.arange(len(text_embeds), device=self.device)
#     important_text_embeds = text_embeds[important_text_indices]
#     relation_vis_text = torch.matmul(original_vision_embeds, important_text_embeds.t())
#     vision_importance_scores = relation_vis_text.max(dim=1).values
#     keep_ratio = getattr(self.config, "vision_keep_ratio", 0.9)
#     merge_ratio_stage2 = getattr(self.config, "vision_merge_stage2_ratio", 0.1)
#     cluster_ratio = getattr(self.config, "vision_cluster_ratio", 0)

#     budget = int(float(os.getenv('BUDGET_RATIO')) * 2 * input_ids.shape[1])
#     r1 = merge_ratio_stage2
#     r2 = cluster_ratio
#     temp = int((budget - r1 * r2 *(vision_end_idx - vision_start_idx)) / (1 - r1 * r2))
#     temp = max(temp, 0)

#     # num_to_keep = min(int(original_vision_embeds.shape[0] * keep_ratio), temp)
#     num_to_keep = int(original_vision_embeds.shape[0] * keep_ratio)
#     pred_score_vis = torch.zeros(original_vision_embeds.shape[0], device=self.device)
#     topk_indices = vision_importance_scores.topk(k=num_to_keep).indices
#     pred_score_vis[topk_indices] = 1

#     merge_token_idx_stage1 = torch.where(pred_score_vis == 0)[0]
#     if len(merge_token_idx_stage1) > 0 and merge_ratio_stage2 > 0:
#         sparse_token_relations = vision_importance_scores[merge_token_idx_stage1]
#         num_to_merge_in_stage2 = int(len(merge_token_idx_stage1) * merge_ratio_stage2) + 1
#         topk_relative_indices = sparse_token_relations.topk(k=min(num_to_merge_in_stage2, len(sparse_token_relations))).indices
#         merge_token_stage2_indices_absolute = merge_token_idx_stage1[topk_relative_indices]
#         tokens_to_merge = original_vision_embeds[merge_token_stage2_indices_absolute].unsqueeze(0)
#         cluster_num = int(tokens_to_merge.shape[1] * cluster_ratio) + 1
#         merged_vision_embeds = cluster_and_merge(tokens_to_merge, cluster_num).squeeze(0)
#         final_keep_indices = torch.where(pred_score_vis == 1)[0]
#     else:
#         merged_vision_embeds = torch.empty(0, original_vision_embeds.shape[-1], device=self.device).to(dtype=kept_vision_embeds.dtype)
#         final_keep_indices = torch.where(pred_score_vis == 1)[0]
    
#     num_merged_tokens = merged_vision_embeds.shape[0]

#     kept_vision_embeds = original_vision_embeds[final_keep_indices]
#     num_kept_before_merge_point = len(kept_vision_embeds) // 2
#     kept_part1 = kept_vision_embeds[:num_kept_before_merge_point]
#     kept_part2 = kept_vision_embeds[num_kept_before_merge_point:]
#     final_vision_embeds = torch.cat([kept_part1, merged_vision_embeds, kept_part2], dim=0)

#     return final_vision_embeds, final_keep_indices, num_merged_tokens

# @torch.no_grad()
# def _prepare_inputs_with_sparsevlm_pruning(
#     self,
#     input_ids,
#     attention_mask,
#     # 统一接收视觉输入
#     pixel_values=None,
#     image_grid_thw=None,
#     pixel_values_videos=None,
#     video_grid_thw=None,
#     second_per_grid_ts=None,
# ):
#     # 统一接口
#     if pixel_values is not None:
#         modality, placeholder_token_id = "image", self.config.image_token_id
#         original_vision_embeds = self.get_image_features(pixel_values, image_grid_thw)
#         grid_thw = image_grid_thw
#     elif pixel_values_videos is not None:
#         modality, placeholder_token_id = "video", self.config.video_token_id
#         original_vision_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
#         grid_thw = video_grid_thw
#     else:
#         raise ValueError("No visual inputs provided.")

#     # 1. 应用剪枝算法，获取所有必要信息
#     compressed_vision_embeds, vision_keep_indices, num_merged_tokens = attention_aware_prune_and_merge(
#         self, original_vision_embeds, input_ids, attention_mask, placeholder_token_id
#     )
#     print(compressed_vision_embeds.shape)

#     batch_size = input_ids.shape[0]
#     active_mask = (attention_mask[0] == 1)
#     original_active_ids = input_ids[0][active_mask]
    
#     vision_placeholder_idx = torch.where(original_active_ids == placeholder_token_id)[0][0]
#     original_num_vision_tokens = original_vision_embeds.shape[0]
    
#     text_part1_ids = original_active_ids[:vision_placeholder_idx]
#     text_part2_ids = original_active_ids[vision_placeholder_idx + original_num_vision_tokens:]
#     text_embeds_part1 = self.get_input_embeddings()(text_part1_ids)
#     text_embeds_part2 = self.get_input_embeddings()(text_part2_ids)
    
#     final_inputs_embeds = torch.cat(
#         [text_embeds_part1, compressed_vision_embeds, text_embeds_part2], dim=0
#     ).unsqueeze(0)

#     full_position_ids, rope_deltas = self.get_rope_index(
#         input_ids=input_ids,
#         image_grid_thw=grid_thw if modality == "image" else None,
#         video_grid_thw=grid_thw if modality == "video" else None,
#         second_per_grid_ts=second_per_grid_ts if modality == "video" else None,
#     )
    
#     pos_ids_1d = full_position_ids.view(batch_size, 3, -1)[:, 0, :]
#     active_pos_ids = pos_ids_1d[:, active_mask]

#     text_pos_ids_part1 = active_pos_ids[:, :vision_placeholder_idx]
#     text_pos_ids_part2 = active_pos_ids[:, vision_placeholder_idx + original_num_vision_tokens:]

#     full_vision_pos_ids = pos_ids_1d[:, torch.where(input_ids[0] == placeholder_token_id)[0]]
#     kept_vision_pos_ids = full_vision_pos_ids[:, vision_keep_indices]

#     num_kept_before_merge_point = len(vision_keep_indices) // 2
#     kept_pos_part1 = kept_vision_pos_ids[:, :num_kept_before_merge_point]
#     kept_pos_part2 = kept_vision_pos_ids[:, num_kept_before_merge_point:]
    
#     if kept_pos_part1.numel() > 0:
#         merged_pos_start = kept_pos_part1.max() + 1
#     elif text_pos_ids_part1.numel() > 0:
#         merged_pos_start = text_pos_ids_part1.max() + 1
#     else:
#         merged_pos_start = 0

#     merged_pos_ids = torch.arange(
#         merged_pos_start, merged_pos_start + num_merged_tokens, device=self.device, dtype=torch.long
#     ).unsqueeze(0).expand(batch_size, -1)

#     final_vision_pos_ids = torch.cat([kept_pos_part1, merged_pos_ids, kept_pos_part2], dim=1)
    
#     final_position_ids = torch.cat([text_pos_ids_part1, final_vision_pos_ids, text_pos_ids_part2], dim=1)

#     new_seq_len = final_inputs_embeds.shape[1]
#     new_attention_mask = torch.ones(batch_size, new_seq_len, device=self.device, dtype=attention_mask.dtype)

#     len_text1 = text_part1_ids.shape[0]
#     len_vision_pruned = compressed_vision_embeds.shape[0]
    
#     vision_mask = torch.zeros_like(new_attention_mask, dtype=torch.bool)
#     text_mask = torch.ones_like(new_attention_mask, dtype=torch.bool)
#     vision_mask[:, len_text1 : len_text1 + len_vision_pruned] = True
#     text_mask[:, len_text1 : len_text1 + len_vision_pruned] = False

#     return (
#         final_inputs_embeds,
#         new_attention_mask,
#         final_position_ids,
#         vision_mask,
#         text_mask,
#         rope_deltas,
#     )

def attention_aware_prune_and_merge(
    self, 
    vision_embeds_chunk: torch.Tensor, 
    context_text_embeds: torch.Tensor
):
    """
    实现了基于V-T注意力的剪枝与合并算法，作用于单个视觉块。
    支持自适应是否进行合并操作。
    
    Args:
        vision_embeds_chunk (torch.Tensor): 当前需要处理的、单个图像/视频的原始视觉特征块。
        context_text_embeds (torch.Tensor): 用于计算相似度的上下文文本特征（通常是整个prompt中的所有文本特征）。

    Returns:
        - final_vision_embeds (torch.Tensor): 最终的、经过压缩和合并的视觉特征。
        - final_keep_indices_relative (torch.Tensor): 在输入的 vision_embeds_chunk 中，被直接保留的token的相对索引。
        - num_merged_tokens (int): 新创建的合并token的数量（若未合并则为0）。
    """
    if vision_embeds_chunk.shape[0] == 0:
        return vision_embeds_chunk, torch.tensor([], device=self.device, dtype=torch.long), 0
        
    # 1. 计算视觉-文本相似度以确定视觉token重要性
    # 使用传入的上下文文本嵌入
    sim_matrix_v_t = torch.matmul(vision_embeds_chunk, context_text_embeds.t())
    
    # 筛选出重要的文本token作为更精确的参照
    m_v_t = sim_matrix_v_t.softmax(dim=1).mean(dim=0)
    important_text_indices = torch.where(m_v_t > m_v_t.mean())[0]
    if len(important_text_indices) == 0:
        # 如果没有高于平均值的，则使用所有文本token
        important_text_embeds = context_text_embeds
    else:
        important_text_embeds = context_text_embeds[important_text_indices]
        
    # 计算最终的重要性分数
    relation_vis_text = torch.matmul(vision_embeds_chunk, important_text_embeds.t())
    vision_importance_scores = relation_vis_text.max(dim=1).values
    
    # 2. 获取配置参数
    keep_ratio = getattr(self.config, "vision_keep_ratio", 0.8)
    merge_ratio_stage2 = getattr(self.config, "vision_merge_stage2_ratio", 1.0)
    cluster_ratio = getattr(self.config, "vision_cluster_ratio", 0.1)
    enable_merge = getattr(self.config, "enable_vision_merge", True)
    
    # 3. 确定要保留的视觉token数量
    num_to_keep = int(vision_embeds_chunk.shape[0] * keep_ratio)
    
    # 标记要保留和要合并的视觉token
    pred_score_vis = torch.zeros(vision_embeds_chunk.shape[0], device=self.device)
    if num_to_keep > 0:
        topk_indices = vision_importance_scores.topk(k=num_to_keep).indices
        pred_score_vis[topk_indices] = 1
    
    # final_keep_indices_relative 是相对于当前 chunk 的索引
    final_keep_indices_relative = torch.where(pred_score_vis == 1)[0]
    kept_vision_embeds = vision_embeds_chunk[final_keep_indices_relative]
    
    merge_token_idx_stage1_relative = torch.where(pred_score_vis == 0)[0]
    num_merged_tokens = 0
    
    # 4. 自适应判断是否需要执行合并
    should_merge = (enable_merge and 
                    len(merge_token_idx_stage1_relative) > 0 and 
                    merge_ratio_stage2 > 0 and 
                    cluster_ratio > 0)
    
    if should_merge:
        sparse_token_relations = vision_importance_scores[merge_token_idx_stage1_relative]
        num_to_merge_in_stage2 = int(len(merge_token_idx_stage1_relative) * merge_ratio_stage2)
        
        if num_to_merge_in_stage2 > 0:
            topk_relative_indices = sparse_token_relations.topk(k=num_to_merge_in_stage2).indices
            merge_token_stage2_indices_absolute = merge_token_idx_stage1_relative[topk_relative_indices]
            tokens_to_merge = vision_embeds_chunk[merge_token_stage2_indices_absolute].unsqueeze(0)
            
            cluster_num = max(int(tokens_to_merge.shape[1] * cluster_ratio), 1)
            cluster_num = min(cluster_num, tokens_to_merge.shape[1])
            
            merged_vision_embeds = cluster_and_merge(tokens_to_merge, cluster_num).squeeze(0)
            merged_vision_embeds = merged_vision_embeds.to(dtype=vision_embeds_chunk.dtype)
            num_merged_tokens = merged_vision_embeds.shape[0]

            # 5. 构建最终的视觉嵌入
            num_kept_before_merge_point = len(kept_vision_embeds) // 2
            kept_part1 = kept_vision_embeds[:num_kept_before_merge_point]
            kept_part2 = kept_vision_embeds[num_kept_before_merge_point:]
            final_vision_embeds = torch.cat([kept_part1, merged_vision_embeds, kept_part2], dim=0)
        else: # 没有token进入第二阶段合并
            final_vision_embeds = kept_vision_embeds
    else: # 不执行合并
        final_vision_embeds = kept_vision_embeds

    # 确保返回的索引是排序的，以保持空间顺序
    final_keep_indices_relative = torch.sort(final_keep_indices_relative).values
    return final_vision_embeds, final_keep_indices_relative, num_merged_tokens
"""Private Qwen multimodal helpers shared by the selected AsymKV port."""

from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F

def _freeze_model_parameters(model) -> None:
    if getattr(model, "_asymkv_paper_parameters_frozen", False):
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._asymkv_paper_parameters_frozen = True


def _scatter_features(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    token_id: int,
    features: torch.Tensor,
    feature_name: str,
) -> torch.Tensor:
    token_mask = input_ids == token_id
    token_count = int(token_mask.sum().item())
    feature_count = int(features.shape[0])
    if token_count != feature_count:
        raise ValueError(
            f"{feature_name} token/feature mismatch: {token_count} tokens, "
            f"{feature_count} features"
        )
    expanded_mask = token_mask.unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(
        expanded_mask,
        features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
    )


@torch.no_grad()
def _prepare_multimodal_inputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pixel_values: Optional[torch.Tensor],
    pixel_values_videos: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    second_per_grid_ts: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.shape[0] != 1:
        raise ValueError("AsymKV lmms-eval integration currently requires batch_size=1")
    if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
        raise ValueError("AsymKV currently requires an unpadded batch; use batch_size=1")

    backbone = model.model
    inputs_embeds = model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_features = backbone.get_image_features(pixel_values, image_grid_thw)
        inputs_embeds = _scatter_features(
            input_ids,
            inputs_embeds,
            model.config.image_token_id,
            image_features,
            "image",
        )

    if pixel_values_videos is not None:
        video_features = backbone.get_video_features(pixel_values_videos, video_grid_thw)
        inputs_embeds = _scatter_features(
            input_ids,
            inputs_embeds,
            model.config.video_token_id,
            video_features,
            "video",
        )

    position_ids, rope_deltas = backbone.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
    )
    backbone.rope_deltas = rope_deltas
    return inputs_embeds, position_ids, rope_deltas


def _chunk_language_model_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> Optional[torch.Tensor]:
    if token_ids.shape[1] < 2:
        return None
    shifted_logits = logits[:, :-1, :].to(torch.float32).contiguous()
    shifted_labels = token_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
    )


def _normalize_eos_ids(value: Optional[Union[int, Sequence[int]]]) -> Set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, torch.Tensor):
        return {int(item) for item in value.flatten().tolist()}
    return {int(item) for item in value}

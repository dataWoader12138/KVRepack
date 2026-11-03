from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from kvrepack._asymkv_helpers import (
    _chunk_language_model_loss,
    _freeze_model_parameters,
    _normalize_eos_ids,
    _prepare_multimodal_inputs,
)

from .cache import AsymKVOfficialCache
from .compression import compress_cache_like_upstream


UPSTREAM_ASYMKV_COMMIT = "771ce4db0292f7e8c9519edf37f7e161beb2c103"


@dataclass(frozen=True)
class AsymKVOfficialRuntimeConfig:
    budget_ratio: float
    absolute_cache_size: Optional[int]
    start_size: int
    chunk_size: int
    audit_path: Optional[str]

    @classmethod
    def from_environment(cls) -> "AsymKVOfficialRuntimeConfig":
        absolute = os.getenv("ASYMKV_OFFICIAL_CACHE_SIZE")
        return cls(
            budget_ratio=float(os.getenv("BUDGET_RATIO", "0.10")),
            absolute_cache_size=(
                int(absolute) if absolute not in {None, ""} else None
            ),
            start_size=int(os.getenv("ASYMKV_START_SIZE", "32")),
            chunk_size=int(os.getenv("ASYMKV_CHUNK_SIZE", "512")),
            audit_path=os.getenv("ASYMKV_AUDIT_PATH") or None,
        )

    def resolve_cache_size(self, prompt_length: int) -> int:
        cache_size = (
            self.absolute_cache_size
            if self.absolute_cache_size is not None
            else int(math.floor(prompt_length * self.budget_ratio))
        )
        cache_size = min(cache_size, prompt_length)
        if cache_size <= self.start_size:
            raise ValueError(
                f"Official AsymKV cache size {cache_size} must exceed its "
                f"{self.start_size}-token sink"
            )
        return cache_size


def _append_audit(path: Optional[str], record: Dict[str, object]) -> None:
    if path is None:
        return
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _forward_chunk(
    model,
    cache: AsymKVOfficialCache,
    chunk_embeddings: torch.Tensor,
    chunk_positions: torch.Tensor,
    cache_position: torch.Tensor,
) -> torch.Tensor:
    cache.begin_step(capture_gradients=False)
    with torch.no_grad():
        outputs = model.model.language_model(
            input_ids=None,
            inputs_embeds=chunk_embeddings,
            position_ids=chunk_positions,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
        )
        logits = model.lm_head(outputs.last_hidden_state)
    cache.detach_()
    return logits[:, -1, :].detach()


def _update_running_hessian_like_upstream(
    current_gradients: Optional[List[torch.Tensor]],
    previous_hessian: Optional[List[torch.Tensor]],
    *,
    delta_idx: int,
    cache_size: int,
    chunk_size: int,
) -> Optional[List[torch.Tensor]]:
    if current_gradients is None:
        return None
    if (
        previous_hessian is None
        or previous_hessian[0].shape[-2] < cache_size + chunk_size
    ):
        return current_gradients

    updated: List[torch.Tensor] = []
    for current, previous in zip(current_gradients, previous_hessian):
        previous_length = previous.shape[-2]
        head = (
            (1.0 / delta_idx) * current[:, :, :previous_length, :]
            + ((delta_idx - 1.0) / delta_idx) * previous
        )
        tail = current[:, :, previous_length:, :]
        updated.append(torch.cat((head, tail), dim=2))
    return updated


def _probe_and_compress_history(
    model,
    cache: AsymKVOfficialCache,
    chunk_ids: torch.Tensor,
    chunk_embeddings: torch.Tensor,
    chunk_positions: torch.Tensor,
    cache_position: torch.Tensor,
    previous_hessian: Optional[List[torch.Tensor]],
    *,
    delta_idx: int,
    cache_size: int,
    runtime: AsymKVOfficialRuntimeConfig,
    num_key_value_groups: int,
) -> Tuple[Optional[List[torch.Tensor]], Dict[str, object]]:
    history_lengths = cache.physical_lengths()
    probe_cache = cache.clone_for_probe()
    with torch.enable_grad():
        outputs = model.model.language_model(
            input_ids=None,
            inputs_embeds=chunk_embeddings,
            position_ids=chunk_positions,
            attention_mask=None,
            past_key_values=probe_cache,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
        )
        logits = model.lm_head(outputs.last_hidden_state)
        loss = _chunk_language_model_loss(logits, chunk_ids)
        fisher_available = loss is not None and bool(torch.isfinite(loss).item())
        if loss is not None:
            loss.backward()

    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Official AsymKV probe did not return attentions")
    history_attentions = [
        attention[:, :, :, : history_lengths[layer_idx]].detach()
        for layer_idx, attention in enumerate(attentions)
    ]
    current_gradients: Optional[List[torch.Tensor]]
    if fisher_available:
        gradients = [
            probe_cache.history_gradient(layer_idx, history_length)
            for layer_idx, history_length in enumerate(history_lengths)
        ]
        current_gradients = (
            [gradient for gradient in gradients if gradient is not None]
            if all(gradient is not None for gradient in gradients)
            else None
        )
    else:
        current_gradients = None

    running_hessian = _update_running_hessian_like_upstream(
        current_gradients,
        previous_hessian,
        delta_idx=delta_idx,
        cache_size=cache_size,
        chunk_size=runtime.chunk_size,
    )
    running_hessian, compression_report = compress_cache_like_upstream(
        cache,
        history_attentions,
        num_key_value_groups,
        running_hessian,
        cache_size=cache_size,
        start_size=runtime.start_size,
    )
    compression_report.update(
        {
            "fisher_available": fisher_available,
            "probe_chunk_length": int(chunk_ids.shape[1]),
        }
    )
    del outputs, logits, loss, probe_cache, attentions, history_attentions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return running_hessian, compression_report


def _prefill_like_upstream(
    model,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    runtime: AsymKVOfficialRuntimeConfig,
) -> Tuple[
    torch.Tensor,
    AsymKVOfficialCache,
    int,
    List[Dict[str, object]],
]:
    prompt_length = int(input_ids.shape[1])
    cache_size = runtime.resolve_cache_size(prompt_length)
    cache = AsymKVOfficialCache()
    running_hessian: Optional[List[torch.Tensor]] = None
    compression_events: List[Dict[str, object]] = []
    final_logits: Optional[torch.Tensor] = None
    delta_idx = 0
    num_key_value_groups = int(
        model.model.language_model.layers[0].self_attn.num_key_value_groups
    )

    # The stop value intentionally matches upstream range(0, context_length - 1, 512).
    for chunk_start in range(0, prompt_length - 1, runtime.chunk_size):
        chunk_end = min(prompt_length, chunk_start + runtime.chunk_size)
        chunk_ids = input_ids[:, chunk_start:chunk_end]
        chunk_embeddings = inputs_embeds[:, chunk_start:chunk_end, :]
        chunk_positions = position_ids[:, :, chunk_start:chunk_end]
        cache_position = torch.arange(
            chunk_start,
            chunk_end,
            dtype=torch.long,
            device=input_ids.device,
        )

        # The released driver probes/compresses only for a complete input
        # window; a final partial chunk is appended without another pass.
        if (
            cache.get_seq_length() > 0
            and chunk_ids.shape[1] == runtime.chunk_size
        ):
            running_hessian, event = _probe_and_compress_history(
                model,
                cache,
                chunk_ids,
                chunk_embeddings,
                chunk_positions,
                cache_position,
                running_hessian,
                delta_idx=delta_idx,
                cache_size=cache_size,
                runtime=runtime,
                num_key_value_groups=num_key_value_groups,
            )
            event.update({"chunk_start": chunk_start, "chunk_end": chunk_end})
            compression_events.append(event)

        final_logits = _forward_chunk(
            model,
            cache,
            chunk_embeddings,
            chunk_positions,
            cache_position,
        )
        delta_idx += 1

    if final_logits is None:
        raise RuntimeError("Official AsymKV prefill did not process any prompt chunk")
    return final_logits, cache, cache_size, compression_events


@torch.no_grad()
def _decode_one_token(
    model,
    token_ids: torch.Tensor,
    cache: AsymKVOfficialCache,
    rope_deltas: torch.Tensor,
) -> torch.Tensor:
    logical_position = cache.original_seen_tokens
    token_embeddings = model.get_input_embeddings()(token_ids)
    scalar_positions = (
        torch.full(
            (token_ids.shape[0], 1),
            logical_position,
            dtype=torch.long,
            device=token_ids.device,
        )
        + rope_deltas.to(device=token_ids.device, dtype=torch.long)
    )
    position_ids = scalar_positions.unsqueeze(0).expand(3, -1, -1)
    cache.begin_step(capture_gradients=False)
    outputs = model.model.language_model(
        input_ids=None,
        inputs_embeds=token_embeddings,
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=cache,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=torch.tensor(
            [logical_position], dtype=torch.long, device=token_ids.device
        ),
    )
    cache.detach_()
    return model.lm_head(outputs.last_hidden_state[:, -1, :])


def asymkv_official_generate(
    self, input_ids: Optional[torch.Tensor] = None, **kwargs
) -> torch.Tensor:
    if input_ids is None:
        raise ValueError("Official AsymKV generation requires input_ids")
    if kwargs.pop("inputs_embeds", None) is not None:
        raise ValueError("Official AsymKV expects token and vision inputs")
    if bool(kwargs.pop("do_sample", False)):
        raise ValueError("Official AsymKV supports greedy decoding only")
    if int(kwargs.pop("num_beams", 1)) != 1:
        raise ValueError("Official AsymKV supports num_beams=1 only")
    if not bool(kwargs.pop("use_cache", True)):
        raise ValueError("Official AsymKV requires use_cache=True")

    max_new_tokens = int(kwargs.pop("max_new_tokens", 20))
    eos_ids: Set[int] = _normalize_eos_ids(
        kwargs.pop("eos_token_id", self.config.eos_token_id)
    )
    kwargs.pop("pad_token_id", None)
    kwargs.pop("temperature", None)
    kwargs.pop("top_p", None)
    kwargs.pop("return_dict_in_generate", None)
    kwargs.pop("output_scores", None)

    attention_mask = kwargs.pop("attention_mask", None)
    pixel_values = kwargs.pop("pixel_values", None)
    pixel_values_videos = kwargs.pop("pixel_values_videos", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    video_grid_thw = kwargs.pop("video_grid_thw", None)
    second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
    kwargs.pop("position_ids", None)
    kwargs.pop("cache_position", None)
    unsupported = {
        key: value
        for key, value in kwargs.items()
        if value is not None and key not in {"synced_gpus"}
    }
    if unsupported:
        raise ValueError(
            f"Unsupported official AsymKV generation arguments: {sorted(unsupported)}"
        )

    runtime = AsymKVOfficialRuntimeConfig.from_environment()
    if runtime.chunk_size <= 0:
        raise ValueError("ASYMKV_CHUNK_SIZE must be positive")
    self.eval()
    _freeze_model_parameters(self)
    inputs_embeds, position_ids, rope_deltas = _prepare_multimodal_inputs(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts,
    )
    del pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    logits, cache, cache_size, compression_events = _prefill_like_upstream(
        self, input_ids, inputs_embeds, position_ids, runtime
    )
    del inputs_embeds, position_ids
    prefill_physical_lengths = cache.physical_lengths()
    prefill_logical_tokens = cache.original_seen_tokens

    generated: List[torch.Tensor] = []
    if max_new_tokens > 0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token)
        # Upstream checks EOS only after a token has passed through decode.
        for _ in range(max_new_tokens - 1):
            logits = _decode_one_token(self, next_token, cache, rope_deltas)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            if all(int(token) in eos_ids for token in next_token.flatten().tolist()):
                break

    generated_ids = (
        torch.cat(generated, dim=1)
        if generated
        else input_ids.new_empty((input_ids.shape[0], 0))
    )
    output_ids = torch.cat((input_ids, generated_ids), dim=1)
    sample_index = int(getattr(self, "_asymkv_official_sample_index", 0))
    self._asymkv_official_sample_index = sample_index + 1
    _append_audit(
        runtime.audit_path,
        {
            "method": "AsymKV-official",
            "upstream_commit": UPSTREAM_ASYMKV_COMMIT,
            "sample_index": sample_index,
            "prompt_length": int(input_ids.shape[1]),
            "configured_cache_size": cache_size,
            "prefill_physical_lengths": prefill_physical_lengths,
            "prefill_logical_tokens": prefill_logical_tokens,
            "cache_physical_lengths_after_generation": cache.physical_lengths(),
            "cache_logical_tokens_after_generation": cache.original_seen_tokens,
            "generated_tokens": len(generated),
            "compression_events": compression_events,
            "runtime": asdict(runtime),
        },
    )
    return output_ids

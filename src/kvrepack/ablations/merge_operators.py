"""Merge-only operators on the host's fixed KVRepack groups.

This module does not select tokens, heads, groups, or positions.  All inputs
describe the exact groups formed at the host's existing merge insertion point.
The returned K is *raw*, before the host reapplies mRoPE at its group anchor.
This is intentionally a controlled aggregation transplant, not a full LookM or
AsymKV implementation.

Sources pinned for the transplanted arithmetic:

* LOOK-M ecf0f51a9c416c2d85e47faf2638502f01a6d748, TextPivotMerge_LayerWise:
  https://github.com/SUSTechBruce/LOOK-M/blob/ecf0f51a9c416c2d85e47faf2638502f01a6d748/LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py
  Pair-average followed by scatter mean with include_self=True is preserved.
  Its global cosine assignment, text-prior pruning, and native post-RoPE K
  space are not transplanted into this fixed-group experiment.
* AsymKV 771ce4db0292f7e8c9519edf37f7e161beb2c103:
  https://github.com/the-scale-lab/Asymkv/blob/771ce4db0292f7e8c9519edf37f7e161beb2c103/asymkv/method/asymkv.py
  Fisher-weighted K and summed V reuse the project's existing group reducer.
  Cardinality is part of the result and MUST accompany locally merged
  attention; summed V must never be read using an uncorrected softmax.

The prompt-only Fisher probe is a separate prerequisite, whose runtime must be
included for the AsymKV arm.  It uses uncompressed native FlashAttention on the
same frozen weights and an LM objective on observed prompt tokens.  It does
not import AsymKV's chunked generation or its pair-selection policy.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
from types import MethodType
from typing import Mapping, Optional

import torch
import torch.nn.functional as F


VARIANTS = ("kvrepack", "lookm", "asymkv")


def reduce_fixed_groups(
    variant: str,
    raw_k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    group_ids: torch.Tensor,
    group_size: torch.Tensor,
    start_mask: torch.Tensor,
    fisher_gradient: Optional[torch.Tensor] = None,
):
    """Reduce one native KV head without changing the supplied groups.

    Inputs are K/V ``[N,D]``, group IDs ``[N]``, sizes ``[G]``, and the
    Boolean ``[N]`` start mask.  ``fisher_gradient`` is dL/d(raw K), also
    ``[N,D]``.  Returns raw K ``[G,D]``, V ``[G,D]``, and counts ``[G]``.

    The caller guarantees that contiguous groups are numbered from zero and
    ``start_mask`` selects their first members.  Value-dependent validation is
    deliberately left to the untimed preflight, avoiding device synchronizes
    in each head's timed merge.  The operator contains no CPU tensor copies,
    token loops, or audit hashing.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown fixed-group merge operator: {variant}")
    if raw_k_seq.ndim != 2 or raw_k_seq.shape != v_seq.shape:
        raise ValueError("Expected matching single-head K/V tensors [N,D]")
    n, dim = raw_k_seq.shape
    if group_ids.shape != (n,) or start_mask.shape != (n,):
        raise ValueError("group_ids and start_mask must each have N entries")
    if group_ids.dtype != torch.long or start_mask.dtype != torch.bool:
        raise TypeError("Require int64 group IDs and a Boolean start mask")
    if group_size.ndim != 1 or not 0 < group_size.shape[0] <= n:
        raise ValueError("Expected a nonempty group_size vector with G <= N")
    if any(x.device != raw_k_seq.device for x in (v_seq, group_ids, group_size, start_mask)):
        raise ValueError("All fixed-group inputs must already share a device")
    count = int(group_size.shape[0])

    if variant == "asymkv":
        if fisher_gradient is None:
            raise ValueError("AsymKV aggregation requires a real raw-K Fisher gradient")
        if fisher_gradient.device != raw_k_seq.device:
            raise ValueError("Fisher gradients must already be on the K device")
        from kvrepack.hybrid_asymkv import reduce_groups_asymkv

        # All groups are independent.  Process this whole host-selected head in
        # one batch instead of inheriting the official generation chunk size.
        return reduce_groups_asymkv(
            raw_k_seq, v_seq, fisher_gradient, group_ids, count, group_size,
            reduce_batch_size=n,
        )

    if variant == "lookm":
        # Match the source's order and dtype: each non-pivot first averages
        # with its *original* pivot, then scatter mean includes that pivot.
        # In particular, a two-token group is 3/4 pivot + 1/4 other, not 1/2.
        pivot_k = raw_k_seq[start_mask]
        pivot_v = v_seq[start_mask]
        nonpivot = ~start_mask
        destinations = group_ids[nonpivot]
        expanded = destinations[:, None].expand(-1, dim)
        source_k = (raw_k_seq[nonpivot] + pivot_k[destinations]) / 2
        source_v = (v_seq[nonpivot] + pivot_v[destinations]) / 2
        merged_k = torch.scatter_reduce(
            pivot_k, 0, expanded, source_k, reduce="mean", include_self=True,
        )
        merged_v = torch.scatter_reduce(
            pivot_v, 0, expanded, source_v, reduce="mean", include_self=True,
        )
        return merged_k, merged_v, group_size.to(torch.float32)

    # Same FP32 index_add/mean as the current KVRepack core, including the cast
    # before the host applies mRoPE.  Kept here for small independent oracles;
    # adapters can leave the native KVRepack reduction entirely untouched.
    denominator = group_size.clamp_min(1.0).unsqueeze(-1)
    key_sum = torch.zeros((count, dim), device=raw_k_seq.device, dtype=torch.float32)
    value_sum = torch.zeros_like(key_sum)
    key_sum.index_add_(0, group_ids, raw_k_seq.float())
    value_sum.index_add_(0, group_ids, v_seq.float())
    return ((key_sum / denominator).to(raw_k_seq.dtype),
            (value_sum / denominator).to(v_seq.dtype),
            group_size.to(torch.float32))


@lru_cache(maxsize=1)
def _native_qwen25_flash_forward():
    """Recover the installed HF method even if its class was monkeypatched.

    Only this forward method is compiled, using the actual installed module's
    globals.  There is no second model or duplicate set of weights.  The
    pinned Transformers 4.52.3 method returns the three values consumed by
    both HF and this project's Hybrid decoder layers.
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as native

    source_path = Path(inspect.getsourcefile(native))
    source = source_path.read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Qwen2_5_VLFlashAttention2")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "forward")
    # The original method has no zero-argument super()/__class__ dependency.
    # Refuse silently compiling a future incompatible HF implementation.
    if any(isinstance(node, ast.Name) and node.id in {"super", "__class__"}
           for node in ast.walk(method)):
        raise RuntimeError("Installed native Qwen FlashAttention method needs a class closure")
    method_source = ast.get_source_segment(source, method)
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(vars(native))
    exec(compile(module, str(source_path) + "<fixed_group_fisher>", "exec"), namespace)
    return namespace["forward"], hashlib.sha256(method_source.encode()).hexdigest()


def _language_layers(model):
    decoder = model.model
    language_model = getattr(decoder, "language_model", decoder)
    if not hasattr(language_model, "layers"):
        raise TypeError("Expected Qwen2.5-VL language_model.layers")
    return list(language_model.layers)


def _collect_raw_key_fisher_full(
    model,
    inputs: Mapping[str, torch.Tensor],
    loss_tail_tokens: int = 32,
    *,
    native_attention_forward=None,
    parameters_already_frozen: bool = False,
):
    """Collect dL/d(raw K) on one uncompressed prompt, with no answer labels.

    Return ``(gradients, stats)``.  Each gradient is ``[B,H_kv,N,D]`` on the
    original device, in layer order.  ``stats`` contains scalar objective and
    provenance metadata, never copies of the prompt or its per-head tensors.

    The objective predicts at most the last ``loss_tail_tokens`` observed
    non-image/video, non-padding prompt tokens.  The final prompt token is
    predicted from its preceding hidden state; no generated token or dataset
    answer participates.  Only these positions pass through ``lm_head``.

    Native, uncompressed FlashAttention temporarily replaces each attention
    instance, including for the custom Hybrid model.  Hooks, parameter flags,
    training flags, use_cache flags, and rope_deltas are restored on failure
    as well as success.  Call outside a generation adapter scope.  Workers
    that have frozen parameters once can set ``parameters_already_frozen``.
    """
    if type(loss_tail_tokens) is not int or loss_tail_tokens <= 0:
        raise ValueError("loss_tail_tokens must be a positive integer")
    if "labels" in inputs or inputs.get("past_key_values") is not None:
        raise ValueError("Fisher probe accepts a fresh observed prompt, without labels or cache")
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("Fisher probe requires one complete prompt [1,N], N >= 2")
    if native_attention_forward is None:
        native_attention_forward, native_sha = _native_qwen25_flash_forward()
    else:
        native_sha = hashlib.sha256(inspect.getsource(native_attention_forward).encode()).hexdigest()
    layers = _language_layers(model)
    if not layers:
        raise ValueError("Cannot probe a model without decoder layers")

    valid = torch.ones_like(input_ids, dtype=torch.bool)
    if inputs.get("attention_mask") is not None:
        mask = inputs["attention_mask"]
        if mask.shape != input_ids.shape:
            raise ValueError("Probe attention_mask must be the prompt's 2D padding mask")
        valid &= mask.bool()
    for name in ("image_token_id", "video_token_id"):
        value = getattr(model.config, name, None)
        if value is not None:
            valid &= input_ids != int(value)
    # Each selected target must have a real preceding source position too.
    eligible = valid[:, 1:]
    if inputs.get("attention_mask") is not None:
        eligible = eligible & inputs["attention_mask"][:, :-1].bool()
    targets = torch.nonzero(eligible[0], as_tuple=False).flatten()[-loss_tail_tokens:] + 1
    if targets.shape[0] == 0:
        raise ValueError("Prompt has no observed text target for the Fisher objective")

    projected = {}
    handles = []
    forwards = []
    parameters = [] if parameters_already_frozen else [(p, p.requires_grad) for p in model.parameters()]
    training = [(module, module.training) for module in model.modules()]
    state_fields = []
    seen = set()
    for obj in (model, model.model):
        for attr in ("rope_deltas",):
            if hasattr(obj, attr):
                state_fields.append((obj, attr, getattr(obj, attr)))
        config = getattr(obj, "config", None)
        if config is not None and id(config) not in seen and hasattr(config, "use_cache"):
            seen.add(id(config))
            state_fields.append((config, "use_cache", config.use_cache))

    def capture(index):
        def hook(_module, _args, output):
            if index in projected:
                raise RuntimeError("Each probe K projection must execute exactly once")
            if not torch.is_tensor(output) or output.ndim != 3:
                raise TypeError("Expected raw k_proj output [B,N,H_kv*D]")
            output.requires_grad_(True)
            output.retain_grad()
            projected[index] = output
            return output
        return hook

    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        model.eval()
        for obj, attr, _ in state_fields:
            if attr == "use_cache":
                setattr(obj, attr, False)
        for index, layer in enumerate(layers):
            attention = layer.self_attn
            # Restore whether an instance override existed, not just the bound
            # callable: removing ours restores the original class dispatch.
            existed = "forward" in vars(attention)
            forwards.append((attention, existed, vars(attention).get("forward")))
            attention.forward = MethodType(native_attention_forward, attention)
            handles.append(attention.k_proj.register_forward_hook(capture(index)))
        arguments = dict(inputs)
        arguments.update(use_cache=False, output_attentions=False,
                         output_hidden_states=False, return_dict=True)
        # Inference-mode generation must not disable the gradient prerequisite.
        with torch.inference_mode(False), torch.enable_grad():
            output = model.model(**arguments)
            hidden = output.last_hidden_state
            logits = model.lm_head(hidden[:, targets - 1, :])
            labels = input_ids[:, targets]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Fisher probe loss is not finite")
            loss.backward()
        if set(projected) != set(range(len(layers))):
            raise RuntimeError("Fisher probe did not capture every language layer")
        gradients = []
        for index, layer in enumerate(layers):
            value = projected[index]
            if value.grad is None:
                raise RuntimeError(f"No Fisher gradient for raw K at layer {index}")
            head_dim = int(layer.self_attn.head_dim)
            batch, sequence, channels = value.shape
            if channels % head_dim:
                raise ValueError("K projection channels do not match the native head dimension")
            gradient = value.grad.detach().view(batch, sequence, channels // head_dim, head_dim)
            gradients.append(gradient.transpose(1, 2).contiguous())
        stats = dict(
            gradient_space="pre_mrope_k_proj",
            objective="next_token_cross_entropy_on_observed_prompt_text_tail",
            loss_tail_tokens_requested=loss_tail_tokens,
            loss_target_tokens=int(targets.shape[0]),
            loss=float(loss.detach().item()),
            layers=len(layers),
            gradient_shapes=[list(gradient.shape) for gradient in gradients],
            native_attention_forward_sha256=native_sha,
            original_asymkv_probe_adaptation=(
                "one uncompressed full-prompt raw-K probe with a text-tail objective; "
                "no iterative chunk averaging, cache compression, answer, or generation"
            ),
        )
        return gradients, stats
    finally:
        for handle in handles:
            handle.remove()
        for attention, existed, original in reversed(forwards):
            if existed:
                attention.forward = original
            else:
                del attention.forward
        for obj, attr, value in state_fields:
            setattr(obj, attr, value)
        for parameter, required in parameters:
            parameter.requires_grad_(required)
        for module, was_training in training:
            module.training = was_training


@lru_cache(maxsize=1)
def _native_qwen25_text_forward():
    """Recover HF's DynamicCache-compatible text forward, without decorators."""
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as native
    source_path = Path(inspect.getsourcefile(native))
    source = source_path.read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Qwen2_5_VLTextModel")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "forward")
    if any(isinstance(node, ast.Name) and node.id in {"super", "__class__"}
           for node in ast.walk(method)):
        raise RuntimeError("Native Qwen text method needs an unsupported class closure")
    method_source = ast.get_source_segment(source, method)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(vars(native))
    exec(compile(module, str(source_path) + "<fixed_group_cached_fisher>", "exec"), namespace)
    return namespace["forward"], hashlib.sha256(method_source.encode()).hexdigest()


def _rebuild_raw_prefix_cache(cache, projected, position_embeddings, layers):
    """Replace equal-valued post-MRoPE K by functions of independent raw leaves.

    This detaches the no-grad prefix and follows the original cache-gradient
    probe idea: past K at different layers are independent inputs, not a full
    backward graph through prefix MLPs/hidden states.  Reapply the very same
    mRoPE embeddings and native helper used during prefix attention.  The
    untimed numerical tests prove that the replacement preserves cache values.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
    if set(projected) != set(range(len(layers))):
        raise RuntimeError("Prefix did not capture every language K projection")
    cos, sin = position_embeddings
    leaves = []
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        value = projected[index]
        batch, sequence, channels = value.shape
        head_dim = int(attention.head_dim)
        if channels % head_dim:
            raise ValueError("Raw prefix projection does not fit native KV heads")
        raw = value.view(batch, sequence, channels // head_dim, head_dim).transpose(1, 2)
        raw = raw.detach().requires_grad_(True)
        # Native code applies the identical helper to Q and K.  Only K's result
        # is retained here; there is no cache-wide rotation-space conversion.
        rotated = apply_multimodal_rotary_pos_emb(
            raw, raw, cos, sin, attention.rope_scaling["mrope_section"],
        )[1]
        if rotated.shape != cache.key_cache[index].shape:
            raise ValueError("Rebuilt raw-prefix K no longer matches the native cache")
        cache.key_cache[index] = rotated
        cache.value_cache[index] = cache.value_cache[index].detach()
        leaves.append(raw)
    return leaves


def collect_raw_key_fisher(
    model,
    inputs: Mapping[str, torch.Tensor],
    loss_tail_tokens: int = 32,
    *,
    native_attention_forward=None,
    parameters_already_frozen: bool = False,
):
    """Memory-bounded prompt-only raw-K Fisher, from a native cached prefix.

    The long multimodal prefix executes once under no_grad, with DynamicCache.
    Prefix raw K becomes independent per-layer leaves and is rotated back into
    an equal-valued post-MRoPE cache.  Only the final contiguous text suffix
    (at most ``loss_tail_tokens+1`` tokens) has a backward graph.  Loss predicts
    its observed tokens 1..end; no generated or dataset answer is used.

    Return ``(gradients, stats)`` with gradients ``[B,H_kv,N,D]`` in the same
    raw projection space as the host merger.  This is a cache-gradient probe,
    not the derivative through every prefix hidden state: the no-grad prefix
    intentionally removes prefix-MLP backward memory, like the official probe.
    Its single suffix objective also differs from official iterative chunks.

    Native attention and native text forwards are temporarily bound to the
    same model instances.  The latter avoids custom HybridCache construction.
    All model state and hooks are restored, including after failures.  This
    temporarily bypasses instance/class compression forwards even inside a
    scoped generation adapter, and restores that adapter before generation.
    Include this prerequisite cost in the Asym arm's end-to-end timing.
    """
    if type(loss_tail_tokens) is not int or loss_tail_tokens <= 0:
        raise ValueError("loss_tail_tokens must be a positive integer")
    if "labels" in inputs or inputs.get("past_key_values") is not None:
        raise ValueError("Fisher probe accepts a fresh observed prompt, without labels or cache")
    ids = inputs.get("input_ids")
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 3:
        raise ValueError("Cached Fisher probe requires an unpadded prompt [1,N], N >= 3")
    n = int(ids.shape[1])
    mask = inputs.get("attention_mask")
    if mask is not None and (mask.shape != ids.shape or not bool(mask.bool().all().item())):
        raise ValueError("Cached Fisher probe supports batch-one unpadded input")
    visual = torch.zeros_like(ids, dtype=torch.bool)
    for name in ("image_token_id", "video_token_id"):
        token_id = getattr(model.config, name, None)
        if token_id is not None:
            visual |= ids == int(token_id)
    visual_indices = torch.nonzero(visual[0], as_tuple=False).flatten()
    last_visual = int(visual_indices[-1].item()) if visual_indices.numel() else -1
    prefix_length = max(1, n - loss_tail_tokens - 1, last_visual + 1)
    tail_length = n - prefix_length
    if tail_length < 2:
        raise ValueError("Cached Fisher probe needs at least two text tokens after the last visual token")

    if native_attention_forward is None:
        native_attention_forward, attention_sha = _native_qwen25_flash_forward()
    else:
        attention_sha = hashlib.sha256(inspect.getsource(native_attention_forward).encode()).hexdigest()
    text_forward, text_sha = _native_qwen25_text_forward()
    layers = _language_layers(model)
    language_model = getattr(model.model, "language_model", model.model)
    from transformers.cache_utils import DynamicCache

    # Preserve the complete-input mRoPE coordinates.  The prefix contains all
    # visual tokens, so the original visual inputs/grids remain valid there.
    positions = inputs.get("position_ids")
    if positions is None:
        with torch.no_grad():
            positions, _ = model.model.get_rope_index(
                input_ids=ids, image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=inputs.get("second_per_grid_ts"), attention_mask=mask,
            )
    if positions.shape[-1] != n:
        raise ValueError("Full-prompt mRoPE positions must cover the full input")
    prefix_projected, tail_projected = {}, {}
    position_embeddings = []
    stage = ["prefix"]
    handles, overrides, state_fields = [], [], []
    parameters = [] if parameters_already_frozen else [(p, p.requires_grad) for p in model.parameters()]
    training = [(module, module.training) for module in model.modules()]
    seen = set()
    for obj in (model, model.model, language_model):
        if hasattr(obj, "rope_deltas"):
            state_fields.append((obj, "rope_deltas", obj.rope_deltas))
        config = getattr(obj, "config", None)
        if config is not None and id(config) not in seen and hasattr(config, "use_cache"):
            seen.add(id(config))
            state_fields.append((config, "use_cache", config.use_cache))

    def bind(obj, function):
        existed = "forward" in vars(obj)
        overrides.append((obj, existed, vars(obj).get("forward")))
        obj.forward = MethodType(function, obj)

    def capture(index):
        def hook(_module, _args, output):
            destination = prefix_projected if stage[0] == "prefix" else tail_projected
            if index in destination:
                raise RuntimeError("Each K projection must execute once per probe stage")
            if stage[0] == "prefix":
                destination[index] = output.detach()
            else:
                output.requires_grad_(True)
                output.retain_grad()
                destination[index] = output
            return output
        return hook

    def capture_positions(_module, _args, kwargs):
        if stage[0] == "prefix" and not position_embeddings:
            cos, sin = kwargs["position_embeddings"]
            position_embeddings.append((cos.detach(), sin.detach()))

    def plain_text_forward(self, *args, **kwargs):
        # Custom outer Qwen models additionally pass these selection masks.
        # Their original construction is retained; native prefix probing simply
        # does not use a compression selection policy.
        kwargs.pop("is_text_token_mask", None)
        kwargs.pop("is_vision_token_mask", None)
        return text_forward(self, *args, **kwargs)

    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        model.eval()
        for obj, attr, _ in state_fields:
            if attr == "use_cache":
                setattr(obj, attr, True)
        bind(language_model, plain_text_forward)
        for index, layer in enumerate(layers):
            bind(layer.self_attn, native_attention_forward)
            handles.append(layer.self_attn.k_proj.register_forward_hook(capture(index)))
        handles.append(layers[0].self_attn.register_forward_pre_hook(capture_positions, with_kwargs=True))
        arguments = dict(inputs)
        arguments.update(input_ids=ids[:, :prefix_length],
                         attention_mask=None if mask is None else mask[:, :prefix_length],
                         position_ids=positions[..., :prefix_length],
                         cache_position=torch.arange(prefix_length, device=ids.device),
                         past_key_values=DynamicCache(), use_cache=True,
                         output_attentions=False, output_hidden_states=False, return_dict=True)
        with torch.inference_mode(False), torch.no_grad():
            output = model.model(**arguments)
        cache = output.past_key_values
        del output
        if not isinstance(cache, DynamicCache) or not position_embeddings:
            raise RuntimeError("Native prefix must return DynamicCache and mRoPE embeddings")
        with torch.inference_mode(False), torch.enable_grad():
            raw_prefix = _rebuild_raw_prefix_cache(cache, prefix_projected, position_embeddings[0], layers)
            prefix_projected.clear()
            stage[0] = "tail"
            # Unpadding was explicitly validated above.  The full prompt mask
            # cannot be passed to the custom Hybrid outer model with suffix
            # input_ids: its modality-mask construction expects equal lengths.
            # With no padding, native cached attention supplies causal masking;
            # full-prompt mRoPE coordinates remain explicit and unchanged.
            tail_args = dict(input_ids=ids[:, prefix_length:], attention_mask=None,
                             position_ids=positions[..., prefix_length:],
                             cache_position=torch.arange(prefix_length, n, device=ids.device),
                             past_key_values=cache, use_cache=True, output_attentions=False,
                             output_hidden_states=False, return_dict=True)
            output = model.model(**tail_args)
            logits = model.lm_head(output.last_hidden_state[:, :-1, :])
            labels = ids[:, prefix_length + 1:]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Cached-prefix Fisher loss is not finite")
            loss.backward()
        if set(tail_projected) != set(range(len(layers))):
            raise RuntimeError("Text suffix did not capture every language K projection")
        gradients = []
        for index, layer in enumerate(layers):
            raw = raw_prefix[index]
            tail = tail_projected[index]
            if raw.grad is None or tail.grad is None:
                raise RuntimeError(f"Missing cached-prefix raw-K Fisher gradient at layer {index}")
            batch, sequence, channels = tail.shape
            dim = int(layer.self_attn.head_dim)
            tail_gradient = tail.grad.detach().view(batch, sequence, channels // dim, dim).transpose(1, 2)
            gradients.append(torch.cat((raw.grad.detach(), tail_gradient), dim=2))
        stats = dict(
            gradient_space="pre_mrope_k_proj",
            objective="next_token_cross_entropy_on_observed_prompt_text_tail",
            loss_tail_tokens_requested=loss_tail_tokens,
            loss_target_tokens=tail_length - 1,
            loss=float(loss.detach().item()), layers=len(layers),
            prefix_tokens=prefix_length, backward_suffix_tokens=tail_length,
            gradient_shapes=[list(gradient.shape) for gradient in gradients],
            native_attention_forward_sha256=attention_sha,
            native_text_forward_sha256=text_sha,
            probe_policy="native_no_grad_prefix_cache_raw_leaves_text_suffix_backward_v2",
            cache_reconstruction="same native mRoPE embeddings and apply_multimodal_rotary_pos_emb; raw leaves and detached prefix V",
            original_asymkv_probe_adaptation=(
                "independent per-layer cached-prefix K gradients from a final observed text suffix; "
                "one probe instead of iterative 512-token chunk probes and running averaging; "
                "no prefix MLP backward, pruning, answer, or generation"
            ),
        )
        return gradients, stats
    finally:
        for handle in handles:
            handle.remove()
        for obj, existed, original in reversed(overrides):
            if existed:
                obj.forward = original
            else:
                del obj.forward
        for obj, attr, value in state_fields:
            setattr(obj, attr, value)
        for parameter, required in parameters:
            parameter.requires_grad_(required)
        for module, was_training in training:
            module.training = was_training


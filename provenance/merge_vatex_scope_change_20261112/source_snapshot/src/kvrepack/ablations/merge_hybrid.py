"""Fixed-group merge rules at HybridKV-KVRepack's existing merge insertion.

The checked-in core is untouched. Source AST adaptations preserve its candidate
and group construction and anchor mRoPE application. All arms share a global
active-prefill budget including the native local window and chunk rounding.
AsymKV retains sum-V and clipped counts and batches the existing official LMA
matmul/exp/count-denominator readout, avoiding duplicate FlashAttention work.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import importlib
import inspect
import math
import textwrap

import torch

VARIANTS = ("kvrepack", "lookm", "asymkv")


def _source_sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _tensor_record(tensor):
    value = tensor.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def shrink_global_budgets(budgets, retrieval_heads, *, seq_len, local_len,
                          ratio, chunk_size):
    """Scale the native Python budgets, preserving classification/relative size.

    Retrieval slots retain the native chunk granularity. Floors leave a small
    unused remainder; allocating it to new heads would change the native policy.
    This is a global budget, not an individual-head cap.
    """
    if not 0 < ratio <= 1 or chunk_size <= 0 or seq_len <= 0:
        raise ValueError("Require positive sequence/chunk and ratio in (0,1]")
    if not budgets or any(len(batch) != 1 for batch in budgets):
        raise ValueError("Hybrid merge comparison supports its native batch size one")
    layers, heads = len(budgets), len(budgets[0][0])
    if any(len(batch[0]) != heads for batch in budgets):
        raise ValueError("Native layer budget shapes disagree")
    slots = layers * heads
    total_capacity = slots * math.floor(seq_len * ratio)
    history_capacity = total_capacity - slots * local_len
    old_total = sum(sum(batch[0]) for batch in budgets)
    if history_capacity < 0 or old_total <= 0:
        raise ValueError("Input too short for the native protected local window")
    scale = min(1.0, history_capacity / old_total)
    result = copy.deepcopy(budgets)
    for layer, batch in enumerate(result):
        dynamic = set(retrieval_heads[layer])
        for head, old in enumerate(budgets[layer][0]):
            value = math.floor(old * scale)
            if head in dynamic:
                value = value // chunk_size * chunk_size
                if value < chunk_size:
                    raise ValueError("Budget cannot fit one retrieval chunk per dynamic head")
            batch[0][head] = value
    history_total = sum(sum(batch[0]) for batch in result)
    active = history_total + slots * local_len
    if active > total_capacity:
        raise RuntimeError("Scaled HybridKV active cache exceeds the global budget")
    metadata = dict(policy="global_prefill_including_local_chunk_floor_v1",
                    sequence_length=int(seq_len), budget_ratio=float(ratio),
                    layers=layers, native_kv_heads=heads, local_len=int(local_len),
                    original_history_budget=copy.deepcopy(budgets),
                    history_budget=copy.deepcopy(result), scale=scale,
                    original_history_positions=old_total,
                    history_positions=history_total, local_positions=slots * local_len,
                    active_prefill_positions=active, capacity_positions=total_capacity,
                    active_ratio=active / (slots * seq_len),
                    per_layer_active_positions=[sum(batch[0]) + heads * local_len for batch in result])
    return result, metadata


def append_flat_cardinality(counts, cu_klen):
    """Append one count=1 per flat head entirely on its current device."""
    heads = cu_klen.numel() - 1
    index = torch.arange(counts.shape[0], device=counts.device)
    owner = torch.searchsorted(cu_klen[1:].to(torch.long), index, right=True)
    output = torch.ones(counts.shape[0] + heads, device=counts.device, dtype=counts.dtype)
    output.index_copy_(0, index + owner, counts)
    return output


def lma_flat_readout(query_states, key_states, value_states, cardinality,
                     cu_klen, head_lens, max_seqlen_k, num_key_value_heads):
    """Batch the repository's official LMA operators without changing Vsum.

    A masked padded batch handles the native heterogeneous per-head lengths.
    Query length is one; every active cached position is causally visible.
    After padding, the operator order/dtype is the same as
    ``asymkv_official.attention.official_lma_attention``: matmul/sqrt, max-30,
    exp in FP32, count-weighted denominator, probability cast, matmul Vsum.
    There is no log-count approximation or division/rounding of cached values.
    """
    bsz, query_heads, query_len, dim = query_states.shape
    if query_len != 1 or query_heads % num_key_value_heads:
        raise ValueError("LMA decode requires one query and native integral GQA")
    flat_heads = bsz * num_key_value_heads
    offsets = torch.arange(max_seqlen_k, device=key_states.device)
    indices = cu_klen[:-1].to(torch.long).unsqueeze(-1) + offsets.unsqueeze(0)
    valid = offsets.unsqueeze(0) < head_lens.unsqueeze(-1)
    indices = indices.clamp_max(key_states.shape[0] - 1)
    keys = key_states[indices].unsqueeze(1)
    values = value_states[indices].unsqueeze(1)
    counts = cardinality[indices].to(torch.float32)
    query = query_states.reshape(flat_heads, query_heads // num_key_value_heads, 1, dim)
    scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(dim)
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    maximum = scores.max(dim=-1, keepdim=True).values - 30
    exponent = torch.exp((scores - maximum).to(torch.float32))
    denominator = (exponent * counts[:, None, None, :]).sum(dim=-1, keepdim=True)
    weights = (exponent / denominator).to(query_states.dtype)
    result = torch.matmul(weights, values)
    return result.reshape(bsz, query_heads, 1, dim)


# Compatibility for external pilot helpers created before the direct official
# readout was selected. This name does not call SDPA.
lma_flat_sdpa = lma_flat_readout


def _compile_merge(function, variant, reduce_callback, record_callback):
    """Keep the native grouping prefix; replace only the reviewed reduction tail."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    node = tree.body[0]
    marker = next((i for i, child in enumerate(node.body)
                   if isinstance(child, ast.If) and ast.dump(child.test) ==
                   ast.dump(ast.parse("self.hybrid_asymkv", mode="eval").body)), None)
    if marker is None:
        raise ValueError("Hybrid merger no longer has its reviewed reducer boundary")
    prefix = copy.deepcopy(node.body[:marker])
    prefix += ast.parse("_merge_group_record(self, k_seq, raw_k_seq, v_seq, importance, positions, group_ids, group_size, start_mask, layer_idx, head_idx, batch_idx)\n").body
    if variant == "kvrepack":
        # The ordinary arithmetic and native return path are byte-for-byte AST
        # identical, including its original false Hybrid-AsymKV branch.
        node.body = prefix + node.body[marker:]
    else:
        raw_branch = next(child for child in node.body[marker:]
                          if isinstance(child, ast.If) and isinstance(child.test, ast.Name)
                          and child.test.id == "use_raw_reapply")
        anchor_start = next(i for i, child in enumerate(raw_branch.body)
                            if isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name)
                            and child.targets[0].id == "anchor_pos")
        replacement = ast.parse("""if not use_raw_reapply:
    raise ValueError('Fixed-group comparison requires raw K and anchor mRoPE')
raw_comp, v_comp, result_cardinality = _merge_group_reduce(self, raw_k_seq, v_seq, group_ids, group_size, start_mask, positions, layer_idx, head_idx, batch_idx)
""").body
        node.body = prefix + replacement + copy.deepcopy(raw_branch.body[anchor_start:])
        node.body += ast.parse("return k_comp, v_comp, result_cardinality\n").body
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, _merge_group_reduce=reduce_callback,
                     _merge_group_record=record_callback)
    code = ast.unparse(tree) + "\n"
    exec(compile(tree, function.__code__.co_filename + "<fixed_group_merge>", "exec"), namespace)
    return namespace[function.__name__], dict(original_sha256=_source_sha(source),
                                             transformed_sha256=_source_sha(code),
                                             original_source=source, transformed_source=code,
                                             adaptation="Native grouping plus lightweight device-reference record; variant reducer only; native anchor mRoPE unchanged")


def _compile_asym_attention(function, callback):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    changed = [0]

    class Readout(ast.NodeTransformer):
        def visit_If(self, node):
            node = self.generic_visit(node)
            # This native block contains the unique varlen call and subsequent
            # per-head LMA correction. Keep all preceding update/retrieval code.
            indices = [i for i, child in enumerate(node.orelse)
                       if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call)
                       and isinstance(child.value.func, ast.Name)
                       and child.value.func.id == "flash_attn_varlen_func"]
            if indices:
                begin = indices[0]
                correction = next(i for i in range(begin, len(node.orelse))
                                  if isinstance(node.orelse[i], ast.If) and
                                  isinstance(node.orelse[i].test, ast.Call) and
                                  any(isinstance(arg, ast.Constant) and arg.value == "hybrid_asymkv"
                                      for arg in node.orelse[i].test.args))
                replacement = ast.parse("attn_output = _merge_lma_readout(self, past_key_value, query_states, key_states, value_states)\n").body
                node.orelse = node.orelse[:begin] + replacement + node.orelse[correction + 1:]
                changed[0] += 1
            return node

    tree = Readout().visit(tree)
    if changed != [1]:
        raise ValueError("Expected one native decode FlashAttention + LMA readout block")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, _merge_lma_readout=callback)
    code = ast.unparse(tree) + "\n"
    exec(compile(tree, function.__code__.co_filename + "<fixed_group_lma>", "exec"), namespace)
    return namespace[function.__name__], dict(original_sha256=_source_sha(source),
                                             transformed_sha256=_source_sha(code),
                                             original_source=source, transformed_source=code,
                                             adaptation="Decode only: padded batched official LMA matmul/exp/count-denominator operators on original sum-V; native prefill, cache updates and retrieval unchanged")


class MergeHybridAdapter:
    def __init__(self, variant, budget_ratio=.1, audit=False, module=None, attention_module=None):
        if variant not in VARIANTS:
            raise ValueError("Unknown fixed-group merge variant")
        self.variant, self.budget_ratio, self.audit = variant, float(budget_ratio), bool(audit)
        self.module = module
        self.attention_module = attention_module
        self._patches = []
        self._entered = False
        self.fisher = None
        self.last_audit = None
        self.transform_spec = {}
        self._cache = None
        self.begin_sample()

    def begin_sample(self, task=None, doc_id=None):
        self._sample = dict(task=task, doc_id=doc_id)
        self._group_refs = []
        self._budget_meta = None
        self.last_audit = None
        self._cache = None
        self.fisher = None

    def set_fisher(self, fisher):
        self.fisher = fisher

    def _record(self, cache, rope, raw, value, importance, positions, group_ids,
                group_size, start_mask, layer_idx, head_idx, batch_idx):
        # Device references only: hashing and transfer occur after synchronized
        # timing in finish_sample. Common overhead is present in every arm.
        row = dict(layer_idx=layer_idx, head_idx=head_idx, batch_idx=batch_idx,
                   before=int(rope.shape[0]), after=int(group_size.shape[0]),
                   positions=positions, group_ids=group_ids, group_size=group_size,
                   start_mask=start_mask)
        if self.audit:
            row.update(raw=raw, rope=rope, value=value, importance=importance)
        self._group_refs.append(row)

    def _reduce(self, cache, raw, value, group_ids, group_size, start_mask,
                positions, layer_idx, head_idx, batch_idx):
        from .merge_operators import reduce_fixed_groups
        fisher_seq = None
        if self.variant == "asymkv":
            if self.fisher is None or layer_idx is None or head_idx is None:
                raise RuntimeError("AsymKV fixed-group merge requires raw-K Fisher and head identity")
            fisher = self.fisher[layer_idx]
            fisher_seq = fisher[batch_idx, head_idx].index_select(0, positions.to(fisher.device))
        return reduce_fixed_groups(self.variant, raw, value, group_ids, group_size,
                                   start_mask, fisher_gradient=fisher_seq)

    def _patch(self, owner, name, value):
        self._patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Merge adapter cannot be entered twice")
        module = self.module or importlib.import_module("kvrepack.hybridkv_utils")
        self.module = module
        owner = module.HybridKVCache
        original_init, original_allocate, original_update = owner.__init__, owner.allocate_budget_by_ratio, owner.update
        original_merge = owner._hybrid_kvrepack_merge_one_sequence
        adapter = self

        @functools.wraps(original_init)
        def init(cache, *args, **kwargs):
            original_init(cache, *args, **kwargs)
            if not cache.hybridkv_kvrepack or cache.hybrid_kvrepack_k_merge_space != "raw_reapply":
                raise ValueError("Use native HybridKV-KVRepack with raw_reapply")
            cache.hybrid_asymkv = adapter.variant == "asymkv"
            cache.hybrid_kvrepack_score_correction = "asymkv_cardinality" if cache.hybrid_asymkv else "none"
            cache.hybrid_kvrepack_value_merge = "sum" if cache.hybrid_asymkv else "mean"
            # Its native force_manual flag is unused by the ordinary path;
            # the AsymKV readout is replaced separately with batched LMA.
            cache.hybrid_kvrepack_force_manual_attention = cache.hybrid_asymkv
            adapter._cache = cache

        @functools.wraps(original_allocate)
        def allocate(cache, key_states):
            result = original_allocate(cache, key_states)
            budgets, meta = shrink_global_budgets(cache.sparse_budget_per_head,
                                                  cache.retrieval_head_mapping,
                                                  seq_len=int(key_states.shape[-2]),
                                                  local_len=cache.local_len,
                                                  ratio=adapter.budget_ratio,
                                                  chunk_size=cache.chunk_size)
            cache.sparse_budget_per_head = budgets
            dynamic = [budgets[layer][0][head] for layer, heads in enumerate(cache.retrieval_head_mapping)
                       for head in heads]
            if dynamic:
                cache.retrieval_budget = min(dynamic)
            adapter._budget_meta = meta
            return result

        @functools.wraps(original_update)
        def update(cache, key_states, value_states, layer_idx, cache_kwargs=None):
            decode = len(cache.key_cache) > layer_idx
            old_counts = cache.layers[layer_idx].self_attn.kv_cluster.last_group_size if decode else None
            old_flag = cache.hybrid_asymkv
            try:
                cache.hybrid_asymkv = False
                result = original_update(cache, key_states, value_states, layer_idx, cache_kwargs)
            finally:
                cache.hybrid_asymkv = old_flag
            if decode:
                cluster = cache.layers[layer_idx].self_attn.kv_cluster
                cluster.last_group_size = append_flat_cardinality(old_counts, cache_kwargs["cu_klen"])
            return result

        try:
            self._patch(owner, "__init__", init)
            self._patch(owner, "allocate_budget_by_ratio", allocate)
            merger, self.transform_spec["merge"] = _compile_merge(original_merge, self.variant, self._reduce, self._record)
            self._patch(owner, "_hybrid_kvrepack_merge_one_sequence", merger)
            if self.variant == "asymkv":
                self._patch(owner, "update", update)
                attention_module = self.attention_module or importlib.import_module("kvrepack.modeling_qwen2_5_vl")
                attn_owner = attention_module.Qwen2_5_VLFlashAttention2

                def readout(attention, cache, query, key, value):
                    cluster = attention.kv_cluster
                    return lma_flat_readout(query, key, value, cluster.last_group_size,
                                            cluster.cu_klen, cluster.head_lens,
                                            cluster.max_seqlen_k, attention.num_key_value_heads)

                attention, self.transform_spec["attention"] = _compile_asym_attention(attn_owner.forward, readout)
                self._patch(attn_owner, "forward", attention)
            self._entered = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def finish_sample(self):
        if self._budget_meta is None or self._cache is None:
            raise RuntimeError("No completed native HybridKV budget allocation")
        rows = []
        for ref in self._group_refs:
            rows.append({name: _tensor_record(value) if isinstance(value, torch.Tensor) else value
                         for name, value in ref.items()})
        mask = self._cache.hybrid_kvrepack_merge_mask
        shapes = [dict(layer=layer, key_shape=list(key.shape), value_shape=list(value.shape))
                  for layer, (key, value) in enumerate(zip(self._cache.key_cache, self._cache.value_cache))]
        self.last_audit = dict(status="complete", variant=self.variant, **self._sample,
                               budget=self._budget_meta, merge_calls=rows,
                               merge_mask=_tensor_record(mask) if mask is not None else None,
                               selected_head_count=int(mask.sum().item()) if mask is not None else None,
                               cache_shapes=shapes,
                               raw_reapply=True,
                               attention="native_varlen_flash" if self.variant != "asymkv" else "batched_padded_official_lma",
                               lma_operator_order="repository_official_lma_attention" if self.variant == "asymkv" else None)
        self._group_refs = []
        self._cache = None
        self.fisher = None
        return self.last_audit

    def __exit__(self, exc_type, exc, tb):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches = []
        self._entered = False
        self._cache = None
        self.fisher = None
        self._group_refs = []
        return False

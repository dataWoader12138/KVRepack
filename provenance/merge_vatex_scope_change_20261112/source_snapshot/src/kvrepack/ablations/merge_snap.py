"""SnapKV fixed-group aggregation at the original KVRepack insertion point.

The core source files are unchanged. Reviewed AST transformations replace only
the raw-K/V reduction block, and (for AsymKV) retain cardinality in the cache and
replace decode attention by the project's released-code LMA implementation.
Candidate selection, head selection, adjacent groups, anchor positions and
anchor mRoPE remain the original functions. The one intentional shared policy
change is a strict total prefill budget, including the recent window.

Official provenance and adaptation boundaries live in ``merge_operators``.
These are fixed-group aggregation transplants, not complete LookM or AsymKV.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import math
import textwrap

VARIANTS = ("kvrepack", "lookm", "asymkv")


def _fingerprint(value):
    import torch
    payload = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(payload).hexdigest())


def _compile(function, tree, extra, label):
    ast.fix_missing_locations(tree)
    source_function = inspect.unwrap(function)
    namespace = dict(source_function.__globals__)
    namespace.update(extra)
    text = ast.unparse(tree) + "\n"
    exec(compile(tree, source_function.__code__.co_filename + "<" + label + ">", "exec"), namespace)
    return namespace[function.__name__], dict(
        original_source_sha256=hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest(),
        transformed_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        transformed_source=text,
    )


def _assigned_name(node):
    return node.targets[0].id if (isinstance(node, ast.Assign) and len(node.targets) == 1
                                and isinstance(node.targets[0], ast.Name)) else None


def strict_history_budget(config, seq_len, history_len, ratio):
    """Historical slots = floor(r*N) minus already protected recent slots."""
    seq_len, history_len = int(seq_len), int(history_len)
    if history_len <= 0:
        raise ValueError("Strict Snap merge experiment requires a historical region")
    recent = seq_len - history_len
    total = math.floor(float(ratio) * seq_len)
    history = total - recent
    if history <= 0:
        raise ValueError("Total budget must exceed the protected recent window")
    return min(history_len, history)


def compile_merger(function, variant, *, audit_callback=None):
    """Keep the actual core group builder and anchor mRoPE block verbatim."""
    if variant not in VARIANTS:
        raise ValueError("Unknown fixed-group aggregation variant")
    if variant == "kvrepack" and audit_callback is None:
        return function, dict(transformation="Unchanged original merger")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    names = [_assigned_name(child) for child in node.body]
    start, end = names.index("raw_sum"), names.index("v_comp") + 1
    expected = ["raw_sum", "raw_comp", "v_sum", "v_comp"]
    if [name for name in names[start:end] if name is not None] != expected:
        raise ValueError("Reviewed Snap reduction block changed")
    block = node.body[start:end]
    extra = {}
    if variant != "kvrepack":
        from .merge_operators import reduce_fixed_groups
        extra["_fixed_reduce"] = reduce_fixed_groups
        node.args.args.append(ast.arg(arg="fisher_gradient"))
        node.args.defaults.append(ast.Constant(value=None))
        block = ast.parse(
            "raw_comp, v_comp, _fixed_cardinality = _fixed_reduce("
            + repr(variant) + ", raw_k_seq, v_seq, group_ids, group_size, start_mask, "
            "fisher_gradient=fisher_gradient)\n"
        ).body
        if variant == "asymkv":
            ret = node.body[-1]
            if not (isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple)
                    and [getattr(item, "id", None) for item in ret.value.elts]
                    == ["k_comp", "v_comp", "group_size"]):
                raise ValueError("Reviewed Snap merger return changed")
            ret.value.elts[-1] = ast.Name(id="_fixed_cardinality", ctx=ast.Load())
    if audit_callback is not None:
        extra["_fixed_audit"] = audit_callback
        block = ast.parse(
            "_fixed_audit(raw_k_seq, rope_k_seq, v_seq, importance, positions, "
            "group_ids, group_size, start_mask)\n"
        ).body + block
    node.body[start:end] = block
    compiled, spec = _compile(function, tree, extra, "fixed_group_" + variant)
    spec["transformation"] = "Only raw K/V aggregation; original groups and anchor mRoPE retained"
    return compiled, spec


def _compile_asym_finalizer(function, fisher_callback, store_callback):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = tree.body[0]
    # This executes under generation no_grad; do not retain core decorator names.
    node.decorator_list = []
    class Transformer(ast.NodeTransformer):
        count_init = count_merge = count_copy = count_store = 0

        def visit_Assign(self, item):
            if _assigned_name(item) == "out_v":
                self.count_init += 1
                return [item, *ast.parse(
                    "out_c = torch.ones((bsz, h_kv, out_len), device=cand_rope.device, dtype=torch.float32)"
                ).body]
            if isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name) and item.value.func.id == "_paired_merge_one_sequence":
                self.count_merge += 1
                targets = item.targets[0]
                if not (isinstance(targets, ast.Tuple) and getattr(targets.elts[-1], "id", None) == "_"):
                    raise ValueError("Reviewed Snap finalizer merge target changed")
                targets.elts[-1] = ast.Name(id="hist_c", ctx=ast.Store())
                item.value.keywords.append(ast.keyword(arg="fisher_gradient", value=ast.parse(
                    "_fixed_fisher(layer_idx, bidx, hidx, cand_idx[bidx, hidx, :candidate_history])", mode="eval").body))
            return self.generic_visit(item)

        def visit_If(self, item):
            if (isinstance(item.test, ast.BoolOp) and any(isinstance(n, ast.Compare) for n in ast.walk(item.test))
                    and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == "_paired_merge_one_sequence" for n in ast.walk(item))):
                # The only merge-vs-prune branch; pure pruning entries have c=1.
                item.orelse += ast.parse("hist_c = torch.ones((final_history,), device=cand_rope.device, dtype=torch.float32)").body
            if ast.unparse(item.test) == "final_history > 0":
                self.count_copy += 1
                item.body += ast.parse("out_c[bidx, hidx, :final_history].copy_(hist_c[:final_history])").body
            return self.generic_visit(item)

        def visit_Expr(self, item):
            if (isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name)
                    and item.value.func.id == "_set_layer_cache" and len(item.value.args) == 4
                    and getattr(item.value.args[2], "id", None) == "out_k"):
                self.count_store += 1
                return [item, *ast.parse("_fixed_store(past_key_value, layer_idx, out_c)").body]
            return self.generic_visit(item)
    tr = Transformer()
    tree = tr.visit(tree)
    if (tr.count_init, tr.count_merge, tr.count_copy, tr.count_store) != (1, 1, 1, 1):
        raise ValueError("Reviewed Snap finalizer structure changed")
    return _compile(function, tree, {"_fixed_fisher": fisher_callback, "_fixed_store": store_callback}, "fixed_group_asym_counts")


def cardinality_lma(query, key, value_sum, cardinality, *, dropout_p=0.0):
    """Reuse released LMA: exp(z)/(sum(c*exp(z))) @ Vsum.

    Inputs Q/K/V use FlashAttention layout [B,N,H,D], counts [B,Hkv,N].
    Decode here is a single unpadded query; it may attend to the entire cache.
    """
    from kvrepack.asymkv_official.attention import official_lma_attention
    if int(query.shape[1]) != 1:
        raise ValueError("Fixed-group LMA adapter supports single-token decode")
    heads = int(query.shape[2])
    groups = heads // int(cardinality.shape[1])
    if groups * int(cardinality.shape[1]) != heads:
        raise ValueError("Counts do not match GQA head dimensions")
    counts = cardinality.repeat_interleave(groups, dim=1) if groups != 1 else cardinality
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value_sum.transpose(1, 2)
    output, _ = official_lma_attention(q, k, v, counts, current_query_length=1,
                                       dropout=float(dropout_p), training=float(dropout_p) > 0)
    return output.transpose(1, 2)


# Backward-compatible helper name from the development-only SDPA pilot. All
# actual arithmetic now calls the existing released LMA, without log(c) or V/c
# reparameterization, additional FlashAttention, or BF16 bias rounding.
cardinality_sdpa = cardinality_lma


def _compile_asym_decode(function, attention_callback):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    class Transformer(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "_flash_attention_forward":
                self.count += 1
                node.func.id = "_fixed_decode_attention"
                node.args = [ast.Name(id="self", ctx=ast.Load()),
                             ast.Name(id="past_key_value", ctx=ast.Load()), *node.args]
            return self.generic_visit(node)
    tr = Transformer(); tree = tr.visit(tree)
    if tr.count != 1:
        raise ValueError("Reviewed original Snap decode attention call changed")
    return _compile(function, tree, {"_fixed_decode_attention": attention_callback}, "fixed_group_asym_lma")


class SnapMergeAdapter:
    """Scoped adaptation; no core source mutation and no timed-path CPU audit."""
    def __init__(self, variant, *, budget_ratio=.1, audit=False, module=None):
        if variant not in VARIANTS or not 0 < float(budget_ratio) <= 1:
            raise ValueError("Require known variant and budget ratio in (0,1]")
        self.variant, self.budget_ratio, self.audit = variant, float(budget_ratio), bool(audit)
        self.module = module or importlib.import_module("kvrepack.qwen_model_snapkv_kvrepack")
        self._patches = []
        self._fisher = None
        self._entered = False
        self.last_audit = None
        self.transform_spec = {}
        self.begin_sample()

    def begin_sample(self, *unused):
        self._fisher = None
        self.last_audit = dict(variant=self.variant, merge_calls=[], layers=[], selected_masks=[], status="pending")
        self._light_groups = []
        self._light_layers = []
        self._light_masks = []
        self._light_shapes = []

    reset = begin_sample

    def set_fisher(self, gradients):
        self._fisher = gradients

    def _fisher_candidate(self, layer, batch, head, positions):
        if self._fisher is None:
            raise RuntimeError("AsymKV merge requires prompt raw-K gradients; fallback is forbidden")
        return self._fisher[int(layer)][int(batch), int(head)].index_select(0, positions)

    @staticmethod
    def _store_cardinality(cache, layer, counts):
        if not hasattr(cache, "_merge_cardinality"):
            cache._merge_cardinality = {}
        cache._merge_cardinality[int(layer)] = counts

    def _decode_attention(self, attention, cache, query, key, value, mask, q_len, **kwargs):
        if kwargs.get("sliding_window") is not None or int(q_len) != 1 or int(query.shape[0]) != 1:
            raise ValueError("LMA benchmark requires batch-one, single-query, non-sliding decode")
        counts = cache._merge_cardinality[int(attention.layer_idx)]
        added = int(key.shape[1]) - int(counts.shape[-1])
        if added < 0:
            raise ValueError("Cardinality outlived its key cache")
        if added:
            import torch
            counts = torch.cat((counts, counts.new_ones((*counts.shape[:-1], added))), dim=-1)
            cache._merge_cardinality[int(attention.layer_idx)] = counts
        # Inputs are fixed batch-one unpadded prompts; original padding mask is
        # all ones and does not describe compressed key positions.
        return cardinality_lma(query, key, value, counts, dropout_p=kwargs.get("dropout", 0.0))

    def _audit_merge(self, raw, rope, value, importance, positions, groups, sizes, starts):
        # These small metadata tensors are already produced by the core. No
        # clone, host transfer, synchronization or profiler in a timed run.
        self._light_groups.append(dict(positions=positions, group_ids=groups,
                                       group_size=sizes, start_mask=starts))
        if self.audit:
            self.last_audit["merge_calls"].append(dict(
                raw=_fingerprint(raw), rope=_fingerprint(rope), value=_fingerprint(value),
                importance=_fingerprint(importance), positions=_fingerprint(positions),
                group_ids=_fingerprint(groups), group_size=_fingerprint(sizes), start_mask=_fingerprint(starts)))

    def _masks(self, records, merge_ratio):
        masks = self._original_masks(records, merge_ratio)
        self._light_masks = masks
        return masks

    def _finalize(self, config, cache):
        records = getattr(config, "_snapkv_kvrepack_records", [])
        self._light_layers = [dict(layer=int(rec["layer_idx"]), candidate_idx=rec["candidate_idx"],
                                  final_history=int(rec["final_history"]),
                                  candidate_history=int(rec["candidate_history"]))
                              for rec in records if "direct_key" not in rec]
        if self.audit:
            self.last_audit["layers"] = [dict(layer=int(rec["layer_idx"]),
                candidates={name:_fingerprint(rec[name]) for name in
                    ("candidate_raw", "candidate_rope", "candidate_value", "candidate_imp", "candidate_idx")},
                final_history=int(rec["final_history"]), candidate_history=int(rec["candidate_history"]))
                for rec in records if "direct_key" not in rec]
        result = self._finalizer(config, cache)
        self._light_shapes = [list(value.shape) for value in cache.key_cache]
        if self.audit:
            self.last_audit["selected_masks"] = [_fingerprint(mask) for mask in self._light_masks]
            self.last_audit["prefill_cache_shapes"] = self._light_shapes
            self.last_audit["status"] = "complete"
        return result

    def finish_sample(self):
        """Synchronize/hash metadata only AFTER the caller's E2E timer stops."""
        if not self._light_shapes:
            raise RuntimeError("No completed Snap prefill to inspect")
        row = self.last_audit
        if not self.audit:
            row["merge_calls"] = [{name:_fingerprint(value) for name,value in rec.items()}
                                  for rec in self._light_groups]
            row["layers"] = [{**{name:value for name,value in rec.items() if name != "candidate_idx"},
                              "candidate_idx":_fingerprint(rec["candidate_idx"])} for rec in self._light_layers]
            row["selected_masks"] = [_fingerprint(mask) for mask in self._light_masks]
            row["prefill_cache_shapes"] = self._light_shapes
        row["status"] = "complete"
        row["actual_merge_calls"] = len(self._light_groups)
        self._light_groups = []; self._light_layers = []; self._light_masks = []
        return row

    def _patch(self, name, value):
        self._patches.append((name, getattr(self.module, name)))
        setattr(self.module, name, value)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Adapter cannot be entered twice")
        self._entered = True
        try:
            self._patch("_history_budget", lambda config, seq_len, history_len:
                        strict_history_budget(config, seq_len, history_len, self.budget_ratio))
            self._original_masks = self.module._global_merge_masks
            self._patch("_global_merge_masks", self._masks)
            merger, spec = compile_merger(self.module._paired_merge_one_sequence, self.variant,
                                         audit_callback=self._audit_merge)
            self.transform_spec["merger"] = spec
            self._patch("_paired_merge_one_sequence", merger)
            self._finalizer = self.module._finalize_prompt_cache
            if self.variant == "asymkv":
                self._finalizer, spec = _compile_asym_finalizer(self._finalizer, self._fisher_candidate, self._store_cardinality)
                self.transform_spec["finalizer"] = spec
                decode, spec = _compile_asym_decode(self.module.qwen_flash_attn2_forward_SnapKV, self._decode_attention)
                self.transform_spec["decode"] = spec
                self._patch("qwen_flash_attn2_forward_SnapKV", decode)
            self._patch("_finalize_prompt_cache", self._finalize)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for name, original in reversed(self._patches):
            setattr(self.module, name, original)
        self._patches.clear()
        self._entered = False
        self._fisher = None


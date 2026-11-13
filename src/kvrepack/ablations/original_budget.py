"""Original benchmark budgets, with after-timer execution evidence only.

The strict-budget experiment and its source files remain frozen. These
adapters reuse its merge operators but deliberately leave the released
SnapKV history capacity and HybridKV allocation unchanged. In particular,
recent/local positions are added by the original methods, not subtracted
from an invented total-cache cap.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import inspect
import math
import textwrap

from .merge_snap import SnapMergeAdapter
from .merge_hybrid import MergeHybridAdapter

BUDGET_POLICY = "original_history_plus_local_v1"


def _source_spec(function):
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(function)))
    return dict(policy=BUDGET_POLICY, native_budget_unchanged=True,
                original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                transformed=False)


def _shape(value):
    shape = [int(x) for x in value.shape]
    if not shape or any(x <= 0 for x in shape):
        raise ValueError("Require a nonempty native cache tensor")
    return shape


def _hybrid_metadata(cache, key_states, ratio):
    """Read only Python allocation values produced by the original function."""
    shape = _shape(key_states)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("Original HybridKV benchmark requires batch-one B,H,N,D")
    budgets = copy.deepcopy(cache.sparse_budget_per_head)
    if not budgets or any(len(batch) != 1 or len(batch[0]) != shape[1] for batch in budgets):
        raise ValueError("Original HybridKV allocation has an unexpected head shape")
    if any(type(n) is not int or n < 0 for batch in budgets for n in batch[0]):
        raise ValueError("Original HybridKV history budgets must be nonnegative integers")
    layers, heads, n = len(budgets), shape[1], shape[2]
    local_len = int(cache.local_len)
    if not 0 <= local_len <= n:
        raise ValueError("Invalid original local window")
    history = sum(sum(batch[0]) for batch in budgets)
    local = layers*heads*local_len
    total = history+local
    return dict(policy=BUDGET_POLICY, native_budget_unchanged=True,
        post_allocation_scaling=False, scale=1.0, sequence_length=n,
        budget_ratio=float(ratio), layers=layers, native_kv_heads=heads,
        local_len=local_len, original_history_budget=copy.deepcopy(budgets),
        history_budget=budgets, original_history_positions=history,
        history_positions=history, local_positions=local,
        active_prefill_positions=total, capacity_positions=total,
        active_ratio=total/(layers*heads*n),
        per_layer_active_positions=[sum(batch[0])+heads*local_len for batch in budgets])


class OriginalSnapMergeAdapter(SnapMergeAdapter):
    """The original history-budget function remains the identical callable."""
    def _patch(self, name, value):
        if name == "_history_budget":
            self.transform_spec["budget"] = _source_spec(getattr(self.module, name))
            return
        super()._patch(name, value)


class OriginalHybridMergeAdapter(MergeHybridAdapter):
    """Run the original allocation; record it without scaling or chunk edits."""
    def _patch(self, owner, name, value):
        if name == "allocate_budget_by_ratio":
            original = getattr(owner, name)
            adapter = self
            self.transform_spec["budget"] = _source_spec(original)

            @functools.wraps(original)
            def allocate(cache, key_states):
                result = original(cache, key_states)
                adapter._budget_meta = _hybrid_metadata(cache, key_states, adapter.budget_ratio)
                return result

            value = allocate
        super()._patch(owner, name, value)


class NativePruningAdapter:
    """Observe the actual native functions without changing their arithmetic.

    The timed path records only shapes and native Python budget lists. It
    neither copies tensors to the CPU nor overwrites any native cache option.
    """
    def __init__(self, method, *, budget_ratio=.1, module=None):
        if method not in ("snapkv", "hybridkv") or not math.isfinite(float(budget_ratio)) or not 0 < float(budget_ratio) <= 1:
            raise ValueError("Require SnapKV/HybridKV and a finite nominal ratio in (0,1]")
        self.method, self.budget_ratio, self.module = method, float(budget_ratio), module
        self._patches, self._snap_rows = [], []
        self._entered = False
        self._cache = self._budget_meta = None
        self.transform_spec = dict(method=method, policy=BUDGET_POLICY,
            native_budget_unchanged=True, scope="Only record calls of unmodified native pruning")
        self.begin_sample()

    def begin_sample(self, task=None, doc_id=None):
        if self._cache is not None or self._snap_rows:
            raise RuntimeError("Finish the previous native sample before beginning another")
        if doc_id is not None and (type(doc_id) is not int or doc_id < 0):
            raise ValueError("doc_id must be a nonnegative integer")
        self._sample = dict(task=task, doc_id=doc_id)
        self._budget_meta = None
        self.last_audit = dict(status="pending", method=self.method, **self._sample)

    reset = begin_sample

    def _patch(self, owner, name, value):
        self._patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def _enter_snap(self):
        module = self.module or importlib.import_module("kvrepack.baseline_utils")
        self.module = module
        owner, adapter = module.SnapKVCluster, self
        original = owner.update_kv
        self.transform_spec["update_kv"] = _source_spec(original)

        @functools.wraps(original)
        def update(cluster, key, query, value):
            if getattr(cluster, "pyram_mode", False):
                raise ValueError("Original SnapKV comparison excludes PyramidKV")
            if float(cluster.budget_ratio) != adapter.budget_ratio:
                raise ValueError("Actual native SnapKV ratio differs from the declared ratio")
            shape = _shape(key)
            if len(shape) != 4 or _shape(value) != shape:
                raise ValueError("Require native matching B,H,N,D keys/values")
            # The original method mutates max_capacity_prompt itself. Preserve
            # that state transition as well as the returned K/V tensors.
            result = original(cluster, key, query, value)
            out_key, out_value = result
            out_shape = _shape(out_key)
            recent = int(cluster.window_size)
            expected = min(shape[-2]-recent, math.floor(adapter.budget_ratio*shape[-2]))+recent
            if out_shape != _shape(out_value) or out_shape != shape[:2]+[expected, shape[-1]]:
                raise ValueError("Native SnapKV output differs from its original history-plus-recent budget")
            adapter._snap_rows.append(dict(layer_idx=getattr(cluster, "layer_idx", None),
                input_shape=shape, prefill_shape=out_shape, input_tokens=shape[-2],
                history_positions=expected-recent, recent_positions=recent,
                capacity_per_head=expected, actual_retained_ratio=expected/shape[-2]))
            return result

        self._patch(owner, "update_kv", update)

    def _enter_hybrid(self):
        module = self.module or importlib.import_module("kvrepack.hybridkv_utils")
        self.module = module
        owner, adapter = module.HybridKVCache, self
        original_init, original_allocate = owner.__init__, owner.allocate_budget_by_ratio
        self.transform_spec["allocate_budget_by_ratio"] = _source_spec(original_allocate)

        @functools.wraps(original_init)
        def init(cache, *args, **kwargs):
            original_init(cache, *args, **kwargs)
            if getattr(cache, "hybridkv_kvrepack", False) or getattr(cache, "hybrid_asymkv", False):
                raise ValueError("Native HybridKV must run without merge")
            if adapter._cache is not None:
                raise ValueError("Multiple native caches created within one sample")
            adapter._cache = cache

        @functools.wraps(original_allocate)
        def allocate(cache, key_states):
            if adapter._cache is not cache:
                raise ValueError("Create the native cache inside the current sample")
            result = original_allocate(cache, key_states)
            adapter._budget_meta = _hybrid_metadata(cache, key_states, adapter.budget_ratio)
            return result

        self._patch(owner, "__init__", init)
        self._patch(owner, "allocate_budget_by_ratio", allocate)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Native pruning adapter cannot be entered twice")
        self._entered = True
        try:
            self._enter_snap() if self.method == "snapkv" else self._enter_hybrid()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def finish_sample(self):
        if self.method == "snapkv":
            if not self._snap_rows:
                raise RuntimeError("No completed original SnapKV prefill")
            result = dict(status="complete", method=self.method, **self._sample,
                budget_policy=BUDGET_POLICY, budget_ratio=self.budget_ratio,
                native_budget_unchanged=True, prefill_calls=copy.deepcopy(self._snap_rows),
                actual_merge_calls=0, value_reduction="none")
        else:
            if self._cache is None or self._budget_meta is None:
                raise RuntimeError("No completed original HybridKV allocation")
            cache = self._cache
            if len(cache.key_cache) != self._budget_meta["layers"] or len(cache.value_cache) != len(cache.key_cache):
                raise ValueError("Native HybridKV final cache is missing allocated layers")
            shapes = []
            for i, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
                shape = _shape(key)
                if shape != _shape(value):
                    raise ValueError("Native HybridKV key/value shapes differ")
                shapes.append(dict(layer=i, key_shape=shape, value_shape=shape))
            result = dict(status="complete", method=self.method, **self._sample,
                budget=copy.deepcopy(self._budget_meta), cache_shapes=shapes,
                native_budget_unchanged=True, actual_merge_calls=0, value_reduction="none")
        self._cache = self._budget_meta = None
        self._snap_rows = []
        self.last_audit = result
        return copy.deepcopy(result)

    def __exit__(self, *exc):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches = []
        self._entered = False
        self._cache = self._budget_meta = None
        self._snap_rows = []
        return False


def active_prefill_original(cache, generated_tokens, input_tokens, backbone, *,
                            shape_fn=None, audit=None, budget_ratio=.1, window_size=32):
    """Validate real cache lengths against the original, unshrunk allocation.

    ``nominal_total_cap`` is a reference number only; it is not enforced as a
    cap. The returned ``capacity`` is the actual original history+local budget.
    """
    if backbone not in ("snapkv", "hybridkv") or type(input_tokens) is not int or input_tokens <= 0:
        raise ValueError("Require an original video-backbone input length")
    if type(generated_tokens) is not int or generated_tokens <= 0:
        raise ValueError("Require nonempty generation")
    if not math.isfinite(float(budget_ratio)) or not 0 < float(budget_ratio) <= 1:
        raise ValueError("Invalid nominal original budget ratio")
    if shape_fn is None:
        from .run_head_selection import cache_shapes
        shape_fn = cache_shapes
    shapes = shape_fn(cache)
    if len(shapes) != 28:
        raise ValueError("Require all 28 Qwen7B cache layers")
    final = 0
    for layer in shapes:
        key, value = layer["key_shape"], layer["value_shape"]
        if key != value or len(key) not in (2, 4) or any(type(x) is not int or x <= 0 for x in key):
            raise ValueError("Invalid native physical K/V cache shapes")
        final += math.prod(key[:-1])
    prefill = final-112*(generated_tokens-1)
    if backbone == "snapkv":
        recent = min(max(int(window_size), 0), input_tokens)
        expected = 112*(min(input_tokens-recent, math.floor(float(budget_ratio)*input_tokens))+recent)
    else:
        meta = (audit or {}).get("budget", {})
        if (meta.get("policy") != BUDGET_POLICY or meta.get("native_budget_unchanged") is not True
                or meta.get("post_allocation_scaling") is not False or meta.get("scale") != 1.0
                or meta.get("sequence_length") != input_tokens
                or meta.get("layers") != 28 or meta.get("native_kv_heads") != 4
                or meta.get("original_history_positions") != meta.get("history_positions")
                or meta.get("original_history_budget") != meta.get("history_budget")):
            raise ValueError("Original HybridKV physical validation requires its unchanged native budget audit")
        expected = meta["history_positions"]+meta["local_positions"]
        if meta.get("active_prefill_positions") != expected or meta.get("capacity_positions") != expected:
            raise ValueError("Original HybridKV budget metadata is inconsistent")
    if not 0 < prefill == expected:
        raise ValueError(f"Physical cache differs from the original budget: {prefill} != {expected}")
    return dict(prefill_tokens_total=prefill, capacity=expected,
        actual_retained_ratio=prefill/(112*input_tokens), final_shapes=shapes,
        budget_policy=BUDGET_POLICY, native_budget_unchanged=True,
        nominal_total_cap=112*math.floor(float(budget_ratio)*input_tokens))

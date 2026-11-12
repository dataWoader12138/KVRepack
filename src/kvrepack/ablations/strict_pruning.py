"""Scoped total-budget controls for the existing native pruning methods.

No model/core file is edited. SnapKV's released scoring/top-k code is compiled
with one reviewed capacity expression changed to include its recent window.
HybridKV's original allocation is followed by the same global budget shrink
used by the merge-only comparison. There is no aggregation or timed CPU hash.
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


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def strict_snap_history(seq_len, window_size, ratio):
    """Return historical slots; protected recent slots are inside floor(r*N)."""
    seq_len, window_size = int(seq_len), int(window_size)
    if seq_len <= 0 or not 0 < window_size < seq_len:
        raise ValueError("Require a nonempty historical region and recent window")
    if not math.isfinite(float(ratio)) or not 0 < float(ratio) <= 1:
        raise ValueError("Budget ratio must be finite and in (0,1]")
    capacity = math.floor(float(ratio)*seq_len)
    history = capacity-window_size
    if history <= 0:
        raise ValueError("Total SnapKV budget must exceed its recent window")
    return history


def _compile_snap_update(function, ratio):
    function = inspect.unwrap(function)
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Require the reviewed SnapKV update function")
    if tree.body[0].decorator_list:
        raise ValueError("Unexpected native SnapKV decorators")
    expected = ast.parse("self.max_capacity_prompt = int(q_len * self.budget_ratio)").body[0]
    expected_ast = ast.dump(expected, include_attributes=False)
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.Assign) and ast.dump(node, include_attributes=False) == expected_ast]
    if len(matches) != 1:
        raise ValueError("Reviewed native SnapKV capacity assignment changed")
    matches[0].value = ast.parse(
        "_strict_history_capacity(q_len, self.window_size, _strict_ratio)", mode="eval").body
    ast.fix_missing_locations(tree)
    transformed = ast.unparse(tree)+"\n"
    namespace = dict(function.__globals__)
    namespace.update(_strict_history_capacity=strict_snap_history, _strict_ratio=float(ratio))
    exec(compile(tree, function.__code__.co_filename+"<strict_total_pruning_budget>", "exec"), namespace)
    return namespace[function.__name__], dict(
        transformation="Only native historical-capacity expression: floor(r*N) minus recent window",
        changed_assignments=1, original_source_sha256=_sha(source),
        transformed_source_sha256=_sha(transformed), transformed_source=transformed)


def _shape(tensor):
    try:
        shape = [int(value) for value in tensor.shape]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Require native cache tensors with known shapes") from error
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("Cache shapes must be nonempty and positive")
    return shape


class StrictPruningAdapter:
    """Original SnapKV/HybridKV with the comparison's strict total KV budget.

    ``module`` is an optional dependency-injection hook for CPU tests. Metadata
    captured in the timed path consists only of Python values and tensor shapes;
    ``finish_sample`` is called after the caller stops its E2E timer.
    """
    def __init__(self, method, *, budget_ratio=.1, module=None):
        if method not in ("snapkv", "hybridkv"):
            raise ValueError("Strict native pruning supports snapkv or hybridkv")
        if not math.isfinite(float(budget_ratio)) or not 0 < float(budget_ratio) <= 1:
            raise ValueError("Budget ratio must be finite and in (0,1]")
        self.method, self.budget_ratio = method, float(budget_ratio)
        self.module = module
        self._patches = []
        self._entered = False
        self._cache = None
        self._budget_meta = None
        self._snap_rows = []
        self.transform_spec = dict(method=method, budget_ratio=self.budget_ratio,
            scope="Budget-only native pruning; no aggregation, candidate or scoring replacement")
        self.begin_sample()

    def begin_sample(self, task=None, doc_id=None):
        if self._cache is not None or self._snap_rows:
            raise RuntimeError("Finish the previous pruning sample before beginning another")
        if doc_id is not None and (type(doc_id) is not int or doc_id < 0):
            raise ValueError("doc_id must be a nonnegative integer")
        self._sample = dict(task=task, doc_id=doc_id)
        self._cache = None
        self._budget_meta = None
        self._snap_rows = []
        self.last_audit = dict(status="pending", method=self.method, **self._sample)

    reset = begin_sample

    def _patch(self, owner, name, value):
        original = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, original))

    def _enter_snap(self):
        module = self.module or importlib.import_module("kvrepack.baseline_utils")
        self.module = module
        owner = module.SnapKVCluster
        original = owner.update_kv
        compiled, spec = _compile_snap_update(original, self.budget_ratio)
        self.transform_spec["update_kv"] = spec
        adapter = self

        @functools.wraps(original)
        def update(cluster, origin_key_states, query_states, origin_value_states):
            if getattr(cluster, "pyram_mode", False):
                raise ValueError("Native strict SnapKV comparison excludes PyramidKV")
            if float(cluster.budget_ratio) != adapter.budget_ratio:
                raise ValueError("Native SnapKV configuration and declared ratio disagree")
            shape, value_shape = _shape(origin_key_states), _shape(origin_value_states)
            if len(shape) != 4 or shape != value_shape:
                raise ValueError("Require matching native [B,Hkv,N,D] K/V tensors")
            history = strict_snap_history(shape[-2], cluster.window_size, adapter.budget_ratio)
            total = history+int(cluster.window_size)
            previous_capacity = cluster.max_capacity_prompt
            try:
                key, value = compiled(cluster, origin_key_states, query_states, origin_value_states)
                out_shape = _shape(key)
                if out_shape != _shape(value) or out_shape != shape[:2]+[total, shape[-1]]:
                    raise ValueError("Native SnapKV output violates its strict total capacity")
                adapter._snap_rows.append(dict(call_index=len(adapter._snap_rows),
                    layer_idx=getattr(cluster, "layer_idx", None), input_shape=shape,
                    prefill_shape=out_shape, input_tokens=shape[-2], recent_positions=int(cluster.window_size),
                    history_positions=history, capacity_per_head=total,
                    actual_retained_ratio=total/shape[-2]))
                return key, value
            finally:
                # Decode reads the returned KV tensors, not this temporary
                # prefill-only capacity; leave no cluster-state intervention.
                cluster.max_capacity_prompt = previous_capacity

        self._patch(owner, "update_kv", update)

    def _enter_hybrid(self):
        from .merge_hybrid import shrink_global_budgets
        module = self.module or importlib.import_module("kvrepack.hybridkv_utils")
        self.module = module
        owner = module.HybridKVCache
        original_init, original_allocate = owner.__init__, owner.allocate_budget_by_ratio
        adapter = self
        self.transform_spec["allocate_budget_by_ratio"] = dict(
            transformation="Original native allocation followed by shared global active-prefill shrink",
            original_source_sha256=_sha(textwrap.dedent(inspect.getsource(original_allocate))),
            shrink_source_sha256=_sha(textwrap.dedent(inspect.getsource(shrink_global_budgets))))

        @functools.wraps(original_init)
        def init(cache, *args, **kwargs):
            original_init(cache, *args, **kwargs)
            if getattr(cache, "hybridkv_kvrepack", False) or getattr(cache, "hybrid_asymkv", False):
                raise ValueError("Strict native HybridKV requires METHOD=hybridkv, without merge")
            if adapter._cache is not None and adapter._cache is not cache:
                raise ValueError("Multiple native caches initialized within one pruning sample")
            adapter._cache = cache

        @functools.wraps(original_allocate)
        def allocate(cache, key_states):
            if adapter._cache is not cache:
                raise ValueError("Native HybridKV cache must be created inside the current context/sample")
            result = original_allocate(cache, key_states)
            shape = _shape(key_states)
            if len(shape) != 4:
                raise ValueError("Require native HybridKV prefill [B,Hkv,N,D]")
            budgets, metadata = shrink_global_budgets(cache.sparse_budget_per_head,
                cache.retrieval_head_mapping, seq_len=shape[-2], local_len=cache.local_len,
                ratio=adapter.budget_ratio, chunk_size=cache.chunk_size)
            cache.sparse_budget_per_head = budgets
            dynamic = [budgets[layer][0][head] for layer, heads in enumerate(cache.retrieval_head_mapping)
                       for head in heads]
            if dynamic:
                cache.retrieval_budget = min(dynamic)
            adapter._budget_meta = metadata
            return result

        self._patch(owner, "__init__", init)
        self._patch(owner, "allocate_budget_by_ratio", allocate)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Strict pruning adapter cannot be entered twice")
        self._entered = True
        try:
            if self.method == "snapkv":
                self._enter_snap()
            else:
                self._enter_hybrid()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def finish_sample(self):
        if not self._entered:
            raise RuntimeError("Pruning audit must finish inside its active context")
        if self.method == "snapkv":
            if not self._snap_rows:
                raise RuntimeError("No completed native SnapKV prefill")
            result = dict(status="complete", method=self.method, **self._sample,
                budget_policy="native_snap_total_prefill_including_recent_v1",
                budget_ratio=self.budget_ratio, prefill_calls=copy.deepcopy(self._snap_rows),
                actual_merge_calls=0, value_reduction="none")
        else:
            if self._cache is None or self._budget_meta is None:
                raise RuntimeError("No completed native HybridKV allocation")
            cache = self._cache
            if len(cache.key_cache) != self._budget_meta["layers"] or len(cache.value_cache) != len(cache.key_cache):
                raise ValueError("Native HybridKV final cache does not cover allocated layers")
            shapes = []
            for layer, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
                key_shape = _shape(key)
                if key_shape != _shape(value):
                    raise ValueError("Native HybridKV K/V cache lengths disagree")
                shapes.append(dict(layer=layer, key_shape=key_shape, value_shape=key_shape))
            result = dict(status="complete", method=self.method, **self._sample,
                budget=copy.deepcopy(self._budget_meta), cache_shapes=shapes,
                actual_merge_calls=0, value_reduction="none")
        self.last_audit = result
        self._cache = None
        self._budget_meta = None
        self._snap_rows = []
        return copy.deepcopy(result)

    def __exit__(self, *exc):
        try:
            for owner, name, original in reversed(self._patches):
                setattr(owner, name, original)
        finally:
            self._patches = []
            self._entered = False
            self._cache = None
            self._budget_meta = None
            self._snap_rows = []
        return False

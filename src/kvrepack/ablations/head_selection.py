"""Count-matched head selection interventions around the original SnapKV code.

The global Random condition has a fixed uniform model-level head ranking per
seed. Only the returned selection mask changes; every candidate, budget, merge
argument and original return object follows the intact core implementation.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import random

STRATEGIES = ('similarity', 'random_global', 'random_layer_matched')
_FUNCTIONS = ('_global_merge_masks', '_paired_merge_one_sequence', '_set_layer_cache')
_CANDIDATES = ('candidate_raw', 'candidate_rope', 'candidate_value', 'candidate_imp', 'candidate_idx', 'head_sim')


def _tensor_record(value):
    import torch
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(raw).hexdigest()}


def _ranking(seed, *dimensions):
    payload = json.dumps([seed, *dimensions], separators=(',', ':')).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')
    return random.Random(value)


class SnapHeadSelectionAdapter:
    def __init__(self, module, *, strategy='similarity', seed=0, audit_path=None):
        if strategy not in STRATEGIES or type(seed) is not int or seed < 0:
            raise ValueError('Require a known strategy and a nonnegative integer local selection seed')
        self.module, self.strategy, self.seed = module, strategy, seed
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self._original = {}
        self._sample = None
        self._row = None
        self._pointers = {}
        self._entered = False
        self._failure = None

    def set_sample(self, task, doc_id):
        if not self._entered or self._sample is not None or not isinstance(task, str) or not task or type(doc_id) is not int or doc_id < 0:
            raise ValueError('Finish the previous sample before setting a valid task/doc_id')
        self._sample = (task, doc_id)
        self._row = None
        self._pointers = {}
        self._failure = None

    def _gate(self, records, merge_ratio):
        import torch
        if self._sample is None or self._row is not None:
            raise RuntimeError('Selection must run exactly once for the explicitly declared sample')
        original = self._original['_global_merge_masks'](records, merge_ratio)
        if not records or len(original) != len(records):
            raise ValueError('Expected complete original layer records and masks')
        baseline = torch.stack(original, dim=1)
        if baseline.ndim != 3 or baseline.dtype != torch.bool:
            raise ValueError('Expected native [batch,layers,KV-heads] Boolean masks')
        batch, layers, heads = baseline.shape
        if [r['layer_idx'] for r in records] != list(range(layers)):
            raise ValueError('Require complete contiguous native layer IDs')
        if any(tuple(r['head_sim'].shape) != (batch, heads) for r in records):
            raise ValueError('Head scores must describe the same native KV-head dimensions')
        ratio = min(max(float(merge_ratio), 0), 1)
        expected_count = min(layers*heads, math.ceil(layers*heads*ratio)) if ratio > 0 else 0
        counts = baseline.sum(dim=(1, 2)).cpu().tolist()
        if any(n != expected_count for n in counts):
            raise ValueError('Original gate count differs from its exact global ceil quota')
        selected = baseline.clone()
        if self.strategy == 'random_global':
            selected.zero_()
            for b in range(batch):
                indices = _ranking(self.seed, 'global', layers, heads, b).sample(range(layers*heads), layers*heads)
                flat = selected[b].view(-1)
                flat[torch.tensor(indices[:counts[b]], device=flat.device, dtype=torch.long)] = True
        elif self.strategy == 'random_layer_matched':
            selected.zero_()
            for b in range(batch):
                for layer in range(layers):
                    count = int(baseline[b, layer].sum().item())
                    indices = _ranking(self.seed, 'layer', layers, heads, b, layer).sample(range(heads), heads)
                    selected[b, layer, torch.tensor(indices[:count], device=selected.device, dtype=torch.long)] = True
        if not torch.equal(selected.sum(dim=(1, 2)), baseline.sum(dim=(1, 2))):
            raise ValueError('Random intervention changed the global selected-head count')
        if self.strategy == 'random_layer_matched' and not torch.equal(selected.sum(-1), baseline.sum(-1)):
            raise ValueError('Layer-matched intervention changed a layer quota')
        candidate_hashes, eligible, work = [], torch.zeros_like(baseline), torch.zeros_like(baseline, dtype=torch.int64)
        for layer, rec in enumerate(records):
            if 'direct_key' in rec:
                candidate_hashes.append({'layer': layer, 'direct': True,
                                          'direct_key': _tensor_record(rec['direct_key']),
                                          'direct_value': _tensor_record(rec['direct_value'])})
                continue
            before, target = int(rec['candidate_history']), int(rec['final_history'])
            if tuple(rec['candidate_raw'].shape[:3]) != (batch, heads, before):
                raise ValueError('Candidate history does not match native KV shape')
            eligible[:, layer] = before > target
            work[:, layer] = max(before-target, 0)
            candidate_hashes.append({'layer': layer, 'candidate_history': before, 'final_history': target,
                                     'history_len': int(rec['history_len']),
                                     **{name: _tensor_record(rec[name]) for name in _CANDIDATES}})
            for b in range(batch):
                for head in range(heads):
                    value = rec['candidate_raw'][b, head, :before]
                    self._pointers[(str(value.device), value.data_ptr(), before)] = (layer, b, head)
        baseline_effective = int((baseline & eligible).sum().item())
        effective = int((selected & eligible).sum().item())
        baseline_work = int((baseline*work).sum().item())
        selected_work = int((selected*work).sum().item())
        if baseline_effective != effective or baseline_work != selected_work:
            raise ValueError('Intervention changed effective merge count/work: use a matched eligible setup')
        intersection = int((baseline & selected).sum().item())
        union = int((baseline | selected).sum().item())
        self._row = {'schema_version': 1, 'status': 'observing', 'task': self._sample[0], 'doc_id': self._sample[1],
                     'strategy': self.strategy, 'seed': self.seed, 'merge_ratio': float(merge_ratio),
                     'native_kv_heads': heads, 'layers': layers, 'batch_size': batch,
                     'baseline_masks': baseline.cpu().tolist(), 'selected_masks': selected.cpu().tolist(),
                     'baseline_selected_count': counts, 'selected_count': selected.sum((1, 2)).cpu().tolist(),
                     'baseline_layer_counts': baseline.sum(-1).cpu().tolist(), 'selected_layer_counts': selected.sum(-1).cpu().tolist(),
                     'eligible_merge_heads': eligible.sum((1, 2)).cpu().tolist(),
                     'expected_effective_merge_count': effective, 'actual_merge_calls': 0,
                     'expected_removed_tokens': selected_work, 'actual_removed_tokens': 0,
                     'mask_hamming': int((baseline != selected).sum().item()),
                     'mask_jaccard': intersection/union if union else 1.0,
                     'layer_mutable_count': int(((baseline.sum(-1)>0) & (baseline.sum(-1)<heads)).sum().item()),
                     'candidate_hashes': candidate_hashes, 'merge_calls': [], 'prefill_cache_shapes': [],
                     'sampling': 'Fixed per-seed model-level ranking, uniform without replacement; no global RNG state changed'}
        self._selected = selected
        # The original condition returns the original list and tensor objects.
        return original if self.strategy == 'similarity' else [selected[:, layer, :] for layer in range(layers)]

    def _merge(self, *args, **kwargs):
        if self._row is None:
            raise RuntimeError('Merge occurred before the audited selection')
        function = self._original['_paired_merge_one_sequence']
        bound = inspect.signature(function).bind(*args, **kwargs).arguments
        raw = bound['raw_k_seq']
        identity = self._pointers.get((str(raw.device), raw.data_ptr(), int(raw.shape[0])))
        if identity is None or not bool(self._selected[identity[1], identity[0], identity[2]]):
            raise ValueError('A merge received a candidate head outside the selected original records')
        result = function(*args, **kwargs)
        target = int(bound['target_len'])
        if not isinstance(result, tuple) or len(result) != 3 or int(result[0].shape[0]) != target or int(result[1].shape[0]) != target:
            raise ValueError('Core merger did not return its declared fixed target length')
        layer, batch, head = identity
        entry = {'layer': layer, 'batch': batch, 'head': head, 'before': int(raw.shape[0]), 'after': target}
        if any((r['layer'],r['batch'],r['head']) == identity for r in self._row['merge_calls']):
            raise ValueError('The same selected head was merged more than once')
        self._row['merge_calls'].append(entry)
        self._row['actual_merge_calls'] += 1
        self._row['actual_removed_tokens'] += int(raw.shape[0])-target
        return result

    def _cache(self, *args, **kwargs):
        function = self._original['_set_layer_cache']
        bound = inspect.signature(function).bind(*args, **kwargs).arguments
        if self._row is None:
            raise RuntimeError('Cache finalized before audited selection')
        key, value = bound['key_states'], bound['value_states']
        layer = int(bound['layer_idx'])
        if layer != len(self._row['prefill_cache_shapes']) or tuple(key.shape) != tuple(value.shape):
            raise ValueError('Expected one ordered K/V cache finalization per layer')
        self._row['prefill_cache_shapes'].append({'layer': layer, 'key_shape': list(key.shape), 'value_shape': list(value.shape),
                                                  'key_dtype': str(key.dtype), 'value_dtype': str(value.dtype)})
        return function(*args, **kwargs)

    def finish_sample(self):
        if self._failure is not None or self._sample is None or self._row is None:
            raise RuntimeError('Sample has no complete successful audited selection')
        row = self._row
        if (row['actual_merge_calls'] != row['expected_effective_merge_count']
                or row['actual_removed_tokens'] != row['expected_removed_tokens']
                or len(row['prefill_cache_shapes']) != row['layers']):
            raise ValueError('Actual merge work or cache coverage differs from the matched intervention')
        row['status'] = 'completed'
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush()
        result = copy.deepcopy(row)
        self._sample = self._row = None
        self._pointers = {}
        return result

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Adapter instances cannot be entered twice')
        self._entered = True
        try:
            for name in _FUNCTIONS:
                function = getattr(self.module, name, None)
                if not callable(function) or getattr(function, '_kvrepack_head_selection_owner', None) is not None:
                    raise RuntimeError('Required Snap function missing or already instrumented')
                self._original[name] = function
            for name, method in zip(_FUNCTIONS, (self._gate, self._merge, self._cache)):
                original = self._original[name]
                @functools.wraps(original)
                def wrapped(*args, _method=method, **kwargs):
                    try:
                        return _method(*args, **kwargs)
                    except BaseException as error:
                        self._failure = f'{type(error).__name__}: {error}'
                        raise
                wrapped._kvrepack_head_selection_owner = self
                setattr(self.module, name, wrapped)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for name, function in self._original.items():
            setattr(self.module, name, function)

    def __exit__(self, *args):
        self._restore()
        return False

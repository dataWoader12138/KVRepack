"""Count-matched global head selection around the intact HybridKV merger.

Only the native [layer,batch,KV-head] merge mask changes. Classification,
per-head retained budgets, candidate generation and the merger stay native.
Different selected heads may legitimately remove different total token counts.
"""
from __future__ import annotations

import copy
import functools
import importlib
import inspect
import math
import sys

from .head_selection import _ranking, _tensor_record

STRATEGIES = ('similarity', 'random_global')
_FUNCTIONS = ('_hybrid_kvrepack_build_simhead_mask', '_hybrid_kvrepack_merge_one_sequence', 'allocate_budget_by_ratio')


def _plain(value):
    """Small metadata only: retain actual budget/classification values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if hasattr(value, 'value'):
        return _plain(value.value)
    raise TypeError(f'Unsupported Hybrid metadata type: {type(value).__name__}')


class HybridHeadSelectionAdapter:
    def __init__(self, module=None, *, strategy='similarity', seed=0):
        if strategy not in STRATEGIES or type(seed) is not int or seed < 0:
            raise ValueError('Require similarity/random_global and a nonnegative integer local seed')
        self.module = module or importlib.import_module('kvrepack.hybridkv_utils')
        self.owner = self.module.HybridKVCache
        self.strategy, self.seed = strategy, seed
        self._original = {}
        self._entered = False
        self._sample = None
        self._row = None
        self._completed = None
        self._cache = None
        self._selected = None
        self._failure = None
        self._tensor_records = {}
        self._classification = None

    def set_sample(self, task, doc_id):
        if not self._entered or self._sample is not None or not isinstance(task, str) or not task or type(doc_id) is not int or doc_id < 0:
            raise ValueError('Declare a valid task/doc_id after finishing the previous sample')
        self._sample = (task, doc_id)
        self._row = self._completed = self._cache = self._selected = self._failure = None
        self._tensor_records = {}
        self._classification = None

    def _record(self, tensor, *, cache=False):
        if tensor is None:
            return None
        if not cache:
            # Merge-local buffers may reuse an allocator address on the next
            # head, so pointer-keyed reuse is only safe for live snapshot inputs.
            return _tensor_record(tensor)
        # All snapshot source tensors remain alive together until finalization.
        # Repeated cos/sin and source views can reuse that snapshot fingerprint.
        key = (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
        if key not in self._tensor_records:
            self._tensor_records[key] = _tensor_record(tensor)
        return self._tensor_records[key]

    def _fingerprint(self, value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if hasattr(value, 'detach'):
            return self._record(value, cache=True)
        if isinstance(value, (list, tuple)):
            return [self._fingerprint(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._fingerprint(item) for key, item in value.items()}
        return _plain(value)

    def _snapshot(self, obj, layers):
        candidates = []
        for layer in range(layers):
            pruned = {}
            for name in ('pruned_raw_history_cache', 'pruned_rope_history_cache',
                         'pruned_value_history_cache', 'pruned_importance_cache', 'pruned_mrope_cache'):
                values = getattr(obj, name, None)
                pruned[name] = self._fingerprint(values[layer] if values is not None else None)
            retrieval = getattr(obj, 'hybrid_kvrepack_retrieval_records', None)
            candidates.append({'layer': layer, 'pruned': pruned,
                'retrieval': self._fingerprint(retrieval[layer] if retrieval is not None else None)})
        parameters = ('max_cache_len', 'avg_budget', 'sparse_budget_ratio', 'dynamic_budget_ratio',
            'retrieval_budget', 'local_len', 'chunk_size', 'budget_method', 'head_classify_method',
            'hybrid_kvrepack_compress_ratio', 'hybrid_kvrepack_k_diff_alpha',
            'hybrid_kvrepack_merge_score_mode', 'hybrid_kvrepack_max_group_size',
            'hybrid_kvrepack_static_merge_scope', 'hybrid_kvrepack_simhead_sim_space',
            'hybrid_kvrepack_k_merge_space', 'hybrid_kvrepack_value_merge')
        return {'head_scores': self._fingerprint(obj.hybrid_kvrepack_head_sim),
            'head_score_values': _plain(obj.hybrid_kvrepack_head_sim),
            'head_category': _plain(getattr(obj, 'head_category', None)),
            'calibrated_head_category': copy.deepcopy(self._classification),
            'retrieval_head_mapping': _plain(obj.retrieval_head_mapping),
            'pruned_head_mapping': _plain(obj.pruned_head_mapping),
            'sparse_budget_per_head': _plain(obj.sparse_budget_per_head),
            'parameters': {name: _plain(getattr(obj, name, None)) for name in parameters},
            'candidate_hashes': candidates}

    def _allocate(self, obj, *args, **kwargs):
        if self._sample is None or self._classification is not None:
            raise RuntimeError('Require one native budget allocation for the declared sample')
        category = getattr(obj, 'head_category', None)
        if category is None:
            raise ValueError('Native allocation did not expose the calibrated head categories')
        self._classification = {'values': _plain(category), 'tensor': self._record(category)}
        return self._original[_FUNCTIONS[2]](obj, *args, **kwargs)

    def _gate(self, obj, *args, **kwargs):
        import torch
        if self._sample is None:
            raise RuntimeError('Undeclared Hybrid head-selection sample')
        previous = getattr(obj, 'hybrid_kvrepack_merge_mask', None)
        result = self._original[_FUNCTIONS[0]](obj, *args, **kwargs)
        mask = getattr(obj, 'hybrid_kvrepack_merge_mask', None)
        if mask is None:
            return result  # Some early layers do not yet have all native scores.
        if self._cache is not None:
            if obj is not self._cache or not torch.equal(mask, self._selected):
                raise ValueError('A sample reused another cache or overwrote its selected mask')
            return result
        if previous is not None:
            raise ValueError('Start each input/condition with a fresh Hybrid cache')
        if not getattr(obj, 'hybridkv_kvrepack', False) or not getattr(obj, 'hybrid_kvrepack_simhead_enable', False):
            raise ValueError('Require the enabled original HybridKV-KVRepack head selector')
        if getattr(obj, 'hybrid_asymkv', False):
            raise ValueError('This intervention is not a Hybrid-AsymKV experiment')
        if getattr(obj, 'hybrid_kvrepack_k_merge_space', None) != 'raw_reapply':
            raise ValueError('Hybrid Random-head keeps the original raw-reapply K merger')
        if mask.ndim != 3 or mask.dtype != torch.bool:
            raise ValueError('Expected native Boolean [layers,batch,KV-heads] mask')
        layers, batch, heads = mask.shape
        ratio = min(max(float(obj.hybrid_kvrepack_simhead_merge_ratio), 0), 1)
        expected = min(layers*heads, math.ceil(layers*heads*ratio)) if ratio > 0 else 0
        counts = mask.sum((0, 2)).cpu().tolist()
        if any(count != expected for count in counts):
            raise ValueError('Original selector did not use its exact global ceil quota')
        baseline = mask
        selected = mask.clone()
        if self.strategy == 'random_global':
            # Flatten in the same [B,L,H] convention as the Snap intervention,
            # then return the native Hybrid [L,B,H] layout without resampling.
            ordered = torch.zeros((batch, layers, heads), dtype=torch.bool, device=mask.device)
            for b in range(batch):
                rank = _ranking(self.seed, 'global', layers, heads, b).sample(range(layers*heads), layers*heads)
                ordered[b].view(-1)[torch.tensor(rank[:counts[b]], device=mask.device, dtype=torch.long)] = True
            selected = ordered.permute(1, 0, 2).contiguous()
        snapshot = self._snapshot(obj, layers)
        if snapshot['sparse_budget_per_head'] is None or self._classification is None:
            raise ValueError('Head intervention requires the original allocated per-head budgets')
        self._cache, self._selected = obj, selected
        intersection, union = int((baseline & selected).sum()), int((baseline | selected).sum())
        self._row = dict(schema_version=1, status='observing', method='hybridkv_kvrepack',
            task=self._sample[0], doc_id=self._sample[1], strategy=self.strategy, seed=self.seed,
            layers=layers, batch_size=batch, native_kv_heads=heads, merge_ratio=ratio,
            mask_layout='layers,batch,kv_heads', baseline_masks=baseline.cpu().tolist(), selected_masks=selected.cpu().tolist(),
            baseline_selected_count=counts, selected_count=selected.sum((0,2)).cpu().tolist(),
            baseline_layer_counts=baseline.sum(-1).transpose(0,1).cpu().tolist(),
            selected_layer_counts=selected.sum(-1).transpose(0,1).cpu().tolist(),
            mask_hamming=int((baseline != selected).sum()), mask_jaccard=intersection/union if union else 1.,
            pre_selection=snapshot, merge_calls=[], actual_merge_calls=0, actual_removed_tokens=0,
            sampling='Fixed per-seed model-level uniform ranking without replacement; total native head quota matched; per-head budgets/classification never randomized')
        if self.strategy == 'random_global':
            obj.hybrid_kvrepack_merge_mask = selected
        else:
            # Keep both the native return value and native mask tensor identity.
            self._selected = baseline
        return result

    def _merge(self, obj, *args, **kwargs):
        import torch
        if self._row is None or obj is not self._cache:
            raise RuntimeError('Merge occurred before this sample’s audited head gate')
        function = self._original[_FUNCTIONS[1]]
        bound = inspect.signature(function).bind(obj, *args, **kwargs)
        bound.apply_defaults(); values = bound.arguments
        layer, b, head = values['layer_idx'], int(values['batch_idx']), values['head_idx']
        if type(layer) is not int or type(head) is not int or not bool(self._selected[layer,b,head]):
            raise ValueError('Original merger received an unselected or unidentified native head')
        if any((row['layer'],row['batch'],row['head']) == (layer,b,head) for row in self._row['merge_calls']):
            raise ValueError('The same selected native head was merged more than once')
        before = int(values['k_seq'].shape[0])
        entry = dict(layer=layer,batch=b,head=head,before=before,requested_target=int(values['target_len']),
            head_kind='retrieval' if head in obj.retrieval_head_mapping[layer] else 'pruned',
            retained_head_budget=int(obj.sparse_budget_per_head[layer][b][head]),
            inputs={name:self._record(values.get(name)) for name in
                ('k_seq','v_seq','raw_k_seq','importance','positions')})
        captured = {}
        def profile(frame, event, arg):
            if event == 'return' and frame.f_code is function.__code__:
                for name in ('merge_idx','group_ids','group_size','start_mask'):
                    if name in frame.f_locals: captured[name] = frame.f_locals[name]
                captured['effective_target'] = frame.f_locals.get('target_len')
        previous = sys.getprofile()
        if previous is not None:
            raise RuntimeError('Original group capture requires no active Python profiler')
        try:
            sys.setprofile(profile)
            result = function(obj, *args, **kwargs)
        finally:
            sys.setprofile(previous)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError('Original merger must return K,V,group-size')
        key, value, sizes = result
        after = int(key.shape[0])
        if key.shape != value.shape or int(sizes.shape[0]) != after or not 0 < after <= before:
            raise ValueError('Invalid original merged K/V/group-size shape')
        if captured.get('effective_target') != after:
            raise ValueError('Actual output does not match the native effective per-head target')
        if after < before and not all(name in captured for name in ('merge_idx','group_ids','group_size')):
            raise ValueError('Failed to capture the original function’s actual groups')
        entry.update(after=after,effective_target=after,removed_tokens=before-after,
            groups={name:self._record(captured[name]) for name in ('merge_idx','group_ids','group_size','start_mask') if name in captured},
            output_key_shape=list(key.shape),output_value_shape=list(value.shape),output_dtype=str(key.dtype))
        self._row['merge_calls'].append(entry)
        self._row['actual_merge_calls'] += int(after < before)
        self._row['actual_removed_tokens'] += before-after
        return result

    def finish_sample(self):
        import torch
        if self._failure is not None or self._sample is None or self._row is None:
            raise RuntimeError('Sample has no completed successful Hybrid head intervention')
        if not torch.equal(self._cache.hybrid_kvrepack_merge_mask, self._selected):
            raise ValueError('Native merge mask drifted after selection')
        # Selection must not secretly reallocate any head budget or classifier.
        original = self._row['pre_selection']
        for name in ('sparse_budget_per_head','head_category','retrieval_head_mapping','pruned_head_mapping'):
            current = _plain(getattr(self._cache,name))
            if current != original[name]:
                raise ValueError('Head intervention changed native classification/retained budgets: '+name)
        if self._row['actual_merge_calls'] <= 0:
            raise ValueError('Selected heads never exercised the original merger')
        self._row['merged_heads'] = [[r['layer'],r['batch'],r['head']] for r in self._row['merge_calls'] if r['after']<r['before']]
        self._row['status'] = 'completed'
        self._completed = copy.deepcopy(self._row)
        self._sample = self._row = self._cache = self._selected = None
        self._tensor_records = {}
        return copy.deepcopy(self._completed)

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Adapter instances cannot be nested/reentered')
        self._entered = True
        try:
            for name in _FUNCTIONS:
                function = getattr(self.owner,name,None)
                if (not callable(function) or getattr(function,'_hybrid_head_selection_owner',None) is not None
                        or getattr(function,'_rope_merge_owner',None) is not None):
                    raise RuntimeError('Required original Hybrid function missing or already instrumented')
                self._original[name] = function
            for name, method in zip(_FUNCTIONS,(self._gate,self._merge,self._allocate)):
                original = self._original[name]
                @functools.wraps(original)
                def wrapped(*args,_method=method,**kwargs):
                    try:return _method(*args,**kwargs)
                    except BaseException as error:
                        self._failure = f'{type(error).__name__}: {error}'
                        raise
                wrapped._hybrid_head_selection_owner = self
                setattr(self.owner,name,wrapped)
        except BaseException:
            self._restore();raise
        return self

    def _restore(self):
        for name,function in self._original.items():setattr(self.owner,name,function)

    def __exit__(self,*args):self._restore()


def compare_hybrid_arms(adapters):
    """Freeze original candidates/classification/budgets; retain work differences."""
    audits = [adapter._completed if isinstance(adapter,HybridHeadSelectionAdapter) else adapter for adapter in adapters]
    if not audits or any(not isinstance(row,dict) or row.get('status')!='completed' for row in audits):
        raise ValueError('Require completed Hybrid selector audits')
    reference = audits[0]
    if reference['strategy'] != 'similarity':raise ValueError('First condition must be original similarity')
    for row in audits:
        for name in ('method','task','doc_id','layers','native_kv_heads','batch_size','merge_ratio',
                     'baseline_masks','baseline_selected_count','selected_count','pre_selection'):
            if row[name] != reference[name]:
                raise ValueError('Uncontrolled Hybrid selector pairing: '+name)
        calls = row['merge_calls']
        if sum(call['before']-call['after'] for call in calls) != row['actual_removed_tokens']:
            raise ValueError('Actual merge work counters disagree')
    return dict(status='passed',method='hybridkv_kvrepack',task=reference['task'],doc_id=reference['doc_id'],
        conditions=len(audits),selected_count=reference['selected_count'],
        actual_merge_calls=[row['actual_merge_calls'] for row in audits],
        actual_removed_tokens=[row['actual_removed_tokens'] for row in audits],
        checks='Same original head scores/classification, full pre-selection candidate bytes and each native retained head budget; same global selected quota; unchanged original merger',
        work_scope='Selected identities can change static/dynamic mixture, candidate expansion and total removed tokens under heterogeneous native head budgets. These measured consequences are retained, not resampled or reallocated.')

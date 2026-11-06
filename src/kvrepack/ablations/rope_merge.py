"""Experimental K aggregation space with unchanged selection and group formation.

Core files stay intact. Baseline calls the original merger unchanged; SnapKV's
post-RoPE function is compiled from its source AST with only the K reduction
changed. HybridKV uses its existing rope reduction branch only inside one merge
call, leaving all cache-wide policy flags and selection paths unchanged.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import importlib
import inspect
import json
import sys
import textwrap

METHODS = {'snapkv': 'snapkv_kvrepack', 'hybridkv': 'hybridkv_kvrepack',
           'snapkv_kvrepack': 'snapkv_kvrepack', 'hybridkv_kvrepack': 'hybridkv_kvrepack'}
VARIANTS = ('raw_reapply', 'post_rope')


def tensor_record(value):
    import torch
    if value is None:
        return None
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(raw).hexdigest()}


def _source_sha(function):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()


def _snap_post_rope(function):
    """Replace only the FP32 K mean source and remove anchor reapplication."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    node = tree.body[0]
    expected = ['raw_sum', 'raw_comp', 'anchor_pos', 'cos_sel', 'sin_sel', 'k_comp']
    found = []
    for child in node.body:
        if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
            name = child.targets[0].id
            if name in expected:
                found.append(name)
    # raw_sum appears once as an assignment; its index_add_ is an expression.
    if found != expected:
        raise ValueError('Snap merger source no longer matches the reviewed K reduction block')
    replacement = ast.parse('''rope_sum = torch.zeros((group_count, rope_k_seq.shape[-1]), device=rope_k_seq.device, dtype=torch.float32)
rope_sum.index_add_(0, group_ids, rope_k_seq.float())
k_comp = (rope_sum / denom).to(rope_k_seq.dtype)
''').body
    output = []
    for child in node.body:
        name = child.targets[0].id if isinstance(child, ast.Assign) and len(child.targets)==1 and isinstance(child.targets[0], ast.Name) else None
        if name == 'raw_sum':
            output.extend(copy.deepcopy(replacement[:2]))
        elif name in {'raw_comp', 'anchor_pos', 'cos_sel', 'sin_sel'}:
            continue
        elif name == 'k_comp':
            output.append(copy.deepcopy(replacement[2]))
        elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call) and isinstance(child.value.func, ast.Attribute) and isinstance(child.value.func.value, ast.Name) and child.value.func.value.id == 'raw_sum':
            continue
        else:
            output.append(child)
    node.body = output
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    transformed = ast.unparse(tree)+'\n'
    exec(compile(tree, function.__code__.co_filename+'<post_rope_experiment>', 'exec'), namespace)
    result = namespace[function.__name__]
    return result, {'original_function_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'post_function_source_sha256': hashlib.sha256(transformed.encode()).hexdigest(),
                    'transformation': 'K only: mean(raw K) then anchor mRoPE -> mean(native post-mRoPE K), using identical original group_ids; V block untouched',
                    'post_function_source': transformed}


class RopeMergeAdapter:
    def __init__(self, method, variant, module=None):
        if method not in METHODS or variant not in VARIANTS:
            raise ValueError('Require SnapKV/HybridKV and raw_reapply/post_rope')
        self.method, self.variant = METHODS[method], variant
        if module is None:
            name = 'kvrepack.qwen_model_snapkv_kvrepack' if self.method=='snapkv_kvrepack' else 'kvrepack.hybridkv_utils'
            module = importlib.import_module(name)
        self.module = module
        self.owner = module if self.method=='snapkv_kvrepack' else module.HybridKVCache
        self.merge_name = '_paired_merge_one_sequence' if self.method=='snapkv_kvrepack' else '_hybrid_kvrepack_merge_one_sequence'
        self.original = getattr(self.owner, self.merge_name)
        self.post = self.original
        self.transform_spec = {'original_function_source_sha256': _source_sha(self.original),
                               'transformation': 'Original function; Hybrid post_rope temporarily selects its native rope K reduction branch only during this function'}
        if self.method=='snapkv_kvrepack':
            self.post, self.transform_spec = _snap_post_rope(self.original)
        self._patches = []
        self._entered = False
        self._sample = None
        self._audit = None
        self._values = []
        self._keys = []
        self._finished = False

    def set_sample(self, task, doc_id):
        if not self._entered or self._sample is not None or not isinstance(task,str) or type(doc_id) is not int or doc_id<0:
            raise ValueError('Declare one valid sample inside the adapter scope')
        self._sample = (task,doc_id)
        self._audit = {'schema_version': 1, 'status': 'observing', 'method': self.method, 'variant': self.variant,
                       'task': task, 'doc_id': doc_id, 'merge_calls': [], 'head_selection': [],
                       'prefill_cache_shapes': [], 'actual_merge_calls': 0, 'actual_removed_tokens': 0,
                       'k_aggregation': 'raw K mean + anchor mRoPE' if self.variant=='raw_reapply' else 'native post-mRoPE K mean; no reapplication'}

    def _merge(self, *args, **kwargs):
        if self._audit is None or self._finished:
            raise RuntimeError('Merge occurred outside a declared sample')
        import torch
        bound = inspect.signature(self.original).bind(*args, **kwargs)
        bound.apply_defaults(); values=bound.arguments
        snap = self.method=='snapkv_kvrepack'
        obj = values.get('self')
        if not snap and (obj.hybrid_kvrepack_k_merge_space!='raw_reapply' or getattr(obj,'hybrid_asymkv',False)):
            raise ValueError('Pure K-space intervention requires ordinary Hybrid-KVRepack raw_reapply baseline')
        rope = values['rope_k_seq' if snap else 'k_seq']
        fields = {'raw_k': values.get('raw_k_seq'), 'rope_k': rope, 'value': values['v_seq'],
                  'importance': values['importance'], 'positions': values.get('positions')}
        call = {'call_id': len(self._audit['merge_calls']), 'before': int(rope.shape[0]),
                'requested_target': int(values['target_len']),
                'layer_idx': values.get('layer_idx'), 'head_idx': values.get('head_idx'),
                'batch_idx': int(values.get('batch_idx',0)), 'inputs': {k:tensor_record(v) for k,v in fields.items()}}
        if snap:
            call.update(alpha=float(values['k_diff_alpha']), merge_score_mode=values['merge_score_mode'])
        else:
            call.update(alpha=float(obj.hybrid_kvrepack_k_diff_alpha), merge_score_mode=obj.hybrid_kvrepack_merge_score_mode)
        target_function = self.post if snap and self.variant=='post_rope' else self.original
        locals_seen = {}
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is target_function.__code__:
                for key in ('merge_idx','group_ids','group_size','start_mask'):
                    if key in frame.f_locals: locals_seen[key]=frame.f_locals[key]
        previous=sys.getprofile()
        if previous is not None:
            raise RuntimeError('Rope merger local capture requires no pre-existing Python profiler')
        old_space = obj.hybrid_kvrepack_k_merge_space if obj is not None else None
        try:
            if not snap and self.variant=='post_rope': obj.hybrid_kvrepack_k_merge_space='rope'
            sys.setprofile(profile)
            result=target_function(*args,**kwargs)
        finally:
            sys.setprofile(previous)
            if obj is not None:obj.hybrid_kvrepack_k_merge_space=old_space
        if not isinstance(result,tuple) or len(result)!=3:
            raise ValueError('Original merger did not return K,V,group_size')
        k,v,sizes=result;after=int(k.shape[0]);call['after']=after
        if tuple(k.shape)!=tuple(v.shape) or sizes.shape[0]!=after:
            raise ValueError('Merged K,V,group_size shapes disagree')
        if after<call['before'] and not all(key in locals_seen for key in ('merge_idx','group_ids','group_size')):
            raise ValueError('Could not capture actual original merge groups')
        call['groups']={key:tensor_record(value) for key,value in locals_seen.items()}
        call.update(output_key=tensor_record(k),output_value=tensor_record(v),output_group_size=tensor_record(sizes))
        self._audit['merge_calls'].append(call)
        self._audit['actual_merge_calls']+=int(after<call['before'])
        self._audit['actual_removed_tokens']+=call['before']-after
        self._values.append(v.detach().cpu().clone())
        self._keys.append(k.detach().cpu().clone())
        return result

    def _patch(self, owner, name, replacement):
        original=getattr(owner,name)
        if getattr(original,'_rope_merge_owner',None) is not None:
            raise RuntimeError('Rope merge adapters cannot be nested on the same target')
        self._patches.append((owner,name,original))
        replacement._rope_merge_owner=self
        setattr(owner,name,replacement)

    def __enter__(self):
        if self._entered:raise RuntimeError('Adapter already entered')
        self._entered=True
        @functools.wraps(self.original)
        def merge(*args,**kwargs):return self._merge(*args,**kwargs)
        try:
            self._patch(self.owner,self.merge_name,merge)
            if self.method=='snapkv_kvrepack':
                original_gate=self.module._global_merge_masks
                @functools.wraps(original_gate)
                def gate(records,ratio):
                    result=original_gate(records,ratio)
                    if self._audit is None:raise RuntimeError('Undeclared head selection')
                    fields=('candidate_raw','candidate_rope','candidate_value','candidate_imp','candidate_idx','head_sim')
                    self._audit['head_selection'].append({'merge_ratio':float(ratio),'masks':[tensor_record(x) for x in result],
                        'candidates':[{'layer':r['layer_idx'],'history_len':r.get('history_len'),
                          'candidate_history':r.get('candidate_history'),'final_history':r.get('final_history'),
                          **{k:tensor_record(r.get(k)) for k in fields}} for r in records]})
                    return result
                self._patch(self.module,'_global_merge_masks',gate)
                original_cache=self.module._set_layer_cache
                @functools.wraps(original_cache)
                def cache(*args,**kwargs):
                    b=inspect.signature(original_cache).bind(*args,**kwargs).arguments
                    self._audit['prefill_cache_shapes'].append({'layer':int(b['layer_idx']),
                        'key_shape':list(b['key_states'].shape),'value_shape':list(b['value_states'].shape)})
                    return original_cache(*args,**kwargs)
                self._patch(self.module,'_set_layer_cache',cache)
            else:
                original_gate=self.owner._hybrid_kvrepack_build_simhead_mask
                @functools.wraps(original_gate)
                def gate(obj,*args,**kwargs):
                    result=original_gate(obj,*args,**kwargs)
                    mask=getattr(obj,'hybrid_kvrepack_merge_mask',None)
                    if mask is not None and not self._audit['head_selection']:
                        self._audit['head_selection'].append({'masks':tensor_record(mask),
                            'scores':[tensor_record(v) for v in obj.hybrid_kvrepack_head_sim],
                            'retrieval_head_mapping':copy.deepcopy(obj.retrieval_head_mapping),
                            'pruned_head_mapping':copy.deepcopy(obj.pruned_head_mapping) if hasattr(obj,'pruned_head_mapping') else None,
                            'merge_ratio':float(obj.hybrid_kvrepack_simhead_merge_ratio)})
                    return result
                self._patch(self.owner,'_hybrid_kvrepack_build_simhead_mask',gate)
        except BaseException:
            self._restore();raise
        return self

    def finish_sample(self):
        if self._audit is None or self._finished or not self._audit['head_selection'] or not self._audit['merge_calls']:
            raise RuntimeError('Sample has no complete selected-head merge audit')
        if self._audit['actual_merge_calls']<=0:
            raise RuntimeError('The declared K aggregation intervention was not exercised')
        self._audit['status']='completed';self._finished=True
        return copy.deepcopy(self._audit)

    def _restore(self):
        for owner,name,original in reversed(self._patches):setattr(owner,name,original)
        self._patches=[]

    def __exit__(self,*args):self._restore()


def compare_merge_arms(raw, post):
    """Compare pre-aggregation facts exactly; quantify only narrow V rounding."""
    import torch
    if not raw._finished or not post._finished or raw.method!=post.method or raw.variant!='raw_reapply' or post.variant!='post_rope':
        raise ValueError('Require completed raw_reapply and post_rope adapters for the same method')
    a,b=raw._audit,post._audit
    for name in ('task','doc_id','method','head_selection','actual_merge_calls','actual_removed_tokens'):
        if a[name]!=b[name]:raise ValueError('Paired aggregation intervention differs before K reduction: '+name)
    if raw.method=='snapkv_kvrepack' and a['prefill_cache_shapes']!=b['prefill_cache_shapes']:
        raise ValueError('SnapKV fixed prefill budget changed')
    if len(a['merge_calls'])!=len(b['merge_calls']):raise ValueError('Merge call coverage differs')
    n_diff=0;max_abs=0.;key_diff=0;key_max=0.;nonexact_calls=0
    for i,(left,right,v0,v1,k0,k1) in enumerate(zip(a['merge_calls'],b['merge_calls'],raw._values,post._values,raw._keys,post._keys)):
        for field in ('call_id','before','requested_target','after','layer_idx','head_idx','batch_idx','inputs','groups','output_group_size','alpha','merge_score_mode'):
            if left[field]!=right[field]:raise ValueError(f'Merge call {i} changed grouping/input/work: {field}')
        if v0.shape!=v1.shape or v0.dtype!=v1.dtype:raise ValueError('V output shapes/dtypes differ')
        delta=(v0.float()-v1.float()).abs();changed=v0!=v1
        n_diff+=int(changed.sum());nonexact_calls+=int(bool(changed.any()));max_abs=max(max_abs,float(delta.max()) if delta.numel() else 0.)
        if v0.dtype in (torch.bfloat16,torch.float16):
            allowed=(torch.nextafter(v0,v1).float()-v0.float()).abs()
            valid=delta<=allowed
        else:
            valid=torch.isclose(v0,v1,rtol=1e-5,atol=1e-6)
        if not bool(valid.all()):raise ValueError(f'Merge call {i} changed V beyond the declared rounding tolerance')
        kd=(k0.float()-k1.float()).abs();key_diff+=int(not torch.equal(k0,k1));key_max=max(key_max,float(kd.max()) if kd.numel() else 0.)
    return {'status':'passed','method':raw.method,'task':a['task'],'doc_id':a['doc_id'],
            'checks':'head selection, exact pre-merge inputs/groups/positions, targets and actual work; Snap prefill shape; unchanged V reduction within narrow rounding bound',
            'merge_calls':len(a['merge_calls']),'actual_merge_calls':a['actual_merge_calls'],
            'actual_removed_tokens':a['actual_removed_tokens'],
            'value_bitwise_equal':n_diff==0,'value_differing_elements':n_diff,'value_nonexact_calls':nonexact_calls,'value_max_abs_difference':max_abs,
            'value_tolerance':'BF16/FP16 at most one representable step toward comparison value; other floating types rtol1e-5,atol1e-6',
            'key_changed_calls':key_diff,'key_max_abs_difference':key_max,
            'hybrid_downstream_retrieval_note':'Retrieval indices and assembled decode caches may legitimately change after K intervention; no final-cache K/V equality asserted'}

"""Use actual core source functions on CPU to isolate K aggregation space."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict,List,Optional,Tuple
import unittest

import torch
from kvrepack.merge_score import merge_score_from_importance
from kvrepack.ablations.rope_merge import RopeMergeAdapter,compare_merge_arms

ROOT=Path(__file__).resolve().parents[1]/'src/kvrepack'

def source_module(method):
    path=ROOT/('qwen_model_snapkv_kvrepack.py' if method=='snapkv' else 'hybridkv_utils.py')
    tree=ast.parse(path.read_text())
    ns=dict(torch=torch,math=math,Tuple=Tuple,List=List,Dict=Dict,Optional=Optional,Cache=object,
            merge_score_from_importance=merge_score_from_importance,__file__=str(path),__name__='cpu_actual_core')
    helpers={'_select_mrope_batch','_rotate_half','_apply_mrope_to_merged_keys'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in helpers]
    if method=='snapkv':
        nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_paired_merge_one_sequence','_global_merge_masks','_set_layer_cache'}]
    else:
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HybridKVCache')
        names={'_hybrid_kvrepack_effective_target_len','_hybrid_kvrepack_select_merge_indices','_hybrid_kvrepack_merge_one_sequence'}
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        nodes += [ast.ClassDef(name='HybridKVCache',bases=[],keywords=[],body=methods,decorator_list=[])]
    module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,str(path),'exec'),ns)
    obj=SimpleNamespace(**ns)
    if method=='hybridkv':
        # Gate itself is unrelated to this arithmetic fixture; record its
        # already-declared native mask while executing the actual merger.
        def gate(self):return None
        obj.HybridKVCache._hybrid_kvrepack_build_simhead_mask=gate
    return obj


def sequence(identity_rope=False):
    raw=torch.randn((6,6),generator=torch.Generator().manual_seed(42))
    values=torch.randn((6,6),generator=torch.Generator().manual_seed(7))
    pos=torch.tensor([0,2,3,5,7,9]);angle=torch.arange(10).float()*.37
    cos=torch.cos(angle).view(1,1,10,1).expand(3,1,10,6).clone()
    sin=torch.sin(angle).view(1,1,10,1).expand(3,1,10,6).clone()
    if identity_rope:cos.fill_(1);sin.zero_()
    half=torch.cat([-raw[:,3:],raw[:,:3]],-1)
    rope=raw*cos[0,0,pos]+half*sin[0,0,pos]
    return dict(raw_k_seq=raw,rope_k_seq=rope,v_seq=values,importance=torch.tensor([.1,.2,1.,2.,3.,4.]),
                target_len=3,positions=pos,cos=cos,sin=sin,batch_idx=0,k_diff_alpha=2.,merge_score_mode='full',mrope_section=[1,1,1])


def evaluate(method,variant,seq):
    module=source_module(method)
    adapter=RopeMergeAdapter(method,variant,module=module)
    with adapter:
        adapter.set_sample('fixture',0)
        if method=='snapkv':
            module._global_merge_masks([{'layer_idx':0,'head_sim':torch.tensor([[.8]])}],1.)
            result=module._paired_merge_one_sequence(**seq)
        else:
            obj=module.HybridKVCache()
            obj.hybrid_kvrepack_k_merge_space='raw_reapply';obj.hybrid_asymkv=False
            obj.hybrid_kvrepack_max_group_size=0;obj.hybrid_kvrepack_k_diff_alpha=2.
            obj.hybrid_kvrepack_merge_score_mode='full';obj.head_dim=6
            obj.hybrid_kvrepack_merge_mask=torch.ones((1,1,1),dtype=torch.bool)
            obj.hybrid_kvrepack_head_sim=[torch.tensor([[.8]])];obj.retrieval_head_mapping=[[0]]
            obj.hybrid_kvrepack_simhead_merge_ratio=1.
            obj._hybrid_kvrepack_build_simhead_mask()
            kw={k:v for k,v in seq.items() if k not in {'rope_k_seq','k_diff_alpha','merge_score_mode'}}
            kw.update(k_seq=seq['rope_k_seq'],layer_idx=0,head_idx=0)
            result=obj._hybrid_kvrepack_merge_one_sequence(**kw)
            assert obj.hybrid_kvrepack_k_merge_space=='raw_reapply'
        audit=adapter.finish_sample()
    return adapter,result,audit


class RopeMergeTests(unittest.TestCase):
    def test_snap_baseline_is_original_and_post_is_exact_grouped_rope_mean(self):
        seq=sequence();module=source_module('snapkv');expected=module._paired_merge_one_sequence(**seq)
        raw,result,a=evaluate('snapkv','raw_reapply',seq)
        for x,y in zip(expected,result):self.assertTrue(torch.equal(x,y))
        post,changed,b=evaluate('snapkv','post_rope',seq)
        fair=compare_merge_arms(raw,post)
        self.assertTrue(fair['value_bitwise_equal']);self.assertEqual(fair['key_changed_calls'],1)
        importance=seq['importance'];scores=merge_score_from_importance(importance,2.,'full')
        edges=torch.topk(scores,k=3,largest=False,sorted=False).indices
        start=torch.ones(6,dtype=torch.bool);start[edges+1]=False;groups=start.long().cumsum(0)-1
        wanted=torch.stack([seq['rope_k_seq'][groups==i].mean(0) for i in range(3)])
        torch.testing.assert_close(changed[0],wanted,rtol=0,atol=0)
        self.assertEqual(a['merge_calls'][0]['groups'],b['merge_calls'][0]['groups'])

    def test_hybrid_native_rope_branch_preserves_groups_values_and_restores_flag(self):
        seq=sequence();raw,result,a=evaluate('hybridkv','raw_reapply',seq)
        post,changed,b=evaluate('hybridkv','post_rope',seq)
        fair=compare_merge_arms(raw,post)
        self.assertTrue(fair['value_bitwise_equal']);self.assertEqual(fair['key_changed_calls'],1)
        self.assertEqual(a['merge_calls'][0]['groups'],b['merge_calls'][0]['groups'])
        torch.testing.assert_close(result[1],changed[1],rtol=0,atol=0)

    def test_identity_position_rotation_makes_the_two_spaces_equal(self):
        for method in ('snapkv','hybridkv'):
            raw,result,_=evaluate(method,'raw_reapply',sequence(True))
            post,changed,_=evaluate(method,'post_rope',sequence(True))
            fair=compare_merge_arms(raw,post)
            self.assertEqual(fair['key_changed_calls'],0)
            for x,y in zip(result,changed):self.assertTrue(torch.equal(x,y))

    def test_source_function_attributes_restore_after_exception(self):
        module=source_module('snapkv');original=module._paired_merge_one_sequence
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):
            with RopeMergeAdapter('snapkv','post_rope',module=module):raise RuntimeError('fixture failure')
        self.assertIs(module._paired_merge_one_sequence,original)

    def test_pairing_rejects_changed_groups_and_values_beyond_one_ulp(self):
        raw,_,_=evaluate('snapkv','raw_reapply',sequence())
        post,_,_=evaluate('snapkv','post_rope',sequence())
        post._values[0][0,0]+=1
        with self.assertRaisesRegex(ValueError,'V beyond'):compare_merge_arms(raw,post)
        post,_,_=evaluate('snapkv','post_rope',sequence())
        post._audit['merge_calls'][0]['groups']['group_ids']['sha256']='changed'
        with self.assertRaisesRegex(ValueError,'grouping/input/work'):compare_merge_arms(raw,post)


if __name__=='__main__':unittest.main()

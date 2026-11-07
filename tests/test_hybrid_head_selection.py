"""Actual native gate/merger on small CPU tensors; no model or CUDA."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import List, Optional, Tuple
import unittest

import torch
from kvrepack.merge_score import merge_score_from_importance
from kvrepack.ablations.hybrid_head_selection import HybridHeadSelectionAdapter, compare_hybrid_arms

SOURCE=Path(__file__).resolve().parents[1]/'src/kvrepack/hybridkv_utils.py'


def fixture():
    tree=ast.parse(SOURCE.read_text())
    original=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='HybridKVCache')
    names={'_hybrid_kvrepack_build_simhead_mask','_hybrid_kvrepack_effective_target_len',
        '_hybrid_kvrepack_select_merge_indices','_hybrid_kvrepack_merge_one_sequence'}
    body=[node for node in original.body if isinstance(node,ast.FunctionDef) and node.name in names]
    body += ast.parse('''def allocate_budget_by_ratio(self, key_states=None):
    self.head_category = None
    return self.budget_return
def _hybrid_kvrepack_dump_merge_info(self, sims, ratio):
    pass
''').body
    helpers=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in
        {'_rotate_half','_select_mrope_batch','_apply_mrope_to_merged_keys'}]
    cls=ast.ClassDef(name='HybridKVCache',bases=[],keywords=[],body=body,decorator_list=[])
    module=ast.Module(body=helpers+[cls],type_ignores=[]);ast.fix_missing_locations(module)
    namespace=dict(torch=torch,math=math,Tuple=Tuple,List=List,Optional=Optional,
        merge_score_from_importance=merge_score_from_importance,__file__=str(SOURCE))
    exec(compile(module,str(SOURCE),'exec'),namespace)
    loaded=SimpleNamespace(**namespace);cache=loaded.HybridKVCache()
    cache.hybridkv_kvrepack=True;cache.hybrid_asymkv=False;cache.hybrid_kvrepack_simhead_enable=True
    cache.hybrid_kvrepack_simhead_merge_ratio=.3;cache.hybrid_kvrepack_merge_mask=None
    cache.hybrid_kvrepack_verbose=False;cache.hybrid_kvrepack_k_merge_space='raw_reapply'
    cache.hybrid_kvrepack_simhead_sim_space='raw';cache.hybrid_kvrepack_static_merge_scope='visual'
    cache.hybrid_kvrepack_max_group_size=0;cache.hybrid_kvrepack_k_diff_alpha=2.
    cache.hybrid_kvrepack_merge_score_mode='full';cache.hybrid_kvrepack_compress_ratio=.5
    cache.head_dim=6;cache.num_layers=4;cache.num_key_value_heads=4
    cache.hybrid_kvrepack_head_sim=[torch.tensor([[16.-4*l-h for h in range(4)]]) for l in range(4)]
    cache.retrieval_head_mapping=[[1,3] for _ in range(4)];cache.pruned_head_mapping=[[0,2] for _ in range(4)]
    cache.head_category=torch.tensor([[[False,True,False,True]]] * 4)
    cache.sparse_budget_per_head=[[[1+(l+h)%4 for h in range(4)]] for l in range(4)]
    cache.budget_return=object()
    cache.pruned_raw_history_cache=[];cache.pruned_rope_history_cache=[];cache.pruned_value_history_cache=[]
    cache.pruned_importance_cache=[];cache.pruned_mrope_cache=[];cache.hybrid_kvrepack_retrieval_records=[]
    sequence={};cos=torch.ones((3,1,8,6));sin=torch.zeros_like(cos)
    for layer in range(4):
        keys=torch.stack([torch.randn(8,6,generator=torch.Generator().manual_seed(100*layer+head)) for head in range(4)]).unsqueeze(0)
        values=keys+0.125;imp=torch.linspace(.1,1.,8).expand(1,4,8).clone()
        cache.pruned_raw_history_cache.append(keys[:,[0,2]])
        cache.pruned_rope_history_cache.append(keys[:,[0,2]])
        cache.pruned_value_history_cache.append(values[:,[0,2]])
        cache.pruned_importance_cache.append(imp[:,[0,2]])
        cache.pruned_mrope_cache.append((cos,sin,[1,1,1]))
        cache.hybrid_kvrepack_retrieval_records.append(dict(key_states_offload=keys[:,[1,3]],
            value_states_offload=values[:,[1,3]],raw_key_states_offload=keys[:,[1,3]],
            importance=imp[:,[1,3]],history_len=8,seq_len=8,cos=cos,sin=sin,mrope_section=[1,1,1]))
        for head in range(4):
            sequence[layer,head]=dict(k_seq=keys[0,head],v_seq=values[0,head],raw_k_seq=keys[0,head],
                importance=imp[0,head],target_len=cache.sparse_budget_per_head[layer][0][head],positions=torch.arange(8),
                cos=cos,sin=sin,batch_idx=0,mrope_section=[1,1,1],layer_idx=layer,head_idx=head)
    return loaded,cache,sequence


def evaluate(strategy='similarity',seed=0):
    module,cache,sequences=fixture();returns={}
    adapter=HybridHeadSelectionAdapter(module,strategy=strategy,seed=seed)
    with adapter:
        adapter.set_sample('fixture',0)
        expected=cache.budget_return
        assert cache.allocate_budget_by_ratio() is expected
        assert cache._hybrid_kvrepack_build_simhead_mask() is None
        selected=cache.hybrid_kvrepack_merge_mask
        assert cache._hybrid_kvrepack_build_simhead_mask() is None
        assert cache.hybrid_kvrepack_merge_mask is selected
        for layer in range(4):
            for head in range(4):
                if bool(selected[layer,0,head]):
                    returns[layer,head]=cache._hybrid_kvrepack_merge_one_sequence(**sequences[layer,head])
        audit=adapter.finish_sample()
    return adapter,audit,returns


class HybridHeadSelectionTests(unittest.TestCase):
    def test_similarity_preserves_original_gate_mask_allocation_and_merge_results(self):
        module,cache,sequences=fixture();cache.allocate_budget_by_ratio();cache._hybrid_kvrepack_build_simhead_mask()
        expected={(l,h):cache._hybrid_kvrepack_merge_one_sequence(**sequences[l,h])
            for l in range(4) for h in range(4) if bool(cache.hybrid_kvrepack_merge_mask[l,0,h])}
        adapter,audit,actual=evaluate()
        self.assertEqual(audit['selected_masks'],cache.hybrid_kvrepack_merge_mask.tolist())
        self.assertEqual(audit['selected_count'],[5]);self.assertEqual(audit['actual_merge_calls'],5)
        self.assertEqual(audit['pre_selection']['calibrated_head_category']['values'],[[[False,True,False,True]]]*4)
        for identity in expected:
            for a,b in zip(expected[identity],actual[identity]):self.assertTrue(torch.equal(a,b))

    def test_random_changes_only_head_identity_under_heterogeneous_native_budgets(self):
        baseline,original,_=evaluate();adapters=[baseline]
        for seed in range(3):
            adapter,audit,_=evaluate('random_global',seed);adapters.append(adapter)
            self.assertEqual(audit['selected_count'],[5]);self.assertEqual(audit['actual_merge_calls'],5)
            self.assertEqual(audit['pre_selection'],original['pre_selection'])
        fair=compare_hybrid_arms(adapters)
        self.assertEqual(fair['status'],'passed')
        self.assertGreater(len(set(fair['actual_removed_tokens'])),1)
        self.assertEqual(len({str(a._completed['selected_masks']) for a in adapters[1:]}),3)
        repeated,again,_=evaluate('random_global',0)
        self.assertEqual(again['selected_masks'],adapters[1]._completed['selected_masks'])

    def test_local_selection_does_not_change_global_rng_or_restore_targets_incorrectly(self):
        module,cache,sequences=fixture();original=module.HybridKVCache._hybrid_kvrepack_build_simhead_mask
        state=random.getstate();torchstate=torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'fixture stop'):
            with HybridHeadSelectionAdapter(module,strategy='random_global',seed=2) as adapter:
                adapter.set_sample('fixture',0);cache.allocate_budget_by_ratio();cache._hybrid_kvrepack_build_simhead_mask()
                self.assertEqual(random.getstate(),state);self.assertTrue(torch.equal(torch.random.get_rng_state(),torchstate))
                raise RuntimeError('fixture stop')
        self.assertIs(module.HybridKVCache._hybrid_kvrepack_build_simhead_mask,original)

    def test_budget_drift_and_unselected_merge_are_rejected(self):
        module,cache,sequences=fixture()
        with HybridHeadSelectionAdapter(module) as adapter:
            adapter.set_sample('fixture',0);cache.allocate_budget_by_ratio();cache._hybrid_kvrepack_build_simhead_mask()
            cache._hybrid_kvrepack_merge_one_sequence(**sequences[0,0])
            cache.sparse_budget_per_head[0][0][0]+=1
            with self.assertRaisesRegex(ValueError,'budgets'):adapter.finish_sample()
        module,cache,sequences=fixture()
        with HybridHeadSelectionAdapter(module) as adapter:
            adapter.set_sample('fixture',0);cache.allocate_budget_by_ratio();cache._hybrid_kvrepack_build_simhead_mask()
            with self.assertRaisesRegex(ValueError,'unselected'):cache._hybrid_kvrepack_merge_one_sequence(**sequences[3,3])

    def test_pairing_rejects_changed_classification_or_candidates(self):
        baseline,_,_=evaluate();other,_,_=evaluate('random_global',0)
        other._completed['pre_selection']['candidate_hashes'][0]['pruned']['pruned_raw_history_cache']['sha256']='changed'
        with self.assertRaisesRegex(ValueError,'pre_selection'):compare_hybrid_arms([baseline,other])


if __name__=='__main__':unittest.main()

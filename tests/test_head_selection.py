"""Run the real source selector/finalizer on CPU with a deterministic merger."""
import ast
import copy
from pathlib import Path
import random
import types
import unittest

import torch

from kvrepack.ablations.head_selection import SnapHeadSelectionAdapter

CORE = Path(__file__).resolve().parents[1]/'src/kvrepack/qwen_model_snapkv_kvrepack.py'


def fixture():
    module = types.ModuleType('snap_test')
    module.__dict__.update(torch=torch, math=__import__('math'), List=list, Dict=dict, Cache=object)
    wanted = {'_global_merge_masks', '_topk_sorted', '_topk_from_candidates_sorted', '_finalize_prompt_cache'}
    tree = ast.parse(CORE.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CORE), 'exec'), module.__dict__)
    def merge(raw_k_seq, rope_k_seq, v_seq, importance, target_len, positions, cos, sin, batch_idx,
              k_diff_alpha, merge_score_mode, mrope_section):
        return rope_k_seq[:target_len].clone()+.25, v_seq[:target_len].clone()+.5, torch.ones(target_len)
    def set_cache(cache, layer_idx, key_states, value_states):
        cache[layer_idx] = (key_states, value_states)
    module._paired_merge_one_sequence = merge
    module._set_layer_cache = set_cache
    records = []
    for layer in range(4):
        values = torch.arange(32, dtype=torch.float32).reshape(1, 4, 4, 2)+100*layer+1
        records.append(dict(layer_idx=layer, candidate_raw=values.clone(), candidate_rope=values.clone()+.1,
                            candidate_value=values.clone()+.2, candidate_imp=torch.arange(4.).reshape(1,1,4).expand(1,4,4).clone(),
                            candidate_idx=torch.arange(4).reshape(1,1,4).expand(1,4,4).clone(),
                            candidate_history=4, final_history=2, history_len=4,
                            recent_key=values[:,:,:1].clone(), recent_value=values[:,:,:1].clone(),
                            head_sim=torch.arange(4.).reshape(1,4)+layer*4,
                            cos=torch.ones(3,1,5,2), sin=torch.zeros(3,1,5,2), mrope_section=[1]))
    config = types.SimpleNamespace(_snapkv_kvrepack_records=records, snapkv_simhead_merge_ratio=.3,
                                   snapkv_kvrepack_k_diff_alpha=2., snapkv_kvrepack_merge_score_mode='logratio',
                                   snapkv_kvrepack_verbose=False)
    return module, records, config


def execute(strategy='similarity', seed=0):
    module, records, config = fixture(); cache={}
    with SnapHeadSelectionAdapter(module, strategy=strategy, seed=seed) as adapter:
        adapter.set_sample('nextqa_oe_test', 0)
        module._finalize_prompt_cache(config, cache)
        audit=adapter.finish_sample()
    return audit, cache


class HeadSelectionTests(unittest.TestCase):
    def test_similarity_leaves_real_finalizer_outputs_identical(self):
        module, records, config = fixture(); expected={}
        module._finalize_prompt_cache(config, expected)
        audit, actual = execute()
        for layer in expected:
            for x, y in zip(expected[layer],actual[layer]):self.assertTrue(torch.equal(x,y))
        self.assertEqual(audit['mask_hamming'],0)
        self.assertEqual(audit['actual_merge_calls'],5)
        self.assertEqual(audit['actual_removed_tokens'],10)

    def test_global_random_matches_count_work_candidates_and_cache_lengths(self):
        baseline, _ = execute()
        masks=[]
        for seed in (0,1,2):
            audit, _=execute('random_global',seed)
            self.assertEqual(audit['selected_count'],baseline['selected_count'])
            self.assertEqual(audit['actual_merge_calls'],baseline['actual_merge_calls'])
            self.assertEqual(audit['actual_removed_tokens'],baseline['actual_removed_tokens'])
            self.assertEqual(audit['candidate_hashes'],baseline['candidate_hashes'])
            self.assertEqual(audit['prefill_cache_shapes'],baseline['prefill_cache_shapes'])
            self.assertGreater(audit['mask_hamming'],0)
            masks.append(audit['selected_masks'])
        self.assertNotEqual(masks[0],masks[1])
        self.assertEqual(execute('random_global',0)[0]['selected_masks'],masks[0])

    def test_local_rng_does_not_change_generation_rngs(self):
        python_state=random.getstate();torch_state=torch.random.get_rng_state().clone()
        execute('random_global',1)
        self.assertEqual(random.getstate(),python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(),torch_state))

    def test_layer_matched_retains_each_quota_and_degenerate_layers(self):
        audit, _=execute('random_layer_matched',2)
        self.assertEqual(audit['baseline_layer_counts'],audit['selected_layer_counts'])
        for before, after, count in zip(audit['baseline_masks'][0],audit['selected_masks'][0],audit['baseline_layer_counts'][0]):
            if count in (0,4):self.assertEqual(before,after)

    def test_model_level_mask_stays_fixed_for_different_inputs(self):
        module,records,config=fixture()
        with SnapHeadSelectionAdapter(module,strategy='random_global',seed=0) as adapter:
            masks=[]
            for doc in (0,17):
                config._snapkv_kvrepack_records=records
                adapter.set_sample('task',doc);module._finalize_prompt_cache(config,{})
                masks.append(adapter.finish_sample()['selected_masks'])
        self.assertEqual(masks[0],masks[1])

    def test_missing_merge_or_repeated_gate_fails_and_functions_restore(self):
        module,records,config=fixture();original={n:getattr(module,n) for n in ('_global_merge_masks','_paired_merge_one_sequence','_set_layer_cache')}
        with self.assertRaises(ValueError):
            with SnapHeadSelectionAdapter(module) as adapter:
                adapter.set_sample('task',0);module._global_merge_masks(records,.3);adapter.finish_sample()
        for n,f in original.items():self.assertIs(getattr(module,n),f)
        with SnapHeadSelectionAdapter(module) as adapter:
            adapter.set_sample('task',0);module._global_merge_masks(records,.3)
            with self.assertRaises(RuntimeError):module._global_merge_masks(records,.3)

    def test_nested_adapter_does_not_disturb_outer(self):
        module,records,config=fixture();original=module._global_merge_masks
        with SnapHeadSelectionAdapter(module) as adapter:
            wrapped=module._global_merge_masks
            with self.assertRaises(RuntimeError):
                with SnapHeadSelectionAdapter(module):pass
            self.assertIs(module._global_merge_masks,wrapped)
            adapter.set_sample('task',0);module._finalize_prompt_cache(config,{})
            adapter.finish_sample()
        self.assertIs(module._global_merge_masks,original)

    def test_random_cannot_silently_change_merge_eligibility_work(self):
        module,records,config=fixture()
        # Heterogeneous capacities are outside this strictly matched setup.
        records[-1]['final_history']=4
        with self.assertRaisesRegex(ValueError,'effective merge'):
            with SnapHeadSelectionAdapter(module,strategy='random_global',seed=0) as adapter:
                adapter.set_sample('task',0);module._global_merge_masks(records,.3)


if __name__=='__main__':unittest.main()

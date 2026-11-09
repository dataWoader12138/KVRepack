"""Execute the native LookM source on CPU; no models, CUDA or downloads."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[1]/'src/kvrepack/baseline_utils.py'


def source_module():
    tree = ast.parse(SOURCE.read_text())
    names = {'repeat_kv', 'merge_kv', 'LookMCluster'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    scope = dict(torch=torch, nn=torch.nn, F=F, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return SimpleNamespace(**scope)


def reference_merge(keys, values, selected, window, mode):
    """Scalar per-head oracle: assign every history complement token once."""
    expected_k = []; expected_v = []
    for batch in range(keys.shape[0]):
        batch_k = []; batch_v = []
        for head in range(keys.shape[1]):
            chosen = selected[batch, head].tolist()
            pivots = chosen + list(range(keys.shape[-2]-window, keys.shape[-2]))
            groups = [[] for _ in pivots]
            for token in range(keys.shape[-2]-window):
                if token in chosen:
                    continue
                scores = []
                for pivot in pivots:
                    a, b = keys[batch,head,token].double(), keys[batch,head,pivot].double()
                    denom = a.norm().item()*b.norm().item()
                    scores.append(a.dot(b).item()/denom if denom else 0.)
                groups[max(range(len(pivots)), key=lambda p:scores[p])].append(token)
            out_k = []; out_v = []
            for pivot, group in zip(pivots, groups):
                pk, pv = keys[batch,head,pivot], values[batch,head,pivot]
                terms_k, terms_v = [pk], [pv]
                for token in group:
                    tk, tv = keys[batch,head,token], values[batch,head,token]
                    terms_k.append((tk+pk)/2 if mode == 'pivot' else tk)
                    terms_v.append((tv+pv)/2 if mode == 'pivot' else tv)
                out_k.append(torch.stack(terms_k).mean(0)); out_v.append(torch.stack(terms_v).mean(0))
            batch_k.append(torch.stack(out_k)); batch_v.append(torch.stack(out_v))
        expected_k.append(torch.stack(batch_k)); expected_v.append(torch.stack(batch_v))
    return torch.stack(expected_k), torch.stack(expected_v)


class LookMBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_total_budget_floor_boundaries_gqa_and_first_layer_policy(self):
        module = source_module()
        cases = [(10,.1),(31,.1),(99,.1),(319,.1),(320,.1),(329,.1),(330,.1),
                 (641,.1),(5,1.),(32,1.),(64,1.)]
        for tokens, ratio in cases:
            for layer in (0, 2):
                for gqa_func in ('mean', 'max'):
                    with self.subTest(tokens=tokens, ratio=ratio, layer=layer, gqa=gqa_func):
                        gen = torch.Generator().manual_seed(19)
                        key = torch.randn((2,2,tokens,7), generator=gen)
                        value = torch.randn(key.shape, generator=gen)
                        query = torch.randn((2,4,tokens,7), generator=gen)
                        cluster = module.LookMCluster(layer_idx=layer, num_hidden_layers=28,
                            num_key_value_groups=2, budget_ratio=ratio, gqa_func=gqa_func,
                            is_text_token_mask=torch.arange(tokens).remainder(7).eq(0).expand(2,-1))
                        out_k, out_v = cluster.update_kv(key, query, value)
                        capacity = math.floor(tokens*ratio)
                        self.assertEqual(out_k.shape, (2,2,capacity,7))
                        self.assertEqual(out_v.shape, out_k.shape)
                        self.assertEqual(cluster.total_prefill_capacity, capacity)
                        self.assertEqual(cluster.effective_window_size, min(32,capacity))
                        self.assertEqual(cluster.select_num, max(0,capacity-32))
                        self.assertEqual(cluster.window_size, 32)
                        self.assertTrue(torch.isfinite(out_k).all()); self.assertTrue(torch.isfinite(out_v).all())
                        if ratio == 1:
                            self.assertIs(out_k,key); self.assertIs(out_v,value)
                        elif capacity <= 32 and layer == 0:
                            torch.testing.assert_close(out_k, key[:,:,-capacity:])
                            torch.testing.assert_close(out_v, value[:,:,-capacity:])
                        elif capacity <= 32:
                            empty = torch.empty((2,2,0),dtype=torch.long)
                            oracle = reference_merge(key,value,empty,capacity,'pivot')
                            torch.testing.assert_close(out_k,oracle[0],atol=2e-6,rtol=2e-6)
                            torch.testing.assert_close(out_v,oracle[1],atol=2e-6,rtol=2e-6)

    def test_recent_only_budget_still_merges_history_in_late_layers(self):
        module = source_module()
        for tokens in (16,320):
            with self.subTest(tokens=tokens):
                key = torch.zeros((1,1,tokens,3)); key[...,0] = 1
                value = torch.arange(float(tokens)).view(1,1,tokens,1).expand(-1,-1,-1,3)
                cluster = module.LookMCluster(layer_idx=27,num_hidden_layers=28,budget_ratio=.1)
                out_k,out_v = cluster.update_kv(key,key,value)
                capacity = tokens//10; pivot = tokens-capacity; dropped = pivot
                expected = (pivot + (sum(range(pivot))+dropped*pivot)/2)/(dropped+1)
                self.assertEqual(out_v[0,0,0,0].item(),expected)
                if tokens == 320:
                    self.assertEqual(expected,216.)
                else:
                    self.assertEqual(expected,11.25)
                torch.testing.assert_close(out_v,reference_merge(key,value,torch.empty((1,1,0),dtype=torch.long),capacity,'pivot')[1])
                if capacity>1:
                    torch.testing.assert_close(out_v[0,0,1:,0],torch.arange(float(pivot+1),float(tokens)))
                self.assertEqual(cluster.select_num,0)
                self.assertEqual(out_k.shape[-2],capacity)

    def test_ratio_one_returns_original_tensor_objects_in_every_layer(self):
        module = source_module()
        for layer in (0,1,2,27):
            for tokens in (5,32,64):
                with self.subTest(layer=layer,tokens=tokens):
                    key=torch.randn((2,2,tokens,3));value=key.clone()
                    cluster=module.LookMCluster(layer_idx=layer,num_hidden_layers=28,budget_ratio=1.)
                    out=cluster.update_kv(key,key,value)
                    self.assertIs(out[0],key);self.assertIs(out[1],value)

    def test_zero_capacity_explicitly_rejects_nonempty_cache(self):
        module = source_module()
        for tokens in (1, 5, 9):
            cluster = module.LookMCluster(layer_idx=0, num_hidden_layers=28, budget_ratio=.1)
            key = torch.ones((1,1,tokens,3))
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, 'budget is zero'):
                cluster.update_kv(key, key, key)

    def test_invalid_ratio_and_window_rejected(self):
        module = source_module(); key = torch.ones((1,1,40,3))
        for ratio in (0., -.1, 1.1, float('nan'), float('inf')):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, 'budget_ratio'):
                module.LookMCluster(budget_ratio=ratio).update_kv(key,key,key)
        with self.assertRaisesRegex(ValueError, 'positive recent window'):
            module.LookMCluster(budget_ratio=.1,window_size=0).update_kv(key,key,key)

    def test_text_priority_gqa_and_native_five_percent_merge_boundary(self):
        module = source_module(); calls = []
        native_merge = module.merge_kv
        def observed_merge(key, value, indices, window, mode):
            calls.append((indices[...,0].clone(), window, mode))
            return native_merge(key, value, indices, window, mode)
        # Extracted functions share this globals dictionary, as the real module does.
        module.LookMCluster.update_kv.__globals__['merge_kv'] = observed_merge
        key = torch.ones((1,2,400,3)); value = torch.arange(400.).view(1,1,400,1).expand(1,2,-1,3)
        query = torch.ones((1,4,400,3)); text = torch.zeros((1,400), dtype=torch.bool)
        expected = list(range(0,80,10)); text[0,expected] = True
        for layer in (0,1,2):
            cluster = module.LookMCluster(layer_idx=layer, num_hidden_layers=28, budget_ratio=.1,
                num_key_value_groups=2, is_text_token_mask=text)
            out_k, out_v = cluster.update_kv(key,query,value)
            if layer < 2:
                self.assertEqual(len(calls), 0)
                for head in range(2):
                    self.assertEqual(sorted(out_v[0,head,:8,0].tolist()), expected)
                    torch.testing.assert_close(out_v[0,head,-32:,0], torch.arange(368.,400.))
            else:
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1:], (32,'pivot'))
                for head in range(2):
                    self.assertEqual(sorted(calls[0][0][0,head].tolist()), expected)

    def test_fresh_prompt_recomputes_capacity_without_mutating_configured_window(self):
        module = source_module()
        cluster = module.LookMCluster(layer_idx=0, num_hidden_layers=28, budget_ratio=.1)
        for tokens in (330, 50, 400, 10):
            key = torch.randn((1,1,tokens,3)); cluster.is_text_token_mask = torch.zeros((1,tokens), dtype=torch.bool)
            out = cluster.update_kv(key,key,key)
            self.assertEqual(out[0].shape[-2], tokens//10)
            self.assertEqual(cluster.select_num, max(0,tokens//10-32))
            self.assertEqual(cluster.window_size, 32)


class LookMMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disjoint_head_and_batch_selection_matches_scalar_oracle(self):
        module = source_module(); gen = torch.Generator().manual_seed(42)
        for dim in (1,3,7,128):
            for mode in ('pivot','averaged'):
                with self.subTest(dim=dim,mode=mode):
                    key = torch.randn((2,3,8,dim), generator=gen)
                    value = torch.randn(key.shape, generator=gen)
                    selected = torch.tensor([[[0,1],[2,3],[4,5]],[[5,4],[3,2],[1,0]]])
                    indices = selected.unsqueeze(-1).expand(-1,-1,-1,dim)
                    actual = module.merge_kv(key,value,indices,2,mode)
                    expected = reference_merge(key,value,selected,2,mode)
                    torch.testing.assert_close(actual[0],expected[0],atol=2e-6,rtol=2e-6)
                    torch.testing.assert_close(actual[1],expected[1],atol=2e-6,rtol=2e-6)
                    self.assertEqual(actual[0].shape,(2,3,4,dim))

    def test_every_unselected_token_contributes_once_without_cross_head_leakage(self):
        module = source_module()
        # All keys have the same direction, so every unselected token maps to
        # its own head's first retained pivot. V exposes the exact contribution.
        key = torch.ones((2,2,6,3))
        value = torch.arange(24.).view(2,2,6,1).expand(-1,-1,-1,3).clone()
        selected = torch.tensor([[[0],[1]],[[2],[3]]])
        indices = selected.unsqueeze(-1).expand(-1,-1,-1,3)
        before = module.merge_kv(key,value,indices,1,'pivot')[1]
        for batch in range(2):
            for head in range(2):
                chosen = selected[batch,head,0].item()
                group = [i for i in range(5) if i != chosen]
                expected = (3*value[batch,head,chosen] + .5*value[batch,head,group].sum(0))/5
                torch.testing.assert_close(before[batch,head,0],expected)
                for token in group:
                    changed = value.clone(); changed[batch,head,token] += 10
                    after = module.merge_kv(key,changed,indices,1,'pivot')[1]
                    delta = torch.zeros_like(before); delta[batch,head,0] = 1.
                    torch.testing.assert_close(after-before,delta,atol=2e-6,rtol=2e-6)

    def test_pivot_weight_preserves_pair_then_mean_include_self(self):
        module = source_module()
        key = torch.ones((1,1,4,3))
        value = torch.tensor([2.,6.,10.,99.]).view(1,1,4,1).expand(-1,-1,-1,3)
        indices = torch.zeros((1,1,1,3), dtype=torch.long)
        out = module.merge_kv(key,value,indices,1,'pivot')[1]
        # ((retained=2) + ((6+2)/2) + ((10+2)/2)) / 3 = 4.
        torch.testing.assert_close(out[0,0,0],torch.full((3,),4.))
        torch.testing.assert_close(out[0,0,1],torch.full((3,),99.))

    def test_zero_keys_finite_for_arbitrary_dimensions_and_tensor_dtypes(self):
        module = source_module()
        for dtype in (torch.float32,torch.float64,torch.float16,torch.bfloat16):
            for dim in (1,3,7):
                with self.subTest(dtype=dtype,dim=dim):
                    key = torch.zeros((2,2,6,dim),dtype=dtype); value = torch.ones_like(key)
                    selected = torch.tensor([[[0],[1]],[[2],[3]]]).unsqueeze(-1).expand(-1,-1,-1,dim)
                    out_k,out_v = module.merge_kv(key,value,selected,1,'pivot')
                    self.assertTrue(torch.isfinite(out_k).all()); self.assertTrue(torch.isfinite(out_v).all())
                    self.assertEqual(out_k.shape,(2,2,2,dim)); self.assertEqual(out_k.dtype,dtype)
                    torch.testing.assert_close(out_k,torch.zeros_like(out_k))
                    torch.testing.assert_close(out_v,torch.ones_like(out_v))

    def test_large_finite_half_cosine_routes_to_fp32_oracle(self):
        module=source_module()
        # Pivot0 is orthogonal to the dropped key. Pivot1 is its positive
        # match, but its four-dimensional FP16 norm overflows to infinity.
        for dtype in (torch.float16,torch.bfloat16):
            with self.subTest(dtype=dtype):
                key=torch.tensor([[0.,20000.,0.,0.],[10000.,0.,0.,0.],
                    [40000.,40000.,40000.,40000.]],dtype=dtype).view(1,1,3,4)
                value=torch.tensor([1.,9.,3.],dtype=dtype).view(1,1,3,1).expand(-1,-1,-1,4)
                if dtype == torch.float16:
                    self.assertTrue(torch.isinf(key[0,0,2].norm()))
                actual=module.merge_kv(key,value,torch.zeros((1,1,1,4),dtype=torch.long),1,'pivot')
                # The only dropped token must go to pivot1, leaving pivot0's
                # value1 unchanged; pivot1 becomes (3 + (9+3)/2)/2 = 4.5.
                torch.testing.assert_close(actual[1][0,0,0],torch.full((4,),1.,dtype=dtype))
                torch.testing.assert_close(actual[1][0,0,1],torch.full((4,),4.5,dtype=dtype))
                self.assertEqual(actual[0].dtype,dtype);self.assertEqual(actual[1].dtype,dtype)
                self.assertTrue(torch.isfinite(actual[0]).all())

    def test_bfloat16_cosine_avoids_quantized_similarity_ties(self):
        module=source_module()
        key=torch.tensor([[60000.,3000.],[60000.,0.],[60000.,2400.]],dtype=torch.bfloat16).view(1,1,3,2)
        value=torch.tensor([1.,9.,3.],dtype=torch.bfloat16).view(1,1,3,1).expand(-1,-1,-1,2)
        actual=module.merge_kv(key,value,torch.zeros((1,1,1,2),dtype=torch.long),1,'pivot')
        torch.testing.assert_close(actual[1][0,0,0],torch.full((2,),1.,dtype=torch.bfloat16))
        torch.testing.assert_close(actual[1][0,0,1],torch.full((2,),4.5,dtype=torch.bfloat16))

    def test_batch_head_and_history_permutation_equivariance(self):
        module = source_module(); gen = torch.Generator().manual_seed(3)
        key = torch.randn((2,3,8,7), generator=gen); value = torch.randn(key.shape, generator=gen)
        selected = torch.tensor([[[0,1],[2,3],[4,5]],[[5,4],[3,2],[1,0]]])
        indices = selected.unsqueeze(-1).expand(-1,-1,-1,7)
        baseline = module.merge_kv(key,value,indices,2,'pivot')
        batches, heads = torch.tensor([1,0]),torch.tensor([2,0,1])
        reordered = module.merge_kv(key[batches][:,heads],value[batches][:,heads],indices[batches][:,heads],2,'pivot')
        for original,permuted in zip(baseline,reordered):
            torch.testing.assert_close(permuted,original[batches][:,heads])
        order = torch.tensor([3,5,0,4,1,2,6,7]); inverse = order.argsort()
        history_permuted = module.merge_kv(key[:,:,order],value[:,:,order],inverse[selected].unsqueeze(-1).expand(-1,-1,-1,7),2,'pivot')
        for original,permuted in zip(baseline,history_permuted):
            torch.testing.assert_close(permuted,original)

    def test_no_dropped_history_and_zero_recent_window(self):
        module = source_module(); key = torch.randn((2,2,5,3)); value = key*2
        selected = torch.tensor([3,1,0,2]).view(1,1,4,1).expand(2,2,-1,3)
        out = module.merge_kv(key,value,selected,1,'pivot')
        torch.testing.assert_close(out[0],key[:,:,[3,1,0,2,4]])
        torch.testing.assert_close(out[1],value[:,:,[3,1,0,2,4]])
        zero_recent_selected = torch.tensor([1,3]).view(1,1,2,1).expand(2,2,-1,3)
        out = module.merge_kv(key,value,zero_recent_selected,0,'pivot')
        expected = reference_merge(key,value,zero_recent_selected[...,0],0,'pivot')
        torch.testing.assert_close(out[0],expected[0]); torch.testing.assert_close(out[1],expected[1])


if __name__ == '__main__':
    unittest.main()

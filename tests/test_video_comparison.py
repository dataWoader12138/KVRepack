"""CPU checks of native capture transparency, budgets and atomic-row validation."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from kvrepack.benchmarks.run_video_comparison import (PrefillBudgetAudit, validate_sample,
    STRICT_LOOKM_POLICY,LEGACY_LOOKM_POLICY,NATIVE_BUDGET_POLICY,validate_budget_audit,validate_resume_contract)

PACKAGE = Path(__file__).resolve().parents[1]/'src/kvrepack'


def actual_lookm():
    path = PACKAGE/'baseline_utils.py'
    names = {'repeat_kv', 'merge_kv', 'LookMCluster'}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return SimpleNamespace(**scope)


def row_fixture(method='lookm', *, strict=False):
    policy=STRICT_LOOKM_POLICY if strict else LEGACY_LOOKM_POLICY if method=='lookm' else NATIVE_BUDGET_POLICY
    inputs = dict(input_hashes={'input_ids':'actual-input-bytes'},
        input_shapes={'input_ids':[1,100]}, input_dtypes={'input_ids':'torch.int64'})
    layers = [dict(layer=i, before_tokens_per_head=100, head_lengths=[42]*4,
        after_tokens_total=168, key_shape=[1,4,42,128], value_shape=[1,4,42,128]) for i in range(28)]
    cache = dict(status='completed', method=method, task='nextqa_oe_test', doc_id=0,
        input_tokens=100, native_kv_heads=4, layers=layers, layers_count=28,
        nominal_budget_ratio=.1, prefill_tokens_total=4704, uncompressed_tokens_total=11200,
        actual_retained_ratio=.42,budget_policy=policy,strict_budget_passed=False if method=='lookm' else None)
    final = [dict(layer=i, key_shape=[1,4,43,128], value_shape=[1,4,43,128]) for i in range(28)]
    row=dict(schema_version=1, status='completed', method=method, task='nextqa_oe_test', doc_id=0,
        contract_sha256='frozen', source_video_id='video', video_id='video',
        reference_metadata=dict(task='nextqa_oe_test', doc_id=0, source_video_id='video', video_id='video'),
        input_fingerprints=inputs, **deepcopy(inputs), generated_ids=[7,2], generated_tokens=2,
        text='answer', prediction='answer', eos_token_id=2, eos_reached=True,
        config=dict(method=method, task='nextqa_oe_test', max_new_tokens=128,
            stream_contract_sha256='frozen', budget_ratio=.1,budget_policy=policy),
        audit=dict(status='completed'), cache_audit=cache, final_cache_shapes=final)
    if strict:
        row['input_tokens']=100;row['reference_metadata']['input_tokens']=100
        cache.update(strict_budget_passed=True,total_prefill_capacity=10,prefill_tokens_total=1120,actual_retained_ratio=.1)
        for layer in cache['layers']:
            layer.update(head_lengths=[10]*4,after_tokens_total=40,key_shape=[1,4,10,128],value_shape=[1,4,10,128],
                total_prefill_capacity=10,configured_recent_window=32,recent_window=10,selected_history_tokens=0)
        row['final_cache_shapes']=[dict(layer=i,key_shape=[1,4,11,128],value_shape=[1,4,11,128]) for i in range(28)]
    return row


class VideoComparisonTests(unittest.TestCase):
    def test_actual_lookm_original_return_and_fresh_length_budget(self):
        module = actual_lookm()
        actual = module.LookMCluster.update_kv
        returned = []
        def spy(*args):
            result = actual(*args)
            returned.append(result)
            return result
        module.LookMCluster.update_kv = spy
        torch.set_num_threads(1)
        with PrefillBudgetAudit('lookm', module=module) as audit:
            for sample, tokens in enumerate((64, 96,400,480)):
                audit.set_sample('nextqa_oe_test', sample, tokens)
                common = torch.randn((1,1,tokens,128), generator=torch.Generator().manual_seed(42))
                key = common.expand(1,4,tokens,128).clone()
                value = key.clone(); query = key.clone()
                final = []
                for layer in range(28):
                    cluster = module.LookMCluster(layer_idx=layer, num_hidden_layers=28,
                        budget_ratio=.1, num_key_value_groups=1,
                        is_text_token_mask=torch.zeros((1,tokens), dtype=torch.bool))
                    result = cluster.update_kv(key, query, value)
                    self.assertIs(result, returned[-1])
                    self.assertEqual(cluster.budget_ratio, .1)
                    self.assertEqual(result[0].shape[-2], int(tokens*.1))
                    final.append(dict(layer=layer, key_shape=list(result[0].shape), value_shape=list(result[1].shape)))
                row = audit.finish_sample(final, 1)
                self.assertEqual(row['prefill_tokens_total'], 112*int(tokens*.1))
                self.assertEqual(row['actual_retained_ratio'], int(tokens*.1)/tokens)
                self.assertEqual(row['budget_policy'],STRICT_LOOKM_POLICY)
                self.assertTrue(row['strict_budget_passed'])
                self.assertTrue(all(layer['recent_window']==min(32,int(tokens*.1)) for layer in row['layers']))
        self.assertEqual(len(returned), 112)
        self.assertIs(module.LookMCluster.update_kv, spy)

    def test_actual_hybrid_prefill_update_original_return_and_heterogeneous_lengths(self):
        path = PACKAGE/'hybridkv_utils.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HybridKVCache')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update')
        node = ast.ClassDef(name='HybridKVCache', bases=[], keywords=[], body=[method], decorator_list=[])
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), scope)
        module = SimpleNamespace(**scope)
        native = module.HybridKVCache.update
        obj = module.HybridKVCache()
        obj.key_cache = []; obj.value_cache = []; obj.local_len = 2
        obj.sparse_budget_per_head = [[[3,4,5,6]] for _ in range(28)]
        obj.layers = [SimpleNamespace(self_attn=SimpleNamespace(kv_cluster=SimpleNamespace(
            head_lens=torch.tensor([5,6,7,8])))) for _ in range(28)]
        with PrefillBudgetAudit('hybridkv_kvrepack', module=module) as audit:
            audit.set_sample('nextqa_oe_test', 0, 100)
            final = []
            for layer in range(28):
                key = torch.randn((26,128)); value = key.clone()
                result = obj.update(key, value, layer)
                self.assertIs(result[0], key); self.assertIs(result[1], value)
                final.append(dict(layer=layer, key_shape=[30,128], value_shape=[30,128]))
            record = audit.finish_sample(final, 2)
            self.assertEqual(record['prefill_tokens_total'], 28*26)
            self.assertEqual(record['layers'][0]['head_lengths'], [5,6,7,8])
        self.assertIs(module.HybridKVCache.update, native)

    def test_explicit_legacy_row_is_readable_but_never_strict(self):
        row = row_fixture()
        self.assertIs(validate_sample(row, doc_id=0, contract_sha256='frozen'), row)
        result=validate_budget_audit(row['cache_audit'],method='lookm')
        self.assertTrue(result['legacy_budget_result']);self.assertFalse(result['strict_budget_passed'])
        row['config'].pop('budget_policy');row['cache_audit'].pop('budget_policy');row['cache_audit'].pop('strict_budget_passed')
        self.assertIs(validate_sample(row,doc_id=0,contract_sha256='frozen'),row)
        self.assertEqual(validate_budget_audit(row['cache_audit'],method='lookm')['budget_policy'],LEGACY_LOOKM_POLICY)

    def test_strict_row_and_posthoc_overbudget_or_decode_tampering(self):
        row=row_fixture(strict=True)
        self.assertIs(validate_sample(row,doc_id=0,contract_sha256='frozen',budget_policy=STRICT_LOOKM_POLICY),row)
        for mode in ('extra_window','capacity','window','denominator','decode','policy'):
            with self.subTest(mode=mode):
                row=row_fixture(strict=True)
                if mode=='extra_window':
                    for layer in row['cache_audit']['layers']:
                        layer.update(head_lengths=[42]*4,after_tokens_total=168,key_shape=[1,4,42,128],value_shape=[1,4,42,128])
                    row['cache_audit'].update(prefill_tokens_total=4704,actual_retained_ratio=.42)
                    row['final_cache_shapes']=[dict(layer=i,key_shape=[1,4,43,128],value_shape=[1,4,43,128]) for i in range(28)]
                elif mode=='capacity':row['cache_audit']['total_prefill_capacity']=11
                elif mode=='window':row['cache_audit']['layers'][0]['recent_window']=32
                elif mode=='denominator':
                    row['input_fingerprints']['input_shapes']['input_ids']=[1,101]
                    row['input_shapes']['input_ids']=[1,101]
                elif mode=='decode':row['final_cache_shapes'][0].update(key_shape=[1,4,12,128],value_shape=[1,4,12,128])
                else:row['config']['budget_policy']=LEGACY_LOOKM_POLICY
                with self.assertRaises(ValueError):validate_sample(row,doc_id=0,contract_sha256='frozen',budget_policy=STRICT_LOOKM_POLICY)

    def test_legacy_resume_cannot_mix_with_strict_sources_or_policy(self):
        strict=dict(method='lookm',budget_policy=STRICT_LOOKM_POLICY,sources_sha256={'core':'new'})
        validate_resume_contract(deepcopy(strict),strict)
        for saved in (dict(method='lookm',sources_sha256={'core':'new'}),dict(method='lookm',budget_policy=LEGACY_LOOKM_POLICY,sources_sha256={'core':'old'}),dict(method='lookm',budget_policy=STRICT_LOOKM_POLICY,sources_sha256={'core':'old'})):
            with self.assertRaises(ValueError):validate_resume_contract(saved,strict)

    def test_identity_generation_and_cache_tampering_rejected(self):
        changes = (
            lambda r:r.update(contract_sha256='different'),
            lambda r:r['reference_metadata'].update(doc_id=1),
            lambda r:r['input_hashes'].update(input_ids='changed'),
            lambda r:r.update(eos_reached=False),
            lambda r:r.update(generated_tokens=3),
            lambda r:r['cache_audit']['layers'].pop(),
            lambda r:r['cache_audit']['layers'][0]['head_lengths'].__setitem__(0, 43),
            lambda r:r['final_cache_shapes'][0].update(key_shape=[1,4,44,128]),
            lambda r:r['cache_audit'].update(actual_retained_ratio=.1),
            lambda r:r['config'].update(method='hybridkv_kvrepack'),
        )
        for change in changes:
            with self.subTest(change=change):
                row = row_fixture(); change(row)
                with self.assertRaises(ValueError):
                    validate_sample(row, doc_id=0, contract_sha256='frozen')

    def test_prefill_decode_disagreement_restores_original(self):
        module = actual_lookm(); original = module.LookMCluster.update_kv
        with self.assertRaises(ValueError):
            with PrefillBudgetAudit('lookm', module=module) as audit:
                audit.set_sample('nextqa_oe_test', 0, 100)
                for layer in range(28):
                    audit._record(layer, torch.zeros((1,4,42,128)), torch.zeros((1,4,42,128)), [42]*4)
                audit.finish_sample([dict(layer=i,key_shape=[1,4,44,128],value_shape=[1,4,44,128]) for i in range(28)], 2)
        self.assertIs(module.LookMCluster.update_kv, original)


if __name__ == '__main__':
    unittest.main()

"""Actual Snap core functions, fixed groups and LMA checked on CPU tensors."""
import ast
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional, Tuple
import unittest

import torch

from kvrepack.ablations.merge_snap import (SnapMergeAdapter, cardinality_sdpa,
                                           compile_merger, strict_history_budget)
from kvrepack.merge_score import merge_score_from_importance

SOURCE = Path(__file__).resolve().parents[1] / "src/kvrepack/qwen_model_snapkv_kvrepack.py"


def core_fixture():
    names = {"_select_mrope_batch", "_rotate_half", "_apply_mrope_to_merged_keys",
             "_paired_merge_one_sequence", "_global_merge_masks", "_set_layer_cache",
             "_history_budget", "_finalize_prompt_cache", "_topk_from_candidates_sorted"}
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ModuleType("actual_snap_core")
    module.__dict__.update(torch=torch, math=math, Tuple=Tuple, Optional=Optional, List=List, Dict=Dict,
              Cache=object, merge_score_from_importance=merge_score_from_importance,
              __name__="actual_snap_core", __file__=str(SOURCE))
    ns = module.__dict__
    ast.fix_missing_locations(tree)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    # Decode code is needed for AST adaptation, not full model execution.
    decode_source = SOURCE.with_name("qwen_model.py")
    decode_tree = ast.parse(decode_source.read_text())
    decode = next(n for n in decode_tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "qwen_flash_attn2_forward_SnapKV")
    exec(compile(ast.Module(body=[decode], type_ignores=[]), str(decode_source), "exec"), ns)
    return module


def sequence():
    raw = torch.randn(6, 6, generator=torch.Generator().manual_seed(3))
    value = torch.randn(6, 6, generator=torch.Generator().manual_seed(7))
    pos = torch.tensor([0, 2, 3, 5, 7, 9])
    angle = torch.arange(10).float() * .37
    cos = angle.cos().view(1, 1, 10, 1).expand(3, 1, 10, 6).clone()
    sin = angle.sin().view(1, 1, 10, 1).expand(3, 1, 10, 6).clone()
    rope = raw*cos[0, 0, pos] + torch.cat((-raw[:, 3:], raw[:, :3]), -1)*sin[0, 0, pos]
    return dict(raw_k_seq=raw, rope_k_seq=rope, v_seq=value,
                importance=torch.tensor([.1, .2, 1., 2., 3., 4.]), target_len=3,
                positions=pos, cos=cos, sin=sin, batch_idx=0, k_diff_alpha=2.,
                merge_score_mode="full", mrope_section=[1, 1, 1])


class SnapMergeTests(unittest.TestCase):
    def test_actual_group_builder_and_anchor_rope_unchanged(self):
        module, seq = core_fixture(), sequence()
        original = module._paired_merge_one_sequence(**seq)
        audits = []
        for variant in ("kvrepack", "lookm", "asymkv"):
            calls = []
            fun, spec = compile_merger(module._paired_merge_one_sequence, variant,
                                      audit_callback=lambda *args: calls.append(args))
            kw = dict(seq)
            if variant != "kvrepack":
                kw["fisher_gradient"] = torch.ones_like(seq["raw_k_seq"])
            result = fun(**kw)
            self.assertEqual(len(calls), 1)
            raw, rope, val, importance, positions, groups, sizes, starts = calls[0]
            audits.append((groups, sizes, starts))
            if variant == "kvrepack":
                for expected, observed in zip(original, result):
                    self.assertTrue(torch.equal(expected, observed))
            elif variant == "lookm":
                k = torch.stack([.5*raw[starts][g] + .5*raw[groups == g].mean(0) for g in range(3)])
                v = torch.stack([.5*val[starts][g] + .5*val[groups == g].mean(0) for g in range(3)])
                cp, sp = seq["cos"][0, 0, positions[starts]], seq["sin"][0, 0, positions[starts]]
                k = k*cp + torch.cat((-k[:, 3:], k[:, :3]), -1)*sp
                torch.testing.assert_close(result[0], k)
                torch.testing.assert_close(result[1], v)
            else:
                torch.testing.assert_close(result[0], original[0])
                want_v = torch.stack([val[groups == g].sum(0) for g in range(3)])
                torch.testing.assert_close(result[1], want_v)
                torch.testing.assert_close(result[2], sizes.clamp(max=5))
        for changed in audits[1:]:
            for a, b in zip(audits[0], changed):
                self.assertTrue(torch.equal(a, b))

    def test_budget_includes_recent_and_short_prompts_fail(self):
        self.assertEqual(strict_history_budget(None, 1000, 968, .1), 68)
        self.assertEqual(strict_history_budget(None, 1000, 968, 1), 968)
        with self.assertRaisesRegex(ValueError, "exceed"):
            strict_history_budget(None, 100, 68, .1)

    def test_lma_sdpa_matches_released_attention_equation_gqa(self):
        gen = torch.Generator().manual_seed(72)
        q = torch.randn(1, 1, 4, 6, generator=gen)
        k = torch.randn(1, 9, 4, 6, generator=gen)
        v = torch.randn(1, 9, 4, 6, generator=gen)
        c = torch.tensor([[[1, 2, 3, 4, 5, 1, 2, 1, 1], [3, 1, 5, 1, 2, 1, 2, 3, 1]]]).float()
        actual = cardinality_sdpa(q, k, v, c)
        z = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(6)
        ex = (z-z.max(-1, keepdim=True).values).exp()
        counts = c.repeat_interleave(2, 1).unsqueeze(-2)
        expected = ((ex/(ex*counts).sum(-1, keepdim=True)) @ v.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)

    def test_bfloat16_lma_is_bitwise_released_batched_readout(self):
        from kvrepack.asymkv_official.attention import official_lma_attention
        for seed in range(8):
            gen = torch.Generator().manual_seed(seed)
            q = torch.randn(1, 1, 4, 64, generator=gen).bfloat16()
            k = torch.randn(1, 37, 4, 64, generator=gen).bfloat16()
            v = torch.randn(1, 37, 4, 64, generator=gen).bfloat16()
            c = torch.randint(1, 6, (1, 2, 37), generator=gen).float()
            observed = cardinality_sdpa(q, k, v, c)
            expected, _ = official_lma_attention(q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), c.repeat_interleave(2, 1), current_query_length=1)
            self.assertTrue(torch.equal(observed, expected.transpose(1, 2)))

    def test_finalizer_gathers_fisher_and_keeps_recent_counts_one(self):
        module, seq = core_fixture(), sequence()
        # Fixture groups: two heads, exactly one selected for merging.
        expand = lambda x: x.view(1, 1, *x.shape).expand(1, 2, *x.shape).clone()
        rec = dict(layer_idx=0, candidate_raw=expand(seq["raw_k_seq"]),
                   candidate_rope=expand(seq["rope_k_seq"]), candidate_value=expand(seq["v_seq"]),
                   candidate_imp=seq["importance"].view(1, 1, 6).expand(1, 2, 6).clone(),
                   candidate_idx=seq["positions"].view(1, 1, 6).expand(1, 2, 6).clone(),
                   recent_key=torch.randn(1, 2, 2, 6), recent_value=torch.randn(1, 2, 2, 6),
                   final_history=3, candidate_history=6, head_sim=torch.tensor([[.9, .1]]),
                   cos=seq["cos"], sin=seq["sin"], mrope_section=[1, 1, 1])
        cfg = SimpleNamespace(_snapkv_kvrepack_records=[rec], snapkv_simhead_merge_ratio=.5,
                              snapkv_kvrepack_verbose=False, snapkv_kvrepack_k_diff_alpha=2.,
                              snapkv_kvrepack_merge_score_mode="full")
        cache = SimpleNamespace(key_cache=[], value_cache=[], _seen_tokens=0)
        original = module._paired_merge_one_sequence
        with SnapMergeAdapter("asymkv", module=module) as adapter:
            adapter.begin_sample()
            adapter.set_fisher([torch.ones(1, 2, 10, 6)])
            module._finalize_prompt_cache(cfg, cache)
            self.assertEqual(list(cache.key_cache[0].shape), [1, 2, 5, 6])
            counts = cache._merge_cardinality[0]
            torch.testing.assert_close(counts[0, 1], torch.ones(5))
            torch.testing.assert_close(counts[0, 0, 3:], torch.ones(2))
            self.assertEqual(float(counts[0, 0, :3].sum()), 6.)
            audit = adapter.finish_sample()
            self.assertEqual(audit["actual_merge_calls"], 1)
            self.assertEqual(audit["status"], "complete")
        self.assertIs(module._paired_merge_one_sequence, original)

    def test_exception_restores_all_source_functions(self):
        module = core_fixture()
        names = ("_paired_merge_one_sequence", "_history_budget", "_global_merge_masks",
                 "_finalize_prompt_cache", "qwen_flash_attn2_forward_SnapKV")
        original = {name:getattr(module, name) for name in names}
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with SnapMergeAdapter("asymkv", module=module):
                raise RuntimeError("fixture")
        for name in names:
            self.assertIs(getattr(module, name), original[name])


if __name__ == "__main__":
    unittest.main()

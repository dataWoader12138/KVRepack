"""CPU numerical oracles for fixed-group aggregation and prompt-only Fisher."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from kvrepack.ablations.merge_operators import (
    _collect_raw_key_fisher_full as collect_raw_key_fisher,
    collect_raw_key_fisher as cached_prefix_fisher,
    reduce_fixed_groups,
    _native_qwen25_flash_forward,
    _rebuild_raw_prefix_cache,
)


class FixedGroupOperatorTests(unittest.TestCase):
    def test_lookm_two_token_exact_three_quarter_pivot(self):
        key = torch.tensor([[4., 8.], [12., 0.]])
        value = torch.tensor([[8., 0.], [0., 16.]])
        result = reduce_fixed_groups(
            "lookm", key, value, torch.tensor([0, 0]),
            torch.tensor([2.]), torch.tensor([True, False]),
        )
        torch.testing.assert_close(result[0], torch.tensor([[6., 6.]]), rtol=0, atol=0)
        torch.testing.assert_close(result[1], torch.tensor([[6., 4.]]), rtol=0, atol=0)
        torch.testing.assert_close(result[2], torch.tensor([2.]), rtol=0, atol=0)

    def test_lookm_multi_member_and_singleton_analytic_oracle(self):
        key = torch.tensor([[4., 8.], [12., 0.], [8., 16.], [9., 3.]], dtype=torch.float64)
        value = key * 3
        actual = reduce_fixed_groups(
            "lookm", key, value, torch.tensor([0, 0, 0, 1]),
            torch.tensor([3., 1.]), torch.tensor([True, False, False, True]),
        )
        expected = torch.stack((.5 * key[0] + .5 * key[:3].mean(0), key[3]))
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=1e-14)
        torch.testing.assert_close(actual[1], expected * 3, rtol=0, atol=1e-14)

    def test_lookm_matches_released_pair_then_scatter_order_in_bfloat16(self):
        # Reference uses upstream's four-dimensional gather/scatter layout;
        # the implementation adapts only dimensions for one fixed host head.
        with torch.random.fork_rng():
            torch.manual_seed(92)
            key = torch.randn(17, 8).to(torch.bfloat16)
            value = torch.randn(17, 8).to(torch.bfloat16)
        starts = torch.tensor([True, False, False, True, False, True, False, False,
                               False, False, True, False, True, False, False, False, True])
        groups = starts.long().cumsum(0) - 1
        sizes = torch.bincount(groups).float()
        actual = reduce_fixed_groups("lookm", key, value, groups, sizes, starts)
        expected = []
        for tensor in (key, value):
            pivots = tensor[starts][None, None]
            dropped = tensor[~starts][None, None]
            index = groups[~starts][None, None, :, None].expand(1, 1, -1, tensor.shape[-1])
            paired = (dropped + pivots.gather(2, index)) / 2
            expected.append(torch.scatter_reduce(
                pivots, 2, index, paired, reduce="mean", include_self=True,
            )[0, 0])
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        self.assertEqual(actual[0].dtype, torch.bfloat16)

    def test_kvrepack_is_group_mean_without_mutating_inputs(self):
        key = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
        value = -key
        before = key.clone()
        actual = reduce_fixed_groups(
            "kvrepack", key, value, torch.tensor([0, 0, 1, 2, 2, 2]),
            torch.tensor([2., 1., 3.]), torch.tensor([True, False, True, True, False, False]),
        )
        expected = torch.stack((key[:2].float().mean(0), key[2].float(), key[3:].float().mean(0))).to(key.dtype)
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], -expected, rtol=0, atol=0)
        torch.testing.assert_close(key, before, rtol=0, atol=0)

    def test_asymkv_coordinate_fisher_sum_values_clip_and_singletons(self):
        key = torch.tensor([[1., 8.], [9., 2.], [11., 13.], *([[3., 7.]] * 6)])
        value = key * 2
        gradient = torch.tensor([[1., 3.], [3., 1.], [0., 0.], *([[1., 1.]] * 6)])
        actual = reduce_fixed_groups(
            "asymkv", key, value, torch.tensor([0, 0, 1, 2, 2, 2, 2, 2, 2]),
            torch.tensor([2., 1., 6.]),
            torch.tensor([True, False, True, True, False, False, False, False, False]),
            fisher_gradient=gradient,
        )
        expected_key = torch.tensor([[(1. + 81.) / 10., (72. + 2.) / 10.], [11., 13.], [3., 7.]])
        expected_value = torch.stack((value[:2].sum(0), value[2], value[3:].sum(0)))
        torch.testing.assert_close(actual[0], expected_key)
        torch.testing.assert_close(actual[1], expected_value)
        torch.testing.assert_close(actual[2], torch.tensor([2., 1., 5.]), rtol=0, atol=0)

    def test_asymkv_does_not_silently_fall_back_without_fisher(self):
        with self.assertRaisesRegex(ValueError, "requires a real"):
            reduce_fixed_groups("asymkv", torch.ones(2, 4), torch.ones(2, 4),
                                torch.tensor([0, 0]), torch.tensor([2.]),
                                torch.tensor([True, False]))


def _toy_native_attention_forward(self, hidden_states):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    scores = q @ k.transpose(-1, -2) / self.head_dim ** .5
    causal = torch.ones_like(scores, dtype=torch.bool).triu(1)
    return self.o_proj(scores.masked_fill(causal, -torch.inf).softmax(-1) @ v)


def _toy_failing_attention_forward(self, hidden_states, **kwargs):
    self.k_proj(hidden_states)
    raise RuntimeError("deliberate native attention failure")


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.q_proj = nn.Linear(4, 2, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(2, 4, bias=False)

    def forward(self, hidden_states):
        # Deliberately different host implementation; the probe must use the
        # provided native method temporarily instead of this patched route.
        return hidden_states / 3


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.ff = nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        hidden = hidden + self.self_attn(hidden)
        return hidden + self.ff(hidden).tanh()


class _ToyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([_ToyLayer(), _ToyLayer()])


class _ToyBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = _ToyLanguage()
        self.rope_deltas = torch.tensor([17])

    def forward(self, input_ids, use_cache, output_attentions, output_hidden_states, return_dict,
                attention_mask=None):
        assert use_cache is False and self.config.use_cache is False
        self.rope_deltas = torch.tensor([999])
        hidden = self.language_model.embed_tokens(input_ids)
        for layer in self.language_model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True, image_token_id=14, video_token_id=15)
        self.model = _ToyBackbone(self.config)
        self.lm_head = nn.Linear(4, 16, bias=False)


class FullGraphPromptFisherOracleTests(unittest.TestCase):
    def make_model(self):
        with torch.random.fork_rng():
            torch.manual_seed(712)
            return _ToyModel()

    def test_native_probe_uses_observed_text_and_restores_all_state(self):
        model = self.make_model()
        model.train()
        # Test restoration of mixed submodule modes and parameter flags too.
        model.model.language_model.layers[0].ff.eval()
        next(model.parameters()).requires_grad_(False)
        training_before = [module.training for module in model.modules()]
        required_before = [parameter.requires_grad for parameter in model.parameters()]
        rope_before = model.model.rope_deltas
        existing_grad = next(model.lm_head.parameters())
        existing_grad.grad = torch.ones_like(existing_grad)
        gradients, stats = collect_raw_key_fisher(
            model, {"input_ids": torch.tensor([[1, 15, 15, 2, 3, 5, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]])},
            loss_tail_tokens=3, native_attention_forward=_toy_native_attention_forward,
        )
        self.assertEqual(stats["loss_target_tokens"], 3)
        self.assertEqual(stats["gradient_space"], "pre_mrope_k_proj")
        self.assertEqual(len(gradients), 2)
        for gradient in gradients:
            self.assertEqual(tuple(gradient.shape), (1, 1, 7, 2))
            self.assertFalse(gradient.requires_grad)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
            # Padded final token follows all targets and has no causal effect.
            torch.testing.assert_close(gradient[:, :, -1], torch.zeros_like(gradient[:, :, -1]), rtol=0, atol=0)
        self.assertEqual(training_before, [module.training for module in model.modules()])
        self.assertEqual(required_before, [parameter.requires_grad for parameter in model.parameters()])
        self.assertTrue(model.config.use_cache)
        self.assertIs(model.model.rope_deltas, rope_before)
        self.assertTrue(torch.equal(existing_grad.grad, torch.ones_like(existing_grad)))
        for layer in model.model.language_model.layers:
            self.assertNotIn("forward", vars(layer.self_attn))
            self.assertEqual(len(layer.self_attn.k_proj._forward_hooks), 0)

    def test_prompt_tail_loss_matches_independent_uncompressed_oracle(self):
        model = self.make_model()
        ids = torch.tensor([[1, 15, 15, 2, 3, 5]])
        actual, stats = collect_raw_key_fisher(
            model, {"input_ids": ids}, 2,
            native_attention_forward=_toy_native_attention_forward,
        )
        # Run an independent explicit full forward.  Targets are known: the
        # last observed text IDs 3 and 5, from hidden positions 3 and 4.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        hidden = model.model.language_model.embed_tokens(ids)
        keys = []
        for layer in model.model.language_model.layers:
            attention = layer.self_attn
            q, k, v = attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden)
            k.requires_grad_(True)
            keys.append(k)
            scores = q @ k.transpose(-1, -2) / 2 ** .5
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            hidden = hidden + attention.o_proj(scores.masked_fill(mask, -torch.inf).softmax(-1) @ v)
            hidden = hidden + layer.ff(hidden).tanh()
        logits = model.lm_head(hidden[:, [3, 4]])
        loss = F.cross_entropy(logits.reshape(-1, 16), ids[:, [4, 5]].reshape(-1))
        expected = torch.autograd.grad(loss, keys)
        self.assertAlmostEqual(stats["loss"], float(loss), places=7)
        for observed, oracle in zip(actual, expected):
            torch.testing.assert_close(observed[:, 0], oracle, rtol=1e-6, atol=1e-8)

    def test_failure_restores_forward_hooks_config_and_parameter_flags(self):
        model = self.make_model()
        model.eval()
        original = lambda hidden_states: hidden_states + 1
        model.model.language_model.layers[0].self_attn.forward = original
        rope_before = model.model.rope_deltas
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            collect_raw_key_fisher(model, {"input_ids": torch.tensor([[1, 2, 3]])},
                                   native_attention_forward=_toy_failing_attention_forward)
        self.assertIs(model.model.language_model.layers[0].self_attn.forward, original)
        self.assertIs(model.model.rope_deltas, rope_before)
        self.assertTrue(model.config.use_cache)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(not layer.self_attn.k_proj._forward_hooks for layer in model.model.language_model.layers))

    def test_answer_labels_rejected_and_absent_text_target_rejected(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "without labels"):
            collect_raw_key_fisher(model, {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[3, 4]])})
        with self.assertRaisesRegex(ValueError, "no observed text"):
            collect_raw_key_fisher(model, {"input_ids": torch.tensor([[15, 15]])},
                                   native_attention_forward=_toy_native_attention_forward)

    def test_native_hf_forward_resolver_is_pinned_source_not_class_dispatch(self):
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as native
        previous = native.Qwen2_5_VLFlashAttention2.forward
        _native_qwen25_flash_forward.cache_clear()
        try:
            native.Qwen2_5_VLFlashAttention2.forward = _toy_failing_attention_forward
            recovered, sha = _native_qwen25_flash_forward()
            self.assertEqual(recovered.__name__, "forward")
            self.assertIn("position_embeddings", recovered.__code__.co_varnames)
            self.assertEqual(len(sha), 64)
            self.assertIsNot(recovered, _toy_failing_attention_forward)
        finally:
            native.Qwen2_5_VLFlashAttention2.forward = previous
            _native_qwen25_flash_forward.cache_clear()


class CachedPrefixFisherTests(unittest.TestCase):
    def make_model(self):
        from transformers import Qwen2_5_VLConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        with torch.random.fork_rng():
            torch.manual_seed(182)
            config = Qwen2_5_VLConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
                image_token_id=30, video_token_id=31, vision_start_token_id=29,
                vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                               "out_hidden_size": 16, "num_heads": 4},
            )
            config._attn_implementation = "eager"
            return Qwen2_5_VLForConditionalGeneration(config).double().eval()

    @staticmethod
    def eager():
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
        return Qwen2_5_VLAttention.forward

    def test_rebuilt_prefix_cache_preserves_native_mrope_values_exactly(self):
        model = self.make_model()
        layers = list(model.model.language_model.layers)
        projected = {}
        positions_seen = []
        handles = []
        for index, layer in enumerate(layers):
            def capture(_module, _args, output, index=index):
                projected[index] = output.detach()
            handles.append(layer.self_attn.k_proj.register_forward_hook(capture))
        def capture_positions(_module, _args, kwargs):
            positions_seen.append(kwargs["position_embeddings"])
        handles.append(layers[0].self_attn.register_forward_pre_hook(capture_positions, with_kwargs=True))
        ids = torch.tensor([[1, 5, 7, 8, 3, 4]])
        positions = torch.arange(6)[None, None].expand(3, 1, -1).clone()
        # Explicitly nonidentical temporal/height/width coordinates exercise
        # the same three-way mRoPE composition used for visual prefixes.
        positions[1] *= 2
        positions[2] *= 3
        try:
            with torch.no_grad():
                output = model.model(input_ids=ids, position_ids=positions,
                                     use_cache=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        cache = output.past_key_values
        before = [key.clone() for key in cache.key_cache]
        with torch.enable_grad():
            leaves = _rebuild_raw_prefix_cache(cache, projected, positions_seen[0], layers)
            for original, replacement in zip(before, cache.key_cache):
                torch.testing.assert_close(replacement, original, rtol=0, atol=0)
                self.assertTrue(replacement.requires_grad)
            loss = sum(key.square().sum() for key in cache.key_cache)
            loss.backward()
        for leaf in leaves:
            self.assertTrue(leaf.is_leaf)
            self.assertIsNotNone(leaf.grad)
        self.assertTrue(all(not value.requires_grad for value in cache.value_cache))

    def test_cached_probe_loss_and_raw_gradients_match_independent_post_cache_oracle(self):
        model = self.make_model()
        ids = torch.tensor([[1, 2, 3, 4, 6, 7, 8, 5, 9, 2, 3, 1]])
        original_forwards = [layer.self_attn.forward.__func__ for layer in model.model.language_model.layers]
        rope_before = model.model.rope_deltas
        masks_seen = []
        def custom_outer_mask_guard(_module, _args, kwargs):
            mask = kwargs.get("attention_mask")
            length = kwargs["input_ids"].shape[-1]
            masks_seen.append((length, None if mask is None else int(mask.shape[-1])))
            # Same shape condition used when the custom Hybrid outer model
            # intersects the padding mask with its current modality mask.
            if mask is not None and mask.shape != kwargs["input_ids"].shape:
                raise RuntimeError("Custom outer mask/input suffix lengths differ")
        guard = model.model.register_forward_pre_hook(custom_outer_mask_guard, with_kwargs=True)
        try:
            actual, stats = cached_prefix_fisher(
                model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                loss_tail_tokens=3, native_attention_forward=self.eager(),
            )
        finally:
            guard.remove()
        self.assertEqual(masks_seen, [(8, 8), (4, None)])
        self.assertEqual(stats["prefix_tokens"], 8)
        self.assertEqual(stats["backward_suffix_tokens"], 4)
        self.assertEqual(stats["loss_target_tokens"], 3)
        self.assertIn("no_grad_prefix_cache", stats["probe_policy"])
        self.assertIs(model.model.rope_deltas, rope_before)
        self.assertEqual(original_forwards, [layer.self_attn.forward.__func__ for layer in model.model.language_model.layers])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        layers = list(model.model.language_model.layers)
        raw_prefix, raw_tail, embeddings = {}, {}, []
        stage = ["prefix"]
        handles = []
        for index, layer in enumerate(layers):
            def capture(_module, _args, output, index=index):
                if stage[0] == "prefix":
                    raw_prefix[index] = output.detach()
                else:
                    output.requires_grad_(True)
                    output.retain_grad()
                    raw_tail[index] = output
            handles.append(layer.self_attn.k_proj.register_forward_hook(capture))
        def capture_embeddings(_module, _args, kwargs):
            if stage[0] == "prefix":
                embeddings.append(kwargs["position_embeddings"])
        handles.append(layers[0].self_attn.register_forward_pre_hook(capture_embeddings, with_kwargs=True))
        positions, _ = model.model.get_rope_index(ids)
        try:
            with torch.no_grad():
                prefix = model.model(input_ids=ids[:, :8], position_ids=positions[..., :8],
                                     use_cache=True, return_dict=True)
            cache = prefix.past_key_values
            # Independent oracle uses the post-RoPE cache as the differentiation
            # input, then converts its gradients to raw space by native VJP.
            post_leaves = [key.detach().requires_grad_(True) for key in cache.key_cache]
            cache.key_cache[:] = post_leaves
            cache.value_cache[:] = [value.detach() for value in cache.value_cache]
            stage[0] = "tail"
            tail = model.model(input_ids=ids[:, 8:], position_ids=positions[..., 8:],
                               cache_position=torch.arange(8, 12), past_key_values=cache,
                               use_cache=True, return_dict=True)
            logits = model.lm_head(tail.last_hidden_state[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, 32), ids[:, 9:].reshape(-1))
            oracle_grads = torch.autograd.grad(loss, post_leaves + [raw_tail[i] for i in range(2)])
        finally:
            for handle in handles:
                handle.remove()
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
        self.assertAlmostEqual(stats["loss"], float(loss), places=7)
        for index, layer in enumerate(layers):
            raw = raw_prefix[index].view(1, 8, 2, 4).transpose(1, 2).detach().requires_grad_(True)
            rotated = apply_multimodal_rotary_pos_emb(raw, raw, *embeddings[0],
                                                     layer.self_attn.rope_scaling["mrope_section"])[1]
            prefix_gradient = torch.autograd.grad(rotated, raw, grad_outputs=oracle_grads[index])[0]
            tail_gradient = oracle_grads[index + 2].view(1, 4, 2, 4).transpose(1, 2)
            expected = torch.cat((prefix_gradient, tail_gradient), dim=2)
            torch.testing.assert_close(actual[index], expected, rtol=1e-9, atol=1e-11)
            torch.testing.assert_close(actual[index][:, :, -1], torch.zeros_like(actual[index][:, :, -1]), rtol=0, atol=0)

    def test_cached_probe_failure_restores_custom_text_and_attention_dispatch(self):
        model = self.make_model()
        language_model = model.model.language_model
        original_text = language_model.forward
        # Simulate an instance-bound host compression override.  The native
        # probe must bypass it and restore it even when attention then fails.
        def host_forward(*args, **kwargs):
            raise AssertionError("Host compression forward entered the Fisher probe")
        language_model.forward = host_forward
        original_rope = model.model.rope_deltas
        model.train()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            cached_prefix_fisher(model, {"input_ids": torch.tensor([[1, 2, 3, 4]])},
                                  native_attention_forward=_toy_failing_attention_forward)
        self.assertIs(language_model.forward, host_forward)
        self.assertTrue(model.training)
        self.assertTrue(language_model.config.use_cache)
        self.assertIs(model.model.rope_deltas, original_rope)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(not layer.self_attn.k_proj._forward_hooks for layer in language_model.layers))
        language_model.forward = original_text

    def test_cached_probe_rejects_padding_answer_labels_and_visual_ending(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "without labels"):
            cached_prefix_fisher(model, {"input_ids": torch.tensor([[1, 2, 3]]),
                                         "labels": torch.tensor([[1, 2, 3]])})
        with self.assertRaisesRegex(ValueError, "unpadded"):
            cached_prefix_fisher(model, {"input_ids": torch.tensor([[1, 2, 3]]),
                                         "attention_mask": torch.tensor([[1, 1, 0]])})
        with self.assertRaisesRegex(ValueError, "at least two text"):
            cached_prefix_fisher(model, {"input_ids": torch.tensor([[1, 31, 2]])})


if __name__ == "__main__":
    unittest.main()

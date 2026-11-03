import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
    flash_attn_supports_top_left_mask,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from .baseline_utils import load_head_score
from .qwen_model import qwen_flash_attn2_forward_SparseMM
from .qwen_model_madakv_kvrepack import (
    _adjacent_similarity,
    _candidate_len,
    _env_float,
    _env_int,
    _merge_adjacent_raw_reapply,
    _topk_sorted,
)


def _topk_sparsemm_order(score: torch.Tensor, keep: int) -> torch.Tensor:
    """Match SparseMM.update_kv: full descending sort, then take the prefix."""
    total = int(score.shape[0])
    keep = min(max(int(keep), 0), total)
    if keep <= 0:
        return torch.empty((0,), device=score.device, dtype=torch.long)
    return torch.sort(score, dim=-1, descending=True).indices[:keep]


def _sparsemm_head_score(
    model_type: str,
    head_score: str,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    if head_score == "random":
        score = np.array([np.random.random() for _ in range(num_hidden_layers * num_attention_heads)])
    elif head_score == "visual":
        raw = load_head_score(model_type)
        score = np.array([np.mean(item[1]) for item in raw.items()])
    else:
        raise ValueError(f"Unsupported SPARSEMM_KVREPACK_HEAD_SCORE={head_score}")

    score = score / max(float(score.sum()), 1e-12)
    score = torch.tensor(score)
    score = score.view(num_hidden_layers, num_attention_heads // num_key_value_groups, num_key_value_groups)
    return score.sum(dim=-1)


class SparseMMKVRepackCluster:
    def __init__(
        self,
        window_size: int,
        kernel_size: int,
        pooling: str,
        budget_ratio: float,
        ratio: float,
        head_score: str,
        layer_idx: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_groups: int,
        gqa_func: str,
        model_type: str,
    ) -> None:
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.budget_ratio = float(budget_ratio)
        self.ratio = float(ratio)
        self.layer_idx = int(layer_idx)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_groups = int(num_key_value_groups)
        self.gqa_func = gqa_func
        self.score = _sparsemm_head_score(
            model_type=model_type,
            head_score=head_score,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_groups=self.num_key_value_groups,
        )
        self.head_adaptive_capacity = None

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None
        self.layer_qlens = None
        self.qlen_sum = 0
        self.cu_qlen = None
        self.cu_head_offset = None

    @torch.no_grad()
    def calcul_attn_sore(self, key_states: torch.Tensor, query_states: torch.Tensor) -> torch.Tensor:
        bsz, num_heads, q_len, head_dim = query_states.shape
        window = min(max(1, self.window_size), q_len)
        history_len = max(0, q_len - window)
        if history_len == 0:
            kv_heads = key_states.shape[1] // self.num_key_value_groups
            return torch.zeros((bsz, kv_heads, 0), device=query_states.device, dtype=query_states.dtype)

        attn_weights = torch.matmul(
            query_states[..., -window:, :],
            key_states.transpose(2, 3),
        ) / math.sqrt(head_dim)
        mask = torch.full(
            (window, window),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -window:, -window:] += mask[None, None, :, :]

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_mean = attn_weights[:, :, -window:, :history_len].mean(dim=-2)
        attn_mean = attn_mean.view(
            attn_mean.shape[0],
            num_heads // self.num_key_value_groups,
            self.num_key_value_groups,
            history_len,
        )
        if self.gqa_func == "max":
            attn_mean = attn_mean.max(dim=-2).values
        elif self.gqa_func == "mean":
            attn_mean = attn_mean.mean(dim=-2)
        else:
            raise ValueError(f"Unsupported SPARSEMM_KVREPACK_GQA_FUNC={self.gqa_func}")

        if self.pooling == "avgpool":
            pooled = F.avg_pool1d(
                attn_mean,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        elif self.pooling == "maxpool":
            pooled = F.max_pool1d(
                attn_mean,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        else:
            raise ValueError(f"Unsupported SPARSEMM_KVREPACK_POOLING={self.pooling}")
        return pooled[:, :, :history_len]

    def ensure_head_capacity(self, q_len: int, device: torch.device) -> torch.Tensor:
        base_capacity = float(q_len) * self.budget_ratio
        if self.head_adaptive_capacity is None:
            min_cache = int(base_capacity * self.ratio)
            remain_capacity = (
                (base_capacity - min_cache)
                * self.num_hidden_layers
                * self.num_attention_heads
                // self.num_key_value_groups
            )
            cap = torch.round(self.score * remain_capacity + min_cache).int()
            # Match opensource SparseMM: redistribute any per-head overflow after
            # building the adaptive capacity table for the current prompt.
            not_allocated_budget = 0
            cap_list = cap.tolist()
            for layer_idx in range(len(cap_list)):
                for head_idx in range(len(cap_list[0])):
                    total_budget = int(cap_list[layer_idx][head_idx])
                    if total_budget > q_len:
                        not_allocated_budget += total_budget - q_len
                        cap_list[layer_idx][head_idx] = int(q_len)

            while not_allocated_budget > 0:
                max_budget = -1
                max_layer_idx = -1
                max_head_idx = -1
                for layer_idx in range(len(cap_list)):
                    for head_idx in range(len(cap_list[0])):
                        current_budget = int(cap_list[layer_idx][head_idx])
                        if current_budget > max_budget and current_budget < q_len:
                            max_budget = current_budget
                            max_layer_idx = layer_idx
                            max_head_idx = head_idx
                if max_layer_idx == -1:
                    break
                can_allocate_budget = int(q_len) - max_budget
                cap_list[max_layer_idx][max_head_idx] += can_allocate_budget
                not_allocated_budget -= can_allocate_budget
            cap = torch.tensor(cap_list, dtype=torch.int32)
            self.head_adaptive_capacity = cap
        return self.head_adaptive_capacity.to(device=device)

    def init_metadata(self, k_lens: List[int], device: torch.device) -> None:
        h_kv = self.num_attention_heads // self.num_key_value_groups
        k_lens = [int(x) for x in k_lens]
        klen_sum = int(sum(k_lens))
        max_seqlen_k = int(max(k_lens)) if k_lens else 0

        self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=device)
        self.klen_sum = klen_sum
        self.max_seqlen_k = max_seqlen_k
        self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
        self.cu_klen = self.cu_headlens - self.head_lens
        self.cu_klen = torch.cat(
            [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=device)],
            dim=0,
        )
        self.layer_qlens = torch.ones(h_kv, dtype=torch.int32, device=device)
        self.qlen_sum = h_kv
        self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
        self.cu_qlen = torch.cat(
            [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=device)],
            dim=0,
        )
        self.cu_offset = torch.arange(0, h_kv + 1, dtype=torch.int32, device=device)
        self.cu_head_offset = torch.arange(1, h_kv + 1, dtype=torch.int32, device=device)


def init_sparsemm_kvrepack(self, rebuild_cluster: bool = False) -> None:
    if not hasattr(self.config, "sparsemm_kvrepack_window_size"):
        self.config.sparsemm_kvrepack_window_size = _env_int("SPARSEMM_KVREPACK_WINDOW_SIZE", 32)
    if not hasattr(self.config, "sparsemm_kvrepack_kernel_size"):
        self.config.sparsemm_kvrepack_kernel_size = _env_int("SPARSEMM_KVREPACK_KERNEL_SIZE", 5)
    if not hasattr(self.config, "sparsemm_kvrepack_pooling"):
        self.config.sparsemm_kvrepack_pooling = os.getenv("SPARSEMM_KVREPACK_POOLING", "avgpool")
    if not hasattr(self.config, "sparsemm_kvrepack_budget_ratio"):
        self.config.sparsemm_kvrepack_budget_ratio = _env_float(
            "SPARSEMM_KVREPACK_BUDGET_RATIO",
            _env_float("BUDGET_RATIO", 0.10),
        )
    if not hasattr(self.config, "sparsemm_kvrepack_ratio"):
        self.config.sparsemm_kvrepack_ratio = _env_float(
            "SPARSEMM_KVREPACK_RATIO",
            _env_float("RATIO", 0.10),
        )
    if not hasattr(self.config, "sparsemm_kvrepack_head_score"):
        self.config.sparsemm_kvrepack_head_score = os.getenv("SPARSEMM_KVREPACK_HEAD_SCORE", "visual")
    if not hasattr(self.config, "sparsemm_kvrepack_gqa_func"):
        self.config.sparsemm_kvrepack_gqa_func = os.getenv("SPARSEMM_KVREPACK_GQA_FUNC", "mean")
    if not hasattr(self.config, "sparsemm_kvrepack_compress_ratio"):
        self.config.sparsemm_kvrepack_compress_ratio = _env_float("SPARSEMM_KVREPACK_COMPRESS_RATIO", 0.80)
        if not (0.0 < float(self.config.sparsemm_kvrepack_compress_ratio) <= 1.0):
            raise ValueError(
                "SPARSEMM_KVREPACK_COMPRESS_RATIO must be in (0, 1], got "
                f"{self.config.sparsemm_kvrepack_compress_ratio}."
            )
    if not hasattr(self.config, "sparsemm_kvrepack_alpha"):
        self.config.sparsemm_kvrepack_alpha = _env_float(
            "SPARSEMM_KVREPACK_K_DIFF_ALPHA",
            _env_float("K_DIFF_ALPHA", 1.0),
        )
    if not hasattr(self.config, "sparsemm_simhead_merge_ratio"):
        self.config.sparsemm_simhead_merge_ratio = _env_float("SPARSEMM_SIMHEAD_MERGE_RATIO", 0.10)
    if not hasattr(self.config, "sparsemm_simhead_sim_space"):
        self.config.sparsemm_simhead_sim_space = os.getenv("SPARSEMM_SIMHEAD_SIM_SPACE", "raw").lower()
    if not hasattr(self.config, "sparsemm_kvrepack_verbose"):
        self.config.sparsemm_kvrepack_verbose = os.getenv(
            "SPARSEMM_KVREPACK_VERBOSE", "0"
        ).lower() in {"1", "true", "on", "yes"}
    # Current SparseMM-KVRepack uses merged K/V as ordinary cache slots and
    # keeps decode on the original varlen FlashAttention path.  log_count is
    # intentionally disabled to avoid switching merged heads to manual attention.
    self.config.sparsemm_kvrepack_score_correction = "none"
    if not hasattr(self, "_flash_attn_uses_top_left_mask"):
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    if rebuild_cluster or not hasattr(self, "kv_cluster"):
        self.kv_cluster = SparseMMKVRepackCluster(
            window_size=self.config.sparsemm_kvrepack_window_size,
            kernel_size=self.config.sparsemm_kvrepack_kernel_size,
            pooling=self.config.sparsemm_kvrepack_pooling,
            budget_ratio=self.config.sparsemm_kvrepack_budget_ratio,
            ratio=self.config.sparsemm_kvrepack_ratio,
            head_score=self.config.sparsemm_kvrepack_head_score,
            layer_idx=self.layer_idx,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups=self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.sparsemm_kvrepack_gqa_func,
            model_type=os.getenv("MODEL_TYPE", self.config.model_type),
        )


def _direct_full_record(attn_module, rope_key_states, value_states, cache_kwargs) -> Dict:
    bsz, h_kv, seq_len, head_dim = rope_key_states.shape
    if bsz != 1:
        raise AssertionError("SparseMM-KVRepack follows SparseMM and supports batch size 1 only.")
    flat_k = rope_key_states.reshape(-1, head_dim).detach()
    flat_v = value_states.reshape(-1, head_dim).detach()
    return {
        "layer_idx": int(attn_module.layer_idx),
        "module": attn_module,
        "direct_flat_key": flat_k,
        "direct_flat_value": flat_v,
        "k_lens": [int(seq_len)] * h_kv,
        "head_sim": torch.zeros((bsz, h_kv), device=flat_k.device, dtype=torch.float32),
        "cache_kwargs": cache_kwargs,
    }


def _select_sparsemm_candidate(
    score: torch.Tensor,
    raw_hist: torch.Tensor,
    rope_hist: torch.Tensor,
    value_hist: torch.Tensor,
    final_len: int,
    compress_ratio: float,
) -> Dict:
    history_len = int(score.shape[0])
    head_dim = int(raw_hist.shape[-1])
    final_len = min(max(int(final_len), 0), history_len)
    if final_len <= 0:
        empty_idx = torch.empty((0,), device=score.device, dtype=torch.long)
        empty_vec = raw_hist.new_empty((0, head_dim))
        return {
            "idx": empty_idx,
            "raw": empty_vec,
            "rope": rope_hist.new_empty((0, head_dim)),
            "value": value_hist.new_empty((0, head_dim)),
            "imp": score.new_empty((0,)),
            "direct_idx": empty_idx,
            "direct_rope": rope_hist.new_empty((0, head_dim)),
            "direct_value": value_hist.new_empty((0, head_dim)),
            "final": 0,
        }

    direct_idx = _topk_sparsemm_order(score, final_len)
    cand_len = _candidate_len(final_len, history_len, compress_ratio)
    cand_idx = _topk_sorted(score, cand_len)
    return {
        "idx": cand_idx,
        "raw": raw_hist.index_select(0, cand_idx),
        "rope": rope_hist.index_select(0, cand_idx),
        "value": value_hist.index_select(0, cand_idx),
        "imp": score.index_select(0, cand_idx),
        "direct_idx": direct_idx,
        "direct_rope": rope_hist.index_select(0, direct_idx),
        "direct_value": value_hist.index_select(0, direct_idx),
        "final": final_len,
    }


@torch.no_grad()
def _build_sparsemm_record(
    attn_module,
    raw_key_states: torch.Tensor,
    rope_key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_states_rope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_kwargs: Dict,
) -> Dict:
    config = attn_module.config
    bsz, h_kv, seq_len, head_dim = rope_key_states.shape
    if bsz != 1:
        raise AssertionError("SparseMM-KVRepack follows SparseMM and supports batch size 1 only.")

    key_rep = repeat_kv(rope_key_states, attn_module.num_key_value_groups)
    attn_score = attn_module.kv_cluster.calcul_attn_sore(key_rep, query_states_rope)
    window = min(max(1, int(config.sparsemm_kvrepack_window_size)), seq_len)
    history_len = max(0, seq_len - window)
    base_capacity = float(seq_len) * float(config.sparsemm_kvrepack_budget_ratio)

    if base_capacity > history_len or history_len == 0:
        return _direct_full_record(attn_module, rope_key_states, value_states, cache_kwargs)

    capacity = attn_module.kv_cluster.ensure_head_capacity(seq_len, attn_score.device)
    hist_raw = raw_key_states[:, :, :history_len, :]
    hist_rope = rope_key_states[:, :, :history_len, :]
    hist_value = value_states[:, :, :history_len, :]
    recent_key = rope_key_states[:, :, history_len:, :].detach()
    recent_value = value_states[:, :, history_len:, :].detach()

    heads: List[List[Dict]] = []
    k_lens: List[int] = []
    head_sim = torch.zeros((bsz, h_kv), device=rope_key_states.device, dtype=torch.float32)
    compress_ratio = float(config.sparsemm_kvrepack_compress_ratio)
    sim_space = (config.sparsemm_simhead_sim_space or "raw").lower()

    for bidx in range(bsz):
        batch_heads: List[Dict] = []
        for hidx in range(h_kv):
            final_len = int(capacity[int(attn_module.layer_idx), hidx].item())
            final_len = min(max(final_len, 0), history_len)
            cand = _select_sparsemm_candidate(
                score=attn_score[bidx, hidx],
                raw_hist=hist_raw[bidx, hidx],
                rope_hist=hist_rope[bidx, hidx],
                value_hist=hist_value[bidx, hidx],
                final_len=final_len,
                compress_ratio=compress_ratio,
            )
            source = cand["raw"] if sim_space == "raw" else cand["rope"]
            head_sim[bidx, hidx] = _adjacent_similarity(source)
            batch_heads.append(cand)
            k_lens.append(final_len + window)
        heads.append(batch_heads)

    return {
        "layer_idx": int(attn_module.layer_idx),
        "module": attn_module,
        "heads": heads,
        "k_lens": k_lens,
        "history_len": int(history_len),
        "recent_key": recent_key,
        "recent_value": recent_value,
        "head_sim": head_sim.detach(),
        "cos": cos.detach(),
        "sin": sin.detach(),
        "mrope_section": attn_module.rope_scaling["mrope_section"],
        "cache_kwargs": cache_kwargs,
    }


def _global_merge_masks(records: List[Dict], merge_ratio: float) -> List[torch.Tensor]:
    if not records:
        return []
    masks = [torch.zeros_like(rec["head_sim"], dtype=torch.bool) for rec in records]
    eligible = [idx for idx, rec in enumerate(records) if "heads" in rec]
    if not eligible:
        return masks

    sims = torch.stack([records[idx]["head_sim"].float() for idx in eligible], dim=1)
    bsz, num_layers, h_kv = sims.shape
    total = num_layers * h_kv
    ratio = min(max(float(merge_ratio), 0.0), 1.0)
    merge_count = int(math.ceil(total * ratio)) if ratio > 0 else 0
    merge_count = min(max(merge_count, 0), total)
    if merge_count == 0:
        return masks

    top = torch.topk(sims.reshape(bsz, total), k=merge_count, largest=True, sorted=False).indices
    for bidx in range(bsz):
        for flat_idx in top[bidx].tolist():
            layer_pos = flat_idx // h_kv
            head_idx = flat_idx % h_kv
            masks[eligible[layer_pos]][bidx, head_idx] = True
    return masks


@torch.no_grad()
def _finalize_prompt_cache(config, past_key_value: Cache) -> None:
    records: List[Dict] = getattr(config, "_sparsemm_kvrepack_records", [])
    if not records or past_key_value is None:
        return

    merge_ratio = float(config.sparsemm_simhead_merge_ratio)
    masks = _global_merge_masks(records, merge_ratio)
    verbose = bool(getattr(config, "sparsemm_kvrepack_verbose", False))
    total_heads = 0
    merge_heads = 0
    layer_counts: List[int] = []
    sim_values = []

    for rec, merge_mask in zip(records, masks):
        module = rec["module"]
        layer_idx = int(rec["layer_idx"])
        device = rec["head_sim"].device

        if "direct_flat_key" in rec:
            module.kv_cluster.init_metadata(rec["k_lens"], device=device)
            past_key_value.update(rec["direct_flat_key"], rec["direct_flat_value"], layer_idx, rec["cache_kwargs"])
            if verbose:
                layer_counts.append(0)
            continue

        heads = rec["heads"]
        recent_key = rec["recent_key"]
        recent_value = rec["recent_value"]
        bsz, h_kv, recent_len, head_dim = recent_key.shape
        head_k: List[torch.Tensor] = []
        head_v: List[torch.Tensor] = []
        k_lens: List[int] = []

        actual_merge_mask = torch.zeros_like(merge_mask, dtype=torch.bool) if verbose else None
        if verbose:
            total_heads += int(merge_mask.numel())
            sim_values.append(rec["head_sim"].reshape(-1).float())

        for bidx in range(bsz):
            for hidx in range(h_kv):
                cand = heads[bidx][hidx]
                final_len = int(cand["final"])
                do_merge = bool(merge_mask[bidx, hidx].item()) and int(cand["idx"].numel()) > final_len
                if do_merge:
                    hist_k, hist_v, hist_g = _merge_adjacent_raw_reapply(
                        raw_k_seq=cand["raw"],
                        rope_k_seq=cand["rope"],
                        v_seq=cand["value"],
                        importance=cand["imp"],
                        target_len=final_len,
                        positions=cand["idx"],
                        cos=rec["cos"],
                        sin=rec["sin"],
                        batch_idx=bidx,
                        alpha=float(config.sparsemm_kvrepack_alpha),
                        mrope_section=rec["mrope_section"],
                    )
                    if verbose:
                        actual_merge_mask[bidx, hidx] = bool(
                            hist_g.numel() > 0 and bool((hist_g > 1).any().item())
                        )
                else:
                    hist_k = cand["direct_rope"].contiguous()
                    hist_v = cand["direct_value"].contiguous()

                if recent_len > 0:
                    cur_k = torch.cat([hist_k, recent_key[bidx, hidx]], dim=0)
                    cur_v = torch.cat([hist_v, recent_value[bidx, hidx]], dim=0)
                else:
                    cur_k, cur_v = hist_k, hist_v
                head_k.append(cur_k)
                head_v.append(cur_v)
                k_lens.append(int(cur_k.shape[0]))

        flat_k = torch.cat(head_k, dim=0)
        flat_v = torch.cat(head_v, dim=0)
        if verbose:
            actual_layer_merge = int(actual_merge_mask.sum().item())
            layer_counts.append(actual_layer_merge)
            merge_heads += actual_layer_merge

        module.kv_cluster.init_metadata(k_lens, device=flat_k.device)
        past_key_value.update(flat_k, flat_v, layer_idx, rec["cache_kwargs"])

    if verbose and total_heads > 0:
        sims = torch.cat(sim_values) if sim_values else torch.zeros((1,))
        ratio = float(merge_heads) / float(total_heads)
        first = records[0]
        print(
            f"[SparseMM-KVRepack] adjacent-sim global gate: "
            f"merge_heads={merge_heads}/{total_heads} merge_ratio={ratio:.4f} "
            f"target_ratio={float(config.sparsemm_simhead_merge_ratio):.4f} "
            f"budget_ratio={float(config.sparsemm_kvrepack_budget_ratio):.4f} "
            f"sparse_min_ratio={float(config.sparsemm_kvrepack_ratio):.4f} "
            f"compress_ratio={float(config.sparsemm_kvrepack_compress_ratio):.4f} "
            f"alpha={float(config.sparsemm_kvrepack_alpha):.4f} "
            f"sim_space={config.sparsemm_simhead_sim_space} "
            f"score_correction=none attention=flash "
            f"first_history={int(first.get('history_len', 0))} "
            f"sim_min/mean/max={float(sims.min().item()):.4f}/{float(sims.mean().item()):.4f}/{float(sims.max().item()):.4f} "
            f"merge_heads_per_layer={layer_counts}"
        )

    config._sparsemm_kvrepack_records = []


def qwen_flash_attn2_forward_SparseMMKVRepack(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    if hidden_states.shape[1] == 1:
        return qwen_flash_attn2_forward_SparseMM(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    output_attentions = False
    bsz, q_len, _ = hidden_states.size()
    if bsz != 1:
        raise AssertionError("SparseMM-KVRepack follows SparseMM and supports batch size 1 only.")
    init_sparsemm_kvrepack(self, rebuild_cluster=True)

    query_states = self.q_proj(hidden_states)
    raw_key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    raw_key_states = raw_key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states_rope, key_states_rope = apply_multimodal_rotary_pos_emb(
        query_states,
        raw_key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
    )

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    if self.layer_idx == 0:
        self.config._sparsemm_kvrepack_records = []

    if use_cache and past_key_value is not None:
        record = _build_sparsemm_record(
            attn_module=self,
            raw_key_states=raw_key_states,
            rope_key_states=key_states_rope,
            value_states=value_states,
            query_states_rope=query_states_rope,
            cos=cos,
            sin=sin,
            cache_kwargs=cache_kwargs,
        )
        self.config._sparsemm_kvrepack_records.append(record)
        if self.layer_idx == self.config.num_hidden_layers - 1:
            _finalize_prompt_cache(self.config, past_key_value)

    key_for_attn = repeat_kv(key_states_rope, self.num_key_value_groups)
    value_for_attn = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states_rope.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states_rope = query_states_rope.to(target_dtype)
        key_for_attn = key_for_attn.to(target_dtype)
        value_for_attn = value_for_attn.to(target_dtype)

    query_fa = query_states_rope.transpose(1, 2)
    key_fa = key_for_attn.transpose(1, 2)
    value_fa = value_for_attn.transpose(1, 2)
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_fa,
        key_fa,
        value_fa,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_weights = None

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value

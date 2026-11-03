import math
import os
import gc
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import json
import numpy as np
from functools import lru_cache

from kvrepack.baseline_utils import load_head_score

# LLaVA-OV uses Qwen2 standard RoPE, so this fork keeps the Qwen2.5 mRoPE
# KVRepack implementation isolated from the standard-RoPE compatibility path.

@lru_cache(maxsize=None)
def load_param_config(model_type):
    param_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "param_config.json")
    with open(param_config_file, 'r') as f:
        param_config = json.load(f)
    if 'llava' in model_type: # llava-ov-7b, llava-hf
        if 'ov' in model_type or 'hf' in model_type:
            return_config = param_config["llava_ov"]
        else:
            raise NotImplementedError
    elif 'qwen2_5' in model_type or 'qwen2.5' in model_type:
        return_config = param_config["qwen2.5_vl"]
    elif 'internvl3' in model_type:
        return_config = param_config["internvl3"]
    else:
        raise NotImplementedError
    return return_config


def nearest_power_of_2(n):
    if n <= 0:
        return 0
    lower = 2 ** math.floor(math.log2(n))
    upper = 2 ** math.ceil(math.log2(n))
    return lower if (n - lower) <= (upper - n) else upper

def largest_power_of_2(n):
    if n <= 0:
        return 0
    return 2 ** math.floor(math.log2(n))

def get_retrieval_budget(seqlen: int, compress_ratio: float, retained_ratio: float) -> int:
    """
    Get the retrieval budget based on the sequence length and compression ratio.
    
    Args:
        seqlen: Sequence length
        compress_ratio: Compression ratio (fraction of tokens to keep)
        retained_ratio: Retained ratio for dynamic heads
    
    Returns:
        Budget for retrieval heads (50% of compressed tokens, rounded to power of 2)
    """
    budget = int(seqlen * compress_ratio * retained_ratio)
    budget = nearest_power_of_2(budget)
    model_type = os.environ.get("MODEL_TYPE")
    if "3b" in model_type: 
        if seqlen < 640: # for qwen-vl-3b
            return max(budget, 32)
    return max(budget, 64)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _select_mrope_batch(pos_emb: torch.Tensor, batch_idx: int, positions: torch.Tensor) -> torch.Tensor:
    if pos_emb.ndim == 4:
        return pos_emb[:, batch_idx, positions, :]
    if pos_emb.ndim == 3 and pos_emb.shape[0] == 3:
        return pos_emb[:, positions, :]
    if pos_emb.ndim == 3:
        return pos_emb[batch_idx, positions, :].unsqueeze(0).expand(3, -1, -1)
    if pos_emb.ndim == 2:
        return pos_emb[positions, :].unsqueeze(0).expand(3, -1, -1)
    raise ValueError(f"Unsupported mRoPE embedding shape: {tuple(pos_emb.shape)}")


def _apply_mrope_to_merged_keys(
    raw_k: torch.Tensor,
    cos_sel: torch.Tensor,
    sin_sel: torch.Tensor,
    mrope_section: Optional[List[int]],
) -> torch.Tensor:
    if mrope_section is None:
        if cos_sel.ndim == 3:
            cos_sel = cos_sel[0]
        if sin_sel.ndim == 3:
            sin_sel = sin_sel[0]
        return raw_k * cos_sel + _rotate_half(raw_k) * sin_sel
    mrope_section = list(mrope_section) * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos_sel.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin_sel.split(mrope_section, dim=-1))], dim=-1)
    return raw_k * cos + _rotate_half(raw_k) * sin


class HybridKV():
    '''
    manage some parameters (hyperparameters, head-wise attention metadata,...) for each layer
    '''
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool', base_capacity=None, ratio=None, normalize=None, 
                 layer_idx = None, num_hidden_layers = None, head_score=None, num_attention_heads=32, num_key_value_groups=1, gqa_func='mean', model_type=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.ratio = ratio
        self.local_len = None
        self.text_len = None

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads  
        self.num_hidden_layers = num_hidden_layers

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.num_key_value_heads = self.num_attention_heads // self.num_key_value_groups
        self.gqa_func = gqa_func

    def update_kv(
        self,
        pruned_k_cache,
        retrieved_k_cache,
        pruned_v_cache,
        retrieved_v_cache,
        pruned_head_mapping,
        retrieval_head_mapping,
        sparse_budget_per_head,
        pruned_group_size_cache=None,
    ):
        assert len(pruned_head_mapping) == 0 or len(pruned_head_mapping) == pruned_k_cache.shape[1]
        assert len(retrieval_head_mapping) == 0 or len(retrieval_head_mapping) == retrieved_k_cache.shape[1]
        n_retrieve = len(retrieval_head_mapping)
        n_prune = len(pruned_head_mapping)
        assert n_retrieve + n_prune == self.num_key_value_heads
        prune_idx = 0
        retrieve_idx = 0
        _device = pruned_k_cache.device if pruned_k_cache is not None else retrieved_k_cache.device
        bsz, _, _, head_dim = pruned_k_cache.shape if pruned_k_cache is not None else retrieved_k_cache.shape

        if pruned_k_cache is not None:
            pruned_k_cache = pruned_k_cache.reshape(bsz * n_prune, -1, head_dim)
            pruned_v_cache = pruned_v_cache.reshape(bsz * n_prune, -1, head_dim)
            if pruned_group_size_cache is not None:
                pruned_group_size_cache = pruned_group_size_cache.reshape(bsz * n_prune, -1)
            # bsz * kv_head * [1, budget, head_dim]
            pruned_k_cache = torch.split(pruned_k_cache, 1, dim=0)
            pruned_v_cache = torch.split(pruned_v_cache, 1, dim=0)
            if pruned_group_size_cache is not None:
                pruned_group_size_cache = torch.split(pruned_group_size_cache, 1, dim=0)
        if retrieved_k_cache is not None:
            retrieved_k_cache = retrieved_k_cache.reshape(bsz * n_retrieve, -1, head_dim)
            retrieved_v_cache = retrieved_v_cache.reshape(bsz * n_retrieve, -1, head_dim)
            # bsz * kv_head * [1, budget, head_dim]
            retrieved_k_cache = torch.split(retrieved_k_cache, 1, dim=0)
            retrieved_v_cache = torch.split(retrieved_v_cache, 1, dim=0)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            num_kv_heads = num_heads // self.num_key_value_groups
            self.layer_qlens = torch.ones(bsz * num_kv_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = bsz * num_kv_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens # qlen全为1, decoding场景
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)

            self.cu_offset = torch.arange(0, bsz * num_kv_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, bsz * num_kv_heads +1, dtype=torch.int32, device=_device)
        
        heads_key_states = [] # bsz * kv_head * [budget + prefill_local, head_dim]
        heads_value_states = []
        heads_group_size = []

        # reinit varlen metadata
        k_lens = [] # bsz * kv_head * [kv_length]
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for bidx in range(bsz):
            for head_idx in range(self.num_key_value_heads):
                budget = sparse_budget_per_head[bidx][head_idx]
                l = budget + self.local_len
                k_lens.append(l)
                max_seqlen_k = max(max_seqlen_k, l)
                klen_sum += l

                if head_idx in retrieval_head_mapping:
                    sparse_Kcache = retrieved_k_cache[retrieve_idx]
                    sparse_Vcache = retrieved_v_cache[retrieve_idx]
                    assert sparse_Kcache.shape[1] == budget + self.local_len, "Retrieved cache length does not match budget + local length"
                    sparse_group_size = torch.ones(
                        (sparse_Kcache.shape[1],),
                        device=sparse_Kcache.device,
                        dtype=torch.float32,
                    )
                    retrieve_idx += 1
                else:
                    sparse_Kcache = torch.cat([
                            pruned_k_cache[prune_idx][:, :budget],
                            pruned_k_cache[prune_idx][:, -self.local_len:],
                        ],
                        dim=1
                    )
                    sparse_Vcache = torch.cat([
                            pruned_v_cache[prune_idx][:, :budget],
                            pruned_v_cache[prune_idx][:, -self.local_len:],
                        ],
                        dim=1
                    )
                    if pruned_group_size_cache is not None:
                        sparse_group_size = torch.cat(
                            [
                                pruned_group_size_cache[prune_idx][:, :budget],
                                pruned_group_size_cache[prune_idx][:, -self.local_len:],
                            ],
                            dim=1,
                        ).reshape(-1).to(device=sparse_Kcache.device, dtype=torch.float32)
                    else:
                        sparse_group_size = torch.ones(
                            (sparse_Kcache.shape[1],),
                            device=sparse_Kcache.device,
                            dtype=torch.float32,
                        )
                    assert sparse_Kcache.shape[1] == budget + self.local_len, "Pruned cache length does not match budget + local token length"
                    prune_idx += 1

                heads_key_states.append(sparse_Kcache.view(-1, head_dim))
                heads_value_states.append(sparse_Vcache.view(-1, head_dim))
                heads_group_size.append(sparse_group_size.reshape(-1))

        init_metadata(self.num_attention_heads, k_lens, klen_sum, max_seqlen_k)

        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)
        self.last_group_size = torch.cat(heads_group_size, dim=0)
        return heads_key_states, heads_value_states

class BudgetAllocMethod(Enum):
    HEAD_ADA = "head_ada"
    UNIFORM = "uniform"
    RANDOM = "random"
    SPARSEMM_INIT = "sparsemm_init"
    SPARSEMM_ATTN = "sparsemm_attn"
    HEAD_LEVEL_ONLY = "head_level_only"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, method_str: str):
        return cls[method_str.upper()]
    
# Ablation: 由于我们方法的prompt动态性，应该跑一个数据集比如NextQA，然后测整体运行时间？
class HeadClassifyMethod(Enum):
    HEAD_ADA = "head_ada" # Ours: online head-wise identification/classification
    DYNAMIC = "dynamic" # fully dynamic pruning
    STATIC = "static" # fully static pruning
    RANDOM = "random" # random classification

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, method_str: str):
        return cls[method_str.upper()]

class HybridKVCache(Cache):
    _hybrid_kvrepack_dump_counter = 0

    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(
        self,
        layers: List[Qwen2_5_VLDecoderLayer],
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: int,
        device: Union[str, torch.device],
        sparse_budget: int = 2048,
        sparse_budget_ratio: float = 0.1,
        dtype: Optional[torch.dtype] = None,
        is_vision_token_mask: Optional[torch.Tensor] = None,
        is_text_token_mask: Optional[torch.Tensor] = None,
    ) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()

        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.device = device
        self.dtype = dtype
        self.layers = layers
        self.model_type = os.environ.get("MODEL_TYPE")

        self.param_config = load_param_config(self.model_type)
        self.verbose = os.environ.get("HYBRIDKV_VERBOSE", "0").lower() in {"1", "true", "on", "yes"}
        self.hybrid_kvrepack_verbose = os.environ.get("HYBRID_KVREPACK_VERBOSE", "0").lower() in {
            "1",
            "true",
            "on",
            "yes",
        }
        if self.verbose:
            print(self.param_config)

        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # NOTE: ====deprecated====
        self.sparse_budget = sparse_budget # budget-based
        # NOTE: ====deprecated====
        self.sparse_budget_ratio = sparse_budget_ratio if sparse_budget_ratio is not None else 0.1
        self.avg_budget = int(max_cache_len * self.sparse_budget_ratio)
        self.dynamic_budget_ratio = 0.75 # retained ratio for budget of dynamic heads, 0.75 for qwen
        self.hybridkv_kvrepack = os.environ.get("METHOD") in {"hybridkv_kvrepack"}
        self.chunk_size = int(os.environ.get("HYBRID_KVREPACK_CHUNK_SIZE", 8)) if self.hybridkv_kvrepack else 8
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")
        self.local_len = 32 # local window size
        self.local_chunk = self.local_len // self.chunk_size
        self.budget_method = BudgetAllocMethod.from_string(os.environ.get("BUDGET_METHOD", "head_ada"))

        # vision token info
        # TODO: 当前假设batch内所有token的video位置都是一样的
        self.vision_token_start = torch.argmax(is_vision_token_mask.long(), dim=1)
        self.vision_token_end = is_vision_token_mask.size(1) - torch.argmax(is_vision_token_mask.long().flip(dims=[1]), dim=1)
        self.is_vision_token_mask = is_vision_token_mask
        self.vision_token_index = self.get_vision_token_index(is_vision_token_mask)

        # NOTE: only used during prefill
        self.ori_key_states: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.ori_value_states: List[torch.Tensor] = [None for _ in range(self.num_layers)]

        # head calibration. 
        # sparsity_ratio的token没有达到score_threshold才能被视为video head
        self.head_classify_method = HeadClassifyMethod.from_string(os.environ.get("HEAD_CLASSIFY", "head_ada"))
        if self.head_classify_method == HeadClassifyMethod.HEAD_ADA:
            self.score_threshold = float(self.param_config["score_threshold"])
        elif self.head_classify_method == HeadClassifyMethod.DYNAMIC:
            self.score_threshold = 1.0
        elif self.head_classify_method == HeadClassifyMethod.STATIC:
            self.score_threshold = 0.0
        self.sparsity_ratio = 0.95 # head calibration

        # budget for retrieval heads
        # self.retrieval_ratio = 0.05 # TODO: 调参, retrieval的budget占比，或调整为固定大小
        if self.head_classify_method == HeadClassifyMethod.HEAD_ADA:
            if self.budget_method == BudgetAllocMethod.UNIFORM: # uniform allocation for both heads
                self.retrieval_budget = self.avg_budget
            else:
                self.retrieval_budget = get_retrieval_budget(max_cache_len, self.sparse_budget_ratio, self.dynamic_budget_ratio) # retrieval budget
        elif self.head_classify_method == HeadClassifyMethod.DYNAMIC:
            # for fixed budget ratio
            chunks = max_cache_len // self.chunk_size
            self.retrieval_budget = int(chunks * self.sparse_budget_ratio) * self.chunk_size
        elif self.head_classify_method == HeadClassifyMethod.STATIC:
            self.retrieval_budget = 0

        self.hybrid_kvrepack_compress_ratio = float(
            os.environ.get("HYBRID_KVREPACK_COMPRESS_RATIO", "0.5")
        )
        if not (0.0 < self.hybrid_kvrepack_compress_ratio <= 1.0):
            raise ValueError(
                "HYBRID_KVREPACK_COMPRESS_RATIO must be in (0, 1], got "
                f"{self.hybrid_kvrepack_compress_ratio}."
            )
        self.hybrid_kvrepack_k_diff_alpha = float(
            os.environ.get(
                "HYBRID_KVREPACK_K_DIFF_ALPHA",
                os.environ.get("K_DIFF_ALPHA", "0.5"),
            )
        )
        self.hybrid_kvrepack_max_group_size = int(
            os.environ.get("HYBRID_KVREPACK_MAX_GROUP_SIZE", "0")
        )
        if self.hybrid_kvrepack_max_group_size < 0:
            raise ValueError(
                "HYBRID_KVREPACK_MAX_GROUP_SIZE must be >= 0, got "
                f"{self.hybrid_kvrepack_max_group_size}."
            )
        # This cleaned Hybrid-KVRepack variant never applies log/count correction;
        # merged K/V always run through the original HybridKV FlashAttention path.
        self.hybrid_kvrepack_score_correction = "none"
        self.hybrid_kvrepack_simhead_enable = (
            os.environ.get("HYBRID_KVREPACK_SIMHEAD_ENABLE", "1").lower()
            not in {"0", "false", "off", "none"}
        )
        self.hybrid_kvrepack_simhead_merge_ratio = float(
            os.environ.get("HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO", "0.5")
        )
        self.hybrid_kvrepack_simhead_sim_space = os.environ.get(
            "HYBRID_KVREPACK_SIMHEAD_SIM_SPACE", "raw"
        ).lower()
        self.hybrid_kvrepack_force_manual_attention = False
        value_merge_env = "mean"
        default_k_merge_space = "raw_reapply"
        k_merge_space = os.environ.get(
            "HYBRID_KVREPACK_K_MERGE_SPACE",
            os.environ.get("K_MERGE_SPACE", default_k_merge_space),
        ).lower()
        if k_merge_space in {"raw", "raw_reapply", "snapkv"}:
            self.hybrid_kvrepack_k_merge_space = "raw_reapply"
        elif k_merge_space in {"rope", "rope_mean", "asym", "asymkv"}:
            self.hybrid_kvrepack_k_merge_space = "rope"
        else:
            raise ValueError(
                "HYBRID_KVREPACK_K_MERGE_SPACE must be raw_reapply/snapkv or rope/asymkv, "
                f"got {k_merge_space}."
            )
        block_meta_weight = os.environ.get(
            "HYBRID_KVREPACK_BLOCK_META_WEIGHT", "uniform"
        ).lower()
        if block_meta_weight in {"uniform", "unweighted", "mean", "none"}:
            self.hybrid_kvrepack_block_meta_weight = "uniform"
        elif block_meta_weight in {"group_size", "weighted", "count"}:
            self.hybrid_kvrepack_block_meta_weight = "group_size"
        else:
            raise ValueError(
                "HYBRID_KVREPACK_BLOCK_META_WEIGHT must be uniform or group_size, "
                f"got {block_meta_weight}."
            )
        static_merge_scope = os.environ.get("HYBRID_KVREPACK_STATIC_MERGE_SCOPE", "visual").lower()
        if static_merge_scope in {"visual", "vision", "video", "image"}:
            self.hybrid_kvrepack_static_merge_scope = "visual"
        elif static_merge_scope in {"all", "full", "both"}:
            self.hybrid_kvrepack_static_merge_scope = "all"
        else:
            raise ValueError(
                "HYBRID_KVREPACK_STATIC_MERGE_SCOPE must be visual or all, "
                f"got {static_merge_scope}."
            )
        self.hybrid_kvrepack_value_merge = "mean"
        if not (0.0 <= self.hybrid_kvrepack_simhead_merge_ratio <= 1.0):
            raise ValueError(
                "HYBRID_KVREPACK_SIMHEAD_MERGE_RATIO must be in [0, 1], got "
                f"{self.hybrid_kvrepack_simhead_merge_ratio}."
            )
        if self.hybridkv_kvrepack and self.retrieval_budget > 0:
            print(
                "[Hybrid-KVRepack] no-log merge enabled: "
                f"chunk_size={self.chunk_size} "
                f"compress_ratio={self.hybrid_kvrepack_compress_ratio} "
                f"simhead_ratio={self.hybrid_kvrepack_simhead_merge_ratio} "
                f"k_diff_alpha={self.hybrid_kvrepack_k_diff_alpha} "
                f"max_group_size={self.hybrid_kvrepack_max_group_size} "
                f"retrieval_budget={self.retrieval_budget} "
                f"score_correction=none "
                f"block_meta={self.hybrid_kvrepack_block_meta_weight} "
                f"static_merge_scope={self.hybrid_kvrepack_static_merge_scope} "
                f"value_merge=mean "
                f"k_merge_space={self.hybrid_kvrepack_k_merge_space} "
                f"force_manual=0"
            )

        # 0: query-agnostic head, 1: query-aware head
        self.head_category: torch.Tensor = torch.zeros(
            self.num_layers, self.max_batch_size,
            self.num_key_value_heads, device=self.device,
            dtype=torch.bool
        )
        
        # budget allocation
        # (num_layers, batch_size, num_kv_heads)
        self.sparse_budget_per_head: List[List[List[int]]] = None
        # NOTE: only used during prefill
        self.cross_attn_head_all_layers : List[torch.Tensor] = [None for _ in range(self.num_layers)]

        # pruned heads
        self.pruned_head_mapping: List[List[int]] = [[] for _ in range(self.num_layers)]
        # NOTE: only used during prefill
        self.pruned_k_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.pruned_v_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.pruned_group_size_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.pruned_importance_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.pruned_rope_history_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.pruned_raw_history_cache: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self.pruned_value_history_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.pruned_mrope_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]] = [
            None for _ in range(self.num_layers)
        ]

        # retrieval heads
        self.retrieval_head_mapping: List[List[int]] = [[] for _ in range(self.num_layers)]
        self.offload_k_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.offload_v_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.offload_k_local_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.block_meta: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.block_valid_mask: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.block_meta_idx: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.layer_chunks: List[int] = [0 for _ in range(self.num_layers)]
        self.retrieval_flat_head_ids: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        # [num_layers, batch_size, num_retrieval_heads, num_chunks]
        self.selection_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.hybrid_kvrepack_head_sim: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self.hybrid_kvrepack_merge_mask: Optional[torch.Tensor] = None
        self.hybrid_kvrepack_retrieval_records: List[Optional[dict]] = [None for _ in range(self.num_layers)]
        self.hybrid_kvrepack_simhead_finalized = False
        self.hybrid_kvrepack_dump_merge_path = os.environ.get("HYBRID_KVREPACK_DUMP_MERGE_PATH", "").strip()
        self.chunk_offsets = torch.arange(self.chunk_size, device=self.device, dtype=torch.long)
        # NOTE: only used during prefill
        self.last_query_temp: List[torch.Tensor] = [None for _ in range(self.num_layers)]

        # GPU KVCache buffer
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            # import nvtx
            # copy_old_rng = nvtx.start_range("copy old")
            from kvrepack_kernel import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)
            # nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    
    def get_vision_token_index(self, is_vision_token_mask: torch.Tensor):
        bsz, seq_len = is_vision_token_mask.shape
        vision_token_len = is_vision_token_mask.sum(dim=1)
        max_vision_tokens = vision_token_len.max()

        # 2. 获取所有视觉令牌的坐标 (batch_idx, seq_idx)
        # 这和之前的方法一样
        nonzero_indices = torch.nonzero(is_vision_token_mask, as_tuple=False)

        # 3. 计算每个视觉令牌在输出张量中的列索引 (这是关键步骤)
        # a. 将布尔掩码转为整数，并沿序列维度计算累积和
        #    这会给每个视觉令牌一个在其序列内的排名 (从1开始)
        col_indices_cumsum = torch.cumsum(is_vision_token_mask.int(), dim=1)

        # b. 从视觉令牌的位置提取这些排名，并减1使其成为从0开始的索引
        col_indices = col_indices_cumsum[is_vision_token_mask] - 1

        # 4. 创建目标索引张量并填充值
        # a. 创建一个用0填充的目标张量
        #    用0填充是安全的，因为它是无效索引或与原始代码中的填充值匹配
        vision_token_index = torch.zeros(bsz, max_vision_tokens, dtype=torch.long, device=is_vision_token_mask.device)

        # b. 使用高级索引一次性将所有视觉令牌的序列索引 (nonzero_indices[:, 1])
        #    放置到它们的目标位置 (nonzero_indices[:, 0] 作为行, col_indices 作为列)
        vision_token_index[nonzero_indices[:, 0], col_indices] = nonzero_indices[:, 1]
        return vision_token_index
    
    def calibrate_head(self, layer_idx: int, key_states: torch.Tensor, query_states: torch.Tensor):
        assert query_states.shape[-2] == key_states.shape[-2], "In prefilling, sequence length of query and key should be the same!"
        bsz, _, seq_len, _ = key_states.shape
        vision_token_end = self.vision_token_end[0].item()
        vision_token_start = self.vision_token_start[0].item()
        current_text_len = seq_len - vision_token_end
        if current_text_len > 512: # avoid OOM
            current_text_len = 512
        
        self.layers[layer_idx].self_attn.kv_cluster.text_len = current_text_len

        # 1. cross-modal attention score
        # TODO: maybe make cross-attention score in a fixed window
        # current_text_len = min(current_text_len, 32)
        # (bsz, q_head, seq_len, head_dim) -> (bsz, q_head, text_len, head_dim)
        text_query_states = query_states[:, :, vision_token_end:vision_token_end+current_text_len:, :]
        # (bsz, kv_head, kv_group, text_len, head_dim) * (bsz, kv_head, head_dim, seq_len) -> (bsz, kv_head, kv_group, text_len, seq_len)
        key = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
        # text-guided
        text_attn_score = text_query_states @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # text causal mask
        attn_mask = torch.triu(
            torch.ones(current_text_len, current_text_len, 
                       dtype=torch.bool, device=text_attn_score.device), 
            diagonal=1
        )
        text_attn_score[:, :, :, vision_token_end:vision_token_end+current_text_len] = text_attn_score[:, :, :, vision_token_end:vision_token_end+current_text_len].masked_fill(attn_mask, float("-inf"))
        # [bsz, q_head, text_len, seq_len]
        text_attn_score = nn.functional.softmax(text_attn_score, dim=-1, dtype=torch.float32).to(self.dtype)

        # 2. classify KV heads with attention score
        vision_len = vision_token_end - vision_token_start
        k = int(vision_len * (1 - self.sparsity_ratio))
        topk_indices = torch.topk(text_attn_score, k=k, dim=-1).indices
        topk_text_score = torch.gather(text_attn_score, dim=-1, index=topk_indices) # [bsz, q_head, text_len, sparse_budget]
        topk_text_score_avg = topk_text_score.sum(-1).mean(-1) # [bsz, q_head]
        # import pdb;pdb.set_trace()
        self.cross_attn_head_all_layers[layer_idx] = topk_text_score_avg.clone()

        # save cdf
        # self.save_cdf(layer_idx, text_attn_score)

        # mean for GQA
        # topk_text_score_avg = topk_text_score_avg.view(
        #     self.max_batch_size, self.num_key_value_heads, self.num_key_value_groups).mean(dim=-1)
        # is_query_aware_head = topk_text_score_avg < self.score_threshold

        # max for each query head and compute any for KV head
        query_aware_heads = topk_text_score_avg < self.score_threshold
        query_aware_heads_groups = query_aware_heads.view(
            self.max_batch_size, self.num_key_value_heads, self.num_key_value_groups
        )
        is_query_aware_head = query_aware_heads_groups.any(dim=-1) # [bsz, kv_head]

        self.head_category[layer_idx] = is_query_aware_head
        if self.verbose:
            print(f"Layer {layer_idx} head category: {self.head_category[layer_idx]}")
            print(f"Layer {layer_idx} - Query-aware heads: {is_query_aware_head.sum().item()}/{is_query_aware_head.numel()}")

        # 3. retrieval head / pruned head mapping
        # TODO: currently assume all batch share the same head categories
        head_cat = self.head_category[layer_idx]
        for head_idx in range(self.num_key_value_heads):
            if head_cat[0][head_idx]:
                self.retrieval_head_mapping[layer_idx].append(head_idx)
            else:
                self.pruned_head_mapping[layer_idx].append(head_idx)

        # store last query states only for retrieval heads
        query = query_states.view(bsz, self.num_key_value_heads, self.num_key_value_groups, -1, self.head_dim)
        if len(self.retrieval_head_mapping[layer_idx]) > 0:
            self.last_query_temp[layer_idx] = query[:, self.retrieval_head_mapping[layer_idx], :, -1, :].reshape(
                bsz, len(self.retrieval_head_mapping[layer_idx]) * self.num_key_value_groups, 1, self.head_dim
            ).clone()

    def _hybrid_kvrepack_target_history_len(self, history_len: int) -> int:
        if history_len <= 0:
            return 0
        target = int(math.ceil(history_len * self.hybrid_kvrepack_compress_ratio))
        target = int(math.ceil(target / self.chunk_size) * self.chunk_size)
        if self.hybrid_kvrepack_max_group_size > 0:
            min_target = int(math.ceil(history_len / self.hybrid_kvrepack_max_group_size))
            target = max(target, min_target)
        target = max(self.chunk_size, min(history_len, target))
        return target

    def _hybrid_kvrepack_effective_target_len(self, seq_len: int, target_len: int) -> int:
        target_len = min(max(int(target_len), 0), int(seq_len))
        if self.hybrid_kvrepack_max_group_size > 0 and seq_len > 0:
            min_target = int(math.ceil(seq_len / self.hybrid_kvrepack_max_group_size))
            target_len = max(target_len, min_target)
        return target_len

    def _hybrid_kvrepack_select_merge_indices(
        self,
        merge_score: torch.Tensor,
        n_to_merge: int,
        seq_len: int,
    ) -> torch.Tensor:
        if n_to_merge <= 0:
            return torch.empty((0,), device=merge_score.device, dtype=torch.long)
        max_group_size = int(self.hybrid_kvrepack_max_group_size)
        if max_group_size <= 0:
            return torch.topk(merge_score, k=n_to_merge, largest=False, sorted=False).indices
        if max_group_size <= 1:
            return torch.empty((0,), device=merge_score.device, dtype=torch.long)

        max_deleted_run = max_group_size - 1
        order = torch.argsort(merge_score.detach().float(), dim=0, descending=False).cpu().tolist()
        deleted = [False] * int(seq_len)
        selected = []
        for idx in order:
            if len(selected) >= n_to_merge:
                break
            pos = int(idx) + 1
            left = 0
            j = pos - 1
            while j >= 0 and deleted[j]:
                left += 1
                j -= 1
            right = 0
            j = pos + 1
            while j < seq_len and deleted[j]:
                right += 1
                j += 1
            if left + 1 + right <= max_deleted_run:
                deleted[pos] = True
                selected.append(int(idx))

        if not selected:
            return torch.empty((0,), device=merge_score.device, dtype=torch.long)
        return torch.tensor(selected, device=merge_score.device, dtype=torch.long)

    def _hybrid_kvrepack_chunk_meta(
        self,
        key_states_indexed: torch.Tensor,
        group_indexed: torch.Tensor,
    ) -> torch.Tensor:
        if self.hybrid_kvrepack_block_meta_weight == "group_size":
            meta_weight = group_indexed.float()
        else:
            # Uniform mode avoids giving larger merged groups extra influence during chunk retrieval.
            meta_weight = (group_indexed > 0).to(dtype=torch.float32)
        meta_denom = meta_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (
            key_states_indexed.float() * meta_weight.unsqueeze(-1)
        ).sum(dim=-2) / meta_denom

    def _hybrid_kvrepack_candidate_len(self, final_len: int, history_len: int) -> int:
        if history_len <= 0:
            return 0
        final_len = min(max(int(final_len), 0), int(history_len))
        if final_len <= 0:
            return 0
        cand_len = int(math.ceil(final_len / max(float(self.hybrid_kvrepack_compress_ratio), 1e-6)))
        return min(max(cand_len, final_len), int(history_len))

    def _hybrid_kvrepack_adjacent_similarity(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.shape[-2] < 2:
            return torch.zeros(seq.shape[:-2], device=seq.device, dtype=torch.float32)
        x = F.normalize(seq.float(), dim=-1, eps=1e-6)
        sim = (x[..., :-1, :] * x[..., 1:, :]).sum(dim=-1)
        return torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=-1)

    def _hybrid_kvrepack_record_head_sim(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        raw_key_states: Optional[torch.Tensor],
        history_len: int,
    ) -> None:
        if not self.hybridkv_kvrepack or not self.hybrid_kvrepack_simhead_enable:
            return
        source = raw_key_states if self.hybrid_kvrepack_simhead_sim_space == "raw" and raw_key_states is not None else key_states
        self.hybrid_kvrepack_head_sim[layer_idx] = self._hybrid_kvrepack_adjacent_similarity(
            source[:, :, :history_len, :]
        ).detach()

    def _hybrid_kvrepack_update_pruned_head_sim_from_candidates(self) -> None:
        """Score pruned heads on the actual expanded candidates that would be merged."""
        if (
            not self.hybridkv_kvrepack
            or not self.hybrid_kvrepack_simhead_enable
            or self.sparse_budget_per_head is None
        ):
            return

        for layer_idx, pruned_heads in enumerate(self.pruned_head_mapping):
            if not pruned_heads:
                continue
            importance = self.pruned_importance_cache[layer_idx]
            rope_history = self.pruned_rope_history_cache[layer_idx]
            if importance is None or rope_history is None:
                continue

            bsz, _, history_len, _ = rope_history.shape
            if self.hybrid_kvrepack_head_sim[layer_idx] is None:
                self.hybrid_kvrepack_head_sim[layer_idx] = torch.zeros(
                    (bsz, self.num_key_value_heads),
                    device=rope_history.device,
                    dtype=torch.float32,
                )
            else:
                self.hybrid_kvrepack_head_sim[layer_idx] = self.hybrid_kvrepack_head_sim[layer_idx].to(
                    device=rope_history.device,
                    dtype=torch.float32,
                )

            raw_history = self.pruned_raw_history_cache[layer_idx]
            sim_source = raw_history if self.hybrid_kvrepack_simhead_sim_space == "raw" and raw_history is not None else rope_history

            for bidx in range(bsz):
                for local_hidx, head_idx in enumerate(pruned_heads):
                    budget = int(self.sparse_budget_per_head[layer_idx][bidx][head_idx])
                    budget = min(max(budget, 0), history_len)
                    imp = importance[bidx, local_hidx].float()
                    if (
                        self.hybrid_kvrepack_static_merge_scope == "visual"
                        and self.is_vision_token_mask is not None
                    ):
                        vision_mask = self.is_vision_token_mask[bidx, :history_len].to(
                            device=rope_history.device,
                            dtype=torch.bool,
                        )
                        if budget <= 0:
                            visual_budget = 0
                        else:
                            base_idx = torch.topk(
                                imp,
                                k=budget,
                                dim=0,
                                largest=True,
                                sorted=True,
                            ).indices
                            base_text = ~vision_mask.index_select(0, base_idx)
                            visual_budget = budget - int(base_text.sum().item())
                        visual_pool = torch.nonzero(vision_mask, as_tuple=False).view(-1)
                        cand_len = self._hybrid_kvrepack_candidate_len(
                            visual_budget,
                            int(visual_pool.numel()),
                        )
                        if visual_budget <= 0 or cand_len < 2 or int(visual_pool.numel()) == 0:
                            sim = torch.zeros((), device=rope_history.device, dtype=torch.float32)
                        else:
                            visual_scores = imp.index_select(0, visual_pool)
                            local_idx = torch.topk(
                                visual_scores,
                                k=cand_len,
                                dim=0,
                                largest=True,
                                sorted=False,
                            ).indices
                            keep_idx = visual_pool.index_select(0, local_idx)
                            keep_idx, _ = torch.sort(keep_idx)
                            sim = self._hybrid_kvrepack_adjacent_similarity(
                                sim_source[bidx, local_hidx, keep_idx].unsqueeze(0)
                            ).reshape(())
                    else:
                        cand_len = self._hybrid_kvrepack_candidate_len(budget, history_len)
                        if cand_len < 2:
                            sim = torch.zeros((), device=rope_history.device, dtype=torch.float32)
                        else:
                            keep_idx = torch.topk(imp, k=cand_len, dim=0, largest=True, sorted=False).indices
                            keep_idx, _ = torch.sort(keep_idx)
                            sim = self._hybrid_kvrepack_adjacent_similarity(
                                sim_source[bidx, local_hidx, keep_idx].unsqueeze(0)
                            ).reshape(())
                    self.hybrid_kvrepack_head_sim[layer_idx][bidx, head_idx] = sim.detach()

    def _hybrid_kvrepack_dump_merge_info(self, sims: torch.Tensor, ratio: float) -> None:
        path = getattr(self, "hybrid_kvrepack_dump_merge_path", "")
        if not path or self.hybrid_kvrepack_merge_mask is None:
            return
        if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) not in {"0", ""}:
            return

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        mask_cpu = self.hybrid_kvrepack_merge_mask.detach().to("cpu")
        sims_cpu = sims.detach().float().to("cpu")
        num_layers, bsz, num_heads = mask_cpu.shape
        dynamic_by_layer = [set(map(int, heads)) for heads in self.retrieval_head_mapping]

        with open(path, "a", encoding="utf-8") as handle:
            for bidx in range(bsz):
                sample_index = int(HybridKVCache._hybrid_kvrepack_dump_counter)
                HybridKVCache._hybrid_kvrepack_dump_counter += 1

                selected_heads = []
                layer_summary = []
                dynamic_selected = 0
                static_selected = 0
                dynamic_total = 0
                static_total = 0

                for layer_idx in range(num_layers):
                    dyn_heads = dynamic_by_layer[layer_idx] if layer_idx < len(dynamic_by_layer) else set()
                    layer_dynamic = []
                    layer_static = []
                    layer_selected = []
                    for head_idx in range(num_heads):
                        is_dynamic = head_idx in dyn_heads
                        if is_dynamic:
                            dynamic_total += 1
                        else:
                            static_total += 1
                        if not bool(mask_cpu[layer_idx, bidx, head_idx].item()):
                            continue
                        item = {
                            "layer": int(layer_idx),
                            "head": int(head_idx),
                            "head_type": "dynamic" if is_dynamic else "static",
                            "sim": float(sims_cpu[bidx, layer_idx, head_idx].item()),
                        }
                        selected_heads.append(item)
                        layer_selected.append(int(head_idx))
                        if is_dynamic:
                            dynamic_selected += 1
                            layer_dynamic.append(int(head_idx))
                        else:
                            static_selected += 1
                            layer_static.append(int(head_idx))
                    layer_summary.append({
                        "layer": int(layer_idx),
                        "dynamic_heads": sorted(int(h) for h in dyn_heads),
                        "merge_heads": layer_selected,
                        "merge_dynamic_heads": layer_dynamic,
                        "merge_static_heads": layer_static,
                    })

                record = {
                    "record_type": "hybrid_kvrepack_merge_mask",
                    "sample_index": sample_index,
                    "batch_index": int(bidx),
                    "num_layers": int(num_layers),
                    "num_kv_heads": int(num_heads),
                    "target_ratio": float(ratio),
                    "selected": int(len(selected_heads)),
                    "total_heads": int(num_layers * num_heads),
                    "dynamic_selected": int(dynamic_selected),
                    "dynamic_total": int(dynamic_total),
                    "static_selected": int(static_selected),
                    "static_total": int(static_total),
                    "sim_min": float(sims_cpu[bidx].min().item()),
                    "sim_mean": float(sims_cpu[bidx].mean().item()),
                    "sim_max": float(sims_cpu[bidx].max().item()),
                    "budget_ratio": float(self.sparse_budget_ratio),
                    "compress_ratio": float(self.hybrid_kvrepack_compress_ratio),
                    "k_diff_alpha": float(self.hybrid_kvrepack_k_diff_alpha),
                    "chunk_size": int(self.chunk_size),
                    "max_group_size": int(self.hybrid_kvrepack_max_group_size),
                    "static_merge_scope": str(self.hybrid_kvrepack_static_merge_scope),
                    "k_merge_space": str(self.hybrid_kvrepack_k_merge_space),
                    "score_correction": "none",
                    "selected_heads": selected_heads,
                    "layer_summary": layer_summary,
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _hybrid_kvrepack_build_simhead_mask(self) -> None:
        if (
            not self.hybridkv_kvrepack
            or not self.hybrid_kvrepack_simhead_enable
            or self.hybrid_kvrepack_merge_mask is not None
        ):
            return
        if any(sim is None for sim in self.hybrid_kvrepack_head_sim):
            return
        sims = torch.stack([sim.float() for sim in self.hybrid_kvrepack_head_sim], dim=1)
        bsz, num_layers, num_heads = sims.shape
        total = num_layers * num_heads
        ratio = min(max(float(self.hybrid_kvrepack_simhead_merge_ratio), 0.0), 1.0)
        merge_count = int(math.ceil(total * ratio)) if ratio > 0 else 0
        merge_count = min(max(merge_count, 0), total)
        mask = torch.zeros_like(sims, dtype=torch.bool)
        if merge_count > 0:
            top = torch.topk(sims.reshape(bsz, total), k=merge_count, largest=True, sorted=False).indices
            for bidx in range(bsz):
                layer_ids = top[bidx] // num_heads
                head_ids = top[bidx] % num_heads
                mask[bidx, layer_ids, head_ids] = True
        self.hybrid_kvrepack_merge_mask = mask.permute(1, 0, 2).contiguous()
        selected = int(self.hybrid_kvrepack_merge_mask.sum().item())
        dynamic_mask = torch.zeros(
            (num_layers, num_heads),
            device=self.hybrid_kvrepack_merge_mask.device,
            dtype=torch.bool,
        )
        for layer_idx, retrieval_heads in enumerate(self.retrieval_head_mapping):
            if retrieval_heads:
                dynamic_mask[layer_idx, retrieval_heads] = True
        dynamic_mask = dynamic_mask[:, None, :].expand_as(self.hybrid_kvrepack_merge_mask)
        dynamic_total = int(dynamic_mask.sum().item())
        dynamic_selected = int((self.hybrid_kvrepack_merge_mask & dynamic_mask).sum().item())
        static_total = int(self.hybrid_kvrepack_merge_mask.numel()) - dynamic_total
        static_selected = selected - dynamic_selected
        print(
            "[Hybrid-KVRepack-SimHead] "
            f"global merge heads={selected}/{self.hybrid_kvrepack_merge_mask.numel()} "
            f"dynamic_merge={dynamic_selected}/{dynamic_total} "
            f"static_merge={static_selected}/{static_total} "
            f"target_ratio={ratio:.4f} "
            f"sim_min/mean/max={float(sims.min().item()):.4f}/"
            f"{float(sims.mean().item()):.4f}/{float(sims.max().item()):.4f}"
        )
        self._hybrid_kvrepack_dump_merge_info(sims, ratio)

    def _hybrid_kvrepack_should_merge_head(self, layer_idx: int, batch_idx: int, head_idx: int) -> bool:
        if not self.hybridkv_kvrepack:
            return False
        if not self.hybrid_kvrepack_simhead_enable:
            return True
        self._hybrid_kvrepack_build_simhead_mask()
        if self.hybrid_kvrepack_merge_mask is None:
            return True
        return bool(self.hybrid_kvrepack_merge_mask[layer_idx, batch_idx, head_idx].item())

    def _hybrid_kvrepack_history_importance(
        self,
        layer_idx: int,
        key_states_offload: torch.Tensor,
        query_states: torch.Tensor,
        history_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        bsz, num_retrieval_head, _, _ = key_states_offload.shape
        local_len = seq_len - history_len
        query = query_states.view(
            bsz,
            self.num_key_value_heads,
            self.num_key_value_groups,
            seq_len,
            self.head_dim,
        )[:, self.retrieval_head_mapping[layer_idx], :, -local_len:, :]

        scores = torch.einsum(
            "bhgld,bhtd->bhglt",
            query.float(),
            key_states_offload.float(),
        ) / math.sqrt(self.head_dim)

        if local_len > 0:
            causal_mask = torch.triu(
                torch.ones(local_len, local_len, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores[..., history_len:] = scores[..., history_len:].masked_fill(
                causal_mask.view(1, 1, 1, local_len, local_len),
                float("-inf"),
            )

        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        return attn[..., :history_len].sum(dim=(2, 3))

    def _hybrid_kvrepack_merge_one_sequence(
        self,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        importance: torch.Tensor,
        target_len: int,
        positions: Optional[torch.Tensor] = None,
        raw_k_seq: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
        mrope_section: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = int(k_seq.shape[0])
        if positions is None:
            positions = torch.arange(seq_len, device=k_seq.device, dtype=torch.long)
        else:
            positions = positions.to(device=k_seq.device, dtype=torch.long)

        use_raw_reapply = (
            self.hybrid_kvrepack_k_merge_space == "raw_reapply"
            and raw_k_seq is not None
            and cos is not None
            and sin is not None
        )
        target_len = self._hybrid_kvrepack_effective_target_len(seq_len, target_len)

        if target_len >= seq_len:
            group_size = torch.ones((seq_len,), device=k_seq.device, dtype=torch.float32)
            return k_seq.contiguous(), v_seq.contiguous(), group_size

        importance = torch.nan_to_num(
            importance.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        left_imp = importance[:-1]
        right_imp = importance[1:]
        pair_sum = left_imp + right_imp
        pair_diff = torch.abs(right_imp - left_imp) / (pair_sum + 1e-6)
        merge_score = pair_sum * (1.0 + self.hybrid_kvrepack_k_diff_alpha * pair_diff)

        n_to_merge = min(max(seq_len - target_len, 0), seq_len - 1)
        merge_idx = self._hybrid_kvrepack_select_merge_indices(
            merge_score,
            n_to_merge,
            seq_len,
        )

        delete_mask = torch.zeros((seq_len,), device=k_seq.device, dtype=torch.bool)
        delete_mask[merge_idx + 1] = True
        start_mask = ~delete_mask
        start_mask[0] = True
        group_ids = torch.cumsum(start_mask.to(torch.long), dim=0) - 1
        group_count = int(group_ids[-1].item()) + 1

        ones = torch.ones((seq_len,), device=k_seq.device, dtype=torch.float32)
        group_size = torch.zeros((group_count,), device=k_seq.device, dtype=torch.float32)
        group_size.index_add_(0, group_ids, ones)

        denom = group_size.clamp_min(1.0).unsqueeze(-1)
        v_sum = torch.zeros((group_count, self.head_dim), device=v_seq.device, dtype=torch.float32)
        v_sum.index_add_(0, group_ids, v_seq.float())
        v_comp = (v_sum / denom).to(v_seq.dtype)

        if self.hybrid_kvrepack_k_merge_space == "raw_reapply" and not use_raw_reapply:
            raise ValueError(
                "Hybrid-KVRepack raw_reapply K merge requires raw_key_states, "
                "cos, and sin."
            )

        if use_raw_reapply:
            raw_k_seq = raw_k_seq.to(device=k_seq.device)
            raw_sum = torch.zeros((group_count, self.head_dim), device=k_seq.device, dtype=torch.float32)
            raw_sum.index_add_(0, group_ids, raw_k_seq.float())
            raw_comp = (raw_sum / denom).to(raw_k_seq.dtype)
            anchor_pos = positions[start_mask]
            cos_src = cos if cos.device == k_seq.device else cos.to(k_seq.device)
            sin_src = sin if sin.device == k_seq.device else sin.to(k_seq.device)
            pos_for_emb = anchor_pos.to(cos_src.device)
            cos_sel = _select_mrope_batch(cos_src, int(batch_idx), pos_for_emb).to(raw_comp.device)
            sin_sel = _select_mrope_batch(sin_src, int(batch_idx), pos_for_emb).to(raw_comp.device)
            k_comp = _apply_mrope_to_merged_keys(
                raw_comp,
                cos_sel,
                sin_sel,
                mrope_section,
            ).to(k_seq.dtype)
        else:
            k_sum = torch.zeros((group_count, self.head_dim), device=k_seq.device, dtype=torch.float32)
            k_sum.index_add_(0, group_ids, k_seq.float())
            k_comp = (k_sum / denom).to(k_seq.dtype)

        return k_comp, v_comp, group_size

    def _hybrid_kvrepack_compress_history(
        self,
        layer_idx: int,
        key_states_offload: torch.Tensor,
        value_states_offload: torch.Tensor,
        query_states: torch.Tensor,
        history_len: int,
        seq_len: int,
        raw_key_states_offload: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mrope_section: Optional[List[int]] = None,
        importance: Optional[torch.Tensor] = None,
        merge_head_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.hybrid_kvrepack_k_merge_space == "raw_reapply"
            and (raw_key_states_offload is None or cos is None or sin is None)
        ):
            raise ValueError(
                "Hybrid-KVRepack raw_reapply merge requires raw_key_states, "
                "cos, and sin from the Qwen attention layer."
            )

        bsz, num_retrieval_head, _, _ = key_states_offload.shape
        if importance is None:
            importance = self._hybrid_kvrepack_history_importance(
                layer_idx=layer_idx,
                key_states_offload=key_states_offload,
                query_states=query_states,
                history_len=history_len,
                seq_len=seq_len,
            )
        else:
            importance = importance.to(device=key_states_offload.device)
        if merge_head_mask is not None:
            merge_head_mask = merge_head_mask.to(device=key_states_offload.device, dtype=torch.bool)

        target_len = self._hybrid_kvrepack_target_history_len(history_len)
        k_budgets = torch.full(
            (bsz, num_retrieval_head),
            target_len,
            device=key_states_offload.device,
            dtype=torch.long,
        )

        if merge_head_mask is not None and not bool(merge_head_mask.all().item()):
            target_len = history_len

        compressed_k = torch.empty(
            (bsz, num_retrieval_head, target_len, self.head_dim),
            device=key_states_offload.device,
            dtype=key_states_offload.dtype,
        )
        compressed_v = torch.empty(
            (bsz, num_retrieval_head, target_len, self.head_dim),
            device=value_states_offload.device,
            dtype=value_states_offload.dtype,
        )
        group_size = torch.empty(
            (bsz, num_retrieval_head, target_len),
            device=key_states_offload.device,
            dtype=torch.float32,
        )
        compressed_k.zero_()
        compressed_v.zero_()
        group_size.zero_()
        valid_lengths = torch.zeros(
            (bsz, num_retrieval_head),
            device=key_states_offload.device,
            dtype=torch.long,
        )

        history_k = key_states_offload[:, :, :history_len, :]
        history_raw_k = (
            raw_key_states_offload[:, :, :history_len, :]
            if raw_key_states_offload is not None
            else None
        )
        history_v = value_states_offload[:, :, :history_len, :]
        history_positions = torch.arange(history_len, device=key_states_offload.device, dtype=torch.long)
        for bidx in range(bsz):
            for hidx in range(num_retrieval_head):
                do_merge = True
                if merge_head_mask is not None:
                    do_merge = do_merge and bool(merge_head_mask[bidx, hidx].item())
                head_k_budget = int(k_budgets[bidx, hidx].item())
                k_seq = history_k[bidx, hidx]
                raw_k_seq = history_raw_k[bidx, hidx] if history_raw_k is not None else None
                v_seq = history_v[bidx, hidx]
                imp_seq = importance[bidx, hidx]
                pos_seq = history_positions

                if not do_merge:
                    head_target_len = int(k_seq.shape[0])
                else:
                    head_target_len = head_k_budget

                k_comp, v_comp, sizes = self._hybrid_kvrepack_merge_one_sequence(
                    k_seq,
                    v_seq,
                    imp_seq,
                    head_target_len,
                    positions=pos_seq,
                    raw_k_seq=raw_k_seq,
                    cos=cos,
                    sin=sin,
                    batch_idx=bidx,
                    mrope_section=mrope_section,
                )
                out_len = int(k_comp.shape[0])
                valid_lengths[bidx, hidx] = out_len
                compressed_k[bidx, hidx, :out_len].copy_(k_comp)
                compressed_v[bidx, hidx, :out_len].copy_(v_comp)
                group_size[bidx, hidx, :out_len].copy_(sizes)

        return compressed_k, compressed_v, group_size, valid_lengths

    def build_index(
        self,
        layer_idx: int,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        query_states: Optional[torch.Tensor] = None,
        raw_key_states: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mrope_section: Optional[List[int]] = None,
        **kwargs,
    ):
        if key_states is None:
            key_states = raw_key_states
            raw_key_states = None
        if value_states is None:
            value_states = kwargs.pop("raw_value_states", None)
        if key_states is None or value_states is None or query_states is None:
            raise ValueError("build_index needs key_states, value_states, and query_states.")
        bsz, _, seq_len, _ = key_states.shape
        self.prefill = seq_len
        self.chunks = seq_len // self.chunk_size - self.local_chunk
        self.local_len = seq_len - self.chunks * self.chunk_size
        self.layers[layer_idx].self_attn.kv_cluster.local_len = self.local_len
        history_len = self.chunks * self.chunk_size
        self._hybrid_kvrepack_record_head_sim(layer_idx, key_states, raw_key_states, history_len)
        
        num_retrieval_head = len(self.retrieval_head_mapping[layer_idx])
        if num_retrieval_head == 0:
            self.layer_chunks[layer_idx] = self.chunks
            return

        batch_ids = torch.arange(bsz, device=self.device, dtype=torch.long)
        retrieval_heads = torch.tensor(
            self.retrieval_head_mapping[layer_idx],
            device=self.device,
            dtype=torch.long,
        )
        self.retrieval_flat_head_ids[layer_idx] = (
            batch_ids[:, None] * self.num_key_value_heads + retrieval_heads[None, :]
        ).reshape(-1)

        # fill offload kv cache
        # [bsz, retrieval_heads, seq_len, head_dim]
        key_states_offload = key_states[:, self.retrieval_head_mapping[layer_idx], :, :]
        raw_key_states_offload = (
            raw_key_states[:, self.retrieval_head_mapping[layer_idx], :, :]
            if raw_key_states is not None
            else None
        )
        value_states_offload = value_states[:, self.retrieval_head_mapping[layer_idx], :, :]

        # build block-wise index
        if self.hybridkv_kvrepack:
            importance = self._hybrid_kvrepack_history_importance(
                layer_idx=layer_idx,
                key_states_offload=key_states_offload,
                query_states=query_states,
                history_len=history_len,
                seq_len=seq_len,
            )
            self.hybrid_kvrepack_retrieval_records[layer_idx] = {
                "key_states_offload": key_states_offload.detach(),
                "value_states_offload": value_states_offload.detach(),
                "raw_key_states_offload": raw_key_states_offload.detach() if raw_key_states_offload is not None else None,
                "importance": importance.detach(),
                "history_len": int(history_len),
                "seq_len": int(seq_len),
                "cos": cos.detach() if cos is not None else None,
                "sin": sin.detach() if sin is not None else None,
                "mrope_section": mrope_section,
            }
            if self.hybrid_kvrepack_simhead_enable:
                self.layer_chunks[layer_idx] = self.chunks
                self.block_meta_idx[layer_idx] = None
                return

            compressed_k, compressed_v, group_size, valid_lengths = self._hybrid_kvrepack_compress_history(
                layer_idx=layer_idx,
                key_states_offload=key_states_offload,
                value_states_offload=value_states_offload,
                query_states=query_states,
                history_len=history_len,
                seq_len=seq_len,
                raw_key_states_offload=raw_key_states_offload,
                cos=cos,
                sin=sin,
                mrope_section=mrope_section,
                importance=importance,
            )
            compressed_history_len = compressed_k.shape[-2]
            self.chunks = compressed_history_len // self.chunk_size
            self.layer_chunks[layer_idx] = self.chunks

            local_k = key_states_offload[:, :, history_len:, :].contiguous()
            local_v = value_states_offload[:, :, history_len:, :].contiguous()
            local_group_size = torch.ones(
                (bsz, num_retrieval_head, local_k.shape[-2]),
                device=group_size.device,
                dtype=torch.float32,
            )

            self.offload_k_cache[layer_idx] = torch.cat([compressed_k, local_k], dim=2).contiguous()
            self.offload_v_cache[layer_idx] = torch.cat([compressed_v, local_v], dim=2).contiguous()
            self.offload_k_local_cache[layer_idx] = local_k

            key_states_indexed = compressed_k.view(
                bsz, num_retrieval_head, self.chunks, self.chunk_size, self.head_dim
            )
            group_indexed = group_size.view(
                bsz, num_retrieval_head, self.chunks, self.chunk_size
            ).float()
            block_meta_all = self._hybrid_kvrepack_chunk_meta(key_states_indexed, group_indexed)
            self.block_meta[layer_idx] = block_meta_all.to(key_states_offload.dtype)
            self.block_valid_mask[layer_idx] = group_indexed.sum(dim=-1) > 0
            if self.hybrid_kvrepack_verbose:
                print(
                    "[Hybrid-KVRepack] "
                    f"layer={layer_idx} history={history_len} "
                    f"compressed={compressed_history_len} chunks={self.chunks} "
                    f"valid_len_min/mean/max="
                    f"{int(valid_lengths.min().item())}/"
                    f"{float(valid_lengths.float().mean().item()):.1f}/"
                    f"{int(valid_lengths.max().item())} "
                    f"alpha={self.hybrid_kvrepack_k_diff_alpha}"
                )
        else:
            self.offload_v_cache[layer_idx] = torch.zeros_like(value_states_offload,
                                                            #    device="cpu",
                                                               device=self.device,
                                                               dtype=value_states_offload.dtype,
                                                            #    pin_memory=True
                                                               )
            self.offload_v_cache[layer_idx].copy_(value_states_offload)
            key_states_indexed = key_states_offload[:,:,:history_len].view(
                bsz, num_retrieval_head, self.chunks, self.chunk_size, self.head_dim
            )
            block_meta_all = key_states_indexed.mean(dim=-2) # [bsz, retrieval_heads, chunks, head_dim]
            # TODO: move to GPU after prefill, maybe avoid OOM? same with ShadowKV
            self.block_meta[layer_idx] = block_meta_all # [bsz, retrieval_heads, chunks, head_dim]
            self.block_valid_mask[layer_idx] = torch.ones(
                (bsz, num_retrieval_head, self.chunks),
                device=self.device,
                dtype=torch.bool,
            )
            self.offload_k_cache[layer_idx] = torch.zeros_like(key_states_offload,
                                                            #    device="cpu",
                                                               device=self.device,
                                                               dtype=key_states_offload.dtype,
                                                            #    pin_memory=True
                                                               )
            self.offload_k_cache[layer_idx].copy_(key_states_offload)
            self.offload_k_local_cache[layer_idx] = None
            self.layer_chunks[layer_idx] = self.chunks

        self.block_meta_idx[layer_idx] = None

    def _hybrid_kvrepack_store_retrieval_index(
        self,
        layer_idx: int,
        compressed_k: torch.Tensor,
        compressed_v: torch.Tensor,
        group_size: torch.Tensor,
        valid_lengths: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
    ) -> None:
        bsz, num_retrieval_head, compressed_history_len, _ = compressed_k.shape
        self.layer_chunks[layer_idx] = compressed_history_len // self.chunk_size
        self.offload_k_cache[layer_idx] = torch.cat([compressed_k, local_k], dim=2).contiguous()
        self.offload_v_cache[layer_idx] = torch.cat([compressed_v, local_v], dim=2).contiguous()
        self.offload_k_local_cache[layer_idx] = local_k

        key_states_indexed = compressed_k.view(
            bsz, num_retrieval_head, self.layer_chunks[layer_idx], self.chunk_size, self.head_dim
        )
        group_indexed = group_size.view(
            bsz, num_retrieval_head, self.layer_chunks[layer_idx], self.chunk_size
        ).float()
        block_meta_all = self._hybrid_kvrepack_chunk_meta(key_states_indexed, group_indexed)
        self.block_meta[layer_idx] = block_meta_all.to(compressed_k.dtype)
        self.block_valid_mask[layer_idx] = group_indexed.sum(dim=-1) > 0
        self.block_meta_idx[layer_idx] = None
        if self.hybrid_kvrepack_verbose:
            print(
                "[Hybrid-KVRepack] "
                f"layer={layer_idx} history={int(valid_lengths.max().item()) if valid_lengths.numel() else 0} "
                f"compressed={compressed_history_len} chunks={self.layer_chunks[layer_idx]} "
                f"valid_len_min/mean/max="
                f"{int(valid_lengths.min().item())}/"
                f"{float(valid_lengths.float().mean().item()):.1f}/"
                f"{int(valid_lengths.max().item())} "
                f"alpha={self.hybrid_kvrepack_k_diff_alpha}"
            )

    def finalize_hybrid_kvrepack_simhead(self) -> None:
        if (
            not self.hybridkv_kvrepack
            or not self.hybrid_kvrepack_simhead_enable
            or self.hybrid_kvrepack_simhead_finalized
        ):
            return
        self._hybrid_kvrepack_update_pruned_head_sim_from_candidates()
        self._hybrid_kvrepack_build_simhead_mask()
        if self.hybrid_kvrepack_merge_mask is None:
            return

        for layer_idx, record in enumerate(self.hybrid_kvrepack_retrieval_records):
            if record is None or len(self.retrieval_head_mapping[layer_idx]) == 0:
                continue
            retrieval_heads = self.retrieval_head_mapping[layer_idx]
            merge_mask = self.hybrid_kvrepack_merge_mask[layer_idx, :, retrieval_heads]
            compressed_k, compressed_v, group_size, valid_lengths = self._hybrid_kvrepack_compress_history(
                layer_idx=layer_idx,
                key_states_offload=record["key_states_offload"],
                value_states_offload=record["value_states_offload"],
                query_states=None,
                history_len=record["history_len"],
                seq_len=record["seq_len"],
                raw_key_states_offload=record["raw_key_states_offload"],
                cos=record["cos"],
                sin=record["sin"],
                mrope_section=record["mrope_section"],
                importance=record["importance"],
                merge_head_mask=merge_mask,
            )
            local_k = record["key_states_offload"][:, :, record["history_len"] :, :].contiguous()
            local_v = record["value_states_offload"][:, :, record["history_len"] :, :].contiguous()
            self._hybrid_kvrepack_store_retrieval_index(
                layer_idx,
                compressed_k,
                compressed_v,
                group_size,
                valid_lengths,
                local_k,
                local_v,
            )

        self._hybrid_kvrepack_prepare_pruned_caches()
        self.hybrid_kvrepack_simhead_finalized = True

    def _hybrid_kvrepack_prepare_pruned_caches(self) -> None:
        if self.hybrid_kvrepack_merge_mask is None or self.sparse_budget_per_head is None:
            return
        for layer_idx in range(self.num_layers):
            pruned_heads = self.pruned_head_mapping[layer_idx]
            if not pruned_heads or self.pruned_k_cache[layer_idx] is None:
                continue
            importance = self.pruned_importance_cache[layer_idx]
            rope_history = self.pruned_rope_history_cache[layer_idx]
            value_history = self.pruned_value_history_cache[layer_idx]
            if importance is None or rope_history is None or value_history is None:
                continue
            raw_history = self.pruned_raw_history_cache[layer_idx]
            mrope = self.pruned_mrope_cache[layer_idx]
            cos, sin, mrope_section = mrope if mrope is not None else (None, None, None)

            out_k = self.pruned_k_cache[layer_idx].clone()
            out_v = self.pruned_v_cache[layer_idx].clone()
            out_g = torch.ones(
                out_k.shape[:-1],
                device=out_k.device,
                dtype=torch.float32,
            )
            bsz, num_pruned, history_len, _ = rope_history.shape
            local_len = self.local_len
            for bidx in range(bsz):
                for local_hidx, head_idx in enumerate(pruned_heads):
                    if not self._hybrid_kvrepack_should_merge_head(layer_idx, bidx, head_idx):
                        continue
                    budget = int(self.sparse_budget_per_head[layer_idx][bidx][head_idx])
                    budget = min(max(budget, 0), history_len)
                    if budget <= 0 or budget >= history_len:
                        continue

                    imp = importance[bidx, local_hidx].float()
                    text_k = rope_history.new_empty((0, self.head_dim))
                    text_v = value_history.new_empty((0, self.head_dim))
                    text_g = torch.empty((0,), device=rope_history.device, dtype=torch.float32)
                    target_len = budget

                    if (
                        self.hybrid_kvrepack_static_merge_scope == "visual"
                        and self.is_vision_token_mask is not None
                    ):
                        vision_mask = self.is_vision_token_mask[bidx, :history_len].to(
                            device=rope_history.device,
                            dtype=torch.bool,
                        )
                        base_idx = torch.topk(
                            imp,
                            k=budget,
                            dim=0,
                            largest=True,
                            sorted=True,
                        ).indices
                        base_text_mask = ~vision_mask.index_select(0, base_idx)
                        text_idx = base_idx[base_text_mask]
                        if int(text_idx.numel()) > 0:
                            text_k = rope_history[bidx, local_hidx, text_idx]
                            text_v = value_history[bidx, local_hidx, text_idx]
                            text_g = torch.ones(
                                (int(text_idx.numel()),),
                                device=rope_history.device,
                                dtype=torch.float32,
                            )
                        target_len = budget - int(text_idx.numel())
                        visual_pool = torch.nonzero(vision_mask, as_tuple=False).view(-1)
                        if target_len <= 0 or int(visual_pool.numel()) == 0:
                            continue
                        cand_len = self._hybrid_kvrepack_candidate_len(
                            target_len,
                            int(visual_pool.numel()),
                        )
                        if cand_len <= target_len:
                            continue
                        visual_scores = imp.index_select(0, visual_pool)
                        local_idx = torch.topk(
                            visual_scores,
                            k=cand_len,
                            dim=0,
                            largest=True,
                            sorted=False,
                        ).indices
                        keep_idx = visual_pool.index_select(0, local_idx)
                        keep_idx, _ = torch.sort(keep_idx)
                    else:
                        cand_len = self._hybrid_kvrepack_candidate_len(budget, history_len)
                        if cand_len <= budget:
                            continue
                        keep_idx = torch.topk(imp, k=cand_len, dim=0, largest=True, sorted=False).indices
                        keep_idx, _ = torch.sort(keep_idx)

                    k_seq = rope_history[bidx, local_hidx, keep_idx]
                    v_seq = value_history[bidx, local_hidx, keep_idx]
                    raw_seq = raw_history[bidx, local_hidx, keep_idx] if raw_history is not None else None
                    imp_seq = imp[keep_idx]

                    k_comp, v_comp, sizes = self._hybrid_kvrepack_merge_one_sequence(
                        k_seq,
                        v_seq,
                        imp_seq,
                        target_len,
                        positions=keep_idx,
                        raw_k_seq=raw_seq,
                        cos=cos,
                        sin=sin,
                        batch_idx=bidx,
                        mrope_section=mrope_section,
                    )
                    if int(k_comp.shape[0]) != target_len:
                        continue
                    if int(text_k.shape[0]) > 0:
                        k_comp = torch.cat([text_k, k_comp], dim=0)
                        v_comp = torch.cat([text_v, v_comp], dim=0)
                        sizes = torch.cat([text_g, sizes], dim=0)
                    if int(k_comp.shape[0]) != budget:
                        continue
                    out_k[bidx, local_hidx, :budget].copy_(k_comp)
                    out_v[bidx, local_hidx, :budget].copy_(v_comp)
                    # No log/count correction: static group sizes are intentionally not recorded.
                    out_g[bidx, local_hidx, -local_len:].fill_(1.0)

            self.pruned_k_cache[layer_idx] = out_k
            self.pruned_v_cache[layer_idx] = out_v
            self.pruned_group_size_cache[layer_idx] = out_g

    def prune_kv(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, query_states: torch.Tensor):
        num_pruned_head = len(self.pruned_head_mapping[layer_idx])
        if num_pruned_head == 0:
            if self.verbose:
                print("No pruning needed for layer {}".format(layer_idx))
            return
        bsz, _, seq_len, _ = key_states.shape
        # pruning based on observation window?
        # TODO: modify observation window to all text tokens
        local_len = self.local_len
        history_len = seq_len - local_len
        key_states_sub = key_states[:, self.pruned_head_mapping[layer_idx], :, :]
        key_states_local = key_states_sub[:, :, -local_len:, :]
        key_states_pruned = key_states_sub[:, :, :seq_len-local_len, :]
        value_states_local = value_states[:, self.pruned_head_mapping[layer_idx], -local_len:, :]
        value_states_pruned = value_states[:, self.pruned_head_mapping[layer_idx], :seq_len-local_len, :]
        query_states = query_states.view(bsz, self.num_key_value_heads, self.num_key_value_groups, -1, self.head_dim)
        local_query_states = query_states[:, self.pruned_head_mapping[layer_idx], :, -local_len:, :].reshape(
            bsz, num_pruned_head * self.num_key_value_groups, local_len, self.head_dim
        )
        key = key_states_sub.repeat_interleave(local_query_states.size(-3)//key_states_sub.size(-3), -3)
        local_attn_score = local_query_states @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # causal mask
        attn_mask = torch.triu(
            torch.ones(local_len, local_len, dtype=torch.bool, device=local_attn_score.device),
            diagonal=1
        )
        local_attn_score[:, :, :, -local_len:] = local_attn_score[:, :, :, -local_len:].masked_fill(attn_mask, float("-inf"))
        # [bsz, pruned_heads * kv_group, local_len, seq_len - local_len]
        local_attn_score = nn.functional.softmax(local_attn_score, dim=-1)
        local_attn_score = local_attn_score[:, :, :, :seq_len - local_len]
        local_attn_score = local_attn_score.view(bsz, num_pruned_head, 
                                                           self.num_key_value_groups, local_len, seq_len - local_len)
        # [bsz, pruned_heads, seq_len - local_len]
        local_attn_score = local_attn_score.sum(2).mean(-2)
        # [bsz, pruned_heads, sparse_budget]
        budget = history_len # TODO: reduce number of tokens to preserve
        local_topk_indices = torch.topk(local_attn_score, k=budget, dim=-1).indices
        # append local kv to the last
        self.pruned_k_cache[layer_idx] = torch.cat([
            torch.gather(key_states_pruned, dim=2, index=local_topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)),
            key_states_local
        ], dim=2)
        self.pruned_v_cache[layer_idx] = torch.cat([
            torch.gather(value_states_pruned, dim=2, index=local_topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)),
            value_states_local
        ], dim=2)

    def prune_kv_by_boosting_text_score_final(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        raw_key_states: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mrope_section: Optional[List[int]] = None,
    ):
        """
        通过提升文本token的注意力分数来优先保留它们，然后从剩余预算中选择最重要的视频token。
        提升的量值为当前注意力分数集内的最大值，以保证数值稳定性和文本内部的顺序。
        """
        pruned_head_indices = self.pruned_head_mapping[layer_idx]
        num_pruned_head = len(pruned_head_indices)
        
        if num_pruned_head == 0:
            # 如果没有需要剪枝的头，直接返回
            return
        
        bsz, _, seq_len, _ = key_states.shape
        local_len = self.local_len
        history_len = seq_len - local_len

        if history_len <= 0:
            return

        key_states_sub = key_states[:, pruned_head_indices, :, :]
        value_states_sub = value_states[:, pruned_head_indices, :, :]
        
        query_states_view = query_states.view(bsz, self.num_key_value_heads, self.num_key_value_groups, -1, self.head_dim)

        local_query_states = query_states_view[:, pruned_head_indices, :, -local_len:, :].reshape(
            bsz, num_pruned_head * self.num_key_value_groups, local_len, self.head_dim
        )

        key_for_attn = key_states_sub.repeat_interleave(self.num_key_value_groups, dim=1)
        local_attn_score = local_query_states @ key_for_attn.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn_mask = torch.triu(
            torch.ones(local_len, local_len, dtype=torch.bool, device=local_attn_score.device),
            diagonal=1
        )
        local_attn_score[:, :, :, -local_len:] = local_attn_score[:, :, :, -local_len:].masked_fill(attn_mask, float("-inf"))
        local_attn_score = F.softmax(local_attn_score, dim=-1, dtype=torch.float32)
        local_attn_score_history = local_attn_score[:, :, :, :history_len]
        
        # 将GQA/MQA的分数聚合到每个KV头
        local_attn_score_history = local_attn_score_history.view(
            bsz, num_pruned_head, self.num_key_value_groups, local_len, history_len
        )

        # 对每个历史token，聚合所有观察窗口query的注意力，得到其重要性总分
        # 形状: [bsz, num_pruned_head, history_len]
        head_attn_scores = local_attn_score_history.sum(dim=[2, 3]) # 在GQA group和local_len维度上求和
        history_text_mask = ~self.is_vision_token_mask[: , :history_len] #[bsz, seq_len]
        max_scores, _ = torch.max(head_attn_scores, dim=-1, keepdim=True)
        boost_mask = history_text_mask.unsqueeze(1).expand_as(head_attn_scores)
        boosted_scores = torch.where(
            boost_mask,
            head_attn_scores + max_scores,
            head_attn_scores
        )

        # budget = self.sparse_budget + 2048
        budget = history_len # TODO: reduce number of tokens to preserve
        k = min(budget, history_len)
        topk_indices = torch.topk(boosted_scores, k=k, dim=-1).indices

        key_states_pruned = key_states_sub[:, :, :history_len, :]
        value_states_pruned = value_states_sub[:, :, :history_len, :]
        key_states_local = key_states_sub[:, :, -local_len:, :]
        value_states_local = value_states_sub[:, :, -local_len:, :]
        raw_key_states_pruned = (
            raw_key_states[:, pruned_head_indices, :history_len, :]
            if raw_key_states is not None
            else None
        )
        
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        
        pruned_k = torch.gather(key_states_pruned, dim=2, index=expanded_indices)
        pruned_v = torch.gather(value_states_pruned, dim=2, index=expanded_indices)

        self.pruned_k_cache[layer_idx] = torch.cat([pruned_k, key_states_local], dim=2)
        self.pruned_v_cache[layer_idx] = torch.cat([pruned_v, value_states_local], dim=2)
        self.pruned_group_size_cache[layer_idx] = torch.ones(
            (bsz, num_pruned_head, history_len + local_len),
            device=key_states.device,
            dtype=torch.float32,
        )
        if self.hybridkv_kvrepack:
            self.pruned_importance_cache[layer_idx] = boosted_scores.detach()
            self.pruned_rope_history_cache[layer_idx] = key_states_pruned.detach()
            self.pruned_raw_history_cache[layer_idx] = (
                raw_key_states_pruned.detach() if raw_key_states_pruned is not None else None
            )
            self.pruned_value_history_cache[layer_idx] = value_states_pruned.detach()
            self.pruned_mrope_cache[layer_idx] = (
                cos.detach() if cos is not None else None,
                sin.detach() if sin is not None else None,
                mrope_section,
            )

    def allocate_budget_by_ratio(self, key_states: torch.Tensor):
        bsz, _, seq_len, _ = key_states.shape
        if self.verbose:
            print(
                f"Sequence Length: {seq_len}, vision token number: {self.vision_token_index.shape[1]}, "
                f"vision token start: {self.vision_token_start}"
            )
            print(f"Using Budget Method: {self.budget_method}")
        total_budget_all_layers = self.num_layers * self.num_key_value_heads * self.avg_budget

        sparse_budget_per_head = torch.zeros(
            self.num_layers,
            self.max_batch_size,
            self.num_key_value_heads, device=self.device,
            dtype=torch.long
        )
        sparse_budget_per_head = sparse_budget_per_head.tolist()

        for i in range(self.max_batch_size):
            n_retrieval = 0
            for layer_idx in range(self.num_layers):
                n_retrieval += len(self.retrieval_head_mapping[layer_idx])
            rest_budget = total_budget_all_layers - n_retrieval * self.retrieval_budget
            n_prune = self.num_layers * self.num_key_value_heads - n_retrieval
            if self.verbose:
                print("=====Head Classification Summary=====")
                print(f"Batch {i}: Dynamic Heads: {n_retrieval}, Static Heads: {n_prune}")

            # allocate budget for each pruning head
            if self.budget_method == BudgetAllocMethod.UNIFORM:
                # if n_prune > 0:
                #     budget_per_prune_head = min(rest_budget // n_prune, seq_len - self.local_len)
                # else:
                #     budget_per_prune_head = 0
                avg_budget = self.avg_budget // self.chunk_size * self.chunk_size
                for layer_idx in range(self.num_layers):
                    for head_idx in range(self.num_key_value_heads):
                        if self.hybridkv_kvrepack and head_idx in self.retrieval_head_mapping[layer_idx]:
                            sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget
                        else:
                            sparse_budget_per_head[layer_idx][i][head_idx] = avg_budget
            
            elif self.budget_method == BudgetAllocMethod.HEAD_LEVEL_ONLY:
                budget_per_prune_head = rest_budget // n_prune if n_prune > 0 else 0
                for layer_idx in range(self.num_layers):
                    for head_idx in range(self.num_key_value_heads):
                        if head_idx in self.retrieval_head_mapping[layer_idx]:
                            sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget
                        else:
                            sparse_budget_per_head[layer_idx][i][head_idx] = min(budget_per_prune_head, seq_len - self.local_len)
            
            elif self.budget_method == BudgetAllocMethod.HEAD_ADA:
                # 按照注意力分数比例分配
                fix_ratio = float(self.param_config["budget_fixed_ratio"])  # 可根据实际需求调整
                fix_budget_total = int(rest_budget * fix_ratio)
                ratio_budget_total = rest_budget - fix_budget_total
                fix_budget_per_head = fix_budget_total // n_prune if n_prune > 0 else 0
                prune_head_scores = []
                prune_head_indices = []  # 记录(层索引, 头索引)用于后续赋值

                temp_head_all_layers = []
                for layer_idx in range(self.num_layers):
                    temp_head_all_layers.append(self.cross_attn_head_all_layers[layer_idx].view(-1, self.num_key_value_heads, self.num_key_value_groups))
                    temp_head_all_layers[layer_idx] = temp_head_all_layers[layer_idx].mean(dim=-1)
                    for head_idx in range(self.num_key_value_heads):
                        if head_idx not in self.retrieval_head_mapping[layer_idx]:
                            # 获取当前头的分数（取第i个batch的分数）
                            score = temp_head_all_layers[layer_idx][i][head_idx]
                            prune_head_scores.append(score)
                            prune_head_indices.append((layer_idx, head_idx))

                scores_tensor = torch.tensor(prune_head_scores, device=self.device, dtype=torch.float32)
                if scores_tensor.sum() == 0:
                    # 分数全为0时退化为均匀分配
                    ratio_budget_per_head = ratio_budget_total // n_prune if n_prune > 0 else 0
                else:
                    score_ratios = scores_tensor / scores_tensor.sum()
                    ratio_budget_raw = (score_ratios * ratio_budget_total).long()
                    
                    # 处理整数分配误差（确保总和等于ratio_budget_total）
                    remaining = ratio_budget_total - ratio_budget_raw.sum()
                    if remaining > 0:
                        top_indices = torch.argsort(score_ratios, descending=True)[:remaining]
                        ratio_budget_raw[top_indices] += 1
                    
                    ratio_budget_per_head = ratio_budget_raw.tolist()

                not_allocated_budget = 0
                for idx, (layer_idx, head_idx) in enumerate(prune_head_indices):
                    total_budget = fix_budget_per_head + ratio_budget_per_head[idx]
                    if total_budget > seq_len - self.local_len:
                        sparse_budget_per_head[layer_idx][i][head_idx] = seq_len - self.local_len
                        not_allocated_budget += total_budget - (seq_len - self.local_len)
                    else:
                        sparse_budget_per_head[layer_idx][i][head_idx] = total_budget
                while not_allocated_budget > 0:
                    # 每次找budget最大的head，分配剩余的budget给他，直到buget为0
                    max_budget = -1
                    max_layer_idx = -1
                    max_head_idx = -1

                    # 找到当前budget最大的head
                    for idx, (layer_idx, head_idx) in enumerate(prune_head_indices):
                        current_budget = sparse_budget_per_head[layer_idx][i][head_idx]
                        if current_budget > max_budget and current_budget < seq_len - self.local_len:
                            max_budget = current_budget
                            max_layer_idx = layer_idx
                            max_head_idx = head_idx

                    # 如果找到了符合条件的head
                    if max_layer_idx != -1:
                        can_allocate_budget = seq_len - self.local_len - max_budget
                        sparse_budget_per_head[max_layer_idx][i][max_head_idx] += can_allocate_budget
                        not_allocated_budget -= can_allocate_budget
                    else:
                        # 如果所有head都达到最大限制，退出循环
                        break
                # if still budget left, allocate for dynamic heads
                while not_allocated_budget > 0:
                    new_budget = self.retrieval_budget * 2
                    old_budget = self.retrieval_budget
                    self.retrieval_budget = self.retrieval_budget * 2
                    not_allocated_budget -= old_budget * n_retrieval
                    self.retrieval_budget = new_budget

                for idx, (layer_idx, head_idx) in enumerate(prune_head_indices):
                    total_budget = fix_budget_per_head + ratio_budget_per_head[idx]
                    sparse_budget_per_head[layer_idx][i][head_idx] = min(total_budget, seq_len - self.local_len)
                    
                    # 为检索头分配固定预算
                for layer_idx in range(self.num_layers):
                    for head_idx in self.retrieval_head_mapping[layer_idx]:
                        sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget

            elif self.budget_method == BudgetAllocMethod.SPARSEMM_INIT:
                head_score = load_head_score(self.model_type)
                head_score_list = [np.mean(l[1]) for l in head_score.items()]
                head_score_list = torch.tensor(head_score_list / sum(head_score_list))
                    # GQA support
                self.score = head_score_list.view(self.num_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
                self.score = self.score.sum(dim=-1)

                # 按照注意力分数比例分配
                fix_ratio = float(self.param_config["budget_fixed_ratio"])  # 可根据实际需求调整
                fix_budget_total = int(rest_budget * fix_ratio)
                ratio_budget_total = rest_budget - fix_budget_total
                fix_budget_per_head = fix_budget_total // n_prune # 固定budget分配给每个剪枝头（保底）
                prune_head_scores = []
                prune_head_indices = []  # 记录(层索引, 头索引)用于后续赋值

                for layer_idx in range(self.num_layers):
                    for head_idx in range(self.num_key_value_heads):
                        if head_idx not in self.retrieval_head_mapping[layer_idx]:
                            # 获取当前头的分数（取第i个batch的分数）
                            score = self.score[layer_idx][head_idx]
                            prune_head_scores.append(score)
                            prune_head_indices.append((layer_idx, head_idx))

                scores_tensor = torch.tensor(prune_head_scores, device=self.device, dtype=torch.float32)
                if scores_tensor.sum() == 0:
                    if n_prune > 0:
                        per_head = ratio_budget_total // n_prune
                    else:
                        per_head = 0
                    ratio_budget_per_head = [per_head] * len(prune_head_indices)
                else:
                    score_ratios = scores_tensor / scores_tensor.sum()
                    ratio_budget_raw = (score_ratios * ratio_budget_total).long()
                    
                    # 处理整数分配误差（确保总和等于ratio_budget_total）
                    remaining = ratio_budget_total - ratio_budget_raw.sum()
                    if remaining > 0:
                        top_indices = torch.argsort(score_ratios, descending=True)[:remaining]
                        ratio_budget_raw[top_indices] += 1
                    
                    ratio_budget_per_head = ratio_budget_raw.tolist()

                not_allocated_budget = 0
                for idx, (layer_idx, head_idx) in enumerate(prune_head_indices):
                    total_budget = fix_budget_per_head + ratio_budget_per_head[idx]
                    if total_budget > seq_len - self.local_len:
                        sparse_budget_per_head[layer_idx][i][head_idx] = seq_len - self.local_len
                        not_allocated_budget += total_budget - (seq_len - self.local_len)
                    else:
                        sparse_budget_per_head[layer_idx][i][head_idx] = total_budget
                while not_allocated_budget > 0:
                    # 每次找budget最大的head，分配剩余的budget给他，直到buget为0
                    max_budget = -1
                    max_layer_idx = -1
                    max_head_idx = -1

                    # 找到当前budget最大的head
                    for idx, (layer_idx, head_idx) in enumerate(prune_head_indices):
                        current_budget = sparse_budget_per_head[layer_idx][i][head_idx]
                        if current_budget > max_budget and current_budget < seq_len - self.local_len:
                            max_budget = current_budget
                            max_layer_idx = layer_idx
                            max_head_idx = head_idx

                    # 如果找到了符合条件的head
                    if max_layer_idx != -1:
                        can_allocate_budget = seq_len - self.local_len - max_budget
                        sparse_budget_per_head[max_layer_idx][i][max_head_idx] += can_allocate_budget
                        not_allocated_budget -= can_allocate_budget
                    else:
                        # 如果所有head都达到最大限制，退出循环
                        break
                # if still budget left, allocate for dynamic heads
                while not_allocated_budget > 0:
                    new_budget = self.retrieval_budget * 2
                    old_budget = self.retrieval_budget
                    self.retrieval_budget = self.retrieval_budget * 2
                    not_allocated_budget -= old_budget * n_retrieval
                    self.retrieval_budget = new_budget
                    # 为检索头分配固定预算
                for layer_idx in range(self.num_layers):
                    for head_idx in self.retrieval_head_mapping[layer_idx]:
                        sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget

            elif self.budget_method == BudgetAllocMethod.SPARSEMM_ATTN:
                # 基础参数设置
                fix_ratio = float(self.param_config["budget_fixed_ratio"])  # 可根据实际需求调整
                fix_budget_total = int(rest_budget * fix_ratio)
                ratio_budget_total = rest_budget - fix_budget_total
                fix_budget_per_head = fix_budget_total // n_prune if n_prune > 0 else 0
                prune_head_scores = []
                prune_head_indices = []  # 记录(层索引, 头索引)用于后续赋值

                head_score = load_head_score(self.model_type)
                head_score_list = [np.mean(l[1]) for l in head_score.items()]
                
                # 归一化SPARSEMM_INIT来源的score
                init_score_sum = sum(head_score_list)
                if init_score_sum == 0:
                    # 避免除以零，全部设为均等值
                    head_score_norm = [1.0 / len(head_score_list) for _ in head_score_list]
                else:
                    head_score_norm = [score / init_score_sum for score in head_score_list]
                
                # 处理GQA结构
                init_score = torch.tensor(head_score_norm, device=self.device)
                init_score = init_score.view(
                    self.num_layers, 
                    self.num_attention_heads // self.num_key_value_groups, 
                    self.num_key_value_groups
                )
                init_score = init_score.sum(dim=-1)  # 维度变为 [num_layers, num_key_value_heads]

                temp_head_all_layers = []
                realtime_scores = []
                
                for layer_idx in range(self.num_layers):
                    # 处理原始实时分数
                    # temp = self.cross_attn_head_all_layers[layer_idx].view(-1, self.num_key_value_groups, self.num_key_value_heads)
                    temp = self.cross_attn_head_all_layers[layer_idx].view(-1, self.num_key_value_heads, self.num_key_value_groups)
                    temp_mean = temp.mean(dim=-1)
                    temp_head_all_layers.append(temp_mean)
                    # 收集当前层所有头的分数
                    for head_idx in range(self.num_key_value_heads):
                        if head_idx not in self.retrieval_head_mapping[layer_idx]:
                            realtime_scores.append(temp_mean[i][head_idx].item())
                
                # 归一化HEAD_ADA来源的score
                realtime_sum = sum(realtime_scores)
                if realtime_sum == 0:
                    # 避免除以零，全部设为均等值
                    realtime_norm = [1.0 / len(realtime_scores) for _ in realtime_scores] if realtime_scores else []
                else:
                    realtime_norm = [score / realtime_sum for score in realtime_scores]

                realtime_idx = 0  # 用于追踪realtime_norm的索引
                alpha = 0.9
                for layer_idx in range(self.num_layers):
                    for head_idx in range(self.num_key_value_heads):
                        if head_idx not in self.retrieval_head_mapping[layer_idx]:
                            # 获取两种归一化后的分数
                            init_score_val = init_score[layer_idx][head_idx]
                            realtime_score_val = realtime_norm[realtime_idx]
                            
                            # 求和得到最终分数
                            combined_score = init_score_val + alpha * realtime_score_val
                            prune_head_scores.append(combined_score)
                            prune_head_indices.append((layer_idx, head_idx))
                            
                            realtime_idx += 1

                scores_tensor = torch.tensor(prune_head_scores, device=self.device, dtype=torch.float32)
                
                if scores_tensor.sum() == 0:
                    # 分数全为0时退化为均匀分配
                    per_head = ratio_budget_total // n_prune if n_prune > 0 else 0
                    ratio_budget_per_head = [per_head] * len(prune_head_indices)
                else:
                    # 对合并后的分数再次归一化用于比例分配
                    score_ratios = scores_tensor / scores_tensor.sum()
                    ratio_budget_raw = (score_ratios * ratio_budget_total).long()
                    
                    # 处理整数分配误差
                    remaining = ratio_budget_total - ratio_budget_raw.sum()
                    if remaining > 0:
                        top_indices = torch.argsort(score_ratios, descending=True)[:remaining]
                        ratio_budget_raw[top_indices] += 1
                    
                    ratio_budget_per_head = ratio_budget_raw.tolist()

                # 应用分配结果
                not_allocated_budget = 0
                for idx, (layer_idx, head_idx) in enumerate(prune_head_indices):
                    total_budget = fix_budget_per_head + ratio_budget_per_head[idx]
                    if total_budget > seq_len - self.local_len:
                        sparse_budget_per_head[layer_idx][i][head_idx] = seq_len - self.local_len
                        not_allocated_budget += total_budget - (seq_len - self.local_len)
                    else:
                        sparse_budget_per_head[layer_idx][i][head_idx] = total_budget
                while not_allocated_budget > 0:
                    # 每次找budget最大的head，分配剩余的budget给他，直到buget为0
                    max_budget = -1
                    max_layer_idx = -1
                    max_head_idx = -1

                    # 找到当前budget最大的head
                    for idx, (layer_idx, head_idx) in enumerate(prune_head_indices):
                        current_budget = sparse_budget_per_head[layer_idx][i][head_idx]
                        if current_budget > max_budget and current_budget < seq_len - self.local_len:
                            max_budget = current_budget
                            max_layer_idx = layer_idx
                            max_head_idx = head_idx

                    # 如果找到了符合条件的head
                    if max_layer_idx != -1:
                        can_allocate_budget = seq_len - self.local_len - max_budget
                        sparse_budget_per_head[max_layer_idx][i][max_head_idx] += can_allocate_budget
                        not_allocated_budget -= can_allocate_budget
                    else:
                        # 如果所有head都达到最大限制，退出循环
                        break
                # if still budget left, allocate for dynamic heads
                while not_allocated_budget > 0:
                    new_budget = self.retrieval_budget * 2
                    old_budget = self.retrieval_budget
                    self.retrieval_budget = self.retrieval_budget * 2
                    not_allocated_budget -= old_budget * n_retrieval
                    self.retrieval_budget = new_budget
                # 为检索头分配固定预算
                for layer_idx in range(self.num_layers):
                    for head_idx in self.retrieval_head_mapping[layer_idx]:
                        sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget

        if self.hybridkv_kvrepack:
            for bidx in range(self.max_batch_size):
                for layer_idx in range(self.num_layers):
                    max_dynamic_budget = (
                        self.layer_chunks[layer_idx]
                        if self.layer_chunks[layer_idx] > 0
                        else self.chunks
                    ) * self.chunk_size
                    for head_idx in self.retrieval_head_mapping[layer_idx]:
                        current = sparse_budget_per_head[layer_idx][bidx][head_idx]
                        clipped = min(int(current), int(max_dynamic_budget))
                        clipped = (clipped // self.chunk_size) * self.chunk_size
                        if clipped == 0 and max_dynamic_budget > 0:
                            clipped = min(self.chunk_size, max_dynamic_budget)
                        sparse_budget_per_head[layer_idx][bidx][head_idx] = clipped

        # [num_layers, bsz, kv_heads]
        self.sparse_budget_per_head = sparse_budget_per_head
        sparse_budget_per_head = torch.tensor(sparse_budget_per_head, device=self.device, dtype=torch.long)
        if self.verbose:
            print("======== Global Budget Allocation Finished ========")
            print(sparse_budget_per_head.to(torch.float32).mean(dim=[-2]))
            print("Mean Budget for all layers: ", sparse_budget_per_head.to(torch.float32).mean())

        self.head_category = None
        torch.cuda.empty_cache()

    def _hybrid_kvrepack_top_chunks(self, layer_idx: int, active_budget: int) -> int:
        top_chunks = max(1, int(active_budget) // self.chunk_size)
        layer_chunks = self.layer_chunks[layer_idx] if self.layer_chunks[layer_idx] > 0 else getattr(self, "chunks", 0)
        return min(int(layer_chunks), top_chunks) if layer_chunks else top_chunks

    def _hybrid_kvrepack_position_ids_from_chunks(self, selected_chunks: torch.Tensor) -> torch.Tensor:
        offsets = self.chunk_offsets
        if offsets.device != selected_chunks.device:
            offsets = offsets.to(selected_chunks.device)
        return (
            selected_chunks.unsqueeze(-1) * self.chunk_size
            + offsets.view(1, 1, 1, -1)
        ).view(selected_chunks.shape[0], selected_chunks.shape[1], -1)

    def _hybrid_kvrepack_needs_log_bias(self, layer_idx: int) -> bool:
        return False

    def get_retrieval_position_ids(self, layer_idx: int, query_states: torch.Tensor):
        if len(self.retrieval_head_mapping[layer_idx]) == 0:
            return None
        self.incoming_q_len = query_states.shape[-2]
        # get query with retrieval head
        num_retrieval_heads = len(self.retrieval_head_mapping[layer_idx])
        query = query_states.view(
            self.max_batch_size, self.num_key_value_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim
        )[:, self.retrieval_head_mapping[layer_idx], :, :, :].reshape(
            self.max_batch_size, num_retrieval_heads * self.num_key_value_groups, self.incoming_q_len, self.head_dim
        )

        # BUG: hardcode for retrieval budget
        budget = self.sparse_budget_per_head[layer_idx][0][self.retrieval_head_mapping[layer_idx][0]]
        top_chunks = self._hybrid_kvrepack_top_chunks(layer_idx, budget)
        # TODO: batch_gemm_softmax kernel integration
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query.view(-1, num_retrieval_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), 
                                  self.block_meta[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(self.head_dim)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, retrieval_heads, kv_group, chunks]
        if self.num_key_value_groups > 1: # 对group中的query head取max来检索, 意思是检索最重要的token
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, kv_head, chunks]
        valid_mask = self.block_valid_mask[layer_idx]
        if valid_mask is not None and valid_mask.shape == chunk_attn.shape:
            chunk_attn = chunk_attn.masked_fill(~valid_mask.to(chunk_attn.device), float("-inf"))
        merged_results = torch.topk(chunk_attn, k=top_chunks, dim=-1).indices # [bsz, 8, select_sets(256)]

        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = merged_results # [bsz, retrieval_heads, select_sets]

        # TODO: reorder_cache_and_compute_offsets
        if (
            self.selection_cache[layer_idx] is None
            or tuple(self.selection_cache[layer_idx].shape) != tuple(selected_chunks.shape)
        ):
            self.selection_cache[layer_idx] = torch.zeros_like(selected_chunks)
        self.selection_cache[layer_idx].copy_(selected_chunks, non_blocking=True)

        position_ids = self._hybrid_kvrepack_position_ids_from_chunks(selected_chunks)
        return position_ids

    def get_key_value_cache(self, layer_idx: int, position_ids: torch.Tensor):
        if position_ids is None: # all pruning heads
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # Gather selected retrieval K/V and scatter them into the existing flat HybridKV buffer.
        position_ids = position_ids.to(self.offload_k_cache[layer_idx].device)
        key_ = self.offload_k_cache[layer_idx].gather(
            dim=-2,
            index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )
        value_ = self.offload_v_cache[layer_idx].gather(
            dim=-2,
            index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )
        batch_size, n_retrive, budget, _ = key_.shape
        key_ = key_.view(-1, self.head_dim).to(self.device)
        value_ = value_.view(-1, self.head_dim).to(self.device)

        cu_klen = self.layers[layer_idx].self_attn.kv_cluster.cu_klen
        head_starts = cu_klen[self.retrieval_flat_head_ids[layer_idx]].to(
            device=self.device,
            dtype=torch.long,
        )
        offsets = torch.arange(budget, device=self.device, dtype=torch.long)
        scatter_index = (head_starts[:, None] + offsets[None, :]).reshape(-1)
        if scatter_index.shape[0] != key_.shape[0]:
            raise ValueError(
                f"Shape mismatch: scatter_index has {scatter_index.shape[0]} elements, "
                f"but key_states has {key_.shape[0]}."
            )

        self.key_cache[layer_idx].index_copy_(0, scatter_index, key_)
        self.value_cache[layer_idx].index_copy_(0, scatter_index, value_)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # Retrieve KVCache with query, without kernel optimization
    # only used during prefill
    # TODO: fill cache for the first time
    def retrieve_kv_naive(self, layer_idx: int, last_query: torch.Tensor):
        if last_query is None:
            return None, None
        # should include local KV at last
        bsz, q_head, _, head_dim = last_query.shape
        n_retrieve = q_head // self.num_key_value_groups
        
        block_attn = torch.einsum(
            'bhgd,bhcd->bhgc',
            last_query.view(-1, n_retrieve, self.num_key_value_groups, self.head_dim),
            self.block_meta[layer_idx].to(last_query.device)
        ) / math.sqrt(self.head_dim)
        # [bsz, n_head, kv_group, chunks]
        block_attn = nn.functional.softmax(block_attn, dim=-1, dtype=torch.float32)
        block_attn, _ = torch.max(block_attn, dim=-2) # [bsz, n_head, chunks]
        valid_mask = self.block_valid_mask[layer_idx]
        if valid_mask is not None and valid_mask.shape == block_attn.shape:
            block_attn = block_attn.masked_fill(~valid_mask.to(block_attn.device), float("-inf"))
        # TODO: currently assume the same layer has the same retrieval budget
        active_budget = self.sparse_budget_per_head[layer_idx][0][self.retrieval_head_mapping[layer_idx][0]]
        n_chunks = self._hybrid_kvrepack_top_chunks(layer_idx, active_budget)
        # init selection cache
        self.selection_cache[layer_idx] = torch.zeros(
            bsz, n_retrieve, n_chunks, device=last_query.device, dtype=torch.long
        )
        # [bsz, n_head, top_chunk]
        topk_indices = torch.topk(block_attn, k=n_chunks, dim=-1).indices
        selected_chunks = topk_indices
        self.selection_cache[layer_idx].copy_(selected_chunks, non_blocking=True)
        position_ids = self._hybrid_kvrepack_position_ids_from_chunks(selected_chunks)
        # gather from CPU KVCache
        position_ids = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        device = self.offload_v_cache[layer_idx].device
        key_ = self.offload_k_cache[layer_idx].gather(dim=-2,
                                                        index=position_ids.to(device)
                                                        ).to(last_query.device)
        value_ = self.offload_v_cache[layer_idx].gather(dim=-2,
                                                        index=position_ids.to(device)
                                                        ).to(last_query.device)
        if self.hybridkv_kvrepack:
            local_key_ = self.offload_k_local_cache[layer_idx].to(last_query.device)
        else:
            local_key_ = self.offload_k_cache[layer_idx][:, :, -self.local_len:, :].to(last_query.device)
        local_value_ = self.offload_v_cache[layer_idx][:, :, -self.local_len:, :].to(last_query.device)
        return torch.cat([key_, local_key_], dim=2), torch.cat([value_, local_value_], dim=2)
    
    def clear_prefill_state(self):
        self.pruned_k_cache = [None for _ in range(self.num_layers)]
        self.pruned_v_cache = [None for _ in range(self.num_layers)]
        self.pruned_group_size_cache = [None for _ in range(self.num_layers)]
        self.pruned_importance_cache = [None for _ in range(self.num_layers)]
        self.pruned_rope_history_cache = [None for _ in range(self.num_layers)]
        self.pruned_raw_history_cache = [None for _ in range(self.num_layers)]
        self.pruned_value_history_cache = [None for _ in range(self.num_layers)]
        self.pruned_mrope_cache = [None for _ in range(self.num_layers)]
        self.last_query_temp = [None for _ in range(self.num_layers)]
        self.cross_attn_head_all_layers = [None for _ in range(self.num_layers)]
        self.hybrid_kvrepack_retrieval_records = [None for _ in range(self.num_layers)]
        self.hybrid_kvrepack_head_sim = [None for _ in range(self.num_layers)]
        torch.cuda.empty_cache()

def init_hybridkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET') or 1024)
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = float(os.getenv('RATIO') or os.getenv('BUDGET_RATIO') or 0.1)
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = HybridKV(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            ratio=self.config.ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            model_type=self.config.model_type
        )

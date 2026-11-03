import math
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
from kvrepack.hybridkv_utils import BudgetAllocMethod, HeadClassifyMethod, get_retrieval_budget, largest_power_of_2, nearest_power_of_2
from kvrepack.baseline_utils import load_head_score

import kvrepack_kernel

class HybridKVCache_CPU(Cache):
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

        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
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
        self.dynamic_budget_ratio = 0.75 # retained ratio for budget of dynamic heads
        self.chunk_size = 8
        self.local_len = 32 # local window size
        self.local_chunk = self.local_len // self.chunk_size

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
            self.score_threshold = 0.9
        elif self.head_classify_method == HeadClassifyMethod.DYNAMIC:
            self.score_threshold = 1.0
        self.sparsity_ratio = 0.95 # head calibration

        # budget for retrieval heads
        # self.retrieval_ratio = 0.05 # TODO: 调参, retrieval的budget占比，或调整为固定大小
        if self.head_classify_method == HeadClassifyMethod.HEAD_ADA:
            self.retrieval_budget = get_retrieval_budget(max_cache_len, self.sparse_budget_ratio, self.dynamic_budget_ratio) # retrieval budget
        elif self.head_classify_method == HeadClassifyMethod.DYNAMIC:
            # for fixed budget ratio
            chunks = max_cache_len // self.chunk_size
            retrieval_budget = int(chunks * self.sparse_budget_ratio) * self.chunk_size
            self.retrieval_budget = nearest_power_of_2(retrieval_budget) # for kernel, budget need to be power of 2
        self.select_sets = self.retrieval_budget // self.chunk_size
        # offload kernels
        self.block_num = int(self.max_batch_size * self.num_key_value_heads)
        self.offsets = torch.zeros(self.block_num*(self.retrieval_budget // self.chunk_size), device=self.device, dtype=torch.int32).contiguous()
        self.cnts = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        self.signals = torch.zeros(self.block_num, device=self.device, dtype=torch.int32).contiguous()
        # [num_layers, batch_size, num_retrieval_heads, num_chunks]
        self.position_ids: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.temp_k = torch.zeros(
            self.max_batch_size, 
            self.num_key_value_heads, 
            self.select_sets, 
            self.chunk_size*self.head_dim, 
            device=self.device, 
            dtype=self.dtype
        ).contiguous()
        self.temp_v = torch.zeros(
            self.max_batch_size, 
            self.num_key_value_heads, 
            self.select_sets, 
            self.chunk_size*self.head_dim, 
            device=self.device, 
            dtype=self.dtype
        ).contiguous()
        self.gpu_base_offset = torch.zeros(
            self.num_layers,
            self.max_batch_size * self.num_key_value_heads,
            device=self.device,
            dtype=torch.int64
        ).contiguous()
        self.dynamic_head_indices = torch.zeros(
            self.num_layers,
            self.max_batch_size * self.num_key_value_heads,
            device=self.device,
            dtype=torch.int32
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
        self.budget_method = BudgetAllocMethod.from_string(os.environ.get("BUDGET_METHOD", "head_ada"))
        self.cross_attn_head_all_layers : List[torch.Tensor] = [None for _ in range(self.num_layers)]

        # pruned heads
        self.pruned_head_mapping: List[List[int]] = [[] for _ in range(self.num_layers)]
        # NOTE: only used during prefill
        self.pruned_k_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.pruned_v_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]

        # retrieval heads
        self.retrieval_head_mapping: List[List[int]] = [[] for _ in range(self.num_layers)]
        self.offload_k_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.offload_v_cache: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.block_meta: List[torch.Tensor] = [None for _ in range(self.num_layers)]
        self.block_meta_idx: List[torch.Tensor] = [None for _ in range(self.num_layers)]
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
            cu_klen = self.layers[layer_idx].self_attn.kv_cluster.cu_klen
            for bsz in range(self.max_batch_size):
                for head_idx in self.retrieval_head_mapping[layer_idx]:
                    idx = bsz * self.num_key_value_heads + head_idx
                    head_start_idx = cu_klen[idx]
                    self.gpu_base_offset[layer_idx, idx] = head_start_idx * self.head_dim
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
    
    def update_base_offset(self, layer_idx: int, cu_offset: torch.Tensor):
        from kvrepack_kernel import update_base_offset as update_base_offset_cuda
        update_base_offset_cuda(self.gpu_base_offset[layer_idx], cu_offset, self.dynamic_head_indices[layer_idx], self.head_dim)
        # self.gpu_base_offset[layer_idx, indices] += cu_offset[indices] * self.head_dim
    
    def fill_dynamic_head_indices(self, layer_idx: int):
        indices = []
        n_head = 0
        for bsz in range(self.max_batch_size):
            for head_idx in self.retrieval_head_mapping[layer_idx]:
                idx = bsz * self.num_key_value_heads + head_idx
                n_head += 1
                indices.append(idx)
        self.dynamic_head_indices[layer_idx, :n_head] = torch.tensor(indices, dtype=torch.int32, device=self.device)

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
    
    def save_cdf(self, layer_idx: int, cross_modal_attn: torch.Tensor):
        bsz, q_head, text_len, vision_len = cross_modal_attn.shape
        avg_cross_modal_attn = cross_modal_attn.view(bsz, self.num_key_value_heads,
                                                     self.num_key_value_groups, text_len, vision_len).mean(dim=[2, 3])
        cdf = torch.sort(avg_cross_modal_attn, dim=-1).values
        self.head_cdfs[layer_idx] = cdf

    def calibrate_head(self, layer_idx: int, key_states: torch.Tensor, query_states: torch.Tensor):
        assert query_states.shape[-2] == key_states.shape[-2], "In prefilling, sequence length of query and key should be the same!"
        bsz, _, seq_len, _ = key_states.shape
        vision_token_end = self.vision_token_end[0].item()
        vision_token_start = self.vision_token_start[0].item()
        current_text_len = seq_len - vision_token_end
        
        self.layers[layer_idx].self_attn.kv_cluster.text_len = current_text_len

        # 1. cross-modal attention score
        # TODO: maybe make cross-attention score in a fixed window
        # current_text_len = min(current_text_len, 32)
        # (bsz, q_head, seq_len, head_dim) -> (bsz, q_head, text_len, head_dim)
        text_query_states = query_states[:, :, vision_token_end:vision_token_end+current_text_len:, :]
        # (bsz, kv_head, kv_group, text_len, head_dim) * (bsz, kv_head, head_dim, seq_len) -> (bsz, kv_head, kv_group, text_len, seq_len)
        key = key_states.repeat_interleave(query_states.size(-3)//key_states.size(-3), -3)
        # Fix: expand index tensor to match the key tensor dimensions
        index_expanded = self.vision_token_index.unsqueeze(1).unsqueeze(-1).expand(-1, key.size(1), -1, self.head_dim)
        vision_key_states = torch.gather(key, dim=-2, index=index_expanded)
        # cross-modal
        # text_attn_score = text_query_states @ vision_key_states.transpose(-2, -1) / math.sqrt(self.head_dim)
        # text-guided
        text_attn_score = text_query_states @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # text causal mask
        attn_mask = torch.triu(
            torch.ones(current_text_len, current_text_len, 
                       dtype=torch.bool, device=text_attn_score.device), 
            diagonal=1
        )
        text_attn_score[:, :, :, vision_token_end:] = text_attn_score[:, :, :, vision_token_end:].masked_fill(attn_mask, float("-inf"))
        attn_mask = torch.ones_like(text_attn_score, dtype=torch.bool)
        for i in range(current_text_len):
            # 对于第i个text token，它不能看到位置i之后的text token
            attn_mask[:, :, i, vision_token_end+i+1:] = False
        text_attn_score = text_attn_score.masked_fill(attn_mask == False, -float('inf'))
        # [bsz, q_head, text_len, seq_len]
        text_attn_score = nn.functional.softmax(text_attn_score, dim=-1, dtype=torch.float32).to(self.dtype)

        # 2. classify KV heads with attention score
        vision_len = vision_token_end - vision_token_start
        k = int(vision_len * (1 - self.sparsity_ratio))
        topk_indices = torch.topk(text_attn_score, k=k, dim=-1).indices
        topk_text_score = torch.gather(text_attn_score, dim=-1, index=topk_indices) # [bsz, q_head, text_len, sparse_budget]
        topk_text_score_avg = topk_text_score.sum(-1).mean(-1) # [bsz, q_head]
        self.cross_attn_head_all_layers[layer_idx] = topk_text_score_avg.clone()

        # save cdf
        # self.save_cdf(layer_idx, text_attn_score)

        # TODO: 更复杂的判断方式，目前只是简单根据cross-modal attention score
        query_aware_heads = topk_text_score_avg < self.score_threshold
        query_aware_heads_groups = query_aware_heads.view(
            self.max_batch_size, self.num_key_value_heads, self.num_key_value_groups
        )
        is_query_aware_head = query_aware_heads_groups.any(dim=-1) # [bsz, kv_head]

        self.head_category[layer_idx] = is_query_aware_head
        print(f"Layer {layer_idx} head category: {self.head_category[layer_idx]}")
        # print(f"Layer {layer_idx} - Avg Entropy: {avg_entropy_kv}, Avg Varentropy: {avg_varentropy_kv}")
        print(f"Layer {layer_idx} - Query-aware heads: {is_query_aware_head.sum().item()}/{is_query_aware_head.numel()}")

        # 3. retrieval head / pruned head mapping
        # TODO: currently assume all batch share the same head categories
        head_cat = self.head_category[layer_idx]
        for head_idx in range(self.num_key_value_heads):
            if head_cat[0][head_idx]:
                self.retrieval_head_mapping[layer_idx].append(head_idx)
            else:
                self.pruned_head_mapping[layer_idx].append(head_idx)

        if len(self.retrieval_head_mapping[layer_idx]) > 0:
            self.fill_dynamic_head_indices(layer_idx)

        # store last query states only for retrieval heads
        query = query_states.view(bsz, self.num_key_value_heads, self.num_key_value_groups, -1, self.head_dim)
        if len(self.retrieval_head_mapping[layer_idx]) > 0:
            self.last_query_temp[layer_idx] = query[:, self.retrieval_head_mapping[layer_idx], :, -1, :].reshape(
                bsz, len(self.retrieval_head_mapping[layer_idx]) * self.num_key_value_groups, 1, self.head_dim
            ).clone()

    def build_index(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, query_states: torch.Tensor):
        bsz, _, seq_len, _ = key_states.shape
        self.prefill = seq_len
        self.chunks = seq_len // self.chunk_size - self.local_chunk
        if layer_idx == 0:
            # gemm softmax kernel
            self.gemm_o = torch.zeros(self.max_batch_size, self.num_key_value_heads, self.num_key_value_groups, self.chunks, 
                                      device=self.device, dtype=self.dtype).contiguous()
            self.softmax_o = torch.zeros(self.max_batch_size, self.num_key_value_heads, self.num_key_value_groups, self.chunks, 
                                         device=self.device, dtype=self.dtype).contiguous()
            self.norm = torch.zeros(self.max_batch_size*self.num_key_value_heads, self.num_key_value_groups, (self.chunks + 256 - 1) // 256, 
                                    device=self.device, dtype=torch.float).contiguous()
            self.sum = torch.zeros(self.max_batch_size*self.num_key_value_heads, self.num_key_value_groups, (self.chunks + 256 - 1) // 256, 
                                   device=self.device, dtype=torch.float).contiguous()

        self.local_len = seq_len - self.chunks * self.chunk_size
        self.layers[layer_idx].self_attn.kv_cluster.local_len = self.local_len
        # init selection cache
        # self.position_ids[layer_idx] = torch.zeros(self.max_batch_size, self.num_key_value_heads, self.select_sets,
        #                                            device=self.device, dtype=torch.int64).fill_(-1).contiguous()
        
        num_retrieval_head = len(self.retrieval_head_mapping[layer_idx])
        if num_retrieval_head == 0:
            return

        # fill offload kv cache
        # [bsz, retrieval_heads, seq_len, head_dim]
        key_states_offload = key_states[:, self.retrieval_head_mapping[layer_idx], :, :]
        value_states_offload = value_states[:, self.retrieval_head_mapping[layer_idx], :, :]
        self.offload_k_cache[layer_idx] = torch.zeros_like(key_states_offload, 
                                                           device="cpu",
                                                        #    device=self.device, 
                                                           dtype=key_states_offload.dtype,
                                                           pin_memory=True
                                                           )
        self.offload_v_cache[layer_idx] = torch.zeros_like(value_states_offload, 
                                                           device="cpu", 
                                                            #   device=self.device,
                                                           dtype=value_states_offload.dtype,
                                                           pin_memory=True
                                                           )
        self.offload_k_cache[layer_idx].copy_(key_states_offload)
        self.offload_v_cache[layer_idx].copy_(value_states_offload)

        # build block-wise index
        key_states_indexed = key_states_offload[:,:,:self.chunks*self.chunk_size].view(
            bsz, num_retrieval_head, self.chunks, self.chunk_size, self.head_dim
        )
        block_meta_all = key_states_indexed.mean(dim=-2) # [bsz, retrieval_heads, chunks, head_dim]
        # TODO: move to GPU after prefill, maybe avoid OOM? same with ShadowKV
        self.block_meta[layer_idx] = block_meta_all # [bsz, retrieval_heads, chunks, head_dim]
        all_idx = torch.arange(self.chunks, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_retrieval_head, -1) # [bsz, retrieval_heads, chunks]
        retrieval_head_idx = all_idx # [bsz, retrieval_heads, chunks]
        self.block_meta_idx[layer_idx] = retrieval_head_idx

    def prune_kv(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, query_states: torch.Tensor):
        num_pruned_head = len(self.pruned_head_mapping[layer_idx])
        if num_pruned_head == 0:
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

    def prune_kv_by_boosting_text_score_final(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, query_states: torch.Tensor):
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
        local_attn_score = F.softmax(local_attn_score, dim=-1)
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
        
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        
        pruned_k = torch.gather(key_states_pruned, dim=2, index=expanded_indices)
        pruned_v = torch.gather(value_states_pruned, dim=2, index=expanded_indices)

        self.pruned_k_cache[layer_idx] = torch.cat([pruned_k, key_states_local], dim=2)
        self.pruned_v_cache[layer_idx] = torch.cat([pruned_v, value_states_local], dim=2)

    def allocate_budget_by_ratio(self, key_states: torch.Tensor):
        bsz, _, seq_len, _ = key_states.shape
        print(f"Sequence Length: {seq_len}, vision token number: {self.vision_token_index.shape[1]}, vision token start: {self.vision_token_start}")
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
            print("=====Head Classification Summary=====")
            print(f"Batch {i}: Dynamic Heads: {n_retrieval}, Static Heads: {n_prune}")

            # allocate budget for each pruning head
            if self.budget_method == BudgetAllocMethod.UNIFORM:
                if n_prune > 0:
                    budget_per_prune_head = min(rest_budget // n_prune, seq_len - self.local_len)
                else:
                    budget_per_prune_head = 0
                
                for layer_idx in range(self.num_layers):
                    for head_idx in range(self.num_key_value_heads):
                        if head_idx in self.retrieval_head_mapping[layer_idx]:
                            sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget
                        else:
                            sparse_budget_per_head[layer_idx][i][head_idx] = budget_per_prune_head
            
            elif self.budget_method == BudgetAllocMethod.HEAD_ADA:
                # 按照注意力分数比例分配
                fix_ratio = 0.1
                fix_budget_total = int(rest_budget * fix_ratio)
                ratio_budget_total = rest_budget - fix_budget_total
                fix_budget_per_head = fix_budget_total // n_prune # 固定budget分配给每个剪枝头（保底）
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
                fix_ratio = 0.1
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
                    
                    # 为检索头分配固定预算
                for layer_idx in range(self.num_layers):
                    for head_idx in self.retrieval_head_mapping[layer_idx]:
                        sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget

            elif self.budget_method == BudgetAllocMethod.SPARSEMM_ATTN:
                # 基础参数设置
                fix_ratio = 0.1  # 可根据实际需求调整
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
                # 为检索头分配固定预算
                for layer_idx in range(self.num_layers):
                    for head_idx in self.retrieval_head_mapping[layer_idx]:
                        sparse_budget_per_head[layer_idx][i][head_idx] = self.retrieval_budget

        # [num_layers, bsz, kv_heads]
        self.sparse_budget_per_head = sparse_budget_per_head
        sparse_budget_per_head = torch.tensor(sparse_budget_per_head, device=self.device, dtype=torch.long)
        print("======== Global Budget Allocation Finished ========")
        print(sparse_budget_per_head.to(torch.float32).mean(dim=[-2]))
        print("Mean Budget for all layers: ", sparse_budget_per_head.to(torch.float32).mean())

        self.head_category = None
        torch.cuda.empty_cache()

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
        # query = query.contiguous()
        # gemm_o = self.gemm_o[:, :num_retrieval_heads, ...].contiguous()
        # softmax_o = self.softmax_o[:, :num_retrieval_heads, ...].contiguous()
        # norm = self.norm[:self.max_batch_size * num_retrieval_heads, ...].contiguous()
        # sum = self.sum[:self.max_batch_size * num_retrieval_heads, ...].contiguous()
        # kvrepack_kernel.gemm_softmax(
        #     query, # [b, nhk_retri * group, q_len, d]
        #     self.block_meta[layer_idx].contiguous(), # [b, nhk_retri, chunk, d]
        #     gemm_o,
        #     norm,
        #     sum,
        #     softmax_o,
        #     self.max_batch_size * num_retrieval_heads,
        #     self.num_key_value_groups * self.incoming_q_len,
        #     self.block_meta[layer_idx].shape[-2],
        #     self.head_dim,
        #     1 / math.sqrt(self.head_dim),
        #     0
        # )
    
        chunk_attn = torch.einsum('bhgqd,bhdc->bhgqc', query.view(-1, num_retrieval_heads, self.num_key_value_groups, self.incoming_q_len, self.head_dim), 
                                  self.block_meta[layer_idx].transpose(2, 3)).squeeze(2) / math.sqrt(self.head_dim)
        chunk_attn = nn.functional.softmax(chunk_attn, dim=-1, dtype=torch.float32).to(self.dtype) # [bsz, 8, 4, q_len, chunks]
        chunk_attn = chunk_attn.sum(dim = -2) # [bsz, retrieval_heads, kv_group, chunks]

        # BUG: hardcode for retrieval budget
        budget = self.sparse_budget_per_head[layer_idx][0][self.retrieval_head_mapping[layer_idx][0]]
        top_chunks = budget // self.chunk_size
        # if self.num_key_value_groups > 1: # 对group中的query head取max来检索, 意思是检索最重要的token
        #     chunk_attn, _ = torch.max(softmax_o.view(
        #         self.max_batch_size, num_retrieval_heads, self.num_key_value_groups, -1), dim=-2) # [bsz, retri_head, chunks]
        if self.num_key_value_groups > 1: # 对group中的query head取max来检索, 意思是检索最重要的token
            chunk_attn, _ = torch.max(chunk_attn, dim=-2) # [bsz, kv_head, chunks]
        merged_results = torch.topk(chunk_attn, k=top_chunks, dim=-1).indices # [bsz, 8, select_sets(256)]

        # use merged_results to gather the position_ids: [bsz, 8, select_sets] --> [bsz, 8, select_sets]
        selected_chunks = self.block_meta_idx[layer_idx].gather(dim=-1, index=merged_results) # [bsz, 8, select_sets]

        # TODO: reorder_cache_and_compute_offsets
        if self.retrieval_budget >= 256:
            kvrepack_kernel.reorder_keys_and_compute_offsets(
                self.position_ids[layer_idx],
                selected_chunks,
                self.offsets,
                self.cnts,
                self.max_batch_size,
                num_retrieval_heads,
                self.select_sets
            )
            return self.position_ids[layer_idx]
        else:
            position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=chunk_attn.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(
                self.max_batch_size, num_retrieval_heads, -1
            ) # [bsz, retrieval_heads, select_sets * chunk_size]
            return position_ids

    def get_key_value_cache(self, layer_idx: int, position_ids: torch.Tensor):
        if position_ids is None: # all pruning heads
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            if self.retrieval_budget >= 256:
                num_retrieval_heads = len(self.retrieval_head_mapping[layer_idx])
                kvrepack_kernel.gather_copy_with_offsets(
                    self.offload_k_cache[layer_idx], self.key_cache[layer_idx], self.temp_k,
                    self.offload_v_cache[layer_idx], self.value_cache[layer_idx], self.temp_v,
                    self.offsets, self.cnts, self.signals, self.gpu_base_offset,
                    self.max_batch_size, num_retrieval_heads,
                    int(self.prefill * self.head_dim), # cpu length
                    int(self.retrieval_budget * self.head_dim), # gpu_length
                    0, # gpu_offset
                    int(self.retrieval_budget * self.head_dim), # gpu_stride
                    self.select_sets # number of chunks
                )
            else:
                # gather from CPU KVCache
                position_ids = position_ids.to(self.offload_k_cache[layer_idx].device)
                # [bsz, retrieval_heads, budget, head_dim]
                key_ = self.offload_k_cache[layer_idx].gather(dim=-2,
                                                            index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                                                            )
                value_ = self.offload_v_cache[layer_idx].gather(dim=-2,
                                                            index=position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                                                            )
                batch_size, n_retrive, budget, _ = key_.shape
                key_ = key_.view(-1, self.head_dim).to(self.device)
                value_ = value_.view(-1, self.head_dim).to(self.device)

                # scatter into GPU-side buffer
                cu_klen = self.layers[layer_idx].self_attn.kv_cluster.cu_klen

                scatter_indices = []
                for bsz in range(batch_size):
                    for head_idx in self.retrieval_head_mapping[layer_idx]:
                        idx = bsz * self.num_key_value_heads + head_idx
                        head_start_idx = cu_klen[idx]
                        # if has initial tokens, should add them to offset
                        indices = torch.arange(head_start_idx, head_start_idx + budget, device=self.device, dtype=torch.long)
                        scatter_indices.append(indices)
            
                scatter_index = torch.cat(scatter_indices, dim=0)
                if scatter_index.shape[0] != key_.shape[0]:
                    raise ValueError(f"Shape mismatch: scatter_index has {scatter_index.shape[0]} elements, but value_states has {key_.shape[0]} elements.")

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
        # TODO: currently assume the same layer has the same retrieval budget
        n_chunks = self.sparse_budget_per_head[layer_idx][0][self.retrieval_head_mapping[layer_idx][0]] // self.chunk_size
        # init selection cache
        # [bsz, n_head, top_chunk]
        topk_indices = torch.topk(block_attn, k=n_chunks, dim=-1).indices
        selected_chunks = self.block_meta_idx[layer_idx].to(last_query.device).gather(dim=-1, index=topk_indices)
        self.position_ids[layer_idx] = selected_chunks.to(torch.int64).contiguous()
        position_ids = (selected_chunks.unsqueeze(-1) * self.chunk_size + torch.arange(self.chunk_size, device=block_attn.device).
                        unsqueeze(0).unsqueeze(0).unsqueeze(0)).view(bsz, n_retrieve, -1)
        # gather from CPU KVCache
        position_ids = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        device = self.offload_v_cache[layer_idx].device
        value_ = self.offload_v_cache[layer_idx].gather(dim=-2,
                                                        index=position_ids.to(device)
                                                        ).to(last_query.device)
        key_ = self.offload_k_cache[layer_idx].gather(dim=-2,
                                                        index=position_ids.to(device)
                                                        ).to(last_query.device)
        local_key_ = self.offload_k_cache[layer_idx][:, :, -self.local_len:, :].to(last_query.device)
        local_value_ = self.offload_v_cache[layer_idx][:, :, -self.local_len:, :].to(last_query.device)
        return torch.cat([key_, local_key_], dim=2), torch.cat([value_, local_value_], dim=2)
    
    def clear_prefill_state(self):
        self.pruned_k_cache = [None for _ in range(self.num_layers)]
        self.pruned_v_cache = [None for _ in range(self.num_layers)]
        self.last_query_temp = [None for _ in range(self.num_layers)]
        self.cross_attn_head_all_layers = [None for _ in range(self.num_layers)]
        torch.cuda.empty_cache()
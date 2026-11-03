import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from typing import List
import random
import numpy as np
import json
import warnings
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache

import math
import einops as ein

def load_head_score(model_type):
    package_root = os.path.dirname(os.path.abspath(__file__))
    head_score_dir = os.path.join(package_root, "data", "head_score")

    if 'llava' in model_type:  # llava-ov-7b, llava-hf
        if 'ov' in model_type or 'hf' in model_type:
            head_score_path = os.path.join(head_score_dir, 'llava-ov.json')
        else:
            raise NotImplementedError
    elif 'qwen2_5' in model_type or 'qwen2.5' in model_type:
        if '3b' in model_type:
            head_score_path = os.path.join(head_score_dir, 'qwen25-vl-3b.json')
        else:
            head_score_path = os.path.join(head_score_dir, 'qwen2_5.json')
    else:
        raise NotImplementedError

    with open(head_score_path, 'r') as f:
        head_score = json.load(f)

    return head_score

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

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len - window_size, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len - window_size)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([selected_keys, key_states[:, :, -window_size:, :], ], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # k_hh_recent = selected_keys
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # v_hh_recent = selected_values
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)
    # pivot merge
    if merge=="pivot":
        # print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    elif merge=="averaged":
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = k_hh_pruned
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = v_hh_pruned
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # weight merge

    return k_hh_recent, v_hh_recent
class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
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

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx = None, num_hidden_layers = None, budget_ratio=None, 
                 pyram_mode = False, pyram_beta = 20,num_key_value_groups = 1, gqa_func='mean'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.pyram_init = False
        self.pyram_mode = pyram_mode
        self.pyram_beta = pyram_beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        self.budget_ratio = budget_ratio


    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        self.max_capacity_prompt = int(q_len * self.budget_ratio)

        # compute pyramidal capacity
        if self.pyram_mode and not self.pyram_init:
            # NOTE: (max_num + min_num) / 2 == base_capacity to restrict the total capacity
            base_capacity = self.max_capacity_prompt - self.window_size
            min_num = base_capacity // self.pyram_beta
            max_num = base_capacity * 2 - min_num
                
            # if the max_num is larger than the query length, we need to adjust the max_num
            if max_num >= q_len - self.window_size:
                max_num = q_len - self.window_size
                min_num = base_capacity * 2 - max_num
        
            # NOTE: compute interval
            steps = (max_num - min_num) // (self.num_hidden_layers - 1)

            self.max_capacity_prompt = max_num - self.layer_idx * steps + self.window_size
            self.pyram_init = True
            print(f"Pyram mode adaptive capacity, layer: {self.layer_idx}, max_capacity_prompt: {self.max_capacity_prompt}, base_capacity: {self.max_capacity_prompt - self.window_size}", flush=True)

        if q_len < self.max_capacity_prompt:
            return origin_key_states, origin_value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)
            
            attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0], -1, self.num_key_value_groups, attn_weights_mean.shape[-1])
            if self.gqa_func == 'max':
                attn_weights_mean = attn_weights_mean.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_mean = attn_weights_mean.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')
                
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_mean, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(self.max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            k_past_compress = origin_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = origin_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = origin_key_states[:, :, -self.window_size:, :]
            v_cur = origin_value_states[:, :, -self.window_size:, :]

            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class SparseMM():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool', base_capacity=None, ratio=None, normalize=None, budget_ratio = None,
                 layer_idx = None, num_hidden_layers = None, head_score=None, num_attention_heads=32, num_key_value_groups=1, gqa_func='mean', model_type=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = base_capacity - window_size
        self.ratio = ratio

        self.normalize = normalize
        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads  
        self.num_hidden_layers = num_hidden_layers
        
        model_type = os.environ.get("MODEL_TYPE")

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

        self.budget_ratio = budget_ratio

        if head_score == 'random':
            head_score_list = np.array([random.random() for _ in range(self.num_hidden_layers * self.num_attention_heads)])
        elif head_score == 'visual':
            head_score = load_head_score(model_type)
            head_score_list = [np.mean(l[1]) for l in head_score.items()]
        head_score_list = torch.tensor(head_score_list / sum(head_score_list))
        # GQA support
        self.score = head_score_list.view(self.num_hidden_layers, self.num_attention_heads//self.num_key_value_groups, self.num_key_value_groups)
        self.score = self.score.sum(dim=-1)

        self.head_adaptive_capacity = None

    # 计算Attn(QK^T)，其中Q取最后window_size个
    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim) #[bsz, heads, window_size, k_len]
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2) #[bsz, heads, k_len]
        # [bsz, kv_heads, groups, k_len]
        attn_weights_mean = attn_weights_mean.view(attn_weights_mean.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        # [bsz, kv_heads, k_len]
        if self.gqa_func == 'max':
            attn_weights_mean = attn_weights_mean.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_mean = attn_weights_mean.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling


    def update_kv(self, origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        # value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states) # 形状[bsz, kv_heads, k_len]
        self.base_capacity = q_len * self.budget_ratio

        if self.head_adaptive_capacity is None:
            min_cache = int(self.base_capacity * self.ratio)
            remain_capacity = (self.base_capacity - min_cache) * self.num_hidden_layers * self.num_attention_heads // self.num_key_value_groups
            self.head_adaptive_capacity = torch.round(self.score * remain_capacity + min_cache).int()

            # Preserve the public SparseMM budget when a head allocation exceeds
            # the current prompt length by redistributing the unavailable slots.
            not_allocated_budget = 0
            head_capacity_list = self.head_adaptive_capacity.tolist()
            for layer_idx in range(len(head_capacity_list)):
                for head_idx in range(len(head_capacity_list[0])):
                    total_budget = head_capacity_list[layer_idx][head_idx]
                    if total_budget > q_len:
                        not_allocated_budget += total_budget - q_len
                        head_capacity_list[layer_idx][head_idx] = q_len

            while not_allocated_budget > 0:
                max_budget = -1
                max_layer_idx = -1
                max_head_idx = -1
                for layer_idx in range(len(head_capacity_list)):
                    for head_idx in range(len(head_capacity_list[0])):
                        current_budget = head_capacity_list[layer_idx][head_idx]
                        if current_budget > max_budget and current_budget < q_len:
                            max_budget = current_budget
                            max_layer_idx = layer_idx
                            max_head_idx = head_idx
                if max_layer_idx == -1:
                    break
                can_allocate_budget = q_len - max_budget
                head_capacity_list[max_layer_idx][max_head_idx] += can_allocate_budget
                not_allocated_budget -= can_allocate_budget
            self.head_adaptive_capacity = torch.tensor(head_capacity_list, dtype=torch.int32)

        # import pdb; pdb.set_trace()
        # ([bsz, 1, k_len, head_dim] * kv_heads)
        origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)
        # Flash-attention元数据初始化
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
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            
            
            self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1): #[k_len]
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # not compress
            return origin_key_states.reshape(-1, head_dim), origin_value_states.reshape(-1, head_dim)
        # [bsz, kv_heads, k_len]
        _,indices = attn_score.sort(dim=-1,descending=True)
        # 元组([bsz, 1, k_len], [bsz, 1, k_len], ... kv_heads个)
        indices = indices.split(1,dim=1)
        heads_key_states = []
        heads_value_states = []
        assert bsz == 1
        # per head

        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0


        for head_idx in range(num_heads//self.num_key_value_groups):
            # [bsz, budget]
            cache_index = indices[head_idx][...,:self.head_adaptive_capacity[self.layer_idx][head_idx]] # 取前top-k个

            l = cache_index.shape[-1] + self.window_size # [budget + window_size]
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim) #[bsz, budget]->[1, 1, bsz*budget, head_dim]
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index) # [1, 1, bsz*budget, head_dim]
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index) # [1, 1, bsz*budget, head_dim]
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2) # [1, 1, bsz*budget + self.window_size, head_dim]
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)# [1, 1, bsz*budget + self.window_size, head_dim]

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim)) #append [bsz*budget + self.window_size, head_dim]
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)       # [budget1 + budget2 + ... , head_dim]
        heads_value_states = torch.cat(heads_value_states, dim=0)   # [budget1 + budget2 + ... , head_dim]

        return heads_key_states, heads_value_states

class LookMCluster():
    def __init__(self, window_size = 32, max_capacity_prompt = 2048, layer_idx = None, num_hidden_layers = None,
                 num_key_value_groups = 1, gqa_func='mean', budget_ratio = 0.2, 
                 is_vision_token_mask = None, is_text_token_mask = None, merge_layer_ratio = 0.05):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.select_num = max_capacity_prompt - window_size # top-k

        self.budget_ratio = budget_ratio

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

        self.is_vision_token_mask = is_vision_token_mask
        self.is_text_token_mask = is_text_token_mask

        # update for dif model
        self.merge_layer_ratio = merge_layer_ratio

    def update_kv(self, origin_key_states, query_states, origin_value_states):
        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups) #[bsz, heads, k_len, dim]
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # self.select_num = int(q_len * self.budget_ratio - self.window_size)
        self.select_num = int(q_len * self.budget_ratio)
        if self.select_num <= 0:
            return origin_key_states, origin_value_states
        else:
            # compute attention
            # [bsz, heads, q_len, k_len]
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]
            # 是否需要softmax？
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # [LOOK-M]
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2) #[bsz, heads, k_len]
            attn_weights_sum = attn_weights_sum.view(attn_weights_sum.shape[0], num_heads//self.num_key_value_groups,self.num_key_value_groups, -1)
            # [bsz, kv_heads, k_len]
            if self.gqa_func == 'max':
                attn_weights_sum = attn_weights_sum.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_sum = attn_weights_sum.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')
            
            # Text token Add Max Score(Text Prefer)
            attn_weights_max = attn_weights_sum.max(dim = -1).values #[bsz, kv_heads]
            text_index = self.is_text_token_mask[:, :-self.window_size] #[bsz, k_len]  k_len = q_len - window_size
            text_index = text_index.unsqueeze(1).expand(attn_weights_sum.shape) #[bsz, kv_heads, k_len]
            text_index = text_index.to(attn_weights_sum.device)
            assert text_index.shape[-1] == attn_weights_sum.shape[-1]
            attn_weights_max = attn_weights_max.unsqueeze(-1).expand(attn_weights_sum.shape) #[bsz, kv_heads, k_len]
            attn_cache = torch.where(
                text_index, # text token
                attn_weights_sum + attn_weights_max, # True: add the max
                attn_weights_sum # False: the same
            ) #[bsz, kv_heads, k_len]

            indices = attn_cache.topk(self.select_num, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            # merge kv 
            if self.layer_idx >= (self.num_hidden_layers * self.merge_layer_ratio):
                key_states, value_states = merge_kv(origin_key_states, origin_value_states, indices, self.window_size, "pivot")
            else:
                k_past_compress = origin_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = origin_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = origin_key_states[:, :, -self.window_size:, :]
                v_cur = origin_value_states[:, :, -self.window_size:, :]

                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            return key_states, value_states
class MadaKVCluster():
    def __init__(self, window_size = 32, base_capacity_prompt = 2048, layer_idx = None, num_hidden_layers = None,
                 num_key_value_groups = 1, gqa_func='mean', threshold = 0.7, budget_ratio = 0.2,
                 is_vision_token_mask = None, is_text_token_mask = None, current_budget = None):
        self.window_size = window_size
        self.base_capacity_prompt = base_capacity_prompt

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func

        self.is_vision_token_mask = is_vision_token_mask
        self.is_text_token_mask = is_text_token_mask
        self.threshold = threshold

        self.budget_ratio = budget_ratio

        self.current_budget = current_budget

    # 疑问点一：proxy token是否需要保留，需要保留
    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim) #[bsz, heads, window_size, k_len]
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2) #[bsz, heads, k_len]
        # [bsz, kv_heads, groups, k_len]
        attn_weights_sum = attn_weights_sum.view(attn_weights_sum.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
        # [bsz, kv_heads, k_len]
        if self.gqa_func == 'max':
            attn_weights_sum = attn_weights_sum.max(dim=-2).values
        elif self.gqa_func == 'mean':
            attn_weights_sum = attn_weights_sum.mean(dim=-2)
        else:
            raise ValueError('gqa_func not supported')
        return attn_weights_sum #[bsz, kv_heads, k_len]
    
    # 论文中未提及如何选择，这里默认topk
    def update_kv(self, origin_key_states, query_states, origin_value_states):
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states) # 形状[bsz, kv_heads, k_len]
        assert key_states.shape[-2] == query_states.shape[-2]
        assert bsz == 1 # 暂不支持 bsz > 1的decoding
        if(self.current_budget is None):
            self.current_budget = torch.tensor([q_len * self.budget_ratio], dtype=torch.int32, device=attn_score.device) # [bsz, 1]

        if(self.current_budget.item() <= 0):
            return origin_key_states, origin_value_states
        # 计算modality prefer
        vision_index = self.is_vision_token_mask[:, :-self.window_size] #[bsz, k_len]
        vision_num = vision_index.sum().item()
        vision_index = vision_index.unsqueeze(1).expand(attn_score.shape) #[bsz, kv_heads, k_len]
        vision_prefer_score = torch.where(
            vision_index,
            attn_score,
            0
        )
        vision_prefer_score = vision_prefer_score.sum(dim = -1) #[bsz, kv_heads]

        text_index = self.is_text_token_mask[:, :-self.window_size] #[bsz, k_len]
        text_num = text_index.sum().item()
        text_index = text_index.unsqueeze(1).expand(attn_score.shape) #[bsz, kv_heads, k_len]
        text_prefer_score = torch.where(
            text_index,
            attn_score,
            0
        )
        text_prefer_score = text_prefer_score.sum(dim = -1) #[bsz, kv_heads]

        total_prefer_score = text_prefer_score + vision_prefer_score

        # 预分配
        target_device = vision_prefer_score.device
        self.current_budget = self.current_budget.to(target_device)

        vision_budget = (self.current_budget.reshape(-1, 1).float() * ( vision_prefer_score / total_prefer_score)).int() # [bsz, kv_heads]
        text_budget = (self.current_budget.reshape(-1, 1).float() - vision_budget).int() # [bsz, kv_heads]

        # 筛选
        # [bsz, kv_heads, k_len]
        vision_attn_score = torch.where(
            vision_index, 
            attn_score,
            torch.finfo(attn_score.dtype).min, # min
        )
        
        text_attn_score = torch.where(
            text_index, 
            attn_score,
            torch.finfo(attn_score.dtype).min,
        )
        head_key_states = []
        head_value_states = []
        for head_idx in range(num_heads // self.num_key_value_groups):
            # print("vision_attn_score.shape:", vision_attn_score.shape)
            # vision_indices = vision_attn_score[:, head_idx, :].topk(min(vision_budget[0][head_idx].int().item(), vision_num), dim=-1).indices #[bsz, vison_num]
            # text_indices = text_attn_score[:, head_idx, :].topk(min(text_budget[0][head_idx].int().item(), text_num), dim=-1).indices #[bsz, text_num]
            # print(f"head_idx = {head_idx}, {vision_budget[0][head_idx].int().item()}, {text_budget[0][head_idx].int().item()}")
            # print(f"vision_num = {vision_num}, text_num = {text_num}")

            vision_indices = vision_attn_score[:, head_idx, :].topk(vision_budget[0][head_idx].int().item(), dim=-1).indices #[bsz, vison_num]
            text_indices = text_attn_score[:, head_idx, :].topk(text_budget[0][head_idx].int().item(), dim=-1).indices #[bsz, text_num]
            
            # print("vision_indices", vision_indices.shape)
            # print("text_indices", text_indices.shape)
            all_indices = torch.cat([vision_indices, text_indices], dim=-1) #[bsz, text+vision num]
            # print("all_indices", all_indices.shape)
            # print("origin_key_states.shape", origin_key_states.shape)
            all_indices = all_indices.unsqueeze(-1).expand(-1, -1, head_dim)#[bsz, text+vision num, head_dim]
            k_compress = origin_key_states[:, head_idx, :-self.window_size, :].gather(dim = -2, index = all_indices)
            v_compress = origin_value_states[:, head_idx, :-self.window_size, :].gather(dim = -2, index = all_indices)
            head_key_states.append(k_compress)
            head_value_states.append(v_compress)

        k_past_compress = torch.stack(head_key_states, dim = 1)
        v_past_compress = torch.stack(head_value_states, dim = 1)

        k_cur = origin_key_states[:, :, -self.window_size:, :]
        v_cur = origin_value_states[:, :, -self.window_size:, :]

        key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        
        compensate_K = 0
        # 更新下一层的budget数量
        for head_idx in range(num_heads // self.num_key_value_groups):
            vision_score_tmp = vision_attn_score[0, head_idx, :] #[k_len]
            vision_score_tmp = vision_score_tmp.sort(dim=0, descending=True).values
            # print("Vision_score_tmp.shape", vision_score_tmp.shape)
            cal_sum, vi = 0, 0
            while cal_sum < self.threshold * vision_prefer_score[0][head_idx].item() and vi < vision_score_tmp.shape[0]:
                cal_sum += vision_score_tmp[vi].item()
                vi = vi + 1
            
            text_score_tmp = text_attn_score[0, head_idx, :] #[k_len]
            text_score_tmp = text_score_tmp.sort(dim=0, descending=True).values
            cal_sum, ti = 0, 0
            while cal_sum < self.threshold * text_prefer_score[0][head_idx].item() and ti < text_attn_score.shape[0]:
                cal_sum += text_score_tmp[ti]
                ti = ti + 1
            compensate_K += vi + ti - self.current_budget.item()
            # print(f"head_idx={head_idx}, vi={vi}, ti={ti}")
        min_budget = int(q_len * self.budget_ratio)
        max_budget = q_len - self.window_size
        self.next_layer_budget = torch.clamp(
            torch.tensor(self.current_budget.item() - compensate_K / (self.num_hidden_layers - self.layer_idx)).int(),
            min=min_budget,
            max=max_budget
        ).int()
        return key_states, value_states
    
    def get_next_layer_kv(self):
        return self.next_layer_budget

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'budget_ratio'):
            self.config.budget_ratio = float(os.getenv('BUDGET_RATIO'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            budget_ratio = self.config.budget_ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func=self.config.gqa_func
        )

def init_sparsemm(self, rebuild_cluster=False):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'base_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'budget_ratio'):
            self.config.budget_ratio = float(os.getenv('BUDGET_RATIO'))
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'head_score'):
            method = os.getenv('METHOD', None)
            if method == 'sparsemm':
                self.config.head_score = 'visual' 
            elif method == 'random':
                self.config.head_score = 'random'
            else:
                raise ValueError('head_score should be set')
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = float(os.getenv('RATIO'))
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'

    if rebuild_cluster or not hasattr(self, "kv_cluster"):
        self.kv_cluster = SparseMM(
            window_size = self.config.window_size,
            base_capacity=self.config.max_capacity_prompt,
            head_score=self.config.head_score,
            ratio=self.config.ratio,
            budget_ratio = self.config.budget_ratio,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            gqa_func = self.config.gqa_func,
            model_type=self.config.model_type
        )

def init_lookm(self, 
    is_vision_token_mask: Optional[torch.Tensor] = None,
    is_text_token_mask: Optional[torch.Tensor] = None,
    merge_layer_ratio: Optional[float] = None,
    ):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'budget_ratio'):
            self.config.budget_ratio = float(os.getenv('BUDGET_RATIO'))
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        if not hasattr(self.config, 'is_vision_token_mask'):
            self.config.is_vision_token_mask = is_vision_token_mask
        if not hasattr(self.config, 'is_text_token_mask'):
            self.config.is_text_token_mask = is_text_token_mask

    # if not hasattr(self, "kv_cluster"):
    self.kv_cluster = LookMCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
        budget_ratio = self.config.budget_ratio,
        gqa_func=self.config.gqa_func,
        is_vision_token_mask=is_vision_token_mask,
        is_text_token_mask=is_text_token_mask,
        layer_idx=self.layer_idx,
        num_hidden_layers = self.config.num_hidden_layers,
        merge_layer_ratio = merge_layer_ratio
    )

def init_madakv(self, 
    is_vision_token_mask: Optional[torch.Tensor] = None,
    is_text_token_mask: Optional[torch.Tensor] = None,
    current_budget: Optional[torch.Tensor] = None,
    ):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'base_capacity_prompt'):
            self.config.base_capacity_prompt = int(os.getenv('BUDGET'))
        if not hasattr(self.config, 'budget_ratio'):
            self.config.budget_ratio = float(os.getenv('BUDGET_RATIO'))
        if not hasattr(self.config,'num_hidden_layers'):
            raise ValueError('num_hidden_layers should be set')
        if not hasattr(self.config,'threshold'):
            self.config.threshold = 0.8
        if not hasattr(self.config,'gqa_func'):
            if 'llama' in self.config.model_type or 'mistral' in self.config.model_type or \
                'llava' in self.config.model_type or 'qwen' in self.config.model_type:
                self.config.gqa_func = 'mean'
        if not hasattr(self.config, 'is_vision_token_mask'):
            self.config.is_vision_token_mask = is_vision_token_mask
        if not hasattr(self.config, 'is_text_token_mask'):
            self.config.is_text_token_mask = is_text_token_mask
        if not hasattr(self.config, 'current_budget'):
            self.config.current_budget = current_budget

    # if not hasattr(self, "kv_cluster"):
    self.kv_cluster = MadaKVCluster(
        window_size = self.config.window_size,
        base_capacity_prompt = self.config.base_capacity_prompt,
        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
        gqa_func=self.config.gqa_func,
        layer_idx = self.layer_idx,
        budget_ratio = self.config.budget_ratio,
        threshold=self.config.threshold,
        num_hidden_layers = self.config.num_hidden_layers,
        is_vision_token_mask=is_vision_token_mask,
        is_text_token_mask=is_text_token_mask,
        current_budget=current_budget
    )

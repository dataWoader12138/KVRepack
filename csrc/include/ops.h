#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/torch.h>

torch::Tensor update_flatten_view(torch::Tensor &cache, torch::Tensor &state, torch::Tensor &headlens, torch::Tensor &cu_headlens);
void update_base_offset(torch::Tensor &base_offset, torch::Tensor &cu_offset, torch::Tensor &indices, int head_dim);

void gemm_softmax(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batch_count,
    int m,
    int n,
    int k,
    float alpha,
    float beta);

void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // input, as cached previous position id as input, also reordered position ids, int64_t type
    torch::Tensor cur_pos_ids,    // input, incoming position id, int64_t type
    torch::Tensor offsets,        // output, offsets for gather_copy_with_offsets, size as numBlocks, int type
    torch::Tensor cnts,           // output, counts to separate d2d and h2d, size as numBlocks, int type
    int batch_size, int heads, 
    int map_size 
);

void gather_copy_with_offsets(
    torch::Tensor keys,             // input, cpu keys
    torch::Tensor k_cache_buffer,   // input, gpu keys
    torch::Tensor temp_k,             // a temp gpu memory for copy, size same as single layer k_cache_buffer 
    torch::Tensor values,           // input, cpu values
    torch::Tensor v_cache_buffer,   // input, gpu values
    torch::Tensor temp_v,             // a temp gpu memory for copy, size same as single layer v_cache_buffer 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    torch::Tensor signals,          // extra internal signals, all zeros sizes as numBlocks, size as numBlocks
    torch::Tensor gpu_base_offset,  // gpu buffer base offset for heads, size as numBlocks
    int batch_size, int heads, int cpu_length, int gpu_length, int gpu_offset, int gpu_stride, int map_size
); // input, torch stream

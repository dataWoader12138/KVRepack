#include <cuda_bf16.h>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef BLOCK_SIZE_CP
#define BLOCK_SIZE_CP 128
#endif

#include "ops.h"
#include "map.cuh"
#include "copy.cuh"

#if BLOCK_SIZE_CP == 128
#define PTYPE int4
#endif

#if BLOCK_SIZE_CP == 256
#define PTYPE int2
#endif

// reorder position ids, and compute offsets and cnts by computing cache hits/misses
// call it before gather_copy_with_offsets
void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // input, as cached previous position id as input, also reordered position ids, int64_t type
    torch::Tensor cur_pos_ids,    // input, incoming position id, int64_t type
    torch::Tensor offsets,        // output, offsets for gather_copy_with_offsets, size as numBlocks, int type
    torch::Tensor cnts,           // output, counts to separate d2d and h2d, size as numBlocks, int type
    int batch_size, int heads, 
    int map_size = 256
)
{
    int blockSize = map_size;
    int numBlocks = batch_size * heads;

    int64_t* cached_pos = cached_pos_ids.data_ptr<int64_t>();
    int64_t* cur_pos = cur_pos_ids.data_ptr<int64_t>();
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());

    if(map_size == 32) {
        reorder_keys_and_mixed_offsets<int64_t, 32, 1024><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 64) {
        reorder_keys_and_mixed_offsets<int64_t, 64, 1024><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 128) {
        reorder_keys_and_mixed_offsets<int64_t, 128, 2048><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    } else if (map_size == 256) {
        reorder_keys_and_mixed_offsets<int64_t, 256, 2048><<<numBlocks, blockSize>>>(
        cached_pos /*in*/, 
        cur_pos, 
        cached_pos /*out*/, 
        offsets_ptr, 
        cnts_ptr);
    }
}

// gather copy with offsets
// call it after reorder_keys_and_compute_offsets
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
    int batch_size, int heads, int cpu_length, int gpu_length, int gpu_offset, int gpu_stride, int map_size = 256) // input, torch stream
{
    int blockSize = BLOCK_SIZE_CP;
    int numBlocks = batch_size * heads * 2;
    int maxSMBytes = CPY_SIZE * 2 * 1024 + map_size * 4 + sizeof(PTYPE); // must less than 160 KB

    // Verify tensor alignment and size requirements
    constexpr int PTYPE_SIZE = sizeof(PTYPE);
    constexpr int BF16_SIZE = sizeof(at::BFloat16);
    
    // Check alignment requirements
    TORCH_CHECK(reinterpret_cast<uintptr_t>(keys.data_ptr<at::BFloat16>()) % PTYPE_SIZE == 0,
                "keys tensor is not aligned for PTYPE access");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(k_cache_buffer.data_ptr<at::BFloat16>()) % PTYPE_SIZE == 0,
                "k_cache_buffer tensor is not aligned for PTYPE access");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(values.data_ptr<at::BFloat16>()) % PTYPE_SIZE == 0,
                "values tensor is not aligned for PTYPE access");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(v_cache_buffer.data_ptr<at::BFloat16>()) % PTYPE_SIZE == 0,
                "v_cache_buffer tensor is not aligned for PTYPE access");
    
    // Check size requirements (number of elements must be divisible by PTYPE element count)
    constexpr int PTYPE_ELEMENTS = PTYPE_SIZE / BF16_SIZE;
    TORCH_CHECK(keys.numel() % PTYPE_ELEMENTS == 0,
                "keys tensor size not compatible with PTYPE vectorization");
    TORCH_CHECK(k_cache_buffer.numel() % PTYPE_ELEMENTS == 0,
                "k_cache_buffer tensor size not compatible with PTYPE vectorization");
    TORCH_CHECK(values.numel() % PTYPE_ELEMENTS == 0,
                "values tensor size not compatible with PTYPE vectorization");
    TORCH_CHECK(v_cache_buffer.numel() % PTYPE_ELEMENTS == 0,
                "v_cache_buffer tensor size not compatible with PTYPE vectorization");

    // Cast bf16 data pointers to int2 or int4
    PTYPE* keys_ptr = reinterpret_cast<PTYPE*>(keys.data_ptr<at::BFloat16>());
    PTYPE* k_cache_buffer_ptr = reinterpret_cast<PTYPE*>(k_cache_buffer.data_ptr<at::BFloat16>());
    PTYPE* temp_k_ptr = reinterpret_cast<PTYPE*>(temp_k.data_ptr<at::BFloat16>());
    PTYPE* values_ptr = reinterpret_cast<PTYPE*>(values.data_ptr<at::BFloat16>());
    PTYPE* v_cache_buffer_ptr = reinterpret_cast<PTYPE*>(v_cache_buffer.data_ptr<at::BFloat16>());
    PTYPE* temp_v_ptr = reinterpret_cast<PTYPE*>(temp_v.data_ptr<at::BFloat16>());
    int* offsets_ptr = reinterpret_cast<int*>(offsets.data_ptr<int32_t>());
    int* cnts_ptr = reinterpret_cast<int*>(cnts.data_ptr<int32_t>());
    unsigned int* signals_ptr = reinterpret_cast<unsigned int*>(signals.data_ptr<int32_t>());
    int64_t* gpu_base_offset_ptr = reinterpret_cast<int64_t*>(gpu_base_offset.data_ptr<int64_t>());

        // Get cudaStream_t from torch::cuda::Stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(map_size == 32) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 32><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            keys_ptr,
            temp_k_ptr,
            k_cache_buffer_ptr,
            values_ptr, 
            temp_v_ptr, 
            v_cache_buffer_ptr, 
            cpu_length, 
            gpu_length,
            gpu_offset,
            gpu_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr,
            gpu_base_offset_ptr);
    } else if(map_size == 64) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 64><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            keys_ptr,
            temp_k_ptr,
            k_cache_buffer_ptr,
            values_ptr, 
            temp_v_ptr, 
            v_cache_buffer_ptr, 
            cpu_length, 
            gpu_length,
            gpu_offset,
            gpu_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr,
            gpu_base_offset_ptr);
    }else if(map_size == 128) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 128><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            keys_ptr,
            temp_k_ptr,
            k_cache_buffer_ptr,
            values_ptr, 
            temp_v_ptr, 
            v_cache_buffer_ptr, 
            cpu_length, 
            gpu_length,
            gpu_offset,
            gpu_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr,
            gpu_base_offset_ptr);
    } else if(map_size == 256) {
        // this only needs to run once
        cudaFuncSetAttribute(gather_copy_var_midpoint_BP<PTYPE, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSMBytes);

        gather_copy_var_midpoint_BP<PTYPE, 256><<<numBlocks, blockSize, maxSMBytes, stream>>>(
            keys_ptr,
            temp_k_ptr,
            k_cache_buffer_ptr,
            values_ptr, 
            temp_v_ptr, 
            v_cache_buffer_ptr, 
            cpu_length, 
            gpu_length,
            gpu_offset,
            gpu_stride, 
            offsets_ptr,
            cnts_ptr,
            signals_ptr,
            gpu_base_offset_ptr);
    }
}
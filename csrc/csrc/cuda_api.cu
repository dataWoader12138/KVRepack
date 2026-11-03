#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/python.h>

#include "cuda_api.h"
#include "ops.h"
#include "static_switch.h"

template<typename tensor_t, int kblock_size=1024>
__global__ void update_flatten_view_kernel(tensor_t* dst_ptr, tensor_t* src_ptr, tensor_t* state_ptr, int* headlens,
        int *cu_headlens,
        int dim) {
    // Create new tensor from cache and insert element into it.

    int head_idx = blockIdx.x;
    int thread_group = blockIdx.y;
    int tid = threadIdx.x + thread_group * blockDim.x;
    int num_threads = blockDim.x * gridDim.y;

    int headlen = headlens[head_idx];

    // get position of src, dst, insert ptr
    int src_cum_off = cu_headlens[head_idx] * dim;
    int dst_cum_off = src_cum_off + head_idx * dim;
    int insert_off = (cu_headlens[head_idx+1] + head_idx) * dim;

    auto old_cache_ptr = src_ptr + src_cum_off;
    auto new_cache_ptr = dst_ptr + dst_cum_off;
    auto insert_cache_ptr = dst_ptr + insert_off;

    // copy old data
    for (int start_addr = 0; start_addr < headlen * dim; start_addr += kblock_size * num_threads) {
        auto src_addr = old_cache_ptr + start_addr + tid * kblock_size;
        auto dst_addr = new_cache_ptr + start_addr + tid * kblock_size;

        #pragma unroll
        for (int i = 0; i < kblock_size; i++) {
            if (start_addr + tid * kblock_size + i >= headlen * dim) {
                return;
            }
            dst_addr[i] = src_addr[i];
        }
    }

    // insert new data
    if (tid < dim) {
        auto insert_src_ptr = state_ptr + head_idx * dim + tid;
        auto insert_dst_addr = insert_cache_ptr + tid;
        *insert_dst_addr = *insert_src_ptr;
    }
}

__global__ void update_base_offset_kernel(int64_t* base_offset, int* cu_offset, int* indices, int head_dim) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (tid == 0) {
        int idx = indices[bid];
        base_offset[idx] += cu_offset[idx] * head_dim;
    }
}

torch::Tensor update_flatten_view(torch::Tensor &cache, torch::Tensor &state, torch::Tensor &headlens, torch::Tensor& cu_headlens) {
  TORCH_CHECK(headlens.dtype() == torch::kInt32, "expected headlens to be int32");
  TORCH_CHECK(cu_headlens.dtype() == torch::kInt32, "expected cu_dst_pos to be int32");

  auto cache_shape = cache.sizes();

  int origin_len = cache_shape[0];
  int head_dim = cache_shape[1];
  int head_num = headlens.sizes()[0];

  torch::Tensor out = torch::empty({origin_len + head_num, head_dim}, cache.options());

  const int kblock_size = 1;
  const int num_threads_group = 1024;
  const int num_threads = 128;

  dim3 grid(head_num, num_threads_group);

  dim3 block(num_threads);
  TORCH_CHECK(num_threads >= head_dim, "num threads should larger than head dim");

  FP16_SWITCH(cache.dtype() == torch::kFloat16, [&] {
      auto kernel = update_flatten_view_kernel<elem_type, kblock_size>;
      kernel<<<grid, block, 0>>>((elem_type*)out.data_ptr(), (elem_type*)cache.data_ptr(), (elem_type*)state.data_ptr(), (int*)headlens.data_ptr(), (int*)cu_headlens.data_ptr(), head_dim);
  });

  // TODO: when to use sync or torch auto
  // cudaDeviceSynchronize();

  return out;
}

void update_base_offset(torch::Tensor &base_offset, torch::Tensor &cu_offset, torch::Tensor &indices, int head_dim) {
  TORCH_CHECK(base_offset.dtype() == torch::kInt64, "expected base_offset to be int64");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "expected indices to be int32");
  TORCH_CHECK(cu_offset.dtype() == torch::kInt32, "expected cu_offset to be int32");

  auto indices_shape = indices.sizes();
  auto n_heads = indices_shape[0];

  const int num_blocks = n_heads;
  const int num_threads = 32;

  dim3 grid(num_blocks);
  dim3 block(num_threads);

  int64_t* base_offset_ptr = base_offset.data_ptr<int64_t>();
  int* cu_offset_ptr = reinterpret_cast<int*>(cu_offset.data_ptr<int32_t>());
  int* indices_ptr = reinterpret_cast<int*>(indices.data_ptr<int32_t>());

  update_base_offset_kernel<<<grid, block, 0>>>(base_offset_ptr, cu_offset_ptr, indices_ptr, head_dim);
}

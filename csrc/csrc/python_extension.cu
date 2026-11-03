#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update_flatten_view", &update_flatten_view, "update flatten view cache");
    m.def("update_base_offset", &update_base_offset, "update base offset for dynamic heads during decoding");
    m.def("gemm_softmax", &gemm_softmax, "GEMM Softmax (CUDA)");
    m.def("reorder_keys_and_compute_offsets", &reorder_keys_and_compute_offsets, "Reorder position ids cache");
    m.def("gather_copy_with_offsets", &gather_copy_with_offsets, "Gather and copy with offsets (CUDA)");
}
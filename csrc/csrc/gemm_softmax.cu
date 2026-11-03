#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_size.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/numeric_size.h" // cutlass::bits_to_bytes

#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "helper.h"
#include "ops.h"
#include "gemm_with_softmax.h"

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementCompute = float;
using ElementNorm = float;
using ElementSum = float;
using ElementD = ElementC;
using ElementSoftmax = ElementC;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 32>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag = cutlass::arch::Sm80; // only have template function for Sm80!

// ApplyShape impacts the final Softmax performance a lot.
// Set ApplyShape::kColumn to be the next multiple of 32 number that is after
// (gemm_N / alignment).
// Set ApplyShape::kRow to max(1, 128 / ApplyShape::kColumn).
using ApplyShape = cutlass::MatrixShape<1, 1024>;

static int const kStages = 3;

/// Linear scaling operator
using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementCompute,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

using GemmSoftmax = cutlass::GemmSoftmax<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    kStages,
    ApplyShape>;

using LayoutC = typename GemmSoftmax::LayoutC;
using LayoutN = typename GemmSoftmax::LayoutN;
using LayoutS = typename GemmSoftmax::LayoutS;
using MatrixCoord = typename LayoutC::TensorCoord;

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
    float beta)
{
    cutlass::gemm::GemmCoord problem = {m, n, k};
    int block_num = (problem.n() + GemmSoftmax::ThreadblockShape::kN - 1) / GemmSoftmax::ThreadblockShape::kN;

    int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);

    // fixed rowmajor for norm and sum
    int64_t ldn = problem.m();
    int64_t lds = ldn;

    int64_t total_elements_A_per_batch = problem.m() * problem.k();
    int64_t total_elements_B_per_batch = problem.k() * problem.n();
    int64_t total_elements_C_per_batch = problem.m() * problem.n();
    int64_t total_elements_D_per_batch = problem.m() * problem.n();
    int64_t total_elements_partial_norm_per_batch = block_num * problem.m();

    GemmSoftmax::Arguments args(
        problem,
        batch_count,
        {reinterpret_cast<ElementA *>(A.data_ptr()), lda},
        {reinterpret_cast<ElementB *>(B.data_ptr()), ldb},
        {nullptr, ldc},
        {reinterpret_cast<ElementD *>(D.data_ptr()), ldc},
        {
            ElementCompute(alpha),
            ElementCompute(beta)
        },
        {reinterpret_cast<ElementNorm *>(Norm.data_ptr()), ldn},
        {reinterpret_cast<ElementSum *>(Sum.data_ptr()), lds},
        {reinterpret_cast<ElementSoftmax *>(Softmax.data_ptr()), ldc},
        total_elements_A_per_batch,
        total_elements_B_per_batch,
        total_elements_C_per_batch,
        total_elements_D_per_batch,
        total_elements_partial_norm_per_batch,
        total_elements_partial_norm_per_batch,
        total_elements_D_per_batch
    );

    GemmSoftmax gemm_softmax_op;

    CUTLASS_CHECK(gemm_softmax_op.initialize(args));

    CUTLASS_CHECK(gemm_softmax_op());
}
"""Build the KVRepack PyTorch extension from the vendored CUDA sources."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="kvrepack-kernels",
    version="0.1.0",
    description="CUDA kernels for KVRepack",
    python_requires=">=3.12",
    license_files=["LICENSE", "deps/cutlass/LICENSE.txt", "deps/cutlass/EULA.txt"],
    packages=[],
    install_requires=["torch==2.6.0"],
    ext_modules=[
        CUDAExtension(
            name="kvrepack_kernel",
            sources=[str(path.relative_to(ROOT)) for path in sorted((ROOT / "csrc").glob("*.cu"))],
            include_dirs=[
                str(ROOT / "csrc"),
                str(ROOT / "include"),
                str(ROOT / "deps/cutlass/include"),
                str(ROOT / "deps/cutlass/tools/util/include"),
                str(ROOT / "deps/cutlass/examples/common"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "--use_fast_math",
                    "-lineinfo",
                    "--ptxas-options=-v",
                    "--ptxas-options=-O2",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ],
            },
        )
    ],
    # PyTorch handles TORCH_CUDA_ARCH_LIST. Do not add our own -gencode flags.
    cmdclass={"build_ext": BuildExtension},
)

# KVRepack CUDA extension

The distribution is `kvrepack-kernels`; the Python import is `kvrepack_kernel`.
CUDA and C++ kernel sources are unchanged from the local HybridKV-artifact
snapshot. The build entry point replaces the original `mmko_kernel` name.

Install the project's Python and PyTorch environment first, then build from the
KVRepack repository root with a matching CUDA toolkit available:

```bash
python -m pip install --no-build-isolation ./csrc
```

The setup script delegates architecture selection to PyTorch's
`TORCH_CUDA_ARCH_LIST` mechanism. Set that variable before building to select
architectures for your deployment machines, for example
`TORCH_CUDA_ARCH_LIST="8.0;8.6"`. `CUDA_HOME` can select the toolkit. There are no
architecture flags hardcoded in this package. No precompiled extension is
provided in the source repository.

The three vendored CUTLASS header directories are a local snapshot, not a
complete CUTLASS checkout or a Git submodule. Its version header identifies
4.5.0; the original Git commit is unavailable. File hashes and provenance are
recorded in `../provenance/vendor_sources.json`. The snapshot includes the
upstream `LICENSE.txt` and `EULA.txt`; kernel licensing is in `LICENSE`.

Source distributions include the headers and CUDA sources. Binary wheels
contain the compiled extension and licenses. A CUDA build and GPU validation
are required on the destination environment; static checks alone do not verify
kernel execution or compatibility.

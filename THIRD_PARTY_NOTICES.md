# Source and third-party notices

This repository reorganizes the user-provided HybridKV-artifact and HybridKV-opensource workspace. File-level origins, transformations and checksums are recorded in `provenance/`.

- The vendored lmms-eval snapshot retains its upstream MIT license in `vendor/lmms-eval/LICENSE`. Its selected Qwen adapter is customized for KVRepack; this is not an unmodified upstream release.
- The custom CUDA extension retains the supplied MIT notice in `csrc/LICENSE`.
- The included NVIDIA CUTLASS header snapshot retains NVIDIA's license under `csrc/deps/cutlass/LICENSE.txt` and individual source headers. Its original Git commit is not available in the supplied workspace; checksums identify the exact bundled snapshot.
- Copied Transformers model files retain their original copyright and Apache license headers.
- MileBench dataset/scoring adapters and the four LLaVA KVRepack ports originate in the supplied HybridKV-opensource tree; exact sources, declared changes and checksums are recorded in milebench_sources.json and llava_image_sources.json. They do not establish a new project-wide license or historical numerical equivalence.
- AsymKV files retain their upstream commit reference and supplied notices. The public `asymkv` method selects the official path only; required shared helper functions are preserved privately.

The supplied HybridKV project roots did not contain a project-wide LICENSE. No new license or authorship claim has been invented for the inherited core or the user's improvements. The repository owner should establish the project-wide license and attribution before a public release. This staging version is not represented as fully licensed for redistribution.

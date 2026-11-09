# LookM comparison preflight, 2026-11-09

Qwen2.5-VL-7B-Instruct, native LookM versus HybridKV + KVRepack, both nominal10% budget. CPU235 tests:234 passed and1 existing skip. Four independent one-input GPU cases and native NextQA/VATEX scoring completed. Input fingerprints match between methods; both Hybrid smoke samples reproduce prior raw baseline input/prediction IDs exactly. Source audit preserves native algorithms and explains actual active-cache budget differences. This is connectivity evidence, not a200-input quality result.

At publication, formal generation in outputs/benchmarks/lookm_vs_kvrepack_20261109 was running. See docs/LOOKM_COMPARISON.md for protocol/status. Prompts and local video/Arrow paths are omitted from smoke reference_metadata; predictions, fingerprints and numerical audits are preserved. Model/video/tensor data and credentials are excluded.

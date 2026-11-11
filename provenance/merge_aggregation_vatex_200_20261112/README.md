# Fixed-group VATEX merge evidence

Complete: Qwen2.5-VL-7B, A800-80GB, first200, LookM/AsymKV aggregation on SnapKV/HybridKV. 800 target results (228 retained +572 new); baseline generation stopped at the user request.

`raw_evidence.tar.gz` preserves exact files at repository-relative paths: 400 new-scope bundles,114 parent three-arm bundles, contracts/states/transformed-source specifications, raw reports/predictions/scoring logs, frozen numerical sources and the parent source snapshot. Model weights, videos, Arrow references, secrets and GPU tensors are excluded. Machine-specific original paths remain recorded; fresh experiments accept explicit model/HF paths and do not require the parent run.

`raw_evidence_index.json` records every original SHA and size. `independent_audit.json` validates structure/source/groups/budgets; `independent_results_check.json` records seven fresh native rescoring passes,914 unique token decodes and zero numerical differences. The independent CPU script and release CPU/source logs are included.

The parent first114 table has its own reference corpus and is separate from first200 results. HybridKV budget covers active slots rather than backing memory. Read the protocol and result page under docs before interpreting method or latency differences.

# Strict LookM:400 fresh video generations,2026-11-09

Qwen2.5-VL-7B-Instruct,64frames,NextQA/VATEX original rows0..199 each. OnlyLookM was rerun. Budget policy total_prefill_including_recent_v2:all28x4 native KV heads retain floor(.1N) prefill positions including the recent window. Decoded tokens append under the original decoding rule and are accounted separately.

See [formal results](../../docs/LOOKM_STRICT_RESULTS_20261109.md) and [source correction/validation](../../docs/LOOKM_STRICT_BUDGET.md). The corrected implementation also fixes head-local drop selection and numerical/boundary behavior; differences from the legacy400 cannot be attributed solely to the32-position budget correction. Hybrid/RoPE stopped runs remain stopped.

Each task has the frozen contract/config/state and native corpus reports. Independent checks audit all400 executions and recompute native task scores while preserving report SHA. samples.tar.gz contains all400 compact prediction/audit records, with declared metadata omissions recorded in sample_manifest.json. archive_validation.json checks the full original/public objects after only that transformation. No videos, weights, tensor caches or credentials are included.

To rescore, extract the archive into a fresh replay folder, copy each task contract/config/state tree there, then invoke kvrepack.benchmarks.single_video_report --run-root TASK_CASE --hf-home MATCHING_LOCAL_CACHE --output TASK_CASE/report --limit 200 for each task. Matching original Arrow/NLTK/Java resources are required; scoring does not use a model/GPU. Original local paths remain historical provenance.

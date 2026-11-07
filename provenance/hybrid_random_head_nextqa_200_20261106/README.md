# HybridKV-KVRepack NExTQA 200-input Random-head ablation

Generated on 2026-11-06, independently checked on 2026-11-07. Qwen2.5-VL-7B + HybridKV-KVRepack; original rows0..199; 4 conditions, 800 completed generations, 184 source videos. No VATEX, All-head rerun, or extension to9178 rows. All conditions select and effectively merge56 of112 heads with unchanged native per-head retained budgets. Heterogeneous budgets and global identity/layer changes mean total removed-token work is not controlled equal; the report preserves these differences.

See [Chinese result and interpretation](../../docs/HYBRID_RANDOM_HEAD_NEXTQA_200_RESULTS_20261106.md) and [protocol](../../docs/HYBRID_RANDOM_HEAD_NEXTQA_200.md). `report/` contains official WUPS scores, seed SD, paired cluster intervals and PNG/PDF/SVG plots. Two independent checks audit execution and recalculate original-task WUPS and source-video intervals.

`samples.tar.gz` contains all200 four-condition bundles. All numerical fields and generated token IDs are preserved; free-text input prompts and local video/Arrow paths are removed from reference_metadata. The exact transformation and original/public hashes are in sample_manifest.json. Video/image data, model weights, tensor caches and credentials are excluded. Unabridged originals remain in local outputs.

To rescore with matching local NextQA reference Arrow and NLTK resources, create a fresh replay directory, copy run_contract.json/run_config.json/run_state.json into it, extract samples.tar.gz there, then run:

```bash
PYTHONPATH=src python -m kvrepack.ablations.hybrid_head_selection_report \
  --run-root /path/to/replay --hf-home /path/to/huggingface_cache \
  --output /path/to/replay/report --limit 200
```

Archived configurations preserve original local paths as provenance. The scorer reads reference data from the HF_HOME argument; no model/GPU is loaded. index.json covers the published files.

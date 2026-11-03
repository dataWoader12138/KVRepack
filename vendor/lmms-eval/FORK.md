# KVRepack evaluation fork

This is the customized lmms-eval 0.3.4 source snapshot from the local
HybridKV-artifact experiment directory. The upstream README and license are
preserved. Its Python package remains `lmms_eval`; the distribution is renamed
`kvrepack-lmms-eval`. Install this fork with the repository's locked environment
instead of installing a separate lmms-eval release.

The `qwen2_5_vl` and `llava_hf` model adapters are registered in this release.
The LLaVA adapter supports the Qwen2-based LLaVA-OneVision model used by the
source experiment scripts. Other model adapters, temporary copies, backups,
logs, bytecode, and notebook checkpoints are omitted. The shared evaluation
framework, task sources, and video-loading utilities are retained. Retaining a
task's source does not mean every optional benchmark dependency is part of the
supported environment.

The Qwen adapter accepts canonical internal method names: `fullkv`, `hybridkv`,
`snapkv`, `sparsemm`, `madakv`, `lookm`, `randomkv`, `asymkv_official`,
`hybrid_decouplekv`, `snapkv_decouplekv`, `sparsemm_decouplekv`,
`madakv_decouplekv`, and `lookm_decouplekv`. Display names and CLI aliases are
resolved by the main KVRepack runner. Unselected development variants raise an
error. Core imports use `kvrepack`.

For Qwen, `fullkv` uses the Transformers model with `torch_dtype="auto"` and the
requested device map, matching the opensource adapter used by the formal
FullKV queue. The compressed Qwen methods preserve the artifact adapter's
BF16, device 0, FlashAttention default. HybridKV and Hybrid-DecoupleKV use the
custom model; the remaining compressed methods patch the Transformers model
as in the source adapter.

LLaVA supports `fullkv`, `snapkv`, `sparsemm`, `madakv`, `lookm`, and `hybridkv`
only. Its loading arguments, processor, and frame sampling are retained from
the artifact adapter. The source `llava_ov_ratio.sh` and `llava_ov_hybridkv.sh`
scripts pass `device_map=auto,dtype=bfloat16,attn_implementation=flash_attention_2`.
HybridKV uses the custom OneVision model; FullKV uses the native Transformers
model; the remaining four methods patch that native model. The unrelated
Mistral branches and their bundled import are removed. No LLaVA DecoupleKV
variant is supplied.

Video preprocessing, prompt construction, generation defaults, and task
scoring are retained from the artifact snapshot for both adapters.

Provenance and changes are recorded in `../../provenance/vendor_sources.json`.

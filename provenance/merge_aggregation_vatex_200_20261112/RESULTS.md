# VATEX fixed-group merge comparison

Qwen2.5-VL-7B, 64 frames, original first200, strict 10% total active prefill KV positions.

| Backbone | Aggregation | BLEU-4 | CIDEr | METEOR | R-L | VATEX Avg. | E2E mean (s) | P50 (s) | P95 (s) | Model E2E (s) | Fisher mean (s) | Generation mean (s) | Generated tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybridkv_kvrepack | lookm | 0.219633 | 0.468165 | 0.221619 | 0.427417 | 0.334209 | 8.173605 | 5.066913 | 25.197800 | 2.357584 | 0.000000 | 2.357575 | 16.930000 |
| hybridkv_kvrepack | asymkv | 0.178995 | 0.387725 | 0.207112 | 0.410441 | 0.296068 | 10.017878 | 6.823433 | 26.741837 | 4.185451 | 1.612517 | 2.572934 | 19.845000 |
| snapkv_kvrepack | lookm | 0.216159 | 0.458444 | 0.213408 | 0.426181 | 0.328548 | 7.909557 | 4.726065 | 25.817936 | 2.000925 | 0.000000 | 2.000916 | 16.000000 |
| snapkv_kvrepack | asymkv | 0.181163 | 0.354224 | 0.199085 | 0.403201 | 0.284418 | 9.640830 | 6.391827 | 27.147535 | 3.738625 | 1.615247 | 2.123378 | 19.645000 |

Separate matched parent prefix: original first 114 inputs, three arms measured before the stop. These rows are not a 200-input KVRepack baseline and no prefix CIDEr is mixed with first200 scores.

| Backbone | Aggregation | N | E2E mean (s) | Model E2E (s) | E2E / KVRepack | Model / KVRepack |
|---|---|---:|---:|---:|---:|---:|
| snapkv_kvrepack | kvrepack | 114 | 8.298082 | 1.959640 | 1.000000 | 1.000000 |
| snapkv_kvrepack | lookm | 114 | 8.266790 | 1.964442 | 0.996229 | 1.002451 |
| snapkv_kvrepack | asymkv | 114 | 9.964290 | 3.658765 | 1.200794 | 1.867060 |

# Matched parent first114 quality and latency

All three arms use the same original first 114 VATEX references. CIDEr document frequencies are recomputed on these 114 references and must not be compared directly with the separate first200 table.

| Backbone | Aggregation | N | BLEU-4 | CIDEr | METEOR | R-L | VATEX Avg. | E2E mean (s) | Model E2E (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| snapkv_kvrepack | kvrepack | 114 | 0.234334 | 0.482117 | 0.220275 | 0.427924 | 0.341163 | 8.298082 | 1.959640 |
| snapkv_kvrepack | lookm | 114 | 0.232737 | 0.480605 | 0.217141 | 0.430255 | 0.340184 | 8.266790 | 1.964442 |
| snapkv_kvrepack | asymkv | 114 | 0.202567 | 0.387079 | 0.207555 | 0.411754 | 0.302239 | 9.964290 | 3.658765 |

video read/decode+processor+H2D+Fisher(if AsymKV)+generate+text decode; synchronized GPU; excludes model load/warmup/audit hashing/writes/scoring

Model E2E is Fisher(if needed)+prefill+decode from GPU-resident input to generated token IDs. Full E2E additionally includes native video decode/preprocessing, H2D and text decoding. Model loading, warmup, audit hashing, result writes and CPU scoring are excluded.

LookM/AsymKV labels denote fixed-group aggregation transplants. All arms share KVRepack groups, heads, anchors and raw-K/reapplied-mRoPE position policy. AsymKV retains the released matmul/exp/cardinality-denominator/Vsum attention arithmetic, batched across heads; full-prompt tail32 Fisher is an adaptation of the released chunked probe.

# Matched parent first114 quality and latency

All three arms use the same original first 114 VATEX references. CIDEr document frequencies are recomputed on these 114 references and must not be compared directly with the separate first200 table.

| Backbone | Aggregation | N | BLEU-4 | CIDEr | METEOR | R-L | VATEX Avg. | E2E mean (s) | Model E2E (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| snapkv_kvrepack | kvrepack | 114 | 0.234334 | 0.482117 | 0.220275 | 0.427924 | 0.341163 | 8.298082 | 1.959640 |
| snapkv_kvrepack | lookm | 114 | 0.232737 | 0.480605 | 0.217141 | 0.430255 | 0.340184 | 8.266790 | 1.964442 |
| snapkv_kvrepack | asymkv | 114 | 0.202567 | 0.387079 | 0.207555 | 0.411754 | 0.302239 | 9.964290 | 3.658765 |

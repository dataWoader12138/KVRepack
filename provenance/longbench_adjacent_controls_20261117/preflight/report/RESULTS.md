# Complete adjacent-key controls on LongBench

Status: **complete**.

Primary: cosine over every original sequence-adjacent pair (i,i+1), exactly N-1 per complete input/head/layer, before current-layer RoPE

Equal adjacent-pair weights within each input/head/layer, then equal native-head/layer and prescribed-input weights; no input-length weighting

| Subset | N | Raw K adjacency | Raw K all-pairs | Centered K adjacency | Bias-subtracted K adjacency |
|---|---:|---:|---:|---:|---:|
| hotpotqa | 2 | 0.851808044 | 0.693659806 | 0.459737948 | 0.691770779 |
| triviaqa | 2 | 0.848849329 | 0.690518347 | 0.458615027 | 0.682746193 |
| lcc | 2 | 0.835218901 | 0.687106867 | 0.419736450 | 0.666154224 |

Raw non-self all-pairs; sequence-mean-subtracted raw-K adjacency; projection-bias-subtracted raw-K adjacency. Transformations operate only on captured FP32 copies of native BF16 outputs and never change the model forward.

112 decoder-layer/native-KV-head means over the prescribed inputs per subset, not individual pair cosines or independent observations

Dataset controls use the same fixed model and prescribed original inputs. Centering is a post-hoc geometry probe; subtracting projection bias from rounded native outputs is not a forward pass with the bias disabled. These measurements do not prove merge safety or isolate a unique causal mechanism.

Post-RoPE K and V are supplemental all-adjacent values in the CSV files. No token/pair sampling, truncation, bootstrap or significance testing was performed.

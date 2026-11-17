# Complete adjacent-key controls on LongBench

Status: **complete**.

Primary: cosine over every original sequence-adjacent pair (i,i+1), exactly N-1 per complete input/head/layer, before current-layer RoPE

Equal adjacent-pair weights within each input/head/layer, then equal native-head/layer and prescribed-input weights; no input-length weighting

| Subset | N | Raw K adjacency | Raw K all-pairs | Centered K adjacency | Bias-subtracted K adjacency |
|---|---:|---:|---:|---:|---:|
| hotpotqa | 200 | 0.850862760 | 0.696223896 | 0.453045831 | 0.690903261 |
| triviaqa | 200 | 0.849427582 | 0.693112363 | 0.455759324 | 0.684255466 |
| lcc | 200 | 0.831355668 | 0.672503112 | 0.439373474 | 0.659694761 |

Raw non-self all-pairs; sequence-mean-subtracted raw-K adjacency; projection-bias-subtracted raw-K adjacency. Transformations operate only on captured FP32 copies of native BF16 outputs and never change the model forward.

112 decoder-layer/native-KV-head means over the prescribed inputs per subset, not individual pair cosines or independent observations

Dataset controls use the same fixed model and prescribed original inputs. Centering is a post-hoc geometry probe; subtracting projection bias from rounded native outputs is not a forward pass with the bias disabled. These measurements do not prove merge safety or isolate a unique causal mechanism.

Post-RoPE K and V are supplemental all-adjacent values in the CSV files. No token/pair sampling, truncation, bootstrap or significance testing was performed.

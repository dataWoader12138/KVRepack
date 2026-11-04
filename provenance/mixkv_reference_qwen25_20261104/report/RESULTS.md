# Reference-style full-input 5+5 pilot

Status: **complete**.

Reference-style adaptation of MixKV Figure 1(b), not an exact reproduction: Qwen2.5 text and vision-language models, fixed LongBench qasper and OCRBench inputs. Models, datasets, input lengths and semantic roles differ; comparison does not isolate a within-model modality effect or establish merge safety.

All distinct-token-pair cosine over the full native prefill sequence in each input; self-pairs excluded; signed values retained

Equal KV-head and layer weights within each input, then equal weights for the five prescribed inputs per model; no token-length weighting

One layer/native-KV-head mean over the five inputs; 28*4=112 descriptive means per model; not independent observations

Descriptive fixed 5+5-input pilot only; no confidence intervals, significance tests, or head-as-sample inference

| Quantity | Text model (5 inputs) | Vision-language model (5 inputs) | Vision − text |
|---|---:|---:|---:|
| k_raw | 0.677177 | 0.595588 | -0.081589 |
| k_rope | 0.503055 | 0.579701 | 0.076646 |
| v | 0.095860 | 0.129328 | 0.033468 |

The main figure uses post-RoPE K and V; raw K is provided separately. Self-pairs are excluded; negative values and opposite-direction results are retained.

| Group | Model | Revision | Valid document IDs | Input token min / max / mean |
|---|---|---|---|---|
| text | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | [0, 1, 2, 3, 4] | 3079 / 4533 / 3921.8 |
| vision | Qwen/Qwen2.5-VL-7B-Instruct | cc594898137f460bfe9f0759e9844b3ce807cfb5 | [0, 200, 400, 600, 800] | 64 / 1013 / 433.2 |

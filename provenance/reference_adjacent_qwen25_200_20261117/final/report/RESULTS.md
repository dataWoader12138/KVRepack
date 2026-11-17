# Complete-sequence adjacent-token 200+200 observation

Status: **complete**.

Mean cosine over every immediate original-sequence pair (i,i+1), i=0..N-2; exactly N-1 pairs per input, per native KV head/layer

Complete unpadded prefill sequence including image, text and structural tokens; includes adjacent pairs across modality, image-grid row and other sequence boundaries

Equal adjacent-pair weights within each input/head/layer; equal head/layer weights and then equal weights for 200 inputs per group; no input-length weighting

One layer/native-KV-head mean over 200 inputs; 28*4=112 descriptive means per model, not independent observations

| Quantity | Qwen2.5-7B / qasper | Qwen2.5-VL-7B / OCRBench | Vision − text |
|---|---:|---:|---:|
| k_raw | 0.843658 | 0.751111 | -0.092547 |
| k_rope | 0.835541 | 0.749498 | -0.086043 |
| v | 0.377914 | 0.368567 | -0.009347 |

The main figures use post-RoPE K and V. Raw K is supplemental. Every immediate adjacent pair enters the observation; there is no token/pair subsampling. Signed values and all prescribed inputs are retained.

| Group | Completed inputs | Adjacent pairs total | Input tokens min / max / mean |
|---|---:|---:|---|
| text | 200 | 1042084 | 2094 / 22064 / 5211.42 |
| vision | 200 | 215056 | 61 / 16308 / 1076.28 |

Adjacent-token observation of Qwen2.5 text and vision-language models on fixed LongBench qasper/OCRBench inputs. Models, datasets, sequence lengths and token roles differ; this does not isolate a within-model modality effect or establish merge safety.

Exploratory 95% source-cluster bootstrap; heads and token pairs are not independent samples

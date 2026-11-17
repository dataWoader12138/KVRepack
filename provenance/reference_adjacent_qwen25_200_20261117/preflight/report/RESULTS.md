# Complete-sequence adjacent-token 2+2 observation

Status: **complete**.

Mean cosine over every immediate original-sequence pair (i,i+1), i=0..N-2; exactly N-1 pairs per input, per native KV head/layer

Complete unpadded prefill sequence including image, text and structural tokens; includes adjacent pairs across modality, image-grid row and other sequence boundaries

Equal adjacent-pair weights within each input/head/layer; equal head/layer weights and then equal weights for 2 inputs per group; no input-length weighting

One layer/native-KV-head mean over 2 inputs; 28*4=112 descriptive means per model, not independent observations

| Quantity | Qwen2.5-7B / qasper | Qwen2.5-VL-7B / OCRBench | Vision − text |
|---|---:|---:|---:|
| k_raw | 0.845194 | 0.770827 | -0.074367 |
| k_rope | 0.837210 | 0.767690 | -0.069520 |
| v | 0.383862 | 0.403773 | 0.019911 |

The main figures use post-RoPE K and V. Raw K is supplemental. Every immediate adjacent pair enters the observation; there is no token/pair subsampling. Signed values and all prescribed inputs are retained.

| Group | Completed inputs | Adjacent pairs total | Input tokens min / max / mean |
|---|---:|---:|---|
| text | 2 | 7517 | 3428 / 4091 / 3759.50 |
| vision | 2 | 134 | 64 / 72 / 68.00 |

Adjacent-token observation of Qwen2.5 text and vision-language models on fixed LongBench qasper/OCRBench inputs. Models, datasets, sequence lengths and token roles differ; this does not isolate a within-model modality effect or establish merge safety.

Descriptive pilot only; no confidence intervals or head-as-sample inference

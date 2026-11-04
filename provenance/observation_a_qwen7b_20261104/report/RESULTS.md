# Adjacent visual/text observations

Status: **complete**.

Prescribed sample: first 200 rows per task (400 documents); records are never replaced or deduplicated by video.
Each document averages its KV heads and layers. Task means weight documents equally. Confidence intervals resample source-video clusters with every member document.
This is adjacent-pair geometry only. It does not establish a modality effect, a distant-pair contrast, or safe merging. Negative cosines are retained.

| Task | Quantity | Paired documents / completed | Videos | Visual | Text | Visual − text (95% CI) | Positive video clusters |
|---|---|---:|---:|---:|---:|---|---:|
| nextqa_oe_test | k_raw | 200 / 200 | 184 | 0.738970 | 0.807565 | -0.068595 [-0.074279, -0.062860] | 7/184 |
| nextqa_oe_test | v | 200 / 200 | 184 | 0.315928 | 0.370496 | -0.054568 [-0.066217, -0.042985] | 43/184 |

nextqa_oe_test: 0 completed records excluded from paired analysis because M=0; 0 prescribed IDs lack a valid completed record.

| vatex_test | k_raw | 200 / 200 | 200 | 0.739972 | 0.784491 | -0.044519 [-0.049584, -0.039596] | 24/200 |
| vatex_test | v | 200 / 200 | 200 | 0.314344 | 0.341388 | -0.027044 [-0.036908, -0.016995] | 67/200 |

vatex_test: 0 completed records excluded from paired analysis because M=0; 0 prescribed IDs lack a valid completed record.

Positive-video counts use the sign of each source video's mean paired difference, without clipping cosines.
Head/layer CSVs contain descriptive means; task-level intervals are the cluster-bootstrap inference. Optional RoPE K summaries are position-dependent and secondary.
Model/frame provenance is included only when supplied in the records. No settings are inferred.


# Raw-K top200 adjacent-token comparison: VATEX versus LongBench

Status: **complete**.

Per input/layer/native KV head: rank all N-1 immediate sequence-pair raw-K cosines before RoPE; take exactly the highest200 and average them

All consecutive positions in each original complete prefill sequence, including text/video/structural and sequence-boundary pairs; no pairing across inputs

Equal weights for the200 selected pairs within each input/head/layer, then equal head/layer and 2 input weights; no length weighting

| Metric | Qwen2.5-7B / LongBench | Qwen2.5-VL-7B / VATEX | Video minus text |
|---|---:|---:|---:|
| raw_k_top200 | 0.949363631 | 0.969155656 | +0.019792025 |
| k_raw_all_adjacent | 0.845194321 | 0.744133398 | -0.101060923 |
| k_rope_all_adjacent | 0.837209962 | 0.743930768 | -0.093279194 |
| v_all_adjacent | 0.383862107 | 0.314822609 | -0.069039498 |

k_raw_all_adjacent, k_rope_all_adjacent and v_all_adjacent cover all N-1 pairs; neither post-RoPE K nor V is measured on the raw-K-selected200 positions

112 layer/native-KV-head means over the prescribed inputs per model, not individual pair cosines or independent observations

Selected upper-tail raw-K similarity, not the average over all token pairs. Equal selected counts do not equalize candidate-pool sizes. Models, datasets, lengths and token roles differ; no isolated modality or merge-safety claim.

No bootstrap or significance test was performed. Input rows are the original prescribed prefix; repeated sources remain.

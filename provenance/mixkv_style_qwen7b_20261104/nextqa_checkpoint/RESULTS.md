# MixKV-style all-pairs observation

Status: **complete**.

Adaptation of MixKV Figure 1(b), not an exact paper reproduction: same Qwen2.5-VL model across conditions; primary visual/text subset lengths are matched, but full forward prompt lengths differ. Source modality, context and token role remain confounded. Cosine alone does not establish merge safety.

Primary: equal token-count subsets in both modalities, all distinct-token-pair cosine within each draw, then equal draw weights. Supplement: entire predefined pools. Self-pairs excluded; negative cosines retained

One decoder-layer/native-KV-head mean over all prescribed rows; layers*KV-heads points per condition/task, not independent samples

Task estimates first average native KV heads and layers within each row, then weight prescribed rows equally. Confidence intervals resample source videos with all member rows, using shared draws across comparisons and metrics.

| Task | Comparison | Metric | Rows / videos | Left | Right | Left − right (95% CI) |
|---|---|---|---:|---:|---:|---|
| nextqa_oe_test | matched_visual_vs_text | k_raw | 200 / 184 | 0.639339 | 0.732235 | -0.092896 [-0.095952, -0.089855] |
| nextqa_oe_test | matched_visual_vs_text | v | 200 / 184 | 0.134056 | 0.218764 | -0.084708 [-0.089791, -0.079829] |
| nextqa_oe_test | matched_visual_vs_text | k_rope | 200 / 184 | 0.571095 | 0.699581 | -0.128486 [-0.132313, -0.124729] |
| nextqa_oe_test | matched_text_context | k_raw | 200 / 184 | 0.732235 | 0.728690 | 0.003545 [0.002352, 0.004709] |
| nextqa_oe_test | matched_text_context | v | 200 / 184 | 0.218764 | 0.245177 | -0.026413 [-0.028065, -0.024747] |
| nextqa_oe_test | matched_text_context | k_rope | 200 / 184 | 0.699581 | 0.700901 | -0.001320 [-0.002526, -0.000122] |
| nextqa_oe_test | full_prompt | k_raw | 200 / 184 | 0.638312 | 0.627331 | 0.010981 [0.008667, 0.013290] |
| nextqa_oe_test | full_prompt | v | 200 / 184 | 0.133436 | 0.134781 | -0.001345 [-0.003251, 0.000567] |
| nextqa_oe_test | full_prompt | k_rope | 200 / 184 | 0.569818 | 0.583576 | -0.013758 [-0.016132, -0.011437] |
| nextqa_oe_test | within_multimodal | k_raw | 200 / 184 | 0.639456 | 0.732261 | -0.092804 [-0.095942, -0.089716] |
| nextqa_oe_test | within_multimodal | v | 200 / 184 | 0.134263 | 0.218790 | -0.084527 [-0.089571, -0.079659] |
| nextqa_oe_test | within_multimodal | k_rope | 200 / 184 | 0.571111 | 0.699603 | -0.128491 [-0.132340, -0.124675] |
| nextqa_oe_test | text_context | k_raw | 200 / 184 | 0.732261 | 0.728718 | 0.003543 [0.002356, 0.004706] |
| nextqa_oe_test | text_context | v | 200 / 184 | 0.218790 | 0.245214 | -0.026424 [-0.028072, -0.024765] |
| nextqa_oe_test | text_context | k_rope | 200 / 184 | 0.699603 | 0.700925 | -0.001322 [-0.002528, -0.000123] |

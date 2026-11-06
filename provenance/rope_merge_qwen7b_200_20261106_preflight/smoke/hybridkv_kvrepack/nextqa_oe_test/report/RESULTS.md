# Key merge-space ablation

Status: **complete**; hybridkv_kvrepack / nextqa_oe_test; first 1 inputs.

Paired key merge-space ablation; raw-reapply minus direct post-RoPE. Inputs, selected heads/groups, parameters and merge budgets are controlled by the recorded adapter fairness gate. Downstream V/cache/retrieval can change after different K choices.

NextQA original WUPS@0 uses 0-100 points. VATEX uses original raw corpus units and the same selected reference corpus/DF; no per-row CIDEr substitute, CIDEr CI or composite score.

| Metric | Raw + reapply RoPE | Direct post-RoPE | Raw − post (95% paired CI for WUPS) |
|---|---:|---:|---|
| WUPS | 100.000000 | 100.000000 | 0.000000 |

reached_length_limit counts any output at max_new_tokens, including EOS exactly at the cap; capped_without_eos counts only capped outputs without EOS

| Arm | EOS | At cap | Capped without EOS | Mean generated tokens |
|---|---:|---:|---:|---:|
| raw_reapply | 1 | 0 | 0 | 2.00 |
| post_rope | 1 | 0 | 0 | 2.00 |

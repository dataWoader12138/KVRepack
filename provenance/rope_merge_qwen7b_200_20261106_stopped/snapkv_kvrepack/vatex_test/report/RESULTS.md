# Key merge-space ablation

Status: **complete**; snapkv_kvrepack / vatex_test; first 200 inputs.

Paired key merge-space ablation; raw-reapply minus direct post-RoPE. Inputs, selected heads/groups, parameters and merge budgets are controlled by the recorded adapter fairness gate. Downstream V/cache/retrieval can change after different K choices.

NextQA original WUPS@0 uses 0-100 points. VATEX uses original raw corpus units and the same selected reference corpus/DF; no per-row CIDEr substitute, CIDEr CI or composite score.

| Metric | Raw + reapply RoPE | Direct post-RoPE | Raw − post (95% paired CI for WUPS) |
|---|---:|---:|---|
| Bleu_4 | 0.219294 | 0.212855 | 0.006439 |
| METEOR | 0.214389 | 0.211665 | 0.002724 |
| ROUGE_L | 0.425972 | 0.423932 | 0.002039 |
| CIDEr | 0.463033 | 0.451014 | 0.012019 |

reached_length_limit counts any output at max_new_tokens, including EOS exactly at the cap; capped_without_eos counts only capped outputs without EOS

| Arm | EOS | At cap | Capped without EOS | Mean generated tokens |
|---|---:|---:|---:|---:|
| raw_reapply | 200 | 0 | 0 | 16.02 |
| post_rope | 200 | 0 | 0 | 16.13 |

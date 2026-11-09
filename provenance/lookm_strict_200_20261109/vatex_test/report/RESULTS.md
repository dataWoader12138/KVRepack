# Single video benchmark result

Complete: `lookm` / `vatex_test`, original first 200 inputs.

**Strict total prefill including recent positions (v2)**; policy `total_prefill_including_recent_v2`; strict_budget_passed=True.

Single completed method/task, exact original first200 inputs; no matrix requirement, contrast, delta or inferred comparison result. Fixed200-input prefix result.

Native NextQA WUPS@0 in 0–100 points; VATEX Bleu_4/METEOR/ROUGE_L/CIDEr in original corpus units using only the same selected200 reference videos for CIDEr DF; no composite or per-sentence CIDEr substitute

| Metric | Native score |
|---|---:|
| Bleu_4 | 0.20829515 |
| METEOR | 0.21383849 |
| ROUGE_L | 0.41671988 |
| CIDEr | 0.45664620 |

EOS: 200/200; at cap: 0; capped without EOS: 0.

At-cap includes EOS exactly at max_new_tokens; capped_without_eos counts capped outputs lacking EOS

Logical active prefill/decode KV slots; excludes backing history/offload stores and is not total VRAM

Configured budget ratio: 0.1; actual active-KV retained fraction min/mean/max: 0.09990917/0.09997525/0.10000000.

Scores describe this fixed prefix only.

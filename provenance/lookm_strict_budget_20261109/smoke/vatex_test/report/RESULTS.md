# Single video benchmark result

Complete: `lookm` / `vatex_test`, original first 1 inputs.

**Strict total prefill including recent positions (v2)**; policy `total_prefill_including_recent_v2`; strict_budget_passed=True.

Single completed method/task, exact original first1 inputs; no matrix requirement, contrast, delta or inferred comparison result. One-input smoke verifies the pipeline only; it is not a formal200 result or a basis for quality/bootstrap conclusions.

Native NextQA WUPS@0 in 0–100 points; VATEX Bleu_4/METEOR/ROUGE_L/CIDEr in original corpus units using only the same selected1 reference videos for CIDEr DF; no composite or per-sentence CIDEr substitute

| Metric | Native score |
|---|---:|
| Bleu_4 | 0.58282340 |
| METEOR | 0.37398949 |
| ROUGE_L | 0.72462203 |
| CIDEr | 0.00000000 |

EOS: 1/1; at cap: 0; capped without EOS: 0.

At-cap includes EOS exactly at max_new_tokens; capped_without_eos counts capped outputs lacking EOS

Logical active prefill/decode KV slots; excludes backing history/offload stores and is not total VRAM

Configured budget ratio: 0.1; actual active-KV retained fraction min/mean/max: 0.09993837/0.09993837/0.09993837.

Scores describe this fixed prefix only.

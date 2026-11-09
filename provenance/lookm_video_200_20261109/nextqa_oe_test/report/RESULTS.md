# Single video benchmark result

Complete: `lookm` / `nextqa_oe_test`, original first 200 inputs.

Single completed method/task, exact original first200 inputs; no matrix requirement, contrast, delta or inferred comparison result

Native NextQA WUPS@0 in 0–100 points; VATEX Bleu_4/METEOR/ROUGE_L/CIDEr in original corpus units using only the same selected200 reference videos for CIDEr DF; no composite or per-sentence CIDEr substitute

| Metric | Native score |
|---|---:|
| WUPS | 33.85714404 |

EOS: 200/200; at cap: 0; capped without EOS: 0.

At-cap includes EOS exactly at max_new_tokens; capped_without_eos counts capped outputs lacking EOS

Logical active prefill/decode KV slots; excludes backing history/offload stores and is not total VRAM

Configured budget ratio: 0.1; actual active-KV retained fraction min/mean/max: 0.10135135/0.10298789/0.10373832.

Scores describe this fixed prefix only.

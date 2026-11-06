# Paired K merge-space comparison

Completed: 1 original inputs per task, 8 generations.

Positive differences favor post-RoPE K aggregation. WUPS uses 0-100 points; VATEX uses raw corpus scores.

| Task | Method | Metric | Raw-reapply | Post-RoPE | Post − raw | Paired 95% interval |
|---|---|---|---:|---:|---:|---|
| nextqa_oe_test | snapkv_kvrepack | WUPS | 100.000000 | 100.000000 | +0.000000 | — |
| nextqa_oe_test | hybridkv_kvrepack | WUPS | 100.000000 | 100.000000 | +0.000000 | — |
| vatex_test | snapkv_kvrepack | Bleu_4 | 0.582823 | 0.582823 | +0.000000 | — |
| vatex_test | snapkv_kvrepack | METEOR | 0.337669 | 0.337669 | +0.000000 | — |
| vatex_test | snapkv_kvrepack | ROUGE_L | 0.724622 | 0.724622 | +0.000000 | — |
| vatex_test | snapkv_kvrepack | CIDEr | 0.000000 | 0.000000 | +0.000000 | — |
| vatex_test | hybridkv_kvrepack | Bleu_4 | 0.340623 | 0.340623 | +0.000000 | — |
| vatex_test | hybridkv_kvrepack | METEOR | 0.260567 | 0.260567 | +0.000000 | — |
| vatex_test | hybridkv_kvrepack | ROUGE_L | 0.473918 | 0.473918 | +0.000000 | — |
| vatex_test | hybridkv_kvrepack | CIDEr | 0.000000 | 0.000000 | +0.000000 | — |

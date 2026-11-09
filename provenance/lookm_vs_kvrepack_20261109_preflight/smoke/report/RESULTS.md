# Video benchmark comparison

Complete: 1 inputs per task/method, 4 generations.

Differences are second matrix condition minus first; signed differences are retained. Configured budgets and actual paired input bytes are checked. Each method retains its native cache/compression semantics; cache audits are recorded without assuming identical actual cache lengths across algorithms.

Native NextQA WUPS@0 in 0–100 points; VATEX Bleu_4/METEOR/ROUGE_L/CIDEr in original raw corpus units with the same selected reference corpus/DF. No composite, per-sentence CIDEr substitute or VATEX bootstrap interval.

| Task | Metric | First condition | Second condition | Second − first | Paired 95% CI |
|---|---|---:|---:|---:|---|
| nextqa_oe_test | WUPS | 100.000000 | 100.000000 | 0.000000 | — |
| vatex_test | Bleu_4 | 0.708588 | 0.340623 | -0.367965 | — |
| vatex_test | METEOR | 0.506089 | 0.260567 | -0.245522 | — |
| vatex_test | ROUGE_L | 0.914086 | 0.473918 | -0.440168 | — |
| vatex_test | CIDEr | 0.000000 | 0.000000 | 0.000000 | — |

First: `lookm`. Second: `hybridkv_kvrepack`.

At-cap includes EOS exactly at the cap; capped_without_eos counts only capped outputs lacking EOS.

NextQA intervals resample source videos with all member questions; these fixed-prefix comparisons do not establish general superiority.

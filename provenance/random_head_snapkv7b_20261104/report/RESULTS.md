# Random-head identity ablation

Status: **complete**; first 20 original rows per task; seeds 0,1,2.

Only global head identities/ranking are changed; original parameters, total selected/effective heads, candidate bytes, merge work, and prefill cache shapes are paired. Random ranking is fixed per model/seed. Layer allocations can differ. Three seeds are not additional independent videos.

WUPS@0 is 0-100; VATEX uses raw corpus scores with the same selected reference pool. No sentence-BLEU or per-row CIDEr replacement; no CIDEr confidence interval or composite score is reported.

| Task | Metric | Similarity | Random seed 0 | Seed 1 | Seed 2 | Random mean ± sample SD | Similarity − random mean |
|---|---|---:|---:|---:|---:|---|---|
| nextqa_oe_test | WUPS | 32.016623 | 31.817793 | 32.016623 | 32.016623 | 31.950346 ± 0.114795 | 0.066277 [0.000000, 0.198830] |
| vatex_test | Bleu_4 | 0.292531 | 0.290201 | 0.295682 | 0.309831 | 0.298571 ± 0.010129 | -0.006040 |
| vatex_test | METEOR | 0.249811 | 0.249394 | 0.250159 | 0.254448 | 0.251334 ± 0.002724 | -0.001523 |
| vatex_test | ROUGE_L | 0.449622 | 0.450307 | 0.455429 | 0.466309 | 0.457348 ± 0.008172 | -0.007727 |
| vatex_test | CIDEr | 0.505653 | 0.504543 | 0.521529 | 0.539502 | 0.521858 ± 0.017482 | -0.016205 |

Sample SD describes variability across exactly three fixed random rankings (ddof=1). WUPS intervals use paired source-video bootstrap, conditional on these three seeds; heads, seed×input rows and token pairs are not independent samples. VATEX corpus metric differences are descriptive.


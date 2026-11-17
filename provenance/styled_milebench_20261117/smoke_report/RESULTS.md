# Qwen7B styled image experiments

Scope: selected_or_smoke_matrix_not_full_experiment; 16 runs; 16 formal generations.

Spot: original ROUGE-L F; MMCoQA: original substring Accuracy. Scores are on a 0–100 scale.

| Arm | Task | N | Score | Mean E2E (s) | P95 E2E (s) | Mean physical prefill ratio |
|---|---|---:|---:|---:|---:|---:|
| hybridkv_asymkv | Spot-the-Diff | 1 | 27.272727 | 1.408511 | 1.408511 | 0.162624 |
| hybridkv_asymkv | MMCoQA | 1 | 100.000000 | 2.844394 | 2.844394 | 0.107550 |
| hybridkv_lookm | Spot-the-Diff | 1 | 25.000000 | 1.116439 | 1.116439 | 0.162624 |
| hybridkv_lookm | MMCoQA | 1 | 100.000000 | 2.043707 | 2.043707 | 0.107550 |
| hybridkv_kvrepack | Spot-the-Diff | 1 | 25.000000 | 1.076554 | 1.076554 | 0.162624 |
| hybridkv_kvrepack | MMCoQA | 1 | 100.000000 | 2.012782 | 2.012782 | 0.107550 |
| snapkv_asymkv | Spot-the-Diff | 1 | 30.000000 | 0.689374 | 0.689374 | 0.157221 |
| snapkv_asymkv | MMCoQA | 1 | 0.000000 | 2.348838 | 2.348838 | 0.106977 |
| snapkv_lookm | Spot-the-Diff | 1 | 16.666666 | 0.625564 | 0.625564 | 0.157221 |
| snapkv_lookm | MMCoQA | 1 | 100.000000 | 2.022328 | 2.022328 | 0.106977 |
| snapkv_kvrepack | Spot-the-Diff | 1 | 17.391304 | 0.592733 | 0.592733 | 0.157221 |
| snapkv_kvrepack | MMCoQA | 1 | 100.000000 | 1.624425 | 1.624425 | 0.106977 |
| asymkv | Spot-the-Diff | 1 | 30.000000 | 0.489864 | 0.489864 | 1.000000 |
| asymkv | MMCoQA | 1 | 100.000000 | 3.658505 | 3.658505 | 0.306091 |
| lookm | Spot-the-Diff | 1 | 27.272727 | 0.652744 | 0.652744 | 0.157221 |
| lookm | MMCoQA | 1 | 0.000000 | 1.920311 | 1.920311 | 0.106977 |

Primary E2E includes image I/O, JPEG conversion, preprocessing, H2D, Fisher where applicable, generation, and text decode. Model loading, warmup, hashes, writes and scoring are excluded.
Standalone AsymKV Fisher is inside original generation; its separate Fisher time is unmeasured (empty CSV), not zero.
Original AsymKV chunk-boundary omissions and original budget/cache differences are retained and reported, not repaired or filtered.

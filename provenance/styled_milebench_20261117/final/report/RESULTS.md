# Qwen7B styled image experiments

Scope: full_16_run_annotation_matrix; 16 runs; 3200 formal generations.

Spot: original ROUGE-L F; MMCoQA: original substring Accuracy. Scores are on a 0–100 scale.

| Arm | Task | N | Score | Mean E2E (s) | P95 E2E (s) | Mean physical prefill ratio |
|---|---|---:|---:|---:|---:|---:|
| hybridkv_asymkv | Spot-the-Diff | 200 | 27.099023 | 1.513908 | 1.730131 | 0.163679 |
| hybridkv_asymkv | MMCoQA | 200 | 53.500000 | 2.897738 | 5.004235 | 0.116938 |
| hybridkv_lookm | Spot-the-Diff | 200 | 29.229593 | 0.798685 | 1.116838 | 0.163679 |
| hybridkv_lookm | MMCoQA | 200 | 59.500000 | 2.293849 | 6.283207 | 0.116938 |
| hybridkv_kvrepack | Spot-the-Diff | 200 | 29.086251 | 0.987383 | 1.289802 | 0.163679 |
| hybridkv_kvrepack | MMCoQA | 200 | 63.500000 | 1.499921 | 3.085907 | 0.116938 |
| snapkv_asymkv | Spot-the-Diff | 200 | 27.781507 | 0.978997 | 1.109952 | 0.157204 |
| snapkv_asymkv | MMCoQA | 200 | 48.000000 | 2.355037 | 14.960004 | 0.114926 |
| snapkv_lookm | Spot-the-Diff | 200 | 29.071116 | 0.578007 | 0.789289 | 0.157204 |
| snapkv_lookm | MMCoQA | 200 | 57.500000 | 1.120930 | 1.996260 | 0.114926 |
| snapkv_kvrepack | Spot-the-Diff | 200 | 28.955760 | 0.557183 | 0.777553 | 0.157204 |
| snapkv_kvrepack | MMCoQA | 200 | 59.000000 | 1.110574 | 1.933911 | 0.114926 |
| asymkv | Spot-the-Diff | 200 | 28.769158 | 0.719469 | 0.969559 | 1.000000 |
| asymkv | MMCoQA | 200 | 49.000000 | 2.688473 | 5.657299 | 0.456298 |
| lookm | Spot-the-Diff | 200 | 28.739377 | 0.740479 | 0.807348 | 0.157204 |
| lookm | MMCoQA | 200 | 45.000000 | 1.323057 | 3.025298 | 0.114926 |

Primary E2E includes image I/O, JPEG conversion, preprocessing, H2D, Fisher where applicable, generation, and text decode. Model loading, warmup, hashes, writes and scoring are excluded.
Standalone AsymKV Fisher is inside original generation; its separate Fisher time is unmeasured (empty CSV), not zero.
Original AsymKV chunk-boundary omissions and original budget/cache differences are retained and reported, not repaired or filtered.

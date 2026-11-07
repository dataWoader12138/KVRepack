# HybridKV-KVRepack NextQA Random-head evaluation

Status: **complete**. Exactly original NextQA rows0..199; four conditions; no VATEX or automatic expansion.

Unchanged native NextQA WUPS@0 aggregate in 0-100 points; negative paired differences are retained

Hybrid Global Random-head changes head identities and layer allocation under the same 56-of-112 global quota and unchanged native per-head retained budgets. Heterogeneous head budgets mean total removed-token work may differ as a consequence of selected identities; work is reported, not forced equal. Three seeds are not additional independent inputs; seed sample SD and input-cluster CI describe different uncertainty. This prefix alone does not establish general superiority.

| Condition | WUPS | Similarity − random (95% paired CI) |
|---|---:|---|
| similarity | 33.219903 | — |
| random_global_seed0 | 33.122167 | 0.097737 [-0.842551, 1.344120] |
| random_global_seed1 | 32.799673 | 0.420231 [-0.796044, 1.860325] |
| random_global_seed2 | 33.342348 | -0.122444 [-1.576544, 1.290349] |

Random mean ± sample SD (ddof=1): **33.088063 ± 0.272940**.
Similarity − random mean: **0.131841**; 95% paired interval: [-0.7788084976530026, 1.3534473617525715].
Each condition has 200 inputs. Bootstrap resamples 184 source videos with all member questions, using common draws across conditions and the fixed three seeds.

| Condition | Mean removed tokens | Total removed tokens | Inputs differing from similarity |
|---|---:|---:|---:|
| similarity | 206339.105 | 41267821 | 0 |
| random_global_seed0 | 195490.295 | 39098059 | 200 |
| random_global_seed1 | 202084.690 | 40416938 | 200 |
| random_global_seed2 | 175985.215 | 35197043 | 200 |

Removed-token totals may differ because the fixed native per-head retained budgets are heterogeneous. This is an identity-selection consequence, not an extra budget intervention.


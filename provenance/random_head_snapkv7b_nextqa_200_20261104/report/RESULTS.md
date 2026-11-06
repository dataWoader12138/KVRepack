# NextQA Random-head prefix evaluation

Status: **complete**. Exactly original NextQA rows0..199; four conditions; no VATEX or automatic expansion.

Unchanged native NextQA WUPS@0 aggregate in 0-100 points; negative paired differences are retained

Global Random-head changes head identities and layer allocation under the same global quotas and merge work. Three seeds are not additional independent inputs; sample SD and input-cluster CI describe different uncertainty. This prefix alone does not establish general superiority.

| Condition | WUPS | Similarity − random (95% paired CI) |
|---|---:|---|
| similarity | 32.917681 | — |
| random_global_seed0 | 32.450740 | 0.466941 [-0.020255, 1.416900] |
| random_global_seed1 | 33.116603 | -0.198922 [-0.781282, 0.147509] |
| random_global_seed2 | 32.865783 | 0.051898 [-0.039178, 0.181910] |

Random mean ± sample SD (ddof=1): **32.811042 ± 0.336289**.
Similarity − random mean: **0.106639**; 95% paired interval: [-0.1918800705795446, 0.5013844817400103].
Each condition has 200 inputs. Bootstrap resamples 184 source videos with all member questions, using common draws across conditions and the fixed three seeds.


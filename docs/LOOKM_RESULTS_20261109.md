# LookM视频基线结果（2026-11-09）

Qwen2.5-VL-7B-Instruct、64帧、原生LookM、名义10%预算；两个任务均取原始前200行。用户已明确无需重跑HybridKV + KVRepack，因此仅补LookM；原双方法矩阵与119条Hybrid部分记录已停止，不混入本结果。复现命令及预算公式见[协议](LOOKM_COMPARISON.md)。

| 数据集 | 样本数 | 指标 | LookM | 状态 |
|---|---:|---|---:|---|
| NExTQA | 200 | WUPS@0（0–100） | **33.857144** | 完成 |
| VATEX | 200（目标） | BLEU4 / METEOR / ROUGE-L / CIDEr | — | 生成中，未评分 |

NExTQA所有200次生成均正常EOS、无长度截断。原任务评分与独立逐条重算完全一致；每头缓存保留`floor(.1*N)+32`，平均活动K/V槽位比例为10.298789%。该比例包含原32位置近期窗口，不能称总显存严格10%。

此前同设置的HybridKV + KVRepack NExTQA200记录为33.219903；LookM数值高0.637241分。此处使用既有Hybrid结果，本次未为对照完成任何新的Hybrid200条；该数值差不等于已证明统计优越。

[原任务报告](../provenance/lookm_nextqa_200_20261109/report/RESULTS.md) · [逐问题分数CSV](../provenance/lookm_nextqa_200_20261109/report/scores.csv) · [机器可读汇总](../provenance/lookm_nextqa_200_20261109/report/summary.json) · [独立复算](../provenance/lookm_nextqa_200_20261109/independent_nextqa_score.json)。

VATEX正在独立LookM进程中生成；本页不会把前缀或单条smoke分数当作200条corpus指标，也不将未评分项填0。完成后追加四项原始corpus分数及实际耗时。

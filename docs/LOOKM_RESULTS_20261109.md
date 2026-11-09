# LookM旧版视频结果（2026-11-09，非严格10%）

**本页是旧版“10%＋32近期位置”结果，不能作为严格10%预算的结果或公平对照。** 用户已要求修正，新代码将近期窗口计入总额度；见[修复说明](LOOKM_STRICT_BUDGET.md)。新正式200条分数尚未重跑。以下原始分数仅作历史保留，不能直接沿用为新结果。

**两任务各200条已完成，共400次LookM生成及原任务评分。** 模型为Qwen2.5-VL-7B-Instruct，64帧、名义10%预算。按用户最新要求，HybridKV + KVRepack重跑已终止，保留119条部分记录但不用于本结果；未恢复任何旧实验。[复现协议与单方法启动脚本](LOOKM_COMPARISON.md)。

| 数据集 | 原始样本数 | 指标 | LookM |
|---|---:|---|---:|
| NExTQA | 200 | WUPS@0（0–100） | **33.857144** |
| VATEX | 200 | BLEU-4 | **0.219333** |
| VATEX | 200 | METEOR | **0.219471** |
| VATEX | 200 | ROUGE-L | **0.432950** |
| VATEX | 200 | CIDEr | **0.472929** |

[五项分数CSV](lookm_scores_20261109.csv) · [全部运行证据与400条预测归档](../provenance/lookm_video_200_20261109/README.md)。VATEX使用原corpus评分及同200条参考计算CIDEr DF；不对四指标作复合平均。

两任务各200次均正常EOS，无达到生成上限或无EOS截断；NextQA最多128/VATEX最多64新token。NextQA原任务报告与独立200条逐次WUPS复算完全一致。每头活动prefill缓存为`floor(.1*N)+32`，平均活动槽位比例分别为10.298789%和10.304682%；额外窗口来自原LookM，逻辑K/V保留量不等于总显存严格10%。原仓库LookM七个关键符号AST与迁入版相同，CPU wrapper检查保持原返回对象和预算语义；未替换为另一种LookM实现。

生成墙钟分别约13分33秒、28分36秒，合计约42分8秒；原任务CPU计分分别9.380秒和8.930秒。计时按各LookM worker起止，不包含已停止的Hybrid工作或前期准备。GPU工作已结束，不自动扩大样本。

## 已有Hybrid结果的数值参考（没有重跑）

| 指标 | LookM（本次） | HybridKV + KVRepack（已有记录） | LookM−已有记录 |
|---|---:|---:|---:|
| NExTQA WUPS | 33.857144 | 33.219903 | +0.637241 |
| VATEX BLEU-4 | 0.219333 | 0.228583 | −0.009251 |
| VATEX METEOR | 0.219471 | 0.224439 | −0.004968 |
| VATEX ROUGE-L | 0.432950 | 0.433812 | −0.000862 |
| VATEX CIDEr | 0.472929 | 0.488914 | −0.015986 |

[参考对照CSV](lookm_vs_existing_hybrid_20261109.csv)。NextQA参考来自11月6日完整200条，已逐条确认本次LookM与其输入张量hash/shape/dtype及原始行/视频身份全部相同，见[输入配对审计](../provenance/lookm_video_200_20261109/sources/nextqa_existing_hybrid_input_pairing.json)。数值差不等于统计优越证明。

VATEX参考来自7月5日已完成200条的日志及保留结果快照；7B、64帧、名义.1、task cap64等宏观协议一致，旧前97条字幕与11月已存raw记录完全相同。历史完整输入tensor SHA、检查点revision、repetition penalty和score mode证据不全，历史seed也与本轮不同。因此只列为历史参考，不能称本轮严格配对重跑，见[参考来源审计](../provenance/lookm_video_200_20261109/sources/lookm_existing_hybrid_reference_20261109.json)。

[NextQA原报告](../provenance/lookm_video_200_20261109/nextqa_oe_test/report/RESULTS.md) · [NextQA逐题CSV](../provenance/lookm_video_200_20261109/nextqa_oe_test/report/scores.csv) · [VATEX原报告](../provenance/lookm_video_200_20261109/vatex_test/report/RESULTS.md) · [VATEX分数CSV](../provenance/lookm_video_200_20261109/vatex_test/report/task_scores.csv)。

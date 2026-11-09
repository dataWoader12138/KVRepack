# LookM严格10%预算重跑结果（2026-11-09）

**两任务各200条、共400次全新生成与原任务评分已完成，并通过独立执行和评分复核。** 模型Qwen2.5-VL-7B-Instruct，64帧；仅重跑LookM，没有恢复Hybrid或RoPE。源代码冻结为`bc8ebee`，严格预算及源码修正见[实现说明](LOOKM_STRICT_BUDGET.md)。

| 数据集 | 原始样本数 | 原任务指标 | 严格预算LookM |
|---|---:|---|---:|
| NExTQA | 200 | WUPS@0（0–100） | **31.111145** |
| VATEX | 200 | BLEU-4 | **0.208295** |
| VATEX | 200 | METEOR | **0.213838** |
| VATEX | 200 | ROUGE-L | **0.416720** |
| VATEX | 200 | CIDEr | **0.456646** |

[下载五项完整精度CSV](lookm_strict_scores_20261109.csv) · [原始报告与400条预测归档](../provenance/lookm_strict_200_20261109/README.md)。VATEX使用原corpus四指标、同200条参考构建CIDEr DF，不作复合平均。数据均为原始行0..199，未随机抽样、替换或过滤。

## 预算与执行检查

每条输入长度N对应每个原生KV head的总prefill容量`C=floor(.1*N)`，32个recent位置计入C，历史只选`C−min(32,C)`。所有400条、28层×4个KV头的实际缓存长度均通过检查，共44,800个逐样本head实例，最大保留比例均为10%。

| 任务 | 平均实际prefill比例 | 最小比例 | 最大比例 |
|---|---:|---:|---:|
| NExTQA | 9.995698% | 9.989485% | 10.000000% |
| VATEX | 9.997525% | 9.990917% | 10.000000% |

400次生成全部正常EOS，达到长度上限及无EOS截断均为0。生成上限NExTQA128、VATEX64；greedy、BF16/FlashAttention2、seed42、repetition penalty1.05。最终cache扣除新解码token增量后，与捕获的prefill容量逐层互验。预算控制针对prefill；新生成token按原规则追加，活动K/V保留量不等于总VRAM。

[独立执行审计](../provenance/lookm_strict_200_20261109/independent_execution_check.json)为passed、0问题，核对原始Arrow行/视频身份、模型与来源哈希、真实缓存尺寸和全部新样本覆盖。与旧记录只核对输入一致性，不复用其预测。

[独立评分复核](../provenance/lookm_strict_200_20261109/independent_scoring_check.json)为passed：200次WUPS不复用报告预测缓存逐次重算，VATEX四项原corpus指标也重新计算；逐题WUPS、五项最终指标、缓存CSV及生成统计的最大绝对误差均为0，正式报告文件SHA前后不变。独立脚本随归档提供。

两worker生成墙钟分别为804.727秒（约13分25秒）与1727.497秒（约28分47秒）。控制器含输入处理、加载和单任务计分的端到端墙钟为2563.089秒，独立复核另计。此次不扩大到完整9178/4478测试集。

## 历史结果的使用边界

旧[10%＋32结果](LOOKM_RESULTS_20261109.md)仍是legacy，不能用于严格10%比较。本次同时包含总预算、逐head合并补集和数值/边界正确性修正；因此新旧分数差不能单独归因于扣除32个位置。已有Hybrid历史记录没有在本轮重跑或重新控制预算，这里不将其作为严格等预算配对对照。

[单独运行LookM的脚本](../scripts/run_lookm_video.py)默认两个任务各200条，必须使用新输出目录。复现命令见[协议](LOOKM_COMPARISON.md)。

# 旧设置独立LOOK-M／AsymKV：NExTQA200结果

两组均已完成并独立复核。Qwen2.5-VL-7B-Instruct、NExTQA开放式测试集原始前200条、64帧、BF16、FlashAttention2、batch1，单张NVIDIA A800-SXM4-80GB。仅运行两种完整独立方法，共400次正式生成；旧四个原版／KVRepack控制未重跑。正式执行UTC2026-11-15T14:41:16至15:28:05，含加载和评分总墙钟2808.284秒，约46.8分钟。

| Method | WUPS | 视频至文本E2E均值（秒/条） | 完整响应循环均值（秒/条） |
|---|---:|---:|---:|
| LOOK-M (standalone) | 33.539684 | 3.577198 | 3.784124 |
| AsymKV (standalone) | 29.164032 | 9.826410 | 10.042068 |

[三列TXT](legacy_standalone_nextqa_200.txt) · [完整CSV](legacy_standalone_nextqa_200.csv) · [协议与启动命令](LEGACY_STANDALONE_NEXTQA_E2E_20261115.md)。

## 计时范围

主要E2E逐题同步测量视频读取/解码、processor、H2D、完整原生成和文本解码。AsymKV所有必要Fisher计算计入。模型加载、任务评分、计时后的输入哈希、缓存审计和写盘不计入该列。无额外预热，首题cold cost保留；每题只测一次，不宣称稳定态性能或统计显著性。

完整响应循环另外包含题间记录、审计、日志和写盘间隙，用于保留可追溯的整段wall time。LOOK-M循环总756.824766秒，AsymKV2008.413514秒；逐题E2E总和分别715.439572、1965.282064秒。旧历史循环均值仍保留；跨旧新结果比较时应注明计时边界，不能把循环均值与排除了记录间隙的逐题E2E默认为同一指标。

逐题E2E的P50/P95：LOOK-M为3.242237/5.741308秒，AsymKV为9.320157/14.380248秒。均值分解中的完整generation分别1.887455/8.075646秒，其余主要是视频处理。AsymKV包含Fisher且生成长度不同，仅由这两项均值不能量化单独merge算子的开销。

## 保留的原实现与预算

LOOK-M恢复旧记录所用精确两函数及旧数值/跨head补集行为，完整原模块SHA为`a7219df197e13cd91e6aee0369bc1e0d8264a2f0fc65c3f863576d83df82722e`。历史位置预算为`floor(.1N)`，额外保留32 recent，实际prefill物理位置平均10.298789%。生成使用原HF greedy与重复惩罚1.05。

AsymKV使用原official完整路径，名义比例0.1、32 sink、512 chunk，保留原Fisher、分块及argmax生成；32 sink在名义预算内，不是额外32 recent。原驱动先压缩再追加当前块，实际prefill物理位置平均18.111221%，范围12.372162%至21.583976%。因此本轮是各自旧设置复测，不能标成严格等总10%预算。原argmax无HF重复惩罚；不为统一外观改变原算法。当前200条未触发已知`N % 512 == 1`边界。

LOOK-M全部200条EOS；AsymKV194条EOS，另6条达到128-token上限且未EOS。所有样本均计入质量与时延，不剔除长生成。两组平均生成token数分别3和7.26。

## LOOK-M旧结果差异

200条的原始row/video/frame和全部5项processor输入tensor指纹与旧LOOK-M记录逐条一致。197条生成ID及文本完全相同，其余如下：

| doc ID | 旧回答 | 本次回答 |
|---|---|---|
| 28 | pick up towel | pick up blanket |
| 44 | look down | look at camera |
| 150 | to take remote control | to take care of helicopter |

本轮原生任务WUPS独立重新计算为33.53968371764908，旧记录为33.8571440351094，下降0.317460。差异原因尚未通过受控实验隔离，不将其直接归因于随机误差或某个算子，也不把旧分数与本次时延拼成同一run。AsymKV本轮WUPS为29.164032378611417。

## 复核与归档

400条原始样本、200对输入、原生评分预测语料与评分身份、模型/Arrow及冻结源码、404项来源索引、每题时延总和与均值独立只读复核通过。独立核验没有再次调用GPU或重跑评分；正式报告中的`gpu_execution: false`指报告生成过程，正式推理在GPU执行。迁移的30份核心实现保持不变。

- 原始运行：`outputs/benchmarks/legacy_standalone_nextqa_200_20261115`。
- [归档清单](../provenance/legacy_standalone_nextqa_20261115/final/release_manifest.json)：425个文件，原始总26,558,371字节，压缩包及逐文件SHA保留；压缩包逐文件回读SHA一致。
- [完整运行压缩包](../provenance/legacy_standalone_nextqa_20261115/final/recorded_runs.tar.gz) 与 [逐文件SHA](../provenance/legacy_standalone_nextqa_20261115/final/recorded_files_sha256.json)。
- [原生报告](../provenance/legacy_standalone_nextqa_20261115/final/report/RESULTS.md)、[逐题时延CSV](../provenance/legacy_standalone_nextqa_20261115/final/report/per_input_latency.csv)、评分语料与metrics均随报告保存。
- [独立最终复核](../provenance/legacy_standalone_nextqa_20261115/final_verification.json) 与 [LOOK-M旧新逐题对照](../provenance/legacy_standalone_nextqa_20261115/lookm_replay_check.json)。

源码执行版本`0b85f67`，正式启动记录版本`4087b02`。入口为`scripts/run_legacy_standalone_nextqa.py`；逐方法worker为`src/kvrepack/benchmarks/run_legacy_nextqa_e2e.py`，原生离线报告为`legacy_nextqa_report.py`。GPU单条联调与27项CPU检查记录保留在preflight，未混入正式200条。此次归档没有新增GPU实验，也未恢复其他已停止任务。

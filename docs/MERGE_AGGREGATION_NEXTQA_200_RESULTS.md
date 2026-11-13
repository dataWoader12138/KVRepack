# NExTQA合并规则对照：前200条完整结果

**八组已全部完成：每组原始前200条，共1600次正式生成。** Qwen2.5-VL-7B、A800-SXM4-80GB、64帧、BF16/FlashAttention2、greedy、最多128个新token；严格总活跃prefill KV预算10%，recent/local位置包含其中。16次预热不计入下表。

| 方法 | WUPS ↑ | 完整E2E（秒/条）↓ |
|---|---:|---:|
| SnapKV | 33.192028 | 3.200852 |
| SnapKV + KVRepack | 32.894385 | 3.253997 |
| SnapKV + LOOK-M-style merging | 32.049940 | 3.278505 |
| SnapKV + AsymKV-style merging | 30.610048 | 4.795997 |
| HybridKV | 32.648727 | 3.302653 |
| HybridKV + KVRepack | 33.230672 | 3.394233 |
| HybridKV + LOOK-M-style merging | 33.201505 | 3.476015 |
| HybridKV + AsymKV-style merging | 31.857199 | 5.051236 |

完整E2E为逐样本同步测量后取均值，包含视频读取/解码、processor、H2D、Fisher（AsymKV适用）、视觉编码/prefill/decode、文本解码；不含模型加载、预热、审计哈希、写盘与评分。两类style只替换KVRepack插入点的聚合规则，候选、选头、分组、anchor保持一致，不代表standalone LOOK-M或AsymKV。

本次HybridKV + KVRepack的WUPS较原版增加0.581944，完整E2E增加约2.77%；SnapKV + KVRepack的WUPS较原版降低0.297643，完整E2E增加约1.66%。两个骨干上，KVRepack均高于另外两种merge规则，完整E2E也更低；Hybrid上KVRepack与LOOK-M-style的WUPS仅差0.029167，不能据此声称统计显著。

AsymKV-style包含平均约1.48秒/条的Fisher估计开销；Snap/Hybrid分别有2/3条达到128token上限而未EOS，均保留并参与分数与时延统计。其余六组全部正常EOS。每个骨干内部，四个条件对同一输入保留的物理prefill位置数完全一致，且都不超过严格10%上限；跨骨干的head预算分配不同。

这些数据属于新的严格总10%实验，不能用旧nominal历史10%另加recent或VATEX实验的分数替代。每输入/条件只测一次；两两轮换顺序，四个worker串行，完整时延同时受预处理和输出长度影响，不等同纯merge算子耗时。

控制器总耗时6447.375秒（约1小时47分，包含加载、预热、正式实验、审计写盘和离线评分），不作为单条E2E。实际UTC起止时间见保存的experiment_state.json。

- [三列表TXT](merge_aggregation_nextqa_200.txt)
- [完整指标及各阶段时延CSV](merge_aggregation_nextqa_200.csv)
- [原始自动报告](../provenance/merge_nextqa_200_20261113/final/report/RESULTS.md)
- [逐样本时延CSV](../provenance/merge_nextqa_200_20261113/final/report/per_input_latency.csv)
- [实际预算CSV](../provenance/merge_nextqa_200_20261113/final/report/budget_comparison.csv)
- [完整原始记录压缩包](../provenance/merge_nextqa_200_20261113/final/recorded_runs.tar.gz)
- [归档SHA与完成身份](../provenance/merge_nextqa_200_20261113/final/release_manifest.json)
- [实验协议与命令](MERGE_AGGREGATION_NEXTQA_200_20261113.md)

独立只读复核及归档逐文件SHA复核均通过，1600条时延重算误差小于1e-9；未重跑GPU、未改变原始预测或时延。见[复核证据](../provenance/merge_nextqa_200_20261113/final/verification.json)。

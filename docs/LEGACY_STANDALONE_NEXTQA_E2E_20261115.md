# 旧设置独立LOOK-M／AsymKV：NExTQA200端到端复测

正式任务已于UTC2026-11-15T15:28:05完成，两方法各200条。[最终分数、E2E、旧LOOK-M差异及归档](LEGACY_STANDALONE_NEXTQA_E2E_RESULTS_20261115.md)；[三列结果TXT](legacy_standalone_nextqa_200.txt)。以下保留实验协议与启动信息。

用户授权仅重跑独立原版LOOK-M和独立原版AsymKV，各使用NExTQA开放式测试集原始doc0..199，共400次正式生成。模型Qwen2.5-VL-7B-Instruct、64帧、BF16、FlashAttention2、batch1、greedy、上限128token，原生prompt和视频处理。两个单条GPU联调及27项CPU检查已通过，联调不混入正式结果。正式任务是否完成以experiment_state.json为准。

## 原实现和原预算

LOOK-M恢复此前产生33.857144的精确旧实现：从可逆来源记录还原merge_kv与LookMCluster.update_kv，原完整模块SHA a7219df197e13cd91e6aee0369bc1e0d8264a2f0fc65c3f863576d83df82722e。历史top-k为floor(.1N)，额外32 recent。包括旧跨head补集和低精度路由等旧行为，不能将修正版代码改回预算后冒充原版。上下文退出恢复当前修正版，不编辑核心文件。

AsymKV运行已选定的released-code完整官方适配路径（commit771ce4db0292），不是HybridKV／SnapKV的style移植。BUDGET_RATIO=.1、START_SIZE=32、CHUNK_SIZE=512，不设absolute cache_size。32是内部sink，不是额外32 recent；原驱动先压历史再追加512或末尾partial，因此实际物理缓存可能明显高于名义10%。完整Fisher、原attention和chunk边界行为均保留，实际缓存写入每题audit，不为凑容量改原算法。当前前200条没有N%512==1，已知该边界仍不擅自修正。

两方法保留各自原解码：LOOK-M使用HF greedy并继承repetition_penalty1.05；原AsymKV为自身argmax生成，不支持HF重复惩罚，不能偷偷加上该参数。模型权重/输入一致不代表这些原实现特性相同。

## E2E范围

主要指标是每题视频读取至文本可用的同步墙钟E2E，再对200题求均值。计入旧lmms多一次decord首帧读取、视频解码/processor、H2D、完整原generate（含AsymKV全部Fisher）、文本解码；模型加载、评分、审计哈希和写盘在外。每题保存分阶段、总时延、生成长度、EOS/截断和峰值显存。

执行顺序复用lmms的prompt-token长度降序与文本字典序tie-break，仍是同一原始前200题，按doc ID分别保存结果。无额外预热，首题cold cost计入，符合旧响应循环的设置。LOOK-M首例全部输入tensor SHA和生成tokens与旧记录一致。

另外独立保存整个正式循环wall time及其除以200的值，它包含审计/日志/写盘gap；不将它与排除了这些gap的逐题E2E混为一列。AsymKV audit仅截获原_append_audit记录，原算法产生记录不变，实际磁盘写入移到计时后。

## 启动

```bash
python scripts/run_legacy_standalone_nextqa.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct/snapshot \
  --hf-home /path/to/huggingface_cache \
  --output outputs/benchmarks/legacy_standalone_nextqa_200_20261115 \
  --limit 200
```

控制器只串行运行LOOK-M和AsymKV，随后自动CPU原任务WUPS评分，并生成report/comparison.csv、per_input_latency.csv、summary.json和RESULTS.md。原分数与新时延成对保存，不把旧33.857的分数冒充本轮已重现。旧四个控制的历史平均E2E保留，不重跑，不恢复任何旧停止任务。

本机正式ROOT为outputs/benchmarks/legacy_standalone_nextqa_200_20261115；pilot ROOT为outputs/benchmarks/legacy_standalone_nextqa_20261115_preflight；来源证据provenance/legacy_standalone_nextqa_20261115。正式源版本0b85f67，controller/PG3865，启动UTC2026-11-15T14:41:16，完成15:28:05，控制器总墙钟2808.284秒（约46.8分钟，含加载与评分）。state为complete；pilot不混入正式200结果。

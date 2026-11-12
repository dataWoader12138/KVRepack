# VATEX200 合并规则与原版／KVRepack 时延对照

> **已停止：用户纠正目标为NExTQA。** 以下为此前误用VATEX的历史安排；保留原始记录，不恢复此任务，不计入当前NExTQA比较。新协议见 [NExTQA前200条](MERGE_AGGREGATION_NEXTQA_200_20261113.md)。

用户授权核查既有结果，补足 SnapKV、HybridKV 的原版和 KVRepack 版完整端到端计时。数据集沿用 VATEX 原始前200条，模型为 Qwen2.5-VL-7B-Instruct。正式比较最终为8组、1,600条输出；本轮新增4组、800条。预热及联调不计入正式分数或时延。

| 骨干 | 条件 | 样本数 | 本轮安排 |
|---|---|---:|---|
| SnapKV | 原生剪枝 | 200 | 新增 |
| SnapKV | KVRepack | 200 | 新增 |
| SnapKV | LOOK-M式合并 | 200 | 复用已完成记录 |
| SnapKV | AsymKV式合并 | 200 | 复用已完成记录 |
| HybridKV | 原生剪枝 | 200 | 新增 |
| HybridKV | KVRepack | 200 | 新增 |
| HybridKV | LOOK-M式合并 | 200 | 复用已完成记录 |
| HybridKV | AsymKV式合并 | 200 | 复用已完成记录 |

四组style已有完整逐样本E2E，并且18个数值依赖的SHA与当前源码一致，保留原预测和原时延；位置为 `outputs/ablations/merge_vatex_mergeonly_200_20261112/`。它们只迁入聚合规则及其必要依赖，候选、选头、分组、anchor与插入位置遵循KVRepack，不是standalone LOOK-M或AsymKV。

原版和KVRepack版也有历史计时，不能说没有测过。但历史200条主要使用nominal历史10%加近期窗口与响应循环计时；同当前严格总10%／完整逐样本同步计时匹配的KVRepack对照只有Snap前114条，其余未找到完整200条。因此本轮对四个对照各重新测量200条，使用全新目录，不恢复任何已停止的旧任务。

## 固定条件

- 模型snapshot：`cc594898137f460bfe9f0759e9844b3ce807cfb5`；BF16、FlashAttention2、batch1，A800-SXM4-80GB。
- VATEX原始`doc_id=0..199`，64帧；沿用现有原生视频处理和prompt。每个条件独立读取处理输入，实际张量指纹逐条核对。
- greedy、最多64个新token、repetition penalty 1.05、seed42；每臂预热2条，正式循环按doc_id轮换原生／KVRepack先后顺序。
- KVRepack参数：Snap gate=.15、cr=.70、alpha=.5；Hybrid gate=.50、cr=.50、alpha=2，raw K聚合后按原anchor重施mRoPE。
- **近期窗口计入10%总活跃prefill KV预算。** 原版保持原生评分、top-k或head分类／分配算法，只用临时适配器统一总预算：Snap仅替换历史容量表达式；Hybrid在原分配后使用同一全局缩放与chunk向下取整。没有把merge_ratio=0当成原生实现，没有修改30份核心算法文件。
- 活跃KV比例不等于总VRAM比例；记录实际保留位置、生成长度、EOS/截断与峰值显存。

## 计时和比较范围

完整E2E从视频处理前开始，包含视频读取／解码、processor、H2D、Fisher（style AsymKV适用）、prefill、decode、文本解码，边界CUDA同步。模型加载、预热、审计哈希、写盘和评分不计入。另保存模型E2E及各阶段时间，避免把generation_seconds误当完整流程时间。

新原版／KVRepack为同进程逐输入交替测量。四组style来自已完成的另一测量session，硬件、模型、输入、算法与预算口径核对一致，但完整时延仍受预处理、输出长度与运行波动影响，不将差值直接称为纯merge算子开销。

原始分数与历史表格保持不变；新8组重新整理为独立结果，不把旧稿0.3447与另一运行时延冒充同次测量。旧四组style评分缓存须通过原始预测语料、Arrow、评分身份、合同与发布哈希核验，新四个对照独立进行原生任务评分。

## 运行

```bash
python scripts/run_baseline_completion.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct/snapshot \
  --hf-home /path/to/huggingface_cache \
  --style-root outputs/ablations/merge_vatex_mergeonly_200_20261112 \
  --output outputs/ablations/baseline_completion_vatex_200_20261113 \
  --limit 200 --warmup 2
```

新worker：`kvrepack.ablations.run_baseline_completion`；原生预算适配器：`strict_pruning.py`；离线汇总：`baseline_completion_report.py`。控制器串行运行Snap／Hybrid，全部完成后自动评分并保存`report/comparison.csv`、`per_input_latency.csv`、`budget_comparison.csv`、`summary.json`和`RESULTS.md`。真实状态以`experiment_state.json`和各骨干的`run_state.json`为准。

GPU联调已完成每骨干各1条原生／改进配对，不当成正式200条结果。Snap实际prefill位置均108976；Hybrid均108659（总cap108976）；输入指纹一致。Snap KVRepack与原严格前缀的token、选头mask及预算完全一致。原生Snap另有真实CPU逐元素oracle，保持原评分／top-k结果和recent token；不通过只刷新哈希掩盖算法变更。

此前误启的正式任务使用控制器PID/PG5481、源版本`1d1edd1`，现因用户纠正数据集为NExTQA已停止。Snap保留116个完整原生/KVRepack配对（232次生成），Hybrid尚未开始；两级状态均禁止恢复，不将它们当作完成的VATEX或NExTQA200结果。原始目录`outputs/ablations/baseline_completion_vatex_200_20261113/`；启动证据保留，停止与样本SHA见`provenance/merge_nextqa_200_20261113/vatex_stop_and_dataset_correction.json`。

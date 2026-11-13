# 恢复原实验预算

用户要求保留原作者代码的运行行为，便于与旧结果对齐。本次将新增NExTQA比较入口的默认预算恢复为原实现；已完成的严格总预算实验单独冻结，不改预测、时延或分数。

## 已恢复的行为

- SnapKV原版及三种merge均保留原历史预算 `floor(r*N)`，再拼接原recent窗口；不再减去32。首例N9607、r=.1时，每头恢复为992个位置。
- HybridKV原版及三种merge均调用原逐头预算分配，不再执行新增global shrink，不覆盖retrieval_budget，也不额外做chunk取整。
- LOOK-M/AsymKV仍只接入聚合方法；候选、head选择、分组和anchor继续用原KVRepack实现，预算恢复后这些计算也使用原容量。
- 30个核心算法文件、通用实验入口及旧严格预算worker/适配器保持原字节。恢复通过新的原预算上下文和默认入口选择完成，方法仍共用原实现。

新增控制器 `scripts/run_nextqa_merge_comparison.py` 默认 `--budget-policy original`，调用run_original_baselines/run_original_merge_comparison和original_nextqa_report。报告明确标注 `original_history_plus_local_v1`，不会把实际约10.3%的原预算误写为严格总10%。若试图把旧严格结果目录作为原预算恢复运行，会在启动模型前拒绝，避免混用。

## 入口

保持原作者accelerate/lmms_eval调用链的常规主实验入口仍是：

```bash
python -m kvrepack.run --method snapkv \
  --task nextqa_oe_test --model-type qwen2_5_vl-7b \
  --pretrained /path/to/model --frames 64 --limit 200 --budget-ratio 0.1
```

其他方法按旧参数使用同一入口；KVRepack的NextQA设置仍需使用此前确认的gate/compress/alpha，不可套用另一数据集参数。通用入口不使用本轮严格预算补丁。

八条件合并与完整E2E比较的恢复后命令：

```bash
python scripts/run_nextqa_merge_comparison.py \
  --pretrained /path/to/model --hf-home /path/to/huggingface_cache \
  --output outputs/ablations/merge_nextqa_original_budget_200 \
  --limit 200 --warmup 2
```

输出必须是新目录。`--budget-policy strict-total`仅用于明确选择历史严格协议，不是默认。

## 核验范围

CPU检查覆盖原生Snap真实K/V逐元素oracle、原预算函数不被替换、Hybrid原分配不缩放、默认路由与跨协议恢复拒绝、原预算评分/缓存校验和计时路径保持一致。另用旧SnapKV+KVRepack和HybridKV+KVRepack各200条的实际缓存形状验证恢复后的预算公式，记录见provenance/original_budget_restore_20261113。

当前机器没有可访问的`/dev/nvidia0`或`/dev/nvidiactl`，因此本次没有重新生成模型结果。恢复代码预算不等于已证明GPU输出与每份历史记录逐字一致。七月旧lmms主实验与后来的自定义诊断worker还存在入口/seed等区别，必须用对应原配置做GPU复核后才能认定具体旧分数复现。

## 原生SnapKV为什么预算更小却分数略高

旧原生WUPS32.913865、本轮严格预算33.192028，增加0.278162。200条时，一题从0分到满分会改变总WUPS 0.5；当前差值不到一题满幅变化的量级，少量回答改变足以产生，不能说明小预算具有稳定优势。

更多KV可能加入干扰，并改变softmax权重分配；生成和最终验证集指标不保证随缓存数量单调改善。这是可能机制，不是本轮已经证明的原因。

旧原生200条逐题预测没有保存下来，现存日志只能核实样本数和聚合分数，所以无法逐题重建这0.278162的来源。此前“6条回答变化、21条选头变化”属于SnapKV+KVRepack；原生SnapKV没有该选头过程，不能用这些统计解释原生分数上升。

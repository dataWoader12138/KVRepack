# 本轮NExTQA与旧实验的协议差异

本轮用新建的 `scripts/run_nextqa_merge_comparison.py` 和两个worker执行，没有原样执行旧主实验sh。虽然沿用核心算法文件，运行时适配器改变了预算及其后续候选/部分选头，属于新的严格总10%实验，不能称为旧主实验的原样复现，也不能直接覆盖旧论文表。

## 已逐条核实的相同项

将本轮两个KVRepack条件与旧Random的similarity、旧RoPE的raw_reapply分别核对200条：doc0..199、Arrow、视频ID、原始行SHA、64帧索引和全部五项输入tensor指纹均相同。模型snapshot/metadata、版本、BF16、greedy、max128、seed42及主要参数数值一致。旧重复惩罚继承模型值1.05，新显式1.05，没有改变。

这证明与上述仓库内旧诊断实验输入相同，不代表已逐条证明所有更早的主实验或论文33.576对应run相同。

## 已改变的部分

| 检查项 | SnapKV + KVRepack | HybridKV + KVRepack |
|---|---:|---:|
| 旧平均实际保留比例 | 10.298789% | 10.329914% |
| 本轮平均实际保留比例 | 9.995698% | 9.974827% |
| 生成token序列变化 | 6/200 | 4/200 |
| 实际选头集合变化 | 21/200 | 6/200 |

Snap旧历史容量为 `floor(.1*N)`，最后另加32个recent；新历史容量为 `floor(.1*N)-32`，拼接后的总量不超过10%。首例N9607：旧每head992位置，新960位置，112个layer-head合计111104→107520。原生方法由strict_pruning适配，三种merge由merge_snap适配。

Hybrid在原分配后执行merge_hybrid.shrink_global_budgets，统一缩放历史预算并对动态头按8-token chunk向下取整。首例111863→107335位置；其local_len实际39（含chunk余数），不能笼统称每head只减32。

预算还影响候选及选头：Snap的_history_budget决定candidate_history，再在这些候选raw K上计算相邻相似度；Hybrid也按实际逐头预算构造候选再算相似度。新八组内部的三种merge仍保持相同heads/groups/anchor，但与旧实验的head集合并非200条全同。

实际例子：Snap doc15旧回答`clean spoon`、新`clean mouth`；Hybrid doc64旧`interested in show`、新`interested in tv`。这已是生成行为差异，而非只改了计时。

## 历史入口和分数的边界

仍可读的旧启动器 `/root/run_qwen7b_common_decouple_sweep_20260706.sh` 第196–205行使用accelerate→lmms_eval、temperature0、limit200；原生方法从opensource启动，改进方法从artifact启动。旧scripts/eval当前仅剩缓存目录，原入口全文保存在provenance/reference_entrypoints.json。

| 旧记录 | WUPS |
|---|---:|
| 七月SnapKV原版 | 32.913865 |
| 七月SnapKV+KVRepack历史最佳（.30/.70/2） | 33.169733 |
| 后来Random/RoPE的SnapKV+KVRepack原策略 | 32.917681 |
| 七月HybridKV原版 | 32.738717 |
| HybridKV+KVRepack旧最佳及后来Random/RoPE原策略 | 33.219903 |

七月原版/Snap最佳保存于原summary JSONL，Hybrid最佳身份保存于provenance/video_best_parameters.json。旧七月lmms记录seed为Python0/NumPy1234/Torch1234，本轮和后来的诊断worker均42。greedy下不能据此断言分数变化由seed造成。旧Snap33.169733与后来32.917681在本轮前已经不同，不能将两者差异归于此次预算修改。

预算/候选/部分选头及输出改变均有证据，但尚未执行只恢复预算的一变量GPU对照，不能断言所有历史分数差异只有预算一个原因。完整E2E边界也与旧loop/generation计时不同；计时边界本身不解释WUPS变化。

若目标是复现旧主实验，应先按原调用链、预算和生成配置对齐原版及KVRepack输出，只在外层增加E2E计时，再接入其他merge。严格总预算这轮单独保留，不替换旧结果。本次只读核对并保存说明，没有重启GPU。

[逐条核对和原始SHA](../provenance/nextqa_protocol_comparison_20261113/comparison.json) · [本轮严格预算结果](MERGE_AGGREGATION_NEXTQA_200_RESULTS.md)

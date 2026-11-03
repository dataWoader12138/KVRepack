# MileBench 来源与行为边界

本次以 `fba9c68` 的 KVRepack 为集成基线，新增图片后端；历史规划保存在 [MILEBENCH_PLAN.txt](MILEBENCH_PLAN.txt)。原 `HybridKV-opensource` 不变，运行时不依赖它。

| 部分 | 处理与证据 |
|---|---|
| MileBenchDataset | 抽取原 `utils.py` 类，去除 torch Dataset 基类；prompt、图片顺序、选项、截断与 collate 的执行 AST 保持相同 |
| 原 Eval | 保留原评分执行 AST，仅改为惰性导入 NumPy/rouge；新包装负责严格ID覆盖、答案校验、子集和排他写入 |
| 图片 worker | 按原 Qwen/LLaVA 协议重组为惰性加载适配器；用固定图片与替身验证消息、JPEG、像素、生成kwargs和token trim，不能称完整模型等价证明 |
| 生成调度 | 保留图片数分组顺序；新增分组前limit、全图预检、失败传播、精确样本与运行来源记录，移除无条件DeepSpeed及分布式初始化假设 |
| 汇总 | 保留逐任务指标及单次百分制转换；新增明确缺失/子集标识与已完成任务宏平均，不静默纳入其他旧结果 |
| Qwen核心 | 复用目标已有唯一实现；未用当前 opensource 整文件覆盖，因此历史实验完整数值等价性仍待核对 |
| LLaVA四改进 | 从当前 opensource port 定向迁入5个缺失模块，复用目标已有outer model；确定性名称/导入变换可恢复原字节并回放 |
| LLaVA SparseMM基线 | 显式修复旧caller未重建prefill容量表的问题，独立记录与CPU回归；相对旧目标行为改变，不能写成仅改名 |
| LLaVA共享配置 | 目标budget_fixed_ratio=0.1，当前opensource为0.5；本轮基线与改进共享目标0.1并记录每次配置快照，不宣称来源参数完全相同 |

Qwen 当前来源 SnapKV 有独立配对缓存和 text-model forward，目标已有 SnapKV 不具有同一结构；SparseMM/MadaKV 的初始化、缓存收尾及 forward 也有差异。Hybrid 代码的若干 AST 差异是命名、日志、重排或未启用分支，不能仅凭文件不同断言默认数值不同。来源 `combined` 与目标 `full` 主评分公式可对应，但 `difference` 与 `difference_only` 不是严格相同；图片入口不开放这些消融映射。

数据与原评分类的比较只允许以下规范化：忽略位置属性及独立字符串说明节点；仅去除原 Dataset 基类；仅忽略 `import numpy as np` 与 `from rouge import Rouge` 两类惰性导入位置。变量、算术、分支、调用和默认参数均参与比较。源类定义、完整源文件哈希和目标哈希保存在 `provenance/milebench_sources.json`，可脱离原机器检查；`--source-root` 可额外对照仍保留的原树。

所有手工重组和新增包装都显式记录为行为变更，不通过更新哈希宣称算法等价。小张量、接口替身与静态调用检查不能证明真实 FlashAttention、完整模型生成或任务分数；后续必须按同模型/同样本/同配置完成 GPU 对照。

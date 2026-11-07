# Qwen7B 图片 Random-head 全量结果（2026-11-08）

已完成 Qwen2.5-VL-7B-Instruct 上的 HybridKV-KVRepack、SnapKV-KVRepack 两方法，对官方 MileBench v1 的 Spot-the-Diff、MMCoQA 两子集运行全部标注，各200条。每个输入运行 similarity、random seed0、random seed1，共800份方法/样本配对记录、2400次正式生成；无第三个选头seed。

MileBench作者从源任务抽样构建每个200条的官方评测子集，见[论文§3.2](https://arxiv.org/pdf/2404.18532)。本轮使用 `--all-samples --head-seeds 0 1`，未对这些子集再截取；这不是两个源数据集全部原始测试集的实验。

分数为百分制：Spot使用原ROUGE-L F，MMCoQA使用原答案子串命中Accuracy。差值为 `similarity − random均值`，单位为百分点。

| 方法 | 数据集 | Similarity | Seed0 | Seed1 | Random均值 ± 样本SD | 差值 |
|---|---|---:|---:|---:|---:|---:|
| HybridKV-KVRepack | Spot-the-Diff | 29.8377 | 29.4362 | 29.2029 | 29.3195 ± 0.1649 | +0.5182 |
| HybridKV-KVRepack | MMCoQA | 65.5000 | 62.5000 | 61.5000 | 62.0000 ± 0.7071 | +3.5000 |
| SnapKV-KVRepack | Spot-the-Diff | 28.9558 | 28.8743 | 28.9118 | 28.8930 ± 0.0265 | +0.0627 |
| SnapKV-KVRepack | MMCoQA | 59.0000 | 58.5000 | 58.0000 | 58.2500 ± 0.3536 | +0.7500 |

两种方法在Spot上的差距都较小；MMCoQA上的观测差值分别为Hybrid +3.50、Snap +0.75。SD仅衡量两个固定模型级选头seed之间的任务分数差异，不是输入不确定性或显著性结论；不把重复seed当作额外独立样本，不报告未定义的bootstrap区间。

## 执行与独立核验

使用 NVIDIA A800-SXM4-80GB、BF16、FlashAttention2、greedy、模型seed42、最多512个新token。模型revision为 `cc594898137f460bfe9f0759e9844b3ce807cfb5`。预算、图片和生成设置见[完整实验协议](MILEBENCH_RANDOM_HEAD.md)及各运行合同。

- 四组均为200条、600次生成，完整ID、固定随机mask、源码/标注SHA、预算/候选配对及原评分重算全部通过。
- 2400/2400以EOS结束；达到512-token上限为0、无EOS截断为0；最长实际生成186 tokens。
- CPU suite为161项，160通过、1跳过；跳过项需要本机缺失的原artifact树做数值差分。选头适配器和两压缩核心与 `6c65bf8` 的逐字节比较通过。
- 本轮覆盖这一个模型、两种方法、两个任务；不能推广为其他组合、原始实现完整数值等价或性能基准已验证。

## 原生合并工作量

选中头配额为Hybrid 56、Snap 34。Hybrid/Spot始终56头有效，Snap两任务始终34头有效且原工作量按输入配对一致。Hybrid/MMCoQA在异质原生预算下，选中56头并不保证56头都进入有效合并：

| Hybrid/MMCoQA条件 | 有效头范围 | 少于56头的输入数 | 有效头累计数 | 移除token累计数 |
|---|---:|---:|---:|---:|
| similarity | 32–56 | 12/200 | 11070 | 11235853 |
| random_global_seed0 | 15–56 | 69/200 | 9916 | 8540928 |
| random_global_seed1 | 19–56 | 70/200 | 10149 | 9411154 |

实际记录中没有no-op调用，上表差异是选中头未进入原生合并调用的观测。各条件仍使用相同原分类、每头预算、候选和固定选中总数；没有跳样、重抽mask或重新分配预算。Hybrid结果应解释为原预算语义下更换选头集合的效果，不能称为有效合并工作量严格相等的比较。

## 修复、复用与产物

首个控制器完成Hybrid/Spot后，在MMCoQA样本18被新增后置门禁拦下：该门禁错误要求56个选中头全部有效合并，超出参考Hybrid适配器只要求存在有效合并的语义。修复仅恢复这一审计语义并独立重算有效计数，算法、预算及随机mask未变；Snap严格工作量门禁保留。旧失败目录、合同及记录保留。完整Hybrid/Spot的222个文件逐字节复用到新结果根，未重复其600次正式生成；旧12份绑定源码保存在归档并按原SHA核验。smoke和失败控制器的额外尝试不计入本表正式结果。

- [完整四行精确分数CSV](../provenance/random_head_qwen7b_all_seed01_20261108/audit/results.csv)
- [独立全量审计JSON](../provenance/random_head_qwen7b_all_seed01_20261108/audit/audit.json)
- [生成长度与EOS统计](../provenance/random_head_qwen7b_all_seed01_20261108/audit/generation_stats.csv)
- [原生合并工作量统计](../provenance/random_head_qwen7b_all_seed01_20261108/audit/merge_work_stats.csv)
- [精简包清单与SHA](../provenance/random_head_qwen7b_all_seed01_20261108/EXPORT.json)
- [环境、构建及验证记录](../provenance/random_head_qwen7b_gpu_validation.json)

精简包含逐样本分数/配对差、2400条轻量预测及生成token、800份原始bundle SHA、合同、报告、旧执行源码和复用记录。完整逐head候选/合并原始记录仍在本机 `outputs/ablations/random_head_qwen7b_all_seed01_v2_20261108`；精简包不含这些大数组，完整合并复审需要本机raw bundles。模型、图片、编译产物和凭证不上传。

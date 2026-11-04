# Random-head 选择策略消融

本实验回答：在选头总数与合并参数相同的条件下，按相似度选择 KV head 是否优于随机选择？用户已授权编写并尝试 GPU 运行。2026-11-04 已完成两任务各20条、4条件共160次真实GPU生成。见[结果报告](RANDOM_HEAD_ABLATION_RESULTS_20261104.md)：当前小样本未显示相似度选头稳定优于Random。

原有 All-head（w/o Head Selection）把所有 head 都纳入合并，同时改变选择策略和合并数量，因此不能代替本对照。首轮采用预算一致、易于审计的 SnapKV-KVRepack，尚不等于完成原文 HybridKV All-head 表的全部消融。

## 对照与固定条件

使用 Qwen2.5-VL-7B-Instruct，在 `nextqa_oe_test`、`vatex_test` 各取测试集原始前20行，不换样、不按结果筛选。每个任务运行4个条件，共160次生成：

| 条件 | 选择规则 | 随机种子 |
|---|---|---|
| `similarity` | 原实现的全局相似度 Top-K | 不使用选头随机数 |
| `random_global` | 全模型 KV head 中均匀无放回选取同样的 K 个 | 0、1、2，分别运行 |

模型有28层、每层4个原生 KV head，共112个候选。Random 为每个种子生成固定的全模型随机排序，所有输入复用该排序，按对应任务的 K 截取；使用局部随机数生成器，不改变模型生成的随机数状态。它保留全局选头总数，允许各层获得不同的选头数，这属于全局选择策略干预的一部分。

| 任务 | 选头比例 `simtop` | 实际选头数 `ceil(112×simtop)` | `cr` | `alpha` |
|---|---:|---:|---:|---:|
| NExTQA OE/test | 0.30 | 34 | 0.70 | 2 |
| VATEX/test | 0.15 | 17 | 0.70 | 0.5 |

两任务均固定64帧、10% KV预算、BF16、FlashAttention 2、batch size 1、greedy generation、`use_cache=True`、模型种子42、`max_new_tokens=128`。参数沿用当前实现的语义。各条件使用同一份 CPU 预处理张量缓存，逐张量校验字节、形状和 dtype；模型、输入、预算、候选构造、合并函数和生成设置保持一致。VATEX 的128 token上限与部分历史64 token运行不同，本轮只在以上四条件内部比较，并记录 EOS 与是否达到上限。

另外实现了 `random_layer_matched`，可保留原策略每层选头数后在层内随机选头。它回答“给定层配额后，层内选哪些 head”的问题，不能替代全局 Random；首轮不运行这个附加条件。

## 实现与公平性检查

独立入口为 [`run_head_selection.py`](../src/kvrepack/ablations/run_head_selection.py)，实验适配器为 [`head_selection.py`](../src/kvrepack/ablations/head_selection.py)。适配器仅在本实验进程中替换选头结果并记录执行情况，继续调用原合并函数；核心压缩源码与常规入口默认行为不变。

逐输入核对四个条件的输入张量与全部层候选张量 SHA-256、一致的选头总数、实际合并调用次数、实际移除 token 数及28层 prefill KV缓存形状。同时记录相对于原策略的 mask 差异与重叠，确认干预实际生效。有效合并数量或工作量不一致时直接报错，不通过重采样掩盖差异。生成长度可能因预测内容不同而变化，因此公平性检查针对 prefill 缓存，不能要求最终 decode 缓存长度完全相同。

## 在其他机器运行

先按项目安装说明准备依赖、模型和两个视频数据集缓存。该入口离线读取已有资源，不自动下载。`MODEL_SNAPSHOT` 应指向本地7B模型 snapshot，`HF_HOME` 指向已准备的数据缓存。以下目录必须是新目录，worker 不会自动续跑部分生成结果。

同一任务的输入缓存须串行准备；不要同时启动 `--prepare-only` 与会准备相同缓存的生成 worker。以下循环按顺序执行各条件。

```bash
set -euo pipefail
export PYTHONPATH=src
export MODEL_SNAPSHOT=/path/to/Qwen2.5-VL-7B-Instruct/snapshot
export HF_HOME=/path/to/huggingface_cache
export HEAD_RUN_ROOT=outputs/ablations/random_head_snapkv7b

for task in nextqa_oe_test vatex_test; do
  python -m kvrepack.ablations.run_head_selection \
    --task "$task" --strategy similarity --limit 20 \
    --pretrained "$MODEL_SNAPSHOT" --hf-home "$HF_HOME" \
    --cache "$HEAD_RUN_ROOT/cache" --output "$HEAD_RUN_ROOT/$task/similarity"
  for seed in 0 1 2; do
    python -m kvrepack.ablations.run_head_selection \
      --task "$task" --strategy random_global --seed "$seed" --limit 20 \
      --pretrained "$MODEL_SNAPSHOT" --hf-home "$HF_HOME" \
      --cache "$HEAD_RUN_ROOT/cache" --output "$HEAD_RUN_ROOT/$task/random_global_seed$seed"
  done
done

python -m kvrepack.ablations.head_selection_report \
  --run-root "$HEAD_RUN_ROOT" --hf-home "$HF_HOME" \
  --output "$HEAD_RUN_ROOT/report" --limit 20
```

可先使用独立输出目录运行 `--limit 1` 检查模型和审计链路；正式20条不能与 smoke 记录混合。只准备 CPU 输入时使用 `--prepare-only`，并选择独立输出目录。评分需预先准备 NLTK 资源；VATEX METEOR 还需要 Java 和对应本地资源，缺项须如实报告。

## 评分与结论边界

NExTQA 使用官方 WUPS；VATEX 使用官方 corpus BLEU-4、METEOR、ROUGE-L 和 CIDEr。分别列出三个 Random 种子的得分，并报告均值与样本标准差（`ddof=1`）。同一批20条复用三个种子，并不产生60个独立输入。

NExTQA 可对“原策略−三个 Random 种子的逐输入均值”计算按源视频聚类的配对 bootstrap 区间；区间条件于这三个固定种子，不把 head 或 token 当作独立样本。VATEX 语料级指标差值作描述性报告；小子集 CIDEr 的文档频率依赖这20条参考集，不能与历史200条或全量的 CIDEr直接比较。

首轮是验证实现与评估效应方向的小样本实验。原策略是否优于 Random 由实际结果决定；差异不明显、方向反转或跨种子不稳定时均应照实记录，不能据此预设全部方法、模型和数据集的结论。

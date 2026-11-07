# Spot / MMCoQA Random-head 图片实验

本入口补充两种方法：`hybridkv_kvrepack` 与 `snapkv_kvrepack`。参考固定提交为 `6c65bf801ab1e5d0cd85f1d24991e3d5399bebcf`；其两个选头适配器及当前两份压缩核心逐字节对应。基础图片流程来自本分支的 MileBench 实现，运行不依赖另一个工作区。当前仅完成脚本准备和 CPU 验证，尚未运行真实模型、GPU 或任务分数。

这里随机的是“执行合并的原生 KV head 集合”。原动态/静态分类、每头预算、候选构造、相邻合并、raw-K 重施位置编码及 V 聚合继续由原核心执行。没有使用 RandomKV token 基线、随机预算或 `HEAD_CLASSIFY=random`。

## 固定实验设置

| 项目 | 设置 |
|---|---|
| 图片任务 | `Spot-the-Diff`、`MMCoQA`，各原标注前200条 |
| 模型 | Qwen2.5-VL-7B-Instruct，28层×4个原生KV头，共112头 |
| 方法/任务运行 | 两种方法×两个任务，共4个独立进程；每进程加载一次模型 |
| 条件 | `similarity`、`random_global_seed0`、`random_global_seed1`、`random_global_seed2` |
| Hybrid 参数 | head gate 0.5，compress ratio 0.5，alpha 2；选56头 |
| SnapKV 参数 | head gate 0.3，compress ratio 0.7，alpha 2；选34头 |
| 预算 | `BUDGET_RATIO=0.1`；图片原入口 `BUDGET=512`、`RATIO=0.1` |
| 生成 | BF16加载并核对实际dtype，FA2，greedy，512个新token，至少1个，repetition penalty 1.05 |
| 随机性 | 模型seed42；选头局部seed0/1/2，不消耗全局模型RNG；每条件重置相同模型RNG |

默认共请求 `2×2×200×4=3200` 次生成，尚未执行。局部seed在全112头上均匀无放回产生一次模型级固定排名，同seed用于全部样本及两个图片任务。只匹配全局头数，不约束每层选头数；不会逐样本重抽或为过门禁改变预算。

每个图片样本仅预处理一次，四条件复用同一份GPU输入tensor mapping；每条件前后检查所有输入tensor指纹，并清理生成缓存、RoPE位置及相关prefill状态。完整四条件通过配对门禁后才原子提交一个样本文件。

与视频参考有意区分的输入设置：沿用图片RGB→JPEG、图前文后、双像素floor、原上下文截断与512-token生成；不是视频64帧/128-token/WUPS协议。绝对BUDGET也保留图片入口512而非视频入口1024，实际每头预算及全部参数会记录，四条件内完全一致。普通图片profile继续默认auto dtype；此专用配方明确使用BF16加载。

## 当前可运行的无 GPU 规划

从仓库根目录执行，按本机数据位置设置：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/run_milebench_random_heads.py \
  --data-root /root/autodl-tmp/HybridKV-opensource/data/milebench \
  --hf-home /root/autodl-tmp/HybridKV-opensource/data/huggingface_cache \
  --output-root outputs/ablations/milebench_random_head_spot_mmcoqa
```

默认只输出4个配对实验的解析计划、参数和请求生成数，不读取模型/数据、不创建目录、不下载、不导入推理包。可以传入本地模型snapshot路径 `--pretrained /path/to/Qwen2.5-VL-7B-Instruct`。本地任意名称路径按7B显式校验；实际模型的28×4头布局还会在加载后核对。

单独选择一个任务或方法：

```bash
PYTHONPATH=src python scripts/run_milebench_random_heads.py \
  --data-root /path/to/milebench --output-root outputs/ablations/random_head_spot \
  --tasks Spot-the-Diff --methods hybridkv_kvrepack snapkv_kvrepack
```

支持显式 `--limit 1` smoke、`--head-seeds 0 1 2`、`--seed 42`、`--device 0`、`--max-new-tokens 512`。这些覆盖会记录进合同，不能将不同配方输出混称同一实验。正式默认limit200；超过200或实际标注不足时拒绝静默扩展/截断。

## GPU 开启后的运行

准备本仓库CUDA内核及依赖，在确认GPU阶段开启后，对同一命令加 `--execute`：

```bash
PYTHONPATH=src python scripts/run_milebench_random_heads.py \
  --data-root /root/autodl-tmp/HybridKV-opensource/data/milebench \
  --hf-home /root/autodl-tmp/HybridKV-opensource/data/huggingface_cache \
  --pretrained Qwen/Qwen2.5-VL-7B-Instruct \
  --output-root outputs/ablations/milebench_random_head_spot_mmcoqa \
  --execute
```

本轮没有执行该命令。先使用独立新目录 `--limit 1` 做两方法/两任务连通性检查，再选择另一新目录跑200条。程序先预检所有选中图片，再加载模型；每method/task生成完毕后立即离线评分，然后进入下一个运行。模型失败、实际合并不足或评分失败时停止，保存诊断状态；不会跳过输入、重抽随机集合或补零。输出必须使用新目录，目前不提供resume或覆盖已有运行。

## 门禁与产物

四条件必须拥有同一输入指纹、同一原生similarity mask、同一预算/候选及非选头配置。报告会独立重算随机排名和真实有效合并集合，不能只凭记录中的 `fairness=passed` 判定。Hybrid要求全部56个选中头实际发生有效合并；SnapKV要求全部34头有效合并，并保持原匹配工作量和prefill缓存形状。短输入若不能满足门禁会保留失败，不能把“选中”当作“执行”。

Hybrid的异质每头预算可能使不同head集合移除的token总量不同，这是选头干预的后果，按参考保留并报告，不重新分配预算以强行对齐。SnapKV的原适配器另要求有效工作量匹配，仍保持原门禁。

目录结构：

```text
<output-root>/controller_state.json
<output-root>/<method>/<task>/
  resolved_config.json / provenance.json / run_state.json / run.log
  run_contract.json / model_runtime.json / ablation_state.json
  samples/000000.json ...
  report/
    conditions/<condition>/eval.json / eval_score.json / evaluation_metadata.json
    scores.csv / paired_scores.csv / selector_audit.csv
    summary.json / RESULTS.md
```

合同冻结参考提交、源码SHA、标注SHA、原样本顺序与图片数分桶生成顺序、模型和生成参数。每样本文件包含四条件预测、生成token/EOS/上限状态、输入指纹和真实merge审计。只完整提交的样本计入进度；子集完成与完整200条在评分元数据中明确区分。

仅离线重评分（无需GPU或模型）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m kvrepack.ablations.milebench_head_selection_report \
  --run-root outputs/ablations/milebench_random_head_spot_mmcoqa/hybridkv_kvrepack/Spot-the-Diff \
  --output /path/to/new-report
```

Spot沿用原Rouge-L F，MMCoQA沿用原答案子串命中Accuracy。报告保存0—1原指标和单次转换的百分制、各seed得分、Random均值/样本标准差及 `similarity − random` 逐输入配对差。原eval_score的三位字符串不用于推算精确配对差；用原评分函数重新获得未舍入分数。三个seed复用同一批输入，不产生600个独立样本；未定义图片聚类单位，因此不套用NExTQA源视频bootstrap置信区间。

来源与CPU证据见 `provenance/random_head_selector_sources.json`、`provenance/milebench_sources.json` 和 `provenance/milebench_random_head_validation.json`。这些检查不能代替真实GPU执行与模型效果验证。

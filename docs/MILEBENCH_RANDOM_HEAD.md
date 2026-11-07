# Spot / MMCoQA Random-head 图片实验

2026-11-08当前协议：用户已授权GPU，并确认使用官方MileBench子集全量。Qwen2.5-VL-7B-Instruct上，HybridKV-KVRepack和SnapKV-KVRepack分别运行Spot-the-Diff、MMCoQA，每个样本包含 `similarity`、`random_global_seed0`、`random_global_seed1` 三个条件。已完成 `2方法×2任务×200样本×3条件=2400` 次正式生成。

MileBench作者从源任务抽样构建基准子集，见[论文§3.2](https://arxiv.org/pdf/2404.18532)；Spot-the-Diff和MMCoQA官方v1子集各200条。本轮读取这两份子集的全部标注。`--all-samples` 完整读取标注，不设条目cap，并冻结完整文件SHA与总条数；它与显式 `--limit` 互斥。本轮没有再对这些子集抽样，也没有运行原始源数据集的全部测试集。

当前正式根为 `outputs/ablations/random_head_qwen7b_all_seed01_v2_20261108`，四组均complete，最终独立审计及完整核分通过。2400次生成全部EOS结束，无512-token上限截断，GPU已释放，不启动重复运行。详见 [Qwen7B结果](MILEBENCH_RANDOM_HEAD_QWEN7B_RESULTS.md)、[精简证据包审计](../provenance/random_head_qwen7b_all_seed01_20261108/audit/audit.json) 和 [四组结果CSV](../provenance/random_head_qwen7b_all_seed01_20261108/audit/results.csv)。两方法/两任务各1条的独立smoke此前已通过，含12次生成及评分；最新CPU测试共161项，160通过、1跳过。这些GPU证据仅覆盖本轮Qwen7B两方法、两个官方子集，不能推广为全部模型/方法已验证，也不构成性能验证。

## 选头干预与固定设置

参考提交固定为 `6c65bf801ab1e5d0cd85f1d24991e3d5399bebcf`。基础图片流程来自本分支的MileBench实现，运行不依赖另一工作区。随机干预只替换“执行合并的原生KV head集合”，原动态/静态分类、每头预算、候选构造、相邻合并、raw-K重施位置编码及V聚合继续由原核心执行。没有使用RandomKV token基线、随机预算或 `HEAD_CLASSIFY=random`。

| 项目 | 当前设置 |
|---|---|
| 图片任务 | 官方MileBench v1的 `Spot-the-Diff`、`MMCoQA` 全部标注，各200条 |
| 模型 | Qwen2.5-VL-7B-Instruct，28层×4个原生KV头，共112头 |
| 方法 | `hybridkv_kvrepack`、`snapkv_kvrepack`；每方法/任务独立进程，每进程加载一次模型 |
| 条件 | `similarity`、`random_global_seed0`、`random_global_seed1` |
| Hybrid参数 | head gate 0.5，compress ratio 0.5，alpha 2；选中56头 |
| SnapKV参数 | head gate 0.3，compress ratio 0.7，alpha 2；选中34头 |
| 预算 | `BUDGET_RATIO=0.1`；图片原入口 `BUDGET=512`、`RATIO=0.1` |
| 生成 | BF16加载并核对实际dtype，FA2，greedy，最多512个新token、至少1个，repetition penalty 1.05 |
| 随机性 | 模型seed42；选头局部seed0/1，不消耗全局模型RNG；每条件重置相同模型RNG |

局部seed在全部112头上均匀无放回产生一次模型级固定排名，同seed用于全部样本及两个图片任务。只匹配全局选中头数，不约束每层选头数；不会逐样本重抽或为满足审计修改预算。

每个图片样本仅预处理一次，三个条件复用同一份GPU输入tensor mapping；每条件前后检查所有输入tensor指纹，并清理生成缓存、RoPE位置及相关prefill状态。完整三个条件通过配对门禁后才原子提交一个样本文件。

输入沿用图片RGB→JPEG、图前文后、双像素floor、原上下文截断与512-token生成。绝对BUDGET保留图片入口512；实际每头预算及全部参数会记录，三个条件内一致。视频64帧/128-token/WUPS设置不适用于本实验。普通图片profile继续默认auto dtype，本专用配方明确使用BF16加载。

## 规划、smoke与执行

从仓库根目录规划本轮协议，按实际路径设置数据和模型缓存，输出使用新目录：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/run_milebench_random_heads.py \
  --data-root /root/autodl-tmp/HybridKV-opensource/data/milebench \
  --hf-home /root/autodl-tmp/HybridKV-opensource/data/huggingface_cache \
  --pretrained Qwen/Qwen2.5-VL-7B-Instruct \
  --all-samples --head-seeds 0 1 \
  --output-root /path/to/new-output
```

不加 `--execute` 只输出4个配对实验的解析计划、参数和请求生成数，不加载模型、不创建目录、不下载。全量规划会读取标注以冻结总数和SHA；实际执行前再核对完整标注、图片及样本覆盖。本地模型snapshot可通过 `--pretrained /path/to/Qwen2.5-VL-7B-Instruct` 指定，加载后会核对28×4的头布局。

需要smoke时，使用独立新目录，将 `--all-samples` 替换为 `--limit 1`，保留 `--head-seeds 0 1`。通过后在另一个新目录使用上述完整命令并加 `--execute` 开始GPU生成及离线评分。可用 `--tasks Spot-the-Diff` 或 `--methods hybridkv_kvrepack` 限定单个运行。显式覆盖 `--seed`、`--device`、`--max-new-tokens` 等会记录进合同，不能将不同配方输出混称同一实验。

CLI默认head seeds0/1/2仅为历史参数兼容；本轮明确使用 `--all-samples --head-seeds 0 1`，只运行seed0/1。通用控制器不提供已有目录的resume或覆盖；不要在当前正式根重启命令，也不要重复生成已完成的Hybrid/Spot。

程序先预检所有选中图片，再加载模型；每method/task生成完毕后立即离线评分，然后进入下一运行。模型、配对审计或评分失败时停止并保存诊断，不跳过样本、不重抽随机集合、不补零。

## 有效合并审计与当前修复

三个条件必须拥有同一输入指纹、同一原生similarity mask、同一预算/候选及非选头配置。报告独立重算随机排名、实际调用和有效合并集合，不能只凭记录中的 `fairness=passed` 判定。

选中头数、合并调用数和有效合并数分别记录。Hybrid原生分支可能使选中头自然不进入合并调用，也可能调用后未减少token（noop）。当前审计恢复参考实现的这一语义：选中头仍为56个，允许上述观测，但每条件必须存在有效合并，且每次调用都须来自所选头、无重复，并核对原生每头预算。缺调用原因只作为观测边界，不能据此断言某头是否具备合并资格。SnapKV仍要求34个选中头全部有效合并，保持原有效工作量匹配及prefill缓存形状。

原控制器在Hybrid/MMCoQA第2个生成样本（原 `sample_id=18`）后置审计中，错误地将56个选中头等同56个有效合并头，因而停止。修复只调整这一过严的后置校验，未改预算、核心算法或随机mask。Hybrid异质每头预算使不同head集合移除token总量可能不同，按参考保留并报告，不重新分配预算强行对齐；SnapKV的原工作量匹配门禁仍保留。

原失败根 `outputs/ablations/random_head_qwen7b_all_seed01_20261108` 保持原样。完整Hybrid/Spot的原合同、200个样本及报告按字节复制到v2根复用，未重复其600次GPU生成。v2根 `reuse_manifest.json` 记录复用文件SHA，`source_archives/first_controller` 冻结旧12份合同绑定源码。绑定源码和manifest保留用于追溯；复用保留原合同身份，最终独立审计已显式校验归档来源。

## 合同、结果与独立审计

每个方法/任务保存：

```text
<output-root>/controller_state.json
<output-root>/reuse_manifest.json
<output-root>/source_archives/first_controller/
<output-root>/<method>/<task>/
  resolved_config.json / provenance.json / run_state.json / run.log
  run_contract.json / model_runtime.json / ablation_state.json
  samples/000000.json ...
  report/
    conditions/<condition>/eval.json / eval_score.json / evaluation_metadata.json
    scores.csv / paired_scores.csv / selector_audit.csv
    summary.json / RESULTS.md
```

合同冻结参考提交、源码SHA、完整标注SHA/总数、原样本顺序与按图片数分桶后的生成顺序、模型和生成参数。每样本文件包含三个条件预测、生成token/EOS/上限状态、输入指纹和真实merge审计。仅完整提交的配对样本计入进度，最终审计要求覆盖完整冻结样本集合。

最终独立审计已通过，原报告位于 `outputs/audits/random_head_qwen7b_all_seed01_v2_20261108_final/audit.json`，精简包中保存同一审计及结果CSV。需要离线复查时，显式提供复用结果所需的旧源码归档，使用另一个尚不存在的输出目录：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python scripts/audit_milebench_random_head_run.py \
  --run-root outputs/ablations/random_head_qwen7b_all_seed01_v2_20261108 \
  --source-archive outputs/ablations/random_head_qwen7b_all_seed01_v2_20261108/source_archives/first_controller \
  --output outputs/audits/random_head_qwen7b_all_seed01_v2_20261108_recheck_new
```

最终审计已验证完整覆盖、合同与来源哈希、逐条件配对、真实合并工作、生成上限状态及分数重算。需要对新合同的单个完整运行离线重评分时，可使用以下入口；它同样要求新的输出目录，来源变化或旧合同复用应交由上述带归档参数的独立审计处理：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m kvrepack.ablations.milebench_head_selection_report \
  --run-root /path/to/complete-method-task-run \
  --output /path/to/new-report
```

Spot沿用原Rouge-L F，MMCoQA沿用原答案子串命中Accuracy。报告保存0—1原指标和单次转换的百分制、各seed得分、Random均值/样本标准差及 `similarity − random` 逐输入配对差。原eval_score中的三位字符串不用于推算精确配对差；重新调用原评分函数获取未舍入分数。两个seed复用相同200个输入，不产生400个独立样本；未定义图片聚类单位，因此不套用NExTQA源视频bootstrap置信区间。

来源与准备期CPU证据见 `provenance/random_head_selector_sources.json`、`provenance/milebench_sources.json` 和 `provenance/milebench_random_head_validation.json`。这些检查与当前限定范围的GPU执行分别记录，不能替代全部模型、算法正确性、历史分数或性能验证。

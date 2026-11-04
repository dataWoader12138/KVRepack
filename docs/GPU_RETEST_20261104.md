# GPU 小样本数值复测（2026-11-04）

在 A800 80GB 上，Qwen2.5-VL-3B 的 **SnapKV-KVRepack 与 MadaKV-KVRepack** 均通过原实现与整理版的严格数值对照。NextQA OE/test 与 VATEX/test 各取 doc_id 0、1，共4个真实视频、8组方法/样本对照、16次模型生成。输入、生成token及文本全部相同；逐步全词表原始logits最大绝对差为 **0**，prefill及生成结束的36层K/V形状、内容SHA-256逐张量一致。

本次执行的是两种改进算法，不包括SnapKV/MadaKV基线。比较对象是本次分别重跑的原artifact实现和KVRepack实现；旧逐样本生成日志已不在本机，不能据此声明历史200样本任务分数已复现。

## 实际设置

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA A800-SXM4-80GB，单卡、独立进程 |
| 模型 | Qwen2.5-VL-3B-Instruct，本地snapshot `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| 输入 | 任务原prompt；64帧；batch=1；min_pixels=200704，max_pixels=1605632；预处理一次后逐tensor核对输入hash |
| 计算 | BF16、FlashAttention 2；seed=0；TF32关闭 |
| 生成 | greedy、use_cache=True、max_new_tokens=64，全部样本在64前生成EOS |
| 预算 | BUDGET_RATIO=0.10，window=32，score_correction=none |
| SnapKV-KVRepack | simtop=0.70，cr=0.90，alpha=2，kernel=5，avgpool |
| MadaKV-KVRepack | simtop=0.50，cr=0.70，alpha=2，threshold=0.8，BUDGET=1024 |
| 比较 | atol=rtol=0；所有步骤×151936词表logits；两阶段全部36层K/V内容hash |

Snap设置与参数表Qwen3B两个数据集条目相同。Mada设置与VATEX条目相同；NextQA用同组参数验证搬迁等价性，**未测试其历史最优cr=0.50**，未重新寻优。输入视频、Arrow、tensor及代码hash见[证据目录](../provenance/gpu_retest_20261104/)。

## 逐样本对照

prefill/final列为每层实际存储的K序列长度，V同长；本次各层长度相同，完整形状及逐层hash见JSON。输入9607的样本压缩后为992，证明执行了实际压缩。

| 方法 | 样本 | 输入token | 生成token（含EOS） | Prefill KV | Final KV | logits最大差 | token及两阶段KV |
|---|---|---:|---:|---:|---:|---:|---|
| snapkv_kvrepack | nextqa_oe_test:0 | 9607 | 2 | 992 | 993 | 0 | 一致 |
| snapkv_kvrepack | nextqa_oe_test:1 | 9616 | 4 | 993 | 996 | 0 | 一致 |
| snapkv_kvrepack | vatex_test:0 | 9736 | 13 | 1005 | 1017 | 0 | 一致 |
| snapkv_kvrepack | vatex_test:1 | 9000 | 12 | 932 | 943 | 0 | 一致 |
| madakv_kvrepack | nextqa_oe_test:0 | 9607 | 2 | 992 | 993 | 0 | 一致 |
| madakv_kvrepack | nextqa_oe_test:1 | 9616 | 4 | 993 | 996 | 0 | 一致 |
| madakv_kvrepack | vatex_test:0 | 9736 | 10 | 1005 | 1014 | 0 | 一致 |
| madakv_kvrepack | vatex_test:1 | 9000 | 12 | 932 | 943 | 0 | 一致 |

合计比较 **8,964,224** 个logit值，无NaN/Inf，0处差异；比较 **1152** 对K/V张量内容hash，全部相同。生成数值来自真实GPU模型运行，比较器在CPU上读取保存结果逐项断言。

## 两样本子集任务分数

下面数值原版和整理版完全相同；NextQA用任务原WUPS（0–100），VATEX保持原始小数单位。评分复用任务原函数与本地WordNet、Stanford PTB和COCO scorer，禁用Python网络与下载。

| 方法 | NextQA WUPS | VATEX BLEU-4 | VATEX ROUGE-L | VATEX CIDEr | 所有指标原新差值 |
|---|---:|---:|---:|---:|---:|
| snapkv_kvrepack | 52.197802198 | 3.83466475e-09 | 0.392679151 | 0.491785491 | 0 |
| madakv_kvrepack | 52.197802198 | 3.92995648e-09 | 0.389591601 | 0.458065734 | 0 |

VATEX两条样本缺少完整4-gram匹配，因此BLEU-4接近零；这是原scorer的子集结果。METEOR及四项综合分未计算。尤其CIDEr的document frequency只由本次两个参考组估计，不能与历史200条或全量分数比较，也不能用于方法优劣判断。[评分JSON](../provenance/gpu_retest_20261104/subset_metrics.json)保留每条预测、数据hash、评分函数AST与依赖版本。

## 证据与本机复查

- [汇总与文件SHA](../provenance/gpu_retest_20261104/artifact_index.json)、[输入与视频SHA](../provenance/gpu_retest_20261104/inputs_manifest.json)。
- [SnapKV对照](../provenance/gpu_retest_20261104/snapkv_comparison.md)、[MadaKV对照](../provenance/gpu_retest_20261104/madakv_comparison.md)，同目录JSON保存全部逐层/逐步检查。
- 同目录保存本次实际运行的worker、比较器、结果JSON及日志快照。worker中的本机路径是历史执行记录，跨机器运行使用项目统一入口；其不是新增的通用benchmark入口。
- 原始logits `.pt` 与预处理输入 `.pt` 留在本机 `/root/kvrepack-gpu-work/`，不随Git发布。无权重、视频、凭证或外部judge调用。

本机可重新检查SnapKV的完整logits（MadaKV替换文件前缀）：

```bash
/root/miniconda3/envs/hybridkv/bin/python /root/kvrepack-gpu-work/compare_gpu_runs.py \
  --reference-json /root/kvrepack-gpu-work/results/snapkv_original.json \
  --candidate-json /root/kvrepack-gpu-work/results/snapkv_repack.json \
  --reference-scores /root/kvrepack-gpu-work/results/snapkv_original_scores.pt \
  --candidate-scores /root/kvrepack-gpu-work/results/snapkv_repack_scores.pt \
  --output-dir /root/kvrepack-gpu-work/results/snapkv_comparison
```

## 范围与启动修复

这项证据覆盖同输入的完整模型forward/generate、真实CUDA与FlashAttention，以及两种改进的压缩数值。隔离worker直接调用项目monkeypatch和模型，**未执行统一CLI/adapter端到端流程**；NextQA上限64与adapter默认128不同，本次均提前EOS。没有据此验证全量分数、其他压缩方法、7B、LLaVA、自定义`kvrepack_kernel`、新机器安装或性能；带CPU拷贝和hash的计时不能作为速度测试。

准备阶段另发现打包manifest已升级schema=2，而runner只接受schema=1，会阻止正式入口启动。本次仅让两处manifest校验同时接受1/2，保留原路径边界及SHA核验，补充真实schema2通过与篡改拒绝测试，并更新runner来源hash；未改任何压缩公式或模型实现。这项修复的CPU测试与来源检查通过，不代表完整CLI已经运行。

最终CPU回归共50项，49通过、1项需要显式原来源参数的检查跳过；`check_sources.py`通过30个核心目标、33个本地模块/adapter、329个vendor Python及2382个build输入。日志与JSON随证据保存；这些计数是结构/来源检查，不扩大上述GPU范围。

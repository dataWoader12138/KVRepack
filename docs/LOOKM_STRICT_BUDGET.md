# LookM严格总预算修复与源码核查

**最新：严格10%两任务各200条重跑已完成，结果及CSV见[正式结果页](LOOKM_STRICT_RESULTS_20261109.md)。** 400条真实缓存和独立原任务评分复核均通过；仅重跑LookM，旧legacy和已停止任务保持原状态。

2026-11-09：用户授权的严格预算正式重跑已完成。UTC02:49:37通过`run_lookm_video.py --limit 200`启动，仅LookM、Qwen2.5-VL-7B、64帧、两个任务各原始前200行，共400次全新生成。控制器PID/PGID58558；`lookm_state.json`状态为`complete`，完成时间`2026-11-09T03:32:20.736820+00:00`，端到端约42分43秒。源码按提交`bc8ebee`及运行契约保持冻结，独立审计与评分复核均通过。

32个近期位置计入10%总prefill额度，契约使用`total_prefill_including_recent_v2`；不复用旧预测。旧400条“10%＋32”结果仅为legacy历史，不能代表本轮strict结果；Hybrid119条及原RoPE任务继续保持停止，不因本轮重跑恢复。启动前已有260项CPU suite（259通过、1项已有跳过）、20项独立oracle及两任务各1条真实7B strict smoke通过，下文保留其验证细节；这些前置检查不是正式两任务各200条分数，未为本次启动重复整套测试。

## 总预算定义

输入prefill长度为N，预算比例r时，每个原生KV head容量`C=floor(r*N)`。近期位置数`w=min(32,C)`，历史top-k为`C−w`，两者相加恰好C。所有28层、每层4个原生KV头都检查真实缓存shape，而非只检查配置值；总活动槽位不超过`28*4*floor(.1N)`。

C小于32时缩短recent窗口。历史top-k为0时，早期prune层只保留recent，后续merge层仍把未保留历史合并到recent pivots；C等于N直接返回原K/V。N过短使C=0时明确报错，不能偷偷退回完整KV。该口径是prefill压缩预算；后续生成token按原解码逻辑追加并单独记账，不额外修改生成时的驱逐策略。

## 原始与官方源码核查

原工作区`HybridKV-artifact/hybridkv/baseline_utils.py`保留不动。官方参照固定为[LOOK-M commit ecf0f51](https://github.com/SUSTechBruce/LOOK-M/tree/ecf0f51a9c416c2d85e47faf2638502f01a6d748)，主要对照[TextPivot实现](https://github.com/SUSTechBruce/LOOK-M/blob/ecf0f51a9c416c2d85e47faf2638502f01a6d748/LLaVA-mix_merge_v1/llava/model/kv_token_merge/modify_llama.py)。

- 原适配先选`floor(.1N)`历史再拼32，超出总额度；现先确定总C，再扣recent。
- 原适配把indices跨batch/head flatten后用全局isin，导致其他head选择的位置被误判为本head已选择。现按每个(batch,head)独立构造补集，每个未选历史token恰好参与一次合并，与官方head-local mask语义一致。
- 保留selected与recent共同作为pivot候选、原TextPivot pair averaging及include_self=True权重。一个pivot有m个evicted时，其结果为`[p + .5*sum(e_j+p)]/(m+1)`，不擅改成普通组均值。
- 移除硬编码128，按真实head_dim运算；零key使用稳定归一化，FP16/BF16的cosine匹配升为FP32，FP64保持精度，K/V聚合及输出dtype保持原样。

这仍是仓库的Qwen LookM适配：动态模态mask、GQA分组mean、last-window query观测分数、前5%层只prune均来自原适配。官方实现基于原LLaVA固定图像布局和累积attention；本次不把这套配置硬移到Qwen视频，也不宣称官方论文模型和完整实验已经复现。

## 来源与验证

`provenance/core_sources.json`保留真实原文件SHA和修改前AST；`provenance/patches/lookm_strict_budget_v2.patch`记录准确行为变化。来源检查会反向恢复旧integration字节，验证旧SHA、两函数before/after AST，并证明其余AST未变，然后与实际原LookM归一化源码比对，避免只刷新哈希掩盖变化。

CPU suite共260项，259通过、1项已有跳过；显式1条评分入口修复后，另有6项受影响报告检查通过；另有20项独立oracle检查，覆盖cross-head/batch隔离、每个evicted贡献、k0 recent-pivot、ratio1 identity、短输入/GQA/zero norm/FP16溢出/FP64精度。完整原工作区比对因无关的opensource LLaVA源文件缺失未完成；发布目录来源检查及本次修改文件的真实原源码定向比对通过。

新契约/配置/缓存记录使用`total_prefill_including_recent_v2`，新报告显式`strict_budget_passed`。旧无v2字段记录只识别legacy；worker续跑和单方法launcher均拒绝将旧样本加入新结果。旧归档和旧报告字节不改。

## 实际GPU缓存核验

模型Qwen2.5-VL-7B、64帧，每任务仅1条，全部28层×4原生KV头验证通过：

| 任务 | 输入N | 总容量C | 近期位置 | 历史候选 | 实际prefill比例 |
|---|---:|---:|---:|---:|---:|
| NExTQA | 9607 | 960 | 32 | 928 | 9.992714% |
| VATEX | 9736 | 973 | 32 | 941 | 9.993837% |

最终缓存扣除新解码token增量后，与真实prefill尺寸互验一致。单条Native评分链路也已跑通，报告标smoke、formal_result=false；不把其分数当作新200条结果。首次小样本评分曾因CLI只允许200条中止；补齐显式1/200入口后续跑，已完成Next生成未重复。此记录保留在日志中。

[验证汇总](../provenance/lookm_strict_budget_20261109/summary.json) · [源码独立检查与官方参照](../provenance/lookm_strict_budget_20261109/independent_source_audit.json) · [原源码与可逆补丁比对](../provenance/lookm_strict_budget_20261109/source_replay_check.json) · [可审阅补丁](../provenance/patches/lookm_strict_budget_v2.patch)。

## 严格预算运行入口

安装依赖并准备模型、视频/Arrow及NLTK/Java资源后，从仓库根目录使用新输出目录：

```bash
python scripts/run_lookm_video.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct \
  --hf-home /path/to/huggingface_cache \
  --output outputs/benchmarks/lookm_strict_200_REPRO
```

默认仅LookM、两个任务各200条；`--limit 1`用于独立小样本检查。不得复用旧“10%＋32”的已完成目录。Hybrid和RoPE已停止实验仍保持停止。

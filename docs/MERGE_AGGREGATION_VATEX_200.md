# VATEX 固定分组合并与端到端时延对照

用户最新指示是不再补跑“原方法＋KVRepack”基线。当前继续 Qwen2.5-VL-7B、VATEX 原始前200条：HybridKV 与 SnapKV 各测 LookM 式聚合、AsymKV 式聚合，共四组、800次新方法生成。原三策略任务已在 SnapKV 完整114条后停止；其中228次 LookM/AsymKV 结果原样继承，只补572次。114条KVRepack基线另作同输入配对前缀对照，不称为完整200条基线。旧的 Random-head、RoPE 与完整 LookM 实验不在此次范围内。

## 固定条件

- 模型：本机固定的 Qwen2.5-VL-7B-Instruct snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5`；BF16，batch size 1。
- 输入：原 VATEX test Arrow 的 `doc_id=0..199`，保留行号，不过滤或替换。64 帧；使用原视频读取、采样、processor 与提示词。
- 每个骨干内，三种聚合共享候选、importance、选头、相邻分组、组 anchor 与 raw-K 聚合后重施 mRoPE 的位置处理。
- 合并插入点：原 KVRepack 的分组构建结束之后、压缩 K/V 写入缓存之前。运行时从原函数 AST 生成适配版本，原核心文件保持不变。
- SnapKV：VATEX 配置 gate=.15、compression=.70、alpha=.5。
- HybridKV：gate=.50、compression=.50、alpha=2，保留原 head 分类与异质预算分布。
- 生成：greedy，最多 64 个新 token，repetition penalty=1.05；报告实际生成长度、EOS 和无 EOS 的截断数。

## 总预算

近期窗口包含在 10% 总活跃 prefill KV 位置预算中，不额外加 32。

SnapKV 每个 head 的容量为 `floor(.1*N)`，历史容量减去原 recent window。HybridKV 先执行原预算分配，然后以共同缩放系数减去实际 local window（包含原 chunk 余量），retrieval budget 保持 chunk 向下取整；28×4 个原生 KV heads 的全局活跃位置数不超过 `112*floor(.1*N)`。HybridKV 保留异质分配，不将其解释为逐 head 10% 上限。

所有三种策略使用相同预算适配，因此本轮 KVRepack 数值需要重新生成，不能拼接旧的 nominal 10% 历史分数。活跃 KV 位置预算也不等于总 GPU 内存预算：HybridKV 的 backing/offload storage 和 AsymKV 的 Fisher 工作空间单独存在。

## 迁入的聚合规则

LookM 迁入 TextPivot 的原算子顺序和 dtype：每个非 anchor 成员先与其原 anchor 平均，再 `scatter_reduce(mean, include_self=True)`。二元组为 75% anchor、25% 另一成员；多成员组由原 scatter 规则确定权重。分组仍来自 KVRepack，不迁入 LookM 的全局余弦分配、文本优先或 pruning。

AsymKV 复用逐维 Fisher 加权 K、V 求和、count clip=5。读取合并 V 时配套使用仓库现有 AsymKV LMA 的 `exp(score) / sum(exp(score)*count)`，再乘原 Vsum。SnapKV 直接调用现有 LMA 函数；HybridKV 对异质长度做 padding/mask 后批处理同一算子顺序。只在单 query decode 使用该读出，不创建完整 prompt 的密集注意力矩阵，不重复计算 FlashAttention 后再覆盖结果，也不引入逐 head 的 CPU 同步循环。

多 token 固定分组的 Fisher 聚合是对官方二元公式的扩展，不等同于其多轮压缩。其 pair 评分、sink、chunk pruning 与生成流程不迁入。

Fisher 必须来自本条已观察 prompt，在相同冻结权重上计算，不读取评测答案。其资源与时间计入 AsymKV。统计方式、cache 空间和完整性记录在每条 `fisher` 及源文件说明中；不能将本轮适配称为完整官方 AsymKV 流程的复现。

## 计时边界与控制

每个骨干用独立进程加载模型，两种新聚合通过可恢复的上下文依次运行。旧114条保留当时三策略轮换的真实顺序与原时延，fresh输入按 `doc_id % 2` 交替执行。复用记录逐条核对父bundle/contract/source快照SHA，预测、时延、审计字典本体不改；输入准备、计时/生成函数AST和全部算法依赖必须保持一致。

当前每个新策略先用前两条独立预热；每骨干四次预热不计入正式200条。正式样本每个策略重新执行视频读取和预处理，计时前后同步 GPU。

报告两种范围：

1. **完整端到端**：视频读取/解码、processor、传入 GPU、Fisher（AsymKV）、prefill、decode、文本解码。
2. **模型端到端**：输入已在 GPU 后，Fisher（AsymKV）加 prefill、decode；另列纯 generation 时间，避免视频处理掩盖方法差异。

模型加载、预热、AST 编译、计时后审计哈希、JSON 写盘和 CPU 评分不计入。每条正式运行只保留轻量分组/候选索引/mask 的 GPU 引用，结束计时后才转 CPU 并计算哈希。诊断 `--audit` 会额外做完整候选哈希，不能作为正式时延结果。

最终表含 mean/P50/P95、200 条累计时延、Fisher 和 generation 分解、输出长度与吞吐。质量指标使用同一 200 条参考的原生 BLEU-4、METEOR、ROUGE-L、CIDEr；subset CIDEr 不直接与全测试集 CIDEr 比较。单次每输入测量不等同于重复稳态微基准。

## 执行与复核

在具备依赖的环境、项目根目录执行；路径由使用者指定：

```bash
python scripts/run_merge_vatex.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct/snapshot \
  --hf-home /path/to/huggingface_cache \
  --output outputs/ablations/merge_vatex_first200 \
  --limit 200 --warmup 2

python -m kvrepack.ablations.merge_comparison_report \
  --run-root outputs/ablations/merge_vatex_first200 \
  --output outputs/ablations/merge_vatex_first200/report
```

`run_state.json` 和 `experiment_state.json` 记录实际状态，不能仅依据协议宣称运行完成。每条完整三条件结果原子提交到 `samples/`；源文件 SHA 冻结，实际输入、预算、候选、选头、分组、anchor 的哈希须匹配。报告再次独立复核这些记录，再调用本地任务评分器。

诊断 pilot 使用独立目录、`--limit 1` 或 `--limit 2`；其结果不混入正式 200 条。失败的联调记录保留，禁止用其时延填入正式结果表。

当前ROOT：`outputs/ablations/merge_vatex_mergeonly_200_20261112`。旧三策略ROOT `outputs/ablations/merge_vatex_200_20261112` 标为 `superseded_by_user_scope`，禁止恢复其baseline生成；它只提供不可变父记录。来源快照与迁移证据保存在 `provenance/merge_vatex_scope_change_20261112/`。

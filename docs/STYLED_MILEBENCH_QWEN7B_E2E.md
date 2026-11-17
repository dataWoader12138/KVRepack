# Qwen7B 图片合并方法与端到端时延

用户于2026-11-17明确授权本轮GPU实验。来源固定为GitHub提交`b9104d8`，工作分支`feat/qwen7b-styled-image-e2e`。旧图片分支与旧结果保持不动，视频实验分数/时延不转录为本轮图片结果。

当前状态：16组独立单条GPU联调及离线重审全部通过（16条评测、16条预热明确区分）。441项CPU suite为440通过、1个原artifact数值差分跳过；新增reporter的10项CPU测试全部通过。正式控制器PID/PG4922于UTC2026-11-17T00:17:26启动，输出根`outputs/benchmarks/styled_milebench_qwen7b_full_20261117`，实际进度见其中`experiment_state.json`与各`run_state.json`。正式3200条尚在运行，不将联调分数当作正式全量结果。准备、启动与检查证据在`provenance/styled_milebench_20261117/`。

## 范围

Spot-the-Diff、MMCoQA采用官方MileBench v1子集全部标注。本机各200条，按实际完整JSON读取，不设置200条截断；每个条件独立完成全部输入。

| 条件 | 聚合/实现 |
|---|---|
| HybridKV + KVRepack | 原KVRepack聚合 |
| HybridKV + LOOK-M-styled | 原head/candidates/groups/anchors下移植TextPivot聚合 |
| HybridKV + AsymKV-styled | 同组下Fisher加权K、V求和及原LMA修正 |
| SnapKV + KVRepack | 原KVRepack聚合 |
| SnapKV + LOOK-M-styled | 同组TextPivot聚合 |
| SnapKV + AsymKV-styled | 同组Fisher加权K、V求和及LMA |
| 独立LOOK-M | `b9104d8`的原版LOOK-M回放路径，保留原历史预算和原函数 |
| 独立AsymKV | 完整官方适配/生成驱动，非styled聚合移植 |

8条件×2任务×200条=3200正式生成。每条件/任务独立进程，模型只加载一次；GPU进程串行，batch1。每组额外1条预热独立保存，不参与分数或正式时延。小批smoke目录也与正式结果隔离。

图片附加依赖可用`python -m pip install -e '.[images]'`安装。模型权重、数据和另行编译的`kvrepack_kernel`仍须准备；当前在已有环境上验证，不将它称为全新机器的完整安装验证。

## 参数和原实现差异

模型为本地Qwen2.5-VL-7B-Instruct snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5`，A800-SXM4-80GB、BF16、FlashAttention2、模型seed42。

Hybrid使用gate=.50、compress=.50、alpha=2；Snap使用gate=.30、compress=.70、alpha=2。三种聚合保留同骨干的原始选头、候选、分组、anchor及raw-K重施mRoPE；采用`OriginalHybridMergeAdapter`、`OriginalSnapMergeAdapter`，不套用strict-total预算修正。名义`BUDGET_RATIO=.1`；图片入口`BUDGET=512`、`RATIO=.1`。实际保留容量逐条记录，不能将名义10%写成实际严格10%。

图像流程沿用已验证MileBench后端：RGB→JPEG、所有图片先于文本，context32768、n_tokens_per_image256、processor min_pixels7168、message floor200704、max_pixels1605632，greedy上限512新token。HF方法采用min1、repetition penalty1.05；原Asym为自身argmax驱动，不强加其不支持的HF min_new_tokens或重复惩罚。此差异写入合同和结果。

独立LOOK-M使用确切原版`LegacyLookMContext`，其history10%加32 recent及旧函数行为保留。独立Asym采用原sink32、chunk512、无absolute cache-size override，不修改其chunk/压缩触发策略。CPU完整预检的实际输入：Spot545–551，MMCoQA595–12657 tokens。原Asym在Spot短输入上可能不触发压缩，实际cache接近100%；MMCoQA样本44（3073 tokens）、196（1537）触发原驱动的N%512==1末token覆盖问题。这两条仍参与原实现全量评测，单列实际覆盖，不改prompt、不跳样、不将修复版冒充原版。

## 计时与核验

主要E2E为单请求同步墙钟：从`dataset[index]`之前开始，计入图片读取、RGB/JPEG、processor、H2D、Fisher（AsymKV适用）、视觉编码/prefill/decode、输出token到CPU及文本解码，停止时CUDA同步。该值按200个请求统计mean/median/p95/total。

模型/processor加载、预热、源码编译、每条输入与group哈希、诊断审计、JSON写盘、离线评分排除在响应E2E之外；另保存加载、正式循环和整个进程wall time。该E2E是预热后服务请求时延，数据预检和重复读取可能命中文件系统缓存，不能称为冷磁盘时延。

辅助GPU peak字段包含停表后的审计工作，不作为纯生成阶段的受控显存峰值比较。

AsymKV-styled的独立Fisher探测计入E2E并提供真实分阶段时延。独立原Asym的所有Fisher在原generate内部，计入generation/E2E，不能把无法单独测量的Fisher开销标为0。E2E还受输出长度及原decoder差异影响，不是纯合并算子耗时。

每组合同冻结源码/模型元数据/标注SHA、完整样本ID、图像参数、算法环境及解码参数。结果保留完整预测、token/EOS/截断、输入指纹、实际merge/group/预算审计、Asym实际token覆盖。按同骨干/任务/输入跨三个merge条件核对输入指纹和原生head/candidates/groups/anchors签名一致。

Spot沿用原ROUGE-L F，MMCoQA沿用原答案子串Accuracy。离线评分要求全部对应样本完整；smoke不能混称全量。最终输出比较CSV、逐输入时延CSV、原评分JSON、完整总结及来源证据。

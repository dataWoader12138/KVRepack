# Qwen7B 图片合并方法与端到端时延

用户于2026-11-17明确授权本轮GPU实验。来源固定为GitHub提交`b9104d8`，工作分支`feat/qwen7b-styled-image-e2e`。旧图片分支与旧结果保持不动，视频实验分数/时延不转录为本轮图片结果。

当前状态：**16/16组、3200条正式生成及评分全部完成并独立核验通过**，另16条预热明确排除。正式UTC 2026-11-17 00:17:27–01:37:37，控制器4810.293秒（约80分钟），GPU已释放。441项CPU suite为440通过、1个原artifact数值差分跳过；另reporter10项、exporter13项测试通过。完整结果和原始证据见下方，准备与启动证据保留于`provenance/styled_milebench_20261117/`，其中`preparation.json`记录的是启动时状态。

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


## 正式结果（2026-11-17）

各组均为完整官方标注200条。分数为0–100，E2E为每请求秒数；这是本轮新运行的分数，不能用历史论文表中的分数替换。

| 方法 | Spot ROUGE-L | E2E均值(s) | MMCoQA Acc. | E2E均值(s) |
|---|---:|---:|---:|---:|
| HybridKV + AsymKV-styled | 27.10 | 1.514 | 53.50 | 2.898 |
| HybridKV + LookM-styled | 29.23 | 0.799 | 59.50 | 2.294 |
| HybridKV + KVRepack | 29.09 | 0.987 | 63.50 | 1.500 |
| SnapKV + AsymKV-styled | 27.78 | 0.979 | 48.00 | 2.355 |
| SnapKV + LookM-styled | 29.07 | 0.578 | 57.50 | 1.121 |
| SnapKV + KVRepack | 28.96 | 0.557 | 59.00 | 1.111 |
| AsymKV (original) | 28.77 | 0.719 | 49.00 | 2.688 |
| LookM (original) | 28.74 | 0.740 | 45.00 | 1.323 |

MMCoQA上Hybrid + KVRepack分别比LookM-styled、AsymKV-styled高4.0和10.0分；Snap + KVRepack分别高1.5和11.0分。Spot上LookM-styled比KVRepack略高（Hybrid约0.143分，Snap约0.115分），本轮不能宣称KVRepack在所有条件中最好。E2E受生成长度与达到512上限的样本影响，不能据此直接推算纯合并算子加速比。

[8行双任务CSV](../provenance/styled_milebench_20261117/results_wide.csv)；[16行完整统计CSV](../provenance/styled_milebench_20261117/final/report/comparison.csv)；[3200条逐输入时延](../provenance/styled_milebench_20261117/final/report/per_input_latency.csv)；[完整机器可读报告](../provenance/styled_milebench_20261117/final/report/summary.json)。

### 实际缓存与原实现差异

同骨干的三种聚合逐输入的选头、候选、group、anchor和实际prefill容量完全一致。平均物理保留率：Hybrid Spot16.368%、MMCoQA11.694%；Snap Spot15.720%、MMCoQA11.493%。独立LookM分别为15.720%、11.493%。

独立原Asym在Spot全部200条均未触发压缩，物理保留率100%；MMCoQA平均45.630%（16.098%–100%），809次压缩事件，10条未压缩。原MMCoQA样本44和196分别覆盖3072/3073、1536/1537输入token；各遗漏末token1个，两条均保留在完整评分中。该原驱动与HF解码参数也有前述差异，不能把独立Asym当成与styled方法严格等物理预算、等decoder的对照。

达到512上限且无EOS的数量如下，全部保留，无删样或调参重跑；所有组均无early EOS/post-EOS生成。

| 条件 | Spot | MMCoQA |
|---|---:|---:|
| Hybrid + AsymKV-styled | 4 | 8 |
| Hybrid + LookM-styled | 0 | 8 |
| Hybrid + KVRepack | 1 | 0 |
| Snap + AsymKV-styled | 4 | 11 |
| Snap + LookM-styled | 0 | 0 |
| Snap + KVRepack | 0 | 0 |
| 原AsymKV | 1 | 6 |
| 原LookM | 3 | 0 |

### 核验与归档

[完整发布](../provenance/styled_milebench_20261117/final/README.txt)含每条预测、生成token、合同、状态、环境、转换审计、原评分JSON和逐文件SHA；[归档](../provenance/styled_milebench_20261117/final/recorded_runs.tar.gz)为27,653,481 bytes，保留3279个原始文件及一个内置SHA索引，不含图片、权重或凭证。SHA256：`fa4ba895a0e648325d6de4bdaf9bff00e28fa83eb4e6aa3435e8a9cc7e6d695c`。

独立CPU核验3200正式+16预热的token重解码、合同/ID/EOS、五阶段E2E与模型E2E求和全部通过，求和最大误差0；12个styled组的2400条原始记录逐sample_id配对、1600对variant比较通过。原Asym另从28层物理长度重算容量与覆盖。证据分别为[decode/E2E](../provenance/styled_milebench_20261117/independent_decode_e2e_audit.json)、[分组配对](../provenance/styled_milebench_20261117/independent_pairing_audit.json)、[Asym缓存与EOS](../provenance/styled_milebench_20261117/independent_asymkv_cache_eos_audit.json)。

发布流程重新核验源码、kernel、标注与原记录，并重新评分全部16组。随后将真实归档解压到新临时目录，仅复制两份标注、禁止导入torch/transformers/kvrepack_kernel，以隔离Python执行归档内CPU评分脚本；16组分数与发布CSV逐项精确一致。见[真实可迁移重评分记录](../provenance/styled_milebench_20261117/real_portable_cpu_replay.json)。该验证不重新测量GPU时延。

GPU任务串行且无其他GPU负载；正式期间有少量CPU测试与只读完整性核验，并未做受控锁频/重复多轮时延实验。详见completion.json中的并行CPU窗口；E2E用于描述本次实际请求耗时。

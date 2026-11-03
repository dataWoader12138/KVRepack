# KVRepack 项目交接（2026-11-03）

本文是随项目保存、可由用户和开发者审阅的交接摘要：记录已接受的要求、实现选择、验证证据和待办事项，供后续会话恢复项目背景。后续修改应以当前代码、报告和用户的新指示为准。密钥、凭证、模型权重、视频和数据集保存在项目外。

当前版本完成了独立项目整理与 CPU 审计，仍有两项已确认但未修复的行为问题。本文不表示 GitHub 上传已经完成，也不表示实验结果已经复现；上传状态以实际远端和提交记录为准。

用户指定的目标为公开仓库 `https://github.com/dataWoader12138/KVRepack`。提交作者为 `dataWoader12138`；Git 的身份和 SSH 凭证在各机器本地配置。

## 目标与已接受的约束

用户希望把原工作区中用于主实验的代码整理成一个独立、可上传 GitHub、可在其他机器使用的版本。主实验原先以 `HybridKV-opensource` 运行，部分扩展、入口或导入与 `HybridKV-artifact` 有关联，因此必须追踪实际调用关系，不能仅凭同名文件选择来源。

- 同时保留 Qwen2.5-VL 与 LLaVA-OneVision 的已选范围。每个模型/方法组合只保留一种明确实现，避免同名包跨目录补齐或运行时切换来源。
- 保持基线和对应改进的算法语义、预算规则及默认参数，不把旧最优参数、当前脚本配置和消融配置混为一谈。发现原问题时先记录来源和影响，不能把修复藏在整理或改名中。
- 当前项目已统一 KVRepack 命名：改进方法为 `*_kvrepack`，对应文件、函数、配置属性和环境变量一起改名；环境变量中的 `DECOUPLE` 改为 `KVREPACK`。真实历史来源路径、参考脚本名及来源记录保持原样。
- 公开 `asymkv` 固定选择 official 实现，内部名仍为 `asymkv_official`。所需私有 helper 不等于另外开放 paper 或 Hybrid-AsymKV 方法。
- 原始实验目录保持不动。项目通过自己的 `kvrepack` 包、定制 `lmms_eval`、资源文件和 `kvrepack_kernel` 源码运行，不依赖原服务器绝对路径、软链接或原 `.so`。
- 用户已授权整理、CPU 检查、代码关系审计，以及将项目和本交接摘要上传到其指定 GitHub 仓库。GitHub 推送采用 SSH，目标仓库、仓库可见性和提交作者须由用户提供；专用私钥留在服务器，公钥由用户添加到目标仓库并授予写入权限。其他机器需自行配置访问凭证。GPU 实验阶段尚未授权，也未在本机完成 GPU 模型或内核测试，后续需要用户明确开启该阶段后再运行。

当前入口假设单进程、batch size=1、greedy generation、`use_cache=True`、多 token 视觉 prompt。每个模型/方法用独立进程，避免 monkeypatch 残留。跨方法对照需显式统一模型、任务、样本、帧数和预算。SnapKV 改进所选旧入口的默认 merge ratio 为 0，不能因为名字是“改进”就默认其已启用合并。

## 来源关系与当前范围

原工作区包含以下树；这些名称是历史定位信息，独立仓库运行时不需要这些目录。

| 原来源 | 与项目的关系 |
|---|---|
| `HybridKV-artifact/` | 多数当前核心与改进实现的来源；`scripts/eval/` 中选定入口的原文本已保存在 provenance。 |
| `HybridKV-opensource/` | 用户主实验的运行底座之一；Qwen FullKV 保留其原加载政策；提供 LLaVA 原有 dispatcher。实际导入组合必须与 artifact 区分。 |
| `HybridKV-llava-ov-decouple-port-20260708/` | 后续发现的独立 LLaVA 改进 port；以 opensource 为底座，并复制 artifact 的部分 Qwen Decouple 依赖。其旁的 `HybridKV-llava-ov-decouple-port-20260708_PORT_SOURCES.txt` 记录了来源。尚未迁入本项目。 |

当前共 **19 个模型/方法组合**：

| 模型 | 已纳入的方法 |
|---|---|
| `qwen2_5_vl`，3B/7B | 13 种：`fullkv`、`hybridkv`、`snapkv`、`sparsemm`、`madakv`、`lookm`、`randomkv`、`asymkv`，以及前五种压缩基线对应的 `hybridkv_kvrepack`、`snapkv_kvrepack`、`sparsemm_kvrepack`、`madakv_kvrepack`、`lookm_kvrepack`。 |
| `llava_ov`，OV-Qwen2-7B | 6 种基线：`fullkv`、`hybridkv`、`snapkv`、`sparsemm`、`madakv`、`lookm`。 |

LLaVA 的 SnapKV、SparseMM、MadaKV、HybridKV 四种改进源码已经找到，尚未迁入或纳入上述检查范围；对应组合会在规划阶段被拒绝。port 中的关键文件为 `qwen2_model_{snapkv,sparsemm,madakv}_repackkv.py`、`modeling_llava_onevision_repack.py`、`modeling_qwen2_repack.py` 和 `hybridkv_repack_utils_llava.py`。未找到对应的 LLaVA LookM 改进。

port 和当前 Qwen 改进共享候选选择、相邻合并、raw K 重施 RoPE 等思路，但同时存在预算边界、score 数值路径、SparseMM 缓存生命周期及额外加权合并选项的差异，不能当作纯架构改名直接拼接。LLaVA 使用普通 Qwen2 RoPE，Qwen2.5-VL 使用 mRoPE；后续迁入需单独记录这些适配和语义差异。

当前任务范围为 NextQA、VATEX、VideoChatGPT。多图 MileBench、历史参数搜索、效率诊断、旧 adjacent/base/blacklist、独立 headsplit 和 AsymKV paper 路线不作为首版可运行方法。

## 最新验证结论

以下数字来自 2026-11-03 的 [汇总报告](../provenance/cpu_audit_summary.json) 和 [CPU 审计说明](CPU_CALL_AUDIT.md)：

| 检查 | 实际结果与边界 |
|---|---|
| 动态分派 | 19 个组合各自独立进程，49 次真实 patch 赋值；9 个 prepare wrapper 各检查 4 种缓存状态。使用模型/压缩执行替身，未运行完整 forward。 |
| 静态调用关系 | 33 个 Python 文件、498 个函数定义、393 个已解析本地调用位置；392 处可核对参数，未确认参数绑定错误。另有 5783 处外部或动态调用不在证明范围内。 |
| 缓存与接口审计 | 按 Transformers 4.52.3 实际源码核对，16 条 decoder→attention 显式调用签名及返回解包匹配。 |
| 原函数 CPU 小张量 | 15 项完成，0 failure、0 error、无 skip；包含原 artifact 同输入差分，并成功复现下列两项问题。CUDA 入口调用为 0。 |
| 完整 unittest suite | 49 项，48 通过、1 跳过；跳过项要求显式提供原工作区的差分检查，已在上述带 `--source-root` 的独立 CPU 检查中通过。 |

“检查通过”表示分派、结构和相应断言符合记录，不能解释为所有方法行为正确。完整模型加载、真实 FlashAttention/CUDA、各层完整 KV 布局、GPU 数值一致性、任务分数、显存、时延和吞吐仍未验证。

## 已知未修问题

| 编号 | 影响与归属 | 下一步验证重点 |
|---|---|---|
| `LLAVA-SPARSE-STATE` | `llava_ov/sparsemm` 同一模型进程处理不同长度 prompt 时复用首个容量表。CPU 8→16 token 已复现；新建对象的第二次容量不同。artifact attention 与 helper 配对时已有该问题；相对每次 prefill 重建的旧 opensource 执行路径，这也是实现组合回归。 | 修复方案需只重建或重算当前 prompt 的预算，避免在 decode 重置元数据；保留两 prompt 与 fresh-object 对照。 |
| `ASYM-PREFILL-BOUNDARY` | 默认 chunk=512，prompt 长度513/1025分别只转发512/1024 token；末 token 未进入 prefill。继承自所选 official 集成中的 `range(0, prompt_length - 1, chunk_size)`，不是改名引入。 | 明确修复与保持旧行为的取舍；覆盖512、513、514、1024、1025及最短输入，检查 token 覆盖与逻辑位置。 |

另有两个低风险直接 API 边界，仍未修复：`HYBRID-TEXT-INPUTIDS-ONLY` 在直接调用自定义 Hybrid 文本模型、仅传 `input_ids` 时先读取空 `inputs_embeds.shape`；`MASK-INPUTEMBEDS-ONLY` 在部分 mask 外层模型仅传 `inputs_embeds` 时读取空 `input_ids.device`。当前视频 adapter 传 token 并由外层构造 embeddings，绕开这两项；不能把它们表述为默认视频入口必然失败。

精确触发条件、影响组合、源码行号和来源解释见 [cache_interface_audit.json](../provenance/cache_interface_audit.json)，CPU 复现输入/输出见 [cpu_compression_audit.json](../provenance/cpu_compression_audit.json)。审计阶段没有修改核心来修复这些问题。

## 从哪里继续读和运行

| 文件 | 用途 |
|---|---|
| [README](../README.md)、[METHODS](METHODS.md) | 安装说明、支持范围、当前方法名、来源入口和默认配置。 |
| [REPACKING](REPACKING.md)、[VALIDATION](VALIDATION.md)、[CPU_CALL_AUDIT](CPU_CALL_AUDIT.md) | 文件收敛与命名映射、阶段边界、调用关系和检查细节。 |
| `src/kvrepack/experiments/registry.py`、`runner.py`、`run.py` | 方法唯一映射、参数白名单、环境隔离、计划与统一 CLI。 |
| `src/kvrepack/monkeypatch.py`、`vendor/lmms-eval/lmms_eval/models/` | 实际方法分派、prepare wrapper 和模型 adapter。 |
| `provenance/core_sources.json`、`vendor_sources.json`、`reference_entrypoints.json` | 原始来源、机械变换、SHA/AST 证据和参考脚本原文。 |
| `provenance/method_registry.json`、`environment_observed.json` | 当前配置快照和原环境的包版本记录。 |
| `provenance/cpu_audit_summary.json`、`validation_report.json` | 最新汇总、打包与来源核验状态；优先于本文的固定统计。 |
| `provenance/dispatch_audit.json`、`call_graph.json` | 19 种实际分派结果和可静态解析的后续调用图。 |
| `scripts/check_{sources,dispatch,call_graph,cpu_compression}.py`、`tests/` | 可复查的检查入口；命令见下方。 |

原观察环境名为 `hybridkv`，使用 Python 3.12、PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124`、Transformers `4.52.3`、flash-attn `2.7.4.post1`、NumPy `1.26.4`、Accelerate `1.13.0`、datasets `2.19.2`、decord `0.6.0`。完整约束见 [constraints-observed.txt](../requirements/constraints-observed.txt)。这是已观察环境快照，尚非全新机器验证过的安装锁；PyTorch 的 CUDA wheel 来源和编译 toolkit 需在目标机器另行固定。环境名不要求跨机器一致。

在项目根目录、已具备相关依赖的环境中复查，无需 GPU、权重或数据：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
python -m kvrepack.run --list-methods
python -m kvrepack.run --method snapkv_kvrepack --limit 2 --dry-run
python scripts/check_sources.py
python scripts/check_dispatch.py > /tmp/kvrepack-dispatch.json
python scripts/check_call_graph.py --dispatch-report /tmp/kvrepack-dispatch.json --output /tmp/kvrepack-call-graph.json
python scripts/check_cpu_compression.py > /tmp/kvrepack-cpu-compression.json
python -m unittest discover -s tests -v
```

保留原工作区时，可给来源检查和 CPU 差分检查显式加 `--source-root <原工作区>`。其他机器不需要原目录即可执行其余检查。缺依赖导致测试跳过时，应记录缺失项，不能沿用本机“15 项完成”的结论。模型与数据通过使用者准备的路径或 `HF_HOME` 配置；NextQA 的 NLTK 资源、VATEX 的 Java/METEOR、VideoChatGPT 外部 judge 需分别准备，评分不等于生成。

## 后续优先级

1. **先处理两项已确认问题。** 明确修复与旧源码等价性的边界，分别加入跨 prompt 和分块边界回归，再修改最小范围实现。更新来源变换、哈希、测试和报告；不要仅更新 manifest 来隐藏行为变化。
2. **在目标机器做安装与 CPU 复查。** 按固定依赖检查包资源、vendor、来源文件和 CLI；重新记录环境与实际通过/跳过项。GitHub 打包只纳入项目及交接/审计证据，SSH 与其他凭证留在项目外。
3. **若扩展 LLaVA 四种改进，单独迁入。** 比较 port 与当前 Qwen 的 RoPE、预算、score 和生命周期；每种方法固定唯一实现，与其基线按同参数对照。当前命名统一没有完成这项迁移。
4. **用户开启 GPU 阶段后再验收。** 先在目标环境构建 `kvrepack_kernel`，逐组合运行1–2个样本，与选中旧实现比较生成 token、KV 长度和预算，再进行 limit=200 的任务检查；质量与数值通过后才测性能。
5. **发布前整理可复现基准与许可。** 冻结模型/数据标识、样本、参数、代码哈希和结果。项目级许可证与继承代码归属仍待仓库所有者确认，见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)；当前不能称已全部复现或已具备完整发布验证。

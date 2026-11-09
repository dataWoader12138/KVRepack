# KVRepack

LookM 已改为[严格10%总prefill预算](docs/LOOKM_STRICT_BUDGET.md)：32个近期位置计入总额度，并修正逐head合并补集。旧400条结果使用“10%＋32”口径，只保留为[历史记录](docs/LOOKM_RESULTS_20261109.md)，不能用于严格10%比较；新正式200条分数尚未重跑。Hybrid与RoPE的已停止任务继续保持停止。

消融数据：[下载统一CSV汇总表](docs/ablation_results_summary.csv)。包含SnapKV/HybridKV的Random-head结果、已完成的RoPE聚合空间对照和20条试跑，共13个已完成指标行；另列HybridKV/VATEX仅97配对的停止状态，未评分项留空。差值统一为“基线−对照”，Random行对照为三seed均值；保留样本数、参数、标准差、区间和来源，UTF-8 BOM可用Excel直接打开。

已完成：[HybridKV + KVRepack 的 Random-head 结果与图表](docs/HYBRID_RANDOM_HEAD_NEXTQA_200_RESULTS_20261106.md) · [协议与复现命令](docs/HYBRID_RANDOM_HEAD_NEXTQA_200.md)。2026-11-06完成Qwen2.5-VL-7B、NExTQA前200条的800次生成，11-07独立全量执行与评分复核均通过。相似度/Random均值WUPS为33.219903/33.088063，差值95%区间为[−0.778808,1.353447]，未显示稳定优势。四条件均选56头、保留原逐头预算，但异质预算使实际移除token量不同，不能称严格等合并工作量。此前[RoPE合并尝试](docs/ROPE_MERGE_ABLATION.md)仍按用户要求终止，不自动续跑。

已完成：[NExTQA前200条 Random-head 消融结果](docs/RANDOM_HEAD_NEXTQA_200_RESULTS_20261104.md) · [协议与复现命令](docs/RANDOM_HEAD_NEXTQA_200.md)。2026-11-04完成Qwen2.5-VL-7B + SnapKV-KVRepack四条件800次生成，11-06复核；原策略/Random均值WUPS为32.917681/32.811042，差值区间跨0，未显示稳定优势。仅前200条，本轮未运行VATEX。

项目交接摘要与后续工作入口：[docs/HANDOFF.md](docs/HANDOFF.md)。

Random-head 消融：[固定选头数量的实验协议与运行命令](docs/RANDOM_HEAD_ABLATION.md)；[首轮160次生成结果](docs/RANDOM_HEAD_ABLATION_RESULTS_20261104.md)。已完成7B SnapKV-KVRepack、两数据集各20条和3个随机种子的160次生成；当前小样本没有显示相似度选头稳定优于Random。

视频实验参数：[按数据集核对的最佳参数表](docs/VIDEO_BEST_PARAMETERS.md) · [CSV](docs/video_best_parameters.csv)。表格区分实际完成记录、单一候选、历史变体及不同复评协议，不修改运行入口的默认值。

扩大 GPU 复测：[200 条配对与完整测试集的运行状态、结果及恢复方式](docs/FULL_VIDEO_RETEST_20261104.md)。该页按阶段更新，明确区分运行中与已通过。

论文观察实验 A：[运行协议](docs/OBSERVATION_A.md) · [实际400条结果与主图](docs/OBSERVATION_A_RESULTS_20261104.md)。使用7B模型、两数据集各原始前200条；独立观察入口和统计绘图位于 `src/kvrepack/observations/`。

MixKV式补充观察：当前为[Qwen2.5两模型各5个完整输入的pilot](docs/MIXKV_REFERENCE_PILOT.md)，比较LongBench长文本与OCRBench图文，统计全部输入token、不抽样token、不自动扩展。[旧等token抽样方案](docs/MIXKV_STYLE_OBSERVATION.md)已按用户要求停止在365/400条，不自动续跑。

该5＋5已完成：[结果和类似MixKV图1(b)的分布图](docs/MIXKV_REFERENCE_PILOT_RESULTS_20261104.md)。RoPE后K/V组均值为图文更高，raw K方向相反；仅为小批量诊断，未扩大实验。

后续已按用户明确授权完成[各200条的完整输入观察](docs/MIXKV_REFERENCE_200_RESULTS_20261104.md)，含主图、raw对照与独立来源聚类区间；原生统计模块不变，400条全部完成，旧5＋5保留原样。

相似度解释：[MixKV对照口径、原结果的层差异及候选机制调研](docs/VISUAL_TEXT_SIMILARITY_RESEARCH.md)。明确区分本机证据、相关论文的结论与尚待验证的原因。

将 Qwen2.5-VL 的 KV cache 基线及其 KVRepack 改进整理到一个独立项目，同时纳入 LLaVA-OneVision-Qwen2-7B 的原有基线。任务包括 NextQA、VATEX、VideoChatGPT。多图 MileBench、历史参数搜索和诊断支线尚未纳入首版。

本版本的代码参考由原工作区 `HybridKV-artifact/scripts/eval` 的实际调用链决定；FullKV 遵循该目录中显式选择 opensource 的对照脚本。项目运行不需要原来的两个目录，也不需要原机器软链接。当前已完成整理、CPU 审计及下述 GPU 子集数值对照；**完整 CLI、完整基准与性能仍待验证**。

2026-11-03 的无 GPU 调用关系审计覆盖了19个组合，并执行原压缩函数的CPU小张量检查；已复现 **LLaVA SparseMM 跨视频复用旧预算**及 **AsymKV 特定分块边界遗漏末 token** 两项问题，核心代码尚未修复。详细调用图、触发条件和复查命令见 [CPU 调用关系审计](docs/CPU_CALL_AUDIT.md)。分派与源码检查通过不能替代这些行为问题的处理。

2026-11-04 已获授权并完成 Qwen3B 的 SnapKV-KVRepack、MadaKV-KVRepack GPU 迁移对照：NextQA、VATEX 各2例、64帧，共8组原实现/新实现配对，生成 token 完全一致，36层 prefill/final KV 的 SHA-256 一致，全部词表的生成 logits 最大绝对差为0。测试使用绕过 runner/adapter 的隔离 worker，未覆盖完整 CLI、完整任务评分、自定义 CUDA 内核、7B、LLaVA、HybridKV、SparseMM 或性能；MadaKV 两个数据集均用 `(simtop, cr, alpha)=(0.5, 0.7, 2)`，NextQA 此处是代表配置，并非其 `cr=0.5` 的历史最佳配置。详情及入口 schema 1/2 兼容修复见 [GPU 子集复测](docs/GPU_RETEST_20261104.md)。

## 文件组织

```text
KVRepack/
├── src/kvrepack/          # 唯一的核心包、方法注册表和实验入口
│   ├── experiments/      # 公开方法 → 内部方法 → 固定实现及环境参数
│   ├── observations/     # 无压缩前向的近邻/等量全配对观察、上下文对照与绘图
│   ├── ablations/        # 固定选头数量的 Random-head 干预、执行审计与离线评分
│   ├── configs/          # 保留原始字节的配置
│   ├── data/head_score/  # 运行必需的 head score，随包分发
│   └── asymkv_official/  # 唯一对外提供的 AsymKV 实现
├── vendor/lmms-eval/     # 本项目的定制评测框架与任务资源
├── csrc/                 # kvrepack_kernel 源码和固定 CUTLASS 头文件快照
├── provenance/           # 原入口、原文件与新文件的 SHA-256、环境版本
├── requirements/         # 原环境的版本约束，不包含凭证或本地安装地址
├── scripts/              # 只读的来源/静态验证工具
├── tests/                # CPU 检查
└── docs/                 # 方法选择、迁移范围、GPU 验证清单
```

## 方法

每个模型与方法组合只映射到一种实现。五种改进方法统一使用 `*_kvrepack`，公开名字与内部 `METHOD` 一致；对应源码文件、函数、配置属性和环境变量也统一使用 KVRepack 名称。命名调整保持算法语义、参数值和默认配置不变，原始来源路径与参考脚本名保留用于追溯，见 [命名对照](docs/REPACKING.md#命名对照)。

| 基线 | 对应 KVRepack 改进 |
|---|---|
| `hybridkv` | `hybridkv_kvrepack` |
| `snapkv` | `snapkv_kvrepack` |
| `sparsemm` | `sparsemm_kvrepack` |
| `madakv` | `madakv_kvrepack` |
| `lookm` | `lookm_kvrepack` |

另提供 `fullkv`、`randomkv` 和 `asymkv`（固定 official 实现）。完整来源及配置见 [方法说明](docs/METHODS.md) 和 `src/kvrepack/experiments/registry.py`。没有把 historical best、当前默认值和消融配置混成一个版本；例如当前 SnapKV 改进入口默认 merge ratio 为 0，项目如实记录，使用非零比例需要显式配置。

| 模型入口 | 本版支持 |
|---|---|
| `--model qwen2_5_vl`（默认） | 上述13种方法，3B/7B |
| `--model llava_ov` | FullKV、HybridKV、SnapKV、SparseMM、MadaKV、LookM，OV-7B |

LLaVA 的六种基线已纳入本版。工作区另已找到 SnapKV、SparseMM、MadaKV、HybridKV 的 LLaVA 改进源码，尚未迁入；这些改进组合在生成计划时会明确拒绝。详见 [迁移范围](docs/REPACKING.md#首版范围)。

## 安装（全新环境安装流程尚未验证）

使用独立的 Python 3.12 环境。以下安装流程尚未在全新机器验证；本轮 GPU 子集复测复用了原有依赖环境。

```bash
cd KVRepack
python -m pip install -c requirements/constraints-observed.txt torch==2.6.0 torchvision==0.21.0 setuptools wheel packaging ninja
python -m pip install -c requirements/constraints-observed.txt --no-build-isolation flash-attn==2.7.4.post1
python -m pip install -c requirements/constraints-observed.txt -e .
python -m pip install -c requirements/constraints-observed.txt -e './vendor/lmms-eval[qwen]' pywsd==1.2.5 rouge==1.0.1
python -m pip install --no-build-isolation ./csrc
```

原实验环境使用 PyTorch `2.6.0+cu124`；上面的普通 PyPI 安装命令不保证得到相同 CUDA wheel，严格对照时需另外固定 PyTorch wheel 来源。CUDA 编译需要匹配的 toolkit、编译器与 GPU 架构；没有 GPU 的编译环境可显式设置 `TORCH_CUDA_ARCH_LIST`。仓库不分发原机器 `.so`，不探测 GPU 后自动修改算法参数。

## 不调用 GPU 的检查

```bash
PYTHONPATH=src python -m kvrepack.run --list-methods
PYTHONPATH=src python -m kvrepack.run --method hybridkv_kvrepack --task nextqa_oe_test --limit 2 --dry-run
PYTHONPATH=src python -m kvrepack.run --model llava_ov --method hybridkv --task nextqa_oe_test --limit 2 --dry-run
python scripts/check_sources.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

`--dry-run` 只展示解析后的计划，不下载数据、不创建实验输出、不启动模型。正式运行使用同一入口去掉 `--dry-run`，例如：

```bash
python -m kvrepack.run --method hybridkv_kvrepack --task nextqa_oe_test \
  --pretrained Qwen/Qwen2.5-VL-3B-Instruct --frames 64 --limit 200 \
  --budget-ratio 0.10 --device 0 --output outputs/nextqa/hybridkv_kvrepack
```

入口固定加载本仓库的 `kvrepack` 和定制 `lmms_eval`，隔离旧的算法环境变量，运行前核对来源与文件哈希，并保存解析配置和来源信息。实验必须逐方法使用独立进程，避免 monkeypatch 残留。

## 模型、数据与评分

权重、视频、数据集和历史日志不包含在仓库中。可显式设置 `HF_HOME` 复用已有缓存；NextQA 的 NLTK 资源需预先准备，VATEX 的 METEOR 评分需要 Java。具体路径及阶段安排见 [验证说明](docs/VALIDATION.md)。VideoChatGPT 的生成和外部 judge 评分是不同环节；评分模型和配置需要单独记录。本次没有发送评分请求。

本项目保留原始算法实现并记录机械迁移证据；现有小样本数值对照不能替代未覆盖方法的 GPU 检查、完整任务评分与性能测试。历史 CSV 的最优结果也不能仅靠当前脚本默认值保证复现。

第三方许可与项目自身许可的待确认项见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

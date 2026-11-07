# KVRepack

项目交接摘要与后续工作入口：[docs/HANDOFF.md](docs/HANDOFF.md)。

当前 [Spot-the-Diff / MMCoQA Random-head 图片实验](docs/MILEBENCH_RANDOM_HEAD.md) 已进入用户授权的 Qwen2.5-VL-7B GPU 阶段：HybridKV-KVRepack 和 SnapKV-KVRepack 均运行 similarity 与 random seed0/1，读取两个官方 MileBench v1 子集全部标注，各200条，计划 `4×200×3=2400` 次生成。MileBench 作者已从源任务抽样构成这些子集，见[论文 §3.2](https://arxiv.org/pdf/2404.18532)；本项目的 `--all-samples` 不设条目上限。

将 Qwen2.5-VL 与 LLaVA-OneVision 的 KV cache 基线及 KVRepack 改进整理到一个独立项目。视频入口包括 NextQA、VATEX、VideoChatGPT；新增 [MileBench 多图主实验](docs/MILEBENCH.md)，包含数据预检、图片生成、原协议评分和 JSON/CSV 汇总。图片配置与视频默认值分别管理，共享核心实现。

本版本的代码参考由原工作区 `HybridKV-artifact/scripts/eval` 的实际调用链决定；FullKV 遵循该目录中显式选择 opensource 的对照脚本。项目运行不需要原来的两个目录，也不需要原机器软链接。首版完成整理和 CPU 静态验证；目前 GPU 实测仅覆盖上述 Qwen7B Random-head 运行范围，其他模型、方法及性能仍需分别验证。

首版无 GPU 审计覆盖19个组合，并复现 LLaVA SparseMM 跨 prompt 预算复用及 AsymKV 分块边界末 token 遗漏。本次已单独修复前者并新增 CPU 回归；AsymKV 问题仍未修复且不在图片入口范围。历史证据见 [CPU 调用关系审计](docs/CPU_CALL_AUDIT.md)，图片扩展与本次实测范围见 [MileBench 验证报告](provenance/milebench_validation.json)。Qwen7B 两方法/两任务各1条的独立 smoke 已通过；正式 Hybrid/Spot 已完成200条、600次生成并重新核分通过，其余三组仍在运行，尚无完整四组结果，也未完成全项目 GPU 数值一致性或性能验证。

正式进度以 `outputs/ablations/random_head_qwen7b_all_seed01_v2_20261108/controller_state.json` 为准。首个控制器在 Hybrid/MMCoQA 第二个生成样本（原 `sample_id=18`）因后置审计错误地要求56个选中头全部有效合并而停止；现已恢复参考 Hybrid 允许自然缺调用或 noop、但有效合并数必须大于0的语义，SnapKV 仍要求34头有效匹配，预算、算法和随机 mask 未改。已完成 Hybrid/Spot 的原合同、样本及报告按字节复制复用，`reuse_manifest.json` 记录 SHA，旧12份绑定源码冻结于 `source_archives/first_controller`；未重复其600次生成。原失败目录保留。

## 文件组织

```text
KVRepack/
├── src/kvrepack/          # 唯一的核心包、方法注册表和实验入口
│   ├── experiments/      # 公开方法 → 内部方法 → 固定实现及环境参数
│   ├── configs/          # 保留原始字节的配置
│   ├── data/head_score/  # 运行必需的 head score，随包分发
│   └── asymkv_official/  # 唯一对外提供的 AsymKV 实现
├── vendor/lmms-eval/     # 本项目的定制评测框架与任务资源
├── csrc/                 # kvrepack_kernel 源码和固定 CUTLASS 头文件快照
├── provenance/           # 原入口、原文件与新文件的 SHA-256、环境版本
├── requirements/         # 原环境的版本约束，不包含凭证或本地安装地址
├── scripts/              # 实验规划、运行、评分及来源审计工具
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

上表为默认视频入口范围。使用 `--benchmark milebench` 时，Qwen和LLaVA各提供六种基线及 HybridKV/SnapKV/SparseMM/MadaKV 四种 KVRepack 改进；新增四个 LLaVA 改进只从图片能力表选择。图片所有 Qwen 方法统一默认3B，7B通过 `--pretrained` 显式选择。详见 [MileBench](docs/MILEBENCH.md) 与 [LLaVA 迁移](docs/LLAVA_MIGRATION.md)。

图片数据预检、无副作用计划和离线评分均可在没有 GPU 时运行：

```bash
PYTHONPATH=src python -m kvrepack.run --benchmark milebench --list-methods
PYTHONPATH=src python -m kvrepack.run --benchmark milebench --method hybridkv_kvrepack \
  --profile main11 --task CLEVR-Change --data-root /path/to/milebench --limit 2 --dry-run
PYTHONPATH=src python -m kvrepack.run --benchmark milebench \
  --task CLEVR-Change --data-root /path/to/milebench --preflight
```

图片附加依赖安装为 `python -m pip install -e '.[images]'`；依赖约束与完整评分/汇总命令见 [使用说明](docs/MILEBENCH.md)。当前目标核心和部分 opensource 历史图片实现有差异，LLaVA共享预算配置也不同，详见 [来源差异](docs/MILEBENCH_SOURCE_DIFF.md)，不能仅凭入口迁移称已复现历史分数。

## 安装与环境约束

使用独立的 Python 3.12 环境。以下为依赖安装参考；本机已从当前 checkout 构建 `kvrepack_kernel` 并开始上述限定范围的 GPU 实测，不代表已验证全新机器安装或所有模型/方法。

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

本项目保留原始算法实现并记录机械迁移证据；这些检查不能替代 GPU 的数值、生成结果与性能对照。历史 CSV 的最优结果也不能仅靠当前脚本默认值保证复现。

第三方许可与项目自身许可的待确认项见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

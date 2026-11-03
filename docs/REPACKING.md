# 文件整理与实现选择

本项目在原工作区旁新建，保留两个原目录。执行来源以 `HybridKV-artifact/scripts/eval` 中的脚本为依据；这些原入口的完整文本作为 JSON 数据保存在 `provenance/reference_entrypoints.json`，不会作为可运行的旧路径入口保留。

## 收敛后的文件

| 文件/目录 | 作用 | 来源与处理 |
|---|---|---|
| `src/kvrepack/qwen_model.py`、`baseline_utils.py`、`qwen_model_croatt.py` | SnapKV/SparseMM/MadaKV/LookM 公共基线 | artifact 主脚本调用的实现；只改包、内核导入和资源位置 |
| `modeling_qwen2_5_vl.py`、`hybridkv_utils.py`、`hybridkv_cpu.py` | HybridKV 与 Hybrid-KVRepack | 使用同一模型/缓存类，由内部 METHOD 控制；不复制旧同名实验实现 |
| `modeling_llava_onevision.py`、`modeling_qwen2.py`、`qwen2_model.py` | LLaVA-OneVision 六种基线 | 原脚本使用的架构实现；四种补丁分支从 opensource 的 OneVision dispatcher 迁入，调用唯一的本地核心 |
| `qwen_model_{snapkv,sparsemm,madakv,lookm}_kvrepack.py` | 四种基线的 KVRepack 改进 | 对应原 simhead/LookM 主入口；保留对公共基线及必要辅助函数的依赖 |
| `lookm_headsplit_kvrepack_utils.py` | LookM 改进必需的辅助函数 | 统一 KVRepack 命名，不另外开放 headsplit 方法 |
| `merge_score.py` | 各方法共用的合并评分 | 保留单份原实现 |
| `randomkv_utils.py`、`qwen_model_randomkv.py` | RandomKV 对照 | 采用原 RandomKV 主入口的实现 |
| `asymkv_official/` | 唯一公开 AsymKV 路线 | 选择 official 实现；paper 路线只抽取所需 5 个函数至 `_asymkv_helpers.py` |
| `hybrid_asymkv.py` | Hybrid 缓存延迟引用的 reducer | 只保留一个原函数，不开放另一套 Hybrid-AsymKV 算法 |
| `monkeypatch.py` | 选中方法的分派与 prepare wrapper | 保留 10 个选中 Qwen 分支及 4 个 LLaVA 补丁分支；按声明的命名映射与原分支比较 AST |
| `configs/param_config.json`、`data/head_score/*.json` | 必需配置和 head score | 字节保持原样，随 Python 包分发 |
| `experiments/registry.py`、`runner.py`、`run.py` | 新的统一实验入口 | 显式方法映射、参数白名单、环境隔离、来源验证和运行记录 |
| `vendor/lmms-eval/` | 任务、指标和 Qwen/LLaVA adapter | 保留任务实现；统一分派，单独保留 Qwen FullKV 原加载政策 |
| `csrc/` | 唯一 CUDA 构建 | 源码/头文件保留原样，模块统一命名 `kvrepack_kernel` |

## 防止整理引入偏移

1. 新增常规 `kvrepack/__init__.py`，取消旧命名空间包跨目录补齐模块的行为。
2. 对源码文件、函数、配置属性、METHOD 与环境变量作声明过的名称映射；`HybridKVCache`、数学表达式、张量操作和参数值保留。
3. 必需资源相对包本身定位，取消原机器软链接和工作目录依赖。
4. 原核心文件的来源/目标 SHA-256 全部留档。迁移中的包名、内核名、资源路径和 KVRepack 名称映射单独声明。
5. 抽取的 6 个函数、Qwen 的 10 个分派分支、LLaVA 的 4 个补丁分支，以及 prepare wrapper 有函数/分支级 AST 指纹。
6. `scripts/check_sources.py --source-root <原工作区>` 能比较原源码与迁移后源码；不提供原目录时仍能检查发布文件哈希、抽取函数指纹、本地导入闭包、vendor 和构建输入。
7. 改进方法绑定明确的基线，运行环境中遗留的 METHOD 和算法参数不参与默认配置。覆盖只能通过 `--set` 的每方法白名单显式指定。

这些措施证明迁移限定在声明范围内。算法原有的数值行为、跨方法公平性、KV预算实际占用和 GPU 执行仍需下一阶段验证；不能由 AST 或哈希替代。

## 命名对照

当前可运行代码统一使用 KVRepack 名称；此次命名调整不改变算法、参数语义、参数值或默认配置，也不增加支持方法。

| 原代码标识 | 本项目标识 |
|---|---|
| `qwen_model_{snapkv,sparsemm,madakv,lookm}_decouplekv.py` | `qwen_model_{snapkv,sparsemm,madakv,lookm}_kvrepack.py` |
| `lookm_headsplit_decouplekv_utils.py` | `lookm_headsplit_kvrepack_utils.py` |
| `hybrid_decouplekv` METHOD | `hybridkv_kvrepack` |
| `{snapkv,sparsemm,madakv,lookm}_decouplekv` METHOD | `{snapkv,sparsemm,madakv,lookm}_kvrepack` |
| `init_snapkv_decouplekv` 等函数名 | `init_snapkv_kvrepack` 等对应的 KVRepack 名称 |
| 配置属性中的 `decouple` | `kvrepack`，例如 `snapkv_kvrepack_compress_ratio` |
| 环境变量名中的 `DECOUPLE` | `KVREPACK`，例如 `HYBRID_KVREPACK_COMPRESS_RATIO` |

原始来源路径、参考脚本文件名和来源记录中的原标识保持原样，以便核对真实历史文件；它们不是本项目的运行接口。使用 `--list-methods` 查看当前配置名与覆盖白名单。

## 首版范围

首版包含 Qwen2.5-VL 3B/7B 的8个基线/对照及5个 KVRepack 改进，以及 LLaVA-OneVision-7B 的6个原有基线。原工作区中名为 DecoupleKV 的 adjacent/base/blacklist 支线、headsplit 独立实现、AsymKV paper 独立实现、参数搜索、效率诊断和多图 MileBench 不作为本版可运行入口。保留的必要 helper 不形成第二个公开算法。

LLaVA 基线来源为 `llava_ov_ratio.sh` 与 `llava_ov_hybridkv.sh`。前者当前循环数组启用 FullKV/LookM/SnapKV；同源 dispatcher 还保留 SparseMM/MadaKV 可选分支。旧 artifact 缺失 OneVision dispatcher，因此按原 opensource 的分支体补齐，分支仍调用本地唯一的 artifact 架构实现，变换与来源单独留档。

工作区另已找到 `HybridKV-llava-ov-decouple-port-20260708/hybridkv/` 下的原有 LLaVA 改进源码：`qwen2_model_{snapkv,sparsemm,madakv}_repackkv.py`，以及 HybridKV 的 `modeling_llava_onevision_repack.py`、`modeling_qwen2_repack.py`、`hybridkv_repack_utils_llava.py`。这些文件尚未迁入本版，不能作为当前已支持的方法；未找到对应的 LLaVA LookM 改进。这里保留发现时的真实来源文件名。

FullKV 是唯一通过原脚本显式选择 opensource 模型加载政策的路径：auto dtype/device_map。压缩方法保留 artifact 的 BF16/逻辑GPU0 政策；单卡由 `CUDA_VISIBLE_DEVICES` 映射。对照时需显式使用同一模型、任务、帧数、样本数、生成参数和数据顺序。

旧主结果 CSV 仅作历史索引，不设为本版验收基准。部分结果使用旧 log_count 逻辑，当前选中源码的 no-correction 路线需要先通过新旧代码对照，再建立新的结果基准。

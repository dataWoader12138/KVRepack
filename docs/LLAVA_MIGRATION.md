# LLaVA 图片实验迁移与验证边界

本次加入四种图片用 KVRepack 改进：HybridKV、SnapKV、SparseMM、MadaKV。原视频入口的19个模型/方法组合保持不变；图片入口由 `experiments/llava_image_methods.py` 提供四种扩展配置，再由 MileBench profile 选择。每种组合仍只有一个核心实现。没有加入 threshold、额外加权合并支线或 LLaVA LookM 改进。

## 核心文件与唯一加载路径

| 当前 opensource 来源 | 目标 |
|---|---|
| `qwen2_model_snapkv_repackkv.py` | `src/kvrepack/qwen2_model_snapkv_kvrepack.py` |
| `qwen2_model_sparsemm_repackkv.py` | `src/kvrepack/qwen2_model_sparsemm_kvrepack.py` |
| `qwen2_model_madakv_repackkv.py` | `src/kvrepack/qwen2_model_madakv_kvrepack.py` |
| `modeling_qwen2_repack.py` | `src/kvrepack/modeling_qwen2_kvrepack.py` |
| `hybridkv_repack_utils_llava.py` | `src/kvrepack/hybridkv_utils_llava.py` |

三个非 Hybrid 改进使用 `monkeypatch.replace_llava_onevision()` 后加载 Transformers 的 `LlavaOnevisionForConditionalGeneration`。分派分别设置 Qwen2 attention；SnapKV另设置专用缓存 forward；SparseMM复用现有 split-cache text/decoder forward；MadaKV另传递视觉mask和预算。

Hybrid基线与改进共享现有 `modeling_llava_onevision.py` 外壳，避免复制相同的多模态处理。只有 `METHOD=hybridkv_kvrepack` 在构造语言模型时选择新 `modeling_qwen2_kvrepack.Qwen2Model`；其它方法保留原构造路径。新语言模型使用普通 Qwen2 RoPE 对应的 LLaVA helper；不能替换为 Qwen2.5-VL 的 mRoPE helper。

移植规则统一包名、CUDA模块名、KVRepack品牌及历史别名集合，不更改张量算术。原 LLaVA SnapKV只读取 `K_DIFF_ALPHA`；为使统一CLI的每方法前缀参数有效，新增 `SNAPKV_KVREPACK_K_DIFF_ALPHA` 优先读取，未设置时仍回退原变量。该接口适配单独留档，并测试显式前缀值与回退值。

资源输入有一项明确适配：当前opensource的 `param_config.json` 为 `llava_ov.budget_fixed_ratio=0.5`，目标仓库已有共享配置为 `0.1`。本次按用户批准的整合口径保留目标 `0.1`，让现有baseline与新upgrade读取同一分配配置，同时保留视频行为。`score_threshold` 同为 `0.9`，LLaVA head score JSON字节相同。来源manifest分别保存两份config的SHA和值；此资源适配意味着不能宣称迁入组合与原opensource按其0.5配置的完整数值结果等价。图片运行另行记录实际目标配置快照。

`provenance/llava_image_sources.json` 记录原始SHA、目标SHA、原始/目标AST、可逆文本变换及上述资源适配。`scripts/check_llava_sources.py` 从当前目标恢复原始字节、核对SHA、重放固定变换，再逐字节比较目标；同时检查实际目标配置与head score，可选 `--source-root` 与当前原目录再核对。仓库运行不需要原目录。

原 `core_sources.json` 中三份已有文件的来源/branding证据保留，通过 `post_migration` 记录精确可逆delta：多模态外壳的构造选择、新分派、SparseMM修复。来源检查先恢复旧目标并执行原证明，再检查新增port。它们不是“只刷新哈希”的来源变更。

## SparseMM跨prompt预算修复

已记录的 `LLAVA-SPARSE-STATE` 来源是 baseline helper 在cluster存在时不重建，而 LLaVA attention每次调用未请求重建。第二个更长prompt因而复用第一个prompt的容量。

本次只改变 LLaVA baseline SparseMM caller：多token prefill调用 `init_sparsemm(..., rebuild_cluster=True)`；单token decode保留同一cluster及其递增的varlen元数据。没有更改公共 SparseMM helper、预算公式、Qwen基线或AsymKV。

CPU回归使用实际caller的初始化语句片段、原始初始化/压缩函数和8→16 token输入；第二次预算与相同长度fresh对象一致，decode保留cluster、head_lens、cu_klen及计数。没有执行完整attention forward。旧 `check_cpu_compression.py` 的历史helper复现测试仍故意调用“不重建”的helper来保存原问题证据；该复现不表示新默认caller仍采用旧路径。

AsymKV分块边界遗漏末token问题不在本次修复范围，图片入口继续不开放AsymKV。

## 与本地Qwen图片历史源码的差异

本次没有替换目标已有的Qwen核心。当前opensource中的Hybrid、SnapKV、SparseMM、MadaKV图片实现与已选目标实现存在结构/选项差异；例如当前opensource SnapKV另有专用cache和text-model forward。因此当前图片后端结果表示目标仓库所选唯一算法在MileBench上的结果，不能直接称为任意历史最优结果的复现。

默认Hybrid评分的source `combined` 与目标 `full` 公式及运算顺序相同。source `sum=abs(left)+abs(right)` 与目标 `importance_only=left+right` 在非负attention分数下相同；source `difference=abs(right-left)` 与目标 `difference_only=(left+right)*abs(right-left)/(left+right+eps)` 不能直接当作等价别名。图片profile只开放主评分路线，不自动迁入这些消融。

## 已验证与未验证

本次仅进行CPU检查。新增七项测试覆盖来源回放、原视频矩阵不变、四种实际图片worker分派，以及四项CPU小张量/状态检查；CPU函数检查以guard拒绝CUDA调用。四种分派逐方法使用独立子进程，只执行实际worker选择与真实patch赋值，模型和算法以带来源信息的替身代替。

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
python scripts/check_llava_sources.py
python scripts/check_llava_dispatch.py
python scripts/check_llava_cpu.py
python -m unittest discover -s tests -p test_llava_image.py -v
```

`check_llava_dispatch.py` 可显式传 `--transformers-root` 读取固定4.52.3源码；不导入Transformers推理实现。报告分别保存在 `provenance/llava_image_dispatch_audit.json` 与 `provenance/llava_image_cpu_audit.json`。原视频19组合分派与静态调用检查另行保留。

完整模型加载、实际generation/FlashAttention/CUDA、任务分数、GPU数值一致性、显存、时延与吞吐均未验证。后续GPU验收应固定模型、数据、样本ID、prompt/像素/生成参数、预算与源码SHA，再逐方法比较旧实现和当前实现。CPU报告不能替代这些结果。

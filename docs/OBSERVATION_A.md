# 观察 A：同一输入的视觉与文本近邻 K/V 相似度

本实验按用户确定的协议执行：**Qwen2.5-VL-7B-Instruct，NExTQA OE/test 和 VATEX/test 原始前200行，各200个输入；只比较近邻**。不构造随机 token 对、间隔对或合并干预。随机种子仅用于从合格近邻中抽取固定位置，不用于挑选相似度较高的结果。

实际400条已完成，耗时约38分钟。[结果、双面板主图与数值表](OBSERVATION_A_RESULTS_20261104.md)。结果保留真实方向，没有据此调整预先确定的统计口径。

## 实现位置

| 文件 | 职责 |
|---|---|
| `src/kvrepack/observations/adjacent.py` | 独立观察 CLI、模型加载、逐条持久化、协议与恢复检查。 |
| `src/kvrepack/observations/inputs.py` | 原任务 prompt、缓存 Arrow/视频、真实 tokenizer offset 与文本分段。 |
| `src/kvrepack/observations/pairs.py` | 时空网格映射、真实近邻配对、等数量抽样、覆盖率。 |
| `src/kvrepack/observations/hooks.py` | 临时捕获语言 decoder 的 raw K、V、实际 RoPE 后 K；结束或异常时恢复。 |
| `src/kvrepack/observations/report.py` | 按源视频整组 bootstrap、表格和 PNG/PDF/SVG 图。 |

观察入口不调用 `replace_qwen` 或任何压缩方法。它通过原生多模态 backbone 做完整提示的因果注意力前向，`use_cache=False`，跳过无关 LM head、解码与任务评分。这里的 FullKV 表示**完整、未压缩的前向表示**；程序不持久保存完整 KV 缓存或原始向量。

## 固定协议

- 单卡、batch=1、BF16、FlashAttention 2；统计以 FP32 计算。64帧，min_pixels=200704，max_pixels=1605632，沿用仓库视频预处理流程。
- 每个任务原始 `doc_id=0..199`，不去重、不打乱、不补选。重复视频与不同问题都保留，统计时按源视频聚类。视频采样索引和重复帧数量另存 metadata。
- 7B 模型包含28个语言 decoder 层、4个 KV 头、head_dim=128；全部参与统计。捕获在 GQA 重复之前，不能把28个 query heads 当成28份独立 K/V。
- 视觉配对是同一实际视觉时间网格、同一行的左右近邻 `(t,r,c)→(t,r,c+1)`，不跨行或跨时间边界。时间网格可能覆盖两帧，不能直接称作同一原始帧。
- NExTQA 主文本组为问题本身；固定答题指令单列为补充。VATEX 主文本组为固定描述指令；原 prompt 的四段文字示例仍参与模型前向，但不混入主文本统计。
- 文本必须在同一自然语言片段中且原序列位置相差1，排除系统文本、聊天/视觉占位标记与 padding。过滤后相邻但原位置不相邻的 token 不组成一对。
- 每个输入取 `M=min(16, 合格视觉近邻数, 合格主文本近邻数)`，两模态各M对。位置在前向前确定，所有层、KV头复用。NExTQA 补充指令组可独立取最多16对，并明确不是主配对样本。
- 主指标为 raw-K、V 的余弦相似度；实际 RoPE 后的 K 为补充结果。仅修正浮点误差到[-1,1]范围，保留所有负值；非有限值或零范数向量会报错，不替换成有利数值。

每条样本先在组内平均近邻对，再等权平均 KV 头和层，最后对规定的200行等权汇总。差值为**视觉近邻均值减主文本近邻均值**，不是“局部性增益”（本轮没有远距对照）。95%置信区间对 `source_video_id` 整组重采样，保留同一视频的全部问题行；默认5000次，bootstrap种子20261104。

主组没有有效邻接对时，仍保留该原始输入的完成记录并报告覆盖不足，不人为扩写文本或用其他行替换。缺行、重复ID、错误层数/头数或非法统计值不会被标为完整实验通过。

`frame_sampling` 的索引是 qwen-vl-utils 初次读取所得帧池中、再重采样至64帧时的索引；其 `original_frames`/`unique_frames` 不应解释为源视频全部解码帧数或独立原始帧数。重复采样情况随结果保留，本轮没有据此作时序冗余的结论。

## 安装与运行

已有项目依赖环境可直接用 `PYTHONPATH=src`；新环境安装观察附加依赖：

```bash
python -m pip install -e '.[observations]'
```

准备相同版本的模型、缓存 Arrow 和视频。`HF_HOME` 默认视频布局为 `nextqa/NExTVideo`、`vatex_test`；可通过 `--nextqa-arrow`、`--vatex-arrow`、`--nextqa-video-root`、`--vatex-video-root` 显式指定。模型复现应使用固定 snapshot 的本地路径，本轮7B snapshot为 `cc594898137f460bfe9f0759e9844b3ce807cfb5`。

```bash
PYTHONPATH=src python -m kvrepack.observations.adjacent \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct-snapshot \
  --hf-home /path/to/huggingface_cache \
  --task both --limit 200 --frames 64 --max-pairs 16 --seed 42 \
  --device 0 --offline --summarize \
  --output outputs/observations/qwen7b_adjacent_first200
```

安装后也可使用 `kvrepack-observe`。中断后使用完全相同的设置和输出目录，增加 `--resume`；请保留原目录与协议文件，不手工混入其他运行的行。计时 pilot 使用独立目录和 `--limit 2`；它不能直接作为400条协议目录续跑，因为样本范围本身也属于协议。

已有结果可在无 GPU 的机器生成图表：

```bash
PYTHONPATH=src python -m kvrepack.observations.report \
  --input outputs/observations/qwen7b_adjacent_first200/records.jsonl \
  --output outputs/observations/qwen7b_adjacent_first200/report \
  --expected-layers 28 --expected-kv-heads 4
```

生成紧凑的论文双面板主图（左raw K、右V，复用已有置信区间，不重新选择样本或重算统计）：

```bash
python scripts/plot_adjacent_observation.py \
  --summary outputs/observations/qwen7b_adjacent_first200/report/summary.json \
  --output outputs/observations/qwen7b_adjacent_first200/report/adjacent_main \
  --title 'Qwen2.5-VL-7B'
```

主图同时输出 PNG（300dpi）、PDF、SVG 和来源 metadata；逐数据集的均值／配对差值图另保留为补充结果。

## 产物与解释范围

`contract.json` 保存设置、观察代码/HF模型源码/模型metadata/任务配置/Arrow的SHA与依赖版本；模型权重payload、视频内容和全部运行库并未逐一做完整字节哈希，不能称完整依赖闭包证明。`records.jsonl` 保存原行ID、文本位置、配对位置、逐层逐KV头均值、输入tensor哈希与时间；`run_state.json` 为即时进度，`errors.jsonl` 保留错误。`report/` 生成完整性检查、逐样本/逐头/逐层/任务CSV、PNG/PDF/SVG及`RESULTS.md`。

视觉/文本标签指**输入位置的来源**；深层表示已经受多模态上下文影响。VATEX指令与NExTQA问题分开报告。本图只能说明这些输入和模型下的相邻表示相关性，不能单独证明合并无损、压缩后任务分数不变或模态自身的因果效应。

本轮实际运行目录为 `outputs/observations/qwen7b_adjacent_first200_20261104`。正式运行前已告知时间估计：4条实测外推约80分钟，考虑慢视频解码预留1–2小时。最终结果与实际耗时以运行状态和完成报告为准。

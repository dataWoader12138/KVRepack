# Qwen2.5 的 MixKV 式完整输入小批量观察

本轮按用户最新要求执行：停止原来的等量token随机子集视频实验，改为类似MixKV图1(b)的**纯文本模型＋长文本 vs 视觉语言模型＋图文**。用户指定沿用2.5系列；**每组只5个完整输入，跑完停止，不自动扩展为全量**。

**5＋5已完成，GPU已释放，没有扩大。** [实际结果、两阶段图与解释](MIXKV_REFERENCE_PILOT_RESULTS_20261104.md)。

## 已停止的旧实验

`outputs/observations/qwen7b_mixkv_style_first200_20261104`已按用户要求停止，状态为`stopped_by_user`，365行完整记录保留：NExTQA200、VATEX165，无尾部不完整行。GPU进程已退出。旧[等token数抽样协议](MIXKV_STYLE_OBSERVATION.md)不再作为当前计划，也不能把365/400标成完整通过。NExTQA200行的阶段报告在其`nextqa_checkpoint/report/`，不是新的5＋5结果。

## 本轮固定设置

| 条件 | 模型及固定revision | 输入 |
|---|---|---|
| 纯文本 | `Qwen/Qwen2.5-7B-Instruct`，`a09a35458c702b33eeacc393d103063234e8bc28` | LongBench v1/qasper/test原始行0–4的完整article、原question及官方prompt。 |
| 图文 | `Qwen/Qwen2.5-VL-7B-Instruct`，`cc594898137f460bfe9f0759e9844b3ce807cfb5` | OCRBench/test固定行0、200、400、600、800，原完整图片和原question。 |

行号在读取激活前固定，不按token长度、相似度或结果选样。本轮OCR行号沿原数据集顺序等间隔取五个输入，避免仅使用最前五个同类单词识别图片；不宣称这是全基准的代表性估计。

- 观察完整未压缩prefill的**全部输入token**，包含聊天结构与系统文本，无padding、无生成token。不随机抽token，不取top-k，不修改原context或图片内容，不静默截断输入或换样本。
- 两模型都使用28层、4个原生KV头、head_dim128；不将GQA扩展后的query头当成独立KV头。
- 单卡BF16、原生FlashAttention2、batch1、`use_cache=False`，直接调用完整backbone跳过LM head。不调用任何压缩补丁，不保存完整KV张量。
- 分别捕获raw K、实际RoPE后K和V；主图上为RoPE后K、下为V，raw K完整另报。K的捕获阶段不能按结果选择。
- 每层每头计算所有不同位置`i≠j`的平均余弦。用向量和的恒等式进行线性内存计算，不构造长文本的T×T矩阵。FP32归一化、FP64归约，精确去除实际归一化后的对角范数和，保留负值。
- 先计算各输入各层各头全配对均值，再对五个输入等权平均，形成每个模型112个层头点的描述分布。每个完整输入不会因为更长而获得更高样本权重。逐输入数值和真实token数同时保存。
- 五条是探索性pilot，不作总体推断或置信区间，不把112个层头点当作112个独立输入。无论小批量差距大小，程序都只执行规定的5条，不续跑全量。

## 图像处理与数据来源

使用Qwen2.5-VL原生图像处理流程保留宽高比，`min_pixels=32×28×28=25088`、`max_pixels=16384×28×28=12845056`。这些数值参考[MixKV公开分析脚本](https://github.com/xuyang-liu16/MixKV/blob/eeb708bc3be65beade00e231a7b2c2160995f3c5/distribution_qwen.py)，不能声称它们是作者已披露的图1设置。原始图片编码字节保持不变，模型实际处理的尺寸、grid和视觉token数单列。

LongBench固定数据revision为`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，使用data.zip中原qasper.jsonl；OCRBench固定为`92a54bd1384387c178d5a07140a2d85e0a3d12e1`的test parquet。读取ZIP白名单成员和parquet，不执行远程dataset脚本。

qasper prompt逐字采用[LongBench官方配置](https://raw.githubusercontent.com/THUDM/LongBench/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json)中的模板，保留完整原article和question；模板文件SHA256为`56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30`。系统提示统一为`You are a helpful assistant.`，两模型使用各自原生聊天模板。

## 与MixKV原图的关系

[MixKV图1及附录A.1](https://arxiv.org/html/2510.20707v2#A1)使用Qwen2-7B/LongBench与Qwen2-VL-7B/OCRBench。它没有完整披露图1的样本ID、数量、Base/Instruct/revision、精确K捕获阶段、prompt和绘图脚本。本轮沿用它的模型/输入条件对照形式和§3.3非对角余弦公式，按用户要求换成Qwen2.5系列，并明确固定上述缺失细节。**这是类似实验，不是原图精确数值复现。**

完整输入与位置长度仍不同，模型参数、训练和语料也不同。结果描述的是两模型在这十个输入上的KV几何，不能直接推导同一VLM中视觉token必然比后置文本token更冗余，更不能单凭余弦证明合并无损。

## 代码和运行

- `reference_inputs.py`：固定数据、官方prompt、完整原图片、SHA及本地十样本manifest。
- `native_all_pairs.py`：两个原生模型的独立临时只读hook、完整输入逐层逐头统计及清理。
- `reference.py`：每次只运行一个固定五输入组，记录协议、输入hash、时间、完整性与状态。
- `reference_report.py`：两个五输入组的严格覆盖检查、逐输入/层/头CSV和PNG/PDF/SVG。

先准备本地数据（manifest含长文章正文，留在忽略的outputs中，不上传Git）：

```python
from kvrepack.observations.reference_inputs import prepare_reference_samples
prepare_reference_samples('/path/to/huggingface_cache',
                          'outputs/observations/qwen25_reference/inputs')
```

每组分别运行，加载完成的固定snapshot。本地纯文本模型需新增下载；现有7B VL snapshot可复用。两个进程顺序运行，不占用两份GPU模型：

```bash
PYTHONPATH=src python -m kvrepack.observations.reference \
  --group text --pretrained /path/to/a09a35458c702b33eeacc393d103063234e8bc28 \
  --hf-home /path/to/huggingface_cache \
  --inputs outputs/observations/qwen25_reference/inputs/reference_samples.json \
  --output outputs/observations/qwen25_reference/text

PYTHONPATH=src python -m kvrepack.observations.reference \
  --group vision --pretrained /path/to/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --hf-home /path/to/huggingface_cache \
  --inputs outputs/observations/qwen25_reference/inputs/reference_samples.json \
  --output outputs/observations/qwen25_reference/vision

cat outputs/observations/qwen25_reference/text/records.jsonl \
    outputs/observations/qwen25_reference/vision/records.jsonl \
    > outputs/observations/qwen25_reference/records.jsonl
PYTHONPATH=src python -m kvrepack.observations.reference_report \
  --input outputs/observations/qwen25_reference/records.jsonl \
  --output outputs/observations/qwen25_reference/report
```

正式GPU小批量开始前告知估计时间，结束后报告实际耗时和全部方向。下载是模型准备时间，单独记录，不与GPU前向时间混为一项。

# Qwen2.5完整输入相似度观察：每组200条

用户在5＋5诊断完成后，明确授权将同一实验扩大到两组各200条并返回数值和绘图。本轮是新的独立运行，不恢复已经停止的旧视频等token子集实验，也不改写已发布pilot。

**400/400已完成并停止，GPU已释放。** [实际数值、两阶段图、置信区间与RoPE解释](MIXKV_REFERENCE_200_RESULTS_20261104.md)。

## 固定设置

| 条件 | 模型 | 原始数据行 |
|---|---|---|
| 纯文本 | Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28` | LongBench v1/qasper/test，0..199，全部200条。 |
| 图文 | Qwen2.5-VL-7B-Instruct，revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` | OCRBench/test，0、5、10…995，共200条。 |

OCR行号沿原测试顺序固定等间隔取样；包含此前5条pilot图像，覆盖各测试区段，不读取激活来选样。Qasper此前5条也包含在本轮中，可作输入和数值一致性对照。这里采样的是完整benchmark输入，**没有token抽样**。

其余设置沿用[5＋5协议](MIXKV_REFERENCE_PILOT.md)：完整原article＋原question＋官方qasper提示；完整原图＋原question，原生图像处理min_pixels25088/max_pixels12845056；统一系统提示，原生聊天模板；BF16、batch1、FlashAttention2、完整未压缩prefill，`use_cache=False`，不生成、不算任务答案分数。

主图为RoPE后K与V，raw K完整另报。每条输入的全部有效token都参与非自身全配对余弦，不包含对角线的1。使用同一个`native_all_pairs.py`和线性内存归约，保留负值。先每条输入等权平均KV头/层，再对200条输入等权汇总；较长输入不获得更高样本权重。每模型28层×4个原生KV头，共112个跨输入平均的层头点，用于分布展示，不作为112个独立输入。

## 来源重复和置信区间

Qasper200条含148个不同context SHA，OCR200条含199个不同原图SHA。重复来源的不同问题均保留，不去重、不补选。200模式新增95%百分位bootstrap区间：文本按context SHA整组采样，图文按原图SHA整组采样，保留每个被抽来源的全部成员，并按实际行数计算等输入权重均值。

两组来自不同数据集，**不按doc_id、层或头进行跨组配对**。默认5000次、seed20261104，使用独立子随机流分别抽两组；组间差为图文均值减纯文本均值。置信区间描述该固定样本选择及其来源重采样，不把OCR固定区段样本或Qasper子任务视为整个视觉/文本世界的随机代表。

## 预检查与范围

CPU预检查：Qasper完整输入2094–22064 token，全部低于当前原生32768窗口；148个源文章。OCR按原生缩放/grid展开预估61–16308 token，199个源图片；最大图约4961×7016，按设定原生缩放。最大图doc700已用真实processor在CPU验证，总16308、视觉16264、grid为`[1,304,214]`。正式运行仍逐条记录实际长度与输入hash，不以预估替代实际结果，不静默截断或删除大输入。

扩大范围只改变显式样本数量、固定行号与统计不确定性，K/V捕获及数值归约模块保持不变。默认CLI仍是5条；200条必须显式声明，不自动扩展。

## 本机运行与复现

本机目录：`outputs/observations/qwen25_mixkv_reference_200_20261104/`。`inputs/reference_samples.json`含完整长文本，留在忽略的outputs中，不上传Git。模型和数据已缓存，离线准备输入：

```python
from kvrepack.observations.reference_inputs import prepare_reference_samples
prepare_reference_samples('/path/to/huggingface_cache',
                          'outputs/observations/qwen25_reference200/inputs',
                          offline=True, sample_count=200)
```

每个模型用独立进程，在同一GPU顺序运行：

```bash
PYTHONPATH=src python -m kvrepack.observations.reference \
  --group text --sample-count 200 \
  --pretrained /path/to/a09a35458c702b33eeacc393d103063234e8bc28 \
  --hf-home /path/to/huggingface_cache \
  --inputs outputs/observations/qwen25_reference200/inputs/reference_samples.json \
  --output outputs/observations/qwen25_reference200/text
```

图文组将group换为vision、snapshot换为`cc594898137f460bfe9f0759e9844b3ce807cfb5`、输出换为vision。均只运行所声明的200条，之后停止。汇总两个完整records后：

```bash
PYTHONPATH=src python -m kvrepack.observations.reference_report \
  --sample-count 200 --bootstrap-samples 5000 --seed 20261104 \
  --input outputs/observations/qwen25_reference200/records.jsonl \
  --output outputs/observations/qwen25_reference200/report
```

报告生成主图和raw补图的PNG/PDF/SVG，逐输入、逐层、逐头、来源聚类和组间比较CSV。模型/任务/输入hash、真实长度、每层捕获次数和时长随记录保存。

这是参考MixKV图1(b)形式的Qwen2.5适配。原图的精确采集点未公开确认，本轮明确采用post-RoPE K主图且同时保留raw K。两模型的参数、语料、位置布局与完整输入长度不同，本实验不隔离同一VLM的模态因果效应，也不证明合并无损。

## 只重绘连续分布图（CPU）

已有归档包含绘图所需的全部层/头均值，无需模型和GPU：

```bash
python scripts/replot_reference_observation.py --report provenance/mixkv_reference_qwen25_200_20261104/report
```

生成 `reference_main` 与 `reference_raw` 的 PNG/PDF/SVG；脚本核对672个唯一层/头/指标身份，并校验原始summary与CSV均未改变。平滑采用Scott带宽（最低0.01）及余弦边界反射；这只是明确的绘图选择，并不声称复刻MixKV未披露的具体平滑实现。物理支持范围保留[-1,1]，当前可视范围[-0.2,1]覆盖全部实际头均值；若出现更小的有符号值，绘图范围自动扩展。

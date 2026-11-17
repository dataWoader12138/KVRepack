# Qwen2.5两模型各200条：完整序列相邻token观察

用户于2026-11-17澄清需要相邻token余弦相似度。此次复用此前全配对图的同一批400个输入及模型，只更换观察统计；旧全配对结果不覆盖。

- Qwen2.5-7B-Instruct：LongBench/qasper test原始0至199行，固定模型revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- Qwen2.5-VL-7B-Instruct：OCRBench固定0、5、10至995行，revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`。
- BF16、FlashAttention2、batch1、原生完整未压缩backbone前向，`use_cache=False`，无生成。沿用原prompt、图片和像素上下限25088/12845056。

## 统计定义

对每个输入、每层、每个原生KV头，计算完整序列所有 `(i,i+1)` 对的余弦均值，共 `N−1` 对。不抽样token或pair，不去除负值，不跨输入拼接，也不包含自身配对。按序列位置定义相邻，图文组包括图像、文本、结构token及它们之间的边界；图像展平后行尾与下一行行首也计入，因此不能称为只统计二维水平视觉近邻。

同时捕获原投影的raw K、V和原生RoPE函数输出的K，均在GQA重复前，28层×4个原生KV头。每个头先对200输入等权平均；总体均值等权平均112层头单元。绘图沿用112个层头均值的KDE，曲线不是逐token对原始值的密度。不用token数给长输入加权。

统计以FP32归一化、FP64邻接点积与归约。新模块`native_adjacent.py`复用旧原生hook的捕获位置及清理逻辑，只替换归约；旧全配对入口和压缩核心保持不变。记录使用独立标识`all_sequence_adjacent_v1`，新报告拒绝混入旧全配对记录。

## 验证与运行

14项CPU测试通过：逐对oracle、与全配对的区别、全部端点/跨边界、BF16和负值、非法向量、两种模型捕获、原生头数量、原前向输出不变及异常清理。独立GPU pilot每模型2条（text0/1、vision0/5）完成，全部输入tensor与旧400条记录一致。pilot不混入正式200条结果。

本机单卡A800-SXM4-80GB。pilot两个进程含依赖导入、加载共22.904秒；样本长度不同，不能简单把pilot总时长乘100。正式400条运行前告知预计5至10分钟，真实耗时以state为准。

```bash
PYTHONPATH=src python scripts/run_reference_adjacent.py \
  --inputs /path/to/reference_samples.json \
  --text-model /path/to/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --vision-model /path/to/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --hf-home /path/to/huggingface_cache \
  --reference-root /path/to/previous_full_input_all_pairs_run \
  --sample-count 200 \
  --output outputs/observations/qwen25_sequence_adjacent_200_20261117
```

`reference_samples.json`复用此前通过`reference_inputs.prepare_reference_samples(..., sample_count=200)`准备的400行manifest。可省略`--reference-root`独立运行；此次实验保留该参数，每条核对旧输入的完整tensor哈希、聊天SHA、长度和来源metadata。图片/原始长文本和权重不上传Git。

worker逐条保存`records.jsonl`、contract和state。控制器串行text、vision，之后CPU统计绘图；`--sample-count 2 --no-report`仅用于独立pilot目录。若采集时使用`--no-report`，可在采集后运行：

```bash
PYTHONPATH=src python -m kvrepack.observations.reference_adjacent_report \
  --text-input outputs/observations/qwen25_sequence_adjacent_200_20261117/text/records.jsonl \
  --vision-input outputs/observations/qwen25_sequence_adjacent_200_20261117/vision/records.jsonl \
  --sample-count 200 \
  --output outputs/observations/qwen25_sequence_adjacent_200_20261117/report
```

正式ROOT：`outputs/observations/qwen25_sequence_adjacent_200_20261117`。pilot ROOT：`outputs/observations/qwen25_sequence_adjacent_20261117_preflight`。来源和发布产物：`provenance/reference_adjacent_qwen25_200_20261117`。是否完成以正式state和报告为准。

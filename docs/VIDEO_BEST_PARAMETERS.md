# 视频 KVRepack 最佳参数核对

核对范围：本机已保存的 NExTQA、VATEX、VideoChatGPT 完成记录，Qwen2.5-VL 3B/7B 与 LLaVA-OneVision 7B。这里的“最好”限定于找到的可比候选；单一已完成配置明确标为候选。没有运行新实验、调用评分API或修改代码默认参数。

下载：[CSV表格](video_best_parameters.csv) · [逐行指标、来源SHA与参数证据](../provenance/video_best_parameters.json) · [原始核查清单](../provenance/video_parameter_audit/)。每行ID可在JSON中搜索。

## 参数与比较条件

- 参数顺序统一为 **simtop / cr / α**：simtop为选择合并的head比例，cr为候选序列压缩后的比例，α为合并重要性差异权重；cr不是最终KV保留比例。
- 主表统一64帧、`--limit 200`、最终预算比例0.10、`score_correction=none`。绝对`BUDGET`并非统一1024：部分Qwen Snap/Mada运行未设置它，按比例计算；历史事实逐条保留在JSON。
- NExTQA按WUPS排序，使用记录中的0–100分值。VATEX沿用原调参规则，按BLEU-4、METEOR、ROUGE-L、CIDEr四项原始小数的等权均值排序；下面四项及综合分均统一乘100展示。CIDEr单项最优另表列出。
- VideoChatGPT主表只收**独立输出评分**，实际judge为`gpt-5.4-mini-2026-03-17`。综合分为正确性、细节、上下文、时序、一致性五项均值，不是三个子任务等权。生成各子任务200条，一致性配对为100个评分单元；五项计数为200/200/200/200/100，共900个评分单元。
- 当前仓库尚未迁入LLaVA四种改进；该模型各行是原port的历史结果参考。Qwen超参数名称可映射到现有CLI，但历史Snap预算取整/缓存协议可能不同，不能据表中参数承诺复现原分数。

## NExTQA

| 模型 | KVRepack方法 | simtop / cr / α | WUPS↑ | 核对状态 | ID |
|---|---|---|---:|---|---|
| Qwen2.5-VL-3B | HybridKV | 0.30 / 0.70 / 2 | 30.12846 | 已观测最好 | P01 |
| Qwen2.5-VL-3B | SnapKV | 0.70 / 0.90 / 2 | 30.49073 | 已观测最好 | P02 |
| Qwen2.5-VL-3B | SparseMM | 0.70 / 0.90 / 2 | 31.08741 | 已观测最好 | P03 |
| Qwen2.5-VL-3B | MadaKV | 0.50 / 0.50 / 2 | 30.59846 | 已观测最好 | P04 |
| Qwen2.5-VL-3B | LookM | — | — | 仅旧log_count | P05 |
| Qwen2.5-VL-7B | HybridKV | 0.50 / 0.50 / 2 | 33.21990 | 已观测最好 | P06 |
| Qwen2.5-VL-7B | SnapKV | 0.30 / 0.70 / 2 | 33.16973 | 已观测最好 | P07 |
| Qwen2.5-VL-7B | SparseMM | 0.50 / 0.70 / 2 | 34.00903 | 已观测最好 | P08 |
| Qwen2.5-VL-7B | MadaKV | 0.70 / 0.70 / 1 | 33.64294 | 已观测最好 | P09 |
| Qwen2.5-VL-7B | LookM | — | — | 无已核实记录 | P10 |
| LLaVA-OV-7B | HybridKV | 0.30 / 0.70 / 2 | 19.64307 | 已观测最好；原port未迁入 | P11 |
| LLaVA-OV-7B | SnapKV | 0.50 / 0.50 / 1 | 19.00936 | 已观测最好；原port未迁入 | P12 |
| LLaVA-OV-7B | SparseMM | 0.70 / 0.70 / 1 | 19.01221 | 已观测最好；原port未迁入 | P13 |
| LLaVA-OV-7B | MadaKV | 0.50 / 0.50 / 2 | 18.71694 | 已观测最好；原port未迁入 | P14 |
| LLaVA-OV-7B | LookM | — | — | 无已核实记录 | P15 |

## VATEX

| 模型 | KVRepack方法 | simtop / cr / α | B-4×100 | METEOR×100 | R-L×100 | CIDEr×100 | 综合×100↑ | 核对状态 | ID |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-VL-3B | HybridKV | 0.50 / 0.50 / 2 | 29.852 | 23.047 | 48.255 | 55.216 | 39.093 | 已观测最好；参数证据较弱 | P16 |
| Qwen2.5-VL-3B | SnapKV | 0.70 / 0.90 / 2 | 27.864 | 21.904 | 47.192 | 51.083 | 37.011 | 已观测最好；参数证据较弱 | P17 |
| Qwen2.5-VL-3B | SparseMM | 0.70 / 0.90 / 2 | 27.881 | 21.000 | 45.650 | 44.226 | 34.689 | 已观测最好 | P18 |
| Qwen2.5-VL-3B | MadaKV | 0.50 / 0.70 / 2 | 26.762 | 20.967 | 45.731 | 48.076 | 35.384 | 已观测最好 | P19 |
| Qwen2.5-VL-3B | LookM | — | — | — | — | — | — | 仅旧log_count | P20 |
| Qwen2.5-VL-7B | HybridKV | 0.50 / 0.50 / 2 | 22.858 | 22.444 | 43.381 | 48.891 | 34.394 | 已观测最好 | P21 |
| Qwen2.5-VL-7B | SnapKV | 0.15 / 0.70 / 0.5 | 21.929 | 21.439 | 42.597 | 46.303 | 33.067 | 已观测最好 | P22 |
| Qwen2.5-VL-7B | SparseMM | 0.50 / 0.50 / 2 | 20.798 | 21.472 | 43.070 | 44.324 | 32.416 | 已观测最好 | P23 |
| Qwen2.5-VL-7B | MadaKV | 0.30 / 0.70 / 2 | 21.232 | 21.711 | 42.720 | 45.316 | 32.745 | 已观测最好 | P24 |
| Qwen2.5-VL-7B | LookM | — | — | — | — | — | — | 无已核实记录 | P25 |
| LLaVA-OV-7B | HybridKV | 0.50 / 0.70 / 2 | 12.675 | 17.940 | 36.782 | 21.766 | 22.291 | 已观测最好；原port未迁入 | P26 |
| LLaVA-OV-7B | SnapKV | 0.50 / 0.50 / 0.5 | 11.396 | 17.125 | 35.803 | 17.916 | 20.560 | 已观测最好；原port未迁入 | P27 |
| LLaVA-OV-7B | SparseMM | 0.50 / 0.70 / 2 | 12.057 | 17.074 | 36.177 | 18.352 | 20.915 | 已观测最好；原port未迁入 | P28 |
| LLaVA-OV-7B | MadaKV | 0.50 / 0.50 / 2 | 11.129 | 16.873 | 35.525 | 17.755 | 20.320 | 已观测最好；原port未迁入 | P29 |
| LLaVA-OV-7B | LookM | — | — | — | — | — | — | 无已核实记录 | P30 |

### 若只优化CIDEr

| 模型 / 方法 | simtop / cr / α | CIDEr×100↑ | 四项综合×100 | ID |
|---|---|---:|---:|---|
| Qwen2.5-VL-3B / SparseMM | 0.50 / 0.70 / 2 | 44.773 | 34.620 | A01 |
| Qwen2.5-VL-7B / SparseMM | 0.50 / 0.70 / 2 | 44.417 | 32.406 | A02 |

这两行与综合分最优不相同，应按报告指标选择，不能把两个选择规则混为一个“最好”。

## VideoChatGPT：独立评分主表

| 模型 | KVRepack方法 | simtop / cr / α | 五指标均值↑ | 评分方式 | 核对状态 | ID |
|---|---|---|---:|---|---|---|
| Qwen2.5-VL-3B | HybridKV | 0.50 / 0.50 / 2 | 2.809 | 逐输出独立 | 单一已完成候选 | P31 |
| Qwen2.5-VL-3B | SnapKV | 0.70 / 0.90 / 2 | 2.771 | 逐输出独立 | 单一已完成候选 | P32 |
| Qwen2.5-VL-3B | SparseMM | 0.70 / 0.90 / 2 | 2.700 | 逐输出独立 | 单一已完成候选 | P33 |
| Qwen2.5-VL-3B | MadaKV | 0.50 / 0.80 / 1 | 2.732 | 逐输出独立 | 单一已完成候选 | P34 |
| Qwen2.5-VL-3B | LookM | — | — | — | 无已核实记录 | P35 |
| Qwen2.5-VL-7B | HybridKV | 0.30 / 0.80 / 2 | 3.065 | 逐输出独立 | 单一已完成候选 | P36 |
| Qwen2.5-VL-7B | SnapKV | 0.30 / 0.70 / 2 | 3.034 | 逐输出独立 | 单一已完成候选 | P37 |
| Qwen2.5-VL-7B | SparseMM | 0.30 / 0.80 / 2 | 2.940 | 逐输出独立 | 单一已完成候选 | P38 |
| Qwen2.5-VL-7B | MadaKV | 0.30 / 0.70 / 2 | 3.001 | 逐输出独立 | 单一已完成候选 | P39 |
| Qwen2.5-VL-7B | LookM | — | — | — | 无已核实记录 | P40 |
| LLaVA-OV-7B | HybridKV | 0.50 / 0.70 / 2 | 3.042 | 独立隔离缓存 | 已观测最好；原port未迁入 | P41 |
| LLaVA-OV-7B | SnapKV | 0.30 / 0.70 / 2 | 3.057 | 逐输出续跑，无跨输出缓存 | 已观测最好；原port未迁入 | P42 |
| LLaVA-OV-7B | SparseMM | 0.30 / 0.80 / 2 | 3.034 | 逐输出续跑，无跨输出缓存 | 已观测最好；原port未迁入 | P43 |
| LLaVA-OV-7B | MadaKV | 0.30 / 0.85 / 2 | 3.047 | 独立隔离缓存 | 已观测最好；原port未迁入 | P44 |
| LLaVA-OV-7B | LookM | — | — | — | 无已核实记录 | P45 |

Qwen主表的8项在该独立评分组中各只有一组已完成参数，属于可核实候选。LLaVA MadaKV第一、第二名分别3.047、3.046，仅差0.001，不能据此断言稳定优势。各五项分数及样本ID摘要保留在JSON/原核查清单。

### 共享缓存或配对复评：单独排名

| 模型 | 方法 | 评分协议 | 本协议内simtop / cr / α | 五指标均值↑ | 参数组数 | ID |
|---|---|---|---|---:|---:|---|
| Qwen2.5-VL-3B | SparseMM | `paired_shared_prompt_cache` | 0.70 / 0.90 / 2 | 2.672 | 1 | C01 |
| Qwen2.5-VL-7B | HybridKV | `single_output_shared_prompt_cache` | 0.50 / 0.50 / 2 | 3.061 | 5 | C02 |
| Qwen2.5-VL-7B | MadaKV | `single_output_shared_prompt_cache` | 0.70 / 0.70 / 1 | 3.031 | 2 | C03 |
| Qwen2.5-VL-7B | SnapKV | `paired_shared_prompt_cache` | 0.30 / 0.70 / 2 | 3.021 | 1 | C04 |
| LLaVA-OV-7B | HybridKV | `single_output_shared_prompt_cache` | 0.30 / 0.85 / 2 | 3.045 | 5 | C05 |
| LLaVA-OV-7B | MadaKV | `single_output_shared_prompt_cache` | 0.50 / 0.70 / 2 | 3.043 | 5 | C06 |

因此，Qwen7B MadaKV在共享缓存组的最佳为**0.70 / 0.70 / 1，3.031**；独立评分主表为**0.30 / 0.70 / 2，3.001**。Qwen7B HybridKV共享组最佳为**0.50 / 0.50 / 2，3.061**，独立组已完成候选为**0.30 / 0.80 / 2，3.065**。这些是不同复评/缓存组，未混为一个统一排名。GPT-5.5旧评分在核查清单中独立保存，也不参与主表。

### 若按三个子任务等权

另一种聚合是先取generic三项均值，再与temporal、consistency等权平均。这与主表的五指标均值不同。已有Qwen7B HybridKV记录中，按这一聚合跨相同judge提示族描述性比较，0.50 / 0.50 / 2约为2.92167，0.30 / 0.80 / 2约为2.91833，排序会变；两条的缓存/评分批次仍需分别保留。子任务独立选优清单保存在`video_parameter_audit/videochatgpt_qwen.json`的`best_by_subtask_same_judge_prompt_family`，不冒充主表目标。

## 历史log_count记录

| 数据集 | 模型 | 方法 | simtop / cr / α | 记录分数 | 状态 | ID |
|---|---|---|---|---:|---|---|
| NExTQA | Qwen2.5-VL-3B | SnapKV | 0.00 / 0.90 / 1 | 30.67239 | 旧log_count；不混入当前none主表 | H01 |
| NExTQA | Qwen2.5-VL-3B | SparseMM | 0.20 / 0.95 / 1 | 31.35364 | 旧log_count；不混入当前none主表 | H02 |
| NExTQA | Qwen2.5-VL-3B | MadaKV | 0.10 / 0.80 / 1 | 30.59846 | 旧log_count；不混入当前none主表 | H03 |
| VATEX | Qwen2.5-VL-3B | SparseMM | 0.00 / 0.80 / 1 | 34.65523 | 旧log_count；不混入当前none主表 | H04 |
| VATEX | Qwen2.5-VL-3B | MadaKV | 0.50 / 0.80 / 1 | 35.58195 | 旧log_count；不混入当前none主表 | H05 |
| NExTQA | Qwen2.5-VL-3B | LookM | 0.50 / 0.80 / 1 | 30.51484 | 旧log_count；不混入当前none主表 | H06 |
| VATEX | Qwen2.5-VL-3B | LookM | 0.50 / 0.80 / 1 | 38.42431 | 旧log_count；不混入当前none主表 | H07 |

历史行的alpha等参数部分仅有目录/旧配置证据，完整可信度见JSON。3B LookM目前只找到这一类历史记录；两模型LookM在VideoChatGPT及7B LookM在两确定性数据集均缺少可核实当前记录，未从别的模型或数据集借值填表。

## 排除项与复现注意

- Qwen7B SnapKV NExTQA记录WUPS=33.313069，目录标0.10预算，但运行日志实为`budget_ratio=0.0000`、固定512历史KV；主表改选合规0.30 / 0.70 / 2的33.169733。原日志证据见JSON的`excluded_or_incomparable`。Hybrid的`retrieval_budget=512`只是动态头的块预算，不以此误判总预算。
- Qwen3B VATEX Hybrid/Snap两行没有找回原始参数日志，结果分数已读JSON，并与明确的online-best引用/转移脚本交叉；已降低参数证据置信度。没有把这一限制藏在“最好”标题中。
- Qwen7B Snap NExTQA同参数后续重跑约32.8983/32.9177，未当作新参数；表中是最高已保存记录，不是多次平均。详细重复结果见JSON。
- LLaVA旧脚本有只设`SNAPKV_DECOUPLE_K_DIFF_ALPHA`而核心实读`K_DIFF_ALPHA`的情况；主表两条Snap确定性最优取自`truealpha`运行，实际α分别1与0.5。
- 不混入6/16小样本筛选、K相似度/score消融、不同预算或不同judge的高分。图片MileBench实验也不属于本表。

## 在当前CLI中设置超参数

例：设置Qwen3B NExTQA SnapKV的三个核心超参数，并先查看CPU计划：

```bash
PYTHONPATH=src python -m kvrepack.run --model qwen2_5_vl --method snapkv_kvrepack \
  --pretrained Qwen/Qwen2.5-VL-3B-Instruct --task nextqa_oe_test \
  --frames 64 --limit 200 --budget-ratio 0.10 \
  --set SNAPKV_SIMHEAD_MERGE_RATIO=0.70 \
  --set SNAPKV_KVREPACK_COMPRESS_RATIO=0.90 --set K_DIFF_ALPHA=2 --dry-run
```

这只映射核心旋钮，不能替代历史实现、额外运行标志和完整环境的核对。LLaVA改进组合仍需原port或先完成迁入；本次只发布参数表与证据，未改runner默认值，也未宣称GPU分数已在新仓库复现。

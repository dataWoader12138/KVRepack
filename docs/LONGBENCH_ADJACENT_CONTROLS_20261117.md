# LongBench其他文本子集的相邻K对照

用户希望检查Qasper的高raw-K邻接相似度是否特有。保持Qwen2.5-7B-Instruct、原生完整chat、BF16、FlashAttention2、batch1与全部N−1相邻对不变，增加HotpotQA、TriviaQA和LCC原始前200条，共600次完整未压缩前向。主结果为当前层RoPE前raw K，不按top200选取；全28层×4原生KV头、每输入等权。

## 输入预检

使用已缓存的固定LongBench ZIP revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`及官方prompt commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`；模型revision为`a09a35458c702b33eeacc393d103063234e8bc28`。任务模板填入完整原context/input，再统一原system/user chat框架。没有生成、压缩、截断、换行补样本或根据相似度筛选数据。

| 子集 | 原始行 | 完整输入token最小/最大/均值 |
|---|---|---|
| HotpotQA | 0–199 | 1917 / 17674 / 13465.455 |
| TriviaQA | 0–199 | 1565 / 24353 / 12227.040 |
| LCC | 0–199 | 1023 / 14240 / 3071.490 |

三组均不超过32768原生窗口。多领域QA的`multifieldqa_en`只有150条，未补造200；选择基于领域差异与完整输入可执行性，未观察cosine后选集。长度证据见`provenance/longbench_adjacent_controls_20261117/length_preflight.json`。

## 同次前向的几何诊断

保留原raw K、RoPE K、V全部邻接统计，额外对捕获的raw K副本计算：非自身全配对均值、按当前输入/头减去序列均值后的邻接余弦、减去`k_proj.bias`后的邻接余弦。后两项在FP32副本上事后变换，不改变实际模型或注意力，减去BF16已舍入投影中的bias也不等同于关闭bias重新运行模型。

这些诊断用于区分局部相邻相关性与表示共同方向的影响。已有研究讨论注意力Q/K表示的各向异性，但它本身不能证明本模型本实验的机制，需结合实际诊断：[Godey et al., EACL 2024](https://aclanthology.org/2024.eacl-long.3/)。全部归约沿原FP32归一化、FP64点积/累加。

各2条GPU pilot独立完成，6条不混入正式600条；模型加载及采集9.450秒。9项新增诊断CPU测试覆盖显式oracle、共享bias例子、BF16、原前向/参数不变及异常清理；另3项输入测试覆盖完整前缀/原文保留、QA与代码模板和不足行拒绝。正式运行前告知约10–15分钟，实际状态以run_state为准。

```bash
PYTHONPATH=src python -m kvrepack.observations.longbench_controls \
  --hf-home /path/to/huggingface_cache \
  --pretrained /path/to/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --subsets hotpotqa triviaqa lcc --sample-count 200 \
  --output outputs/observations/longbench_adjacent_controls_200_20261117
```

用`--sample-count 2`和新目录做pilot，恢复显式`--resume`。原Qasper与VATEX结果保留，本轮不重跑视频。正式ROOT为`outputs/observations/longbench_adjacent_controls_200_20261117`，pilot ROOT为`outputs/observations/longbench_adjacent_controls_20261117_preflight`；`provenance/longbench_adjacent_controls_20261117`保存证据，不上传长文本原文和权重。

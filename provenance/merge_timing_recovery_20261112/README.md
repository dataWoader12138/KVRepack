# VATEX 历史时延追溯与表格修正

2026-11-12 只读核对既有运行，未运行 GPU。旧日志确实保留了 SnapKV/HybridKV + KVRepack 的 200 条模型生成端到端计时，分别为 2.100160 / 2.443860 秒。此前把新 VATEX 时延除以另一数据集图中 FullKV 的百分比已经撤回。

- `*.log.gz` 是两份原日志的逐字压缩副本；解压后 SHA-256 见 `recovery_summary.json`。Hybrid 只统计 `[2/2]` 后半段；前半原生 Hybrid 没有 Generation 打印，但有响应循环总计时。
- `historical_per_sample_stages.csv` 保存两次运行各 200 条原样解析的阶段值；sample_order 为日志顺序，不假设旧日志保存了可靠 doc_id。
- Snap 质量保留专用 summary.jsonl 原件；Hybrid 精确质量取既有参数审计中同一结果路径的记录，避免对日志四位小数再求均值。
- `historical_adapter_timing_excerpt.txt` 来自当前残存旧 adapter，用于核查计时位置；没有把它冒充 7 月冻结源码。Generation 不含预处理、H2D、文本解码，也没有显式 CUDA 同步。两阶段求和仍不是连续完整 E2E。
- 旧预算为 nominal history 10% 另加近期位置，新实验为近期位置计入严格总 10%；不能据此构造同协议加速结论。

论文表为 `docs/merge_compensation_table_with_timing.txt`，恢复原三列与方法分组，只追加 SnapKV 三种合并行。原稿分数保留；不同运行的历史时延参考标†并明确脚注。配对114条、模型阶段计时继续保留在实验报告/本目录证据中；standalone完整E2E缺失仍--。原始来源与所有数值见论文表同名 `_sources.json`。

历史响应循环计时亦已恢复：原生 HybridKV **26:28 / 200 ≈ 7.94s/条**，HybridKV+KVRepack **26:37 / 200 ≈ 7.99s/条**，SnapKV+KVRepack **24:31 / 200 ≈ 7.36s/条**。这段 tqdm 计时覆盖读视频、processor、H2D、generate、文本解码、响应缓存和循环工作，不含模型加载和任务评分；有整秒显示精度限制，不能冒充本轮精确逐样本同步 E2E。末行证据见 `historical_response_loop_excerpts.txt`。旧冻结 inventory 所说原结果 JSON 缺失仍然正确，但“仅剩质量摘要/没有时延”的结论已由此次日志追溯补正。

原生 SnapKV（7B、VATEX200、64帧、nominal0.10）亦已恢复：Avg **0.331264461087091**，响应循环 **23:59 / 200 ≈ 7.20s/条**。来源为 `.log.stopped_20260706_095548` 快照首个 RUN block，补充完整日志gzip、精确summary和计时摘录；与SnapKV+KVRepack明确区分，主表只新增一行基线并沿用†脚注。

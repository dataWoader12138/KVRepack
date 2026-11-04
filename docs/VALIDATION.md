# 验证阶段

2026-11-03 完成了 CPU 可运行的结构、来源、配置、动态分派与原函数小张量验证；该阶段未调用 GPU、未编译 CUDA、未下载权重或数据、未运行外部评分。检查复现了两项行为问题，尚未修复，见 [CPU 调用关系审计](CPU_CALL_AUDIT.md)。

## 2026-11-04 GPU 子集复测

用户已授权并完成 Qwen3B 的 SnapKV-KVRepack、MadaKV-KVRepack 原实现/迁移实现对照。NextQA、VATEX 各2例、64帧，共8组配对；生成 token 完全一致，36层 prefill/final KV 的 SHA-256 一致，全部词表的生成 logits 最大绝对差为0。SnapKV 的 `(simtop, cr, alpha)` 为 `(0.7, 0.9, 2)`；MadaKV 在两个数据集均为 `(0.5, 0.7, 2)`，NextQA 此处是代表配置，并非其 `cr=0.5` 的历史最佳配置。详情见 [GPU_RETEST_20261104.md](GPU_RETEST_20261104.md)。

复测使用独立进程中的真实模型与压缩函数，复用已准备好的相同输入张量；隔离 worker 绕过 runner/adapter。因此结论限于所列输入和配置的数值迁移一致性，尚未覆盖完整 CLI、完整基准与任务评分、自定义 CUDA 内核、7B、LLaVA、HybridKV、SparseMM 或性能。runner 的 schema 1/2 来源清单兼容问题已修复，原有哈希和路径验证保留；此修复尚不等于完整入口 GPU 验收。

## 当前可复查的检查

```bash
python scripts/check_sources.py
python scripts/check_dispatch.py > /tmp/kvrepack-dispatch.json
python scripts/check_call_graph.py --dispatch-report /tmp/kvrepack-dispatch.json --output /tmp/kvrepack-call-graph.json
CUDA_VISIBLE_DEVICES='' python scripts/check_cpu_compression.py
# 仅在原工作区仍可用时：
python scripts/check_sources.py --source-root ..
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m kvrepack.run --method snapkv_kvrepack --limit 2 --dry-run
```

打包检查使用 `pip wheel --no-deps --no-build-isolation` 构建纯 Python wheel，并核对 head score、param_config、任务 YAML、无扩展名模板和 stopwords.csv 是否在包内。CUDA 构建不属于当前检查。实验入口按源码 checkout 运行，安装方式为 README 中的 editable 安装；单独一个核心 wheel 不包含整个 vendor/内核和来源目录。

最终静态检查的实际统计保存在 `provenance/validation_report.json`。

## 模型与数据准备

原机器模型/数据不复制到项目。`HF_HOME` 可指向已准备好的缓存；未设置时使用 checkout 内 `.cache/huggingface`。首次拉取与离线运行由使用者显式配置，入口不会强制启用原机器的镜像或离线模式。

- NextQA：沿用任务模板指定的数据集；视频目录由原 `nextqa` 模板和 `_task_utils/video_loader.py` 共同确定。准备 NLTK 的 averaged_perceptron_tagger、averaged_perceptron_tagger_eng、punkt、punkt_tab、wordnet、omw-1.4。原评分工具有加载时下载语料的逻辑，离线机器要预先满足这些依赖。
- VATEX：视频路径为 `HF_HOME` 下 `vatex_test.yaml` 所指定的 cache_dir。METEOR 需要 Java，CIDEr 等指标沿用原定制评测框架。
- VideoChatGPT：视频目录为 `${HF_HOME}/videochatgpt/Test_Videos`。外部 judge 的模型及参数记录在任务模板中；生成与判分分别核验。任何凭证只在子进程环境中透传，不写入计划或来源文件。本次没有发起 API 调用。

## 后续 GPU 验收顺序

1. 在新环境中验证安装流程；对需要自定义内核的方法，从源码构建 `kvrepack_kernel`，验证模块来源、导出函数和实际GPU架构。
2. 为 Qwen 的13种方法和 LLaVA 的6种基线分别运行1–2个样本；核对模型加载、prefill、decode、KV更新和结果保存。每个模型/方法组合独立进程。
3. 用相同模型、同一小批样本和完全相同的环境参数，对照旧脚本选中的原实现与 KVRepack，比较生成 token、输出文本、压缩后 KV 长度与预算。
4. 每组基线与改进方法分别核验：gate=0/关闭merge时的回退行为、启用merge后的KV布局和预算。原代码本来有差异时先定位，不通过改动公式来掩盖迁移问题。
5. 通过小样本后再运行 NextQA/VATEX 的 limit=200；最终冻结完整模型与数据标识、样本索引、参数、指标和代码哈希。
6. 数值和质量验证完成后，再测显存、时延和吞吐。首次编译、下载与评分API耗时不计入推理性能。

现有 GPU 小样本一致性不等于完整实验结果已复现或性能达标。

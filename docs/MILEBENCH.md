# MileBench 多图主实验

本后端将 `HybridKV-opensource` 的 MileBench 数据、图片生成、评分及汇总流程接入 KVRepack。Qwen2.5-VL-3B/7B 和 LLaVA-OneVision-7B 各提供六种基线（FullKV、HybridKV、SnapKV、SparseMM、MadaKV、LookM）及前四种压缩方法的 KVRepack 改进。RandomKV、AsymKV、LookM-KVRepack、threshold、OCR 和历史消融不在图片入口范围。

这是完成 CPU 整理和检查的实现。**GPU 模型生成、CUDA 正确性、任务分数与性能尚未验证**。CPU 假模型集成测试只验证数据与产物流程，其分数不属于模型实验结果。当前验证证据见 [milebench_validation.json](../provenance/milebench_validation.json)、[数据与协议报告](../provenance/milebench_data_cpu_audit.json) 和 [LLaVA 迁移说明](LLAVA_MIGRATION.md)。

## 环境与数据

沿用仓库 Python 3.12、Torch 2.6.0、Transformers 4.52.3 的约束。图片附加依赖：

```bash
python -m pip install -c requirements/constraints-observed.txt -e '.[images]'
```

配置规划不需要推理依赖；全图预检需要 Pillow；评分需要 NumPy、rouge。汇总使用标准库 CSV，不依赖 pandas。生成还需主项目依赖、FlashAttention 及所选方法所需的本仓库 CUDA 内核；本阶段没有构建或执行 CUDA 内核。使用当前激活的 Python，不强制 conda 环境、HF mirror、系统 CUDA 路径或旧机器库路径。

数据放在仓库外，通过显式路径选择：

```text
<data-root>/
  CLEVR-Change/CLEVR-Change.json
  CLEVR-Change/images/...
  DocVQA/DocVQA.json
  DocVQA/images/...
  ...
```

`image8` 保留原图片入口的八任务清单。默认 `main11` 包含 CLEVR-Change、Spot-the-Diff、MMCoQA、IEdit、ALFRED、SceneTransition、DocVQA、MultiModalQA、SlideVQA、WebQA、WikiVQA。SceneTransition 按原 runner 的图片序列协议执行。

本机只读预检覆盖11项、2200条样本、9504次图片引用及9010张去重图片，路径/非空/可读/Pillow验证均通过。这不证明其他数据副本完整，也不验证 processor 或模型；换机器后应重新预检。仓库不包含真实题目、答案、图片、模型权重或历史预测。

## 不使用 GPU 的命令

```bash
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''

python -m kvrepack.run --benchmark milebench --list-tasks
python -m kvrepack.run --benchmark milebench --list-methods

python -m kvrepack.run --benchmark milebench --method hybridkv_kvrepack \
  --model qwen2_5_vl --pretrained Qwen/Qwen2.5-VL-3B-Instruct \
  --profile main11 --task CLEVR-Change --data-root /path/to/milebench \
  --budget-ratio 0.1 --limit 2 --dry-run

python -m kvrepack.run --benchmark milebench --profile main11 \
  --task CLEVR-Change --data-root /path/to/milebench --preflight

python scripts/plan_milebench.py --data-root /path/to/milebench \
  --model llava_ov --profile main11 --limit 200

python scripts/check_milebench_sources.py
python scripts/check_llava_sources.py
```

`--dry-run` 不导入 torch/Transformers/Accelerate/Pillow，不读取数据、不联网、不创建缓存或输出。`--preflight` 是显式读取数据和图片的检查，没有模型调用或输出写入；省略 `--limit` 检查整个任务。矩阵规划脚本只将计划及可执行命令输出至 stdout，默认规划四种改进、全部任务，**不会执行命令**。

## 图片配置及核心来源

`configs/milebench/{image8,main11}.json` 显式存储图片与生成设置、各模型/方法的参数及来源标签。Qwen 使用所选原图片脚本的配方，main11 扩展任务集合；3B 配方也可显式用于7B，但不能解释为复现了历史7B最优设置。LLaVA 四改进使用当前对应 runner 的默认配方，脚本文件名中的 best 仅是历史来源名称。

- 原图片生成默认512个新 token、至少1个，greedy、单进程、实际模型 batch=1。
- Qwen 按原 worker 转 RGB→JPEG，全部图片置于文字之前并移除 `<ImageHere>`；LLaVA 按原 worker 使用 RGB PIL 图片和无 system turn 的 user prompt。
- Qwen processor 的 `min_pixels=7168`，原消息实际像素下限为200704，最大1605632，两者都显式记录。`--min-pixels` 是有意同时覆盖二者的新配置；默认保持来源行为。
- 图片所有 Qwen 基线和改进统一默认3B；使用 `--pretrained Qwen/Qwen2.5-VL-7B-Instruct` 可显式统一到7B。视频入口的原模型默认值不变。
- LLaVA 的图片尺寸由其原 processor 处理，profile中沿用的像素字段不参与运行，`pixel_settings_effective=false` 明确标识；对LLaVA传 `--min-pixels/--max-pixels` 会报错，不静默接受无效参数。
- 原截断按32768上下文与每图256个估算 token处理。实际 processor token 数另记，不用估算代替实际长度，也不悄然增加新截断策略。
- `RATIO` 默认0.1，独立于 `BUDGET_RATIO`。改变后者不会自动改变前者；方法特有的 `*_BUDGET_RATIO` 会显式联动。绝对 `BUDGET` 按原实际命令分别为512或1024，不能只根据 shell 中被覆盖的 export 判断。
- 图片只开放主实验评分模式；历史 difference/sum 消融别名不做等价映射。

核心统一从目标 `kvrepack` 加载，图片不导入旧目录。原视频命令、默认参数和19个方法组合保持其原注册范围；新增 LLaVA 四改进通过图片能力表选择。Qwen 当前 opensource 的部分 cache/forward 与目标已有核心不同，本次没有整文件替换；此版本应描述为“目标实现的 MileBench 入口”，尚不能称与全部原历史图片实验数值等价。差异说明见 [MILEBENCH_SOURCE_DIFF.md](MILEBENCH_SOURCE_DIFF.md)。

共享 `param_config.json` 中 LLaVA 的 `budget_fixed_ratio` 继续使用目标值0.1，基线与改进使用同一资源配置；当前 opensource 为0.5，score_threshold同为0.9。该差异在来源记录及每次运行provenance的配置快照中列明，未把来源0.5作为本轮实际运行值。视觉头分数字节相同。

## 生成与产物（GPU 阶段）

用户明确开启 GPU 阶段、准备匹配依赖并构建内核后，使用上述同一运行命令去掉 `--dry-run`。运行进程需要 CUDA；离线评分与数据检查不需要。每种模型/方法独立进程，禁止同一进程连续切换 monkeypatch。

默认目录为 `outputs/milebench/<model-type>/<method>/<profile>/<task>/<run-id>/`，可以显式选择仓库外的新目录；仓库内输出限 `outputs/`。拒绝覆盖非空目录，拒绝写入数据根目录。

每次运行包含解析配置、代码来源、状态/日志、精确样本清单与标注哈希、`pred.json`、`runtime.jsonl`、`scoring/` 和成功后才产生的 `generation_summary.json`。输入 token 哈希、可用图片网格、实际 token 数和可见缓存形状会记录；无法获得的统计明确标为不可用。逻辑 tensor 字节不等于 GPU allocator 占用或独立存储大小，不能据此直接声称省显存。

`--limit` 先按原标注顺序选择，再按原图片数量分组顺序生成，两个 ID 顺序均记录。图片预检先于模型构造。缺图、重复/缺失预测或模型异常会导致失败，不补零或伪造完整结果。状态中的 returncode 及 scoring 元数据区分失败、子集与完整任务。

## 离线评分与汇总

```bash
PYTHONPATH=src python -m kvrepack.benchmarks.milebench.evaluate \
  --data-root /path/to/milebench --task CLEVR-Change \
  --predictions /path/to/run/pred.json --limit 2 --output /path/to/new-scoring

PYTHONPATH=src python -m kvrepack.benchmarks.milebench.score \
  --tasks CLEVR-Change MMCoQA \
  --run CLEVR-Change=/path/to/clevr-run \
  --run MMCoQA=/path/to/mmcoqa-run --output /path/to/new-summary
```

离线评分的 `--limit` 必须与预测对应的原标注子集一致，默认是完整任务。评分严格核对 ID、真实答案及完整覆盖。保留原开放式 Rouge-L F、多选匹配 Accuracy、MMCoQA 答案子串命中规则；原 `Eval` 的字符串清洗和选择逻辑未顺便改算法。

`eval.json` 是0—1值，CSV只转换一次为百分制。汇总显式列出预期任务、缺失任务、子集任务及样本数未证实任务；已完成任务均值与完整数据集均值分别命名。缺任务不补零，`--limit 2` 不会被标为完整200条主实验。不会自动选最新结果或混入其他配置的旧运行。

## CPU 验证与 GPU 验证边界

CPU 检查包含数据/评分的原函数固定输入差分、假模型端到端产物、图片预处理与参数传递、全部候选配置、LLaVA四改进来源变换回放和小张量压缩检查，以及 SparseMM 跨 prompt 预算回归。完整命令、环境、通过/跳过范围在 [验证报告](../provenance/milebench_validation.json)。

后续 GPU 阶段先固定模型版本、样本ID、像素、dtype及生成参数，对照输入token、位置、各层KV预算与多步decode；逐方法1—2条样本通过后扩展子集，再做11项完整任务，最后测显存与速度。本轮 CPU 结果不替代这些验收。

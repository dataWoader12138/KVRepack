# HybridKV-KVRepack：NExTQA 前200条 Random-head 消融

2026-11-06 状态：**正式200条、四条件共800次生成及离线评分已完成**，2026-11-07独立全量执行与评分复核均通过，见[实际结果与图表](HYBRID_RANDOM_HEAD_NEXTQA_200_RESULTS_20261106.md)。相似度策略WUPS为33.219903，Random三个种子均值为33.088063±0.272940（样本标准差）；差值为+0.131841，184源视频配对聚类95%区间[−0.778808,1.353447]，未显示稳定优势，也不构成等价证明。正式worker墙钟约47分27秒，全部800次生成均EOS、无上限命中。本轮仅包含 NExTQA 原始测试集前200行，不扩大到9178行，不包含VATEX。

用户已停止此前RoPE合并空间实验；该旧矩阵不自动续跑。本文是新的HybridKV Random-head实验协议，原SnapKV Random-head结果保留在[单独文档](RANDOM_HEAD_NEXTQA_200.md)，不与本轮预测拼接。

## 实验条件

| 项目 | 固定设置 |
|---|---|
| 模型 | Qwen/Qwen2.5-VL-7B-Instruct；本机snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| 方法 | `hybridkv_kvrepack`，使用项目保留的唯一HybridKV-KVRepack实现 |
| 数据 | NExTQA OE `test`，原始 `doc_id=0..199`；200问题、184源视频 |
| 输入 | 64帧；同一原始行只准备一次输入，四策略共享相同CPU/GPU张量 |
| 预算 | `budget_ratio=0.1`；原生分类及每头保留预算由原实现计算 |
| 原生KV头 | 28层×4头，共112头；四策略均选56头 |
| 合并参数 | `simhead_merge_ratio=0.5`，`compress_ratio=0.5`，`K_DIFF_ALPHA=2` |
| 分类及预算分配 | 原生 `head_ada`；不因Random选择重分配预算 |
| K与V合并 | 保留原 `raw_reapply` K路径、原相邻分组及V聚合；原始raw K相似度选头 |
| 生成 | greedy，`do_sample=false`，`use_cache=true`，`max_new_tokens=128`，`repetition_penalty=1.05` |
| 数值及模型随机种子 | BF16，FlashAttention2，model seed42；各策略生成前重置相同模型随机状态 |

四种策略标签为 `similarity`、`random_global_seed0`、`random_global_seed1`、`random_global_seed2`。`similarity`原样使用原实现全局相似度排名得到的56头。每个Random seed在完整112个原生KV头上生成一次均匀无放回排名，取前56头，并将该模型级集合固定用于全部200个输入；不逐输入重抽，不先限制每层配额，也不加入新的可合并头筛选。Random选头使用独立局部RNG，不消耗全局模型RNG。

本轮不是每层头数匹配实验。不同策略可以改变选中的头身份、层间分布及静态/动态头组成，但每个头原有的类别、保留预算、候选构造、压缩参数和实际merger算法保持原样。

## 配对和有效执行门禁

每个输入均核验四策略的实际输入指纹、原Arrow行及视频身份、模型与生成配置、原相似度mask、原分类、全部每头预算及合并前候选指纹。配置中仅 `strategy` 和 `mask_seed` 允许不同。

Hybrid原生mask形状为 `[layer=28,batch=1,KVhead=4]`。不仅要求选中56头，还要求56个不同的选中头实际执行 `after < before` 的合并，且实际移除量等于所有真实调用的 `sum(before-after)`。若某条件不足56次有效合并则保留失败诊断，不把“选中56”当作“有效合并56”，也不重抽或改变参数使其过门禁。

原生 `allocate_budget_by_ratio` 结束时会将 `head_category` 置为 `None`。观察适配器在释放前保存 `calibrated_head_category={values,tensor}`；四策略的完整 `pre_selection` 快照必须相同，其中包含这份分类副本、稀疏每头预算、检索/裁剪映射及候选指纹。门禁使用实际分类副本，不要求已释放字段重新非空。

**不要求四策略的移除token总量相同。** Hybrid每头预算异质，选择不同的头会改变候选扩展及实际移除总量，这是头身份干预的后果，必须保留并报告。适配器不为对齐总工作量改变原分类、每头预算、候选或原合并代码；也不把不同head集合的分组指纹硬判为应相同。

## 流式执行与复现入口

从项目根目录运行，使用已安装本项目依赖和Hybrid CUDA内核的环境。模型、原视频、数据集Arrow缓存及内核按[项目说明](../README.md)准备；以下路径替换为本机路径，输出使用新目录。

```bash
python scripts/run_hybrid_random_head_nextqa_200.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct \
  --hf-home /path/to/huggingface_cache \
  --output outputs/ablations/hybrid_random_head_nextqa_200_REPRO
```

[控制器](../scripts/run_hybrid_random_head_nextqa_200.py)固定传入 `--limit 200`，先执行[GPU worker](../src/kvrepack/ablations/run_hybrid_head_selection.py)，生成全部完成后唯一一次调用[离线报告模块](../src/kvrepack/ablations/hybrid_head_selection_report.py)。四策略由同一个已加载模型服务；每输入完成并通过审计后，才原子提交一个 `samples/000000.json` 形式的四条件bundle，不持久保存大型输入张量缓存。

精确续跑使用同一命令加 `--resume`，仅在原worker已退出后执行。续跑要求冻结契约、源码SHA、模型和数据来源一致，已提交bundle必须构成连续完整原始前缀；不会覆盖已有样本。控制器状态与日志位于输出目录旁的 `_controller_state.json`、`_worker.log`、`_scoring.log`；worker进度在输出内 `run_state.json`。

仅当完整生成已结束且尚不存在报告时，可单独执行CPU评分：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m kvrepack.ablations.hybrid_head_selection_report \
  --run-root outputs/ablations/hybrid_random_head_nextqa_200_REPRO \
  --hf-home /path/to/huggingface_cache \
  --output outputs/ablations/hybrid_random_head_nextqa_200_REPRO/report \
  --limit 200
```

该命令拒绝不完整状态及已有报告，使用本地NLTK/WordNet资源，不下载、不调用模型或GPU。`--partial`仅打印进度，不评分。worker和报告还支持独立新目录的 `--limit 1` smoke；控制器不自动扩展到其他数据集或更多样本。

## 评分、统计和产物

WUPS使用项目原NExTQA任务源码函数AST，保留原问题类型、备选参考答案及停用词/词形处理规则，主值为原始任务返回的WUPS@0，单位0–100。精确预测与其原始 `answer/type/additional_ref_answer` 作为缓存键；每策略逐问题分数均值须复现原任务聚合结果。

报告保留三个seed各自分数及相对原策略差值，另给Random三seed均值和样本标准差（`ddof=1`）。配对差值统一为 `similarity − random`，负值保留。95%区间使用固定seed20261104、5000次源视频聚类bootstrap：共同重采样184个源视频并保留各视频所有问题，问题等权；四条件共用抽样，Random均值区间条件于三个固定seed。不能将三个seed×200问题或112个头当作新增独立输入，也不能混淆seed标准差与输入聚类置信区间。

主要产物：

- `run_contract.json`、`run_config.json`、`run_state.json`：模型、数据、生成配置、核心/适配器/报告源码及内核SHA、完整计数与运行状态。
- `samples/{doc_id:06d}.json`：四策略预测token和文本、输入/来源指纹、逐次真实merge审计、完整每头预算及配对门禁。
- `report/scores.csv`、`paired_scores.csv`、`run_scores.csv`：逐问题分数、逐seed配对差值和任务分数/区间。
- `report/selector_audit.csv`、`merge_work.csv`：56头及56有效合并核验、mask差异、各策略实际移除量和相对原策略的工作量差异。
- `report/generation_summary.csv`：EOS、达到生成上限、未EOS截断计数和生成长度；恰好在上限输出EOS与未EOS截断分别记录。
- `report/sample_files.csv`、`summary.json`、`RESULTS.md`、`nextqa_scores.{png,pdf,svg}`：完整来源、统计及图表。

已完成的正式本机输出为 `outputs/ablations/hybrid_random_head_nextqa_200_20261106`，其中`run_state.json`和控制器状态均为`complete`。运行内配对门禁与报告记录显示200条均覆盖四条件、每条件均56头及56次有效合并；实际移除token总量随策略变化，详见[结果页](HYBRID_RANDOM_HEAD_NEXTQA_200_RESULTS_20261106.md)。正式结果与smoke、旧SnapKV及已停止的RoPE实验分别保存，不从本前缀实验直接推断所有任务/模型上的普遍优势。

## 已完成的验证边界

正式启动前CPU suite为224项：223通过，1项已有跳过；其中包含5项本报告测试，另有Hybrid适配器检查。CPU测试覆盖异质预算导致不同移除量的合法情况、真实mask布局和56有效合并门禁、负差值、来源/覆盖和篡改拒绝，不能替代完整GPU效果评估。

独立smoke目录为 `outputs/ablations/hybrid_random_head_nextqa_200_20261106_smoke`。四策略各1次真实生成已完成，每条件均56选头、56有效合并；同一份每头预算范围512–1313，实际移除量依次为169598、158923、172939、141273，按协议保留差异。四输出均为`no`，该问题原WUPS均100；这是单输入连通性检查，不是策略效果证据，只有1个源视频时不生成bootstrap CI。

该smoke报告已唯一一次真实CPU离线评分完成，计分7.381秒；CSV/JSON分数、预算工作量及EOS统计一致，12项表格/报告/PNG/PDF/SVG产物存在且PNG目检正常，冻结源SHA匹配。正式报告随后由控制器在800次生成完成后产生，正式离线计分10.771秒，未与smoke合并。2026-11-07的[独立全量执行审计](../provenance/hybrid_random_head_nextqa_200_20261106/independent_execution_check.json)通过：800条生成记录、600组Random/原策略配对、固定随机集合及冻结来源SHA核对均无问题；原策略与旧Hybrid raw基线200条逐输入/token/text/native mask/分组完全一致。该执行审计未重新运行GPU或评分。另有[独立评分复核](../provenance/hybrid_random_head_nextqa_200_20261106/independent_scoring_check.json)通过，800次原生WUPS未复用预测缓存，逐条分数、三seed样本SD及视频聚类CI均与正式报告一致。

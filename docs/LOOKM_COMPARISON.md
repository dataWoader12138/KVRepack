# LookM：两个视频任务的10%名义预算测试

2026-11-09最新指示：**“你不用重跑HybridKV+KVRepack，把LookM结果给我。”** 已停止原双方法控制器及Hybrid worker，原矩阵状态为`stopped_by_user`、`resume_authorized=false`。当前只补齐LookM：NExTQA原始前200条已完整生成并完成CPU评分，WUPS=33.857144；VATEX原始前200条由独立LookM worker运行中，尚未完成。Hybrid部分记录与此前RoPE实验均保留，不自动续跑。

本轮使用Qwen2.5-VL-7B-Instruct和原仓库LookM，在NExTQA OE/test及VATEX/test各运行原始前200行，共400次LookM生成；复用本轮已经完成的LookM/NExTQA预测，不再生成一遍，不拼接旧实验预测。该设置沿用此前明确约定的“全量指200条”，不扩大到NExTQA 9178条或VATEX 4478条。最初的LookM/Hybrid四格800次计划已被用户收窄，本页保留其预算来源说明供追溯，不继续执行双方法矩阵。

| 项目 | 设置 |
|---|---|
| 模型 | Qwen2.5-VL-7B-Instruct；固定本机snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| 当前方法 | 仅`lookm`，按任务使用独立进程加载原生模型类 |
| 输入 | 各任务原始行0..199；64帧，相同任务提示词、processor与像素范围 |
| 预算 | 原方法的 `budget_ratio=0.1`；记录实际prefill缓存占用 |
| LookM | 原text优先打分、32近期位置窗口、GQA mean、merge layer ratio .05 |
| 生成 | BF16、FlashAttention2、greedy、seed42、repetition penalty1.05；NExTQA最多128/VATEX最多64个新token |
| 评分 | 原NExTQA WUPS@0（0–100）；原VATEX corpus BLEU4/METEOR/ROUGE-L/CIDEr，不对四指标混合平均 |

## 原代码与预算口径

LookM 基线已存在于项目：`monkeypatch.replace_qwen('lookm')` → `qwen_model.qwen_flash_attn2_forward_LookM` → `baseline_utils.init_lookm` / `LookMCluster.update_kv` / `merge_kv`。原始参考入口来自 `HybridKV-artifact/scripts/eval/qwen_ratio.sh` 的已保存快照，并非把 `lookm_kvrepack` 当作LookM基线。

2026-11-09的只读来源审计核对了7个关键符号：`merge_kv`、`LookMCluster`、`init_lookm`、`qwen_flash_attn2_forward_LookM`，以及三个视觉/文本mask转发函数；其与原工作区对应定义的AST均一致。整文件因打包与导入变换而非字节相同。这证明本次所检查的迁移来源，不等于官方论文在所有设置下的精确复现。保留的`qwen_ratio.sh`快照能证明存在7B/10%/64帧/200条入口；当前定向查找未找到可读的已完成LookM7B日志，不能由此断言历史实验未运行。

下面的`N`均指完整prefill序列长度，包含展开视觉token、文本和特殊token，不是只统计视觉token。28层×4个原生KV头给出112个层/头槽位；一个位置对应的K/V对在位置计数中只计一次。Hybrid一列仅为已停止原比较协议的预算说明，不表示当前仍在运行该方法；其原设置为simhead ratio .5、compress ratio .5、alpha2、原分类与分配、raw相似度及`raw_reapply`路径。

| 项目 | 当前LookM | 原拟比较的HybridKV + KVRepack（已停止） |
|---|---|---|
| 名义历史预算 | 每头`k=floor(0.1*N)` | 全模型目标`112*a`，其中`a=floor(0.1*N)` |
| 实际历史分配 | 每头选`k`个历史位置 | 原`head_ada`分类及预算分配得到每头`b[l,h]`，保留取整、裁剪与检索8槽对齐 |
| 近期位置数 | 每头32 | 每头`local=32+(N%8)`，范围32–39 |
| 全层活跃prefill位置数 | `112*(k+32)` | `sum(b[l,h])+112*local` |
| 活跃位置占完整缓存比例 | `(k+32)/N` | `(sum(b[l,h])+112*local)/(112*N)` |

公式描述本轮长视频输入的prefill活跃缓存。Hybrid的`112*a`是分配目标，整数余数、头级容量裁剪及检索对齐可改变最终总和，因此不能直接把`a`当成每头实际长度。实际数值由执行记录测量，不能仅从名义比例推断。

LookM每次prefill重建prompt相关cluster，并以当前`N`覆盖历史选择数量。构造器中的2048或环境`BUDGET=1024`不决定最终缓存长度；`BUDGET`仍为初始化所需兼容参数。文本分数获得优先加分后进入同一个top-k，没有另加无限保留的文本区。7B的第0/1层仅裁剪，第2–27层执行原pivot merge；保留32个近期位置不表示这些位置的K/V值原样不变，因为被丢弃历史可能合并到它们。

原`merge_kv`通过跨head展平的索引成员集合计算drop，而非分别对每头计算成员关系；这一行为在原来源与迁移实现中均存在，本轮保留。pivot实现的`head_dim=128`假设与目标7B相符；不在对照中更换算法或暗中修复这些边界。

两方法的窗口、候选构造及跨头分配不同；**相同10%参数不意味着每层、每头和总字节严格相等**，加入近期窗口后也并非恰好10%的总活跃缓存。实验记录实际prefill K/V槽位数、占完整缓存比例、每层形状/dtype及生成后缓存形状；目标模型BF16、head_dim128时，每个计入的K/V位置对对应`128*2*2=512`字节的逻辑payload。Hybrid还保留模型设备上的检索历史/backing张量与索引等元数据，活跃attention缓存的payload不能当作全部K/V存储或总GPU显存；GPU峰值另有记录。若需严格等字节预算，应另行定义匹配方案，不能把本轮原生设置称作已完成等字节控制。

## 仅运行LookM的可复现入口

[单方法启动脚本](../scripts/run_lookm_video.py)只运行LookM，不启动Hybrid。新机器安装依赖、准备模型和数据后，在仓库根目录使用新输出目录：

```bash
python scripts/run_lookm_video.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct \
  --hf-home /path/to/huggingface_cache \
  --output outputs/benchmarks/lookm_only_REPRO
```

默认两个数据集各200条，可用`--tasks nextqa_oe_test`或`--tasks vatex_test`只选一个。完成单任务生成后自动调用其CPU评分。`--resume`只用于该单方法入口自己的连续前缀，完成的任务跳过重新生成；原已停止双方法控制器仍不得恢复。

已完成的NExTQA分数及CSV见[结果页](LOOKM_RESULTS_20261109.md)。

## 当前运行与单方法评分

当前服务器只继续已启动的LookM/VATEX进程，已完成的LookM/NExTQA无需重新生成。两个任务分别使用`kvrepack.benchmarks.run_video_comparison --method lookm`，使用原Transformers模型及其LookM patch、新KV缓存和prompt状态。模型、原视频/Arrow缓存、NLTK/WordNet及VATEX Java评分资源按项目说明准备。

任务达到200个完整样本后，使用独立单方法CPU报告入口`kvrepack.benchmarks.single_video_report`；该入口不要求Hybrid记录或原四格矩阵完成。以下`CASE`指向一个已完成任务的目录，已有报告时遵循报告工具的覆盖限制，不为评分重跑生成：

```bash
export CASE=/path/to/completed/lookm/nextqa_oe_test
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m kvrepack.benchmarks.single_video_report \
  --run-root "$CASE" \
  --hf-home /path/to/huggingface_cache \
  --output "$CASE/report"
```

每个任务保留独立的`run_contract.json`、`run_config.json`、`run_state.json`及`samples/000000.json`等原子样本。当前LookM-only总状态为本轮根目录的`lookm_only_state.json`，各任务的准确进度以自己的`run_state.json`为准；原`experiment_state.json`描述的是已停止双方法矩阵，不能用它判断LookM-only是否完成。

原双方法脚本`scripts/run_lookm_vs_kvrepack.py`、静态`matrix.json`及双方法报告模块保留作历史实现；**本轮不再启动该控制器，也不使用其`--resume`恢复Hybrid**。若后续需要恢复任何已停止方法，须有用户的新指示。完整LookM样本不因原矩阵停止而删除或重复生成，Hybrid的119条部分样本不计入LookM结果。

本轮先报告LookM自身的原任务分数：NExTQA WUPS@0和VATEX corpus四指标，不发布未完成双方法矩阵的差值或配对区间，不把CIDEr单句贡献错误当作可独立求和的区间。记录EOS、达到长度上限及无EOS截断，不把尚未完成的VATEX前缀当成正式200条结果。

## 本轮进度

输出根目录继续保留为`outputs/benchmarks/lookm_vs_kvrepack_20261109/`，目录名不表示仍在运行双方法对照。原控制器于UTC01:17:42启动，在用户收窄范围后于**01:41:49停止**；原PID/PGID4690及Hybrid worker已终止，状态`stopped_by_user`、`resume_authorized=false`。

LookM/NExTQA的200条于UTC01:31:19生成完成，保存在`lookm/nextqa_oe_test/`，原WUPS@0为33.8571440351094（0–100）；200次全部EOS、无截断，独立逐条复算与正式报告完全一致，平均活动缓存比例10.298789%。Hybrid/NExTQA留下119个完整样本，位于独立方法目录，停止且不续跑；不计为完成200条，也不混入LookM。

新的LookM-only进程于UTC**01:42:55**启动，PID/PGID36304，只运行`lookm/vatex_test/`的200条。`lookm_only_state.json`记录该独立任务；VATEX仍在运行，实际剩余时间及计时以worker状态为准。结束后再补充LookM正式分数、实际计时与验证范围，不预判VATEX得分。

启动前验证已写入同级`lookm_vs_kvrepack_20261109_preflight/summary.json`：

- CPU suite共235项，234通过、1跳过。
- 独立`lookm_vs_kvrepack_20261109_smoke/`完成两方法×两任务各1条，共4次真实生成，原NExTQA WUPS及VATEX四指标的离线评分链路均完成。
- 两方法在各任务上的首条实际输入张量哈希配对一致；Hybrid两任务首条的输入和生成token序列分别与旧raw路径精确一致。
- 控制器及benchmark入口4份源码SHA已记录；只读预算/来源审计保存在同级`lookm_vs_kvrepack_20261109_source_audit.json`。

这些是原矩阵启动前的连通性与迁移检查，不是当前仍应继续执行Hybrid的授权。smoke的单条分数不用于推断正式优劣，也不把1条VATEX参考语料的CIDEr与200条语料混比。当前只等待LookM/VATEX完整生成与评分；原四格800次计划不再补齐，RoPE等历史停止任务同样不恢复。

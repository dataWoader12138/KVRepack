# LookM 与 KVRepack：两个视频任务的10%预算对照

2026-11-09：**正式四格矩阵已启动，尚未完成800次生成与正式评分。** Qwen2.5-VL-7B、两任务各200条的总运行预计约60–90分钟起，实际进度和计时以控制器及worker状态文件为准。启动前235项CPU检查中234通过、1跳过；独立四格GPU smoke及原任务离线评分链路已完成，不能将这些检查当成正式200条结果。此前 RoPE、随机抽样等已停止实验保持停止。

首轮采用 Qwen2.5-VL-7B-Instruct，NExTQA OE/test 与 VATEX/test 的原始前200行，各自比较原仓库 LookM 与 HybridKV + KVRepack，共800次重新生成；不拼接旧实验预测。该设置沿用此前明确约定的“全量指200条”，不扩大到NExTQA 9178条或VATEX 4478条。当前按已说明的默认主方法`hybridkv_kvrepack`执行；如用户进一步指定方法，以其最新回复为准。

| 项目 | 设置 |
|---|---|
| 模型 | Qwen2.5-VL-7B-Instruct；固定本机snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| 方法 | `lookm` 与 `hybridkv_kvrepack`，分别在独立进程加载原生模型类 |
| 输入 | 各任务原始行0..199；64帧，相同任务提示词、processor与像素范围 |
| 预算 | 原方法的 `budget_ratio=0.1`；记录实际prefill缓存占用 |
| LookM | 原text优先打分、32近期位置窗口、GQA mean、merge layer ratio .05 |
| HybridKV + KVRepack | simhead ratio .5、compress ratio .5、alpha2、原分类与分配、raw相似度、raw_reapply K路径 |
| 生成 | BF16、FlashAttention2、greedy、seed42、repetition penalty1.05；NExTQA最多128/VATEX最多64个新token |
| 评分 | 原NExTQA WUPS@0（0–100）；原VATEX corpus BLEU4/METEOR/ROUGE-L/CIDEr，不对四指标混合平均 |

## 原代码与预算口径

LookM 基线已存在于项目：`monkeypatch.replace_qwen('lookm')` → `qwen_model.qwen_flash_attn2_forward_LookM` → `baseline_utils.init_lookm` / `LookMCluster.update_kv` / `merge_kv`。原始参考入口来自 `HybridKV-artifact/scripts/eval/qwen_ratio.sh` 的已保存快照，并非把 `lookm_kvrepack` 当作LookM基线。

2026-11-09的只读来源审计核对了7个关键符号：`merge_kv`、`LookMCluster`、`init_lookm`、`qwen_flash_attn2_forward_LookM`，以及三个视觉/文本mask转发函数；其与原工作区对应定义的AST均一致。整文件因打包与导入变换而非字节相同。这证明本次所检查的迁移来源，不等于官方论文在所有设置下的精确复现。保留的`qwen_ratio.sh`快照能证明存在7B/10%/64帧/200条入口；当前定向查找未找到可读的已完成LookM7B日志，不能由此断言历史实验未运行。

下面的`N`均指完整prefill序列长度，包含展开视觉token、文本和特殊token，不是只统计视觉token。28层×4个原生KV头给出112个层/头槽位；一个位置对应的K/V对在位置计数中只计一次。

| 项目 | LookM | HybridKV + KVRepack |
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

## 启动和断点

安装仓库依赖及Hybrid CUDA内核，准备模型、本地视频/原Arrow缓存、NLTK/WordNet和VATEX Java评分资源后，从仓库根目录运行：

```bash
python scripts/run_lookm_vs_kvrepack.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct \
  --hf-home /path/to/huggingface_cache \
  --output outputs/benchmarks/lookm_vs_kvrepack_REPRO
```

`--limit 1` 用于独立目录的四格连通性检查，默认 `--limit 200`。显式续跑使用同一命令加 `--resume`，要求原进程退出、相同输入/源码契约和完整连续前缀。控制器不会覆盖旧样本，不会恢复历史已停止任务。

控制器逐任务、逐方法串行执行新worker `kvrepack.benchmarks.run_video_comparison`，每格均为全新进程并使用各自原生模型类；LookM使用原Transformers模型及其patch，Hybrid使用原定制类，不在Hybrid类上套用LookM分派。同一worker每输入采用新KV缓存并重置prompt相关状态。这样也避免LookM配置字段在改变方法或比例时沿用旧值。全部生成完成后调用 `kvrepack.benchmarks.video_comparison_report`。每个方法/任务有独立的 `run_contract.json`、`run_config.json`、`run_state.json` 与 `samples/000000.json` 等原子样本；总状态在 `experiment_state.json`，静态协议在 `matrix.json`。输入分别按相同规则准备，评分前逐条核验两方法的实际张量哈希及原行/视频身份；不持久保存大型输入张量。

主差值统一为 **HybridKV + KVRepack − LookM**，正负均保留。NExTQA按源视频聚类配对bootstrap，VATEX保留原corpus四指标；不把CIDEr单句贡献错误当作可独立求和的区间。记录EOS、达到长度上限及无EOS截断，不把尚未完成的前缀当成正式200条结果。

## 本轮进度

正式已于2026-11-09 UTC **01:17:42**启动，输出目录为`outputs/benchmarks/lookm_vs_kvrepack_20261109/`。控制器PID/PGID为4690，初始worker为4691；记录状态为`generating`，首格是LookM/NExTQA。后续格由控制器顺序启动，不应重复发起相同矩阵。源码已冻结，生成中不修改`src/`、`scripts/`或`tests/`。已告知正式耗时估计约60–90分钟起，结束后再补充实际计时、表格、图与验证范围。

启动前验证已写入同级`lookm_vs_kvrepack_20261109_preflight/summary.json`：

- CPU suite共235项，234通过、1跳过。
- 独立`lookm_vs_kvrepack_20261109_smoke/`完成两方法×两任务各1条，共4次真实生成，原NExTQA WUPS及VATEX四指标的离线评分链路均完成。
- 两方法在各任务上的首条实际输入张量哈希配对一致；Hybrid两任务首条的输入和生成token序列分别与旧raw路径精确一致。
- 控制器及benchmark入口4份源码SHA已记录；只读预算/来源审计保存在同级`lookm_vs_kvrepack_20261109_source_audit.json`。

这些是启动前的连通性与迁移检查。smoke的单条分数不用于推断正式优劣，也不把1条VATEX参考语料的CIDEr与200条语料混比。正式800次生成、最终输入配对门禁、评分及统计尚待完成；现阶段不发布预期得分。

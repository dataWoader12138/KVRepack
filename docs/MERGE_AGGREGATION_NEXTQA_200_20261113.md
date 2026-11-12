# NExTQA 前200条：合并规则与完整端到端时延

用户已明确纠正数据集为 **NExTQA**。此前误启的 VATEX 补跑已停止，原始记录保留，不计入此实验，不恢复旧停止任务。

## 实验矩阵

Qwen2.5-VL-7B-Instruct，NExTQA 开放式测试集原始 `doc_id=0..199`。

| 骨干 | 原版 | + KVRepack | + LOOK-M-style | + AsymKV-style |
|---|---:|---:|---:|---:|
| SnapKV | 200 | 200 | 200 | 200 |
| HybridKV | 200 | 200 | 200 | 200 |

**共8组、1600次正式生成；每组另预热2条，共16次预热。** 小批联调独立目录，8个输出及其预热不并入正式结果。优先运行四组style，再补两骨干的原版/KVRepack基线。所有结果均新生成，不将VATEX或旧不同预算的NExTQA记录迁入。

历史NExTQA原版和KVRepack的200条分数与loop/generation计时确实存在，但预算为nominal历史10%加recent，或计时混入诊断，不能当作本轮严格总10%的完整E2E。新style结果也没有历史200条。因此补齐八条件同口径的质量与时延。

## 固定条件

- 模型snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5`；A800-SXM4-80GB、BF16、FlashAttention2、batch1、seed42。
- 原生NExTQA prompt和64帧采样；greedy、`max_new_tokens=128`、repetition penalty 1.05。保留实际输出长度、EOS/截断与显存峰值。
- SnapKV：gate=.30、compress=.70、alpha=2；HybridKV：gate=.50、compress=.50、alpha=2。均沿用 raw K 合并后按anchor重施mRoPE，插入点与KVRepack一致。
- 总活跃prefill缓存上限为 `floor(.1*N)`，**recent/local位置包含在10%之内**。Snap原版仅适配历史容量表达式；Hybrid在原head预算分配后采用共用的全局缩放/chunk向下取整。保留原生剪枝实现，不将merge ratio置零冒充原版。
- 三种merge共用候选、head选择、groups和anchor，只替换聚合规则。LOOK-M-style移植TextPivot聚合，不引入其pruning或全局路由；AsymKV-style移植Fisher加权K、V求和和源码LMA，Fisher不读取答案。
- 每骨干的style两臂以及原版/KVRepack两臂分别共享一个加载模型，逐doc轮换先后顺序，独立读取处理输入。跨进程逐样本核对实际输入指纹、任务/行/视频元数据、三种merge的真实选头/分组/预算以及共享算法源码。Hybrid原版实际token数因异质预算可能不同，逐行报告，不伪称完全相同。

## 计时和评分

完整E2E从视频读取/解码/processor之前开始，包含H2D、Fisher（AsymKV适用）、视觉编码/prefill/decode和文本解码；GPU计时边界显式同步。模型加载、预热、源码编译、审计哈希、写盘及评分在计时之外。另保存模型E2E和每阶段时延，不把generation_seconds当成完整E2E。

质量采用原生NExTQA WUPS，离线用相同Arrow原始前200条独立评分八组，评分前拒绝样本不全、任务错误、输入/算法来源变化和超预算。完整E2E比较同时受输出长度和预处理影响，不能直接称为纯merge算子开销。结果只代表前200条，不代表完整测试集。

## 运行与输出

```bash
python scripts/run_nextqa_merge_comparison.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct/snapshot \
  --hf-home /path/to/huggingface_cache \
  --output outputs/ablations/merge_nextqa_200_20261113 \
  --limit 200 --warmup 2
```

新style worker为`kvrepack.ablations.run_style_comparison`，原版/KVRepack worker为`run_baseline_completion --task nextqa_oe_test`，二者共用`video_e2e_inputs.prepare_input`无offset/二次tokenization原生输入流程。原VATEX style worker保持冻结。三十份核心算法文件不修改。

四个GPU进程串行运行，根`experiment_state.json`和`{style,baseline}/{snapkv,hybridkv}/run_state.json`记录实际进度。全部生成后控制器自动调用`nextqa_merge_report`，保存WUPS、逐样本完整E2E/模型E2E及预算比较。正式目录为`outputs/ablations/merge_nextqa_200_20261113/`；联调为`outputs/ablations/merge_nextqa_20261113_preflight/pilot1/`；来源与检查证据为`provenance/merge_nextqa_200_20261113/`。当前是否启动/完成以状态文件为准，不将联调称为200条结果。

正式任务已启动，控制器PID/PG34513，源版本`baa323f`；实际UTC启动记录见`provenance/merge_nextqa_200_20261113/formal_launch.json`。先运行SnapKV的LOOK-M/AsymKV两臂各200条，后续三case及最终评分由同一控制器自动衔接；当前是运行中，尚未完成200条结果。

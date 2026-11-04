# Qwen3B 视频复测与后台进度（2026-11-04）

状态：**running**；阶段快照 UTC：2026-11-04T05:20:32.294430+00:00。

本页仅覆盖下表计划。pending/running 均未通过；200 条是子集，完整 NextQA、VATEX 分别为9178、4478条。阶段变化时更新，逐样本实时计数请查看本机 matrix_state.json。

| 方法 | 任务 | 样本数 | 范围 | 状态 | 实际评分 |
|---|---|---:|---|---|---|
| snapkv_kvrepack | nextqa_oe_test | 200 | 前200条子集 | running | — |
| snapkv_kvrepack | vatex_test | 200 | 前200条子集 | pending | — |
| madakv_kvrepack | nextqa_oe_test | 200 | 前200条子集 | pending | — |
| madakv_kvrepack | vatex_test | 200 | 前200条子集 | pending | — |
| snapkv_kvrepack | nextqa_oe_test | 9178 | 完整split | pending | — |
| snapkv_kvrepack | vatex_test | 4478 | 完整split | pending | — |

配对两端使用同一当前 CLI/adapter、模型快照、BF16、64帧、batch=1、greedy 和 budget_ratio=0.1；原版端仅替换为原压缩源码。Snap参数为simtop=0.7/cr=0.9/alpha=2；Mada为simtop=0.5/alpha=2，NextQA使用历史最佳cr=0.5，VATEX用cr=0.7；correction=none。

通过阶段核对实际输入tensor、生成token/text、每步完整词表logits及36层prefill/final KV的SHA-256。此轮是哈希相等检查，未测max_abs_error；追踪开销使其不能作为性能测试。NextQA为任务原WUPS（0–100），VATEX为真实BLEU-4/METEOR/ROUGE-L/CIDEr及四项原始小数均值mean4；CIDEr随参考子集变化，子集和全量不能混比。

核心相对冻结基准`169aa1f`未变化：**True**。本报告生成器只维护本页与专属provenance目录。

## 普通 CLI 与历史上下文

- cli_nextqa_snapkv_smoke20：passed，有效样本20，WUPS=28.003499816653978。仅candidate普通CLI结果，没有配对原版E2E相等证明。
- cli_nextqa_madakv_200：running，有效样本None，WUPS=None。仅candidate普通CLI结果，没有配对原版E2E相等证明。
- 历史200条 snapkv_kvrepack WUPS=30.490725191798408，cr=0.9；仅作旧汇总分上下文，即使同分也不能证明历史逐样本一致。
- 历史200条 madakv_kvrepack WUPS=30.598461646042235，cr=0.5；仅作旧汇总分上下文，即使同分也不能证明历史逐样本一致。

普通Snap20正式CLI的已完成WUPS为28.003499816653978；它不是200条或全量结果。普通Mada200只在其run_state成功且结果有效计数为200时标passed。

## 证据与边界

[阶段汇总](../provenance/full_video_retest_20261104/summary.json)、[冻结资源摘要](../provenance/full_video_retest_20261104/frozen_assets_summary.json)及同目录completed/、helpers/保存完成对比、真实评分和运行助手。权重、视频、完整trace JSONL、logits均不上传，仅保留本机路径、SHA和快照大小。

冻结ledger记录9877个文件；冻结完成后摘要列出两个模型权重和两个Arrow SHA，并保留完整ledger的SHA。资源初始字节哈希后按stat/软链接目标监测；视频仅存在/非零/stat检查，不是视频内容哈希、完整解码验证或全部依赖闭包。

## 复查与恢复

在已准备同版本模型、数据和依赖的工作目录运行；helpers快照保留本次具体逻辑，本机路径需按新机器调整。

```bash
cd /root/autodl-tmp/HybridKV-artifact/kvrepack-full-work
cat matrix_state.json
cat matrix_report.md
# 仅在没有现存driver进程时恢复；复用已有trace并重新验证冻结资源
python -B -u run_matrix.py --plan matrix_plan.json
# 复查完整NextQA；VATEX改expected-count为4478并换文件路径
python compare_full_runs.py --reference snapkv_nextqa_original/trace.jsonl --candidate snapkv_nextqa_repack/trace.jsonl --expected-count 9178 --output check_nextqa_full.json
# 复查前200条须额外传 --limit 200，完整expected-count仍为9178
python score_full_runs.py --records snapkv_nextqa_original/trace.jsonl --task nextqa_oe_test --expected-count 9178 --output check_nextqa_metrics.json --repo-root ../KVRepack
# 后台汇总；--publish为显式可选的Git提交/推送授权开关
python watch_publish.py --watch --interval 30
```

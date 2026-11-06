# NExTQA 前200条 Random-head 消融

2026-11-04，用户将本轮“推全”的范围明确为 **NExTQA 原始前200条**。本轮运行 `doc_id=0..199`，比较原相似度选头与3个 Random-head 种子，共800次生成；不扩大至完整9178条测试集，也不运行VATEX。实验已于2026-11-04完成200条×4条件共800次生成及自动评分，并于2026-11-06复核。见[结果、图表与诚实结论](RANDOM_HEAD_NEXTQA_200_RESULTS_20261104.md)；本轮GPU已释放。

本轮继续使用已验证的 SnapKV-KVRepack 分支，尚不等于完成正文 HybridKV 那一行的消融。此前各20条的试跑及其结果保留，正式200条在独立目录重新生成，不拼接旧预测。

## 对照与固定设置

| 项目 | 本轮设置 |
|---|---|
| 模型 | Qwen2.5-VL-7B-Instruct，本地固定snapshot |
| 数据 | NExTQA OE/test，原始前200行，不筛选、不换样 |
| 条件 | `similarity`、`random_global_seed0`、`random_global_seed1`、`random_global_seed2` |
| head配额 | 28层×4个原生KV head中选择34个，`ceil(112×.30)` |
| 参数 | `simtop=.30`、`cr=.70`、`alpha=2`、KV预算比例`.10` |
| 输入与推理 | 64帧、BF16、FlashAttention 2、batch size 1、greedy、模型种子42 |
| 生成 | `max_new_tokens=128`、`num_beams=1`、`use_cache=True`、显式tokenizer EOS |
| repetition penalty | 继承固定模型配置中的`1.05`，四条件相同 |

Random在全部112个原生KV head间均匀无放回选取34个；每个种子使用固定模型级排序，所有输入复用。全局数量保持一致，允许层间配额不同。本轮不运行逐层配额匹配策略，也不重新运行All-head。

每条输入只在CPU预处理一次，四条件共享同一份输入张量；模型只加载一次。每次生成清理前一次的缓存与位置状态，独立安装和恢复选头适配器。四组逐输入核对输入SHA、28层候选SHA、实际34次合并、移除token总数、prefill缓存形状及其他配置，全部通过后才原子保存一个完整样本。压缩核心源码和已冻结的旧试跑worker不变。

## 在其他机器运行

先按项目说明准备依赖、7B模型snapshot、NExTQA视频及测试Arrow缓存；该入口离线读取现有资源，不下载模型或数据。WUPS评分还需要本地NLTK等评分资源。推荐使用固定200条并自动评分的控制器：

```bash
export PYTHONPATH=src
export MODEL_SNAPSHOT=/path/to/Qwen2.5-VL-7B-Instruct/snapshot
export HF_HOME=/path/to/huggingface_cache
export RANDOM_HEAD_RUN=outputs/ablations/random_head_snapkv7b_nextqa_200

python scripts/run_nextqa_random_head_200.py \
  --pretrained "$MODEL_SNAPSHOT" --hf-home "$HF_HOME" \
  --output "$RANDOM_HEAD_RUN"
```

控制器明确传入`--limit 200`，不会从数据集长度推断或扩大样本数。仅调用生成worker时，同样必须显式指定：

```bash
python -m kvrepack.ablations.run_head_selection_full \
  --limit 200 --pretrained "$MODEL_SNAPSHOT" --hf-home "$HF_HOME" \
  --output "$RANDOM_HEAD_RUN"
```

以上是两种入口，选择一种执行。新运行使用新输出目录；同一目录由文件锁限制为单个worker。使用控制器时，生成成功后会自动调用`full_head_selection_report`，将WUPS分数、三个种子的均值与样本标准差、按源视频聚类的配对区间及图表写入`$RANDOM_HEAD_RUN/report/`。直接调用worker不会自动评分。

## 状态、日志与恢复

```bash
cat "$RANDOM_HEAD_RUN/run_state.json"
cat "${RANDOM_HEAD_RUN}_controller_state.json"
tail -n 10 "${RANDOM_HEAD_RUN}_worker.log"
```

worker状态中的`target_docs=200`、`target_generations=800`固定；`completed_docs`只计算四条件均已完成且配对检查通过的输入，`completed_generations=4×completed_docs`。`current_doc_id`和`current_strategy`表示当前工作，`estimated_remaining_seconds`随真实完成速度更新。控制器还会区分`generating`、`scoring`和`complete`。评分日志为`${RANDOM_HEAD_RUN}_scoring.log`。

每条完整输入保存为`samples/000000.json`至`samples/000199.json`，包含四组预测、生成token、输入与候选指纹及实际执行审计；不会持续存储全部CPU视频张量。`run_contract.json`固定模型配置、数据来源、参数和代码SHA。若进程中断，在确认原进程已退出后使用相同入口和路径显式恢复：

```bash
python scripts/run_nextqa_random_head_200.py \
  --pretrained "$MODEL_SNAPSHOT" --hf-home "$HF_HOME" \
  --output "$RANDOM_HEAD_RUN" --resume
```

恢复会校验合同与已有样本的连续原始行覆盖。某条输入若只完成了部分条件，不作为完成样本提交，恢复时重新执行该输入的四组；不会跳过失败行。修改模型、数据、参数或绑定源码后，原合同不匹配的恢复会被拒绝。

## 本机运行与结论边界

本机正式输出为`outputs/ablations/random_head_snapkv7b_nextqa_200_20261104/`，运行状态为`complete`，`completed_docs=200`、`completed_generations=800`。生成worker墙钟约30分9.7秒，自动评分约10.8537秒，控制器端到端约30分29.55秒。此前独立单条检查及本轮完整执行审计均通过；实际检查范围和证据见[结果报告](RANDOM_HEAD_NEXTQA_200_RESULTS_20261104.md)。

实际WUPS为原策略32.917681、Random均值32.811042±样本SD0.336289，配对差为+0.106639点，95%源视频聚类区间[−0.191880,0.501384]。区间跨0，当前结果不支持稳定优势，也不是两者等效的证明。200条输入复用3个随机排序，不产生600个独立输入；完整分种子得分见[结果报告](RANDOM_HEAD_NEXTQA_200_RESULTS_20261104.md)。本轮已结束，不自动追加实验。

[原实验协议](RANDOM_HEAD_ABLATION.md) · [旧20条试跑结果](RANDOM_HEAD_ABLATION_RESULTS_20261104.md) · [流式worker](../src/kvrepack/ablations/run_head_selection_full.py) · [200条控制器](../scripts/run_nextqa_random_head_200.py)

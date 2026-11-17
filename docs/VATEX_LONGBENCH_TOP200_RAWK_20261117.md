# VATEX与LongBench：RoPE前raw K的最高200对相邻token

2026-11-17用户将对照明确为VATEX与LongBench，不再使用OCRBench。本实验以每输入、每层、每个原生KV头为单位，从全部序列相邻位置`(i,i+1)`中选取raw K余弦最高的200对。先对200对取均值，再等权平均200条输入、层和头。不同头和输入各自排序；不是跨样本汇总后只取200，也不是任意两两全配对或前20%。

- LongBench/qasper：Qwen2.5-7B-Instruct原始test行0..199，revision `a09a35458c702b33eeacc393d103063234e8bc28`，沿用完整长文本和固定原生chat模板。
- VATEX：Qwen2.5-VL-7B-Instruct原始test行0..199，revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`，64帧、min_pixels200704、max_pixels1605632、原caption指令及四段示例。复用`observations.inputs.prepare_sample`；不调用旧16对抽样plan。
- 两组BF16、FlashAttention2、batch1、原生未压缩backbone，`use_cache=False`，不生成或评分。

全部规定输入都足够200对：LongBench最短2094 token（2093对），VATEX最短8360 token（8359对）；不复制pair、不删短样本、不换行、不截断。序列相邻包含文本/视频/结构token及其边界，视频展平跨行也包含，因此不是单独的二维视觉近邻池。

## 选择与统计

在`k_proj`原输出捕获RoPE前raw K，GQA重复之前的28层×4头。FP32归一化，FP64点积与均值，保留负值，误差范围裁剪到[-1,1]。对`N−1`个候选做稳定降序排序，同分取原pair起点索引较小者，精确取200。

RoPE前指当前层施加旋转前的投影，不表示关闭前层注意力或视觉编码器中的全部位置作用。

每条保留选中的`pair_starts`、`scores`、左右token ID，形状均为[28,4,200]，保存为压缩NPZ并记录SHA。每层每头记录top200均值/最低选中值；同时记录全部相邻对的raw K、post-RoPE K、V均值作为参照。**V与post-K补充列是全邻接均值，不是按raw K所选200对的V或post-K，也不是各自top200。**

主图用112个层/头的200输入等权均值做连续KDE，金色LongBench、蓝色VATEX；并非逐pair余弦原始分布。固定选200只控制选中数量，两侧候选总数/长度与被选比例仍不同；两模型和语料也不同，结果不能单独隔离模态或证明合并无损。

## 检查与时长

新增13项CPU observer检查通过，包括显式排序oracle、稳定同分、BF16/负值、全部相邻候选、短输入拒绝、GQA前捕获、两种原生路径输出不变及异常清理。GPU pilot各2条独立完成，全部输入与旧LongBench全配对/旧VATEX近邻实验hash一致；pilot不混入正式200。

pilot含导入和加载共58.553秒。VATEX前两条约10.05/29.40秒，与此前同两条约9.52/30.15秒相近；这两条不是全体均速代表。旧VATEX200条总约1548秒，结合LongBench新采集预计正式400条30–45分钟，已在运行前告知；真实进度以state为准。

## 入口

```bash
PYTHONPATH=src python scripts/run_vatex_longbench_top_adjacent.py \
  --inputs /path/to/reference_samples.json \
  --text-model /path/to/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --video-model /path/to/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --hf-home /path/to/huggingface_cache \
  --text-reference /path/to/old/LongBench/text/records.jsonl \
  --video-reference /path/to/old/video_observation/records.jsonl \
  --sample-count 200 \
  --output outputs/observations/vatex_longbench_top200_rawk_200_20261117
```

`reference_samples.json`使用已准备的LongBench/OCRBench400行manifest，只读取其中text原前200行；不会读取OCR图片或运行OCR模型输入。两个reference记录用于强制核对原始输入，不复用其相似度数值。VATEX reference可包含两任务，worker只接受过滤后的200条VATEX。

正式ROOT：`outputs/observations/vatex_longbench_top200_rawk_200_20261117`。pilot ROOT：`outputs/observations/vatex_longbench_top200_rawk_20261117_preflight`。来源/成果：`provenance/vatex_longbench_top200_rawk_20261117`。控制器串行两个模型，随后自动CPU校验NPZ、统计并绘图。独立目录运行`--sample-count 2 --no-report`可计时；恢复必须显式`--resume`且源码/协议/输入一致。旧实验保持原样，不恢复其他停止任务。

## 首次中断与恢复

首个控制器UTC01:02:02启动；LongBench200条于约01:03完成，VATEX首条完成前worker6070收到SIGTERM并退出，信号发送方未确定，cgroup没有OOM事件。控制器的旧信号处理在主`wait()`持锁时再次调用`wait()`，导致退出等待卡住。已记录首attempt并仅结束确认卡住且无运行GPU子进程的控制器。

控制器修复只调整回收和恢复，不改七个数值依赖文件。两项真实CPU子进程回归验证SIGTERM时正常退出143/回收子进程，以及完整组不启动worker即可复用；篡改source/model/reference/NPZ均拒绝。真实LongBench200条的完整协议/来源/模型metadata/依赖/全部NPZ再次校验通过。

UTC01:15:28以独立后台session恢复，controller6809，`reused_completed_groups=["text"]`，仅启动VATEX worker6838。输出保留`original_started_utc`、`previous_attempts`，并单独记录恢复进程elapsed；不能把恢复进程elapsed误作含前一attempt的总耗时。首中断与恢复记录在同一provenance目录下的`attempt1_interruption.json`、`completed_text_reuse_check.json`和`resume_launch.json`。数值源码仍为`48df172`，控制器修复为`dd1c797`。

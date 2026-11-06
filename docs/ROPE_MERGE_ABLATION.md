# RoPE前后K聚合空间配对消融

2026-11-06状态：四个方法/任务组合各一条、两条件共8次真实GPU检查及离线评分已通过。正式200条矩阵于UTC04:53:48启动，计划共1,600次生成，目前运行中，不能视为正式结果已完成。实际状态以矩阵experiment_state.json与各worker的run_state.json为准。

模型固定为Qwen/Qwen2.5-VL-7B-Instruct，快照cc594898137f460bfe9f0759e9844b3ce807cfb5。任务为NExTQA OE test与VATEX test，各使用原始doc_id 0..199，保持原顺序，不去重、不按答案筛选。每个任务分别运行HybridKV-KVRepack和SnapKV-KVRepack，每个方法包含两种K聚合条件，共200 × 2任务 × 2方法 × 2条件 = 1,600次生成。此次不是整个9,178题NExTQA或整个VATEX测试集。

设同一合并组为G，组内原生KV头的RoPE前向量为k_i，V为v_i，R(p_i)表示该token实际三轴mRoPE旋转，a为组在候选顺序中的首个token。

基线raw_reapply先平均raw K，再使用左锚token的位置旋转：

$$
\widetilde{k}^{\mathrm{raw}}_G
=R(p_a)\left(\frac{1}{|G|}\sum_{i\in G}k_i\right).
$$

实验条件post_rope直接平均已经应用各自位置旋转的K，不再次应用RoPE：

$$
\widetilde{k}^{\mathrm{post}}_G
=\frac{1}{|G|}\sum_{i\in G}R(p_i)k_i.
$$

两条件的V公式相同：

$$
\widetilde{v}_G=\frac{1}{|G|}\sum_{i\in G}v_i.
$$

实际实现采用FP32累加并转回模型dtype；raw_reapply在重新旋转前也会按原实现将均值转回dtype。因此上式描述计算语义，不额外改变原有BF16舍入步骤。候选内相邻token不一定是原输入中的相邻token。两条件不改变候选顺序、重要度、合并边评分、分组、头选择或预算；头选择仍使用raw K的相邻相似度，不把SIMHEAD_SIM_SPACE改成rope。

| 方法 | 任务 | 合并头比例simtop | 候选压缩比例cr | K_DIFF_ALPHA | 生成上限 |
|---|---|---:|---:|---:|---:|
| HybridKV-KVRepack | NExTQA | 0.50 | 0.50 | 2 | 128 |
| HybridKV-KVRepack | VATEX | 0.50 | 0.50 | 2 | 64 |
| SnapKV-KVRepack | NExTQA | 0.30 | 0.70 | 2 | 128 |
| SnapKV-KVRepack | VATEX | 0.15 | 0.70 | 0.5 | 64 |

参数依据provenance/video_best_parameters.json中的P06、P07、P21、P22，是已观察到的同比例预算候选结果，不声称穷尽所有参数的最优值。所有条件固定64帧、10%历史预算、batch size 1、BF16、FlashAttention2、greedy生成、num_beams=1、无log-count校正、模型随机种子42。处理器min_pixels=200704、max_pixels=1605632，复用原任务提示和原视频采样实现。生成显式设置tokenizer的EOS/PAD；保留同一检查点generation_config中的repetition_penalty=1.05。checkpoint配置与显式generation_overrides均保存在run_config中。

Hybrid另固定chunk size 8、max group size 0、head_ada分类与预算分配、static merge scope visual、block meta uniform。其原生全局合并比例0.50对应56/112个KV头；Snap的两个任务分别为ceil(112 × 0.30)=34头、ceil(112 × 0.15)=17头。实际发生合并的数量和移除token数另由每次调用记录核验，不能只依赖配置值。

本实验没有编辑原核心算法文件。新增adapter位于src/kvrepack/ablations/rope_merge.py。raw_reapply直接调用原合并函数；Snap的post_rope函数从原函数AST构造，只替换K归约及锚位置重施RoPE的语句，V和分组语句保持原样，转换后的源码与哈希写入intervention.json。Hybrid仅在单次合并函数调用内临时选择原生rope归约分支，并在finally中恢复缓存级raw_reapply标志；其他缓存策略保持原值。

每个method/task进程加载一个模型。每条原始样本只做一次CPU预处理，记录输入指纹后，将相同GPU输入依次提供给两条件。每个条件前清理RoPE与生成临时状态；Hybrid还删除每层的kv_cluster，使新生成重建其元数据。生成不传入前一条件的past_key_values，每个条件使用新缓存。adapter离开作用域时恢复原函数。

配对校验严格比较头选择、候选/合并输入及位置、原函数形成的merge_idx/group_ids/group_size、目标长度和实际合并工作量。V使用相同归约代码；考虑CUDA原子累加次序的舍入，BF16/FP16的配对合并V最多允许朝比较值方向一个可表示步长，并记录是否逐位相等、非精确元素数及最大绝对差。超过此界限即拒绝提交。此容差不是改变V公式的授权。K输出差异另行记录。

Hybrid用合并K建立索引，新的K可能改变初始及后续检索的chunk与最终组装缓存，因此允许下游检索结果及组装后的V缓存变化；不能据此要求完整prefill/final K/V缓存哈希相等。严格配对的V检查位于同一合并组的输出，而非检索后缓存。Snap的固定prefill缓存形状须一致。

所有路径由命令行提供，避免依赖原HybridKV目录。先按仓库README安装匹配的环境；Hybrid需要从本仓库csrc编译kvrepack_kernel，不能回退旧mmko扩展。当前机器针对sm_80完成了本仓库编译和CPU导入/符号检查，GPU执行兼容性仍需smoke确认。其他GPU须显式使用其适当架构。

~~~bash
cd /path/to/KVRepack
export KVREPACK_REPO="$PWD"
export KVREPACK_HF_HOME="/path/to/huggingface_cache"
export KVREPACK_MODEL="/path/to/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
export KVREPACK_OUTPUT="/path/to/outputs/rope_merge_qwen7b_200"
export PYTHONPATH="$KVREPACK_REPO/src:$KVREPACK_REPO/vendor/lmms-eval"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
~~~

单条smoke必须使用独立输出目录。下面的命令只运行一个方法/任务的两个条件：

~~~bash
python -m kvrepack.ablations.run_rope_merge \
  --method snapkv_kvrepack --task nextqa_oe_test --limit 1 \
  --pretrained "$KVREPACK_MODEL" --hf-home "$KVREPACK_HF_HOME" \
  --output "$KVREPACK_OUTPUT/smoke/snapkv_kvrepack/nextqa_oe_test"
~~~

正式任务由调度方逐个启动，保持单GPU串行执行。将method依次设为snapkv_kvrepack、hybridkv_kvrepack，将task依次设为nextqa_oe_test、vatex_test，共四个独立目录：

~~~bash
python -m kvrepack.ablations.run_rope_merge \
  --method snapkv_kvrepack --task nextqa_oe_test --limit 200 \
  --pretrained "$KVREPACK_MODEL" --hf-home "$KVREPACK_HF_HOME" \
  --output "$KVREPACK_OUTPUT/snapkv_kvrepack/nextqa_oe_test"
~~~

worker不保存全量CPU tensor缓存。一个样本的两条件都生成成功且配对校验通过，才原子提交samples/000000.json等文件；未提交的半个样本不计为完成。显式加入--resume可复用同一输出目录中连续完整的样本前缀，但必须保持目标数量、输入、源码、模型和配置合同完全一致。不能把limit=1的smoke目录改成limit=200续跑。

每个目录保存run_contract.json、run_config.json、intervention.json、run_state.json及完整配对样本。合同包括数据Arrow、任务源码、模型metadata和本次执行源码的哈希；Hybrid另记录实际内核路径与SHA。验收应同时检查完成数量、输入与候选配对、原生KV头掩码、实际合并工作量、分组、V舍入界限、K是否真正发生变化，以及EOS与长度上限命中情况。CPU测试和单条smoke都不能替代这1,600次正式生成的结果。

完成生成后可离线生成单个method/task报告；该步骤不调用模型或外部评分API：

~~~bash
python -m kvrepack.ablations.rope_merge_report \
  --run-root "$KVREPACK_OUTPUT/snapkv_kvrepack/nextqa_oe_test" \
  --hf-home "$KVREPACK_HF_HOME" \
  --output "$KVREPACK_OUTPUT/snapkv_kvrepack/nextqa_oe_test/report" \
  --limit 200
~~~

NExTQA报告原口径WUPS及按source video成组的配对bootstrap区间。VATEX报告相同参考集/文档频率下的corpus BLEU-4、METEOR、ROUGE-L、CIDEr原始单位，不用行级CIDEr区间或复合均分代替四指标。报告中的差值统一为raw_reapply减post_rope。同时保留两条件的生成长度、EOS和cap统计，区分达到cap与达到cap但无EOS。报告默认使用5,000次bootstrap、seed 20261104。正式耗时与方向性结论等待实测，不在协议中预填。

## 本机启动与自动汇总

正式根目录为 `outputs/ablations/rope_merge_qwen7b_200_20261106/`。矩阵控制器依次执行NExTQA SnapKV、NExTQA HybridKV、VATEX SnapKV、VATEX HybridKV，每个组合独立加载模型，并在本进程中为每条输入运行两种K合并。完成一个组合后自动离线评分；全部完成后单独的汇总进程写根目录`RESULTS.md`、`comparison.csv`和`comparison.json`。

```bash
python scripts/run_rope_merge_qwen7b.py \
  --pretrained "$KVREPACK_MODEL" --hf-home "$KVREPACK_HF_HOME" \
  --output "$KVREPACK_OUTPUT" --limit 200

# 可单独运行；--wait等待矩阵结束后汇总，不执行GPU推理或重新评分。
python scripts/summarize_rope_merge.py \
  --run-root "$KVREPACK_OUTPUT" --limit 200 --wait
```

单组合报告沿用 `raw_reapply − post_rope`；汇总表明确改用 `post_rope − raw_reapply`，正值代表本次新对照更高，区间端点相应取反并交换顺序。NExTQA使用0–100 WUPS点，VATEX使用原始语料分数；不混用两种量纲。

初步估计1–2小时，包含视频预处理、生成、执行审计与评分，不是纯GPU性能计时。CPU suite共214项，213通过、1项既有来源差分测试按其条件跳过；另有20项离线评分fixture检查及四组合真实smoke评分。核心来源检查30个文件、55个本地模块通过，核心源码未改。Hybrid所需CUDA扩展由本仓库csrc编译并用于GPU smoke；二进制不纳入Git。

[启动前小样本证据与来源](../provenance/rope_merge_qwen7b_200_20261106_preflight/README.md)只证明流程可运行与干预配对，不支持效果结论。正式报告必须等待各200条完整生成。

# 借鉴 MixKV 图1(b) 的等 token 数相似度观察

**此方案已于2026-11-04按用户要求停止，不续跑。** 本机保留365行：NExTQA200、VATEX165；当前任务改为[Qwen2.5两模型各5个完整输入的pilot](MIXKV_REFERENCE_PILOT.md)。下文保留为已执行方案的历史说明，不代表最新计划。

本实验补充[观察A](OBSERVATION_A.md)，不替换或改写其结果。用户指出视觉序列远长于文本，要求不直接以全部近邻的均值作为模态对照。本实验在前向前固定两侧相同的测量 token 数、重复次数和非自身配对数，使用 MixKV 的 head 内全配对公式，而非增加视觉侧的统计权重。

原结果与MixKV的区别、逐层差异、文献支持的几何解释及其限制，见[相似度调研](VISUAL_TEXT_SIMILARITY_RESEARCH.md)。

## 文献依据及复现边界

[MixKV 原文图1及附录A.1](https://arxiv.org/html/2510.20707v2#A1)比较 **Qwen2-7B + LongBench纯文本**与 **Qwen2-VL-7B + OCRBench图文**，图1(b)展示所有层/头的平均相似度分布。这不是同一个多模态输入中的视觉/文本近邻比较。附录称其使用 representative samples，但没有给出图1的完整样本ID、样本数、精确prompt及所有统计细节；图2的100样本数不能套用到图1。

本实验保留已经指定的 Qwen2.5-VL-7B、NExTQA OE/test 和 VATEX/test 各原始前200行、64帧和原始任务提示。借鉴 MixKV §3.3 公式(4)，不宣称逐项复现原论文数值。实际缓存中的K在mRoPE之后，因此主分布使用 **post-RoPE K 与 V**，同时完整报告raw K。论文图1本身的精确捕获位置未能从已核对材料中确定，不能替作者补设为事实。

图中的“Cached K”指原生注意力/缓存所使用的mRoPE后K位置。本实验`use_cache=False`，没有持久保存完整KV缓存；观察器读取完整未压缩前向中的相应向量，只写小型统计和选定位置。

核对[官方代码仓库](https://github.com/xuyang-liu16/MixKV)截至`eeb708bc3be65beade00e231a7b2c2160995f3c5`时，未定位到可完整复现图1(b)的公开生成脚本。因此图1(b)是否排除对角线、如何跨样本汇总等细节也不能直接由算法§3.3反推。下文明确采用公开公式(4)重建统计方法。仓内已有相似度JSON缺少样本/语料metadata，数组长度不作为图1样本数证据。

## 主比较：等 token 数的组内全配对

每条输入先确定视觉位置和主文本位置。NExTQA主文本为问题本身；VATEX主文本为固定描述指令，其四段文字示例保留在模型输入中，另外计入完整user文本补充池。视觉位置为真实展开的视频token，排除视觉起止标记。

1. 预先确定 `T=min(16, 视觉token数, 主文本token数)`。不足2的输入报错，不复制token、不换行补样本。
2. 两侧各做32次均匀无放回抽样，每次选T个token。**同一次抽样内部没有重复位置**；不同次可以重用位置。抽样不读取向量或相似度。种子为 `42 + doc_id + task_offset + 2000000`，VATEX的task_offset为1000000。
3. 对每层、每个原生KV头、每次抽样，计算T个token全部 `T(T−1)` 个有向非自身对的平均余弦。两种模态有完全相同的T、抽样次数与配对数；不把对角线的1混入主值。
4. 每头先等权平均32次抽样，再对规定输入行等权汇总。分布中的每个点是一个层/KV头的行均值；7B共28层×4个原生KV头=112点。不能把GQA重复后的28个query头视作28份独立K/V。
5. 同时保留每次抽样值、全部层/头、负相似度和完整抽样位置。对任务平均差值按源视频整组bootstrap5000次，种子20261104；抽样重复、token对、层/头不充当独立视频样本。

设归一化后的向量为u，则用线性内存计算：

```
r = ( ||sum_i u_i||² - T ) / (T * (T - 1))
```

实现以FP32归一化、FP64累加，并按实际FP32归一化向量的对角范数和去除对角线，避免舍入误差。与小张量显式构造非对角相似度矩阵的结果做CPU对照。不构造长视频的T×T矩阵。

这是**全局、等量抽样**，不是空间/文本近邻实验。视觉patch距离与语言token距离仍不可视为相同单位；结果不能代替原近邻观察。

## 上下文及完整池补充对照

每条样本另做一次原生文本前向：移除video内容，仅保留完全相同的系统提示、完整user任务文本和assistant生成标记。主文本token ID必须逐个一致，使用与多模态条件相同的文本抽样位置映射。

- 等量文本在有/无视频时的逐层逐头差异，用来观察上下文和位置共同改变后的结果。
- 完整多模态prompt与完整纯文本prompt的全配对分布，作为图1(b)形式的补充。
- 同一输入完整visual池与完整主text池的全配对，作为等量抽样的参照。
- 完整user文本池单列，避免把VATEX固定四段示例悄悄混入主文本。

等量抽样**只匹配测量token数及配对数**，没有匹配全部因果输入长度、文本角色或空间/语言结构。文本去视频对照同时改变了上下文和位置，不能独立归因为其中一个因素。固定短指令也不是LongBench长文本。样本池长短本身不会在数学上强制余弦均值升高；等量抽样主要使比较口径及抽样不确定性更清晰。

原观察A中的相同M对近邻也在新的多模态前向中重算。指定 `--reference-adjacent` 时，检查原输入张量字节hash、原始行/帧采样/grid及全部逐层逐头近邻结果（容差1e-6），防止新增观察器改变输入或静默偏移统计。

## 实现与运行

- `src/kvrepack/observations/mixkv_style.py`：独立CLI、精确文本对照、固定抽样和原观察复核。
- `src/kvrepack/observations/all_pairs.py`：临时只读hook、逐头全配对及等量重复抽样统计；复用原hook的恢复逻辑，不改压缩核心。
- `src/kvrepack/observations/all_pairs_report.py`：完整性检查、聚类bootstrap、CSV和PNG/PDF/SVG分布。

```bash
PYTHONPATH=src python -m kvrepack.observations.mixkv_style \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct-snapshot \
  --hf-home /path/to/huggingface_cache \
  --limit 200 --frames 64 --matched-token-cap 16 --matched-draws 32 \
  --reference-adjacent /path/to/adjacent/records.jsonl \
  --offline --summarize \
  --output outputs/observations/qwen7b_mixkv_style_first200
```

先用独立目录、`--limit 2`进行输入/统计复核和计时，不把pilot续入200行协议。恢复正式运行时使用相同目录和全部参数，添加 `--resume`；源代码、模型metadata、Arrow、协议或依赖变化都会拒绝混合恢复。

本地保存`contract.json`、400行双条件`records.jsonl`、逐层逐头与32次抽样值、完整位置、输入hash、进度和错误。报告区分主等量分布及完整池补充图，不按结果挑模型、任务、层、头、token或样本。只观察表示相似度，不据此声明合并无损。

# KVRepack 项目交接（2026-11-04）

**2026-11-13 新授权：补齐VATEX原生／KVRepack完整E2E，四style复用。** 用户要求Snap/Hybrid各LookMstyle和Asymstyle200，并核查原版和KVRepack版，缺同口径E2E则补跑。已核四style共800完整、18数值依赖SHA一致，所以保留旧raw及原时延；旧原版／KVRepack虽有loop200历史计时，但预算/计时不同，同严格口径仅SnapKV前114，因此新四基线各200共800，最终8组1600。运行前已向用户说明正式800、总比较1600和约2小时（联调另计），预热8与preflight8另计。新strict_pruning仅预算适配原生函数，核心30文件不改；新run_baseline_completion成对逐doc轮换native/KV，同process/model，完整视频→输出计时与旧style同边界，审计/写盘计时外。新controller自动两骨干串行+离线8组评分，ROOT outputs/ablations/baseline_completion_vatex_200_20261113（状态以experiment_state为准）；旧停止任务绝不resume。18项定向CPU测试与local来源检查通过、两真实GPU各1配对通过，SnapKV输出/输入/headmask/预算与原strict父首例完全一致，Hybrid原生无merge且两版budget108659/108659。preflight证据provenance/baseline_completion_vatex_200_20261113，协议MERGE_BASELINE_E2E_VATEX_200_20261113.md。当前formal尚待启动，不能报已完成；用户TXT7.30和配色PDF不碰。

**2026-11-12 分图标题/图例与K/V叠加试绘完成：** 用户指定Key Similarity Distribution Comparison、Value Similarity Distribution Comparison两个标题，每面板各有Cosine Similarity横轴和模型legend，主图/原raw图已同步。另存reference_key/value/combined三组PNG/PDF/SVG；叠加同模型同色，K实线斜纹、V虚线点纹，face.025/hatch.28、填充z1曲线z4，交叠纹理不遮轮廓；叠加无均值竖线以免和V虚线混淆，不做峰高重缩放。原400raw/CSV/summary及主图四曲线density metadata全不变，_density_curve和统计函数AST不变。2项现有渲染测试（含15个输出文件）通过，独立视觉审阅通过；来源comparison_layout_update_20261112.json，index仅更新图和绘图meta并追加新资产。outputs副本也已更新，无GPU；用户TXT和配色PDF未改。

**2026-11-12 图内文字精简：** 用户要求尽可能去掉不必要灰字，已删除200＋200 Qwen连续主图/补图的全局总标题与底部两行说明，legend仅模型名，横轴缩为Mean cosine similarity并用深色；保留K before/afterRoPE、V、Density、均值虚线。PNG/PDF/SVG与outputs副本均重绘。输入/统计文件SHA、全部密度metadata和_density_curve函数AST均不变，仅_plot可见文字/布局改变；2项现有render测试通过。证据minimal_labels_update_20261112.json，artifact_index同步；没有GPU或新实验。用户时延TXT改动和输入调色PDF仍未碰。

**2026-11-12 Qwen2.5文本/VL观察图已按用户要求改成连续曲线：** 最终200＋200的 `provenance/mixkv_reference_qwen25_200_20261104/report/reference_main/raw.{png,pdf,svg}` 已重绘，outputs同名图同步。主图post-RoPEK/V、补图rawK/V不变；每曲线仍112真实head均值，ScottGaussianKDE、余弦边界反射，金棕text#B98A2F与深蓝VL#2F4373来自用户 `fullkv_k_head_envelope (1).pdf`。只改渲染，400raw、summary、所有CSV/均值/CI逐文件SHA不变；AST确认原统计函数全不变。重绘入口 `scripts/replot_reference_observation.py`；14项已有reportCPU测试及来源检查通过，未运行GPU。数据/旧新图SHA与密度积分验证见continuous_plot_update_20261112.json，artifact_index仅更新真实改动的六图并追加此次证据。用户原PDF与其修改的时延TXT保持不动，不将它们随此绘图更新提交。

**原生 SnapKV 基线已补齐（2026-11-12）：** 用户要求自身质量/时延也写进原表，现 `merge_compensation_table_with_timing.{txt,tex}` 在Snap三个merge前加 `SnapKV .3313 / ≈7.20†`，其三merge改为与Hybrid相同的缩进行；总10数据行三列，其余分数/时延不变。精确Avg .331264461087091，旧完整响应循环200/20023:59（日志7.20s/it），均来自 `/tmp/run_qwen7b_common_decouple_sweep_20260706.log.stopped_20260706_095548` 首block；summary第1行同4指标独立核对一致。原记录是7B/64帧/前200/nominal.1、纯snapkv，不是3B或表示观察。完整gzip/summary/摘录及来源已随既有追溯证据保存；此前“未找到纯Snap基线”结论已补正。本次只恢复旧记录，没有GPU实验。

**2026-11-12 用户纠正排版：只按原表直接填结果，保留原稿分数，不要多 panel、多版本或114条辅助表。** `docs/merge_compensation_table_with_timing.{txt,tex}` 已恢复原三列 Method / VATEX Avg. / E2E秒，Hybrid原4行＋Snap三种merge＋两standalone共9行。原Hybrid+KVRepack .3447保持，时延≈7.99†仅旧run参考（该timed-run Avg .343937），caption明确分数与时延不同run；原Hybrid≈7.94†，Snap+KV .3307/≈7.36†。新4组各200条全部完成，new quality/time分别HAsym .2961/10.018、HLook .3342/8.174、SAsym .2844/9.641、SLook .3285/7.910。standalone两行分数已有但完整E2E缺失，保留--，不启动补跑。删除额外模型E2E表，114与历史证据仍保留在原实验报告/归档内；不再引用旧20样本图或另改原稿分数。

**2026-11-12 论文时延表修正：此前将 VATEX 时延除以另一任务 FullKV 图值的百分比已撤回。** 用户要求按真实端到端秒数比较并补上 SnapKV 各 merge。完整流程三列表为 `docs/merge_compensation_table_with_timing.{txt,tex}`；模型端到端三列表另存 `docs/merge_compensation_table_model_timing.{txt,tex}`。当前 first200 四种新 merge 和 Snap first114 三种同输入聚合分 panel，不拼样本数/计时口径。旧原生Hybrid响应循环E2E已恢复约7.94s；原生SnapKV/FullKV及manuscript Hybrid+KVRepack=.3447对应的兼容时延仍未找到，不拼协议声称加merge更快。

**已补找出历史 SnapKV/HybridKV+KVRepack VATEX7B first200 的模型端到端记录，不能再说完全没有时延。** `/tmp/snapkv_repackkv_vatex_tune_20260710_r2_vatex_s0p15_cr0p70_a0p5.log` Generation 均2.100160s，Avg=.330672；`/tmp/run_vatex_hybridkv_pair_qwen7b_20260705.log`只取后半段，Generation均2.443860s，Avg=.343937（不是另一 manuscript=.3447）。旧 Generation 包视觉编码/prefill/decode，无显式同步，不含视频预处理/H2D/文本解码；nominal10%另加近期位置，与新严格总10%/显式同步/生成设置不同，单列历史参考不算同协议加速。两份原日志逐字gzip、400条阶段CSV、精确质量和计时边界已存 `provenance/merge_timing_recovery_20261112/`。另恢复旧响应循环总计时：原生Hybrid26:28/200≈7.94s，Hybrid+KV26:37/200≈7.99s，Snap+KV24:31/200≈7.36s；这是真实video至输出循环E2E近似值，不能说完全没有原生Hybrid端到端记录；无逐样本同步且旧协议不同，单列≈历史panel。旧inventory关于Only archived summary的结论由本次追溯补正，冻结原件不改。本次未运行GPU，七组新E2E均值已从raw再核对，原实验记录未改。

**本轮已全部完成并独立复核，GPU已释放。** VATEX first200、Qwen2.5-VL-7B/A80080G，HybridKV与SnapKV各LookM/AsymKV式聚合共800结果（228原样复用＋572新生成）；没有继续生成baseline。结果见[完整表与图](MERGE_AGGREGATION_VATEX_200_RESULTS.md)，compact CSV为merge_aggregation_vatex_200.csv，父Snap114同输入三arm质量/时延另见merge_aggregation_snapkv_114_matched.csv。Hybrid LookM/Asym CIDEr=.468165/.387725，模型E2E=2.357584/4.185451s，完整E2E=8.173605/10.017878s；Snap对应CIDEr=.458444/.354224，模型=2.000925/3.738625s，完整=7.909557/9.640830s。父114 KV/LookM/Asym CIDEr=.482117/.480605/.387079，模型=1.959640/1.964442/3.658765s；其CIDEr参考语料仅114，不能混入200表。Asym H/S分别8/9条无EOS到64上限、LookM H1/S0，原输出保留。结构源码审计400doc/800gen passed_complete；独立CPU对raw预测重新计算4x200＋3x114七corpora、496数值比较最大误差0，914唯一token解码全匹配，原514raw/33report SHA不变。最终CPU313项312pass1原skip，本地30core/67local来源核验passed；core算法文件未改，scope改动仅新worker orchestration，旧source快照与one/input/timing AST等价门禁已过。controller47875 UTC19:18:48正常结束，postprocess49154 UTC19:19:53完成，独立评分UTC19:21:20完成；本机ROOT outputs/ablations/merge_vatex_mergeonly_200_20261112，证据provenance/merge_aggregation_vatex_200_20261112含10.6MiB精确raw归档。所有本轮与旧RoPE/Random/LookM任务均保持结束，不自动重启。

**最新范围调整：用户明确不再重跑“原方法＋KVRepack”。** 旧三策略controller/PG10868及postprocess13550已停止，旧ROOT标superseded_by_user_scope、resume_authorized=false，保留Snap114完整三arm。新ROOT `outputs/ablations/merge_vatex_mergeonly_200_20261112`，controller/PG47875、UTC2026-11-11T17:48:59启动；只LookM/AsymKV两骨干各200，目标800，已复用228、只补572，不再生成baseline。父row本体/时延不改，父source18＋额外快照存provenance/merge_vatex_scope_change_20261112；迁移门禁已核对四个输入/计时/生成函数AST与全部算法依赖一致。35项合并/report测试通过，独立来源链13项测试通过。新postprocessor仅在全部complete后计分（见当前preflight/task_state.json的PID）；会另列父Snap114的三arm配对质量/时延，不将它冒充200基线。旧Snap200 raw预算均10.3047%、计时含审计，旧Hybrid raw只有97；不能拼进同协议strict200表。所有旧RoPE/Random/legacyLookM继续禁止恢复。

**最新授权：VATEX 前200条 merge 聚合与端到端时延对照已启动，尚无完整正式结果。** 用户确认固定候选/heads/groups/anchor/raw-K空间，只迁入LookM TextPivot与AsymKV Fisher-K/Vsum/LMA，分别接入HybridKV与SnapKV，merge插入点与原KVRepack相同；要求尽量沿用源码算子与效率。见[协议](MERGE_AGGREGATION_VATEX_200.md)。代码均在新ablations适配器内，原核心未改；每骨干三策略共1,200次正式生成，完整E2E计入视频处理/H2D/Fisher/生成/文本解码，同时单列模型E2E。全局严格10%含local窗口；Hybrid保留异质head预算、共同比例缩减，不能拼接旧nominal预算结果。正式数据逐条轮换顺序、aftertimer核对真实groups/masks/candidates，不将审计哈希计时。首条Snap/Hybrid三策略生成路径已联调；最初SDPA被改为直接原LMA以消除额外BF16log-count舍入。完整prefix反传probe首条N9736峰60.56GiB，而前200最长N24744，现已改为no-grad前缀cache与短文本suffix反传，遵循缓存梯度思路；最长doc39 N24744已实测通过，峰27.999GiB。Fisher不读答案，必须计入AsymE2E。预检记录在outputs/ablations/merge_vatex_20261112_preflight，pilot1数据类型失败/pilot2源冻结拒绝均不作正式结果；最终CPU293项292通过1原skip、本地来源核对通过（原参考目录缺文件仍单独记录）。Snap/Hybrid各2条三策略配对已过；正式ROOT outputs/ablations/merge_vatex_200_20261112，控制器PID/PGID10868、UTC2026-11-11T16:46:57启动，先Snap后Hybrid；postprocess PID13550仅等待全部complete后CPU计分，不恢复GPU。各case run_state与总experiment_state为准，formal200共1200，源已冻结。旧RoPE/Random/完整LookM停止限制仍有效，不恢复旧worker。

**2026-11-09严格LookM重跑已全部完成。** 用户明确授权“完成重跑”，仅Qwen2.5-VL7B LookM、64帧、两任务原始前200，共400 fresh。ROOT outputs/benchmarks/lookm_strict_200_20261109，controller/PGID58558，UTC02:49:37启动，现lookm_state complete；源bc8ebee/coreee1baad保持冻结。WUPS=31.111145486845444；VATEX Bleu4=0.2082951516285398、METEOR=0.2138384859549153、ROUGE-L=0.4167198796600370、CIDEr=0.4566462018942800。所有28×4 head的C=floor(.1N)，recent已计入，400条最大比率10%；全部EOS、0cap。独立stdlib执行审计与原生评分复算均passed，分数/CSV最大误差0，原报告SHA未改。生成墙钟804.727s/1727.497s；控制器端到端2563.089s。成果见[严格结果](LOOKM_STRICT_RESULTS_20261109.md)、CSV和provenance/lookm_strict_200_20261109(400预测归档/双审计/来源冻结)。代码与输入没有混入旧legacy预测，旧400仍不能当strict结果；Hybrid119和RoPE禁止恢复。本轮GPU已结束，不自动扩大。

**2026-11-09最新修正：LookM的近期窗口必须计入10%总prefill预算。** 用户明确否定此前10%之外再加32的设置。唯一LookM实现现在C=floor(r*N)、w=min32C、history=C−w；C0明确拒绝，k0 late层仍将历史merge进recent pivots，ratio1原tensor直返。依据LOOK-M官方TextPivot修复原Qwen适配跨batch/head全局isin漏合并，任意head_dim、零范数与低精度cosine路由也修正；保留原pivot权重、文本加分、GQAmean、前5%层prune及近邻窗口观测评分。只改baseline_utils.merge_kv及LookMCluster.update_kv，并以显式可逆补丁记录：原sourceSHA保留，恢复修改前integration后与真实原LookM文件AST比对，其他函数AST不变。新policy total_prefill_including_recent_v2，worker/report逐28层×4头检查C，旧记录legacy且不允许resume混入；单方法launcher也拒绝已complete的legacy case。CPU260项259pass1原skip，独立20项oracle检查通过；2个真实7B GPU smoke均通过（各task1条），112个head的实际prefill比例分别9.992714%/9.993837%，不是新正式200条结果；显式1条报告CLI修复后6项受影响检查通过；验证归档provenance/lookm_strict_budget_20261109，原源码完整归一化比对与两函数补丁校验通过。旧400条分数保留为超预算历史，不得标为strict10%，用户后续授权的严格两任务各200条现已完成，正式分数及复核见最新结果。只涉及LookM，Hybrid119条停止及旧RoPE不得恢复。详见[修正与检查](LOOKM_STRICT_BUDGET.md)。

**旧LookM两任务各200条已完成并评分（10%＋32 legacy，非本轮strict）；Hybrid重跑及原双方法矩阵保持停止。** 2026-11-09用户明确“你不用重跑HybridKV+KVRepack，把LookM结果给我”，见[当前状态与预算口径](LOOKM_COMPARISON.md)。原controller PID/PGID4690及Hybrid worker已核对命令行后SIGTERM退出；ROOT `outputs/benchmarks/lookm_vs_kvrepack_20261109/experiment_state.json`与Hybrid/NExTQA `run_state.json`均`stopped_by_user`、`resume_authorized=false`。Hybrid的119条完整部分记录保留，不续跑、不当成200条或混入LookM。LookM/NExTQA原始0..199的200条已完整生成并完成原任务CPU评分，WUPS=33.8571440351094；独立逐条重算一致，200次全部EOS、无截断，平均活动缓存比例10.298789%。另起且仅起LookM/VATEX200，PID/PGID36304，UTC01:42:55启动，状态在同ROOT的`lookm_only_state.json`；其各任务run_state给出准确进度，VATEX已完成200条，四指标为BLEU4 .2193327212159965、METEOR .21947138133147506、ROUGE-L .4329500116215331、CIDEr .4729287767910361，全部EOS无cap；LookM-only状态complete，400条预测与预算审计已归档provenance/lookm_video_200_20261109。当前目标为400次LookM生成，复用本轮已完成Next200，不重跑；不扩大9178/4478。原双方法脚本与matrix保留历史，不再执行或resume；原RoPE仍禁止自动续跑。新单任务CPU评分入口为`kvrepack.benchmarks.single_video_report --run-root CASE --hf-home HF --output CASE/report`，结果已保存于[结果页](LOOKM_RESULTS_20261109.md)，不等待停止矩阵完整。源文件保持冻结，本轮本段只更新文档。原启动前235项CPU检查234通过1跳过、4格smoke及旧Hybrid首例迁移一致性作为历史检查保留。LookM保留原实现，10%历史预算另加32近期位置；实际缓存与总VRAM分开报告，不称官方论文精确复现。

**2026-11-08：用户要求的消融数据已整理为[一个CSV](ablation_results_summary.csv)。** 同步并保留远端截至6c65bf8的更新后，汇总13个已完成指标行：Random-head的Hybrid NextQA200、Snap NextQA200、Snap NextQA20及VATEX20四指标；RoPE聚合的Hybrid NextQA200、Snap NextQA200、Snap VATEX200四指标。另列Hybrid VATEX RoPE仅97/200配对的未评分停止状态，分数空白不填0。本表共14行36列、UTF-8 BOM，保留三Random seed、均值/样本SD、基线−对照差值与原报告CI、样本/源视频数、head配额、预算、cr/alpha、生成上限及来源。RoPE统一raw_reapply−post_rope（与停止汇总completed_comparisons.csv反向），Random统一Similarity−Random均值；Hybrid Random只同56头不等移除工作量，VATEX Random20 max128与RoPE200 max64亦明确区分。不合并重叠前缀或重复指标，不把相似度观察/迁移核验当消融，也不重启任何已停止任务。本轮仅汇总与核对现有记录，未运行GPU。

**最新HybridKV+KVRepack Random-head已完成，2026-11-07独立执行与评分复核均通过。** 2026-11-06 UTC07:00–07:47完成NExTQA原始前200条、184源视频、四条件800次生成及正式离线评分，见[结果与图表](HYBRID_RANDOM_HEAD_NEXTQA_200_RESULTS_20261106.md)和[协议](HYBRID_RANDOM_HEAD_NEXTQA_200.md)。Qwen2.5-VL7B＋原native Hybrid，gate.5/cr.5/alpha2、budget.1、64帧、greedy、max128、repetition1.05；similarity对比固定模型级global uniform Random seeds0/1/2，从112原生KV头选56个，各条件每输入均56次有效合并。原分类、每头保留预算及完整pre_selection快照一致；异质预算及层间分布变化使实际移除量不同，不能称严格等合并工作量。WUPS原33.219903；Random0/1/2=33.122167/32.799673/33.342348，均值33.088063±sampleSD.272940，原−均值=+.131841，184源视频/5000次配对聚类95%区间[−.778808,1.353447]。seed2更高，不支持稳定优于Random，也不是等价证明。800次全部EOS、0上限命中；worker墙钟约47分27秒，正式计分10.771秒。ROOT为outputs/ablations/hybrid_random_head_nextqa_200_20261106，worker及controller均complete；发布证据目标provenance/hybrid_random_head_nextqa_200_20261106/。原核心及旧实验workers未改。运行内门禁与报告audit记录passed；2026-11-07[独立全量执行审计](../provenance/hybrid_random_head_nextqa_200_20261106/independent_execution_check.json)亦passed、0问题，覆盖800记录、600组完整配对、固定rank重建及16源码与内核SHA；similarity与旧Hybrid raw200条输入/IDs/text/native mask/groups完全一致，Random0/1/2预测改变10/16/9条、全部首token相同。[独立评分复核](../provenance/hybrid_random_head_nextqa_200_20261106/independent_scoring_check.json)亦passed，800次原生WUPS无预测缓存重算、600组配对差值、三seed样本SD及源视频聚类CI均匹配，逐条分数与CI最大差为0；原报告12文件SHA保持不变。本轮不含VATEX或全测试集，不恢复下述已停止RoPE任务，旧SnapKV结果独立保留。

**RoPE尝试已按用户要求停止，禁止自动续跑。** 原controller组9353、aggregation组13514已终止且确认GPU0后才启动上述新任务。旧ROOT outputs/ablations/rope_merge_qwen7b_200_20261106状态stopped_by_user、resume_authorized=false。三组各200已完成：NextSnap WUPS32.917681→32.707920（raw−post CI[-.019996,.634253]）；NextHybrid33.219903持平；VATEX Snap四项均下降（CIDEr.463033→.451014）。最后HybridVATEX仅97个完整paired样本，未评分，保留且不换样。697pairs/1394完整gens不是1600全完成。已停止快照provenance/rope_merge_qwen7b_200_20261106_stopped；原始所有bundles仍outputs，旧全量打包helper因未全部完成不可执行。旧Rope compile核心/adapter/worker/report保留，不为补Random而重启。

**此前NExTQA前200条 SnapKV Random-head消融已完成并复核。** 实验2026-11-04运行、2026-11-06复核；原始`doc_id=0..199`，四条件800次生成全部EOS、0上限命中，只NExTQA、不VATEX。见[结果与图表](RANDOM_HEAD_NEXTQA_200_RESULTS_20261104.md)及[协议与复现命令](RANDOM_HEAD_NEXTQA_200.md)。Qwen2.5-VL7B + SnapKV-KVRepack，原策略与固定模型级global Random seeds0/1/2，保持34/112原生KV头、`.30/.70/2`、10%预算、64帧、128生成上限和repetition penalty1.05；允许层间配额变化，未运行逐层matched或All-head，不拼接旧Hybrid行。WUPS原策略32.917681；Random0/1/2=32.450740/33.116603/32.865783，均值32.811042±sampleSD.336289，原策略−均值=+.106639，184源视频/5000次聚类配对95%区间[−.191880,.501384]。seed1反而高，不能写稳定优于Random，也不能证明等效。report及独立执行审计检查输入/candidate/配置/work/cache形状配对通过，8源码SHA匹配；预测改变6/6/4条。worker墙钟1809.701921秒，自动评分10.8537秒，控制器端到端30分29.55秒。输出`outputs/ablations/random_head_snapkv7b_nextqa_200_20261104/`及其`report/`；发布证据位于`provenance/random_head_snapkv7b_nextqa_200_20261104/`。状态complete、GPU已释放，本轮不运行更多实验，不恢复下述历史停止任务。

**Random-head首轮已完成并停止**：Qwen2.5-VL7B + SnapKV-KVRepack，NExTQA/VATEX各原前20行，原策略与global Random seeds0/1/2共160次生成。见[实际结果](RANDOM_HEAD_ABLATION_RESULTS_20261104.md)与[协议](RANDOM_HEAD_ABLATION.md)。仅换mask，原生112头中分别选34/17，输入及全部候选SHA、实际merge数量/移除量、28层prefill形状配对通过，核心文件未改。NExTQA原/Random均值=32.016623/31.950346（差+.066277，视频聚类95%CI[0,.198830]）；VATEX CIDEr=.505653/.521858，Random均值更高，其余三指标也同向。**没有稳定优于Random的证据，不能写成Head Selection的正向消融结论**；这里只是Snap7B/n20，未完成原文Hybrid表或总体有效性证明。正式17分24秒，累计generate约298.229秒，全部EOS、无cap。VATEX CPU准备曾误并行，在生成0条时被cache伴随文件检查中止，已存16条复验后串行补齐，失败记录保留。新CPU30项通过；全suite205项204过1旧skip；来源30core/50local通过。证据位于provenance/random_head_snapkv7b_20261104，GPU已释放，不自动扩大或恢复旧任务。

本文是随项目保存、可由用户和开发者审阅的交接摘要：记录已接受的要求、实现选择、验证证据和待办事项，供后续会话恢复项目背景。后续修改应以当前代码、报告和用户的新指示为准。密钥、凭证、模型权重、视频和数据集保存在项目外。

当前版本完成了独立项目整理、CPU 审计、Qwen3B 两种改进方法的 GPU 小样本迁移对照及 Qwen7B 的400条未压缩表示观察，仍有两项已确认但未修复的行为问题。小样本一致性和表示观察不表示完整压缩基准已经复现；GitHub 上传状态以实际远端和提交记录为准。

用户指定的目标为公开仓库 `https://github.com/dataWoader12138/KVRepack`。提交作者为 `dataWoader12138`；Git 的身份和 SSH 凭证在各机器本地配置。

**最新授权已更新为两组各200条完整输入观察**，见[200条协议](MIXKV_REFERENCE_200.md)：Qasper0..199与OCRBench0,5,..995；原模型snapshot、像素设定、所有token精确全配对及原生观察器不变，主post-K/V、raw-K补充。新的独立路径`outputs/observations/qwen25_mixkv_reference_200_20261104/`，两个worker顺序运行，状态看各group/run_state.json及总experiment_state.json。旧“只5＋5不扩大”的限制已由此次用户明确请求覆盖，但不重启旧视频任务。样本含148篇不同article/199张不同原图，报告采用两组独立来源聚类bootstrap，不能按行号或head跨组配对。原生统计模块未改，只扩展reference输入/CLI/report显式sample_count(5,200)，默认仍5；此前pilot及其contract/产物保持原样。正式GPU已开始，完成前不能把计划写成200通过；后续必须更新实际结果与计时。

**上述200＋200现已全部完成并停止**，见[结果](MIXKV_REFERENCE_200_RESULTS_20261104.md)。text200=91.569秒，vision200=75.832秒，合计2分47秒；调度端到端2分59秒，均不含准备和绘图。真实输入无截断/换样/失败，长度text2094–22064、vision61–16308。rawK text/VL=0.6779314109/0.5920369405，postK=0.4986427404/0.5744482152，V=0.0966598961/0.1301542318；VL−text区间分别[−.089431,−.082514]、[+.072580,+.078912]、[+.030272,+.036592]。10条旧pilot重叠输入hash及所有层头三指标完全一致，最大差0；来源SHA冻结，原生模块不变。CPU175项174通过1旧skip，来源检查通过；独立重算均值误差≤2.22e-16、独立聚类区间误差0。小型来源/契约/400条统计/图表与完整性索引位于`provenance/mixkv_reference_qwen25_200_20261104`。最新用户问RoPE前后为何不同：已说明不同位置的旋转会改变K–K余弦；实测text降.179289、VL降.017589；图像mRoPE与长文本位置布局不同，但不能单独归因长度/模态，rawK也含前层位置影响。任务只剩结果/文档发布，**不要重启任何GPU/旧视频任务或扩大实验**。

2026-11-04 最新用户指示是**停止等T随机子集实验，只做类似MixKV图1(b)的小批量，且使用2.5系列**。旧路径`outputs/observations/qwen7b_mixkv_style_first200_20261104`已停止，state=`stopped_by_user`、resume_authorized=false；保留365行（NExTQA200、VATEX165），无尾部不完整行，GPU进程已退出。不得自动续跑或把365/400当作完整实验。NExTQA200的完整阶段统计在该目录`nextqa_checkpoint/report`：matched rawK visual/text=0.6393388834/0.7322346623、postK=0.5710947896/0.6995809908、V=0.1340562239/0.2187641099；去视频后的相同文本postK=0.7009006648、V=0.2451769803，未支持“共享video前缀是文本更高主要原因”的简单解释。旧[等token协议](MIXKV_STYLE_OBSERVATION.md)和源码保留，仅作历史记录。

最新任务采用[完整输入5＋5协议](MIXKV_REFERENCE_PILOT.md)：Qwen2.5-7B-Instruct（revision`a09a35458c702b33eeacc393d103063234e8bc28`）处理LongBench/qasper原0..4，Qwen2.5-VL-7B-Instruct（本机既有`cc594898137f460bfe9f0759e9844b3ce807cfb5`）处理OCRBench固定0/200/400/600/800。全部prefill token精确非对角全配对，不token抽样、生成、压缩或截断；每组仅5条，不自动扩大。新增`observations/reference_inputs.py`、`native_all_pairs.py`、`reference.py`、`reference_report.py`。本地目录`outputs/observations/qwen25_mixkv_reference_pilot_20261104/`；完整输入manifest在inputs/reference_samples.json，仅留本机。旧Qwen2两模型下载已取消；纯文本Qwen2.5已通过官方ModelScope镜像下载并按HF固定revision的实际payloadSHA验证，缓存已就绪，完成标志为`outputs/observations/mixkv_qwen25_reference_model_path_20261104.json`。下载记录见`outputs/observations/mixkv_qwen25_segmented_download_20261104.log`，准备过程与GPU前向时间分开记录。不要重启任何已取消的下载或旧GPU实验。

上述5＋5现已**完成并停止**，实际结果见[报告](MIXKV_REFERENCE_PILOT_RESULTS_20261104.md)。vision进程6.712秒、text进程5.789秒（均含加载、不含依赖导入及下载），GPU已释放。全部10条无截断/替换/缺失；文本token数4091/3428/4478/3079/4533，图文64/69/356/664/1013。rawK text/VL=0.6771765946/0.5955875879，postK=0.5030549992/0.5797005708，V=0.0958604792/0.1293283778。postK/V方向与MixKV相符但rawK相反，分布重叠，不能称原图精确复现或单一模态因果证明。CPU suite166项、165通过1项旧差分跳过；来源检查46个本地模块通过，核心未改。独立重算和CPU输入hash复核通过，均值最大重算误差约1.11e-16；冻结sourceSHA未变。小型结果、10条原始统计、两个contract/state、不含正文的来源manifest、图表与完整性索引在`provenance/mixkv_reference_qwen25_20261104`，不含文章全文、图片或模型权重；旧停止状态/NExTQA阶段报告在`provenance/mixkv_style_qwen7b_20261104`。**新旧GPU任务和下载均已结束，不自动续跑或扩大。** 用户最新追问图1采集时点；已说明公开代码先RoPE后压缩/入cache，但Fig1本身未披露精确采集阶段，不能由PCA/量化脚本断言原图是prefill/postRoPE。当前只收尾文档与证据上传，不运行更多GPU。

最新用户授权的独立任务是[观察实验A](OBSERVATION_A.md)：Qwen2.5-VL-7B、NExTQA/VATEX各原始前200行，只比较视觉空间近邻和自然文本近邻，保留原行且不去重；不做随机/远距对照或合并干预。代码位于 `src/kvrepack/observations/`，正式400条原始记录和状态位于 `outputs/observations/qwen7b_adjacent_first200_20261104/`；4条计时pilot使用独立目录，不纳入正式400条。本轮不改压缩核心。下述旧大规模矩阵已在VATEX200因1例V缓存字节哈希不同而按严格门禁停止，两端输入、生成token/text和logits未报差，尚未解释该V差异幅度或完成全量；不能认为旧矩阵仍在后台运行。

该观察实验现已**400/400完成**，UTC07:16–07:54，正式耗时约38分钟，无失败或M=0排除；完整结果见 [OBSERVATION_A_RESULTS_20261104.md](OBSERVATION_A_RESULTS_20261104.md)。两任务的视觉近邻raw K和V均值均低于相应文本近邻，四个主配对区间均低于0；不能写成视觉整体更相似或合并安全。存在层/头异质性，不是每层每头同向。模型28层、4个KV头；NExTQA184个源视频聚类、VATEX200。运行协议及图表证据位于 `provenance/observation_a_qwen7b_20261104/`；完整原始records保存在outputs，未加入Git。当前观察任务已结束，GPU已释放；不要因旧段落中的计划而自动恢复旧失败矩阵。

2026-11-04 后续用户已授权扩大 GPU 测试。当前计划先经正式 CLI/adapter 完成 SnapKV-KVRepack、MadaKV-KVRepack 在 NextQA/VATEX 各200条的原压缩实现与整理版配对，再自动续跑 SnapKV-KVRepack 的两个完整测试集（9178/4478条，两端均跑）。本机后台控制器、断点、资源冻结记录与进度位于 `/root/autodl-tmp/HybridKV-artifact/kvrepack-full-work/`；不要重复启动已有控制器或改动其冻结代码/资源。阶段完成、错误及全量结果会由独立发布器更新到 [扩大复测报告](FULL_VIDEO_RETEST_20261104.md) 和相应 provenance；以该页及本机 `matrix_state.json` 的实际状态为准，不能把计划或启动视为通过。核心算法保持 `169aa1f`，本轮新增的是外部追踪、对照与断点工具。

初始版本已推送到 `main`，提交为 `fba9c68`。后续视频参数核查见 [VIDEO_BEST_PARAMETERS.md](VIDEO_BEST_PARAMETERS.md)、[CSV](video_best_parameters.csv) 和 [机器可读证据](../provenance/video_best_parameters.json)：主表45个数据集/模型/方法格，36格有完成记录、9格明确缺失；独立评分与共享/配对缓存分开，历史 `log_count` 不混入当前 `none` 表。部分配置只有一个已完成候选，不能称充分寻优。此次发布只新增参数和证据文档，未改变算法或CLI默认值。

## 目标与已接受的约束

用户希望把原工作区中用于主实验的代码整理成一个独立、可上传 GitHub、可在其他机器使用的版本。主实验原先以 `HybridKV-opensource` 运行，部分扩展、入口或导入与 `HybridKV-artifact` 有关联，因此必须追踪实际调用关系，不能仅凭同名文件选择来源。

- 同时保留 Qwen2.5-VL 与 LLaVA-OneVision 的已选范围。每个模型/方法组合只保留一种明确实现，避免同名包跨目录补齐或运行时切换来源。
- 保持基线和对应改进的算法语义、预算规则及默认参数，不把旧最优参数、当前脚本配置和消融配置混为一谈。发现原问题时先记录来源和影响，不能把修复藏在整理或改名中。
- 当前项目已统一 KVRepack 命名：改进方法为 `*_kvrepack`，对应文件、函数、配置属性和环境变量一起改名；环境变量中的 `DECOUPLE` 改为 `KVREPACK`。真实历史来源路径、参考脚本名及来源记录保持原样。
- 公开 `asymkv` 固定选择 official 实现，内部名仍为 `asymkv_official`。所需私有 helper 不等于另外开放 paper 或 Hybrid-AsymKV 方法。
- 原始实验目录保持不动。项目通过自己的 `kvrepack` 包、定制 `lmms_eval`、资源文件和 `kvrepack_kernel` 源码运行，不依赖原服务器绝对路径、软链接或原 `.so`。
- 用户已授权整理、CPU 检查、代码关系审计，以及将项目和本交接摘要上传到其指定 GitHub 仓库。GitHub 推送采用 SSH，目标仓库、仓库可见性和提交作者须由用户提供；专用私钥留在服务器，公钥由用户添加到目标仓库并授予写入权限。其他机器需自行配置访问凭证。2026-11-04 用户另已明确授权 GPU 小样本复测，完成范围见下文；自定义 CUDA 内核尚未构建或验证。

当前入口假设单进程、batch size=1、greedy generation、`use_cache=True`、多 token 视觉 prompt。每个模型/方法用独立进程，避免 monkeypatch 残留。跨方法对照需显式统一模型、任务、样本、帧数和预算。SnapKV 改进所选旧入口的默认 merge ratio 为 0，不能因为名字是“改进”就默认其已启用合并。

## 来源关系与当前范围

原工作区包含以下树；这些名称是历史定位信息，独立仓库运行时不需要这些目录。

| 原来源 | 与项目的关系 |
|---|---|
| `HybridKV-artifact/` | 多数当前核心与改进实现的来源；`scripts/eval/` 中选定入口的原文本已保存在 provenance。 |
| `HybridKV-opensource/` | 用户主实验的运行底座之一；Qwen FullKV 保留其原加载政策；提供 LLaVA 原有 dispatcher。实际导入组合必须与 artifact 区分。 |
| `HybridKV-llava-ov-decouple-port-20260708/` | 后续发现的独立 LLaVA 改进 port；以 opensource 为底座，并复制 artifact 的部分 Qwen Decouple 依赖。其旁的 `HybridKV-llava-ov-decouple-port-20260708_PORT_SOURCES.txt` 记录了来源。尚未迁入本项目。 |

当前共 **19 个模型/方法组合**：

| 模型 | 已纳入的方法 |
|---|---|
| `qwen2_5_vl`，3B/7B | 13 种：`fullkv`、`hybridkv`、`snapkv`、`sparsemm`、`madakv`、`lookm`、`randomkv`、`asymkv`，以及前五种压缩基线对应的 `hybridkv_kvrepack`、`snapkv_kvrepack`、`sparsemm_kvrepack`、`madakv_kvrepack`、`lookm_kvrepack`。 |
| `llava_ov`，OV-Qwen2-7B | 6 种基线：`fullkv`、`hybridkv`、`snapkv`、`sparsemm`、`madakv`、`lookm`。 |

LLaVA 的 SnapKV、SparseMM、MadaKV、HybridKV 四种改进源码已经找到，尚未迁入或纳入上述检查范围；对应组合会在规划阶段被拒绝。port 中的关键文件为 `qwen2_model_{snapkv,sparsemm,madakv}_repackkv.py`、`modeling_llava_onevision_repack.py`、`modeling_qwen2_repack.py` 和 `hybridkv_repack_utils_llava.py`。未找到对应的 LLaVA LookM 改进。

port 和当前 Qwen 改进共享候选选择、相邻合并、raw K 重施 RoPE 等思路，但同时存在预算边界、score 数值路径、SparseMM 缓存生命周期及额外加权合并选项的差异，不能当作纯架构改名直接拼接。LLaVA 使用普通 Qwen2 RoPE，Qwen2.5-VL 使用 mRoPE；后续迁入需单独记录这些适配和语义差异。

当前任务范围为 NextQA、VATEX、VideoChatGPT。多图 MileBench、历史参数搜索、效率诊断、旧 adjacent/base/blacklist、独立 headsplit 和 AsymKV paper 路线不作为首版可运行方法。

后续已在原工作区 `HybridKV-artifact/hybridkv (1).tar` 找到真实 MileBench 图片实验结果：770份预测、768组完整评分文件，覆盖16任务和Qwen3B/7B，含参数组及备份。图片脚本和四种改进也在归档；当前展开目录的脚本不代表归档版本，尤其Snap预算取整/缓存协议有差异。图片入口仍未迁入本仓库；不能因当前目录无输出而再次断言没有图片运行记录。

## 最新验证结论

以下数字来自 2026-11-03 的 [汇总报告](../provenance/cpu_audit_summary.json) 和 [CPU 审计说明](CPU_CALL_AUDIT.md)：

| 检查 | 实际结果与边界 |
|---|---|
| 动态分派 | 19 个组合各自独立进程，49 次真实 patch 赋值；9 个 prepare wrapper 各检查 4 种缓存状态。使用模型/压缩执行替身，未运行完整 forward。 |
| 静态调用关系 | 33 个 Python 文件、498 个函数定义、393 个已解析本地调用位置；392 处可核对参数，未确认参数绑定错误。另有 5783 处外部或动态调用不在证明范围内。 |
| 缓存与接口审计 | 按 Transformers 4.52.3 实际源码核对，16 条 decoder→attention 显式调用签名及返回解包匹配。 |
| 原函数 CPU 小张量 | 15 项完成，0 failure、0 error、无 skip；包含原 artifact 同输入差分，并成功复现下列两项问题。CUDA 入口调用为 0。 |
| 完整 unittest suite | 49 项，48 通过、1 跳过；跳过项要求显式提供原工作区的差分检查，已在上述带 `--source-root` 的独立 CPU 检查中通过。 |

上述 CPU 检查通过表示分派、结构和相应断言符合记录，不能解释为所有方法行为正确。

2026-11-04 的 GPU 子集复测使用真实 Qwen3B 模型与 FlashAttention，比较 SnapKV-KVRepack、MadaKV-KVRepack 各自的原实现和迁移实现。NextQA、VATEX 各2例、64帧，共8组配对：生成 token 完全一致，36层 prefill/final KV 的 SHA-256 一致，全部词表的生成 logits 最大绝对差为0。SnapKV 使用 `(simtop, cr, alpha)=(0.7, 0.9, 2)`；MadaKV 两个数据集均使用 `(0.5, 0.7, 2)`，其中 NextQA 是代表配置，并非其 `cr=0.5` 的历史最佳配置。

本轮使用隔离 worker 直接执行真实模型、monkeypatch 与压缩函数，绕过统一 runner 和 adapter；因此尚未验证完整 CLI、完整基准任务评分、自定义 CUDA 内核、7B、LLaVA、HybridKV、SparseMM 及其他未覆盖方法，也未验证显存、时延或吞吐。另已修复 runner 只接受 schema 1、与现有 schema 2 来源清单不兼容的启动问题，保留原有哈希与路径验证。详细证据见 [GPU_RETEST_20261104.md](GPU_RETEST_20261104.md)。

## 已知未修问题

| 编号 | 影响与归属 | 下一步验证重点 |
|---|---|---|
| `LLAVA-SPARSE-STATE` | `llava_ov/sparsemm` 同一模型进程处理不同长度 prompt 时复用首个容量表。CPU 8→16 token 已复现；新建对象的第二次容量不同。artifact attention 与 helper 配对时已有该问题；相对每次 prefill 重建的旧 opensource 执行路径，这也是实现组合回归。 | 修复方案需只重建或重算当前 prompt 的预算，避免在 decode 重置元数据；保留两 prompt 与 fresh-object 对照。 |
| `ASYM-PREFILL-BOUNDARY` | 默认 chunk=512，prompt 长度513/1025分别只转发512/1024 token；末 token 未进入 prefill。继承自所选 official 集成中的 `range(0, prompt_length - 1, chunk_size)`，不是改名引入。 | 明确修复与保持旧行为的取舍；覆盖512、513、514、1024、1025及最短输入，检查 token 覆盖与逻辑位置。 |

另有两个低风险直接 API 边界，仍未修复：`HYBRID-TEXT-INPUTIDS-ONLY` 在直接调用自定义 Hybrid 文本模型、仅传 `input_ids` 时先读取空 `inputs_embeds.shape`；`MASK-INPUTEMBEDS-ONLY` 在部分 mask 外层模型仅传 `inputs_embeds` 时读取空 `input_ids.device`。当前视频 adapter 传 token 并由外层构造 embeddings，绕开这两项；不能把它们表述为默认视频入口必然失败。

精确触发条件、影响组合、源码行号和来源解释见 [cache_interface_audit.json](../provenance/cache_interface_audit.json)，CPU 复现输入/输出见 [cpu_compression_audit.json](../provenance/cpu_compression_audit.json)。审计阶段没有修改核心来修复这些问题。

## 从哪里继续读和运行

| 文件 | 用途 |
|---|---|
| [README](../README.md)、[METHODS](METHODS.md) | 安装说明、支持范围、当前方法名、来源入口和默认配置。 |
| [REPACKING](REPACKING.md)、[VALIDATION](VALIDATION.md)、[CPU_CALL_AUDIT](CPU_CALL_AUDIT.md) | 文件收敛与命名映射、阶段边界、调用关系和检查细节。 |
| [GPU_RETEST_20261104](GPU_RETEST_20261104.md) | Qwen3B 两种改进方法的 GPU 小样本迁移对照、参数、证据与未覆盖范围。 |
| `src/kvrepack/experiments/registry.py`、`runner.py`、`run.py` | 方法唯一映射、参数白名单、环境隔离、计划与统一 CLI。 |
| `src/kvrepack/monkeypatch.py`、`vendor/lmms-eval/lmms_eval/models/` | 实际方法分派、prepare wrapper 和模型 adapter。 |
| `provenance/core_sources.json`、`vendor_sources.json`、`reference_entrypoints.json` | 原始来源、机械变换、SHA/AST 证据和参考脚本原文。 |
| `provenance/method_registry.json`、`environment_observed.json` | 当前配置快照和原环境的包版本记录。 |
| `provenance/cpu_audit_summary.json`、`validation_report.json` | 最新汇总、打包与来源核验状态；优先于本文的固定统计。 |
| `provenance/dispatch_audit.json`、`call_graph.json` | 19 种实际分派结果和可静态解析的后续调用图。 |
| `scripts/check_{sources,dispatch,call_graph,cpu_compression}.py`、`tests/` | 可复查的检查入口；命令见下方。 |

原观察环境名为 `hybridkv`，使用 Python 3.12、PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124`、Transformers `4.52.3`、flash-attn `2.7.4.post1`、NumPy `1.26.4`、Accelerate `1.13.0`、datasets `2.19.2`、decord `0.6.0`。完整约束见 [constraints-observed.txt](../requirements/constraints-observed.txt)。这是已观察环境快照，尚非全新机器验证过的安装锁；PyTorch 的 CUDA wheel 来源和编译 toolkit 需在目标机器另行固定。环境名不要求跨机器一致。

在项目根目录、已具备相关依赖的环境中复查，无需 GPU、权重或数据：

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
python -m kvrepack.run --list-methods
python -m kvrepack.run --method snapkv_kvrepack --limit 2 --dry-run
python scripts/check_sources.py
python scripts/check_dispatch.py > /tmp/kvrepack-dispatch.json
python scripts/check_call_graph.py --dispatch-report /tmp/kvrepack-dispatch.json --output /tmp/kvrepack-call-graph.json
python scripts/check_cpu_compression.py > /tmp/kvrepack-cpu-compression.json
python -m unittest discover -s tests -v
```

保留原工作区时，可给来源检查和 CPU 差分检查显式加 `--source-root <原工作区>`。其他机器不需要原目录即可执行其余检查。缺依赖导致测试跳过时，应记录缺失项，不能沿用本机“15 项完成”的结论。模型与数据通过使用者准备的路径或 `HF_HOME` 配置；NextQA 的 NLTK 资源、VATEX 的 Java/METEOR、VideoChatGPT 外部 judge 需分别准备，评分不等于生成。

## 后续优先级

1. **先处理两项已确认问题。** 明确修复与旧源码等价性的边界，分别加入跨 prompt 和分块边界回归，再修改最小范围实现。更新来源变换、哈希、测试和报告；不要仅更新 manifest 来隐藏行为变化。
2. **在目标机器做安装与 CPU 复查。** 按固定依赖检查包资源、vendor、来源文件和 CLI；重新记录环境与实际通过/跳过项。GitHub 打包只纳入项目及交接/审计证据，SSH 与其他凭证留在项目外。
3. **若扩展 LLaVA 四种改进，单独迁入。** 比较 port 与当前 Qwen 的 RoPE、预算、score 和生命周期；每种方法固定唯一实现，与其基线按同参数对照。当前命名统一没有完成这项迁移。
4. **继续完成未覆盖的 GPU 验收。** 对需要自定义内核的方法，先在目标环境构建 `kvrepack_kernel`；补齐统一 CLI/adapter、其余模型和方法的1–2例对照，再进行 limit=200 的任务检查。本轮两种方法的隔离 worker 对照不能替代这些检查；质量与数值通过后才测性能。
5. **发布前整理可复现基准与许可。** 冻结模型/数据标识、样本、参数、代码哈希和结果。项目级许可证与继承代码归属仍待仓库所有者确认，见 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)；当前不能称已全部复现或已具备完整发布验证。

# KVRepack 工作约定

开始工作时阅读 `docs/HANDOFF.md`、`README.md`、`provenance/milebench_validation.json` 和首版 `provenance/cpu_audit_summary.json`，再核对当前源码与用户的新指示。

- 视频注册范围为 Qwen2.5-VL 13种及 LLaVA-OneVision 6种基线；MileBench图片范围为两类模型各6基线+4改进，新增LLaVA改进仅通过图片能力表选择。
- 每种模型/方法只保留一种实现；保持对应基线与 KVRepack 改进的预算与参数语义可对照。
- 运行接口使用 KVRepack 命名；历史来源路径和原始 SHA 保留原样。
- LLaVA SparseMM 跨prompt预算已独立修复并做CPU回归；AsymKV分块边界问题仍未修复。旧审计是历史证据，当前状态见MileBench/LLaVA文档。
- 图片默认Qwen3B与512个生成token，RATIO独立；LLaVA共享目标budget_fixed_ratio=0.1而当前来源为0.5。配置差异明确记录，不将CPU迁移称为历史分数复现。
- Spot/MMCoQA专用Random-head消融参考6c65bf8，只干预merge-head mask。2026-11-08用户已授权GPU并确认官方MileBench子集全量：Qwen7B、Hybrid/Snap两方法、random seeds0/1及similarity，4组×200条×3条件=2400次正式生成已全部完成。各200条由MileBench作者从源任务抽样构成（[论文§3.2](https://arxiv.org/pdf/2404.18532)）；本项目--all-samples读取全部标注，不设条目cap，与显式--limit互斥。本轮明确--all-samples --head-seeds 0 1；CLI默认三seed仅为历史参数兼容，不代表本轮范围，不启动seed2。
- 当前正式根为outputs/ablations/random_head_qwen7b_all_seed01_v2_20261108，四组均complete，最终独立审计及完整核分passed；2400次生成全部EOS，无512-token上限截断。GPU已释放，不启动重复运行。最新CPU测试161项：160通过、1跳过；GPU结论仅覆盖本轮Qwen7B两方法/两官方子集，不能泛化为全部方法或性能验证。结果见 [Qwen7B结果](docs/MILEBENCH_RANDOM_HEAD_QWEN7B_RESULTS.md)、[精简包审计](provenance/random_head_qwen7b_all_seed01_20261108/audit/audit.json)、[结果CSV](provenance/random_head_qwen7b_all_seed01_20261108/audit/results.csv)。
- 原控制器在Hybrid/MMCoQA第2个生成样本（原sample_id=18）后置审计失败：将56个选中头误等同56个有效合并头。修复恢复参考Hybrid允许自然缺调用/noop且每条件有效合并>0；Snap仍要求34头有效匹配，预算、算法、随机mask均未改。不要为满足有效头数而改选头或预算。
- 原失败根outputs/ablations/random_head_qwen7b_all_seed01_20261108保持原样；完整Hybrid/Spot原合同/样本/报告按字节拷贝复用，未重复600次生成。v2根reuse_manifest.json记录SHA，source_archives/first_controller冻结旧12份绑定源码。保留绑定源码和manifest用于追溯；最终独立审计已显式提供--source-archive <v2根>/source_archives/first_controller，后续离线复查仍须提供该归档并写入新目录。
- 修改核心时同时维护来源变换、必要回归检查与报告，不通过只刷新哈希掩盖行为变化。
- GPU 实验须有用户明确授权。已完成的 CPU 检查不能替代完整模型、CUDA、生成质量或性能验证。
- GitHub 上传包含项目、交接和审计文档；SSH 私钥及其他凭证保存在仓库外。

复查命令、已验证范围和后续优先级见 `docs/HANDOFF.md`。缺依赖导致跳过时如实记录。

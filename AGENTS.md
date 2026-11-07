# KVRepack 工作约定

开始工作时阅读 `docs/HANDOFF.md`、`README.md`、`provenance/milebench_validation.json` 和首版 `provenance/cpu_audit_summary.json`，再核对当前源码与用户的新指示。

- 视频注册范围为 Qwen2.5-VL 13种及 LLaVA-OneVision 6种基线；MileBench图片范围为两类模型各6基线+4改进，新增LLaVA改进仅通过图片能力表选择。
- 每种模型/方法只保留一种实现；保持对应基线与 KVRepack 改进的预算与参数语义可对照。
- 运行接口使用 KVRepack 命名；历史来源路径和原始 SHA 保留原样。
- LLaVA SparseMM 跨prompt预算已独立修复并做CPU回归；AsymKV分块边界问题仍未修复。旧审计是历史证据，当前状态见MileBench/LLaVA文档。
- 图片默认Qwen3B与512个生成token，RATIO独立；LLaVA共享目标budget_fixed_ratio=0.1而当前来源为0.5。配置差异明确记录，不将CPU迁移称为历史分数复现。
- Spot/MMCoQA专用Random-head消融使用Qwen7B、Hybrid/Snap两种改进、原选头+3局部seed；参考6c65bf8，只干预merge-head mask。入口默认只规划；当前用户要求GPU后续再开启，勿执行--execute。见docs/MILEBENCH_RANDOM_HEAD.md与provenance/milebench_random_head_validation.json。
- 修改核心时同时维护来源变换、必要回归检查与报告，不通过只刷新哈希掩盖行为变化。
- GPU 实验须有用户明确授权。已完成的 CPU 检查不能替代完整模型、CUDA、生成质量或性能验证。
- GitHub 上传包含项目、交接和审计文档；SSH 私钥及其他凭证保存在仓库外。

复查命令、已验证范围和后续优先级见 `docs/HANDOFF.md`。缺依赖导致跳过时如实记录。

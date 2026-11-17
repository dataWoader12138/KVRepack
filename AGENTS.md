# KVRepack 工作约定

2026-11-17当前图像任务已完成：`b9104d8`来源，Qwen7B、Spot/MMCoQA官方完整各200条，Hybrid/Snap各三聚合与独立原版LOOK-M/AsymKV，16组3200正式生成+16预热排除。E2E与全部原始证据已独立复核、CPU重评分并归档；GPU已释放，不自动重跑或扩大。结果和参数见`docs/STYLED_MILEBENCH_QWEN7B_E2E.md`，发布在`provenance/styled_milebench_20261117/final`。保持来源原预算与30份核心，独立Asym Spot未压缩、MM44/196原边界遗漏如实报告；旧图片分支和旧结果不改，已停止视频任务不恢复。

开始工作时阅读 `docs/HANDOFF.md`、`README.md` 和 `provenance/cpu_audit_summary.json`，再核对当前源码与用户的新指示。

- 当前支持 Qwen2.5-VL 的13种组合及 LLaVA-OneVision 的6种基线。四种已有 LLaVA 改进源码尚未迁入。
- 每种模型/方法只保留一种实现；保持对应基线与 KVRepack 改进的预算与参数语义可对照。
- 运行接口使用 KVRepack 命名；历史来源路径和原始 SHA 保留原样。
- 两项已复现问题尚未修复：LLaVA SparseMM 跨 prompt 预算复用、AsymKV 分块边界遗漏末 token。详情见 `docs/CPU_CALL_AUDIT.md`。
- 修改核心时同时维护来源变换、必要回归检查与报告，不通过只刷新哈希掩盖行为变化。
- GPU 实验须有用户明确授权。已完成的 CPU 检查不能替代完整模型、CUDA、生成质量或性能验证。
- GitHub 上传包含项目、交接和审计文档；SSH 私钥及其他凭证保存在仓库外。

复查命令、已验证范围和后续优先级见 `docs/HANDOFF.md`。缺依赖导致跳过时如实记录。

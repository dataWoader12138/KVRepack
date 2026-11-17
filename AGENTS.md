# KVRepack 工作约定

2026-11-17当前任务：用户已明确授权以`b9104d8`为基础，在Qwen2.5-VL-7B上跑Spot-the-Diff、MMCoQA官方MileBench子集全量（本机各200条）。HybridKV/SnapKV各配KVRepack、LOOK-M-styled、AsymKV-styled三种聚合，另跑独立原版LOOK-M/AsymKV，共16运行、3200正式生成。真实E2E覆盖图片读取/JPEG/processor/H2D/Fisher（适用时）/生成/文本解码，模型加载、预热、审计散列、写盘与评分分开统计。保持来源原预算，不擅自改成strict-total。独立Asym保留原驱动并报告实际缓存及MMCoQA44/196的已知末token边界；不跳样、不改prompt规避。旧图片分支和旧已完成结果不修改。正式进度以新输出根的experiment_state.json为准；运行期间禁止更改绑定src源码。协议见`docs/STYLED_MILEBENCH_QWEN7B_E2E.md`。

开始工作时阅读 `docs/HANDOFF.md`、`README.md` 和 `provenance/cpu_audit_summary.json`，再核对当前源码与用户的新指示。

- 当前支持 Qwen2.5-VL 的13种组合及 LLaVA-OneVision 的6种基线。四种已有 LLaVA 改进源码尚未迁入。
- 每种模型/方法只保留一种实现；保持对应基线与 KVRepack 改进的预算与参数语义可对照。
- 运行接口使用 KVRepack 命名；历史来源路径和原始 SHA 保留原样。
- 两项已复现问题尚未修复：LLaVA SparseMM 跨 prompt 预算复用、AsymKV 分块边界遗漏末 token。详情见 `docs/CPU_CALL_AUDIT.md`。
- 修改核心时同时维护来源变换、必要回归检查与报告，不通过只刷新哈希掩盖行为变化。
- GPU 实验须有用户明确授权。已完成的 CPU 检查不能替代完整模型、CUDA、生成质量或性能验证。
- GitHub 上传包含项目、交接和审计文档；SSH 私钥及其他凭证保存在仓库外。

复查命令、已验证范围和后续优先级见 `docs/HANDOFF.md`。缺依赖导致跳过时如实记录。

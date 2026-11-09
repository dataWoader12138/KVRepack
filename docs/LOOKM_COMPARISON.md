# LookM视频测试：当前使用严格10%总prefill预算

**最新：严格10%两任务各200条重跑已完成，结果及CSV见[正式结果页](LOOKM_STRICT_RESULTS_20261109.md)。** 400条真实缓存和独立原任务评分复核均通过；仅重跑LookM，旧legacy和已停止任务保持原状态。

用户已要求近期32位置也计入总额度。**当前唯一LookM实现已修为`C=floor(.1*N)`，recent与历史选择总和不再超过C。** 源码核对、边界处理及验证见[严格预算说明](LOOKM_STRICT_BUDGET.md)。

[旧400条结果](LOOKM_RESULTS_20261109.md)使用“10%＋32”，不能作为严格预算结果；新的严格两任务各200条现已完成，见上方正式结果链接。旧源码/报告/预测归档保留，不伪造新分数。新结果必须使用新输出目录和v2契约，禁止续入旧样本。

仅运行LookM的入口：

```bash
python scripts/run_lookm_video.py \
  --pretrained /path/to/Qwen2.5-VL-7B-Instruct \
  --hf-home /path/to/huggingface_cache \
  --output outputs/benchmarks/lookm_strict_200_REPRO
```

模型Qwen2.5-VL-7B，64帧，两个任务原始前200行；`--tasks nextqa_oe_test`或`--tasks vatex_test`可单选。生成greedy、seed42、repetition penalty1.05、NextQA上限128/VATEX上限64。默认两个任务各200，`--limit 1`只做各1条检查；完成单任务后自动调用CPU原任务评分。这里只启动LookM。

原双方法控制器及Hybrid重跑已按用户要求终止，保留119条部分记录、状态stopped_by_user/resume_authorized=false，不自动恢复。此前RoPE实验同样保持停止。原LookM400条已完成记录为历史，不因预算修复而删除或直接改写其分数。

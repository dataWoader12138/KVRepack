# 方法选择与默认配置

每个模型/公开方法组合只保留一条实现路径。来源脚本是证据标签，运行时不读取旧目录。默认值来自所选脚本当前配置，不能直接视为历史最优参数。完整机器可读快照见 [method_registry.json](../provenance/method_registry.json)。

## qwen2_5_vl

| 公开方法 | 内部 METHOD | 配对基线 | 默认模型 | 默认 limit | 参考脚本 |
|---|---|---|---|---|---|
| `fullkv` | `fullkv` | — | Qwen2.5-VL-7B-Instruct | 200 | `run_qwen7b_fullkv_vatex_nextqa_limit200.sh` |
| `hybridkv` | `hybridkv` | — | Qwen2.5-VL-7B-Instruct | 200 | `qwen_ratio_hybridkv.sh` |
| `snapkv` | `snapkv` | — | Qwen2.5-VL-7B-Instruct | 200 | `qwen_ratio.sh` |
| `sparsemm` | `sparsemm` | — | Qwen2.5-VL-7B-Instruct | 200 | `qwen_ratio.sh` |
| `madakv` | `madakv` | — | Qwen2.5-VL-7B-Instruct | 200 | `qwen_ratio.sh` |
| `lookm` | `lookm` | — | Qwen2.5-VL-7B-Instruct | 200 | `qwen_ratio.sh` |
| `randomkv` | `randomkv` | — | Qwen2.5-VL-3B-Instruct | 20000 | `qwen_ratio_randomkv_3b.sh` |
| `asymkv` | `asymkv_official` | — | Qwen2.5-VL-3B-Instruct | 200 | `qwen_asymkv_official_lmms.sh` |
| `hybridkv_kvrepack` | `hybridkv_kvrepack` | hybridkv | Qwen2.5-VL-3B-Instruct | 10000 | `qwen_ratio_hybrid_decouplekv_3b.sh` |
| `snapkv_kvrepack` | `snapkv_kvrepack` | snapkv | Qwen2.5-VL-3B-Instruct | 200 | `qwen_ratio_snapkv_decouplekv_simhead_3b.sh` |
| `sparsemm_kvrepack` | `sparsemm_kvrepack` | sparsemm | Qwen2.5-VL-3B-Instruct | 200 | `qwen_ratio_sparsemm_decouplekv_simhead_3b.sh` |
| `madakv_kvrepack` | `madakv_kvrepack` | madakv | Qwen2.5-VL-3B-Instruct | 20000 | `qwen_ratio_madakv_decouplekv_simhead_3b.sh` |
| `lookm_kvrepack` | `lookm_kvrepack` | lookm | Qwen2.5-VL-3B-Instruct | 9000 | `qwen_ratio_lookm_decouplekv_3b.sh` |

参考脚本栏保留原工作区中的真实文件名，其中的历史命名不表示本项目仍接受旧 METHOD 或环境变量名；当前代码命名见 [命名对照](REPACKING.md#命名对照)。

## llava_ov

| 公开方法 | 内部 METHOD | 配对基线 | 默认模型 | 默认 limit | 参考脚本 |
|---|---|---|---|---|---|
| `fullkv` | `fullkv` | — | llava-onevision-qwen2-7b-ov-hf | 200 | `llava_ov_ratio.sh` |
| `snapkv` | `snapkv` | — | llava-onevision-qwen2-7b-ov-hf | 200 | `llava_ov_ratio.sh` |
| `sparsemm` | `sparsemm` | — | llava-onevision-qwen2-7b-ov-hf | 200 | `llava_ov_ratio.sh` |
| `madakv` | `madakv` | — | llava-onevision-qwen2-7b-ov-hf | 200 | `llava_ov_ratio.sh` |
| `lookm` | `lookm` | — | llava-onevision-qwen2-7b-ov-hf | 200 | `llava_ov_ratio.sh` |
| `hybridkv` | `hybridkv` | — | llava-onevision-qwen2-7b-ov-hf | 200 | `llava_ov_hybridkv.sh` |

上述 LLaVA 基线已经纳入。工作区另有四种 LLaVA 改进源码（SnapKV、SparseMM、MadaKV、HybridKV），尚未迁入本版；这些组合会在 CPU 规划阶段拒绝。SparseMM/MadaKV 的基线分支来自同源 dispatcher，原 ratio 脚本当前数组没有激活这两项。

## 默认值与公平对照

帧数默认为64、batch size为1，生成temperature=0。Qwen可以显式选择3B或7B，LLaVA固定OV-7B。每组基线和改进必须显式使用同一模型、任务、limit、帧数和预算，不能混用各脚本的默认模型/limit。

列出每个模型/方法的精确默认环境变量、实现与覆盖白名单：

```bash
PYTHONPATH=src python -m kvrepack.run --list-methods
```

原Snap改进入口当前默认merge ratio为0；需要显式启用合并时：

```bash
PYTHONPATH=src python -m kvrepack.run --model qwen2_5_vl --method snapkv_kvrepack \
  --task nextqa_oe_test --pretrained Qwen/Qwen2.5-VL-3B-Instruct --limit 200 \
  --set SNAPKV_SIMHEAD_MERGE_RATIO=0.30 --dry-run
```

LLaVA 基线示例：

```bash
PYTHONPATH=src python -m kvrepack.run --model llava_ov --method hybridkv \
  --task nextqa_oe_test --limit 2 --dry-run
```

原有环境变量不会静默覆盖方法设置，只有显式 `--set KEY=VALUE` 生效。未知参数、跨方法参数、重复参数、非法数值在启动前报错。五种改进方法的内部 METHOD 与公开名字一致；原环境变量名中的 `DECOUPLE` 已统一为 `KVREPACK`，参数语义和值保持不变，覆盖参数使用 `--list-methods` 列出的当前名称。

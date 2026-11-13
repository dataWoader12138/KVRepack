# NExTQA merge comparison

Original first 200 inputs per arm; Qwen2.5-VL-7B, 64 frames, strict total-active-prefill 10% budget.

| Backbone | Variant | N | WUPS | Full E2E (s) | Model E2E (s) | P50 (s) | P95 (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| snapkv | native | 200 | 33.192028 | 3.200852 | 1.490850 | 2.825479 | 5.339109 |
| snapkv | kvrepack | 200 | 32.894385 | 3.253997 | 1.542912 | 2.902323 | 5.299678 |
| snapkv | lookm | 200 | 32.049940 | 3.278505 | 1.566501 | 2.895809 | 5.354654 |
| snapkv | asymkv | 200 | 30.610048 | 4.795997 | 3.087641 | 4.456035 | 7.486852 |
| hybridkv | native | 200 | 32.648727 | 3.302653 | 1.585962 | 2.929506 | 5.420880 |
| hybridkv | kvrepack | 200 | 33.230672 | 3.394233 | 1.684881 | 3.023416 | 5.573138 |
| hybridkv | lookm | 200 | 33.201505 | 3.476015 | 1.726665 | 3.132840 | 5.609358 |
| hybridkv | asymkv | 200 | 31.857199 | 5.051236 | 3.300154 | 4.681929 | 7.991823 |

Full E2E includes video read/decode, processor, H2D, Fisher where required, generation and text decoding, with synchronized GPU boundaries. Model E2E includes Fisher and generation. Model load, warmup, diagnostic hashing, writes and scoring are excluded.

- One measurement per original input/arm; execution order rotates within each paired worker.
- Four workers are measured serially; this is not repeated steady-state timing.
- LOOK-M/AsymKV rows transplant only merging at the KVRepack insertion point; they are not standalone methods.
- HybridKV native and merge arms can have different actual slot counts due to heterogeneous chunk allocation; differences are reported.
- The active-cache position budget is distinct from total GPU memory and HybridKV backing storage.

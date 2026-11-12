# NExTQA merge comparison

Original first 1 inputs per arm; Qwen2.5-VL-7B, 64 frames, strict total-active-prefill 10% budget.

| Backbone | Variant | N | WUPS | Full E2E (s) | Model E2E (s) | P50 (s) | P95 (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| snapkv | native | 1 | 100.000000 | 2.878325 | 1.282545 | 2.878325 | 2.878325 |
| snapkv | kvrepack | 1 | 100.000000 | 2.820245 | 1.322898 | 2.820245 | 2.820245 |
| snapkv | lookm | 1 | 100.000000 | 2.975285 | 1.347950 | 2.975285 | 2.975285 |
| snapkv | asymkv | 1 | 100.000000 | 4.216505 | 2.610893 | 4.216505 | 4.216505 |
| hybridkv | native | 1 | 100.000000 | 2.959463 | 1.330217 | 2.959463 | 2.959463 |
| hybridkv | kvrepack | 1 | 100.000000 | 3.084119 | 1.450776 | 3.084119 | 3.084119 |
| hybridkv | lookm | 1 | 100.000000 | 3.118716 | 1.489201 | 3.118716 | 3.118716 |
| hybridkv | asymkv | 1 | 100.000000 | 4.322769 | 2.732395 | 4.322769 | 4.322769 |

Full E2E includes video read/decode, processor, H2D, Fisher where required, generation and text decoding, with synchronized GPU boundaries. Model E2E includes Fisher and generation. Model load, warmup, diagnostic hashing, writes and scoring are excluded.

- One measurement per original input/arm; execution order rotates within each paired worker.
- Four workers are measured serially; this is not repeated steady-state timing.
- LOOK-M/AsymKV rows transplant only merging at the KVRepack insertion point; they are not standalone methods.
- HybridKV native and merge arms can have different actual slot counts due to heterogeneous chunk allocation; differences are reported.
- The active-cache position budget is distinct from total GPU memory and HybridKV backing storage.

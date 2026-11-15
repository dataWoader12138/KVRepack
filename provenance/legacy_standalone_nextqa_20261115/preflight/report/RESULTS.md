# Original standalone LOOK-M / AsymKV: NExTQA

Qwen2.5-VL-7B, original first 1 rows, 64 frames. Original method budgets and complete algorithms.

| Method | N | WUPS | Timed response E2E mean (s) | P50 (s) | P95 (s) | Formal loop / input (s) |
|---|---:|---:|---:|---:|---:|---:|
| lookm | 1 | 100.000000 | 4.107358 | 4.107358 | 4.107358 | 4.305438 |
| asymkv | 1 | 100.000000 | 8.774524 | 8.774524 | 8.774524 | 8.933534 |

Per-input full E2E: video reading/decoding, processor, H2D, complete original method generation (including AsymKV Fisher where required), and text decoding. Model load and after-timer fingerprints/audits/writes/scoring are excluded.

Formal loop wall time includes the measured input E2E intervals plus intervening record/audit/log/write overhead. Report this separately from timed E2E.

- Original standalone methods, not fixed-group merge transplants.
- LOOK-M uses original historical budget plus 32 recent; AsymKV uses its own original budget and algorithm, without an invented +32 extension.
- Original generation behavior is method-specific: LOOK-M uses HF greedy decoding with repetition penalty1.05; official AsymKV uses its native argmax loop without that HF penalty.
- Single latency observation per input; no repeated steady-state timing or significance claim.
- Historical response-loop averages and these two reported timing scopes must not be silently interchanged.

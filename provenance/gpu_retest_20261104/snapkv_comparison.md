# GPU worker artifact comparison

Overall status: **passed**. Comparison runs on CPU only.

Score tolerance: atol=0.0, rtol=0.0. Matching signed infinities count as equal; any NaN fails.

| Record | Config | Input hashes | Tokens | Text | Prefill shapes | Prefill K/V SHA | Final shapes | Final K/V SHA | Score dtype | Full-vocab max abs error | Scores | Status |
|---|---|---|---|---|---|---|---|---|---|---:|---|---|
| nextqa_oe_test:0 | passed | passed | passed | passed | passed | passed | passed | passed | equal | 0.0 | passed | passed |
| nextqa_oe_test:1 | passed | passed | passed | passed | passed | passed | passed | passed | equal | 0.0 | passed | passed |
| vatex_test:0 | passed | passed | passed | passed | passed | passed | passed | passed | equal | 0.0 | passed | passed |
| vatex_test:1 | passed | passed | passed | passed | passed | passed | passed | passed | equal | 0.0 | passed | passed |

The JSON report includes every generation step’s maximum absolute error, complete tensor shapes, record/config differences, source artifact hashes and unchecked metadata.

Prefill cache snapshots are mandatory: missing or empty snapshots fail. Both prefill and final cache shapes/stored lengths are compared. Per-layer K/V tensor content hashes are compared only when all hashes are present and valid SHA-256 values; malformed or partial hashes fail, and absent hashes remain explicitly not checked.

A missing/empty score artifact is explicitly not checked and makes the result incomplete. Equal generated tokens alone cannot make the full comparison pass.

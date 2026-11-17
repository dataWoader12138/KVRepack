Complete Qwen2.5-VL-7B styled image experiment: 16 conditions x 200 official samples.
The tar contains every raw sample, warmup and run contract/environment/final state, without images or weights.
Verify recorded_runs.tar.gz SHA256 against release_manifest.json before extraction.
CPU-only rescore (same original annotation JSON bytes; image files are unnecessary):
  python -m pip install -r cpu_rescore/requirements.txt
  mkdir unpacked
  tar -xzf recorded_runs.tar.gz -C unpacked
  python cpu_rescore/scripts/export_styled_milebench.py --rescore-extracted unpacked --data-root /path/to/milebench --output new_cpu_scores
Original absolute paths remain in unmodified contracts as provenance; --data-root only relocates annotations.
Rescoring checks all archived file hashes, exact IDs and annotation hashes, and computes original ROUGE-L F / Accuracy.
It does not rerun inference, remeasure latency, or require the original GPU/kernel/model installation.
Quality scores use the full official 200-item annotation of each task; separate warmups are excluded.
Standalone AsymKV Fisher stays within official generation time (separate Fisher is unmeasured).
Original cache policies, AsymKV chunk-boundary omissions and early EOS behavior are preserved and reported.
E2E includes image I/O, preprocessing, H2D, applicable Fisher, generation and text decoding; model load is separate.

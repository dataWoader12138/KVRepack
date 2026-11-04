# SnapKV-KVRepack Random-head pilot evidence

Two original test prefixes of 20 inputs, Qwen2.5-VL-7B, similarity and global random seeds0/1/2: 160 completed generations. See ../../docs/RANDOM_HEAD_ABLATION_RESULTS_20261104.md for interpretation.

`generations/` preserves predictions, token IDs, all numerical audits, original run states/configs, and source/input fingerprints. Public records remove prompt/text segments and local input video/Arrow paths from reference_metadata; original local JSONL hashes are in index.json. No tensor, image/video, weights, SSH material or dataset is included. Original unabridged records remain under outputs locally.

`input_contracts/` freezes the input preparation contract. `report/` contains official local task scores, three-seed statistics, paired source-video WUPS intervals, audits and plots. Scores can be recalculated against an identically prepared local reference Arrow using the project head_selection_report CLI with --run-root pointing at generations/.

`failed_preparation/` records a CPU-only overlap: two input preparers briefly wrote the same cache; companion-file checking stopped VATEX before generation. The duplicate was terminated, all16 retained input records were independently revalidated, and the final20 inputs and all4 VATEX conditions were processed serially. No generated record or sample was discarded or replaced.

Local orchestration helpers are retained for provenance; portable commands are in the protocol document. They must not be launched against the archived paths.

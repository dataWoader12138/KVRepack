"""Continue this fixed matrix, preserving the completed Spot run byte for byte.

The first controller stopped at a reporting gate, after its complete Spot run.
No contract, prediction, sample or report in that controller is rewritten.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_milebench_random_heads import execute_matrix
from audit_milebench_random_head_run import audit_one, file_sha
from kvrepack.experiments.random_head_images import build_matrix
from kvrepack.experiments.runner import run_plan
from kvrepack.ablations.milebench_head_selection_report import summarize_run

OLD = ROOT / 'outputs/ablations/random_head_qwen7b_all_seed01_20261108'
OUT = ROOT / 'outputs/ablations/random_head_qwen7b_all_seed01_v2_20261108'
SOURCE = OLD / 'hybridkv_kvrepack/Spot-the-Diff'
SNAPSHOT = OLD / 'source_snapshot'
MODEL = '/root/autodl-tmp/HybridKV-opensource/data/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'


def reuse(plan):
    config = plan.resolved_config
    return (config['public_method'], config['task']) == ('hybridkv_kvrepack', 'Spot-the-Diff')


def launch(plan):
    if not reuse(plan):
        return run_plan(plan)
    evidence, scores, stats = audit_one(SOURCE, 'hybridkv_kvrepack', 'Spot-the-Diff', [SNAPSHOT])
    shutil.copytree(SOURCE, plan.output)
    archive = OUT / 'source_archives/first_controller'
    shutil.copytree(SNAPSHOT, archive)
    manifest = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Reuse fully completed and independently rescored Hybrid/Spot without repeating 600 GPU generations. Continue remaining three full runs after correcting a Hybrid post-generation audit stricter than reference.',
        'source_run': str(SOURCE), 'destination_run': str(plan.output),
        'original_controller': str(OLD / 'controller_state.json'),
        'original_controller_sha256': file_sha(OLD / 'controller_state.json'),
        'source_archive': str(archive),
        'contract_sha256': evidence['contract_sha256'],
        'sample_files': evidence['sample_files'],
        'report_files_sha256': evidence['report_files_sha256'],
        'scores': scores, 'generation_stats': stats,
        'continuation_driver_sha256': file_sha(__file__),
        'reuse_policy': 'Exact bytes copied; original contract, source hashes, predictions and reports unchanged. Current CPU validator and unchanged original scorer recompute coverage, masks, work and scores. Archived original executed source hashes verified explicitly.',
    }
    (OUT / 'reuse_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return 0


def report(run, output):
    if Path(run) == OUT / 'hybridkv_kvrepack/Spot-the-Diff':
        return json.loads((Path(run) / 'report/summary.json').read_text())
    return summarize_run(run, output)


if __name__ == '__main__':
    plans = build_matrix('/root/autodl-tmp/HybridKV-opensource/data/milebench', OUT,
                         pretrained=MODEL,
                         hf_home='/root/autodl-tmp/HybridKV-opensource/data/huggingface_cache',
                         head_seeds=(0, 1), all_samples=True)
    print(json.dumps(execute_matrix(plans, OUT, runner=launch, reporter=report), ensure_ascii=False, indent=2))

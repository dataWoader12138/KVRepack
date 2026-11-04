#!/usr/bin/env python3
"""Launch the unmodified KVRepack CLI through an external, resumable trace shim."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("snapkv_kvrepack", "madakv_kvrepack"), required=True)
    p.add_argument("--task", choices=("nextqa_oe_test", "vatex_test"), required=True)
    p.add_argument("--limit", default="200", help="Positive integer or 'all'")
    p.add_argument("--implementation", choices=("original", "repack"), required=True)
    p.add_argument("--output", type=Path, required=True, help="Persistent trace root; each invocation gets a fresh official attempt directory")
    p.add_argument("--inventory", type=Path, default=Path("/tmp/gpu_retest_data_inventory.json"))
    p.add_argument("--repo", type=Path, default=HERE.parent / "KVRepack")
    p.add_argument("--source-root", type=Path, default=HERE.parent / "HybridKV-artifact")
    p.add_argument("--device", default="0")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    repo, original, output = args.repo.resolve(), args.source_root.resolve(), args.output.resolve()
    inventory = json.loads(args.inventory.read_text())
    data = next(row for row in inventory["datasets"] if row["task"] == args.task)
    count = int(data["rows"]) if args.limit == "all" else int(args.limit)
    if not 0 < count <= int(data["rows"]):
        p.error("--limit must be positive and no greater than the actual cached split")
    model = next(row for row in inventory["models"] if row["repo_id"] == "Qwen/Qwen2.5-VL-3B-Instruct")
    sys.path.insert(0, str(repo / "src"))
    from kvrepack.experiments.runner import build_plan, run_plan

    # These explicit settings match the audited historical best profile per task.
    prefix = "SNAPKV" if args.method.startswith("snapkv") else "MADAKV"
    gate = "0.7" if prefix == "SNAPKV" else "0.5"
    cr = "0.9" if prefix == "SNAPKV" else ("0.5" if args.task == "nextqa_oe_test" else "0.7")
    overrides = (f"{prefix}_SIMHEAD_MERGE_RATIO={gate}", f"{prefix}_KVREPACK_COMPRESS_RATIO={cr}", "K_DIFF_ALPHA=2")
    hf_home = Path(inventory["recommended_HF_HOME"]).resolve()
    for key, value in {
        "HF_HOME": str(hf_home), "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HF_HUB_CACHE": str(hf_home / "hub"), "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "NLTK_DATA": inventory["scoring_resources"]["nltk_resource_root"],
    }.items():
        os.environ[key] = value
    java_dir = Path(inventory["scoring_resources"]["java_path"]).parent
    os.environ["PATH"] = str(java_dir) + os.pathsep + os.environ.get("PATH", "")
    attempt = 1
    while (output / f"attempt_{attempt:04d}").exists() or (output / f"attempt_{attempt:04d}.trace_config.json").exists():
        attempt += 1
    attempt_dir = output / f"attempt_{attempt:04d}"
    plan = build_plan(args.method, task=args.task, pretrained=model["snapshot_path"], model_type="qwen2_5_vl-3b",
                      frames=64, limit=count, budget_ratio=0.1, device=args.device, output=attempt_dir,
                      overrides=overrides, repo_root=repo)
    # Intentionally source-neutral and limit-neutral: a 200-row run can resume to all rows.
    common = {"model": "Qwen2.5-VL-3B-Instruct", "model_snapshot": model["snapshot"], "method": args.method,
              "task": args.task, "dataset_arrow_sha256": data["arrow_sha256"], "frames": 64,
              "dtype": "bfloat16", "attention": "flash_attention_2", "batch_size": 1,
              "algorithm_env": plan.resolved_config["algorithm_env"], "budget_ratio": 0.1,
              "min_pixels": 256 * 28 * 28, "max_pixels": 1605632, "use_cache": True,
              "max_new_tokens": 128 if args.task == "nextqa_oe_test" else 64,
              "generation": {"do_sample": False, "temperature": None, "top_p": None, "num_beams": 1},
              "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234}}
    config_sha = hashlib.sha256(json.dumps(common, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    contract = {"schema_version": 1, "implementation": args.implementation, "repo_root": str(repo),
                "source_root": str(original) if args.implementation == "original" else str(repo),
                "config": common, "config_sha256": config_sha,
                "helper_sha256": {name: sha(HERE / name) for name in ("trace_eval.py", "launch_traced.py")}}
    runtime = {**contract, "trace_path": str(output / "trace.jsonl"), "requested_limit": count,
               "created_utc": datetime.now(timezone.utc).isoformat()}
    command = list(plan.command)
    target = command.index("lmms_eval")
    if command[target - 1] != "-m":
        raise ValueError("Unexpected official launch command; refusing to rewrite it")
    command[target - 1:target + 1] = [str(HERE / "trace_eval.py")]
    trace_config_path = output / f"attempt_{attempt:04d}.trace_config.json"
    provenance = {**plan.provenance, "trace_contract": contract, "trace_sources_sha256": contract["helper_sha256"],
                  "trace_scope": "Real CLI/adapter; old compression substituted only for original control; tracing overhead invalidates performance timing."}
    if args.implementation == "original":
        reference_files = [original / "hybridkv" / name for name in (
            "monkeypatch.py", "qwen_model.py", "qwen_model_croatt.py",
            f"qwen_model_{prefix.lower()}_decouplekv.py")]
        provenance["source_policy"] = (
            "Controlled reference: current checkout runner/adapter/tasks and explicitly selected original artifact compression. "
            "The normal checkout-only algorithm policy is intentionally overridden for this reference process.")
        provenance["reference_source_files"] = [{"path": str(path), "sha256": sha(path)} for path in reference_files]
    plan = replace(plan, command=tuple(command), env={**plan.env, "KVREPACK_TRACE_CONFIG": str(trace_config_path)},
                   provenance=provenance)
    if args.dry_run:
        print(json.dumps({"trace_contract": contract, "requested_limit": count, "plan": plan.to_dict()}, indent=2, ensure_ascii=False))
        return 0
    if output.exists() and not (output / "trace_contract.json").exists():
        if any(output.iterdir()):
            raise ValueError("Nonempty output has no trace contract; refusing unsafe reuse")
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "trace_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Trace contract/config/source identity changed; select another output")
    else:
        atomic_json(contract_path, contract)
    atomic_json(trace_config_path, runtime)
    print(json.dumps({"implementation": args.implementation, "task": args.task, "method": args.method,
                      "limit": count, "trace": runtime["trace_path"], "attempt": str(attempt_dir)}, ensure_ascii=False), flush=True)
    return run_plan(plan)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"launch_traced: {error}", file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
"""Read local CLI/cache metadata and CPU Java prerequisites; never launch inference.

This helper lives outside the published project. It does not certify a full run:
cache metadata does not decode every video or exercise model/adapter generation.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent / "KVRepack")
    parser.add_argument("--hf-home", type=Path, default=Path(__file__).resolve().parent.parent / "HybridKV-artifact/scripts/data/huggingface_cache")
    parser.add_argument("--java", type=Path, default=Path(sys.prefix) / "lib/jvm/bin/java")
    parser.add_argument("--java-smoke", action="store_true", help="Execute local PTBTokenizer and METEOR jar SCORE protocol on CPU")
    args = parser.parse_args()
    root, hf_home = args.repo.resolve(), args.hf_home.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def deny_network(event, values):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise RuntimeError("Network access prohibited by CPU readiness helper")

    sys.addaudithook(deny_network)
    sys.path.insert(0, str(root / "src"))
    from kvrepack.experiments.runner import build_plan, verify_manifest, verify_vendor_sources

    report = {
        "scope": {"gpu_launched": False, "model_loaded": False, "network_permitted": False,
                  "repo_modified": False, "dataset_load_exercised": False,
                  "video_decoding_exercised": False},
        "repo": str(root), "python": sys.executable, "hf_home": str(hf_home),
        "core_verified_targets": len(verify_manifest(root)["verified_targets"]),
        "vendor_verified_files": len(verify_vendor_sources(root)["verified_targets"]),
        "tasks": [], "nltk_download_calls": [],
        "notes": ["Metadata and CPU Java checks do not establish full adapter/CLI completion.",
                  "Full split counts must be explicitly passed as --limit; registry defaults are smaller.",
                  "HF offline flags do not disable nltk.download's index request; cached resources may let the task continue after a downloader error.",
                  "VATEX doc_to_target is the literal 'answer', not the enCap column; current scoring reads enCap directly, so this is misleading sample target metadata rather than the caption metric reference."],
    }
    target_sources = (("nextqa_oe_test", "lmms-lab___n_ex_tqa/OE", 9178),
                      ("vatex_test", "lmms-lab___vatex/vatex_test", 4478))
    model_base = hf_home / "hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
    model_revision = (model_base / "refs/main").read_text().strip()
    model_path = model_base / "snapshots" / model_revision
    for task, cache_name, expected in target_sources:
        matches = sorted((hf_home / "datasets" / cache_name).glob("*/**/dataset_info.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one cache metadata file for {task}; found {len(matches)}")
        metadata = json.loads(matches[0].read_text())
        count = metadata["splits"]["test"]["num_examples"]
        if count != expected:
            raise ValueError(f"Unexpected current split count for {task}: {count} != {expected}")
        output = Path(__file__).resolve().parent / (task + "_snapkv_kvrepack_full")
        plan = build_plan("snapkv_kvrepack", task=task, pretrained=str(model_path),
                          model_type="qwen2_5_vl-3b", frames=64, limit=count,
                          budget_ratio=0.1, output=output,
                          overrides=("SNAPKV_SIMHEAD_MERGE_RATIO=0.7",
                                     "SNAPKV_KVREPACK_COMPRESS_RATIO=0.9", "K_DIFF_ALPHA=2"),
                          environ={"HF_HOME": str(hf_home)}, repo_root=root)
        official = [sys.executable, "-m", "kvrepack.run", "--method", "snapkv_kvrepack",
                    "--task", task, "--pretrained", str(model_path), "--model-type", "qwen2_5_vl-3b",
                    "--frames", "64", "--limit", str(count), "--budget-ratio", "0.10",
                    "--device", "0", "--output", str(output),
                    "--set", "SNAPKV_SIMHEAD_MERGE_RATIO=0.7", "--set", "SNAPKV_KVREPACK_COMPRESS_RATIO=0.9",
                    "--set", "K_DIFF_ALPHA=2"]
        report["tasks"].append({"task": task, "rows": count, "metadata": str(matches[0]),
                                "features": sorted(metadata["features"]),
                                "algorithm_env": plan.resolved_config["algorithm_env"],
                                "official_command": shlex.join(official),
                                "resolved_child_command": shlex.join(plan.command)})
    source = root / "vendor/lmms-eval/lmms_eval/tasks/nextqa/utils.py"
    for node in ast.walk(ast.parse(source.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "nltk"
                and node.func.attr == "download"):
            report["nltk_download_calls"].append({"source": str(source), "line": node.lineno,
                                                  "resource": ast.literal_eval(node.args[0])})
    import nltk
    resources = ["taggers/averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger_eng",
                 "tokenizers/punkt", "tokenizers/punkt_tab", "corpora/wordnet", "corpora/omw-1.4"]
    report["nltk_resources"] = {}
    for name in resources:
        try:
            found = nltk.data.find(name)
        except LookupError:
            found = nltk.data.find(name + ".zip")
        report["nltk_resources"][name] = str(found)
    site = Path(nltk.__file__).resolve().parent.parent
    coco = site / "pycocoevalcap"
    report["java"] = {"default_PATH_java": shutil.which("java"), "explicit_java": str(args.java),
                      "exists": args.java.is_file(), "smoke": []}
    report["required_environment"] = {"HF_HOME": str(hf_home), "HF_DATASETS_CACHE": str(hf_home / "datasets"),
                                      "HF_HUB_CACHE": str(hf_home / "hub"),
                                      "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
                                      "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                      "NLTK_DATA": "/root/nltk_data", "PATH_prefix": str(args.java.parent),
                                      "PYTHONPATH": str(root / "src")}
    if args.java_smoke:
        java = str(args.java)
        commands = [
            ("PTBTokenizer", [java, "-cp", "stanford-corenlp-3.4.1.jar", "edu.stanford.nlp.process.PTBTokenizer",
                              "-preserveLines", "-lowerCase"], coco / "tokenizer", "A person walks.\n"),
            ("METEOR_SCORE", [java, "-jar", "-Xmx2G", "meteor-1.5.jar", "-", "-", "-stdio", "-l", "en", "-norm"],
             coco / "meteor", "SCORE ||| A person walks . ||| A person walks .\n"),
        ]
        for label, command, cwd, stdin in commands:
            result = subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=30, cwd=cwd)
            report["java"]["smoke"].append({"name": label, "returncode": result.returncode,
                                            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()})
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(f"Local Java smoke failed: {label}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

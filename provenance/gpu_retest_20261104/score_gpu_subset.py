#!/usr/bin/env python3
"""Offline CPU task metrics for the four original/repack GPU subset runs.

Run with the existing hybridkv Python environment. This reads Arrow references
and JSON predictions only: no model, tensor checkpoint, GPU, API, or download.
The default subset is doc_ids 0 and 1 of each task, not the historical 200 docs.
"""

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import urllib.request
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deny_network(*args, **kwargs):
    raise RuntimeError("Offline subset scorer: Python networking/download is disabled")


def offline_environment(java_home):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ.setdefault("NLTK_DATA", "/root/nltk_data")
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ["PATH"]
    if not (java_home / "bin/java").is_file():
        raise FileNotFoundError(java_home / "bin/java")
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    urllib.request.urlopen = deny_network
    urllib.request.urlretrieve = deny_network


def load_source_functions(path, names, namespace):
    """Execute unchanged function ASTs without task-module import side effects."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing/duplicate source functions in {path}: {names}")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return {
        "path": str(path), "sha256": sha256(path),
        "unchanged_function_ast_sha256": {
            node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in selected
        },
    }


def setup_scorers(project_root, temporary_dir):
    import nltk
    nltk.download = deny_network
    nltk.downloader.Downloader.download = deny_network
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    # Real task dependency, preserving pywsd's installed lemmatization behavior.
    from pywsd.utils import lemmatize_sentence
    import pandas as pd
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb

    # Use the installed tokenizer's original function bytecode and Stanford jar.
    # Only relocate its temporary files out of site-packages; flags and token
    # filtering remain unchanged. Memoization avoids repeated Java invocations.
    jar = Path(ptb.__file__).parent / ptb.STANFORD_CORENLP_3_4_1_JAR
    if not jar.is_file():
        raise FileNotFoundError(jar)
    original = ptb.PTBTokenizer.tokenize
    globals_copy = dict(original.__globals__)
    globals_copy["__file__"] = str(temporary_dir / "ptbtokenizer.py")
    globals_copy["STANFORD_CORENLP_3_4_1_JAR"] = str(jar)
    relocated_tokenize = types.FunctionType(original.__code__, globals_copy, original.__name__, original.__defaults__)
    tokenization_cache = {}

    class OfflinePTBTokenizer:
        def tokenize(self, captions):
            key = json.dumps(captions, ensure_ascii=False, sort_keys=True)
            if key not in tokenization_cache:
                tokenization_cache[key] = relocated_tokenize(self, captions)
            return tokenization_cache[key]

    class UnselectedMeteor:
        # The task creates every scorer even when requesting only BLEU/CIDEr.
        # No METEOR score is requested; suppress only this unused constructor.
        def compute_score(self, *args, **kwargs):
            raise RuntimeError("METEOR is deliberately omitted from this subset run")

    class QuietLogger:
        def info(self, *args, **kwargs):
            pass

    tasks_root = project_root / "vendor/lmms-eval/lmms_eval/tasks"
    nextqa_path = tasks_root / "nextqa/utils.py"
    stopwords_path = nextqa_path.parent / "stopwords.csv"
    nextqa = {
        "lemmatize_sentence": lemmatize_sentence,
        "wordnet": wordnet, "word_tokenize": word_tokenize,
        "stopwords": set(pd.read_csv(stopwords_path).squeeze()),
    }
    nextqa_source = load_source_functions(nextqa_path, (
        "remove_stop", "wup", "wups", "get_wups",
        "nextqa_process_results", "nextqa_aggregate_results",
    ), nextqa)
    nextqa_source["stopwords_path"] = str(stopwords_path)
    nextqa_source["stopwords_sha256"] = sha256(stopwords_path)

    vatex = {
        "Bleu": Bleu, "Cider": Cider, "Rouge": Rouge,
        "Meteor": UnselectedMeteor, "COCO": COCO, "COCOEvalCap": COCOEvalCap,
        "PTBTokenizer": OfflinePTBTokenizer, "eval_logger": QuietLogger(),
        "VATEX_METRICS": ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"],
        "json": json,
        "generate_submission_file": lambda filename, args: str(temporary_dir / filename),
    }
    vatex_source = load_source_functions(tasks_root / "vatex/utils.py", (
        "vatex_process_result", "vatex_aggregation_result",
    ), vatex)
    vatex_source["adaptations"] = [
        "Unchanged task aggregation AST; unused METEOR constructor replaced with an unscorable placeholder.",
        "Only Bleu_4, ROUGE_L, CIDEr are requested; no COCOEvalCap.evaluate(), METEOR or SPICE execution.",
        "PTB tokenize bytecode unchanged; only jar path and temporary-file directory relocated, with exact-input memoization.",
        "Task submission side effect redirected into an automatically deleted temporary directory.",
    ]
    vatex_source["ptb_source_path"] = ptb.__file__
    vatex_source["ptb_source_sha256"] = sha256(ptb.__file__)
    vatex_source["ptb_jar_path"] = str(jar)
    vatex_source["ptb_jar_sha256"] = sha256(jar)
    return nextqa, vatex, {"nextqa": nextqa_source, "vatex": vatex_source}


def load_references(inventory_path, manifest_path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    inventory = json.loads(inventory_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    selected_inputs = {row["id"]: row for row in manifest["records"]}
    expected_ids = {f"{task}:{doc_id}" for task in ("nextqa_oe_test", "vatex_test") for doc_id in (0, 1)}
    if set(selected_inputs) != expected_ids:
        raise ValueError("Expected exactly doc_ids 0,1 for NextQA and VATEX in inputs manifest")
    docs, sources = {}, {}
    for dataset in inventory["datasets"]:
        task = dataset["task"]
        if task not in ("nextqa_oe_test", "vatex_test"):
            continue
        path = Path(dataset["arrow_path"])
        digest = sha256(path)
        if digest != dataset["arrow_sha256"]:
            raise ValueError(f"Dataset SHA256 does not match inventory: {path}")
        with pa.memory_map(str(path), "r") as stream:
            rows = ipc.open_stream(stream).read_all().slice(0, 2).to_pylist()
        if len(rows) != 2:
            raise ValueError(f"Missing reference rows: {path}")
        for doc_id, doc in enumerate(rows):
            ident = f"{task}:{doc_id}"
            if selected_inputs[ident]["source_dataset_arrow_sha256"] != digest:
                raise ValueError(f"Dataset hash does not match GPU input manifest: {ident}")
            video_id = str(doc["video"] if task == "nextqa_oe_test" else doc["videoID"])
            if Path(selected_inputs[ident]["video_path"]).stem != video_id:
                raise ValueError(f"Reference video ID does not match GPU input: {ident}")
            docs[ident] = doc
        sources[task] = {"path": str(path), "sha256": digest, "doc_ids": [0, 1], "row_count": dataset["rows"]}
    if set(docs) != expected_ids:
        raise ValueError("Missing cached reference dataset")
    return docs, selected_inputs, sources


def score_run(path, references, inputs, nextqa, vatex):
    payload = json.loads(path.read_text())
    records = {row["id"]: row for row in payload["records"]}
    if len(payload["records"]) != 4 or set(records) != set(references):
        raise ValueError(f"Expected four unique, complete GPU predictions: {path}")
    for ident, record in records.items():
        if record["input_hashes"] != inputs[ident]["input_hashes"]:
            raise ValueError(f"GPU input hash does not match input manifest: {path.name}/{ident}")
    wups_records = []
    nextqa_docs = []
    for doc_id in (0, 1):
        ident = f"nextqa_oe_test:{doc_id}"
        score = nextqa["nextqa_process_results"](references[ident], [records[ident]["text"]])["WUPS"]
        wups_records.append(score)
        nextqa_docs.append({"id": ident, "prediction": records[ident]["text"], "task_result": score})
    vatex_results = [vatex["vatex_process_result"](references[f"vatex_test:{doc_id}"], [records[f"vatex_test:{doc_id}"]["text"]])["vatex_Bleu_4"] for doc_id in (0, 1)]
    vatex_scores = {
        metric: float(vatex["vatex_aggregation_result"](vatex_results, metric))
        for metric in ("Bleu_4", "ROUGE_L", "CIDEr")
    }
    return {
        "file": str(path), "sha256": sha256(path),
        "implementation": payload["implementation"],
        "generation_source_module": payload["source_module"],
        "generation_source_sha256": payload["source_sha256"],
        "config": records["nextqa_oe_test:0"]["config"],
        "tasks": {
            "nextqa_oe_test": {"n": 2, "doc_ids": [0, 1], "metrics": {"WUPS": float(nextqa["nextqa_aggregate_results"](wups_records))}, "scale": "0-100; task returns WUPS@0", "records": nextqa_docs},
            "vatex_test": {"n": 2, "doc_ids": [0, 1], "metrics": vatex_scores, "scale": "Raw pycocoevalcap units; no multiplication by 100", "records": [{"id": f"vatex_test:{doc_id}", "video_id": vatex_results[doc_id]["video_id"], "reference_count": len(vatex_results[doc_id]["answer"]), "prediction": vatex_results[doc_id]["pred"]} for doc_id in (0, 1)]},
        },
    }


def main():
    # Conda's sqlite/ICU needs its bundled C++ runtime; pyarrow can otherwise
    # preload an older system libstdc++ before nltk imports sqlite3.
    runtime_dir = str(Path(sys.prefix) / "lib")
    runtime_paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if (Path(runtime_dir) / "libstdc++.so.6").exists() and runtime_paths[0] != runtime_dir:
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = runtime_dir + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=base / "results")
    parser.add_argument("--inventory", type=Path, default=Path("/tmp/gpu_retest_data_inventory.json"))
    parser.add_argument("--inputs-manifest", type=Path, default=base / "inputs_manifest.json")
    parser.add_argument("--project-root", type=Path, default=base / "KVRepack")
    parser.add_argument("--java-home", type=Path, default=Path(sys.prefix) / "lib/jvm")
    parser.add_argument("--output", type=Path, default=base / "results/subset_metrics.json")
    args = parser.parse_args()
    offline_environment(args.java_home)
    references, inputs, reference_sources = load_references(args.inventory, args.inputs_manifest)
    with tempfile.TemporaryDirectory(prefix="offline-subset-", dir=base) as directory:
        with contextlib.redirect_stdout(io.StringIO()):
            nextqa, vatex, scorer_sources = setup_scorers(args.project_root, Path(directory))
            runs = {
                f"{method}_{implementation}": score_run(args.results_dir / f"{method}_{implementation}.json", references, inputs, nextqa, vatex)
                for method in ("snapkv", "madakv") for implementation in ("original", "repack")
            }
    comparisons = {}
    for method in ("snapkv", "madakv"):
        original, repack = runs[f"{method}_original"], runs[f"{method}_repack"]
        comparisons[method] = {}
        for task in ("nextqa_oe_test", "vatex_test"):
            left, right = original["tasks"][task], repack["tasks"][task]
            comparisons[method][task] = {
                "same_doc_ids": left["doc_ids"] == right["doc_ids"],
                "identical_prediction_texts": [r["prediction"] for r in left["records"]] == [r["prediction"] for r in right["records"]],
                "metric_deltas_repack_minus_original": {name: right["metrics"][name] - value for name, value in left["metrics"].items()},
            }
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"cpu_metrics_only": True, "gpu_or_model_loaded": False, "network_or_download_requested": False, "python_network_blocked": True, "java_use": "Only local Stanford PTB tokenizer; Python block does not sandbox subprocess networking", "subset": "Exactly doc_ids 0 and 1 of NextQA-OE test and VATEX test"},
        "limitations": [
            "Compares newly rerun original/repack outputs only; historical prediction JSONL files are unavailable.",
            "Two-doc subset metrics are not historical 200-doc aggregates, a benchmark reproduction, a quality ranking, or evidence of general algorithm equivalence.",
            "CIDEr document frequencies are computed over these two references groups, so subset CIDEr is especially incomparable with 200-doc/full-corpus CIDEr.",
            "METEOR and SPICE are deliberately omitted; no VATEX mean4/composite score is reported.",
            "This CPU script verifies input hashes/text/metrics; token/logit/cache equivalence is checked separately by compare_gpu_runs.py.",
        ],
        "versions": {name: importlib.metadata.version(name) for name in ("nltk", "pywsd", "pycocoevalcap", "pycocotools", "pyarrow", "pandas")},
        "provenance": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "inventory": str(args.inventory), "inventory_sha256": sha256(args.inventory), "inputs_manifest": str(args.inputs_manifest), "inputs_manifest_sha256": sha256(args.inputs_manifest), "references": reference_sources, "scorers": scorer_sources},
        "runs": runs, "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for name, run in runs.items():
        print(name, json.dumps({task: value["metrics"] for task, value in run["tasks"].items()}))
    print(f"Saved: {args.output}")
    if any(delta != 0 for tasks in comparisons.values() for row in tasks.values() for delta in row["metric_deltas_repack_minus_original"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

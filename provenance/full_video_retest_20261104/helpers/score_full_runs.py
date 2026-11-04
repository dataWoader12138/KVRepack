#!/usr/bin/env python3
"""Score completed NextQA/VATEX prediction JSONL using offline task metrics.

Example:
  python score_full_runs.py --records predictions.jsonl --task vatex_test \
    --expected-count 4478 --output metrics.json --repo-root /path/to/KVRepack

Every nonempty JSONL line must contain task, integer doc_id, text and optionally
id="task:doc_id". Arbitrary doc_ids are accepted, with duplicate/range/count
checks. A full-split run additionally requires every Arrow row exactly once.
No model, torch, network service or download is used. Java is used only locally
by Stanford PTBTokenizer and METEOR. SPICE is never instantiated.
"""

import argparse
import ast
import contextlib
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
import types
import urllib.request

SCHEMA_VERSION = 1
TASKS = {"nextqa": "nextqa_oe_test", "nextqa_oe_test": "nextqa_oe_test", "vatex": "vatex_test", "vatex_test": "vatex_test"}
NEXTQA_FUNCTIONS = ("remove_stop", "wup", "wups", "get_wups", "nextqa_process_results", "nextqa_aggregate_results")
VATEX_FUNCTIONS = ("vatex_process_result", "vatex_aggregation_result")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def deny_network(*args, **kwargs):
    raise RuntimeError("Offline scorer: Python networking and downloads are disabled")


def environment_setup(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    if args.nltk_data:
        os.environ["NLTK_DATA"] = str(args.nltk_data)
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    urllib.request.urlopen = deny_network
    urllib.request.urlretrieve = deny_network
    if args.task == "vatex_test":
        if args.java_home:
            os.environ["JAVA_HOME"] = str(args.java_home)
            os.environ["PATH"] = str(args.java_home / "bin") + os.pathsep + os.environ["PATH"]
        if not shutil.which("java"):
            raise FileNotFoundError("VATEX requires a local Java executable; pass --java-home")


def read_records(path, task, expected_count):
    before = sha256(path)
    records = {}
    with path.open() as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{lineno}: incomplete/invalid JSONL: {error}") from error
            doc_id = row.get("doc_id")
            if row.get("task") != task or type(doc_id) is not int or doc_id < 0 or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{lineno}: requires matching task, nonnegative integer doc_id, and string text")
            if row.get("id", f"{task}:{doc_id}") != f"{task}:{doc_id}":
                raise ValueError(f"{path}:{lineno}: id/task/doc_id mismatch")
            if doc_id in records:
                raise ValueError(f"{path}:{lineno}: duplicate doc_id {doc_id}")
            records[doc_id] = {"task": task, "doc_id": doc_id, "text": row["text"]}
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} predictions, found {len(records)}; refusing partial scoring")
    if sha256(path) != before:
        raise ValueError("Prediction JSONL changed while being read; score only completed runs")
    ordered = [records[key] for key in sorted(records)]
    return ordered, before, canonical_hash(ordered)


def reference_rows(args, records):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    expected_digest = None
    if args.reference_arrow:
        path = args.reference_arrow
    else:
        inventory = json.loads(args.inventory.read_text())
        dataset = next(row for row in inventory["datasets"] if row["task"] == args.task)
        path = Path(dataset["arrow_path"])
        expected_digest = dataset["arrow_sha256"]
    digest = sha256(path)
    if expected_digest and digest != expected_digest:
        raise ValueError("Reference Arrow SHA256 does not match inventory")
    with pa.memory_map(str(path), "r") as stream:
        table = ipc.open_stream(stream).read_all()
    doc_ids = [row["doc_id"] for row in records]
    if not doc_ids or max(doc_ids) >= table.num_rows:
        raise ValueError(f"doc_id out of reference range 0..{table.num_rows - 1}")
    docs = table.take(pa.array(doc_ids, type=pa.int64())).to_pylist()
    required = {"answer", "type", "additional_ref_answer"} if args.task == "nextqa_oe_test" else {"videoID", "enCap"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Reference Arrow missing required task columns {required}")
    if args.expected_count == table.num_rows and doc_ids != list(range(table.num_rows)):
        raise ValueError("Full split requires each reference doc_id exactly once")
    return docs, {
        "path": str(path.resolve()), "sha256": digest, "split_rows": table.num_rows,
        "selected_rows": len(doc_ids), "full_split": doc_ids == list(range(table.num_rows)),
        "doc_ids": doc_ids, "doc_ids_sha256": canonical_hash(doc_ids),
        "selected_reference_rows_sha256": canonical_hash(docs),
    }


def function_spec(path, names):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names) or {node.name for node in nodes} != set(names):
        raise ValueError(f"Missing/duplicate task function in {path}")
    return nodes, {
        "path": str(path.resolve()), "source_sha256": sha256(path),
        "function_ast_sha256": {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() for node in nodes},
    }


def execute_functions(nodes, path, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def scorer_spec(args):
    task_directory = "nextqa" if args.task == "nextqa_oe_test" else "vatex"
    task_path = args.repo_root / "vendor/lmms-eval/lmms_eval/tasks" / task_directory / "utils.py"
    nodes, source = function_spec(task_path, NEXTQA_FUNCTIONS if args.task == "nextqa_oe_test" else VATEX_FUNCTIONS)
    packages = ("nltk", "pywsd", "wn", "pandas", "pyarrow") if args.task == "nextqa_oe_test" else ("pycocoevalcap", "pycocotools", "pyarrow")
    versions = {name: importlib.metadata.version(name) for name in packages}
    resources = {}
    # Fingerprint installed scoring Python code, not only package version labels.
    for name in packages:
        if name == "pyarrow":
            continue
        distribution = importlib.metadata.distribution(name)
        files = {}
        for item in distribution.files or ():
            if str(item).endswith(".py") or (name == "wn" and "/data/" in str(item)):
                candidate = Path(distribution.locate_file(item))
                if candidate.is_file():
                    files[str(item)] = sha256(candidate)
        resources[f"{name}_python_tree"] = canonical_hash(files)
    if args.task == "nextqa_oe_test":
        resources["stopwords.csv"] = sha256(task_path.parent / "stopwords.csv")
        nltk_root = args.nltk_data or Path(os.environ.get("NLTK_DATA", str(Path.home() / "nltk_data")))
        # Hash all installed NLTK resource files for safe reuse. They are local
        # linguistic resources, not model checkpoints or video payloads.
        if not nltk_root.is_dir():
            raise FileNotFoundError(f"Local NLTK resources missing: {nltk_root}")
        resources["nltk_data_tree"] = canonical_hash({str(path.relative_to(nltk_root)): sha256(path) for path in sorted(nltk_root.rglob("*")) if path.is_file()})
    else:
        distribution = importlib.metadata.distribution("pycocoevalcap")
        for relative in ("pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar", "pycocoevalcap/meteor/meteor-1.5.jar", "pycocoevalcap/meteor/data/paraphrase-en.gz"):
            path = Path(distribution.locate_file(relative))
            resources[relative] = sha256(path)
    identity = {
        "task": args.task, "task_source_sha256": source["source_sha256"],
        "function_ast_sha256": source["function_ast_sha256"], "versions": versions,
        "resources": resources, "implementation": "unchanged_task_function_ast_v1",
        "adapter_sha256": sha256(__file__),
        "metric_set": ["WUPS"] if args.task == "nextqa_oe_test" else ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "mean4"],
    }
    return task_path, nodes, {"task_source": source, "identity": identity, "identity_sha256": canonical_hash(identity)}


def nextqa_score(task_path, nodes, records, docs):
    import nltk
    nltk.download = deny_network
    nltk.downloader.Downloader.download = deny_network
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from pywsd.utils import lemmatize_sentence
    import pandas as pd
    namespace = {
        "wordnet": wordnet, "word_tokenize": word_tokenize,
        "lemmatize_sentence": lemmatize_sentence,
        "stopwords": set(pd.read_csv(task_path.parent / "stopwords.csv").squeeze()),
    }
    execute_functions(nodes, task_path, namespace)
    namespace["remove_stop"] = functools.lru_cache(maxsize=None)(namespace["remove_stop"])
    results = []
    for index, (row, doc) in enumerate(zip(records, docs), 1):
        results.append(namespace["nextqa_process_results"](doc, [row["text"]])["WUPS"])
        if index % 1000 == 0:
            print(f"NextQA WUPS: {index}/{len(records)}", file=sys.stderr, flush=True)
    return {"WUPS": float(namespace["nextqa_aggregate_results"](results))}, {
        "metric_scale": "WUPS is 0-100; the task aggregate returns WUPS@0, including its original question-type/reference rules",
        "execution": "Unchanged task function AST; deterministic remove_stop results memoized per exact text",
    }


def vatex_score(task_path, nodes, records, docs, temporary_dir):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb

    original = ptb.PTBTokenizer.tokenize
    adjusted_globals = dict(original.__globals__)
    adjusted_globals["__file__"] = str(temporary_dir / "ptbtokenizer.py")
    adjusted_globals["STANFORD_CORENLP_3_4_1_JAR"] = str(Path(ptb.__file__).parent / ptb.STANFORD_CORENLP_3_4_1_JAR)
    relocated_tokenize = types.FunctionType(original.__code__, adjusted_globals, original.__name__, original.__defaults__)
    tokenization_cache = {}

    class OfflinePTBTokenizer:
        def tokenize(self, captions):
            key = canonical_hash(captions)
            if key not in tokenization_cache:
                tokenization_cache[key] = relocated_tokenize(self, captions)
            return tokenization_cache[key]

    class QuietLogger:
        def info(self, *args, **kwargs):
            pass

    meteor_instances = []

    def shared_meteor():
        # The task constructs all scorers every time it scores one metric.
        # Reuse the real, unmodified Meteor scorer instead of starting four JVMs.
        if not meteor_instances:
            meteor_instances.append(Meteor())
        return meteor_instances[0]

    namespace = {
        "Bleu": Bleu, "Cider": Cider, "Rouge": Rouge, "Meteor": shared_meteor,
        "COCO": COCO, "COCOEvalCap": COCOEvalCap, "PTBTokenizer": OfflinePTBTokenizer,
        "VATEX_METRICS": ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"],
        "eval_logger": QuietLogger(), "json": json,
        "generate_submission_file": lambda name, args: str(temporary_dir / name),
    }
    execute_functions(nodes, task_path, namespace)
    results = [namespace["vatex_process_result"](doc, [row["text"]])["vatex_Bleu_4"] for row, doc in zip(records, docs)]
    if len({result["video_id"] for result in results}) != len(results):
        raise ValueError("Duplicate VATEX reference video IDs would overwrite COCO results")
    try:
        metrics = {}
        for name in ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr"):
            print(f"VATEX scoring {name} on {len(records)} records", file=sys.stderr, flush=True)
            metrics[name] = float(namespace["vatex_aggregation_result"](results, name))
        metrics["mean4"] = sum(metrics.values()) / 4
        return metrics, {
            "metric_scale": "All VATEX values are raw pycocoevalcap units; no multiplication by 100",
            "mean4_definition": "(Bleu_4 + METEOR + ROUGE_L + CIDEr) / 4, matching docs/VIDEO_BEST_PARAMETERS.md",
            "execution": "Unchanged task aggregation AST and installed scorers; one shared real Meteor instance; unchanged PTB tokenization bytecode with relocated temp files and memoization",
            "meteor_command": meteor_instances[0].meteor_cmd,
            "meteor_process_alive_after_scoring": meteor_instances[0].meteor_p.poll() is None,
            "spice": "Not instantiated or scored; task's configured metric list excludes SPICE",
        }
    finally:
        if meteor_instances:
            scorer = meteor_instances.pop()
            # Release the last reference so the installed destructor closes,
            # kills, and waits for its Java process exactly once.
            del scorer


def main():
    # Ensure conda SQLite/ICU and pyarrow share conda's C++ runtime. This is a
    # process-local restart and does not modify the installed environment.
    runtime = str(Path(sys.prefix) / "lib")
    if (Path(runtime) / "libstdc++.so.6").exists() and os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0] != runtime:
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = runtime + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=Path("/tmp/gpu_retest_data_inventory.json"))
    parser.add_argument("--reference-arrow", type=Path, help="Explicit Arrow path overrides inventory (for portability)")
    nltk_default = Path(os.environ.get("NLTK_DATA", str(Path.home() / "nltk_data")))
    parser.add_argument("--nltk-data", type=Path, default=nltk_default)
    bundled_java = Path(sys.prefix) / "lib/jvm"
    parser.add_argument("--java-home", type=Path, default=bundled_java if (bundled_java / "bin/java").is_file() else None)
    parser.add_argument("--reuse-metrics-from", type=Path, help="Explicitly reuse scores only if prediction/reference/scorer identity hashes match")
    args = parser.parse_args()
    args.task = TASKS[args.task]
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    environment_setup(args)
    started = time.monotonic()
    records, raw_hash, corpus_hash = read_records(args.records, args.task, args.expected_count)
    docs, references = reference_rows(args, records)
    task_path, nodes, scorers = scorer_spec(args)
    identity = {
        "task": args.task, "n": len(records), "prediction_corpus_sha256": corpus_hash,
        "reference_arrow_sha256": references["sha256"],
        "selected_reference_rows_sha256": references["selected_reference_rows_sha256"],
        "doc_ids_sha256": references["doc_ids_sha256"], "scorer_identity_sha256": scorers["identity_sha256"],
    }
    identity_hash = canonical_hash(identity)
    reuse = {"requested": bool(args.reuse_metrics_from), "reused": False}
    if args.reuse_metrics_from:
        previous = json.loads(args.reuse_metrics_from.read_text())
        if previous.get("scoring_identity_sha256") != identity_hash or previous.get("scoring_identity") != identity:
            raise ValueError("Refusing metrics reuse: predictions, selected references, or scorer identity differ")
        if previous.get("status") != "complete" or not isinstance(previous.get("metrics"), dict):
            raise ValueError("Refusing metrics reuse: source report is incomplete")
        metrics, details = previous["metrics"], previous["metric_details"]
        reuse.update({"reused": True, "source": str(args.reuse_metrics_from.resolve()), "source_sha256": sha256(args.reuse_metrics_from)})
    else:
        with tempfile.TemporaryDirectory(prefix="offline-score-", dir=Path(__file__).resolve().parent) as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                if args.task == "nextqa_oe_test":
                    metrics, details = nextqa_score(task_path, nodes, records, docs)
                else:
                    metrics, details = vatex_score(task_path, nodes, records, docs, Path(directory))
    report = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "task": args.task,
        "n": len(records), "expected_count": args.expected_count,
        "metrics": metrics, "metric_details": details, "reuse": reuse,
        "scoring_identity": identity, "scoring_identity_sha256": identity_hash,
        "records": {"path": str(args.records.resolve()), "file_sha256": raw_hash, "canonical_prediction_corpus_sha256": corpus_hash, "canonical_order": "ascending doc_id; hash includes task, doc_id and exact text"},
        "references": references, "scorers": scorers,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "scope": {"cpu_only": True, "model_or_gpu_loaded": False, "network_or_download_requested": False, "python_networking_blocked": True, "java": "Local PTBTokenizer/METEOR only; Python network block does not sandbox Java subprocesses"},
        "limitations": [
            "Metrics describe exactly these doc_ids and this reference cache; compare methods only on identical task/reference/doc_id coverage.",
            "CIDEr document frequencies are recomputed from the selected reference corpus; subset CIDEr is not comparable to full-corpus CIDEr.",
            "This scores newly generated predictions; it does not recover missing historical predictions or reproduce historical 200-doc results automatically.",
            "Metrics reuse, when explicitly requested, is documented and requires exact prediction/reference/scorer identity hashes; no scorer is silently skipped.",
        ],
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    temporary_output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary_output.replace(args.output)
    print(json.dumps({"status": "complete", "task": args.task, "n": len(records), "full_split": references["full_split"], "metrics": metrics, "reused": reuse["reused"], "seconds": report["seconds"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a verified full styled-image experiment, or rescore its extracted raw JSON.

Export never launches inference and accepts only the complete 16 x 200 matrix.
The extracted archive can be rescored using only the matching annotations and
the bundled CPU scorer; image files, model weights and CUDA are unnecessary.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import numbers
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("hybridkv_asymkv", "hybridkv_lookm", "hybridkv_kvrepack",
        "snapkv_asymkv", "snapkv_lookm", "snapkv_kvrepack", "asymkv", "lookm")
TASKS = ("Spot-the-Diff", "MMCoQA")
EXPECTED_SAMPLES = 200
MAX_ARCHIVE_BYTES = 90_000_000
RUN_METADATA = ("run_contract.json", "run_state.json", "environment.json")
REPORT_TOP = ("comparison.csv", "per_input_latency.csv", "summary.json", "RESULTS.md")
EVALUATION_FILES = ("eval.json", "eval_score.json", "evaluation_metadata.json")
PROVENANCE_FILES = ("preparation.json", "formal_launch.json", "input_preflight.json", "source_check.json")
CPU_FILES = ("scripts/export_styled_milebench.py", "src/kvrepack/__init__.py",
             "src/kvrepack/benchmarks/__init__.py", "src/kvrepack/benchmarks/milebench/__init__.py",
             "src/kvrepack/benchmarks/milebench/dataset.py", "src/kvrepack/benchmarks/milebench/evaluate.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked recorded file: {path}")
    return path


def safe_relative(name):
    require(isinstance(name, str) and "\\" not in name, "Invalid archive path")
    value = PurePosixPath(name)
    require(not value.is_absolute() and name == value.as_posix()
            and all(part not in ("", ".", "..") for part in value.parts), f"Unsafe archive path: {name}")
    return Path(*value.parts)


def unique_by(rows, columns, name):
    require(isinstance(rows, list), f"Missing {name}")
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"Invalid {name} row")
        key = tuple(row.get(column) for column in columns)
        require(key not in result, f"Duplicate {name}: {key}")
        result[key] = row
    return result


def equal_values(actual, expected, label):
    """Compare all persisted JSON fields, including booleans and nulls."""
    if isinstance(expected, numbers.Real) and not isinstance(expected, (int, bool)):
        require(isinstance(actual, numbers.Real) and not isinstance(actual, bool) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), f"Changed {label}")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Changed keys in {label}")
        for key, value in expected.items():
            equal_values(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Changed length in {label}")
        for index, value in enumerate(expected):
            equal_values(actual[index], value, f"{label}[{index}]")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Changed {label}")


def check_csv(path, expected, columns, keys):
    with plain_file(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(columns), f"CSV columns differ: {path.name}")
        rows = list(reader)
    # DictWriter serializes None as empty and booleans as True/False. Compare
    # exact serialized values so every timing and per-input score is checked.
    expected_strings = [{column: "" if row[column] is None else str(row[column]) for column in columns}
                        for row in expected]
    actual_map = unique_by(rows, keys, path.name)
    expected_map = unique_by(expected_strings, keys, f"recomputed {path.name}")
    require(actual_map == expected_map, f"CSV values differ from raw rows: {path.name}")


def load_reporter():
    spec = importlib.util.spec_from_file_location("styled_export_reporter", ROOT / "scripts/summarize_styled_milebench.py")
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    require(reporter.ARMS == ARMS and reporter.TASKS == TASKS, "Exporter/reporter condition matrix differs")
    return reporter


def matrix_pairs(controller, *, relocated=False, root=None):
    expected = {(arm, task) for arm in ARMS for task in TASKS}
    require(controller.get("status") == "complete", "Only a complete controller can be exported/rescored")
    require(controller.get("arms") == list(ARMS) and controller.get("tasks") == list(TASKS),
            "Require exactly all eight arms and both tasks")
    require(controller.get("source_commit") == "b9104d8" and controller.get("all_samples") is True
            and controller.get("planned_runs") == len(expected)
            and controller.get("target_generations") == len(expected)*EXPECTED_SAMPLES,
            "Controller protocol/counts differ from the full official matrix")
    rows = unique_by(controller.get("runs"), ("arm", "task"), "controller conditions")
    require(set(rows) == expected and controller.get("completed_runs") == len(expected)
            and all(row.get("status") == "complete" for row in rows.values()), "Incomplete 16-condition matrix")
    for (arm, task), row in rows.items():
        require(row.get("target_samples") == EXPECTED_SAMPLES
                and row.get("annotation_total_samples") == EXPECTED_SAMPLES and row.get("all_samples") is True,
                "Controller child is not the complete official annotation")
        if not relocated:
            require(Path(row["output"]).resolve() == root / arm / task, "Controller run path differs from source root")
    return [(arm, task) for arm in ARMS for task in TASKS]


def record_sources(paths):
    return {Path(path): {"sha256": file_sha(path), "bytes": Path(path).stat().st_size}
            for path in set(paths)}


def assert_unchanged(snapshot):
    for path, record in snapshot.items():
        require(path.is_file() and path.stat().st_size == record["bytes"] and file_sha(path) == record["sha256"],
                f"Source changed during export/rescore: {path}")


def raw_files_for_run(run, contract):
    names = list(RUN_METADATA)
    if not contract["standalone"]:
        names.append("adapter_transform.json")
    elif contract["arm"] == "lookm":
        names.append("legacy_lookm_transform.json")
    paths = [plain_file(run / name) for name in names]
    for directory, count in (("samples", EXPECTED_SAMPLES), ("warmup", contract["warmup_per_arm"])):
        require(type(count) is int and count >= 0, "Invalid declared warmup count")
        expected = [f"{i:06d}.json" for i in range(count)]
        folder = run / directory
        files = sorted(folder.iterdir()) if folder.exists() else []
        require([path.name for path in files] == expected, f"Unexpected {directory} coverage: {run}")
        paths.extend(plain_file(path) for path in files)
    return paths


def check_full_contract(contract):
    require(contract.get("all_samples") is True and contract.get("limit") == EXPECTED_SAMPLES
            and contract.get("annotation_total_samples") == EXPECTED_SAMPLES
            and len(contract.get("selected_sample_ids", [])) == EXPECTED_SAMPLES
            and len(set(contract["selected_sample_ids"])) == EXPECTED_SAMPLES,
            f"Require the complete official {EXPECTED_SAMPLES}-sample annotation for every condition")


def validate_export(root, report):
    """Recompute raw scores, E2E/cache/EOS accounting and source/pairing audits."""
    root, report = Path(root).resolve(), Path(report).resolve()
    controller_path = plain_file(root / "experiment_state.json")
    controller = read_json(controller_path)
    pairs = matrix_pairs(controller, root=root)
    summary = read_json(plain_file(report / "summary.json"))
    require(summary.get("status") == "passed" and summary.get("full_experiment") is True
            and summary.get("completed_runs") == len(pairs)
            and summary.get("formal_generations") == len(pairs) * EXPECTED_SAMPLES
            and summary.get("run_root") == str(root), "Report is partial or belongs to another experiment")
    reporter = load_reporter()
    result_map = unique_by(summary.get("results"), ("arm", "task"), "summary results")
    evidence_map = unique_by(summary.get("source_and_sample_evidence"), ("arm", "task"), "report evidence")
    require(set(result_map) == set(pairs) and set(evidence_map) == set(pairs), "Report condition set differs")
    raw_paths = [controller_path]
    report_paths = [plain_file(report / name) for name in REPORT_TOP]
    for arm, task in pairs:
        contract = read_json(root / arm / task / "run_contract.json")
        check_full_contract(contract)
        raw_paths.extend(raw_files_for_run(root / arm / task, contract))
        report_paths.extend(plain_file(report / "conditions" / arm / task / name) for name in EVALUATION_FILES)
    initial = record_sources(raw_paths + report_paths + [ROOT / "scripts/summarize_styled_milebench.py"])
    runs, latency, summaries, referenced = {}, [], [], []
    for arm, task in pairs:
        run = root / arm / task
        record = reporter.read_run(run, arm, task)
        contract, _, computed, per_input, evaluation, evidence = record
        runs[arm, task] = record
        equal_values(result_map[arm, task], computed, f"summary {arm}/{task}")
        equal_values(evidence_map[arm, task], evidence, f"SHA evidence {arm}/{task}")
        metadata = {key: evaluation.get(key) for key in ("task", "sample_ids", "sample_count", "annotation_total_samples", "full_task")}
        for name, expected in (("eval.json", evaluation["eval"]), ("eval_score.json", evaluation["eval_score"]),
                               ("evaluation_metadata.json", metadata)):
            equal_values(read_json(report / "conditions" / arm / task / name), expected, f"{arm}/{task}/{name}")
        latency.extend(per_input); summaries.append(computed)
        referenced.extend(ROOT / relative for relative in contract["source_sha256"])
        from kvrepack.benchmarks.milebench.dataset import annotation_path
        referenced.append(annotation_path(contract["data_root"], task))
        if contract.get("kernel_binary"):
            referenced.append(Path(contract["kernel_binary"]["path"]))
    equal_values(summary.get("pairing"), reporter.validate_pairing(runs), "pairing evidence")
    require(summary.get("warmup_generations_excluded") == sum(len(record[5]["excluded_warmup_files"]) for record in runs.values()),
            "Warmup exclusion count differs")
    check_csv(report / "comparison.csv", summaries, reporter.SUMMARY_COLUMNS, ("arm", "task"))
    check_csv(report / "per_input_latency.csv", latency, reporter.PER_INPUT_COLUMNS, ("arm", "task", "sample_id"))
    assert_unchanged(initial)
    return controller, summary, runs, raw_paths, report_paths, initial, referenced


def new_output(output, protected):
    output = Path(output).expanduser().resolve()
    require(not output.exists(), "Output already exists; choose a new directory")
    for source in protected:
        source = Path(source).resolve()
        require(output != source and not source.is_relative_to(output) and not output.is_relative_to(source),
                f"Output must be separate from source: {source}")
    require(not output.is_relative_to(ROOT / "src") and not output.is_relative_to(ROOT / "scripts")
            and not output.is_relative_to(ROOT / "tests"), "Output cannot be inside implementation directories")
    return output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=True)


def archive_files(path, root, raw_paths, index_bytes):
    with tarfile.open(path, "w:gz", compresslevel=6) as archive:
        for file in sorted(raw_paths):
            archive.add(file, arcname=file.relative_to(root).as_posix(), recursive=False)
        entry = tarfile.TarInfo("recorded_files_sha256.json")
        entry.size = len(index_bytes); entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(index_bytes))
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES,
            f"Raw archive exceeds the {MAX_ARCHIVE_BYTES}-byte (90 MB) limit; no samples were removed")


def check_archive(path, index):
    expected = {entry["path"]: entry for entry in index["files"]}
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        require(len({member.name for member in members}) == len(members)
                and set(member.name for member in members) == set(expected) | {"recorded_files_sha256.json"},
                "Archive member coverage differs")
        for member in members:
            safe_relative(member.name)
            require(member.isfile(), "Archive contains a link or non-file")
            with archive.extractfile(member) as stream:
                digest = hashlib.sha256(); length = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk); length += len(chunk)
            if member.name == "recorded_files_sha256.json":
                require(digest.hexdigest() == hashlib.sha256(json_bytes(index)).hexdigest(), "Embedded index differs")
            else:
                require(length == expected[member.name]["bytes"] and digest.hexdigest() == expected[member.name]["sha256"],
                        f"Archive round-trip differs: {member.name}")


def export_experiment(root, report, output, *, provenance=None):
    root, report = Path(root).expanduser().resolve(), Path(report).expanduser().resolve()
    # A cheap complete/full check happens before any expensive CPU validation.
    controller = read_json(root / "experiment_state.json")
    matrix_pairs(controller, root=root)
    protected = [root, report]
    if provenance is not None:
        provenance = Path(provenance).expanduser().resolve()
        protected.extend(provenance / name for name in PROVENANCE_FILES if (provenance / name).exists())
    output = new_output(output, protected)
    controller, summary, runs, raw_paths, report_paths, initial, referenced = validate_export(root, report)
    model_map = next(iter(runs.values()))[0].get("model_metadata_sha256", {})
    configuration = {}
    for record in runs.values():
        contract = record[0]
        require(contract.get("model_metadata_sha256") == model_map, "Model configuration identities differ")
        model_root = Path(contract["pretrained"]).expanduser().resolve()
        require(not output.is_relative_to(model_root) and not model_root.is_relative_to(output), "Output overlaps model metadata")
        protected.append(Path(contract["data_root"]).expanduser().resolve())
        require(not output.is_relative_to(protected[-1]) and not protected[-1].is_relative_to(output), "Output overlaps benchmark data")
        for name, expected in model_map.items():
            relative = safe_relative(name)
            require(relative.suffix in (".json", ".txt"), "Model metadata allowlist rejects non-config file")
            source = model_root / relative
            require(source.is_file() and file_sha(source) == expected, f"Model config changed: {name}")
            configuration[Path("configuration/model") / relative] = source
        for relative, expected in contract["source_sha256"].items():
            if relative.endswith(".json") or relative.endswith(".patch"):
                source = ROOT / safe_relative(relative)
                require(file_sha(source) == expected, "Bound configuration source changed")
                configuration[Path("configuration/repository") / relative] = source
    metadata_paths = ([plain_file(provenance / name) for name in PROVENANCE_FILES if (provenance / name).exists()]
                      if provenance is not None else [])
    cpu_sources = [plain_file(ROOT / name) for name in CPU_FILES]
    snapshot = {**record_sources(referenced + list(configuration.values()) + metadata_paths + cpu_sources), **initial}
    raw_index = {"schema_version": 1, "scope": "all_raw_archive_members_except_this_index",
                 "condition_count": len(runs), "formal_samples": len(runs) * EXPECTED_SAMPLES,
                 "warmup_samples": summary["warmup_generations_excluded"],
                 "files": [{"path": path.relative_to(root).as_posix(), **initial[path]} for path in sorted(raw_paths)]}
    # Publish atomically only after raw, report, tar round-trip and end-of-export
    # source checks all pass. A failed export leaves no misleading final output.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".styled-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "release"; stage.mkdir()
        for file in report_paths:
            copy_file(file, stage / "report" / file.relative_to(report))
        for file in raw_paths:
            if file.parent.name not in ("samples", "warmup"):
                copy_file(file, stage / "metadata" / file.relative_to(root))
        for relative, source in configuration.items():
            copy_file(source, stage / relative)
        for source in metadata_paths:
            copy_file(source, stage / "provenance" / source.name)
        for source in cpu_sources:
            copy_file(source, stage / "cpu_rescore" / source.relative_to(ROOT))
        (stage / "cpu_rescore/requirements.txt").write_text("numpy==1.26.4\nrouge==1.0.1\n", encoding="utf-8")
        write_json(stage / "recorded_files_sha256.json", raw_index)
        archive = stage / "recorded_runs.tar.gz"
        archive_files(archive, root, raw_paths, json_bytes(raw_index))
        check_archive(archive, raw_index)
        write_json(stage / "source_files_sha256.json", {"schema_version": 1, "files": [
            {"source": str(path), **entry} for path, entry in sorted(snapshot.items())]})
        write_json(stage / "verification.json", {"status": "passed", "independent_raw_rescore": True,
            "all_summary_and_per_input_csv_values_checked": True, "all_condition_evaluation_json_checked": True,
            "all_contract_sample_and_warmup_sha256_checked": True, "native_pairing_rechecked": True,
            "archive_round_trip_every_member_sha256_checked": True,
            "summary_file_sha256": initial[report / "summary.json"]["sha256"],
            "report_evidence_sha256": json_sha(summary["source_and_sample_evidence"]),
            "final_controller_sha256": initial[root / "experiment_state.json"]["sha256"],
            "controller_at_scoring_sha256": summary.get("controller_generation_sha256"),
            "results": summary["results"]})
        write_json(stage / "protocol.json", {"schema_version": 1, "source_commit": "b9104d8",
            "arms": list(ARMS), "tasks": list(TASKS), "samples_per_condition": EXPECTED_SAMPLES,
            "image": next(iter(runs.values()))[0]["image"],
            "launch_command": controller.get("command"),
            "conditions": [{"arm": arm, "task": task, **{key: record[0].get(key) for key in
                ("algorithm_env", "generation", "budget_policy", "decoding_policy", "timing_scope", "stage_semantics")}}
                for (arm, task), record in runs.items()],
            "timing_scope": summary.get("timing_scope"), "fisher_scope": summary.get("fisher_scope"),
            "warmup_policy": summary.get("warmup_policy"), "eos_policy": summary.get("eos_policy"),
            "budget_scope": summary.get("budget_scope"), "limits": summary.get("limits", [])})
        (stage / "README.txt").write_text(
            "Complete Qwen2.5-VL-7B styled image experiment: 16 conditions x 200 official samples.\n"
            "The tar contains every raw sample, warmup and run contract/environment/final state, without images or weights.\n"
            "Verify recorded_runs.tar.gz SHA256 against release_manifest.json before extraction.\n"
            "CPU-only rescore (same original annotation JSON bytes; image files are unnecessary):\n"
            "  python -m pip install -r cpu_rescore/requirements.txt\n"
            "  mkdir unpacked\n"
            "  tar -xzf recorded_runs.tar.gz -C unpacked\n"
            "  python cpu_rescore/scripts/export_styled_milebench.py --rescore-extracted unpacked --data-root /path/to/milebench --output new_cpu_scores\n"
            "Original absolute paths remain in unmodified contracts as provenance; --data-root only relocates annotations.\n"
            "Rescoring checks all archived file hashes, exact IDs and annotation hashes, and computes original ROUGE-L F / Accuracy.\n"
            "It does not rerun inference, remeasure latency, or require the original GPU/kernel/model installation.\n"
            "Quality scores use the full official 200-item annotation of each task; separate warmups are excluded.\n"
            "Standalone AsymKV Fisher stays within official generation time (separate Fisher is unmeasured).\n"
            "Original cache policies, AsymKV chunk-boundary omissions and early EOS behavior are preserved and reported.\n"
            "E2E includes image I/O, preprocessing, H2D, applicable Fisher, generation and text decoding; model load is separate.\n",
            encoding="utf-8")
        release_files = {file.relative_to(stage).as_posix(): {"sha256": file_sha(file), "bytes": file.stat().st_size}
                         for file in sorted(stage.rglob("*")) if file.is_file()}
        write_json(stage / "release_files_sha256.json", {"schema_version": 1, "files": release_files,
            "excludes": ["release_files_sha256.json", "release_manifest.json"]})
        manifest = {"schema_version": 1, "status": "complete_and_independently_verified",
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(root), "source_report": str(report),
            "source_commit": "b9104d8", "arms": list(ARMS), "tasks": list(TASKS),
            "condition_count": len(runs), "samples_per_condition": EXPECTED_SAMPLES,
            "formal_generations": len(runs)*EXPECTED_SAMPLES, "warmup_generations_excluded": summary["warmup_generations_excluded"],
            "archive_file": archive.name, "archive_files": len(raw_paths)+1,
            "raw_source_files": len(raw_paths), "raw_source_uncompressed_bytes": sum(initial[path]["bytes"] for path in raw_paths),
            "archive_bytes": archive.stat().st_size, "archive_max_bytes": MAX_ARCHIVE_BYTES,
            "archive_sha256": file_sha(archive), "recorded_files_index_sha256": file_sha(stage / "recorded_files_sha256.json"),
            "release_files_index_sha256": file_sha(stage / "release_files_sha256.json"),
            "source_files_index_sha256": file_sha(stage / "source_files_sha256.json"),
            "round_trip_every_archived_file_sha256_verified": True, "all_sources_unchanged_at_end": True,
            "report_summary_sha256": initial[report / "summary.json"]["sha256"],
            "exporter_sha256": file_sha(Path(__file__)), "reporter_sha256": file_sha(ROOT / "scripts/summarize_styled_milebench.py"),
            "gpu_execution_by_exporter": False,
            "limits": summary.get("limits", []) + [
                "CPU rescore reproduces quality metrics from recorded predictions; it does not reproduce GPU timing.",
                "Standalone AsymKV Fisher is included in official generation, not independently measured.",
                "Cache ratios reflect original implementations; nominal 10% is not a shared strict physical cap.",
                "Released AsymKV chunk omissions and post-initial-EOS generation are preserved, with affected counts reported."]}
        write_json(stage / "release_manifest.json", manifest)
        # Recheck contract-bound references against their declared identities too,
        # closing the interval between read_run's check and snapshot creation.
        for record in runs.values():
            contract = record[0]
            for relative, expected in contract["source_sha256"].items():
                require(file_sha(ROOT / relative) == expected, "Frozen source changed during export")
            from kvrepack.benchmarks.milebench.dataset import annotation_path
            require(file_sha(annotation_path(contract["data_root"], contract["task"])) == contract["annotation_sha256"],
                    "Annotation changed during export")
            if contract.get("kernel_binary"):
                require(file_sha(contract["kernel_binary"]["path"]) == contract["kernel_binary"]["sha256"],
                        "Hybrid kernel changed during export")
            for name, expected in contract["model_metadata_sha256"].items():
                require(file_sha(Path(contract["pretrained"]) / name) == expected, "Model metadata changed during export")
        final_raw_paths = {root / "experiment_state.json"}
        for (arm, task), record in runs.items():
            final_raw_paths.update(raw_files_for_run(root / arm / task, record[0]))
        require(final_raw_paths == set(raw_paths), "Raw file coverage changed during export")
        assert_unchanged(snapshot)
        require(not output.exists(), "Output appeared while exporting; refusing to overwrite")
        stage.rename(output)
    return manifest


def verify_extracted(root):
    root = Path(root).resolve()
    index_path = plain_file(root / "recorded_files_sha256.json")
    index = read_json(index_path)
    require(index.get("schema_version") == 1 and index.get("condition_count") == len(ARMS)*len(TASKS)
            and index.get("formal_samples") == len(ARMS)*len(TASKS)*EXPECTED_SAMPLES, "Not a full recorded archive")
    entries = unique_by(index.get("files"), ("path",), "recorded file index")
    snapshot = record_sources([index_path])
    for (name,), entry in entries.items():
        path = root / safe_relative(name)
        # Check every parent too: an extracted directory symlink must not redirect
        # a permitted relative member to an unrelated location.
        require(path.resolve().is_relative_to(root) and not any(p.is_symlink() for p in path.parents if p != root.parent),
                f"Extracted path escapes raw directory: {name}")
        plain_file(path)
        require(path.stat().st_size == entry.get("bytes") and file_sha(path) == entry.get("sha256"),
                f"Recorded file hash differs: {name}")
        snapshot[path] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
    all_paths = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in all_paths), "Extracted raw directory contains a symlink")
    require(set(path.relative_to(root).as_posix() for path in all_paths if path.is_file())
            == {name[0] for name in entries} | {index_path.name}, "Extracted raw directory contains unindexed files")
    return index, snapshot


def rescore_extracted(root, data_root, output):
    """No original source/model/kernel paths are accessed in this portable mode."""
    from kvrepack.benchmarks.milebench.dataset import annotation_path, load_annotation
    from kvrepack.benchmarks.milebench.evaluate import evaluate_predictions
    root, data_root = Path(root).expanduser().resolve(), Path(data_root).expanduser().resolve()
    output = new_output(output, [root, data_root])
    index, snapshot = verify_extracted(root)
    controller = read_json(root / "experiment_state.json")
    pairs = matrix_pairs(controller, relocated=True)
    annotations = {task: load_annotation(data_root, task, limit=None) for task in TASKS}
    annotations_paths = [annotation_path(data_root, task) for task in TASKS]
    snapshot.update(record_sources(annotations_paths))
    results, evaluations, expected_raw, warmup_count = [], {}, {root / "experiment_state.json"}, 0
    for arm, task in pairs:
        run = root / arm / task
        contract, state = read_json(run / "run_contract.json"), read_json(run / "run_state.json")
        check_full_contract(contract)
        require(contract.get("experiment") == "styled_milebench_image_response_e2e" and contract.get("source_commit") == "b9104d8"
                and contract.get("arm") == arm and contract.get("task") == task, "Unexpected archived run identity")
        require(state.get("status") == "complete" and state.get("contract_sha256") == json_sha(contract)
                and all(state.get(key) == EXPECTED_SAMPLES for key in ("target_samples", "completed_samples", "target_generations", "completed_generations")),
                "Archived run state is incomplete")
        annotation = annotations[task]
        require(file_sha(annotation_path(data_root, task)) == contract.get("annotation_sha256")
                and len(annotation["data"]) == EXPECTED_SAMPLES
                and [row["sample_id"] for row in annotation["data"]] == contract["selected_sample_ids"], "Relocated annotation bytes/IDs differ")
        groups = {}
        for sample in annotation["data"]:
            groups.setdefault(len(sample["task_instance"]["images_path"]), []).append(sample["sample_id"])
        ordered = [sample_id for group in groups.values() for sample_id in group]
        require(ordered == contract.get("generation_sample_ids"), "Archived generation IDs/order differ")
        paths = raw_files_for_run(run, contract); expected_raw.update(paths)
        predictions = []
        for position, sample_id in enumerate(ordered):
            row = read_json(run / "samples" / f"{position:06d}.json")
            require(row.get("status") == "completed" and row.get("warmup") is False and row.get("arm") == arm and row.get("task") == task
                    and row.get("sample_index") == position and row.get("sample_id") == sample_id
                    and row.get("contract_sha256") == json_sha(contract), "Archived formal row identity differs")
            require(row.get("prediction", {}).get("sample_id") == sample_id, "Archived prediction ID differs")
            predictions.append(row["prediction"])
        for position in range(contract["warmup_per_arm"]):
            row = read_json(run / "warmup" / f"{position:06d}.json")
            require(row.get("warmup") is True and row.get("status") == "completed"
                    and row.get("contract_sha256") == json_sha(contract) and row.get("arm") == arm and row.get("task") == task,
                    "Archived warmup identity differs")
            warmup_count += 1
        evaluation = evaluate_predictions(predictions, annotation, task)
        evaluations[arm, task] = evaluation
        metric = "Accuracy" if task == "MMCoQA" else "Rouge-L f"
        results.append({"arm": arm, "task": task, "metric": metric, "score_percent": float(evaluation["eval"][metric])*100,
                        "sample_count": evaluation["sample_count"], "annotation_sha256": contract["annotation_sha256"],
                        "contract_sha256": json_sha(contract)})
    require(set(snapshot) - {root / "recorded_files_sha256.json"} - set(annotations_paths) == expected_raw
            and warmup_count == index.get("warmup_samples"), "Archive allowlist/count differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".styled-rescore-", dir=output.parent) as temporary:
        stage = Path(temporary) / "scores"; stage.mkdir()
        for (arm, task), evaluation in evaluations.items():
            directory = stage / "conditions" / arm / task
            write_json(directory / "eval.json", evaluation["eval"])
            write_json(directory / "eval_score.json", evaluation["eval_score"])
            write_json(directory / "evaluation_metadata.json", {key: evaluation.get(key) for key in
                       ("task", "sample_ids", "sample_count", "annotation_total_samples", "full_task")})
        with (stage / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
        summary = {"schema_version": 1, "status": "passed", "full_quality_rescore": True,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "formal_generations": len(pairs)*EXPECTED_SAMPLES,
                   "warmup_generations_excluded": warmup_count, "conditions": len(pairs), "results": results,
                   "recorded_files_index_sha256": snapshot[root / "recorded_files_sha256.json"]["sha256"],
                   "gpu_execution": False, "latency_remeasured": False,
                   "scope": "original metrics only; archived raw hashes and exact annotation bytes/IDs verified"}
        write_json(stage / "summary.json", summary)
        require({path for path in root.rglob("*") if path.is_file()} == expected_raw | {root / "recorded_files_sha256.json"},
                "Raw file coverage changed during CPU rescore")
        assert_unchanged(snapshot)
        require(not output.exists(), "Output appeared during CPU rescore; refusing to overwrite")
        stage.rename(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="completed original 16-run experiment")
    source.add_argument("--rescore-extracted", type=Path, help="extracted recorded_runs.tar.gz, CPU scoring only")
    parser.add_argument("--report", type=Path, help="complete passed report, required with --root")
    parser.add_argument("--data-root", type=Path, help="same original annotations, required with --rescore-extracted")
    parser.add_argument("--provenance", type=Path, help="optional directory containing allowlisted preparation/launch metadata")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are rejected")
    args = parser.parse_args(argv)
    if args.root is not None and (args.report is None or args.data_root is not None):
        parser.error("--root requires --report and does not accept --data-root")
    if args.rescore_extracted is not None and (args.data_root is None or args.report is not None or args.provenance is not None):
        parser.error("--rescore-extracted requires --data-root and does not accept --report/--provenance")
    try:
        if args.root is not None:
            result = export_experiment(args.root, args.report, args.output, provenance=args.provenance)
        else:
            result = rescore_extracted(args.rescore_extracted, args.data_root, args.output)
    except (ValueError, OSError, KeyError, TypeError, ImportError, tarfile.TarError) as error:
        parser.exit(2, f"Styled MileBench export/rescore failed: {error}\n")
    print(json.dumps({"status": result["status"], "formal_generations": result["formal_generations"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

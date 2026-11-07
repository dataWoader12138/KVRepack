#!/usr/bin/env python3
"""Export compact evidence from a complete, independently audited Qwen7B run.

This CPU-only command never runs inference or scoring and never changes its
inputs. Full per-head merge validation still requires the original raw bundles.
Only the full four-run, similarity / random-seed-0 / random-seed-1 matrix is
accepted. The output directory must not already exist.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile

METHODS = ("hybridkv_kvrepack", "snapkv_kvrepack")
TASKS = ("Spot-the-Diff", "MMCoQA")
LABELS = ("similarity", "random_global_seed0", "random_global_seed1")
REFERENCE = "6c65bf801ab1e5d0cd85f1d24991e3d5399bebcf"
RUN_FILES = ("run_contract.json", "resolved_config.json", "provenance.json",
             "model_runtime.json", "run_state.json", "ablation_state.json")
SMALL_FILE_LIMIT = 16 * 1024 * 1024
SMALL_TOTAL_LIMIT = 128 * 1024 * 1024
AUDIT_TABLES = {"results.csv": "results", "generation_stats.csv": "generation_stats",
                "merge_work_stats.csv": "merge_work_stats"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_sha(value, location):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"Missing or invalid SHA-256: {location}")
    return value


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def json_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def safe_path(path, *, directory=False):
    """Reject source links, including links in any ancestor directory."""
    path = Path(path).expanduser().absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlink is not allowed: {path}")
    require(path.is_dir() if directory else path.is_file(), f"Missing {'directory' if directory else 'file'}: {path}")
    return path.resolve(strict=True)


def file_sha(path):
    path = safe_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json(data):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(data, parse_constant=invalid)


class Exporter:
    def __init__(self, run_root, audit_root, staging):
        self.root, self.audit_root, self.staging = run_root, audit_root, staging
        self.sources = {}
        self.copied_bytes = 0

    def read(self, path, *, small=True, expected=None):
        path = safe_path(path)
        if small:
            require(path.stat().st_size <= SMALL_FILE_LIMIT, f"Evidence file exceeds small-file limit: {path}")
        data = path.read_bytes()
        identity = hashlib.sha256(data).hexdigest()
        require(expected is None or identity == expected, f"SHA-256 mismatch: {path}")
        require(self.sources.setdefault(path, identity) == identity, f"Source changed during export: {path}")
        return data

    def read_json(self, path, **kwargs):
        return parse_json(self.read(path, **kwargs))

    def copy(self, path, relative, *, expected=None):
        data = self.read(path, expected=expected)
        self.copied_bytes += len(data)
        require(self.copied_bytes <= SMALL_TOTAL_LIMIT, "Copied evidence exceeds total small-file limit")
        target = self.staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)

    def copy_audit(self, audit):
        require({p.name for p in self.audit_root.iterdir()} == {"audit.json", *AUDIT_TABLES},
                "Independent audit must contain exactly audit.json and its three CSV tables")
        self.copy(self.audit_root / "audit.json", "audit/audit.json")
        for name, key in AUDIT_TABLES.items():
            path, rows = self.audit_root / name, audit.get(key)
            require(isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows),
                    f"Independent audit is missing its embedded {key} table")
            reader = csv.DictReader(io.StringIO(self.read(path).decode("utf-8"), newline=""))
            actual = list(reader)
            require(reader.fieldnames == list(rows[0]) and len(actual) == len(rows),
                    f"Independent audit {name} columns/coverage mismatch")
            for index, (row, wanted) in enumerate(zip(actual, rows)):
                require(list(wanted) == list(rows[0]), f"Independent audit {key} row columns mismatch")
                for field, expected in wanted.items():
                    if type(expected) in (int, float):
                        try:
                            value = float(row[field])
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"Independent audit {name}:{index}:{field} is not numeric") from exc
                        require(math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-10),
                                f"Independent audit {name}:{index}:{field} value mismatch")
                    else:
                        require(row[field] == ("" if expected is None else str(expected)),
                                f"Independent audit {name}:{index}:{field} value mismatch")
            self.copy(path, Path("audit") / name)

    def copy_source_archives(self, source, name, evidence):
        """Copy only frozen source manifests, their bound files and package stubs."""
        source = safe_path(source, directory=True)
        require(not any(p.is_symlink() for p in source.rglob("*")), f"Symlink is not allowed in source archive: {source}")
        manifests = sorted(source.rglob("SNAPSHOT.json"))
        require(bool(manifests), f"Source archive requires a SNAPSHOT.json: {source}")
        covered = set()
        for marker in manifests:
            snapshot = safe_path(marker.parent, directory=True)
            bound = self.read_json(marker).get("source_files_sha256")
            require(isinstance(bound, dict) and bound
                    and any(bound == row.get("bound_source_files") for row in evidence),
                    "Source snapshot must match a final-audit bound source map")
            allowed = {"SNAPSHOT.json", *bound}
            for relative in bound:
                path = Path(relative)
                require(not path.is_absolute() and ".." not in path.parts and path.suffix in {".py", ".json"},
                        "Unsafe or unsupported frozen source path")
                for parent in path.parents:
                    if parent != Path("."):
                        allowed.add((parent / "__init__.py").as_posix())
            for path in sorted(snapshot.rglob("*")):
                require(not path.is_symlink(), f"Symlink is not allowed: {path}")
                if path.is_dir():
                    continue
                relative = path.relative_to(snapshot).as_posix()
                require(path.is_file() and relative in allowed, f"Unexpected file in source snapshot: {path}")
                require(path not in covered, "Nested source snapshots are not supported")
                expected = require_sha(bound[relative], str(path)) if relative in bound else None
                self.copy(path, Path(name) / path.relative_to(source), expected=expected)
                covered.add(path)
            require(all(snapshot.joinpath(relative) in covered for relative in bound), "Source snapshot is incomplete")
        require({p for p in source.rglob("*") if not p.is_dir()} == covered,
                "Source archive contains files outside its frozen snapshots")

    def launch(self):
        candidates = [self.root / "launch.json", self.root.parent / (self.root.name + "_launch.json"),
                      self.root.parent / (self.root.name + ".launch.json")]
        found = [p for p in candidates if p.exists() or p.is_symlink()]
        require(len(found) == 1, "Require exactly one launch.json or sibling <run-root>_launch.json / .launch.json")
        launch = self.read_json(found[0])
        outputs = [launch[key] for key in ("output", "output_root") if key in launch]
        require(bool(outputs) and all(Path(value).expanduser().resolve() == self.root for value in outputs),
                "Launch output does not identify --run-root")
        self.copy(found[0], "launch.json")
        if "continuation_driver_sha256" in launch:
            command = launch.get("command")
            require(isinstance(command, list) and all(isinstance(arg, str) for arg in command),
                    "Continuation launch command must be an argument list")
            drivers = [Path(arg).expanduser() for arg in command if Path(arg).suffix == ".py"]
            require(len(drivers) == 1, "Continuation launch must identify exactly one Python driver")
            driver = safe_path(drivers[0])
            require(driver.is_relative_to(self.root.parent), "Continuation driver must reside beside the experiment tree")
            self.copy(driver, "continuation_driver.py",
                      expected=require_sha(launch["continuation_driver_sha256"], "launch.continuation_driver_sha256"))
        return str(found[0])

    def copy_report(self, run, relative, evidence):
        report = safe_path(run / "report", directory=True)
        top_files = {"summary.json", "scores.csv", "paired_scores.csv", "selector_audit.csv", "RESULTS.md"}
        require({p.name for p in report.iterdir()} == top_files | {"conditions"},
                "Report tree has missing or unexpected files")
        conditions = safe_path(report / "conditions", directory=True)
        require({p.name for p in conditions.iterdir()} == set(LABELS), "Report condition coverage mismatch")
        names = {"eval.json", "eval_score.json", "evaluation_metadata.json"}
        expected_files = top_files | {f"conditions/{label}/{name}" for label in LABELS for name in names}
        hashes = evidence.get("report_files_sha256")
        require(isinstance(hashes, dict) and set(hashes) == expected_files,
                "Final audit must bind every report file in report_files_sha256")
        require(hashes["summary.json"] == evidence["report_summary_sha256"], "Audit report summary hashes disagree")
        for name in sorted(top_files):
            self.copy(report / name, relative / "report" / name, expected=require_sha(hashes[name], name))
        for label in LABELS:
            folder = safe_path(conditions / label, directory=True)
            require({p.name for p in folder.iterdir()} == names, f"Report {label} has missing or unexpected files")
            for name in sorted(names):
                expected = require_sha(hashes[f"conditions/{label}/{name}"], f"{label}/{name}")
                self.copy(folder / name, relative / "report" / "conditions" / label / name, expected=expected)

    def check_controller(self, controller, audit):
        matrix = [(method, task) for method in METHODS for task in TASKS]
        require(controller.get("schema_version") == 1 and controller.get("reference_commit") == REFERENCE,
                "Controller schema/reference mismatch")
        require(controller.get("status") == "complete" and controller.get("planned_runs") == 4
                and controller.get("completed_runs") == 4, "Controller must be complete with four completed runs")
        runs = controller.get("runs")
        require(isinstance(runs, list) and [(r.get("method"), r.get("task")) for r in runs] == matrix,
                "Controller matrix coverage/order mismatch")
        require(audit.get("schema_version") == 1 and audit.get("status") == "passed"
                and audit.get("controller_status") == "complete" and audit.get("scoring_performed") is True,
                "Independent final audit must have passed with complete controller and scoring")
        require(Path(audit.get("run_root", "")).expanduser().resolve() == self.root,
                "Independent audit belongs to another run root")
        require(require_sha(audit.get("controller_sha256"), "audit.controller_sha256")
                == self.sources[self.root / "controller_state.json"],
                "Independent audit has a stale controller SHA-256")
        evidence = audit.get("runs")
        require(isinstance(evidence, list) and [(r.get("method"), r.get("task")) for r in evidence] == matrix,
                "Independent audit matrix coverage/order mismatch")
        actual_contracts = {safe_path(p) for p in self.root.rglob("run_contract.json")}
        expected_contracts = {self.root / m / t / "run_contract.json" for m, t in matrix}
        require(actual_contracts == expected_contracts, "Run contract matrix has missing or extra runs")
        for row, checked in zip(runs, evidence):
            run = self.root / row["method"] / row["task"]
            require(row.get("status") == "complete" and type(row.get("returncode")) is int
                    and row["returncode"] == 0, "Controller contains an incomplete or failed run")
            require(Path(row.get("output", "")).resolve() == run
                    and Path(row.get("report", "")).resolve() == run / "report", "Controller run paths mismatch")
            require(checked.get("status") == "passed" and Path(checked.get("run", "")).resolve() == run
                    and Path(checked.get("report", "")).resolve() == run / "report", "Audit run status/path mismatch")
        return evidence

    def run(self, evidence, predictions, manifest):
        method, task = evidence["method"], evidence["task"]
        relative, run = Path(method) / task, self.root / method / task
        contract = self.read_json(run / "run_contract.json")
        identity = json_sha(contract)
        count = contract.get("annotation_total_samples")
        require(type(count) is int and count > 0 and contract.get("all_samples") is True
                and type(contract.get("limit")) is int and contract["limit"] == count,
                "Export requires a full-annotation contract")
        require(contract.get("method") == method and contract.get("task") == task
                and contract.get("reference_commit") == REFERENCE and contract.get("conditions") == list(LABELS)
                and contract.get("model_type") == "qwen2_5_vl-7b"
                and contract.get("head_geometry") == {"layers": 28, "native_kv_heads": 4},
                "Run contract identity/protocol mismatch")
        settings = [{"label": label, "strategy": "similarity" if i == 0 else "random_global",
                     "mask_seed": max(0, i - 1)} for i, label in enumerate(LABELS)]
        require(contract.get("condition_settings") == settings, "Contract condition settings mismatch")
        require(evidence.get("contract_sha256") == identity
                and evidence.get("annotation_sha256") == contract.get("annotation_sha256")
                and evidence.get("annotation_total_samples") == count
                and evidence.get("validated_bundles") == count and evidence.get("validated_generations") == count * 3
                and evidence.get("bound_source_files") == contract.get("source_files_sha256"),
                "Independent audit contract/coverage mismatch")
        sample_ids, generation_ids = contract.get("selected_sample_ids"), contract.get("generation_sample_ids")
        require(isinstance(sample_ids, list) and len(sample_ids) == count and len(set(sample_ids)) == count
                and all(type(i) is int for i in sample_ids) and isinstance(generation_ids, list)
                and len(generation_ids) == count and set(generation_ids) == set(sample_ids),
                "Contract sample ID coverage mismatch")
        maximum = contract.get("generation", {}).get("max_new_tokens")
        require(type(maximum) is int and maximum > 0, "Invalid generation maximum")
        state = self.read_json(run / "ablation_state.json")
        require(state.get("status") == "complete" and state.get("contract_sha256") == identity,
                "Ablation state is incomplete or bound to a different contract")
        for name, expected in (("target_samples", count), ("completed_samples", count),
                               ("target_generations", count * 3), ("completed_generations", count * 3)):
            require(type(state.get(name)) is int and state[name] == expected, f"Ablation state {name} mismatch")
        final = self.read_json(run / "run_state.json")
        require(type(final.get("returncode")) is int and final["returncode"] == 0
                and bool(final.get("finished_at")), "Runner final state must record successful completion")
        summary_hash = require_sha(evidence.get("report_summary_sha256"), "audit.report_summary_sha256")
        summary = self.read_json(run / "report" / "summary.json", expected=summary_hash)
        require(summary.get("status") == "complete" and summary.get("contract_sha256") == identity
                and summary.get("sample_count") == count and summary.get("generation_count") == count * 3
                and summary.get("full_task") is True, "Final report coverage/status mismatch")
        for name in RUN_FILES:
            self.copy(run / name, relative / name)
        self.copy_report(run, relative, evidence)
        samples = safe_path(run / "samples", directory=True)
        filenames = [f"{index:06d}.json" for index in range(count)]
        require(sorted(p.name for p in samples.iterdir()) == filenames, "Raw bundle filename coverage mismatch")
        listed = evidence.get("sample_files")
        require(isinstance(listed, list) and [r.get("sample_file") for r in listed] == filenames,
                "Independent audit sample manifest coverage mismatch")
        require(summary.get("sample_files") == listed, "Report/audit sample manifest mismatch")
        for index, item in enumerate(listed):
            path = samples / filenames[index]
            raw = self.read(path, small=False, expected=require_sha(item.get("sample_sha256"), str(path)))
            bundle = parse_json(raw)
            sample_id = generation_ids[index]
            require(bundle.get("schema_version") == 1 and bundle.get("status") == "completed"
                    and bundle.get("method") == method and bundle.get("task") == task
                    and bundle.get("contract_sha256") == identity and bundle.get("sample_index") == index
                    and bundle.get("sample_id") == item.get("sample_id") == sample_id
                    and bundle.get("fairness", {}).get("status") == "passed",
                    f"Raw bundle identity/completion mismatch: {path}")
            require(json_sha(bundle.get("input_fingerprints"))
                    == require_sha(item.get("input_fingerprints_sha256"), "audit.input_fingerprints_sha256"),
                    f"Raw bundle input fingerprint mismatch: {path}")
            records = bundle.get("records")
            require(isinstance(records, dict) and set(records) == set(LABELS), "Raw bundle condition coverage mismatch")
            for setting in settings:
                label = setting["label"]
                record = records[label]
                require(record.get("condition") == {k: setting[k] for k in ("strategy", "mask_seed")},
                        "Raw bundle condition settings mismatch")
                prediction, runtime = record.get("prediction", {}), record.get("runtime", {})
                require(prediction.get("sample_id") == sample_id and isinstance(prediction.get("pred_response"), str),
                        "Prediction identity/text mismatch")
                ids, eos = runtime.get("generated_ids"), runtime.get("eos_token_id")
                require(isinstance(ids, list) and 0 < len(ids) <= maximum
                        and all(type(i) is int and i >= 0 for i in ids)
                        and type(eos) is int and eos >= 0 and eos not in ids[:-1], "Invalid generated IDs/EOS")
                require(type(runtime.get("generated_tokens")) is int and runtime["generated_tokens"] == len(ids),
                        "Generation length disagrees with token IDs")
                for field, expected in (("eos_reached", ids[-1] == eos), ("hit_max_new_tokens", len(ids) == maximum),
                                        ("truncated_without_eos", len(ids) == maximum and ids[-1] != eos)):
                    require(type(runtime.get(field)) is bool and runtime[field] == expected,
                            f"Generation runtime {field} disagrees with token IDs")
                require(type(runtime.get("input_tokens")) is int and runtime["input_tokens"] > 0
                        and type(runtime.get("generation_seconds")) in (int, float)
                        and math.isfinite(runtime["generation_seconds"]) and runtime["generation_seconds"] >= 0,
                        "Invalid generation input count/duration")
                projected = {"method": method, "task": task, "sample_id": sample_id, "condition": label,
                             "pred_response": prediction["pred_response"], "max_new_tokens": maximum,
                             "bundle_sha256": item["sample_sha256"]}
                for key in ("generated_ids", "generated_tokens", "eos_token_id", "eos_reached",
                            "hit_max_new_tokens", "truncated_without_eos", "input_tokens", "generation_seconds"):
                    projected[key] = runtime[key]
                predictions.write(json_bytes(projected).decode("utf-8") + "\n")
            entry = {"method": method, "task": task, "sample_index": index, "sample_id": sample_id,
                     "sample_file": (relative / "samples" / path.name).as_posix(), "bytes": len(raw),
                     "sha256": item["sample_sha256"], "input_fingerprints_sha256": item["input_fingerprints_sha256"]}
            manifest.write(json_bytes(entry).decode("utf-8") + "\n")
        return {"method": method, "task": task, "sample_count": count, "generation_count": count * 3,
                "contract_sha256": identity, "annotation_sha256": contract["annotation_sha256"],
                "annotation_total_samples": count}


def export_results(run_root, audit_root, output):
    root, audit_root = safe_path(run_root, directory=True), safe_path(audit_root, directory=True)
    output = Path(output).expanduser().absolute()
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Output path must not contain symlinks")
    require(not output.exists(), f"Output directory already exists: {output}")
    output = output.resolve()
    for protected in (root, audit_root):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                "Output must be outside the experiment and independent audit trees")
    for path in root.glob("*/*/run_contract.json"):
        data = Path(parse_json(safe_path(path).read_bytes())["data_root"]).expanduser().resolve()
        require(not output.is_relative_to(data) and not data.is_relative_to(output),
                "Output must be outside annotation data")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent))
    reserved = False
    try:
        exporter = Exporter(root, audit_root, staging)
        controller = exporter.read_json(root / "controller_state.json")
        audit = exporter.read_json(audit_root / "audit.json")
        evidence = exporter.check_controller(controller, audit)
        exporter.copy(root / "controller_state.json", "controller_state.json")
        launch_source = exporter.launch()
        if (root / "reuse_manifest.json").exists() or (root / "reuse_manifest.json").is_symlink():
            exporter.copy(root / "reuse_manifest.json", "reuse_manifest.json")
        source_archives = []
        for name in ("source_archives", "source_snapshot"):
            if (root / name).exists() or (root / name).is_symlink():
                exporter.copy_source_archives(root / name, name, evidence)
                source_archives.append(name)
        exporter.copy_audit(audit)
        with (staging / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
                (staging / "bundle_manifest.jsonl").open("x", encoding="utf-8") as manifest:
            runs = [exporter.run(row, predictions, manifest) for row in evidence]
        # Recheck every source, including raw bundles, before publishing the archive.
        for path, identity in exporter.sources.items():
            require(file_sha(path) == identity, f"Source changed during export: {path}")
        files = [{"path": path.relative_to(staging).as_posix(), "bytes": path.stat().st_size,
                  "sha256": file_sha(path)} for path in sorted(staging.rglob("*")) if path.is_file()]
        metadata = {"schema_version": 1, "status": "complete", "exported_utc": datetime.now(timezone.utc).isoformat(),
                    "run_root": str(root), "audit_root": str(audit_root), "launch_source": launch_source,
                    "reference_commit": REFERENCE, "controller_status": "complete", "audit_status": "passed",
                    "controller_sha256": exporter.sources[root / "controller_state.json"],
                    "audit_sha256": exporter.sources[audit_root / "audit.json"],
                    "exporter_sha256": file_sha(Path(__file__)), "conditions": list(LABELS), "runs": runs,
                    "source_archive_directories": source_archives,
                    "bundle_count": sum(r["sample_count"] for r in runs),
                    "prediction_count": sum(r["generation_count"] for r in runs),
                    "raw_bundles_included": False, "merge_audit_reproducible_from_export": False,
                    "limitations": [
                        "Full per-head merge and fairness re-audit requires the original local raw samples/*.json bundles, verified by bundle_manifest.jsonl.",
                        "Predictions permit offline rescoring with the original external task annotations matching the recorded annotation SHA-256 values and the original scorer.",
                        "The final independent audit is copied evidence; this exporter checks identities, coverage, runtime fields and hashes, but does not rerun scoring or per-head merge validation.",
                        "Weights, images, questions, ground-truth text, raw bundles and per-head arrays are excluded from predictions.jsonl.",
                        "files records the SHA-256 and size of every exported file except EXPORT.json itself."],
                    "files": files}
        (staging / "EXPORT.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                             encoding="utf-8")
        # Reserve exclusively before rename: an existing user directory is never replaced.
        output.mkdir(exist_ok=False)
        reserved = True
        staging.rename(output)
        reserved = False
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if reserved:
            output.rmdir()  # Only remove our own empty reservation, never another tree.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export_results(args.run_root, args.audit_root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Image Random-head export failed: {error}\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "bundle_count": result["bundle_count"], "prediction_count": result["prediction_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

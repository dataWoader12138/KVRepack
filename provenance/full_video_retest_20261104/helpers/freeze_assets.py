#!/usr/bin/env python3
"""Freeze this retest's assets once; subsequent checks use file metadata only.

Run with the pinned experiment Python (pyarrow is needed only at creation).
No model/GPU imports, downloads, video decoding, or dependency-closure claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile


def critical_stat(s):
    return dict(st_dev=s.st_dev, st_ino=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def file_stat(path):
    return critical_stat(os.stat(path))


def inspect_file(path, category, hashed=True):
    path = Path(os.path.abspath(path))  # Preserve symlink spelling as well as target.
    resolved, before = str(path.resolve(strict=True)), file_stat(path)
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    digest = hashlib.sha256()
    if hashed:
        with path.open("rb") as stream:
            if critical_stat(os.fstat(stream.fileno())) != before:
                raise ValueError(f"File changed while opening: {path}")
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
            if critical_stat(os.fstat(stream.fileno())) != before:
                raise ValueError(f"Open file changed while hashing: {path}")
    if before != file_stat(path) or resolved != str(path.resolve(strict=True)):
        raise ValueError(f"File changed while recording: {path}")
    return dict(path=str(path), resolved_path=resolved, category=category,
                verification="sha256_at_freeze_then_stat" if hashed else "metadata_only",
                stat=before, **({"sha256": digest.hexdigest()} if hashed else {}))


def verify_manifest(path):
    """Return changed resources; stat/resolve only, never rehash or import ML code."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported/corrupt asset manifest")
    changed = []
    for entry in manifest["files"]:
        p = Path(entry["path"])
        try:
            now = file_stat(p)
            resolved = str(p.resolve(strict=True))
            if now != entry["stat"] or resolved != entry["resolved_path"] or not p.is_file():
                changed.append(dict(path=str(p), reason="stat_or_target_changed",
                                    before=entry["stat"], after=now,
                                    resolved_before=entry["resolved_path"], resolved_after=resolved))
        except OSError as exc:
            changed.append(dict(path=str(p), reason=type(exc).__name__))
    for absent in manifest.get("absent_paths", []):
        if os.path.lexists(absent):
            changed.append(dict(path=absent, reason="previously_absent_path_appeared"))
    return changed


def add_tree(add, root, category, suffixes=None):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Missing required directory: {root}")
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(x in {".git", "__pycache__", "build"} for x in p.parts):
            if suffixes is None or p.suffix in suffixes:
                add(p, category)


def add_dataset_videos(add, dataset):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    task, root = dataset["task"], Path(dataset["video_root"])
    column = "video" if task == "nextqa_oe_test" else "videoID"
    if task not in {"nextqa_oe_test", "vatex_test"}:
        raise ValueError(f"Unimplemented video path policy: {task}")
    with pa.memory_map(dataset["arrow_path"], "r") as source:
        try:
            table = ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            table = ipc.open_file(source).read_all()
    if table.num_rows != dataset["rows"]:
        raise ValueError(f"Arrow row count differs from inventory: {task}")
    references = table[column].to_pylist()
    paths = set()
    for value in references:
        name = str(value)
        if task == "nextqa_oe_test":
            choices = [root / (name if name.endswith(ext) else name + "." + ext)
                       for ext in ("mp4", "MP4", "mp4")]
        else:
            original = str(root / (name + ".mp4"))
            choices = [Path(original), Path(original.replace("mp4", "MP4")),
                       Path(original.replace("mp4", "mkv"))]
        selected = next((p for p in choices if p.is_file()), None)
        if selected is None or selected.stat().st_size == 0:
            raise ValueError(f"Missing/empty Arrow-referenced video: {task}:{name}")
        paths.add(selected)
    for p in sorted(paths):
        add(p, "video_metadata", hashed=False)
    return dict(task=task, rows=table.num_rows, arrow_path=dataset["arrow_path"],
                video_root=str(root), unique_referenced_videos=len(paths),
                video_path_policy="selected lmms task utility path/extension rules")


def freeze(inventory_path, repo, source_root):
    inventory_path, repo, source_root = map(lambda p: Path(p).absolute(),
                                          (inventory_path, repo, source_root))
    entries, absent, packages = {}, [], {}

    def add(p, category, hashed=True):
        key = os.path.abspath(p)
        if key not in entries:
            entries[key] = inspect_file(p, category, hashed)
        return entries[key]

    add(inventory_path, "inventory")
    inventory = json.loads(inventory_path.read_text())
    model = next(m for m in inventory["models"] if m["repo_id"] == "Qwen/Qwen2.5-VL-3B-Instruct")
    add_tree(add, model["snapshot_path"], "model_snapshot_all_files")
    if not any(e["path"].endswith(".safetensors") for e in entries.values()):
        raise ValueError("Model snapshot has no safetensors weight files")
    for shard in model["weight_shards"]:
        p = Path(model["snapshot_path"]) / shard["name"]
        if file_stat(p)["size"] != shard["bytes"]:
            raise ValueError(f"Model shard size differs from inventory: {p}")
    tasks = []
    for dataset in inventory["datasets"]:
        entry = add(dataset["arrow_path"], "dataset_arrow")
        if entry["sha256"] != dataset["arrow_sha256"]:
            raise ValueError(f"Arrow SHA differs from inventory: {dataset['task']}")
        tasks.append(add_dataset_videos(add, dataset))
    add_tree(add, repo / "src/kvrepack", "candidate_core_frontend", {".py", ".json"})
    add_tree(add, source_root / "hybridkv", "original_core", {".py", ".json"})
    heads = source_root / "visual_head/head_score"
    if heads.exists():
        add_tree(add, heads, "original_visual_head", {".json"})
    else:
        absent.append(str(heads))
    lmms = repo / "vendor/lmms-eval/lmms_eval"
    for p in sorted(lmms.glob("*.py")):
        add(p, "evaluation_frontend")
    for folder in ("api", "caching", "filters", "tasks/_task_utils"):
        add_tree(add, lmms / folder, "evaluation_frontend", {".py"})
    for folder in ("nextqa", "vatex"):
        add_tree(add, lmms / "tasks" / folder, "selected_task",
                 {".py", ".json", ".yaml", ".yml", ".txt", ".csv", ""})
    for name in ("models/__init__.py", "models/qwen2_5_vl.py", "tasks/__init__.py"):
        add(lmms / name, "adapter_frontend")
    for name in ("transformers", "qwen-vl-utils", "flash-attn", "torch"):
        dist = metadata.distribution(name)
        packages[name] = dict(version=dist.version,
                              payload=".py only; torch shared objects omitted" if name == "torch" else ".py and .so")
        for f in sorted(dist.files or [], key=str):
            suffix = Path(f).suffix
            if suffix == ".py" or (suffix == ".so" and name != "torch") or f.name in {"METADATA", "direct_url.json"}:
                add(dist.locate_file(f), "installed_" + name)
    nltk_root = Path(inventory["scoring_resources"]["nltk_resource_root"])
    if nltk_root.is_dir():
        for p in sorted(nltk_root.rglob("*")):
            if p.is_file():
                add(p, "nltk_metadata", hashed=False)
    else:
        absent.append(str(nltk_root))
    files = sorted(entries.values(), key=lambda e: e["path"])
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  python=dict(executable=sys.executable, version=platform.python_version()),
                  repo=str(repo), source_root=str(source_root), inventory_path=str(inventory_path),
                  model_snapshot=model["snapshot_path"],
                  tasks=tasks, installed_packages=packages, absent_paths=absent, files=files,
                  coverage={"initial_hash": "Full bytes of weights, Arrow, selected source and package payloads.",
                            "monitoring": "stat and symlink target only; no periodic byte rehash or proof against metadata-preserving byte changes",
                            "videos": "All unique Arrow-referenced video files: existence/nonzero/stat only; not decoded or hashed.",
                            "per_doc_proof": "Tracer separately hashes actual model input tensors, logits and KV cache tensors.",
                            "limitations": "Not all possible dependencies. torch .so/runtime/system libraries omitted; NLTK metadata only; new unlisted files are not monitored."},
                  summary=dict(file_count=len(files), by_category=dict(Counter(e["category"] for e in files)),
                               hashed_bytes=sum(e["stat"]["size"] for e in files if "sha256" in e)))
    # Ensure nothing drifted while the remaining assets were being hashed.
    for e in files:
        if file_stat(e["path"]) != e["stat"] or str(Path(e["path"]).resolve(strict=True)) != e["resolved_path"]:
            raise ValueError(f"Asset changed before manifest commit: {e['path']}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        saved = json.loads(args.output.read_text())
        if saved["repo"] != str(args.repo.absolute()) or saved["source_root"] != str(args.source_root.absolute()):
            raise ValueError("Existing manifest belongs to different source roots")
        if saved.get("inventory_path") != str(args.inventory.absolute()):
            raise ValueError("Existing manifest belongs to a different inventory")
        changes = verify_manifest(args.output)
        print(json.dumps(dict(status="changed" if changes else "unchanged", reused=True, changes=changes)))
        return 1 if changes else 0
    result = freeze(args.inventory, args.repo, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".asset_manifest-", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)  # Atomic create-only; never replace a prior ledger.
    finally:
        os.unlink(temporary)
    print(json.dumps(dict(status="frozen", output=str(args.output), **result["summary"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

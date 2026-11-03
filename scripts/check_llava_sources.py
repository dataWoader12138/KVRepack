#!/usr/bin/env python3
"""Replay the LLaVA image port without importing inference dependencies."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PORTS = {
    "qwen2_model_snapkv_repackkv.py": "qwen2_model_snapkv_kvrepack.py",
    "qwen2_model_sparsemm_repackkv.py": "qwen2_model_sparsemm_kvrepack.py",
    "qwen2_model_madakv_repackkv.py": "qwen2_model_madakv_kvrepack.py",
    "modeling_qwen2_repack.py": "modeling_qwen2_kvrepack.py",
    "hybridkv_repack_utils_llava.py": "hybridkv_utils_llava.py",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def restore_recorded_edits(text, record):
    """Recover exact prior bytes; never assume a global rename is invertible."""
    edits = record["reversible_edits"]
    assert edits == sorted(edits, key=lambda item: item["target_start"])
    end = 0
    for edit in edits:
        start, stop = edit["target_start"], edit["target_end"]
        assert end <= start <= stop <= len(text)
        assert text[start:stop] == edit["after"], "Recorded migration edit drift"
        end = stop
    for edit in reversed(edits):
        text = text[:edit["target_start"]] + edit["before"] + text[edit["target_end"]:]
    assert digest(text) == record["previous_target_sha256"], "Previous source bytes changed"
    return text


def transform_port_source(name, text):
    """Names/imports/aliases only, plus the declared SnapKV parameter adapter."""
    for old, new in PORTS.items():
        text = text.replace(old[:-3], new[:-3])
    text = text.replace("hybrid_decouplekv", "hybridkv_kvrepack").replace("decouplekv", "kvrepack")
    text = text.replace("DecoupleKV", "KVRepack").replace("DECOUPLEKV", "KVREPACK")
    text = re.sub(r"(?<![A-Za-z0-9])decouple(?![A-Za-z0-9])", "kvrepack", text)
    text = text.replace("DECOUPLE", "KVREPACK")
    text = re.sub(r"Decouple(?!d)", "KVRepack", text)
    text = text.replace("RepackKV", "KVRepack").replace("repackKV", "kvrepack").replace("repackkv", "kvrepack")
    text = re.sub(r"(?m)^(\s*(?:from|import)\s+)hybridkv(?=[.\s])", r"\1kvrepack", text)
    text = re.sub(r"\b(?:hybridkv_kernel|mmko_kernel)\b", "kvrepack_kernel", text)
    text = text.replace('{"hybridkv_kvrepack", "hybridkv_kvrepack", "hybridkv_kvrepack"}', '{"hybridkv_kvrepack"}')
    text = text.replace("{'hybridkv', 'hybridkv_kvrepack', 'hybridkv_kvrepack', 'hybridkv_kvrepack'}", "{'hybridkv', 'hybridkv_kvrepack'}")
    if name == "qwen2_model_snapkv_repackkv.py":
        old = '        alpha = _safe_get_env_float("K_DIFF_ALPHA")\n'
        new = ('        alpha = _safe_get_env_float("SNAPKV_KVREPACK_K_DIFF_ALPHA")\n'
               '        if alpha is None:\n'
               '            alpha = _safe_get_env_float("K_DIFF_ALPHA")\n')
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text


def check(source_root=None):
    manifest = json.loads((ROOT / "provenance/llava_image_sources.json").read_text())
    core = json.loads((ROOT / "provenance/core_sources.json").read_text())
    core_by_target = {entry["target_path"]: entry for entry in core["entries"]}
    checked = []
    for entry in manifest["entries"]:
        assert core_by_target[entry["target_path"]] == entry, "LLaVA/core provenance mismatch"
        target = (ROOT / entry["target_path"]).resolve()
        assert target.is_relative_to(ROOT / "src/kvrepack")
        text = target.read_text()
        assert digest(text) == entry["target_sha256"]
        before = restore_recorded_edits(text, entry["port_replay"])
        assert digest(before) == entry["source_sha256"]
        source_name = Path(entry["source_path"]).name
        assert PORTS[source_name] == target.name
        assert transform_port_source(source_name, before) == text, "Undeclared port transformation"
        assert digest(ast.dump(ast.parse(before), include_attributes=False)) == entry["source_ast_sha256"]
        assert digest(ast.dump(ast.parse(text), include_attributes=False)) == entry["renamed_source_ast_sha256"]
        if source_root is not None:
            assert digest((Path(source_root) / entry["source_path"]).read_text()) == entry["source_sha256"]
        checked.append(entry["target_path"])
    assert len(checked) == len(PORTS) == 5
    for resource in manifest["resource_adaptations"]:
        target = ROOT / resource["target_path"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == resource["target_sha256"]
        assert json.loads(target.read_text())["llava_ov"] == resource["target_llava_ov"]
        assert resource["target_llava_ov"]["budget_fixed_ratio"] == .1
        assert resource["source_llava_ov"]["budget_fixed_ratio"] == .5
        if source_root is not None:
            source = Path(source_root) / resource["source_path"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == resource["source_sha256"]
            assert json.loads(source.read_text())["llava_ov"] == resource["source_llava_ov"]
    for resource in manifest["reused_identical_resources"]:
        assert hashlib.sha256((ROOT / resource["target_path"]).read_bytes()).hexdigest() == resource["sha256"]
        if source_root is not None:
            assert hashlib.sha256((Path(source_root) / resource["source_path"]).read_bytes()).hexdigest() == resource["sha256"]
    return {"status": "passed", "ported_modules": checked, "source_comparison": source_root is not None,
            "scope": "Exact source recovery and deterministic declared transform replay; no model/GPU execution."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Optional parent of HybridKV-opensource")
    args = parser.parse_args()
    print(json.dumps(check(args.source_root), indent=2))

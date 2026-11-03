#!/usr/bin/env python3
"""Verify frozen MileBench inputs, target files, and preserved protocol ASTs."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvrepack.experiments.runner import verify_milebench_sources


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protocol_ast(text: str, class_name: str, *, source: bool = False) -> str:
    """Apply only declared base/import/docstring normalization, never arithmetic."""
    node = next(node for node in ast.parse(text).body
                if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = copy.deepcopy(node)
    if class_name == "MileBenchDataset":
        if source:
            assert len(node.bases) == 1 and isinstance(node.bases[0], ast.Name) and node.bases[0].id == "Dataset"
        else:
            assert not node.bases
        node.bases = []
    class Normalize(ast.NodeTransformer):
        def visit_Expr(self, value):
            if isinstance(value.value, ast.Constant) and isinstance(value.value.value, str):
                return None
            return self.generic_visit(value)

        def visit_Import(self, value):
            if class_name == "Eval" and len(value.names) == 1 and value.names[0].name == "numpy" and value.names[0].asname == "np":
                return None
            return value

        def visit_ImportFrom(self, value):
            if (class_name == "Eval" and value.level == 0 and value.module == "rouge"
                    and len(value.names) == 1 and value.names[0].name == "Rouge" and value.names[0].asname is None):
                return None
            return value
    return ast.dump(Normalize().visit(node), include_attributes=False)


def check(source_root: Path | None = None) -> dict:
    integrity = verify_milebench_sources(ROOT)
    manifest = json.loads((ROOT / "provenance/milebench_sources.json").read_text(encoding="utf-8"))
    preserved = []
    for entry in manifest["protocol_definitions"]:
        original = entry["source_definition"]
        assert digest(original.encode()) == entry["source_definition_sha256"]
        original_ast = protocol_ast(original, entry["class_name"], source=True)
        assert digest(original_ast.encode()) == entry["normalized_ast_sha256"]
        target_ast = protocol_ast((ROOT / entry["target_path"]).read_text(encoding="utf-8"), entry["class_name"])
        assert target_ast == original_ast, f"MileBench protocol changed: {entry['class_name']}"
        preserved.append(entry["class_name"])
    for entry in manifest["source_snapshots"]:
        if "text" in entry:
            assert digest(entry["text"].encode()) == entry["sha256"], f"Source snapshot changed: {entry['path']}"
        if source_root is not None:
            assert digest((source_root / entry["path"]).read_bytes()) == entry["sha256"], f"Original input changed: {entry['path']}"
    return {"status": "passed", "targets": len(integrity["verified_targets"]),
            "preserved_protocols": preserved, "source_snapshots": len(manifest["source_snapshots"]),
            "original_source_comparison": source_root is not None,
            "scope": "Source integrity and declared protocol AST equality; no model or CUDA execution"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Optional parent directory of HybridKV-opensource")
    args = parser.parse_args()
    print(json.dumps(check(args.source_root), indent=2))

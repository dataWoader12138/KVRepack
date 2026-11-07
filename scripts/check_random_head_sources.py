#!/usr/bin/env python3
"""Verify the exact 6c65bf8 head selectors and their unchanged canonical cores.

The default check needs only the shipped manifest and source files. With
``--compare-git`` it also compares the preserved files to the local reference
commit object. Neither mode loads inference dependencies or contacts GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def check(compare_git: bool = False) -> dict:
    manifest = json.loads((ROOT / "provenance/random_head_selector_sources.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Random-head source manifest")
    reference = manifest["source_commit"]
    checked = []
    for entry in manifest["entries"]:
        target = (ROOT / entry["target_path"]).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f"Missing local Random-head source: {entry['target_path']}")
        text = target.read_bytes()
        if digest(text) != entry["target_sha256"]:
            raise ValueError(f"Random-head target source drift: {entry['target_path']}")
        if entry["transformation"] == "exact_copy":
            recovered = text
        elif entry["transformation"] == "optional_torch_test_skip":
            before, after = entry["replacement"]["before"], entry["replacement"]["after"]
            decoded = text.decode("utf-8")
            if decoded.count(after) != 1:
                raise ValueError("Declared optional torch import guard changed")
            recovered = decoded.replace(after, before).encode("utf-8")
        else:
            raise ValueError("Undeclared Random-head source transformation")
        if digest(recovered) != entry["source_sha256"]:
            raise ValueError(f"Source recovery drift: {entry['target_path']}")
        if compare_git:
            original = subprocess.run(["git", "show", f"{reference}:{entry['source_path']}"],
                                      cwd=ROOT, capture_output=True, check=True).stdout
            if original != recovered:
                raise ValueError(f"Local reference commit differs: {entry['source_path']}")
        checked.append(entry["target_path"])
    compatible = []
    for entry in manifest["core_compatibility"]:
        actual = digest((ROOT / entry["path"]).read_bytes())
        if actual != entry["reference_sha256"] or actual != entry["target_sha256"]:
            raise ValueError(f"Canonical core differs from the reference selector contract: {entry['path']}")
        if compare_git:
            original = subprocess.run(["git", "show", f"{reference}:{entry['path']}"],
                                      cwd=ROOT, capture_output=True, check=True).stdout
            if digest(original) != actual:
                raise ValueError(f"Reference core changed: {entry['path']}")
        compatible.append(entry["path"])
    return {"status": "passed", "source_commit": reference, "files": checked,
            "unchanged_cores": compatible, "git_reference_comparison": compare_git,
            "boundary": "Source bytes, declared test import guard and core compatibility only; no model or GPU execution."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-git", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.compare_git), indent=2))

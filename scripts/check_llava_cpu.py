#!/usr/bin/env python3
"""Emit a CPU-only report for the LLaVA image migration's numerical/state tests."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def audit():
    spec = importlib.util.spec_from_file_location("kvrepack_llava_cpu_tests", ROOT / "tests/test_llava_image.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.LlavaCPUChecks)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    cls = module.LlavaCPUChecks
    extract = getattr(cls, "extract", None)
    guard = getattr(cls, "guard", None)
    return {
        "schema_version": 1, "passed": result.wasSuccessful() and not result.skipped,
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "cuda_entry_point_attempts": getattr(guard, "attempts", []),
        "test_output": stream.getvalue().splitlines(),
        "source_files": getattr(extract, "files", {}),
        "extracted_definitions": getattr(extract, "definitions", {}),
        "validated": [
            "LLaVA SparseMM caller initialization prefix + real baseline init/update: 8->16 tokens matches fresh object; decode keeps cluster/varlen metadata.",
            "LLaVA SparseMM upgrade recomputes per-prompt capacity; decode init keeps existing metadata.",
            "LLaVA SnapKV prefixed alpha takes effect and K_DIFF_ALPHA fallback remains effective.",
            "Original-source vs port tiny CPU merge outputs for SnapKV, MadaKV and HybridKV with ordinary Qwen2 RoPE, target lengths 1/2/3/4.",
        ],
        "boundary": "Extracted real function ASTs with tiny CPU tensors, not complete model construction/forward, attention kernels, generation, datasets, scores, GPU numerics, memory or latency. CUDA calls are blocked. Missing dependencies are skips, never success.",
    }


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)

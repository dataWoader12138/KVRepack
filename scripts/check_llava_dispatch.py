#!/usr/bin/env python3
"""Execute image-upgrade selection and real patches with no inference imports.

Each method has a fresh interpreter and source-labelled class/function doubles.
No tensors, complete forwards, weights, attention, CUDA or downloads execute.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dispatch import DispatchWorld, _NoInferenceImports, _execute, _function, _tree, find_transformers_root

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("hybridkv_kvrepack", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack")


def hybrid_cache_selection(world):
    """Execute the actual outer constructor's selection, then both cache gates."""
    path = ROOT / "src/kvrepack/modeling_llava_onevision.py"
    cls = next(n for n in _tree(path).body if isinstance(n, ast.ClassDef) and n.name == "LlavaOnevisionModel")
    init = _function(cls, "__init__")
    gate = next(n for n in init.body if isinstance(n, ast.If) and "METHOD" in ast.unparse(n.test))
    selected = []
    for name in ("kvrepack.modeling_qwen2", "kvrepack.modeling_qwen2_kvrepack"):
        model = world._core_class(name, "Qwen2Model")
        type.__setattr__(model, "_from_config", classmethod(lambda cls, config: selected.append(cls.__module__) or object()))
    instance = SimpleNamespace()
    _execute([gate], path, {"self": instance, "os": os, "config": SimpleNamespace(text_config=object()),
                           "Qwen2Model": world.modules["kvrepack.modeling_qwen2"].Qwen2Model})
    assert selected == ["kvrepack.modeling_qwen2_kvrepack"], selected
    model_path = ROOT / "src/kvrepack/modeling_qwen2_kvrepack.py"
    branch = next(n for n in ast.walk(_tree(model_path)) if isinstance(n, ast.If)
                  and any(isinstance(s, ast.Assign) and isinstance(s.value, ast.Call)
                          and isinstance(s.value.func, ast.Name) and s.value.func.id == "HybridKVCache" for s in n.body))
    calls = []
    _execute([branch], model_path, {
        "self": SimpleNamespace(layers=[], config=object()), "method": os.environ["METHOD"],
        "inputs_embeds": SimpleNamespace(shape=(1, 8), device="cpu-double", dtype="double"),
        "MAX_LEN": 8, "sparse_budget": 1024, "sparse_budget_ratio": .1,
        "is_vision_token_mask": object(), "is_text_token_mask": object(),
        "HybridKVCache": lambda *a, **k: calls.append("HybridKVCache") or object(),
        "HybridKVCache_CPU": lambda *a, **k: calls.append("CPU") or object(),
    })
    assert calls == ["HybridKVCache"], calls
    cache_path = ROOT / "src/kvrepack/hybridkv_utils_llava.py"
    cache_cls = next(n for n in _tree(cache_path).body if isinstance(n, ast.ClassDef) and n.name == "HybridKVCache")
    cache_init = _function(cache_cls, "__init__")
    switches = [n for n in cache_init.body if isinstance(n, ast.Assign)
                and any(ast.unparse(t) in {"self.hybridkv_kvrepack", "self.chunk_size"} for t in n.targets)]
    state = SimpleNamespace()
    _execute(switches, cache_path, {"self": state, "os": os})
    assert state.hybridkv_kvrepack is True
    return {"language_backbone": selected[0], "cache_class": "kvrepack.hybridkv_utils_llava.HybridKVCache",
            "kvrepack_enabled": state.hybridkv_kvrepack, "chunk_size": state.chunk_size}


def worker(method, transformers_root):
    sys.path.insert(0, str(ROOT / "src"))
    from kvrepack.experiments.milebench import method_spec
    selected = method_spec(method, "llava_ov", "main11")
    os.environ.clear()
    os.environ.update({key: param.default for key, param in selected.parameters.items()},
                      METHOD=method, MODEL_TYPE="llava-ov-7b")
    sys.meta_path.insert(0, _NoInferenceImports())
    result = {"method": method, "worker_pid": os.getpid(), "passed": False, "errors": []}
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            world = DispatchWorld(transformers_root)
            assert not world.events
            workers = importlib.import_module("kvrepack.benchmarks.milebench.workers")
            cls = workers.select_model_class("llava_ov", method, profile="main11")
            cls.from_pretrained("<loader-double>")
            result["adapter_class"] = world.loads[0]["class"]
            result["patch_targets"] = world.events
            if method == "hybridkv_kvrepack":
                assert cls.__module__ == "kvrepack.modeling_llava_onevision"
                result["hybrid_cache_check"] = hybrid_cache_selection(world)
                assert not world.events
            else:
                assert cls.__module__.startswith("transformers.")
                assert world.events
                assert any(p["target"].startswith("kvrepack.qwen2_model_") and "KVRepack" in p["target"] for p in world.events)
            assert world.algorithm_calls == 0 and "torch" not in sys.modules
            result["algorithm_functions_executed"] = world.algorithm_calls
            result["passed"] = True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["dispatch_stdout"] = captured.getvalue().splitlines()
    return result


def audit(transformers_root=None):
    root = find_transformers_root(transformers_root)
    rows = []
    for method in METHODS:
        result = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                 "--worker", method, "--transformers-root", str(root)],
                                capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
        rows.append(json.loads(result.stdout) if result.returncode == 0 else {
            "method": method, "passed": False, "errors": [result.stderr or result.stdout]})
    pids = [row.get("worker_pid") for row in rows]
    return {"schema_version": 1, "scope": "Four image-only LLaVA upgrades; real loader/patch/cache selection with dependency doubles. No model/CPU tensor/GPU execution.",
            "combinations": rows, "passed": all(row["passed"] for row in rows) and None not in pids and len(set(pids)) == 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers-root")
    parser.add_argument("--worker", choices=METHODS)
    args = parser.parse_args()
    root = find_transformers_root(args.transformers_root)
    result = worker(args.worker, root) if args.worker else audit(root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if args.worker or result["passed"] else 1)

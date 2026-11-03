#!/usr/bin/env python3
"""Audit real dispatch code with standard-library dependency doubles, never tensors.

Each model/method pair runs in its own isolated Python process. Adapter selection
statements are compiled directly from their AST; monkeypatch.py and the official
AsymKV installer execute unchanged. Algorithm implementations and model loaders
are source-labelled doubles and are deliberately never executed.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = (
    "Executes actual adapter AST dispatch, monkeypatch assignments, official AsymKV "
    "installer, prepare-input forwarding/reset, and selected Hybrid cache switches. "
    "Compression functions, model loading, attention, tensors and CUDA are NOT executed. "
    "Transformers class/method existence and selected model construction links are "
    "checked against the supplied installed source, without importing transformers."
)


@lru_cache(maxsize=None)
def _tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))


def _label(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _source(path, node):
    return {"source_path": _label(path), "line": node.lineno}


def _function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _registry():
    name = "_kvrepack_dispatch_registry"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/kvrepack/experiments/registry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_transformers_root(value=None):
    if value:
        root = Path(value).expanduser().resolve()
    else:
        try:
            root = Path(importlib.metadata.distribution("transformers").locate_file("transformers")).resolve()
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError("Pass --transformers-root /path/to/site-packages/transformers; only source files are read") from exc
    if not (root / "__init__.py").is_file():
        raise ValueError(f"No transformers source package at {root}")
    return root


def _execute(statements, path, namespace):
    tree = ast.Module(body=statements, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


class _NoInferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "accelerate", "flash_attn", "decord", "numpy", "PIL"}:
            raise RuntimeError(f"Dispatch audit forbids inference dependency import: {fullname}")
        return None


class DispatchWorld:
    """Source-derived classes make patches to nonexistent HF targets fail loudly."""

    def __init__(self, transformers_root):
        self.transformers_root = Path(transformers_root)
        self.events = []
        self.loads = []
        self.original_calls = []
        self.modules = {}
        self.hf_classes = {}
        self.paths = {}
        self.algorithm_calls = 0
        self._module("kvrepack", ROOT / "src/kvrepack")
        self._module("transformers", self.transformers_root)
        for family in ("qwen2_5_vl", "qwen2", "llava_onevision"):
            name = f"transformers.models.{family}.modeling_{family}"
            path = self.transformers_root / "models" / family / f"modeling_{family}.py"
            self.paths[name] = path
            self._hf_module(name, path)
        llava = self.hf_classes["transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration"]
        self.modules["transformers"].LlavaOnevisionForConditionalGeneration = llava
        self._core_imports(ROOT / "src/kvrepack/monkeypatch.py", "kvrepack.monkeypatch")
        self._core_class("kvrepack.modeling_qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration")
        self._core_class("kvrepack.modeling_llava_onevision", "LlavaOnevisionForConditionalGeneration")
        self._install_real_asymkv()
        self.monkeypatch = self._real_module("kvrepack.monkeypatch", ROOT / "src/kvrepack/monkeypatch.py")

    def _module(self, name, package_path=None):
        if name in self.modules:
            return self.modules[name]
        module = ModuleType(name)
        module.__path__ = [str(package_path)] if package_path else []
        module.__package__ = name if package_path else name.rpartition(".")[0]
        self.modules[name] = sys.modules[name] = module
        if "." in name:
            parent, _, child = name.rpartition(".")
            setattr(self._module(parent), child, module)
        return module

    def _callable(self, name, path, node):
        def algorithm_double(*args, **kwargs):
            self.algorithm_calls += 1
            raise AssertionError(f"Compression double must not execute: {name}")
        algorithm_double.__module__, _, algorithm_double.__qualname__ = name.rpartition(".")
        algorithm_double._dispatch_source = {"target": name, **_source(path, node)}
        return algorithm_double

    def _class(self, name, path, node, bases=()):
        world = self

        if not hasattr(self, "_record_meta"):
            class RecordWrites(type):
                def __setattr__(cls, key, value):
                    class_name = cls.__module__ + "." + cls.__qualname__
                    if key in {"forward", "generate", "prepare_inputs_for_generation"} and not hasattr(cls, key):
                        raise AssertionError(f"Patch target does not exist in transformers source: {class_name}.{key}")
                    caller = sys._getframe(1)
                    if callable(value):
                        ref = getattr(value, "_dispatch_source", {
                            "target": value.__module__ + "." + value.__qualname__,
                            "source_path": _label(caller.f_code.co_filename),
                            "line": getattr(value, "__code__", caller.f_code).co_firstlineno,
                        })
                        world.events.append({"attribute": class_name + "." + key, **ref,
                                             "patch_source_path": _label(caller.f_code.co_filename),
                                             "patch_line": caller.f_lineno})
                    type.__setattr__(cls, key, value)
            self._record_meta = RecordWrites

        attrs = {n.name: self._callable(name + "." + n.name, path, n)
                 for n in node.body if isinstance(n, ast.FunctionDef)}
        attrs.update({"__module__": name.rpartition(".")[0], "__qualname__": name.rpartition(".")[2],
                      "_dispatch_source": {"target": name, **_source(path, node)}})
        if name.endswith("ForConditionalGeneration"):
            def from_pretrained(cls, *args, **kwargs):
                world.loads.append({"class": name, **_source(path, node), "kwargs": kwargs})
                return SimpleNamespace(eval=lambda: SimpleNamespace(), dtype="model-loader-double")
            attrs["from_pretrained"] = classmethod(from_pretrained)
        return self._record_meta(name.rpartition(".")[2], bases or (object,), attrs)

    def _hf_module(self, name, path):
        module = self._module(name)
        classes = {}
        # GenerationMixin is an inherited public HF method, not a model-local def.
        gen_path = self.transformers_root / "generation/utils.py"
        generate = _function(_tree(gen_path), "generate")
        generation_base = type("GenerationMixinDouble", (), {
            "generate": self._callable("transformers.generation.utils.GenerationMixin.generate", gen_path, generate)
        })
        for node in _tree(path).body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [classes[b.id] for b in node.bases if isinstance(b, ast.Name) and b.id in classes]
            if any(isinstance(b, ast.Name) and b.id == "GenerationMixin" for b in node.bases):
                bases.append(generation_base)
            cls = self._class(name + "." + node.name, path, node, tuple(bases))
            classes[node.name] = cls
            setattr(module, node.name, cls)
            self.hf_classes[name + "." + node.name] = cls
        conditional = classes.get("Qwen2_5_VLForConditionalGeneration")
        if conditional is not None:
            original = conditional.prepare_inputs_for_generation

            def original_prepare(instance, *args, **kwargs):
                self.original_calls.append((instance, args, kwargs))
                return self.original_calls[-1]

            original_prepare._dispatch_source = original._dispatch_source
            # Baseline sentinel replacement is fixture setup, never an audited patch.
            type.__setattr__(conditional, "prepare_inputs_for_generation", original_prepare)

    def _core_class(self, module_name, symbol):
        module = self._module(module_name)
        if hasattr(module, symbol):
            return getattr(module, symbol)
        path = ROOT / "src" / Path(*module_name.split(".")).with_suffix(".py")
        node = next(n for n in _tree(path).body if isinstance(n, ast.ClassDef) and n.name == symbol)
        cls = self._class(module_name + "." + symbol, path, node)
        setattr(module, symbol, cls)
        return cls

    def _core_imports(self, path, module_name):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom):
                continue
            name = importlib.util.resolve_name("." * node.level + (node.module or ""), module_name.rpartition(".")[0]) if node.level else node.module
            if not name or not name.startswith("kvrepack.") or name.endswith(".integration"):
                continue
            module = self._module(name)
            source = ROOT / "src" / Path(*name.split(".")).with_suffix(".py")
            definitions = {n.name: n for n in _tree(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for alias in node.names:
                definition = definitions[alias.name]
                if isinstance(definition, ast.ClassDef):
                    self._core_class(name, alias.name)
                else:
                    setattr(module, alias.name, self._callable(name + "." + alias.name, source, definition))

    def _real_module(self, name, path):
        module = self._module(name)
        module.__file__ = str(path)
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        return module

    def _install_real_asymkv(self):
        path = ROOT / "src/kvrepack/asymkv_official/integration.py"
        self._core_imports(path, "kvrepack.asymkv_official.integration")
        self._real_module("kvrepack.asymkv_official.integration", path)

    def adapter(self, family, method):
        adapter = "llava_hf" if family == "llava_ov" else "qwen2_5_vl"
        path = ROOT / "vendor/lmms-eval/lmms_eval/models" / (adapter + ".py")
        tree = _tree(path)
        init = _function(tree, "__init__")
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Import) and any(a.name == "os" for a in n.names))
        end = next(i for i in range(start, len(init.body))
                   if any(isinstance(n, ast.Assign) and any(ast.unparse(t) == "self._model" for t in n.targets)
                          for n in ast.walk(init.body[i])))
        statements = init.body[start:end + 1]
        namespace = {
            "self": SimpleNamespace(device_map="audit-device"), "method": method,
            "pretrained": "<model-loader-double>", "attn_implementation": "flash_attention_2",
            "use_custom_video_loader": False, "fps": None, "max_image_size": None,
            "Accelerator": lambda: SimpleNamespace(num_processes=1),
            "torch": SimpleNamespace(device=lambda value: value, bfloat16="bfloat16-double"),
            "device": "cpu-double", "device_map": "audit-device", "dtype": "bfloat16-double",
            "max_frames_num": 64, "revision": "main", "trust_remote_code": False,
            "AutoConfig": SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(model_type="llava_onevision")),
            "LlavaOnevisionForConditionalGeneration": self.modules["transformers"].LlavaOnevisionForConditionalGeneration,
        }
        if family == "llava_ov":
            maps = [n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "model_map" for t in n.targets)]
            _execute(maps, path, namespace)
        _execute(statements, path, namespace)
        if len(self.loads) != 1:
            raise AssertionError(f"Adapter must choose exactly one model class, got {self.loads}")
        return {"path": _label(path), "line": init.lineno,
                "extracted_ranges": [[statements[0].lineno, statements[-1].end_lineno]]}

    def prepare_check(self):
        patches = [p for p in self.events if p["attribute"].endswith(".prepare_inputs_for_generation")]
        if not patches:
            return {"applicable": False, "reason": "Selected route installs no prepare-input wrapper"}
        cls = self.hf_classes["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration"]
        wrapper = cls.prepare_inputs_for_generation
        fields = ("attention_mask", "inputs_embeds", "cache_position", "position_ids", "pixel_values",
                  "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts", "extra_sentinel")
        cases = []
        for case, cache in (("none", None), ("legacy_tuple", (object(),)),
                            ("empty", SimpleNamespace(key_cache=[])), ("nonempty", SimpleNamespace(key_cache=[object()]))):
            layers = [SimpleNamespace(self_attn=SimpleNamespace(kv_seq_len=value)) for value in (23, 24)]
            instance = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
            token = object()
            args = {name: object() for name in fields}
            args.update(past_key_values=cache, use_cache=False)
            result = wrapper(instance, token, **args)
            forwarded = result[0] is instance and result[1] == (token,) and all(result[2].get(k) is v for k, v in args.items())
            lengths = [layer.self_attn.kv_seq_len for layer in layers]
            reset = lengths == ([0, 0] if case == "empty" else [23, 24])
            if not forwarded or not reset:
                raise AssertionError(f"prepare-input wrapper failed case={case}: forwarding={forwarded}, reset={lengths}")
            cases.append({"cache_case": case, "all_arguments_forwarded_by_identity": forwarded,
                          "kv_seq_len_after": lengths, "empty_cache_reset_correct": reset})
        return {"applicable": True, "cases": cases, "source": patches[-1]}

    def hybrid_check(self, family, method, adapter_class):
        if not adapter_class.startswith("kvrepack."):
            return {"applicable": False}
        model_path = ROOT / "src/kvrepack" / ("modeling_qwen2.py" if family == "llava_ov" else "modeling_qwen2_5_vl.py")
        tree = _tree(model_path)
        cache_branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                            and any(isinstance(s, ast.Assign) and isinstance(s.value, ast.Call)
                                    and isinstance(s.value.func, ast.Name) and s.value.func.id == "HybridKVCache" for s in n.body))
        selected = []
        namespace = {"method": method, "self": SimpleNamespace(layers=[], config=SimpleNamespace()),
                     "inputs_embeds": SimpleNamespace(shape=(1, 4), device="cpu-double", dtype="double"),
                     "MAX_LEN": 4, "sparse_budget_ratio": 0.1, "is_vision_token_mask": object(),
                     "HybridKVCache": lambda *a, **k: selected.append("HybridKVCache") or "HybridKVCache",
                     "HybridKVCache_CPU": lambda *a, **k: selected.append("HybridKVCache_CPU") or "HybridKVCache_CPU"}
        _execute([cache_branch], model_path, namespace)
        if selected != ["HybridKVCache"]:
            raise AssertionError(f"Custom model selected wrong cache: {selected}")
        result = {"applicable": True, "cache_class": "kvrepack.hybridkv_utils.HybridKVCache",
                  "model_switch_source": _source(model_path, cache_branch)}
        if family == "qwen2_5_vl":
            path = ROOT / "src/kvrepack/hybridkv_utils.py"
            cache_cls = next(n for n in _tree(path).body if isinstance(n, ast.ClassDef) and n.name == "HybridKVCache")
            init = next(n for n in cache_cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            statements = [n for n in init.body if isinstance(n, ast.Assign)
                          and any(ast.unparse(t) in {"method", "self.hybrid_asymkv", "self.hybridkv_kvrepack", "self.chunk_size"} for t in n.targets)]
            state = SimpleNamespace()
            _execute(statements, path, {"self": state, "os": os})
            expected = method == "hybridkv_kvrepack"
            if state.hybridkv_kvrepack != expected or state.hybrid_asymkv:
                raise AssertionError("Hybrid KVRepack cache gate disagrees with selected METHOD")
            result.update(kvrepack_enabled=state.hybridkv_kvrepack, asymkv_enabled=state.hybrid_asymkv,
                          chunk_size=state.chunk_size, cache_switch_source=_source(path, statements[0]))
        return result


def transformers_structure(root):
    root = Path(root)
    version_nodes = [n for n in _tree(root / "__init__.py").body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    checks = []
    checks_requested = (
        ("qwen2", "Qwen2DecoderLayer", "self.self_attn", "Qwen2Attention"),
        ("qwen2", "Qwen2Model", "self.layers", "nn.ModuleList"),
        ("llava_onevision", "LlavaOnevisionModel", "self.language_model", "AutoModel.from_config"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration", "self.model", "LlavaOnevisionModel"),
        ("qwen2_5_vl", "Qwen2_5_VLModel", "self.language_model", "Qwen2_5_VLTextModel._from_config"),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration", "self.model", "Qwen2_5_VLModel"),
    )
    files = {"__init__.py", "generation/utils.py"}
    for family, cls, attribute, constructor in checks_requested:
        relative = f"models/{family}/modeling_{family}.py"
        files.add(relative)
        path = root / relative
        node = next(n for n in _tree(path).body if isinstance(n, ast.ClassDef) and n.name == cls)
        assignment = next(n for n in ast.walk(node) if isinstance(n, ast.Assign)
                          and any(ast.unparse(t) == attribute for t in n.targets))
        observed = ast.unparse(assignment.value.func) if isinstance(assignment.value, ast.Call) else None
        checks.append({"class": cls, "attribute": attribute, "constructor": observed,
                       "expected_constructor": constructor, "passed": observed == constructor,
                       **_source(path, assignment)})
    auto_relative = "models/auto/modeling_auto.py"
    files.add(auto_relative)
    auto_path = root / auto_relative
    auto_map = next(n for n in _tree(auto_path).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "MODEL_MAPPING_NAMES" for t in n.targets))
    qwen2 = dict(ast.literal_eval(auto_map.value.args[0])).get("qwen2")
    checks.append({"class": "AutoModel", "attribute": "MODEL_MAPPING_NAMES[qwen2]", "constructor": qwen2,
                   "expected_constructor": "Qwen2Model", "passed": qwen2 == "Qwen2Model", **_source(auto_path, auto_map)})
    qwen_path = root / "models/qwen2_5_vl/modeling_qwen2_5_vl.py"
    attention_map = next(n for n in _tree(qwen_path).body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "QWEN2_5_VL_ATTENTION_CLASSES" for t in n.targets))
    attention = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(attention_map.value.keys, attention_map.value.values)}
    checks.append({"class": "Qwen2_5_VLDecoderLayer", "attribute": "QWEN2_5_VL_ATTENTION_CLASSES[flash_attention_2]",
                   "constructor": attention.get("flash_attention_2"), "expected_constructor": "Qwen2_5_VLFlashAttention2",
                   "passed": attention.get("flash_attention_2") == "Qwen2_5_VLFlashAttention2", **_source(qwen_path, attention_map)})
    version = ast.literal_eval(version_nodes[0].value)
    return {"root": str(root), "version": version, "expected_version": "4.52.3",
            "files_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sorted(files)},
            "structure_checks": checks, "passed": version == "4.52.3" and all(c["passed"] for c in checks)}


def worker(family, public_method, transformers_root):
    registry = _registry()
    spec, parameters, _ = registry.resolve_experiment(public_method, model=family)
    os.environ.clear()
    os.environ.update(parameters, METHOD=spec.internal_method, MODEL_TYPE="llava-ov-7b" if family == "llava_ov" else "qwen2_5_vl-3b")
    sys.meta_path.insert(0, _NoInferenceImports())
    result = {"model": family, "public_method": public_method, "internal_method": spec.internal_method,
              "worker_pid": os.getpid(), "algorithm_functions_executed": 0, "errors": [], "passed": False}
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            world = DispatchWorld(transformers_root)
            result["initial_patch_count"] = len(world.events)
            if world.events:
                raise AssertionError("Fresh worker already contains dispatch patches")
            result["adapter_source"] = world.adapter(family, spec.internal_method)
            result["adapter_class"] = world.loads[0]["class"]
            result["adapter_class_source"] = world.loads[0]
            result["patch_targets"] = [p for p in world.events if p["attribute"].rsplit(".", 1)[-1]
                                       in {"forward", "generate", "prepare_inputs_for_generation"}]
            result["prepare_inputs_check"] = world.prepare_check()
            result["hybrid_cache_check"] = world.hybrid_check(family, spec.internal_method, result["adapter_class"])
            result["algorithm_functions_executed"] = world.algorithm_calls
            if world.algorithm_calls or "torch" in sys.modules:
                raise AssertionError("Inference code was executed or torch imported")
            expected_custom = spec.model_path.startswith("custom_")
            if result["adapter_class"].startswith("kvrepack.") != expected_custom:
                raise AssertionError("Adapter class family conflicts with registry model_path")
            if public_method == "fullkv" and result["patch_targets"]:
                raise AssertionError("FullKV unexpectedly patched transformers")
            if public_method != "fullkv" and not expected_custom and not result["patch_targets"]:
                raise AssertionError("Compressed native route installed no patches")
            result["passed"] = True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["dispatch_stdout"] = captured.getvalue().splitlines()
    return result


def audit(transformers_root=None):
    root = find_transformers_root(transformers_root)
    structure = transformers_structure(root)
    registry = _registry()
    combinations = []
    for family, methods in registry.METHODS_BY_MODEL.items():
        for method in methods:
            command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                       "--worker", family, method, "--transformers-root", str(root)]
            process = subprocess.run(command, capture_output=True, text=True, check=False,
                                     env={"PATH": os.environ.get("PATH", "")})
            if process.returncode:
                combinations.append({"model": family, "public_method": method, "passed": False,
                                     "errors": [process.stderr.strip() or "Isolated worker failed"]})
            else:
                combinations.append(json.loads(process.stdout))
    pids = [c.get("worker_pid") for c in combinations]
    isolated = None not in pids and len(set(pids)) == len(pids)
    return {"schema_version": 1, "evidence_boundary": BOUNDARY,
            "transformers_source": structure, "combinations": combinations,
            "separate_process_per_combination": isolated,
            "passed": structure["passed"] and isolated and all(c["passed"] for c in combinations)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers-root", help="Installed transformers package directory (read-only AST inspection)")
    parser.add_argument("--worker", nargs=2, metavar=("MODEL", "METHOD"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        root = find_transformers_root(args.transformers_root)
        result = worker(*args.worker, root) if args.worker else audit(root)
    except (OSError, ValueError, StopIteration) as exc:
        result = {"schema_version": 1, "evidence_boundary": BOUNDARY, "passed": False,
                  "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # Workers report a failed audit as JSON so the parent can retain its evidence.
    return 0 if args.worker or result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

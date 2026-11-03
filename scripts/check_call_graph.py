#!/usr/bin/env python3
"""Measure local Python call relationships without importing inference packages.

This is a conservative direct-call graph, not Python/PyTorch whole-program
analysis. Receiver types, framework callbacks and **kwargs can remain unresolved.
Use check_dispatch.py for the runtime method-selection evidence.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return prefix + "." + node.attr if prefix else None
    return None


def import_bindings(tree, module, is_package=False):
    bindings = {}
    parent = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pairs = [(a.asname or a.name.split(".")[0], a.name if a.asname else a.name.split(".")[0])
                     for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            if node.level:
                parts = parent.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                prefix = base + ("." + prefix if prefix else "")
            pairs = [(a.asname or a.name, prefix + "." + a.name) for a in node.names if a.name != "*"]
        else:
            continue
        for local, target in pairs:
            bindings.setdefault(local, set()).add(target)
    return {key: next(iter(values)) for key, values in bindings.items() if len(values) == 1}


def direct_nodes(body):
    """Walk a scope, excluding the bodies of child definitions."""
    for node in body:
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from direct_nodes(ast.iter_child_nodes(node))


def binding_errors(function, call, *, bound=False):
    """Only diagnose calls whose provided arguments are statically unambiguous."""
    args = function.args
    positional = list(args.posonlyargs) + list(args.args)
    required = len(positional) - len(args.defaults)
    if bound and positional:
        positional = positional[1:]
        required = max(0, required - 1)
    if any(isinstance(a, ast.Starred) for a in call.args):
        return [], "unresolved_star_args"
    if any(k.arg is None for k in call.keywords):
        return [], "unresolved_star_kwargs"
    keywords = [k.arg for k in call.keywords]
    errors = []
    if not args.vararg and len(call.args) > len(positional):
        errors.append("too_many_positional_arguments")
    posnames = [a.arg for a in positional]
    posonly = {a.arg for a in args.posonlyargs}
    accepted = (set(posnames) - posonly) | {a.arg for a in args.kwonlyargs}
    if not args.kwarg and set(keywords) - accepted:
        errors.append("unexpected_keywords:" + ",".join(sorted(set(keywords) - accepted)))
    duplicate = set(keywords) & set(posnames[:len(call.args)])
    if duplicate:
        errors.append("duplicate_arguments:" + ",".join(sorted(duplicate)))
    missing = set(posnames[len(call.args):required]) - set(keywords)
    missing |= {a.arg for a, default in zip(args.kwonlyargs, args.kw_defaults)
                if default is None and a.arg not in keywords}
    if missing:
        errors.append("missing_required_arguments:" + ",".join(sorted(missing)))
    return errors, "checked"


def audit(root=ROOT, dispatch=None):
    root = Path(root).resolve()
    sources = {"kvrepack." + ".".join(p.relative_to(root / "src/kvrepack").with_suffix("").parts): p
               for p in (root / "src/kvrepack").rglob("*.py")}
    sources = {name.removesuffix(".__init__"): path for name, path in sources.items()}
    for adapter in ("qwen2_5_vl", "llava_hf"):
        sources["lmms_eval.models." + adapter] = root / f"vendor/lmms-eval/lmms_eval/models/{adapter}.py"
    trees = {name: ast.parse(path.read_text()) for name, path in sources.items()}
    functions, classes, imports, module_exports = {}, {}, {}, {}

    def index(module, body, prefix="", owner=None):
        for node in body:
            if isinstance(node, ast.ClassDef):
                key = module + "." + prefix + node.name
                classes[key] = node
                index(module, node.body, prefix + node.name + ".", key)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = module + "." + prefix + node.name
                functions[key] = (module, node, owner)
                index(module, node.body, prefix + node.name + ".", owner)

    for module, tree in trees.items():
        index(module, tree.body)
        imports[module] = import_bindings(tree, module, sources[module].name == "__init__.py")
        exports = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exports.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                exports.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                exports.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
        module_exports[module] = exports

    errors, resolved, unresolved = [], [], []
    import_count = 0
    for module, tree in trees.items():
        for local, target in imports[module].items():
            if target.startswith("kvrepack."):
                if target in sources:
                    import_count += 1
                    continue
                parent, _, symbol = target.rpartition(".")
                if parent in module_exports and symbol in module_exports[parent]:
                    import_count += 1
                else:
                    errors.append({"kind": "unresolved_local_import", "module": module, "name": target})

    def resolve(callee, caller, module, owner):
        def descriptor_bound(target, via_instance=False):
            decorators = {dotted(d) for d in functions[target][1].decorator_list}
            if "staticmethod" in decorators:
                return False
            return "classmethod" in decorators or via_instance

        name = dotted(callee)
        if not name:
            return None, False, "dynamic_expression"
        if owner and name.startswith(("self.", "cls.")):
            candidate = owner + "." + name.partition(".")[2]
            return (candidate, descriptor_bound(candidate, True), "self_method") if candidate in functions else (None, False, "receiver_type_or_inherited_method")
        first, dot, rest = name.partition(".")
        candidate = imports[module].get(first, module + "." + first) + (dot + rest)
        # A function nested in this lexical scope takes precedence over globals.
        for prefix in (caller, caller.rpartition(".")[0]):
            nested = prefix + "." + name
            if nested in functions:
                return nested, descriptor_bound(nested), "lexical_function"
        if candidate in functions:
            return candidate, descriptor_bound(candidate), "direct_or_imported_function"
        if candidate in classes and candidate + ".__init__" in functions:
            return candidate + ".__init__", True, "local_constructor"
        return None, False, "external_or_dynamic_receiver"

    for caller, (module, function, owner) in functions.items():
        for node in direct_nodes(function.body):
            if not isinstance(node, ast.Call):
                continue
            target, bound, kind = resolve(node.func, caller, module, owner)
            location = {"source_path": str(sources[module].relative_to(root)), "line": node.lineno}
            if target:
                issues, status = binding_errors(functions[target][1], node, bound=bound)
                edge = {"caller": caller, "callee": target, **location,
                        "resolution": kind, "argument_binding": status}
                resolved.append(edge)
                errors.extend({"kind": "local_call_signature", **edge, "error": issue} for issue in issues)
            else:
                unresolved.append({"caller": caller, "expression": ast.unparse(node.func), **location, "reason": kind})

    extension = root / "csrc/csrc/python_extension.cu"
    exports = sorted(set(re.findall(r'm\.def\("([A-Za-z_0-9]+)"', extension.read_text())))
    kernel_calls = []
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not (name := dotted(node.func)):
                continue
            first, dot, rest = name.partition(".")
            target = imports[module].get(first, first) + dot + rest
            if target.startswith("kvrepack_kernel."):
                symbol = target.split(".", 1)[1]
                kernel_calls.append({"source_path": str(sources[module].relative_to(root)), "line": node.lineno, "symbol": symbol})
                if symbol not in exports:
                    errors.append({"kind": "missing_kernel_export", **kernel_calls[-1]})

    adjacency = {}
    for edge in resolved:
        adjacency.setdefault(edge["caller"], set()).add(edge["callee"])
    method_graphs = []
    if dispatch:
        for combo in dispatch["combinations"]:
            seeds = {p["target"] for p in combo["patch_targets"] if p["target"] in functions}
            model = combo["adapter_class"]
            seeds.update(model + "." + method for method in ("__init__", "forward", "prepare_inputs_for_generation")
                         if model + "." + method in functions)
            pending, reached = list(seeds), set()
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                pending.extend(adjacency.get(current, set()) - reached)
            method_graphs.append({"model": combo["model"], "public_method": combo["public_method"],
                                  "seeds": sorted(seeds), "reachable_local_functions": sorted(reached),
                                  "boundary": "Direct statically resolved calls from dynamically selected patch/model functions; framework callback edges are not inferred."})

    return {
        "schema_version": 1, "passed": not errors,
        "evidence_boundary": "AST/import/call-signature/kernel-export checks. Dynamic receiver, inheritance, PyTorch __call__, tensor shapes and CUDA execution are not proven by this graph.",
        "summary": {"python_files": len(sources), "function_definitions": len(functions),
                    "verified_unique_local_import_bindings": import_count, "resolved_local_call_sites": len(resolved),
                    "argument_binding_checked": sum(e["argument_binding"] == "checked" for e in resolved),
                    "external_or_dynamic_call_sites": len(unresolved), "kernel_call_sites": len(kernel_calls),
                    "kernel_exports": len(exports), "errors": len(errors)},
        "files_sha256": {str(path.relative_to(root)): sha(path) for path in sources.values()},
        "functions": {key: {"source_path": str(sources[module].relative_to(root)), "line": node.lineno,
                             "end_line": node.end_lineno, "signature": ast.unparse(node.args)}
                      for key, (module, node, owner) in functions.items()},
        "resolved_local_calls": resolved, "external_or_dynamic_calls": unresolved,
        "dynamic_call_reasons": dict(Counter(e["reason"] for e in unresolved)),
        "kernel": {"binding_source": str(extension.relative_to(root)), "binding_sha256": sha(extension),
                   "exports": exports, "call_sites": kernel_calls},
        "method_graphs": method_graphs, "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(dispatch=json.loads(args.dispatch_report.read_text()) if args.dispatch_report else None)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps({"passed": report["passed"], **report["summary"], "report": str(args.output)}, indent=2))
    else:
        print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

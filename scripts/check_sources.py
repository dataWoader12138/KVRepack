#!/usr/bin/env python3
"""Read-only provenance, syntax and import checks. Uses no ML dependencies."""

from __future__ import annotations

import argparse
import copy
import ast
import hashlib
import json
import re
from pathlib import Path
import warnings

# Preserve upstream source text while keeping this check's output readable.
warnings.filterwarnings("ignore", category=SyntaxWarning)


ROOT = Path(__file__).resolve().parents[1]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ast_digest(node: ast.AST) -> str:
    return digest(ast.dump(node, include_attributes=False).encode())


class NormalizeSource(ast.NodeTransformer):
    def visit_Import(self, node):
        for alias in node.names:
            alias.name = self.module(alias.name)
        return node

    def visit_ImportFrom(self, node):
        if node.module:
            node.module = self.module(node.module)
        return node

    def visit_Name(self, node):
        if node.id == "mmko_kernel":
            node.id = "kvrepack_kernel"
        return node

    @staticmethod
    def module(name):
        if name == "hybridkv.asymkv_paper.generation":
            return "kvrepack._asymkv_helpers"
        if name == "mmko_kernel":
            return "kvrepack_kernel"
        return "kvrepack" + name[len("hybridkv"):] if name == "hybridkv" or name.startswith("hybridkv.") else name


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def branches(tree, function_name="replace_qwen"):
    result = {}
    for n in ast.walk(function(tree, function_name)):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare):
            c = n.test
            if isinstance(c.left, ast.Name) and c.left.id == "method" and len(c.ops) == 1 and isinstance(c.ops[0], ast.Eq) and isinstance(c.comparators[0], ast.Constant):
                result[c.comparators[0].value] = ast.Module(body=n.body, type_ignores=[])
    return result


def rename_brand(text):
    """Replay the ordered naming-only transformation; preserve RandomDecoupledKV."""
    text = text.replace("hybrid_decouplekv", "hybridkv_kvrepack")
    text = text.replace("decouplekv", "kvrepack")
    text = text.replace("DecoupleKV", "KVRepack")
    text = text.replace("DECOUPLEKV", "KVREPACK")
    text = re.sub(r"(?<![A-Za-z0-9])decouple(?![A-Za-z0-9])", "kvrepack", text)
    text = text.replace("DECOUPLE", "KVREPACK")
    return re.sub(r"Decouple(?!d)", "KVRepack", text)


def verify_branding_record(target_text, record):
    """Reverse recorded character spans, then prove exact forward replay and old hash."""
    previous = target_text
    edits = record["reversible_edits"]
    assert edits == sorted(edits, key=lambda e: e["target_start"])
    last_end = 0
    for edit in edits:
        start, end = edit["target_start"], edit["target_end"]
        assert last_end <= start <= end <= len(target_text)
        assert target_text[start:end] == edit["after"]
        last_end = end
    for edit in reversed(edits):
        start, end = edit["target_start"], edit["target_end"]
        previous = previous[:start] + edit["before"] + previous[end:]
    assert digest(previous.encode()) == record["previous_target_sha256"], "Pre-branding target changed"
    assert rename_brand(previous) == target_text, "Branding introduced an undeclared edit"
    return ast.parse(previous)


def normalized_source(text, *, branded=False):
    return NormalizeSource().visit(ast.parse(rename_brand(text) if branded else text))


def selected_wrapper(tree, removed_test):
    result = copy.deepcopy(function(tree, "prepare_inputs_for_generation_qwen"))
    test_ast = ast_digest(ast.parse(removed_test, mode="eval").body)
    removed = [n for n in result.body if isinstance(n, ast.If) and ast_digest(n.test) == test_ast]
    assert len(removed) == 1, "Recorded wrapper guard is missing or ambiguous"
    result.body = [n for n in result.body if n not in removed]
    return result


def normalized_llava_function(tree):
    result = copy.deepcopy(function(tree, "replace_llava_onevision"))
    assert len(result.body) >= 3 and all(isinstance(n, ast.ImportFrom) for n in result.body[:2])
    result.body = result.body[2:]
    tail = result.body[0]
    while len(tail.orelse) == 1 and isinstance(tail.orelse[0], ast.If):
        tail = tail.orelse[0]
    assert len(tail.orelse) == 1 and isinstance(tail.orelse[0], ast.Raise)
    tail.orelse = []
    return result


def check_core(source_root):
    manifest = json.loads((ROOT / "provenance/core_sources.json").read_text())
    count = 0
    for e in manifest["entries"]:
        target = ROOT / e["target_path"]
        assert target.is_file(), f"Missing {e['target_path']}"
        assert digest(target.read_bytes()) == e["target_sha256"], f"Changed target: {target}"
        target_tree = ast.parse(target.read_text()) if target.suffix == ".py" else None
        previous_tree = target_tree
        if "branding" in e:
            previous_tree = verify_branding_record(target.read_text(), e["branding"])
        source_tree = renamed_tree = None
        if source_root:
            source = source_root / e["source_path"]
            assert digest(source.read_bytes()) == e["source_sha256"], f"Changed reference: {source}"
            if source.suffix == ".py":
                source_tree = normalized_source(source.read_text())
                renamed_tree = normalized_source(source.read_text(), branded=True)
        if e["kind"] == "byte_copy":
            assert e["source_sha256"] == e["target_sha256"]
        elif e["kind"] == "mechanical_copy":
            if source_tree is not None:
                for tree in (source_tree, renamed_tree):
                    if "relocate_head_score_resources" in e["transformations"]:
                        function(tree, "load_head_score").body[:2] = ast.parse('package_root = os.path.dirname(os.path.abspath(__file__))\nhead_score_dir = os.path.join(package_root, "data", "head_score")').body
                assert ast_digest(source_tree) == e["source_ast_sha256"], f"Original source AST drift: {target}"
                assert ast_digest(source_tree) == ast_digest(previous_tree), f"Pre-branding algorithm drift: {target}"
                assert ast_digest(renamed_tree) == e["renamed_source_ast_sha256"]
                assert ast_digest(renamed_tree) == ast_digest(target_tree), f"Unexpected algorithm change: {target}"
            assert ast_digest(target_tree) == e["renamed_source_ast_sha256"], f"Algorithm drift: {target}"
        elif e["kind"] == "selected_functions":
            for f in e["function_verification"]:
                assert ast_digest(function(previous_tree, f["name"])) == f["source_ast_sha256"]
                assert ast_digest(function(target_tree, rename_brand(f["name"]))) == f["renamed_source_ast_sha256"], f"Function drift: {f['name']}"
                if source_tree is not None:
                    assert ast_digest(function(source_tree, f["name"])) == f["source_ast_sha256"]
                    assert ast_digest(function(renamed_tree, rename_brand(f["name"]))) == f["renamed_source_ast_sha256"]
        elif e["kind"] == "selected_dispatcher":
            selected = branches(target_tree)
            previous = branches(previous_tree)
            assert set(selected) == set(e["retained_branches"])
            assert not set(selected).intersection(rename_brand(m) for m in e["removed_branches"])
            for b in e["branch_verification"]:
                method = b["method"]
                assert b["target_body_ast_sha256"] == b["renamed_source_body_ast_sha256"]
                assert ast_digest(previous[method]) == b["source_body_ast_sha256"]
                assert ast_digest(selected[b["target_method"]]) == b["renamed_source_body_ast_sha256"], f"Dispatch drift: {method}"
                if source_tree is not None:
                    assert ast_digest(branches(source_tree)[method]) == b["source_body_ast_sha256"]
                    assert ast_digest(branches(renamed_tree)[b["target_method"]]) == b["renamed_source_body_ast_sha256"]
            w = e["wrapper_verification"]
            wrapper = function(target_tree, "prepare_inputs_for_generation_qwen")
            assert ast_digest(function(previous_tree, "prepare_inputs_for_generation_qwen")) == w["expected_selected_wrapper_ast_sha256"]
            assert ast_digest(wrapper) == w["renamed_selected_wrapper_ast_sha256"]
            if source_tree is not None:
                assert ast_digest(selected_wrapper(source_tree, w["removed_unreachable_if"])) == w["expected_selected_wrapper_ast_sha256"]
                assert ast_digest(selected_wrapper(renamed_tree, rename_brand(w["removed_unreachable_if"]))) == ast_digest(wrapper), "Generation wrapper drift"
            if "llava_source" in e:
                selected_llava = branches(target_tree, "replace_llava_onevision")
                assert set(selected_llava) == set(e["llava_retained_branches"])
                original_llava = None
                if source_root:
                    llava_source = source_root / e["llava_source"]["source_path"]
                    assert digest(llava_source.read_bytes()) == e["llava_source"]["source_sha256"]
                    original_llava = normalized_source(llava_source.read_text(), branded=True)
                for b in e["llava_branch_verification"]:
                    assert ast_digest(selected_llava[b["method"]]) == b["source_body_ast_sha256"], f"LLaVA dispatch drift: {b['method']}"
                    if original_llava is not None:
                        assert ast_digest(branches(original_llava, "replace_llava_onevision")[b["method"]]) == b["source_body_ast_sha256"]
                llava = e["llava_function_verification"]
                assert ast_digest(function(target_tree, "replace_llava_onevision")) == llava["target_function_ast_sha256"]
                assert ast_digest(normalized_llava_function(target_tree)) == llava["source_function_ast_sha256"]
                if original_llava is not None:
                    assert ast_digest(function(original_llava, "replace_llava_onevision")) == llava["source_function_ast_sha256"]
                assert ast_digest(function(previous_tree, "replace_qwen")) == llava["qwen_function_before_ast_sha256"]
                assert ast_digest(function(target_tree, "replace_qwen")) == llava["renamed_qwen_function_ast_sha256"]
        count += 1
    frontend = manifest["frontend_integration"]
    for relative, expected in frontend["files_sha256"].items():
        assert digest((ROOT / relative).read_bytes()) == expected, f"Frontend changed: {relative}"
    assert digest((ROOT / frontend["registry_snapshot_path"]).read_bytes()) == frontend["registry_snapshot_sha256"]
    return count


def check_imports_and_syntax():
    package = ROOT / "src/kvrepack"
    assert (package / "__init__.py").is_file(), "Namespace-package fallback must be disabled"
    modules = set()
    for p in package.rglob("*.py"):
        name = ".".join(p.relative_to(package.parent).with_suffix("").parts)
        modules.add(name.removesuffix(".__init__"))
    count = 0
    trees = [(p, ast.parse(p.read_text())) for p in package.rglob("*.py")]
    for name in ("qwen2_5_vl", "llava_hf"):
        adapter = ROOT / f"vendor/lmms-eval/lmms_eval/models/{name}.py"
        trees.append((adapter, ast.parse(adapter.read_text())))
    for p, tree in trees:
        count += 1
        current = ".".join(p.relative_to(package.parent).with_suffix("").parts) if p.is_relative_to(package) else "lmms_eval.models.qwen2_5_vl"
        parent = current.removesuffix(".__init__") if p.name == "__init__.py" else current.rsplit(".", 1)[0]
        for n in ast.walk(tree):
            targets = []
            if isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    parts = parent.split(".")
                    prefix = ".".join(parts[:len(parts) - n.level + 1])
                    targets = [prefix + ("." + n.module if n.module else "")]
                    if not n.module:
                        targets = [prefix + "." + a.name for a in n.names]
                else:
                    targets = [n.module or ""]
            for name in targets:
                assert name.split(".")[0] not in {"hybridkv", "mmko_kernel", "hybridkv_kernel"}, f"Legacy import in {p}:{n.lineno}: {name}"
                if name == "kvrepack" or name.startswith("kvrepack."):
                    assert name in modules, f"Unresolved local module in {p}:{n.lineno}: {name}"
    vendor_count = 0
    for p in (ROOT / "vendor/lmms-eval/lmms_eval").rglob("*.py"):
        ast.parse(p.read_text())
        vendor_count += 1
    return count, vendor_count


def check_vendor(source_root):
    manifest = json.loads((ROOT / "provenance/vendor_sources.json").read_text())
    components = manifest["components"]
    specs = (
        (components["lmms_eval"], "vendored_files_sha256"),
        (components["cuda_kernels"], "unmodified_source_files_sha256"),
        (components["cuda_kernels"], "integration_build_files_sha256"),
        (components["cutlass_headers"], "files_sha256"),
    )
    count = 0
    for component, field in specs:
        for relative, expected in component[field].items():
            p = ROOT / component["destination"] / relative
            assert p.is_file() and digest(p.read_bytes()) == expected, f"Vendor/build input changed: {p}"
            count += 1
    for relative, record in components["lmms_eval"].get("branding", {}).items():
        target_text = (ROOT / components["lmms_eval"]["destination"] / relative).read_text()
        previous = verify_branding_record(target_text, record)
        assert ast_digest(previous) == record["previous_ast_sha256"]
        assert ast_digest(ast.parse(target_text)) == record["renamed_ast_sha256"]
    if source_root:
        lmms = components["lmms_eval"]
        for relative, expected in lmms["source_selected_files_sha256"].items():
            p = source_root / lmms["source"] / relative
            assert digest(p.read_bytes()) == expected, f"Changed vendor reference: {p}"
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Optional original workspace, for source-to-target AST comparison")
    args = parser.parse_args()
    core = check_core(args.source_root.resolve() if args.source_root else None)
    code, vendor = check_imports_and_syntax()
    vendor_hashes = check_vendor(args.source_root.resolve() if args.source_root else None)
    print(json.dumps({"status": "passed", "core_files": core, "local_modules_and_adapter": code, "vendor_python_files": vendor, "vendor_build_files": vendor_hashes, "original_source_comparison": args.source_root is not None}, indent=2))


if __name__ == "__main__":
    main()

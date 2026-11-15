"""Scoped replay of the exact archived Qwen LOOK-M implementation.

The historical run used both history10%+recent and the pre-correction merge
implementation. Recover its two functions from the reversible provenance
record rather than reimplementing either rule or changing a core file.
Source inspection is stdlib-only; model imports happen only on context entry.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import textwrap

TARGET = "src/kvrepack/baseline_utils.py"
FUNCTIONS = ("merge_kv", "LookMCluster.update_kv")
POLICY = "archived_lookm_history10percent_plus_recent32_v1"


def _sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _ast_sha(node):
    return _sha(ast.dump(node, include_attributes=False))


def _definition(tree, name):
    nodes = tree.body
    for part in name.split("."):
        found = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name == part]
        if len(found) != 1:
            raise ValueError("Missing or ambiguous archived LOOK-M definition: "+name)
        node = found[0]
        nodes = node.body
    return node


def _recover(repo_root=None):
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[3]
    manifest_path = root/"provenance/core_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [item for item in manifest["entries"] if item.get("target_path") == TARGET]
    if len(entries) != 1:
        raise ValueError("Require one provenance entry for the original LOOK-M module")
    entry = entries[0]
    current = (root/TARGET).read_text(encoding="utf-8")
    if _sha(current) != entry["target_sha256"]:
        raise ValueError("Current LOOK-M source differs from its recorded SHA")
    record = entry["behavior_patch"]
    if _sha((root/record["patch_path"]).read_bytes()) != record["patch_sha256"]:
        raise ValueError("LOOK-M reversible patch file differs from its recorded SHA")
    changes = record["function_changes"]
    if [item["name"] for item in changes] != list(FUNCTIONS):
        raise ValueError("Only the two recorded LOOK-M functions may be replayed")
    edits = record["reversible_edits"]
    if edits != sorted(edits, key=lambda item: item["target_start"]):
        raise ValueError("LOOK-M reversible edits are not ordered")
    last_end = 0
    for edit in edits:
        start, end = edit["target_start"], edit["target_end"]
        if (not 0 <= last_end <= start <= end <= len(current)
                or current[start:end] != edit["after"]):
            raise ValueError("LOOK-M reversible edit does not match current source")
        last_end = end
    historical = current
    for edit in reversed(edits):
        historical = historical[:edit["target_start"]]+edit["before"]+historical[edit["target_end"]:]
    if _sha(historical) != record["previous_target_sha256"]:
        raise ValueError("Recovered LOOK-M source differs from the archived module SHA")
    tree, current_tree = ast.parse(historical), ast.parse(current)
    if _ast_sha(tree) != record["previous_target_ast_sha256"]:
        raise ValueError("Recovered LOOK-M module AST differs from the archived AST")
    for item in changes:
        if (_ast_sha(_definition(tree, item["name"])) != item["before_ast_sha256"]
                or _ast_sha(_definition(current_tree, item["name"])) != item["after_ast_sha256"]):
            raise ValueError("Recorded LOOK-M function AST identity differs")
    identity = dict(policy=POLICY, target_path=TARGET,
        current_module_sha256=entry["target_sha256"],
        archived_module_sha256=record["previous_target_sha256"],
        archived_module_ast_sha256=record["previous_target_ast_sha256"],
        manifest_sha256=_sha(manifest_path.read_bytes()),
        reversible_patch_path=record["patch_path"], reversible_patch_sha256=record["patch_sha256"],
        transformation="Replay the two exact pre-correction functions; no core source mutation",
        functions={name: dict(archived_source=ast.get_source_segment(historical, _definition(tree, name)),
            archived_ast_sha256=_ast_sha(_definition(tree, name)),
            current_ast_sha256=_ast_sha(_definition(current_tree, name))) for name in FUNCTIONS},
        historical_behavior="Original history ratio plus recent; original cross-head complement and native-dtype cosine behavior retained")
    return root, tree, identity


def source_identity(repo_root=None):
    """JSON-serializable source evidence, without importing Torch or a model."""
    return _recover(repo_root)[2]


class LegacyLookMContext:
    """Temporarily use the exact LOOK-M functions behind the legacy run.

    Enter outside the measured response loop, after installing the canonical
    LOOK-M model attention. Any per-sample observer belongs inside this scope.
    This intentionally preserves archived behavior rather than mixing an old
    budget with the later corrected merge implementation.
    """
    def __init__(self, *, module=None, repo_root=None):
        self.module, self.repo_root = module, repo_root
        self._patches = []
        self._entered = False
        self.identity = self.transform_spec = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Legacy LOOK-M context cannot be entered twice")
        root, tree, identity = _recover(self.repo_root)
        module = self.module or importlib.import_module("kvrepack.baseline_utils")
        if self.module is None and Path(module.__file__).resolve() != root/TARGET:
            raise ValueError("Imported LOOK-M module is from a different checkout")
        self.module = module
        owners = ((module, "merge_kv"), (module.LookMCluster, "update_kv"))
        # A nested/foreign patch must not be silently replaced or mistaken for
        # the recorded corrected module. Check both before mutating either.
        for name, (owner, attr) in zip(FUNCTIONS, owners):
            original = getattr(owner, attr)
            try:
                node = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
            except (OSError, TypeError, SyntaxError) as error:
                raise ValueError("Cannot identify currently installed LOOK-M function") from error
            if _ast_sha(node) != identity["functions"][name]["current_ast_sha256"]:
                raise ValueError("LOOK-M function already patched or differs from its recorded AST")
        compiled = []
        for name in FUNCTIONS:
            node = copy.deepcopy(_definition(tree, name))
            namespace = {}
            # The updater must resolve merge_kv through the real module global
            # dictionary, exactly like the original class implementation.
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                str(root/TARGET)+"<archived_lookm>", "exec"), module.__dict__, namespace)
            compiled.append(namespace[node.name])
        try:
            for (owner, attr), function in zip(owners, compiled):
                self._patches.append((owner, attr, getattr(owner, attr)))
                setattr(owner, attr, function)
            self.identity = self.transform_spec = identity
            self._entered = True
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches.clear()
        self._entered = False
        return False

"""Build inspectable launch plans and execute only this checkout's implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

from .registry import MODEL_TYPES, TASKS, resolve_experiment


# Do not inherit algorithm flags or Python import paths. Credentials are passed only
# to the child, never copied into a plan, resolved_config, provenance, or a log.
_INFRA_KEYS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "TERM",
    "TMPDIR", "TMP", "TEMP", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
    "CPLUS_INCLUDE_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_CACHE_PATH", "JAVA_HOME",
    "NLTK_DATA", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
    "all_proxy", "no_proxy", "API_TYPE", "TRANSFORMERS_OFFLINE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
})
_INFRA_PREFIXES = ("HF_", "HUGGINGFACE_", "XDG_", "OPENAI_", "AZURE_", "LC_")
_CACHE_KEYS = ("HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE")


def _infrastructure(key: str) -> bool:
    return key in _INFRA_KEYS or key.startswith(_INFRA_PREFIXES)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _infer_model_type(pretrained: str, model_type: str | None, model: str) -> str:
    if model == "llava_ov":
        if model_type is not None and model_type != "llava-ov-7b":
            raise ValueError("LLaVA OneVision requires --model-type llava-ov-7b")
        if "3b" in pretrained.lower() or ("qwen" in pretrained.lower() and "llava" not in pretrained.lower()):
            raise ValueError("--model llava_ov requires LLaVA OneVision 7B weights")
        return "llava-ov-7b"
    size = re.search(r"(?:^|[^0-9])(3|7)b(?:[^a-z0-9]|$)", pretrained.lower())
    inferred = f"qwen2_5_vl-{size.group(1)}b" if size else None
    if model_type is not None and model_type not in MODEL_TYPES[:2]:
        raise ValueError(f"Qwen model_type must be one of {', '.join(MODEL_TYPES[:2])}")
    if inferred and model_type and inferred != model_type:
        raise ValueError("--model-type conflicts with the model size in --pretrained")
    if not model_type and not inferred:
        raise ValueError("Cannot infer model size from --pretrained; set --model-type explicitly")
    if any(family in pretrained.lower() for family in ("llava", "internvl", "qwen3")):
        raise ValueError("This registry supports Qwen2.5-VL 3B/7B only")
    return model_type or inferred  # type: ignore[return-value]


def _device(value: str) -> str:
    value = str(value).removeprefix("cuda:")
    if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+", value):
        raise ValueError("--device must select one CUDA device index or GPU/MIG UUID")
    return value


@dataclass(frozen=True)
class ExperimentPlan:
    repo_root: Path
    output: Path
    command: tuple[str, ...]
    env: Mapping[str, str]
    resolved_config: dict
    provenance: dict

    def to_dict(self) -> dict:
        return {
            "resolved_config": self.resolved_config, "provenance": self.provenance,
            "cwd": str(self.repo_root), "env": dict(self.env),
            "command": list(self.command), "shell_command": shlex.join(self.command),
        }


def build_plan(method: str, *, model: str = "qwen2_5_vl", task: str | None = None, pretrained: str | None = None,
               model_type: str | None = None, frames: int | None = None,
               limit: int | None = None, budget_ratio: float | None = None,
               device: str = "0", output: str | Path | None = None,
               overrides: Sequence[str] = (), environ: Mapping[str, str] | None = None,
               repo_root: str | Path | None = None) -> ExperimentPlan:
    """Pure planning: no imports of inference packages, network access, or file writes."""
    spec, algorithm, explicit = resolve_experiment(method, overrides, budget_ratio=budget_ratio, model=model)
    root = Path(repo_root).resolve() if repo_root is not None else _root()
    ambient = os.environ if environ is None else environ
    task = task or spec.default_tasks[0]
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}; supported tasks: {', '.join(TASKS)}")
    pretrained = pretrained or spec.default_pretrained
    if not pretrained or any(c in pretrained for c in ",\r\n\x00"):
        raise ValueError("--pretrained must be a model ID or local path without commas or newlines")
    model_type = _infer_model_type(pretrained, model_type, model)
    frames = _positive_integer("frames", frames if frames is not None else 64)
    limit = _positive_integer("limit", limit if limit is not None else spec.default_limit)
    gpu = _device(device)
    out = Path(output).expanduser().resolve() if output is not None else root / "outputs" / method / task / model_type
    if out == root or any(out.is_relative_to(root / part) for part in ("src", "vendor", "csrc", "provenance")):
        raise ValueError("--output must be outside source, vendor, csrc, and provenance directories")
    src = root / "src"
    vendor = root / "vendor" / "lmms-eval"
    env = {
        **algorithm, **spec.runtime_defaults,
        "METHOD": spec.internal_method, "MODEL_TYPE": model_type,
        "CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (src, vendor, root / "csrc"))),
    }
    if spec.method_name:
        env["METHOD_NAME"] = spec.method_name
    hf_home = Path(ambient.get("HF_HOME") or str(root / ".cache" / "huggingface")).expanduser().resolve()
    env["HF_HOME"] = str(hf_home)
    env["HF_DATASETS_CACHE"] = ambient.get("HF_DATASETS_CACHE") or str(hf_home / "datasets")
    env["HUGGINGFACE_HUB_CACHE"] = ambient.get("HUGGINGFACE_HUB_CACHE") or ambient.get("HF_HUB_CACHE") or str(hf_home / "hub")
    env["HF_HUB_CACHE"] = env["HUGGINGFACE_HUB_CACHE"]
    if method == "asymkv":
        env["ASYMKV_AUDIT_PATH"] = str(out / "asymkv_audit.jsonl")

    adapter = "llava_hf" if model == "llava_ov" else "qwen2_5_vl"
    if model == "llava_ov":
        model_args = f"pretrained={pretrained},max_frames_num={frames},device_map=auto,dtype=bfloat16,attn_implementation=flash_attention_2"
    else:
        model_args = f"pretrained={pretrained},max_num_frames={frames},attn_implementation=flash_attention_2,method={spec.internal_method}"
    # The selected FullKV reference passes no compression parameters. Other values
    # remain environment-driven exactly as in the selected experiment scripts.
    if method == "asymkv":
        model_args += f",budget={algorithm['BUDGET']},ratio={algorithm['RATIO']},mask_ratio={algorithm['MASK_RATIO']}"
    command = (
        sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1",
        "--main_process_port", "29500", "-m", "lmms_eval", "--model", adapter,
        "--model_args", model_args, "--tasks", task, "--batch_size", "1", "--log_samples",
        "--log_samples_suffix", method, "--output_path", str(out / "eval_outputs"),
        "--gen_kwargs", "temperature=0", "--limit", str(limit),
    )
    manifest_path = root / "provenance" / "core_sources.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
    reference_path = root / "provenance" / "reference_entrypoints.json"
    reference = None
    if reference_path.is_file():
        references = json.loads(reference_path.read_text(encoding="utf-8"))
        reference = next((entry for entry in references.get("entries", [])
                          if entry.get("source_path") == spec.source_script), None)
    config_sources = {}
    for relative in ("src/kvrepack/experiments/registry.py", "src/kvrepack/experiments/runner.py", "src/kvrepack/run.py"):
        path = root / relative
        if path.is_file():
            config_sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    vendor_manifest_path = root / "provenance" / "vendor_sources.json"
    config = {
        "schema_version": 1, "public_method": method, "internal_method": spec.internal_method,
        "model": model, "lmms_adapter": adapter,
        "baseline": spec.baseline, "task": task, "pretrained": pretrained,
        "model_type": model_type, "frames": frames, "limit": limit,
        "budget_ratio": float(algorithm["BUDGET_RATIO"]) if "BUDGET_RATIO" in algorithm else None,
        "device": gpu, "output": str(out), "batch_size": 1,
        "attention": "flash_attention_2", "generation": {"temperature": 0},
        "algorithm_env": algorithm, "runtime_env": {key: env[key] for key in (*spec.runtime_defaults, *_CACHE_KEYS, "CUDA_VISIBLE_DEVICES")},
        "explicit_env_overrides": explicit,
    }
    provenance = {
        "schema_version": 1, "selected_method": spec.describe(),
        "core_manifest": "provenance/core_sources.json", "core_manifest_sha256": manifest_hash,
        "reference_entrypoint_snapshot": reference,
        "configuration_sources_sha256": config_sources,
        "vendor_manifest_sha256": hashlib.sha256(vendor_manifest_path.read_bytes()).hexdigest() if vendor_manifest_path.is_file() else None,
        "source_policy": "Only targets inside this checkout are imported; legacy source trees are never consulted.",
        "environment_policy": "Only infrastructure variables are inherited; all algorithm flags are rebuilt explicitly. Secret values are never serialized.",
        "discarded_environment_names": sorted(key for key in ambient if not _infrastructure(key)),
        "inherited_infrastructure_names": sorted(key for key in ambient if _infrastructure(key)),
        "runtime_normalization": {
            "python": "current sys.executable", "gpu": gpu, "processes": 1,
            "cache_default": "<checkout>/.cache/huggingface (HF_HOME may override)",
            "conda": "No activation, absolute conda path, forced mirror, or forced offline mode.",
            "task_selection": "One requested task; default is the first task in the selected source script.",
            "model_type": "Qwen size is inferred; LLaVA OneVision uses llava-ov-7b. The adapter is recorded in resolved_config.",
            "other_runtime_env": "OMP_NUM_THREADS=4, TOKENIZERS_PARALLELISM=false, explicit CUDA_LAUNCH_BLOCKING, fixed launch port 29500.",
        },
        "python_executable": sys.executable,
    }
    return ExperimentPlan(root, out, command, env, config, provenance)


def child_environment(plan: ExperimentPlan, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the private child environment; callers must not serialize its values."""
    ambient = os.environ if environ is None else environ
    return {**{key: value for key, value in ambient.items() if _infrastructure(key)}, **plan.env}


def verify_manifest(root: Path) -> dict:
    """Validate copied targets only: a released checkout needs no original source tree."""
    path = root / "provenance" / "core_sources.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Missing or invalid provenance/core_sources.json") from exc
    entries = manifest.get("entries", [])
    # Schema 2 adds branding provenance; copied-target fields retain schema 1's
    # path/sha256 contract and must pass the same checks below.
    if manifest.get("schema_version") not in (1, 2) or not entries:
        raise ValueError("Unsupported or empty core source manifest")
    checked = []
    for entry in entries:
        target = (root / entry["target_path"]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"Manifest target is missing or outside this checkout: {entry['target_path']}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != entry["target_sha256"]:
            raise ValueError(f"Source hash mismatch: {entry['target_path']}")
        checked.append({"target_path": entry["target_path"], "sha256": actual})
    return {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "verified_targets": checked}


def verify_vendor_sources(root: Path, adapter: str = "qwen2_5_vl") -> dict:
    """Check the frontend and supported model adapters against their shipped manifest."""
    path = root / "provenance" / "vendor_sources.json"
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        component = manifest["components"]["lmms_eval"]
        hashes = component["vendored_files_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("Missing or invalid provenance/vendor_sources.json") from exc
    if manifest.get("schema_version") not in (1, 2) or component.get("destination") != "vendor/lmms-eval":
        raise ValueError("Unexpected evaluation frontend in vendor source manifest")
    checked = []
    critical = ["lmms_eval/__init__.py", "lmms_eval/__main__.py", "lmms_eval/evaluator.py",
                "lmms_eval/models/__init__.py", "lmms_eval/models/qwen2_5_vl.py"]
    if adapter == "llava_hf" or "lmms_eval/models/llava_hf.py" in hashes:
        critical.append("lmms_eval/models/llava_hf.py")
    for relative in critical:
        target = (root / "vendor/lmms-eval" / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"Missing local evaluation frontend: {relative}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != hashes.get(relative):
            raise ValueError(f"Evaluation frontend hash mismatch: {relative}")
        checked.append({"target_path": str(target.relative_to(root)), "sha256": actual})
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "verified_targets": checked}


def verify_import_sources(plan: ExperimentPlan, env: Mapping[str, str]) -> dict:
    """Inspect import specs in the actual child interpreter without importing torch."""
    expected = {"kvrepack": str(plan.repo_root / "src" / "kvrepack"),
                "lmms_eval": str(plan.repo_root / "vendor" / "lmms-eval" / "lmms_eval")}
    if plan.provenance["selected_method"]["kernel"]:
        expected["kvrepack_kernel"] = str(plan.repo_root / "csrc")
    script = """
import hashlib, importlib.util, importlib.metadata, json, pathlib, re, sys
from urllib.parse import unquote, urlparse
expected = json.loads(sys.argv[1])
resolved = {}
kernel_distribution = None
for name, allowed in expected.items():
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit(f'Missing {name}; install this checkout and build csrc when required')
    locations = ([spec.origin] if spec.origin and spec.origin != 'namespace' else list(spec.submodule_search_locations or []))
    root = pathlib.Path(allowed).resolve()
    local = locations and all(pathlib.Path(p).resolve().is_relative_to(root) for p in locations)
    if not local and name == 'kvrepack_kernel' and len(locations) == 1:
        # A local-source pip build installs its extension in site-packages.
        # Accept it only when the exact named distribution owns this origin and
        # its recorded build source is this checkout, never an older code tree.
        origin = pathlib.Path(locations[0]).resolve()
        # A setuptools source .egg-info can coexist with the installed dist-info
        # on PYTHONPATH. Only metadata that owns the resolved extension counts.
        candidates = [d for d in importlib.metadata.distributions()
                      if re.sub(r'[-_.]+', '-', d.metadata.get('Name', '')).lower() == 'kvrepack-kernels'
                      and any(pathlib.Path(d.locate_file(p)).resolve() == origin for p in (d.files or []))]
        if len(candidates) != 1:
            raise SystemExit('kvrepack_kernel requires exactly one owning kvrepack-kernels distribution')
        dist = candidates[0]
        owned = any(pathlib.Path(dist.locate_file(p)).resolve() == origin for p in (dist.files or []))
        try: direct = json.loads(dist.read_text('direct_url.json') or '{}')
        except ValueError: direct = {}
        source = urlparse(direct.get('url', ''))
        source_matches = (source.scheme == 'file' and source.netloc in ('', 'localhost')
                          and pathlib.Path(unquote(source.path)).resolve() == root)
        if not owned or not source_matches:
            raise SystemExit('kvrepack_kernel distribution does not own this module or was built from another checkout; rebuild ./csrc')
        kernel_distribution = {'name': dist.metadata['Name'], 'version': dist.version,
                               'build_source': direct['url'], 'origin': str(origin),
                               'extension_sha256': hashlib.sha256(origin.read_bytes()).hexdigest()}
        local = True
    if not local:
        raise SystemExit(f'{name} resolves outside this checkout: {locations}')
    resolved[name] = locations
versions = {}
for name in ('torch', 'transformers', 'accelerate', 'flash-attn', 'decord', 'datasets', 'qwen-vl-utils'):
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name] = None
print(json.dumps({'resolved_modules': resolved, 'dependency_versions': versions,
                  'kernel_distribution': kernel_distribution, 'python_version': sys.version}))
"""
    result = subprocess.run([sys.executable, "-c", script, json.dumps(expected)], cwd=plan.repo_root,
                            env=dict(env), capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "Import source verification failed")
    return json.loads(result.stdout)


def _git_commit(root: Path) -> str | None:
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=False)
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            return None
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def run_plan(plan: ExperimentPlan) -> int:
    """Check source provenance, save metadata, then run one evaluation with a run log."""
    manifest = verify_manifest(plan.repo_root)
    if plan.provenance["core_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Core manifest changed after this plan was built; rebuild the plan")
    vendor = verify_vendor_sources(plan.repo_root, plan.resolved_config["lmms_adapter"])
    if plan.provenance["vendor_manifest_sha256"] != vendor["manifest_sha256"]:
        raise ValueError("Vendor manifest changed after this plan was built; rebuild the plan")
    for relative, expected in plan.provenance["configuration_sources_sha256"].items():
        target = (plan.repo_root / relative).resolve()
        if not target.is_relative_to(plan.repo_root) or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Experiment configuration source changed after planning: {relative}")
    env = child_environment(plan)
    imports = verify_import_sources(plan, env)
    if plan.output.exists() and any(plan.output.iterdir()):
        raise ValueError(f"Output directory is not empty; select a fresh --output: {plan.output}")
    plan.output.mkdir(parents=True, exist_ok=True)
    provenance = {**plan.provenance, **imports, "source_verification": manifest,
                  "vendor_source_verification": vendor,
                  "git_commit": _git_commit(plan.repo_root), "command": list(plan.command)}
    for name, payload in (("resolved_config.json", plan.resolved_config), ("provenance.json", provenance)):
        (plan.output / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    state = {"started_at": datetime.now(timezone.utc).isoformat(), "returncode": None}
    state_path = plan.output / "run_state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    try:
        with (plan.output / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(plan.command, cwd=plan.repo_root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        state["returncode"] = result.returncode
        return result.returncode
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

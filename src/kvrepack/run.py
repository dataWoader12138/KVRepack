"""Command-line entry point: ``python -m kvrepack.run --list-methods``."""

from __future__ import annotations

import argparse
import json
import sys

from .experiments.registry import METHODS, METHODS_BY_MODEL, MODELS, MODEL_TYPES, TASKS
from .experiments.runner import build_plan, run_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one source-pinned Qwen2.5-VL or LLaVA OneVision KV cache experiment")
    parser.add_argument("--benchmark", choices=("video", "milebench"), default="video")
    parser.add_argument("--list-methods", action="store_true", help="Print methods, sources, defaults, and allowed --set keys as JSON")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument("--model", choices=MODELS, default="qwen2_5_vl")
    parser.add_argument("--task", help="Video task or case-sensitive MileBench dataset name")
    parser.add_argument("--data-root", help="External MileBench root containing TASK/TASK.json and TASK/images")
    parser.add_argument("--profile", choices=("image8", "main11"), help="MileBench image recipe and task suite (default main11)")
    parser.add_argument("--preflight", action="store_true", help="Read-only MileBench annotation and image integrity check; no model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-context-len", type=int)
    parser.add_argument("--min-pixels", type=int, help="Explicitly override both Qwen processor and message image minimum")
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--pretrained", help="Hugging Face model ID or local model path")
    parser.add_argument("--model-type", choices=MODEL_TYPES, help="Head score/model size selector; inferred from --pretrained by default")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget-ratio", type=float)
    parser.add_argument("--device", default="0", help="Single CUDA device index, cuda:N, or GPU/MIG UUID")
    parser.add_argument("--output", help="Fresh output directory; defaults to outputs/METHOD/TASK/MODEL_TYPE in this checkout")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides", help="Explicit per-method whitelisted algorithm setting")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved JSON and launch command without creating outputs/caches or launching evaluation")
    args = parser.parse_args(argv)
    if args.list_tasks:
        if args.benchmark == "milebench":
            from .experiments.milebench import load_profile
            tasks = load_profile(args.profile or "main11")["tasks"]
        else:
            tasks = TASKS
        print(json.dumps(list(tasks), indent=2))
        return 0
    if args.list_methods:
        if args.benchmark == "milebench":
            from .experiments.milebench import describe_methods
            print(json.dumps(describe_methods(args.profile or "main11"), indent=2, ensure_ascii=False))
            return 0
        listing = {name: {**spec.describe(), "model_variants": {
            model: methods[name].describe() for model, methods in METHODS_BY_MODEL.items() if name in methods
        }} for name, spec in METHODS.items()}
        print(json.dumps(listing, indent=2, ensure_ascii=False))
        return 0
    if args.preflight:
        if args.benchmark != "milebench" or not args.task or not args.data_root:
            parser.error("--preflight requires --benchmark milebench, --task, and --data-root")
        if args.dry_run:
            parser.error("--preflight reads data; use it separately from --dry-run")
        from .experiments.milebench import load_profile
        from .benchmarks.milebench.dataset import validate_data
        if args.task not in load_profile(args.profile or "main11")["tasks"]:
            parser.error("--task is not in the selected MileBench profile")
        try:
            report = validate_data(args.data_root, args.task, limit=args.limit, verify_images=True)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["valid"] else 2
        except (OSError, ValueError) as exc:
            print(f"kvrepack: {exc}", file=sys.stderr)
            return 2
    if not args.method:
        parser.error("--method is required unless --list-methods is used")
    try:
        plan = build_plan(args.method, model=args.model, task=args.task, pretrained=args.pretrained, model_type=args.model_type,
                          frames=args.frames, limit=args.limit, budget_ratio=args.budget_ratio,
                          device=args.device, output=args.output, overrides=args.overrides,
                          benchmark=args.benchmark, data_root=args.data_root, profile=args.profile,
                          seed=args.seed, max_new_tokens=args.max_new_tokens, max_context_len=args.max_context_len,
                          min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        if args.dry_run:
            print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
            return 0
        print(f"Running {args.method}; resolved configuration and log: {plan.output}", flush=True)
        return run_plan(plan)
    except (OSError, ValueError) as exc:
        print(f"kvrepack: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

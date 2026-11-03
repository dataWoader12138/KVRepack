"""Command-line entry point: ``python -m kvrepack.run --list-methods``."""

from __future__ import annotations

import argparse
import json
import sys

from .experiments.registry import METHODS, METHODS_BY_MODEL, MODELS, MODEL_TYPES, TASKS
from .experiments.runner import build_plan, run_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one source-pinned Qwen2.5-VL or LLaVA OneVision KV cache experiment")
    parser.add_argument("--list-methods", action="store_true", help="Print methods, sources, defaults, and allowed --set keys as JSON")
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument("--model", choices=MODELS, default="qwen2_5_vl")
    parser.add_argument("--task", choices=TASKS)
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
    if args.list_methods:
        listing = {name: {**spec.describe(), "model_variants": {
            model: methods[name].describe() for model, methods in METHODS_BY_MODEL.items() if name in methods
        }} for name, spec in METHODS.items()}
        print(json.dumps(listing, indent=2, ensure_ascii=False))
        return 0
    if not args.method:
        parser.error("--method is required unless --list-methods is used")
    try:
        plan = build_plan(args.method, model=args.model, task=args.task, pretrained=args.pretrained, model_type=args.model_type,
                          frames=args.frames, limit=args.limit, budget_ratio=args.budget_ratio,
                          device=args.device, output=args.output, overrides=args.overrides)
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

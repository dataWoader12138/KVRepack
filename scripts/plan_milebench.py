#!/usr/bin/env python3
"""Print a MileBench experiment matrix; never launch, download or write outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvrepack.experiments.milebench import IMAGE_METHODS, PROFILES, load_profile
from kvrepack.experiments.runner import build_plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen2_5_vl", "llava_ov"), default="qwen2_5_vl")
    parser.add_argument("--pretrained")
    parser.add_argument("--model-type")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="main11")
    parser.add_argument("--methods", nargs="+", choices=IMAGE_METHODS,
                        default=("hybridkv_kvrepack", "snapkv_kvrepack", "sparsemm_kvrepack", "madakv_kvrepack"))
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--budget-ratio", type=float)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    tasks = args.tasks or load_profile(args.profile)["tasks"]
    plans = []
    try:
        for method in args.methods:
            for task in tasks:
                plan = build_plan(method, benchmark="milebench", model=args.model, task=task,
                                  pretrained=args.pretrained, model_type=args.model_type,
                                  profile=args.profile, data_root=args.data_root, limit=args.limit,
                                  budget_ratio=args.budget_ratio, device=args.device)
                launch = [sys.executable, "-m", "kvrepack.run", "--benchmark", "milebench", "--model", args.model,
                          "--method", method, "--task", task, "--data-root", args.data_root,
                          "--profile", args.profile, "--limit", str(args.limit), "--device", args.device,
                          "--output", str(plan.output)]
                for option, value in (("--pretrained", args.pretrained), ("--model-type", args.model_type),
                                      ("--budget-ratio", args.budget_ratio)):
                    if value is not None:
                        launch.extend((option, str(value)))
                plans.append({"launch_command": launch, "launch_shell_command": shlex.join(launch), "plan": plan.to_dict()})
    except (OSError, ValueError) as exc:
        parser.exit(2, f"MileBench planning failed: {exc}\n")
    print(json.dumps({"benchmark": "milebench", "executed": False, "count": len(plans), "experiments": plans}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

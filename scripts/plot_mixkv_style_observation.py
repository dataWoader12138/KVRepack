#!/usr/bin/env python3
"""Plot the frozen, count-matched MixKV-style head means as four paper panels.

Inputs are an existing report's summary.json and head_summary.csv. This helper
does not perform inference, select token positions, fit statistics, or bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


TASKS = (("nextqa_oe_test", "NExTQA"), ("vatex_test", "VATEX"))
POOLS = (("visual_matched", "Visual", "left"), ("text_matched", "Text", "right"))
COLORS = {"visual_matched": "#377EB8", "text_matched": "#E68A2E"}
MODEL_TITLE = "Qwen2.5-VL-7B"
COMPARISON = "matched_visual_vs_text"


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def _csv_integer(value, label):
    if not isinstance(value, str) or not value or not value.isdecimal():
        raise ValueError(f"{label}: expected a nonnegative integer CSV field")
    return int(value)


def _cosine(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError(f"{label}: expected finite cosine in [-1,1]")
    return value


def load_report(report, *, key_stage="rope", expected_limit=200):
    """Validate complete coverage and reconcile plotted head means with summary."""
    _integer(expected_limit, "expected_limit", 1)
    if key_stage not in ("rope", "raw"):
        raise ValueError("key_stage must be rope or raw")
    report = Path(report).expanduser().resolve()
    summary_path, heads_path = report/"summary.json", report/"head_summary.csv"
    summary_bytes, heads_bytes = summary_path.read_bytes(), heads_path.read_bytes()
    summary = json.loads(summary_bytes)
    if (summary.get("schema_version") != 1 or summary.get("status") != "complete"
            or summary.get("expected_total_records") != 2*expected_limit
            or summary.get("validation_issues")):
        raise ValueError("Require a complete, valid report with the prescribed two-task row count")
    if summary.get("dimensions") != {"layers": 28, "native_kv_heads": 4}:
        raise ValueError("Require 28 decoder layers and 4 native KV heads; GQA repeats are not independent heads")
    sampling = summary.get("matched_sampling", {})
    if sampling.get("draws") != 32 or sampling.get("token_cap") != 16:
        raise ValueError("Require the frozen count-matched protocol: 32 draws and token cap 16")
    definition = summary.get("comparisons", {}).get(COMPARISON, {})
    if (definition.get("left") != ["multimodal", "visual_matched"]
            or definition.get("right") != ["multimodal", "text_matched"]):
        raise ValueError("Summary comparison does not identify the prescribed two matched multimodal pools")
    tasks = summary.get("tasks", {})
    if set(tasks) != {task for task, _ in TASKS}:
        raise ValueError("Require exactly NExTQA OE/test and VATEX/test")
    metrics = ("k_rope" if key_stage == "rope" else "k_raw", "v")
    for task, _ in TASKS:
        data = tasks[task]
        if (data.get("expected_records") != expected_limit or data.get("valid_completed_records") != expected_limit
                or data.get("missing_completed_doc_ids")):
            raise ValueError(f"{task}: require every one of the prescribed first {expected_limit} rows")
        for metric in metrics:
            values = data.get("comparisons", {}).get(COMPARISON, {}).get(metric, {})
            if values.get("n_records") != expected_limit:
                raise ValueError(f"{task}/{metric}: summary row coverage differs from expected_limit")
            for _, _, side in POOLS:
                _cosine(values.get(side), f"{task}/{metric}/{side}")
            difference = values.get("difference")
            if (type(difference) not in (int, float) or not math.isfinite(difference)
                    or abs(difference-(values["left"]-values["right"])) > 1e-12):
                raise ValueError(f"{task}/{metric}: inconsistent recorded paired difference")
    reader = csv.DictReader(io.StringIO(heads_bytes.decode("utf-8")))
    required = {"task", "condition", "pool", "metric", "layer", "kv_head", "n_records", "mean"}
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or required-set(reader.fieldnames):
        raise ValueError("head_summary.csv lacks unique required header fields")
    curves = {(task, pool, metric): {} for task, _ in TASKS for pool, _, _ in POOLS for metric in metrics}
    for line, row in enumerate(reader, 2):
        if row.get("condition") != "multimodal" or row.get("pool") not in COLORS or row.get("metric") not in metrics:
            continue
        key = (row.get("task"), row["pool"], row["metric"])
        if key not in curves:
            raise ValueError(f"CSV line {line}: unexpected task in a plotted matched pool")
        layer, head = _csv_integer(row.get("layer"), "layer"), _csv_integer(row.get("kv_head"), "kv_head")
        if not (0 <= layer < 28 and 0 <= head < 4):
            raise ValueError(f"CSV line {line}: layer/native-KV-head index outside 28x4 grid")
        if _csv_integer(row.get("n_records"), "n_records") != expected_limit:
            raise ValueError(f"CSV line {line}: head mean does not cover every prescribed row")
        try:
            mean = float(row["mean"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV line {line}: invalid cosine mean") from exc
        _cosine(mean, f"CSV line {line}")
        if (layer, head) in curves[key]:
            raise ValueError(f"CSV line {line}: duplicate layer/native-KV-head mean")
        curves[key][layer, head] = mean
    expected_grid = {(layer, head) for layer in range(28) for head in range(4)}
    max_difference = 0.0
    for (task, pool, metric), values in curves.items():
        if set(values) != expected_grid:
            raise ValueError(f"{task}/{pool}/{metric}: require all 112 unique layer/native-KV-head means")
        side = "left" if pool == "visual_matched" else "right"
        recorded = tasks[task]["comparisons"][COMPARISON][metric][side]
        error = abs(statistics.fmean(values.values())-recorded)
        max_difference = max(max_difference, error)
        if error > 1e-12:
            raise ValueError(f"{task}/{pool}/{metric}: CSV head means disagree with summary (absolute difference {error})")
    provenance = {
        "source_summary": str(summary_path), "source_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "source_head_summary": str(heads_path), "source_head_summary_sha256": hashlib.sha256(heads_bytes).hexdigest(),
        "source_provenance": summary.get("provenance", {}), "source_dimensions": summary["dimensions"],
        "csv_summary_mean_max_absolute_difference": max_difference,
    }
    return summary, curves, provenance


def draw(report, output, *, key_stage="rope", expected_limit=200):
    summary, curves, provenance = load_report(report, key_stage=key_stage, expected_limit=expected_limit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = (("k_rope", "K after RoPE") if key_stage == "rope" else ("k_raw", "K before RoPE"), ("v", "V"))
    edges, histograms = np.linspace(-1, 1, 31), []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.15), sharex=True, sharey="row")
        for column, (task, label) in enumerate(TASKS):
            for row, (metric, stage_label) in enumerate(metric_rows):
                axis = axes[row, column]
                for pool, _, side in POOLS:
                    values = list(curves[task, pool, metric].values())
                    counts = np.histogram(values, bins=edges)[0]
                    if int(counts.sum()) != 112:
                        raise ValueError("Fixed [-1,1] histogram bins did not retain all 112 signed head means")
                    histograms.append(dict(task=task, pool=pool, metric=metric, n_head_means=112,
                                           bin_counts=counts.tolist(), recorded_mean=summary["tasks"][task]["comparisons"][COMPARISON][metric][side]))
                    axis.hist(values, bins=edges, density=True, histtype="stepfilled", alpha=.35, color=COLORS[pool])
                    axis.hist(values, bins=edges, density=True, histtype="step", linewidth=.9, color=COLORS[pool])
                    axis.axvline(summary["tasks"][task]["comparisons"][COMPARISON][metric][side],
                                 color=COLORS[pool], linestyle="--", linewidth=1)
                axis.text(.025, .94, f"({chr(97+row*2+column)})", transform=axis.transAxes, va="top", fontsize=9)
                axis.set(xlim=(-1, 1), xticks=[-1, -.5, 0, .5, 1])
                if row == 0:
                    axis.set_title(f"{label} (n={expected_limit})", pad=7)
                else:
                    axis.set_xlabel("Mean within-head cosine")
                if column == 0:
                    axis.set_ylabel(f"{stage_label}\nDensity")
                axis.spines[["top", "right"]].set_visible(False)
                axis.grid(axis="y", alpha=.16, linewidth=.5)
                axis.set_axisbelow(True)
        fig.suptitle(f"{MODEL_TITLE}: equal-token global subsets", fontsize=11, y=.995)
        fig.legend(handles=[Patch(facecolor=COLORS[pool], edgecolor=COLORS[pool], alpha=.5, label=label)
                            for pool, label, _ in POOLS], loc="upper center", bbox_to_anchor=(.5, .955),
                   frameon=False, ncol=2, fontsize=9)
        fig.text(.5, .012, "112 layer/KV-head means per curve; descriptive units, not independent videos. Dashed lines: means.\n"
                 f"T≤16; R=32; n={expected_limit} inputs/dataset. Full causal context lengths are not matched.\n"
                 "Text: NExTQA questions / VATEX instruction. Similarity alone does not establish merge safety.",
                 ha="center", va="bottom", fontsize=7.2, linespacing=1.45)
        fig.subplots_adjust(left=.09, right=.985, bottom=.23, top=.81, wspace=.16, hspace=.19)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(str(output)+"."+extension, dpi=300, bbox_inches="tight")
        plt.close(fig)
    metadata = dict(schema_version=1, model_title=MODEL_TITLE, key_stage=key_stage,
                    intended_model_scope="User-specified Qwen2.5-VL-7B experiment; model identity is not inferred from absent record metadata",
                    source_status=summary["status"], expected_rows_per_task=expected_limit,
                    expected_total_inputs=2*expected_limit, comparison=COMPARISON,
                    matched_sampling=summary["matched_sampling"],
                    histogram=dict(bin_edges=edges.tolist(), bins=30, range=[-1, 1], density=True,
                                   unit="One layer/native-KV-head mean over the prescribed rows, after equal draw weights",
                                   curves=histograms),
                    statistics="Reuses recorded head means and recorded task means; no inference, selection, fitting, or bootstrap recomputation",
                    interpretation="Global sampling with equal measured token/pair counts; full causal context length, token roles, and spatial/linguistic distances remain unmatched. Not an exact MixKV Figure 1(b) replication or a merge-safety experiment.",
                    plotting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **provenance)
    Path(str(output)+".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix, without extension")
    parser.add_argument("--key-stage", choices=("rope", "raw"), default="rope")
    parser.add_argument("--expected-limit", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        metadata = draw(args.report, args.output, key_stage=args.key_stage, expected_limit=args.expected_limit)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(output=str(args.output), key_stage=args.key_stage,
                          inputs=metadata["expected_total_inputs"], head_means_per_curve=112)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

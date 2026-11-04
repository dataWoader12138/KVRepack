#!/usr/bin/env python3
"""Make one paper figure from a completed 400-input adjacent observation report.

Uses existing document means and source-video cluster intervals, without fitting
new statistics. Matplotlib is optional and imported only when drawing the figure.
The model name is not inferred from missing provenance; supply --title as needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


TASKS = (("nextqa_oe_test", "NExTQA"), ("vatex_test", "VATEX"))
METRICS = (("k_raw", "(a) K before RoPE"), ("v", "(b) V"))
COLORS = {"visual": "#377EB8", "text": "#E68A2E"}


def validate_summary(summary):
    if (summary.get("status") != "complete" or summary.get("expected_total_records") != 400
            or summary.get("validation_issues")):
        raise ValueError("Require a complete, valid 400-input report; incomplete results are not plotted")
    bootstrap = summary.get("bootstrap", {})
    if bootstrap.get("unit") != "source_video_id" or bootstrap.get("confidence") != .95:
        raise ValueError("Require existing 95% source-video cluster confidence intervals")
    for task, _ in TASKS:
        data = summary.get("tasks", {}).get(task, {})
        if (data.get("expected_records") != 200 or data.get("valid_completed_records") != 200
                or data.get("missing_completed_doc_ids")):
            raise ValueError(f"{task}: require all 200 valid completed prescribed inputs")
        cluster_counts = []
        for metric, _ in METRICS:
            values = data.get("metrics", {}).get(metric, {})
            n, clusters = values.get("n_records"), values.get("n_source_videos")
            if (type(n) is not int or not 1 <= n <= 200 or n != data.get("paired_records")
                    or type(clusters) is not int or not 2 <= clusters <= n):
                raise ValueError(f"{task}/{metric}: invalid paired-row/video coverage")
            cluster_counts.append(clusters)
            for group in ("visual", "text"):
                mean = values.get(group)
                ci = (values.get("ci95") or {}).get(group)
                if (type(mean) not in (int, float) or not math.isfinite(mean)
                        or not isinstance(ci, list) or len(ci) != 2
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in ci)
                        or ci[0] > ci[1] or max(abs(x) for x in [mean, *ci]) > 1.00001):
                    raise ValueError(f"{task}/{metric}/{group}: invalid finite cosine mean/interval")
        if len(set(cluster_counts)) != 1:
            raise ValueError(f"{task}: raw K and V must cover the same source-video clusters")
    return summary


def draw(summary_path, output_prefix, title=None):
    summary_path, output_prefix = Path(summary_path), Path(output_prefix)
    payload = summary_path.read_bytes()
    summary = validate_summary(json.loads(payload))
    # No ML imports, inference, pair selection, or statistical recomputation here.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError("This optional plotting helper needs matplotlib") from exc
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
        coverage = {}
        for ax, (metric, label) in zip(axes, METRICS):
            ticklabels = []
            for index, (task, task_label) in enumerate(TASKS):
                data, values = summary["tasks"][task], summary["tasks"][task]["metrics"][metric]
                n, clusters = values["n_records"], values["n_source_videos"]
                coverage[task] = dict(n_inputs=200, n_paired_inputs=n, n_source_video_clusters=clusters)
                eligible = f"\npaired={n}" if n != 200 else ""
                ticklabels.append(f"{task_label}\nn=200; G={clusters}{eligible}")
                for group, offset, marker in (("visual", -.12, "o"), ("text", .12, "s")):
                    x, ci = index + offset, values["ci95"][group]
                    # Draw interval endpoints directly: percentile intervals need
                    # not contain the point estimate; never clip signed values.
                    ax.vlines(x, ci[0], ci[1], color=COLORS[group], linewidth=1.3)
                    ax.hlines(ci, x - .035, x + .035, color=COLORS[group], linewidth=1.3)
                    ax.plot(x, values[group], marker=marker, markersize=4.6,
                            color=COLORS[group], linestyle="none")
            ax.set(title=label, xticks=[0, 1], xticklabels=ticklabels, xlim=(-.5, 1.5),
                   ylabel="Mean cosine similarity")
            # Automatic limits include every signed point and interval. A zero
            # line is helpful when visible, without forcing positive data to zero.
            low, high = ax.get_ylim()
            if low <= 0 <= high:
                ax.axhline(0, color=".65", linewidth=.7, linestyle="--", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.18, linewidth=.6)
            ax.set_axisbelow(True)
        handles = [Line2D([], [], color=COLORS[g], marker=m, markersize=4.6,
                          linestyle="none", label=label)
                   for g, m, label in (("visual", "o", "Visual spatial adjacent"),
                                       ("text", "s", "Text adjacent (task primary)"))]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .98),
                   ncol=2, frameon=False, fontsize=8.5)
        if title:
            fig.suptitle(title, fontsize=10.5, y=1.035)
        fig.text(.5, .025, "95% source-video cluster CIs; equal input weights. n=inputs; G=source videos.\n"
                 "Text: NExTQA question / VATEX instruction.",
                 ha="center", fontsize=7.4)
        fig.subplots_adjust(left=.085, right=.985, top=.80, bottom=.25, wspace=.32)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(str(output_prefix) + "." + extension, dpi=300, bbox_inches="tight")
        plt.close(fig)
    provenance = dict(schema_version=1, source_summary=str(summary_path.resolve()),
                      source_summary_sha256=hashlib.sha256(payload).hexdigest(), title=title,
                      source_status=summary["status"], n_inputs=400, coverage=coverage,
                      source_dimensions=summary.get("dimensions"),
                      intended_run_scope={"model": "Qwen2.5-VL-7B", "expected_layers": 28, "expected_kv_heads": 4,
                                          "note": "Scope only; model identity is not inferred from absent record metadata."},
                      source_provenance=summary.get("provenance", {}),
                      statistics="Existing per-input means and source-video 95% intervals; no new statistics computed",
                      values="Signed cosine values retained; no positivity clipping",
                      interpretation="Adjacent visual/text comparison; no locality-gain, causal, or merge-safety claim")
    Path(str(output_prefix) + ".metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix (without extension)")
    parser.add_argument("--title", help="Optional scope title; model identity is never inferred")
    args = parser.parse_args(argv)
    try:
        result = draw(args.summary, args.output, args.title)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(output=str(args.output), n_inputs=result["n_inputs"], coverage=result["coverage"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

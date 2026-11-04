"""CPU reports of adjacent visual/text cosine observations, without causal claims.

The independent resampling unit is source_video_id. Means give equal weight to
each prescribed document, after averaging its KV heads and layers. No cosine is
clipped and no token pairs, heads, or layers are treated as independent samples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


TASKS = {"nextqa_oe_test": "text_question", "vatex_test": "text_instruction"}
METRICS = ("k_raw", "v", "k_rope")


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _vector(values, heads, label):
    if not isinstance(values, list) or len(values) != heads:
        raise ValueError(f"{label} requires exactly {heads} KV-head means")
    if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1.00001 for x in values):
        raise ValueError(f"{label} must contain finite cosine means in [-1,1] (roundoff tolerance 1e-5)")
    return values


def _collapse(record, primary, expected_layers, expected_heads):
    layers = record.get("layer_stats")
    if not isinstance(layers, list) or not layers:
        raise ValueError("eligible record requires layer_stats")
    count = expected_layers or len(layers)
    ids = [_integer(x.get("layer"), "layer") for x in layers]
    if sorted(ids) != list(range(count)):
        raise ValueError(f"layers must cover exactly 0..{count - 1}")
    heads = expected_heads or _integer(layers[0].get("kv_heads"), "kv_heads", 1)
    detail, values = [], defaultdict(lambda: {"visual": [], "text": []})
    rope_presence = []
    for layer in sorted(layers, key=lambda x: x["layer"]):
        if layer.get("kv_heads") != heads:
            raise ValueError("KV-head count differs between records/layers")
        groups = layer.get("groups", {})
        visual, text = groups.get("visual_spatial", {}), groups.get(primary, {})
        rope_presence.extend(("k_rope" in visual, "k_rope" in text))
        for metric in METRICS:
            if metric == "k_rope" and (metric not in visual or metric not in text):
                continue
            vv = _vector(visual.get(metric), heads, f"visual_spatial.{metric}")
            tt = _vector(text.get(metric), heads, f"{primary}.{metric}")
            values[metric]["visual"].append(statistics.fmean(vv))
            values[metric]["text"].append(statistics.fmean(tt))
            for head, (v, t) in enumerate(zip(vv, tt)):
                detail.append(dict(metric=metric, layer=layer["layer"], kv_head=head,
                                   visual=v, text=t, difference=v - t))
    if any(rope_presence) and not all(rope_presence):
        raise ValueError("optional k_rope must cover both groups and every layer when supplied")
    means = {metric: dict(visual=statistics.fmean(v["visual"]), text=statistics.fmean(v["text"]))
             for metric, v in values.items()}
    for v in means.values():
        v["difference"] = v["visual"] - v["text"]
    return count, heads, means, detail


def _percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    return values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower)


def cluster_statistics(rows, bootstrap_samples=5000, seed=20261104):
    """Resample videos with all document members, preserving row-weighted means."""
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["source_video_id"]].append(row)
    if not rows:
        return dict(n_records=0, n_source_videos=0, visual=None, text=None, difference=None,
                    ci95=None, positive_source_video_fraction=None, positive_source_video_count=0,
                    uncertainty="no eligible paired records")
    point = {k: statistics.fmean(r[k] for r in rows) for k in ("visual", "text", "difference")}
    positive = sum(statistics.fmean(r["difference"] for r in members) > 0 for members in clusters.values())
    result = dict(n_records=len(rows), n_source_videos=len(clusters), **point,
                  positive_source_video_fraction=positive / len(clusters),
                  positive_source_video_count=positive, ci95=None)
    if len(clusters) < 2:
        result["uncertainty"] = "CI unavailable: fewer than two independent source-video clusters"
        return result
    # Sum/count form carries every member of each drawn video, including repeats.
    totals = [(len(members), sum(r["visual"] for r in members), sum(r["text"] for r in members))
              for _, members in sorted(clusters.items())]
    rng, draws = random.Random(seed), {k: [] for k in point}
    for _ in range(bootstrap_samples):
        sampled = [totals[rng.randrange(len(totals))] for _ in totals]
        denominator = sum(x[0] for x in sampled)
        visual = sum(x[1] for x in sampled) / denominator
        text = sum(x[2] for x in sampled) / denominator
        for key, value in (("visual", visual), ("text", text), ("difference", visual - text)):
            draws[key].append(value)
    result["ci95"] = {k: [_percentile(v, 0.025), _percentile(v, 0.975)] for k, v in draws.items()}
    result["uncertainty"] = "95% percentile source-video cluster bootstrap"
    return result


def build_summary(records, *, expected_count=200, expected_tasks=None, expected_layers=None,
                  expected_kv_heads=None, bootstrap_samples=5000, seed=20261104, read_errors=None):
    """Validate coverage and return stats plus compact per-record/head/layer tables."""
    tasks = tuple(expected_tasks or TASKS)
    _integer(expected_count, "expected_count", 1)
    if any(task not in TASKS for task in tasks):
        raise ValueError("Only NextQA OE test and VATEX test observations are supported")
    if expected_layers is not None:
        _integer(expected_layers, "expected_layers", 1)
    if expected_kv_heads is not None:
        _integer(expected_kv_heads, "expected_kv_heads", 1)
    issues, seen, completed, excluded = list(read_errors or []), defaultdict(set), defaultdict(set), defaultdict(list)
    paired, head_rows, context, status_counts = defaultdict(list), [], set(), defaultdict(Counter)
    dimensions = (expected_layers, expected_kv_heads)
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            task = record.get("task")
            if task not in tasks:
                raise ValueError(f"unexpected task {task!r}")
            doc_id = _integer(record.get("doc_id"), "doc_id")
            if doc_id >= expected_count:
                raise ValueError(f"doc_id outside prescribed first {expected_count} rows")
            if doc_id in seen[task]:
                raise ValueError(f"duplicate doc_id {doc_id}")
            seen[task].add(doc_id)
            status_counts[task][str(record.get("status"))] += 1
            if record.get("status") != "completed":
                raise ValueError(f"record status is {record.get('status')!r}: {record.get('error', '')}")
            _integer(record.get("input_tokens"), "input_tokens", 1)
            for key in ("video_id", "source_video_id"):
                if not isinstance(record.get(key), (str, int)) or str(record[key]) == "" or isinstance(record[key], bool):
                    raise ValueError(f"missing/invalid {key}")
            primary, counts = TASKS[task], record.get("pair_counts", {})
            visual_count = _integer(counts.get("visual_spatial"), "visual pair count")
            text_count = _integer(counts.get(primary), "primary text pair count")
            if visual_count != text_count:
                raise ValueError("visual and primary text pair counts must be equal per record")
            if record.get("eligibility", {}).get("primary") is False and visual_count:
                raise ValueError("eligibility.primary=False conflicts with positive matched pair count")
            info = {k: record[k] for k in ("model", "model_id", "model_name", "model_path", "frames",
                                         "num_frames", "frame_flags", "provenance", "run_metadata") if k in record}
            if info:
                context.add(json.dumps(info, sort_keys=True))
            if not visual_count:
                completed[task].add(doc_id)
                excluded[task].append(dict(doc_id=doc_id, source_video_id=str(record["source_video_id"]),
                                           reason="M=0: no primary matched adjacent pairs"))
                continue
            count, heads, means, details = _collapse(record, primary, *dimensions)
            dimensions = (count, heads)
            completed[task].add(doc_id)
            common = dict(task=task, doc_id=doc_id, video_id=str(record["video_id"]),
                          source_video_id=str(record["source_video_id"]), matched_pairs=visual_count,
                          input_tokens=record["input_tokens"])
            for metric, values in means.items():
                paired[task].append(dict(**common, metric=metric, **values))
            for detail in details:
                head_rows.append(dict(**common, **detail))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append(dict(record_index=index, task=record.get("task") if isinstance(record, dict) else None,
                               doc_id=record.get("doc_id") if isinstance(record, dict) else None, error=str(exc)))
    by_task = {}
    for task in tasks:
        missing = sorted(set(range(expected_count)) - completed[task])
        if missing:
            issues.append(dict(task=task, error="missing valid completed prescribed doc IDs", doc_ids=missing))
        metrics = {}
        for metric in METRICS:
            rows = [r for r in paired[task] if r["metric"] == metric]
            if rows or metric != "k_rope":
                metrics[metric] = cluster_statistics(rows, bootstrap_samples, seed)
        by_task[task] = dict(primary_text_group=TASKS[task], expected_records=expected_count,
                             seen_unique_records=len(seen[task]), valid_completed_records=len(completed[task]),
                             paired_records=metrics["k_raw"]["n_records"], excluded_primary_M0=excluded[task],
                             missing_completed_doc_ids=missing, status_counts=dict(status_counts[task]), metrics=metrics)
    return dict(schema_version=1, status="incomplete" if issues else "complete",
                expected_total_records=expected_count * len(tasks), tasks=by_task, validation_issues=issues,
                dimensions=dict(layers=dimensions[0], kv_heads=dimensions[1]),
                bootstrap=dict(samples=bootstrap_samples, seed=seed, confidence=0.95, unit="source_video_id",
                               estimator="equal document weights; equal layer weights then equal KV-head weights",
                               resampling="sample source videos with replacement; retain every member document"),
                interpretation="Adjacent visual-minus-text cosine difference is descriptive association; no distant controls, merge-safety or causal claim. Negative cosines are retained.",
                provenance={"record_metadata_variants": [json.loads(x) for x in sorted(context)],
                            "missing_metadata": "Model/frame settings are not inferred when absent from input records."},
                record_table=[row for task in tasks for row in paired[task]], head_table=head_rows)


def read_jsonl(path):
    records, issues = [], []
    with Path(path).open() as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                if not line.endswith("\n"):
                    raise ValueError("unterminated JSONL record")
                records.append(json.loads(line))
            except (ValueError, json.JSONDecodeError) as exc:
                issues.append(dict(line=line_no, error=str(exc)))
    return records, issues


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _detail_summary(rows, include_head):
    groups = defaultdict(list)
    keys = ["task", "metric", "layer"] + (["kv_head"] if include_head else [])
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        result.append(dict(zip(keys, key), n_records=len({r["doc_id"] for r in values}),
                           n_source_videos=len({r["source_video_id"] for r in values}),
                           **{k: statistics.fmean(r[k] for r in values) for k in ("visual", "text", "difference")}))
    return result


def _plot_task(task, data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for row, (metric, label) in enumerate((("k_raw", "Raw K"), ("v", "V"))):
        values = data["metrics"][metric]
        left, right = axes[row]
        if values["n_records"]:
            ci = values["ci95"]
            for x, key, color in ((0, "visual", "#377eb8"), (1, "text", "#e28e2c")):
                left.plot(x, values[key], "o", color=color)
                if ci:
                    left.vlines(x, *ci[key], color=color, linewidth=2)
                    left.hlines(ci[key], x - 0.06, x + 0.06, color=color)
            right.plot(0, values["difference"], "o", color="#555555")
            if ci:
                right.vlines(0, *ci["difference"], color="#555555", linewidth=2)
                right.hlines(ci["difference"], -0.06, 0.06, color="#555555")
        else:
            left.text(0.5, 0.5, "No eligible paired records", ha="center", transform=left.transAxes)
        left.set(xticks=[0, 1], xticklabels=["Visual spatial", data["primary_text_group"].replace("text_", "Text ")],
                 xlim=(-0.5, 1.5), ylabel=f"{label} mean cosine", title=f"{label}: matched document means")
        right.axhline(0, color="#999999", linestyle="--", linewidth=1)
        right.set(xticks=[0], xticklabels=["Visual − text"], xlim=(-0.5, 0.5), ylabel="Paired cosine difference",
                  title=f"n={values['n_records']}; videos={values['n_source_videos']}")
        for ax in (left, right):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{task}: adjacent pairs; 95% source-video cluster intervals")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{task}.{extension}", dpi=180)
    plt.close(fig)


def write_report(summary, output, *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "record_summary.csv", summary["record_table"],
               ["task", "doc_id", "video_id", "source_video_id", "matched_pairs", "input_tokens", "metric", "visual", "text", "difference"])
    for filename, include_head in (("head_summary.csv", True), ("layer_summary.csv", False)):
        _write_csv(output / filename, _detail_summary(summary["head_table"], include_head),
                   ["task", "metric", "layer"] + (["kv_head"] if include_head else []) +
                   ["n_records", "n_source_videos", "visual", "text", "difference"])
    task_rows = []
    for task, data in summary["tasks"].items():
        for metric, values in data["metrics"].items():
            row = dict(task=task, metric=metric, **{k: v for k, v in values.items() if k not in {"ci95", "uncertainty"}})
            for key in ("visual", "text", "difference"):
                interval = values["ci95"][key] if values["ci95"] else (None, None)
                row[key + "_ci95_low"], row[key + "_ci95_high"] = interval
            task_rows.append(row)
    _write_csv(output / "task_summary.csv", task_rows,
               ["task", "metric", "n_records", "n_source_videos", "visual", "visual_ci95_low", "visual_ci95_high",
                "text", "text_ci95_low", "text_ci95_high", "difference", "difference_ci95_low", "difference_ci95_high",
                "positive_source_video_count", "positive_source_video_fraction"])
    compact = {k: v for k, v in summary.items() if k not in {"head_table", "record_table"}}
    lines = ["# Adjacent visual/text observations", "", f"Status: **{summary['status']}**.", "",
             f"Prescribed sample: first {next(iter(summary['tasks'].values()))['expected_records']} rows per task ({summary['expected_total_records']} documents); records are never replaced or deduplicated by video.",
             "Each document averages its KV heads and layers. Task means weight documents equally. Confidence intervals resample source-video clusters with every member document.",
             "This is adjacent-pair geometry only. It does not establish a modality effect, a distant-pair contrast, or safe merging. Negative cosines are retained.", "",
             "| Task | Quantity | Paired documents / completed | Videos | Visual | Text | Visual − text (95% CI) | Positive video clusters |",
             "|---|---|---:|---:|---:|---:|---|---:|"]
    for task, data in summary["tasks"].items():
        for metric in ("k_raw", "v"):
            values = data["metrics"][metric]
            fmt = lambda x: "unavailable" if x is None else f"{x:.6f}"
            ci = values["ci95"]
            interval = ", ".join(fmt(x) for x in ci["difference"]) if ci else "CI unavailable"
            positive = f"{values['positive_source_video_count']}/{values['n_source_videos']}"
            lines.append(f"| {task} | {metric} | {values['n_records']} / {data['valid_completed_records']} | {values['n_source_videos']} | {fmt(values['visual'])} | {fmt(values['text'])} | {fmt(values['difference'])} [{interval}] | {positive} |")
        lines.extend(["", f"{task}: {len(data['excluded_primary_M0'])} completed records excluded from paired analysis because M=0; {len(data['missing_completed_doc_ids'])} prescribed IDs lack a valid completed record.", ""])
        if plots:
            _plot_task(task, data, output)
    lines.extend(["Positive-video counts use the sign of each source video's mean paired difference, without clipping cosines.",
                  "Head/layer CSVs contain descriptive means; task-level intervals are the cluster-bootstrap inference. Optional RoPE K summaries are position-dependent and secondary.",
                  "Model/frame provenance is included only when supplied in the records. No settings are inferred.", ""])
    if summary["validation_issues"]:
        lines += ["Coverage or validation problems (statistics are provisional):", "", "```json",
                  json.dumps(summary["validation_issues"], indent=2), "```", ""]
    compact["artifacts"] = dict(plots="PNG/PDF/SVG per task" if plots else "disabled",
                                tables=["task_summary.csv", "record_summary.csv", "head_summary.csv", "layer_summary.csv"])
    (output / "summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int)
    parser.add_argument("--expected-kv-heads", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--no-plots", action="store_true", help="Explicitly omit optional matplotlib figures")
    args = parser.parse_args(argv)
    records, errors = read_jsonl(args.input)
    summary = build_summary(records, expected_layers=args.expected_layers, expected_kv_heads=args.expected_kv_heads,
                            bootstrap_samples=args.bootstrap_samples, seed=args.seed, read_errors=errors)
    summary["provenance"]["input_jsonl"] = str(args.input.resolve())
    summary["provenance"]["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    try:
        write_report(summary, args.output, plots=not args.no_plots)
    except ImportError as exc:
        parser.error(f"Plotting needs matplotlib; install the optional dependency or explicitly pass --no-plots: {exc}")
    print(json.dumps(dict(status=summary["status"], output=str(args.output), validation_issues=len(summary["validation_issues"]))))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

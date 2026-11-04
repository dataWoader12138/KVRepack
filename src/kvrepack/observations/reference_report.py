"""Descriptive reports for five full text inputs versus five image/text inputs.

This reference-style pilot follows MixKV's per-layer/per-head distribution
format while comparing Qwen2.5 models. It never bootstraps heads or claims a
modality effect: the models, datasets, inputs and full prompt lengths differ.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .report import read_jsonl


GROUPS = {
    "text": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "doc_ids": (0, 1, 2, 3, 4),
             "label": "Qwen2.5-7B · LongBench qasper", "color": "#E68A2E"},
    "vision": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "doc_ids": (0, 200, 400, 600, 800),
               "label": "Qwen2.5-VL-7B · OCRBench", "color": "#377EB8"},
}
METRICS = ("k_raw", "k_rope", "v")
LAYERS, HEADS, HEAD_DIM = 28, 4, 128


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _vector(value, label):
    if not isinstance(value, list) or len(value) != HEADS:
        raise ValueError(f"{label} requires exactly {HEADS} native KV heads")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not -1 <= x <= 1 for x in value):
        raise ValueError(f"{label} requires finite signed cosines in [-1,1]")
    return value


def _collapse(record):
    length = _integer(record.get("input_tokens"), "input_tokens", 2)
    image_tokens = _integer(record.get("image_tokens"), "image_tokens")
    if image_tokens > length or (record["group"] == "text" and image_tokens != 0) or (record["group"] == "vision" and image_tokens == 0):
        raise ValueError("image_tokens must be zero for text and positive/bounded by input length for vision")
    layers = record.get("layer_stats")
    if not isinstance(layers, list) or len(layers) != LAYERS:
        raise ValueError(f"Require exactly {LAYERS} decoder layers")
    if sorted(_integer(layer.get("layer"), "layer") for layer in layers) != list(range(LAYERS)):
        raise ValueError("Decoder layer IDs must cover 0..27 exactly once")
    values, inclusive, details = defaultdict(list), defaultdict(list), []
    for layer in layers:
        if layer.get("kv_heads") != HEADS or layer.get("head_dim") != HEAD_DIM:
            raise ValueError("Require 4 native KV heads of dimension 128, before GQA repetition")
        if layer.get("sequence_length") != length:
            raise ValueError("Every observed layer must cover the full recorded input_tokens")
        captures = layer.get("capture_counts", {})
        if set(captures) != set(METRICS) or any(type(captures[metric]) is not int or captures[metric] != 1 for metric in METRICS):
            raise ValueError("Raw K, post-RoPE K and V must each be captured exactly once per layer")
        included = layer.get("inclusive")
        if not isinstance(included, dict) or set(included) != set(METRICS):
            raise ValueError("Each layer requires explicit inclusive summaries for every metric")
        for metric in METRICS:
            off = _vector(layer.get(metric), metric)
            inc = _vector(included[metric], "inclusive."+metric)
            values[metric].append(statistics.fmean(off))
            inclusive[metric].append(statistics.fmean(inc))
            for head in range(HEADS):
                details.append(dict(metric=metric, layer=layer["layer"], kv_head=head,
                                    mean=off[head], inclusive_mean=inc[head]))
    return ({metric: statistics.fmean(values[metric]) for metric in METRICS},
            {metric: statistics.fmean(inclusive[metric]) for metric in METRICS}, details)


def build_summary(records, *, read_errors=None):
    """Require the prescribed 5+5 inputs; average inputs equally, regardless of N."""
    issues, seen, valid = list(read_errors or []), defaultdict(set), defaultdict(set)
    record_rows, head_values, group_meta, source_meta = [], defaultdict(list), {}, defaultdict(list)
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            if record.get("schema_version") != 1 or record.get("status") != "completed":
                raise ValueError("Require completed schema_version=1 records")
            group = record.get("group")
            if group not in GROUPS:
                raise ValueError("group must be text or vision")
            doc_id = _integer(record.get("doc_id"), "doc_id")
            if doc_id not in GROUPS[group]["doc_ids"]:
                raise ValueError("doc_id outside the prescribed five-input pilot")
            if doc_id in seen[group]:
                raise ValueError("duplicate group/doc_id")
            seen[group].add(doc_id)
            if record.get("model_id") != GROUPS[group]["model_id"]:
                raise ValueError("Model ID differs from the prescribed Qwen2.5 group/model mapping")
            revision = record.get("model_revision")
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("Record must declare its model_revision")
            metadata = record.get("source_metadata")
            if not isinstance(metadata, dict) or not metadata:
                raise ValueError("Record must preserve nonempty source_metadata")
            serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
            means, inclusive, details = _collapse(record)
            identity = dict(model_id=record["model_id"], model_revision=revision)
            if group in group_meta and identity != group_meta[group]:
                raise ValueError("Model ID/revision changed within the same five-input group")
            group_meta[group] = identity
            common = dict(group=group, doc_id=doc_id, **identity,
                          input_tokens=record["input_tokens"], image_tokens=record["image_tokens"])
            for metric in METRICS:
                record_rows.append(dict(**common, metric=metric, mean=means[metric], inclusive_mean=inclusive[metric],
                                        ordered_nonself_pairs=record["input_tokens"]*(record["input_tokens"]-1),
                                        source_metadata_json=serialized))
            for detail in details:
                key = (group, detail["metric"], detail["layer"], detail["kv_head"])
                head_values[key].append((detail["mean"], detail["inclusive_mean"]))
            source_meta[group].append(dict(doc_id=doc_id, input_tokens=record["input_tokens"],
                                          image_tokens=record["image_tokens"], metadata=metadata))
            valid[group].add(doc_id)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append(dict(record_index=index, group=record.get("group") if isinstance(record, dict) else None,
                               doc_id=record.get("doc_id") if isinstance(record, dict) else None, error=str(exc)))
    heads, layer_values = [], defaultdict(list)
    for (group, metric, layer, head), values in sorted(head_values.items()):
        item = dict(group=group, metric=metric, layer=layer, kv_head=head, n_inputs=len(values),
                    mean=statistics.fmean(value[0] for value in values),
                    inclusive_mean=statistics.fmean(value[1] for value in values))
        heads.append(item)
        layer_values[group, metric, layer].append(item)
    layers = [dict(group=key[0], metric=key[1], layer=key[2], kv_heads=len(values),
                   n_inputs=len(valid[key[0]]), mean=statistics.fmean(row["mean"] for row in values),
                   inclusive_mean=statistics.fmean(row["inclusive_mean"] for row in values))
              for key, values in sorted(layer_values.items())]
    by_group = {}
    for group, specification in GROUPS.items():
        missing = sorted(set(specification["doc_ids"])-valid[group])
        if missing:
            issues.append(dict(group=group, error="missing valid completed prescribed inputs", doc_ids=missing))
        primary = [row for row in record_rows if row["group"] == group and row["metric"] == METRICS[0]]
        by_group[group] = dict(expected_doc_ids=list(specification["doc_ids"]), valid_doc_ids=sorted(valid[group]),
                               seen_unique_inputs=len(seen[group]), n_inputs=len(valid[group]), missing_doc_ids=missing,
                               **group_meta.get(group, dict(model_id=specification["model_id"], model_revision=None)),
                               input_tokens={"min": min(row["input_tokens"] for row in primary),
                                             "max": max(row["input_tokens"] for row in primary),
                                             "mean": statistics.fmean(row["input_tokens"] for row in primary)} if primary else None,
                               metrics={metric: dict(mean=statistics.fmean(row["mean"] for row in record_rows if row["group"] == group and row["metric"] == metric),
                                                     inclusive_mean=statistics.fmean(row["inclusive_mean"] for row in record_rows if row["group"] == group and row["metric"] == metric)) for metric in METRICS} if primary else {})
    comparisons = {}
    for metric in METRICS:
        if all(metric in by_group[group]["metrics"] for group in GROUPS):
            text, vision = (by_group[group]["metrics"][metric]["mean"] for group in ("text", "vision"))
            comparisons[metric] = dict(text=text, vision=vision, vision_minus_text=vision-text,
                                       text_inputs=by_group["text"]["n_inputs"], vision_inputs=by_group["vision"]["n_inputs"])
    return dict(schema_version=1, experiment="mixkv_reference_style_qwen25_full_input_5plus5",
                status="incomplete" if issues else "complete", expected_total_inputs=10,
                dimensions=dict(layers=LAYERS, native_kv_heads=HEADS, head_dim=HEAD_DIM),
                groups=by_group, comparisons=comparisons, validation_issues=issues,
                statistic="All distinct-token-pair cosine over the full native prefill sequence in each input; self-pairs excluded; signed values retained",
                aggregation="Equal KV-head and layer weights within each input, then equal weights for the five prescribed inputs per model; no token-length weighting",
                histogram_unit="One layer/native-KV-head mean over the five inputs; 28*4=112 descriptive means per model; not independent observations",
                uncertainty="Descriptive fixed 5+5-input pilot only; no confidence intervals, significance tests, or head-as-sample inference",
                interpretation="Reference-style adaptation of MixKV Figure 1(b), not an exact reproduction: Qwen2.5 text and vision-language models, fixed LongBench qasper and OCRBench inputs. Models, datasets, input lengths and semantic roles differ; comparison does not isolate a within-model modality effect or establish merge safety.",
                source_inputs={group: sorted(values, key=lambda row: row["doc_id"]) for group, values in source_meta.items()},
                record_table=record_rows, head_table=heads, layer_table=layers)


def _csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary, output, *, raw=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    histogram_metadata = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.5), sharex=True)
        for axis, (metric, label) in zip(axes, (("k_raw" if raw else "k_rope", "K before RoPE" if raw else "K after RoPE"), ("v", "V"))):
            for group, specification in GROUPS.items():
                values = [row["mean"] for row in summary["head_table"] if row["group"] == group and row["metric"] == metric]
                if len(values) != LAYERS*HEADS:
                    raise ValueError("Complete figures require all 112 native layer/head means per model")
                counts, edges = np.histogram(values, bins=np.linspace(-1, 1, 31))
                if int(counts.sum()) != LAYERS*HEADS:
                    raise ValueError("Histogram failed to retain all signed head means")
                histogram_metadata.append(dict(group=group, metric=metric, n_head_means=len(values), bin_counts=counts.tolist()))
                axis.hist(values, bins=edges, density=True, histtype="stepfilled", color=specification["color"], alpha=.35, label=specification["label"])
                axis.hist(values, bins=edges, density=True, histtype="step", color=specification["color"], linewidth=.9)
                axis.axvline(summary["groups"][group]["metrics"][metric]["mean"], color=specification["color"], linestyle="--", linewidth=1)
            axis.set(ylabel="Density", title=label, xlim=(-1, 1), xticks=[-1, -.5, 0, .5, 1])
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=.15)
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        axes[-1].set_xlabel("Full-input, within-head mean cosine")
        fig.suptitle("Qwen2.5-7B vs Qwen2.5-VL-7B: 5 + 5 full-input pilot", fontsize=10.5)
        fig.text(.5, .015, "112 layer/KV-head means per curve; equal weights for 5 full inputs/model. Dashed lines: means.\n"
                 "Descriptive pilot; no confidence intervals. Models, datasets and full context lengths differ.",
                 ha="center", va="bottom", fontsize=7.4)
        fig.subplots_adjust(left=.095, right=.985, bottom=.16, top=.88, hspace=.34)
        name = "reference_raw" if raw else "reference_main"
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output/f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return dict(bin_edges=edges.tolist(), bins=30, range=[-1, 1], curves=histogram_metadata)


def write_report(summary, output, *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output/"record_summary.csv", summary["record_table"], ["group", "doc_id", "model_id", "model_revision", "input_tokens", "image_tokens", "ordered_nonself_pairs", "metric", "mean", "inclusive_mean", "source_metadata_json"])
    for table, fields in (("head", ["kv_head"]), ("layer", ["kv_heads"])):
        _csv(output/f"{table}_summary.csv", summary[f"{table}_table"], ["group", "metric", "layer", *fields, "n_inputs", "mean", "inclusive_mean"])
    group_rows = [dict(group=group, model_id=data["model_id"], model_revision=data["model_revision"], n_inputs=data["n_inputs"],
                       metric=metric, **values, **{"input_tokens_"+key: value for key, value in (data["input_tokens"] or {}).items()})
                  for group, data in summary["groups"].items() for metric, values in data["metrics"].items()]
    _csv(output/"group_summary.csv", group_rows, ["group", "model_id", "model_revision", "n_inputs", "metric", "mean", "inclusive_mean", "input_tokens_min", "input_tokens_max", "input_tokens_mean"])
    _csv(output/"comparison_summary.csv", [dict(metric=metric, **values) for metric, values in summary["comparisons"].items()],
         ["metric", "text", "vision", "vision_minus_text", "text_inputs", "vision_inputs"])
    compact = {key: value for key, value in summary.items() if not key.endswith("_table")}
    compact["artifacts"] = dict(tables=["record_summary.csv", "head_summary.csv", "layer_summary.csv", "group_summary.csv", "comparison_summary.csv"], plots="disabled")
    if plots and summary["status"] == "complete":
        compact["artifacts"]["plots"] = dict(main=_plot(summary, output), raw_supplement=_plot(summary, output, raw=True))
    lines = ["# Reference-style full-input 5+5 pilot", "", f"Status: **{summary['status']}**.", "", summary["interpretation"], "", summary["statistic"], "", summary["aggregation"], "", summary["histogram_unit"], "", summary["uncertainty"], "",
             "| Quantity | Text model (5 inputs) | Vision-language model (5 inputs) | Vision − text |", "|---|---:|---:|---:|"]
    for metric, values in summary["comparisons"].items():
        lines.append(f"| {metric} | {values['text']:.6f} | {values['vision']:.6f} | {values['vision_minus_text']:.6f} |")
    lines += ["", "The main figure uses post-RoPE K and V; raw K is provided separately. Self-pairs are excluded; negative values and opposite-direction results are retained.", "",
              "| Group | Model | Revision | Valid document IDs | Input token min / max / mean |", "|---|---|---|---|---|"]
    for group, data in summary["groups"].items():
        lengths = data["input_tokens"]
        label = f"{lengths['min']} / {lengths['max']} / {lengths['mean']:.1f}" if lengths else "unavailable"
        lines.append(f"| {group} | {data['model_id']} | {data['model_revision']} | {data['valid_doc_ids']} | {label} |")
    if summary["validation_issues"]:
        lines += ["", "Incomplete coverage or validation errors; all displayed aggregates are provisional:", "", "```json", json.dumps(summary["validation_issues"], indent=2), "```"]
    (output/"summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    records, issues = read_jsonl(args.input)
    summary = build_summary(records, read_errors=issues)
    summary["provenance"] = dict(input_jsonl=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    write_report(summary, args.output, plots=not args.no_plots)
    print(json.dumps(dict(status=summary["status"], output=str(args.output), validation_issues=len(summary["validation_issues"]))))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

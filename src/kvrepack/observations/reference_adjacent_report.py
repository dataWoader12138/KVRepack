"""Reports for complete-sequence adjacent-token Qwen2.5 observations.

Only (i, i+1), i=0..N-2, enter each recorded head mean. All-pair records and
the earlier sampled within-VLM modality-neighbor records are rejected. Models
and data remain different across the two groups; heads are descriptive units.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile

from .report import read_jsonl
from .reference_report import (
    GROUPS, HEAD_DIM, HEADS, LAYERS, METRICS, _cluster_bootstrap, _csv,
    _integer, _plot as _reference_plot, _vector,
)


STATISTICS_KIND = "all_sequence_adjacent_v1"


def expected_ids(sample_count):
    """Use prefixes of the same prescribed 200-input reference sources."""
    if type(sample_count) is not int or not 1 <= sample_count <= 200:
        raise ValueError("sample_count must be an integer from 1 through 200")
    return {"text": tuple(range(sample_count)),
            "vision": tuple(range(0, 5*sample_count, 5))}


def _collapse(record):
    if record.get("statistics_kind") != STATISTICS_KIND:
        raise ValueError("Require all_sequence_adjacent_v1; all-pair or sampled-neighbor records cannot be used")
    length = _integer(record.get("input_tokens"), "input_tokens", 2)
    pair_count = _integer(record.get("adjacent_pair_count"), "adjacent_pair_count", 1)
    if pair_count != length-1:
        raise ValueError("Complete immediate adjacency requires exactly input_tokens-1 pairs")
    image_tokens = _integer(record.get("image_tokens"), "image_tokens")
    if image_tokens > length or (record["group"] == "text" and image_tokens != 0) or (record["group"] == "vision" and image_tokens == 0):
        raise ValueError("image_tokens must be zero for text and positive/bounded by input length for vision")
    layers = record.get("layer_stats")
    if not isinstance(layers, list) or len(layers) != LAYERS:
        raise ValueError(f"Require exactly {LAYERS} decoder layers")
    if sorted(_integer(layer.get("layer"), "layer") for layer in layers) != list(range(LAYERS)):
        raise ValueError("Decoder layer IDs must cover 0..27 exactly once")
    values, details = defaultdict(list), []
    for layer in layers:
        if type(layer.get("kv_heads")) is not int or layer["kv_heads"] != HEADS or type(layer.get("head_dim")) is not int or layer["head_dim"] != HEAD_DIM:
            raise ValueError("Require 4 native KV heads of dimension 128, before GQA repetition")
        if type(layer.get("sequence_length")) is not int or layer["sequence_length"] != length:
            raise ValueError("Every layer must cover the complete recorded input length")
        if type(layer.get("adjacent_pair_count")) is not int or layer["adjacent_pair_count"] != pair_count:
            raise ValueError("Layer adjacency count disagrees with the complete sequence")
        captures = layer.get("capture_counts", {})
        if set(captures) != set(METRICS) or any(type(captures[metric]) is not int or captures[metric] != 1 for metric in METRICS):
            raise ValueError("Raw K, post-RoPE K and V must each be captured exactly once per layer")
        for metric in METRICS:
            heads = _vector(layer.get(metric), metric)
            values[metric].append(statistics.fmean(heads))
            details.extend(dict(metric=metric, layer=layer["layer"], kv_head=head, mean=value)
                           for head, value in enumerate(heads))
    return {metric: statistics.fmean(values[metric]) for metric in METRICS}, details


def build_summary(records, *, sample_count=200, bootstrap_samples=5000, seed=20261117,
                  read_errors=None):
    """Aggregate validated complete sequences with equal input/head/layer weights."""
    prescribed = expected_ids(sample_count)
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    _integer(seed, "seed")
    issues, seen, valid = list(read_errors or []), defaultdict(set), defaultdict(set)
    record_rows, head_values, source_meta, group_meta = [], defaultdict(list), defaultdict(list), {}
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            if type(record.get("schema_version")) is not int or record["schema_version"] != 1 or record.get("status") != "completed":
                raise ValueError("Require completed schema_version=1 records")
            group = record.get("group")
            if group not in GROUPS:
                raise ValueError("group must be text or vision")
            doc_id = _integer(record.get("doc_id"), "doc_id")
            if doc_id not in prescribed[group]:
                raise ValueError("doc_id outside the prescribed source prefix")
            if doc_id in seen[group]:
                raise ValueError("duplicate group/doc_id")
            seen[group].add(doc_id)
            if record.get("model_id") != GROUPS[group]["model_id"]:
                raise ValueError("Model ID differs from the prescribed group/model mapping")
            revision = record.get("model_revision")
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("Record must declare its model_revision")
            metadata = record.get("source_metadata")
            if not isinstance(metadata, dict) or not metadata:
                raise ValueError("Record must preserve nonempty source_metadata")
            serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
            cluster_key = "context_sha256" if group == "text" else "original_image_sha256"
            cluster = metadata.get(cluster_key)
            if sample_count == 200 and (not isinstance(cluster, str) or not cluster):
                raise ValueError(f"200-input source-cluster statistics require {cluster_key}")
            cluster = cluster if isinstance(cluster, str) and cluster else f"input:{doc_id}"
            means, details = _collapse(record)
            identity = dict(model_id=record["model_id"], model_revision=revision)
            if group in group_meta and identity != group_meta[group]:
                raise ValueError("Model revision changed within the group")
            # Only fully validated records enter any aggregate.
            group_meta[group] = identity
            common = dict(group=group, doc_id=doc_id, **identity,
                          input_tokens=record["input_tokens"], image_tokens=record["image_tokens"],
                          adjacent_pair_count=record["adjacent_pair_count"], source_cluster_id=cluster)
            record_rows.extend(dict(**common, metric=metric, mean=means[metric], source_metadata_json=serialized)
                               for metric in METRICS)
            for detail in details:
                head_values[group, detail["metric"], detail["layer"], detail["kv_head"]].append(detail["mean"])
            source_meta[group].append(dict(doc_id=doc_id, input_tokens=record["input_tokens"],
                                           image_tokens=record["image_tokens"], adjacent_pair_count=record["adjacent_pair_count"],
                                           source_cluster_id=cluster, metadata=metadata))
            valid[group].add(doc_id)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append(dict(record_index=index, group=record.get("group") if isinstance(record, dict) else None,
                               doc_id=record.get("doc_id") if isinstance(record, dict) else None, error=str(exc)))

    heads, layer_values = [], defaultdict(list)
    for (group, metric, layer, head), values in sorted(head_values.items()):
        item = dict(group=group, metric=metric, layer=layer, kv_head=head,
                    n_inputs=len(values), mean=statistics.fmean(values))
        heads.append(item)
        layer_values[group, metric, layer].append(item)
    layers = [dict(group=key[0], metric=key[1], layer=key[2], kv_heads=len(values),
                   n_inputs=len(valid[key[0]]), mean=statistics.fmean(row["mean"] for row in values))
              for key, values in sorted(layer_values.items())]
    by_group = {}
    for group, specification in GROUPS.items():
        missing = sorted(set(prescribed[group])-valid[group])
        if missing:
            issues.append(dict(group=group, error="missing valid completed prescribed inputs", doc_ids=missing))
        primary = [row for row in record_rows if row["group"] == group and row["metric"] == METRICS[0]]
        by_group[group] = dict(expected_doc_ids=list(prescribed[group]), valid_doc_ids=sorted(valid[group]),
                               seen_unique_inputs=len(seen[group]), n_inputs=len(valid[group]), missing_doc_ids=missing,
                               **group_meta.get(group, dict(model_id=specification["model_id"], model_revision=None)),
                               input_tokens=dict(min=min(row["input_tokens"] for row in primary),
                                                 max=max(row["input_tokens"] for row in primary),
                                                 mean=statistics.fmean(row["input_tokens"] for row in primary)) if primary else None,
                               adjacent_pairs=dict(total=sum(row["adjacent_pair_count"] for row in primary),
                                                   min=min(row["adjacent_pair_count"] for row in primary),
                                                   max=max(row["adjacent_pair_count"] for row in primary),
                                                   mean=statistics.fmean(row["adjacent_pair_count"] for row in primary)) if primary else None,
                               metrics={metric: dict(mean=statistics.fmean(row["mean"] for row in record_rows
                                                                           if row["group"] == group and row["metric"] == metric))
                                        for metric in METRICS} if primary else {})
    comparisons = {}
    for metric in METRICS:
        if all(metric in by_group[group]["metrics"] for group in GROUPS):
            text, vision = (by_group[group]["metrics"][metric]["mean"] for group in GROUPS)
            comparisons[metric] = dict(text=text, vision=vision, vision_minus_text=vision-text,
                                       text_inputs=by_group["text"]["n_inputs"], vision_inputs=by_group["vision"]["n_inputs"])
    bootstrap = dict(enabled=sample_count == 200, samples=bootstrap_samples, seed=seed,
                     unit="Source context/image, not tokens, adjacent pairs, layers or heads",
                     estimator="Independent source-cluster resampling across groups, retaining all member inputs and equal input weights")
    if sample_count == 200:
        group_ci, difference_ci, cluster_metadata = _cluster_bootstrap(record_rows, bootstrap_samples, seed)
        for group, data in by_group.items():
            data.update(cluster_metadata[group])
            for metric, values in data["metrics"].items():
                values["ci95"] = group_ci[group][metric] if group_ci[group] is not None else None
        for metric, values in comparisons.items():
            values["ci95"] = difference_ci[metric] if difference_ci is not None else None
        bootstrap["confidence"] = .95
    return dict(schema_version=1, experiment=f"qwen25_complete_sequence_adjacent_{sample_count}pergroup",
                statistics_kind=STATISTICS_KIND, status="incomplete" if issues else "complete",
                sample_count=sample_count, expected_total_inputs=2*sample_count,
                dimensions=dict(layers=LAYERS, native_kv_heads=HEADS, head_dim=HEAD_DIM),
                groups=by_group, comparisons=comparisons, validation_issues=issues,
                statistic="Mean cosine over every immediate original-sequence pair (i,i+1), i=0..N-2; exactly N-1 pairs per input, per native KV head/layer",
                pair_scope="Complete unpadded prefill sequence including image, text and structural tokens; includes adjacent pairs across modality, image-grid row and other sequence boundaries",
                token_sampling=False,
                aggregation=f"Equal adjacent-pair weights within each input/head/layer; equal head/layer weights and then equal weights for {sample_count} inputs per group; no input-length weighting",
                histogram_unit=f"One layer/native-KV-head mean over {sample_count} inputs; 28*4=112 descriptive means per model, not independent observations",
                uncertainty="Exploratory 95% source-cluster bootstrap; heads and token pairs are not independent samples" if sample_count == 200 else "Descriptive pilot only; no confidence intervals or head-as-sample inference",
                bootstrap=bootstrap,
                interpretation="Adjacent-token observation of Qwen2.5 text and vision-language models on fixed LongBench qasper/OCRBench inputs. Models, datasets, sequence lengths and token roles differ; this does not isolate a within-model modality effect or establish merge safety.",
                source_inputs={group: sorted(values, key=lambda row: row["doc_id"]) for group, values in source_meta.items()},
                record_table=record_rows, head_table=heads, layer_table=layers)


def _combined_plot(summary, output, *, raw=False):
    """Keep the full four-curve legend above the data axes, below the title."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    from .reference_report import _density_curve

    key_metric = "k_raw" if raw else "k_rope"
    channel_styles = {key_metric: dict(linestyle="-", hatch="/", label="Key"),
                      "v": dict(linestyle=(0, (4, 2)), hatch=".", label="Value")}
    grid = np.linspace(-1, 1, 2049)
    minimum = min(row["mean"] for row in summary["head_table"] if row["metric"] in channel_styles)
    display_left = max(-1., min(-.2, math.floor((minimum-.1)*10)/10))
    curve_metadata, legend_handles, legend_labels = [], [], []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "text.color": "#30343B",
                         "axes.labelcolor": "#30343B", "xtick.color": "#70747A", "ytick.color": "#70747A",
                         "axes.edgecolor": "#C8CBD0", "hatch.linewidth": .4}):
        fig, axis = plt.subplots(figsize=(7.2, 4.5))
        fig.subplots_adjust(left=.095, right=.985, bottom=.14, top=.73)
        for metric, style in channel_styles.items():
            for group, specification in GROUPS.items():
                values = [row["mean"] for row in summary["head_table"] if row["group"] == group and row["metric"] == metric]
                if len(values) != LAYERS*HEADS:
                    raise ValueError("Complete figures require all 112 native layer/head means per model")
                counts, edges = np.histogram(values, bins=np.linspace(-1, 1, 31))
                density, bandwidth = _density_curve(values, grid)
                color = specification["color"]
                curve_metadata.append(dict(group=group, metric=metric, n_head_means=len(values),
                    bin_counts=counts.tolist(), bandwidth=bandwidth,
                    density_area=float(np.sum((density[:-1]+density[1:])*np.diff(grid))/2),
                    observed_min=min(values), observed_max=max(values), color=color))
                axis.fill_between(grid, 0, density, facecolor=to_rgba(color, .025),
                                  edgecolor=to_rgba(color, .28), hatch=style["hatch"], linewidth=0, zorder=1)
                axis.plot(grid, density, color=color, linewidth=2.1, linestyle=style["linestyle"], zorder=4)
                legend_handles.append((Patch(facecolor=to_rgba(color, .04), edgecolor=to_rgba(color, .45),
                                             hatch=style["hatch"], linewidth=.4),
                                       Line2D([], [], color=color, linewidth=2.1, linestyle=style["linestyle"])))
                model_label = specification["model_id"].split("/")[-1].removesuffix("-Instruct")
                legend_labels.append(model_label+" · "+style["label"])
        axis.set(ylabel="Density", xlabel="Cosine Similarity", xlim=(display_left, 1), ylim=(0, None))
        axis.set_xticks([x for x in (-1, -.5, -.2, 0, .2, .4, .6, .8, 1) if x >= display_left])
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(color="#E2E4E7", linewidth=.65, linestyle=(0, (2, 2)))
        legend = axis.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=1)},
                             frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.02),
                             ncol=2, columnspacing=1.2)
        title = fig.suptitle("Key and Value Similarity Distribution Comparison", y=.97, fontsize=10)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_bounds = axis.get_window_extent(renderer)
        legend_bounds = legend.get_window_extent(renderer)
        title_bounds = title.get_window_extent(renderer)
        # Save measured display positions so a layout test can check this
        # property using actual rendered text, rather than just anchor values.
        legend_axis_bounds = legend_bounds.transformed(axis.transAxes.inverted()).bounds
        if legend_bounds.y0 <= axes_bounds.y1 or title_bounds.y0 <= legend_bounds.y1:
            raise ValueError("Combined legend must sit above the data axes and below the title")
        name = "adjacent_raw_combined" if raw else "adjacent_combined"
        for extension in ("png", "pdf", "svg"):
            path = Path(output)/f"{name}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
        plt.close(fig)
    return dict(representation="boundary-reflected Gaussian KDE", bandwidth_rule="Scott; minimum 0.01",
                bin_edges=edges.tolist(), bins=30, range=[-1, 1], display_range=[display_left, 1],
                grid_points=len(grid), histogram_counts="audit only; no histogram drawn", curves=curve_metadata,
                view="combined", cache_styles=channel_styles, statistics_kind=STATISTICS_KIND,
                legend=dict(location="outside_above_data_axes", bounds_in_axes_coordinates=list(legend_axis_bounds),
                            entirely_above_data_axes=True, below_title=True),
                overlapping_fills=dict(face_alpha=.025, hatch_alpha=.28, hatch_linewidth=.4,
                                       fills_zorder=1, curves_zorder=4, mean_markers=False))


def _plot(summary, output, *, raw=False, view="panels"):
    """Reuse the unchanged reference KDE renderer, keeping adjacent names separate."""
    if view == "combined":
        return _combined_plot(summary, output, raw=raw)
    with tempfile.TemporaryDirectory(prefix=".adjacent-plots-", dir=output) as directory:
        temporary = Path(directory)
        metadata = _reference_plot(summary, temporary, raw=raw, view=view)
        for source in temporary.iterdir():
            shutil.copyfile(source, Path(output)/source.name.replace("reference_", "adjacent_", 1))
    metadata["statistics_kind"] = STATISTICS_KIND
    return metadata


def write_report(summary, output, *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _csv(output/"record_summary.csv", summary["record_table"],
         ["group", "doc_id", "model_id", "model_revision", "input_tokens", "image_tokens", "adjacent_pair_count", "source_cluster_id", "metric", "mean", "source_metadata_json"])
    _csv(output/"source_inputs.csv", [dict(group=group, **{key: value[key] for key in ("doc_id", "input_tokens", "image_tokens", "adjacent_pair_count", "source_cluster_id")},
                                          source_metadata_json=json.dumps(value["metadata"], ensure_ascii=False, sort_keys=True))
                                     for group, values in summary["source_inputs"].items() for value in values],
         ["group", "doc_id", "input_tokens", "image_tokens", "adjacent_pair_count", "source_cluster_id", "source_metadata_json"])
    for table, fields in (("head", ["kv_head"]), ("layer", ["kv_heads"])):
        _csv(output/f"{table}_summary.csv", summary[f"{table}_table"], ["group", "metric", "layer", *fields, "n_inputs", "mean"])
    group_rows = []
    for group, data in summary["groups"].items():
        for metric, values in data["metrics"].items():
            row = dict(group=group, model_id=data["model_id"], model_revision=data["model_revision"], n_inputs=data["n_inputs"],
                       n_source_clusters=data.get("n_source_clusters"), metric=metric, **values,
                       **{"input_tokens_"+key: value for key, value in (data["input_tokens"] or {}).items()},
                       **{"adjacent_pairs_"+key: value for key, value in (data["adjacent_pairs"] or {}).items()})
            row["ci95_low"], row["ci95_high"] = values.get("ci95") or (None, None)
            group_rows.append(row)
    _csv(output/"group_summary.csv", group_rows,
         ["group", "model_id", "model_revision", "n_inputs", "n_source_clusters", "metric", "mean", "ci95_low", "ci95_high",
          "input_tokens_min", "input_tokens_max", "input_tokens_mean", "adjacent_pairs_total", "adjacent_pairs_min", "adjacent_pairs_max", "adjacent_pairs_mean"])
    comparison_rows = []
    for metric, values in summary["comparisons"].items():
        row = dict(metric=metric, **values)
        row["ci95_low"], row["ci95_high"] = values.get("ci95") or (None, None)
        comparison_rows.append(row)
    _csv(output/"comparison_summary.csv", comparison_rows,
         ["metric", "text", "vision", "vision_minus_text", "ci95_low", "ci95_high", "text_inputs", "vision_inputs"])
    compact = {key: value for key, value in summary.items() if not key.endswith("_table")}
    compact["artifacts"] = dict(tables=["record_summary.csv", "head_summary.csv", "layer_summary.csv", "group_summary.csv", "comparison_summary.csv", "source_inputs.csv"], plots="disabled")
    if plots and summary["status"] == "complete":
        compact["artifacts"]["plots"] = dict(main=_plot(summary, output), raw_supplement=_plot(summary, output, raw=True),
            key=_plot(summary, output, view="key"), value=_plot(summary, output, view="value"), combined=_plot(summary, output, view="combined"))
    count = summary["sample_count"]
    lines = [f"# Complete-sequence adjacent-token {count}+{count} observation", "", f"Status: **{summary['status']}**.", "",
             summary["statistic"], "", summary["pair_scope"], "", summary["aggregation"], "", summary["histogram_unit"], "",
             "| Quantity | Qwen2.5-7B / qasper | Qwen2.5-VL-7B / OCRBench | Vision − text |", "|---|---:|---:|---:|"]
    for metric, values in summary["comparisons"].items():
        lines.append(f"| {metric} | {values['text']:.6f} | {values['vision']:.6f} | {values['vision_minus_text']:.6f} |")
    lines += ["", "The main figures use post-RoPE K and V. Raw K is supplemental. Every immediate adjacent pair enters the observation; there is no token/pair subsampling. Signed values and all prescribed inputs are retained.", "",
              "| Group | Completed inputs | Adjacent pairs total | Input tokens min / max / mean |", "|---|---:|---:|---|"]
    for group, data in summary["groups"].items():
        lengths, pairs = data["input_tokens"], data["adjacent_pairs"]
        length_label = f"{lengths['min']} / {lengths['max']} / {lengths['mean']:.2f}" if lengths else "unavailable"
        lines.append(f"| {group} | {data['n_inputs']} | {pairs['total'] if pairs else 0} | {length_label} |")
    lines += ["", summary["interpretation"], "", summary["uncertainty"]]
    if summary["validation_issues"]:
        lines += ["", "Incomplete coverage or invalid records; displayed aggregates are provisional:", "", "```json", json.dumps(summary["validation_issues"], indent=2), "```"]
    (output/"summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-input", type=Path, required=True)
    parser.add_argument("--vision-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    records, errors, source_files = [], [], {}
    for group, path in (("text", args.text_input), ("vision", args.vision_input)):
        loaded, issues = read_jsonl(path)
        errors.extend(dict(input_group=group, **issue) for issue in issues)
        for row in loaded:
            if not isinstance(row, dict) or row.get("group") != group:
                errors.append(dict(input_group=group, error="Input file contains a row belonging to the wrong model group"))
            else:
                records.append(row)
        source_files[group] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary = build_summary(records, sample_count=args.sample_count, bootstrap_samples=args.bootstrap_samples,
                            seed=args.seed, read_errors=errors)
    summary["provenance"] = dict(input_files=source_files)
    write_report(summary, args.output, plots=not args.no_plots)
    print(json.dumps(dict(status=summary["status"], output=str(args.output), validation_issues=len(summary["validation_issues"]))))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU-only reports for the preregistered MixKV-style all-pairs observation.

Every histogram point is one native KV head in one decoder layer, averaged
equally over prescribed input rows. Statistical intervals instead resample
source videos with all their rows. Heads are never independent task samples.
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


TASKS = ("nextqa_oe_test", "vatex_test")
METRICS = ("k_raw", "v", "k_rope")
POOLS = {"multimodal": ("prompt_all", "visual_all", "text_primary", "natural_user"),
         "text_only": ("prompt_all", "text_primary", "natural_user")}
COMPARISONS = {
    "matched_visual_vs_text": (("multimodal", "visual_matched"), ("multimodal", "text_matched")),
    "matched_text_context": (("multimodal", "text_matched"), ("text_only", "text_matched")),
    "full_prompt": (("multimodal", "prompt_all"), ("text_only", "prompt_all")),
    "within_multimodal": (("multimodal", "visual_all"), ("multimodal", "text_primary")),
    "text_context": (("multimodal", "text_primary"), ("text_only", "text_primary")),
}
LABELS = {
    "matched_visual_vs_text": ("Visual: equal-token subsets", "Text: equal-token subsets"),
    "matched_text_context": ("Matched text with preceding video", "Matched text without video"),
    "full_prompt": ("Video + text: entire prompt", "Text only: entire prompt"),
    "within_multimodal": ("Visual positions in video + text", "Primary text in video + text"),
    "text_context": ("Primary text with preceding video", "Primary text without video"),
}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _vector(value, heads, label):
    if not isinstance(value, list) or len(value) != heads:
        raise ValueError(f"{label} must contain exactly {heads} native KV heads")
    if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1.000001 for x in value):
        raise ValueError(f"{label} requires finite cosine values in [-1,1] (roundoff tolerance 1e-6)")
    return value


def _collapse(record, expected_layers, expected_heads, expected_matched_draws, matched_token_cap):
    """Return validated means/details only after the entire record is valid."""
    conditions = record.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(POOLS):
        raise ValueError("conditions must contain exactly multimodal and text_only")
    layers_count, heads = expected_layers, expected_heads
    means, details, matched_tokens = {}, [], None
    for condition, required_pools in POOLS.items():
        value = conditions[condition]
        length = _integer(value.get("input_tokens"), f"{condition}.input_tokens", 2)
        pools = value.get("pools")
        if not isinstance(pools, dict) or set(pools) != set(required_pools):
            raise ValueError(f"{condition}: requires exactly the prescribed token pools")
        for pool, positions in pools.items():
            if not isinstance(positions, list) or len(positions) < 2:
                raise ValueError(f"{condition}/{pool}: fewer than two token positions")
            if any(type(p) is not int or not 0 <= p < length for p in positions):
                raise ValueError(f"{condition}/{pool}: invalid native token position")
            if positions != sorted(set(positions)):
                raise ValueError(f"{condition}/{pool}: positions must be sorted and unique")
        if pools["prompt_all"] != list(range(length)):
            raise ValueError(f"{condition}: prompt_all must cover the entire native prompt")
        if not set(pools["text_primary"]).issubset(pools["natural_user"]):
            raise ValueError("primary text pool must lie inside the natural user-text pool")
        if condition == "multimodal" and set(pools["visual_all"]) & set(pools["natural_user"]):
            raise ValueError("visual and natural text positions must not overlap")
        matched = value.get("matched_pools")
        matched_names = ("visual_matched", "text_matched") if condition == "multimodal" else ("text_matched",)
        if not isinstance(matched, dict) or set(matched) != set(matched_names):
            raise ValueError(f"{condition}: missing exactly prescribed matched_pools")
        if condition == "multimodal":
            matched_tokens = min(matched_token_cap, len(pools["visual_all"]), len(pools["text_primary"]))
        token_count = matched_tokens
        for name, draws in matched.items():
            base_pool = pools["visual_all" if name == "visual_matched" else "text_primary"]
            if not isinstance(draws, list) or len(draws) != expected_matched_draws:
                raise ValueError(f"{condition}/{name}: wrong number of matched subset draws")
            for draw in draws:
                if not isinstance(draw, list) or len(draw) != token_count or any(type(p) is not int for p in draw):
                    raise ValueError(f"{condition}/{name}: matched draws must have the same prescribed token count")
                if draw != sorted(set(draw)) or not set(draw).issubset(base_pool):
                    raise ValueError(f"{condition}/{name}: each matched draw must be unique, sorted, and inside its source pool")
        layers = value.get("layer_stats")
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"{condition}: missing layer_stats")
        layers_count = layers_count or len(layers)
        if sorted(_integer(x.get("layer"), "layer") for x in layers) != list(range(layers_count)):
            raise ValueError(f"{condition}: layer IDs must cover exactly 0..{layers_count-1}")
        heads = heads or _integer(layers[0].get("kv_heads"), "kv_heads", 1)
        accumulation = defaultdict(list)
        for layer in layers:
            if layer.get("kv_heads") != heads:
                raise ValueError("native KV-head dimensions differ across layers/conditions/records")
            if "sequence_length" in layer and layer["sequence_length"] != length:
                raise ValueError("observed sequence length differs from native input length")
            if layer.get("capture_counts") != dict.fromkeys(METRICS, 1):
                raise ValueError("each projection/rotary metric must be captured exactly once")
            all_pairs = layer.get("all_pairs")
            if not isinstance(all_pairs, dict) or set(all_pairs) != set(required_pools):
                raise ValueError("all_pairs lacks an exact prescribed pool set")
            for pool, positions in pools.items():
                group = all_pairs[pool]
                if group.get("token_count") != len(positions):
                    raise ValueError("all-pairs token_count differs from selected pool")
                if group.get("status") != "supported" or group.get("pair_count") != len(positions)*(len(positions)-1):
                    raise ValueError("all-pairs pool must support exactly T*(T-1) ordered distinct-token pairs")
                if group.get("capture_counts") != dict.fromkeys(METRICS, 1):
                    raise ValueError("each pool/metric must be captured exactly once")
                for metric in METRICS:
                    vector = _vector(group.get(metric), heads, f"{condition}/{pool}/{metric}")
                    accumulation[(pool, metric)].append(statistics.fmean(vector))
                    for head, number in enumerate(vector):
                        details.append((condition, pool, metric, layer["layer"], head, number))
            matched_stats = layer.get("matched_all_pairs")
            if not isinstance(matched_stats, dict) or set(matched_stats) != set(matched_names):
                raise ValueError("matched_all_pairs requires exactly the prescribed equal-token pools")
            for pool in matched_names:
                group = matched_stats[pool]
                if group.get("token_count") != token_count or group.get("draw_count") != expected_matched_draws:
                    raise ValueError("matched_all_pairs token/draw count differs from selected matched subsets")
                if group.get("status") != "supported":
                    raise ValueError("matched pool must have supported all-pairs statistics")
                if group.get("pair_count_per_draw") != token_count*(token_count-1):
                    raise ValueError("matched pool must use all T*(T-1) ordered distinct-token pairs per draw")
                if group.get("capture_counts") != dict.fromkeys(METRICS, 1):
                    raise ValueError("each matched pool/metric must be captured exactly once")
                draw_values = group.get("draw_values")
                if not isinstance(draw_values, dict) or set(draw_values) != set(METRICS):
                    raise ValueError("matched pool requires every metric's per-draw values")
                for metric in METRICS:
                    vector = _vector(group.get(metric), heads, f"{condition}/{pool}/{metric}")
                    values = draw_values[metric]
                    if not isinstance(values, list) or len(values) != expected_matched_draws:
                        raise ValueError("matched per-draw values have inconsistent draw count")
                    for draw in values:
                        _vector(draw, heads, f"{condition}/{pool}/{metric} per-draw")
                    if any(abs(statistics.fmean(draw[h] for draw in values)-vector[h]) > 1e-6 for h in range(heads)):
                        raise ValueError("matched head mean differs from its equal-weight draw mean")
                    accumulation[(pool, metric)].append(statistics.fmean(vector))
                    for head, number in enumerate(vector):
                        details.append((condition, pool, metric, layer["layer"], head, number))
        for (pool, metric), numbers in accumulation.items():
            means[(condition, pool, metric)] = statistics.fmean(numbers)
    for pool in ("text_primary", "natural_user"):
        if len(conditions["multimodal"]["pools"][pool]) != len(conditions["text_only"]["pools"][pool]):
            raise ValueError(f"{pool}: token counts differ between matched context conditions")
    mm, tt = conditions["multimodal"], conditions["text_only"]
    mm_primary, tt_primary = mm["pools"]["text_primary"], tt["pools"]["text_primary"]
    for mm_draw, tt_draw in zip(mm["matched_pools"]["text_matched"], tt["matched_pools"]["text_matched"]):
        if [mm_primary.index(p) for p in mm_draw] != [tt_primary.index(p) for p in tt_draw]:
            raise ValueError("text-only matched draws must preserve primary-token indices from the multimodal condition")
    return layers_count, heads, means, details


def _joint_statistics(rows, bootstrap_samples, seed):
    """Same source-video bootstrap draw for every comparison and metric."""
    import numpy as np

    keys = [(comparison, metric) for comparison in COMPARISONS for metric in METRICS]
    if not rows:
        return {name: {} for name in COMPARISONS}
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["source_video_id"]].append(row)
    totals, counts = [], []
    for _, members in sorted(clusters.items()):
        counts.append(len(members))
        totals.append([[sum(member["comparisons"][key][side] for member in members)
                        for side in ("left", "right")] for key in keys])
    totals, counts = np.asarray(totals, dtype=np.float64), np.asarray(counts, dtype=np.int64)
    point = totals.sum(axis=0)/counts.sum()
    intervals = None
    if len(clusters) >= 2:
        rng = np.random.default_rng(seed)
        draws = np.empty((bootstrap_samples, len(keys), 3), dtype=np.float64)
        for start in range(0, bootstrap_samples, 256):
            size = min(256, bootstrap_samples-start)
            indices = rng.integers(0, len(clusters), size=(size, len(clusters)))
            sampled = totals[indices].sum(axis=1)/counts[indices].sum(axis=1)[:, None, None]
            draws[start:start+size, :, :2] = sampled
            draws[start:start+size, :, 2] = sampled[:, :, 0]-sampled[:, :, 1]
        intervals = np.quantile(draws, [.025, .975], axis=0)
    result = {name: {} for name in COMPARISONS}
    for index, (comparison, metric) in enumerate(keys):
        left, right = point[index].tolist()
        positive = sum(statistics.fmean(member["comparisons"][(comparison, metric)]["difference"]
                                        for member in members) > 0 for members in clusters.values())
        result[comparison][metric] = dict(
            n_records=len(rows), n_source_videos=len(clusters), left=left, right=right,
            difference=left-right, positive_source_video_count=positive,
            ci95={side: intervals[:, index, i].tolist() for i, side in enumerate(("left", "right", "difference"))}
                 if intervals is not None else None,
            uncertainty="95% percentile source-video cluster bootstrap" if intervals is not None
                        else "CI unavailable: fewer than two source-video clusters")
    return result


def build_summary(records, *, expected_limit=200, expected_tasks=TASKS,
                  expected_layers=None, expected_kv_heads=None, bootstrap_samples=5000,
                  seed=20261104, read_errors=None, expected_matched_draws=32, matched_token_cap=16):
    """Strict coverage validation; malformed rows never partially enter statistics."""
    _integer(expected_limit, "expected_limit", 1)
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    _integer(expected_matched_draws, "expected_matched_draws", 1)
    _integer(matched_token_cap, "matched_token_cap", 2)
    for key, value in (("expected_layers", expected_layers), ("expected_kv_heads", expected_kv_heads)):
        if value is not None:
            _integer(value, key, 1)
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks) or set(expected_tasks)-set(TASKS):
        raise ValueError("expected_tasks must be a unique nonempty subset of the supported tasks")
    issues, dimensions = list(read_errors or []), (expected_layers, expected_kv_heads)
    seen, valid, paired = defaultdict(set), defaultdict(set), defaultdict(list)
    record_rows, comparison_rows, head_values = [], [], defaultdict(list)
    contexts = set()
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            task, doc = record.get("task"), _integer(record.get("doc_id"), "doc_id")
            if task not in expected_tasks or doc >= expected_limit:
                raise ValueError("record lies outside prescribed task/first-row range")
            if doc in seen[task]:
                raise ValueError("duplicate doc_id")
            seen[task].add(doc)
            if record.get("status") != "completed":
                raise ValueError(f"record status is {record.get('status')!r}")
            source = record.get("source_video_id")
            if not isinstance(source, (str, int)) or isinstance(source, bool) or not str(source):
                raise ValueError("missing source_video_id")
            info = {key: record[key] for key in ("model", "model_id", "model_path", "frames", "run_metadata") if key in record}
            context = json.dumps(info, sort_keys=True, allow_nan=False) if info else None
            count, heads, means, details = _collapse(record, *dimensions, expected_matched_draws, matched_token_cap)
            dimensions = (count, heads)
            common = dict(task=task, doc_id=doc, source_video_id=str(source))
            row = dict(**common, comparisons={})
            for comparison, (lhs, rhs) in COMPARISONS.items():
                for metric in METRICS:
                    left, right = means[(*lhs, metric)], means[(*rhs, metric)]
                    values = dict(left=left, right=right, difference=left-right)
                    row["comparisons"][(comparison, metric)] = values
                    comparison_rows.append(dict(**common, comparison=comparison, metric=metric, **values))
            paired[task].append(row)
            for (condition, pool, metric), mean in means.items():
                condition_data = record["conditions"][condition]
                matched_pool = condition_data["matched_pools"].get(pool)
                record_rows.append(dict(**common, condition=condition, pool=pool, metric=metric, mean=mean,
                                        token_count=len(matched_pool[0]) if matched_pool else len(condition_data["pools"][pool]),
                                        draw_count=len(matched_pool) if matched_pool else 1,
                                        input_tokens=condition_data["input_tokens"]))
            for condition, pool, metric, layer, head, number in details:
                head_values[(task, condition, pool, metric, layer, head)].append(number)
            if context is not None:
                contexts.add(context)
            valid[task].add(doc)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append(dict(record_index=index, task=record.get("task") if isinstance(record, dict) else None,
                               doc_id=record.get("doc_id") if isinstance(record, dict) else None, error=str(exc)))
    by_task = {}
    for task in expected_tasks:
        missing = sorted(set(range(expected_limit))-valid[task])
        if missing:
            issues.append(dict(task=task, error="missing valid completed prescribed doc IDs", doc_ids=missing))
        by_task[task] = dict(expected_records=expected_limit, seen_unique_records=len(seen[task]),
                             valid_completed_records=len(valid[task]), missing_completed_doc_ids=missing,
                             comparisons=_joint_statistics(paired[task], bootstrap_samples, seed))
    head_rows, layer_values = [], defaultdict(list)
    for key, values in sorted(head_values.items()):
        task, condition, pool, metric, layer, head = key
        mean = statistics.fmean(values)
        head_rows.append(dict(task=task, condition=condition, pool=pool, metric=metric, layer=layer,
                              kv_head=head, n_records=len(values), mean=mean))
        layer_values[key[:-1]].append(mean)
    layer_rows = [dict(zip(("task", "condition", "pool", "metric", "layer"), key), mean=statistics.fmean(values),
                       kv_heads=len(values), n_records=len(valid[key[0]])) for key, values in sorted(layer_values.items())]
    return dict(schema_version=1, experiment="mixkv_style_all_pairs_context_control",
                status="incomplete" if issues else "complete", expected_total_records=expected_limit*len(expected_tasks),
                dimensions=dict(layers=dimensions[0], native_kv_heads=dimensions[1]), tasks=by_task,
                validation_issues=issues, comparisons={name: dict(left=list(pair[0]), right=list(pair[1]),
                                                                  labels=list(LABELS[name])) for name, pair in COMPARISONS.items()},
                statistic="Primary: equal token-count subsets in both modalities, all distinct-token-pair cosine within each draw, then equal draw weights. Supplement: entire predefined pools. Self-pairs excluded; negative cosines retained",
                matched_sampling=dict(draws=expected_matched_draws, token_cap=matched_token_cap,
                                      tokens="min(token_cap, visual token count, primary text token count); same T across visual and text pools/contexts",
                                      resampling="uniform subsets without replacement within a draw; draws can overlap; outcome independent"),
                histogram_unit="One decoder-layer/native-KV-head mean over all prescribed rows; layers*KV-heads points per condition/task, not independent samples",
                bootstrap=dict(samples=bootstrap_samples, seed=seed, confidence=.95, unit="source_video_id",
                               estimator="Equal documents, equal layers, equal native KV heads",
                               rng="numpy.default_rng; shared cluster draws across comparisons/metrics"),
                interpretation="Adaptation of MixKV Figure 1(b), not an exact paper reproduction: same Qwen2.5-VL model across conditions; primary visual/text subset lengths are matched, but full forward prompt lengths differ. Source modality, context and token role remain confounded. Cosine alone does not establish merge safety.",
                provenance=dict(record_metadata_variants=[json.loads(x) for x in sorted(contexts)]),
                record_table=record_rows, comparison_table=comparison_rows, head_table=head_rows, layer_table=layer_rows)


def _csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(task, summary, output, comparison, *, raw=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6), sharex=True, constrained_layout=True)
    lhs, rhs = COMPARISONS[comparison]
    for axis, (metric, title) in zip(axes, (("k_raw" if raw else "k_rope", "K before RoPE" if raw else "Cached K after RoPE"), ("v", "V"))):
        for (condition, pool), label, color in zip((lhs, rhs), LABELS[comparison], ("#2878b5", "#e18727")):
            values = [row["mean"] for row in summary["head_table"] if row["task"] == task and row["metric"] == metric
                      and row["condition"] == condition and row["pool"] == pool]
            if values:
                axis.hist(values, bins=np.linspace(-1, 1, 31), density=True, histtype="stepfilled", alpha=.35,
                          color=color, label=f"{label} (n={len(values)} heads)")
                axis.axvline(statistics.fmean(values), color=color, linewidth=1.3, linestyle="--")
        axis.set(ylabel="Head distribution density", title=title, xlim=(-1, 1))
        axis.legend(fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("All distinct-token-pair mean cosine; each head averaged over input rows")
    fig.suptitle(f"{task}: {summary['tasks'][task]['valid_completed_records']} rows; {comparison}")
    name = task + ("_raw" if raw else "") + ("" if comparison == "matched_visual_vs_text" else "_supp_"+comparison)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output/f"{name}.{extension}", dpi=300)
    plt.close(fig)


def write_report(summary, output, *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["task", "doc_id", "source_video_id"]
    _csv(output/"record_summary.csv", summary["record_table"], fields+["condition", "pool", "metric", "token_count", "draw_count", "input_tokens", "mean"])
    _csv(output/"record_comparisons.csv", summary["comparison_table"], fields+["comparison", "metric", "left", "right", "difference"])
    _csv(output/"head_summary.csv", summary["head_table"], ["task", "condition", "pool", "metric", "layer", "kv_head", "n_records", "mean"])
    _csv(output/"layer_summary.csv", summary["layer_table"], ["task", "condition", "pool", "metric", "layer", "kv_heads", "n_records", "mean"])
    task_rows, lines = [], ["# MixKV-style all-pairs observation", "", f"Status: **{summary['status']}**.", "", summary["interpretation"], "",
                            summary["statistic"], "", summary["histogram_unit"], "",
                            "Task estimates first average native KV heads and layers within each row, then weight prescribed rows equally. Confidence intervals resample source videos with all member rows, using shared draws across comparisons and metrics.", "",
                            "| Task | Comparison | Metric | Rows / videos | Left | Right | Left − right (95% CI) |",
                            "|---|---|---|---:|---:|---:|---|"]
    for task, data in summary["tasks"].items():
        for comparison, metrics in data["comparisons"].items():
            for metric, values in metrics.items():
                row = dict(task=task, comparison=comparison, metric=metric, **{k: v for k, v in values.items() if k not in ("ci95", "uncertainty")})
                for key in ("left", "right", "difference"):
                    bounds = values["ci95"][key] if values["ci95"] else (None, None)
                    row[key+"_ci95_low"], row[key+"_ci95_high"] = bounds
                task_rows.append(row)
                bounds = values["ci95"]["difference"] if values["ci95"] else None
                interval = f"[{bounds[0]:.6f}, {bounds[1]:.6f}]" if bounds else "CI unavailable"
                lines.append(f"| {task} | {comparison} | {metric} | {values['n_records']} / {values['n_source_videos']} | {values['left']:.6f} | {values['right']:.6f} | {values['difference']:.6f} {interval} |")
        if plots and data["valid_completed_records"]:
            for comparison in COMPARISONS:
                _plot(task, summary, output, comparison)
            _plot(task, summary, output, "matched_visual_vs_text", raw=True)
    _csv(output/"task_summary.csv", task_rows, ["task", "comparison", "metric", "n_records", "n_source_videos"]+
         [item for key in ("left", "right", "difference") for item in (key, key+"_ci95_low", key+"_ci95_high")]+["positive_source_video_count"])
    compact = {key: value for key, value in summary.items() if not key.endswith("_table")}
    compact["artifacts"] = dict(tables=["record_summary.csv", "record_comparisons.csv", "head_summary.csv", "layer_summary.csv", "task_summary.csv"],
                                histogram_bins=dict(minimum=-1, maximum=1, bins=30, shared=True),
                                plots="PNG/PDF/SVG" if plots else "disabled")
    if summary["validation_issues"]:
        lines += ["", "Results are provisional because validation failed:", "", "```json", json.dumps(summary["validation_issues"], indent=2), "```"]
    (output/"summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-limit", type=int, default=200)
    parser.add_argument("--expected-layers", type=int)
    parser.add_argument("--expected-kv-heads", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--matched-draws", type=int, default=32)
    parser.add_argument("--matched-token-cap", type=int, default=16)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    records, errors = read_jsonl(args.input)
    summary = build_summary(records, expected_limit=args.expected_limit, expected_layers=args.expected_layers,
                            expected_kv_heads=args.expected_kv_heads, bootstrap_samples=args.bootstrap_samples,
                            seed=args.seed, read_errors=errors, expected_matched_draws=args.matched_draws,
                            matched_token_cap=args.matched_token_cap)
    summary["provenance"].update(input_jsonl=str(args.input.resolve()),
                                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    write_report(summary, args.output, plots=not args.no_plots)
    print(json.dumps(dict(status=summary["status"], output=str(args.output), validation_issues=len(summary["validation_issues"]))))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline, paired-input scores for similarity versus three random-head seeds.

No model/GPU is loaded. Task functions execute from their unchanged source AST;
VATEX uses the selected reference corpus's document frequencies. Only NextQA's
additive WUPS is given a paired source-video bootstrap interval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout
import csv
import functools
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import sys
import tempfile
from types import SimpleNamespace
import urllib.request


TASKS = ("nextqa_oe_test", "vatex_test")
VARIANTS = ("similarity", "random_global_seed0", "random_global_seed1", "random_global_seed2")
VATEX_METRICS = ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr")
COUNTS = {"nextqa_oe_test": 34, "vatex_test": 17}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} requires an integer >= {minimum}")
    return value


def _mask(value, name):
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list) or len(value[0]) != 28:
        raise ValueError(f"{name} requires [batch=1,layers=28,native KV heads=4]")
    for layer in value[0]:
        if not isinstance(layer, list) or len(layer) != 4 or any(type(v) is not bool for v in layer):
            raise ValueError(f"{name} requires native-head boolean masks")
    return [bit for layer in value[0] for bit in layer]


def _audit(row, variant):
    audit = row.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("Missing selector audit")
    strategy = "similarity" if variant == "similarity" else "random_global"
    if (audit.get("status") != "completed" or audit.get("strategy") != strategy
            or (variant != "similarity" and audit.get("seed") != int(variant[-1]))
            or audit.get("native_kv_heads") != 4 or audit.get("layers") != 28 or audit.get("batch_size") != 1):
        raise ValueError("Selector audit strategy/seed/status/native dimensions differ from the declared run")
    baseline = _mask(audit.get("baseline_masks"), "baseline_masks")
    selected = _mask(audit.get("selected_masks"), "selected_masks")
    expected = COUNTS[row["task"]]
    for prefix, mask in (("baseline", baseline), ("selected", selected)):
        if sum(mask) != expected or audit.get(prefix+"_selected_count" if prefix == "baseline" else "selected_count") != [expected]:
            raise ValueError("Global selected-head count differs from the frozen method parameters")
        layer_counts = [[sum(mask[layer*4:layer*4+4]) for layer in range(28)]]
        if audit.get(prefix+"_layer_counts") != layer_counts:
            raise ValueError("Recorded layer counts disagree with actual mask identities")
    if variant == "similarity" and selected != baseline:
        raise ValueError("Similarity strategy does not preserve the original selector")
    for name in ("expected_effective_merge_count", "actual_merge_calls", "expected_removed_tokens", "actual_removed_tokens"):
        _integer(audit.get(name), name)
    if audit["expected_effective_merge_count"] != expected or audit["actual_merge_calls"] != expected:
        raise ValueError("Selected-head count and actual effective merge calls differ")
    if audit["expected_removed_tokens"] != audit["actual_removed_tokens"]:
        raise ValueError("Actual token merge work differs from the frozen budget")
    shapes = audit.get("prefill_cache_shapes")
    if not isinstance(shapes, list) or len(shapes) != 28 or [s.get("layer") for s in shapes] != list(range(28)):
        raise ValueError("Require every layer's prefill cache shapes")
    for shape in shapes:
        key, value = shape.get("key_shape"), shape.get("value_shape")
        if (not isinstance(key, list) or len(key) != 4 or key != value or key[:2] != [1, 4]
                or type(key[2]) is not int or key[2] < 1 or key[3] != 128
                or not shape.get("key_dtype") or shape.get("key_dtype") != shape.get("value_dtype")):
            raise ValueError("Invalid native prefill K/V cache shape or dtype")
    hashes = audit.get("candidate_hashes")
    if (not isinstance(hashes, list) or len(hashes) != 28
            or [h.get("layer", h.get("layer_idx")) for h in hashes] != list(range(28))):
        raise ValueError("Require unchanged pre-gate candidate fingerprints for all layers")
    union = sum(a or b for a, b in zip(baseline, selected))
    return dict(selected_count=expected, effective_merge_count=audit["actual_merge_calls"],
                removed_tokens=audit["actual_removed_tokens"], mask_hamming=sum(a != b for a, b in zip(baseline, selected)),
                mask_jaccard=sum(a and b for a, b in zip(baseline, selected))/union if union else 1.,
                baseline_mask=baseline, selected_mask=selected)


def load_runs(run_root, *, limit=20):
    """Require eight completed, matched runs; reject any fairness-audit mismatch."""
    if type(limit) is not int or limit not in (1, 20):
        raise ValueError("limit must be the authorized smoke=1 or first-prefix=20")
    root, runs, sources, audit_rows = Path(run_root), {}, {}, []
    for task in TASKS:
        runs[task] = {}
        for variant in VARIANTS:
            path = root/task/variant/"records.jsonl"
            payload = path.read_bytes()
            rows = {}
            for line_no, line in enumerate(payload.decode().splitlines(keepends=True), 1):
                if not line.endswith("\n") or not line.strip():
                    raise ValueError(f"{path}:{line_no}: partial/empty JSONL line")
                row = json.loads(line)
                doc = _integer(row.get("doc_id"), "doc_id")
                if row.get("task") != task or row.get("status") != "completed" or not 0 <= doc < limit or doc in rows:
                    raise ValueError("Task/status/first-prefix coverage or duplicate doc_id mismatch")
                texts = [row[key] for key in ("prediction", "generated_text", "text") if key in row]
                if not texts or any(not isinstance(text, str) or text != texts[0] for text in texts):
                    raise ValueError("Require one unambiguous exact generated prediction string")
                if (not isinstance(row.get("input_hashes"), dict) or not row["input_hashes"]
                        or not isinstance(row.get("source_video_id"), str) or not row["source_video_id"]
                        or not isinstance(row.get("generated_ids"), list)
                        or any(type(token) is not int or token < 0 for token in row["generated_ids"])):
                    raise ValueError("Missing input hashes, source-video identity or generated token IDs")
                row["text"] = texts[0]
                config = row.get("config")
                if not isinstance(config, dict) or not config:
                    raise ValueError("Missing frozen run configuration")
                if config.get("strategy", config.get("mask_strategy")) != row["audit"].get("strategy"):
                    raise ValueError("Configuration and actual selector strategy disagree")
                detail = _audit(row, variant)
                audit_rows.append(dict(task=task, variant=variant, doc_id=doc, **detail))
                rows[doc] = row
            if set(rows) != set(range(limit)) or _sha(path) != hashlib.sha256(payload).hexdigest():
                raise ValueError("Incomplete first-prefix coverage or predictions changed during reading")
            state_path = path.parent/"run_state.json"
            if state_path.exists() and json.loads(state_path.read_text()).get("status") != "complete":
                raise ValueError("Score only completed generation runs")
            runs[task][variant] = [rows[doc] for doc in range(limit)]
            sources[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
        for doc in range(limit):
            reference = runs[task]["similarity"][doc]
            for variant in VARIANTS[1:]:
                row = runs[task][variant][doc]
                if row["input_hashes"] != reference["input_hashes"] or row["source_video_id"] != reference["source_video_id"]:
                    raise ValueError("Input tensor bytes/source video differ across paired conditions")
                paired = lambda config: {key: value for key, value in config.items() if key not in ("strategy", "mask_strategy", "mask_seed")}
                if paired(row["config"]) != paired(reference["config"]):
                    raise ValueError("Method/model/generation settings differ beyond the head selector")
                for field in ("baseline_masks", "baseline_selected_count", "expected_effective_merge_count", "expected_removed_tokens", "prefill_cache_shapes", "candidate_hashes"):
                    if row["audit"][field] != reference["audit"][field]:
                        raise ValueError(f"Paired selector audit differs before intervention: {field}")
        for variant in VARIANTS[1:]:
            masks = [row["audit"]["selected_masks"] for row in runs[task][variant]]
            if any(mask != masks[0] for mask in masks):
                raise ValueError("random_global must use the frozen model-level ranking across inputs")
    # The same model-level ranking yields nested prefixes for K=17 and K=34.
    for variant in VARIANTS[1:]:
        smaller = _mask(runs["vatex_test"][variant][0]["audit"]["selected_masks"], "VATEX random mask")
        larger = _mask(runs["nextqa_oe_test"][variant][0]["audit"]["selected_masks"], "NExTQA random mask")
        if any(small and not large for small, large in zip(smaller, larger)):
            raise ValueError("The two tasks do not share the same fixed model-level random ranking")
    return runs, dict(input_files_sha256=sources, audit_rows=audit_rows, limit=limit)


def _helper(repo_root):
    path = Path(repo_root)/"provenance/full_video_retest_20261104/helpers/score_full_runs.py"
    spec = importlib.util.spec_from_file_location("_kvrepack_versioned_offline_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


@contextmanager
def offline_resources(*, nltk_data, java_home):
    """Block Python socket/download calls and restore process-local state on exit."""
    from unittest.mock import patch

    def deny(*args, **kwargs):
        raise RuntimeError("Head-selection scoring is offline; networking/downloads are disabled")

    environment = dict(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NLTK_DATA=str(nltk_data))
    if java_home is not None:
        environment.update(JAVA_HOME=str(java_home), PATH=str(Path(java_home)/"bin")+os.pathsep+os.environ.get("PATH", ""))
    with patch.dict(os.environ, environment), patch.object(socket.socket, "connect", deny), patch.object(socket.socket, "connect_ex", deny), patch.object(socket, "create_connection", deny), patch.object(urllib.request, "urlopen", deny), patch.object(urllib.request, "urlretrieve", deny):
        import nltk
        with patch.object(nltk, "download", deny), patch.object(nltk.downloader.Downloader, "download", deny):
            yield


def _nextqa_namespace(helper, repo_root):
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from pywsd.utils import lemmatize_sentence
    import pandas as pd

    path = Path(repo_root)/"vendor/lmms-eval/lmms_eval/tasks/nextqa/utils.py"
    nodes, source = helper.function_spec(path, helper.NEXTQA_FUNCTIONS)
    namespace = dict(wordnet=wordnet, word_tokenize=word_tokenize, lemmatize_sentence=lemmatize_sentence,
                     stopwords=set(pd.read_csv(path.parent/"stopwords.csv").squeeze()))
    helper.execute_functions(nodes, path, namespace)
    namespace["remove_stop"] = functools.lru_cache(maxsize=None)(namespace["remove_stop"])
    return namespace, source


def _vatex_score(helper, path, nodes, rows, docs, temporary_dir, repo_root):
    if len({doc["videoID"] for doc in docs}) != len(docs):
        raise ValueError("Duplicate VATEX reference video IDs would overwrite corpus results")
    try:
        metrics, details = helper.vatex_score(path, nodes, rows, docs, Path(temporary_dir))
        return dict(metrics={key: metrics[key] for key in VATEX_METRICS}, missing_metrics={}, status="complete", scorer_details=details)
    except Exception as error:
        # Reuse the separately versioned, already audited three-metric helper.
        # It executes the same original task AST, replacing only the *unused*
        # Meteor constructor while keeping the real BLEU/ROUGE/CIDEr scorers.
        fallback_path = Path(repo_root)/"provenance/gpu_retest_20261104/score_gpu_subset.py"
        spec = importlib.util.spec_from_file_location("_kvrepack_offline_three_metric_fallback", fallback_path)
        fallback = importlib.util.module_from_spec(spec); spec.loader.exec_module(fallback)
        _, namespace, sources = fallback.setup_scorers(Path(repo_root), Path(temporary_dir))
        processed = [namespace["vatex_process_result"](doc, [row["text"]])["vatex_Bleu_4"] for row, doc in zip(rows, docs)]
        metrics = {metric: float(namespace["vatex_aggregation_result"](processed, metric)) for metric in VATEX_METRICS if metric != "METEOR"}
        return dict(status="partial_metrics", metrics=metrics,
                    missing_metrics={"METEOR": f"Full local scorer failed; three remaining original scorers succeeded: {type(error).__name__}: {error}"},
                    scorer_details=dict(fallback_path=str(fallback_path), fallback_sha256=_sha(fallback_path), sources=sources))


def score_runs(runs, *, hf_home, repo_root, nltk_data, java_home, temporary_dir):
    """Original local task scorers; references stay identical for all four runs."""
    helper, helper_path = _helper(repo_root)
    from ..observations.inputs import TASKS as DATA_TASKS

    scores, per_input, provenance = {}, {}, dict(helper_path=str(helper_path.resolve()), helper_sha256=_sha(helper_path), references={}, task_sources={})
    with offline_resources(nltk_data=nltk_data, java_home=java_home), redirect_stdout(io.StringIO()):
        nextqa, nextqa_source = _nextqa_namespace(helper, repo_root)
        provenance["task_sources"]["nextqa_oe_test"] = nextqa_source
        for task in TASKS:
            _, _, cache_name, arrow_name = DATA_TASKS[task]
            matches = list((Path(hf_home)/"datasets"/cache_name).glob(f"**/{arrow_name}"))
            if len(matches) != 1:
                raise ValueError(f"Require exactly one cached reference Arrow for {task}")
            count = len(runs[task]["similarity"])
            args = SimpleNamespace(reference_arrow=matches[0], task=task, expected_count=count)
            docs, reference = helper.reference_rows(args, runs[task]["similarity"])
            provenance["references"][task] = reference
            scores[task], per_input[task] = {}, {}
            if task == "vatex_test":
                path = Path(repo_root)/"vendor/lmms-eval/lmms_eval/tasks/vatex/utils.py"
                nodes, source = helper.function_spec(path, helper.VATEX_FUNCTIONS)
                provenance["task_sources"][task] = source
            for variant in VARIANTS:
                rows = runs[task][variant]
                for row, doc in zip(rows, docs):
                    metadata = row.get("reference_metadata", {})
                    if metadata.get("arrow_sha256", reference["sha256"]) != reference["sha256"]:
                        raise ValueError("Scoring Arrow differs from the actual model input reference")
                    digest = hashlib.sha256(_canonical(doc).encode()).hexdigest()
                    if metadata.get("original_row_sha256", digest) != digest:
                        raise ValueError("Scoring reference row differs from the actual model input row")
                if task == "nextqa_oe_test":
                    values = [nextqa["nextqa_process_results"](doc, [row["text"]])["WUPS"] for row, doc in zip(rows, docs)]
                    score = float(nextqa["nextqa_aggregate_results"](values))
                    individual = [100.*value["0"] for value in values]
                    if abs(score-statistics.fmean(individual)) > 1e-10:
                        raise ValueError("Per-input WUPS does not reproduce original task aggregate")
                    scores[task][variant] = dict(metrics={"WUPS": score}, missing_metrics={}, status="complete")
                    per_input[task][variant] = individual
                else:
                    scores[task][variant] = _vatex_score(helper, path, nodes, rows, docs, temporary_dir, repo_root)
        provenance["interpretation"] = "Unchanged original task AST; WUPS@0 in 0-100 points; VATEX in raw corpus units. CIDEr DF comes only from these selected references, never historical/full-corpus references."
        provenance["versions"] = {name: importlib.metadata.version(name) for name in ("nltk", "pywsd", "wn", "pandas", "pyarrow", "pycocoevalcap", "pycocotools")}
        distribution = importlib.metadata.distribution("pycocoevalcap")
        provenance["local_resources_sha256"] = {relative: _sha(distribution.locate_file(relative))
                                                for relative in ("pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar", "pycocoevalcap/meteor/meteor-1.5.jar", "pycocoevalcap/meteor/data/paraphrase-en.gz")
                                                if Path(distribution.locate_file(relative)).is_file()}
        provenance["cpu_offline_scope"] = "No model or GPU loaded; Python networking/downloads blocked; Java used only for installed local PTB/METEOR jars"
    return scores, per_input, provenance


def paired_wups_intervals(runs, per_input, *, samples=5000, seed=20261104):
    """Same source-video draws for original and all three seeds; no seed pooling."""
    import numpy as np

    _integer(samples, "bootstrap samples", 1)
    task = "nextqa_oe_test"
    originals = per_input[task]["similarity"]
    delta = np.asarray([[originals[i]-per_input[task][variant][i] for variant in VARIANTS[1:]] for i in range(len(originals))], dtype=np.float64)
    grouped = defaultdict(list)
    for index, row in enumerate(runs[task]["similarity"]):
        grouped[row["source_video_id"]].append(index)
    result = dict(n_inputs=len(originals), n_source_videos=len(grouped),
                  variants={variant: dict(difference=float(delta[:, index].mean()), ci95=None) for index, variant in enumerate(VARIANTS[1:])},
                  random_mean=dict(difference=float(delta.mean()), ci95=None),
                  estimand="Similarity minus random; bootstrap over source-video clusters, conditional on fixed seeds 0,1,2")
    if len(grouped) < 2:
        result["uncertainty"] = "CI unavailable: fewer than two independent source videos"
        return result
    counts = np.asarray([len(indices) for _, indices in sorted(grouped.items())], dtype=np.int64)
    totals = np.asarray([delta[indices].sum(axis=0) for _, indices in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, 4), dtype=np.float64)
    for start in range(0, samples, 256):
        size = min(256, samples-start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        values = totals[indices].sum(axis=1)/counts[indices].sum(axis=1)[:, None]
        draws[start:start+size, :3] = values
        draws[start:start+size, 3] = values.mean(axis=1)
    for index, variant in enumerate(VARIANTS[1:]):
        result["variants"][variant]["ci95"] = np.quantile(draws[:, index], [.025, .975]).tolist()
    result["random_mean"]["ci95"] = np.quantile(draws[:, 3], [.025, .975]).tolist()
    return result


def build_summary(runs, scores, per_input, *, audit, bootstrap_samples=5000, seed=20261104):
    rows, spread = [], []
    for variant in VARIANTS:
        values = per_input.get("nextqa_oe_test", {}).get(variant)
        if (not isinstance(values, list) or len(values) != audit["limit"]
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100 for value in values)):
            raise ValueError("Each condition requires every paired finite per-input WUPS value")
        if abs(statistics.fmean(values)-scores["nextqa_oe_test"][variant]["metrics"]["WUPS"]) > 1e-10:
            raise ValueError("Per-input WUPS values do not reproduce the recorded task score")
    for task in TASKS:
        expected = ("WUPS",) if task == "nextqa_oe_test" else VATEX_METRICS
        for variant in VARIANTS:
            result = scores[task][variant]
            for metric in expected:
                value = result["metrics"].get(metric)
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError("Nonfinite task metric")
                if value is None and metric not in result.get("missing_metrics", {}):
                    raise ValueError("Unreported task metric missing without an explicit reason")
                if value is None and metric != "METEOR":
                    raise ValueError("Required local task metric is unavailable")
                rows.append(dict(task=task, variant=variant, seed=None if variant == "similarity" else int(variant[-1]),
                                  metric=metric, score=value, missing_reason=result.get("missing_metrics", {}).get(metric),
                                  n_inputs=audit["limit"]))
        for metric in expected:
            original = scores[task]["similarity"]["metrics"].get(metric)
            random = [scores[task][variant]["metrics"].get(metric) for variant in VARIANTS[1:]]
            complete = original is not None and all(value is not None for value in random)
            mean = statistics.fmean(random) if complete else None
            spread.append(dict(task=task, metric=metric, original=original, random_seed0=random[0], random_seed1=random[1], random_seed2=random[2],
                               random_mean=mean, random_sample_std=statistics.stdev(random) if complete else None,
                               similarity_minus_random_mean=original-mean if complete else None, n_seeds=3, ci95=None))
    intervals = paired_wups_intervals(runs, per_input, samples=bootstrap_samples, seed=seed)
    next(row for row in spread if row["task"] == "nextqa_oe_test" and row["metric"] == "WUPS")["ci95"] = intervals["random_mean"]["ci95"]
    paired_rows = [dict(task="nextqa_oe_test", doc_id=index, source_video_id=row["source_video_id"], variant=variant,
                        similarity=per_input["nextqa_oe_test"]["similarity"][index], random=per_input["nextqa_oe_test"][variant][index],
                        difference=per_input["nextqa_oe_test"]["similarity"][index]-per_input["nextqa_oe_test"][variant][index])
                   for index, row in enumerate(runs["nextqa_oe_test"]["similarity"]) for variant in VARIANTS[1:]]
    return dict(schema_version=1, status="complete" if all(row["score"] is not None for row in rows) else "partial_metrics",
                experiment="snapkv_kvrepack_global_random_head_identity", limit=audit["limit"], seeds=[0, 1, 2],
                runs=rows, seed_summary=spread, paired_WUPS=intervals, paired_input_WUPS=paired_rows, audit=audit,
                bootstrap=dict(samples=bootstrap_samples, seed=seed, unit="source_video_id", pairing="Same input resampling for similarity and every seed"),
                interpretation="Only global head identities/ranking are changed; original parameters, total selected/effective heads, candidate bytes, merge work, and prefill cache shapes are paired. Random ranking is fixed per model/seed. Layer allocations can differ. Three seeds are not additional independent videos.",
                metric_scope="WUPS@0 is 0-100; VATEX uses raw corpus scores with the same selected reference pool. No sentence-BLEU or per-row CIDEr replacement; no CIDEr confidence interval or composite score is reported.")


def _csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.6))
        for axis, (task, metric) in zip(axes, (("nextqa_oe_test", "WUPS"), ("vatex_test", "CIDEr"))):
            data = next(row for row in summary["seed_summary"] if row["task"] == task and row["metric"] == metric)
            values = [data["original"], data["random_seed0"], data["random_seed1"], data["random_seed2"]]
            axis.plot(range(4), values, "o", color="#377EB8", markersize=5)
            axis.errorbar(4, data["random_mean"], yerr=data["random_sample_std"], fmt="s", color="#E68A2E", capsize=3)
            axis.set(xticks=range(5), xticklabels=["Similarity", "Seed 0", "Seed 1", "Seed 2", "Mean ± SD"], ylabel=metric,
                     title=f"{task}: n={summary['limit']}")
            axis.tick_params(axis="x", labelrotation=25)
            axis.spines[["top", "right"]].set_visible(False)
        fig.text(.5, .015, "Error bars: sample SD across three fixed rankings, not input confidence intervals.\nSame inputs, parameters, global selected heads and token merge work.", ha="center", fontsize=7.2)
        fig.subplots_adjust(left=.08, right=.985, bottom=.29, top=.88, wspace=.3)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output/f"task_scores.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        audits = summary["audit"]["audit_rows"]
        if audits:
            fig, axes = plt.subplots(2, 4, figsize=(9, 6.8), sharex=True, sharey=True)
            for r, task in enumerate(TASKS):
                for c, variant in enumerate(VARIANTS):
                    values = [row["selected_mask"] for row in audits if row["task"] == task and row["variant"] == variant]
                    frequency = np.asarray(values, dtype=float).mean(axis=0).reshape(28, 4)
                    artist = axes[r,c].imshow(frequency, vmin=0, vmax=1, aspect="auto", interpolation="nearest", cmap="Blues")
                    axes[r,c].set_title("Similarity" if variant == "similarity" else "Random seed "+variant[-1], fontsize=9)
                    axes[r,c].set(xticks=range(4), xlabel="Native KV head")
                    if c == 0:axes[r,c].set_ylabel(task+"\nDecoder layer")
            fig.subplots_adjust(left=.1, right=.9, bottom=.13, top=.91, wspace=.22, hspace=.42)
            fig.colorbar(artist, cax=fig.add_axes([.92,.2,.014,.6]), label="Selection frequency across inputs")
            fig.text(.5,.015,"Global random ranking is fixed per seed; total quotas match, while layer allocation can differ.",ha="center",fontsize=7.5)
            for extension in ("png", "pdf", "svg"):
                fig.savefig(output/f"head_selection.{extension}", dpi=300, bbox_inches="tight")
            plt.close(fig)


def write_report(summary, output, *, plots=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    _csv(output/"run_metrics.csv", summary["runs"], ["task", "variant", "seed", "metric", "score", "missing_reason", "n_inputs"])
    spread = [dict(row, ci95_low=row["ci95"][0] if row["ci95"] else None, ci95_high=row["ci95"][1] if row["ci95"] else None) for row in summary["seed_summary"]]
    _csv(output/"seed_summary.csv", spread, ["task", "metric", "original", "random_seed0", "random_seed1", "random_seed2", "random_mean", "random_sample_std", "similarity_minus_random_mean", "ci95_low", "ci95_high", "n_seeds"])
    _csv(output/"selector_audit.csv", summary["audit"]["audit_rows"], ["task", "variant", "doc_id", "selected_count", "effective_merge_count", "removed_tokens", "mask_hamming", "mask_jaccard"])
    _csv(output/"paired_input_WUPS.csv", summary["paired_input_WUPS"], ["task", "doc_id", "source_video_id", "variant", "similarity", "random", "difference"])
    lines = ["# Random-head identity ablation", "", f"Status: **{summary['status']}**; first {summary['limit']} original rows per task; seeds 0,1,2.", "", summary["interpretation"], "", summary["metric_scope"], "",
             "| Task | Metric | Similarity | Random seed 0 | Seed 1 | Seed 2 | Random mean ± sample SD | Similarity − random mean |", "|---|---|---:|---:|---:|---:|---|---|"]
    fmt = lambda value: "missing" if value is None else f"{value:.6f}"
    for row in spread:
        interval = f" [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]" if row["ci95_low"] is not None else ""
        lines.append(f"| {row['task']} | {row['metric']} | {fmt(row['original'])} | {fmt(row['random_seed0'])} | {fmt(row['random_seed1'])} | {fmt(row['random_seed2'])} | {fmt(row['random_mean'])} ± {fmt(row['random_sample_std'])} | {fmt(row['similarity_minus_random_mean'])}{interval} |")
    lines += ["", "Sample SD describes variability across exactly three fixed random rankings (ddof=1). WUPS intervals use paired source-video bootstrap, conditional on these three seeds; heads, seed×input rows and token pairs are not independent samples. VATEX corpus metric differences are descriptive.", ""]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if plots:
        _plots(summary, output)


def main(argv=None):
    runtime = str(Path(sys.prefix)/"lib")
    if argv is None and (Path(runtime)/"libstdc++.so.6").is_file() and os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0] != runtime:
        environment = dict(os.environ, LD_LIBRARY_PATH=runtime+os.pathsep+os.environ.get("LD_LIBRARY_PATH", ""))
        os.execve(sys.executable, [sys.executable, "-m", "kvrepack.ablations.head_selection_report", *sys.argv[1:]], environment)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(1, 20), default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--nltk-data", type=Path, default=Path(os.environ.get("NLTK_DATA", str(Path.home()/"nltk_data"))))
    bundled_java = Path(sys.prefix)/"lib/jvm"
    parser.add_argument("--java-home", type=Path, default=bundled_java if (bundled_java/"bin/java").is_file() else None)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    runs, audit = load_runs(args.run_root, limit=args.limit)
    with tempfile.TemporaryDirectory(prefix="head-score-", dir=args.run_root) as temporary:
        scores, per_input, provenance = score_runs(runs, hf_home=args.hf_home, repo_root=repo,
                                                   nltk_data=args.nltk_data, java_home=args.java_home, temporary_dir=temporary)
    summary = build_summary(runs, scores, per_input, audit=audit, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    summary["scorer_provenance"] = provenance
    write_report(summary, args.output, plots=not args.no_plots)
    print(json.dumps(dict(status=summary["status"], limit=args.limit, output=str(args.output))))
    return 0  # Missing optional METEOR is explicit; mandatory metric failures raise.


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Redraw saved Qwen2.5 reference plots without inference or recomputing statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from kvrepack.observations.reference_report import GROUPS, HEADS, LAYERS, METRICS, _plot


def redraw(report):
    report = report.expanduser().resolve()
    data_files = [report/"summary.json", *sorted(report.glob("*.csv"))]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}
    summary = json.loads((report/"summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("sample_count") not in (5, 200):
        raise ValueError("Require an existing complete fixed-input reference report")
    count = summary["sample_count"]
    with (report/"head_summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = [dict(group=r["group"], metric=r["metric"], layer=int(r["layer"]),
                     kv_head=int(r["kv_head"]), n_inputs=int(r["n_inputs"]), mean=float(r["mean"]))
                for r in csv.DictReader(stream)]
    expected = {(g, m, layer, head) for g in GROUPS for m in METRICS
                for layer in range(LAYERS) for head in range(HEADS)}
    identities = {(r["group"], r["metric"], r["layer"], r["kv_head"]) for r in rows}
    if identities != expected or len(rows) != len(expected) or any(r["n_inputs"] != count for r in rows):
        raise ValueError("Require all 672 unique layer/head/metric means at the saved input count")
    for group in GROUPS:
        for metric in METRICS:
            mean = statistics.fmean(r["mean"] for r in rows if r["group"] == group and r["metric"] == metric)
            if abs(mean-summary["groups"][group]["metrics"][metric]["mean"]) > 1e-12:
                raise ValueError("Saved head means disagree with the original group statistic")
    summary["head_table"] = rows
    plots = dict(main=_plot(summary, report), raw_supplement=_plot(summary, report, raw=True),
        key=_plot(summary, report, view="key"), value=_plot(summary, report, view="value"),
        combined=_plot(summary, report, view="combined"))
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}
    if before != after:
        raise ValueError("Replot changed an original statistics file")
    evidence = dict(status="passed", gpu_execution=False, sample_count_per_group=count,
        source_files_sha256={Path(p).name: digest for p, digest in before.items()},
        original_statistics_unchanged=True, plots=plots,
        plotting_source_sha256=hashlib.sha256((ROOT/"src/kvrepack/observations/reference_report.py").read_bytes()).hexdigest(),
        reference="MixKV Figure 1(b) distribution appearance; KDE is an explicit plotting choice, not a claimed reproduction of undisclosed author code.")
    (report/"continuous_plot_metadata.json").write_text(json.dumps(evidence, indent=2)+"\n")
    return dict(report=str(report), status=evidence["status"], sample_count_per_group=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for report in args.report:
        print(json.dumps(redraw(report)))


if __name__ == "__main__":
    main()

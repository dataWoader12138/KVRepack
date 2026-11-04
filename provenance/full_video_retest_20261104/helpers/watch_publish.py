#!/usr/bin/env python3
"""Publish compact stage-boundary evidence; default writes files, never pushes.

Use --once to render once; --watch --interval 30 to watch the external matrix.
Git writes require explicit --publish. Only the two owned report paths below
are staged/committed. No model/GPU calls, trace payload copies, pulls or rebases.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
DOC = "docs/FULL_VIDEO_RETEST_20261104.md"
EVIDENCE = "provenance/full_video_retest_20261104"
BASE = "169aa1f2e44d76d7205bd40b86bf08a6a7705b6d"
OPTIONAL_HISTORY = ("snap_history_reconstruction.json", "snap_history_recovered_nextqa200_metrics.json",
                    "snap_rounding_floor_3docs.json", "snap_rounding_ceil_3docs.json")


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    payload = value if isinstance(value, bytes) else (value.encode() if isinstance(value, str) else (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    if path.exists() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def git(repo, *args, check=True):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, timeout=55)
    if check and result.returncode:
        raise RuntimeError(f"git {args[0]} exited {result.returncode}: {result.stderr.strip()[-2000:]}")
    return result


def candidate_cli(work, name, expected):
    directory = work / name
    state = load(directory / "run_state.json", {})
    results = sorted(directory.glob("eval_outputs/**/*results.json"))
    result = load(results[-1], {}) if results else {}
    score = result.get("results", {}).get("nextqa_oe_test", {}).get("WUPS,none")
    count = result.get("n-samples", {}).get("nextqa_oe_test", {}).get("effective")
    status = "passed" if state.get("returncode") == 0 and score is not None and count == expected else ("failed" if state.get("returncode") not in (None, 0) else ("incomplete" if state.get("returncode") == 0 else ("running" if state else "not_started")))
    payload = {"status": status, "scope": "candidate-only ordinary CLI; no paired original equality proof",
               "expected_count": expected, "effective_count": count, "WUPS": score,
               "run_state_path": str(directory / "run_state.json"),
               "algorithm_env": load(directory / "resolved_config.json", {}).get("algorithm_env", {})}
    if results:
        payload.update(result_path=str(results[-1]), result_sha256=sha(results[-1]))
    return payload


def planned_jobs(plan, state):
    known = {job["key"]: job for job in state.get("stages", [])}
    rows = []
    counts = state.get("dataset_split_counts", {"nextqa_oe_test": 9178, "vatex_test": 4478})
    for methods, full in ((plan.get("subset_methods", []), False), (plan.get("full_methods", []), True)):
        for method in methods:
            for task, total in counts.items():
                count = total if full else min(plan["subset_limit"], total)
                key = f"{method}:{task}:{count}"
                rows.append(known.pop(key, {"key": key, "method": method, "task": task, "count": count,
                                           "full_split": count == total, "status": "pending"}))
    return rows + list(known.values())


def freeze_summary(work):
    path = work / "asset_manifest.json"
    ledger = load(path, {})
    selected = [{"category": row["category"], "path": row["path"], "sha256": row.get("sha256"), "bytes": row.get("stat", {}).get("size")}
                for row in ledger.get("files", []) if row["category"] == "dataset_arrow" or
                (row["category"] == "model_snapshot_all_files" and row["path"].endswith(".safetensors"))]
    return {"ledger_path": str(path), "ledger_sha256": sha(path) if path.exists() else None,
            "created_utc": ledger.get("created_utc"), "summary": ledger.get("summary"),
            "model_snapshot": ledger.get("model_snapshot"), "coverage": ledger.get("coverage"),
            "weight_and_arrow_hashes": selected,
            "selected_hashes_aggregate_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
            "ledger_copied": False, "videos": "Existence/nonzero/stat metadata only, no video payload hashes or decoding proof; no complete dependency closure claim"}


def historical_budget_evidence(work, repo, jobs, history, candidates):
    """Preserve narrow historical observations without changing test semantics."""
    log = Path("/tmp/run_nextqa_vatexbest_decouplekv_20260704.log")
    pairs, previous = [], None
    if log.exists():
        # Binary iteration counts LF physical lines; progress-bar CR characters
        # give different line numbers in str.splitlines(). Never copy the log.
        with log.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if text.startswith("Token Length: "):
                    previous = (number, text, int(text.split(":", 1)[1]))
                if previous and "[SnapKV-DecoupleKV] adjacent-sim global gate:" in text:
                    observed = {12567: 1257, 9621: 963}.get(previous[2])
                    if observed and f"final_history={observed} " in text and not any(p["seq_len"] == previous[2] for p in pairs):
                        pairs.append({"seq_len": previous[2], "budget_ratio": 0.1, "historical_final_history": observed,
                                      "current_int_budget": int(previous[2] * 0.1),
                                      "log_lines": [{"line": previous[0], "text": previous[1]}, {"line": number, "text": text}]})
                        if len(pairs) == 2:
                            break
    sources = []
    for path in (work.parent / "HybridKV-artifact/hybridkv/qwen_model_snapkv_decouplekv.py",
                 repo / "src/kvrepack/qwen_model_snapkv_kvrepack.py"):
        if path.exists():
            text = path.read_text()
            function = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == "_history_budget")
            sources.append({"path": str(path), "sha256": sha(path), "function": function.name,
                            "line_start": function.lineno, "line_end": function.end_lineno,
                            "excerpt": ast.get_source_segment(text, function)})
    snap = next((job for job in jobs if job["method"] == "snapkv_kvrepack" and job["task"] == "nextqa_oe_test" and job["count"] == 200 and job["status"] == "passed"), {})
    historical = {row["method"]: row for row in history}
    ordinary_mada = candidates.get("cli_nextqa_madakv_200", {})
    official = []
    for implementation in ("original", "repack"):
        for path in sorted((work / f"snapkv_nextqa_{implementation}").glob("attempt_*/eval_outputs/**/*results.json")):
            result = load(path, {})
            if result.get("n-samples", {}).get("nextqa_oe_test", {}).get("effective") == 200:
                official.append({"implementation": implementation, "path": str(path), "sha256": sha(path),
                                 "WUPS": result.get("results", {}).get("nextqa_oe_test", {}).get("WUPS,none")})
    excerpt_hash = hashlib.sha256(json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    paired_wups = (snap.get("metrics") or {}).get("WUPS")
    return {"status": "confirmed_observed_budget_difference" if len(pairs) == 2 else "historical_excerpt_unavailable",
            "historical_snap_result_id": "P02", "historical_snap_WUPS": historical.get("snapkv_kvrepack", {}).get("score_raw"),
            "current_snap_paired_WUPS": paired_wups, "current_official_cli_results": official,
            "current_official_offline_WUPS_agree": paired_wups is not None and len(official) >= 2 and all(row["WUPS"] == paired_wups for row in official),
            "historical_log": {"path": str(log), "line_number_convention": "LF physical lines; CR progress bars count differently under str.splitlines()", "excerpt_only": True, "selected_excerpt_sha256": excerpt_hash},
            "budget_examples": pairs, "current_source_functions": sources,
            "interpretation": "Old log budgets match ceiling for these examples; current original and migrated code use int(seq_len*ratio), yielding one fewer history token. This is a confirmed older-run versus current-original behavior difference, not a migration-introduced difference. It does not prove the cause of the entire WUPS gap.",
            "alpha_evidence": "The selected historical Snap log rows explicitly report alpha=2.0000; current paired config also sets alpha=2. No missing-alpha explanation is established.",
            "pending_checks": "Historical task-source/video/input identity may also differ and remains under investigation; no historical per-sample equivalence claim.",
            "ordinary_mada_context": {"historical_result_id": "P04", "historical_WUPS": historical.get("madakv_kvrepack", {}).get("score_raw"),
                                      "candidate_WUPS": ordinary_mada.get("WUPS"), "candidate_status": ordinary_mada.get("status"),
                                      "aggregate_equal": ordinary_mada.get("status") == "passed" and ordinary_mada.get("WUPS") == historical.get("madakv_kvrepack", {}).get("score_raw"),
                                      "limitation": "Exact aggregate agreement only; ordinary candidate CLI has no paired original E2E proof or historical per-sample identity proof"}}


def display(value):
    return "—" if value is None else str(value)


def optional_history(work):
    items = {}
    for name in OPTIONAL_HISTORY:
        path = work / name
        item = {"source_path": str(path), "status": "pending"}
        if path.exists() and path.stat().st_size <= 500_000:
            try:
                payload = path.read_bytes()
                data = json.loads(payload)
                ready = (data.get("n") == 200 and len(data.get("records", [])) == 200 if name == OPTIONAL_HISTORY[0] else
                         data.get("status") == "complete" and data.get("n") == 200 if name == OPTIONAL_HISTORY[1] else
                         data.get("diagnostic_only") is True and data.get("production_source_modified") is False and len(data.get("records", [])) == 3)
                if ready:
                    item.update(status="ready", sha256=hashlib.sha256(payload).hexdigest(), payload=payload, data=data)
            except (OSError, ValueError):
                pass  # A diagnostic can still be writing its final JSON.
        elif path.exists():
            item["status"] = "oversize_not_copied"
        items[name] = item
    return items


def history_diagnostics(evidence, optional):
    copied = {}
    for name, item in optional.items():
        copied[name] = {key: value for key, value in item.items() if key not in {"payload", "data"}}
        if item["status"] == "ready":
            atomic(evidence / "historical_diagnostics" / name, item["payload"])
            copied[name]["repository_path"] = f"{EVIDENCE}/historical_diagnostics/{name}"
    reconstruction = optional[OPTIONAL_HISTORY[0]].get("data", {})
    rescoring = optional[OPTIONAL_HISTORY[1]].get("data", {})
    floor = optional[OPTIONAL_HISTORY[2]].get("data", {})
    ceil = optional[OPTIONAL_HISTORY[3]].get("data", {})
    floor_rows, ceil_rows = ({row["doc_id"]: row for row in data.get("records", [])} for data in (floor, ceil))
    comparisons = [{"doc_id": doc_id, "input_hashes_equal": floor_rows[doc_id]["input_hashes"] == ceil_rows[doc_id]["input_hashes"],
                    "floor_input_matches_production": floor_rows[doc_id].get("input_matches_production"),
                    "ceil_input_matches_production": ceil_rows[doc_id].get("input_matches_production"),
                    "floor_matches_production_text": floor_rows[doc_id].get("matches_production_text"),
                    "ceil_matches_historical_text": ceil_rows[doc_id].get("matches_historical_text"),
                    "floor_ceil_text_equal": floor_rows[doc_id]["text"] == ceil_rows[doc_id]["text"]}
                   for doc_id in sorted(floor_rows.keys() & ceil_rows.keys())]
    old_wups = reconstruction.get("rescoring", {}).get("historical_P02_WUPS")
    return {"artifacts": copied, "historical_mapping_200": reconstruction.get("summary"),
            "recovered_old_text_WUPS": rescoring.get("metrics", {}).get("WUPS"),
            "rescoring_matches_P02_exactly": old_wups is not None and rescoring.get("metrics", {}).get("WUPS") == old_wups,
            "reconstruction_rescoring": reconstruction.get("rescoring"), "historical_tensor_or_token_id_equality_proved": False,
            "rounding_diagnostic": {"scope": "Only the three previously changed docs; not a full 200-row ceil replay",
                                    "floor_status": optional[OPTIONAL_HISTORY[2]]["status"], "ceil_status": optional[OPTIONAL_HISTORY[3]]["status"],
                                    "comparisons": comparisons},
            "limitations": ["Video/question and input-length matches do not establish historical pixel/input tensors, generation token IDs, logits or KV identity.",
                            "Recovered historical stdout text can be rescored exactly, without proving complete historical execution equivalence.",
                            "The external floor/ceil diagnostic changes only its process-local helper and is not a production source or stage-parameter change."]}


def render(work, repo, state, plan, candidates, optional=None):
    jobs = planned_jobs(plan, state)
    evidence = repo / EVIDENCE
    artifacts, helpers, trace_hashes = [], {}, {}
    # This function runs only at stage/status boundaries, never per-doc polling.
    for job in jobs:
        if job["status"] != "passed":
            continue
        total = state.get("dataset_split_counts", {"nextqa_oe_test": 9178, "vatex_test": 4478})[job["task"]]
        if job["full_split"] != (job["count"] == total):
            raise ValueError(f"Stage count/full-split claim mismatch: {job['key']}")
        paths = [job.get("comparison"), *job.get("metrics_files", {}).values()]
        for raw in filter(None, paths):
            source = Path(raw).resolve()
            if not source.is_relative_to(work) or source.suffix != ".json" or source.stat().st_size > 8 * 1024 * 1024:
                raise ValueError(f"Unexpected completed artifact: {source}")
            digest = sha(source)
            if digest != job.get("artifacts_sha256", {}).get(str(source)):
                raise ValueError(f"Completed artifact hash mismatch: {source}")
            target = evidence / "completed" / source.name
            atomic(target, source.read_bytes())
            artifacts.append({"stage": job["key"], "source_path": str(source), "sha256": digest, "repository_path": str(target.relative_to(repo))})
        comparison = load(Path(job["comparison"]))
        if (comparison.get("status") != "passed" or comparison.get("expected_count_per_side") != job["count"]
                or comparison.get("declared_full_count_per_side") != total
                or bool(comparison.get("full_benchmark_validated")) != job["full_split"]):
            raise ValueError(f"Comparison count/scope mismatch: {job['key']}")
        for raw in job.get("metrics_files", {}).values():
            metrics = load(Path(raw))
            if (metrics.get("status") != "complete" or metrics.get("n") != job["count"]
                    or metrics.get("task") != job["task"] or metrics.get("metrics") != job["metrics"]):
                raise ValueError(f"Completed metrics mismatch: {raw}")
        for side in ("reference", "candidate"):
            exported = comparison.get(side, {})
            raw = exported.get("path")
            trace_key = f"{job['key']}:{side}"
            if raw and trace_key not in trace_hashes:
                path = Path(raw).resolve()
                if not path.is_relative_to(work):
                    raise ValueError("Trace path escaped work directory")
                # The comparison already hashes its finite byte snapshot. Keep
                # that stage-specific proof even when a shared trace later grows.
                trace_hashes[trace_key] = {"path": raw, "snapshot_sha256": exported.get("scanned_prefix_sha256"),
                                     "snapshot_size_bytes": exported.get("snapshot_size_bytes"),
                                     "source_stage": job["key"], "copied": False,
                                     "note": "Comparison's finite full-file snapshot SHA; shared trace may grow on full-split resume"}
    selected_helpers = {Path(path).name: Path(path) for path in state.get("helper_sha256", {})}
    selected_helpers.update({"watch_publish.py": Path(__file__).resolve(), "matrix_plan.json": work / "matrix_plan.json"})
    if (work / "check_snap_rounding.py").exists():
        selected_helpers["check_snap_rounding.py"] = work / "check_snap_rounding.py"
    for name, source in sorted(selected_helpers.items()):
        if not source.resolve().is_relative_to(work) or source.suffix not in (".py", ".json"):
            raise ValueError(f"Unexpected helper snapshot path: {source}")
        digest = sha(source)
        expected = state.get("helper_sha256", {}).get(str(source))
        if expected and expected != digest:
            raise ValueError(f"Frozen helper changed: {source}")
        atomic(evidence / "helpers" / name, source.read_bytes())
        helpers[name] = {"source_path": str(source), "sha256": digest, "driver_frozen": expected is not None}
    assets = freeze_summary(work)
    atomic(evidence / "frozen_assets_summary.json", assets)
    history_path = repo / "provenance/video_best_parameters.json"
    history = [{key: row.get(key) for key in ("id", "dataset", "model", "method", "simtop", "cr", "alpha", "score_raw", "metric", "generation_limit", "evidence")}
               for row in load(history_path, {}).get("main_rows", []) if row.get("model") == "Qwen2.5-VL-3B" and row.get("source_task") == "nextqa_oe_test" and row.get("method") in {"snapkv_kvrepack", "madakv_kvrepack"}]
    historical_budget = historical_budget_evidence(work, repo, jobs, history, candidates)
    atomic(evidence / "historical_budget_evidence.json", historical_budget)
    diagnostics = history_diagnostics(evidence, optional if optional is not None else optional_history(work))
    if diagnostics["historical_mapping_200"]:
        historical_budget["pending_checks"] = "All200 historical video/question IDs uniquely mapped and input token lengths matched; historical video content/pixel tensors, generation token IDs, logits, KV, and complete task/dependency identities remain unproved."
        atomic(evidence / "historical_budget_evidence.json", historical_budget)
    unchanged = git(repo, "diff", "--quiet", BASE, "--", "src", "vendor", "csrc", "requirements", "pyproject.toml", check=False).returncode == 0
    summary = {"schema_version": 1, "updated_utc": state.get("updated_utc"), "matrix_status": state.get("status", "waiting"),
               "base_commit": BASE, "core_unchanged_from_base": unchanged, "matrix_plan": plan, "stages": jobs,
               "current_snapshot": state.get("current"), "error": state.get("error"), "candidate_only_cli": candidates,
               "completed_artifacts": artifacts, "trace_snapshot_references": list(trace_hashes.values()), "helper_snapshots": helpers,
               "frozen_assets_summary": "frozen_assets_summary.json", "historical_context_only": history,
               "historical_budget_evidence": historical_budget,
               "historical_reconstruction_and_rounding": diagnostics,
               "historical_source_sha256": sha(history_path) if history_path.exists() else None,
               "limitations": ["Only the explicitly listed Qwen3B methods/tasks are planned; no all-methods claim.",
                               "Original control substitutes original compression into the current runner/adapter/tasks; it is not an untouched historical frontend rerun.",
                               "Exact logits/KV equality uses exported full-byte SHA-256; no max_abs_error was measured in these traced runs.",
                               "200 docs are a first-doc subset, not full NextQA9178/VATEX4478; only passed full_split stages validate the full split.",
                               "Historical aggregates are context only: matching aggregate scores would not prove historical per-sample identity.",
                               "Trace transfer/hashing invalidates timing/performance benchmarks; videos are metadata-monitored and dependency coverage is incomplete."]}
    atomic(evidence / "summary.json", summary)
    lines = ["# Qwen3B 视频复测与后台进度（2026-11-04）", "", f"状态：**{summary['matrix_status']}**；阶段快照 UTC：{summary['updated_utc']}。", "",
             "本页仅覆盖下表计划。pending/running 均未通过；200 条是子集，完整 NextQA、VATEX 分别为9178、4478条。阶段变化时更新，逐样本实时计数请查看本机 matrix_state.json。", "",
             "| 方法 | 任务 | 样本数 | 范围 | 状态 | 实际评分 |", "|---|---|---:|---|---|---|"]
    for job in jobs:
        scores = "; ".join(f"{key}={display(value)}" for key, value in (job.get("metrics") or {}).items()) or "—"
        scope = "完整split" if job["full_split"] else f"前{job['count']}条子集"
        lines.append(f"| {job['method']} | {job['task']} | {job['count']} | {scope} | {job['status']} | {scores} |")
    lines += ["", "配对两端使用同一当前 CLI/adapter、模型快照、BF16、64帧、batch=1、greedy 和 budget_ratio=0.1；原版端仅替换为原压缩源码。Snap参数为simtop=0.7/cr=0.9/alpha=2；Mada为simtop=0.5/alpha=2，NextQA使用历史最佳cr=0.5，VATEX用cr=0.7；correction=none。", "",
              "通过阶段核对实际输入tensor、生成token/text、每步完整词表logits及36层prefill/final KV的SHA-256。此轮是哈希相等检查，未测max_abs_error；追踪开销使其不能作为性能测试。NextQA为任务原WUPS（0–100），VATEX为真实BLEU-4/METEOR/ROUGE-L/CIDEr及四项原始小数均值mean4；CIDEr随参考子集变化，子集和全量不能混比。", "",
              f"核心相对冻结基准`{BASE[:7]}`未变化：**{unchanged}**。本报告生成器只维护本页与专属provenance目录。", "", "## 普通 CLI 与历史上下文", ""]
    for name, candidate in candidates.items():
        lines.append(f"- {name}：{candidate['status']}，有效样本{display(candidate['effective_count'])}，WUPS={display(candidate['WUPS'])}。仅candidate普通CLI结果，没有配对原版E2E相等证明。")
    for row in history:
        lines.append(f"- 历史200条 {row['method']} WUPS={row['score_raw']}，cr={row['cr']}；仅作旧汇总分上下文，即使同分也不能证明历史逐样本一致。")
    lines += ["", "普通Snap20正式CLI的已完成WUPS为28.003499816653978；它不是200条或全量结果。普通Mada200只在其run_state成功且结果有效计数为200时标passed。", "", "## 历史Snap预算差异（已确认，原因仍有边界）", "",
              f"当前Snap200原版/迁移版配对WUPS={display(historical_budget['current_snap_paired_WUPS'])}，历史P02为{display(historical_budget['historical_snap_WUPS'])}；当前两端官方CLI与离线评分相等：{historical_budget['current_official_offline_WUPS_agree']}。旧日志实际记录12567 token→final_history1257、9621→963，均符合向上取整；当前原版`_history_budget`使用`int(seq_len*ratio)`，对应1256、962，迁移版保留同一规则。选中旧日志也明确alpha=2.0000，当前为alpha=2。", "",
              "这是**旧实验与当前原实现之间已确认的预算行为差异，不是此次迁移引入的差异**。现有证据不能证明它解释全部WUPS分差；历史任务源码、视频或输入身份也可能不同，仍待核对。未调整当前算法、预算规则或阶段参数。", "",
              f"普通Mada200 candidate WUPS={display(historical_budget['ordinary_mada_context']['candidate_WUPS'])}，与历史P04={display(historical_budget['ordinary_mada_context']['historical_WUPS'])}的汇总相等状态为{historical_budget['ordinary_mada_context']['aggregate_equal']}；这不证明历史逐样本一致，也不是普通Mada200的原版配对E2E验证。", "",
              f"[预算短摘录与源码函数证据](../{EVIDENCE}/historical_budget_evidence.json)仅保存两组旧日志行及当前原版/迁移版函数片段，不保存大日志。", "", "## 证据与边界", "",
              f"[阶段汇总](../{EVIDENCE}/summary.json)、[冻结资源摘要](../{EVIDENCE}/frozen_assets_summary.json)及同目录completed/、helpers/保存完成对比、真实评分和运行助手。权重、视频、完整trace JSONL、logits均不上传，仅保留本机路径、SHA和快照大小。", "",
              f"冻结ledger记录{(assets.get('summary') or {}).get('file_count', '未知')}个文件；冻结完成后摘要列出两个模型权重和两个Arrow SHA，并保留完整ledger的SHA。资源初始字节哈希后按stat/软链接目标监测；视频仅存在/非零/stat检查，不是视频内容哈希、完整解码验证或全部依赖闭包。", "", "## 复查与恢复", "", "在已准备同版本模型、数据和依赖的工作目录运行；helpers快照保留本次具体逻辑，本机路径需按新机器调整。", "", "```bash",
              f"cd {work}", "cat matrix_state.json", "cat matrix_report.md", "# 仅在没有现存driver进程时恢复；复用已有trace并重新验证冻结资源", "python -B -u run_matrix.py --plan matrix_plan.json", "# 复查完整NextQA；VATEX改expected-count为4478并换文件路径", "python compare_full_runs.py --reference snapkv_nextqa_original/trace.jsonl --candidate snapkv_nextqa_repack/trace.jsonl --expected-count 9178 --output check_nextqa_full.json", "# 复查前200条须额外传 --limit 200，完整expected-count仍为9178", "python score_full_runs.py --records snapkv_nextqa_original/trace.jsonl --task nextqa_oe_test --expected-count 9178 --output check_nextqa_metrics.json --repo-root ../KVRepack", "# 后台汇总；--publish为显式可选的Git提交/推送授权开关", "python watch_publish.py --watch --interval 30", "```", ""]
    mapping = diagnostics["historical_mapping_200"]
    if mapping:
        lines += ["## 历史200条文本重建与三条取整诊断", "",
                  f"从旧stdout恢复的200条中，视频/问题唯一映射{mapping.get('unique_video_question_matches')}/200，输入token长度相等{mapping.get('input_token_lengths_equal')}/200；旧预算符合ceil共{mapping.get('old_budget_equals_ceil')}条，当前符合floor共{mapping.get('current_budget_equals_floor')}条，其中{mapping.get('budget_smaller_by_one')}条当前预算少1。答案文本相同{mapping.get('answers_equal')}条，不同doc_id为{mapping.get('changed_doc_ids')}。", "",
                  f"恢复旧文本按当前真实scorer重算WUPS={display(diagnostics['recovered_old_text_WUPS'])}，与旧P02精确相等：{diagnostics['rescoring_matches_P02_exactly']}。这确认了恢复文本及分数差，不能证明历史输入tensor、生成token IDs、logits或KV一致。", ""]
    rounding = diagnostics["rounding_diagnostic"]
    if rounding["floor_status"] == "ready" or rounding["ceil_status"] == "ready":
        lines += [f"独立外部诊断：floor={rounding['floor_status']}，ceil={rounding['ceil_status']}。仅针对已发现差异的doc61/117/131，生产源码和matrix阶段参数均未修改；**不是完整200条ceil重跑**。", ""]
        for row in rounding["comparisons"]:
            lines.append(f"- doc{row['doc_id']}：两诊断输入hash相等={row['input_hashes_equal']}；floor匹配当前文本={row['floor_matches_production_text']}；ceil匹配旧文本={row['ceil_matches_historical_text']}。")
        lines.append("")
    if mapping or rounding["comparisons"]:
        lines += [f"小型源JSON及SHA见[summary](../{EVIDENCE}/summary.json)中的historical_reconstruction_and_rounding及historical_diagnostics/；不上传恢复预测JSONL或大日志。", ""]
    if summary["error"]:
        lines += ["矩阵错误：", "", "```text", str(summary["error"]), "```", ""]
    atomic(repo / DOC, "\n".join(lines))
    return summary


def publish(repo, status, retry=False):
    if git(repo, "branch", "--show-current").stdout.strip() != "main":
        raise ValueError("Publisher requires current branch main")
    owned = [DOC, EVIDENCE]
    git(repo, "add", "--", *owned)
    changed = git(repo, "diff", "--cached", "--quiet", "--", *owned, check=False).returncode
    if changed not in (0, 1):
        raise RuntimeError("Cannot inspect staged owned report changes")
    if changed:
        git(repo, "commit", "--only", "-m", f"Record video retest stages ({status})", "--", *owned)
    elif not retry:
        return {"no_owned_changes": True, "publish_pending": False}
    # Also retry a prior rejected push even when its local commit has no diff.
    git(repo, "push", "origin", "main")
    return {"published_commit": git(repo, "rev-parse", "HEAD").stdout.strip(), "publish_pending": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--interval", type=float, default=30)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least one second")
    work = args.work_dir.resolve()
    repo = args.repo.resolve() if args.repo else work.parent / "KVRepack"
    local_status = work / "watch_publish_state.json"
    previous_signature = None
    pending_push = bool(args.publish and load(local_status, {}).get("publish_pending"))
    retry_push = pending_push
    while True:
        try:
            state = load(work / "matrix_state.json", {"status": "waiting"})
            plan = load(work / "matrix_plan.json", {})
            candidates = {name: candidate_cli(work, name, count) for name, count in (("cli_nextqa_snapkv_smoke20", 20), ("cli_nextqa_madakv_200", 200))}
            optional = optional_history(work)
            signature = json.dumps([state.get("status"), [(job["key"], job["status"]) for job in state.get("stages", [])], state.get("error"), plan, {key: row["status"] for key, row in candidates.items()}, {name: (item['status'], item.get('sha256')) for name, item in optional.items()}], sort_keys=True)
            if signature != previous_signature:
                render(work, repo, state, plan, candidates, optional)
                previous_signature, pending_push = signature, args.publish
                atomic(local_status, {"updated_utc": datetime.now(timezone.utc).isoformat(), "matrix_status": state.get("status"), "report_written": True, "publish_requested": args.publish, "publish_pending": pending_push})
            if pending_push:
                try:
                    publication = publish(repo, state.get("status", "waiting"), retry=retry_push)
                    pending_push = False
                    retry_push = False
                    atomic(local_status, {"updated_utc": datetime.now(timezone.utc).isoformat(), "matrix_status": state.get("status"), **publication})
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    retry_push = True
                    atomic(local_status, {"updated_utc": datetime.now(timezone.utc).isoformat(), "matrix_status": state.get("status"), "publish_error": str(error), "publish_pending": True, "gpu_driver_untouched": True})
                    print(f"Publication failed; matrix continues independently: {error}", file=sys.stderr, flush=True)
            if not args.watch or state.get("status") in {"complete", "failed", "interrupted"}:
                return 1 if pending_push or state.get("status") in {"failed", "interrupted"} else 0
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
            atomic(local_status, {"updated_utc": datetime.now(timezone.utc).isoformat(), "report_error": str(error), "gpu_driver_untouched": True})
            print(f"Report update failed; matrix continues independently: {error}", file=sys.stderr, flush=True)
            if not args.watch:
                return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

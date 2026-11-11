#!/usr/bin/env python3
"""Independent, stdlib-only audit of recorded VATEX merge executions.

This does not import the worker, torch or Transformers. An optional reference
directory is used ONLY to match input fingerprints. Explicitly reused parent
LookM/AsymKV rows are checked verbatim against their stopped-run source chain;
the old KVRepack baseline is auxiliary evidence, never a new 200-row result.
The sole mutation is writing the explicitly requested independent audit JSON.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

METHODS = ("snapkv_kvrepack", "hybridkv_kvrepack")
ARMS = ("kvrepack", "lookm", "asymkv")
MERGE_ONLY_ARMS = ("lookm", "asymkv")
POLICY = "native_no_grad_prefix_cache_raw_leaves_text_suffix_backward_v2"
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            result.update(piece)
    return result.hexdigest()


def canonical_sha(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def load(path):
    return json.loads(Path(path).read_text())


def ast_equal(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def qualified_function(path, name):
    source = Path(path).read_text()
    nodes = ast.parse(source).body
    for part in name.split("."):
        found = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        require(len(found) == 1, "Missing/ambiguous source definition: " + name)
        node = found[0]
        nodes = node.body
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    text = textwrap.dedent("".join(source.splitlines(keepends=True)[start-1:node.end_lineno]))
    return text, ast.parse(text).body[0]


def assignment_name(node):
    return node.targets[0].id if (isinstance(node, ast.Assign) and len(node.targets) == 1
                                and isinstance(node.targets[0], ast.Name)) else None


def expr_call(node, name):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == name)


def anchor_tail(node):
    for i, item in enumerate(node.body):
        if assignment_name(item) == "anchor_pos":
            return node.body[i:]
    return None


def validate_merge_ast(original, changed, method, arm):
    """Independently prove native group prefix and anchor arithmetic survived."""
    hook = "_fixed_audit" if method == "snapkv_kvrepack" else "_merge_group_record"
    hook_indices = [i for i,n in enumerate(changed.body) if expr_call(n, hook)]
    require(len(hook_indices) == 1, "Expected one lightweight group-record insertion")
    hook_at = hook_indices[0]
    if method == "snapkv_kvrepack":
        boundary = next(i for i,n in enumerate(original.body) if assignment_name(n) == "raw_sum")
        raw_anchor = anchor_tail(original)
        new_anchor = anchor_tail(changed)
        require(raw_anchor is not None and new_anchor is not None, "Missing native anchor mRoPE")
        # Return cardinality can change for AsymKV; all anchor arithmetic stays.
        require(ast_equal(ast.Module(body=raw_anchor[:-1], type_ignores=[]),
                          ast.Module(body=new_anchor[:-1], type_ignores=[])), "Anchor mRoPE block drift")
        if arm != "kvrepack":
            reduce_calls = [n for n in ast.walk(changed) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name) and n.func.id == "_fixed_reduce"]
            require(len(reduce_calls) == 1 and isinstance(reduce_calls[0].args[0], ast.Constant)
                    and reduce_calls[0].args[0].value == arm, "Wrong transplanted reducer")
    else:
        boundary = next(i for i,n in enumerate(original.body) if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "self.hybrid_asymkv")
        raw_branch = next(n for n in original.body[boundary:] if isinstance(n, ast.If)
                          and ast.unparse(n.test) == "use_raw_reapply")
        raw_anchor = anchor_tail(raw_branch)
        if arm != "kvrepack":
            new_anchor = anchor_tail(changed)
            require(raw_anchor is not None and new_anchor is not None
                    and ast_equal(ast.Module(body=raw_anchor, type_ignores=[]),
                                  ast.Module(body=new_anchor[:-1], type_ignores=[])), "Anchor mRoPE block drift")
    require(ast_equal(ast.Module(body=original.body[:boundary], type_ignores=[]),
                      ast.Module(body=changed.body[:hook_at], type_ignores=[])), "Group construction prefix drift")
    if arm == "kvrepack":
        cleaned = copy.deepcopy(changed)
        cleaned.body.pop(hook_at)
        require(ast_equal(original, cleaned), "Native KVRepack arithmetic was changed")


def validate_transforms(case, method, repo_root, arms=ARMS):
    """Hash real original functions and inspect saved executable transformed AST."""
    if method == "snapkv_kvrepack":
        specs = {"merger": ("qwen_model_snapkv_kvrepack.py", "_paired_merge_one_sequence"),
                 "finalizer": ("qwen_model_snapkv_kvrepack.py", "_finalize_prompt_cache"),
                 "decode": ("qwen_model.py", "qwen_flash_attn2_forward_SnapKV")}
        merge_name = "merger"
    else:
        specs = {"merge": ("hybridkv_utils.py", "HybridKVCache._hybrid_kvrepack_merge_one_sequence"),
                 "attention": ("modeling_qwen2_5_vl.py", "Qwen2_5_VLFlashAttention2.forward")}
        merge_name = "merge"
    results, original_hashes = {}, []
    for arm in arms:
        path = case / ("transform_" + arm + ".json")
        document = load(path)
        expected_keys = {merge_name} | ((set(specs)-{merge_name}) if arm == "asymkv" else set())
        require(set(document) == expected_keys, "Missing/extra recorded source transformations")
        results[arm] = dict(path=str(path), sha256=file_sha(path), functions={})
        for name, spec in document.items():
            original_text, original_ast = qualified_function(repo_root/"src/kvrepack"/specs[name][0], specs[name][1])
            original_hash = spec.get("original_source_sha256", spec.get("original_sha256"))
            compiled_hash = spec.get("transformed_source_sha256", spec.get("transformed_sha256"))
            text = spec["transformed_source"]
            require(digest_bytes(original_text.encode()) == original_hash, "Original function source SHA drift: " + name)
            require(digest_bytes(text.encode()) == compiled_hash, "Compiled function source SHA drift: " + name)
            if "original_source" in spec:
                require(spec["original_source"] == original_text, "Original function text drift")
            tree = ast.parse(text)
            require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), "Invalid compiled source")
            changed = tree.body[0]
            if name == merge_name:
                validate_merge_ast(original_ast, changed, method, arm)
                original_hashes.append(original_hash)
            if name == "decode":
                # Reverse precisely the extra callback arguments/name and prove
                # the original projection/RoPE/cache/decode body is unchanged.
                calls = [n for n in ast.walk(changed) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "_fixed_decode_attention"]
                require(len(calls) == 1, "Missing AsymKV decode readout")
                calls[0].func.id = "_flash_attention_forward"
                require([ast.unparse(n) for n in calls[0].args[:2]] == ["self", "past_key_value"], "Decode callback context drift")
                calls[0].args = calls[0].args[2:]
                require(ast_equal(original_ast, changed), "Decode changes beyond LMA readout")
            results[arm]["functions"][name] = dict(original_sha256=original_hash,
                compiled_sha256=compiled_hash, grouping_and_anchor_checked=name == merge_name)
    require(len(set(original_hashes)) == 1, "Aggregation arms used different original merger sources")
    return results


def check_fingerprint(value, shape=None, dtype=None):
    require(isinstance(value, dict) and re.fullmatch("[0-9a-f]{64}", value.get("sha256", "")) is not None,
            "Invalid recorded tensor hash")
    require(isinstance(value.get("shape"), list) and all(type(n) is int and n >= 0 for n in value["shape"]), "Invalid tensor shape")
    if shape is not None:
        require(value["shape"] == shape, "Recorded tensor shape mismatch")
    if dtype is not None:
        require(value.get("dtype") == dtype, "Recorded tensor dtype mismatch")


def independent_signature(audit, method):
    require(audit.get("status") == "complete" and audit.get("merge_calls"), "Missing real merge/group evidence")
    groups = []
    for call in audit["merge_calls"]:
        before, after = call["positions"]["shape"][0], call["group_size"]["shape"][0]
        require(0 < after < before, "Merge group lengths are not compressive")
        fields = {name:call[name] for name in ("positions", "group_ids", "group_size", "start_mask")}
        for name, value in fields.items():
            check_fingerprint(value, [after] if name == "group_size" else [before],
                              "torch.float32" if name == "group_size" else "torch.bool" if name == "start_mask" else "torch.int64")
        groups.append(fields)
    if method == "snapkv_kvrepack":
        require(len(audit["layers"]) == len(audit["selected_masks"]) == len(audit["prefill_cache_shapes"]) == 28,
                "Missing Snap layer/candidate evidence")
        layers = []
        for index, layer in enumerate(audit["layers"]):
            require(layer["layer"] == index, "Snap layer order differs")
            candidate = layer.get("candidate_idx", layer.get("candidates", {}).get("candidate_idx"))
            check_fingerprint(candidate, [1, 4, layer["candidate_history"]], "torch.int64")
            check_fingerprint(audit["selected_masks"][index], [1, 4], "torch.bool")
            layers.append(dict(layer=index, candidate_idx=candidate, final_history=layer["final_history"],
                               candidate_history=layer["candidate_history"]))
        require(audit.get("actual_merge_calls") == len(groups), "Snap merge-call count differs")
        return dict(groups=groups, selected_masks=audit["selected_masks"], layers=layers,
                    prefill_shapes=audit["prefill_cache_shapes"])
    check_fingerprint(audit["merge_mask"], [28, 1, 4], "torch.bool")
    identities = []
    for call in audit["merge_calls"]:
        identity = {k:call[k] for k in ("layer_idx", "head_idx", "batch_idx", "before", "after")}
        require(0 <= identity["layer_idx"] < 28 and 0 <= identity["head_idx"] < 4
                and identity["batch_idx"] == 0, "Hybrid native head identity differs")
        require(identity["before"] == call["positions"]["shape"][0]
                and identity["after"] == call["group_size"]["shape"][0], "Hybrid group size metadata differs")
        identities.append(identity)
    require(len({(x["layer_idx"], x["head_idx"], x["batch_idx"]) for x in identities}) == len(identities), "Hybrid duplicated merge head")
    return dict(groups=groups, identities=identities, merge_mask=audit["merge_mask"], budget=audit["budget"])


def validate_budget(row, method):
    n, tokens = row["input_tokens"], row["generated_tokens"]
    require(type(n) is int and n > 0, "Invalid input length")
    cache = row["budget"]
    shapes = cache["final_shapes"]
    require(len(shapes) == 28, "Missing final cache layers")
    slots = 0
    for layer, shape in enumerate(shapes):
        k, v = shape["key_shape"], shape["value_shape"]
        require(shape["layer"] == layer and k == v and k[-1] == 128, "Invalid final K/V shape")
        require(all(type(x) is int and x > 0 for x in k), "Invalid cache dimensions")
        require((method == "snapkv_kvrepack" and len(k) == 4 and k[:2] == [1, 4])
                or (method == "hybridkv_kvrepack" and len(k) == 2), "Wrong native cache layout")
        slots += math.prod(k[:-1])
    retained = slots-112*(tokens-1)
    limit = 112*math.floor(n*.1)
    require(0 < retained <= limit, "Budget exceeds strict 10% including recent positions")
    require(cache["prefill_tokens_total"] == retained and cache["capacity"] == limit, "Recorded budget does not match physical cache")
    require(math.isclose(cache["actual_retained_ratio"], retained/(112*n), rel_tol=0, abs_tol=1e-12), "Retained ratio differs")
    if method == "snapkv_kvrepack":
        require(retained == limit, "Snap uniform budget underfilled")
        for layer in row["audit"]["layers"]:
            require(layer["final_history"] == math.floor(n*.1)-32, "Snap recent window not included in budget")
    else:
        meta = row["audit"]["budget"]
        require(meta["sequence_length"] == n and meta["local_len"] == 32 and meta["budget_ratio"] == .1,
                "Hybrid budget protocol differs")
        budgets = meta["history_budget"]
        require(len(budgets) == 28 and all(len(x) == 1 and len(x[0]) == 4 for x in budgets), "Hybrid head budget shape differs")
        history = sum(sum(x[0]) for x in budgets)
        require(history+112*32 == retained == meta["active_prefill_positions"]
                and meta["capacity_positions"] == limit, "Hybrid local/history budget mismatch")
    return retained, limit


def validate_fisher(row, contract):
    stats = row.get("fisher")
    if row["variant"] != "asymkv":
        require(stats is None and row["runtime"]["fisher_seconds"] == 0, "Unexpected Fisher branch on ordinary merger")
        return
    require(isinstance(stats, dict), "Missing Fisher; fallback aggregation is forbidden")
    require(contract.get("fisher_probe_policy") == POLICY == stats.get("probe_policy"), "Fisher probe policy differs")
    require(stats.get("gradient_space") == "pre_mrope_k_proj"
            and stats.get("objective") == "next_token_cross_entropy_on_observed_prompt_text_tail", "Wrong Fisher key space/objective")
    require(stats.get("layers") == 28 and stats.get("gradient_shapes") == [[1, 4, row["input_tokens"], 128]]*28,
            "Fisher gradient shapes differ from complete raw prompt")
    require(stats.get("loss_tail_tokens_requested") == 32 and 0 < stats.get("loss_target_tokens", 0) <= 32,
            "Fisher target suffix length differs")
    require(stats.get("prefix_tokens", -1) + stats.get("backward_suffix_tokens", -1) == row["input_tokens"], "Fisher prompt token coverage differs")
    require(math.isfinite(stats.get("loss", math.nan)) and stats["loss"] >= 0
            and row["runtime"]["fisher_seconds"] > 0, "Invalid Fisher objective/runtime")
    for name in ("native_attention_forward_sha256", "native_text_forward_sha256"):
        require(re.fullmatch("[0-9a-f]{64}", stats.get(name, "")) is not None, "Missing native Fisher source SHA")


def contract_arms(contract):
    arms = tuple(contract.get("variants", ARMS))
    require(arms in (ARMS, MERGE_ONLY_ARMS), "Unsupported aggregation arm scope")
    return arms


def validate_reused_rows(bundle, parent_bundle, parent_contract):
    """Validate immutable selected rows; parent baseline is evidence only."""
    require(contract_arms(parent_contract) == ARMS, "Reuse parent is not the preserved three-arm run")
    require(bundle.get("method") == parent_bundle.get("method")
            and bundle.get("doc_id") == parent_bundle.get("doc_id")
            and bundle.get("task") == parent_bundle.get("task"), "Reuse parent sample identity differs")
    validate_bundle(parent_bundle, parent_contract)
    require(set(bundle["records"]) == set(MERGE_ONLY_ARMS), "Baseline must not be migrated into merge-only results")
    for arm in MERGE_ONLY_ARMS:
        require(bundle["records"][arm] == parent_bundle["records"][arm],
                "Reused parent row changed: " + arm)
    order = [arm for arm in parent_bundle["order"] if arm in MERGE_ONLY_ARMS]
    require(bundle.get("order") == order, "Reused sample must preserve measured parent arm order")
    return order


def validate_bundle(bundle, contract, *, eos_token_id=151645, vocab_size=152064, expected_order=None):
    """Pure validation entry point, also used by tampering regression tests."""
    doc_id, method = bundle["doc_id"], contract["method"]
    require(bundle.get("status") == "completed" and bundle.get("task") == "vatex_test"
            and bundle.get("method") == method and bundle.get("contract_sha256") == canonical_sha(contract), "Bundle identity/contract mismatch")
    arms = contract_arms(contract)
    require(set(bundle["records"]) == set(arms), "Missing/extra aggregation arm")
    default_order = list(arms[doc_id%len(arms):]+arms[:doc_id%len(arms)])
    require(bundle.get("order") == (expected_order if expected_order is not None else default_order), "Arm rotation differs")
    signatures, retentions = [], []
    reference = bundle["records"][arms[0]]
    for arm in arms:
        row = bundle["records"][arm]
        require(row.get("status") == "completed" and row.get("variant") == arm and row.get("doc_id") == doc_id
                and row.get("task") == "vatex_test" and row.get("method") == method and row.get("warmup") is False,
                "Wrong arm/sample identity or warmup contamination")
        require(row.get("text") == row.get("prediction") and isinstance(row.get("text"), str), "Prediction text differs")
        ids = row.get("generated_ids")
        require(isinstance(ids, list) and 0 < len(ids) <= 64 and all(type(x) is int and 0 <= x < vocab_size for x in ids), "Invalid generated token IDs")
        require(row.get("generated_tokens") == len(ids) and eos_token_id not in ids[:-1], "Generated count/early EOS differs")
        eos = ids[-1] == eos_token_id
        capped = len(ids) == 64 and not eos
        require(row.get("eos_reached") is eos and row.get("capped_without_eos") is capped
                and (eos or capped), "EOS/cap termination flags differ")
        require(row["reference_metadata"] == reference["reference_metadata"]
                and row["input_fingerprints"] == reference["input_fingerprints"], "Input fingerprints or source metadata differ")
        require(row["reference_metadata"]["doc_id"] == doc_id and row["reference_metadata"]["input_tokens"] == row["input_tokens"], "Source row/input length differs")
        require(row["input_fingerprints"]["input_shapes"]["input_ids"] == [1, row["input_tokens"]], "Input tensor shape differs")
        runtime = row["runtime"]
        fields = ("preprocessing_seconds", "h2d_seconds", "fisher_seconds", "generation_seconds",
                  "text_decode_seconds", "model_end_to_end_seconds", "end_to_end_seconds", "peak_allocated_gib")
        require(all(isinstance(runtime.get(k), (float, int)) and math.isfinite(runtime[k]) and runtime[k] >= 0 for k in fields), "Invalid runtime")
        parts = sum(runtime[k] for k in fields[:5])
        require(abs(parts-runtime["end_to_end_seconds"]) <= .01, "E2E timer decomposition differs")
        require(abs(runtime["fisher_seconds"]+runtime["generation_seconds"]-runtime["model_end_to_end_seconds"]) <= .01, "Model E2E timer decomposition differs")
        validate_fisher(row, contract)
        retentions.append(validate_budget(row, method))
        signatures.append(independent_signature(row["audit"], method))
    require(all(x == signatures[0] for x in signatures[1:]), "Recorded candidates/heads/groups/anchors differ across arms")
    require(len(set(retentions)) == 1, "Final budgets differ across arms")
    # Host-specific invariant: these two implementations still compute the
    # prompt attention from the original full K/V. Their merged cache first
    # affects subsequent decode, so the first generated token must agree.
    # This is a necessary state-contamination check, not a general property of
    # arbitrary compression methods or proof that all model state is intact.
    first_ids = {bundle["records"][arm]["generated_ids"][0] for arm in arms}
    require(len(first_ids) == 1, "Host prefill first generated token differs across arms")
    fairness = bundle.get("fairness", {})
    require(fairness.get("status") == "passed" and fairness.get("signature_sha256") == canonical_sha(signatures[0]),
            "Worker fairness evidence disagrees with independent group signature")
    return dict(doc_id=doc_id, input_tokens=reference["input_tokens"], prefill_positions=retentions[0][0],
                capacity_positions=retentions[0][1], signature_sha256=canonical_sha(signatures[0]),
                first_generated_id=next(iter(first_ids)), first_generated_id_matched=True,
                eos={v:bundle["records"][v]["eos_reached"] for v in arms},
                capped={v:bundle["records"][v]["capped_without_eos"] for v in arms})


MIGRATION_CONFIG_FIELDS = (
    "method", "task", "target_docs", "frames", "min_pixels", "max_pixels", "model_snapshot", "hf_home",
    "budget_ratio", "k_space", "budget_policy", "algorithm_env", "max_new_tokens", "repetition_penalty",
    "do_sample", "dtype", "model_seed", "arrow_sha256", "dataset_rows", "task_yaml_sha256", "task_utils_sha256",
    "warmup_per_arm", "diagnostic_audit", "fisher_prompt_tail_tokens", "fisher_probe_policy",
    "model_metadata_sha256", "versions", "timing_scope",
)


def validate_parent_prefix(prefix, contract, repo_root):
    """Freeze the stopped protocol via snapshots, never resume its worker."""
    if prefix is None:
        return None
    require(contract_arms(contract) == MERGE_ONLY_ARMS, "Parent reuse only belongs to merge-only scope")
    repo_root = Path(repo_root).resolve()
    case = Path(prefix["case"]).resolve()
    run_root = Path(prefix["run_root"]).resolve()
    require(case == run_root/contract["method"], "Parent case/method path differs")
    contract_path = Path(prefix["parent_contract_path"]).resolve()
    require(contract_path == case/"run_contract.json", "Parent contract path differs")
    old = load(contract_path)
    require(canonical_sha(old) == prefix["parent_contract_sha256"]
            and file_sha(contract_path) == prefix["parent_contract_file_sha256"], "Parent contract identity changed")
    require(contract_arms(old) == ARMS, "Parent contract scope differs")
    stopped = load(run_root/"experiment_state.json")
    require(stopped.get("status") == "superseded_by_user_scope" and stopped.get("resume_authorized") is False,
            "Parent protocol is not explicitly stopped and superseded")
    state = load(case/"run_state.json")
    require(state.get("resume_authorized") is False, "Parent worker can still resume")
    n = prefix["completed_docs"]
    files = sorted((case/"samples").glob("*.json"))
    require(type(n) is int and 0 < n <= 200 and [p.name for p in files] == [f"{i:06d}.json" for i in range(n)],
            "Parent completed prefix coverage changed")
    require(state.get("completed_docs") == n, "Parent prefix counter differs")
    snapshot_path = Path(prefix["source_snapshot_path"]).resolve()
    require(file_sha(snapshot_path) == prefix["source_snapshot_sha256"], "Parent source snapshot manifest changed")
    snapshot = load(snapshot_path)
    require(Path(snapshot.get("parent_run_root", "")).resolve() == run_root
            and snapshot.get("parent_completed_docs") == n, "Parent source snapshot scope differs")
    recorded = snapshot["files"]
    for relative, entry in recorded.items():
        path = (repo_root/entry["path"]).resolve()
        require(path.is_relative_to(repo_root) and file_sha(path) == entry["sha256"], "Immutable parent source snapshot changed: " + relative)
    for relative, old_sha in old["source_sha256"].items():
        require(relative in recorded and recorded[relative]["sha256"] == old_sha,
                "Parent source snapshot does not cover original contract: " + relative)
        if relative != "src/kvrepack/ablations/run_merge_comparison.py":
            require(contract["source_sha256"].get(relative) == old_sha, "Algorithm source changed between parent and merge-only scope")
    for name in MIGRATION_CONFIG_FIELDS:
        require(contract.get(name) == old.get(name), "Parent/new experiment configuration differs: " + name)
    # Only orchestration/arm count changes are allowed in the worker. The
    # actual input, budget, grouping signature and timed execution functions
    # must have exactly the same AST as in the immutable stopped source.
    worker = "src/kvrepack/ablations/run_merge_comparison.py"
    old_tree = ast.parse((repo_root/recorded[worker]["path"]).read_text())
    new_tree = ast.parse((repo_root/worker).read_text())
    function_hashes = {}
    for name in ("prepare_input", "active_prefill", "pairing_signature", "one"):
        left = [x for x in ast.walk(old_tree) if isinstance(x, ast.FunctionDef) and x.name == name]
        right = [x for x in ast.walk(new_tree) if isinstance(x, ast.FunctionDef) and x.name == name]
        require(len(left) == len(right) == 1 and ast_equal(left[0], right[0]), "Timed/algorithm worker function changed during migration: " + name)
        function_hashes[name] = digest_bytes(ast.dump(left[0], include_attributes=False).encode())
    return dict(prefix=prefix, contract=old, case=case, run_root=run_root, source_snapshot=snapshot,
                unchanged_worker_function_ast_sha256=function_hashes)


def validate_parent_bundle(bundle, context):
    """Resolve recorded lineage and prove both reused row dictionaries intact."""
    prefix = context["prefix"]
    doc_id = bundle["doc_id"]
    require(0 <= doc_id < prefix["completed_docs"], "Reuse outside committed parent prefix")
    origin = bundle.get("reused_from_parent")
    require(isinstance(origin, dict), "Missing parent lineage on reused prefix row")
    path = context["case"]/"samples"/f"{doc_id:06d}.json"
    require(Path(origin.get("parent_bundle_path", "")).resolve() == path
            and origin.get("parent_bundle_sha256") == file_sha(path), "Parent bundle path/SHA changed")
    for name in ("parent_contract_path", "parent_contract_sha256", "parent_contract_file_sha256",
                 "source_snapshot_path", "source_snapshot_sha256"):
        require(origin.get(name) == prefix[name], "Parent lineage metadata differs: " + name)
    parent = load(path)
    require(bundle.get("original_parent_order") == parent.get("order"), "Original parent measured order missing/changed")
    return validate_reused_rows(bundle, parent, context["contract"])


def audit_run(run_root, repo_root=ROOT, *, allow_partial=False, reference_inputs=None,
              require_postprocess_complete=False):
    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    controller = load(run_root/"experiment_state.json")
    total_generations = controller.get("target_generations")
    require(total_generations in (800, 1200), "Unexpected controller generation count")
    arms = MERGE_ONLY_ARMS if total_generations == 800 else ARMS
    require(controller.get("methods") == list(METHODS) and controller.get("docs_per_arm") == 200
            and total_generations == 400*len(arms), "Controller experiment scope differs")
    require(controller.get("status") not in ("failed", "stopped_by_user"), "Controller failed/stopped")
    if not allow_partial:
        require(controller.get("status") == "complete", "Controller has not completed the matrix")
        require(set(controller.get("completed_methods", [])) == set(METHODS), "Controller completion list differs")
    result = dict(schema_version=1, auditor="independent_stdlib_v1", audited_utc=datetime.now(timezone.utc).isoformat(),
        run_root=str(run_root), repo_root=str(repo_root), allow_partial=allow_partial,
        controller_state=controller, variants=list(arms), methods={}, completed_docs=0, completed_generations=0,
        reused_generations=0, fresh_generations=0,
        reference_input_policy="Input fingerprint matching only; old predictions/scores/timing are never read into results",
        parent_reuse_policy="Only explicitly chained LookM/AsymKV rows may be reused verbatim; parent KVRepack baseline is auxiliary evidence and excluded from new arm totals",
        limitations=["Checks recorded tensor hashes, frozen source and AST transformations; does not replay model/GPU arithmetic.",
                     "HybridKV budget covers active cache positions, excluding backing/offload storage and Fisher metadata.",
                     "First-token equality is specific to these hosts' full-K/V prefill attention; it is not a generic compression invariant or sufficient proof of unchanged model state."])
    for method in METHODS:
        case = run_root/method
        required = ("run_contract.json", "run_state.json", "environment.json")
        if not all((case/name).is_file() for name in required):
            require(allow_partial, "Missing required backbone: " + method)
            result["methods"][method] = dict(status="initializing" if case.is_dir() else "not_started", completed_docs=0)
            continue
        contract, state, env = (load(case/name) for name in ("run_contract.json", "run_state.json", "environment.json"))
        require(contract.get("method") == method and contract.get("task") == "vatex_test"
                and contract.get("target_docs") == 200 and contract.get("target_generations") == 200*len(arms)
                and contract.get("variants") == list(arms) and contract.get("diagnostic_audit") is False,
                "Wrong formal run scope")
        require(contract.get("frames") == 64 and contract.get("budget_ratio") == .1
                and contract.get("k_space") == "raw_reapply" and contract.get("max_new_tokens") == 64
                and contract.get("fisher_probe_policy") == POLICY, "Model/generation/merge protocol differs")
        require(state.get("contract_sha256") == canonical_sha(contract), "Worker contract SHA differs")
        require(state.get("status") not in ("failed", "stopped_by_user"), "Worker failed/stopped")
        sources = contract.get("source_sha256", {})
        require(bool(sources), "Missing frozen source hashes")
        for relative, sha in sources.items():
            source = (repo_root/relative).resolve()
            require(source.is_relative_to(repo_root) and file_sha(source) == sha, "Frozen source SHA differs: " + relative)
        for filename, sha in contract.get("model_metadata_sha256", {}).items():
            require(file_sha(Path(contract["model_snapshot"])/filename) == sha, "Model metadata SHA differs: " + filename)
        kwargs = env["generation_kwargs"]
        require(kwargs.get("eos_token_id") == 151645 and kwargs.get("max_new_tokens") == 64
                and kwargs.get("do_sample") is False and kwargs.get("num_beams") == 1, "Generation environment differs")
        transformations = validate_transforms(case, method, repo_root, arms)
        parent = validate_parent_prefix(contract.get("parent_prefix"), contract, repo_root)
        if arms == MERGE_ONLY_ARMS:
            require((method == "snapkv_kvrepack" and parent is not None and parent["prefix"]["completed_docs"] == 114)
                    or (method == "hybridkv_kvrepack" and parent is None), "Merge-only scope must reuse exactly the committed Snap114 parent prefix")
        files = sorted((case/"samples").glob("*.json"))
        require([p.name for p in files] == [f"{i:06d}.json" for i in range(len(files))]
                and len(files) <= 200, "Committed samples are not contiguous original prefix")
        if not allow_partial:
            require(len(files) == state.get("completed_docs") == 200 and state.get("status") == "complete"
                    and state.get("completed_generations") == 200*len(arms), "Full audit requires 200 complete rows per backbone")
        rows, fingerprints, reused_rows = [], [], 0
        for index, path in enumerate(files):
            bundle = load(path)
            require(bundle.get("doc_id") == index, "Filename/doc_id mismatch")
            expected_order = None
            reused = parent is not None and index < parent["prefix"]["completed_docs"]
            if reused:
                expected_order = validate_parent_bundle(bundle, parent)
                reused_rows += 1
            else:
                require(not bundle.get("reused_from_parent"), "Unvalidated reuse lineage outside authorized prefix")
            rows.append({**validate_bundle(bundle, contract, eos_token_id=kwargs["eos_token_id"], expected_order=expected_order),
                         "origin":"reused_from_parent" if reused else "fresh"})
            fp = bundle["records"][arms[0]]["input_fingerprints"]
            fingerprints.append(canonical_sha(fp))
            if reference_inputs is not None:
                old = load(Path(reference_inputs)/path.name)
                require(old.get("doc_id") == index and old.get("task") == "vatex_test"
                        and old.get("input_fingerprints") == fp, "Reference input fingerprints differ")
        result["methods"][method] = dict(status="complete" if len(rows) == 200 else "verified_prefix",
            completed_docs=len(rows), worker_state=state, source_files_verified=len(sources),
            reused_generations=reused_rows*len(arms), fresh_generations=(len(rows)-reused_rows)*len(arms),
            contract_sha256=canonical_sha(contract), contract_file_sha256=file_sha(case/"run_contract.json"),
            transformations=transformations, rows=rows, input_fingerprint_sha256=fingerprints)
        if parent is not None:
            result["methods"][method]["parent_lineage"] = dict(**parent["prefix"],
                unchanged_worker_function_ast_sha256=parent["unchanged_worker_function_ast_sha256"],
                baseline_auxiliary_prefix_docs=parent["prefix"]["completed_docs"], baseline_counted_as_new_result=False)
        result["completed_docs"] += len(rows)
        result["completed_generations"] += len(rows)*len(arms)
        result["reused_generations"] += reused_rows*len(arms)
        result["fresh_generations"] += (len(rows)-reused_rows)*len(arms)
    available = [v["input_fingerprint_sha256"] for v in result["methods"].values() if "input_fingerprint_sha256" in v]
    if len(available) == 2:
        count = min(map(len, available))
        require(available[0][:count] == available[1][:count], "Backbones used different input fingerprints")
        result["cross_backbone_matched_docs"] = count
    post_path = run_root/"postprocess_state.json"
    post = load(post_path) if post_path.exists() else dict(status="absent")
    require(post.get("status") != "failed", "Postprocess failed")
    if require_postprocess_complete:
        require(post.get("status") == "complete", "Postprocess has not completed")
    result["postprocess_state"] = post
    complete = (result["completed_docs"] == 400 and controller.get("status") == "complete"
                and all(item.get("worker_state", {}).get("status") == "complete" for item in result["methods"].values()))
    result["status"] = "passed_complete" if complete else "passed_partial"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--reference-inputs", type=Path)
    parser.add_argument("--require-postprocess-complete", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.name not in {"run_contract.json", "run_state.json", "environment.json", "experiment_state.json", "postprocess_state.json"}
            and not any(part == "samples" for part in output.parts), "Audit output must not replace recorded generation data")
    try:
        result = audit_run(args.run_root, args.repo_root, allow_partial=args.allow_partial,
                           reference_inputs=args.reference_inputs, require_postprocess_complete=args.require_postprocess_complete)
    except Exception as error:
        result = dict(status="failed", auditor="independent_stdlib_v1", error=f"{type(error).__name__}: {error}",
                      run_root=str(args.run_root.resolve()), audited_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status", "completed_docs", "completed_generations", "error") if k in result}))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())

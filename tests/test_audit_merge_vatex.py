"""Independent auditor tampering checks; standard library only, no GPU/model."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_merge_auditor", ROOT/"scripts/audit_merge_vatex.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fingerprint(shape, dtype):
    return dict(shape=shape, dtype=dtype, sha256="a"*64)


def fixture():
    contract = dict(method="snapkv_kvrepack", fisher_probe_policy=audit.POLICY, max_new_tokens=64)
    group = dict(positions=fingerprint([98], "torch.int64"), group_ids=fingerprint([98], "torch.int64"),
                 group_size=fingerprint([68], "torch.float32"), start_mask=fingerprint([98], "torch.bool"))
    grouping = dict(status="complete", variant="kvrepack", merge_calls=[copy.deepcopy(group) for _ in range(17)],
        actual_merge_calls=17, layers=[dict(layer=i, final_history=68, candidate_history=98,
            candidate_idx=fingerprint([1, 4, 98], "torch.int64")) for i in range(28)],
        selected_masks=[fingerprint([1, 4], "torch.bool") for _ in range(28)],
        prefill_cache_shapes=[[1, 4, 100, 128] for _ in range(28)])
    runtime = dict(preprocessing_seconds=.3, h2d_seconds=.01, fisher_seconds=0.,
                   generation_seconds=.5, text_decode_seconds=.001, model_end_to_end_seconds=.5,
                   end_to_end_seconds=.811, peak_allocated_gib=18.)
    row = dict(status="completed", task="vatex_test", method="snapkv_kvrepack", variant="kvrepack", doc_id=0,
        text="Example caption.", prediction="Example caption.", warmup=False,
        generated_ids=[42, 151645], generated_tokens=2, input_tokens=1000,
        eos_reached=True, capped_without_eos=False,
        reference_metadata=dict(doc_id=0, input_tokens=1000),
        input_fingerprints=dict(input_shapes=dict(input_ids=[1, 1000]), input_hashes=dict(input_ids="b"*64)),
        runtime=runtime, audit=grouping, fisher=None,
        budget=dict(prefill_tokens_total=11200, capacity=11200, actual_retained_ratio=.1,
            final_shapes=[dict(layer=i, key_shape=[1, 4, 101, 128], value_shape=[1, 4, 101, 128]) for i in range(28)]))
    rows = {arm:copy.deepcopy(row) for arm in audit.ARMS}
    for arm, record in rows.items():
        record["variant"] = record["audit"]["variant"] = arm
    asym = rows["asymkv"]
    asym["fisher"] = dict(gradient_space="pre_mrope_k_proj", objective="next_token_cross_entropy_on_observed_prompt_text_tail",
        layers=28, gradient_shapes=[[1, 4, 1000, 128]]*28, loss_tail_tokens_requested=32, loss_target_tokens=32,
        prefix_tokens=967, backward_suffix_tokens=33, loss=5., probe_policy=audit.POLICY,
        native_attention_forward_sha256="d"*64, native_text_forward_sha256="e"*64)
    asym["runtime"].update(fisher_seconds=.2, model_end_to_end_seconds=.7, end_to_end_seconds=1.011)
    signature = audit.independent_signature(rows["kvrepack"]["audit"], contract["method"])
    bundle = dict(status="completed", task="vatex_test", method="snapkv_kvrepack", doc_id=0,
        contract_sha256=audit.canonical_sha(contract), order=list(audit.ARMS), records=rows,
        fairness=dict(status="passed", signature_sha256=audit.canonical_sha(signature)))
    return contract, bundle


class IndependentMergeAuditTests(unittest.TestCase):
    def test_valid_synthetic_bundle_passes_without_model_dependencies(self):
        contract, bundle = fixture()
        result = audit.validate_bundle(bundle, contract)
        self.assertEqual(result["prefill_positions"], 11200)
        self.assertEqual(result["capacity_positions"], 11200)
        self.assertEqual(result["first_generated_id"], 42)
        self.assertTrue(result["first_generated_id_matched"])

    def test_two_arm_fresh_bundle_excludes_baseline_and_uses_rotate_two(self):
        contract, parent = fixture()
        new = copy.deepcopy(contract)
        new["variants"] = list(audit.MERGE_ONLY_ARMS)
        child = copy.deepcopy(parent)
        child["records"].pop("kvrepack")
        child["order"] = list(audit.MERGE_ONLY_ARMS)
        child["contract_sha256"] = audit.canonical_sha(new)
        result = audit.validate_bundle(child, new)
        self.assertEqual(set(result["eos"]), set(audit.MERGE_ONLY_ARMS))
        with self.assertRaisesRegex(ValueError, "Missing/extra"):
            child["records"]["kvrepack"] = parent["records"]["kvrepack"]
            audit.validate_bundle(child, new)

    def test_parent_two_arm_rows_and_original_measured_order_are_immutable(self):
        contract, parent = fixture()
        # Row1's old rotate-three order filters to LookM/Asym, while fresh
        # rotate-two would be Asym/LookM. Reuse must preserve its real order.
        parent["doc_id"] = 1
        parent["order"] = ["lookm", "asymkv", "kvrepack"]
        for row in parent["records"].values():
            row["doc_id"] = row["reference_metadata"]["doc_id"] = 1
        child_contract = {**contract, "variants":list(audit.MERGE_ONLY_ARMS)}
        child = copy.deepcopy(parent)
        child["records"].pop("kvrepack")
        child["order"] = ["lookm", "asymkv"]
        child["contract_sha256"] = audit.canonical_sha(child_contract)
        order = audit.validate_reused_rows(child, parent, contract)
        audit.validate_bundle(child, child_contract, expected_order=order)
        with self.assertRaisesRegex(ValueError, "Arm rotation"):
            audit.validate_bundle(child, child_contract)
        child["records"]["lookm"]["runtime"]["generation_seconds"] += .1
        with self.assertRaisesRegex(ValueError, "Reused parent row changed"):
            audit.validate_reused_rows(child, parent, contract)

    def test_parent_prediction_and_parent_contract_identity_tampering_rejected(self):
        contract, parent = fixture()
        child = copy.deepcopy(parent)
        child["records"].pop("kvrepack")
        child["order"] = list(audit.MERGE_ONLY_ARMS)
        child["records"]["asymkv"]["text"] = "Altered reused caption"
        with self.assertRaisesRegex(ValueError, "Reused parent row changed"):
            audit.validate_reused_rows(child, parent, contract)
        parent["contract_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            audit.validate_reused_rows(child, parent, contract)

    def test_reused_bundle_file_sha_and_source_lineage_metadata_are_verified(self):
        contract, parent = fixture()
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)/"snapkv_kvrepack"
            (case/"samples").mkdir(parents=True)
            path = case/"samples/000000.json"
            path.write_text(json.dumps(parent))
            prefix = dict(completed_docs=1, parent_contract_path=str(case/"run_contract.json"),
                parent_contract_sha256=audit.canonical_sha(contract), parent_contract_file_sha256="c"*64,
                source_snapshot_path=str(Path(directory)/"source_snapshot.json"), source_snapshot_sha256="d"*64)
            context = dict(case=case, prefix=prefix, contract=contract)
            child = copy.deepcopy(parent)
            child["records"].pop("kvrepack")
            child["order"] = list(audit.MERGE_ONLY_ARMS)
            child["original_parent_order"] = parent["order"]
            child["reused_from_parent"] = {k:v for k,v in prefix.items() if k != "completed_docs"}
            child["reused_from_parent"].update(parent_bundle_path=str(path), parent_bundle_sha256=audit.file_sha(path))
            audit.validate_parent_bundle(child, context)
            child["reused_from_parent"]["parent_bundle_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "Parent bundle path/SHA"):
                audit.validate_parent_bundle(child, context)
            child["reused_from_parent"]["parent_bundle_sha256"] = audit.file_sha(path)
            child["reused_from_parent"]["source_snapshot_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "lineage metadata"):
                audit.validate_parent_bundle(child, context)

    def test_host_specific_first_prefill_token_drift_is_rejected(self):
        contract, bundle = fixture()
        bundle["records"]["asymkv"]["generated_ids"][0] = 43
        with self.assertRaisesRegex(ValueError, "first generated token differs"):
            audit.validate_bundle(bundle, contract)

    def test_changed_group_hash_rejected_despite_worker_passed_flag(self):
        contract, bundle = fixture()
        bundle["records"]["lookm"]["audit"]["merge_calls"][0]["group_ids"]["sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "groups/anchors differ"):
            audit.validate_bundle(bundle, contract)

    def test_extra_32_recent_positions_rejected_even_if_all_arms_match(self):
        contract, bundle = fixture()
        for row in bundle["records"].values():
            row["budget"].update(prefill_tokens_total=112*132, capacity=112*132, actual_retained_ratio=.132)
            for layer in row["budget"]["final_shapes"]:
                layer["key_shape"][2] += 32
                layer["value_shape"][2] += 32
        with self.assertRaisesRegex(ValueError, "strict 10%"):
            audit.validate_bundle(bundle, contract)

    def test_missing_fisher_cannot_silently_use_mean_fallback(self):
        contract, bundle = fixture()
        bundle["records"]["asymkv"]["fisher"] = None
        with self.assertRaisesRegex(ValueError, "Missing Fisher"):
            audit.validate_bundle(bundle, contract)

    def test_post_rope_or_incomplete_fisher_shapes_are_rejected(self):
        for kind in ("gradient_space", "gradient_shapes", "probe_policy"):
            contract, bundle = fixture()
            stats = bundle["records"]["asymkv"]["fisher"]
            if kind == "gradient_space":
                stats[kind] = "post_mrope"
            elif kind == "gradient_shapes":
                stats[kind] = [[1, 4, 999, 128]]*28
            else:
                stats[kind] = "legacy_no_fisher"
            with self.assertRaisesRegex(ValueError, "Fisher"):
                audit.validate_bundle(bundle, contract)

    def test_excluding_probe_from_asym_e2e_is_rejected(self):
        contract, bundle = fixture()
        bundle["records"]["asymkv"]["runtime"]["end_to_end_seconds"] -= .2
        with self.assertRaisesRegex(ValueError, "E2E timer"):
            audit.validate_bundle(bundle, contract)

    def test_incorrect_eos_flags_and_warmup_rows_are_rejected(self):
        for key, value in (("eos_reached", False), ("warmup", True), ("generated_tokens", 3)):
            contract, bundle = fixture()
            bundle["records"]["lookm"][key] = value
            with self.assertRaises(ValueError):
                audit.validate_bundle(bundle, contract)

    def test_group_prefix_and_anchor_ast_tampering_are_rejected(self):
        _, original = audit.qualified_function(ROOT/"src/kvrepack/qwen_model_snapkv_kvrepack.py", "_paired_merge_one_sequence")
        changed = copy.deepcopy(original)
        index = next(i for i,n in enumerate(changed.body) if audit.assignment_name(n) == "raw_sum")
        changed.body[index:index] = ast.parse("_fixed_audit(raw_k_seq, rope_k_seq, v_seq, importance, positions, group_ids, group_size, start_mask)").body
        audit.validate_merge_ast(original, changed, "snapkv_kvrepack", "kvrepack")
        prefix = copy.deepcopy(changed)
        prefix.body[0].value = ast.Constant(value=999)
        with self.assertRaisesRegex(ValueError, "prefix drift"):
            audit.validate_merge_ast(original, prefix, "snapkv_kvrepack", "kvrepack")
        anchor = copy.deepcopy(changed)
        target = next(n for n in anchor.body if audit.assignment_name(n) == "anchor_pos")
        target.value = ast.parse("positions[~start_mask]", mode="eval").body
        with self.assertRaisesRegex(ValueError, "Anchor mRoPE block drift"):
            audit.validate_merge_ast(original, anchor, "snapkv_kvrepack", "kvrepack")


if __name__ == "__main__":
    unittest.main()

"""Contracts that prevent unmatched groups or inflated latency comparisons."""
import copy
import unittest

from kvrepack.ablations.run_merge_comparison import pairing_signature
from kvrepack.ablations.merge_comparison_report import latency_summary


class MergeExperimentTests(unittest.TestCase):
    def test_group_change_is_detected_even_with_same_inputs_and_final_budget(self):
        audit = dict(status="complete", merge_calls=[dict(positions="positions", group_ids="groups",
            group_size="sizes", start_mask="starts")], selected_masks=["mask"],
            layers=[dict(layer=0, candidate_idx="candidates", final_history=3, candidate_history=6)],
            prefill_cache_shapes=[[1, 4, 35, 128]])
        changed = copy.deepcopy(audit)
        changed["merge_calls"][0]["group_ids"] = "different_actual_groups"
        self.assertNotEqual(pairing_signature(audit, "snapkv_kvrepack"),
                            pairing_signature(changed, "snapkv_kvrepack"))
        # Aggregated values intentionally differ between operators and must
        # not cause a false rejection of genuinely matched selection/groups.
        changed = copy.deepcopy(audit)
        changed["variant"] = "lookm"
        changed["merge_calls"][0]["raw"] = "optional diagnostic"
        self.assertEqual(pairing_signature(audit, "snapkv_kvrepack"),
                         pairing_signature(changed, "snapkv_kvrepack"))

    def test_fisher_time_is_included_in_e2e_and_kept_separate(self):
        rows = []
        for total, fisher, generated in ((6., 3., 2.), (10., 5., 4.)):
            rows.append(dict(runtime=dict(end_to_end_seconds=total, fisher_seconds=fisher,
                generation_seconds=generated, preprocessing_seconds=.8, h2d_seconds=.1,
                text_decode_seconds=.1, peak_allocated_gib=20.), generated_tokens=8,
                eos_reached=True, capped_without_eos=False,
                budget=dict(actual_retained_ratio=.099)))
        summary = latency_summary(rows)
        self.assertEqual(summary["end_to_end_seconds_mean"], 8.)
        self.assertEqual(summary["end_to_end_seconds_total"], 16.)
        self.assertEqual(summary["fisher_seconds_mean"], 4.)
        self.assertEqual(summary["tokens_per_generation_second"], 16/6)
        self.assertEqual(summary["e2e_p50_seconds"], 8.)
        self.assertEqual(summary["eos_count"], 2)


if __name__ == "__main__":
    unittest.main()

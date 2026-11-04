"""CPU tests of observation-A token/grid pairing, with no torch dependency."""
import copy
import json
import random
import unittest

from kvrepack.observations.pairs import build_pair_plan


class PairPlanTests(unittest.TestCase):
    def plan(self, *, ids=None, grid=(2, 4, 6), merge=2, segments=None,
             primary="question", seed=17, cap=16):
        if ids is None:
            ids = [1, 2] + [99] * 12 + list(range(10, 30))
        if segments is None:
            segments = [{"name": "question", "positions": list(range(14, 24))},
                        {"name": "instruction", "positions": list(range(24, 34))}]
        return build_pair_plan(ids, list(grid), merge, 99, segments, seed, cap,
                               primary_text_group=primary)

    def test_grid_maps_horizontal_neighbors_without_row_or_time_wrap(self):
        plan = self.plan()
        visual = plan["groups"]["visual_spatial"]
        self.assertEqual(plan["grid"]["merged_thw"], [2, 2, 3])
        self.assertEqual(visual["eligible_pair_count"], 8)
        self.assertEqual(visual["pairs"], [[2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]])
        self.assertNotIn([4, 5], visual["pairs"])
        self.assertNotIn([7, 8], visual["pairs"])
        for pair in visual["pair_metadata"]:
            left, right = pair["coordinates"]
            self.assertEqual(left[:2], right[:2])
            self.assertEqual(right[2], left[2] + 1)

    def test_noncontiguous_visual_placeholders_keep_actual_sequence_positions(self):
        ids = [99, 5, 99, 6, 99, 7, 99, 8, 10, 11, 12]
        plan = self.plan(ids=ids, grid=(1, 2, 2), merge=1,
                         segments=[{"name": "question", "positions": [8, 9, 10]}])
        self.assertEqual(plan["video_positions"], [0, 2, 4, 6])
        self.assertEqual(plan["groups"]["visual_spatial"]["pairs"], [[0, 2], [4, 6]])

    def test_text_native_adjacency_not_filtered_sequence_adjacency(self):
        plan = self.plan(segments=[{"name": "question", "positions": [14, 16, 17, 20, 21]}])
        self.assertEqual(plan["groups"]["text_question"]["pairs"], [[16, 17], [20, 21]])
        self.assertEqual(plan["matched_pair_count"], 2)
        self.assertEqual(len(plan["groups"]["visual_spatial"]["pairs"]), 2)

    def test_adjacent_segments_do_not_join_across_block_boundary(self):
        plan = self.plan(segments=[{"name": "question", "positions": [14, 15]},
                                   {"name": "text_question", "positions": [16, 17]}])
        self.assertEqual(plan["groups"]["text_question"]["pairs"], [[14, 15], [16, 17]])
        self.assertNotIn([15, 16], plan["groups"]["text_question"]["pairs"])

    def test_equal_main_counts_and_independent_instruction_supplement(self):
        plan = self.plan(segments=[{"name": "question", "positions": [14, 15, 16]},
                                   {"name": "instruction", "positions": list(range(17, 34))}])
        self.assertEqual(plan["matched_pair_count"], 2)
        self.assertEqual(plan["groups"]["visual_spatial"]["selected_pair_count"], 2)
        self.assertEqual(plan["groups"]["text_question"]["selected_pair_count"], 2)
        supplement = plan["groups"]["text_instruction"]
        self.assertEqual(supplement["selected_pair_count"], 16)
        self.assertFalse(supplement["matched_to_primary"])
        self.assertEqual(supplement["selection_policy"], "independent_supplement")

    def test_instruction_can_be_primary_for_vatex(self):
        plan = self.plan(primary="text_instruction")
        self.assertEqual(plan["primary_text_group"], "text_instruction")
        self.assertEqual(plan["groups"]["text_instruction"]["selected_pair_count"], 8)
        self.assertFalse(plan["groups"]["text_question"]["matched_to_primary"])

    def test_short_or_empty_primary_is_reported_without_failure(self):
        for positions in ([], [14], [14, 16]):
            with self.subTest(positions=positions):
                plan = self.plan(segments=[{"name": "question", "positions": positions},
                                           {"name": "instruction", "positions": [20, 21]}])
                self.assertEqual(plan["matched_pair_count"], 0)
                self.assertFalse(plan["coverage"]["main_eligible"])
                self.assertEqual(plan["coverage"]["status"], "no_primary_text_adjacent_pairs")
                self.assertEqual(plan["groups"]["visual_spatial"]["pairs"], [])
                self.assertEqual(plan["groups"]["text_question"]["pairs"], [])
                self.assertEqual(plan["groups"]["text_instruction"]["pairs"], [[20, 21]])

    def test_width_one_has_no_horizontal_neighbors(self):
        plan = self.plan(ids=[99] * 3 + [1, 2, 3], grid=(3, 1, 1), merge=1,
                         segments=[{"name": "question", "positions": [3, 4, 5]}])
        self.assertEqual(plan["matched_pair_count"], 0)
        self.assertEqual(plan["coverage"]["status"], "no_visual_spatial_adjacent_pairs")
        self.assertEqual(plan["groups"]["text_question"]["pairs"], [])

    def test_same_seed_reuses_identical_pair_ids_and_ignores_global_rng(self):
        first = self.plan(cap=3)
        random.seed(849)
        for _ in range(100):
            random.random()
        self.assertEqual(first, self.plan(cap=3))
        self.assertNotEqual(first["groups"]["visual_spatial"]["pairs"], self.plan(cap=3, seed=71)["groups"]["visual_spatial"]["pairs"])

    def test_adding_supplement_does_not_change_main_pair_selection(self):
        question = {"name": "question", "positions": list(range(14, 24))}
        first = self.plan(cap=3, segments=[question])
        second = self.plan(cap=3, segments=[question, {"name": "instruction", "positions": list(range(24, 34))}])
        for name in ("visual_spatial", "text_question"):
            self.assertEqual(first["groups"][name]["pair_ids"], second["groups"][name]["pair_ids"])

    def test_only_provided_nonvisual_text_positions_can_be_selected(self):
        plan = self.plan(segments=[{"name": "question", "positions": [16, 17, 18]},
                                   {"name": "instruction", "positions": [25, 26]}])
        for name, allowed in (("text_question", {16, 17, 18}), ("text_instruction", {25, 26})):
            for left, right in plan["groups"][name]["pairs"]:
                self.assertTrue({left, right}.issubset(allowed))
                self.assertEqual(right - left, 1)

    def test_inputs_are_not_mutated_and_plan_is_json_serializable(self):
        ids = [99] * 4 + [1, 2, 3]
        segments = [{"name": "question", "positions": [6, 4, 5]}]
        before = copy.deepcopy((ids, segments))
        plan = self.plan(ids=ids, grid=(1, 2, 2), merge=1, segments=segments)
        self.assertEqual((ids, segments), before)
        self.assertEqual(plan, json.loads(json.dumps(plan)))
        group = plan["groups"]["text_question"]
        self.assertEqual(group["pair_ids"], [row["pair_id"] for row in group["pair_metadata"]])
        self.assertFalse(plan["sampling"]["dataset_rows_deduplicated"])

    def test_invalid_grid_and_cardinality_are_rejected(self):
        for overrides in ({"grid": (1, 4)}, {"grid": (2, 3, 6)}, {"grid": (2, 4, 8)},
                          {"grid": (0, 4, 6)}, {"merge": 0}, {"merge": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.plan(**overrides)

    def test_impure_overlapping_duplicate_and_out_of_range_text_rejected(self):
        bad_segments = [
            [{"name": "question", "positions": [13, 14]}],
            [{"name": "question", "positions": [14, 14]}],
            [{"name": "question", "positions": [14, 15]}, {"name": "instruction", "positions": [15, 16]}],
            [{"name": "question", "positions": [34]}],
            [{"name": "question", "positions": [-1]}],
            [{"name": "system", "positions": [14, 15]}],
            [{"name": "question", "positions": [True, 15]}],
        ]
        for segments in bad_segments:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                self.plan(segments=segments)

    def test_invalid_primary_seed_and_limit_rejected(self):
        for overrides in ({"primary": "visual_spatial"}, {"seed": True}, {"cap": 0}, {"cap": 1.5}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.plan(**overrides)


if __name__ == "__main__":
    unittest.main()

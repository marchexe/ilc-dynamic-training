import math
import unittest

from training.runtime import (
    BTAG_REFERENCE_WORKING_POINTS,
    CTAG_REFERENCE_WORKING_POINTS,
    _working_point_metrics,
    combine_group_scores,
    geometric_mean_of_four,
    mistag_percent_key,
)


class GeometricMeanOfFourTest(unittest.TestCase):
    def test_all_equal_inputs_return_that_value(self):
        product, score = geometric_mean_of_four(16, 16, 16, 16)
        self.assertEqual(product, 16 ** 4)
        self.assertAlmostEqual(score, 16.0)

    def test_unequal_inputs_match_the_formula(self):
        x1, x2, x3, x4 = 1.0, 2.0, 4.0, 8.0
        product, score = geometric_mean_of_four(x1, x2, x3, x4)
        self.assertAlmostEqual(product, x1 * x2 * x3 * x4)
        self.assertAlmostEqual(score, (x1 * x2 * x3 * x4) ** 0.25)

    def test_zero_input_gives_zero_score_without_epsilon_substitution(self):
        product, score = geometric_mean_of_four(0.0, 1.0, 2.0, 3.0)
        self.assertEqual(product, 0.0)
        self.assertEqual(score, 0.0)

    def test_lower_score_is_better_ordering_example(self):
        _, better = geometric_mean_of_four(1.0, 1.0, 1.0, 1.0)
        _, worse = geometric_mean_of_four(2.0, 2.0, 2.0, 2.0)
        self.assertLess(better, worse)

    def test_every_one_of_the_four_inputs_affects_the_score(self):
        baseline = geometric_mean_of_four(1.0, 1.0, 1.0, 1.0)[1]
        self.assertNotEqual(geometric_mean_of_four(2.0, 1.0, 1.0, 1.0)[1], baseline)
        self.assertNotEqual(geometric_mean_of_four(1.0, 2.0, 1.0, 1.0)[1], baseline)
        self.assertNotEqual(geometric_mean_of_four(1.0, 1.0, 2.0, 1.0)[1], baseline)
        self.assertNotEqual(geometric_mean_of_four(1.0, 1.0, 1.0, 2.0)[1], baseline)

    def test_negative_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mean_of_four(-1.0, 1.0, 1.0, 1.0)

    def test_missing_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mean_of_four(None, 1.0, 1.0, 1.0)

    def test_nan_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mean_of_four(float("nan"), 1.0, 1.0, 1.0)

    def test_infinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mean_of_four(float("inf"), 1.0, 1.0, 1.0)


class CombineGroupScoresTest(unittest.TestCase):
    def test_equals_sqrt_of_the_product(self):
        self.assertAlmostEqual(combine_group_scores(4.0, 9.0), 6.0)

    def test_matches_the_equivalent_eight_value_geometric_mean(self):
        ctag_values = (1.0, 2.0, 3.0, 4.0)
        btag_values = (5.0, 6.0, 7.0, 8.0)
        _, ctag_score = geometric_mean_of_four(*ctag_values)
        _, btag_score = geometric_mean_of_four(*btag_values)
        total = combine_group_scores(ctag_score, btag_score)
        product_of_eight = 1.0
        for value in (*ctag_values, *btag_values):
            product_of_eight *= value
        self.assertAlmostEqual(total, product_of_eight ** 0.125)

    def test_none_when_ctag_score_missing(self):
        self.assertIsNone(combine_group_scores(None, 5.0))

    def test_none_when_btag_score_missing(self):
        self.assertIsNone(combine_group_scores(5.0, None))


class WorkingPointMetricsGeometricScoreWiringTest(unittest.TestCase):
    """Verifies the ctag_score/btag_score/total_mistag_score wiring inside
    _working_point_metrics against synthetic curves data, not just the pure
    functions in isolation."""

    def _curves(self):
        return {
            "efficiencies": [0.5, 0.8, 0.9],
            "pairs": {
                "cb": [10.0, 20.0, 30.0],  # cb@0.5 -> mistag 100/10=10.0; cb@0.8 -> 100/20=5.0
                "cd": [8.0, 4.0, 2.0],     # cd@0.5 -> mistag 100/8=12.5; cd@0.8 -> 100/4=25.0
                "bc": [40.0, 25.0, 8.0],   # bc@0.8 -> mistag 100/25=4.0; bc@0.9 -> 100/8=12.5
                "bd": [50.0, 20.0, 5.0],   # bd@0.8 -> mistag 100/20=5.0; bd@0.9 -> 100/5=20.0
            },
        }

    def test_ctag_and_btag_scores_are_recorded(self):
        out = _working_point_metrics(self._curves())
        expected_ctag_product, expected_ctag_score = geometric_mean_of_four(10.0, 12.5, 5.0, 25.0)
        expected_btag_product, expected_btag_score = geometric_mean_of_four(4.0, 5.0, 12.5, 20.0)
        self.assertAlmostEqual(out["validation_ctag_reference_mistag_geomean_percent"], expected_ctag_score)
        self.assertAlmostEqual(out["validation_btag_reference_mistag_geomean_percent"], expected_btag_score)

    def test_total_score_equals_sqrt_of_ctag_times_btag(self):
        out = _working_point_metrics(self._curves())
        expected_total = combine_group_scores(
            out["validation_ctag_reference_mistag_geomean_percent"],
            out["validation_btag_reference_mistag_geomean_percent"],
        )
        self.assertAlmostEqual(out["validation_total_reference_mistag_geomean_percent"], expected_total)
        # cb is not bc, cd is not dc/bd -- confirm the raw per-pair keys are
        # genuinely distinct series, not aliases of one another.
        self.assertNotEqual(
            out[mistag_percent_key("cb", 0.8)],
            out[mistag_percent_key("bc", 0.8)],
        )

    def test_missing_working_point_yields_none_group_score_not_a_crash(self):
        curves = self._curves()
        curves["efficiencies"] = [0.5]  # cb@0.8/cd@0.8/bc@0.8/bd@0.8/bc@0.9/bd@0.9 all now unavailable
        out = _working_point_metrics(curves)
        self.assertIsNone(out["validation_ctag_reference_mistag_geomean_percent"])
        self.assertIsNone(out["validation_btag_reference_mistag_geomean_percent"])
        self.assertIsNone(out["validation_total_reference_mistag_geomean_percent"])

    def test_ctag_group_still_available_when_only_btag_is_missing(self):
        curves = self._curves()
        del curves["pairs"]["bc"]
        out = _working_point_metrics(curves)
        self.assertIsNotNone(out["validation_ctag_reference_mistag_geomean_percent"])
        self.assertIsNone(out["validation_btag_reference_mistag_geomean_percent"])
        self.assertIsNone(out["validation_total_reference_mistag_geomean_percent"])

    def test_canonical_working_point_tuples_partition_cleanly(self):
        # ctag_score's 4 points + btag_score's 4 points must be exactly the
        # 8 points that make up total_mistag_score, with no overlap.
        self.assertEqual(len(CTAG_REFERENCE_WORKING_POINTS), 4)
        self.assertEqual(len(BTAG_REFERENCE_WORKING_POINTS), 4)
        self.assertEqual(set(CTAG_REFERENCE_WORKING_POINTS) & set(BTAG_REFERENCE_WORKING_POINTS), set())


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from training.runtime import _working_point_metrics, geometric_mistag_score


class GeometricMistagScoreTest(unittest.TestCase):
    def test_all_equal_inputs_return_that_value(self):
        product, score = geometric_mistag_score(16, 16, 16, 16)
        self.assertEqual(product, 16 ** 4)
        self.assertAlmostEqual(score, 16.0)

    def test_unequal_inputs_match_the_formula(self):
        x1, x2, x3, x4 = 1.0, 2.0, 4.0, 8.0
        product, score = geometric_mistag_score(x1, x2, x3, x4)
        self.assertAlmostEqual(product, x1 * x2 * x3 * x4)
        self.assertAlmostEqual(score, (x1 * x2 * x3 * x4) ** 0.25)

    def test_zero_input_gives_zero_score_without_epsilon_substitution(self):
        product, score = geometric_mistag_score(0.0, 1.0, 2.0, 3.0)
        self.assertEqual(product, 0.0)
        self.assertEqual(score, 0.0)

    def test_lower_score_is_better_ordering_example(self):
        _, better = geometric_mistag_score(1.0, 1.0, 1.0, 1.0)
        _, worse = geometric_mistag_score(2.0, 2.0, 2.0, 2.0)
        self.assertLess(better, worse)

    def test_every_one_of_the_four_inputs_affects_the_score(self):
        baseline = geometric_mistag_score(1.0, 1.0, 1.0, 1.0)[1]
        self.assertNotEqual(geometric_mistag_score(2.0, 1.0, 1.0, 1.0)[1], baseline)
        self.assertNotEqual(geometric_mistag_score(1.0, 2.0, 1.0, 1.0)[1], baseline)
        self.assertNotEqual(geometric_mistag_score(1.0, 1.0, 2.0, 1.0)[1], baseline)
        self.assertNotEqual(geometric_mistag_score(1.0, 1.0, 1.0, 2.0)[1], baseline)

    def test_negative_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mistag_score(-1.0, 1.0, 1.0, 1.0)

    def test_missing_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mistag_score(None, 1.0, 1.0, 1.0)

    def test_nan_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mistag_score(float("nan"), 1.0, 1.0, 1.0)

    def test_infinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            geometric_mistag_score(float("inf"), 1.0, 1.0, 1.0)


class WorkingPointMetricsGeometricScoreWiringTest(unittest.TestCase):
    """Verifies the x1..x4 -> product -> score wiring inside
    _working_point_metrics against synthetic curves data, not just the pure
    geometric_mistag_score function in isolation."""

    def _curves(self):
        return {
            "efficiencies": [0.5, 0.8],
            "pairs": {
                "cb": [10.0, 20.0],  # cb@0.5 -> mistag 100/10=10.0 (x1); cb@0.8 -> 100/20=5.0 (x3)
                "cd": [8.0, 4.0],    # cd@0.5 -> mistag 100/8=12.5 (x2); cd@0.8 -> 100/4=25.0 (x4)
                "bc": [],
                "bd": [],
            },
        }

    def test_x1_through_x4_are_recorded_and_aliased_from_the_underlying_working_points(self):
        out = _working_point_metrics(self._curves())
        self.assertAlmostEqual(out["validation_mistag_geometric_score_x1_percent"], 10.0)
        self.assertAlmostEqual(out["validation_mistag_geometric_score_x2_percent"], 12.5)
        self.assertAlmostEqual(out["validation_mistag_geometric_score_x3_percent"], 5.0)
        self.assertAlmostEqual(out["validation_mistag_geometric_score_x4_percent"], 25.0)
        self.assertAlmostEqual(out["validation_cb_mistag_eff_0.50_percent"], 10.0)
        self.assertAlmostEqual(out["validation_cd_mistag_eff_0.80_percent"], 25.0)

    def test_product_and_score_match_the_pure_function(self):
        out = _working_point_metrics(self._curves())
        expected_product, expected_score = geometric_mistag_score(10.0, 12.5, 5.0, 25.0)
        self.assertAlmostEqual(out["validation_mistag_geometric_score_product"], expected_product)
        self.assertAlmostEqual(out["validation_mistag_geometric_score_percent"], expected_score)

    def test_missing_working_point_yields_none_score_not_a_crash(self):
        curves = self._curves()
        curves["efficiencies"] = [0.5]  # cb@0.8/cd@0.8 (x3/x4) now unavailable
        out = _working_point_metrics(curves)
        self.assertIsNone(out["validation_mistag_geometric_score_x3_percent"])
        self.assertIsNone(out["validation_mistag_geometric_score_x4_percent"])
        self.assertIsNone(out["validation_mistag_geometric_score_product"])
        self.assertIsNone(out["validation_mistag_geometric_score_percent"])


if __name__ == "__main__":
    unittest.main()

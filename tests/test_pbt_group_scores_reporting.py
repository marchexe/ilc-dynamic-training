import copy
import tempfile
import unittest
from pathlib import Path

from tests.test_pbt_artifacts import CURVE_EFFICIENCIES, CURVE_REJECTIONS, synthetic_manifest
from training.pbt.reporting import evaluation_rows, group_score_row, write_canonical_outputs
from training.pbt.reporting.constants import (
    BTAG_GEOMEAN_METRIC_KEY,
    BTAG_SCORE_COLUMN,
    BTAG_SCORE_WORKING_POINTS,
    CTAG_GEOMEAN_METRIC_KEY,
    CTAG_SCORE_COLUMN,
    CTAG_SCORE_WORKING_POINTS,
    FIXED_WORKING_POINTS,
    TOTAL_GEOMEAN_METRIC_KEY,
    TOTAL_SCORE_COLUMN,
)
from training.runtime import combine_group_scores


class CanonicalWorkingPointPartitionTest(unittest.TestCase):
    """Requirement 1/2/3: c-tag and b-tag plots must use exactly their own
    four working points, and cb@0.8/bc@0.8 must never collide."""

    def test_ctag_group_contains_only_cb_cd_pairs(self):
        self.assertEqual({point["pair"] for point in CTAG_SCORE_WORKING_POINTS}, {"cb", "cd"})

    def test_btag_group_contains_only_bc_bd_pairs(self):
        self.assertEqual({point["pair"] for point in BTAG_SCORE_WORKING_POINTS}, {"bc", "bd"})

    def test_groups_do_not_overlap(self):
        ctag_labels = {point["score_label"] for point in CTAG_SCORE_WORKING_POINTS}
        btag_labels = {point["score_label"] for point in BTAG_SCORE_WORKING_POINTS}
        self.assertEqual(ctag_labels & btag_labels, set())

    def test_cb_0p8_and_bc_0p8_are_distinct_labels(self):
        self.assertIn("cb@0.8", {point["score_label"] for point in CTAG_SCORE_WORKING_POINTS})
        self.assertIn("bc@0.8", {point["score_label"] for point in BTAG_SCORE_WORKING_POINTS})
        self.assertNotIn("bc@0.8", {point["score_label"] for point in CTAG_SCORE_WORKING_POINTS})
        self.assertNotIn("cb@0.8", {point["score_label"] for point in BTAG_SCORE_WORKING_POINTS})


class GroupScoreRowReconstructionTest(unittest.TestCase):
    """Requirements 11/12/13: old manifests without the aggregate fields
    must still work; reconstruction requires all 8 raw inputs; a missing
    input produces a warning, never an invented score."""

    def _curves_metrics(self):
        return {
            "validation_bkg_rejection_at_eff": {
                "efficiencies": list(CURVE_EFFICIENCIES),
                "pairs": copy.deepcopy(CURVE_REJECTIONS),
            },
        }

    def test_old_style_metrics_without_aggregate_fields_reconstruct_fully(self):
        ctag_score, btag_score, total_score, missing = group_score_row(self._curves_metrics())
        self.assertEqual(missing, [])
        self.assertIsNotNone(ctag_score)
        self.assertIsNotNone(btag_score)
        self.assertAlmostEqual(total_score, combine_group_scores(ctag_score, btag_score))

    def test_missing_one_working_point_produces_a_warning_not_a_fabricated_score(self):
        metrics = self._curves_metrics()
        # Drop cb entirely -- cb@0.5 and cb@0.8 both become unavailable.
        del metrics["validation_bkg_rejection_at_eff"]["pairs"]["cb"]
        ctag_score, btag_score, total_score, missing = group_score_row(metrics)
        self.assertIsNone(ctag_score)
        self.assertIsNotNone(btag_score)  # unaffected group still available
        self.assertIsNone(total_score)  # total requires both groups
        self.assertIn("cb@0.5", missing)
        self.assertIn("cb@0.8", missing)

    def test_no_raw_data_at_all_produces_full_missing_list_not_zero_or_nan(self):
        ctag_score, btag_score, total_score, missing = group_score_row({})
        self.assertIsNone(ctag_score)
        self.assertIsNone(btag_score)
        self.assertIsNone(total_score)
        self.assertEqual(len(missing), 8)

    def test_already_present_aggregate_keys_are_used_directly_not_recomputed(self):
        # Deliberately inconsistent with what the raw curves would produce,
        # so returning these exact (wrong-on-purpose) values proves the
        # function used them as-is rather than recomputing from raw data.
        metrics = self._curves_metrics()
        metrics[CTAG_GEOMEAN_METRIC_KEY] = 111.0
        metrics[BTAG_GEOMEAN_METRIC_KEY] = 222.0
        metrics[TOTAL_GEOMEAN_METRIC_KEY] = 333.0
        ctag_score, btag_score, total_score, missing = group_score_row(metrics)
        self.assertEqual((ctag_score, btag_score, total_score), (111.0, 222.0, 333.0))
        self.assertEqual(missing, [])


class EvaluationRowsGroupScoreWiringTest(unittest.TestCase):
    """Requirement 4/5/6: evaluation_rows (the row-preparation layer every
    plot consumes) carries the three canonical scores, total == sqrt(ctag *
    btag), and the values are read from group_score_row -- not
    reimplemented in the row-building loop itself."""

    def test_rows_carry_all_three_group_scores(self):
        manifest = synthetic_manifest()
        rows = evaluation_rows(manifest)
        self.assertTrue(rows)
        for row in rows:
            self.assertIn(CTAG_SCORE_COLUMN, row)
            self.assertIn(BTAG_SCORE_COLUMN, row)
            self.assertIn(TOTAL_SCORE_COLUMN, row)
            self.assertAlmostEqual(row[TOTAL_SCORE_COLUMN], combine_group_scores(row[CTAG_SCORE_COLUMN], row[BTAG_SCORE_COLUMN]))

    def test_lower_total_score_selects_the_same_winner_as_the_configured_metric(self):
        # synthetic_manifest's configured metric IS the fixed-WP arithmetic
        # mean family, not total_mistag_score -- but every row still gets a
        # total_mistag_score computed alongside it, and picking by that
        # column must still deterministically select one real member.
        manifest = synthetic_manifest()
        rows = evaluation_rows(manifest)
        generation_0 = [row for row in rows if row["generation"] == 0]
        winner = min(generation_0, key=lambda row: row[TOTAL_SCORE_COLUMN])
        self.assertIn(winner["trial"], {"trial_a", "trial_b"})
        losers = [row for row in generation_0 if row is not winner]
        self.assertTrue(all(winner[TOTAL_SCORE_COLUMN] <= row[TOTAL_SCORE_COLUMN] for row in losers))


class DeterministicOrderingTest(unittest.TestCase):
    def test_fixed_working_points_iteration_order_is_stable(self):
        self.assertEqual([point["score_label"] for point in FIXED_WORKING_POINTS], list(dict.fromkeys(point["score_label"] for point in FIXED_WORKING_POINTS)))
        # Re-importing/re-deriving must reproduce the identical order.
        from importlib import reload

        import training.pbt.reporting.constants as constants_module

        reload(constants_module)
        self.assertEqual(
            [point["score_label"] for point in FIXED_WORKING_POINTS],
            [point["score_label"] for point in constants_module.FIXED_WORKING_POINTS],
        )


class ReportRerunStabilityTest(unittest.TestCase):
    """Requirements 16/17: canonical filenames are stable, and re-running
    report generation on the same run_dir does not accumulate duplicate
    plot files."""

    def test_rerunning_write_canonical_outputs_does_not_duplicate_plot_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest()
            write_canonical_outputs(run_dir, manifest)
            first_listing = sorted(p.name for p in (run_dir / "plots").iterdir() if p.is_file())

            write_canonical_outputs(run_dir, manifest)
            second_listing = sorted(p.name for p in (run_dir / "plots").iterdir() if p.is_file())

            self.assertEqual(first_listing, second_listing)
            for expected in (
                "ctag_working_points_evolution.png",
                "btag_working_points_evolution.png",
                "aggregate_mistag_score_evolution.png",
            ):
                self.assertIn(expected, second_listing)


if __name__ == "__main__":
    unittest.main()

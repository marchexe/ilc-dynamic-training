import unittest

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from research.proxy_statistics import (
    best_checkpoint_agreement,
    build_synthetic_manifest,
    full_summary,
    kendall_tau,
    pairwise_direction_agreement,
)
from training.pbt.reporting.statistics import tier_correlation, ranking_agreement


def _result(value):
    return {"status": "completed", "metrics": {"validation_working_point_mistag_percent": value}}


class BuildSyntheticManifestTest(unittest.TestCase):
    def test_manifest_shape_matches_reporting_statistics_expectations(self):
        control = {"a": _result(1.0), "b": _result(0.5)}
        full = {"a": _result(1.1), "b": _result(0.6)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")

        tiers = {round_record["tier"] for round_record in manifest["tiered_evaluations"]}
        self.assertEqual(tiers, {"control", "full_holdout"})
        self.assertEqual(manifest["config"]["pbt"]["mode"], "min")
        # min mode: "b" (0.5) should rank first in control.
        control_round = next(r for r in manifest["tiered_evaluations"] if r["tier"] == "control")
        self.assertEqual(control_round["ranking"], ["b", "a"])

    def test_non_finite_metrics_excluded_from_ranking(self):
        control = {"a": _result(1.0), "b": _result(float("nan")), "c": _result(float("inf"))}
        manifest = build_synthetic_manifest(control, control, "validation_working_point_mistag_percent", "min")
        control_round = next(r for r in manifest["tiered_evaluations"] if r["tier"] == "control")
        self.assertEqual(control_round["ranking"], ["a"])


class KendallTauTest(unittest.TestCase):
    def test_perfect_agreement_gives_tau_one(self):
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0), "d": _result(4.0)}
        full = {"a": _result(1.5), "b": _result(2.5), "c": _result(3.5), "d": _result(4.5)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        result = kendall_tau(manifest)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["tau"], 1.0)
        self.assertIsNone(result["reason"])

    def test_perfect_disagreement_gives_tau_negative_one(self):
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0), "d": _result(4.0)}
        full = {"a": _result(4.0), "b": _result(3.0), "c": _result(2.0), "d": _result(1.0)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        result = kendall_tau(manifest)
        self.assertAlmostEqual(result["tau"], -1.0)

    def test_insufficient_points_reports_reason_not_a_fabricated_value(self):
        control = {"a": _result(1.0), "b": _result(2.0)}
        manifest = build_synthetic_manifest(control, control, "validation_working_point_mistag_percent", "min")
        result = kendall_tau(manifest)
        self.assertIsNone(result["tau"])
        self.assertEqual(result["reason"], "insufficient_paired_observations")


class PairwiseDirectionAgreementTest(unittest.TestCase):
    def test_full_agreement(self):
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0)}
        full = {"a": _result(1.1), "b": _result(2.1), "c": _result(3.1)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        result = pairwise_direction_agreement(manifest)
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["agreeing_pairs"], 3)
        self.assertEqual(result["agreement_fraction"], 1.0)
        self.assertEqual(result["disagreements"], [])

    def test_detects_a_single_flipped_pair(self):
        # min mode: control ranks a < b < c; full flips b and c.
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0)}
        full = {"a": _result(1.0), "b": _result(3.0), "c": _result(2.0)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        result = pairwise_direction_agreement(manifest)
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["agreeing_pairs"], 2)
        self.assertEqual(len(result["disagreements"]), 1)
        disagreement = result["disagreements"][0]
        self.assertEqual({disagreement["checkpoint_a"], disagreement["checkpoint_b"]}, {"b", "c"})


    def test_tied_values_are_excluded_not_forced_to_agree_or_disagree(self):
        control = {"a": _result(1.0), "b": _result(1.0), "c": _result(2.0)}
        full = {"a": _result(5.0), "b": _result(6.0), "c": _result(7.0)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        result = pairwise_direction_agreement(manifest)
        # a vs b tied on control -> excluded; only a-c and b-c pairs counted.
        self.assertEqual(result["n_pairs"], 2)

    def test_insufficient_checkpoints(self):
        control = {"a": _result(1.0)}
        manifest = build_synthetic_manifest(control, control, "validation_working_point_mistag_percent", "min")
        result = pairwise_direction_agreement(manifest)
        self.assertEqual(result["reason"], "insufficient_paired_observations")
        self.assertIsNone(result["agreement_fraction"])


class BestCheckpointAgreementTest(unittest.TestCase):
    def test_agrees_when_same_checkpoint_wins_both_tiers(self):
        control = {"a": _result(1.0), "b": _result(0.5)}
        full = {"a": _result(2.0), "b": _result(0.8)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        result = best_checkpoint_agreement(manifest)
        self.assertTrue(result["agrees"])
        self.assertEqual(result["control_best"]["member"], "b")
        self.assertEqual(result["full_best"]["member"], "b")

    def test_disagrees_when_different_checkpoints_win(self):
        control = {"a": _result(0.5), "b": _result(1.0)}
        full = {"a": _result(2.0), "b": _result(0.8)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        result = best_checkpoint_agreement(manifest)
        self.assertFalse(result["agrees"])

    def test_min_max_mode_orientation_flips_which_checkpoint_wins(self):
        """Same raw values, opposite mode -- the "best" checkpoint must
        flip too (min favors the lower value, max favors the higher one).
        A bug that ignored `mode` would report the same winner either way."""
        control = {"a": _result(1.0), "b": _result(2.0)}
        min_manifest = build_synthetic_manifest(control, control, "validation_working_point_mistag_percent", "min")
        max_manifest = build_synthetic_manifest(control, control, "validation_working_point_mistag_percent", "max")
        self.assertEqual(best_checkpoint_agreement(min_manifest)["control_best"]["member"], "a")
        self.assertEqual(best_checkpoint_agreement(max_manifest)["control_best"]["member"], "b")


class FullSummaryTest(unittest.TestCase):
    def test_reports_insufficient_evidence_below_three_checkpoints(self):
        control = {"a": _result(1.0), "b": _result(2.0)}
        full = {"a": _result(1.5), "b": _result(2.5)}
        result = full_summary(control, full, "validation_working_point_mistag_percent", "min")
        self.assertTrue(result["insufficient_evidence"])

    def test_does_not_claim_insufficient_evidence_at_three_or_more(self):
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0)}
        full = {"a": _result(1.1), "b": _result(2.1), "c": _result(3.1)}
        result = full_summary(control, full, "validation_working_point_mistag_percent", "min")
        self.assertFalse(result["insufficient_evidence"])

    def test_bootstrap_confidence_intervals_are_explicitly_unavailable(self):
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0)}
        result = full_summary(control, control, "validation_working_point_mistag_percent", "min")
        self.assertIn("unavailable", result["bootstrap_confidence_intervals"])

    def test_does_not_duplicate_reporting_statistics_correlation_math(self):
        """full_summary's pearson/spearman numbers must come from the same
        tier_correlation function reporting/statistics.py's live-run
        diagnostics use, not a re-implementation that could silently drift
        from it."""
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0), "d": _result(0.5)}
        full = {"a": _result(1.2), "b": _result(1.8), "c": _result(3.3), "d": _result(0.4)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        direct = tier_correlation(manifest, "control", "full_holdout")
        result = full_summary(control, full, "validation_working_point_mistag_percent", "min")
        self.assertEqual(result["pearson_spearman"], direct)

    def test_ranking_agreement_reuses_reporting_statistics(self):
        control = {"a": _result(1.0), "b": _result(2.0), "c": _result(3.0)}
        full = {"a": _result(1.1), "b": _result(2.2), "c": _result(3.3)}
        manifest = build_synthetic_manifest(control, full, "validation_working_point_mistag_percent", "min")
        direct = ranking_agreement(manifest, "control", "full_holdout")
        result = full_summary(control, full, "validation_working_point_mistag_percent", "min")
        self.assertEqual(result["ranking_agreement"], direct[0])


if __name__ == "__main__":
    unittest.main()

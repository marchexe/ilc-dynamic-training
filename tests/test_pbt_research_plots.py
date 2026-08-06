import math
import tempfile
import unittest
from pathlib import Path

from tests.test_pbt_artifacts import fixed_curve_metrics, synthetic_manifest
from training.pbt.reporting.constants import (
    BTAG_SCORE_COLUMN,
    BTAG_SCORE_WORKING_POINTS,
    CONDITIONAL_RESEARCH_PLOT_NAMES,
    CTAG_SCORE_COLUMN,
    CTAG_SCORE_WORKING_POINTS,
    RESEARCH_PLOT_NAMES,
    TOTAL_SCORE_COLUMN,
)
from training.pbt.reporting.research_plots import (
    build_generation_decision_rows,
    build_member_metric_rows,
    plot_aggregate_scores,
    plot_baseline_ratio,
    plot_btag_working_points,
    plot_ctag_working_points,
    plot_decision_history,
    plot_lr_population,
    plot_score_vs_lr,
    plot_tag_tradeoff,
    validate_metric_rows,
    write_research_plots,
)
from training.runtime import combine_group_scores


def _anchor_copy_manifest():
    """A small, fully hand-specified anchor_copy_lr_recenter manifest: 3
    generations, 2 members, one occurrence of each decision outcome, and
    known LR/metric values throughout -- so numeric-verification tests can
    assert exact expected results rather than just "did not crash"."""
    manifest = synthetic_manifest()
    manifest["config"]["pbt"]["strategy"] = "anchor_copy_lr_recenter"
    manifest["config"]["pbt"]["min_lr"] = 3.0e-6
    manifest["config"]["pbt"]["max_lr"] = 1.4e-5
    manifest["members"] = {
        "m_a": {"name": "m_a", "lr": 1.0e-5, "parent": None},
        "m_b": {"name": "m_b", "lr": 8.0e-6, "parent": None},
    }
    manifest["generations"] = [
        {
            "index": 0,
            "epoch": 20,
            "status": "completed",
            "workers": {
                "m_a": {"status": "completed", "lr": 1.0e-5, "metrics": fixed_curve_metrics(0.9, 0.9, 0.97, 0.3)},
                "m_b": {"status": "completed", "lr": 8.0e-6, "metrics": fixed_curve_metrics(1.1, 0.88, 0.96, 0.32)},
            },
            "ranking": ["m_a", "m_b"],
            "exploit": [],
            "anchor_copy_lr_recenter": {
                "decision": "accepted_new_anchor",
                "winner": "m_a",
                "winner_lr": 1.0e-5,
                "previous_lr_center": 1.0e-5,
                "new_lr_center": 1.0e-5,
                "assigned_lrs": {"m_a": 1.0e-5, "m_b": 9.0e-6},
                "spread_collapsed": False,
            },
        },
        {
            "index": 1,
            "epoch": 21,
            "status": "completed",
            "workers": {
                "m_a": {"status": "completed", "lr": 1.0e-5, "metrics": fixed_curve_metrics(0.95, 0.9, 0.97, 0.3)},
                "m_b": {"status": "completed", "lr": 9.0e-6, "metrics": fixed_curve_metrics(0.85, 0.9, 0.98, 0.29)},
            },
            "ranking": ["m_b", "m_a"],
            "exploit": [],
            "anchor_copy_lr_recenter": {
                "decision": "reused_previous_anchor",
                "winner": "m_b",
                "winner_lr": 9.0e-6,
                "previous_lr_center": 1.0e-5,
                "new_lr_center": 9.0e-6,
                "assigned_lrs": {"m_a": 3.0e-6, "m_b": 1.4e-5},  # deliberately at the min/max bounds
                "spread_collapsed": True,
            },
        },
        {
            "index": 2,
            "epoch": 22,
            "status": "completed",
            "workers": {
                "m_a": {"status": "completed", "lr": 3.0e-6, "metrics": fixed_curve_metrics(1.3, 0.8, 0.9, 0.4)},
                "m_b": {"status": "completed", "lr": 1.4e-5, "metrics": fixed_curve_metrics(1.4, 0.79, 0.89, 0.41)},
            },
            "ranking": ["m_a", "m_b"],
            "exploit": [],
            "anchor_copy_lr_recenter": {
                "decision": "rewound_to_previous_anchor",
                "winner": "m_a",
                "winner_lr": 3.0e-6,
                "previous_lr_center": 9.0e-6,
                "new_lr_center": 9.0e-6,
                "assigned_lrs": {"m_a": 8.1e-6, "m_b": 9.9e-6},
                "spread_collapsed": False,
            },
        },
    ]
    return manifest


class DataLayerTest(unittest.TestCase):
    def test_member_rows_flag_the_actual_configured_metric_winner(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        generation_0 = [row for row in rows if row["generation"] == 0]
        winners = [row for row in generation_0 if row["is_winner"]]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["trial"], "m_a")  # 0.9 < 1.1, mode=min

    def test_decision_rows_empty_for_non_anchor_copy_strategy(self):
        manifest = synthetic_manifest()  # strategy=anchored_lr_sweep
        rows = build_member_metric_rows(manifest)
        self.assertEqual(build_generation_decision_rows(manifest, rows), [])

    def test_decision_rows_carry_the_three_decision_types(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        self.assertEqual([d["decision"] for d in decisions], ["accepted_new_anchor", "reused_previous_anchor", "rewound_to_previous_anchor"])

    def test_anchor_row_carries_forward_unchanged_across_reuse_and_rewind(self):
        # gen0 accepts m_a as anchor; gen1/gen2 never accept again, so the
        # anchor's *own* row must stay gen0's m_a row throughout, even
        # though a different member wins gen1/gen2.
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        anchor_rows = [d["anchor_row"] for d in decisions]
        self.assertTrue(all(row is anchor_rows[0] for row in anchor_rows))
        self.assertEqual(anchor_rows[0]["trial"], "m_a")
        self.assertEqual(anchor_rows[0]["generation"], 0)

    def test_anchor_score_before_decision_is_none_for_the_first_ever_decision(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        self.assertIsNone(decisions[0]["anchor_total_score_before_decision"])
        # gen1/gen2 compare against gen0's now-established anchor score.
        self.assertIsNotNone(decisions[1]["anchor_total_score_before_decision"])
        self.assertAlmostEqual(decisions[1]["anchor_total_score_before_decision"], decisions[0]["anchor_row"][TOTAL_SCORE_COLUMN])

    def test_validate_metric_rows_keeps_a_genuine_zero(self):
        rows = [{"trial": "m", "generation": 0, "x": 0.0}]
        valid, warnings = validate_metric_rows(rows, ["x"])
        self.assertEqual(valid, rows)
        self.assertEqual(warnings, [])

    def test_validate_metric_rows_rejects_negative_missing_and_non_finite(self):
        rows = [
            {"trial": "a", "generation": 0, "x": -1.0},
            {"trial": "b", "generation": 0, "x": None},
            {"trial": "c", "generation": 0, "x": float("nan")},
            {"trial": "d", "generation": 0, "x": float("inf")},
            {"trial": "e", "generation": 0, "x": 5.0},
        ]
        valid, warnings = validate_metric_rows(rows, ["x"])
        self.assertEqual([row["trial"] for row in valid], ["e"])
        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("member=a" in warning and "negative" in warning for warning in warnings))
        self.assertTrue(any("member=b" in warning and "missing" in warning for warning in warnings))
        self.assertTrue(any("member=c" in warning and "non-finite" in warning for warning in warnings))
        self.assertTrue(any("member=d" in warning and "non-finite" in warning for warning in warnings))


class PanelToMetricMappingTest(unittest.TestCase):
    """Requirement: correct panel-to-metric mapping (c-tag plot uses only
    c-tag columns, b-tag only b-tag columns)."""

    def test_ctag_plot_metric_keys_are_exactly_the_ctag_working_points(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_ctag_working_points(temporary, manifest, rows, decisions)
        self.assertEqual(set(result["metric_keys"]), {point["column"] for point in CTAG_SCORE_WORKING_POINTS})

    def test_btag_plot_metric_keys_are_exactly_the_btag_working_points(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_btag_working_points(temporary, manifest, rows, decisions)
        self.assertEqual(set(result["metric_keys"]), {point["column"] for point in BTAG_SCORE_WORKING_POINTS})


class AggregateScoresPlotTest(unittest.TestCase):
    def test_contains_all_three_group_score_columns(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_aggregate_scores(temporary, manifest, rows, decisions)
        self.assertEqual(set(result["metric_keys"]), {CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN, TOTAL_SCORE_COLUMN})

    def test_total_score_equals_sqrt_ctag_times_btag_for_every_plotted_row(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        for row in rows:
            self.assertAlmostEqual(row[TOTAL_SCORE_COLUMN], combine_group_scores(row[CTAG_SCORE_COLUMN], row[BTAG_SCORE_COLUMN]))

    def test_produces_a_png(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_aggregate_scores(temporary, manifest, rows, decisions)
            self.assertTrue(Path(result["png"]).is_file())
            self.assertEqual(Path(result["png"]).suffix, ".png")
            self.assertNotIn("pdf", result)


class TagTradeoffPlotTest(unittest.TestCase):
    """Requirement: tag-tradeoff coordinates are exactly (ctag_score,
    btag_score) per row -- not swapped, not re-derived."""

    def test_metric_keys_are_ctag_then_btag(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_tag_tradeoff(temporary, manifest, rows, decisions)
        self.assertEqual(result["metric_keys"], [CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN])

    def test_covers_every_generation_and_member(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_tag_tradeoff(temporary, manifest, rows, decisions)
        self.assertEqual(result["generations"], 3)
        self.assertEqual(result["members"], 2)


class ScoreVsLrPlotTest(unittest.TestCase):
    def test_uses_total_score_and_lr_columns(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_score_vs_lr(temporary, manifest, rows, decisions)
        self.assertEqual(result["metric_keys"], [TOTAL_SCORE_COLUMN, "LR"])

    def test_generation_subsetting_warns_when_over_the_limit(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_score_vs_lr(temporary, manifest, rows, decisions, max_generations=2)
        self.assertEqual(result["generations"], 2)
        self.assertTrue(any("omitted" in warning for warning in result["warnings"]))


class LrPopulationPlotTest(unittest.TestCase):
    """Requirements: LR center matches persisted new_lr_center; member LR
    points match persisted assigned LRs; returns None outside
    anchor_copy_lr_recenter."""

    def test_none_for_non_anchor_copy_strategy(self):
        manifest = synthetic_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        self.assertEqual(decisions, [])
        self.assertIsNone(plot_lr_population("/nonexistent", manifest, rows, decisions))

    def test_lr_center_values_match_the_persisted_new_lr_center(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        expected_centers = [generation["anchor_copy_lr_recenter"]["new_lr_center"] for generation in manifest["generations"]]
        actual_centers = [decision["new_lr_center"] for decision in decisions]
        self.assertEqual(actual_centers, expected_centers)

    def test_member_lr_values_match_the_persisted_assigned_lrs(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        for generation, decision in zip(manifest["generations"], decisions):
            self.assertEqual(decision["assigned_lrs"], generation["anchor_copy_lr_recenter"]["assigned_lrs"])

    def test_renders_a_real_figure_with_a_spread_collapsed_generation_present(self):
        manifest = _anchor_copy_manifest()  # generation 1 has spread_collapsed=True
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_lr_population(temporary, manifest, rows, decisions)
            self.assertTrue(Path(result["png"]).is_file())


class DecisionHistoryPlotTest(unittest.TestCase):
    def test_none_for_non_anchor_copy_strategy(self):
        self.assertIsNone(plot_decision_history("/nonexistent", synthetic_manifest(), []))

    def test_covers_all_three_decision_types(self):
        manifest = _anchor_copy_manifest()
        rows = build_member_metric_rows(manifest)
        decisions = build_generation_decision_rows(manifest, rows)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_decision_history(temporary, manifest, decisions)
            self.assertEqual(result["generations"], 3)
            self.assertTrue(Path(result["png"]).is_file())


class BaselineRatioPlotTest(unittest.TestCase):
    """Requirement: baseline ratios use the same baseline checkpoint for
    every metric (all ratios computed against the one measured-baseline
    evaluation, never mixed sources)."""

    def test_ratio_uses_the_same_baseline_source_for_every_metric(self):
        manifest = _anchor_copy_manifest()
        manifest["best"] = {
            "generation": 2, "member": "m_a", "metrics": fixed_curve_metrics(0.5, 0.95, 0.99, 0.1), "lr": 3.0e-6,
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_baseline_ratio(temporary, manifest)
            self.assertTrue(Path(result["png"]).is_file())
            self.assertIn(TOTAL_SCORE_COLUMN, result["metric_keys"])
        # 8 raw + 3 aggregate columns, all sourced from the one baseline
        # evaluation and the one final-selected (manifest["best"]) evaluation.
        self.assertEqual(len(result["metric_keys"]), 11)

    def test_missing_baseline_or_final_produces_a_warning_not_a_crash(self):
        manifest = synthetic_manifest(measured_baseline=False)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_baseline_ratio(temporary, manifest)
        self.assertIsNone(result["png"])
        self.assertTrue(result["warnings"])


class DeterministicOutputTest(unittest.TestCase):
    def test_research_plot_filenames_are_stable_across_two_runs(self):
        manifest = _anchor_copy_manifest()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_results = write_research_plots(first, manifest)
            second_results = write_research_plots(second, manifest)
            first_names = {key: Path(value["png"]).name for key, value in first_results.items()}
            second_names = {key: Path(value["png"]).name for key, value in second_results.items()}
            self.assertEqual(first_names, second_names)

    def test_filenames_match_the_canonical_registry(self):
        manifest = _anchor_copy_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            results = write_research_plots(temporary, manifest)
        for key, base in RESEARCH_PLOT_NAMES.items():
            if key not in results:
                self.assertIn(key, CONDITIONAL_RESEARCH_PLOT_NAMES)
                continue
            self.assertEqual(Path(results[key]["png"]).name, f"{base}.png")
            self.assertNotIn("pdf", results[key])

    def test_no_combined_contact_sheet_output_exists(self):
        # Every research figure is its own standalone PNG; nothing in this
        # repo combines them into one image.
        for base in RESEARCH_PLOT_NAMES.values():
            self.assertNotIn("contact", base.lower())
            self.assertNotIn("sheet", base.lower())
            self.assertNotIn("dashboard", base.lower())


class LongRunReadabilityTest(unittest.TestCase):
    """Requirement: figures remain readable for long runs (many
    generations) without crashing or producing an unreadable
    small-multiples layout."""

    def _many_generation_manifest(self, n):
        manifest = _anchor_copy_manifest()
        base_generation = manifest["generations"][0]
        generations = []
        for index in range(n):
            generation = {
                **base_generation,
                "index": index,
                "epoch": 20 + index,
                "anchor_copy_lr_recenter": {**base_generation["anchor_copy_lr_recenter"]},
            }
            generations.append(generation)
        manifest["generations"] = generations
        return manifest

    def test_twenty_generations_render_without_error(self):
        manifest = self._many_generation_manifest(20)
        with tempfile.TemporaryDirectory() as temporary:
            results = write_research_plots(temporary, manifest)
            self.assertTrue(Path(results["score_vs_lr"]["png"]).is_file())
        self.assertLessEqual(results["score_vs_lr"]["generations"], 12)  # default max_generations subset

    def test_three_members_up_to_ten_supported(self):
        manifest = _anchor_copy_manifest()
        manifest["members"]["m_c"] = {"name": "m_c", "lr": 5.0e-6, "parent": None}
        for generation in manifest["generations"]:
            generation["workers"]["m_c"] = {"status": "completed", "lr": 5.0e-6, "metrics": fixed_curve_metrics(1.2, 0.85, 0.95, 0.35)}
            generation["anchor_copy_lr_recenter"]["assigned_lrs"]["m_c"] = 5.0e-6
        with tempfile.TemporaryDirectory() as temporary:
            results = write_research_plots(temporary, manifest)
        self.assertEqual(results["ctag_working_points"]["members"], 3)


class OldFormatManifestTest(unittest.TestCase):
    """Requirement: old manifests (predating the aggregate score fields,
    but with full raw curve data) still produce every figure via
    reconstruction, with no warnings when reconstruction succeeds."""

    def test_old_format_manifest_reconstructs_and_renders_every_figure(self):
        manifest = synthetic_manifest()  # fixed_curve_metrics never sets the aggregate keys
        with tempfile.TemporaryDirectory() as temporary:
            results = write_research_plots(temporary, manifest)
            for key in ("ctag_working_points", "btag_working_points", "aggregate_scores", "tag_tradeoff", "score_vs_lr"):
                self.assertTrue(Path(results[key]["png"]).is_file(), key)
                self.assertEqual(results[key]["warnings"], [], key)


if __name__ == "__main__":
    unittest.main()

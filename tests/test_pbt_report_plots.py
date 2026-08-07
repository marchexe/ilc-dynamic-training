import tempfile
import unittest
from pathlib import Path

from tests.test_pbt_artifacts import fixed_curve_metrics, synthetic_manifest
from tests.test_pbt_research_plots import _anchor_copy_manifest
from training.pbt.reporting.constants import (
    BTAG_SCORE_COLUMN,
    CTAG_SCORE_COLUMN,
    REPORT_PLOT_NAMES,
    TOTAL_GEOMEAN_METRIC_KEY,
    TOTAL_SCORE_COLUMN,
)
from training.pbt.reporting.report_plots import (
    _contiguous_runs,
    plot_learning_rate_lineage,
    plot_learning_rate_mistag_correlation,
    plot_mistag_score_evolution,
    plot_pbt_population_selection,
    plot_proxy_validation,
    write_report_plots,
)
from training.pbt.reporting.research_plots import (
    build_generation_decision_rows,
    build_member_metric_rows,
    shared_lr_center_series,
)
from training.pbt.reporting.statistics import lr_mistag_correlation


def _exploit_mutate_manifest():
    manifest = synthetic_manifest()
    manifest["config"]["pbt"]["strategy"] = "exploit_mutate"
    manifest["config"]["pbt"]["min_lr"] = 1.0e-5
    manifest["config"]["pbt"]["max_lr"] = 5.0e-4
    manifest["members"] = {
        "member_00": {"name": "member_00", "lr": 1.0e-4, "parent": None},
        "member_01": {"name": "member_01", "lr": 2.0e-4, "parent": None},
    }
    manifest["generations"] = [
        {
            "index": 0,
            "epoch": 0,
            "status": "completed",
            "workers": {
                "member_00": {"status": "completed", "lr": 1.0e-4, "metrics": fixed_curve_metrics(0.8, 0.9, 0.97, 0.3)},
                "member_01": {"status": "completed", "lr": 2.0e-4, "metrics": fixed_curve_metrics(1.4, 0.85, 0.94, 0.35)},
            },
            "ranking": ["member_00", "member_01"],
            "exploit": [],
        },
        {
            "index": 1,
            "epoch": 1,
            "status": "completed",
            "workers": {
                "member_00": {"status": "completed", "lr": 1.0e-4, "metrics": fixed_curve_metrics(0.75, 0.91, 0.975, 0.28)},
                "member_01": {"status": "completed", "lr": 8.0e-5, "metrics": fixed_curve_metrics(0.9, 0.9, 0.96, 0.29)},
            },
            "ranking": ["member_00", "member_01"],
            "exploit": [],
        },
    ]
    manifest["best"]["generation"] = 1
    manifest["best"]["member"] = "member_00"
    return manifest


_EXPLOIT_MUTATE_APPLIED_EVENT = {
    "event_type": "exploit", "source": "population", "generation": 0, "applied": True,
    "donor": "member_00", "recipient": "member_01", "old_lr": 2.0e-4, "new_lr": 8.0e-5,
}
_EXPLOIT_MUTATE_UNAPPLIED_EVENT = {**_EXPLOIT_MUTATE_APPLIED_EVENT, "applied": False}


def _fixed_lr_grid_manifest():
    manifest = synthetic_manifest()
    manifest["config"]["pbt"]["strategy"] = "fixed_lr_grid"
    manifest["config"]["pbt"]["min_lr"] = 1.0e-5
    manifest["config"]["pbt"]["max_lr"] = 5.0e-4
    for generation in manifest["generations"]:
        generation["exploit"] = []
    return manifest


def _single_member_manifest():
    manifest = synthetic_manifest()
    manifest["members"] = {"trial_a": manifest["members"]["trial_a"]}
    for generation in manifest["generations"]:
        generation["workers"] = {"trial_a": generation["workers"]["trial_a"]}
        generation["ranking"] = ["trial_a"]
        generation["exploit"] = []
    manifest["best"]["member"] = "trial_a"
    manifest["best"]["generation"] = 1
    return manifest


def _many_generation_manifest(n):
    manifest = _anchor_copy_manifest()
    base = manifest["generations"][0]
    manifest["generations"] = [
        {
            **base,
            "index": index,
            "epoch": 20 + index,
            "anchor_copy_lr_recenter": {**base["anchor_copy_lr_recenter"]},
            "exploit": [dict(event) for event in base["exploit"]],
        }
        for index in range(n)
    ]
    return manifest


def _rows_and_decisions(manifest):
    rows = build_member_metric_rows(manifest)
    decisions = build_generation_decision_rows(manifest, rows)
    return rows, decisions


def _group_by_generation(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["generation"], []).append(row)
    return grouped


class PopulationSelectionPlotTest(unittest.TestCase):
    def test_produces_a_png_no_pdf(self):
        manifest = _anchor_copy_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_pbt_population_selection(temporary, manifest, rows, decisions)
            self.assertTrue(Path(result["png"]).is_file())
            self.assertEqual(Path(result["png"]).suffix, ".png")
        self.assertNotIn("pdf", result)

    def test_all_members_present(self):
        manifest = _anchor_copy_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_pbt_population_selection(temporary, manifest, rows, decisions)
        self.assertEqual(result["members"], len(manifest["members"]))

    def test_winner_marker_matches_authoritative_decision_winner_for_multiple_strategies(self):
        for manifest in (_anchor_copy_manifest(), _exploit_mutate_manifest(), _fixed_lr_grid_manifest()):
            with self.subTest(strategy=manifest["config"]["pbt"]["strategy"]):
                rows, decisions = _rows_and_decisions(manifest)
                winner_by_generation = {row["generation"]: row["trial"] for row in rows if row["is_winner"]}
                with tempfile.TemporaryDirectory() as temporary:
                    result = plot_pbt_population_selection(temporary, manifest, rows, decisions)
                    self.assertTrue(Path(result["png"]).is_file())
                self.assertTrue(winner_by_generation)

    def test_missing_generation_creates_a_gap_not_a_fabricated_point(self):
        manifest = _anchor_copy_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        rows = [row for row in rows if not (row["trial"] == "m_a" and row["generation"] == 1)]
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_pbt_population_selection(temporary, manifest, rows, decisions)
            self.assertTrue(Path(result["png"]).is_file())

        m_a_rows = [row for row in rows if row["trial"] == "m_a"]
        runs = _contiguous_runs(m_a_rows)
        self.assertEqual(len(runs), 2)
        self.assertEqual([row["generation"] for row in runs[0]], [0])
        self.assertEqual([row["generation"] for row in runs[1]], [2])

    def test_genuine_zero_score_not_treated_as_missing(self):
        manifest = _anchor_copy_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        for row in rows:
            if row["trial"] == "m_a" and row["generation"] == 0:
                row["optimization_metric_value"] = 0.0
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_pbt_population_selection(temporary, manifest, rows, decisions)
        self.assertEqual(result["generations"], 3)
        self.assertEqual(result["members"], 2)

    def test_anchor_copy_lr_recenter_bottom_panel_has_three_decision_marker_types(self):
        manifest = _anchor_copy_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        self.assertEqual({d["decision"] for d in decisions}, {"accepted_new_anchor", "reused_previous_anchor", "rewound_to_previous_anchor"})
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_pbt_population_selection(temporary, manifest, rows, decisions)
            self.assertTrue(Path(result["png"]).is_file())

    def test_no_data_returns_none_png(self):
        manifest = synthetic_manifest()
        manifest["generations"] = []
        rows, decisions = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_pbt_population_selection(temporary, manifest, rows, decisions)
        self.assertIsNone(result["png"])
        self.assertEqual(result["generations"], 0)
        self.assertEqual(result["members"], 0)


class MistagScoreEvolutionPlotTest(unittest.TestCase):
    def test_three_lines_present_ctag_btag_total(self):
        manifest = _anchor_copy_manifest()
        rows, _ = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_mistag_score_evolution(temporary, manifest, rows)
            self.assertTrue(Path(result["png"]).is_file())
        self.assertEqual(result["metric_keys"], [CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN, TOTAL_SCORE_COLUMN])

    def test_measured_baseline_becomes_the_baseline_point(self):
        manifest = synthetic_manifest(measured_baseline=True)
        rows, _ = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_mistag_score_evolution(temporary, manifest, rows)
        self.assertIs(result["has_baseline_point"], True)

    def test_no_baseline_still_renders(self):
        manifest = synthetic_manifest(measured_baseline=False)
        rows, _ = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_mistag_score_evolution(temporary, manifest, rows)
            self.assertTrue(Path(result["png"]).is_file())
        self.assertIs(result["has_baseline_point"], False)

    def test_ranking_metric_is_total_score_flag_true_when_configured(self):
        manifest = synthetic_manifest()
        manifest["config"]["pbt"]["metric"] = TOTAL_GEOMEAN_METRIC_KEY
        rows, _ = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_mistag_score_evolution(temporary, manifest, rows)
        self.assertIs(result["ranking_metric_is_total_score"], True)

    def test_ranking_metric_is_total_score_flag_false_otherwise_and_winner_by_real_metric(self):
        manifest = synthetic_manifest()  # metric=validation_working_point_mistag_percent
        rows, _ = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_mistag_score_evolution(temporary, manifest, rows)
        self.assertIs(result["ranking_metric_is_total_score"], False)
        winners = {row["generation"]: row["trial"] for row in rows if row["is_winner"]}
        for generation, group in _group_by_generation(rows).items():
            expected = min(group, key=lambda row: row["optimization_metric_value"])["trial"]
            self.assertEqual(winners[generation], expected)


class LearningRateLineagePlotTest(unittest.TestCase):
    def test_applied_donor_recipient_produces_a_heavy_edge(self):
        manifest = _exploit_mutate_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        center_series = shared_lr_center_series(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_lineage(temporary, manifest, rows, decisions, center_series, [_EXPLOIT_MUTATE_APPLIED_EVENT])
        copy_edges = [edge for edge in result["edges"] if edge[4] == "copy"]
        self.assertEqual(copy_edges, [("member_00", 0, "member_01", 1, "copy")])

    def test_unapplied_or_skipped_exploit_produces_no_edge(self):
        manifest = _exploit_mutate_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        center_series = shared_lr_center_series(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_lineage(temporary, manifest, rows, decisions, center_series, [_EXPLOIT_MUTATE_UNAPPLIED_EVENT])
        copy_edges = [edge for edge in result["edges"] if edge[4] == "copy"]
        self.assertEqual(copy_edges, [])
        self.assertTrue(all(edge[4] == "self" for edge in result["edges"]))

    def test_anchor_copy_lr_recenter_edges_use_assigned_lrs_and_persisted_anchor(self):
        manifest = _anchor_copy_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        center_series = shared_lr_center_series(manifest)
        # Deliberately NO __anchor__ events in this events list -- the
        # anchor pseudo-recipient is never written to events.jsonl (see
        # state/transitions.py::apply_exploit's bundle special-case), so
        # the anchor line must come from center_series/decision_rows alone.
        events = [dict(event, event_type="exploit") for generation in manifest["generations"] for event in generation["exploit"]]
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_lineage(temporary, manifest, rows, decisions, center_series, events)
            self.assertTrue(Path(result["png"]).is_file())
        self.assertEqual(center_series, [(g["index"], g["anchor_copy_lr_recenter"]["new_lr_center"]) for g in manifest["generations"]])
        lr_by_trial_generation = {(row["trial"], row["generation"]): row["LR"] for row in rows}
        last_generation = max(generation["index"] for generation in manifest["generations"])
        for generation in manifest["generations"]:
            if generation["index"] == last_generation:
                continue  # the LR assigned this generation trains *next* generation, which doesn't exist yet
            for member, lr in generation["anchor_copy_lr_recenter"]["assigned_lrs"].items():
                self.assertEqual(lr_by_trial_generation.get((member, generation["index"] + 1)), lr)

    def test_fixed_lr_grid_produces_independent_branches_no_copy_edges(self):
        manifest = _fixed_lr_grid_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        center_series = shared_lr_center_series(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_lineage(temporary, manifest, rows, decisions, center_series, [])
        self.assertTrue(result["edges"])
        self.assertTrue(all(edge[4] == "self" for edge in result["edges"]))
        self.assertEqual(center_series, [])

    def test_single_member_run_renders(self):
        manifest = _single_member_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        center_series = shared_lr_center_series(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_lineage(temporary, manifest, rows, decisions, center_series, [])
            self.assertTrue(Path(result["png"]).is_file())
        self.assertEqual(result["members"], 1)

    def test_zero_exploit_events_renders(self):
        manifest = _anchor_copy_manifest()
        rows, decisions = _rows_and_decisions(manifest)
        center_series = shared_lr_center_series(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_lineage(temporary, manifest, rows, decisions, center_series, [])
            self.assertTrue(Path(result["png"]).is_file())
        self.assertTrue(all(edge[4] == "self" for edge in result["edges"]))

    def test_min_max_lr_lines_present_for_every_strategy(self):
        for manifest in (_anchor_copy_manifest(), _exploit_mutate_manifest(), _fixed_lr_grid_manifest()):
            with self.subTest(strategy=manifest["config"]["pbt"]["strategy"]):
                self.assertIn("min_lr", manifest["config"]["pbt"])
                self.assertIn("max_lr", manifest["config"]["pbt"])
                rows, decisions = _rows_and_decisions(manifest)
                center_series = shared_lr_center_series(manifest)
                with tempfile.TemporaryDirectory() as temporary:
                    result = plot_learning_rate_lineage(temporary, manifest, rows, decisions, center_series, [])
                    self.assertTrue(Path(result["png"]).is_file())

    def test_spread_collapse_marker_present_only_where_flagged(self):
        manifest = _anchor_copy_manifest()  # generation 1 has spread_collapsed=True
        rows, decisions = _rows_and_decisions(manifest)
        collapsed = [decision["generation"] for decision in decisions if decision["spread_collapsed"]]
        self.assertEqual(collapsed, [1])


class LrMistagCorrelationStatsTest(unittest.TestCase):
    def test_insufficient_paired_observations_below_three(self):
        manifest = _single_member_manifest()  # 1 member x 2 generations = 2 rows
        rows, _ = _rows_and_decisions(manifest)
        result = lr_mistag_correlation(rows)
        self.assertEqual(result["n"], 2)
        self.assertIsNone(result["pearson_r"])
        self.assertEqual(result["reason"], "insufficient_paired_observations")

    def test_zero_variance_total_mistag_score_reports_named_reason(self):
        # _anchor_copy_manifest() reuses fixed_curve_metrics()'s fixed
        # rejection curves for every generation/member (see
        # tests/test_pbt_artifacts.py's identical caveat about
        # total_mistag_score), so total_mistag_score is the exact same
        # constant across all 6 rows here -- scipy would return nan without
        # raising; this must surface as a named reason, never a fabricated
        # coefficient (nan silently passing an isinstance(..., float) check).
        manifest = _anchor_copy_manifest()  # 2 members x 3 generations = 6 rows
        rows, _ = _rows_and_decisions(manifest)
        result = lr_mistag_correlation(rows)
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["reason"], "zero_variance")
        self.assertIsNone(result["pearson_r"])
        self.assertIsNone(result["spearman_rho"])

    def test_correlation_computed_with_varying_scores(self):
        manifest = _anchor_copy_manifest()
        rows, _ = _rows_and_decisions(manifest)
        rows = [dict(row) for row in rows]
        # Monotonic in LR by construction, so the sign is known ahead of
        # time -- a real assertion on the computed value, not just "did not
        # crash".
        for row in rows:
            row[TOTAL_SCORE_COLUMN] = row["LR"] * 1.0e4
        result = lr_mistag_correlation(rows)
        self.assertEqual(result["n"], 6)
        self.assertIsNone(result["reason"])
        self.assertGreater(result["pearson_r"], 0.9)
        self.assertGreater(result["spearman_rho"], 0.9)

    def test_rows_with_missing_or_nonpositive_lr_are_excluded(self):
        manifest = _anchor_copy_manifest()
        rows, _ = _rows_and_decisions(manifest)
        rows = [dict(row) for row in rows]
        rows[0]["LR"] = None
        rows[1]["LR"] = 0.0
        result = lr_mistag_correlation(rows)
        self.assertEqual(result["n"], 4)


class LearningRateMistagCorrelationPlotTest(unittest.TestCase):
    def test_renders_and_reports_correlation_metric_keys(self):
        manifest = _anchor_copy_manifest()
        rows, _ = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_mistag_correlation(temporary, manifest, rows)
            self.assertTrue(Path(result["png"]).is_file())
        self.assertEqual(result["metric_keys"], ["LR", TOTAL_SCORE_COLUMN])
        self.assertEqual(result["correlation"]["n"], 6)

    def test_renders_with_a_real_computed_correlation(self):
        manifest = _anchor_copy_manifest()
        rows, _ = _rows_and_decisions(manifest)
        for row in rows:
            row[TOTAL_SCORE_COLUMN] = row["LR"] * 1.0e4
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_mistag_correlation(temporary, manifest, rows)
            self.assertTrue(Path(result["png"]).is_file())
        self.assertIsNone(result["correlation"]["reason"])
        self.assertGreater(result["correlation"]["pearson_r"], 0.9)

    def test_single_member_run_still_renders_with_insufficient_correlation(self):
        manifest = _single_member_manifest()
        rows, _ = _rows_and_decisions(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_mistag_correlation(temporary, manifest, rows)
            self.assertTrue(Path(result["png"]).is_file())
        self.assertEqual(result["correlation"]["reason"], "insufficient_paired_observations")

    def test_no_lr_data_returns_none_png(self):
        manifest = _anchor_copy_manifest()
        rows, _ = _rows_and_decisions(manifest)
        for row in rows:
            row["LR"] = None
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_learning_rate_mistag_correlation(temporary, manifest, rows)
        self.assertIsNone(result["png"])
        self.assertEqual(result["generations"], 0)


class ProxyValidationPlotTest(unittest.TestCase):
    def _manifest_with_tiers(self, tiers):
        manifest = _anchor_copy_manifest()
        manifest["tiered_evaluations"] = []
        for generation in manifest["generations"]:
            index = generation["index"]
            for tier in tiers:
                manifest["tiered_evaluations"].append(
                    {
                        "generation": index,
                        "tier": tier,
                        "dataset": "d",
                        "suffix": "s",
                        "metric_name": "validation_working_point_mistag_percent",
                        "mode": "min",
                        "members": {
                            "m_a": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.0 + 0.1 * index}},
                            "m_b": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 0.5 + 0.1 * index}},
                        },
                        # Deliberately opposite of the decision winner, so a
                        # left panel that (incorrectly) used this round's own
                        # ranking[0] instead of the decision winner would
                        # plot a different value.
                        "ranking": ["m_b", "m_a"],
                    }
                )
        return manifest

    def test_none_when_no_monitor_or_full_holdout_tiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(plot_proxy_validation(temporary, _anchor_copy_manifest()))
            self.assertIsNone(plot_proxy_validation(temporary, self._manifest_with_tiers(["control", "full"])))

    def test_present_with_monitor_and_full_holdout(self):
        manifest = self._manifest_with_tiers(["control", "monitor", "full_holdout"])
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_proxy_validation(temporary, manifest)
            self.assertIsNotNone(result)
            self.assertTrue(Path(result["png"]).is_file())
        self.assertIs(result["independent"], True)
        self.assertEqual(result["tier_b"], "full_holdout")

    def test_falls_back_to_monitor_and_annotates_not_independent_when_full_holdout_absent(self):
        manifest = self._manifest_with_tiers(["control", "monitor"])
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_proxy_validation(temporary, manifest)
        self.assertIs(result["independent"], False)
        self.assertEqual(result["tier_b"], "monitor")

    def test_left_panel_uses_decision_winner_value_not_plain_round_ranking(self):
        manifest = self._manifest_with_tiers(["control", "monitor"])
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_proxy_validation(temporary, manifest)
        # Generation 0's decision winner is m_a (see _anchor_copy_manifest's
        # generation-0 "ranking": ["m_a", "m_b"]) -> control value 1.0, not
        # m_b's 0.5 (which the round's own reversed "ranking" would imply).
        self.assertIn("control", result["winner_series"])
        self.assertIn((0, 1.0), result["winner_series"]["control"])
        self.assertNotIn((0, 0.5), result["winner_series"]["control"])


class OrchestrationTest(unittest.TestCase):
    def _tiered_manifest(self):
        manifest = _anchor_copy_manifest()
        manifest["tiered_evaluations"] = [
            {
                "generation": 0,
                "tier": "monitor",
                "dataset": "d",
                "suffix": "s",
                "metric_name": "validation_working_point_mistag_percent",
                "mode": "min",
                "members": {"m_a": {"status": "completed", "metrics": {"validation_working_point_mistag_percent": 1.0}}},
                "ranking": ["m_a"],
            }
        ]
        return manifest

    def test_write_report_plots_produces_all_five_core_names(self):
        manifest = self._tiered_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            results = write_report_plots(temporary, manifest)
            for key in (
                "pbt_population_selection",
                "mistag_score_evolution",
                "learning_rate_lineage",
                "learning_rate_mistag_correlation",
                "proxy_validation",
            ):
                self.assertIn(key, results)
                self.assertEqual(Path(results[key]["png"]).name, f"{REPORT_PLOT_NAMES[key]}.png")
                self.assertTrue(Path(results[key]["png"]).is_file())
                self.assertNotIn("pdf", results[key])

    def test_write_report_plots_omits_proxy_validation_key_when_absent(self):
        manifest = _anchor_copy_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            results = write_report_plots(temporary, manifest)
        self.assertNotIn("proxy_validation", results)
        for key in ("pbt_population_selection", "mistag_score_evolution", "learning_rate_lineage", "learning_rate_mistag_correlation"):
            self.assertIn(key, results)

    def test_no_combined_contact_sheet_output_exists(self):
        for base in REPORT_PLOT_NAMES.values():
            self.assertNotIn("contact", base.lower())
            self.assertNotIn("sheet", base.lower())
            self.assertNotIn("dashboard", base.lower())

    def test_filenames_stable_across_two_runs(self):
        manifest = _anchor_copy_manifest()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_results = write_report_plots(first, manifest)
            second_results = write_report_plots(second, manifest)
            first_names = {key: Path(value["png"]).name for key, value in first_results.items()}
            second_names = {key: Path(value["png"]).name for key, value in second_results.items()}
        self.assertEqual(first_names, second_names)

    def test_fifty_generation_run_stays_readable(self):
        manifest = _many_generation_manifest(52)
        with tempfile.TemporaryDirectory() as temporary:
            results = write_report_plots(temporary, manifest)
            for key in (
                "pbt_population_selection",
                "mistag_score_evolution",
                "learning_rate_lineage",
                "learning_rate_mistag_correlation",
            ):
                self.assertTrue(Path(results[key]["png"]).is_file(), key)
                self.assertEqual(results[key]["generations"], 52)


if __name__ == "__main__":
    unittest.main()

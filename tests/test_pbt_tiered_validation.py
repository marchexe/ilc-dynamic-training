"""Tests for the control/monitor/full proxy-validation tier machinery:
paired tiered_metrics.csv, correlation/ranking-agreement diagnostics,
best-checkpoint agreement, proxy-overfitting detection, corroboration
labeling, and the finite-metric guard used before a worker's result is
accepted into ranking.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_DIR  # noqa: F401
from training.pbt.artifacts import (
    _proxy_diagnostics_report_lines,
    best_checkpoint_by_tier,
    corroboration_status,
    proxy_overfitting_cases,
    proxy_selected_checkpoint_other_tiers,
    ranking_agreement,
    record_tiered_evaluation_round,
    tier_correlation,
    tiered_evaluation_rows,
    write_canonical_outputs,
    write_tiered_metrics_csv,
)
from training.pbt.execution.backend import finite_metric_ok


METRIC = "validation_working_point_mistag_percent"


def curves_for(value):
    # A minimal-but-complete curve set the fixed-WP helpers can parse, with
    # matching background pass/total counts for Wilson uncertainty.
    efficiencies = [0.5, 0.8, 0.9]
    pairs = {
        "bc": [800.0, 500.0, 40.0],
        "bd": [900.0, 700.0, 120.0],
        "cb": [250.0, 80.0, 20.0],
        "cd": [900.0, 120.0, 12.0],
    }
    total = 20000
    counts = {}
    for pair, rejections in pairs.items():
        counts[pair] = [
            {
                "signal_efficiency": eff,
                "background_passed": max(1, round(total / rejection)),
                "background_total": total,
                "background_efficiency": max(1, round(total / rejection)) / total,
            }
            for eff, rejection in zip(efficiencies, rejections)
        ]
    return {
        METRIC: value,
        "validation_bkg_rejection_at_eff": {"efficiencies": efficiencies, "pairs": pairs},
        "validation_bkg_rejection_at_eff_counts": counts,
    }


def base_manifest():
    return {
        "schema_version": 1,
        "experiment": "tiered_test",
        "status": "completed",
        "method": "exploit_mutate",
        "run": {
            "method_name": "exploit_mutate",
            "datasets": {},
            "schedule": {
                "training_interval": {"epochs_per_generation": 1, "samples_per_epoch": 100, "samples_per_trial_chunk": 100},
                "evaluation_interval": {"training_chunks": 1},
                "exploit_interval": {"enabled": True, "training_chunks": 1},
            },
        },
        "datasets": {},
        "checkpoint": {"path": "checkpoint.pt", "sha256": "abc"},
        "initial_evaluation": {"status": "completed", "checkpoint": "checkpoint.pt", "metrics": curves_for(1.5)},
        "config": {
            "shared": {
                "samples_per_epoch": 100,
                "epochs_per_generation": 1,
                "samples_per_epoch_val": 3000,
                "validation_dataset": "control_ds",
                "validation_suffix": "val5k_tail",
            },
            "pbt": {"metric": METRIC, "mode": "min", "strategy": "exploit_mutate", "baseline_metric_value": 1.5},
        },
        "members": {"member_00": {"name": "member_00", "lr": 1e-4, "parent": None}, "member_01": {"name": "member_01", "lr": 1e-4, "parent": None}},
        "generations": [
            {
                "index": 0,
                "epoch": 0,
                "status": "completed",
                "workers": {
                    "member_00": {"status": "completed", "lr": 1e-4, "metrics": curves_for(1.2)},
                    "member_01": {"status": "completed", "lr": 1e-4, "metrics": curves_for(1.3)},
                },
                "ranking": ["member_00", "member_01"],
                "exploit": [],
            }
        ],
        "best": {
            "generation": 0,
            "epoch": 0,
            "member": "member_00",
            "metric": METRIC,
            "metric_value": 1.2,
            "lr": 1e-4,
            "metrics": curves_for(1.2),
            "state_path": "checkpoints/global_best_state.pt",
            "optimizer_path": "checkpoints/global_best_optimizer.pt",
            "metadata_path": "checkpoints/global_best_metadata.json",
        },
        "tiered_evaluations": [],
    }


class FiniteMetricOkTest(unittest.TestCase):
    def test_rejects_missing_none_nan_and_inf(self):
        self.assertFalse(finite_metric_ok(None, METRIC))
        self.assertFalse(finite_metric_ok({}, METRIC))
        self.assertFalse(finite_metric_ok({METRIC: None}, METRIC))
        self.assertFalse(finite_metric_ok({METRIC: float("nan")}, METRIC))
        self.assertFalse(finite_metric_ok({METRIC: float("inf")}, METRIC))

    def test_accepts_finite_value(self):
        self.assertTrue(finite_metric_ok({METRIC: 1.23}, METRIC))


class RecordTieredEvaluationRoundTest(unittest.TestCase):
    def test_ranking_reflects_mode_and_excludes_missing_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = {}
            member_results = {
                "member_00": {"status": "completed", "metrics": {METRIC: 1.2}},
                "member_01": {"status": "completed", "metrics": {METRIC: 0.9}},
                "member_02": {"status": "failed", "metrics": None},
            }

            round_record = record_tiered_evaluation_round(
                run_dir, manifest, {"index": 0}, "monitor", "monitor_ds", "val50k_tail", member_results, METRIC, "min"
            )

            self.assertEqual(round_record["ranking"], ["member_01", "member_00"])
            self.assertEqual(manifest["tiered_evaluations"], [round_record])
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["event_type"], "tiered_evaluation")
            self.assertEqual(events[0]["ranking"], ["member_01", "member_00"])

    def test_max_mode_ranks_higher_value_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = {}
            member_results = {
                "member_00": {"status": "completed", "metrics": {"validation_bkg_rejection_score": 5.0}},
                "member_01": {"status": "completed", "metrics": {"validation_bkg_rejection_score": 9.0}},
            }

            round_record = record_tiered_evaluation_round(
                run_dir, manifest, {"index": 0}, "full", "full_ds", "val1000k",
                member_results, "validation_bkg_rejection_score", "max",
            )

            self.assertEqual(round_record["ranking"], ["member_01", "member_00"])


class TierCorrelationAndAgreementTest(unittest.TestCase):
    def _manifest_with_paired_rounds(self):
        manifest = base_manifest()
        # Three paired (control, monitor) generations with a clean positive
        # relationship so Pearson/Spearman are well defined and non-trivial.
        for generation, (c0, c1, m0, m1) in enumerate([(1.0, 1.4, 1.1, 1.5), (0.8, 1.2, 0.9, 1.3), (0.6, 1.0, 0.7, 1.1)]):
            gen_record = {"index": generation}
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "control", "control_ds", "val5k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(c0)}, "member_01": {"status": "completed", "metrics": curves_for(c1)}},
                METRIC, "min",
            )
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "monitor", "monitor_ds", "val50k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(m0)}, "member_01": {"status": "completed", "metrics": curves_for(m1)}},
                METRIC, "min",
            )
        return manifest

    def test_correlation_reports_insufficient_data_below_three_points(self):
        manifest = base_manifest()
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "control", "control_ds", "val5k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.0)}}, METRIC, "min",
        )
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.1)}}, METRIC, "min",
        )

        result = tier_correlation(manifest, "control", "monitor")

        self.assertEqual(result["n"], 1)
        self.assertIsNone(result["pearson_r"])
        self.assertEqual(result["reason"], "insufficient_paired_observations")

    def test_correlation_computed_with_enough_paired_points(self):
        manifest = self._manifest_with_paired_rounds()

        result = tier_correlation(manifest, "control", "monitor")

        self.assertEqual(result["n"], 6)
        self.assertIsNone(result["reason"])
        self.assertGreater(result["pearson_r"], 0.9)
        self.assertGreater(result["spearman_rho"], 0.9)

    def test_ranking_agreement_detects_perfect_top1_and_overlap(self):
        manifest = self._manifest_with_paired_rounds()

        rows = ranking_agreement(manifest, "control", "monitor")

        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(row["top1_agrees"])
            self.assertEqual(row["top_k_overlap_fraction"], 1.0)

    def test_ranking_agreement_flags_disagreement(self):
        manifest = base_manifest()
        gen_record = {"index": 0}
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, gen_record, "control", "control_ds", "val5k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.0)}, "member_01": {"status": "completed", "metrics": curves_for(2.0)}},
            METRIC, "min",
        )
        # Monitor disagrees on the winner: member_01 is now best.
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, gen_record, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(2.0)}, "member_01": {"status": "completed", "metrics": curves_for(1.0)}},
            METRIC, "min",
        )

        rows = ranking_agreement(manifest, "control", "monitor")

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["top1_agrees"])


class BestCheckpointAndOverfittingTest(unittest.TestCase):
    def test_best_checkpoint_by_tier_respects_min_mode(self):
        manifest = base_manifest()
        gen0 = {"index": 0}
        gen1 = {"index": 1}
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, gen0, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.5)}}, METRIC, "min",
        )
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, gen1, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(0.8)}}, METRIC, "min",
        )

        best = best_checkpoint_by_tier(manifest)

        self.assertEqual(best["monitor"]["generation"], 1)
        self.assertAlmostEqual(best["monitor"]["metric_value"], 0.8)

    def test_proxy_overfitting_case_detected_when_control_improves_and_monitor_does_not(self):
        manifest = base_manifest()
        for generation, (control_value, monitor_value) in enumerate([(1.0, 1.0), (0.5, 1.3)]):
            gen_record = {"index": generation}
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "control", "control_ds", "val5k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(control_value)}}, METRIC, "min",
            )
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "monitor", "monitor_ds", "val50k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(monitor_value)}}, METRIC, "min",
            )

        cases = proxy_overfitting_cases(manifest)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["member"], "member_00")
        self.assertLess(cases[0]["control_after"], cases[0]["control_before"])
        self.assertGreaterEqual(cases[0]["monitor_after"], cases[0]["monitor_before"])

    def test_no_overfitting_case_when_both_tiers_improve_together(self):
        manifest = base_manifest()
        for generation, (control_value, monitor_value) in enumerate([(1.0, 1.0), (0.5, 0.6)]):
            gen_record = {"index": generation}
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "control", "control_ds", "val5k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(control_value)}}, METRIC, "min",
            )
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "monitor", "monitor_ds", "val50k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(monitor_value)}}, METRIC, "min",
            )

        self.assertEqual(proxy_overfitting_cases(manifest), [])


class CorroborationStatusTest(unittest.TestCase):
    def test_provisional_when_no_other_tier_data_exists(self):
        manifest = base_manifest()

        status, details = corroboration_status(manifest)

        self.assertEqual(status, "provisional")
        self.assertFalse(details["monitor"]["available"])
        self.assertFalse(details["full"]["available"])

    def test_monitor_corroborated_when_monitor_agrees(self):
        manifest = base_manifest()
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": -1}, "monitor", "monitor_ds", "val50k_tail",
            {"initial_resume": {"status": "completed", "metrics": curves_for(1.6)}}, METRIC, "min",
        )
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.0)}}, METRIC, "min",
        )

        status, details = corroboration_status(manifest)

        self.assertEqual(status, "monitor-corroborated")
        self.assertTrue(details["monitor"]["improved"])

    def test_not_corroborated_when_monitor_disagrees_with_control(self):
        manifest = base_manifest()
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": -1}, "monitor", "monitor_ds", "val50k_tail",
            {"initial_resume": {"status": "completed", "metrics": curves_for(1.0)}}, METRIC, "min",
        )
        # Monitor says the selected checkpoint is *worse* than baseline,
        # even though control (which drove selection) said 1.2 < 1.5.
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.8)}}, METRIC, "min",
        )

        status, details = corroboration_status(manifest)

        self.assertEqual(status, "provisional")
        self.assertFalse(details["monitor"]["improved"])


class TieredMetricsCsvTest(unittest.TestCase):
    def test_rows_include_rank_and_are_paired_by_generation_and_member(self):
        manifest = base_manifest()
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "control", "control_ds", "val5k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.0)}, "member_01": {"status": "completed", "metrics": curves_for(1.5)}},
            METRIC, "min",
        )
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.1)}, "member_01": {"status": "completed", "metrics": curves_for(1.6)}},
            METRIC, "min",
        )

        rows = tiered_evaluation_rows(manifest)
        by_key = {(row["generation"], row["tier"], row["member"]): row for row in rows}

        self.assertEqual(by_key[(0, "control", "member_00")]["rank"], 1)
        self.assertEqual(by_key[(0, "control", "member_01")]["rank"], 2)
        self.assertAlmostEqual(by_key[(0, "control", "member_00")]["metric_value"], 1.0)
        self.assertAlmostEqual(by_key[(0, "monitor", "member_00")]["metric_value"], 1.1)

    def test_write_tiered_metrics_csv_round_trips(self):
        manifest = base_manifest()
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "control", "control_ds", "val5k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.0)}}, METRIC, "min",
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            path = write_tiered_metrics_csv(run_dir, manifest)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["tier"], "control")
            self.assertEqual(rows[0]["member"], "member_00")


class ProxySelectedCheckpointOtherTiersTest(unittest.TestCase):
    def test_reports_monitor_value_for_the_control_selected_checkpoint(self):
        manifest = base_manifest()  # best = member_00, generation 0
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": 0}, "monitor", "monitor_ds", "val50k_tail",
            {"member_00": {"status": "completed", "metrics": curves_for(1.25)}}, METRIC, "min",
        )

        result = proxy_selected_checkpoint_other_tiers(manifest)

        self.assertEqual(result["member"], "member_00")
        self.assertEqual(result["generation"], 0)
        self.assertAlmostEqual(result["tiers"]["monitor"]["metric_value"], 1.25)

    def test_empty_when_selected_checkpoint_never_evaluated_on_other_tiers(self):
        manifest = base_manifest()

        result = proxy_selected_checkpoint_other_tiers(manifest)

        self.assertEqual(result["tiers"], {})


class FullHoldoutUsedNotPlainFullTest(unittest.TestCase):
    def test_report_correlation_uses_full_holdout_not_plain_full(self):
        # control and full_holdout move together (clean positive
        # correlation); "full" is deliberately given a scrambled,
        # uncorrelated sequence. If the diagnostics used plain "full" for
        # the fidelity check, the reported correlation would be weak/absent;
        # since it must use full_holdout instead, it should be strong.
        manifest = base_manifest()
        control_values = [1.0, 0.8, 0.6]
        holdout_values = [1.05, 0.82, 0.58]  # tracks control closely
        full_values = [0.5, 1.2, 0.7]  # scrambled, not correlated with control
        for generation, (c, h, f) in enumerate(zip(control_values, holdout_values, full_values)):
            gen_record = {"index": generation}
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "control", "control_ds", "val5k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(c)}}, METRIC, "min",
            )
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "full_holdout", "full_ds", "val_holdout",
                {"member_00": {"status": "completed", "metrics": curves_for(h)}}, METRIC, "min",
            )
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, gen_record, "full", "full_ds", "val1000k",
                {"member_00": {"status": "completed", "metrics": curves_for(f)}}, METRIC, "min",
            )

        holdout_correlation = tier_correlation(manifest, "control", "full_holdout")
        full_correlation = tier_correlation(manifest, "control", "full")
        self.assertGreater(holdout_correlation["pearson_r"], 0.95)
        self.assertLess(full_correlation["pearson_r"], 0.5)

        lines = _proxy_diagnostics_report_lines(manifest, {}, "plots/proxy_diagnostics.png")
        report_text = "\n".join(lines)
        self.assertIn("full_holdout", report_text)
        self.assertIn(f"n={holdout_correlation['n']}, Pearson r={holdout_correlation['pearson_r']:.3f}", report_text)
        self.assertNotIn(f"Pearson r={full_correlation['pearson_r']:.3f}", report_text)


class EndToEndTieredArtifactsTest(unittest.TestCase):
    def test_write_canonical_outputs_produces_proxy_diagnostics_when_tiers_present(self):
        manifest = base_manifest()
        record_tiered_evaluation_round(
            Path(tempfile.gettempdir()), manifest, {"index": -1}, "monitor", "monitor_ds", "val50k_tail",
            {"initial_resume": {"status": "completed", "metrics": curves_for(1.6)}}, METRIC, "min",
        )
        for generation, value in ((0, 1.25),):
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, {"index": generation}, "control", "control_ds", "val5k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(value)}, "member_01": {"status": "completed", "metrics": curves_for(value + 0.2)}},
                METRIC, "min",
            )
            record_tiered_evaluation_round(
                Path(tempfile.gettempdir()), manifest, {"index": generation}, "monitor", "monitor_ds", "val50k_tail",
                {"member_00": {"status": "completed", "metrics": curves_for(value + 0.05)}, "member_01": {"status": "completed", "metrics": curves_for(value + 0.25)}},
                METRIC, "min",
            )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            artifacts = write_canonical_outputs(run_dir, manifest)

            self.assertTrue((run_dir / "tiered_metrics.csv").is_file())
            self.assertTrue((run_dir / "plots" / "proxy_diagnostics.png").is_file())
            self.assertTrue((run_dir / "plots" / "report" / "skipped_exploits.csv").is_file())
            self.assertIn("proxy_diagnostics", artifacts["plots"])
            report = (run_dir / "report.md").read_text()
            self.assertIn("## Proxy Validation Diagnostics", report)
            self.assertIn("Corroboration status", report)

    def test_no_proxy_diagnostics_plot_when_no_tiers_recorded(self):
        manifest = base_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            artifacts = write_canonical_outputs(run_dir, manifest)

            self.assertNotIn("proxy_diagnostics", artifacts["plots"])
            self.assertFalse((run_dir / "plots" / "proxy_diagnostics.png").exists())
            report = (run_dir / "report.md").read_text()
            self.assertNotIn("## Proxy Validation Diagnostics", report)


if __name__ == "__main__":
    unittest.main()

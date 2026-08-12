import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_DIR  # noqa: F401
from training.pbt.state import checkpointing
from training.pbt.state import transitions
from training.pbt.reporting import (
    append_event,
    fixed_working_point_uncertainty,
    format_mistag_value,
    run_contract,
    wilson_interval,
    write_canonical_outputs,
)
from training.pbt.reporting.constants import METRICS_COLUMNS, TIERED_METRICS_COLUMNS
from training.pbt.reporting.plots import _baseline_fixed_working_point_values


METRIC = "validation_working_point_mistag_percent"
CURVE_EFFICIENCIES = [0.5, 0.8, 0.9]
CURVE_REJECTIONS = {
    "bc": [800.0, 500.0, 40.0],
    "bd": [900.0, 700.0, 120.0],
    "cb": [250.0, 80.0, 20.0],
    "cd": [900.0, 120.0, 12.0],
}


def counts_from_rejections(rejections, efficiencies=CURVE_EFFICIENCIES, total=100_000):
    counts = {}
    for pair, values in rejections.items():
        rows = []
        for eff, rejection in zip(efficiencies, values):
            passed = max(1, round(total / rejection))
            rows.append(
                {
                    "signal_efficiency": eff,
                    "background_passed": passed,
                    "background_total": total,
                    "background_efficiency": passed / total,
                }
            )
        counts[pair] = rows
    return counts


def fixed_curve_metrics(metric_value, accuracy, auc, train_loss):
    return {
        METRIC: metric_value,
        "validation_accuracy": accuracy,
        "validation_auc": auc,
        "train_loss": train_loss,
        "validation_bkg_rejection_at_eff": {
            "efficiencies": CURVE_EFFICIENCIES,
            "pairs": CURVE_REJECTIONS,
        },
        "validation_bkg_rejection_at_eff_counts": counts_from_rejections(CURVE_REJECTIONS),
    }


def synthetic_manifest(measured_baseline=True, configured_baseline=1.7, dataset_size=1000):
    shared = {
        "samples_per_epoch": 100,
        "weaver_epochs_per_generation": 1,
        "samples_per_epoch_val": 3000,
    }
    if dataset_size is not None:
        shared["training_dataset_size"] = dataset_size
    pbt_config = {
        "metric": METRIC,
        "mode": "min",
        "strategy": "anchored_lr_sweep",
        "baseline_metric_value": configured_baseline,
    }
    if measured_baseline:
        pbt_config["configured_baseline_metric_value"] = configured_baseline
        pbt_config["runtime_baseline_metric_value"] = 1.5
        pbt_config["baseline_metric_value"] = 1.5
    manifest = {
        "schema_version": 1,
        "experiment": "synthetic_publication_run",
        "status": "completed",
        "method": "anchored_lr_sweep",
        "run": {
            "method_name": "anchored_lr_sweep",
            "datasets": {"validation_dataset": "synthetic_proxy", "validation_suffix": "val5k_tail"},
            "schedule": {
                "training_interval": {"weaver_epochs_per_generation": 1, "samples_per_epoch": 100, "samples_per_trial_chunk": 100},
                "evaluation_interval": {"training_chunks": 1, "epochs": 1, "samples_per_trial": 100, "samples_per_epoch_val": 3000},
                "exploit_interval": {"enabled": True, "training_chunks": 1, "epochs": 1, "samples_per_trial": 100},
            },
        },
        "datasets": {"validation_dataset": "synthetic_proxy", "validation_suffix": "val5k_tail"},
        "checkpoint": {"path": "checkpoint.pt", "sha256": "abc123"},
        "initial_evaluation": {
            "status": "completed" if measured_baseline else "skipped",
            "checkpoint": "checkpoint.pt",
            "metrics": fixed_curve_metrics(1.5, 0.86, 0.96, None) if measured_baseline else {},
        },
        "config": {
            "shared": shared,
            "pbt": pbt_config,
        },
        "members": {
            "trial_a": {"name": "trial_a", "lr": 1.0e-4, "parent": None},
            "trial_b": {"name": "trial_b", "lr": 5.0e-5, "parent": "trial_a"},
        },
        "generations": [
            {
                "index": 0,
                "epoch": 18,
                "status": "completed",
                "workers": {
                    "trial_a": {"status": "completed", "lr": 1.0e-4, "metrics": fixed_curve_metrics(1.1, 0.88, 0.97, 0.32)},
                    "trial_b": {"status": "completed", "lr": 5.0e-5, "metrics": fixed_curve_metrics(1.3, 0.87, 0.965, 0.35)},
                },
                "ranking": ["trial_a", "trial_b"],
                "exploit": [],
            },
            {
                "index": 1,
                "epoch": 19,
                "status": "completed",
                "workers": {
                    "trial_a": {"status": "completed", "lr": 1.0e-4, "metrics": fixed_curve_metrics(1.0, 0.89, 0.972, 0.28)},
                    "trial_b": {"status": "completed", "lr": 7.5e-5, "metrics": fixed_curve_metrics(0.9, 0.9, 0.975, 0.27)},
                },
                "ranking": ["trial_b", "trial_a"],
                "exploit": [],
            },
        ],
        "best": {
            "generation": 1,
            "epoch": 19,
            "member": "trial_b",
            "metric": METRIC,
            "metric_value": 0.9,
            "lr": 7.5e-5,
            "metrics": fixed_curve_metrics(0.9, 0.9, 0.975, 0.27),
            "state_path": "checkpoints/global_best_state.pt",
            "optimizer_path": "checkpoints/global_best_optimizer.pt",
            "metadata_path": "checkpoints/global_best_metadata.json",
        },
    }
    return manifest


class PBTArtifactsTest(unittest.TestCase):
    def test_run_contract_records_resolved_inputs_and_configured_intervals(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR) as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            for suffix in ("trainTiny", "valTiny"):
                for flavor in ("bb", "cc", "dd"):
                    (dataset / f"sample_{flavor}_{suffix}.root").write_bytes(b"root")
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "shared": {
                    "checkpoint": str(checkpoint),
                    "dataset": str(dataset),
                    "data_extension": "root",
                    "train_suffix": "trainTiny",
                    "validation_suffix": "valTiny",
                    "seed": 123,
                    "weaver_epochs_per_generation": 2,
                    "samples_per_epoch": 50,
                    "samples_per_epoch_val": 30,
                },
                "pbt": {
                    "metric": METRIC,
                    "mode": "min",
                    "strategy": "exploit_mutate",
                    "exploit_fraction": 0.25,
                    "evaluation_interval_generations": 2,
                    "exploit_interval_generations": 3,
                },
                "slots": [{"gpu": "0", "label": "0"}],
            }

            contract = run_contract(config, ["python", "runner.py"], "local_weaver")

            self.assertEqual(contract["command"], ["python", "runner.py"])
            self.assertIn("commit", contract["git"])
            self.assertIn("dirty", contract["git"])
            self.assertEqual(contract["schedule"]["training_interval"]["samples_per_trial_chunk"], 100)
            self.assertEqual(contract["schedule"]["evaluation_interval"]["training_chunks"], 2)
            self.assertEqual(contract["schedule"]["exploit_interval"]["training_chunks"], 3)
            train_files = contract["datasets"]["resolved_files"]["train"]
            val_files = contract["datasets"]["resolved_files"]["val"]
            self.assertEqual(sum(len(row["files"]) for row in train_files), 3)
            self.assertEqual(sum(len(row["files"]) for row in val_files), 3)

    def test_canonical_outputs_from_synthetic_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest()
            append_event(
                run_dir,
                "exploit",
                {
                    "generation": 0,
                    "donor": "trial_a",
                    "recipient": "trial_b",
                    "donor_metric": 1.1,
                    "recipient_metric": 1.3,
                    "weight_source": "anchor",
                    "optimizer_source": "anchor",
                    "old_lr": 5.0e-5,
                    "new_lr": 7.5e-5,
                    "mutation": 0.75,
                },
            )

            artifacts = write_canonical_outputs(run_dir, manifest)

            for relative in (
                "manifest.json",
                "resolved_config.yaml",
                "events.jsonl",
                "metrics.csv",
                "summary.json",
                "report.md",
            ):
                self.assertTrue((run_dir / relative).is_file(), relative)
            for name in (
                "pbt_population_selection.png",
                "mistag_score_evolution.png",
                "learning_rate_lineage.png",
                "report/physics_performance.png",
                "diagnostics/background_efficiency_curves.png",
                "report/btag_mistag_tables.csv",
                "report/ctag_mistag_tables.csv",
            ):
                self.assertTrue((run_dir / "plots" / name).is_file(), name)
            for removed in (
                "training_evolution.png",
                "ctag_working_points_evolution.png",
                "btag_working_points_evolution.png",
                "aggregate_mistag_score_evolution.png",
                "ctag_vs_btag_tradeoff.png",
                "total_score_vs_learning_rate.png",
                "learning_rate_population_evolution.png",
                "final_vs_baseline_mistag_ratio.png",
                "pbt_decision_history.png",
                "baseline_vs_selected.png",
                "proxy_diagnostics.png",
            ):
                self.assertFalse((run_dir / "plots" / removed).exists(), removed)
            self.assertFalse(any(path.suffix == ".pdf" for path in (run_dir / "plots").rglob("*")))
            self.assertEqual(artifacts["report"], str(run_dir / "report.md"))

            with (run_dir / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["trial"], "trial_a")
            self.assertEqual(rows[0]["generation"], "0")
            self.assertEqual(rows[0]["training_chunk"], "0")
            self.assertEqual(rows[0]["samples_seen"], "100")
            self.assertEqual(rows[0]["epoch_fraction"], "0.1")
            self.assertEqual(rows[0]["optimization_metric_name"], METRIC)
            self.assertEqual(rows[0]["optimization_metric_value"], "1.1")
            self.assertEqual(rows[0]["optimization_metric_mode"], "min")
            self.assertEqual(rows[0]["validation_working_point_mistag_percent"], "1.1")
            self.assertAlmostEqual(float(rows[0]["controller_objective_mistag_percent"]), 0.7838293650793651)
            self.assertEqual(rows[0]["btag_c_mistag_percent_at_0p80"], "0.2")
            self.assertEqual(rows[0]["ctag_d_mistag_percent_at_0p80"], "0.8333333333333334")
            self.assertEqual(rows[0]["btag_c_mistag_percent_at_0p80_passed"], "200")
            self.assertEqual(rows[0]["btag_c_mistag_percent_at_0p80_total"], "100000")
            expected_lower, expected_upper = wilson_interval(200, 100_000)
            self.assertAlmostEqual(float(rows[0]["btag_c_mistag_percent_at_0p80_err_low"]), 100.0 * expected_lower)
            self.assertAlmostEqual(float(rows[0]["btag_c_mistag_percent_at_0p80_err_high"]), 100.0 * expected_upper)
            self.assertEqual(rows[0]["validation_accuracy"], "0.88")
            self.assertEqual(rows[0]["validation_auc"], "0.97")
            self.assertEqual(rows[0]["validation_dataset"], "synthetic_proxy")
            self.assertEqual(rows[0]["validation_suffix"], "val5k_tail")
            self.assertEqual(rows[0]["validation_sample_count"], "3000")
            self.assertEqual(rows[0]["evaluation_type"], "proxy")
            self.assertEqual(rows[-1]["samples_seen"], "200")
            self.assertEqual(rows[-1]["epoch_fraction"], "0.2")
            self.assertEqual(rows[-1]["best_so_far"], "0.9")
            self.assertTrue((run_dir / "plots" / "report" / "exploit_table.csv").is_file())

            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["winning_trial"], "trial_b")
            self.assertEqual(summary["baseline"]["kind"], "measured")
            self.assertEqual(summary["configured_baseline"]["kind"], "configured")
            self.assertEqual(summary["configured_baseline"]["metric_value"], 1.7)
            self.assertAlmostEqual(summary["best_improvement_vs_baseline"], (1.5 - 0.9) / 1.5)
            self.assertEqual(summary["evaluation"]["evaluation_type"], "proxy")
            self.assertIn("physics_performance", summary["plots"])
            self.assertIn("pbt_population_selection", summary["plots"])
            self.assertIn("mistag_score_evolution", summary["plots"])
            self.assertIn("learning_rate_lineage", summary["plots"])
            self.assertIn("learning_rate_mistag_correlation", summary["plots"])
            self.assertNotIn("training_evolution", summary["plots"])
            self.assertNotIn("baseline_comparison", summary["plots"])
            self.assertNotIn("working_point_evolution", summary["plots"])
            self.assertNotIn("lr_vs_metric", summary["plots"])
            # The report-facing plots (population/selection, mistag score
            # evolution, LR lineage, ...) live under the same manifest key
            # as the physics-performance bridge outputs now -- see
            # canonical.py/write_canonical_outputs -- each carrying its
            # richer {png, warnings, generations, members, metric_keys}
            # result dict, not just a path string.
            plots_artifacts = manifest["canonical_artifacts"]["plots"]
            for key in ("pbt_population_selection", "mistag_score_evolution", "learning_rate_lineage", "learning_rate_mistag_correlation"):
                self.assertIn(key, plots_artifacts)
                self.assertTrue(Path(plots_artifacts[key]["png"]).is_file())
            self.assertNotIn("research_plots", manifest["canonical_artifacts"])

            # synthetic_manifest()'s fixed_curve_metrics() reuses the exact
            # same rejection curves for every generation/member, so
            # "best_physics" (first-seen tie-break) resolves to trial_a/gen
            # 0 while manifest["best"] (the real PBT selection) is
            # trial_b/gen 1 -- this deliberately exercises the
            # global_best-vs-best_physics disagreement the report-facing
            # checkpoint-role fix must surface, not hide.
            selection = manifest["checkpoint_selection_for_report"]
            self.assertEqual(selection["role"], "global_best")
            self.assertEqual(selection["member"], "trial_b")
            self.assertEqual(selection["generation"], 1)
            self.assertEqual(selection["best_physics_member"], "trial_a")
            self.assertIs(selection["agrees_with_best_physics"], False)

            report = (run_dir / "report.md").read_text()
            self.assertLess(report.index("## Results"), report.index("## Method"))
            self.assertLess(report.index("## Final Physics Performance"), report.index("## PBT Population and Selection"))
            self.assertLess(report.index("## PBT Population and Selection"), report.index("## Mistag Score Evolution"))
            self.assertLess(report.index("## Mistag Score Evolution"), report.index("## Learning-Rate Lineage"))
            self.assertLess(report.index("## Learning-Rate Lineage"), report.index("## Learning Rate vs. Mistag Score Correlation"))
            self.assertLess(report.index("## Learning Rate vs. Mistag Score Correlation"), report.index("## Proxy Validation"))
            self.assertLess(report.index("## Proxy Validation"), report.index("## Model Selection Scores"))
            self.assertIn("anchored_lr_sweep", report)
            self.assertIn("Controller objective: mean predefined fixed-WP mistag percent", report)
            self.assertNotIn("## Training Evolution", report)
            self.assertNotIn("## Research Figures", report)
            self.assertNotIn("## Baseline vs. Selected Model", report)
            self.assertNotIn("samples_seen:LR =", report)
            self.assertIn("global best (PBT selection)", report)
            self.assertIn("Differs from the separate best-physics-score checkpoint", report)
            self.assertNotIn("LR vs metric", report)
            self.assertIn("samples/trial chunk", report)
            self.assertNotIn("epoch(s)", report)

    def test_report_links_only_to_pngs_that_exist_on_disk(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest()

            write_canonical_outputs(run_dir, manifest)

            report = (run_dir / "report.md").read_text()
            for match in re.finditer(r"\]\((plots/[^)]+\.png)\)", report):
                relative = match.group(1)
                self.assertTrue((run_dir / relative).is_file(), relative)

    def test_metrics_columns_and_tiered_metrics_columns_constants_unchanged(self):
        # The plot-set redesign must not touch the CSV schema -- pin the
        # exact column tuples so any accidental drift fails loudly here.
        self.assertEqual(
            METRICS_COLUMNS,
            (
                "generation", "training_chunk", "samples_seen", "epoch_fraction", "trial", "LR",
                "optimization_metric_name", "optimization_metric_value", "optimization_metric_mode",
                "controller_objective_mistag_percent",
                "validation_working_point_mistag_percent",
                "ctag_score", "btag_score", "total_mistag_score", "group_score_warning",
                "btag_c_mistag_percent_at_0p80", "btag_d_mistag_percent_at_0p80",
                "btag_c_mistag_percent_at_0p90", "btag_d_mistag_percent_at_0p90",
                "ctag_b_mistag_percent_at_0p50", "ctag_d_mistag_percent_at_0p50",
                "ctag_b_mistag_percent_at_0p80", "ctag_d_mistag_percent_at_0p80",
                "btag_c_mistag_percent_at_0p80_err_low", "btag_c_mistag_percent_at_0p80_err_high",
                "btag_c_mistag_percent_at_0p80_passed", "btag_c_mistag_percent_at_0p80_total",
                "btag_d_mistag_percent_at_0p80_err_low", "btag_d_mistag_percent_at_0p80_err_high",
                "btag_d_mistag_percent_at_0p80_passed", "btag_d_mistag_percent_at_0p80_total",
                "btag_c_mistag_percent_at_0p90_err_low", "btag_c_mistag_percent_at_0p90_err_high",
                "btag_c_mistag_percent_at_0p90_passed", "btag_c_mistag_percent_at_0p90_total",
                "btag_d_mistag_percent_at_0p90_err_low", "btag_d_mistag_percent_at_0p90_err_high",
                "btag_d_mistag_percent_at_0p90_passed", "btag_d_mistag_percent_at_0p90_total",
                "ctag_b_mistag_percent_at_0p50_err_low", "ctag_b_mistag_percent_at_0p50_err_high",
                "ctag_b_mistag_percent_at_0p50_passed", "ctag_b_mistag_percent_at_0p50_total",
                "ctag_d_mistag_percent_at_0p50_err_low", "ctag_d_mistag_percent_at_0p50_err_high",
                "ctag_d_mistag_percent_at_0p50_passed", "ctag_d_mistag_percent_at_0p50_total",
                "ctag_b_mistag_percent_at_0p80_err_low", "ctag_b_mistag_percent_at_0p80_err_high",
                "ctag_b_mistag_percent_at_0p80_passed", "ctag_b_mistag_percent_at_0p80_total",
                "ctag_d_mistag_percent_at_0p80_err_low", "ctag_d_mistag_percent_at_0p80_err_high",
                "ctag_d_mistag_percent_at_0p80_passed", "ctag_d_mistag_percent_at_0p80_total",
                "validation_accuracy", "validation_auc", "validation_loss", "best_so_far", "training_loss",
                "validation_shutdown_warning", "validation_dataset", "validation_suffix",
                "validation_sample_count", "evaluation_type",
            ),
        )
        self.assertEqual(
            TIERED_METRICS_COLUMNS,
            (
                "generation", "samples_seen", "tier", "member", "dataset", "suffix", "status", "rank",
                "population_size", "metric_name", "metric_value",
                "controller_objective_mistag_percent",
                "validation_working_point_mistag_percent",
                "ctag_score", "btag_score", "total_mistag_score", "group_score_warning",
                "btag_c_mistag_percent_at_0p80", "btag_d_mistag_percent_at_0p80",
                "btag_c_mistag_percent_at_0p90", "btag_d_mistag_percent_at_0p90",
                "ctag_b_mistag_percent_at_0p50", "ctag_d_mistag_percent_at_0p50",
                "ctag_b_mistag_percent_at_0p80", "ctag_d_mistag_percent_at_0p80",
                "btag_c_mistag_percent_at_0p80_err_low", "btag_c_mistag_percent_at_0p80_err_high",
                "btag_c_mistag_percent_at_0p80_passed", "btag_c_mistag_percent_at_0p80_total",
                "btag_d_mistag_percent_at_0p80_err_low", "btag_d_mistag_percent_at_0p80_err_high",
                "btag_d_mistag_percent_at_0p80_passed", "btag_d_mistag_percent_at_0p80_total",
                "btag_c_mistag_percent_at_0p90_err_low", "btag_c_mistag_percent_at_0p90_err_high",
                "btag_c_mistag_percent_at_0p90_passed", "btag_c_mistag_percent_at_0p90_total",
                "btag_d_mistag_percent_at_0p90_err_low", "btag_d_mistag_percent_at_0p90_err_high",
                "btag_d_mistag_percent_at_0p90_passed", "btag_d_mistag_percent_at_0p90_total",
                "ctag_b_mistag_percent_at_0p50_err_low", "ctag_b_mistag_percent_at_0p50_err_high",
                "ctag_b_mistag_percent_at_0p50_passed", "ctag_b_mistag_percent_at_0p50_total",
                "ctag_d_mistag_percent_at_0p50_err_low", "ctag_d_mistag_percent_at_0p50_err_high",
                "ctag_d_mistag_percent_at_0p50_passed", "ctag_d_mistag_percent_at_0p50_total",
                "ctag_b_mistag_percent_at_0p80_err_low", "ctag_b_mistag_percent_at_0p80_err_high",
                "ctag_b_mistag_percent_at_0p80_passed", "ctag_b_mistag_percent_at_0p80_total",
                "ctag_d_mistag_percent_at_0p80_err_low", "ctag_d_mistag_percent_at_0p80_err_high",
                "ctag_d_mistag_percent_at_0p80_passed", "ctag_d_mistag_percent_at_0p80_total",
            ),
        )

    def test_baseline_evaluation_feeds_working_point_plot_start(self):
        manifest = synthetic_manifest()

        baseline_values = _baseline_fixed_working_point_values(manifest)

        self.assertIsNotNone(baseline_values)
        self.assertAlmostEqual(baseline_values["btag_c_mistag_percent_at_0p80"], 0.2)
        self.assertAlmostEqual(baseline_values["ctag_d_mistag_percent_at_0p80"], 0.8333333333333334)

    def test_baseline_evaluation_is_absent_without_measured_initial_evaluation(self):
        manifest = synthetic_manifest(measured_baseline=False)

        self.assertIsNone(_baseline_fixed_working_point_values(manifest))

    def test_wilson_interval_is_bounded_and_asymmetric_for_rare_mistags(self):
        # A tiny observed rate (2 passes out of 100000) is the regime fixed-WP
        # mistags usually live in. The interval must stay within [0, 1] and
        # must not be forced symmetric around p.
        bounds = wilson_interval(2, 100_000)

        self.assertIsNotNone(bounds)
        lower, upper = bounds
        p = 2 / 100_000
        self.assertGreaterEqual(p - lower, 0.0)
        self.assertLessEqual(p - lower, p)  # lower bound of interval stays >= 0
        self.assertNotAlmostEqual(lower, upper)  # asymmetric, not a naive symmetric stderr

    def test_wilson_interval_handles_zero_passes_without_going_negative(self):
        bounds = wilson_interval(0, 1000)

        self.assertIsNotNone(bounds)
        lower, upper = bounds
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_wilson_interval_approaches_normal_approximation_for_large_n_moderate_p(self):
        # Sanity check against the textbook large-n regime: for p away from
        # the edges and large n, Wilson should be close to the symmetric
        # sqrt(p(1-p)/n) approximation.
        passed, total = 30_000, 100_000
        lower, upper = wilson_interval(passed, total)
        p = passed / total
        naive_stderr = math.sqrt(p * (1 - p) / total)

        self.assertAlmostEqual(lower, naive_stderr, delta=naive_stderr * 0.05)
        self.assertAlmostEqual(upper, naive_stderr, delta=naive_stderr * 0.05)

    def test_fixed_working_point_uncertainty_reads_counts_from_metrics(self):
        metrics = fixed_curve_metrics(1.1, 0.88, 0.97, 0.32)

        result = fixed_working_point_uncertainty(metrics, "b", 0.80, "c")

        self.assertIsNotNone(result)
        lower, upper, passed, total = result
        self.assertEqual(passed, 200)
        self.assertEqual(total, 100_000)
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_fixed_working_point_uncertainty_is_none_without_counts(self):
        self.assertIsNone(fixed_working_point_uncertainty({}, "b", 0.80, "c"))

    def test_format_mistag_value_rounds_to_uncertainty_precision(self):
        # A tiny central value with a much larger uncertainty must not be
        # printed with false precision (e.g. not "0.001700%").
        self.assertEqual(format_mistag_value(0.0017, 0.05, 0.05), "0.002±0.050%")
        # Comfortably-measured values keep the default 3-decimal precision.
        self.assertEqual(format_mistag_value(0.2), "0.200%")
        # A visibly asymmetric interval is reported as +/- rather than
        # symmetrized away.
        self.assertEqual(format_mistag_value(1.0, 0.05, 0.4), "1.00% (+0.40/-0.05)")

    def test_format_mistag_value_handles_missing_value(self):
        self.assertEqual(format_mistag_value(None), "n/a")

    def test_fixed_efficiency_mistag_plots_support_many_generations(self):
        # The evolution plots must not be hard-coded around the two-point
        # smoke-test shape; verify a longer, single-trial run round-trips
        # cleanly through the full canonical-artifact pipeline.
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest()
            manifest["generations"] = [
                {
                    "index": index,
                    "epoch": index,
                    "status": "completed",
                    "workers": {
                        "trial_a": {
                            "status": "completed",
                            "lr": 1.0e-4,
                            "metrics": fixed_curve_metrics(1.2 - 0.05 * index, 0.85, 0.96, 0.3),
                        },
                    },
                    "ranking": ["trial_a"],
                    "exploit": [],
                }
                for index in range(8)
            ]
            manifest["best"]["generation"] = 7
            manifest["best"]["member"] = "trial_a"

            write_canonical_outputs(run_dir, manifest)

            with (run_dir / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertTrue((run_dir / "plots" / "pbt_population_selection.png").is_file())
            self.assertTrue((run_dir / "plots" / "learning_rate_lineage.png").is_file())

    def test_rebuild_command_regenerates_report_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest()
            write_canonical_outputs(run_dir, manifest)
            (run_dir / "report.md").unlink()

            result = subprocess.run(
                [sys.executable, str(PROJECT_DIR / "scripts/training/pbt/rebuild_artifacts.py"), str(run_dir)],
                cwd=PROJECT_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((run_dir / "report.md").is_file())
            self.assertIn("report.md", result.stdout)

    def test_configured_baseline_is_not_measured_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest(measured_baseline=False, configured_baseline=1.5)

            write_canonical_outputs(run_dir, manifest)

            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertIsNone(summary["baseline"])
            self.assertEqual(summary["configured_baseline"]["metric_value"], 1.5)
            self.assertIsNone(summary["best_improvement_vs_baseline"])
            report = (run_dir / "report.md").read_text()
            self.assertIn("Measured baseline: n/a", report)
            self.assertIn("Configured reference: 1.5", report)
            self.assertNotIn("## Baseline vs. Selected Model", report)
            self.assertFalse((run_dir / "plots" / "baseline_vs_selected.png").exists())
            self.assertIn("No corroboration-tier evaluation was scheduled during this short run.", report)
            mistag_evolution = manifest["canonical_artifacts"]["plots"]["mistag_score_evolution"]
            self.assertIs(mistag_evolution["has_baseline_point"], False)
            self.assertIn("not available for this run", report)

    def test_epoch_fraction_is_null_when_training_dataset_size_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = synthetic_manifest(dataset_size=None)

            write_canonical_outputs(run_dir, manifest)

            with (run_dir / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["training_chunk"], "0")
            self.assertEqual(rows[0]["samples_seen"], "100")
            self.assertEqual(rows[0]["epoch_fraction"], "")

    def test_apply_exploit_records_structured_copy_and_lr_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("strong", "weak"):
                (root / name).mkdir()
            strong_state, strong_optimizer = checkpointing.checkpoint_paths(root / "strong", 0)
            weak_state, weak_optimizer = checkpointing.checkpoint_paths(root / "weak", 0)
            strong_state.write_bytes(b"strong-state")
            strong_optimizer.write_bytes(b"strong-optimizer")
            weak_state.write_bytes(b"weak-state")
            weak_optimizer.write_bytes(b"weak-optimizer")
            manifest_path = root / "manifest.json"
            manifest = {
                "config": {
                    "shared": {},
                    "pbt": {"metric": METRIC, "mode": "min"},
                },
                "members": {
                    "strong": {"lr": 1.0e-4, "parent": None},
                    "weak": {"lr": 1.5e-4, "parent": None},
                },
            }
            generation = {
                "index": 0,
                "epoch": 0,
                "workers": {
                    "strong": {"metrics": {METRIC: 0.9}},
                    "weak": {"metrics": {METRIC: 1.4}},
                },
                "exploit": [
                    {
                        "donor": "strong",
                        "recipient": "weak",
                        "recipient_lr": 1.5e-4,
                        "donor_lr": 1.0e-4,
                        "mutation_factor": 0.8,
                        "new_lr": 8.0e-5,
                        "applied": False,
                    }
                ],
            }

            transitions.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(weak_state.read_bytes(), b"strong-state")
            self.assertEqual(weak_optimizer.read_bytes(), b"strong-optimizer")
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual([event["event_type"] for event in events], ["exploit", "weight_copy", "optimizer_copy", "lr_change"])
            self.assertEqual(events[0]["donor_metric"], 0.9)
            self.assertEqual(events[0]["recipient_metric"], 1.4)
            self.assertEqual(events[-1]["old_lr"], 1.5e-4)
            self.assertEqual(events[-1]["new_lr"], 8.0e-5)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_pbt_research_plots import _anchor_copy_manifest
from training.pbt.reporting.compare_runs import (
    load_run_correlation,
    plot_run_comparison,
    render_comparison_table,
)
from training.pbt.reporting.metrics_rows import refresh_metrics_csv


def _write_run(temporary, manifest, name="run"):
    run_dir = Path(temporary) / name
    refresh_metrics_csv(run_dir, manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


class LoadRunCorrelationTest(unittest.TestCase):
    def test_reads_config_summary_and_correlation_from_disk(self):
        manifest = _anchor_copy_manifest()
        manifest["config"]["shared"]["generations"] = 3
        manifest["config"]["shared"]["weaver_epochs_per_generation"] = 1
        manifest["config"]["shared"]["proxy_validation"] = {"control_rows_per_class": 50000}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _write_run(temporary, manifest)
            row = load_run_correlation(run_dir, label="baseline")
        self.assertEqual(row["label"], "baseline")
        self.assertEqual(row["generations"], 3)
        self.assertEqual(row["weaver_epochs_per_generation"], 1)
        self.assertEqual(row["control_rows_per_class"], 50000)
        self.assertEqual(row["n"], 6)  # 2 members x 3 generations, matches _anchor_copy_manifest

    def test_accepts_manifest_json_path_directly(self):
        manifest = _anchor_copy_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _write_run(temporary, manifest)
            row = load_run_correlation(run_dir / "manifest.json")
        self.assertEqual(row["n"], 6)


class RenderComparisonTableTest(unittest.TestCase):
    def test_formats_ci_and_reason_rows(self):
        rows = [
            {
                "label": "48gen",
                "generations": 48,
                "weaver_epochs_per_generation": 1,
                "control_rows_per_class": 50000,
                "n": 192,
                "pearson_r": -0.162,
                "pearson_r_ci": (-0.257, -0.067),
                "spearman_rho": -0.181,
                "spearman_rho_ci": (-0.290, -0.080),
                "reason": None,
            },
            {
                "label": "too_short",
                "generations": 2,
                "weaver_epochs_per_generation": 1,
                "control_rows_per_class": 50000,
                "n": 2,
                "pearson_r": None,
                "pearson_r_ci": None,
                "spearman_rho": None,
                "spearman_rho_ci": None,
                "reason": "insufficient_paired_observations",
            },
        ]
        table = render_comparison_table(rows)
        self.assertIn("-0.162 [-0.257, -0.067]", table)
        self.assertIn("insufficient_paired_observations", table)
        self.assertIn("| 48gen |", table)

    def test_row_with_no_ci_falls_back_to_bare_point_estimate(self):
        rows = [
            {
                "label": "two_gen",
                "generations": 2,
                "weaver_epochs_per_generation": 1,
                "control_rows_per_class": 50000,
                "n": 6,
                "pearson_r": 0.5,
                "pearson_r_ci": None,
                "spearman_rho": 0.6,
                "spearman_rho_ci": None,
                "reason": None,
            }
        ]
        table = render_comparison_table(rows)
        self.assertIn("0.500", table)
        self.assertNotIn("0.500 [", table)


class PlotRunComparisonTest(unittest.TestCase):
    def test_writes_forest_plot_png(self):
        rows = [
            {"label": "a", "n": 10, "pearson_r": -0.2, "pearson_r_ci": (-0.3, -0.1), "reason": None},
            {"label": "b", "n": 4, "pearson_r": None, "pearson_r_ci": None, "reason": "insufficient_paired_observations"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "comparison.png"
            written = plot_run_comparison(rows, output_path)
            self.assertEqual(written, output_path)
            self.assertTrue(output_path.is_file())

    def test_returns_none_when_nothing_plottable(self):
        rows = [{"label": "a", "n": 2, "pearson_r": None, "pearson_r_ci": None, "reason": "insufficient_paired_observations"}]
        with tempfile.TemporaryDirectory() as temporary:
            written = plot_run_comparison(rows, Path(temporary) / "comparison.png")
        self.assertIsNone(written)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports.write_metrics_summary import write_summary


class MetricsSummaryTest(unittest.TestCase):
    def test_write_summary_includes_fixed_efficiency_tables(self):
        manifest = {
            "experiment": "unit_run",
            "status": "completed",
            "config": {
                "shared": {
                    "samples_per_epoch": 100000,
                    "dataset": "/tmp/data",
                    "checkpoint": "/tmp/net.pt",
                    "data_config": "/tmp/data.yaml",
                },
                "pbt": {
                    "metric": "validation_working_point_mistag_percent",
                    "mode": "min",
                    "strategy": "anchored_lr_sweep",
                },
            },
            "members": {"member_00": {"lr": 1.0e-4}},
            "best": {
                "generation": 0,
                "member": "member_00",
                "epoch": 0,
                "metric": "validation_working_point_mistag_percent",
                "metric_value": 1.846875,
                "state_path": "/tmp/state.pt",
            },
            "generations": [
                {
                    "index": 0,
                    "epoch": 0,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "lr": 1.0e-4,
                            "metrics": {
                                "validation_bkg_rejection_score": 2.0,
                                "validation_working_point_mistag_percent": 2.51875,
                                "validation_ctag_reference_mistag_percent": 2.4375,
                                "validation_bkg_rejection_at_eff": {
                                    "efficiencies": [0.5, 0.8, 0.9],
                                    "pairs": {
                                        "bc": [10.0, 20.0, 25.0],
                                        "bd": [100.0, 200.0, 250.0],
                                        "cb": [25.0, 50.0, 100.0],
                                        "cd": [40.0, 80.0, 160.0],
                                    },
                                },
                            },
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            plot_paths = {
                "physics_performance_plot": Path(temporary) / "plots/report/physics_performance.png",
                "training_diagnostics_plot": Path(temporary) / "plots/report/training_diagnostics.png",
                "btag_mistag_table_csv": Path(temporary) / "plots/report/btag_mistag_tables.csv",
                "ctag_mistag_table_csv": Path(temporary) / "plots/report/ctag_mistag_tables.csv",
                "background_rejection_curves_plot": Path(temporary) / "plots/diagnostics/background_rejection_curves.png",
            }
            for key, plot_path in plot_paths.items():
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plot_path.write_text("", encoding="utf-8")
                manifest[key] = str(plot_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            output = write_summary(manifest_path)
            summary = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(output.name, "metrics_summary.json")
            self.assertEqual(summary["global_best"]["member"], "member_00")
            self.assertEqual(
                summary["metric"]["definition"]["display_name"],
                "Selection metric: average mistag at reference working points",
            )
            self.assertIn("mistag percentages", summary["metric"]["definition"]["formula"])
            self.assertEqual(summary["metric"]["mode"], "min")
            self.assertIn("c-tag efficiencies 0.50/0.80", summary["showcase_metric"]["formula"])
            self.assertEqual(
                summary["showcase_metric"]["display_name"],
                "Average mistag at fixed working points",
            )
            self.assertAlmostEqual(
                summary["global_best"]["details"]["showcase_metrics"]["average_mistag_percent"],
                (5.0 + 0.5 + 4.0 + 0.4 + 4.0 + 2.5 + 2.0 + 1.25) / 8,
            )
            self.assertEqual(summary["generations"][0]["training_events"], 100000)
            self.assertEqual(
                summary["plots"]["btag_mistag_table_csv"],
                str(plot_paths["btag_mistag_table_csv"]),
            )
            self.assertEqual(
                summary["report_plots"]["physics_performance_plot"],
                str(plot_paths["physics_performance_plot"]),
            )
            self.assertEqual(
                summary["diagnostic_plots"]["background_rejection_curves_plot"],
                str(plot_paths["background_rejection_curves_plot"]),
            )
            self.assertEqual(list(summary["diagnostic_plots"]), ["background_rejection_curves_plot"])
            c_table = next(item for item in summary["mistag_percentages"] if item["tag"] == "c")
            b_table = next(item for item in summary["mistag_percentages"] if item["tag"] == "b")
            self.assertEqual(c_table["rows"][0]["fixed_efficiency"], 0.5)
            self.assertEqual(c_table["rows"][0]["mistag_percent"]["b_bkg_percent"], 4.0)
            self.assertEqual(c_table["rows"][0]["mistag_percent"]["d_bkg_percent"], 2.5)
            self.assertEqual(b_table["rows"][0]["mistag_percent"]["d_bkg_percent"], 0.5)


if __name__ == "__main__":
    unittest.main()

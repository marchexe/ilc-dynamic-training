import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports.plot_controller_diagnostics import collect_rows, default_output, plot_manifest


def controller_manifest():
    return {
        "experiment": "controller_unit",
        "generations": [
            {
                "index": 0,
                "epoch": 18,
                "status": "completed",
                "controller_observations": {
                    "member_00": {
                        "generation": 0,
                        "member": "member_00",
                        "epoch_fraction": 17.2,
                        "lr": 2.0e-5,
                        "metric_name": "validation_working_point_mistag_percent",
                        "metric_value": 1.04,
                        "metric_ema": 1.04,
                        "metric_uncertainty": 0.02,
                        "metric_delta_sigma": 0.0,
                        "baseline_metric_value": 1.05,
                        "baseline_delta": 0.01,
                        "train_loss_ema": 0.27,
                        "grad_norm": 0.7,
                        "adaptive_direction_norm": 1.8,
                    }
                },
                "controller_actions": {
                    "member_00": {
                        "state_label": "flat",
                        "action": "keep",
                        "safety_check": "passed",
                        "applied": False,
                    }
                },
            },
            {
                "index": 1,
                "epoch": 18,
                "status": "completed",
                "controller_observations": {
                    "member_00": {
                        "generation": 1,
                        "member": "member_00",
                        "epoch_fraction": 17.4,
                        "lr": 1.9e-5,
                        "metric_name": "validation_working_point_mistag_percent",
                        "metric_value": 1.06,
                        "metric_ema": 1.046,
                        "metric_uncertainty": 0.025,
                        "metric_delta_sigma": -0.62,
                        "baseline_metric_value": 1.05,
                        "baseline_delta": -0.01,
                        "train_loss_ema": 0.26,
                        "grad_norm": 0.8,
                    }
                },
                "controller_actions": {
                    "member_00": {
                        "state_label": "unsafe",
                        "action": "lr_mul_0_95",
                        "safety_check": "passed",
                        "applied": True,
                    }
                },
            },
        ],
    }


class PlotControllerDiagnosticsTest(unittest.TestCase):
    def test_default_output_for_manifest_goes_under_diagnostics(self):
        output = default_output(Path("/tmp/run/manifest.json"))

        self.assertEqual(output, Path("/tmp/run/plots/diagnostics/controller_diagnostics.png"))

    def test_collect_rows_keeps_controller_signals_and_actions(self):
        rows = collect_rows(controller_manifest())

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["member"], "member_00")
        self.assertAlmostEqual(rows[0]["metric_uncertainty"], 0.02)
        self.assertAlmostEqual(rows[0]["train_loss_ema"], 0.27)
        self.assertAlmostEqual(rows[0]["grad_norm"], 0.7)
        self.assertEqual(rows[1]["action"], "lr_mul_0_95")
        self.assertTrue(rows[1]["applied"])

    def test_plot_manifest_writes_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(controller_manifest()), encoding="utf-8")

            output = plot_manifest(manifest_path)

            self.assertEqual(output, Path(temporary) / "plots/diagnostics/controller_diagnostics.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_background_efficiency_curves


class PlotBackgroundEfficiencyCurvesTest(unittest.TestCase):
    def test_plot_manifest_writes_diagnostic_png(self):
        manifest = {
            "generations": [
                {
                    "index": 0,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "metrics": {
                                "validation_bkg_rejection_at_eff": {
                                    "efficiencies": [0.5, 0.8, 0.9],
                                    "pairs": {
                                        "bc": [10.0, 20.0, 25.0],
                                        "bd": [100.0, 200.0, 250.0],
                                        "cb": [25.0, 50.0, 100.0],
                                        "cd": [40.0, 80.0, 160.0],
                                    },
                                }
                            }
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = plot_background_efficiency_curves.plot_manifest(manifest_path)

            self.assertEqual(output, Path(temporary) / "plots/diagnostics/background_efficiency_curves.png")
            self.assertTrue(output.exists())

    def test_background_efficiency_is_inverse_rejection(self):
        self.assertEqual(plot_background_efficiency_curves.background_efficiency(20.0), 0.05)


if __name__ == "__main__":
    unittest.main()

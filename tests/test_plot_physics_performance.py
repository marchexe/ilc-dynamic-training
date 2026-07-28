import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_physics_performance


class PlotPhysicsPerformanceTest(unittest.TestCase):

    def test_best_physics_ignores_manifest_best_score(self):
        def worker(scale):
            return {
                "metrics": {
                    "validation_bkg_rejection_at_eff": {
                        "efficiencies": [0.5, 0.8, 0.9],
                        "pairs": {
                            "bc": [10.0 * scale, 20.0 * scale, 25.0 * scale],
                            "bd": [100.0 * scale, 200.0 * scale, 250.0 * scale],
                            "cb": [25.0 * scale, 50.0 * scale, 100.0 * scale],
                            "cd": [40.0 * scale, 80.0 * scale, 160.0 * scale],
                        },
                    }
                }
            }

        manifest = {
            "best": {"generation": 0, "member": "member_bad"},
            "generations": [
                {
                    "index": 0,
                    "status": "completed",
                    "workers": {
                        "member_bad": worker(1.0),
                        "member_good": worker(2.0),
                    },
                }
            ],
        }

        _worker, generation, member_name, score = plot_physics_performance.worker_for_report(
            manifest, "best_physics"
        )

        self.assertEqual(generation["index"], 0)
        self.assertEqual(member_name, "member_good")
        self.assertLess(score, plot_physics_performance.physics_mistag_score(worker(1.0)["metrics"]))

    def test_plot_manifest_writes_single_report_png(self):
        manifest = {
            "experiment": "unit_run",
            "config": {"pbt": {"metric": "validation_working_point_mistag_percent", "mode": "min"}},
            "best": {"generation": 0, "member": "member_00"},
            "generations": [
                {
                    "index": 0,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "metrics": {
                                "validation_working_point_mistag_percent": 1.0,
                                "validation_bkg_rejection_at_eff": {
                                    "efficiencies": [0.5, 0.8, 0.9],
                                    "pairs": {
                                        "bc": [10.0, 20.0, 25.0],
                                        "bd": [100.0, 200.0, 250.0],
                                        "cb": [25.0, 50.0, 100.0],
                                        "cd": [40.0, 80.0, 160.0],
                                    },
                                },
                            }
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = plot_physics_performance.plot_manifest(manifest_path)

            self.assertEqual(output, Path(temporary) / "plots/report/physics_performance.png")
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

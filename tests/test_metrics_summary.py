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
                    "metric": "validation_bkg_rejection_score",
                    "mode": "max",
                    "strategy": "anchored_lr_sweep",
                },
            },
            "members": {"member_00": {"lr": 1.0e-4}},
            "best": {
                "generation": 0,
                "member": "member_00",
                "epoch": 0,
                "metric": "validation_bkg_rejection_score",
                "metric_value": 2.0,
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
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            output = write_summary(manifest_path)
            summary = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(output.name, "metrics_summary.json")
            self.assertEqual(summary["global_best"]["member"], "member_00")
            self.assertEqual(
                summary["metric"]["definition"]["display_name"],
                "PBT objective: mean ln(BGrej), all pairs",
            )
            self.assertIn("mean(log(BGrej_pair(eff)))", summary["metric"]["definition"]["formula"])
            self.assertEqual(summary["generations"][0]["training_events"], 100000)
            c_table = next(item for item in summary["mistag_percentages"] if item["tag"] == "c")
            b_table = next(item for item in summary["mistag_percentages"] if item["tag"] == "b")
            self.assertEqual(c_table["rows"][0]["fixed_efficiency"], 0.5)
            self.assertEqual(c_table["rows"][0]["mistag_percent"]["b_bkg_percent"], 4.0)
            self.assertEqual(c_table["rows"][0]["mistag_percent"]["d_bkg_percent"], 2.5)
            self.assertEqual(b_table["rows"][0]["mistag_percent"]["d_bkg_percent"], 0.5)


if __name__ == "__main__":
    unittest.main()

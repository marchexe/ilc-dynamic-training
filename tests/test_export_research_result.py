import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports.export_research_result import build_export, write_export


def research_manifest(metric_value=0.95, seed=12345):
    return {
        "experiment": f"adaptive_seed_{seed}",
        "status": "completed",
        "config": {
            "shared": {
                "seed": seed,
                "samples_per_epoch": 480000,
                "samples_per_epoch_val": 150000,
            },
            "pbt": {
                "metric": "validation_working_point_mistag_percent",
                "mode": "min",
                "backend": "local_weaver",
                "strategy": "anchored_lr_sweep",
                "confidence_aware_selection": True,
                "selection_uncertainty_sigma": 1.0,
                "anchored_weight_source": "self",
                "baseline_metric_value": 1.0,
                "dynamic_controller": {"mode": "active"},
            },
        },
        "best": {
            "generation": 1,
            "member": "member_00",
            "metric": "validation_working_point_mistag_percent",
            "metric_value": metric_value,
        },
        "generations": [
            {
                "index": 0,
                "status": "completed",
                "workers": {
                    "member_00": {
                        "metrics": {"validation_working_point_mistag_percent": 1.02}
                    }
                },
            },
            {
                "index": 1,
                "status": "completed",
                "workers": {
                    "member_00": {
                        "metrics": {"validation_working_point_mistag_percent": metric_value}
                    }
                },
            },
        ],
    }


class ExportResearchResultTest(unittest.TestCase):
    def test_build_export_summarizes_baseline_improvement(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(research_manifest()), encoding="utf-8")

            payload = build_export([manifest_path])

            run = payload["runs"][0]
            self.assertEqual(run["anchored_weight_source"], "self")
            self.assertAlmostEqual(run["checkpoint_baseline_metric"], 1.0)
            self.assertAlmostEqual(run["best_metric"], 0.95)
            self.assertAlmostEqual(run["improvement_vs_checkpoint_abs"], 0.05)
            self.assertAlmostEqual(run["improvement_vs_checkpoint_percent"], 5.0)
            self.assertEqual(payload["aggregate"]["run_count"], 1)

    def test_write_export_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(research_manifest(seed=7)), encoding="utf-8")
            output = Path(temporary) / "research/result.json"
            csv_output = Path(temporary) / "research/result.csv"

            result = write_export([manifest_path], output, csv_output)

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertTrue(csv_output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["aggregate"]["seeds"], [7])


if __name__ == "__main__":
    unittest.main()

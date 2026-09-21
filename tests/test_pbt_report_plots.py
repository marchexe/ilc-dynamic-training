import tempfile
import unittest
from pathlib import Path

from tests.test_pbt_research_plots import _anchor_copy_manifest
from training.pbt.reporting.constants import REPORT_PLOT_NAMES
from training.pbt.reporting.report_plots import plot_proxy_validation, write_report_plots


class ProxyValidationPlotTest(unittest.TestCase):
    @staticmethod
    def tiered_manifest():
        manifest = _anchor_copy_manifest()
        manifest["tiered_evaluations"] = [{
            "generation": 0,
            "tier": "monitor",
            "dataset": "d",
            "suffix": "s",
            "metric_name": "validation_working_point_mistag_percent",
            "mode": "min",
            "members": {"m_a": {"status": "completed", "metrics": {
                "validation_working_point_mistag_percent": 1.0,
            }}},
            "ranking": ["m_a"],
        }]
        return manifest

    def test_proxy_plot_is_conditional(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(plot_proxy_validation(temporary, _anchor_copy_manifest()))

    def test_proxy_plot_is_written_when_tier_data_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = plot_proxy_validation(temporary, self.tiered_manifest())
            self.assertEqual(Path(result["png"]).name, "proxy_validation.png")
            self.assertTrue(Path(result["png"]).is_file())
            self.assertFalse(result["independent"])


class OrchestrationTest(unittest.TestCase):
    def test_only_proxy_validation_is_generated(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = write_report_plots(temporary, ProxyValidationPlotTest.tiered_manifest())
            self.assertEqual(set(results), {"proxy_validation"})
            self.assertEqual(Path(results["proxy_validation"]["png"]).name,
                             f"{REPORT_PLOT_NAMES['proxy_validation']}.png")

    def test_no_report_plot_is_generated_without_tier_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = write_report_plots(temporary, _anchor_copy_manifest())
            self.assertEqual(results, {})
            self.assertEqual(list((Path(temporary) / "plots").glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()

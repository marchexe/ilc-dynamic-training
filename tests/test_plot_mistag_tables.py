import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_mistag_tables


class PlotMistagTablesTest(unittest.TestCase):

    def test_default_output_for_manifest_goes_under_plots(self):
        output = plot_mistag_tables.default_output_path(("run", Path("/tmp/run/manifest.json")), "c")
        self.assertEqual(output, Path("/tmp/run/plots/diagnostics/ctag_mistag_tables.png"))

    def test_collect_tables_uses_c_tag_efficiencies_as_percent_mistag(self):
        manifest = {
            "best": {"generation": 0, "member": "member_00"},
            "generations": [
                {
                    "index": 0,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "metrics": {
                                "validation_bkg_rejection_at_eff": {
                                    "efficiencies": [0.5, 0.8],
                                    "pairs": {
                                        "bc": [10.0, 20.0],
                                        "bd": [100.0, 200.0],
                                        "cb": [25.0, 50.0],
                                        "cd": [40.0, 80.0],
                                    },
                                }
                            }
                        }
                    },
                }
            ],
        }

        manifest_key = object()
        tables = plot_mistag_tables.collect_tables(
            [("demo", manifest_key)], "c", (0.5, 0.8), "global_best", manifests={manifest_key: manifest}
        )

        self.assertEqual(tables[0.5][0]["b_bkg_percent"], 4.0)
        self.assertEqual(tables[0.5][0]["d_bkg_percent"], 2.5)
        self.assertEqual(tables[0.8][0]["b_bkg_percent"], 2.0)
        self.assertEqual(tables[0.8][0]["d_bkg_percent"], 1.25)


if __name__ == "__main__":
    unittest.main()

import unittest

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_pbt_lr_response


class PlotPBTLRResponseTest(unittest.TestCase):
    def test_collect_points_reads_lr_and_mistag_values(self):
        manifest = {
            "best": {"generation": 0, "member": "member_01"},
            "generations": [
                {
                    "index": 0,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "command": ["weaver", "--start-lr", "0.1"],
                            "metrics": {
                                "validation_bkg_rejection_at_eff_lookup": {
                                    "b_tag_eff_0.80": {
                                        "c_bkg_rejection": 10.0,
                                        "d_bkg_rejection": 100.0,
                                    }
                                }
                            },
                        },
                        "member_01": {
                            "command": ["weaver", "--start-lr", "0.2"],
                            "metrics": {
                                "validation_bkg_rejection_at_eff_lookup": {
                                    "b_tag_eff_0.80": {
                                        "c_bkg_rejection": 20.0,
                                        "d_bkg_rejection": 200.0,
                                    }
                                }
                            },
                        },
                    },
                }
            ],
        }

        points = plot_pbt_lr_response.collect_points(manifest, b_efficiencies=(0.8,))

        self.assertEqual([point["lr"] for point in points], [0.1, 0.2])
        self.assertEqual(points[0]["mistag_percent"][0.8]["c"], 10.0)
        self.assertEqual(points[0]["mistag_percent"][0.8]["d"], 1.0)
        self.assertFalse(points[0]["is_global_best"])
        self.assertTrue(points[1]["is_global_best"])


if __name__ == "__main__":
    unittest.main()

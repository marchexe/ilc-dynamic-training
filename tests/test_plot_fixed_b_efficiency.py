import unittest

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_fixed_b_efficiency


class PlotFixedBEfficiencyTest(unittest.TestCase):
    def test_collect_series_converts_rejection_to_background_efficiency(self):
        manifest = {
            "config": {
                "shared": {"samples_per_epoch": 100000},
                "pbt": {"metric": "validation_bkg_rejection_score", "mode": "max"},
            },
            "generations": [
                {
                    "index": 0,
                    "epoch": 0,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "metrics": {
                                "validation_bkg_rejection_score": 1.0,
                                "validation_bkg_rejection_at_eff_lookup": {
                                    "b_tag_eff_0.80": {
                                        "c_bkg_rejection": 10.0,
                                        "d_bkg_rejection": 100.0,
                                    }
                                },
                            }
                        },
                        "member_01": {
                            "metrics": {
                                "validation_bkg_rejection_score": 2.0,
                                "validation_bkg_rejection_at_eff_lookup": {
                                    "b_tag_eff_0.80": {
                                        "c_bkg_rejection": 20.0,
                                        "d_bkg_rejection": 200.0,
                                    }
                                },
                            }
                        },
                    },
                },
                {
                    "index": 1,
                    "epoch": 1,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "metrics": {
                                "validation_bkg_rejection_score": 3.0,
                                "validation_bkg_rejection_at_eff_lookup": {
                                    "b_tag_eff_0.80": {
                                        "c_bkg_rejection": 40.0,
                                        "d_bkg_rejection": 400.0,
                                    }
                                },
                            }
                        }
                    },
                },
            ],
        }

        series, labels = plot_fixed_b_efficiency.collect_series(manifest, (0.8,))

        self.assertEqual(series[("c", 0.8)]["x"], [100000, 200000])
        self.assertEqual(series[("c", 0.8)]["y"], [0.05, 0.025])
        self.assertEqual(series[("d", 0.8)]["y"], [0.005, 0.0025])
        self.assertEqual(labels[0], (100000, "member_01", 0))


if __name__ == "__main__":
    unittest.main()

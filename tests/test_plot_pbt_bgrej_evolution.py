import unittest

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_pbt_bgrej_evolution


class PlotPBTBgrejEvolutionTest(unittest.TestCase):
    def test_values_for_quantity_converts_rejection_to_mistag_percent(self):
        self.assertEqual(
            plot_pbt_bgrej_evolution.values_for_quantity([10.0, 100.0], "mistag"),
            [10.0, 1.0],
        )
        self.assertEqual(
            plot_pbt_bgrej_evolution.values_for_quantity([10.0, 100.0], "rejection"),
            [10.0, 100.0],
        )

    def test_collect_curves_uses_generation_winners_and_lrs(self):
        manifest = {
            "config": {"pbt": {"metric": "validation_bkg_rejection_score", "mode": "max"}},
            "best": {"generation": 1, "member": "member_01"},
            "generations": [
                {
                    "index": 0,
                    "epoch": 0,
                    "status": "completed",
                    "workers": {
                        "member_00": {
                            "command": ["weaver", "--start-lr", "0.1"],
                            "metrics": {
                                "validation_bkg_rejection_score": 1.0,
                                "validation_bkg_rejection_at_eff": {
                                    "efficiencies": [0.8, 0.9],
                                    "pairs": {"bc": [10, 5], "bd": [20, 10]},
                                },
                            },
                        },
                        "member_01": {
                            "command": ["weaver", "--start-lr", "0.2"],
                            "metrics": {
                                "validation_bkg_rejection_score": 2.0,
                                "validation_bkg_rejection_at_eff": {
                                    "efficiencies": [0.8, 0.9],
                                    "pairs": {"bc": [30, 15], "bd": [40, 20]},
                                },
                            },
                        },
                    },
                },
                {
                    "index": 1,
                    "epoch": 1,
                    "status": "completed",
                    "workers": {
                        "member_01": {
                            "command": ["weaver", "--start-lr", "0.3"],
                            "metrics": {
                                "validation_bkg_rejection_score": 3.0,
                                "validation_bkg_rejection_at_eff": {
                                    "efficiencies": [0.8, 0.9],
                                    "pairs": {"bc": [50, 25], "bd": [60, 30]},
                                },
                            },
                        }
                    },
                },
            ],
        }

        rows = plot_pbt_bgrej_evolution.collect_curves(manifest, tag="b")

        self.assertEqual([row["generation"] for row in rows], [0, 1])
        self.assertEqual([row["member"] for row in rows], ["member_01", "member_01"])
        self.assertEqual([row["lr"] for row in rows], [0.2, 0.3])
        self.assertEqual(rows[0]["pairs"]["bc"], [30.0, 15.0])


if __name__ == "__main__":
    unittest.main()

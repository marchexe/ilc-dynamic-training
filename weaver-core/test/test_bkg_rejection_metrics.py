import numpy as np
import unittest

from weaver.utils.nn.metrics import (
    b_tag_rejection_score,
    bkg_rejection_at_eff,
    bkg_rejection_bc_score,
    bkg_rejection_bd_score,
    bkg_rejection_cb_score,
    bkg_rejection_cd_score,
    bkg_rejection_score,
    c_tag_rejection_score,
)


class BkgRejectionMetricsTest(unittest.TestCase):
    def test_bkg_rejection_metrics_report_expected_pairs(self):
        y_true = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_score = np.asarray(
            [
                [0.95, 0.04, 0.01],
                [0.85, 0.10, 0.05],
                [0.70, 0.20, 0.10],
                [0.20, 0.70, 0.10],
                [0.10, 0.80, 0.10],
                [0.05, 0.90, 0.05],
                [0.20, 0.10, 0.70],
                [0.10, 0.20, 0.70],
                [0.05, 0.10, 0.85],
            ]
        )

        curves = bkg_rejection_at_eff(y_true, y_score)

        self.assertEqual(set(curves), {"bc", "bd", "cb", "cd"})
        self.assertTrue(all(len(values) == 7 for values in curves.values()))
        self.assertTrue(np.isfinite(bkg_rejection_bc_score(y_true, y_score)))
        self.assertTrue(np.isfinite(bkg_rejection_bd_score(y_true, y_score)))
        self.assertTrue(np.isfinite(bkg_rejection_cb_score(y_true, y_score)))
        self.assertTrue(np.isfinite(bkg_rejection_cd_score(y_true, y_score)))
        self.assertTrue(np.isfinite(b_tag_rejection_score(y_true, y_score)))
        self.assertTrue(np.isfinite(c_tag_rejection_score(y_true, y_score)))
        self.assertTrue(np.isfinite(bkg_rejection_score(y_true, y_score)))


if __name__ == "__main__":
    unittest.main()

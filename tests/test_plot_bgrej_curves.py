import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports import plot_bgrej_curves


class PlotBgrejCurvesTest(unittest.TestCase):
    def test_bgrej_curves_parser_reads_last_eval_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "generation-001.log"
            log_path.write_text(
                """
INFO: Evaluation metrics:
    - bkg_rejection_at_eff:
{'bc': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'bd': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'cb': [3, 4, 5, 6, 7, 8, 9, 10, 11], 'cd': [4, 5, 6, 7, 8, 9, 10, 11, 12]}
    - bkg_rejection_score:
1.0
INFO: Evaluation metrics:
    - bkg_rejection_at_eff:
{'bc': [11, 12, 13, 14, 15, 16, 17, 18, 19], 'bd': [12, 13, 14, 15, 16, 17, 18, 19, 20], 'cb': [13, 14, 15, 16, 17, 18, 19, 20, 21], 'cd': [14, 15, 16, 17, 18, 19, 20, 21, 22]}
    - bkg_rejection_score:
2.0
""",
                encoding="utf-8",
            )

            curves = plot_bgrej_curves.parse_bgrej_curves(log_path)

            self.assertEqual(curves["bc"][0], 11)
            self.assertEqual(curves["cd"][-1], 22)
            self.assertEqual(plot_bgrej_curves.efficiency_points_for_curves(curves)[-1], 1.0)


if __name__ == "__main__":
    unittest.main()

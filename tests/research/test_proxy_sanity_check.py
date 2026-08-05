import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from research.proxy_sanity_check import (
    class_balance,
    positional_bias_note,
    run_sanity_check,
    source_file_check,
    train_overlap_check,
)


def _manifest(control_rows=None, full_rows=None):
    control_rows = control_rows or {"bb": 5000, "cc": 5000, "dd": 5000}
    full_rows = full_rows or {"bb": 900000, "cc": 900000, "dd": 900000}
    return {
        "levels": {
            "monitor": {
                "suffix": "val50k_tail",
                "rows_by_flavor": control_rows,
                "files": [
                    {"flavor": flavor, "path": f"x_{flavor}_val50k_tail.parquet", "source": f"x_{flavor}_val1000k.parquet", "source_start_row": 100, "source_stop_row": 200}
                    for flavor in control_rows
                ],
            },
            "full_holdout": {
                "suffix": "val_holdout",
                "rows_by_flavor": full_rows,
                "files": [
                    {"flavor": flavor, "path": f"x_{flavor}_val_holdout.parquet", "source": f"x_{flavor}_val1000k.parquet", "source_start_row": 0, "source_stop_row": 100}
                    for flavor in full_rows
                ],
            },
        },
        "strategy": {"source_suffix": "val1000k"},
    }


class ClassBalanceTest(unittest.TestCase):
    def test_balanced_classes_report_equal_fractions(self):
        manifest = _manifest()
        result = class_balance(manifest, "monitor")
        self.assertEqual(result["rows_total"], 15000)
        for fraction in result["fraction_by_flavor"].values():
            self.assertAlmostEqual(fraction, 1 / 3)

    def test_dominated_class_is_visible_in_fractions(self):
        manifest = _manifest(control_rows={"bb": 90000, "cc": 5000, "dd": 5000})
        result = class_balance(manifest, "monitor")
        self.assertGreater(result["fraction_by_flavor"]["bb"], 0.85)


class SourceFileCheckTest(unittest.TestCase):
    def test_single_source_per_flavor_is_reported_not_applicable(self):
        manifest = _manifest()
        result = source_file_check(manifest, "monitor")
        self.assertTrue(result["single_source_file_per_flavor"])
        self.assertIn("not_applicable", result["note"])


class TrainOverlapCheckTest(unittest.TestCase):
    def test_no_train800k_reference_means_no_overlap(self):
        manifest = _manifest()
        result = train_overlap_check(manifest)
        self.assertFalse(result["train800k_referenced_anywhere_in_manifest"])

    def test_train800k_reference_is_flagged(self):
        manifest = _manifest()
        manifest["levels"]["monitor"]["files"][0]["source"] = "x_bb_train800k.parquet"
        result = train_overlap_check(manifest)
        self.assertTrue(result["train800k_referenced_anywhere_in_manifest"])
        self.assertIn("re-examine", result["note"])


class PositionalBiasNoteTest(unittest.TestCase):
    def test_reports_row_ranges_and_marks_hypothesis_not_a_finding(self):
        manifest = _manifest()
        result = positional_bias_note(manifest)
        self.assertIn("bb", result["row_ranges_by_flavor"])
        self.assertEqual(result["row_ranges_by_flavor"]["bb"]["control_proxy_50k_rows"], [100, 200])
        self.assertEqual(result["row_ranges_by_flavor"]["bb"]["full_validation_rows"], [0, 100])
        self.assertIn("hypothesis", result["hypothesis"].lower())


class RunSanityCheckTest(unittest.TestCase):
    def test_produces_every_required_section_from_a_real_manifest_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()))
            result = run_sanity_check(manifest_path)

        for key in (
            "control_proxy_50k_class_balance",
            "full_validation_class_balance",
            "control_proxy_50k_source_files",
            "full_validation_source_files",
            "positional_bias",
            "train_overlap",
            "kinematic_variables",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["kinematic_variables"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

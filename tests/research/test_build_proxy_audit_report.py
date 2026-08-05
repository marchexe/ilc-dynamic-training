import tempfile
import unittest
from pathlib import Path

from tests.helpers import SCRIPTS_DIR  # noqa: F401

from reports.build_proxy_audit_report import build_nightly_result_md, determine_verdict


def _row(checkpoint_id, control, full, control_status="completed", full_status="completed"):
    return {
        "checkpoint_id": checkpoint_id,
        "control_proxy_50k_status": control_status,
        "control_proxy_50k_working_point_mistag_percent": str(control) if control is not None else "",
        "full_validation_status": full_status,
        "full_validation_working_point_mistag_percent": str(full) if full is not None else "",
    }


def _summary(checkpoints_distinct, proxy_vs_full, sanity=None):
    return {
        "checkpoints_distinct": checkpoints_distinct,
        "checkpoints_requested": checkpoints_distinct,
        "checkpoints_duplicate": [],
        "proxy_vs_full": proxy_vs_full,
        "proxy_sanity_check": sanity or {},
        "runtime_seconds": {"control_proxy_50k": 60.0, "full_validation": 600.0, "total": 700.0},
        "git": {"commit": "abc123", "dirty": False},
        "audit_config": {"metric_name": "validation_working_point_mistag_percent", "metric_mode": "min", "dataset": "d", "data_config": "c"},
        "config_path": "configs/research/nightly_proxy_audit.yaml",
        "run_id": "test_run",
    }


class DetermineVerdictTest(unittest.TestCase):
    def test_zero_checkpoints_is_incomplete(self):
        summary = _summary(0, {})
        verdict, _ = determine_verdict(summary, [])
        self.assertEqual(verdict, "experiment incomplete")

    def test_fewer_than_six_completed_is_incomplete(self):
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(3)]
        summary = _summary(3, {"insufficient_evidence": False})
        verdict, _ = determine_verdict(summary, rows)
        self.assertEqual(verdict, "experiment incomplete")

    def test_severe_class_imbalance_is_data_integrity_failed(self):
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(8)]
        sanity = {
            "control_proxy_50k_class_balance": {"fraction_by_flavor": {"bb": 0.9, "cc": 0.05, "dd": 0.05}},
            "full_validation_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "train_overlap": {"train800k_referenced_anywhere_in_manifest": False},
        }
        summary = _summary(8, {"insufficient_evidence": False}, sanity=sanity)
        verdict, reason = determine_verdict(summary, rows)
        self.assertEqual(verdict, "validation data integrity failed")
        self.assertIn("imbalanced", reason)

    def test_train_overlap_is_data_integrity_failed(self):
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(8)]
        sanity = {
            "control_proxy_50k_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "full_validation_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "train_overlap": {"train800k_referenced_anywhere_in_manifest": True},
        }
        summary = _summary(8, {"insufficient_evidence": False}, sanity=sanity)
        verdict, _ = determine_verdict(summary, rows)
        self.assertEqual(verdict, "validation data integrity failed")

    def test_strong_correlation_and_agreement_is_supported(self):
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(8)]
        sanity = {
            "control_proxy_50k_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "full_validation_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "train_overlap": {"train800k_referenced_anywhere_in_manifest": False},
        }
        proxy_vs_full = {
            "insufficient_evidence": False,
            "pearson_spearman": {"spearman_rho": 0.95},
            "pairwise_direction_agreement": {"agreement_fraction": 0.9},
            "best_checkpoint_agreement": {"agrees": True},
        }
        summary = _summary(8, proxy_vs_full, sanity=sanity)
        verdict, _ = determine_verdict(summary, rows)
        self.assertEqual(verdict, "50k proxy supported for limited control use")

    def test_weak_correlation_is_unreliable(self):
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(8)]
        sanity = {
            "control_proxy_50k_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "full_validation_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "train_overlap": {"train800k_referenced_anywhere_in_manifest": False},
        }
        proxy_vs_full = {
            "insufficient_evidence": False,
            "pearson_spearman": {"spearman_rho": 0.1},
            "pairwise_direction_agreement": {"agreement_fraction": 0.4},
            "best_checkpoint_agreement": {"agrees": False},
        }
        summary = _summary(8, proxy_vs_full, sanity=sanity)
        verdict, _ = determine_verdict(summary, rows)
        self.assertEqual(verdict, "50k proxy ranking unreliable")

    def test_moderate_correlation_is_promising_but_insufficient(self):
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(8)]
        sanity = {
            "control_proxy_50k_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "full_validation_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "train_overlap": {"train800k_referenced_anywhere_in_manifest": False},
        }
        proxy_vs_full = {
            "insufficient_evidence": False,
            "pearson_spearman": {"spearman_rho": 0.6},
            "pairwise_direction_agreement": {"agreement_fraction": 0.65},
            "best_checkpoint_agreement": {"agrees": True},
        }
        summary = _summary(8, proxy_vs_full, sanity=sanity)
        verdict, _ = determine_verdict(summary, rows)
        self.assertEqual(verdict, "50k proxy promising but evidence insufficient")

    def test_never_claims_supported_use_from_insufficient_evidence(self):
        """Even if the (too-few-points) correlation happens to look
        perfect, insufficient_evidence must win -- never overclaim from a
        small sample."""
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(8)]
        sanity = {
            "control_proxy_50k_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "full_validation_class_balance": {"fraction_by_flavor": {"bb": 0.33, "cc": 0.33, "dd": 0.34}},
            "train_overlap": {"train800k_referenced_anywhere_in_manifest": False},
        }
        proxy_vs_full = {
            "insufficient_evidence": True,
            "pearson_spearman": {"spearman_rho": 1.0},
            "pairwise_direction_agreement": {"agreement_fraction": 1.0},
            "best_checkpoint_agreement": {"agrees": True},
        }
        summary = _summary(8, proxy_vs_full, sanity=sanity)
        verdict, _ = determine_verdict(summary, rows)
        self.assertNotEqual(verdict, "50k proxy supported for limited control use")
        self.assertEqual(verdict, "50k proxy promising but evidence insufficient")


class BuildNightlyResultMdPartialResultsTest(unittest.TestCase):
    def test_renders_from_empty_summary_and_rows_without_crashing(self):
        """A run interrupted before summary.json was even written must
        still get a report -- build_nightly_result_md must not KeyError on
        an empty dict."""
        content = build_nightly_result_md(Path("/tmp/does-not-matter"), {}, [])
        self.assertIn("# Nightly Result", content)
        self.assertIn("experiment incomplete", content)

    def test_renders_from_partial_rows_with_missing_tiers(self):
        summary = _summary(2, {"insufficient_evidence": True})
        rows = [
            _row("c0", 1.0, 1.1),
            _row("c1", None, None, control_status="failed", full_status="failed"),
        ]
        content = build_nightly_result_md(Path("/tmp/does-not-matter"), summary, rows)
        self.assertIn("c0", content)
        self.assertIn("c1", content)
        self.assertIn("FAILED", content)

    def test_reproduction_command_is_included(self):
        summary = _summary(8, {"insufficient_evidence": False, "pearson_spearman": {}, "pairwise_direction_agreement": {}, "best_checkpoint_agreement": {}})
        rows = [_row(f"c{i}", 1.0, 1.1) for i in range(8)]
        content = build_nightly_result_md(Path("/tmp/does-not-matter"), summary, rows)
        self.assertIn("run_proxy_audit.py", content)
        self.assertIn("build_proxy_audit_report.py", content)


if __name__ == "__main__":
    unittest.main()

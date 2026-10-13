import math
import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_DIR, namespace

from training import runtime, weaver


def resolved_config(**shared):
    """Minimal resolved configuration accepted by weaver.build_command."""
    base = dict(dataset="/tmp/sgv", data_extension="root", checkpoint=str(PROJECT_DIR / "checkpoint.pt"),
                data_config="/tmp/data.yaml", network_config="/tmp/network.py", seed=12345, epochs=1,
                samples_per_epoch=1024, samples_per_epoch_val=1024, batch_size=512, start_lr=1.0e-4,
                optimizer="ranger", lr_scheduler="none", num_workers=1, fetch_step=0.01, use_amp=True,
                amp_dtype="fp16", no_remake_weights=True)
    return dict(shared=dict(base, **shared))


class WeaverCommandTest(unittest.TestCase):
    def test_worker_overrides_seed_and_lr(self):
        worker = dict(name="fixed_lr_125e-4", gpu="0", seed=12347, start_lr=1.25e-4, controller=None)
        command = weaver.build_command(resolved_config(), worker, Path("/tmp/unit/fixed_lr_125e-4"), resume_epoch=None)

        self.assertEqual(command[command.index("--start-lr") + 1], "0.000125")
        self.assertEqual(command[command.index("--seed") + 1], "12347")

    def test_project_path_expands_portable_aliases(self):
        self.assertEqual(
            runtime.project_path("${PROJECT_DIR}/configs").resolve(),
            PROJECT_DIR / "configs",
        )
        self.assertEqual(
            runtime.project_path("${PART_DATA_DIR}/data.yaml"),
            PROJECT_DIR.parent / "data/data.yaml",
        )

    def test_data_command_args_are_shared_by_launchers(self):
        args = runtime.data_command_args("/tmp/sgv", "parquet")

        self.assertEqual(args[0], "--data-train")
        self.assertEqual(args[4], "--data-val")
        self.assertEqual(args[1], "nnbb:/tmp/sgv/*_bb_train800k.parquet")
        self.assertEqual(args[6], "nncc:/tmp/sgv/*_cc_val50k.parquet")
        self.assertNotIn(".root", " ".join(args))

    def test_data_command_args_can_use_separate_validation_dataset_and_suffix(self):
        args = runtime.data_command_args(
            "/tmp/train_sgv",
            "parquet",
            "/tmp/proxy_sgv",
            train_suffix="train800k",
            validation_suffix="val5k_tail",
        )

        self.assertEqual(args[0], "--data-train")
        self.assertEqual(args[4], "--data-val")
        self.assertEqual(args[1], "nnbb:/tmp/train_sgv/*_bb_train800k.parquet")
        self.assertEqual(args[5], "nnbb:/tmp/proxy_sgv/*_bb_val5k_tail.parquet")
        self.assertEqual(args[7], "nndd:/tmp/proxy_sgv/*_dd_val5k_tail.parquet")

    def test_command_can_use_parquet_data_and_prefetch(self):
        resolved = resolved_config(dataset="/tmp/sgv_parquet", data_extension="parquet", prefetch_factor=4)
        worker = dict(name="fixed_lr_100e-4", gpu="0", seed=12345, start_lr=1.0e-4, controller=None)
        command = weaver.build_command(resolved, worker, Path("/tmp/unit/fixed_lr_100e-4"), resume_epoch=None)

        joined = " ".join(command)
        self.assertIn("nnbb:/tmp/sgv_parquet/*_bb_train800k.parquet", joined)
        self.assertIn("--prefetch-factor", command)
        self.assertEqual(command[command.index("--prefetch-factor") + 1], "4")

    def test_read_metrics_keeps_bgrej_curves_and_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "train.log"
            log_path.write_text(
                """
Overrode loaded optimizer learning rate: 2e-05 -> 1.9e-05
Train AvgLoss: 0.27196, AvgAcc: 0.89479
Max Grad Norm: 0.69661
AMP skipped optimizer steps: 0
Max CUDA memory: 12345.0 MB
Eval AvgLoss: 0.5, AvgAcc: 0.75
INFO: Evaluation metrics:
    - bkg_rejection_at_eff:
{'bc': [1, 2, 3, 4, 5, 6, 80, 90, 100], 'bd': [2, 3, 4, 5, 6, 7, 81, 91, 101], 'cb': [3, 4, 5, 6, 7, 8, 82, 92, 102], 'cd': [4, 5, 6, 7, 8, 9, 83, 93, 103]}
    - bkg_rejection_score:
2.0
""",
                encoding="utf-8",
            )

            metrics = runtime.read_metrics(log_path)

            self.assertEqual(metrics["validation_bkg_rejection_at_eff"]["efficiencies"][-1], 1.0)
            lookup = metrics["validation_bkg_rejection_at_eff_lookup"]
            self.assertEqual(lookup["c_tag_eff_0.50"]["b_bkg_rejection"], 6)
            self.assertEqual(lookup["b_tag_eff_0.80"]["c_bkg_rejection"], 80)
            self.assertEqual(lookup["b_tag_eff_0.90"]["d_bkg_rejection"], 91)
            self.assertEqual(lookup["c_tag_eff_1.00"]["d_bkg_rejection"], 103)
            self.assertAlmostEqual(
                metrics["validation_working_point_mistag_percent"],
                sum(100.0 / value for value in (80, 81, 90, 91, 6, 7, 82, 83)) / 8,
            )
            self.assertAlmostEqual(
                metrics["validation_ctag_reference_mistag_percent"],
                sum(100.0 / value for value in (6, 7, 82, 83)) / 4,
            )
            self.assertAlmostEqual(metrics["validation_bc_mistag_eff_0.80_percent"], 100.0 / 80)
            self.assertAlmostEqual(metrics["train_loss"], 0.27196)
            self.assertAlmostEqual(metrics["train_accuracy"], 0.89479)
            self.assertAlmostEqual(metrics["train_max_grad_norm"], 0.69661)
            self.assertEqual(metrics["train_amp_skipped_optimizer_steps"], 0)
            self.assertAlmostEqual(metrics["train_max_cuda_memory_mb"], 12345.0)
            self.assertAlmostEqual(metrics["train_loaded_optimizer_lr"], 1.9e-5)

    def test_read_metrics_accepts_nan_in_bgrej_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "train.log"
            log_path.write_text(
                """
Eval AvgLoss: 0.50000, AvgAcc: 0.80000
INFO: Evaluation metrics:
    - bkg_rejection_at_eff:
{'bc': [nan, 2.0], 'bd': [3.0, 4.0]}
    - bkg_rejection_score:
1.0
""",
                encoding="utf-8",
            )

            metrics = runtime.read_metrics(log_path)

            value = metrics["validation_bkg_rejection_at_eff"]["pairs"]["bc"][0]
            self.assertNotEqual(value, value)

    def test_read_metrics_excludes_non_finite_values_from_working_point_mean(self):
        # A NaN rejection at one of the eight fixed working points (e.g. a
        # zero-background-event edge case) must not poison the mean fed to
        # PBT ranking -- it should be dropped, not silently turn the whole
        # mean into NaN.
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "train.log"
            log_path.write_text(
                """
Eval AvgLoss: 0.50000, AvgAcc: 0.80000
INFO: Evaluation metrics:
    - bkg_rejection_at_eff:
{'bc': [3, 4, 5, 6, 7, 8, nan, 91, 100], 'bd': [2, 3, 4, 5, 6, 7, 81, 91, 101], 'cb': [3, 4, 5, 6, 7, 8, 82, 92, 102], 'cd': [4, 5, 6, 7, 8, 9, 83, 93, 103]}
    - bkg_rejection_score:
1.0
""",
                encoding="utf-8",
            )

            metrics = runtime.read_metrics(log_path)

            self.assertIsNone(metrics["validation_bc_mistag_eff_0.80_percent"])
            finite_working_points = (81, 91, 91, 6, 7, 82, 83)
            self.assertAlmostEqual(
                metrics["validation_working_point_mistag_percent"],
                sum(100.0 / value for value in finite_working_points) / len(finite_working_points),
            )
            self.assertTrue(math.isfinite(metrics["validation_working_point_mistag_percent"]))

    def test_read_metrics_keeps_bgrej_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "train.log"
            log_path.write_text(
                """
Eval AvgLoss: 0.50000, AvgAcc: 0.80000
INFO: Evaluation metrics:
    - bkg_rejection_at_eff_counts:
{'bc': [{'signal_efficiency': 0.8, 'background_passed': 12, 'background_total': 1000, 'background_efficiency': 0.012}], 'bd': []}
    - bkg_rejection_score:
1.0
""",
                encoding="utf-8",
            )

            metrics = runtime.read_metrics(log_path)

            counts = metrics["validation_bkg_rejection_at_eff_counts"]
            self.assertEqual(counts["bc"][0]["background_passed"], 12)
            self.assertEqual(counts["bc"][0]["background_total"], 1000)
            self.assertEqual(counts["bc"][0]["background_efficiency"], 0.012)
            self.assertEqual(metrics["validation_working_point_mistag_percent_uncertainty_points"], 1)
            alpha, beta = 12.5, 988.5
            self.assertAlmostEqual(
                metrics["validation_working_point_mistag_percent_uncertainty"],
                100.0 * (alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) ** 0.5,
            )


if __name__ == "__main__":
    unittest.main()

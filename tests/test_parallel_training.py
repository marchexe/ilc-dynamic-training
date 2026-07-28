import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_DIR, namespace

from training.comparison import runner as comparison_runner
from training import runtime, weaver


class ParallelTrainingTest(unittest.TestCase):
    def test_parallel_worker_can_override_seed_and_lr(self):
        resolved = comparison_runner.load_and_resolve(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pbt_control_fixed_lr.yaml",
                experiment_name="unit_fixed_sweep",
                gpus=None,
                smoke=True,
            )
        )
        worker = next(
            item for item in resolved["workers"]
            if item["name"] == "fixed_lr_125e-4"
        )
        command = weaver.build_command(
            resolved,
            worker,
            PROJECT_DIR / "runs/parallel/unit_fixed_sweep/fixed_lr_125e-4",
            resume_epoch=None,
        )

        self.assertEqual(command[command.index("--start-lr") + 1], "0.000125")
        self.assertEqual(command[command.index("--seed") + 1], "12347")

    def test_data_command_args_are_shared_by_launchers(self):
        args = runtime.data_command_args("/tmp/sgv", "parquet")

        self.assertEqual(args[0], "--data-train")
        self.assertEqual(args[4], "--data-val")
        self.assertEqual(args[1], "nnbb:/tmp/sgv/*_bb_train800k.parquet")
        self.assertEqual(args[6], "nncc:/tmp/sgv/*_cc_val50k.parquet")
        self.assertNotIn(".root", " ".join(args))

    def test_parallel_command_can_use_parquet_data_and_prefetch(self):
        resolved = comparison_runner.load_and_resolve(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pbt_control_fixed_lr.yaml",
                experiment_name="unit_fixed_sweep",
                gpus=None,
                smoke=True,
            )
        )
        resolved["shared"]["dataset"] = "/tmp/sgv_parquet"
        resolved["shared"]["data_extension"] = "parquet"
        resolved["shared"]["prefetch_factor"] = 4
        worker = resolved["workers"][0]

        command = weaver.build_command(
            resolved,
            worker,
            PROJECT_DIR / "runs/parallel/unit_fixed_sweep/fixed_lr_100e-4",
            resume_epoch=None,
        )

        joined = " ".join(command)
        self.assertIn("nnbb:/tmp/sgv_parquet/*_bb_train800k.parquet", joined)
        self.assertIn("--prefetch-factor", command)
        self.assertEqual(command[command.index("--prefetch-factor") + 1], "4")

    def test_read_metrics_keeps_bgrej_curves_and_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "train.log"
            log_path.write_text(
                """
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


if __name__ == "__main__":
    unittest.main()

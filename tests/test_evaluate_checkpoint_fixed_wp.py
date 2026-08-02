import tempfile
import unittest
from pathlib import Path

from tests.helpers import namespace
from training.runtime import PROJECT_DIR
from validation.evaluate_checkpoint_fixed_wp import build_manifest, build_test_command


class EvaluateCheckpointFixedWPTest(unittest.TestCase):
    def test_build_test_command_accepts_validation_suffix(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            args = namespace(
                checkpoint=Path("checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt"),
                dataset=Path("datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet"),
                data_extension="parquet",
                data_config=Path("checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/data_config.auto.yaml"),
                network_config=Path("networks/pretrained_sgv_particle_transformer.py"),
                batch_size=1024,
                num_workers=1,
                fetch_step="0.01",
                gpu="0",
                amp_dtype="fp16",
                validation_suffix="val50k_tail",
            )

            command = build_test_command(args, Path(temporary) / "eval.log")

        joined = " ".join(command)
        self.assertIn("*_bb_val50k_tail.parquet", joined)
        self.assertIn("*_cc_val50k_tail.parquet", joined)
        self.assertIn("*_dd_val50k_tail.parquet", joined)
        self.assertNotIn("*_bb_val50k.parquet", joined)

    def test_build_manifest_records_validation_suffix(self):
        args = namespace(
            checkpoint=Path("checkpoint.pt"),
            name="unit_eval",
            output_root=Path("runs/eval"),
            dataset=Path("datasets/proxy"),
            data_extension="parquet",
            data_config=Path("data.yaml"),
            network_config=Path("network.py"),
            samples_per_epoch_val=150000,
            batch_size=1024,
            num_workers=1,
            fetch_step="0.01",
            amp_dtype="fp16",
            validation_suffix="val1000k",
        )

        manifest = build_manifest(
            args,
            {
                "validation_working_point_mistag_percent": 1.0,
                "validation_bkg_rejection_at_eff": {"bb": {}},
            },
            PROJECT_DIR / "runs/eval/unit_eval/checkpoint_eval.log",
            ["weaver"],
        )

        self.assertEqual(manifest["config"]["shared"]["validation_suffix"], "val1000k")


if __name__ == "__main__":
    unittest.main()

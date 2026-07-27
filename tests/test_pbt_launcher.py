import unittest

from tests.helpers import PROJECT_DIR, namespace, pbt_smoke_config
from training.pbt import config as config_module, weaver


class PBTLauncherTest(unittest.TestCase):
    def test_smoke_config_and_resume_command(self):
        config = pbt_smoke_config()
        self.assertEqual(config["shared"]["generations"], 2)
        self.assertEqual(len(config["population"]), 2)
        self.assertEqual(config["shared"]["data_extension"], "root")

        member = {"name": "member_00", "lr": 9.0e-5}
        command, log_path, target_epoch = weaver.make_command(
            config,
            member,
            "0",
            PROJECT_DIR / "runs/pbt/unit_test/member_00",
            generation=1,
        )

        self.assertEqual(target_epoch, 1)
        self.assertEqual(command[command.index("--load-epoch") + 1], "0")
        self.assertIn("--override-load-lr", command)
        self.assertIn("--training-controller", command)
        self.assertEqual(
            command[command.index("--training-controller") + 1],
            config["shared"]["training_controller"],
        )
        self.assertEqual(command[command.index("--start-lr") + 1], "9e-05")
        self.assertEqual(command[command.index("--seed") + 1], "12346")
        self.assertEqual(log_path.name, "generation-001.log")

    def test_pbt_command_can_use_parquet_data(self):
        config = pbt_smoke_config()
        config["shared"]["dataset"] = "/tmp/sgv_parquet"
        config["shared"]["data_extension"] = "parquet"

        command, _, _ = weaver.make_command(
            config,
            {"name": "member_00", "lr": 9.0e-5},
            "0",
            PROJECT_DIR / "runs/pbt/unit_test/member_00",
            generation=0,
        )

        joined = " ".join(command)
        self.assertIn("nnbb:/tmp/sgv_parquet/*_bb_train800k.parquet", joined)
        self.assertIn("nncc:/tmp/sgv_parquet/*_cc_val50k.parquet", joined)
        self.assertNotIn(".root", joined)

    def test_remote_slots_wrap_worker_command_in_ssh(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pbt_smoke.yaml",
                experiment_name="unit_test_remote",
                gpus=None,
                slots="iutgpu01:6,iutgpu05:4",
                smoke=True,
            )
        )

        self.assertEqual(
            [slot["label"] for slot in config["slots"]],
            ["iutgpu01:6", "iutgpu05:4"],
        )

        member = {"name": "member_00", "lr": 9.0e-5}
        command, _, _ = weaver.make_command(
            config,
            member,
            config["slots"][0],
            PROJECT_DIR / "runs/pbt/unit_test_remote/member_00",
            generation=0,
        )

        self.assertEqual(command[:2], ["ssh", "iutgpu01"])
        self.assertIn(".venv/bin/python", command[2])
        self.assertIn("remote venv python is not executable", command[2])
        self.assertIn("--gpus 6", command[2])


if __name__ == "__main__":
    unittest.main()

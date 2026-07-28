import importlib.util
import unittest

from tests.helpers import PROJECT_DIR, namespace, pbt_smoke_config
from training.pbt import config as config_module
from training.pbt.backend import LocalWeaverBackend, backend_from_config
from training.pbt.ray_backend import RayWeaverBackend


class PBTLauncherTest(unittest.TestCase):
    def setUp(self):
        self.backend = LocalWeaverBackend()

    def test_default_backend_is_local_weaver(self):
        config = pbt_smoke_config()

        self.assertIsInstance(backend_from_config(config), LocalWeaverBackend)
        self.assertEqual(config["pbt"]["backend"], "local_weaver")

    def test_ray_backend_uses_common_runner_contract(self):
        config = pbt_smoke_config()
        config["pbt"]["backend"] = "ray_weaver"
        backend = backend_from_config(config)

        self.assertIsInstance(backend, RayWeaverBackend)
        self.assertFalse(getattr(backend, "handles_run", False))
        command, _, _ = backend.command_for(
            config,
            {"name": "member_00", "lr": 9.0e-5},
            "0",
            PROJECT_DIR / "runs/pbt/unit_ray/member_00",
            generation=0,
        )
        self.assertIn("--start-lr", command)

    def test_ray_backend_dependency_error_is_clear(self):
        if importlib.util.find_spec("ray") is not None:
            self.skipTest("Ray is installed in this environment")
        from training.pbt.ray_backend import _ray_import

        with self.assertRaisesRegex(RuntimeError, "Ray backend requires"):
            _ray_import()


    def test_anchored_ray_config_uses_ray_weaver_backend(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pbt_anchored_lr_sweep_ray.yaml",
                experiment_name="unit_anchored_ray",
                gpus="0,1",
                slots=None,
                smoke=True,
            )
        )

        self.assertEqual(config["pbt"]["backend"], "ray_weaver")
        self.assertEqual(config["pbt"]["strategy"], "anchored_lr_sweep")
        self.assertIsInstance(backend_from_config(config), RayWeaverBackend)

    def test_legacy_ray_tune_name_aliases_to_ray_weaver_backend(self):
        config = pbt_smoke_config()
        config["pbt"]["backend"] = "ray_tune"

        self.assertIsInstance(backend_from_config(config), RayWeaverBackend)


    def test_anchored_lr_sweep_config_generates_start_lrs(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pbt_anchored_lr_sweep.yaml",
                experiment_name="unit_anchored",
                gpus="0,1,2,3",
                slots=None,
                smoke=False,
            )
        )

        self.assertEqual(config["pbt"]["strategy"], "anchored_lr_sweep")
        self.assertEqual(
            [round(member["start_lr"], 10) for member in config["population"]],
            [0.00011025, 0.000107625, 0.000102375, 0.00009975],
        )

    def test_anchored_lr_sweep_smoke_uses_high_and_low_branches(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pbt_anchored_lr_sweep.yaml",
                experiment_name="unit_anchored_smoke",
                gpus="0,1",
                slots=None,
                smoke=True,
            )
        )

        self.assertEqual(
            [round(member["start_lr"], 10) for member in config["population"]],
            [0.00011025, 0.00009975],
        )

    def test_smoke_config_and_resume_command(self):
        config = pbt_smoke_config()
        self.assertEqual(config["shared"]["generations"], 2)
        self.assertEqual(len(config["population"]), 2)
        self.assertEqual(config["shared"]["data_extension"], "root")

        member = {"name": "member_00", "lr": 9.0e-5}
        command, log_path, target_epoch = self.backend.command_for(
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
        self.assertEqual(command[command.index("--optimizer") + 1], "adamw")
        self.assertEqual(command[command.index("--start-lr") + 1], "9e-05")
        self.assertEqual(command[command.index("--seed") + 1], "12346")
        self.assertEqual(log_path.name, "generation-001.log")

    def test_pbt_command_can_use_parquet_data(self):
        config = pbt_smoke_config()
        config["shared"]["dataset"] = "/tmp/sgv_parquet"
        config["shared"]["data_extension"] = "parquet"

        command, _, _ = self.backend.command_for(
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
        command, _, _ = self.backend.command_for(
            config,
            member,
            config["slots"][0],
            PROJECT_DIR / "runs/pbt/unit_test_remote/member_00",
            generation=0,
        )

        self.assertEqual(command[:2], ["ssh", "iutgpu01"])
        self.assertIn(".venv/bin/python", command[2])
        self.assertIn("sys.version_info[:2] != (", command[2])
        self.assertIn("command -v python", command[2])
        self.assertIn("remote Python", command[2])
        self.assertIn("--gpus 6", command[2])

    def test_fixed_lr_grid_config_preserves_low_lr_grid(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/finetune_fixed_lr_grid_adamw.yaml",
                experiment_name="unit_fixed_grid",
                gpus="0,1,2,3",
                slots=None,
                smoke=False,
            )
        )

        self.assertEqual(config["pbt"]["strategy"], "fixed_lr_grid")
        self.assertEqual(config["pbt"]["metric"], "validation_working_point_mistag_percent")
        self.assertEqual(config["pbt"]["mode"], "min")
        self.assertEqual(config["pbt"]["early_stop_degraded_generations"], 4)
        self.assertEqual(
            [member["start_lr"] for member in config["population"]],
            [1.0e-5, 2.0e-5, 3.0e-5, 5.0e-5],
        )

    def test_head_warmup_freezes_backbone_only_for_configured_generations(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/finetune_head_warmup_fixed_lr_grid.yaml",
                experiment_name="unit_head_warmup",
                gpus="0,1,2,3",
                slots=None,
                smoke=False,
            )
        )
        member = {"name": "lr_1e_5", "lr": 1.0e-5}

        command_gen0, _, _ = self.backend.command_for(
            config,
            member,
            "0",
            PROJECT_DIR / "runs/pbt/unit_head_warmup/lr_1e_5",
            generation=0,
        )
        command_gen2, _, _ = self.backend.command_for(
            config,
            member,
            "0",
            PROJECT_DIR / "runs/pbt/unit_head_warmup/lr_1e_5",
            generation=2,
        )

        self.assertIn("--freeze-model-weights", command_gen0)
        self.assertIn("part\\.blocks", command_gen0[command_gen0.index("--freeze-model-weights") + 1])
        self.assertNotIn("--freeze-model-weights", command_gen2)

    def test_optimizer_options_are_forwarded_to_weaver(self):
        config = pbt_smoke_config()
        config["shared"]["optimizer_options"] = {"weight_decay": "1e-4"}

        command, _, _ = self.backend.command_for(
            config,
            {"name": "member_00", "lr": 9.0e-5},
            "0",
            PROJECT_DIR / "runs/pbt/unit_optimizer_options/member_00",
            generation=0,
        )

        self.assertIn("--optimizer-option", command)
        index = command.index("--optimizer-option")
        self.assertEqual(command[index + 1:index + 3], ["weight_decay", "1e-4"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_DIR, namespace, pbt_smoke_config
from training.pbt import config as config_module
from training.pbt import strategy
from training.pbt.backend import LocalWeaverBackend, backend_from_config
from training.pbt.ray_backend import RayWeaverBackend
from training.pbt.tune_runner import build_trial_specs, ray_runtime_env, small_tune_payload
from training.pbt.tune_trainable import (
    TUNE_CONTROLLER_NAME,
    TUNE_METADATA_NAME,
    TUNE_OPTIMIZER_NAME,
    TUNE_STATE_NAME,
    config_from_tune_payload,
    package_tune_checkpoint,
)


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

    def test_ray_tune_trial_specs_use_resolved_population_and_logical_gpu(self):
        config = pbt_smoke_config()

        trial_specs = build_trial_specs(config, generations=1)

        self.assertEqual([trial["member_name"] for trial in trial_specs], ["member_00", "member_01"])
        self.assertEqual([trial["lr"] for trial in trial_specs], [7.5e-5, 1.0e-4])
        self.assertEqual({trial["generations"] for trial in trial_specs}, {1})
        self.assertEqual({trial["slot"]["gpu"] for trial in trial_specs}, {"0"})
        self.assertEqual({trial["slot"]["label"] for trial in trial_specs}, {"ray:gpu0"})

    def test_ray_tune_payload_includes_output_root(self):
        config = pbt_smoke_config()
        trial = build_trial_specs(config, generations=1)[0]

        payload = small_tune_payload(config, trial)

        self.assertEqual(payload["output_root"], str(config["output_root"]))

    def test_ray_runtime_env_makes_project_modules_importable(self):
        pythonpath = ray_runtime_env()["env_vars"]["PYTHONPATH"].split(":")

        self.assertIn(str(PROJECT_DIR / "scripts"), pythonpath)
        self.assertIn(str(PROJECT_DIR / "weaver-core"), pythonpath)

    def test_ray_tune_trainable_can_rebuild_config_from_small_payload(self):
        config = config_from_tune_payload(
            {
                "config_path": str(PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"),
                "experiment_name": "unit_tune_payload",
                "gpus": "0,2",
                "smoke": True,
            }
        )

        self.assertEqual(config["experiment_name"], "unit_tune_payload")
        self.assertEqual(config["gpus"], ["0", "2"])
        self.assertEqual(len(config["population"]), 2)

    def test_ray_tune_checkpoint_package_is_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member_dir = root / "member_00"
            checkpoint_dir = root / "ray_checkpoint"
            member_dir.mkdir()
            state_path, optimizer_path = strategy.checkpoint_paths(member_dir, 3)
            controller_path = strategy.controller_checkpoint_path(member_dir, 3)
            state_path.write_bytes(b"state")
            optimizer_path.write_bytes(b"optimizer")
            controller_path.write_bytes(b"controller")

            metadata = package_tune_checkpoint(
                member_dir,
                3,
                checkpoint_dir,
                {"member": "member_00", "generation": 2, "lr": 2.0e-5},
            )

            self.assertEqual((checkpoint_dir / TUNE_STATE_NAME).read_bytes(), b"state")
            self.assertEqual((checkpoint_dir / TUNE_OPTIMIZER_NAME).read_bytes(), b"optimizer")
            self.assertEqual((checkpoint_dir / TUNE_CONTROLLER_NAME).read_bytes(), b"controller")
            self.assertTrue((checkpoint_dir / TUNE_METADATA_NAME).is_file())
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["member"], "member_00")
            self.assertTrue(metadata["has_controller"])


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

    def test_initial_epoch_requires_optimizer_resume_files(self):
        source = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "bad_initial_epoch.yaml"
            payload = source.read_text().replace(
                "  checkpoint: checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt\n",
                "  checkpoint: checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt\n  initial_epoch: 17\n",
            )
            config_path.write_text(payload)

            with self.assertRaisesRegex(ValueError, "initial_state and initial_optimizer"):
                config_module.load_config(
                    namespace(
                        config=config_path,
                        experiment_name="unit_bad_initial_epoch",
                        gpus="0,1",
                        slots=None,
                        smoke=True,
                    )
                )

    def test_initial_optimizer_mode_is_validated(self):
        source = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "bad_initial_optimizer_mode.yaml"
            payload = source.read_text().replace(
                "  checkpoint: checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt\n",
                "  checkpoint: checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt\n  initial_optimizer_mode: chaotic\n",
            )
            config_path.write_text(payload)

            with self.assertRaisesRegex(ValueError, "initial_optimizer_mode"):
                config_module.load_config(
                    namespace(
                        config=config_path,
                        experiment_name="unit_bad_initial_optimizer_mode",
                        gpus="0,1",
                        slots=None,
                        smoke=True,
                    )
                )

    def test_pydantic_parses_string_false_without_enabling_guard(self):
        source = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "string_false_guard.yaml"
            payload = source.read_text().replace(
                "  seed: 2026\n",
                "  baseline_guard_reject_global_best: \"false\"\n  seed: 2026\n",
            )
            config_path.write_text(payload)

            config = config_module.load_config(
                namespace(
                    config=config_path,
                    experiment_name="unit_string_false_guard",
                    gpus="0,1",
                    slots=None,
                    smoke=True,
                )
            )

            self.assertFalse(config["pbt"]["baseline_guard_reject_global_best"])

    def test_pydantic_rejects_unknown_pbt_field(self):
        source = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "unknown_pbt_field.yaml"
            payload = source.read_text().replace(
                "  seed: 2026\n",
                "  typo_metric: validation_auc\n  seed: 2026\n",
            )
            config_path.write_text(payload)

            with self.assertRaisesRegex(ValueError, "typo_metric"):
                config_module.load_config(
                    namespace(
                        config=config_path,
                        experiment_name="unit_unknown_pbt_field",
                        gpus="0,1",
                        slots=None,
                        smoke=True,
                    )
                )

    def test_baseline_rollback_requires_initial_resume_checkpoint(self):
        source = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "bad_baseline_guard.yaml"
            payload = source.read_text().replace(
                "  seed: 2026\n",
                "  baseline_metric_value: 1.0\n  baseline_guard_action: rollback_to_initial\n  seed: 2026\n",
            )
            config_path.write_text(payload)

            with self.assertRaisesRegex(ValueError, "rollback_to_initial requires initial_state"):
                config_module.load_config(
                    namespace(
                        config=config_path,
                        experiment_name="unit_bad_baseline_guard",
                        gpus="0,1",
                        slots=None,
                        smoke=True,
                    )
                )

    def test_initial_optimizer_resume_bootstraps_generation_zero(self):
        config = pbt_smoke_config()
        config["shared"].update(
            initial_epoch=17,
            initial_state="/tmp/source_state.pt",
            initial_optimizer="/tmp/source_optimizer.pt",
        )

        command, _, target_epoch = self.backend.command_for(
            config,
            {"name": "member_00", "lr": 2.1544346900318847e-5},
            "0",
            PROJECT_DIR / "runs/pbt/unit_initial_resume/member_00",
            generation=0,
        )

        self.assertEqual(target_epoch, 18)
        self.assertEqual(command[command.index("--load-epoch") + 1], "17")
        self.assertIn("--override-load-lr", command)
        self.assertNotIn("--load-model-weights", command)

    def test_initial_optimizer_files_are_copied_to_member_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = PROJECT_DIR / temporary if not temporary.startswith("/") else Path(temporary)
            source_state = root / "source_state.pt"
            source_optimizer = root / "source_optimizer.pt"
            member_dir = root / "member_00"
            member_dir.mkdir()
            source_state.write_bytes(b"state")
            source_optimizer.write_bytes(b"optimizer")
            config = pbt_smoke_config()
            config["shared"].update(
                initial_epoch=17,
                initial_state=str(source_state),
                initial_optimizer=str(source_optimizer),
            )

            bootstrapped_epoch = strategy.bootstrap_initial_checkpoint(config, member_dir)
            state_path, optimizer_path = strategy.checkpoint_paths(member_dir, 17)

            self.assertEqual(bootstrapped_epoch, 17)
            self.assertEqual(state_path.read_bytes(), b"state")
            self.assertEqual(optimizer_path.read_bytes(), b"optimizer")

    def test_initial_optimizer_can_be_damped_to_reduce_old_momentum(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = PROJECT_DIR / temporary if not temporary.startswith("/") else Path(temporary)
            source_state = root / "source_state.pt"
            source_optimizer = root / "source_optimizer.pt"
            member_dir = root / "member_00"
            member_dir.mkdir()
            source_state.write_bytes(b"state")
            torch.save(
                {
                    "state": {
                        0: {
                            "step": 17,
                            "exp_avg": torch.ones(2),
                            "exp_avg_sq": torch.full((2,), 4.0),
                        }
                    },
                    "param_groups": [{"lr": 2.0e-5, "params": [0]}],
                },
                source_optimizer,
            )
            config = pbt_smoke_config()
            config["shared"].update(
                initial_epoch=17,
                initial_state=str(source_state),
                initial_optimizer=str(source_optimizer),
                initial_optimizer_mode="damped",
                initial_optimizer_damping=0.25,
            )

            strategy.bootstrap_initial_checkpoint(config, member_dir)
            _, optimizer_path = strategy.checkpoint_paths(member_dir, 17)
            loaded = torch.load(optimizer_path, map_location="cpu")
            metadata = member_dir / "net_epoch-17_optimizer_resume.json"

            self.assertTrue(torch.equal(loaded["state"][0]["exp_avg"], torch.full((2,), 0.25)))
            self.assertTrue(torch.equal(loaded["state"][0]["exp_avg_sq"], torch.full((2,), 4.0)))
            self.assertEqual(loaded["state"][0]["step"], 17)
            self.assertIn('"mode": "damped"', metadata.read_text())

    def test_initial_optimizer_can_be_reset_for_clean_resume_shell(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = PROJECT_DIR / temporary if not temporary.startswith("/") else Path(temporary)
            source_state = root / "source_state.pt"
            source_optimizer = root / "source_optimizer.pt"
            member_dir = root / "member_00"
            member_dir.mkdir()
            source_state.write_bytes(b"state")
            torch.save(
                {
                    "state": {
                        0: {
                            "step": torch.tensor(17),
                            "exp_avg": torch.ones(2),
                            "exp_avg_sq": torch.full((2,), 4.0),
                        }
                    },
                    "param_groups": [{"lr": 2.0e-5, "params": [0]}],
                },
                source_optimizer,
            )
            config = pbt_smoke_config()
            config["shared"].update(
                initial_epoch=17,
                initial_state=str(source_state),
                initial_optimizer=str(source_optimizer),
                initial_optimizer_mode="reset",
            )

            strategy.bootstrap_initial_checkpoint(config, member_dir)
            _, optimizer_path = strategy.checkpoint_paths(member_dir, 17)
            loaded = torch.load(optimizer_path, map_location="cpu")

            self.assertTrue(torch.equal(loaded["state"][0]["exp_avg"], torch.zeros(2)))
            self.assertTrue(torch.equal(loaded["state"][0]["exp_avg_sq"], torch.zeros(2)))
            self.assertTrue(torch.equal(loaded["state"][0]["step"], torch.tensor(0)))

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

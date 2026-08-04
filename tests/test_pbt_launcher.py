import importlib.util
import tempfile
import unittest
from pathlib import Path

from tests.helpers import PROJECT_DIR, namespace, pbt_smoke_config
from training.pbt import config as config_module
from training.pbt import strategy
from training.pbt.execution.backend import LocalWeaverBackend, backend_from_config
from training.pbt.execution.ray_backend import RayWeaverBackend
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
        from training.pbt.execution.ray_backend import _ray_import

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
                config=PROJECT_DIR / "configs/experiments/archive/legacy_finetune/finetune_fixed_lr_grid_adamw.yaml",
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

    def test_finetune_config_enables_smooth_lr_controller(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pretrained_guarded_4gpu_smooth_lr.yaml",
                experiment_name="unit_smooth_lr",
                gpus="0,1,2,3",
                slots=None,
                smoke=False,
            )
        )

        controller = config["pbt"]["lr_controller"]
        self.assertEqual(config["pbt"]["strategy"], "anchored_lr_sweep")
        self.assertEqual(controller["mode"], "smooth")
        self.assertAlmostEqual(controller["smoothing"], 0.20)
        self.assertAlmostEqual(controller["max_member_decrease"], 0.87)


    def test_pretrained_guarded_config_uses_presets(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pretrained_guarded_8gpu_smooth_lr.yaml",
                experiment_name="unit_pretrained_guarded",
                gpus=None,
                slots="iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3,iutgpu01:4,iutgpu01:5,iutgpu01:6,iutgpu01:7",
                smoke=False,
            )
        )

        self.assertEqual(config["shared"]["initial_epoch"], 17)
        self.assertTrue(config["shared"]["initial_optimizer"].endswith("net_epoch-17_optimizer.pt"))
        self.assertEqual(config["shared"]["initial_optimizer_mode"], "damped")
        self.assertEqual(config["shared"]["data_extension"], "parquet")
        self.assertEqual(config["pbt"]["baseline_metric_value"], 1.0426476744821431)
        self.assertTrue(config["pbt"]["baseline_guard_seed_initial_best"])
        self.assertEqual(len(config["population"]), 8)
        self.assertEqual(config["slots"][0]["label"], "iutgpu01:0")

    def test_proxy_validation_config_sets_active_validation_dataset(self):
        source = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "proxy_control.yaml"
            payload = source.read_text().replace(
                "  no_remake_weights: true\n",
                "  no_remake_weights: true\n"
                "  proxy_validation:\n"
                "    manifest: datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json\n"
                "    active_subset: control\n"
                "    control_dataset: datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet\n"
                "    monitor_dataset: datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet\n"
                "    full_dataset: datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet\n"
                "    train_suffix: train800k\n"
                "    control_suffix: val5k_tail\n"
                "    monitor_suffix: val50k_tail\n"
                "    full_suffix: val1000k\n"
                "    control_rows_per_class: 5000\n"
                "    monitor_rows_per_class: 50000\n"
                "    full_rows_per_class: 1000000\n"
                "    strategy: disjoint_tail_windows_from_full_validation\n",
            )
            config_path.write_text(payload)

            config = config_module.load_config(
                namespace(
                    config=config_path,
                    experiment_name="unit_proxy_control",
                    gpus="0,1",
                    slots=None,
                    smoke=True,
                )
            )

        proxy = config["shared"]["proxy_validation"]
        self.assertEqual(proxy["active_subset"], "control")
        self.assertTrue(config["shared"]["validation_dataset"].endswith("20250711_ilc_nnqq_sgv_10m_3cat_parquet"))
        self.assertEqual(config["shared"]["train_suffix"], "train800k")
        self.assertEqual(config["shared"]["validation_suffix"], "val5k_tail")
        self.assertTrue(proxy["manifest"].endswith("datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json"))


    def test_pretrained_10m_proxy_control_config_uses_val5k_tail_control(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/pretrained_guarded_8gpu_smooth_lr_10m_proxy_control.yaml",
                experiment_name="unit_10m_proxy_control",
                gpus="0,1",
                slots=None,
                smoke=False,
            )
        )

        shared = config["shared"]
        self.assertTrue(shared["dataset"].endswith("20250711_ilc_nnqq_sgv_10m_3cat_parquet"))
        self.assertTrue(shared["validation_dataset"].endswith("20250711_ilc_nnqq_sgv_10m_3cat_parquet"))
        self.assertEqual(shared["train_suffix"], "train800k")
        self.assertEqual(shared["validation_suffix"], "val5k_tail")
        self.assertEqual(shared["samples_per_epoch"], 120000)
        self.assertEqual(shared["generations"], 96)
        self.assertEqual(shared["proxy_validation"]["monitor_suffix"], "val50k_tail")
        self.assertEqual(shared["proxy_validation"]["full_suffix"], "val1000k")

        pbt = config["pbt"]
        self.assertEqual(pbt["baseline_metric_value"], 1.1400000057487494)
        self.assertEqual(pbt["min_lr"], 3.0e-6)
        self.assertEqual(pbt["max_lr"], 1.2e-5)
        self.assertEqual(pbt["base_start_lr"], 8.0e-6)
        self.assertTrue(pbt["dynamic_controller"]["evaluate_initial_checkpoint"])
        self.assertAlmostEqual(config["population"][0]["start_lr"], 1.0e-5)
        self.assertAlmostEqual(config["population"][-1]["start_lr"], 6.0e-6)

    def test_initial_evaluation_command_uses_initial_state_and_test_mode(self):
        config = pbt_smoke_config()
        config["shared"].update(
            initial_epoch=17,
            initial_state="checkpoints/pretrained/epoch17_state.pt",
            initial_optimizer="checkpoints/pretrained/epoch17_optimizer.pt",
        )
        config["pbt"]["dynamic_controller"] = {
            "mode": "active",
            "evaluate_initial_checkpoint": True,
        }

        command, log_path = self.backend.initial_evaluation_command_for(
            config,
            "0",
            PROJECT_DIR / "runs/pbt/unit_initial_eval",
        )

        self.assertIn("--run-mode", command)
        self.assertEqual(command[command.index("--run-mode") + 1], "test")
        self.assertIn("--data-test", command)
        self.assertEqual(
            command[command.index("--model-prefix") + 1],
            "checkpoints/pretrained/epoch17_state.pt",
        )
        self.assertEqual(log_path.name, "initial-evaluation.log")

    def test_initial_evaluation_data_test_merges_flavors_into_one_group(self):
        # Regression test: Weaver's `--run-mode test` loader (test_load in
        # weaver-core) builds one independent DataLoader per keyword-prefixed
        # `--data-test` group, evaluated as its own separate test pass --
        # unlike `--data-val` during train/val mode, which merges every
        # group into a single combined evaluation. If the per-flavor labels
        # data_paths() uses for train/val (nnbb:/nncc:/nndd:) leak into
        # --data-test unchanged, Weaver runs three single-class passes back
        # to back, each structurally unable to compute a background-
        # rejection curve (needs signal and background classes present
        # together), so validation_working_point_mistag_percent comes out
        # None and initial evaluation fails even though Weaver exits 0. See
        # the 2026-08-04 smoke failure: "finished initial_evaluation
        # returncode=0 metric=n/a".
        config = pbt_smoke_config()
        config["shared"].update(
            initial_epoch=17,
            initial_state="checkpoints/pretrained/epoch17_state.pt",
            initial_optimizer="checkpoints/pretrained/epoch17_optimizer.pt",
        )
        config["pbt"]["dynamic_controller"] = {
            "mode": "active",
            "evaluate_initial_checkpoint": True,
        }

        command, _ = self.backend.initial_evaluation_command_for(
            config,
            "0",
            PROJECT_DIR / "runs/pbt/unit_initial_eval",
        )

        start = command.index("--data-test") + 1
        end = start
        while end < len(command) and not command[end].startswith("--"):
            end += 1
        data_test_args = command[start:end]

        self.assertEqual(len(data_test_args), 3)
        for arg in data_test_args:
            self.assertNotRegex(
                arg,
                r"^nn(bb|cc|dd):",
                f"--data-test arg {arg!r} still carries a per-flavor group label; "
                "Weaver will evaluate it as an independent single-class test pass",
            )
        joined = " ".join(data_test_args)
        self.assertIn("_bb_", joined)
        self.assertIn("_cc_", joined)
        self.assertIn("_dd_", joined)

    def test_lr_controller_is_rejected_outside_anchored_sweep(self):
        source = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "bad_lr_controller.yaml"
            payload = source.read_text().replace(
                "  seed: 2026\n",
                "  seed: 2026\n  lr_controller:\n    mode: smooth\n",
            )
            config_path.write_text(payload)

            with self.assertRaisesRegex(ValueError, "lr_controller is only supported"):
                config_module.load_config(
                    namespace(
                        config=config_path,
                        experiment_name="unit_bad_lr_controller",
                        gpus="0,1",
                        slots=None,
                        smoke=True,
                    )
                )

    def test_head_warmup_freezes_backbone_only_for_configured_generations(self):
        config = config_module.load_config(
            namespace(
                config=PROJECT_DIR / "configs/experiments/archive/legacy_finetune/finetune_head_warmup_fixed_lr_grid.yaml",
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

    def test_baseline_seed_initial_best_creates_safe_recommended_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = PROJECT_DIR / temporary if not temporary.startswith("/") else Path(temporary)
            source_state = root / "source_state.pt"
            source_optimizer = root / "source_optimizer.pt"
            experiment_dir = root / "run"
            experiment_dir.mkdir()
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
            config["pbt"].update(
                baseline_metric_value=1.0367,
                baseline_guard_seed_initial_best=True,
            )
            manifest = {"best": None}

            seeded = strategy.seed_initial_global_best(config, experiment_dir, manifest)

            self.assertTrue(seeded)
            self.assertEqual((experiment_dir / "checkpoints" / "global_best_state.pt").read_bytes(), b"state")
            loaded = torch.load(experiment_dir / "checkpoints" / "global_best_optimizer.pt", map_location="cpu")
            self.assertTrue(torch.equal(loaded["state"][0]["exp_avg"], torch.full((2,), 0.25)))
            self.assertEqual(manifest["best"]["member"], "initial_resume")
            self.assertEqual(manifest["best"]["generation"], -1)
            self.assertAlmostEqual(manifest["best"]["metric_value"], 1.0367)
            self.assertTrue((experiment_dir / "checkpoints" / "global_best_metadata.json").is_file())

    def test_freeze_batch_norm_defaults_to_true(self):
        # Every experiment in this repo resumes from a checkpoint (pretrained
        # or a prior PBT run's global_best), never a random init, so BN
        # running stats are always worth preserving by default (see
        # bn_freeze_diag_baseline/frozen.yaml for the confirmed regression
        # this prevents). Configs must opt OUT explicitly, not opt in.
        config = pbt_smoke_config()
        self.assertTrue(config["shared"]["freeze_batch_norm"])

        command, _, _ = self.backend.command_for(
            config,
            {"name": "member_00", "lr": 9.0e-5},
            "0",
            PROJECT_DIR / "runs/pbt/unit_test/member_00",
            generation=0,
        )

        self.assertIn("--freeze-batch-norm", command)

    def test_freeze_batch_norm_can_be_explicitly_disabled(self):
        config = pbt_smoke_config()
        config["shared"]["freeze_batch_norm"] = False

        command, _, _ = self.backend.command_for(
            config,
            {"name": "member_00", "lr": 9.0e-5},
            "0",
            PROJECT_DIR / "runs/pbt/unit_test/member_00",
            generation=0,
        )

        self.assertNotIn("--freeze-batch-norm", command)

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

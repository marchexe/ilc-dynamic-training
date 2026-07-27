import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import run_pbt  # noqa: E402
import run_parallel_training  # noqa: E402
import plot_bgrej_curves  # noqa: E402


class PBTTest(unittest.TestCase):
    def smoke_config(self):
        return run_pbt.load_config(
            SimpleNamespace(
                config=PROJECT_DIR / "configs/experiments/pp_pbt.yaml",
                experiment_name="unit_test",
                gpus="0,2",
                slots=None,
                smoke=True,
            )
        )

    def test_smoke_config_and_resume_command(self):
        config = self.smoke_config()
        self.assertEqual(config["shared"]["generations"], 2)
        self.assertEqual(len(config["population"]), 2)
        self.assertEqual(config["shared"]["data_extension"], "root")

        member = {"name": "member_00", "lr": 9.0e-5}
        command, log_path, target_epoch = run_pbt.make_command(
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
        config = self.smoke_config()
        config["shared"]["dataset"] = "/tmp/sgv_parquet"
        config["shared"]["data_extension"] = "parquet"

        command, _, _ = run_pbt.make_command(
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

    def test_ranking_and_exploit_plan_are_deterministic(self):
        config = self.smoke_config()
        generation = {
            "index": 0,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 2.0}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 1.0}},
            },
        }
        members = {
            "member_00": {"lr": 7.5e-5},
            "member_01": {"lr": 1.0e-4},
        }

        ranking, plan = run_pbt.ranking_and_plan(config, generation, members)

        self.assertEqual(ranking, ["member_00", "member_01"])
        self.assertEqual(plan[0]["donor"], "member_00")
        self.assertEqual(plan[0]["recipient"], "member_01")
        self.assertGreaterEqual(plan[0]["new_lr"], config["pbt"]["min_lr"])
        self.assertLessEqual(plan[0]["new_lr"], config["pbt"]["max_lr"])

    def test_exploit_copies_both_states_and_updates_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("strong", "weak"):
                (root / name).mkdir()
            strong_state, strong_optimizer = run_pbt.checkpoint_paths(root / "strong", 0)
            weak_state, weak_optimizer = run_pbt.checkpoint_paths(root / "weak", 0)
            strong_state.write_bytes(b"strong-state")
            strong_optimizer.write_bytes(b"strong-optimizer")
            weak_state.write_bytes(b"weak-state")
            weak_optimizer.write_bytes(b"weak-optimizer")
            strong_controller = run_pbt.controller_checkpoint_path(root / "strong", 0)
            weak_controller = run_pbt.controller_checkpoint_path(root / "weak", 0)
            strong_controller.write_bytes(b"strong-controller")
            weak_controller.write_bytes(b"weak-controller")
            manifest_path = root / "manifest.json"
            manifest = {
                "config": {"shared": {"training_controller": "controller.yaml"}},
                "members": {
                    "strong": {"lr": 1.0e-4, "parent": None},
                    "weak": {"lr": 1.5e-4, "parent": None},
                }
            }
            generation = {
                "index": 0,
                "epoch": 0,
                "exploit": [
                    {
                        "donor": "strong",
                        "recipient": "weak",
                        "new_lr": 8.0e-5,
                        "applied": False,
                    }
                ],
            }

            run_pbt.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(weak_state.read_bytes(), b"strong-state")
            self.assertEqual(weak_optimizer.read_bytes(), b"strong-optimizer")
            self.assertEqual(weak_controller.read_bytes(), b"strong-controller")
            self.assertTrue(generation["exploit"][0]["applied"])
            self.assertEqual(manifest["members"]["weak"]["parent"], "strong")
            self.assertEqual(manifest["members"]["weak"]["lr"], 8.0e-5)

    def test_exploit_can_reset_controller_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("strong", "weak"):
                (root / name).mkdir()
            strong_state, strong_optimizer = run_pbt.checkpoint_paths(root / "strong", 0)
            weak_state, weak_optimizer = run_pbt.checkpoint_paths(root / "weak", 0)
            strong_state.write_bytes(b"strong-state")
            strong_optimizer.write_bytes(b"strong-optimizer")
            weak_state.write_bytes(b"weak-state")
            weak_optimizer.write_bytes(b"weak-optimizer")
            strong_controller = run_pbt.controller_checkpoint_path(root / "strong", 0)
            weak_controller = run_pbt.controller_checkpoint_path(root / "weak", 0)
            strong_controller.write_bytes(b"strong-controller")
            weak_controller.write_bytes(b"weak-controller")
            manifest_path = root / "manifest.json"
            manifest = {
                "config": {
                    "shared": {"training_controller": "controller.yaml"},
                    "pbt": {"controller_state_on_exploit": "reset"},
                },
                "members": {
                    "strong": {"lr": 1.0e-4, "parent": None},
                    "weak": {"lr": 1.5e-4, "parent": None},
                },
            }
            generation = {
                "index": 0,
                "epoch": 0,
                "exploit": [
                    {
                        "donor": "strong",
                        "recipient": "weak",
                        "new_lr": 8.0e-5,
                        "applied": False,
                    }
                ],
            }

            run_pbt.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(weak_state.read_bytes(), b"strong-state")
            self.assertEqual(weak_optimizer.read_bytes(), b"strong-optimizer")
            self.assertFalse(weak_controller.exists())

    def test_global_best_is_copied_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member_dir = root / "member_00"
            member_dir.mkdir()
            state, optimizer = run_pbt.checkpoint_paths(member_dir, 3)
            state.write_bytes(b"best-state")
            optimizer.write_bytes(b"best-optimizer")
            manifest_path = root / "manifest.json"
            config = self.smoke_config()
            manifest = {
                "config": config,
                "members": {"member_00": {"lr": 9.0e-5}},
                "generations": [],
                "best": None,
            }
            generation = {
                "index": 1,
                "epoch": 3,
                "ranking": ["member_00"],
                "workers": {
                    "member_00": {
                        "metrics": {
                            "validation_bkg_rejection_score": 7.5,
                            "validation_accuracy": 0.88,
                        }
                    }
                },
            }

            improved = run_pbt.update_global_best(root, manifest, generation, manifest_path)

            self.assertTrue(improved)
            self.assertEqual(manifest["best"]["member"], "member_00")
            self.assertEqual(manifest["best"]["generation"], 1)
            self.assertEqual((root / "global_best_state.pt").read_bytes(), b"best-state")
            self.assertEqual((root / "global_best_optimizer.pt").read_bytes(), b"best-optimizer")
            self.assertTrue((root / "global_best_metadata.json").is_file())

    def test_degraded_generation_rolls_back_worst_member_from_global_best(self):
        config = self.smoke_config()
        config["pbt"]["rollback_fraction"] = 0.5
        config["pbt"]["degradation_tolerance"] = 0.02
        config["pbt"]["degradation_window"] = 1
        manifest = {
            "best": {
                "member": "member_00",
                "generation": 0,
                "metric_value": 10.0,
                "lr": 9.0e-5,
            },
            "generations": [],
        }
        generation = {
            "index": 0,
            "ranking": ["member_00", "member_01"],
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 9.0}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 8.0}},
            },
        }
        members = {
            "member_00": {"lr": 9.0e-5},
            "member_01": {"lr": 1.2e-4},
        }

        health = run_pbt.update_generation_health(config, manifest, generation)
        plan = run_pbt.add_global_best_rollbacks(config, manifest, generation, members, [])

        self.assertEqual(health["status"], "degraded")
        self.assertEqual(plan[0]["source"], "global_best")
        self.assertEqual(plan[0]["recipient"], "member_01")
        self.assertEqual(plan[0]["new_lr"], 9.0e-5)

    def test_remote_slots_wrap_worker_command_in_ssh(self):
        config = run_pbt.load_config(
            SimpleNamespace(
                config=PROJECT_DIR / "configs/experiments/pp_pbt.yaml",
                experiment_name="unit_test_remote",
                gpus=None,
                slots="iutgpu01:6@.venv-iutgpu01,iutgpu05:4",
                smoke=True,
            )
        )

        self.assertEqual(
            [slot["label"] for slot in config["slots"]],
            ["iutgpu01:6@.venv-iutgpu01", "iutgpu05:4"],
        )

        member = {"name": "member_00", "lr": 9.0e-5}
        command, _, _ = run_pbt.make_command(
            config,
            member,
            config["slots"][0],
            PROJECT_DIR / "runs/pbt/unit_test_remote/member_00",
            generation=0,
        )

        self.assertEqual(command[:2], ["ssh", "iutgpu01"])
        self.assertIn(".venv-iutgpu01/bin/python", command[2])
        self.assertIn("remote venv python is not executable", command[2])
        self.assertIn("--gpus 6", command[2])

    def test_parallel_worker_can_override_seed_and_lr(self):
        resolved = run_parallel_training.load_and_resolve(
            SimpleNamespace(
                config=PROJECT_DIR / "configs/experiments/pp_fixed_lr_12h.yaml",
                experiment_name="unit_fixed_sweep",
                gpus=None,
                smoke=True,
            )
        )
        worker = next(
            item for item in resolved["workers"]
            if item["name"] == "fixed_lr_125e-4"
        )
        command = run_parallel_training.build_command(
            resolved,
            worker,
            PROJECT_DIR / "runs/parallel/unit_fixed_sweep/fixed_lr_125e-4",
            resume_epoch=None,
        )

        self.assertEqual(command[command.index("--start-lr") + 1], "0.000125")
        self.assertEqual(command[command.index("--seed") + 1], "12347")

    def test_parallel_command_can_use_parquet_data_and_prefetch(self):
        resolved = run_parallel_training.load_and_resolve(
            SimpleNamespace(
                config=PROJECT_DIR / "configs/experiments/pp_fixed_lr_12h.yaml",
                experiment_name="unit_fixed_sweep",
                gpus=None,
                smoke=True,
            )
        )
        resolved["shared"]["dataset"] = "/tmp/sgv_parquet"
        resolved["shared"]["data_extension"] = "parquet"
        resolved["shared"]["prefetch_factor"] = 4
        worker = resolved["workers"][0]

        command = run_parallel_training.build_command(
            resolved,
            worker,
            PROJECT_DIR / "runs/parallel/unit_fixed_sweep/fixed_lr_100e-4",
            resume_epoch=None,
        )

        joined = " ".join(command)
        self.assertIn("nnbb:/tmp/sgv_parquet/*_bb_train800k.parquet", joined)
        self.assertIn("--prefetch-factor", command)
        self.assertEqual(command[command.index("--prefetch-factor") + 1], "4")

    def test_bgrej_curves_parser_reads_last_eval_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "generation-001.log"
            log_path.write_text(
                """
INFO: Evaluation metrics:
    - bkg_rejection_at_eff:
{'bc': [1, 2, 3, 4, 5, 6, 7], 'bd': [2, 3, 4, 5, 6, 7, 8], 'cb': [3, 4, 5, 6, 7, 8, 9], 'cd': [4, 5, 6, 7, 8, 9, 10]}
    - bkg_rejection_score:
1.0
INFO: Evaluation metrics:
    - bkg_rejection_at_eff:
{'bc': [11, 12, 13, 14, 15, 16, 17], 'bd': [12, 13, 14, 15, 16, 17, 18], 'cb': [13, 14, 15, 16, 17, 18, 19], 'cd': [14, 15, 16, 17, 18, 19, 20]}
    - bkg_rejection_score:
2.0
""",
                encoding="utf-8",
            )

            curves = plot_bgrej_curves.parse_bgrej_curves(log_path)

            self.assertEqual(curves["bc"][0], 11)
            self.assertEqual(curves["cd"][-1], 20)


if __name__ == "__main__":
    unittest.main()

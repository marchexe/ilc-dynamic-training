import tempfile
import unittest
from pathlib import Path

from tests.helpers import pbt_smoke_config
from training.pbt import strategy


class PBTAlgorithmTest(unittest.TestCase):
    def test_ranking_and_exploit_plan_are_deterministic(self):
        config = pbt_smoke_config()
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

        ranking, plan = strategy.ranking_and_plan(config, generation, members)

        self.assertEqual(ranking, ["member_00", "member_01"])
        self.assertEqual(plan[0]["donor"], "member_00")
        self.assertEqual(plan[0]["recipient"], "member_01")
        self.assertGreaterEqual(plan[0]["new_lr"], config["pbt"]["min_lr"])
        self.assertLessEqual(plan[0]["new_lr"], config["pbt"]["max_lr"])


    def test_anchored_lr_sweep_uses_fixed_worker_positions(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            strategy="anchored_lr_sweep",
            lr_factors=[1.05, 1.025, 0.975, 0.95],
        )
        generation = {
            "index": 0,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 7.0}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 7.5}},
                "member_02": {"metrics": {"validation_bkg_rejection_score": 8.0}},
                "member_03": {"metrics": {"validation_bkg_rejection_score": 7.2}},
            },
        }
        members = {
            "member_00": {"lr": 1.10e-4},
            "member_01": {"lr": 1.05e-4},
            "member_02": {"lr": 0.95e-4},
            "member_03": {"lr": 0.90e-4},
        }

        ranking, plan = strategy.anchored_lr_sweep_plan(config, generation, members)

        self.assertEqual(ranking[0], "member_02")
        self.assertEqual([event["recipient"] for event in plan], list(members))
        self.assertTrue(all(event["donor"] == "member_02" for event in plan))
        self.assertAlmostEqual(plan[0]["new_lr"], 0.95e-4 * 1.05)
        self.assertAlmostEqual(plan[1]["new_lr"], 0.95e-4 * 1.025)
        self.assertAlmostEqual(plan[2]["new_lr"], 0.95e-4 * 0.975)
        self.assertAlmostEqual(plan[3]["new_lr"], 0.95e-4 * 0.95)

    def test_anchored_lr_sweep_smoke_uses_outer_lr_factors(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            strategy="anchored_lr_sweep",
            lr_factors=[1.05, 1.025, 0.975, 0.95],
        )

        factors = strategy.lr_factors_for_population(config, ["member_00", "member_01"])

        self.assertEqual(factors, [1.05, 0.95])

    def test_anchored_lr_radius_generates_symmetric_factors(self):
        self.assertEqual(
            strategy.factors_from_radius(0.05, 4),
            [1.05, 1.025, 0.975, 0.95],
        )
        self.assertEqual(strategy.factors_from_radius(0.05, 2), [1.05, 0.95])

    def test_anchored_lr_radius_shrinks_after_inner_wins(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            strategy="anchored_lr_sweep",
            lr_radius={
                "initial": 0.05,
                "minimum": 0.015,
                "shrink_factor": 0.7,
                "shrink_after_inner_wins": 3,
                "keep_if_edge_wins": True,
            },
        )
        members = {
            "member_00": {"lr": 1.10e-4},
            "member_01": {"lr": 1.05e-4},
            "member_02": {"lr": 0.95e-4},
            "member_03": {"lr": 0.90e-4},
        }
        manifest = {
            "best": {"metric_value": 10.0},
            "generations": [
                {"index": 0, "lr_radius": {"next_radius": 0.05, "inner_win_generations": 2}}
            ],
        }
        generation = {
            "index": 1,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 9.8}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 9.9}},
                "member_02": {"metrics": {"validation_bkg_rejection_score": 9.7}},
                "member_03": {"metrics": {"validation_bkg_rejection_score": 9.6}},
            },
        }

        ranking, plan = strategy.anchored_lr_sweep_plan(config, generation, members, manifest)

        self.assertEqual(ranking[0], "member_01")
        self.assertEqual(generation["lr_radius"]["action"], "shrink_inner_winners")
        self.assertAlmostEqual(generation["lr_radius"]["radius"], 0.05)
        self.assertAlmostEqual(generation["lr_radius"]["next_radius"], 0.035)
        self.assertAlmostEqual(plan[0]["lr_radius"], 0.05)

    def test_anchored_lr_radius_keeps_radius_when_edge_wins(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            strategy="anchored_lr_sweep",
            lr_radius={
                "initial": 0.05,
                "minimum": 0.015,
                "shrink_factor": 0.7,
                "shrink_after_inner_wins": 3,
                "keep_if_edge_wins": True,
            },
        )
        members = {
            "member_00": {"lr": 1.10e-4},
            "member_01": {"lr": 1.05e-4},
            "member_02": {"lr": 0.95e-4},
            "member_03": {"lr": 0.90e-4},
        }
        manifest = {
            "best": {"metric_value": 10.0},
            "generations": [
                {"index": 0, "lr_radius": {"next_radius": 0.05, "inner_win_generations": 2}}
            ],
        }
        generation = {
            "index": 1,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 9.9}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 9.8}},
                "member_02": {"metrics": {"validation_bkg_rejection_score": 9.7}},
                "member_03": {"metrics": {"validation_bkg_rejection_score": 9.6}},
            },
        }

        strategy.anchored_lr_sweep_plan(config, generation, members, manifest)

        self.assertEqual(generation["lr_radius"]["action"], "keep_edge_winner")
        self.assertAlmostEqual(generation["lr_radius"]["next_radius"], 0.05)
        self.assertEqual(generation["lr_radius"]["inner_win_generations"], 0)

    def test_exploit_copies_both_states_and_updates_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("strong", "weak"):
                (root / name).mkdir()
            strong_state, strong_optimizer = strategy.checkpoint_paths(root / "strong", 0)
            weak_state, weak_optimizer = strategy.checkpoint_paths(root / "weak", 0)
            strong_state.write_bytes(b"strong-state")
            strong_optimizer.write_bytes(b"strong-optimizer")
            weak_state.write_bytes(b"weak-state")
            weak_optimizer.write_bytes(b"weak-optimizer")
            strong_controller = strategy.controller_checkpoint_path(root / "strong", 0)
            weak_controller = strategy.controller_checkpoint_path(root / "weak", 0)
            strong_controller.write_bytes(b"strong-controller")
            weak_controller.write_bytes(b"weak-controller")
            manifest_path = root / "manifest.json"
            manifest = {
                "config": {"shared": {"training_controller": "controller.yaml"}},
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

            strategy.apply_exploit(root, manifest, generation, manifest_path)

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
            strong_state, strong_optimizer = strategy.checkpoint_paths(root / "strong", 0)
            weak_state, weak_optimizer = strategy.checkpoint_paths(root / "weak", 0)
            strong_state.write_bytes(b"strong-state")
            strong_optimizer.write_bytes(b"strong-optimizer")
            weak_state.write_bytes(b"weak-state")
            weak_optimizer.write_bytes(b"weak-optimizer")
            strong_controller = strategy.controller_checkpoint_path(root / "strong", 0)
            weak_controller = strategy.controller_checkpoint_path(root / "weak", 0)
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

            strategy.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(weak_state.read_bytes(), b"strong-state")
            self.assertEqual(weak_optimizer.read_bytes(), b"strong-optimizer")
            self.assertFalse(weak_controller.exists())

    def test_global_best_is_copied_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member_dir = root / "member_00"
            member_dir.mkdir()
            state, optimizer = strategy.checkpoint_paths(member_dir, 3)
            state.write_bytes(b"best-state")
            optimizer.write_bytes(b"best-optimizer")
            manifest_path = root / "manifest.json"
            config = pbt_smoke_config()
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

            improved = strategy.update_global_best(root, manifest, generation, manifest_path)

            self.assertTrue(improved)
            self.assertEqual(manifest["best"]["member"], "member_00")
            self.assertEqual(manifest["best"]["generation"], 1)
            self.assertEqual((root / "global_best_state.pt").read_bytes(), b"best-state")
            self.assertEqual((root / "global_best_optimizer.pt").read_bytes(), b"best-optimizer")
            self.assertTrue((root / "global_best_metadata.json").is_file())

    def test_degraded_generation_rolls_back_worst_member_from_global_best(self):
        config = pbt_smoke_config()
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

        health = strategy.update_generation_health(config, manifest, generation)
        plan = strategy.add_global_best_rollbacks(config, manifest, generation, members, [])

        self.assertEqual(health["status"], "degraded")
        self.assertEqual(plan[0]["source"], "global_best")
        self.assertEqual(plan[0]["recipient"], "member_01")
        self.assertEqual(plan[0]["new_lr"], 9.0e-5)


if __name__ == "__main__":
    unittest.main()

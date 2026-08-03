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

    def test_exploit_significance_gating_skips_insignificant_replacement(self):
        config = pbt_smoke_config()
        config["pbt"]["mode"] = "min"
        config["pbt"]["metric"] = "validation_working_point_mistag_percent"
        config["pbt"]["exploit_significance_sigma"] = 1.0
        # Donor is nominally better (1.10 < 1.15) but the gap is tiny relative
        # to the combined uncertainty -- a real PBT should not overwrite the
        # recipient's weights on noise this small.
        generation = {
            "index": 0,
            "workers": {
                "member_00": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 1.10,
                        "validation_working_point_mistag_percent_uncertainty": 0.20,
                    }
                },
                "member_01": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 1.15,
                        "validation_working_point_mistag_percent_uncertainty": 0.20,
                    }
                },
            },
        }
        members = {"member_00": {"lr": 7.5e-5}, "member_01": {"lr": 1.0e-4}}

        ranking, plan = strategy.ranking_and_plan(config, generation, members)

        self.assertEqual(plan, [])
        self.assertEqual(len(generation["skipped_exploits"]), 1)
        skipped = generation["skipped_exploits"][0]
        self.assertEqual(skipped["donor"], "member_00")
        self.assertEqual(skipped["recipient"], "member_01")
        self.assertEqual(skipped["reason"], "not_significant")
        self.assertIsNotNone(skipped["margin_sigma"])
        self.assertLess(skipped["margin_sigma"], 1.0)

    def test_exploit_significance_gating_applies_clearly_significant_replacement(self):
        config = pbt_smoke_config()
        config["pbt"]["mode"] = "min"
        config["pbt"]["metric"] = "validation_working_point_mistag_percent"
        config["pbt"]["exploit_significance_sigma"] = 1.0
        generation = {
            "index": 0,
            "workers": {
                "member_00": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 0.50,
                        "validation_working_point_mistag_percent_uncertainty": 0.02,
                    }
                },
                "member_01": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 2.00,
                        "validation_working_point_mistag_percent_uncertainty": 0.02,
                    }
                },
            },
        }
        members = {"member_00": {"lr": 7.5e-5}, "member_01": {"lr": 1.0e-4}}

        ranking, plan = strategy.ranking_and_plan(config, generation, members)

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["donor"], "member_00")
        self.assertEqual(plan[0]["recipient"], "member_01")
        self.assertGreater(plan[0]["significance_margin_sigma"], 1.0)
        self.assertEqual(generation["skipped_exploits"], [])

    def test_exploit_significance_gating_treats_missing_uncertainty_as_inconclusive(self):
        config = pbt_smoke_config()
        config["pbt"]["mode"] = "min"
        config["pbt"]["metric"] = "validation_working_point_mistag_percent"
        config["pbt"]["exploit_significance_sigma"] = 1.0
        # Donor is nominally much better, but neither worker reports an
        # uncertainty -- a missing uncertainty must be treated as
        # inconclusive (skip), not silently fall back to a nominal compare.
        generation = {
            "index": 0,
            "workers": {
                "member_00": {"metrics": {"validation_working_point_mistag_percent": 0.10}},
                "member_01": {"metrics": {"validation_working_point_mistag_percent": 5.0}},
            },
        }
        members = {"member_00": {"lr": 7.5e-5}, "member_01": {"lr": 1.0e-4}}

        ranking, plan = strategy.ranking_and_plan(config, generation, members)

        self.assertEqual(plan, [])
        self.assertEqual(generation["skipped_exploits"][0]["reason"], "missing_uncertainty")

    def test_exploit_significance_gating_disabled_by_default_keeps_legacy_behavior(self):
        config = pbt_smoke_config()
        config["pbt"]["mode"] = "min"
        config["pbt"]["metric"] = "validation_working_point_mistag_percent"
        self.assertIsNone(config["pbt"].get("exploit_significance_sigma"))
        generation = {
            "index": 0,
            "workers": {
                "member_00": {"metrics": {"validation_working_point_mistag_percent": 1.10}},
                "member_01": {"metrics": {"validation_working_point_mistag_percent": 1.15}},
            },
        }
        members = {"member_00": {"lr": 7.5e-5}, "member_01": {"lr": 1.0e-4}}

        ranking, plan = strategy.ranking_and_plan(config, generation, members)

        self.assertEqual(len(plan), 1)
        self.assertIsNone(plan[0]["significance_margin_sigma"])

    def test_raw_metric_ranking_is_a_plain_sort_unaffected_by_incumbent_persistence(self):
        config = pbt_smoke_config()
        config["pbt"]["mode"] = "max"
        config["pbt"]["metric"] = "validation_bkg_rejection_score"
        generation = {
            "index": 1,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 9.9}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 10.0}},
            },
        }
        members = {"member_00": {"lr": 7.5e-5}, "member_01": {"lr": 1.0e-4}}

        ranking = strategy.raw_metric_ranking(config, generation, members)

        self.assertEqual(ranking, ["member_01", "member_00"])

    def test_burn_in_suppresses_exploit_but_final_and_early_stop_still_win(self):
        config = pbt_smoke_config()
        config["pbt"]["burn_in_generations"] = 2

        self.assertTrue(strategy.in_burn_in(config, 0))
        self.assertTrue(strategy.in_burn_in(config, 1))
        self.assertFalse(strategy.in_burn_in(config, 2))

        # Generation 0 and 1 are within burn-in: no exploit even though
        # nothing else would block it.
        self.assertFalse(strategy.should_apply_exploit(config, 0, is_final_generation=False, early_stop_triggered=False))
        self.assertFalse(strategy.should_apply_exploit(config, 1, is_final_generation=False, early_stop_triggered=False))
        # Generation 2 is past burn-in: exploit resumes.
        self.assertTrue(strategy.should_apply_exploit(config, 2, is_final_generation=False, early_stop_triggered=False))
        # Still respects the existing final-generation / early-stop guards.
        self.assertFalse(strategy.should_apply_exploit(config, 5, is_final_generation=True, early_stop_triggered=False))
        self.assertFalse(strategy.should_apply_exploit(config, 5, is_final_generation=False, early_stop_triggered=True))

    def test_burn_in_zero_by_default_never_suppresses_exploit(self):
        config = pbt_smoke_config()
        self.assertEqual(config["pbt"].get("burn_in_generations", 0), 0)
        self.assertFalse(strategy.in_burn_in(config, 0))
        self.assertTrue(strategy.should_apply_exploit(config, 0, is_final_generation=False, early_stop_triggered=False))

    def test_exploit_interval_generations_makes_exploit_genuinely_less_frequent(self):
        # exploit_interval_generations previously existed only as a
        # reporting field nobody consulted -- this locks in that it now
        # actually gates the exploit-application decision.
        config = pbt_smoke_config()
        config["pbt"]["exploit_interval_generations"] = 3

        due = [
            strategy.should_apply_exploit(config, generation, is_final_generation=False, early_stop_triggered=False)
            for generation in range(9)
        ]

        # Due on generations where (generation+1) % 3 == 0: indices 2,5,8.
        self.assertEqual(due, [False, False, True, False, False, True, False, False, True])

    def test_exploit_interval_generations_unset_means_every_generation(self):
        config = pbt_smoke_config()
        self.assertIsNone(config["pbt"].get("exploit_interval_generations"))

        for generation in range(4):
            self.assertTrue(
                strategy.should_apply_exploit(config, generation, is_final_generation=False, early_stop_triggered=False)
            )

    def test_exploit_interval_and_burn_in_compose(self):
        config = pbt_smoke_config()
        config["pbt"]["burn_in_generations"] = 2
        config["pbt"]["exploit_interval_generations"] = 3

        # Generation 2 is past burn-in and satisfies (2+1)%3==0 -> due.
        self.assertTrue(strategy.should_apply_exploit(config, 2, is_final_generation=False, early_stop_triggered=False))
        # Generation 1 is past... no, still in burn-in (burn_in=2 covers 0,1).
        self.assertFalse(strategy.should_apply_exploit(config, 1, is_final_generation=False, early_stop_triggered=False))
        # Generation 3 is past burn-in but not interval-due ((3+1)%3=1).
        self.assertFalse(strategy.should_apply_exploit(config, 3, is_final_generation=False, early_stop_triggered=False))


    def test_anchored_lr_sweep_uses_fixed_worker_positions(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            strategy="anchored_lr_sweep",
            anchored_weight_source="anchor",
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

    def test_smooth_lr_controller_limits_center_and_member_jumps(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            strategy="anchored_lr_sweep",
            lr_factors=[1.0, 1.0, 1.0, 1.0],
            min_lr=3.0e-6,
            max_lr=4.0e-5,
            lr_controller={
                "mode": "smooth",
                "smoothing": 1.0,
                "max_center_increase": 1.20,
                "max_center_decrease": 0.85,
                "max_member_increase": 1.25,
                "max_member_decrease": 0.80,
                "decay_bias": 1.0,
            },
        )
        members = {
            "member_00": {"lr": 4.0e-5},
            "member_01": {"lr": 3.0e-6},
            "member_02": {"lr": 1.0e-5},
            "member_03": {"lr": 1.2e-5},
        }
        manifest = {
            "generations": [
                {"index": 1, "lr_controller": {"generation": 1, "center_lr": 4.0e-5}}
            ]
        }
        generation = {
            "index": 2,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 9.0}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 10.0}},
                "member_02": {"metrics": {"validation_bkg_rejection_score": 8.0}},
                "member_03": {"metrics": {"validation_bkg_rejection_score": 7.0}},
            },
        }

        ranking, plan = strategy.anchored_lr_sweep_plan(config, generation, members, manifest)

        self.assertEqual(ranking[0], "member_01")
        self.assertAlmostEqual(generation["lr_controller"]["target_lr"], 3.0e-6)
        self.assertAlmostEqual(generation["lr_controller"]["center_lr"], 3.4e-5)
        self.assertAlmostEqual(plan[0]["new_lr"], 3.4e-5)
        self.assertAlmostEqual(plan[1]["new_lr"], 3.75e-6)
        self.assertTrue(plan[1]["member_lr_step_clamped"])

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
            self.assertEqual(generation["exploit"][0]["source"], "population")
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
            self.assertEqual((root / "checkpoints" / "global_best_state.pt").read_bytes(), b"best-state")
            self.assertEqual((root / "checkpoints" / "global_best_optimizer.pt").read_bytes(), b"best-optimizer")
            self.assertTrue((root / "checkpoints" / "global_best_metadata.json").is_file())

    def test_global_best_selection_respects_min_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, epoch in (("member_00", 1), ("member_01", 1), ("member_00", 2)):
                member_dir = root / name
                member_dir.mkdir(exist_ok=True)
                state, optimizer = strategy.checkpoint_paths(member_dir, epoch)
                state.write_bytes(f"{name}-epoch{epoch}-state".encode())
                optimizer.write_bytes(f"{name}-epoch{epoch}-optimizer".encode())
            manifest_path = root / "manifest.json"
            config = pbt_smoke_config()
            config["pbt"]["metric"] = "validation_working_point_mistag_percent"
            config["pbt"]["mode"] = "min"
            manifest = {
                "config": config,
                "members": {"member_00": {"lr": 9.0e-5}, "member_01": {"lr": 1.0e-4}},
                "generations": [],
                "best": None,
            }
            generation_one = {
                "index": 0,
                "epoch": 1,
                "ranking": ["member_01", "member_00"],
                "workers": {
                    "member_00": {"metrics": {"validation_working_point_mistag_percent": 1.2}},
                    "member_01": {"metrics": {"validation_working_point_mistag_percent": 0.8}},
                },
            }

            improved = strategy.update_global_best(root, manifest, generation_one, manifest_path)

            self.assertTrue(improved)
            self.assertEqual(manifest["best"]["member"], "member_01")
            self.assertEqual(manifest["best"]["metric_value"], 0.8)

            # A worse (higher) mistag rate in a later generation must not
            # overwrite the lower-is-better incumbent.
            generation_two = {
                "index": 1,
                "epoch": 2,
                "ranking": ["member_00"],
                "workers": {
                    "member_00": {"metrics": {"validation_working_point_mistag_percent": 1.0}},
                },
            }

            improved_again = strategy.update_global_best(root, manifest, generation_two, manifest_path)

            self.assertFalse(improved_again)
            self.assertEqual(manifest["best"]["member"], "member_01")
            self.assertEqual(manifest["best"]["metric_value"], 0.8)
            self.assertEqual((root / "checkpoints" / "global_best_state.pt").read_bytes(), b"member_01-epoch1-state")

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

    def test_confidence_aware_selection_keeps_previous_winner_inside_uncertainty(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            confidence_aware_selection=True,
            selection_uncertainty_sigma=1.0,
        )
        generation = {
            "index": 1,
            "workers": {
                "member_00": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 1.00,
                        "validation_working_point_mistag_percent_uncertainty": 0.05,
                    }
                },
                "member_01": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 0.98,
                        "validation_working_point_mistag_percent_uncertainty": 0.05,
                    }
                },
            },
        }
        members = {"member_00": {"lr": 2.0e-5}, "member_01": {"lr": 2.2e-5}}
        manifest = {"generations": [{"index": 0, "ranking": ["member_00", "member_01"]}]}

        ranking = strategy.confidence_aware_ranking(config, generation, members, manifest)

        self.assertEqual(ranking, ["member_00", "member_01"])
        self.assertEqual(generation["selection"]["mode"], "confidence_aware")

    def test_confidence_aware_selection_switches_when_delta_exceeds_uncertainty(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            confidence_aware_selection=True,
            selection_uncertainty_sigma=1.0,
        )
        generation = {
            "index": 1,
            "workers": {
                "member_00": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 1.00,
                        "validation_working_point_mistag_percent_uncertainty": 0.02,
                    }
                },
                "member_01": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 0.90,
                        "validation_working_point_mistag_percent_uncertainty": 0.02,
                    }
                },
            },
        }
        members = {"member_00": {"lr": 2.0e-5}, "member_01": {"lr": 2.2e-5}}
        manifest = {"generations": [{"index": 0, "ranking": ["member_00", "member_01"]}]}

        ranking = strategy.confidence_aware_ranking(config, generation, members, manifest)

        self.assertEqual(ranking, ["member_01", "member_00"])

    def test_confidence_aware_selection_uses_incumbent_not_pairwise_ties(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            confidence_aware_selection=True,
            selection_uncertainty_sigma=1.0,
        )
        generation = {
            "index": 1,
            "workers": {
                "member_00": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 1.00,
                        "validation_working_point_mistag_percent_uncertainty": 0.03,
                    }
                },
                "member_01": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 0.96,
                        "validation_working_point_mistag_percent_uncertainty": 0.03,
                    }
                },
                "member_02": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 0.92,
                        "validation_working_point_mistag_percent_uncertainty": 0.03,
                    }
                },
            },
        }
        members = {
            "member_00": {"lr": 2.0e-5},
            "member_01": {"lr": 2.1e-5},
            "member_02": {"lr": 2.2e-5},
        }
        manifest = {"generations": [{"index": 0, "ranking": ["member_00", "member_01", "member_02"]}]}

        ranking = strategy.confidence_aware_ranking(config, generation, members, manifest)

        self.assertEqual(ranking, ["member_02", "member_01", "member_00"])
        self.assertEqual(generation["selection"]["anchor_policy"], "incumbent_significance")
        self.assertEqual(generation["selection"]["score"], "conservative_confidence_bound")


    def test_confidence_aware_selection_preserves_incumbent_without_significant_win(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            confidence_aware_selection=True,
            selection_uncertainty_sigma=1.0,
        )
        generation = {
            "index": 1,
            "workers": {
                "member_00": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 1.00,
                        "validation_working_point_mistag_percent_uncertainty": 0.03,
                    }
                },
                "member_01": {
                    "metrics": {
                        "validation_working_point_mistag_percent": 0.96,
                        "validation_working_point_mistag_percent_uncertainty": 0.03,
                    }
                },
            },
        }
        members = {"member_00": {"lr": 2.0e-5}, "member_01": {"lr": 2.1e-5}}
        manifest = {"generations": [{"index": 0, "ranking": ["member_00", "member_01"]}]}

        ranking = strategy.confidence_aware_ranking(config, generation, members, manifest)

        self.assertEqual(ranking[0], "member_00")


    def test_anchored_lr_sweep_can_preserve_branch_weights(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            strategy="anchored_lr_sweep",
            anchored_weight_source="self",
            lr_factors=[1.05, 0.95],
        )
        generation = {
            "index": 0,
            "workers": {
                "member_00": {"metrics": {"validation_bkg_rejection_score": 7.0}},
                "member_01": {"metrics": {"validation_bkg_rejection_score": 8.0}},
            },
        }
        members = {
            "member_00": {"lr": 1.0e-4},
            "member_01": {"lr": 0.9e-4},
        }

        ranking, plan = strategy.anchored_lr_sweep_plan(config, generation, members)

        self.assertEqual(ranking[0], "member_01")
        self.assertEqual([event["donor"] for event in plan], ["member_00", "member_01"])
        self.assertTrue(all(event["weight_source"] == "self" for event in plan))

    def test_generation_health_records_baseline_regression(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            baseline_metric_value=1.04,
            baseline_guard_tolerance=0.01,
        )
        manifest = {"best": None, "generations": []}
        generation = {
            "index": 0,
            "workers": {
                "member_00": {"metrics": {"validation_working_point_mistag_percent": 1.08}},
                "member_01": {"metrics": {"validation_working_point_mistag_percent": 1.12}},
            },
        }

        health = strategy.update_generation_health(config, manifest, generation)

        self.assertEqual(health["current_best_member"], "member_00")
        self.assertTrue(health["baseline_degraded"])
        self.assertLess(health["relative_to_baseline"], 0)
        self.assertFalse(health["degraded"])
        self.assertEqual(health["status"], "ok")

    def test_global_best_rejects_baseline_regression_when_guarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member_dir = root / "member_00"
            member_dir.mkdir()
            state, optimizer = strategy.checkpoint_paths(member_dir, 18)
            state.write_bytes(b"bad-state")
            optimizer.write_bytes(b"bad-optimizer")
            manifest_path = root / "manifest.json"
            config = pbt_smoke_config()
            config["pbt"].update(
                metric="validation_working_point_mistag_percent",
                mode="min",
                baseline_metric_value=1.04,
                baseline_guard_tolerance=0.0,
                baseline_guard_reject_global_best=True,
            )
            manifest = {
                "config": config,
                "members": {"member_00": {"lr": 2.0e-5}},
                "generations": [],
                "best": None,
            }
            generation = {
                "index": 0,
                "epoch": 18,
                "ranking": ["member_00"],
                "workers": {
                    "member_00": {
                        "metrics": {"validation_working_point_mistag_percent": 1.08}
                    }
                },
            }

            improved = strategy.update_global_best(root, manifest, generation, manifest_path)

            self.assertFalse(improved)
            self.assertIsNone(manifest["best"])
            self.assertEqual(generation["baseline_rejected_global_best"]["reason"], "worse_than_baseline")

    def test_baseline_guard_rolls_back_to_initial_checkpoint_and_reduces_lr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member_dir = root / "member_00"
            member_dir.mkdir()
            initial_state, initial_optimizer = strategy.checkpoint_paths(member_dir, 17)
            current_state, current_optimizer = strategy.checkpoint_paths(member_dir, 18)
            initial_state.write_bytes(b"initial-state")
            initial_optimizer.write_bytes(b"initial-optimizer")
            current_state.write_bytes(b"bad-state")
            current_optimizer.write_bytes(b"bad-optimizer")
            manifest_path = root / "manifest.json"
            config = pbt_smoke_config()
            config["shared"].update(
                generations=3,
                initial_epoch=17,
                initial_state="/tmp/initial_state.pt",
                initial_optimizer="/tmp/initial_optimizer.pt",
                training_controller=None,
            )
            config["pbt"].update(
                metric="validation_working_point_mistag_percent",
                mode="min",
                baseline_metric_value=1.04,
                baseline_guard_tolerance=0.0,
                baseline_guard_action="rollback_to_initial",
                baseline_guard_lr_factor=0.7,
                min_lr=1.0e-5,
                max_lr=3.0e-5,
            )
            manifest = {
                "config": config,
                "members": {"member_00": {"lr": 2.0e-5, "parent": None}},
            }
            generation = {
                "index": 0,
                "epoch": 18,
                "workers": {
                    "member_00": {
                        "metrics": {"validation_working_point_mistag_percent": 1.08}
                    }
                },
                "exploit": [],
            }

            plan = strategy.add_baseline_guard_rollbacks(
                config, manifest, generation, manifest["members"], []
            )
            generation["exploit"] = plan
            strategy.apply_exploit(root, manifest, generation, manifest_path)

            self.assertEqual(current_state.read_bytes(), b"initial-state")
            self.assertEqual(current_optimizer.read_bytes(), b"initial-optimizer")
            self.assertAlmostEqual(manifest["members"]["member_00"]["lr"], 1.4e-5)
            self.assertEqual(manifest["members"]["member_00"]["parent_source"], "initial_resume")
            self.assertIsNone(manifest["members"]["member_00"]["parent"])
            self.assertEqual(plan[0]["source"], "initial_resume")
            self.assertTrue(plan[0]["applied"])

    def test_atomic_copy_pair_commits_all_or_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_state = root / "source_state.pt"
            source_optimizer = root / "source_optimizer.pt"
            source_state.write_bytes(b"new-state")
            source_optimizer.write_bytes(b"new-optimizer")
            destination_state = root / "dest_state.pt"
            destination_optimizer = root / "dest_optimizer.pt"
            destination_state.write_bytes(b"old-state")
            destination_optimizer.write_bytes(b"old-optimizer")

            strategy.atomic_copy_pair(
                [(source_state, destination_state), (source_optimizer, destination_optimizer)]
            )

            self.assertEqual(destination_state.read_bytes(), b"new-state")
            self.assertEqual(destination_optimizer.read_bytes(), b"new-optimizer")

    def test_atomic_copy_pair_leaves_destinations_untouched_if_any_source_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_state = root / "source_state.pt"
            source_state.write_bytes(b"new-state")
            missing_optimizer_source = root / "does_not_exist.pt"
            destination_state = root / "dest_state.pt"
            destination_optimizer = root / "dest_optimizer.pt"
            destination_state.write_bytes(b"old-state")
            destination_optimizer.write_bytes(b"old-optimizer")

            with self.assertRaises(OSError):
                strategy.atomic_copy_pair(
                    [(source_state, destination_state), (missing_optimizer_source, destination_optimizer)]
                )

            # Neither destination changed: a recipient must never end up with
            # a new weight file paired with its old, unrelated optimizer (or
            # vice versa) -- weight+optimizer copy is one coherent transition.
            self.assertEqual(destination_state.read_bytes(), b"old-state")
            self.assertEqual(destination_optimizer.read_bytes(), b"old-optimizer")


if __name__ == "__main__":
    unittest.main()

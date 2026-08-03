import unittest

from tests.helpers import pbt_smoke_config
from training.pbt.controller import (
    apply_actions_to_plan,
    apply_controller_actions_to_members,
    run_generation_controller,
    oriented_delta,
    observation_epoch_fraction,
)


class PBTDynamicControllerTest(unittest.TestCase):
    def test_oriented_delta_is_positive_when_metric_improves(self):
        config = pbt_smoke_config()
        config["pbt"]["mode"] = "min"
        self.assertAlmostEqual(oriented_delta(config, 0.9, 1.0), 0.1)
        config["pbt"]["mode"] = "max"
        self.assertAlmostEqual(oriented_delta(config, 1.1, 1.0), 0.1)


    def test_generation_epoch_fraction_defines_logical_epoch_fraction(self):
        config = pbt_smoke_config()
        config["shared"]["initial_epoch"] = 17
        config["pbt"]["dynamic_controller"] = {
            "mode": "active",
            "generation_epoch_fraction": 0.20,
        }

        self.assertAlmostEqual(
            observation_epoch_fraction(config, {"index": 0, "epoch": 18}),
            17.2,
        )
        self.assertAlmostEqual(
            observation_epoch_fraction(config, {"index": 2, "epoch": 20}),
            17.6,
        )

    def test_run_generation_controller_records_actions_without_changing_lr(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            min_lr=1.0e-5,
            max_lr=4.0e-5,
            baseline_metric_value=1.0,
            dynamic_controller={
                "mode": "active",
                "allowed_actions": ["keep", "lr_mul_0_95", "lr_mul_0_9"],
                "metric_delta_tolerance": 0.0,
            },
        )
        manifest = {
            "members": {
                "member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None},
                "member_01": {"name": "member_01", "lr": 2.2e-5, "parent": None},
            },
            "generations": [
                {
                    "index": 0,
                    "workers": {
                        "member_00": {
                            "status": "completed",
                            "metrics": {"validation_working_point_mistag_percent": 0.95},
                        },
                        "member_01": {
                            "status": "completed",
                            "metrics": {"validation_working_point_mistag_percent": 1.05},
                        },
                    },
                }
            ],
            "best": {"metric_value": 0.95},
        }
        generation = {
            "index": 1,
            "epoch": 2,
            "workers": {
                "member_00": {
                    "status": "completed",
                    "metrics": {"validation_working_point_mistag_percent": 1.02, "train_loss": 0.30, "train_accuracy": 0.89, "train_max_grad_norm": 1.4},
                },
                "member_01": {
                    "status": "completed",
                    "metrics": {"validation_working_point_mistag_percent": 0.98},
                },
            },
        }

        original_lrs = {name: dict(member) for name, member in manifest["members"].items()}
        record = run_generation_controller(config, manifest, generation)

        self.assertEqual(record["mode"], "active")
        self.assertFalse(record["applied"])
        self.assertEqual(manifest["members"], original_lrs)
        self.assertAlmostEqual(
            generation["controller_observations"]["member_00"]["metric_delta"],
            -0.07,
        )
        self.assertEqual(
            generation["controller_actions"]["member_00"]["state_label"],
            "unsafe",
        )
        self.assertAlmostEqual(generation["controller_observations"]["member_00"]["train_loss_ema"], 0.30)
        self.assertAlmostEqual(generation["controller_observations"]["member_00"]["grad_norm"], 1.4)
        self.assertEqual(
            generation["controller_actions"]["member_00"]["action"],
            "lr_mul_0_9",
        )
        self.assertEqual(
            generation["controller_actions"]["member_01"]["safety_check"],
            "passed",
        )

    def test_disabled_controller_clears_controller_fields(self):
        config = pbt_smoke_config()
        config["pbt"]["dynamic_controller"] = {"mode": "disabled"}
        manifest = {"members": {}, "generations": []}
        generation = {
            "controller_observations": {"member_00": {}},
            "controller_actions": {"member_00": {}},
            "dynamic_controller": {"mode": "active"},
        }

        self.assertIsNone(run_generation_controller(config, manifest, generation))
        self.assertNotIn("controller_observations", generation)
        self.assertNotIn("controller_actions", generation)
        self.assertNotIn("dynamic_controller", generation)

    def test_active_controller_applies_bounded_lr_to_existing_plan(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            min_lr=1.0e-5,
            max_lr=4.0e-5,
            baseline_metric_value=1.0,
            dynamic_controller={
                "mode": "active",
                "allowed_actions": ["keep", "lr_mul_0_9"],
                "metric_delta_tolerance": 0.0,
                "ema_beta": 0.5,
                "trend_window": 2,
                "action_interval_fraction": 0.1,
                "max_cumulative_lr_factor_per_epoch": 1.05,
            },
        )
        manifest = {
            "members": {
                "member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None},
            },
            "generations": [],
            "best": {"metric_value": 0.95},
        }
        generation = {
            "index": 0,
            "epoch": 18,
            "workers": {
                "member_00": {
                    "status": "completed",
                    "metrics": {"validation_working_point_mistag_percent": 1.02, "train_loss": 0.30, "train_accuracy": 0.89, "train_max_grad_norm": 1.4},
                },
            },
        }
        plan = [
            {
                "source": "anchored_lr_sweep",
                "recipient": "member_00",
                "donor": "member_00",
                "recipient_lr": 2.0e-5,
                "donor_lr": 2.0e-5,
                "anchor_member": "member_00",
                "anchor_metric": 1.02,
                "lr_factor": 1.0,
                "new_lr": 2.5e-5,
                "applied": False,
            }
        ]

        run_generation_controller(config, manifest, generation)
        updated = apply_actions_to_plan(config, generation, plan)

        self.assertTrue(generation["controller_actions"]["member_00"]["applied"])
        self.assertEqual(updated[0]["dynamic_controller_action"], "lr_mul_0_9")
        self.assertAlmostEqual(updated[0]["new_lr"], 1.9e-5)
        self.assertEqual(generation["dynamic_controller"]["applied_action_count"], 1)

    def test_active_controller_respects_cooldown(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            baseline_metric_value=1.0,
            dynamic_controller={
                "mode": "active",
                "allowed_actions": ["keep", "lr_mul_0_9"],
                "action_interval_fraction": 10.0,
                "metric_delta_tolerance": 0.0,
            },
        )
        manifest = {
            "members": {"member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None}},
            "generations": [
                {
                    "index": 0,
                    "epoch": 18,
                    "controller_observations": {
                        "member_00": {"epoch_fraction": 18.0, "metric_value": 0.95}
                    },
                    "controller_actions": {
                        "member_00": {"action": "lr_mul_0_9", "action_ready": True}
                    },
                    "workers": {
                        "member_00": {
                            "metrics": {"validation_working_point_mistag_percent": 0.95}
                        }
                    },
                }
            ],
            "best": {"metric_value": 0.95},
        }
        generation = {
            "index": 1,
            "epoch": 19,
            "workers": {
                "member_00": {
                    "status": "completed",
                    "metrics": {"validation_working_point_mistag_percent": 1.05},
                },
            },
        }

        run_generation_controller(config, manifest, generation)
        action = generation["controller_actions"]["member_00"]

        self.assertEqual(action["action"], "keep")
        self.assertEqual(action["safety_check"], "cooldown")
        self.assertFalse(action["action_ready"])
        self.assertGreater(action["cooldown_remaining"], 0.0)

    def test_controller_marks_statistically_weak_delta_as_noisy(self):
        config = pbt_smoke_config()
        config["pbt"].update(
            metric="validation_working_point_mistag_percent",
            mode="min",
            min_lr=1.0e-5,
            max_lr=4.0e-5,
            baseline_metric_value=1.2,
            dynamic_controller={
                "mode": "active",
                "allowed_actions": ["keep", "lr_mul_0_95"],
                "metric_delta_tolerance": 0.0,
                "min_delta_sigma_for_action": 1.0,
            },
        )
        manifest = {
            "members": {"member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None}},
            "generations": [
                {
                    "index": 0,
                    "epoch": 18,
                    "controller_observations": {
                        "member_00": {
                            "metric_value": 1.00,
                            "metric_ema": 1.00,
                            "metric_uncertainty": 0.05,
                        }
                    },
                    "workers": {
                        "member_00": {
                            "metrics": {"validation_working_point_mistag_percent": 1.00}
                        }
                    },
                }
            ],
            "best": {"metric_value": 1.0},
        }
        generation = {
            "index": 1,
            "epoch": 18,
            "workers": {
                "member_00": {
                    "status": "completed",
                    "metrics": {
                        "validation_working_point_mistag_percent": 1.02,
                        "validation_working_point_mistag_percent_uncertainty": 0.05,
                    },
                },
            },
        }

        run_generation_controller(config, manifest, generation)
        observation = generation["controller_observations"]["member_00"]
        action = generation["controller_actions"]["member_00"]

        self.assertAlmostEqual(observation["metric_delta_sigma"], -0.02 / (0.05 ** 2 + 0.05 ** 2) ** 0.5)
        self.assertEqual(action["state_label"], "noisy")
        self.assertEqual(action["action"], "keep")

    def test_apply_controller_actions_to_members_updates_lr_independent_of_exploit(self):
        # Two members both get a ready, non-keep action this generation, and
        # NEITHER is an exploit recipient -- this is exactly the case that
        # was previously silently dropped (only plan/exploit recipients ever
        # got a controller action applied).
        config = pbt_smoke_config()
        config["pbt"]["dynamic_controller"] = {"mode": "active"}
        manifest = {
            "members": {
                "member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None},
                "member_01": {"name": "member_01", "lr": 3.0e-5, "parent": None},
            }
        }
        generation = {
            "index": 0,
            "controller_actions": {
                "member_00": {"action": "lr_mul_0_9", "action_ready": True, "safety_check": "passed", "bounded_lr": 1.8e-5},
                "member_01": {"action": "lr_mul_1_05", "action_ready": True, "safety_check": "passed", "bounded_lr": 3.15e-5},
            },
        }

        applied = apply_controller_actions_to_members(config, manifest, generation)

        self.assertAlmostEqual(manifest["members"]["member_00"]["lr"], 1.8e-5)
        self.assertAlmostEqual(manifest["members"]["member_01"]["lr"], 3.15e-5)
        self.assertEqual(set(applied), {"member_00", "member_01"})
        self.assertTrue(generation["controller_actions"]["member_00"]["applied"])

    def test_apply_controller_actions_to_members_skips_excluded_members(self):
        config = pbt_smoke_config()
        config["pbt"]["dynamic_controller"] = {"mode": "active"}
        manifest = {"members": {"member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None}}}
        generation = {
            "index": 0,
            "controller_actions": {
                "member_00": {"action": "lr_mul_0_9", "action_ready": True, "safety_check": "passed", "bounded_lr": 1.8e-5},
            },
        }

        applied = apply_controller_actions_to_members(config, manifest, generation, exclude_members={"member_00"})

        self.assertEqual(applied, {})
        self.assertAlmostEqual(manifest["members"]["member_00"]["lr"], 2.0e-5)

    def test_apply_controller_actions_to_members_skips_keep_and_not_ready(self):
        config = pbt_smoke_config()
        config["pbt"]["dynamic_controller"] = {"mode": "active"}
        manifest = {
            "members": {
                "member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None},
                "member_01": {"name": "member_01", "lr": 3.0e-5, "parent": None},
            }
        }
        generation = {
            "index": 0,
            "controller_actions": {
                "member_00": {"action": "keep", "action_ready": True, "safety_check": "passed", "bounded_lr": 2.0e-5},
                "member_01": {"action": "lr_mul_0_9", "action_ready": False, "safety_check": "cooldown", "bounded_lr": 2.7e-5},
            },
        }

        applied = apply_controller_actions_to_members(config, manifest, generation)

        self.assertEqual(applied, {})
        self.assertAlmostEqual(manifest["members"]["member_00"]["lr"], 2.0e-5)
        self.assertAlmostEqual(manifest["members"]["member_01"]["lr"], 3.0e-5)

    def test_apply_controller_actions_to_members_noop_when_controller_disabled(self):
        config = pbt_smoke_config()
        config["pbt"]["dynamic_controller"] = {"mode": "disabled"}
        manifest = {"members": {"member_00": {"name": "member_00", "lr": 2.0e-5, "parent": None}}}
        generation = {
            "index": 0,
            "controller_actions": {
                "member_00": {"action": "lr_mul_0_9", "action_ready": True, "safety_check": "passed", "bounded_lr": 1.8e-5},
            },
        }

        applied = apply_controller_actions_to_members(config, manifest, generation)

        self.assertEqual(applied, {})
        self.assertAlmostEqual(manifest["members"]["member_00"]["lr"], 2.0e-5)


if __name__ == "__main__":
    unittest.main()

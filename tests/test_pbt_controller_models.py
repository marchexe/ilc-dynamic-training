import unittest

from tests.helpers import SCRIPTS_DIR  # noqa: F401
from training.pbt.models.controller import (
    ControllerAction,
    ControllerObservation,
    dump_controller_action,
    dump_controller_observation,
    parse_controller_action,
    parse_controller_observation,
)


def valid_observation_payload():
    return {
        "schema_version": 1,
        "generation": 2,
        "member": "member_03",
        "epoch": 19,
        "epoch_fraction": 19.05,
        "step": 1234,
        "lr": 2.0e-5,
        "metric_name": "validation_working_point_mistag_percent",
        "metric_value": 1.04,
        "previous_metric_value": 1.05,
        "metric_delta": -0.01,
        "metric_uncertainty": 0.02,
        "previous_metric_uncertainty": 0.01,
        "metric_delta_sigma": -0.4472135955,
        "baseline_metric_value": 1.0367,
        "baseline_delta": 0.0033,
        "global_best_metric_value": 1.0367,
        "global_best_delta": 0.0033,
        "train_loss": 0.23,
        "train_accuracy": 0.91,
        "train_loss_ema": 0.21,
        "train_loss_ema_delta": 0.02,
        "grad_norm": 1.5,
        "amp_skipped_optimizer_steps": 0,
        "max_cuda_memory_mb": 12345.0,
        "optimizer_step": 168700.0,
        "optimizer_param_groups": 1,
        "optimizer_lr_mean": 2.0e-5,
        "optimizer_lr_min": 2.0e-5,
        "optimizer_lr_max": 2.0e-5,
        "optimizer_weight_decay_mean": 0.01,
        "momentum_norm": 0.8,
        "second_moment_norm": 0.3,
        "adaptive_direction_norm": 2.1,
        "adaptive_direction_norm_max": 3.2,
        "action_ready": False,
        "allowed_actions": ["keep", "lr_mul_0_95", "lr_mul_1_05"],
    }


def valid_action_payload():
    return {
        "schema_version": 1,
        "generation": 2,
        "member": "member_03",
        "state_label": "flat",
        "confidence": 0.75,
        "action": "keep",
        "reason": "metric trend is inconclusive",
        "safety_check": "passed",
        "applied": False,
        "action_ready": False,
    }


class PBTControllerModelsTest(unittest.TestCase):
    def test_valid_observation_and_action_parse(self):
        observation = parse_controller_observation(valid_observation_payload())
        action = parse_controller_action(valid_action_payload())

        self.assertIsInstance(observation, ControllerObservation)
        self.assertIsInstance(action, ControllerAction)
        self.assertEqual(observation.allowed_actions[0], "keep")
        self.assertEqual(action.action, "keep")

    def test_unknown_action_is_rejected(self):
        payload = valid_action_payload()
        payload["action"] = "lr_mul_2_0"

        with self.assertRaises(ValueError):
            parse_controller_action(payload)

    def test_confidence_outside_unit_interval_is_rejected(self):
        payload = valid_action_payload()
        payload["confidence"] = 1.5

        with self.assertRaises(ValueError):
            parse_controller_action(payload)

    def test_non_positive_lr_is_rejected(self):
        payload = valid_observation_payload()
        payload["lr"] = 0.0

        with self.assertRaises(ValueError):
            parse_controller_observation(payload)

    def test_extra_fields_are_forbidden(self):
        observation_payload = valid_observation_payload()
        observation_payload["raw_log_line"] = "free-form controller input"
        action_payload = valid_action_payload()
        action_payload["unvalidated_action"] = "raise_lr_aggressively"

        with self.assertRaises(ValueError):
            parse_controller_observation(observation_payload)
        with self.assertRaises(ValueError):
            parse_controller_action(action_payload)

    def test_dump_parse_roundtrip_preserves_payload(self):
        observation_payload = valid_observation_payload()
        action_payload = valid_action_payload()

        observation_dump = dump_controller_observation(observation_payload)
        action_dump = dump_controller_action(action_payload)

        self.assertEqual(observation_dump, observation_payload)
        self.assertEqual(action_dump, action_payload)
        self.assertEqual(
            dump_controller_observation(parse_controller_observation(observation_dump)),
            observation_payload,
        )
        self.assertEqual(
            dump_controller_action(parse_controller_action(action_dump)),
            action_payload,
        )


if __name__ == "__main__":
    unittest.main()

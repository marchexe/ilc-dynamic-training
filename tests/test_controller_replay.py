import argparse
import hashlib
import json
import unittest
from pathlib import Path

from training.pbt.config import load_config
from validation.controller_replay import (
    ReplayPolicy,
    estimate_change_noise,
    replay,
    signal_for_delta,
)


def row(epoch, proxy, reference):
    return {
        "epoch": epoch,
        "proxy_metric": proxy,
        "reference_metric": reference,
        "checkpoint_sha256": f"sha-{epoch}",
    }


class ControllerReplayTest(unittest.TestCase):
    def test_signal_uses_lower_is_better_orientation_and_threshold(self):
        self.assertEqual(signal_for_delta(-0.02, 0.01), "UP")
        self.assertEqual(signal_for_delta(0.02, 0.01), "DOWN")
        self.assertEqual(signal_for_delta(0.005, 0.01), "FLAT")
        self.assertEqual(signal_for_delta(None, 0.01), "FLAT")

    def test_patience_prevents_reaction_to_one_isolated_bad_point(self):
        trajectories = {
            ("run", "member"): [
                row(1, 1.00, 1.00),
                row(2, 1.03, 1.03),
                row(3, 1.00, 1.00),
            ]
        }
        result = replay(
            trajectories,
            0.01,
            ReplayPolicy(ema_beta=0.0, patience=2, cooldown_observations=2),
        )
        self.assertEqual(result["action_counts"], {"UP": 0, "KEEP": 3, "DOWN": 0})

    def test_sustained_direction_acts_once_then_cools_down(self):
        trajectories = {
            ("run", "member"): [
                row(1, 1.00, 1.00),
                row(2, 0.97, 0.97),
                row(3, 0.94, 0.94),
                row(4, 0.91, 0.91),
                row(5, 0.88, 0.88),
            ]
        }
        result = replay(
            trajectories,
            0.01,
            ReplayPolicy(ema_beta=0.0, patience=2, cooldown_observations=2),
        )
        self.assertEqual(result["action_counts"], {"UP": 1, "KEEP": 4, "DOWN": 0})
        self.assertEqual(result["action_reversals"], 0)
        self.assertEqual(result["active_action_disagreements"], 0)

    def test_noise_scale_is_paired_delta_residual_mad(self):
        trajectories = {
            ("run", "member"): [
                row(1, 1.00, 1.00),
                row(2, 0.90, 0.91),
                row(3, 0.82, 0.81),
                row(4, 0.70, 0.72),
            ]
        }
        estimate = estimate_change_noise(trajectories)
        self.assertEqual(estimate["adjacent_pair_count"], 3)
        self.assertGreater(estimate["robust_sigma"], 0.0)


class Representative60kShadowConfigTest(unittest.TestCase):
    def test_config_is_fixed_lr_deterministic_and_shadow_only(self):
        config_path = Path(
            "configs/experiments/pretrained_fixed_lr_8gpu_representative60k_shadow.yaml"
        )
        config = load_config(
            argparse.Namespace(
                config=config_path,
                experiment_name=None,
                gpus=None,
                slots=None,
                smoke=False,
            )
        )

        self.assertEqual(config["pbt"]["strategy"], "fixed_lr_grid")
        self.assertEqual(config["pbt"]["metric"], "validation_total_reference_mistag_geomean_percent")
        self.assertEqual(config["shared"]["generations"], 96)
        self.assertEqual(config["shared"]["samples_per_epoch"], 120000)
        self.assertEqual(config["shared"]["samples_per_epoch_val"], 60000)
        self.assertTrue(config["shared"]["deterministic"])
        self.assertEqual(len(config["population"]), 8)
        self.assertEqual(
            [member["start_lr"] for member in config["population"]],
            [3e-6, 4.5e-6, 6e-6, 7.5e-6, 9e-6, 10.5e-6, 12e-6, 14e-6],
        )

        controller = config["pbt"]["dynamic_controller"]
        self.assertEqual(controller["mode"], "shadow")
        self.assertEqual(controller["policy"], "patient_bidirectional")
        self.assertEqual(controller["allowed_actions"], ["keep", "lr_mul_0_95", "lr_mul_1_05"])
        self.assertEqual(controller["ema_beta"], 0.5)
        self.assertAlmostEqual(controller["metric_delta_tolerance"], 0.004579530054761879)
        self.assertEqual(controller["direction_patience"], 2)
        self.assertEqual(controller["action_interval_fraction"], 0.6)
        self.assertEqual(controller["generation_epoch_fraction"], 0.2)
        self.assertEqual(controller["max_cumulative_lr_factor_per_epoch"], 1.0)
        self.assertEqual(config["pbt"]["tiered_validation"]["monitor_interval_generations"], 5)
        self.assertIsNone(config["pbt"]["tiered_validation"].get("full_interval_generations"))

    def test_membership_is_the_qualified_representative_60k_manifest(self):
        proxy_sets = json.loads(
            Path("runs/eval/proxy_qualification_v1/proxy_sets.json").read_text()
        )
        candidate = next(
            item for item in proxy_sets["candidates"] if item["id"] == "representative_60k"
        )
        membership = Path(candidate["membership_manifest"])
        payload = json.loads(membership.read_text())
        digest = hashlib.sha256(membership.read_bytes()).hexdigest()

        self.assertEqual(digest, candidate["proxy_manifest_sha256"])
        self.assertEqual(payload["candidate_id"], "representative_60k")
        self.assertEqual(payload["rows_total"], 60000)
        self.assertEqual(payload["class_counts"], {"bb": 20000, "cc": 20000, "dd": 20000})
        for file_record in payload["files"].values():
            proxy_file = Path(file_record["path"])
            self.assertTrue(proxy_file.is_file())
            self.assertEqual(
                hashlib.sha256(proxy_file.read_bytes()).hexdigest(),
                file_record["sha256"],
            )


if __name__ == "__main__":
    unittest.main()

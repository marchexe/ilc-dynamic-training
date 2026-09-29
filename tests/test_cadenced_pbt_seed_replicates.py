"""CPU-only contract tests for paired cadenced-PBT seed replicates."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

from tests.helpers import PROJECT_DIR
from scripts.launch.experiment import configuration
from training.pbt.execution.backend import LocalWeaverBackend


CONFIG_DIR = PROJECT_DIR / "configs/experiments"
CONFIGS = {
    (22345, 1): CONFIG_DIR / "cadenced_pbt_v1_seed22345.yaml",
    (22345, 5): CONFIG_DIR / "cadenced_pbt_v1_cadence5_seed22345.yaml",
    (32345, 1): CONFIG_DIR / "cadenced_pbt_v1_seed32345.yaml",
    (32345, 5): CONFIG_DIR / "cadenced_pbt_v1_cadence5_seed32345.yaml",
}


def normalized(config, *, drop_seed=False, drop_interval=False):
    result = copy.deepcopy(config)
    for key in ("config_path", "experiment_name"):
        result.pop(key, None)
    if drop_seed:
        result["shared"].pop("seed")
    if drop_interval:
        result["pbt"].pop("exploit_interval_generations")
    return result


def option(command, flag):
    return command[command.index(flag) + 1]


class CadencedPBTSeedReplicateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            key: configuration(path, "0,1,2,3,4")
            for key, path in CONFIGS.items()
        }

    def test_each_pair_differs_only_by_name_path_and_cadence(self):
        for seed in (22345, 32345):
            cadence1 = self.configs[(seed, 1)]
            cadence5 = self.configs[(seed, 5)]
            self.assertEqual(cadence1["shared"]["seed"], seed)
            self.assertEqual(cadence5["shared"]["seed"], seed)
            self.assertEqual(cadence1["pbt"]["exploit_interval_generations"], 1)
            self.assertEqual(cadence5["pbt"]["exploit_interval_generations"], 5)
            self.assertEqual(
                normalized(cadence1, drop_interval=True),
                normalized(cadence5, drop_interval=True),
            )

    def test_replicates_differ_only_by_name_path_and_training_seed(self):
        for cadence in (1, 5):
            seed2 = self.configs[(22345, cadence)]
            seed3 = self.configs[(32345, cadence)]
            self.assertEqual(
                normalized(seed2, drop_seed=True),
                normalized(seed3, drop_seed=True),
            )

    def test_production_contract_and_pretrained_initialization_are_fixed(self):
        reference = self.configs[(22345, 1)]
        expected_lrs = [3e-6, 5.75e-6, 8.5e-6, 11.25e-6, 14e-6]
        for config in self.configs.values():
            shared = config["shared"]
            pbt = config["pbt"]
            self.assertEqual(shared["generations"], 50)
            self.assertEqual(shared["weaver_epochs_per_generation"], 1)
            self.assertEqual([item["start_lr"] for item in config["population"]], expected_lrs)
            self.assertEqual(shared["initial_state"], reference["shared"]["initial_state"])
            self.assertEqual(shared["initial_optimizer"], reference["shared"]["initial_optimizer"])
            self.assertEqual(shared["initial_epoch"], 17)
            self.assertEqual(shared["initial_optimizer_mode"], "raw")
            self.assertTrue(shared["deterministic"])
            self.assertTrue(shared["data_audit"])
            self.assertEqual(pbt["metric"], "validation_total_reference_mistag_geomean_percent")
            self.assertEqual(pbt["cadenced_pbt_v1"]["decision_margin"], 0.002)
            self.assertEqual(pbt["seed"], 2026)

    def test_shared_seed_drives_worker_seed_and_pbt_seed_does_not(self):
        backend = LocalWeaverBackend()
        for (seed, cadence), config in self.configs.items():
            member = {"name": "lr_3e-6", "lr": 3e-6}
            for generation in (0, 17, 49):
                command, _, _ = backend.command_for(
                    config,
                    member,
                    "0",
                    Path("/tmp/cadenced_pbt_seed_contract") / config["experiment_name"] / member["name"],
                    generation,
                )
                self.assertEqual(int(option(command, "--seed")), seed + generation)
                changed_pbt_seed = copy.deepcopy(config)
                changed_pbt_seed["pbt"]["seed"] = 987654
                changed, _, _ = backend.command_for(
                    changed_pbt_seed,
                    member,
                    "0",
                    Path("/tmp/cadenced_pbt_seed_contract") / config["experiment_name"] / member["name"],
                    generation,
                )
                self.assertEqual(option(changed, "--seed"), option(command, "--seed"))
        self.assertTrue(set(range(22345, 22395)).isdisjoint(range(32345, 32395)))


if __name__ == "__main__":
    unittest.main()

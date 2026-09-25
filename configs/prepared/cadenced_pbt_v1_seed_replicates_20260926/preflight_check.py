#!/usr/bin/env python3
"""Resolve and prove the paired-seed cadence-control contracts without training."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from training.pbt.config import load_config, validate_inputs  # noqa: E402
from training.pbt.execution.backend import LocalWeaverBackend  # noqa: E402
from training.runtime import sha256  # noqa: E402


OUT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs/experiments"
CONFIGS = {
    "seed22345_cadence1": CONFIG_DIR / "cadenced_pbt_v1_seed22345.yaml",
    "seed22345_cadence5": CONFIG_DIR / "cadenced_pbt_v1_cadence5_seed22345.yaml",
    "seed32345_cadence1": CONFIG_DIR / "cadenced_pbt_v1_seed32345.yaml",
    "seed32345_cadence5": CONFIG_DIR / "cadenced_pbt_v1_cadence5_seed32345.yaml",
}
PAIRWISE_DIFFS = {
    "seed22345_cadence1_vs_cadence5": ("seed22345_cadence1", "seed22345_cadence5"),
    "seed32345_cadence1_vs_cadence5": ("seed32345_cadence1", "seed32345_cadence5"),
    "cadence1_seed22345_vs_seed32345": ("seed22345_cadence1", "seed32345_cadence1"),
    "cadence5_seed22345_vs_seed32345": ("seed22345_cadence5", "seed32345_cadence5"),
}


def resolve(path):
    config = load_config(
        SimpleNamespace(
            config=path,
            experiment_name=None,
            gpus="0,1,2,3,4",
            slots=None,
            smoke=False,
        )
    )
    validate_inputs(config)
    return config


def normalize(config, *paths):
    result = copy.deepcopy(config)
    for path in paths:
        node = result
        pieces = path.split(".")
        for piece in pieces[:-1]:
            node = node[piece]
        node.pop(pieces[-1], None)
    return result


def option(command, flag):
    return command[command.index(flag) + 1]


def rng_witness(seed):
    # SimpleIterDataset uses numpy.default_rng from the configured seed base.
    payload = np.random.default_rng(seed).permutation(4096).tobytes()
    return hashlib.sha256(payload).hexdigest()


def main():
    resolved = {name: resolve(path) for name, path in CONFIGS.items()}
    expected_lrs = [3e-6, 5.75e-6, 8.5e-6, 11.25e-6, 14e-6]
    initial_hashes = {}
    backend = LocalWeaverBackend()
    command_seeds = {}

    for name, config in resolved.items():
        seed = int(config["shared"]["seed"])
        interval = int(config["pbt"]["exploit_interval_generations"])
        assert seed in (22345, 32345)
        assert interval in (1, 5)
        assert config["shared"]["generations"] == 50
        assert config["shared"]["weaver_epochs_per_generation"] == 1
        assert [member["start_lr"] for member in config["population"]] == expected_lrs
        assert config["pbt"]["metric"] == "validation_total_reference_mistag_geomean_percent"
        assert config["pbt"]["cadenced_pbt_v1"] == {
            "warmup_epochs": 2,
            "decision_margin": 0.002,
            "max_recipients": 1,
        }
        assert config["pbt"]["seed"] == 2026
        assert config["shared"]["initial_epoch"] == 17
        assert config["shared"]["initial_optimizer_mode"] == "raw"
        assert config["shared"]["deterministic"] is True
        assert config["shared"]["data_audit"] is True
        assert not (Path(config["output_root"]) / config["experiment_name"]).exists()
        initial_hashes[name] = {
            key: sha256(Path(config["shared"][key]))
            for key in ("initial_state", "initial_optimizer", "data_config", "network_config")
        }
        member = {"name": "lr_3e-6", "lr": 3e-6}
        command_seeds[name] = []
        for generation in range(50):
            command, _, target_epoch = backend.command_for(
                config,
                member,
                "0",
                Path("/tmp/cadenced_pbt_seed_preflight") / config["experiment_name"] / member["name"],
                generation,
            )
            worker_seed = int(option(command, "--seed"))
            assert worker_seed == seed + generation
            assert target_epoch == 18 + generation
            command_seeds[name].append(worker_seed)

    assert len({json.dumps(value, sort_keys=True) for value in initial_hashes.values()}) == 1
    for seed in (22345, 32345):
        left = resolved[f"seed{seed}_cadence1"]
        right = resolved[f"seed{seed}_cadence5"]
        assert normalize(left, "config_path", "experiment_name", "pbt.exploit_interval_generations") == normalize(
            right, "config_path", "experiment_name", "pbt.exploit_interval_generations"
        )
    for cadence in (1, 5):
        left = resolved[f"seed22345_cadence{cadence}"]
        right = resolved[f"seed32345_cadence{cadence}"]
        assert normalize(left, "config_path", "experiment_name", "shared.seed") == normalize(
            right, "config_path", "experiment_name", "shared.seed"
        )
    assert set(command_seeds["seed22345_cadence1"]).isdisjoint(
        command_seeds["seed32345_cadence1"]
    )
    assert command_seeds["seed22345_cadence1"] == command_seeds["seed22345_cadence5"]
    assert command_seeds["seed32345_cadence1"] == command_seeds["seed32345_cadence5"]
    witnesses = {str(seed): rng_witness(seed) for seed in (22345, 32345)}
    assert len(set(witnesses.values())) == 2

    resolved_dir = OUT / "resolved"
    diffs_dir = OUT / "diffs"
    resolved_dir.mkdir(exist_ok=True)
    diffs_dir.mkdir(exist_ok=True)
    rendered = {}
    for name, config in resolved.items():
        text = yaml.safe_dump(config, sort_keys=False)
        rendered[name] = text
        (resolved_dir / f"{name}.yaml").write_text(text)
    for label, (left, right) in PAIRWISE_DIFFS.items():
        diff = difflib.unified_diff(
            rendered[left].splitlines(keepends=True),
            rendered[right].splitlines(keepends=True),
            fromfile=f"{left}.yaml",
            tofile=f"{right}.yaml",
        )
        (diffs_dir / f"{label}.diff").write_text("".join(diff))

    result = {
        "status": "PREPARED_NOT_LAUNCHED",
        "training_seed_field": "shared.seed",
        "generation_seed_rule": "shared.seed + zero_based_generation",
        "pbt_seed_fixed": 2026,
        "training_seed_ranges": {"22345": [22345, 22394], "32345": [32345, 32394]},
        "dataset_rng_witness_sha256": witnesses,
        "initialization_hashes": next(iter(initial_hashes.values())),
        "resolved_config_sha256": {
            name: hashlib.sha256(rendered[name].encode()).hexdigest()
            for name in rendered
        },
        "proofs": {
            "within_pair_only_name_path_and_interval": True,
            "between_pairs_only_name_path_and_training_seed": True,
            "same_worker_seed_sequence_within_pair": True,
            "disjoint_worker_seed_sequences_between_pairs": True,
            "all_initialization_hashes_identical": True,
        },
    }
    (OUT / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

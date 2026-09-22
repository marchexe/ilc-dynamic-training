#!/usr/bin/env python3
"""CPU-only provenance preflight for the 50-epoch pretrained cadenced run.

This reads and hashes configuration, source, checkpoint, and dataset inputs. It
does not invoke Weaver, validation, the PBT runner, nvidia-smi, CUDA, or GPUs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.pbt.config import load_config, validate_inputs
from training.pbt.reporting.io import run_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/cadenced_pbt_v1.yaml"),
    )
    parser.add_argument("--gpus", help="Configured physical GPU IDs; no devices are queried")
    args = parser.parse_args()
    config = load_config(argparse.Namespace(
        config=args.config,
        experiment_name=None,
        gpus=args.gpus,
        slots=None,
        smoke=False,
    ))
    if config["pbt"]["strategy"] != "cadenced_pbt_v1":
        parser.error("config must select cadenced_pbt_v1")
    if config["shared"]["generations"] != 50 or config["shared"].get("initialization_mode") == "scratch":
        parser.error("pre-production contract requires the 50-epoch pretrained configuration")
    validate_inputs(config)
    gpu_ids = [str(gpu) for gpu in config["gpus"]]
    launch = [
        str(Path(sys.executable)),
        "scripts/launch/experiment.py",
        "start",
        "--config",
        str(args.config),
        "--gpus",
        ",".join(gpu_ids),
    ]
    contract = run_contract(config, launch, "local_weaver")
    print(json.dumps({
        "check": "CPU-only; no training, validation, PBT, CUDA, or GPU query executed",
        "launch_command_not_executed": launch,
        "experiment": config["experiment_name"],
        "generations": config["shared"]["generations"],
        "contract": contract,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only synthetic decision schedule for cadenced_pbt_v1 (no Weaver/GPU)."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.pbt.config import load_config
from training.pbt.planning.cadenced_pbt_v1 import cadenced_pbt_v1_plan
from training.pbt.state.checkpointing import epoch_for_generation


def synthetic_schedule(config, generations):
    config = copy.deepcopy(config)
    config["shared"]["generations"] = generations
    members = {item["name"]: {"lr": float(item["start_lr"])} for item in config["population"]}
    names = list(members)
    for index in range(generations):
        # Deliberately fixed, clearly synthetic scores with one gap > 0.002.
        scores = [0.400, 0.401, 0.402, 0.403, 0.410]
        generation = {
            "index": index,
            "epoch": epoch_for_generation(config, index),
            "workers": {
                name: {"status": "completed", "returncode": 0,
                       "metrics": {config["pbt"]["metric"]: scores[position]}}
                for position, name in enumerate(names)
            },
        }
        _, plan = cadenced_pbt_v1_plan(config, generation, members)
        decision = generation["cadenced_pbt_v1"]
        yield {
            "generation": index,
            "completed_epoch": decision["completed_epoch"],
            "weaver_epoch": decision["weaver_epoch"],
            "validation": "synthetic_full_reference",
            "cadence_interval_epochs": decision["cadence_interval_epochs"],
            "cadence_boundary": decision["cadence_boundary"],
            "warmup_active_during_training": decision["warmup_active_during_training"],
            "warmup_complete_at_boundary": decision["warmup_complete_at_boundary"],
            "terminal": decision["terminal"],
            "copy_opportunity": decision["copy_opportunity"],
            "donor": decision["donor"],
            "recipient": decision["recipient"],
            "metric_gap": decision["metric_gap"],
            "copy_planned": decision["copy_planned"],
            "old_lr": decision.get("old_lr"),
            "new_lr": decision.get("new_lr"),
            "mutation_applied": decision.get("mutation_applied", False),
            "mutation_reason": decision.get("mutation_reason", "not_attempted"),
            "reason": decision["reason"],
        }
        if plan:
            members[plan[0]["recipient"]]["lr"] = plan[0]["new_lr"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=6)
    args = parser.parse_args()
    if args.generations < 1:
        parser.error("--generations must be positive")
    config = load_config(argparse.Namespace(
        config=args.config, experiment_name=None, gpus=None, slots=None, smoke=False,
    ))
    if config["pbt"]["strategy"] != "cadenced_pbt_v1":
        parser.error("config must select cadenced_pbt_v1")
    print("Synthetic schedule only: no model, dataset, checkpoint, or GPU is opened.")
    for row in synthetic_schedule(config, args.generations):
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()

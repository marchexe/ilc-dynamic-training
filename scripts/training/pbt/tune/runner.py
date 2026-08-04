#!/usr/bin/env python3
"""Ray Tune CLI runner for Weaver fine-tuning trials."""

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path

import yaml

from training.pbt.config import load_config, validate_inputs
from training.pbt.tune.trainable import run_weaver_trial, run_weaver_trial_direct, tune_trial_slot
from training.pbt.execution.weaver_command import make_command
from training.runtime import PROJECT_DIR


DEFAULT_CONFIG = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"


def _ray_tune_import():
    try:
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler, FIFOScheduler, MedianStoppingRule
    except ImportError as error:
        raise RuntimeError("Ray Tune runner requires ray[tune]. Install project requirements first.") from error
    return ray, tune, ASHAScheduler, FIFOScheduler, MedianStoppingRule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-name")
    parser.add_argument("--gpus", help="Comma-separated physical GPUs visible to Ray, e.g. 0,2")
    parser.add_argument(
        "--slots",
        help="Accepted for config compatibility, but Tune trials use Ray-managed local GPU resources.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two members, two generations and small train/validation budgets",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scheduler", choices=("fifo", "asha", "median"), default="fifo")
    parser.add_argument("--generations", type=int, help="Override generations per Tune trial")
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--gpus-per-trial", type=float, default=1.0)
    parser.add_argument("--max-concurrent-trials", type=int)
    parser.add_argument("--direct-trial-smoke", action="store_true", help="Run one Tune-style trial in this process without Ray actor scheduling")
    parser.add_argument("--direct-trial-index", type=int, default=0)
    parser.add_argument("--object-store-memory-mb", type=int, default=1024)
    parser.add_argument(
        "--ray-address",
        default="local",
        help="Ray address. Default local starts a fresh local runtime; use auto to attach to an existing one.",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        help="Ray Tune storage path. Defaults to <output_root>/ray_tune.",
    )
    return parser.parse_args()


def build_trial_specs(config, generations=None):
    trial_generations = int(generations or config["shared"]["generations"])
    slot = tune_trial_slot()
    return [
        {
            "member_name": member["name"],
            "lr": float(member["start_lr"]),
            "generations": trial_generations,
            "slot": slot,
        }
        for member in config["population"]
    ]


def small_tune_payload(config, trial):
    return {
        "config_path": str(Path(config["config_path"])),
        "experiment_name": config["experiment_name"],
        "gpus": ",".join(config["gpus"]),
        "smoke": bool(config["smoke"]),
        "output_root": str(config["output_root"]),
        "trial": trial,
    }


def ray_runtime_env():
    """Make project modules importable inside Ray worker processes."""
    paths = [str(PROJECT_DIR / "scripts"), str(PROJECT_DIR / "weaver-core")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    return {"env_vars": {"PYTHONPATH": os.pathsep.join(paths)}}


def build_scheduler(name, config):
    _, _, ASHAScheduler, FIFOScheduler, MedianStoppingRule = _ray_tune_import()
    metric = config["pbt"]["metric"]
    mode = config["pbt"]["mode"]
    if name == "fifo":
        return FIFOScheduler()
    if name == "asha":
        return ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=int(config["shared"]["generations"]),
            time_attr="generation",
            grace_period=1,
            reduction_factor=2,
        )
    if name == "median":
        return MedianStoppingRule(metric=metric, mode=mode, time_attr="generation", grace_period=1)
    raise ValueError(f"Unsupported Tune scheduler: {name}")


def dry_run(config, trial_specs):
    print(yaml.safe_dump(config, sort_keys=False).rstrip())
    print("# Ray Tune trial commands")
    for trial in trial_specs:
        member_dir = (
            Path(config["output_root"])
            / f"{config['experiment_name']}_tune"
            / trial["member_name"]
        )
        command, _, _ = make_command(
            config,
            {"name": trial["member_name"], "lr": trial["lr"]},
            trial["slot"],
            member_dir,
            0,
        )
        print(f"[{trial['member_name']}] {shlex.join(command)}")
    return 0


def run(args):
    if args.slots:
        raise ValueError("Ray Tune runner uses Ray-managed local GPU resources; use --gpus for Ray visibility")
    config = load_config(args)
    validate_inputs(config)
    trial_specs = build_trial_specs(config, generations=args.generations)
    if args.dry_run:
        return dry_run(config, trial_specs)
    if args.direct_trial_smoke:
        if not 0 <= args.direct_trial_index < len(trial_specs):
            raise ValueError("--direct-trial-index is outside the resolved population")
        results = run_weaver_trial_direct(
            small_tune_payload(config, trial_specs[args.direct_trial_index])
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ray, tune, _, _, _ = _ray_tune_import()
    scheduler = build_scheduler(args.scheduler, config)
    storage_path = args.storage_path or Path(config["output_root"]) / "ray_tune"
    max_concurrent = args.max_concurrent_trials or max(1, len(config["slots"]))
    num_cpus = max(1, math.ceil(args.cpus_per_trial * max_concurrent))
    num_gpus = max(0.0, float(args.gpus_per_trial) * max_concurrent)
    if not ray.is_initialized():
        ray_address = None if args.ray_address == "none" else args.ray_address
        ray.init(
            address=ray_address,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            include_dashboard=False,
            ignore_reinit_error=True,
            object_store_memory=max(256, args.object_store_memory_mb) * 1024 * 1024,
            runtime_env=ray_runtime_env(),
        )
    trainable = tune.with_resources(
        run_weaver_trial,
        {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
    )
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            metric=config["pbt"]["metric"],
            mode=config["pbt"]["mode"],
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            name=config["experiment_name"],
            storage_path=str(storage_path),
        ),
        param_space={
            "config_path": str(Path(config["config_path"])),
            "experiment_name": config["experiment_name"],
            "gpus": ",".join(config["gpus"]),
            "smoke": bool(config["smoke"]),
            "output_root": str(config["output_root"]),
            "trial": tune.grid_search(trial_specs),
        },
    )
    try:
        results = tuner.fit()
    finally:
        ray.shutdown()
    if results.errors:
        for error in results.errors:
            print(f"trial error: {error}", file=sys.stderr)
        return 1
    return 0


def main():
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

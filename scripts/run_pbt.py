#!/usr/bin/env python3
"""Run epoch-level Population Based Training on independent Weaver workers."""

import argparse
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from run_parallel_training import (
    PROJECT_DIR,
    atomic_json,
    build_command,
    git_metadata,
    project_path,
    read_metrics,
    sha256,
    terminate,
    utc_now,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/experiments/pp_pbt.yaml"
MANIFEST_NAME = "manifest.json"
MEMBER_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-name")
    parser.add_argument("--gpus", help="Comma-separated GPU slots, e.g. 0,2")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two members, two generations and small train/validation budgets",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute_project_path(value, *, resolve=True):
    path = project_path(value)
    return str(path.resolve() if resolve else path.absolute())


def load_config(args):
    config_path = args.config.resolve()
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 PBT configuration")

    mapping_sections = ("experiment", "shared", "resources", "pbt")
    if any(not isinstance(payload.get(key), dict) for key in mapping_sections):
        raise ValueError(f"PBT configuration requires mappings: {', '.join(mapping_sections)}")
    if not isinstance(payload.get("population"), list):
        raise ValueError("PBT configuration requires a population list")

    experiment = dict(payload["experiment"])
    shared = dict(payload["shared"])
    resources = dict(payload["resources"])
    population = [dict(member) for member in payload["population"]]
    pbt = dict(payload["pbt"])

    required_shared = {
        "dataset",
        "checkpoint",
        "data_config",
        "network_config",
        "seed",
        "generations",
        "epochs_per_generation",
        "samples_per_epoch",
        "samples_per_epoch_val",
        "batch_size",
        "optimizer",
        "lr_scheduler",
        "num_workers",
        "fetch_step",
        "use_amp",
        "amp_dtype",
        "no_remake_weights",
    }
    missing = sorted(required_shared - shared.keys())
    if missing:
        raise ValueError(f"Missing shared options: {', '.join(missing)}")

    for key in ("dataset", "data_config", "network_config"):
        shared[key] = absolute_project_path(shared[key])
    shared["checkpoint"] = absolute_project_path(shared["checkpoint"], resolve=False)

    gpus = args.gpus.split(",") if args.gpus else [str(gpu) for gpu in resources.get("gpus", [])]
    gpus = [gpu.strip() for gpu in gpus]
    if not gpus or any(not gpu for gpu in gpus):
        raise ValueError("At least one non-empty GPU slot is required")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU slots must be unique")

    if args.smoke:
        shared.update(
            generations=2,
            epochs_per_generation=1,
            samples_per_epoch=7680,
            samples_per_epoch_val=3000,
        )
        population = population[:2]

    if len(population) < 2:
        raise ValueError("PBT requires at least two population members")
    names = [member.get("name") for member in population]
    if any(not isinstance(name, str) or not MEMBER_NAME_RE.fullmatch(name) for name in names):
        raise ValueError("Every population member requires a filesystem-safe name")
    if len(set(names)) != len(names):
        raise ValueError("Population member names must be unique")
    if any(float(member.get("start_lr", 0)) <= 0 for member in population):
        raise ValueError("Every population member requires a positive start_lr")

    required_pbt = {
        "metric",
        "mode",
        "exploit_fraction",
        "mutation_factors",
        "min_lr",
        "max_lr",
        "seed",
    }
    missing = sorted(required_pbt - pbt.keys())
    if missing:
        raise ValueError(f"Missing PBT options: {', '.join(missing)}")
    if pbt["metric"] not in {"validation_accuracy", "validation_auc", "validation_loss"}:
        raise ValueError("Unsupported PBT metric")
    if pbt["mode"] not in {"max", "min"}:
        raise ValueError("PBT mode must be 'max' or 'min'")
    fraction = float(pbt["exploit_fraction"])
    if not 0 < fraction <= 0.5:
        raise ValueError("exploit_fraction must be in (0, 0.5]")
    factors = [float(value) for value in pbt["mutation_factors"]]
    if not factors or any(value <= 0 for value in factors):
        raise ValueError("mutation_factors must contain positive values")
    pbt["mutation_factors"] = factors
    pbt["exploit_fraction"] = fraction
    pbt["min_lr"] = float(pbt["min_lr"])
    pbt["max_lr"] = float(pbt["max_lr"])
    if not 0 < pbt["min_lr"] < pbt["max_lr"]:
        raise ValueError("Expected 0 < min_lr < max_lr")
    if any(
        not pbt["min_lr"] <= float(member["start_lr"]) <= pbt["max_lr"]
        for member in population
    ):
        raise ValueError("Population start_lr values must lie within PBT LR bounds")

    integer_options = ("generations", "epochs_per_generation", "samples_per_epoch",
                       "samples_per_epoch_val", "batch_size", "num_workers")
    if any(int(shared[key]) < 1 for key in integer_options):
        raise ValueError("Generation, epoch, sample, batch and worker counts must be positive")
    if shared["lr_scheduler"] != "none":
        raise ValueError("PBT learning-rate mutation requires lr_scheduler: none")

    name = args.experiment_name or experiment.get("name")
    if args.smoke and not args.experiment_name:
        name = f"{name}_smoke"
    if not isinstance(name, str) or not MEMBER_NAME_RE.fullmatch(name):
        raise ValueError("Experiment requires a filesystem-safe name")

    return {
        "schema_version": 1,
        "config_path": str(config_path),
        "experiment_name": name,
        "output_root": absolute_project_path(experiment["output_root"]),
        "shared": shared,
        "gpus": gpus,
        "population": population,
        "pbt": pbt,
        "smoke": args.smoke,
    }


def validate_inputs(config):
    shared = config["shared"]
    files = ("checkpoint", "data_config", "network_config")
    for key in files:
        if not Path(shared[key]).is_file():
            raise FileNotFoundError(f"{key} not found: {shared[key]}")
    dataset = Path(shared["dataset"])
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    patterns = (
        "*_bb_train800k.root",
        "*_cc_train800k.root",
        "*_dd_train800k.root",
        "*_bb_val50k.root",
        "*_cc_val50k.root",
        "*_dd_val50k.root",
    )
    missing = [pattern for pattern in patterns if not any(dataset.glob(pattern))]
    if missing:
        raise FileNotFoundError(f"dataset is missing required samples: {', '.join(missing)}")
    if not (PROJECT_DIR / ".venv/bin/weaver").is_file():
        raise FileNotFoundError("Project Weaver executable is missing")


def contract_fingerprint(config):
    contract = {
        "schema_version": config["schema_version"],
        "shared": config["shared"],
        "population": config["population"],
        "pbt": config["pbt"],
        "smoke": config["smoke"],
    }
    encoded = json.dumps(contract, sort_keys=True).encode()
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def epoch_for_generation(config, generation):
    return (generation + 1) * int(config["shared"]["epochs_per_generation"]) - 1


def make_command(config, member, gpu, member_dir, generation):
    target_epoch = epoch_for_generation(config, generation)
    resume_epoch = None if generation == 0 else target_epoch - int(
        config["shared"]["epochs_per_generation"]
    )
    shared = dict(config["shared"])
    shared.update(
        epochs=target_epoch + 1,
        seed=int(config["shared"]["seed"]) + generation,
        start_lr=member["lr"],
    )
    resolved = {"shared": shared}
    worker = {"name": member["name"], "gpu": gpu, "controller": None}
    log_path = member_dir / f"generation-{generation:03d}.log"
    command = build_command(
        resolved,
        worker,
        member_dir,
        resume_epoch,
        log_path=log_path,
        override_load_lr=resume_epoch is not None,
    )
    return command, log_path, target_epoch


def atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def checkpoint_paths(member_dir, epoch):
    prefix = member_dir / f"net_epoch-{epoch}"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")


def ranking_and_plan(config, generation_record, members):
    metric_name = config["pbt"]["metric"]
    reverse = config["pbt"]["mode"] == "max"
    ranking = sorted(
        members,
        key=lambda name: generation_record["workers"][name]["metrics"][metric_name],
        reverse=reverse,
    )
    count = max(1, math.floor(len(ranking) * config["pbt"]["exploit_fraction"]))
    count = min(count, len(ranking) // 2)
    donors = ranking[:count]
    recipients = ranking[-count:]
    rng = random.Random(int(config["pbt"]["seed"]) + generation_record["index"])
    plan = []
    for index, recipient in enumerate(recipients):
        donor = donors[index % len(donors)]
        factor = rng.choice(config["pbt"]["mutation_factors"])
        old_lr = float(members[recipient]["lr"])
        donor_lr = float(members[donor]["lr"])
        new_lr = min(
            config["pbt"]["max_lr"],
            max(config["pbt"]["min_lr"], donor_lr * factor),
        )
        plan.append(
            {
                "recipient": recipient,
                "donor": donor,
                "recipient_lr": old_lr,
                "donor_lr": donor_lr,
                "mutation_factor": factor,
                "new_lr": new_lr,
                "applied": False,
            }
        )
    return ranking, plan


def apply_exploit(experiment_dir, manifest, generation_record, manifest_path):
    epoch = generation_record["epoch"]
    for event in generation_record["exploit"]:
        if event["applied"]:
            continue
        donor_dir = experiment_dir / event["donor"]
        recipient_dir = experiment_dir / event["recipient"]
        donor_state, donor_optimizer = checkpoint_paths(donor_dir, epoch)
        recipient_state, recipient_optimizer = checkpoint_paths(recipient_dir, epoch)
        if not donor_state.is_file() or not donor_optimizer.is_file():
            raise FileNotFoundError(f"Donor checkpoint is incomplete: {event['donor']}")
        atomic_copy(donor_state, recipient_state)
        atomic_copy(donor_optimizer, recipient_optimizer)
        member = manifest["members"][event["recipient"]]
        member["lr"] = event["new_lr"]
        member["parent"] = event["donor"]
        member["last_exploit_generation"] = generation_record["index"]
        event["applied"] = True
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)


def launch_chunk(config, experiment_dir, manifest, generation_record, names, manifest_path):
    processes = {}
    streams = {}
    for name, gpu in zip(names, config["gpus"]):
        member = manifest["members"][name]
        member_dir = experiment_dir / name
        command, log_path, target_epoch = make_command(
            config, member, gpu, member_dir, generation_record["index"]
        )
        console_path = member_dir / f"generation-{generation_record['index']:03d}.console.log"
        stream = console_path.open("w")
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            stream.close()
            terminate(processes)
            for running_name, running_process in processes.items():
                streams[running_name].close()
                generation_record["workers"][running_name].update(
                    status="terminated",
                    returncode=running_process.poll(),
                    finished_at=utc_now(),
                )
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            raise
        processes[name] = process
        streams[name] = stream
        generation_record["workers"][name].update(
            status="running",
            gpu=gpu,
            pid=process.pid,
            command=command,
            log=str(log_path),
            console_log=str(console_path),
            target_epoch=target_epoch,
            started_at=utc_now(),
        )
        print(f"started generation={generation_record['index']} {name} gpu={gpu} pid={process.pid}")
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)

    failure = None
    try:
        while processes:
            for name, process in list(processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                streams.pop(name).close()
                processes.pop(name)
                record = generation_record["workers"][name]
                metrics = read_metrics(Path(record["log"]))
                metric_name = config["pbt"]["metric"]
                metric_ok = metrics is not None and metrics.get(metric_name) is not None
                status = "completed" if returncode == 0 and metric_ok else "failed"
                record.update(
                    status=status,
                    returncode=returncode,
                    metrics=metrics,
                    finished_at=utc_now(),
                )
                print(
                    f"finished generation={generation_record['index']} "
                    f"{name} returncode={returncode}"
                )
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
                if status == "failed":
                    failure = name
                    break
            if failure:
                break
            time.sleep(0.5)
    except BaseException:
        terminate(processes)
        for name, process in processes.items():
            streams[name].close()
            generation_record["workers"][name].update(
                status="terminated",
                returncode=process.poll(),
                finished_at=utc_now(),
            )
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        raise

    if failure:
        terminate(processes)
        for name, process in processes.items():
            streams[name].close()
            generation_record["workers"][name].update(
                status="terminated",
                returncode=process.poll(),
                finished_at=utc_now(),
            )
        raise RuntimeError(f"PBT worker failed: {failure}")


def initial_manifest(config, fingerprint):
    checkpoint = Path(config["shared"]["checkpoint"])
    return {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "fingerprint": fingerprint,
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "next_generation": 0,
        "config": config,
        "checkpoint": {
            "path": str(checkpoint),
            "resolved_path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        },
        "git": git_metadata(),
        "members": {
            member["name"]: {
                "name": member["name"],
                "lr": float(member["start_lr"]),
                "parent": None,
            }
            for member in config["population"]
        },
        "generations": [],
    }


def run(args):
    config = load_config(args)
    validate_inputs(config)
    fingerprint = contract_fingerprint(config)
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    manifest_path = experiment_dir / MANIFEST_NAME

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False).rstrip())
        for index, member_config in enumerate(config["population"]):
            member = {"name": member_config["name"], "lr": float(member_config["start_lr"])}
            command, _, _ = make_command(
                config,
                member,
                config["gpus"][index % len(config["gpus"])],
                experiment_dir / member["name"],
                0,
            )
            print(f"[{member['name']}] {shlex.join(command)}")
        return 0

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resolved PBT contract differs from the saved manifest")
        if manifest["status"] == "completed":
            print(f"already completed: {experiment_dir}")
            return 0
        manifest["status"] = "running"
        manifest.pop("failure", None)
        manifest["updated_at"] = utc_now()
    else:
        if experiment_dir.exists():
            raise FileExistsError(f"Experiment already exists: {experiment_dir}")
        experiment_dir.mkdir(parents=True)
        manifest = initial_manifest(config, fingerprint)

    for name in manifest["members"]:
        (experiment_dir / name).mkdir(parents=True, exist_ok=True)
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    atomic_json(manifest_path, manifest)

    try:
        for generation in range(
            int(manifest["next_generation"]), int(config["shared"]["generations"])
        ):
            existing = next(
                (item for item in manifest["generations"] if item["index"] == generation),
                None,
            )
            if existing is None:
                existing = {
                    "index": generation,
                    "epoch": epoch_for_generation(config, generation),
                    "seed": int(config["shared"]["seed"]) + generation,
                    "status": "training",
                    "workers": {
                        name: {"status": "pending"} for name in manifest["members"]
                    },
                    "ranking": None,
                    "exploit": None,
                    "started_at": utc_now(),
                }
                manifest["generations"].append(existing)
                atomic_json(manifest_path, manifest)
            existing["status"] = "training"
            pending = [
                name
                for name, record in existing["workers"].items()
                if record["status"] != "completed"
            ]
            for offset in range(0, len(pending), len(config["gpus"])):
                launch_chunk(
                    config,
                    experiment_dir,
                    manifest,
                    existing,
                    pending[offset : offset + len(config["gpus"])],
                    manifest_path,
                )

            if existing["exploit"] is None:
                ranking, plan = ranking_and_plan(
                    config, existing, manifest["members"]
                )
                existing["ranking"] = ranking
                existing["exploit"] = (
                    [] if generation == int(config["shared"]["generations"]) - 1 else plan
                )
                existing["status"] = "exploiting"
                atomic_json(manifest_path, manifest)

            apply_exploit(experiment_dir, manifest, existing, manifest_path)
            existing["status"] = "completed"
            existing["finished_at"] = utc_now()
            manifest["next_generation"] = generation + 1
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)

        manifest["status"] = "completed"
        manifest.pop("failure", None)
        manifest["finished_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(f"completed: {experiment_dir}")
        return 0
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        raise


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

#!/usr/bin/env python3
"""Launch reproducible, independent Weaver training workers in parallel."""

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from training.runtime import (
    PROJECT_DIR,
    atomic_json,
    data_command_args,
    data_paths,
    git_metadata,
    normalize_data_extension,
    project_path,
    read_metrics,
    required_sample_patterns,
    sha256,
    terminate,
    utc_now,
    validate_dataset,
    weaver_executable,
)
from training.weaver import build_command, latest_resumable_epoch


DEFAULT_CONFIG = PROJECT_DIR / "configs/experiments/parallel_baseline_vs_controller.yaml"
MANIFEST_NAME = "manifest.json"
# (Weaver input name, jet-flavor file stem)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU assignment in worker order, e.g. 0,2",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override the budget with 7680 train and 3000 validation samples",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume workers from their latest complete epoch checkpoints",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_and_resolve(args):
    config_path = args.config.resolve()
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 experiment configuration")

    experiment = payload.get("experiment")
    shared = payload.get("shared")
    workers = payload.get("workers")
    if not isinstance(experiment, dict) or not isinstance(shared, dict):
        raise ValueError("Configuration requires experiment and shared mappings")
    if not isinstance(workers, list) or len(workers) < 2:
        raise ValueError("Configuration requires at least two workers")

    required_shared = {
        "dataset",
        "checkpoint",
        "data_config",
        "network_config",
        "seed",
        "epochs",
        "samples_per_epoch",
        "samples_per_epoch_val",
        "batch_size",
        "start_lr",
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

    resolved_shared = dict(shared)
    for key in ("dataset", "data_config", "network_config"):
        resolved_shared[key] = str(project_path(shared[key]).resolve())
    # Keep the project-local symlink in commands. The manifest records its
    # resolved target and content checksum separately.
    resolved_shared["checkpoint"] = str(project_path(shared["checkpoint"]).absolute())

    if args.smoke:
        resolved_shared["epochs"] = 1
        resolved_shared["samples_per_epoch"] = 7680
        resolved_shared["samples_per_epoch_val"] = 3000
    resolved_shared["data_extension"] = normalize_data_extension(
        resolved_shared.get("data_extension", "root")
    )

    resolved_workers = []
    override_gpus = args.gpus.split(",") if args.gpus else None
    if override_gpus and len(override_gpus) != len(workers):
        raise ValueError("--gpus must provide exactly one GPU per configured worker")

    for index, worker in enumerate(workers):
        if not isinstance(worker, dict) or not worker.get("name"):
            raise ValueError("Every worker requires a non-empty name")
        resolved = dict(worker)
        resolved["gpu"] = override_gpus[index] if override_gpus else str(worker["gpu"])
        controller = worker.get("controller")
        resolved["controller"] = (
            None if controller is None else str(project_path(controller).resolve())
        )
        if "start_lr" in worker:
            resolved["start_lr"] = float(worker["start_lr"])
        if "seed" in worker:
            resolved["seed"] = int(worker["seed"])
        resolved_workers.append(resolved)

    names = [worker["name"] for worker in resolved_workers]
    gpus = [worker["gpu"] for worker in resolved_workers]
    if len(set(names)) != len(names):
        raise ValueError("Worker names must be unique")
    if len(set(gpus)) != len(gpus):
        raise ValueError("Parallel workers must use different GPUs")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
        raise ValueError("Worker names may contain only letters, digits, '.', '_' and '-'")

    base_name = args.experiment_name or experiment.get("name")
    if not base_name:
        raise ValueError("Experiment name is required")
    experiment_name = base_name
    if args.smoke and not args.experiment_name and not str(experiment_name).endswith("_smoke"):
        experiment_name = f"{experiment_name}_smoke"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment_name):
        raise ValueError("Experiment name contains unsupported characters")

    output_root = project_path(experiment["output_root"]).resolve()
    return {
        "schema_version": 1,
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "output_root": str(output_root),
        "shared": resolved_shared,
        "workers": resolved_workers,
        "smoke": args.smoke,
    }


def validate_files(resolved):
    shared = resolved["shared"]
    weaver = weaver_executable()
    if not weaver.is_file():
        raise FileNotFoundError(f"Weaver executable not found: {weaver}")
    required_files = ("checkpoint", "data_config", "network_config")
    for key in required_files:
        path = Path(shared[key])
        if not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")
    validate_dataset(shared["dataset"], shared.get("data_extension", "root"))
    for worker in resolved["workers"]:
        if worker["controller"] and not Path(worker["controller"]).is_file():
            raise FileNotFoundError(
                f"controller not found for {worker['name']}: {worker['controller']}"
            )
    if int(shared["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    if int(shared["samples_per_epoch"]) < 1 or int(shared["samples_per_epoch_val"]) < 1:
        raise ValueError("sample budgets must be positive")


def fingerprint(resolved):
    reproducibility_contract = {
        "schema_version": resolved["schema_version"],
        "shared": resolved["shared"],
        "workers": [
            {
                "name": worker["name"],
                "controller": worker["controller"],
                "start_lr": worker.get("start_lr"),
                "seed": worker.get("seed"),
            }
            for worker in resolved["workers"]
        ],
    }
    encoded = json.dumps(reproducibility_contract, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    args = parse_args()
    resolved = load_and_resolve(args)
    validate_files(resolved)
    run_fingerprint = fingerprint(resolved)
    experiment_dir = Path(resolved["output_root"]) / resolved["experiment_name"]
    manifest_path = experiment_dir / MANIFEST_NAME

    worker_specs = []
    for worker in resolved["workers"]:
        worker_dir = experiment_dir / worker["name"]
        resume_epoch = (
            latest_resumable_epoch(worker_dir, bool(worker["controller"]))
            if args.resume
            else None
        )
        command = build_command(resolved, worker, worker_dir, resume_epoch)
        worker_specs.append((worker, worker_dir, resume_epoch, command))

    if args.dry_run:
        for worker, worker_dir, resume_epoch, command in worker_specs:
            print(f"[{worker['name']}] gpu={worker['gpu']} resume_epoch={resume_epoch}")
            print(shlex.join(command))
            print(f"output: {worker_dir}")
        return 0

    if experiment_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}. "
            "Choose another name or use --resume."
        )
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        previous = json.loads(manifest_path.read_text())
        if previous.get("fingerprint") != run_fingerprint:
            raise ValueError("Resolved training contract differs from the saved manifest")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )
    checkpoint = Path(resolved["shared"]["checkpoint"])
    manifest = {
        "schema_version": 1,
        "experiment": resolved["experiment_name"],
        "fingerprint": run_fingerprint,
        "status": "starting",
        "attempt": previous.get("attempt", 1) + 1 if args.resume else 1,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "config": resolved,
        "checkpoint": {
            "path": str(checkpoint),
            "resolved_path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        },
        "git": git_metadata(),
        "workers": {},
    }
    for worker, worker_dir, resume_epoch, command in worker_specs:
        worker_dir.mkdir(parents=True, exist_ok=True)
        manifest["workers"][worker["name"]] = {
            "gpu": worker["gpu"],
            "controller": worker["controller"],
            "directory": str(worker_dir),
            "command": command,
            "resume_epoch": resume_epoch,
            "status": "pending",
        }
    atomic_json(manifest_path, manifest)

    processes = {}
    streams = {}
    interrupted = False

    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    old_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for worker, worker_dir, resume_epoch, command in worker_specs:
            if resume_epoch is not None and resume_epoch >= int(resolved["shared"]["epochs"]) - 1:
                manifest["workers"][worker["name"]]["status"] = "already_complete"
                manifest["workers"][worker["name"]]["metrics"] = read_metrics(worker_dir)
                continue
            stream = (worker_dir / "console.log").open("a" if args.resume else "w")
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
                raise
            processes[worker["name"]] = process
            streams[worker["name"]] = stream
            manifest["workers"][worker["name"]].update(
                status="running", pid=process.pid, started_at=utc_now()
            )
            print(
                f"started {worker['name']}: pid={process.pid} gpu={worker['gpu']} "
                f"log={worker_dir / 'console.log'}"
            )
        manifest["status"] = "running"
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)

        failure = None
        while processes:
            if interrupted:
                failure = ("launcher", 128)
                break
            for name, process in list(processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                streams[name].close()
                streams.pop(name)
                processes.pop(name)
                worker_dir = Path(manifest["workers"][name]["directory"])
                manifest["workers"][name].update(
                    status="completed" if returncode == 0 else "failed",
                    returncode=returncode,
                    finished_at=utc_now(),
                    metrics=read_metrics(worker_dir),
                )
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
                print(f"finished {name}: returncode={returncode}")
                if returncode != 0:
                    failure = (name, returncode)
                    break
            if failure:
                break
            time.sleep(0.5)

        if failure:
            terminate(processes)
            for name, process in processes.items():
                streams[name].close()
                manifest["workers"][name].update(
                    status="terminated",
                    returncode=process.poll(),
                    finished_at=utc_now(),
                )
            manifest["status"] = "interrupted" if interrupted else "failed"
            manifest["failure"] = {"worker": failure[0], "returncode": failure[1]}
            manifest["finished_at"] = utc_now()
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            return 130 if interrupted else 1

        manifest["status"] = "completed"
        manifest["finished_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        print(f"completed: {experiment_dir}")
        return 0
    except Exception as error:
        terminate(processes)
        for name, process in processes.items():
            manifest["workers"][name].update(
                status="terminated" if process.poll() is not None else "failed",
                returncode=process.poll(),
                finished_at=utc_now(),
            )
        manifest["status"] = "failed"
        manifest["failure"] = {
            "worker": "launcher",
            "error": f"{type(error).__name__}: {error}",
        }
        manifest["finished_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        raise
    finally:
        terminate(processes)
        for stream in streams.values():
            stream.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
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
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from plot_pbt_summary import plot_manifest
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


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def log_event(log_path, message):
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-name")
    parser.add_argument("--gpus", help="Comma-separated GPU slots, e.g. 0,2")
    parser.add_argument(
        "--slots",
        help=(
            "Comma-separated host:gpu slots for multi-node runs. "
            "Use host:gpu@venv to override the Python environment on a host, "
            "e.g. iutgpu01:6@.venv-iutgpu01,iutgpu05:4"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two members, two generations and small train/validation budgets",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def local_hostnames():
    names = {"localhost", "127.0.0.1"}
    for value in (socket.gethostname(), socket.getfqdn()):
        if value:
            names.add(value)
            names.add(value.split(".")[0])
    return names


def parse_slots(args, resources):
    if args.gpus and args.slots:
        raise ValueError("Use either --gpus or --slots, not both")

    if args.slots:
        raw_slots = [slot.strip() for slot in args.slots.split(",")]
        slots = []
        for raw in raw_slots:
            if not raw:
                raise ValueError("--slots contains an empty slot")
            slot_part, _, venv_part = raw.partition("@")
            if ":" not in slot_part:
                raise ValueError(f"Expected host:gpu slot, got: {raw}")
            host, gpu = slot_part.rsplit(":", 1)
            host = host.strip()
            gpu = gpu.strip()
            venv = venv_part.strip() or None
            if not host or not gpu:
                raise ValueError(f"Expected host:gpu slot, got: {raw}")
            label = f"{host}:{gpu}" + (f"@{venv}" if venv else "")
            slots.append({"host": host, "gpu": gpu, "venv": venv, "label": label})
    else:
        raw_gpus = args.gpus.split(",") if args.gpus else [str(gpu) for gpu in resources.get("gpus", [])]
        slots = []
        for raw in raw_gpus:
            gpu = raw.strip()
            if not gpu:
                raise ValueError("GPU slots must be non-empty")
            slots.append({"host": None, "gpu": gpu, "label": gpu})

    if not slots:
        raise ValueError("At least one GPU slot is required")
    labels = [slot["label"] for slot in slots]
    if len(set(labels)) != len(labels):
        raise ValueError("GPU slots must be unique")
    return slots


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
    if shared.get("training_controller"):
        shared["training_controller"] = absolute_project_path(shared["training_controller"])
    shared["checkpoint"] = absolute_project_path(shared["checkpoint"], resolve=False)

    slots = parse_slots(args, resources)

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
    if pbt["metric"] not in {
        "validation_accuracy",
        "validation_auc",
        "validation_loss",
        "validation_bkg_rejection_bc_score",
        "validation_bkg_rejection_bd_score",
        "validation_bkg_rejection_cb_score",
        "validation_bkg_rejection_cd_score",
        "validation_b_tag_rejection_score",
        "validation_c_tag_rejection_score",
        "validation_bkg_rejection_score",
    }:
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
    pbt["degradation_tolerance"] = float(pbt.get("degradation_tolerance", 0.02))
    pbt["degradation_window"] = int(pbt.get("degradation_window", 3))
    pbt["rollback_fraction"] = float(pbt.get("rollback_fraction", 0.0))
    pbt["controller_state_on_exploit"] = pbt.get("controller_state_on_exploit", "copy")
    if not 0 < pbt["min_lr"] < pbt["max_lr"]:
        raise ValueError("Expected 0 < min_lr < max_lr")
    if not 0 <= pbt["degradation_tolerance"] < 1:
        raise ValueError("degradation_tolerance must be in [0, 1)")
    if pbt["degradation_window"] < 1:
        raise ValueError("degradation_window must be positive")
    if not 0 <= pbt["rollback_fraction"] <= 0.5:
        raise ValueError("rollback_fraction must be in [0, 0.5]")
    if pbt["controller_state_on_exploit"] not in {"copy", "reset"}:
        raise ValueError("controller_state_on_exploit must be 'copy' or 'reset'")
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
        "gpus": [slot["gpu"] for slot in slots],
        "slots": slots,
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
    if shared.get("training_controller") and not Path(shared["training_controller"]).is_file():
        raise FileNotFoundError(
            f"training_controller not found: {shared['training_controller']}"
        )
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


def slot_label(slot):
    return slot["label"] if isinstance(slot, dict) else str(slot)


def remote_host(slot):
    if not isinstance(slot, dict):
        return None
    host = slot.get("host")
    if not host or host in local_hostnames():
        return None
    return host


def wrap_remote_command(command, slot):
    host = remote_host(slot)
    if not host:
        return command
    if command and Path(command[0]).name == "weaver":
        venv_root = project_path(slot.get("venv") or ".venv")
        venv_python = venv_root / "bin/python"
        venv_weaver = venv_root / "bin/weaver"
        py310_site = venv_root / "lib/python3.10/site-packages"
        py312_site = venv_root / "lib/python3.12/site-packages"
        args = shlex.join(command[1:])
        remote = (
            f"cd {shlex.quote(str(PROJECT_DIR))} && "
            f"if [ -x {shlex.quote(str(venv_python))} ]; then "
            f"exec {shlex.quote(str(venv_python))} {shlex.quote(str(venv_weaver))} {args}; "
            f"elif command -v python3.10 >/dev/null 2>&1; then "
            f"export PYTHONPATH={shlex.quote(str(py310_site))}:"
            f"{shlex.quote(str(PROJECT_DIR / 'weaver-core'))}:"
            f"${{PYTHONPATH:-}}; "
            f"exec python3.10 {shlex.quote(str(venv_weaver))} {args}; "
            f"elif command -v python3.12 >/dev/null 2>&1 && [ -d {shlex.quote(str(py312_site))} ]; then "
            f"export PYTHONPATH={shlex.quote(str(py312_site))}:"
            f"{shlex.quote(str(PROJECT_DIR / 'weaver-core'))}:"
            f"${{PYTHONPATH:-}}; "
            f"exec python3.12 {shlex.quote(str(venv_weaver))} {args}; "
            f"else "
            f"echo 'remote venv python is not executable on this host: "
            f"{shlex.quote(str(venv_python))}' >&2; "
            f"echo 'Create a compatible host-local venv and pass it as host:gpu@venv.' >&2; "
            f"exit 127; "
            f"fi"
        )
    else:
        remote = f"cd {shlex.quote(str(PROJECT_DIR))} && exec {shlex.join(command)}"
    return ["ssh", host, remote]


def make_command(config, member, slot, member_dir, generation):
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
    worker = {
        "name": member["name"],
        "gpu": slot["gpu"] if isinstance(slot, dict) else str(slot),
        "controller": shared.get("training_controller"),
    }
    log_path = member_dir / f"generation-{generation:03d}.log"
    command = build_command(
        resolved,
        worker,
        member_dir,
        resume_epoch,
        log_path=log_path,
        override_load_lr=resume_epoch is not None,
    )
    return wrap_remote_command(command, slot), log_path, target_epoch


def atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def checkpoint_paths(member_dir, epoch):
    prefix = member_dir / f"net_epoch-{epoch}"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")


def controller_checkpoint_path(member_dir, epoch):
    return member_dir / f"net_epoch-{epoch}_controller.pt"


def global_best_paths(experiment_dir):
    return {
        "state_path": str(experiment_dir / "global_best_state.pt"),
        "optimizer_path": str(experiment_dir / "global_best_optimizer.pt"),
        "controller_path": str(experiment_dir / "global_best_controller.pt"),
        "metadata_path": str(experiment_dir / "global_best_metadata.json"),
    }


def metric_is_better(config, candidate, incumbent):
    if incumbent is None:
        return True
    if config["pbt"]["mode"] == "max":
        return candidate > incumbent
    return candidate < incumbent


def metric_has_degraded(config, current, best):
    if best is None:
        return False
    tolerance = config["pbt"]["degradation_tolerance"]
    if config["pbt"]["mode"] == "max":
        return current < best * (1 - tolerance)
    return current > best * (1 + tolerance)


def relative_to_best(config, current, best):
    if best in (None, 0):
        return None
    if config["pbt"]["mode"] == "max":
        return current / best - 1
    return best / current - 1


def best_worker_in_generation(config, generation_record):
    metric_name = config["pbt"]["metric"]
    ranking = generation_record.get("ranking")
    if not ranking:
        reverse = config["pbt"]["mode"] == "max"
        ranking = sorted(
            generation_record["workers"],
            key=lambda name: generation_record["workers"][name]["metrics"][metric_name],
            reverse=reverse,
        )
    name = ranking[0]
    metrics = generation_record["workers"][name]["metrics"]
    return name, metrics[metric_name], metrics


def update_global_best(experiment_dir, manifest, generation_record, manifest_path):
    config = manifest["config"]
    metric_name = config["pbt"]["metric"]
    member_name, value, metrics = best_worker_in_generation(config, generation_record)
    current_best = manifest.get("best")
    if not metric_is_better(
        config,
        value,
        None if current_best is None else current_best["metric_value"],
    ):
        return False

    member_dir = experiment_dir / member_name
    state_path, optimizer_path = checkpoint_paths(member_dir, generation_record["epoch"])
    if not state_path.is_file() or not optimizer_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is incomplete: {member_name}")

    paths = global_best_paths(experiment_dir)
    atomic_copy(state_path, Path(paths["state_path"]))
    atomic_copy(optimizer_path, Path(paths["optimizer_path"]))

    controller_source = controller_checkpoint_path(member_dir, generation_record["epoch"])
    controller_path = Path(paths["controller_path"])
    has_controller = bool(config["shared"].get("training_controller")) and controller_source.is_file()
    if has_controller:
        atomic_copy(controller_source, controller_path)
    elif controller_path.exists():
        controller_path.unlink()

    best_record = {
        "generation": generation_record["index"],
        "epoch": generation_record["epoch"],
        "member": member_name,
        "metric": metric_name,
        "metric_value": value,
        "lr": float(manifest["members"][member_name]["lr"]),
        "metrics": metrics,
        "updated_at": utc_now(),
        "source_state_path": str(state_path),
        "source_optimizer_path": str(optimizer_path),
        "source_controller_path": str(controller_source) if has_controller else None,
        **paths,
    }
    manifest["best"] = best_record
    atomic_json(Path(paths["metadata_path"]), best_record)
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    return True


def update_generation_health(config, manifest, generation_record):
    member_name, value, _ = best_worker_in_generation(config, generation_record)
    best = manifest.get("best")
    best_value = None if best is None else best["metric_value"]
    degraded = metric_has_degraded(config, value, best_value)
    previous = [
        item
        for item in manifest.get("generations", [])
        if item["index"] < generation_record["index"]
    ]
    consecutive = 1 if degraded else 0
    if degraded:
        for item in reversed(previous):
            health = item.get("health") or {}
            if not health.get("degraded"):
                break
            consecutive += 1
    generation_record["health"] = {
        "current_best_member": member_name,
        "current_best_metric": value,
        "global_best_member": None if best is None else best["member"],
        "global_best_generation": None if best is None else best["generation"],
        "global_best_metric": best_value,
        "relative_to_global_best": relative_to_best(config, value, best_value),
        "degraded": degraded,
        "consecutive_degraded_generations": consecutive,
        "status": "degraded" if consecutive >= config["pbt"]["degradation_window"] else "ok",
        "member_lrs": {
            name: float(record["lr"])
            for name, record in manifest.get("members", {}).items()
        },
    }
    return generation_record["health"]


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


def add_global_best_rollbacks(config, manifest, generation_record, members, plan):
    best = manifest.get("best")
    if not best or generation_record["index"] == int(config["shared"]["generations"]) - 1:
        return plan
    health = generation_record.get("health") or {}
    if health.get("status") != "degraded":
        return plan
    count = math.floor(len(generation_record["ranking"]) * config["pbt"]["rollback_fraction"])
    count = min(max(0, count), len(generation_record["ranking"]) // 2)
    if count == 0:
        return plan

    recipients = generation_record["ranking"][-count:]
    filtered = [event for event in plan if event["recipient"] not in recipients]
    for recipient in recipients:
        filtered.append(
            {
                "source": "global_best",
                "recipient": recipient,
                "donor": best["member"],
                "recipient_lr": float(members[recipient]["lr"]),
                "donor_lr": float(best["lr"]),
                "mutation_factor": 1.0,
                "new_lr": float(best["lr"]),
                "applied": False,
                "reason": "rollback_from_global_best",
                "global_best_generation": best["generation"],
            }
        )
    return filtered


def apply_exploit(experiment_dir, manifest, generation_record, manifest_path):
    epoch = generation_record["epoch"]
    for event in generation_record["exploit"]:
        if event["applied"]:
            continue
        recipient_dir = experiment_dir / event["recipient"]
        if event.get("source") == "global_best":
            paths = global_best_paths(experiment_dir)
            donor_state = Path(paths["state_path"])
            donor_optimizer = Path(paths["optimizer_path"])
            donor_controller = Path(paths["controller_path"])
        else:
            donor_dir = experiment_dir / event["donor"]
            donor_state, donor_optimizer = checkpoint_paths(donor_dir, epoch)
            donor_controller = controller_checkpoint_path(donor_dir, epoch)
        recipient_state, recipient_optimizer = checkpoint_paths(recipient_dir, epoch)
        if not donor_state.is_file() or not donor_optimizer.is_file():
            raise FileNotFoundError(f"Donor checkpoint is incomplete: {event['donor']}")
        atomic_copy(donor_state, recipient_state)
        atomic_copy(donor_optimizer, recipient_optimizer)
        shared_config = manifest.get("config", {}).get("shared", {})
        pbt_config = manifest.get("config", {}).get("pbt", {})
        if shared_config.get("training_controller"):
            recipient_controller = controller_checkpoint_path(recipient_dir, epoch)
            if pbt_config.get("controller_state_on_exploit", "copy") == "reset":
                if recipient_controller.exists():
                    recipient_controller.unlink()
            else:
                if not donor_controller.is_file():
                    raise FileNotFoundError(
                        f"Donor controller checkpoint is incomplete: {event['donor']}"
                    )
                atomic_copy(donor_controller, recipient_controller)
        member = manifest["members"][event["recipient"]]
        member["lr"] = event["new_lr"]
        member["parent"] = event["donor"]
        member["parent_source"] = event.get("source", "population")
        member["last_exploit_generation"] = generation_record["index"]
        event["applied"] = True
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)


def launch_workers(config, experiment_dir, manifest, generation_record, names, manifest_path):
    processes = {}
    streams = {}
    started_monotonic = {}
    process_slots = {}
    pbt_log_path = manifest_path.with_name("pbt.log")
    pending_names = list(names)
    free_slots = list(config["slots"])

    def start_worker(name, slot):
        member = manifest["members"][name]
        member_dir = experiment_dir / name
        command, log_path, target_epoch = make_command(
            config, member, slot, member_dir, generation_record["index"]
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
        process_slots[name] = slot
        started_monotonic[name] = time.monotonic()
        generation_record["workers"][name].update(
            status="running",
            gpu=slot["gpu"],
            host=slot.get("host") if isinstance(slot, dict) else None,
            slot=slot_label(slot),
            pid=process.pid,
            command=command,
            log=str(log_path),
            console_log=str(console_path),
            target_epoch=target_epoch,
            started_at=utc_now(),
        )
        log_event(
            pbt_log_path,
            f"started generation={generation_record['index']} worker={name} "
            f"slot={slot_label(slot)} pid={process.pid}",
        )

    while pending_names and free_slots:
        start_worker(pending_names.pop(0), free_slots.pop(0))
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)

    failure = None
    try:
        while processes or pending_names:
            for name, process in list(processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                streams.pop(name).close()
                processes.pop(name)
                finished_slot = process_slots.pop(name)
                elapsed = format_duration(time.monotonic() - started_monotonic.pop(name))
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
                log_event(
                    pbt_log_path,
                    f"finished generation={generation_record['index']} "
                    f"worker={name} returncode={returncode} elapsed={elapsed}",
                )
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
                if status == "failed":
                    failure = name
                    break
                if pending_names:
                    start_worker(pending_names.pop(0), finished_slot)
                    manifest["updated_at"] = utc_now()
                    atomic_json(manifest_path, manifest)
                else:
                    free_slots.append(finished_slot)
            if failure:
                break
            time.sleep(0.5)
    except BaseException:
        terminate(processes)
        for name, process in processes.items():
            streams[name].close()
            elapsed = format_duration(time.monotonic() - started_monotonic.get(name, time.monotonic()))
            generation_record["workers"][name].update(
                status="terminated",
                returncode=process.poll(),
                finished_at=utc_now(),
            )
            log_event(
                pbt_log_path,
                f"terminated generation={generation_record['index']} "
                f"worker={name} returncode={process.poll()} elapsed={elapsed}",
            )
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        raise

    if failure:
        terminate(processes)
        for name, process in processes.items():
            streams[name].close()
            elapsed = format_duration(time.monotonic() - started_monotonic.get(name, time.monotonic()))
            generation_record["workers"][name].update(
                status="terminated",
                returncode=process.poll(),
                finished_at=utc_now(),
            )
            log_event(
                pbt_log_path,
                f"terminated generation={generation_record['index']} "
                f"worker={name} returncode={process.poll()} elapsed={elapsed}",
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
        "best": None,
        "generations": [],
    }


def run(args):
    config = load_config(args)
    validate_inputs(config)
    fingerprint = contract_fingerprint(config)
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    manifest_path = experiment_dir / MANIFEST_NAME
    pbt_log_path = manifest_path.with_name("pbt.log")

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False).rstrip())
        for index, member_config in enumerate(config["population"]):
            member = {"name": member_config["name"], "lr": float(member_config["start_lr"])}
            command, _, _ = make_command(
                config,
                member,
                config["slots"][index % len(config["slots"])],
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
    run_started_monotonic = time.monotonic()

    for name in manifest["members"]:
        (experiment_dir / name).mkdir(parents=True, exist_ok=True)
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    atomic_json(manifest_path, manifest)
    log_event(
        pbt_log_path,
        f"run started experiment={config['experiment_name']} "
        f"slots={','.join(slot_label(slot) for slot in config['slots'])} "
        f"members={len(manifest['members'])} "
        f"batch_size={config['shared']['batch_size']} metric={config['pbt']['metric']}",
    )

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
            launch_workers(
                config,
                experiment_dir,
                manifest,
                existing,
                pending,
                manifest_path,
            )

            if existing["exploit"] is None:
                ranking, plan = ranking_and_plan(
                    config, existing, manifest["members"]
                )
                existing["ranking"] = ranking
                improved = update_global_best(
                    experiment_dir, manifest, existing, manifest_path
                )
                health = update_generation_health(config, manifest, existing)
                plan = add_global_best_rollbacks(
                    config, manifest, existing, manifest["members"], plan
                )
                existing["exploit"] = (
                    [] if generation == int(config["shared"]["generations"]) - 1 else plan
                )
                existing["status"] = "exploiting"
                best = manifest.get("best") or {}
                log_event(
                    pbt_log_path,
                    "health generation=%d current_best=%s %.6g global_best=%s %.6g "
                    "delta=%s degraded=%s consecutive=%d improved=%s lrs=%s"
                    % (
                        existing["index"],
                        health["current_best_member"],
                        health["current_best_metric"],
                        best.get("member"),
                        best.get("metric_value", float("nan")),
                        (
                            "n/a"
                            if health["relative_to_global_best"] is None
                            else f"{health['relative_to_global_best']:+.3%}"
                        ),
                        health["status"],
                        health["consecutive_degraded_generations"],
                        improved,
                        ",".join(
                            f"{name}:{lr:.3g}"
                            for name, lr in sorted(health["member_lrs"].items())
                        ),
                    ),
                )
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
        try:
            plot_path = plot_manifest(manifest_path)
            manifest["summary_plot"] = str(plot_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"summary plot: {plot_path}")
        except Exception as plot_error:
            log_event(pbt_log_path, f"warning: failed to create PBT summary plot: {plot_error}")
        log_event(
            pbt_log_path,
            f"run completed experiment={config['experiment_name']} "
            f"elapsed={format_duration(time.monotonic() - run_started_monotonic)} "
            f"directory={experiment_dir} "
            f"recommended_checkpoint={(manifest.get('best') or {}).get('state_path')}",
        )
        return 0
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        log_event(
            pbt_log_path,
            f"run {manifest['status']} experiment={config['experiment_name']} "
            f"elapsed={format_duration(time.monotonic() - run_started_monotonic)} "
            f"error={type(error).__name__}: {error}",
        )
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

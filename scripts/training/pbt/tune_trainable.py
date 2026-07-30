#!/usr/bin/env python3
"""Ray Tune trainable adapter for PBT Weaver trials."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parents[3]

SCRIPTS_DIR = PROJECT_DIR / "scripts"
WEAVER_CORE_DIR = PROJECT_DIR / "weaver-core"
for path in (SCRIPTS_DIR, WEAVER_CORE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

TUNE_METADATA_NAME = "metadata.json"
TUNE_STATE_NAME = "net_state.pt"
TUNE_OPTIMIZER_NAME = "net_optimizer.pt"
TUNE_CONTROLLER_NAME = "net_controller.pt"
TUNE_EVENTS_NAME = "tune_trial_events.jsonl"
TUNE_STATUS_NAME = "tune_trial_status.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def record_trial_event(member_dir, event, **payload):
    member_dir = Path(member_dir)
    member_dir.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), "event": event}
    record.update(payload)
    with (member_dir / TUNE_EVENTS_NAME).open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    atomic_json(member_dir / TUNE_STATUS_NAME, record)
    print(f"[ray-tune-weaver] {event}: {json.dumps(payload, sort_keys=True)}", flush=True)


def checkpoint_paths(member_dir, epoch):
    prefix = Path(member_dir) / f"net_epoch-{epoch}"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")


def controller_checkpoint_path(member_dir, epoch):
    return Path(member_dir) / f"net_epoch-{epoch}_controller.pt"


def _ray_tune_import():
    try:
        from ray import tune
    except ImportError as error:
        raise RuntimeError("Ray Tune runner requires ray[tune]. Install project requirements first.") from error
    return tune, tune.Checkpoint


def _copy_if_exists(source, destination):
    source = Path(source)
    if source.is_file():
        shutil.copy2(source, destination)
        return str(destination)
    return None


def package_tune_checkpoint(member_dir, epoch, checkpoint_dir, metadata=None):
    """Copy Weaver epoch artifacts into a portable Ray Tune checkpoint directory."""
    member_dir = Path(member_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path, optimizer_path = checkpoint_paths(member_dir, epoch)
    if not state_path.is_file():
        raise FileNotFoundError(f"state checkpoint not found: {state_path}")
    if not optimizer_path.is_file():
        raise FileNotFoundError(f"optimizer checkpoint not found: {optimizer_path}")

    shutil.copy2(state_path, checkpoint_dir / TUNE_STATE_NAME)
    shutil.copy2(optimizer_path, checkpoint_dir / TUNE_OPTIMIZER_NAME)
    controller_source = controller_checkpoint_path(member_dir, epoch)
    controller_copy = _copy_if_exists(controller_source, checkpoint_dir / TUNE_CONTROLLER_NAME)
    payload = {
        "schema_version": 1,
        "epoch": int(epoch),
        "member_dir": str(member_dir),
        "created_at": utc_now(),
        "has_controller": controller_copy is not None,
    }
    payload.update(metadata or {})
    atomic_json(checkpoint_dir / TUNE_METADATA_NAME, payload)
    return payload


def restore_tune_checkpoint(checkpoint, member_dir):
    """Restore a Ray Tune checkpoint into the Weaver epoch naming convention."""
    member_dir = Path(member_dir)
    member_dir.mkdir(parents=True, exist_ok=True)
    with checkpoint.as_directory() as checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        metadata_path = checkpoint_dir / TUNE_METADATA_NAME
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Ray Tune checkpoint metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        epoch = int(metadata["epoch"])
        state_path, optimizer_path = checkpoint_paths(member_dir, epoch)
        shutil.copy2(checkpoint_dir / TUNE_STATE_NAME, state_path)
        shutil.copy2(checkpoint_dir / TUNE_OPTIMIZER_NAME, optimizer_path)
        controller_source = checkpoint_dir / TUNE_CONTROLLER_NAME
        if controller_source.is_file():
            shutil.copy2(controller_source, controller_checkpoint_path(member_dir, epoch))
        return metadata


def tune_member_dir(config, member_name):
    return Path(config["output_root"]) / f"{config['experiment_name']}_tune" / member_name


def tune_trial_slot():
    # Ray sets CUDA_VISIBLE_DEVICES for the trial when GPU resources are used.
    # Weaver should therefore address the first visible device inside that trial.
    return {"host": None, "gpu": "0", "label": "ray:gpu0"}


def scalar_report_metrics(metrics):
    scalars = {}
    for key, value in (metrics or {}).items():
        if value is None or isinstance(value, (int, float, bool)):
            scalars[key] = value
    return scalars


def run_weaver_trial_direct(tune_config, checkpoint_root=None):
    """Run one Tune-style Weaver trial in the current process for diagnostics."""
    config = config_from_tune_payload(tune_config)
    trial = dict(tune_config["trial"])
    member_name = trial["member_name"]
    member = {"name": member_name, "lr": float(trial["lr"])}
    member_dir = Path(trial.get("member_dir") or tune_member_dir(config, member_name))
    member_dir.mkdir(parents=True, exist_ok=True)
    record_trial_event(member_dir, "direct_trial_start", member=member_name, lr=member["lr"])

    from training.pbt.checkpointing import bootstrap_initial_checkpoint

    record_trial_event(member_dir, "bootstrap_start")
    bootstrap_initial_checkpoint(config, member_dir)
    record_trial_event(member_dir, "bootstrap_done")
    generations = int(trial.get("generations", config["shared"]["generations"]))
    start_generation = int(trial.get("start_generation", 0))
    slot = trial.get("slot") or tune_trial_slot()
    metric_name = config["pbt"]["metric"]
    results = []

    for generation in range(start_generation, generations):
        from training.pbt.weaver import make_command
        from training.runtime import read_metrics

        started = time.monotonic()
        command, log_path, target_epoch = make_command(
            config,
            member,
            slot,
            member_dir,
            generation,
        )
        record_trial_event(
            member_dir,
            "weaver_command_ready",
            generation=generation,
            epoch=target_epoch,
            log=str(log_path),
            command=command,
        )
        console_path = member_dir / f"generation-{generation:03d}.tune.console.log"
        with console_path.open("w") as stream:
            record_trial_event(member_dir, "weaver_subprocess_start", generation=generation, console_log=str(console_path))
            result = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record_trial_event(member_dir, "weaver_subprocess_done", generation=generation, returncode=result.returncode)
        metrics = read_metrics(log_path)
        metric_ok = metrics is not None and metrics.get(metric_name) is not None
        if result.returncode != 0 or not metric_ok:
            record_trial_event(
                member_dir,
                "trial_failed",
                generation=generation,
                returncode=result.returncode,
                metric=metric_name,
                metric_ok=metric_ok,
            )
            raise RuntimeError(
                f"Weaver trial failed member={member_name} generation={generation} "
                f"returncode={result.returncode} metric={metric_name}"
            )

        report = scalar_report_metrics(metrics)
        report.update(
            {
                "generation": generation,
                "epoch": target_epoch,
                "lr": member["lr"],
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        metadata = {
            "member": member_name,
            "generation": generation,
            "lr": member["lr"],
            "metric": metric_name,
            "metric_value": metrics[metric_name],
            "log": str(log_path),
            "console_log": str(console_path),
            "direct_trial_smoke": True,
        }
        checkpoint_dir = Path(checkpoint_root or member_dir / "tune_checkpoints") / f"generation-{generation:03d}"
        checkpoint_metadata = package_tune_checkpoint(member_dir, target_epoch, checkpoint_dir, metadata)
        results.append({"report": report, "checkpoint": checkpoint_metadata})
        record_trial_event(member_dir, "direct_trial_report_ready", generation=generation, metric_value=metrics[metric_name])
    return results


def config_from_tune_payload(tune_config):
    if tune_config.get("pbt_config") is not None:
        return tune_config["pbt_config"]
    from training.pbt.config import load_config

    return load_config(
        SimpleNamespace(
            config=Path(tune_config["config_path"]),
            experiment_name=tune_config.get("experiment_name"),
            gpus=tune_config.get("gpus"),
            slots=None,
            smoke=bool(tune_config.get("smoke", False)),
        )
    )


def run_weaver_trial(tune_config):
    """Ray Tune Function API entrypoint.

    The trial owns one Weaver member directory, runs one or more PBT generations,
    reports scalar validation metrics to Tune, and publishes portable checkpoints.
    """
    member_dir = None
    try:
        tune, Checkpoint = _ray_tune_import()
        config = config_from_tune_payload(tune_config)
        trial = dict(tune_config["trial"])
        member_name = trial["member_name"]
        member = {"name": member_name, "lr": float(trial["lr"])}
        member_dir = Path(trial.get("member_dir") or tune_member_dir(config, member_name))
        member_dir.mkdir(parents=True, exist_ok=True)
        record_trial_event(member_dir, "trial_start", member=member_name, lr=member["lr"])

        restored = None
        current_checkpoint = tune.get_checkpoint()
        if current_checkpoint is not None:
            record_trial_event(member_dir, "restore_start")
            restored = restore_tune_checkpoint(current_checkpoint, member_dir)
            start_generation = int(restored.get("generation", -1)) + 1
            record_trial_event(member_dir, "restore_done", start_generation=start_generation)
        else:
            from training.pbt.checkpointing import bootstrap_initial_checkpoint

            record_trial_event(member_dir, "bootstrap_start")
            bootstrap_initial_checkpoint(config, member_dir)
            start_generation = int(trial.get("start_generation", 0))
            record_trial_event(member_dir, "bootstrap_done", start_generation=start_generation)

        generations = int(trial.get("generations", config["shared"]["generations"]))
        slot = trial.get("slot") or tune_trial_slot()
        metric_name = config["pbt"]["metric"]

        for generation in range(start_generation, generations):
            started = time.monotonic()
            from training.pbt.weaver import make_command
            from training.runtime import read_metrics

            record_trial_event(member_dir, "make_command_start", generation=generation)
            command, log_path, target_epoch = make_command(
                config,
                member,
                slot,
                member_dir,
                generation,
            )
            record_trial_event(
                member_dir,
                "weaver_command_ready",
                generation=generation,
                epoch=target_epoch,
                log=str(log_path),
                command=command,
            )
            console_path = member_dir / f"generation-{generation:03d}.tune.console.log"
            with console_path.open("w") as stream:
                record_trial_event(member_dir, "weaver_subprocess_start", generation=generation, console_log=str(console_path))
                result = subprocess.run(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record_trial_event(member_dir, "weaver_subprocess_done", generation=generation, returncode=result.returncode)
            metrics = read_metrics(log_path)
            metric_ok = metrics is not None and metrics.get(metric_name) is not None
            if result.returncode != 0 or not metric_ok:
                record_trial_event(
                    member_dir,
                    "trial_failed",
                    generation=generation,
                    returncode=result.returncode,
                    metric=metric_name,
                    metric_ok=metric_ok,
                )
                raise RuntimeError(
                    f"Weaver trial failed member={member_name} generation={generation} "
                    f"returncode={result.returncode} metric={metric_name}"
                )

            report = scalar_report_metrics(metrics)
            report.update(
                {
                    "generation": generation,
                    "epoch": target_epoch,
                    "lr": member["lr"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            metadata = {
                "member": member_name,
                "generation": generation,
                "lr": member["lr"],
                "metric": metric_name,
                "metric_value": metrics[metric_name],
                "log": str(log_path),
                "console_log": str(console_path),
                "restored_from": restored,
            }
            with tempfile.TemporaryDirectory() as temporary:
                package_tune_checkpoint(member_dir, target_epoch, temporary, metadata)
                record_trial_event(member_dir, "tune_report", generation=generation, metric_value=metrics[metric_name])
                tune.report(report, checkpoint=Checkpoint.from_directory(temporary))
    except Exception as error:
        if member_dir is not None:
            record_trial_event(member_dir, "trial_exception", error=repr(error))
        raise

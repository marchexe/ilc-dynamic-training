#!/usr/bin/env python3
"""Checkpoint path, copy, and bootstrap helpers for PBT."""

import os
import shutil
from pathlib import Path

from training.pbt.reporting import record_new_best
from training.pbt.state.optimizer_state import atomic_copy, prepare_initial_optimizer
from training.runtime import atomic_json, utc_now


def epoch_for_generation(config, generation):
    initial_epoch = int(config["shared"].get("initial_epoch", -1))
    return initial_epoch + (generation + 1) * int(config["shared"]["weaver_epochs_per_generation"])

def generations_before(manifest, generation_index):
    """Manifest generations strictly before `generation_index`, in original order."""
    return [
        generation
        for generation in manifest.get("generations", [])
        if int(generation.get("index", -1)) < int(generation_index)
    ]

def atomic_copy_pair(pairs):
    """Copy multiple (source, destination) pairs as one all-or-nothing unit.

    Every source is staged to a temporary file first; only once every
    staging copy has succeeded are the temp files committed in place via
    os.replace. This guarantees an exploit recipient never ends up with a
    donor's weights paired with its own unrelated, pre-copy optimizer state
    (or vice versa) -- weight and optimizer copy are one coherent transition.
    """
    staged = []
    try:
        for source, destination in pairs:
            destination = Path(destination)
            temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
            shutil.copy2(source, temporary)
            staged.append((temporary, destination))
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    for temporary, destination in staged:
        os.replace(temporary, destination)

def checkpoint_paths(member_dir, epoch):
    prefix = member_dir / f"net_epoch-{epoch}"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")

def controller_checkpoint_path(member_dir, epoch):
    return member_dir / f"net_epoch-{epoch}_controller.pt"

def population_lr_policy_snapshot_paths(member_dir, epoch):
    """Path for a recipient's own pre-copy checkpoint, snapshotted right
    before a population_lr_policy donor copy overwrites net_epoch-{epoch}_*
    in place. Without this, a rollback would have nothing distinct from the
    donor's state to restore -- the plain per-epoch path is the copy
    destination itself, so it no longer holds the recipient's own state by
    the time a rollback might need it.
    """
    prefix = member_dir / f"net_epoch-{epoch}_population_lr_policy_pre"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")

def bootstrap_initial_checkpoint(config, member_dir):
    shared = config["shared"]
    if not shared.get("initial_state"):
        return None
    initial_epoch = int(shared["initial_epoch"])
    state_path, optimizer_path = checkpoint_paths(member_dir, initial_epoch)
    if not state_path.exists():
        atomic_copy(Path(shared["initial_state"]), state_path)
    optimizer_metadata_path = member_dir / f"net_epoch-{initial_epoch}_optimizer_resume.json"
    if not optimizer_path.exists():
        optimizer_metadata = prepare_initial_optimizer(
            Path(shared["initial_optimizer"]),
            optimizer_path,
            mode=shared.get("initial_optimizer_mode", "raw"),
            damping_factor=shared.get("initial_optimizer_damping", 0.1),
        )
        optimizer_metadata.update(
            {
                "source_optimizer": str(shared["initial_optimizer"]),
                "destination_optimizer": str(optimizer_path),
                "initial_epoch": initial_epoch,
            }
        )
        atomic_json(optimizer_metadata_path, optimizer_metadata)
    elif not optimizer_metadata_path.exists():
        atomic_json(
            optimizer_metadata_path,
            {
                "mode": shared.get("initial_optimizer_mode", "raw"),
                "damping_factor": shared.get("initial_optimizer_damping", 0.1),
                "transformed": shared.get("initial_optimizer_mode", "raw") not in {"raw", "copy"},
                "source_optimizer": str(shared["initial_optimizer"]),
                "destination_optimizer": str(optimizer_path),
                "initial_epoch": initial_epoch,
                "reused_existing_destination": True,
            },
        )
    if shared.get("initial_controller"):
        controller_path = controller_checkpoint_path(member_dir, initial_epoch)
        if not controller_path.exists():
            atomic_copy(Path(shared["initial_controller"]), controller_path)
    return initial_epoch

def global_best_paths(experiment_dir):
    checkpoint_dir = Path(experiment_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "state_path": str(checkpoint_dir / "global_best_state.pt"),
        "optimizer_path": str(checkpoint_dir / "global_best_optimizer.pt"),
        "controller_path": str(checkpoint_dir / "global_best_controller.pt"),
        "metadata_path": str(checkpoint_dir / "global_best_metadata.json"),
    }

def seed_initial_global_best(config, experiment_dir, manifest):
    if not config["pbt"].get("baseline_guard_seed_initial_best"):
        return False
    if manifest.get("best"):
        return False

    shared = config["shared"]
    pbt = config["pbt"]
    baseline_metric = pbt.get("baseline_metric_value")
    if baseline_metric is None:
        raise ValueError("baseline_guard_seed_initial_best requires baseline_metric_value")
    if not shared.get("initial_state") or not shared.get("initial_optimizer"):
        raise ValueError("baseline_guard_seed_initial_best requires initial_state/initial_optimizer")

    experiment_dir = Path(experiment_dir)
    paths = global_best_paths(experiment_dir)
    atomic_copy(Path(shared["initial_state"]), Path(paths["state_path"]))
    optimizer_metadata = prepare_initial_optimizer(
        Path(shared["initial_optimizer"]),
        Path(paths["optimizer_path"]),
        mode=shared.get("initial_optimizer_mode", "raw"),
        damping_factor=shared.get("initial_optimizer_damping", 0.1),
    )

    controller_path = Path(paths["controller_path"])
    has_controller = False
    if shared.get("initial_controller"):
        atomic_copy(Path(shared["initial_controller"]), controller_path)
        has_controller = True
    elif controller_path.exists():
        controller_path.unlink()

    best_record = {
        "generation": -1,
        "epoch": int(shared["initial_epoch"]),
        "member": "initial_resume",
        "metric": pbt["metric"],
        "metric_value": float(baseline_metric),
        "lr": float(pbt.get("base_start_lr") or config["population"][0]["start_lr"]),
        "metrics": {pbt["metric"]: float(baseline_metric)},
        "updated_at": utc_now(),
        "source_state_path": str(shared["initial_state"]),
        "source_optimizer_path": str(shared["initial_optimizer"]),
        "source_controller_path": str(shared.get("initial_controller")) if has_controller else None,
        "optimizer_resume": optimizer_metadata,
        **paths,
    }
    manifest["best"] = best_record
    record_new_best(experiment_dir, manifest, {"index": -1}, best_record)
    manifest["updated_at"] = utc_now()
    atomic_json(Path(paths["metadata_path"]), best_record)
    return True

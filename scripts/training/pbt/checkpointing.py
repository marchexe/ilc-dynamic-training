#!/usr/bin/env python3
"""Checkpoint path, copy, and bootstrap helpers for PBT."""

import os
import shutil
from pathlib import Path

from training.pbt.optimizer_state import prepare_initial_optimizer
from training.runtime import atomic_json


def epoch_for_generation(config, generation):
    initial_epoch = int(config["shared"].get("initial_epoch", -1))
    return initial_epoch + (generation + 1) * int(config["shared"]["epochs_per_generation"])

def atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)

def checkpoint_paths(member_dir, epoch):
    prefix = member_dir / f"net_epoch-{epoch}"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")

def controller_checkpoint_path(member_dir, epoch):
    return member_dir / f"net_epoch-{epoch}_controller.pt"

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
    return {
        "state_path": str(experiment_dir / "global_best_state.pt"),
        "optimizer_path": str(experiment_dir / "global_best_optimizer.pt"),
        "controller_path": str(experiment_dir / "global_best_controller.pt"),
        "metadata_path": str(experiment_dir / "global_best_metadata.json"),
    }

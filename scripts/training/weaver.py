#!/usr/bin/env python3
"""Weaver command construction helpers."""

import re
from pathlib import Path

from training.runtime import data_command_args, weaver_executable


EPOCH_RE = re.compile(r"net_epoch-(\d+)_state\.pt$")


def latest_resumable_epoch(worker_dir, controller_enabled):
    candidates = []
    for state_path in worker_dir.glob("net_epoch-*_state.pt"):
        match = EPOCH_RE.search(state_path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        optimizer = worker_dir / f"net_epoch-{epoch}_optimizer.pt"
        controller = worker_dir / f"net_epoch-{epoch}_controller.pt"
        if optimizer.is_file() and (not controller_enabled or controller.is_file()):
            candidates.append(epoch)
    return max(candidates, default=None)


def build_command(
    resolved,
    worker,
    worker_dir,
    resume_epoch,
    *,
    log_path=None,
    override_load_lr=False,
):
    shared = dict(resolved["shared"])
    if worker.get("start_lr") is not None:
        shared["start_lr"] = worker["start_lr"]
    if worker.get("seed") is not None:
        shared["seed"] = worker["seed"]
    command = [
        str(weaver_executable()),
        "--run-mode",
        "train,val",
        *data_command_args(shared["dataset"], shared.get("data_extension", "root")),
        "--data-config",
        shared["data_config"],
        "--network-config",
        shared["network_config"],
        "--model-prefix",
        str(worker_dir / "net"),
        "--log-file",
        str(log_path or worker_dir / "train.log"),
        "--lr-scheduler",
        str(shared["lr_scheduler"]),
        "--seed",
        str(shared["seed"]),
        "--optimizer",
        str(shared["optimizer"]),
        "--batch-size",
        str(shared["batch_size"]),
        "--start-lr",
        str(shared["start_lr"]),
        "--samples-per-epoch",
        str(shared["samples_per_epoch"]),
        "--samples-per-epoch-val",
        str(shared["samples_per_epoch_val"]),
        "--num-epochs",
        str(shared["epochs"]),
        "--num-workers",
        str(shared["num_workers"]),
        "--fetch-step",
        str(shared["fetch_step"]),
        "--gpus",
        str(worker["gpu"]),
    ]
    if resume_epoch is None:
        command.extend(["--load-model-weights", shared["checkpoint"]])
    else:
        command.extend(["--load-epoch", str(resume_epoch)])
        if override_load_lr:
            command.append("--override-load-lr")
    for key, value in (shared.get("optimizer_options") or {}).items():
        command.extend(["--optimizer-option", str(key), str(value)])
    if shared.get("freeze_model_weights"):
        command.extend(["--freeze-model-weights", str(shared["freeze_model_weights"])])
    if worker["controller"]:
        command.extend(["--training-controller", worker["controller"]])
    if shared["no_remake_weights"]:
        command.append("--no-remake-weights")
    if shared["use_amp"]:
        command.extend(["--use-amp", "--amp-dtype", str(shared["amp_dtype"])])
    if shared.get("prefetch_factor") is not None:
        command.extend(["--prefetch-factor", str(shared["prefetch_factor"])])
    return command

#!/usr/bin/env python3
"""PBT-specific Weaver command construction."""

import shlex
import socket
import re
from pathlib import Path

from training.pbt.checkpointing import epoch_for_generation
from training.weaver import build_command
from training.runtime import PROJECT_DIR, data_paths, project_path, weaver_executable


def local_hostnames():
    names = {"localhost", "127.0.0.1"}
    for value in (socket.gethostname(), socket.getfqdn()):
        if value:
            names.add(value)
            names.add(value.split(".")[0])
    return names


def slot_label(slot):
    return slot["label"] if isinstance(slot, dict) else str(slot)


def remote_host(slot):
    if not isinstance(slot, dict):
        return None
    host = slot.get("host")
    if not host or host in local_hostnames():
        return None
    return host


def venv_python_version():
    config_path = project_path(".venv/pyvenv.cfg")
    if not config_path.exists():
        return "3.10"
    match = re.search(r"^version\s*=\s*(\d+\.\d+)", config_path.read_text(), re.MULTILINE)
    return match.group(1) if match else "3.10"


def wrap_remote_command(command, slot):
    host = remote_host(slot)
    if not host:
        return command
    if command and Path(command[0]).name == "weaver":
        venv_root = project_path(".venv")
        venv_python = venv_root / "bin/python"
        venv_weaver = venv_root / "bin/weaver"
        python_version = venv_python_version()
        expected_major, expected_minor = (int(part) for part in python_version.split(".", 1))
        site_packages = venv_root / f"lib/python{python_version}/site-packages"
        python_cmd = f"python{python_version}"
        args = shlex.join(command[1:])
        remote_pythonpath = (
            f"{shlex.quote(str(site_packages))}:"
            f"{shlex.quote(str(PROJECT_DIR / 'weaver-core'))}:"
            f"${{PYTHONPATH:-}}"
        )
        version_check = (
            f"{shlex.quote(str(venv_python))} -c "
            f"'import sys; raise SystemExit(sys.version_info[:2] != ({expected_major}, {expected_minor}))'"
        )
        remote = (
            f"cd {shlex.quote(str(PROJECT_DIR))} && "
            f"if [ -x {shlex.quote(str(venv_python))} ] && {version_check}; then "
            f"export PYTHONPATH={remote_pythonpath}; "
            f"exec {shlex.quote(str(venv_python))} {shlex.quote(str(venv_weaver))} {args}; "
            f"elif command -v {python_cmd} >/dev/null 2>&1 && [ -d {shlex.quote(str(site_packages))} ]; then "
            f"export PYTHONPATH={remote_pythonpath}; "
            f"exec {python_cmd} {shlex.quote(str(venv_weaver))} {args}; "
            f"else "
            f"actual=$({shlex.quote(str(venv_python))} --version 2>&1 || true); "
            f"echo \"found .venv/bin/python: $actual\" >&2; "
            f"echo 'remote Python {python_version} environment is not available for: "
            f"{shlex.quote(str(venv_weaver))}' >&2; "
            f"echo 'Expected Python {python_version} with packages in {shlex.quote(str(site_packages))}.' >&2; "
            f"exit 127; "
            f"fi"
        )
    else:
        remote = f"cd {shlex.quote(str(PROJECT_DIR))} && exec {shlex.join(command)}"
    return ["ssh", host, remote]


def _test_mode_command(shared, slot, validation_paths, checkpoint, log_path):
    """Build a Weaver `--run-mode test` command evaluating one checkpoint
    against one already-resolved set of validation data paths.

    Shared by baseline evaluation and every proxy-tier (control/monitor/full)
    evaluation, so there is exactly one place that knows how to turn
    (checkpoint, dataset, suffix) into a Weaver invocation.
    """
    gpu = slot["gpu"] if isinstance(slot, dict) else str(slot)
    command = [
        str(weaver_executable()),
        "--run-mode",
        "test",
        "--data-test",
        *validation_paths,
        "--data-config",
        shared["data_config"],
        "--network-config",
        shared["network_config"],
        "--model-prefix",
        str(checkpoint),
        "--log-file",
        str(log_path),
        "--batch-size",
        str(shared["batch_size"]),
        "--num-workers",
        str(shared["num_workers"]),
        "--fetch-step",
        str(shared["fetch_step"]),
        "--gpus",
        str(gpu),
        "--predict-gpus",
        str(gpu),
    ]
    if shared["use_amp"]:
        command.extend(["--use-amp", "--amp-dtype", str(shared["amp_dtype"])])
    if shared.get("prefetch_factor") is not None:
        command.extend(["--prefetch-factor", str(shared["prefetch_factor"])])
    return wrap_remote_command(command, slot)


def make_initial_evaluation_command(config, slot, experiment_dir):
    shared = config["shared"]
    validation_paths = data_paths(
        shared["dataset"],
        shared.get("data_extension", "root"),
        shared.get("validation_dataset"),
        shared.get("train_suffix"),
        shared.get("validation_suffix"),
    )["val"]
    eval_dir = Path(experiment_dir) / "logs" / "initial_evaluation"
    log_path = eval_dir / "initial-evaluation.log"
    checkpoint = shared.get("initial_state") or shared["checkpoint"]
    command = _test_mode_command(shared, slot, validation_paths, checkpoint, log_path)
    return command, log_path


def make_tiered_evaluation_command(config, slot, checkpoint, dataset, suffix, log_path):
    """Build a `--run-mode test` command evaluating an arbitrary checkpoint
    against an arbitrary (dataset, suffix) pair -- the primitive used for
    automatic monitor/full proxy-tier evaluation of every population member.
    """
    shared = config["shared"]
    validation_paths = data_paths(
        dataset,
        shared.get("data_extension", "root"),
        validation_suffix=suffix,
    )["val"]
    command = _test_mode_command(shared, slot, validation_paths, checkpoint, log_path)
    return command, log_path


def make_command(config, member, slot, member_dir, generation):
    target_epoch = epoch_for_generation(config, generation)
    resume_epoch = target_epoch - int(config["shared"]["epochs_per_generation"])
    if generation == 0 and not config["shared"].get("initial_state"):
        resume_epoch = None
    shared = dict(config["shared"])
    shared.update(
        epochs=target_epoch + 1,
        seed=int(config["shared"]["seed"]) + generation,
        start_lr=member["lr"],
    )
    freeze_generations = int(shared.get("freeze_model_weights_generations", 0) or 0)
    if freeze_generations and generation >= freeze_generations:
        shared.pop("freeze_model_weights", None)
    resolved = {"shared": shared}
    worker = {
        "name": member["name"],
        "gpu": slot["gpu"] if isinstance(slot, dict) else str(slot),
        "controller": shared.get("training_controller"),
    }
    log_path = member_dir.parent / "logs" / member["name"] / f"generation-{generation:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        resolved,
        worker,
        member_dir,
        resume_epoch,
        log_path=log_path,
        override_load_lr=resume_epoch is not None,
    )
    return wrap_remote_command(command, slot), log_path, target_epoch

#!/usr/bin/env python3
"""PBT-specific Weaver command construction."""

import shlex
import socket
from pathlib import Path

from training.pbt.strategy import epoch_for_generation
from training.weaver import build_command
from training.runtime import PROJECT_DIR, project_path


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


def wrap_remote_command(command, slot):
    host = remote_host(slot)
    if not host:
        return command
    if command and Path(command[0]).name == "weaver":
        venv_root = project_path(".venv")
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
            f"echo 'Make the project .venv available on this host at the same path.' >&2; "
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

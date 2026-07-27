#!/usr/bin/env python3
"""PBT-specific Weaver command construction."""

import shlex
import socket
import re
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

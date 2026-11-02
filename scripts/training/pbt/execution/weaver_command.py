#!/usr/bin/env python3
"""PBT-specific Weaver command construction."""

import shlex
import socket
import re
from pathlib import Path

from training.pbt.state.checkpointing import epoch_for_generation
from training.weaver import build_command
from training.members import MemberState
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


def _combine_test_group(validation_paths):
    """Strip the per-flavor keyword prefix from every `--data-test` glob.

    Weaver's `--run-mode test` loader (`test_load` in weaver-core) builds one
    independent DataLoader per keyword-prefixed group -- each is evaluated as
    its own separate test pass with its own confusion matrix, unlike
    `--data-val` during train/val mode, which merges every group into a
    single combined SimpleIterDataset. `data_paths()` labels each flavor
    glob `nnbb:`/`nncc:`/`nndd:` for that train/val merge; passed unchanged
    to `--data-test`, those three distinct labels instead make Weaver run
    three single-class passes back to back, each unable to compute a
    background-rejection curve since that requires signal and background
    classes present together. Dropping the prefix (Weaver's plain-list
    `--data-test` form) puts every glob under the same unlabelled group, so
    weaver's `test_load` merges them into one multi-class evaluation pass --
    the same approach already used by
    scripts/validation/evaluate_checkpoint_fixed_wp.py. Per-event class
    labels come from the data config, not the group name, so this does not
    change what each event is labelled.
    """
    return [path.split(":", 1)[1] if ":" in path else path for path in validation_paths]


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
        *_combine_test_group(validation_paths),
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
    for flag in ("data_audit", "deterministic"):
        if shared.get(flag):
            command.append("--" + flag.replace("_", "-"))
    if shared["use_amp"]:
        command.extend(["--use-amp", "--amp-dtype", str(shared["amp_dtype"])])
    if shared.get("save_predictions"):
        prediction_path = Path(log_path).with_suffix(".predictions.parquet")
        command.extend(["--predict-output", str(prediction_path)])
    if shared.get("prefetch_factor") is not None:
        command.extend(["--prefetch-factor", str(shared["prefetch_factor"])])
    return wrap_remote_command(command, slot)


def make_initial_evaluation_command(config, slot, experiment_dir):
    shared = config["shared"]
    if shared.get("initialization_mode") == "scratch":
        raise ValueError("scratch initialization has no initial checkpoint to evaluate")
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


def generation_seed(config, generation):
    """Weaver --seed of a generation: every slice of one data epoch shares that epoch's seed."""
    return int(config["shared"]["seed"]) + generation // int(config["shared"].get("generations_per_epoch") or 1)


def continuation_arguments(config, member_dir, generation, target_epoch):
    """Entry-point command prefix of the exact-restart contract, or None for the historical contract.

    With generations_per_epoch = n, generation g trains slice g % n of data epoch d = g // n. All slices of
    one data epoch share the seed and data stream of the one-epoch generation d would have, and each slice
    after the first restores the random and model process state its predecessor saved. The first slice of
    an epoch starts as a fresh process, like a one-epoch generation, so with no PBT actions the run equals
    the corresponding one-epoch run.
    """
    shared = config["shared"]
    if not shared.get("optimizer_state_contract"):
        return None
    entry = [str(weaver_executable().parent / "python"), str(PROJECT_DIR / "scripts/training/weaver_full_state.py")]
    slices = shared.get("generations_per_epoch")
    if not slices:
        return entry
    data_epoch, part = divmod(generation, slices)
    initial = shared.get("initial_epoch")
    batches = int(shared["batches_per_generation"])
    entry += ["--cursor-data-epoch", str((-1 if initial is None else int(initial)) + 1 + data_epoch),
              "--cursor-skip", str(part * batches)]
    if part < slices - 1:
        entry += ["--cursor-take", str(batches),
                  "--cursor-state-out", str(member_dir / f"net_epoch-{target_epoch}_continuation.pt")]
    if part:
        entry += ["--cursor-state-in", str(member_dir / f"net_epoch-{target_epoch - 1}_continuation.pt")]
    return entry


def make_command(config, member, slot, member_dir, generation):
    member = member if isinstance(member, MemberState) else MemberState.from_legacy(member)
    target_epoch = epoch_for_generation(config, generation)
    resume_epoch = target_epoch - int(config["shared"]["weaver_epochs_per_generation"])
    if generation == 0 and not config["shared"].get("initial_state"):
        resume_epoch = None
    entry = continuation_arguments(config, Path(member_dir), generation, target_epoch)
    shared = dict(config["shared"])
    shared.update(
        epochs=target_epoch + 1,
        seed=generation_seed(config, generation),
        start_lr=member.current_lr,
    )
    freeze_generations = int(shared.get("freeze_model_weights_generations", 0) or 0)
    if freeze_generations and generation >= freeze_generations:
        shared.pop("freeze_model_weights", None)
    resolved = {"shared": shared}
    worker = {
        "name": member.member_id,
        "gpu": slot["gpu"] if isinstance(slot, dict) else str(slot),
        "controller": shared.get("training_controller"),
    }
    log_path = member_dir.parent / "logs" / member.member_id / f"generation-{generation:03d}.log"
    # The backend creates the log directory when it starts a worker. Command
    # construction must remain read-only for dry runs and continuation planning.
    command = build_command(
        resolved,
        worker,
        member_dir,
        resume_epoch,
        log_path=log_path,
        override_load_lr=resume_epoch is not None,
    )
    if entry:
        command = [*entry, *command[1:]]  # the entry point takes Weaver's own arguments
    return wrap_remote_command(command, slot), log_path, target_epoch

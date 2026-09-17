#!/usr/bin/env python3
"""Start/resume/status/stop a local experiment using the production runner.

Requires --config; logs and process identity live in runs/launch_logs/<name>.
GPU IDs refer to physical nvidia-smi indices, pinned by UUID before launch.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.pbt.config import contract_fingerprint, load_config, validate_inputs
from training.runtime import PROJECT_DIR, atomic_json, utc_now

TOKEN_KEY = "MARCH_EXPERIMENT_TOKEN"


def configuration(config_path, gpus=None):
    return load_config(argparse.Namespace(config=config_path, experiment_name=None,
                                         gpus=gpus, slots=None, smoke=False))


def locations(config):
    run = Path(config["output_root"]) / config["experiment_name"]
    logs = PROJECT_DIR / "runs/launch_logs" / config["experiment_name"]
    return run, logs


def launch_command(config, gpu_ids):
    # CUDA_VISIBLE_DEVICES maps physical UUIDs to these contiguous local IDs.
    return [sys.executable, str(PROJECT_DIR / "scripts/training/run_pbt.py"),
            "--config", str(config["config_path"]), "--gpus",
            ",".join(str(i) for i in range(len(gpu_ids)))]


def available_gpus(gpu_ids, min_free_mib=26624):
    gpu_ids = [int(gpu) for gpu in gpu_ids]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Select distinct physical GPU indices")
    def query(kind, fields):
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True)
        return [[v.strip() for v in row] for row in csv.reader(output.splitlines()) if row]

    rows = query("gpu", "index,uuid,memory.used,memory.free,utilization.gpu")
    apps = query("compute-apps", "gpu_uuid,pid")
    busy = {row[0] for row in apps}
    selected = {int(row[0]): row for row in rows if int(row[0]) in gpu_ids}
    print("GPU availability (index, UUID, used MiB, free MiB, utilization %):", flush=True)
    for row in selected.values():
        print(", ".join(row), flush=True)
    if set(selected) != set(gpu_ids):
        raise RuntimeError("All selected GPUs must be present")
    for gpu, row in selected.items():
        if row[1] in busy or float(row[2]) > 1024 or float(row[3]) < min_free_mib or float(row[4]) > 5:
            raise RuntimeError(f"GPU {gpu} is occupied, active, or has less than {min_free_mib} MiB free; nothing launched")
    return [selected[i][1] for i in gpu_ids]


def start(config, min_free_mib=26624, *, resume=False):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support is required for identity-safe stopping")
    run, output = locations(config)
    previous = None
    if resume:
        previous = metadata(output)
        if live_pids(previous):
            raise RuntimeError('Refusing resume while launch-token-matched processes are alive')
        manifest = json.loads((run / 'manifest.json').read_text())
        if manifest['fingerprint'] != contract_fingerprint(config):
            raise ValueError('Resume configuration fingerprint differs')
        if Path(previous['run']).resolve() != run.resolve() or manifest.get('status') == 'completed':
            raise ValueError('Resume requires the same unfinished run')
    elif run.exists() or output.exists():
        raise FileExistsError(f"Refusing existing run or launcher output: {run}, {output}")
    if any(slot.get("host") for slot in config["slots"]):
        raise ValueError("This launcher supports local GPU slots only")
    validate_inputs(config)
    gpu_ids = config["gpus"]
    gpu_uuids = available_gpus(gpu_ids, min_free_mib)
    token = previous['token'] if previous else uuid.uuid4().hex
    command = launch_command(config, gpu_ids) + (['--resume'] if resume else [])
    env = dict(os.environ, **{TOKEN_KEY: token, "PYTHONUNBUFFERED": "1",
                             "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids)})
    output.mkdir(parents=True, exist_ok=resume)
    record = dict(token=token, host=socket.gethostname(), uid=os.getuid(),
                    started_at=utc_now(), command=command, gpu_uuids=gpu_uuids,
                    run=str(run), arms=[m["name"] for m in config["population"]])
    atomic_json(output / "launcher.json", record)
    with (output / "main.log").open("a" if resume else "x") as stream:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record["pid"] = process.pid
    atomic_json(output / "launcher.json", record)
    (output / "launcher.pid").write_text(f"{process.pid}\n")
    print(f"Started production runner PID {process.pid}; {len(config['population'])} arms configured.")
    print(f"Log: {output / 'main.log'}\nStudy: {run}")


def metadata(output):
    record = json.loads((output / "launcher.json").read_text())
    if record["host"] != socket.gethostname() or record["uid"] != os.getuid():
        raise RuntimeError("Status/stop must use the original launch host and user")
    if len(record.get("token", "")) != 32:
        raise RuntimeError("Invalid launch identity")
    return record


def owns_process(path, record):
    """Inherited token finds workers even when they create separate sessions."""
    try:
        if path.stat().st_uid != record["uid"]:
            return False
        expected = f"{TOKEN_KEY}={record['token']}".encode()
        return expected in (path / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def live_pids(record):
    return sorted(int(p.name) for p in Path("/proc").iterdir()
                  if p.name.isdigit() and owns_process(p, record))


def status(output):
    record = metadata(output)
    pids = live_pids(record)
    manifest_path = Path(record["run"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    generations = manifest.get("generations", [])
    workers = generations[-1].get("workers", {}) if generations else {}
    names = record["arms"]
    arms = {name: dict(status=workers.get(name, {}).get("status", "pending"),
                      pid=workers.get(name, {}).get("pid"),
                      alive=workers.get(name, {}).get("pid") in pids) for name in names}
    print(json.dumps(dict(run_status=manifest.get("status", "starting"),
                          live_pids=pids, arms=arms,
                          incomplete_without_live_processes=not pids and manifest.get("status") != "completed"),
                     indent=2))


def stop(output):
    record = metadata(output)
    # Stop the launcher first so it cannot intentionally schedule another epoch.
    # Re-scan descendants after signaling it; never signal by a saved PID alone.
    signaled = set()
    for _ in range(3):
        pids = live_pids(record)
        pids.sort(key=lambda pid: pid != record.get("pid"))
        for pid in pids:
            if pid in signaled:
                continue
            try:
                fd = os.pidfd_open(pid)
                try:
                    if owns_process(Path(f"/proc/{pid}"), record):
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                        signaled.add(pid)
                finally:
                    os.close(fd)
            except ProcessLookupError:
                pass
    print(f"Sent SIGTERM only to launch-token-matched PIDs: {sorted(signaled)}")
    print("Use status after shutdown; repeat stop if descendants are still exiting. No unrelated jobs signaled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "resume", "status", "stop"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", help="Comma-separated physical GPU indices; defaults to config")
    parser.add_argument("--min-free-mib", type=int, default=26624)
    args = parser.parse_args()
    os.chdir(PROJECT_DIR)
    config = configuration(args.config, args.gpus)
    if args.action in ('start', 'resume'):
        if args.min_free_mib <= 0:
            parser.error("--min-free-mib must be positive")
        start(config, args.min_free_mib, resume=args.action == 'resume')
    else:
        {"status": status, "stop": stop}[args.action](locations(config)[1])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"error: {error}")

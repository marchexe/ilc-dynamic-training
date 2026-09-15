#!/usr/bin/env python3
"""User-run full-data smoke: start, status, or stop only this launch's processes.

Run on iutgpu01 with the project .venv. Uses run_pbt.py without --smoke.
The production runner owns study/; launcher logs and identity live beside it.
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
from training.pbt.config import load_config, validate_inputs
from training.runtime import PROJECT_DIR, atomic_json, utc_now

CONFIG = PROJECT_DIR / "configs/experiments/foundation_fixed_lr_smoke.yaml"
OUTPUT = PROJECT_DIR / "runs/pbt/foundation_fixed_lr_smoke_20260916"
TOKEN_KEY = "MARCH_FIXED_LR_SMOKE_TOKEN"


def configuration():
    return load_config(argparse.Namespace(config=CONFIG, experiment_name=None,
                                         gpus="0,1,2,3,4", slots=None, smoke=False))


def launch_command():
    return [sys.executable, str(PROJECT_DIR / "scripts/training/run_pbt.py"),
            "--config", str(CONFIG), "--gpus", "0,1,2,3,4"]


def available_gpus():
    def query(kind, fields):
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True)
        return [[v.strip() for v in row] for row in csv.reader(output.splitlines()) if row]

    rows = query("gpu", "index,uuid,memory.used,memory.free,utilization.gpu")
    apps = query("compute-apps", "gpu_uuid,pid")
    busy = {row[0] for row in apps}
    selected = {int(row[0]): row for row in rows if int(row[0]) in range(5)}
    print("GPU availability (index, UUID, used MiB, free MiB, utilization %):", flush=True)
    for row in selected.values():
        print(", ".join(row), flush=True)
    if set(selected) != set(range(5)):
        raise RuntimeError("GPUs 0–4 must all be present")
    for gpu, row in selected.items():
        if row[1] in busy or float(row[2]) > 1024 or float(row[3]) < 26624 or float(row[4]) > 5:
            raise RuntimeError(f"GPU {gpu} is occupied, active, or has less than 26 GiB free; nothing launched")
    return [selected[i][1] for i in range(5)]


def start():
    if socket.gethostname().split(".")[0] != "iutgpu01":
        raise RuntimeError("Run this launcher directly on iutgpu01")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support is required for identity-safe stopping")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {OUTPUT}")
    config = configuration()
    validate_inputs(config)
    if Path(config["output_root"]) != OUTPUT or config["experiment_name"] != "study":
        raise RuntimeError("Smoke config output no longer matches launcher")
    gpu_uuids = available_gpus()
    token = uuid.uuid4().hex
    command = launch_command()
    env = dict(os.environ, **{TOKEN_KEY: token, "PYTHONUNBUFFERED": "1",
                             "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids)})
    OUTPUT.mkdir(parents=True, exist_ok=False)
    metadata = dict(token=token, host=socket.gethostname(), uid=os.getuid(),
                    started_at=utc_now(), command=command, gpu_uuids=gpu_uuids)
    atomic_json(OUTPUT / "launcher.json", metadata)
    with (OUTPUT / "main.log").open("x") as stream:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    metadata["pid"] = process.pid
    atomic_json(OUTPUT / "launcher.json", metadata)
    (OUTPUT / "launcher.pid").write_text(f"{process.pid}\n")
    print(f"Started production runner PID {process.pid}; five training arms follow initial evaluation.")
    print(f"Log: {OUTPUT / 'main.log'}\nStudy: {OUTPUT / 'study'}")


def metadata():
    record = json.loads((OUTPUT / "launcher.json").read_text())
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


def status():
    record = metadata()
    pids = live_pids(record)
    manifest_path = OUTPUT / "study/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    generations = manifest.get("generations", [])
    workers = generations[-1].get("workers", {}) if generations else {}
    names = [m["name"] for m in configuration()["population"]]
    arms = {name: dict(status=workers.get(name, {}).get("status", "pending"),
                      pid=workers.get(name, {}).get("pid"),
                      alive=workers.get(name, {}).get("pid") in pids) for name in names}
    print(json.dumps(dict(run_status=manifest.get("status", "starting"),
                          live_smoke_pids=pids, arms=arms,
                          incomplete_without_live_processes=not pids and manifest.get("status") != "completed"),
                     indent=2))


def stop():
    record = metadata()
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
    print(f"Sent SIGTERM only to launch-token-matched smoke PIDs: {sorted(signaled)}")
    print("Use status after shutdown; repeat stop if descendants are still exiting. No unrelated jobs signaled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop"])
    args = parser.parse_args()
    os.chdir(PROJECT_DIR)
    {"start": start, "status": status, "stop": stop}[args.action]()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"error: {error}")

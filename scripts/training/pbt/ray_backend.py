"""Ray backend for Weaver PBT worker execution."""

import os
import subprocess
import sys
import time
from pathlib import Path

from training.pbt.backend import PBTBackend, format_duration, log_event
from training.pbt.weaver import make_command
from training.runtime import PROJECT_DIR, atomic_json, read_metrics, utc_now

SCRIPTS_DIR = PROJECT_DIR / "scripts"
WEAVER_CORE_DIR = PROJECT_DIR / "weaver-core"
for path in (SCRIPTS_DIR, WEAVER_CORE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _ray_import():
    try:
        import ray
    except ImportError as error:
        raise RuntimeError("Ray backend requires ray[tune]. Install project requirements first.") from error
    return ray


def _run_weaver_command(payload):
    started = time.monotonic()
    console_path = Path(payload["console_log"])
    console_path.parent.mkdir(parents=True, exist_ok=True)
    with console_path.open("w") as stream:
        result = subprocess.run(
            payload["command"],
            cwd=PROJECT_DIR,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metrics = read_metrics(Path(payload["log"]))
    return {
        "name": payload["name"],
        "returncode": result.returncode,
        "metrics": metrics,
        "pid": os.getpid(),
        "elapsed_seconds": time.monotonic() - started,
    }


class RayWeaverBackend(PBTBackend):
    """Run each generation's Weaver workers as Ray tasks.

    Ray is used only as the distributed execution layer. The PBT contract, ranking,
    anchored LR sweep, checkpoint exploit, manifest, and reports stay in the common
    runner so Ray and local runs produce the same artifacts.
    """

    name = "ray_weaver"

    def command_for(self, config, member, slot, member_dir, generation):
        return make_command(config, member, slot, member_dir, generation)

    def run_generation(self, config, experiment_dir, manifest, generation_record, names, manifest_path):
        ray = _ray_import()
        runtime_pythonpath = os.pathsep.join(
            [str(SCRIPTS_DIR), str(WEAVER_CORE_DIR), os.environ.get("PYTHONPATH", "")]
        )
        os.environ["PYTHONPATH"] = runtime_pythonpath
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"env_vars": {"PYTHONPATH": runtime_pythonpath}},
            )

        pbt_log_path = manifest_path.with_name("pbt.log")
        pending_names = list(names)
        free_slots = list(config["slots"])
        running = {}
        task_for_ref = {}
        remote_worker = ray.remote(_run_weaver_command)

        def start_worker(name, slot):
            member = manifest["members"][name]
            member_dir = experiment_dir / name
            command, log_path, target_epoch = self.command_for(
                config, member, slot, member_dir, generation_record["index"]
            )
            console_path = member_dir / f"generation-{generation_record['index']:03d}.console.log"
            payload = {
                "name": name,
                "command": command,
                "log": str(log_path),
                "console_log": str(console_path),
            }
            ref = remote_worker.remote(payload)
            running[name] = {
                "ref": ref,
                "slot": slot,
                "started_monotonic": time.monotonic(),
            }
            task_for_ref[ref] = name
            generation_record["workers"][name].update(
                status="running",
                gpu=slot["gpu"] if isinstance(slot, dict) else str(slot),
                host=slot.get("host") if isinstance(slot, dict) else None,
                slot=self.slot_label(slot),
                pid=None,
                ray_task=str(ref),
                command=command,
                log=str(log_path),
                console_log=str(console_path),
                target_epoch=target_epoch,
                started_at=utc_now(),
                finished_at=None,
                returncode=None,
                metrics=None,
            )
            log_event(
                pbt_log_path,
                f"started generation={generation_record['index']} worker={name} "
                f"slot={self.slot_label(slot)} ray_task={ref}",
            )

        while pending_names and free_slots:
            start_worker(pending_names.pop(0), free_slots.pop(0))
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)

        failure = None
        try:
            while running or pending_names:
                ready_refs, _ = ray.wait(
                    [item["ref"] for item in running.values()],
                    num_returns=1,
                    timeout=0.5,
                )
                if not ready_refs:
                    continue
                for ref in ready_refs:
                    name = task_for_ref.pop(ref)
                    task = running.pop(name)
                    slot = task["slot"]
                    try:
                        result = ray.get(ref)
                    except Exception as error:
                        result = {
                            "name": name,
                            "returncode": 1,
                            "metrics": None,
                            "pid": None,
                            "error": f"{type(error).__name__}: {error}",
                            "elapsed_seconds": time.monotonic() - task["started_monotonic"],
                        }
                    elapsed = format_duration(result.get("elapsed_seconds", 0))
                    metrics = result.get("metrics")
                    metric_name = config["pbt"]["metric"]
                    metric_ok = metrics is not None and metrics.get(metric_name) is not None
                    status = "completed" if result["returncode"] == 0 and metric_ok else "failed"
                    record = generation_record["workers"][name]
                    record.update(
                        status=status,
                        pid=result.get("pid"),
                        returncode=result["returncode"],
                        metrics=metrics,
                        finished_at=utc_now(),
                    )
                    if result.get("error"):
                        record["error"] = result["error"]
                    log_event(
                        pbt_log_path,
                        f"finished generation={generation_record['index']} "
                        f"worker={name} returncode={result['returncode']} elapsed={elapsed}",
                    )
                    manifest["updated_at"] = utc_now()
                    atomic_json(manifest_path, manifest)
                    if status == "failed":
                        failure = name
                        break
                    if pending_names:
                        start_worker(pending_names.pop(0), slot)
                        manifest["updated_at"] = utc_now()
                        atomic_json(manifest_path, manifest)
                    else:
                        free_slots.append(slot)
                if failure:
                    break
        except BaseException:
            self._cancel_running(ray, running, generation_record, manifest, manifest_path, pbt_log_path)
            raise

        if failure:
            self._cancel_running(ray, running, generation_record, manifest, manifest_path, pbt_log_path)
            raise RuntimeError(f"PBT worker failed: {failure}")

    @staticmethod
    def _cancel_running(ray, running, generation_record, manifest, manifest_path, pbt_log_path):
        for name, task in running.items():
            ray.cancel(task["ref"], force=True)
            elapsed = format_duration(time.monotonic() - task["started_monotonic"])
            generation_record["workers"][name].update(
                status="terminated",
                returncode=None,
                finished_at=utc_now(),
            )
            log_event(
                pbt_log_path,
                f"terminated generation={generation_record['index']} "
                f"worker={name} returncode=None elapsed={elapsed}",
            )
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)

"""Execution backends for Population Based Training workers."""

import subprocess
import time
from pathlib import Path

from training.pbt.artifacts import refresh_metrics_csv, record_evaluation, record_train_finish, record_train_start
from training.pbt.weaver import make_command, make_initial_evaluation_command, slot_label
from training.runtime import PROJECT_DIR, atomic_json, read_metrics, terminate, utc_now


class PBTBackend:
    """Interface for launching and monitoring one PBT generation."""

    name = "abstract"

    def command_for(self, config, member, slot, member_dir, generation):
        raise NotImplementedError

    def slot_label(self, slot):
        return slot_label(slot)

    def initial_evaluation_command_for(self, config, slot, experiment_dir):
        raise NotImplementedError

    def run_generation(self, config, experiment_dir, manifest, generation_record, names, manifest_path):
        raise NotImplementedError


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def log_event(log_path, message):
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


class LocalWeaverBackend(PBTBackend):
    """Run each population member as a local or SSH-wrapped Weaver process."""

    name = "local_weaver"

    def command_for(self, config, member, slot, member_dir, generation):
        return make_command(config, member, slot, member_dir, generation)

    def initial_evaluation_command_for(self, config, slot, experiment_dir):
        return make_initial_evaluation_command(config, slot, experiment_dir)

    def run_generation(self, config, experiment_dir, manifest, generation_record, names, manifest_path):
        processes = {}
        streams = {}
        started_monotonic = {}
        process_slots = {}
        pbt_log_path = manifest_path.parent / "logs" / "pbt.log"
        pending_names = list(names)
        free_slots = list(config["slots"])

        def start_worker(name, slot):
            member = manifest["members"][name]
            member_dir = experiment_dir / name
            command, log_path, target_epoch = self.command_for(
                config, member, slot, member_dir, generation_record["index"]
            )
            console_path = experiment_dir / "logs" / name / f"generation-{generation_record['index']:03d}.console.log"
            console_path.parent.mkdir(parents=True, exist_ok=True)
            stream = console_path.open("w")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                stream.close()
                terminate(processes)
                for running_name, running_process in processes.items():
                    streams[running_name].close()
                    generation_record["workers"][running_name].update(
                        status="terminated",
                        returncode=running_process.poll(),
                        finished_at=utc_now(),
                    )
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
                raise
            processes[name] = process
            streams[name] = stream
            process_slots[name] = slot
            started_monotonic[name] = time.monotonic()
            generation_record["workers"][name].update(
                status="running",
                gpu=slot["gpu"] if isinstance(slot, dict) else str(slot),
                host=slot.get("host") if isinstance(slot, dict) else None,
                slot=self.slot_label(slot),
                pid=process.pid,
                lr=float(member["lr"]),
                command=command,
                log=str(log_path),
                console_log=str(console_path),
                target_epoch=target_epoch,
                started_at=utc_now(),
                finished_at=None,
                returncode=None,
                metrics=None,
            )
            record_train_start(experiment_dir, config, generation_record, name, generation_record["workers"][name])
            log_event(
                pbt_log_path,
                f"started generation={generation_record['index']} worker={name} "
                f"slot={self.slot_label(slot)} pid={process.pid}",
            )

        while pending_names and free_slots:
            start_worker(pending_names.pop(0), free_slots.pop(0))
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)

        failure = None
        try:
            while processes or pending_names:
                for name, process in list(processes.items()):
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    streams.pop(name).close()
                    processes.pop(name)
                    finished_slot = process_slots.pop(name)
                    elapsed = format_duration(time.monotonic() - started_monotonic.pop(name))
                    record = generation_record["workers"][name]
                    metrics = read_metrics(Path(record["log"]))
                    metric_name = config["pbt"]["metric"]
                    metric_ok = metrics is not None and metrics.get(metric_name) is not None
                    status = "completed" if returncode == 0 and metric_ok else "failed"
                    record.update(
                        status=status,
                        returncode=returncode,
                        metrics=metrics,
                        finished_at=utc_now(),
                    )
                    record_train_finish(experiment_dir, config, generation_record, name, record)
                    if metric_ok:
                        record_evaluation(experiment_dir, config, generation_record, name, record)
                    refresh_metrics_csv(experiment_dir, manifest)
                    log_event(
                        pbt_log_path,
                        f"finished generation={generation_record['index']} "
                        f"worker={name} returncode={returncode} elapsed={elapsed}",
                    )
                    manifest["updated_at"] = utc_now()
                    atomic_json(manifest_path, manifest)
                    if status == "failed":
                        failure = name
                        break
                    if pending_names:
                        start_worker(pending_names.pop(0), finished_slot)
                        manifest["updated_at"] = utc_now()
                        atomic_json(manifest_path, manifest)
                    else:
                        free_slots.append(finished_slot)
                if failure:
                    break
                time.sleep(0.5)
        except BaseException:
            self._terminate_running(
                processes, streams, started_monotonic, generation_record, manifest,
                manifest_path, pbt_log_path
            )
            raise

        if failure:
            self._terminate_running(
                processes, streams, started_monotonic, generation_record, manifest,
                manifest_path, pbt_log_path
            )
            raise RuntimeError(f"PBT worker failed: {failure}")

    @staticmethod
    def _terminate_running(processes, streams, started_monotonic, generation_record, manifest, manifest_path, pbt_log_path):
        terminate(processes)
        for name, process in processes.items():
            streams[name].close()
            elapsed = format_duration(time.monotonic() - started_monotonic.get(name, time.monotonic()))
            generation_record["workers"][name].update(
                status="terminated",
                returncode=process.poll(),
                finished_at=utc_now(),
            )
            log_event(
                pbt_log_path,
                f"terminated generation={generation_record['index']} "
                f"worker={name} returncode={process.poll()} elapsed={elapsed}",
            )
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)


def backend_from_config(config):
    backend_name = config.get("pbt", {}).get("backend", "local_weaver")
    if backend_name == "local_weaver":
        return LocalWeaverBackend()
    if backend_name in {"ray_weaver", "ray_tune"}:
        from training.pbt.ray_backend import RayWeaverBackend

        return RayWeaverBackend()
    raise ValueError(f"Unsupported PBT backend: {backend_name}")

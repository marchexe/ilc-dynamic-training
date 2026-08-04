"""Execution backends for Population Based Training workers."""

import math
import subprocess
import time
from pathlib import Path

from training.pbt.reporting import refresh_metrics_csv, record_evaluation, record_train_finish, record_train_start
from training.pbt.execution.weaver_command import make_command, make_initial_evaluation_command, make_tiered_evaluation_command, slot_label
from training.runtime import PROJECT_DIR, atomic_json, read_metrics, terminate, utc_now


def finite_metric_ok(metrics, metric_name):
    """True only if `metric_name` is present and a finite (non-NaN/inf) number.

    A non-finite value (e.g. NaN from a zero-count fixed-WP ratio) must never
    reach ranking/exploit/controller decisions -- treat it the same as a
    missing metric: the worker is marked failed rather than silently
    poisoning the population's ranking with a NaN comparison.
    """
    if metrics is None:
        return False
    value = metrics.get(metric_name)
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def run_tiered_evaluation(config, experiment_dir, generation_index, tier, dataset, suffix, member_checkpoints, pbt_log_path):
    """Evaluate `tier` (monitor/full) for every (member, checkpoint) pair in
    `member_checkpoints`, in parallel across the population's GPU slots,
    sequentially with respect to population training (no dedicated
    evaluation GPU is reserved -- this runs after a generation's population
    work frees every slot, and before the next generation's training claims
    them again).

    Read-only and best-effort: a single member's evaluation failing here
    never raises and never blocks the others -- monitor/full are
    diagnostics, not part of the training critical path, and must not be
    able to abort a PBT run.
    """
    experiment_dir = Path(experiment_dir)
    results = {}
    processes = {}
    streams = {}
    process_context = {}
    started_monotonic = {}
    pending = list(member_checkpoints.items())
    free_slots = list(config["slots"])

    def start(member_name, checkpoint_path, slot):
        eval_dir = experiment_dir / "logs" / "tiered_evaluation" / tier / member_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        log_path = eval_dir / f"generation-{generation_index:03d}.log"
        console_path = log_path.with_suffix(".console.log")
        command, _ = make_tiered_evaluation_command(config, slot, checkpoint_path, dataset, suffix, log_path)
        stream = console_path.open("w")
        try:
            process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as error:
            stream.close()
            results[member_name] = {
                "status": "failed",
                "error": str(error),
                "metrics": None,
                "log": str(log_path),
                "checkpoint": str(checkpoint_path),
            }
            log_event(pbt_log_path, f"tiered_evaluation tier={tier} generation={generation_index} worker={member_name} failed_to_launch={error}")
            return False
        processes[member_name] = process
        streams[member_name] = stream
        process_context[member_name] = (slot, log_path, checkpoint_path)
        started_monotonic[member_name] = time.monotonic()
        return True

    while pending and free_slots:
        name, checkpoint_path = pending.pop(0)
        slot = free_slots.pop(0)
        if not start(name, checkpoint_path, slot):
            free_slots.append(slot)

    while processes or pending:
        for name, process in list(processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            streams.pop(name).close()
            processes.pop(name)
            slot, log_path, checkpoint_path = process_context.pop(name)
            elapsed = format_duration(time.monotonic() - started_monotonic.pop(name))
            metrics = read_metrics(log_path)
            metric_ok = metrics is not None and metrics.get("validation_bkg_rejection_at_eff") is not None
            status = "completed" if returncode == 0 and metric_ok else "failed"
            results[name] = {
                "status": status,
                "returncode": returncode,
                "metrics": metrics,
                "log": str(log_path),
                "checkpoint": str(checkpoint_path),
            }
            log_event(
                pbt_log_path,
                f"tiered_evaluation tier={tier} generation={generation_index} worker={name} status={status} elapsed={elapsed}",
            )
            if metrics is not None and metrics.get("validation_shutdown_warning"):
                log_event(
                    pbt_log_path,
                    f"WARNING: validation_shutdown_warning tier={tier} generation={generation_index} worker={name}",
                )
            free_slots.append(slot)
            while pending and free_slots:
                next_name, next_checkpoint = pending.pop(0)
                next_slot = free_slots.pop(0)
                if not start(next_name, next_checkpoint, next_slot):
                    free_slots.append(next_slot)
        if processes or pending:
            time.sleep(0.5)
    return results


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
                    metric_ok = finite_metric_ok(metrics, metric_name)
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
                        f"worker={name} returncode={returncode} elapsed={elapsed}"
                        + ("" if metric_ok or metrics is None or metrics.get(metric_name) is None else " non_finite_metric=true"),
                    )
                    if metrics is not None and metrics.get("validation_shutdown_warning"):
                        log_event(
                            pbt_log_path,
                            f"WARNING: validation_shutdown_warning generation={generation_record['index']} "
                            f"worker={name} -- data-loader threads reported a shutdown race during evaluation; "
                            "treat this worker's metrics for this generation with extra caution",
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
        from training.pbt.execution.ray_backend import RayWeaverBackend

        return RayWeaverBackend()
    raise ValueError(f"Unsupported PBT backend: {backend_name}")

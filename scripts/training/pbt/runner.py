#!/usr/bin/env python3
"""Run epoch-level Population Based Training on independent Weaver workers."""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from training.pbt.weaver import make_command, slot_label
from training.pbt.config import contract_fingerprint, load_config, validate_inputs
from training.runtime import (
    PROJECT_DIR,
    atomic_json,
    git_metadata,
    read_metrics,
    sha256,
    terminate,
    utc_now,
)
from training.pbt.strategy import (
    add_global_best_rollbacks,
    apply_exploit,
    checkpoint_paths,
    controller_checkpoint_path,
    epoch_for_generation,
    global_best_paths,
    ranking_and_plan,
    update_generation_health,
    update_global_best,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
MANIFEST_NAME = "manifest.json"
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-name")
    parser.add_argument("--gpus", help="Comma-separated GPU slots, e.g. 0,2")
    parser.add_argument(
        "--slots",
        help=(
            "Comma-separated host:gpu slots for multi-node runs. "
            "Every host must see the project .venv at the same path, "
            "e.g. iutgpu01:6,iutgpu05:4"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two members, two generations and small train/validation budgets",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def launch_workers(config, experiment_dir, manifest, generation_record, names, manifest_path):
    processes = {}
    streams = {}
    started_monotonic = {}
    process_slots = {}
    pbt_log_path = manifest_path.with_name("pbt.log")
    pending_names = list(names)
    free_slots = list(config["slots"])

    def start_worker(name, slot):
        member = manifest["members"][name]
        member_dir = experiment_dir / name
        command, log_path, target_epoch = make_command(
            config, member, slot, member_dir, generation_record["index"]
        )
        console_path = member_dir / f"generation-{generation_record['index']:03d}.console.log"
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
            gpu=slot["gpu"],
            host=slot.get("host") if isinstance(slot, dict) else None,
            slot=slot_label(slot),
            pid=process.pid,
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
            f"slot={slot_label(slot)} pid={process.pid}",
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
        raise

    if failure:
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
        raise RuntimeError(f"PBT worker failed: {failure}")


def initial_manifest(config, fingerprint):
    checkpoint = Path(config["shared"]["checkpoint"])
    return {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "fingerprint": fingerprint,
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "next_generation": 0,
        "config": config,
        "checkpoint": {
            "path": str(checkpoint),
            "resolved_path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        },
        "git": git_metadata(),
        "members": {
            member["name"]: {
                "name": member["name"],
                "lr": float(member["start_lr"]),
                "parent": None,
            }
            for member in config["population"]
        },
        "best": None,
        "generations": [],
    }


def run(args):
    config = load_config(args)
    validate_inputs(config)
    fingerprint = contract_fingerprint(config)
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    manifest_path = experiment_dir / MANIFEST_NAME
    pbt_log_path = manifest_path.with_name("pbt.log")

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False).rstrip())
        for index, member_config in enumerate(config["population"]):
            member = {"name": member_config["name"], "lr": float(member_config["start_lr"])}
            command, _, _ = make_command(
                config,
                member,
                config["slots"][index % len(config["slots"])],
                experiment_dir / member["name"],
                0,
            )
            print(f"[{member['name']}] {shlex.join(command)}")
        return 0

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resolved PBT contract differs from the saved manifest")
        if manifest["status"] == "completed":
            print(f"already completed: {experiment_dir}")
            return 0
        manifest["status"] = "running"
        manifest.pop("failure", None)
        manifest["updated_at"] = utc_now()
    else:
        if experiment_dir.exists():
            raise FileExistsError(f"Experiment already exists: {experiment_dir}")
        experiment_dir.mkdir(parents=True)
        manifest = initial_manifest(config, fingerprint)
    run_started_monotonic = time.monotonic()

    for name in manifest["members"]:
        (experiment_dir / name).mkdir(parents=True, exist_ok=True)
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    atomic_json(manifest_path, manifest)
    log_event(
        pbt_log_path,
        f"run started experiment={config['experiment_name']} "
        f"slots={','.join(slot_label(slot) for slot in config['slots'])} "
        f"members={len(manifest['members'])} "
        f"batch_size={config['shared']['batch_size']} metric={config['pbt']['metric']}",
    )

    try:
        for generation in range(
            int(manifest["next_generation"]), int(config["shared"]["generations"])
        ):
            existing = next(
                (item for item in manifest["generations"] if item["index"] == generation),
                None,
            )
            if existing is None:
                existing = {
                    "index": generation,
                    "epoch": epoch_for_generation(config, generation),
                    "seed": int(config["shared"]["seed"]) + generation,
                    "status": "training",
                    "workers": {
                        name: {"status": "pending"} for name in manifest["members"]
                    },
                    "ranking": None,
                    "exploit": None,
                    "started_at": utc_now(),
                }
                manifest["generations"].append(existing)
                atomic_json(manifest_path, manifest)
            existing["status"] = "training"
            pending = [
                name
                for name, record in existing["workers"].items()
                if record["status"] != "completed"
            ]
            launch_workers(
                config,
                experiment_dir,
                manifest,
                existing,
                pending,
                manifest_path,
            )

            if existing["exploit"] is None:
                ranking, plan = ranking_and_plan(
                    config, existing, manifest["members"]
                )
                existing["ranking"] = ranking
                improved = update_global_best(
                    experiment_dir, manifest, existing, manifest_path
                )
                health = update_generation_health(config, manifest, existing)
                plan = add_global_best_rollbacks(
                    config, manifest, existing, manifest["members"], plan
                )
                existing["exploit"] = (
                    [] if generation == int(config["shared"]["generations"]) - 1 else plan
                )
                existing["status"] = "exploiting"
                best = manifest.get("best") or {}
                log_event(
                    pbt_log_path,
                    "health generation=%d current_best=%s %.6g global_best=%s %.6g "
                    "delta=%s degraded=%s consecutive=%d improved=%s lrs=%s"
                    % (
                        existing["index"],
                        health["current_best_member"],
                        health["current_best_metric"],
                        best.get("member"),
                        best.get("metric_value", float("nan")),
                        (
                            "n/a"
                            if health["relative_to_global_best"] is None
                            else f"{health['relative_to_global_best']:+.3%}"
                        ),
                        health["status"],
                        health["consecutive_degraded_generations"],
                        improved,
                        ",".join(
                            f"{name}:{lr:.3g}"
                            for name, lr in sorted(health["member_lrs"].items())
                        ),
                    ),
                )
                atomic_json(manifest_path, manifest)

            apply_exploit(experiment_dir, manifest, existing, manifest_path)
            existing["status"] = "completed"
            existing["finished_at"] = utc_now()
            manifest["next_generation"] = generation + 1
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)

        manifest["status"] = "completed"
        manifest.pop("failure", None)
        manifest["finished_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        try:
            from reports.plot_pbt_summary import plot_manifest

            plot_path = plot_manifest(manifest_path)
            manifest["summary_plot"] = str(plot_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"summary plot: {plot_path}")
        except Exception as plot_error:
            log_event(pbt_log_path, f"warning: failed to create PBT summary plot: {plot_error}")
        log_event(
            pbt_log_path,
            f"run completed experiment={config['experiment_name']} "
            f"elapsed={format_duration(time.monotonic() - run_started_monotonic)} "
            f"directory={experiment_dir} "
            f"recommended_checkpoint={(manifest.get('best') or {}).get('state_path')}",
        )
        return 0
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        log_event(
            pbt_log_path,
            f"run {manifest['status']} experiment={config['experiment_name']} "
            f"elapsed={format_duration(time.monotonic() - run_started_monotonic)} "
            f"error={type(error).__name__}: {error}",
        )
        raise


def main():
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

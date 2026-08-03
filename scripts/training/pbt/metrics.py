#!/usr/bin/env python3
"""Metric comparison, health tracking, and global-best bookkeeping for PBT."""

from pathlib import Path

from training.pbt.artifacts import record_new_best
from training.pbt.checkpointing import (
    atomic_copy,
    checkpoint_paths,
    controller_checkpoint_path,
    global_best_paths,
)
from training.runtime import atomic_json, utc_now


def metric_is_better(config, candidate, incumbent):
    if incumbent is None:
        return True
    if config["pbt"]["mode"] == "max":
        return candidate > incumbent
    return candidate < incumbent

def metric_has_degraded(config, current, best):
    if best is None:
        return False
    tolerance = config["pbt"]["degradation_tolerance"]
    if config["pbt"]["mode"] == "max":
        return current < best * (1 - tolerance)
    return current > best * (1 + tolerance)

def relative_to_best(config, current, best):
    if best in (None, 0):
        return None
    if config["pbt"]["mode"] == "max":
        return current / best - 1
    return best / current - 1

def metric_is_worse_than_reference(config, current, reference, tolerance=0.0):
    if reference is None:
        return False
    tolerance = float(tolerance)
    if config["pbt"]["mode"] == "max":
        return current < reference * (1 - tolerance)
    return current > reference * (1 + tolerance)

def best_worker_in_generation(config, generation_record):
    metric_name = config["pbt"]["metric"]
    ranking = generation_record.get("ranking")
    if not ranking:
        reverse = config["pbt"]["mode"] == "max"
        ranking = sorted(
            generation_record["workers"],
            key=lambda name: generation_record["workers"][name]["metrics"][metric_name],
            reverse=reverse,
        )
    name = ranking[0]
    metrics = generation_record["workers"][name]["metrics"]
    return name, metrics[metric_name], metrics

def update_global_best(experiment_dir, manifest, generation_record, manifest_path):
    config = manifest["config"]
    metric_name = config["pbt"]["metric"]
    member_name, value, metrics = best_worker_in_generation(config, generation_record)
    if config["pbt"].get("baseline_guard_reject_global_best") and metric_is_worse_than_reference(
        config,
        value,
        config["pbt"].get("baseline_metric_value"),
        config["pbt"].get("baseline_guard_tolerance", 0.0),
    ):
        generation_record["baseline_rejected_global_best"] = {
            "member": member_name,
            "metric": metric_name,
            "metric_value": value,
            "baseline_metric": config["pbt"].get("baseline_metric_value"),
            "baseline_guard_tolerance": config["pbt"].get("baseline_guard_tolerance", 0.0),
            "reason": "worse_than_baseline",
        }
        return False
    current_best = manifest.get("best")
    if not metric_is_better(
        config,
        value,
        None if current_best is None else current_best["metric_value"],
    ):
        return False

    member_dir = experiment_dir / member_name
    state_path, optimizer_path = checkpoint_paths(member_dir, generation_record["epoch"])
    if not state_path.is_file() or not optimizer_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is incomplete: {member_name}")

    paths = global_best_paths(experiment_dir)
    atomic_copy(state_path, Path(paths["state_path"]))
    atomic_copy(optimizer_path, Path(paths["optimizer_path"]))

    controller_source = controller_checkpoint_path(member_dir, generation_record["epoch"])
    controller_path = Path(paths["controller_path"])
    has_controller = bool(config["shared"].get("training_controller")) and controller_source.is_file()
    if has_controller:
        atomic_copy(controller_source, controller_path)
    elif controller_path.exists():
        controller_path.unlink()

    best_record = {
        "generation": generation_record["index"],
        "epoch": generation_record["epoch"],
        "member": member_name,
        "metric": metric_name,
        "metric_value": value,
        "lr": float(manifest["members"][member_name]["lr"]),
        "metrics": metrics,
        "updated_at": utc_now(),
        "source_state_path": str(state_path),
        "source_optimizer_path": str(optimizer_path),
        "source_controller_path": str(controller_source) if has_controller else None,
        **paths,
    }
    manifest["best"] = best_record
    atomic_json(Path(paths["metadata_path"]), best_record)
    record_new_best(experiment_dir, manifest, generation_record, best_record)
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    return True

def update_generation_health(config, manifest, generation_record):
    member_name, value, _ = best_worker_in_generation(config, generation_record)
    best = manifest.get("best")
    best_value = None if best is None else best["metric_value"]
    degraded = metric_has_degraded(config, value, best_value)
    baseline_value = config["pbt"].get("baseline_metric_value")
    baseline_degraded = metric_is_worse_than_reference(
        config,
        value,
        baseline_value,
        config["pbt"].get("baseline_guard_tolerance", 0.0),
    )
    previous = [
        item
        for item in manifest.get("generations", [])
        if item["index"] < generation_record["index"]
    ]
    consecutive = 1 if degraded else 0
    if degraded:
        for item in reversed(previous):
            health = item.get("health") or {}
            if not health.get("degraded"):
                break
            consecutive += 1
    generation_record["health"] = {
        "current_best_member": member_name,
        "current_best_metric": value,
        "global_best_member": None if best is None else best["member"],
        "global_best_generation": None if best is None else best["generation"],
        "global_best_metric": best_value,
        "relative_to_global_best": relative_to_best(config, value, best_value),
        "baseline_metric": baseline_value,
        "relative_to_baseline": relative_to_best(config, value, baseline_value),
        "baseline_degraded": baseline_degraded,
        "baseline_guard_tolerance": config["pbt"].get("baseline_guard_tolerance", 0.0),
        "degraded": degraded,
        "consecutive_degraded_generations": consecutive,
        "status": "degraded" if consecutive >= config["pbt"]["degradation_window"] else "ok",
        "member_lrs": {
            name: float(record["lr"])
            for name, record in manifest.get("members", {}).items()
        },
    }
    return generation_record["health"]

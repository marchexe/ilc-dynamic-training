#!/usr/bin/env python3
"""Population Based Training state transitions and checkpoint bookkeeping."""

import math
import os
import random
import shutil
from pathlib import Path

from training.runtime import atomic_json, utc_now


def epoch_for_generation(config, generation):
    return (generation + 1) * int(config["shared"]["epochs_per_generation"]) - 1


def atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".pbt-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def checkpoint_paths(member_dir, epoch):
    prefix = member_dir / f"net_epoch-{epoch}"
    return Path(f"{prefix}_state.pt"), Path(f"{prefix}_optimizer.pt")


def controller_checkpoint_path(member_dir, epoch):
    return member_dir / f"net_epoch-{epoch}_controller.pt"


def global_best_paths(experiment_dir):
    return {
        "state_path": str(experiment_dir / "global_best_state.pt"),
        "optimizer_path": str(experiment_dir / "global_best_optimizer.pt"),
        "controller_path": str(experiment_dir / "global_best_controller.pt"),
        "metadata_path": str(experiment_dir / "global_best_metadata.json"),
    }


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
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    return True


def update_generation_health(config, manifest, generation_record):
    member_name, value, _ = best_worker_in_generation(config, generation_record)
    best = manifest.get("best")
    best_value = None if best is None else best["metric_value"]
    degraded = metric_has_degraded(config, value, best_value)
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
        "degraded": degraded,
        "consecutive_degraded_generations": consecutive,
        "status": "degraded" if consecutive >= config["pbt"]["degradation_window"] else "ok",
        "member_lrs": {
            name: float(record["lr"])
            for name, record in manifest.get("members", {}).items()
        },
    }
    return generation_record["health"]


def ranking_and_plan(config, generation_record, members):
    metric_name = config["pbt"]["metric"]
    reverse = config["pbt"]["mode"] == "max"
    ranking = sorted(
        members,
        key=lambda name: generation_record["workers"][name]["metrics"][metric_name],
        reverse=reverse,
    )
    count = max(1, math.floor(len(ranking) * config["pbt"]["exploit_fraction"]))
    count = min(count, len(ranking) // 2)
    donors = ranking[:count]
    recipients = ranking[-count:]
    rng = random.Random(int(config["pbt"]["seed"]) + generation_record["index"])
    plan = []
    for index, recipient in enumerate(recipients):
        donor = donors[index % len(donors)]
        factor = rng.choice(config["pbt"]["mutation_factors"])
        old_lr = float(members[recipient]["lr"])
        donor_lr = float(members[donor]["lr"])
        new_lr = min(
            config["pbt"]["max_lr"],
            max(config["pbt"]["min_lr"], donor_lr * factor),
        )
        plan.append(
            {
                "recipient": recipient,
                "donor": donor,
                "recipient_lr": old_lr,
                "donor_lr": donor_lr,
                "mutation_factor": factor,
                "new_lr": new_lr,
                "applied": False,
            }
        )
    return ranking, plan


def add_global_best_rollbacks(config, manifest, generation_record, members, plan):
    best = manifest.get("best")
    if not best or generation_record["index"] == int(config["shared"]["generations"]) - 1:
        return plan
    health = generation_record.get("health") or {}
    if health.get("status") != "degraded":
        return plan
    count = math.floor(len(generation_record["ranking"]) * config["pbt"]["rollback_fraction"])
    count = min(max(0, count), len(generation_record["ranking"]) // 2)
    if count == 0:
        return plan

    recipients = generation_record["ranking"][-count:]
    filtered = [event for event in plan if event["recipient"] not in recipients]
    for recipient in recipients:
        filtered.append(
            {
                "source": "global_best",
                "recipient": recipient,
                "donor": best["member"],
                "recipient_lr": float(members[recipient]["lr"]),
                "donor_lr": float(best["lr"]),
                "mutation_factor": 1.0,
                "new_lr": float(best["lr"]),
                "applied": False,
                "reason": "rollback_from_global_best",
                "global_best_generation": best["generation"],
            }
        )
    return filtered


def apply_exploit(experiment_dir, manifest, generation_record, manifest_path):
    epoch = generation_record["epoch"]
    for event in generation_record["exploit"]:
        if event["applied"]:
            continue
        recipient_dir = experiment_dir / event["recipient"]
        if event.get("source") == "global_best":
            paths = global_best_paths(experiment_dir)
            donor_state = Path(paths["state_path"])
            donor_optimizer = Path(paths["optimizer_path"])
            donor_controller = Path(paths["controller_path"])
        else:
            donor_dir = experiment_dir / event["donor"]
            donor_state, donor_optimizer = checkpoint_paths(donor_dir, epoch)
            donor_controller = controller_checkpoint_path(donor_dir, epoch)
        recipient_state, recipient_optimizer = checkpoint_paths(recipient_dir, epoch)
        if not donor_state.is_file() or not donor_optimizer.is_file():
            raise FileNotFoundError(f"Donor checkpoint is incomplete: {event['donor']}")
        atomic_copy(donor_state, recipient_state)
        atomic_copy(donor_optimizer, recipient_optimizer)
        shared_config = manifest.get("config", {}).get("shared", {})
        config_payload = manifest.get("config", {}).get("pbt", {})
        if shared_config.get("training_controller"):
            recipient_controller = controller_checkpoint_path(recipient_dir, epoch)
            if config_payload.get("controller_state_on_exploit", "copy") == "reset":
                if recipient_controller.exists():
                    recipient_controller.unlink()
            else:
                if not donor_controller.is_file():
                    raise FileNotFoundError(
                        f"Donor controller checkpoint is incomplete: {event['donor']}"
                    )
                atomic_copy(donor_controller, recipient_controller)
        member = manifest["members"][event["recipient"]]
        member["lr"] = event["new_lr"]
        member["parent"] = event["donor"]
        member["parent_source"] = event.get("source", "population")
        member["last_exploit_generation"] = generation_record["index"]
        event["applied"] = True
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)

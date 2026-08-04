#!/usr/bin/env python3
"""Population- and baseline-guard-triggered rollback injection into an exploit plan."""

import math

from training.pbt.metrics import clamp as _clamp
from training.pbt.metrics import metric_is_worse_than_reference
from training.pbt.models.events import normalize_exploit_plan


def strategy_uses_population_rollbacks(config):
    return config["pbt"].get("strategy", "exploit_mutate") != "fixed_lr_grid"


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
    return normalize_exploit_plan(filtered)


def add_baseline_guard_rollbacks(config, manifest, generation_record, members, plan):
    if config["pbt"].get("baseline_guard_action") != "rollback_to_initial":
        return plan
    if generation_record["index"] == int(config["shared"]["generations"]) - 1:
        return plan
    if config["pbt"].get("baseline_metric_value") is None or not config["shared"].get("initial_state"):
        return plan

    metric_name = config["pbt"]["metric"]
    baseline_value = config["pbt"].get("baseline_metric_value")
    tolerance = config["pbt"].get("baseline_guard_tolerance", 0.0)
    factor = float(config["pbt"].get("baseline_guard_lr_factor", 0.7))
    recipients = []
    for name, worker in generation_record.get("workers", {}).items():
        metrics = worker.get("metrics") or {}
        if metric_name not in metrics:
            continue
        value = metrics[metric_name]
        if metric_is_worse_than_reference(config, value, baseline_value, tolerance):
            recipients.append((name, value))
    if not recipients:
        return plan

    recipient_names = {name for name, _ in recipients}
    filtered = [event for event in plan if event["recipient"] not in recipient_names]
    guard_events = []
    for recipient, value in recipients:
        old_lr = float(members[recipient]["lr"])
        new_lr = _clamp(old_lr * factor, config["pbt"]["min_lr"], config["pbt"]["max_lr"])
        event = {
            "source": "initial_resume",
            "recipient": recipient,
            "donor": recipient,
            "recipient_lr": old_lr,
            "donor_lr": old_lr,
            "mutation_factor": factor,
            "new_lr": new_lr,
            "applied": False,
            "reason": "rollback_from_initial_resume_baseline_guard",
            "initial_epoch": int(config["shared"]["initial_epoch"]),
            "metric": metric_name,
            "metric_value": value,
            "baseline_metric": baseline_value,
            "baseline_guard_tolerance": tolerance,
        }
        filtered.append(event)
        guard_events.append(event.copy())
    generation_record["baseline_guard"] = {
        "action": "rollback_to_initial",
        "events": guard_events,
    }
    return normalize_exploit_plan(filtered)

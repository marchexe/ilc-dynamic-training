#!/usr/bin/env python3
"""PBT ranking, LR planning, and rollback policy helpers."""

import math
import random

from training.pbt.metrics import metric_is_worse_than_reference
from training.pbt.models.events import normalize_exploit_plan


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
                "source": "population",
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

def factors_from_radius(radius, count):
    radius = float(radius)
    if count < 2:
        raise ValueError("anchored_lr_sweep requires at least two members")
    if radius < 0:
        raise ValueError("LR radius must be non-negative")
    if count == 2:
        offsets = (radius, -radius)
    elif count == 4:
        offsets = (radius, radius / 2, -radius / 2, -radius)
    else:
        step = 2 * radius / (count - 1)
        offsets = [radius - index * step for index in range(count)]
    return [1.0 + offset for offset in offsets]

def lr_factors_for_population(config, member_names):
    count = len(member_names)
    radius_config = config["pbt"].get("lr_radius")
    if radius_config:
        return factors_from_radius(radius_config["initial"], count)

    factors = [float(value) for value in config["pbt"]["lr_factors"]]
    if len(factors) == count:
        return factors
    if count == 2 and len(factors) >= 2:
        return [factors[0], factors[-1]]
    if len(factors) < count:
        raise ValueError("anchored_lr_sweep requires at least one lr_factor per member")
    return factors[:count]

def previous_lr_radius_record(manifest, generation_index):
    if not manifest:
        return None
    previous = [
        item
        for item in manifest.get("generations", [])
        if item.get("index", -1) < generation_index and item.get("lr_radius")
    ]
    return previous[-1].get("lr_radius") if previous else None

def adaptive_lr_radius_state(config, generation_record, ranking, member_names, manifest=None):
    radius_config = config["pbt"].get("lr_radius")
    if not radius_config:
        return None

    previous = previous_lr_radius_record(manifest, generation_record["index"])
    radius = float(previous.get("next_radius", radius_config["initial"])) if previous else float(radius_config["initial"])
    minimum = float(radius_config["minimum"])
    shrink_factor = float(radius_config["shrink_factor"])
    shrink_after = int(radius_config["shrink_after_inner_wins"])
    keep_if_edge_wins = bool(radius_config.get("keep_if_edge_wins", True))

    anchor = ranking[0]
    anchor_position = member_names.index(anchor)
    edge_winner = anchor_position in {0, len(member_names) - 1}
    inner_winner = not edge_winner

    inner_win_generations = 1 if inner_winner else 0
    if previous and inner_winner:
        inner_win_generations += int(previous.get("inner_win_generations", 0))

    action = "keep"
    next_radius = radius
    if edge_winner and keep_if_edge_wins:
        inner_win_generations = 0
        action = "keep_edge_winner"
    elif inner_win_generations >= shrink_after and radius > minimum:
        next_radius = max(minimum, radius * shrink_factor)
        inner_win_generations = 0
        action = "shrink_inner_winners"

    factors = factors_from_radius(radius, len(member_names))
    return {
        "mode": "adaptive",
        "radius": radius,
        "next_radius": next_radius,
        "minimum": minimum,
        "shrink_factor": shrink_factor,
        "shrink_after_inner_wins": shrink_after,
        "inner_win_generations": inner_win_generations,
        "inner_winner": inner_winner,
        "edge_winner": edge_winner,
        "anchor_position": anchor_position,
        "action": action,
        "factors": factors,
    }

def anchored_lr_sweep_plan(config, generation_record, members, manifest=None):
    metric_name = config["pbt"]["metric"]
    reverse = config["pbt"]["mode"] == "max"
    member_names = list(members)
    ranking = sorted(
        member_names,
        key=lambda name: generation_record["workers"][name]["metrics"][metric_name],
        reverse=reverse,
    )
    anchor = ranking[0]
    anchor_lr = float(members[anchor]["lr"])
    anchor_metric = generation_record["workers"][anchor]["metrics"][metric_name]
    radius_state = adaptive_lr_radius_state(config, generation_record, ranking, member_names, manifest)
    factors = radius_state["factors"] if radius_state else lr_factors_for_population(config, member_names)
    if radius_state:
        generation_record["lr_radius"] = radius_state
    plan = []
    for index, recipient in enumerate(member_names):
        factor = factors[index]
        new_lr = min(
            config["pbt"]["max_lr"],
            max(config["pbt"]["min_lr"], anchor_lr * factor),
        )
        plan.append(
            {
                "source": "anchored_lr_sweep",
                "recipient": recipient,
                "donor": anchor,
                "anchor_member": anchor,
                "anchor_metric": anchor_metric,
                "recipient_lr": float(members[recipient]["lr"]),
                "donor_lr": anchor_lr,
                "lr_factor": factor,
                "lr_radius": None if radius_state is None else radius_state["radius"],
                "new_lr": new_lr,
                "applied": False,
            }
        )
    return ranking, plan

def fixed_lr_grid_plan(config, generation_record, members, manifest=None):
    ranking, _ = ranking_and_plan(config, generation_record, members)
    return ranking, []

def exploit_mutate_plan(config, generation_record, members, manifest=None):
    return ranking_and_plan(config, generation_record, members)

STRATEGY_PLANNERS = {
    "anchored_lr_sweep": anchored_lr_sweep_plan,
    "fixed_lr_grid": fixed_lr_grid_plan,
    "exploit_mutate": exploit_mutate_plan,
}


def plan_for_strategy(config, generation_record, members, manifest=None):
    strategy_name = config["pbt"].get("strategy", "exploit_mutate")
    try:
        planner = STRATEGY_PLANNERS[strategy_name]
    except KeyError as error:
        raise ValueError(f"Unsupported PBT strategy: {strategy_name}") from error
    ranking, plan = planner(config, generation_record, members, manifest)
    return ranking, normalize_exploit_plan(plan)

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
        new_lr = min(
            config["pbt"]["max_lr"],
            max(config["pbt"]["min_lr"], old_lr * factor),
        )
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

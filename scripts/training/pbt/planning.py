#!/usr/bin/env python3
"""PBT ranking, LR planning, and rollback policy helpers."""

import math
import random

from training.pbt.metrics import metric_is_worse_than_reference
from training.pbt.models.events import normalize_exploit_plan


def metric_uncertainty(metrics, metric_name):
    value = metrics.get(f"{metric_name}_uncertainty")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def member_metric_value(generation_record, member_name, metric_name):
    return float(generation_record["workers"][member_name]["metrics"][metric_name])


def previous_ranking_index(manifest, member_name):
    if not manifest:
        return None
    for generation in reversed(manifest.get("generations", [])):
        ranking = generation.get("ranking") or []
        if member_name in ranking:
            return ranking.index(member_name)
    return None


def stable_member_index(members, member_name):
    return list(members).index(member_name)


def ranking_tie_break_key(manifest, members, member_name):
    previous_index = previous_ranking_index(manifest, member_name)
    if previous_index is None:
        previous_index = stable_member_index(members, member_name)
    return previous_index, stable_member_index(members, member_name), member_name


def metric_difference(config, left_value, right_value):
    if config["pbt"]["mode"] == "max":
        return float(left_value) - float(right_value)
    return float(right_value) - float(left_value)


def raw_metric_sort_key(config, generation_record, member_name, metric_name):
    value = member_metric_value(generation_record, member_name, metric_name)
    if config["pbt"]["mode"] == "max":
        value = -value
    return value


def previous_anchor(manifest, members):
    if not manifest:
        return None
    for generation in reversed(manifest.get("generations", [])):
        ranking = generation.get("ranking") or []
        for member_name in ranking:
            if member_name in members:
                return member_name
    return None


def member_metric_uncertainty(generation_record, member_name, metric_name):
    metrics = generation_record["workers"][member_name]["metrics"]
    return metric_uncertainty(metrics, metric_name)


def has_all_member_uncertainties(generation_record, members, metric_name):
    return all(
        member_metric_uncertainty(generation_record, member_name, metric_name) is not None
        for member_name in members
    )


def statistically_beats(config, generation_record, candidate, incumbent, metric_name, sigma):
    candidate_value = member_metric_value(generation_record, candidate, metric_name)
    incumbent_value = member_metric_value(generation_record, incumbent, metric_name)
    candidate_uncertainty = member_metric_uncertainty(generation_record, candidate, metric_name)
    incumbent_uncertainty = member_metric_uncertainty(generation_record, incumbent, metric_name)
    if candidate_uncertainty is None or incumbent_uncertainty is None or sigma is None:
        return metric_difference(config, candidate_value, incumbent_value) > 0.0

    margin = float(sigma)
    if config["pbt"]["mode"] == "max":
        return (
            candidate_value - margin * candidate_uncertainty
            > incumbent_value + margin * incumbent_uncertainty
        )
    return (
        candidate_value + margin * candidate_uncertainty
        < incumbent_value - margin * incumbent_uncertainty
    )


def confidence_bound_sort_key(config, generation_record, members, member_name, metric_name, sigma, manifest=None):
    value = member_metric_value(generation_record, member_name, metric_name)
    uncertainty = member_metric_uncertainty(generation_record, member_name, metric_name) or 0.0
    margin = 0.0 if sigma is None else float(sigma) * uncertainty
    if config["pbt"]["mode"] == "max":
        score = -(value - margin)
    else:
        score = value + margin
    return score, ranking_tie_break_key(manifest, members, member_name)


def confidence_aware_ranking(config, generation_record, members, manifest=None):
    metric_name = config["pbt"]["metric"]
    pbt = config["pbt"]
    if not pbt.get("confidence_aware_selection", True):
        return sorted(
            members,
            key=lambda name: member_metric_value(generation_record, name, metric_name),
            reverse=pbt["mode"] == "max",
        )

    sigma = pbt.get("selection_uncertainty_sigma")
    if sigma is None or not has_all_member_uncertainties(generation_record, members, metric_name):
        ranking = sorted(
            members,
            key=lambda name: (
                raw_metric_sort_key(config, generation_record, name, metric_name),
                ranking_tie_break_key(manifest, members, name),
            ),
        )
        generation_record["selection"] = {
            "schema_version": 1,
            "mode": "raw_metric",
            "metric": metric_name,
            "uncertainty_sigma": sigma,
            "reason": "missing_uncertainty",
        }
        return ranking

    incumbent = previous_anchor(manifest, members)
    if incumbent is None:
        incumbent = min(
            members,
            key=lambda name: (
                raw_metric_sort_key(config, generation_record, name, metric_name),
                ranking_tie_break_key(manifest, members, name),
            ),
        )
    challenger_order = sorted(
        (name for name in members if name != incumbent),
        key=lambda name: (
            raw_metric_sort_key(config, generation_record, name, metric_name),
            ranking_tie_break_key(manifest, members, name),
        ),
    )
    for candidate in challenger_order:
        if statistically_beats(config, generation_record, candidate, incumbent, metric_name, sigma):
            incumbent = candidate

    remaining = sorted(
        (name for name in members if name != incumbent),
        key=lambda name: confidence_bound_sort_key(
            config,
            generation_record,
            members,
            name,
            metric_name,
            sigma,
            manifest,
        ),
    )
    ranking = [incumbent, *remaining]
    generation_record["selection"] = {
        "schema_version": 1,
        "mode": "confidence_aware",
        "metric": metric_name,
        "uncertainty_sigma": sigma,
        "anchor_policy": "incumbent_significance",
        "score": "conservative_confidence_bound",
        "anchor_member": incumbent,
    }
    return ranking


def ranking_and_plan(config, generation_record, members, manifest=None):
    metric_name = config["pbt"]["metric"]
    ranking = confidence_aware_ranking(config, generation_record, members, manifest)
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

def previous_lr_controller_record(manifest, generation_index):
    if not manifest:
        return None
    previous = [
        item
        for item in manifest.get("generations", [])
        if item.get("index", -1) < generation_index and item.get("lr_controller")
    ]
    return previous[-1].get("lr_controller") if previous else None

def _clamp(value, lower, upper):
    return min(upper, max(lower, value))

def _geometric_mean(values):
    positive_values = [float(value) for value in values if float(value) > 0]
    if not positive_values:
        raise ValueError("LR controller requires at least one positive learning rate")
    log_sum = sum(math.log(value) for value in positive_values)
    return math.exp(log_sum / len(positive_values))

def smooth_lr_controller_state(
    config,
    generation_record,
    ranking,
    member_names,
    members,
    anchor_lr,
    manifest=None,
):
    controller = config["pbt"].get("lr_controller")
    if not controller:
        return None

    previous = previous_lr_controller_record(manifest, generation_record["index"])
    min_lr = float(config["pbt"]["min_lr"])
    max_lr = float(config["pbt"]["max_lr"])
    if previous:
        old_center_lr = float(previous.get("center_lr", anchor_lr))
    else:
        old_center_lr = _geometric_mean(members[name]["lr"] for name in member_names)
    old_center_lr = _clamp(old_center_lr, min_lr, max_lr)

    target_lr = _clamp(float(anchor_lr), min_lr, max_lr)
    smoothing = float(controller.get("smoothing", 0.25))
    decay_bias = float(controller.get("decay_bias", 1.0))
    raw_center_lr = old_center_lr * math.pow(target_lr / old_center_lr, smoothing)
    biased_center_lr = raw_center_lr * decay_bias

    max_center_decrease = float(controller.get("max_center_decrease", 0.80))
    max_center_increase = float(controller.get("max_center_increase", 1.25))
    center_lr = _clamp(
        biased_center_lr,
        old_center_lr * max_center_decrease,
        old_center_lr * max_center_increase,
    )
    center_lr = _clamp(center_lr, min_lr, max_lr)

    return {
        "mode": controller.get("mode", "smooth"),
        "anchor_member": ranking[0],
        "old_center_lr": old_center_lr,
        "target_lr": target_lr,
        "raw_center_lr": raw_center_lr,
        "center_lr": center_lr,
        "smoothing": smoothing,
        "decay_bias": decay_bias,
        "max_center_increase": max_center_increase,
        "max_center_decrease": max_center_decrease,
        "max_member_increase": float(controller.get("max_member_increase", 1.35)),
        "max_member_decrease": float(controller.get("max_member_decrease", 0.75)),
        "previous_generation": None if previous is None else previous.get("generation"),
    }

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
    member_names = list(members)
    ranking = confidence_aware_ranking(config, generation_record, members, manifest)
    anchor = ranking[0]
    anchor_lr = float(members[anchor]["lr"])
    anchor_metric = generation_record["workers"][anchor]["metrics"][metric_name]
    radius_state = adaptive_lr_radius_state(config, generation_record, ranking, member_names, manifest)
    factors = radius_state["factors"] if radius_state else lr_factors_for_population(config, member_names)
    if radius_state:
        generation_record["lr_radius"] = radius_state
    controller_state = smooth_lr_controller_state(
        config,
        generation_record,
        ranking,
        member_names,
        members,
        anchor_lr,
        manifest,
    )
    if controller_state:
        controller_state["generation"] = generation_record["index"]
        generation_record["lr_controller"] = controller_state
    center_lr = controller_state["center_lr"] if controller_state else anchor_lr
    plan = []
    for index, recipient in enumerate(member_names):
        factor = factors[index]
        proposed_lr = center_lr * factor
        member_step_clamped = False
        if controller_state:
            old_lr = float(members[recipient]["lr"])
            proposed_lr_before_member_clamp = proposed_lr
            proposed_lr = _clamp(
                proposed_lr,
                old_lr * controller_state["max_member_decrease"],
                old_lr * controller_state["max_member_increase"],
            )
            member_step_clamped = proposed_lr != proposed_lr_before_member_clamp
        new_lr = _clamp(proposed_lr, config["pbt"]["min_lr"], config["pbt"]["max_lr"])
        donor = anchor if config["pbt"].get("anchored_weight_source", "anchor") == "anchor" else recipient
        plan.append(
            {
                "source": "anchored_lr_sweep",
                "recipient": recipient,
                "donor": donor,
                "anchor_member": anchor,
                "anchor_metric": anchor_metric,
                "recipient_lr": float(members[recipient]["lr"]),
                "donor_lr": float(members[donor]["lr"]),
                "weight_source": config["pbt"].get("anchored_weight_source", "anchor"),
                "lr_center": center_lr,
                "lr_factor": factor,
                "lr_radius": None if radius_state is None else radius_state["radius"],
                "lr_controller": None if controller_state is None else controller_state["mode"],
                "member_lr_step_clamped": member_step_clamped,
                "new_lr": new_lr,
                "applied": False,
            }
        )
    return ranking, plan

def fixed_lr_grid_plan(config, generation_record, members, manifest=None):
    ranking, _ = ranking_and_plan(config, generation_record, members)
    return ranking, []

def exploit_mutate_plan(config, generation_record, members, manifest=None):
    return ranking_and_plan(config, generation_record, members, manifest)

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

#!/usr/bin/env python3
"""The anchored_lr_sweep strategy: LR grid centered on the ranking anchor, optionally smoothed."""

import math

from training.pbt.metrics import clamp as _clamp
from training.pbt.planning.ranking import confidence_aware_ranking
from training.pbt.state.checkpointing import generations_before


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
    previous = [item for item in generations_before(manifest, generation_index) if item.get("lr_radius")]
    return previous[-1].get("lr_radius") if previous else None


def previous_lr_controller_record(manifest, generation_index):
    if not manifest:
        return None
    previous = [item for item in generations_before(manifest, generation_index) if item.get("lr_controller")]
    return previous[-1].get("lr_controller") if previous else None


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

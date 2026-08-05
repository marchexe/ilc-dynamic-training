#!/usr/bin/env python3
"""The population_lr_policy strategy: infer an LR direction from a matched,
low-noise population-wide proxy-tier round, copy weights+optimizer from the
winning direction's best member to everyone unconditionally, and roll back
to the pre-decision checkpoint if the following round is worse for the whole
population than it was before the change.

Unlike exploit_mutate (per-pair significance-gated donor->recipient copy,
random mutation_factors, driven by the every-generation control-tier eval),
this strategy makes one population-wide decision per proxy-tier round
(monitor by default -- already computed on tiered_validation.
monitor_interval_generations cadence, at a much larger, lower-noise sample
than control), never gates the copy itself on significance, and always
either confirms or reverts the previous decision before considering a new
one. See planning/ranking.py and state/transitions.py -- this module builds
plan events in the same shape those already understand; it adds no new
copy/checkpoint machinery.
"""

import math

from training.pbt.metrics import clamp as _clamp
from training.pbt.metrics import metric_is_worse_than_reference
from training.pbt.planning.ranking import (
    confidence_aware_ranking,
    in_burn_in,
    metric_difference,
    metric_uncertainty,
)
from training.pbt.state.checkpointing import generations_before


def population_lr_policy_config(config):
    policy = config.get("pbt", {}).get("population_lr_policy")
    if not policy or policy.get("mode") == "disabled":
        return None
    return policy


def tier_round_for_generation(manifest, generation_index, tier):
    for round_record in manifest.get("tiered_evaluations", []) or []:
        if round_record.get("generation") == generation_index and round_record.get("tier") == tier:
            return round_record
    return None


def _round_member_metric(round_record, member_name, metric_name):
    member = (round_record.get("members") or {}).get(member_name) or {}
    metrics = member.get("metrics") or {}
    value = metrics.get(metric_name)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _round_member_uncertainty(round_record, member_name, metric_name):
    member = (round_record.get("members") or {}).get(member_name) or {}
    return metric_uncertainty(member.get("metrics") or {}, metric_name)


def _best_in_group(config, round_record, group, metric_name):
    mode = config["pbt"]["mode"]
    candidates = [(name, _round_member_metric(round_record, name, metric_name)) for name in group]
    candidates = [(name, value) for name, value in candidates if value is not None]
    if not candidates:
        return None, None
    picker = max if mode == "max" else min
    return picker(candidates, key=lambda pair: pair[1])


def _split_by_lr(members):
    """Lower/upper halves of the population by current LR (stable tie-break
    by name). A member sitting exactly on the median with an odd population
    size belongs to neither half -- it simply isn't compared this round.
    """
    names = sorted(members, key=lambda name: (float(members[name]["lr"]), name))
    half = len(names) // 2
    if half == 0:
        return [], []
    return names[:half], names[-half:]


def infer_direction(config, round_record, members, policy):
    metric_name = config["pbt"]["metric"]
    low, high = _split_by_lr(members)
    if not low or not high:
        return "keep", None, None
    low_name, low_value = _best_in_group(config, round_record, low, metric_name)
    high_name, high_value = _best_in_group(config, round_record, high, metric_name)
    if low_value is None or high_value is None:
        return "keep", None, None

    # Positive => the high-LR half's best beats the low-LR half's best,
    # oriented for config["pbt"]["mode"] the same way ranking.py's
    # statistically_beats/confidence_bound_sort_key already are.
    diff = metric_difference(config, high_value, low_value)
    sigma = policy.get("direction_sigma")
    threshold = 0.0
    if sigma is not None:
        low_unc = _round_member_uncertainty(round_record, low_name, metric_name) or 0.0
        high_unc = _round_member_uncertainty(round_record, high_name, metric_name) or 0.0
        threshold = float(sigma) * math.sqrt(low_unc**2 + high_unc**2)

    if diff > threshold:
        return "up", high_name, diff
    if -diff > threshold:
        return "down", low_name, diff
    return "keep", None, diff


def _build_forward_events(config, generation_record, members, direction, donor_name, margin, metric_before, policy):
    factor = float(policy["up_factor"] if direction == "up" else policy["down_factor"])
    min_lr = float(config["pbt"]["min_lr"])
    max_lr = float(config["pbt"]["max_lr"])
    donor_lr = float(members[donor_name]["lr"])
    events = []
    for name, member in members.items():
        old_lr = float(member["lr"])
        new_lr = _clamp(old_lr * factor, min_lr, max_lr)
        events.append(
            {
                "source": "population_lr_policy",
                "recipient": name,
                "donor": donor_name,
                "recipient_lr": old_lr,
                "donor_lr": donor_lr,
                "new_lr": new_lr,
                "direction": direction,
                "factor": factor,
                "margin_sigma": margin,
                "decision_generation": generation_record["index"],
                "decision_epoch": generation_record["epoch"],
                "metric_before": metric_before,
                "eval_tier": policy.get("eval_tier", "monitor"),
                "applied": False,
            }
        )
    return events


def find_pending_decision(manifest, generation_index):
    """Most recent applied population_lr_policy decision that has not yet
    been resolved by a later applied population_lr_policy_resolution event
    for the same decision_generation -- reconstructed by scanning history
    (same pattern as ranking.py's previous_anchor / observation.py's
    last_action_epoch_fraction) instead of separate mutable manifest state,
    so a decision whose plan never actually got applied (burn-in/final
    generation/early-stop all discard the whole plan in runner.py) is
    correctly invisible here too.
    """
    pending_generation = None
    events_by_generation = {}
    for generation in generations_before(manifest, generation_index):
        for event in generation.get("exploit") or []:
            if not event.get("applied"):
                continue
            source = event.get("source")
            if source == "population_lr_policy":
                gen = event.get("decision_generation")
                events_by_generation.setdefault(gen, []).append(event)
                pending_generation = gen
            elif source == "population_lr_policy_resolution":
                if event.get("decision_generation") == pending_generation:
                    pending_generation = None
    if pending_generation is None:
        return None
    return events_by_generation.get(pending_generation)


def _build_resolution_events(config, manifest, decision_events, round_record):
    metric_name = config["pbt"]["metric"]
    metric_before = decision_events[0]["metric_before"]
    decision_generation = decision_events[0]["decision_generation"]
    rollback_epoch = decision_events[0]["decision_epoch"]
    recipients = [event["recipient"] for event in decision_events]

    best_name, best_value = _best_in_group(config, round_record, recipients, metric_name)
    if best_value is None:
        return [], None

    accepted = not metric_is_worse_than_reference(config, best_value, metric_before, tolerance=0.0)
    outcome = "accepted" if accepted else "rolled_back"
    events = []
    for event in decision_events:
        name = event["recipient"]
        current_lr = float(manifest["members"][name]["lr"])
        events.append(
            {
                "source": "population_lr_policy_resolution",
                "outcome": outcome,
                "recipient": name,
                "donor": name,
                "recipient_lr": current_lr,
                "donor_lr": current_lr,
                "new_lr": current_lr if accepted else event["recipient_lr"],
                "decision_generation": decision_generation,
                "rollback_epoch": rollback_epoch,
                "metric_before": metric_before,
                "metric_after": best_value,
                "applied": False,
            }
        )
    return events, outcome


def population_lr_policy_plan(config, generation_record, members, manifest=None):
    ranking = confidence_aware_ranking(config, generation_record, members, manifest)
    policy = population_lr_policy_config(config)
    if not policy or manifest is None:
        return ranking, []
    if in_burn_in(config, generation_record["index"]):
        return ranking, []

    tier = policy.get("eval_tier", "monitor")
    round_record = tier_round_for_generation(manifest, generation_record["index"], tier)
    if round_record is None:
        # No fresh proxy-tier round this generation -- nothing to decide on;
        # LR/weights stay exactly as they are until the next one is due.
        return ranking, []

    plan = []
    status = {"schema_version": 1, "eval_tier": tier, "generation": generation_record["index"]}

    pending = find_pending_decision(manifest, generation_record["index"])
    if pending is not None:
        resolution_events, outcome = _build_resolution_events(config, manifest, pending, round_record)
        plan.extend(resolution_events)
        status["resolved"] = {
            "decision_generation": pending[0]["decision_generation"],
            "outcome": outcome,
            "metric_before": pending[0]["metric_before"],
        }
        if outcome == "rolled_back":
            # The just-computed round describes the failed post-decision
            # state we're discarding -- it says nothing about the restored
            # state's own trajectory, so don't use it to start a new
            # decision. Wait for the next fresh round.
            generation_record["population_lr_policy"] = status
            return ranking, plan

    metric_name = config["pbt"]["metric"]
    _, population_best = _best_in_group(config, round_record, list(members), metric_name)
    direction, donor_name, margin = infer_direction(config, round_record, members, policy)
    if direction == "keep" or population_best is None:
        status["decision"] = "keep"
        generation_record["population_lr_policy"] = status
        return ranking, plan

    plan.extend(
        _build_forward_events(
            config, generation_record, members, direction, donor_name, margin, population_best, policy
        )
    )
    status["decision"] = direction
    status["donor"] = donor_name
    status["margin_sigma"] = margin
    status["metric_before"] = population_best
    generation_record["population_lr_policy"] = status
    return ranking, plan

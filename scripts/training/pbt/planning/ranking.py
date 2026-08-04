#!/usr/bin/env python3
"""Population ranking primitives and PBT exploit cadence gates."""

import math


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


def raw_metric_ranking(config, generation_record, members):
    """Plain metric-value ordering (best first), no confidence adjustment.

    Used for cross-tier (control/monitor/full) ranking-agreement diagnostics,
    where comparing against the decision-making confidence-aware ranking
    would bias the comparison via its incumbent-persistence behavior.
    """
    metric_name = config["pbt"]["metric"]
    return sorted(
        members,
        key=lambda name: raw_metric_sort_key(config, generation_record, name, metric_name),
    )


def in_burn_in(config, generation):
    burn_in_generations = int(config["pbt"].get("burn_in_generations", 0) or 0)
    return int(generation) < burn_in_generations


def should_apply_exploit(config, generation, is_final_generation, early_stop_triggered):
    """Whether this generation's PBT exploit plan (donor->recipient
    weight+optimizer copy) should actually be applied. False during burn-in,
    on the final generation, once early stopping has triggered, or -- the
    real cadence gate -- on any generation not due per
    `exploit_interval_generations` (default: every generation, if unset).

    This is deliberately independent of the fine dynamic_controller's LR
    nudges, which are applied every non-burn-in generation regardless (see
    runner.py/apply_controller_actions_to_members) -- that's what makes the
    controller genuinely more frequent than PBT exploit when
    exploit_interval_generations > 1, not just a config field nobody reads.
    """
    if in_burn_in(config, generation):
        return False
    if early_stop_triggered:
        return False
    if is_final_generation:
        return False
    interval = int(config["pbt"].get("exploit_interval_generations") or 1)
    if (int(generation) + 1) % interval != 0:
        return False
    return True

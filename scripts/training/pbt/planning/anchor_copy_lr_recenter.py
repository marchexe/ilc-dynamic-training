#!/usr/bin/env python3
"""anchor_copy_lr_recenter: an isolated PBT strategy.

Every generation (no burn-in exception, no significance gate): select the
best-finite-metric stream, classify it against a single persisted
population anchor (accepted_new_anchor / reused_previous_anchor /
rewound_to_previous_anchor), recenter one common LR center toward -- or, on
rewind, away from -- the winner's LR, copy the resulting anchor state to
every stream including the winner, and assign a deterministic LR spread
around the new center.

Reuses rather than reimplements: raw_metric_ranking (planning/ranking.py --
the existing plain best-finite-metric ranking already used for cross-tier
diagnostics), finite_metric_ok (execution/backend.py -- the existing
NaN/Inf/missing-metric guard), metric_is_worse_than_reference (metrics.py --
the existing orientation-safe tolerance comparator, applied symmetrically in
both directions here to get a three-way split instead of its native
two-way one), and clamp (metrics.py).

Explicitly does NOT reuse population_lr_policy's algorithm (median-LR-half
split, per-member own-LR*factor scaling, per-recipient snapshot rollback,
periodic-tier-gated decisions) -- population_lr_policy is a technical
template for "planner -> typed plan -> unmodified apply_exploit" wiring
only, not for this strategy's semantics. In particular: the LR center here
is a single shared value multiplied out to every stream
(new_center * spread_multipliers[i]), never a per-stream independent scale
of that stream's own previous LR; and the anchor is one persisted bundle
restored to everyone on regression, never per-recipient snapshots.
"""

from training.pbt.execution.backend import finite_metric_ok
from training.pbt.metrics import clamp, metric_is_worse_than_reference
from training.pbt.planning.ranking import member_metric_value, raw_metric_ranking

ANCHOR_PSEUDO_RECIPIENT = "__anchor__"
EVAL_TIER = "control"  # always the per-generation control-tier metric (control_proxy_50k); never a separate periodic tier


def anchor_copy_lr_recenter_config(config):
    policy = config.get("pbt", {}).get("anchor_copy_lr_recenter")
    if not policy or policy.get("mode") == "disabled":
        return None
    return policy


def _finite_members(config, generation_record, members):
    """Members with a completed worker and a finite metric this
    generation, in the same order `members` was given. A missing, failed,
    NaN, or Inf metric excludes a member here -- it can never become the
    winner and can never itself trigger an LR/anchor action."""
    metric_name = config["pbt"]["metric"]
    finite = []
    for name in members:
        worker = (generation_record.get("workers") or {}).get(name)
        if not worker or worker.get("status") != "completed":
            continue
        if not finite_metric_ok(worker.get("metrics"), metric_name):
            continue
        finite.append(name)
    return finite


def classify_anchor_decision(config, tolerance, winner_metric, anchor_metric):
    """Three-way accept/reuse/rewind split, built from two applications of
    the existing metric_is_worse_than_reference (never a new absolute-units
    comparator): winner strictly worse than anchor beyond tolerance ->
    rewind; anchor strictly worse than winner beyond tolerance (i.e. winner
    strictly better) -> accept; neither -> reuse (tie zone)."""
    if anchor_metric is None:
        return "accepted_new_anchor"
    if metric_is_worse_than_reference(config, winner_metric, anchor_metric, tolerance):
        return "rewound_to_previous_anchor"
    if metric_is_worse_than_reference(config, anchor_metric, winner_metric, tolerance):
        return "accepted_new_anchor"
    return "reused_previous_anchor"


def assign_lr_spread(center, multipliers, min_lr, max_lr, member_order):
    """Deterministic per-member LR = clamp(center * multiplier). Never a
    per-member scale of that member's own previous LR -- every stream's
    next LR is derived from the one shared center, matching the "true
    common LR center" requirement."""
    return {
        name: clamp(center * multiplier, min_lr, max_lr)
        for name, multiplier in zip(member_order, multipliers)
    }


def anchor_copy_lr_recenter_plan(config, generation_record, members, manifest=None):
    policy = anchor_copy_lr_recenter_config(config)
    if not policy or manifest is None:
        return [], []

    metric_name = config["pbt"]["metric"]
    finite_names = _finite_members(config, generation_record, members)
    if not finite_names:
        generation_record["anchor_copy_lr_recenter"] = {
            "decision": "no_finite_metric",
            "reason": "no member produced a finite control-tier metric this generation",
        }
        return [], []

    ranking = raw_metric_ranking(config, generation_record, finite_names)
    winner_name = ranking[0]
    winner_metric = member_metric_value(generation_record, winner_name, metric_name)
    winner_lr = float(members[winner_name]["lr"])

    anchor = manifest.get("anchor")
    anchor_metric = anchor["metric_value"] if anchor else None
    prev_center = float(anchor["lr_center"]) if anchor else winner_lr

    tolerance = float(policy.get("accept_tolerance", 0.0))
    decision = classify_anchor_decision(config, tolerance, winner_metric, anchor_metric)

    min_lr = float(config["pbt"]["min_lr"])
    max_lr = float(config["pbt"]["max_lr"])
    center_step_fraction = float(policy.get("center_step_fraction", 0.3))

    if decision == "rewound_to_previous_anchor":
        # Restored, not moved: this generation's result carries no
        # information we act on for the center when it's being rejected.
        new_center = float(anchor["lr_center"])
    else:
        # accepted_new_anchor and reused_previous_anchor move the center
        # identically -- the winner's LR still carries information about
        # which LR performed best even when its checkpoint isn't (yet)
        # better enough to replace the anchor outright.
        new_center = clamp(prev_center + center_step_fraction * (winner_lr - prev_center), min_lr, max_lr)

    if decision == "accepted_new_anchor":
        anchor_donor_label = winner_name
        anchor_generation = generation_record["index"]
        anchor_metric_for_events = winner_metric
    elif anchor is not None:
        anchor_donor_label = anchor["member"]
        anchor_generation = anchor["generation"]
        anchor_metric_for_events = anchor["metric_value"]
    else:  # pragma: no cover -- anchor is None only alongside "accepted_new_anchor" above
        anchor_donor_label = winner_name
        anchor_generation = generation_record["index"]
        anchor_metric_for_events = winner_metric

    common_fields = {
        "source": "anchor_copy_lr_recenter",
        "decision": decision,
        "anchor_generation": anchor_generation,
        "anchor_metric_value": anchor_metric_for_events,
        "winner": winner_name,
        "winner_metric_value": winner_metric,
        "lr_center": new_center,
    }

    # The special "__anchor__" event must be first: apply_exploit processes
    # events in order and persists the manifest after each one, so by the
    # time any real member's event runs, manifest["anchor"] already
    # reflects this generation's decision -- accept and rewind then share
    # identical per-member distribution code below.
    anchor_event = {
        **common_fields,
        "recipient": ANCHOR_PSEUDO_RECIPIENT,
        "donor": anchor_donor_label,
        "new_lr": new_center,
    }

    member_order = list(members)
    spread = assign_lr_spread(new_center, policy["spread_multipliers"], min_lr, max_lr, member_order)

    plan = [anchor_event]
    for name in member_order:
        plan.append(
            {
                **common_fields,
                "recipient": name,
                "donor": anchor_donor_label,
                "new_lr": spread[name],
            }
        )

    generation_record["anchor_copy_lr_recenter"] = {
        "decision": decision,
        "winner": winner_name,
        "winner_metric_value": winner_metric,
        "anchor_metric_value": anchor_metric_for_events,
        "previous_lr_center": prev_center,
        "new_lr_center": new_center,
        "assigned_lrs": spread,
        "eval_tier": EVAL_TIER,
    }

    return ranking, plan

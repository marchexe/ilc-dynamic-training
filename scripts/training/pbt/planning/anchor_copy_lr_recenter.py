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
from training.pbt.planning.ranking import member_metric_value, raw_metric_ranking, should_apply_exploit

ANCHOR_PSEUDO_RECIPIENT = "__anchor__"
EVAL_TIER = "control"  # always the per-generation control-tier metric (control_proxy_50k); never a separate periodic tier
STRATEGY_NAME = "anchor_copy_lr_recenter"


def anchor_copy_lr_recenter_config(config):
    policy = config.get("pbt", {}).get("anchor_copy_lr_recenter")
    if not policy or policy.get("mode") == "disabled":
        return None
    return policy


def should_apply_exploit_for_strategy(config, generation, is_final_generation, early_stop_triggered):
    """should_apply_exploit (planning/ranking.py, unmodified) skips applying
    the exploit plan on the run's final generation for every strategy --
    correct for exploit_mutate/population_lr_policy/anchored_lr_sweep,
    where nothing trains after a copy that would otherwise happen there, so
    skipping it is harmless. anchor_copy_lr_recenter's spec requires the
    complete plan (the "__anchor__" bundle update/reuse/rewind AND every
    member's copy) to apply on every generation including the last one, so
    this strategy alone gets is_final_generation forced to False before
    delegating to the real, untouched should_apply_exploit -- burn-in,
    early-stop, and cadence-interval gating all still apply to it exactly
    as before (this strategy's own preset already sets burn_in_generations:
    0 and exploit_interval_generations: 1, so those gates are no-ops for it
    in practice, but they are not bypassed here)."""
    if config.get("pbt", {}).get("strategy") == STRATEGY_NAME:
        is_final_generation = False
    return should_apply_exploit(config, generation, is_final_generation, early_stop_triggered)


def detect_spread_collapse(spread):
    """min_lr/max_lr clamping can push two or more members' otherwise-
    distinct multiplier*center values onto the exact same bound, silently
    collapsing what looks like a full distinct LR grid into fewer genuinely
    different values. clamp() returns the bound itself exactly (no
    floating-point fuzz) when it clamps, so exact equality is the correct,
    sufficient check -- never treat a collapsed spread as if every member
    still got a distinct LR. Returns (collapsed: bool, duplicate_groups:
    list[list[str]]) -- duplicate_groups lists only the member-name groups
    that actually share a value, sorted for deterministic output; empty
    when nothing collapsed."""
    by_value = {}
    for name, lr in spread.items():
        by_value.setdefault(lr, []).append(name)
    duplicate_groups = sorted(
        (sorted(names) for names in by_value.values() if len(names) > 1),
        key=lambda group: group[0],
    )
    return bool(duplicate_groups), duplicate_groups


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
    common LR center" requirement. Returns (clamped, unclamped) dicts --
    both recorded so a collapsed spread (see detect_spread_collapse) can be
    explained from the manifest alone, without cross-referencing
    spread_multipliers/min_lr/max_lr from the resolved config."""
    unclamped = {name: center * multiplier for name, multiplier in zip(member_order, multipliers)}
    clamped = {name: clamp(value, min_lr, max_lr) for name, value in unclamped.items()}
    return clamped, unclamped


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

    if decision == "rewound_to_previous_anchor":
        # Restored, not moved: this generation's result carries no
        # information we act on for the center when it's being rejected.
        new_center = float(anchor["lr_center"])
    else:
        # accepted_new_anchor and reused_previous_anchor both set the
        # center to exactly the winner's own LR -- no damping/blending
        # toward it. Direction is never inferred separately: it falls out
        # naturally from whether winner_lr sits above, below, or at the
        # old center.
        new_center = clamp(winner_lr, min_lr, max_lr)

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

    member_order = list(members)
    spread, unclamped_spread = assign_lr_spread(new_center, policy["spread_multipliers"], min_lr, max_lr, member_order)
    spread_collapsed, duplicate_lr_groups = detect_spread_collapse(spread)

    common_fields = {
        "source": "anchor_copy_lr_recenter",
        "decision": decision,
        "anchor_generation": anchor_generation,
        "anchor_metric_value": anchor_metric_for_events,
        "winner": winner_name,
        "winner_metric_value": winner_metric,
        "winner_lr": winner_lr,
        "lr_center": new_center,
        "spread_collapsed": spread_collapsed,
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
        "recipient_lr": prev_center,
        "new_lr": new_center,
        "unclamped_lr": new_center,
    }

    plan = [anchor_event]
    for name in member_order:
        plan.append(
            {
                **common_fields,
                "recipient": name,
                "donor": anchor_donor_label,
                "recipient_lr": float(members[name]["lr"]),
                "new_lr": spread[name],
                "unclamped_lr": unclamped_spread[name],
            }
        )

    generation_record["anchor_copy_lr_recenter"] = {
        "decision": decision,
        "winner": winner_name,
        "winner_metric_value": winner_metric,
        "winner_lr": winner_lr,
        "anchor_metric_value": anchor_metric_for_events,
        "previous_lr_center": prev_center,
        "new_lr_center": new_center,
        "assigned_lrs": spread,
        "unclamped_lrs": unclamped_spread,
        "eval_tier": EVAL_TIER,
        # min_lr/max_lr clamping can collapse two or more members onto the
        # exact same LR (e.g. center near a bound with a wide multiplier
        # spread) -- recorded explicitly here rather than silently letting
        # the assigned_lrs dict look like a full distinct grid when it
        # isn't. See detect_spread_collapse's docstring.
        "spread_collapsed": spread_collapsed,
        "duplicate_lr_groups": duplicate_lr_groups,
    }

    return ranking, plan

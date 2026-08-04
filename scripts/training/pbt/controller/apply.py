#!/usr/bin/env python3
"""Per-generation controller entrypoints: observe the population, then apply decisions."""

from training.pbt.controller.decision import build_controller_action, dynamic_controller_config
from training.pbt.controller.observation import build_observation
from training.pbt.models.events import normalize_exploit_plan


def run_generation_controller(config, manifest, generation_record, experiment_dir=None):
    controller = dynamic_controller_config(config)
    if not controller:
        generation_record.pop("controller_observations", None)
        generation_record.pop("controller_actions", None)
        generation_record.pop("dynamic_controller", None)
        return None

    observations = {}
    actions = {}
    metric_name = config["pbt"]["metric"]
    for member_name, worker in generation_record.get("workers", {}).items():
        metrics = worker.get("metrics") or {}
        if worker.get("status") != "completed" or metric_name not in metrics:
            continue
        observation = build_observation(config, manifest, generation_record, member_name, experiment_dir)
        observations[member_name] = observation
        actions[member_name] = build_controller_action(config, observation)

    generation_record["controller_observations"] = observations
    generation_record["controller_actions"] = actions
    generation_record["dynamic_controller"] = {
        "schema_version": 1,
        "mode": controller.get("mode", "active"),
        "eval_interval_fraction": controller.get("eval_interval_fraction", 0.20),
        "action_interval_fraction": controller.get("action_interval_fraction", 0.20),
        "generation_epoch_fraction": controller.get("generation_epoch_fraction"),
        "ema_beta": controller.get("ema_beta", 0.7),
        "trend_window": controller.get("trend_window", 3),
        "applied": False,
        "action_count": len(actions),
    }
    return generation_record["dynamic_controller"]


def apply_actions_to_plan(config, generation_record, plan):
    """PBT exploit recipients are fully owned by the PBT plan: model state,
    optimizer state, and LR (donor_lr * mutation_factor) all come from the
    donor. The dynamic controller must never touch a recipient's LR here --
    it would otherwise pair donor weights with an LR computed from the
    recipient's own pre-exploit trend, unrelated to the donor (real incident:
    generation 2 of the bnfreeze pilot replaced a PBT proposal of 1.32e-5
    with a controller value of 4.05e-6 derived from the recipient's stale
    LR). This is therefore a pure annotation pass -- it records that the
    controller was excluded, but it never modifies `new_lr`.

    Must run before any rollback events (global_best / baseline_guard) are
    appended to `plan`: those events declare their own `reason` field, and
    this pass would clobber it if it ran after.
    """
    controller = dynamic_controller_config(config)
    if not controller:
        return plan

    updated = []
    for event in plan:
        event = dict(event)
        final_lr = float(event["new_lr"])
        event["pbt_proposed_lr"] = final_lr
        event["final_lr"] = final_lr
        event["controller_applied"] = False
        event["reason"] = "exploit_recipient_owned_by_pbt"
        updated.append(event)

    dynamic_controller_record = generation_record.get("dynamic_controller")
    if dynamic_controller_record is not None:
        dynamic_controller_record["applied"] = False
        dynamic_controller_record["applied_action_count"] = 0

    return normalize_exploit_plan(updated)


def apply_controller_actions_to_members(config, manifest, generation_record, exclude_members=None):
    """Apply each eligible member's ready controller action directly to its
    own LR (manifest["members"][name]["lr"]), independent of PBT exploit.

    `apply_actions_to_plan` above never applies a controller action to a
    plan event -- exploit recipients' LR is owned entirely by the PBT plan
    (donor_lr * mutation_factor), never by the controller. This function is
    the *only* place a controller action ever changes a member's LR, and it
    runs every non-burn-in generation, not just the (less frequent, once
    exploit_interval_generations > 1) generations where population-level
    exploit also happens.

    `exclude_members` should be the current generation's exploit recipients
    (if exploit is firing this generation) -- they're about to have their
    weights, optimizer state, AND lr overwritten by the donor copy, so
    nudging them here first would just be immediately-discarded work and a
    confusing duplicate log entry.
    """
    controller = dynamic_controller_config(config)
    if not controller or controller.get("mode") != "active":
        return {}
    exclude_members = set(exclude_members or ())
    actions = generation_record.get("controller_actions") or {}
    applied = {}
    for member_name, action in actions.items():
        if member_name in exclude_members:
            continue
        if not action.get("action_ready") or action.get("safety_check") not in {"passed", "clamped"}:
            continue
        if action.get("action") in {"keep", "flag_review"}:
            continue
        member = manifest["members"].get(member_name)
        if member is None:
            continue
        old_lr = float(member["lr"])
        new_lr = float(action["bounded_lr"])
        if new_lr == old_lr:
            continue
        member["lr"] = new_lr
        action["applied"] = True
        applied[member_name] = {"action": action["action"], "old_lr": old_lr, "new_lr": new_lr}
    return applied

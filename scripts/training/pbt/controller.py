#!/usr/bin/env python3
"""Dynamic hyperparameter controller helpers for PBT runs."""

import math
from pathlib import Path

from training.pbt.metrics import clamp
from training.pbt.models.controller import (
    DEFAULT_CONTROLLER_ACTIONS,
    dump_controller_action,
    dump_controller_observation,
)
from training.pbt.models.events import normalize_exploit_plan
from training.pbt.state.checkpointing import checkpoint_paths, generations_before
from training.pbt.state.optimizer_state import load_optimizer_state, summarize_optimizer_state


LR_ACTION_FACTORS = {
    "keep": 1.0,
    "lr_mul_0_95": 0.95,
    "lr_mul_0_9": 0.90,
    "lr_mul_1_05": 1.05,
    "lr_mul_1_1": 1.10,
}


def dynamic_controller_config(config):
    controller = config.get("pbt", {}).get("dynamic_controller")
    if not controller or controller.get("mode") == "disabled":
        return None
    return controller


def oriented_delta(config, current, reference):
    if current is None or reference is None:
        return None
    if config["pbt"]["mode"] == "max":
        return float(current) - float(reference)
    return float(reference) - float(current)


def _sample_std(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(finite) < 2:
        return 0.0 if finite else None
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))


def previous_member_observations(manifest, generation_index, member_name):
    observations = []
    for generation in generations_before(manifest, generation_index):
        observation = (generation.get("controller_observations") or {}).get(member_name)
        if observation:
            observations.append(observation)
    return observations


def previous_member_metric(manifest, generation_index, member_name, metric_name):
    observations = previous_member_observations(manifest, generation_index, member_name)
    if observations and observations[-1].get("metric_value") is not None:
        return float(observations[-1]["metric_value"])
    for generation in reversed(generations_before(manifest, generation_index)):
        metrics = (generation.get("workers", {}).get(member_name, {}) or {}).get("metrics", {})
        if metric_name in metrics:
            return float(metrics[metric_name])
    return None


def previous_observation_value(manifest, generation_index, member_name, field):
    observations = previous_member_observations(manifest, generation_index, member_name)
    if observations and observations[-1].get(field) is not None:
        return float(observations[-1][field])
    return None


def previous_metric_ema(manifest, generation_index, member_name):
    return previous_observation_value(manifest, generation_index, member_name, "metric_ema")


def _delta_sigma(delta, current_uncertainty, previous_uncertainty):
    if delta is None or current_uncertainty is None or previous_uncertainty is None:
        return None
    combined = math.sqrt(float(current_uncertainty) ** 2 + float(previous_uncertainty) ** 2)
    if combined <= 0.0:
        return None
    return float(delta) / combined


def last_action_epoch_fraction(manifest, generation_index, member_name):
    for generation in reversed(generations_before(manifest, generation_index)):
        action = (generation.get("controller_actions") or {}).get(member_name)
        observation = (generation.get("controller_observations") or {}).get(member_name)
        if action and action.get("action_ready") and action.get("action") != "keep":
            if observation and observation.get("epoch_fraction") is not None:
                return float(observation["epoch_fraction"])
            if generation.get("epoch") is not None:
                return float(generation["epoch"])
    return None


def metric_history(manifest, generation_record, member_name, metric_name, current_metric, window):
    history = []
    for generation in generations_before(manifest, generation_record["index"]):
        metrics = (generation.get("workers", {}).get(member_name, {}) or {}).get("metrics", {})
        if metric_name in metrics:
            history.append(float(metrics[metric_name]))
    history.append(float(current_metric))
    return history[-int(window):]


def _stat_value(summary, name, key="mean"):
    stats = summary.get(name)
    if not stats:
        return None
    return stats.get(key)


def _numeric_stats(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "min": min(finite),
        "max": max(finite),
        "mean": sum(finite) / len(finite),
    }


def optimizer_summary_for_member(experiment_dir, generation_record, member_name):
    if experiment_dir is None or generation_record.get("epoch") is None:
        return {}
    member_dir = Path(experiment_dir) / member_name
    _, optimizer_path = checkpoint_paths(member_dir, int(generation_record["epoch"]))
    if not optimizer_path.is_file():
        return {}
    try:
        state = load_optimizer_state(optimizer_path)
        summary = summarize_optimizer_state(state, top_k=0)
    except Exception:
        return {}

    lr_stats = _numeric_stats(
        group.get("lr")
        for group in state.get("param_groups", [])
        if isinstance(group, dict)
    )
    weight_decay_stats = _numeric_stats(
        group.get("weight_decay")
        for group in state.get("param_groups", [])
        if isinstance(group, dict) and group.get("weight_decay") is not None
    )
    return {
        "optimizer_step": _stat_value(summary, "steps", "max"),
        "optimizer_param_groups": summary.get("param_groups"),
        "optimizer_lr_mean": None if lr_stats is None else lr_stats["mean"],
        "optimizer_lr_min": None if lr_stats is None else lr_stats["min"],
        "optimizer_lr_max": None if lr_stats is None else lr_stats["max"],
        "optimizer_weight_decay_mean": (
            None if weight_decay_stats is None else weight_decay_stats["mean"]
        ),
        "momentum_norm": _stat_value(summary, "momentum_norm"),
        "second_moment_norm": _stat_value(summary, "sqrt_second_moment_norm"),
        "adaptive_direction_norm": _stat_value(summary, "adaptive_direction_norm"),
        "adaptive_direction_norm_max": _stat_value(summary, "adaptive_direction_norm", "max"),
    }


def epoch_start_lr(manifest, generation_index, member_name, current_lr, epoch_fraction):
    epoch_index = int(math.floor(float(epoch_fraction or 0.0)))
    for generation in reversed(generations_before(manifest, generation_index)):
        observation = (generation.get("controller_observations") or {}).get(member_name)
        if not observation or observation.get("epoch_fraction") is None:
            continue
        if int(math.floor(float(observation["epoch_fraction"]))) == epoch_index:
            return float(observation.get("epoch_start_lr") or observation["lr"])
    return float(current_lr)


def cooldown_state(config, manifest, generation_record, member_name, epoch_fraction):
    controller = dynamic_controller_config(config) or {}
    interval = float(controller.get("action_interval_fraction", 0.20))
    previous_action_epoch = last_action_epoch_fraction(
        manifest,
        generation_record["index"],
        member_name,
    )
    if previous_action_epoch is None:
        return True, 0.0
    elapsed = float(epoch_fraction) - previous_action_epoch
    if elapsed >= interval:
        return True, 0.0
    return False, max(0.0, interval - elapsed)


def observation_epoch_fraction(config, generation_record):
    controller = dynamic_controller_config(config) or {}
    fraction = controller.get("generation_epoch_fraction")
    if fraction is None:
        return float(generation_record["epoch"])
    initial_epoch = float(config.get("shared", {}).get("initial_epoch") or 0.0)
    return initial_epoch + (int(generation_record["index"]) + 1) * float(fraction)


def build_observation(config, manifest, generation_record, member_name, experiment_dir=None):
    metric_name = config["pbt"]["metric"]
    worker = generation_record["workers"][member_name]
    metrics = worker.get("metrics") or {}
    current_metric = float(metrics[metric_name])
    controller = dynamic_controller_config(config) or {}
    trend_window = int(controller.get("trend_window", 3))
    history = metric_history(
        manifest,
        generation_record,
        member_name,
        metric_name,
        current_metric,
        trend_window,
    )
    previous_metric = previous_member_metric(
        manifest,
        generation_record["index"],
        member_name,
        metric_name,
    )
    metric_delta = oriented_delta(config, current_metric, previous_metric)
    metric_uncertainty = metrics.get(f"{metric_name}_uncertainty")
    metric_uncertainty = None if metric_uncertainty is None else float(metric_uncertainty)
    previous_metric_uncertainty = previous_observation_value(
        manifest, generation_record["index"], member_name, "metric_uncertainty"
    )
    old_ema = previous_metric_ema(manifest, generation_record["index"], member_name)
    ema_beta = float(controller.get("ema_beta", 0.7))
    metric_ema = current_metric if old_ema is None else ema_beta * old_ema + (1.0 - ema_beta) * current_metric
    train_loss = metrics.get("train_loss")
    train_loss = None if train_loss is None else float(train_loss)
    old_train_loss_ema = previous_observation_value(
        manifest, generation_record["index"], member_name, "train_loss_ema"
    )
    train_loss_ema = (
        None
        if train_loss is None
        else train_loss
        if old_train_loss_ema is None
        else ema_beta * old_train_loss_ema + (1.0 - ema_beta) * train_loss
    )
    best = manifest.get("best") or {}
    baseline_metric = config["pbt"].get("baseline_metric_value")
    global_best_metric = best.get("metric_value")
    epoch_fraction = observation_epoch_fraction(config, generation_record)
    current_lr = float(manifest["members"][member_name]["lr"])
    start_lr = epoch_start_lr(
        manifest,
        generation_record["index"],
        member_name,
        current_lr,
        epoch_fraction,
    )
    action_ready, cooldown_remaining = cooldown_state(
        config,
        manifest,
        generation_record,
        member_name,
        epoch_fraction,
    )
    optimizer_summary = optimizer_summary_for_member(experiment_dir, generation_record, member_name)

    return dump_controller_observation(
        {
            "schema_version": 1,
            "generation": int(generation_record["index"]),
            "member": member_name,
            "epoch": int(generation_record["epoch"]),
            "epoch_fraction": epoch_fraction,
            "step": None,
            "lr": current_lr,
            "epoch_start_lr": start_lr,
            "cumulative_lr_factor": current_lr / start_lr,
            "metric_name": metric_name,
            "metric_value": current_metric,
            "previous_metric_value": previous_metric,
            "metric_delta": metric_delta,
            "metric_ema": metric_ema,
            "previous_metric_ema": old_ema,
            "metric_ema_delta": oriented_delta(config, metric_ema, old_ema),
            "metric_trend": oriented_delta(config, history[-1], history[0]) if len(history) >= 2 else None,
            "metric_noise": _sample_std(history),
            "metric_uncertainty": metric_uncertainty,
            "previous_metric_uncertainty": previous_metric_uncertainty,
            "metric_delta_sigma": _delta_sigma(
                metric_delta, metric_uncertainty, previous_metric_uncertainty
            ),
            "trend_window": len(history),
            "baseline_metric_value": baseline_metric,
            "baseline_delta": oriented_delta(config, current_metric, baseline_metric),
            "global_best_metric_value": global_best_metric,
            "global_best_delta": oriented_delta(config, current_metric, global_best_metric),
            "train_loss": train_loss,
            "train_accuracy": metrics.get("train_accuracy"),
            "train_loss_ema": train_loss_ema,
            "train_loss_ema_delta": oriented_delta(config, train_loss_ema, old_train_loss_ema),
            "grad_norm": metrics.get("train_max_grad_norm"),
            "amp_skipped_optimizer_steps": metrics.get("train_amp_skipped_optimizer_steps"),
            "max_cuda_memory_mb": metrics.get("train_max_cuda_memory_mb"),
            **optimizer_summary,
            "action_ready": action_ready,
            "cooldown_remaining": cooldown_remaining,
            "allowed_actions": list(controller.get("allowed_actions") or DEFAULT_CONTROLLER_ACTIONS),
        }
    )


def classify_observation(config, observation):
    controller = dynamic_controller_config(config) or {}
    tolerance = float(controller.get("metric_delta_tolerance", 0.0))
    noise_threshold = controller.get("noisy_metric_threshold")
    min_delta_sigma = controller.get("min_delta_sigma_for_action")
    baseline_delta = observation.get("baseline_delta")
    metric_ema_delta = observation.get("metric_ema_delta")
    metric_trend = observation.get("metric_trend")
    metric_delta = observation.get("metric_delta")
    metric_noise = observation.get("metric_noise")
    metric_delta_sigma = observation.get("metric_delta_sigma")
    if baseline_delta is not None and baseline_delta < -tolerance:
        return "unsafe"
    if noise_threshold is not None and metric_noise is not None and metric_noise > float(noise_threshold):
        return "noisy"
    if (
        min_delta_sigma is not None
        and metric_delta_sigma is not None
        and abs(float(metric_delta_sigma)) < float(min_delta_sigma)
    ):
        return "noisy"
    signal = metric_ema_delta if metric_ema_delta is not None else metric_trend
    if signal is None:
        signal = metric_delta
    if signal is None:
        return "flat"
    if signal > tolerance:
        return "improving"
    if signal < -tolerance:
        return "degraded"
    return "flat"


def select_controller_action(observation, state_label):
    allowed = set(observation["allowed_actions"])
    if not observation.get("action_ready", False):
        return "keep" if "keep" in allowed else observation["allowed_actions"][0]
    if state_label == "unsafe":
        for action in ("lr_mul_0_9", "lr_mul_0_95", "keep"):
            if action in allowed:
                return action
    if state_label == "degraded":
        for action in ("lr_mul_0_95", "lr_mul_0_9", "keep"):
            if action in allowed:
                return action
    return "keep" if "keep" in allowed else observation["allowed_actions"][0]


def action_reason(state_label, action, action_ready=True):
    if not action_ready:
        return "action cooldown is active"
    if state_label == "unsafe":
        return "metric is worse than the configured baseline; controller avoids LR increase"
    if state_label == "degraded":
        return "EMA/trend degraded relative to previous observations"
    if state_label == "noisy":
        return "metric window is too noisy for a safe LR change"
    if state_label == "improving":
        return "EMA/trend improved; controller keeps LR stable"
    return "metric trend is flat or unavailable"


def bounded_lr(config, observation, action):
    lr_before = float(observation["lr"])
    factor = LR_ACTION_FACTORS.get(action, 1.0)
    proposed = lr_before * factor
    pbt = config["pbt"]
    controller = dynamic_controller_config(config) or {}
    min_lr = float(pbt["min_lr"])
    max_lr = float(pbt["max_lr"])
    epoch_start = float(observation.get("epoch_start_lr") or lr_before)
    max_factor = float(controller.get("max_cumulative_lr_factor_per_epoch", 2.0))
    lower = max(min_lr, epoch_start / max_factor)
    upper = min(max_lr, epoch_start * max_factor)
    return proposed, clamp(proposed, lower, upper)


def build_controller_action(config, observation):
    state_label = classify_observation(config, observation)
    action = select_controller_action(observation, state_label)
    proposed_lr, next_lr = bounded_lr(config, observation, action)
    if not observation.get("action_ready", False):
        safety_check = "cooldown"
    elif action == "flag_review":
        safety_check = "blocked"
    elif proposed_lr != next_lr:
        safety_check = "clamped"
    else:
        safety_check = "passed"
    confidence = 0.5 if observation.get("metric_ema_delta") is None else 0.75
    if state_label == "unsafe":
        confidence = 0.9
    elif state_label == "noisy":
        confidence = 0.4
    return dump_controller_action(
        {
            "schema_version": 1,
            "generation": observation["generation"],
            "member": observation["member"],
            "state_label": state_label,
            "confidence": confidence,
            "action": action,
            "reason": action_reason(state_label, action, observation.get("action_ready", False)),
            "safety_check": safety_check,
            "applied": False,
            "lr_before": observation["lr"],
            "proposed_lr": proposed_lr,
            "bounded_lr": next_lr,
            "action_ready": observation.get("action_ready", False),
            "cooldown_remaining": observation.get("cooldown_remaining", 0.0),
        }
    )


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

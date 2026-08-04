#!/usr/bin/env python3
"""Turn a controller observation into a bounded LR action."""

from training.pbt.metrics import clamp
from training.pbt.models.controller import dump_controller_action


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

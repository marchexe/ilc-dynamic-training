#!/usr/bin/env python3
"""Build a per-member, per-generation controller observation from manifest history."""

import math
from pathlib import Path

from training.pbt.controller.decision import dynamic_controller_config, oriented_delta
from training.pbt.models.controller import DEFAULT_CONTROLLER_ACTIONS, dump_controller_observation
from training.pbt.state.checkpointing import checkpoint_paths, generations_before
from training.pbt.state.optimizer_state import load_optimizer_state, summarize_optimizer_state


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

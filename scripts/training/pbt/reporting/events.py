#!/usr/bin/env python3
"""Event-log writers: one record_*() per lifecycle event (train start/finish,
evaluation, exploit application, controller LR change, new best, ...).
"""

import math

from training.runtime import utc_now
from training.pbt.reporting.io import append_event

def record_initial_evaluation(run_dir, config, record):
    metric = config["pbt"]["metric"]
    metrics = record.get("metrics") or {}
    append_event(
        run_dir,
        "evaluation",
        {
            "phase": "baseline",
            "trial": "initial_resume",
            "step": -1,
            "generation": -1,
            "checkpoint": record.get("checkpoint"),
            "slot": record.get("slot"),
            "metric": metric,
            "proxy_metric": metrics.get(metric),
            "metrics": metrics,
            "command": record.get("command"),
            "log": record.get("log"),
            "status": record.get("status"),
            "returncode": record.get("returncode"),
        },
    )


def record_train_start(run_dir, config, generation_record, trial, worker):
    append_event(
        run_dir,
        "train_interval",
        {
            "status": "started",
            "generation": generation_record["index"],
            "step": generation_record["index"],
            "trial": trial,
            "epoch": generation_record.get("epoch"),
            "lr": worker.get("lr"),
            "samples_per_epoch": config["shared"].get("samples_per_epoch"),
            "epochs_per_generation": config["shared"].get("epochs_per_generation"),
            "slot": worker.get("slot"),
            "command": worker.get("command"),
            "log": worker.get("log"),
        },
    )


def record_train_finish(run_dir, config, generation_record, trial, worker):
    metrics = worker.get("metrics") or {}
    append_event(
        run_dir,
        "train_interval",
        {
            "status": worker.get("status"),
            "generation": generation_record["index"],
            "step": generation_record["index"],
            "trial": trial,
            "epoch": generation_record.get("epoch"),
            "lr": worker.get("lr"),
            "train_loss": metrics.get("train_loss"),
            "train_accuracy": metrics.get("train_accuracy"),
            "returncode": worker.get("returncode"),
            "log": worker.get("log"),
        },
    )


def record_evaluation(run_dir, config, generation_record, trial, worker):
    metric = config["pbt"]["metric"]
    metrics = worker.get("metrics") or {}
    append_event(
        run_dir,
        "evaluation",
        {
            "phase": "proxy_validation",
            "generation": generation_record["index"],
            "step": generation_record["index"],
            "trial": trial,
            "epoch": generation_record.get("epoch"),
            "lr": worker.get("lr"),
            "metric": metric,
            "proxy_metric": metrics.get(metric),
            "metrics": metrics,
            "log": worker.get("log"),
            "status": worker.get("status"),
            "returncode": worker.get("returncode"),
        },
    )


def _worker_metric(config, generation_record, member):
    metric = (config.get("pbt") or {}).get("metric")
    if not metric:
        return None
    worker = (generation_record.get("workers") or {}).get(member) or {}
    metrics = worker.get("metrics") or {}
    return metrics.get(metric)


def record_skipped_exploit(run_dir, generation_record, skipped_event):
    """Log a donor->recipient replacement that significance gating declined
    to apply, so the audit trail shows not just what happened but what
    almost happened and why it didn't -- see planning.py:exploit_significance.
    """
    append_event(
        run_dir,
        "exploit_skipped",
        {
            "generation": generation_record.get("index"),
            "donor": skipped_event.get("donor"),
            "recipient": skipped_event.get("recipient"),
            "donor_metric": skipped_event.get("donor_metric"),
            "recipient_metric": skipped_event.get("recipient_metric"),
            "margin_sigma": skipped_event.get("margin_sigma"),
            "required_sigma": skipped_event.get("required_sigma"),
            "reason": skipped_event.get("reason"),
        },
    )


def record_controller_lr_change(run_dir, generation_record, member_name, change):
    """Log a fine dynamic-controller LR nudge applied directly to a member's
    own LR, independent of PBT exploit -- a distinct event type from
    "exploit"/"lr_change" (population layer) so the two adaptation layers
    stay clearly separated in the event log, not just in the config.
    """
    append_event(
        run_dir,
        "controller_lr_change",
        {
            "generation": generation_record.get("index"),
            "member": member_name,
            "action": change.get("action"),
            "old_lr": change.get("old_lr"),
            "new_lr": change.get("new_lr"),
        },
    )


def record_tiered_evaluation_round(run_dir, manifest, generation_record, tier, dataset, suffix, member_results, metric_name, mode):
    """Persist one control/monitor/full evaluation round: every member's
    result at this generation, plus the full rank ordering (not just #1),
    so post-hoc ranking-agreement/correlation analysis has real paired data.
    Read-only bookkeeping -- nothing in planning.py/controller.py ever reads
    manifest["tiered_evaluations"]; it must stay that way for `full` (and
    `monitor`) to remain a genuine, non-leaking check on the control proxy.
    """
    ranking = sorted(
        (
            name
            for name, record in member_results.items()
            if (record.get("metrics") or {}).get(metric_name) is not None
            and math.isfinite(float(record["metrics"][metric_name]))
        ),
        key=lambda name: float(member_results[name]["metrics"][metric_name]),
        reverse=(mode == "max"),
    )
    round_record = {
        "schema_version": 1,
        "generation": generation_record.get("index"),
        "tier": tier,
        "dataset": dataset,
        "suffix": suffix,
        "metric_name": metric_name,
        "mode": mode,
        "members": member_results,
        "ranking": ranking,
        "recorded_at": utc_now(),
    }
    manifest.setdefault("tiered_evaluations", []).append(round_record)
    append_event(
        run_dir,
        "tiered_evaluation",
        {
            "generation": generation_record.get("index"),
            "tier": tier,
            "dataset": dataset,
            "suffix": suffix,
            "ranking": ranking,
            "member_count": len(member_results),
        },
    )
    return round_record


def record_new_best(run_dir, manifest, generation_record, best_record):
    append_event(
        run_dir,
        "new_best",
        {
            "generation": generation_record.get("index"),
            "step": generation_record.get("index"),
            "trial": best_record.get("member"),
            "metric": best_record.get("metric"),
            "metric_value": best_record.get("metric_value"),
            "lr": best_record.get("lr"),
            "source_state_path": best_record.get("source_state_path"),
            "source_optimizer_path": best_record.get("source_optimizer_path"),
            "state_path": best_record.get("state_path"),
            "optimizer_path": best_record.get("optimizer_path"),
        },
    )


def record_exploit_application(
    run_dir,
    config,
    generation_record,
    event,
    donor_state,
    donor_optimizer,
    recipient_state,
    recipient_optimizer,
    *,
    weight_copied,
    optimizer_copied,
):
    donor = event.get("donor")
    recipient = event.get("recipient")
    generation = generation_record["index"]
    old_lr = event.get("recipient_lr")
    new_lr = event.get("new_lr")
    mutation = event.get("mutation_factor", event.get("lr_factor"))
    payload = {
        "generation": generation,
        "step": generation,
        "donor": donor,
        "recipient": recipient,
        "donor_metric": _worker_metric(config, generation_record, donor),
        "recipient_metric": _worker_metric(config, generation_record, recipient),
        "weight_source": event.get("weight_source", event.get("source")),
        "optimizer_source": event.get("optimizer_source", event.get("source")),
        "old_lr": old_lr,
        "new_lr": new_lr,
        # Exploit recipients are owned entirely by the PBT plan
        # (donor_lr * mutation_factor); the dynamic controller never applies
        # to them (see apply_actions_to_plan in controller.py). pbt_proposed_lr
        # and final_lr are always equal to new_lr here -- controller_applied
        # is always False -- kept as explicit fields so ownership is visible
        # in the artifact without needing to read the code.
        "mutation": mutation,
        "significance_margin_sigma": event.get("significance_margin_sigma"),
        "significance_sigma_required": event.get("significance_sigma_required"),
        "pbt_proposed_lr": event.get("pbt_proposed_lr"),
        "final_lr": event.get("final_lr"),
        "controller_applied": event.get("controller_applied"),
        "reason": event.get("reason"),
        "source": event.get("source"),
        "applied": event.get("applied"),
        # population_lr_policy-only fields; None for every other source.
        "direction": event.get("direction"),
        "margin_sigma": event.get("margin_sigma"),
        "outcome": event.get("outcome"),
        "metric_before": event.get("metric_before"),
        "metric_after": event.get("metric_after"),
    }
    append_event(run_dir, "exploit", payload)
    append_event(
        run_dir,
        "weight_copy",
        {
            **payload,
            "source_path": str(donor_state),
            "destination_path": str(recipient_state),
            "copied": bool(weight_copied),
        },
    )
    append_event(
        run_dir,
        "optimizer_copy",
        {
            **payload,
            "source_path": str(donor_optimizer),
            "destination_path": str(recipient_optimizer),
            "copied": bool(optimizer_copied),
        },
    )
    append_event(
        run_dir,
        "lr_change",
        {
            **payload,
            "old_lr": old_lr,
            "new_lr": new_lr,
            "changed": old_lr != new_lr,
        },
    )

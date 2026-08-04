#!/usr/bin/env python3
"""Fixed-working-point physics metrics (Wilson intervals, mistag percentages)
and the row-building / CSV-writing layer built on top of them
(metrics.csv, tiered_metrics.csv, exploit_table.csv, skipped_exploits.csv).
"""

import csv
import math
from pathlib import Path

from training.pbt.reporting.constants import (
    CONTROLLER_OBJECTIVE_COLUMN,
    EXPLOIT_TABLE_NAME,
    FIXED_WORKING_POINTS,
    FIXED_WORKING_POINT_COLUMNS,
    FIXED_WORKING_POINT_COUNT_COLUMNS,
    FIXED_WORKING_POINT_ERROR_COLUMNS,
    METRICS_COLUMNS,
    METRICS_NAME,
    SKIPPED_EXPLOIT_TABLE_NAME,
    TIERED_METRICS_COLUMNS,
    TIERED_METRICS_NAME,
    TIER_ORDER,
)
from training.pbt.reporting.io import ensure_run_layout, write_atomic_csv

def _metric_mode(manifest):
    return manifest.get("config", {}).get("pbt", {}).get("mode", "max")


def _metric_name(manifest):
    return manifest.get("config", {}).get("pbt", {}).get("metric", "validation_bkg_rejection_score")


def _better(mode, candidate, incumbent):
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return candidate > incumbent if mode == "max" else candidate < incumbent


def _rejection_at(metrics, tag, eff, background):
    lookup = metrics.get("validation_bkg_rejection_at_eff_lookup") or {}
    row = lookup.get(f"{tag}_tag_eff_{eff:.2f}") or {}
    value = row.get(f"{background}_bkg_rejection")
    if value is not None:
        return float(value)

    curves = metrics.get("validation_bkg_rejection_at_eff") or {}
    efficiencies = [float(value) for value in curves.get("efficiencies") or []]
    pairs = curves.get("pairs") or {}
    pair = f"{tag}{background}"
    if eff not in efficiencies or pair not in pairs:
        return None
    index = efficiencies.index(eff)
    values = pairs.get(pair) or []
    return float(values[index]) if index < len(values) else None


def _mistag_percent(metrics, tag, eff, background):
    rejection = _rejection_at(metrics, tag, eff, background)
    if rejection is None or rejection <= 0 or not math.isfinite(rejection):
        return None
    return 100.0 / rejection


def fixed_working_point_values(metrics):
    return {
        point["column"]: _mistag_percent(metrics, point["tag"], point["efficiency"], point["background"])
        for point in FIXED_WORKING_POINTS
    }


def _working_point_counts(metrics, tag, eff, background):
    rows = (metrics.get("validation_bkg_rejection_at_eff_counts") or {}).get(f"{tag}{background}") or []
    for row in rows:
        if abs(float(row.get("signal_efficiency", -1.0)) - float(eff)) < 1.0e-6:
            return row.get("background_passed"), row.get("background_total")
    return None, None


def wilson_interval(passed, total, z=1.0):
    """Asymmetric binomial confidence half-widths (fractional, 0-1) via the
    Wilson score interval, centered on the observed rate p = passed/total.

    z=1.0 gives the ~68.27% (1 sigma) interval. Unlike the naive
    sqrt(p(1-p)/n) normal approximation, this never produces bounds outside
    [0, 1] and stays meaningful when p is close to 0 -- the regime fixed-WP
    mistag rates usually sit in, so it avoids implying misleadingly tight or
    symmetric uncertainty for rare mistags.
    """
    if passed is None or total is None:
        return None
    total = int(total)
    passed = int(passed)
    if total <= 0 or passed < 0 or passed > total:
        return None
    p = passed / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    lower = max(0.0, centre - half)
    upper = min(1.0, centre + half)
    return max(0.0, p - lower), max(0.0, upper - p)


def fixed_working_point_uncertainty(metrics, tag, eff, background):
    """(lower_err, upper_err, passed, total) in mistag-percent units, or None."""
    passed, total = _working_point_counts(metrics, tag, eff, background)
    bounds = wilson_interval(passed, total)
    if bounds is None:
        return None
    lower, upper = bounds
    return 100.0 * lower, 100.0 * upper, int(passed), int(total)


def fixed_working_point_uncertainties(metrics):
    out = {}
    for point in FIXED_WORKING_POINTS:
        result = fixed_working_point_uncertainty(metrics, point["tag"], point["efficiency"], point["background"])
        lower, upper, passed, total = result if result else (None, None, None, None)
        column = point["column"]
        out[f"{column}_err_low"] = lower
        out[f"{column}_err_high"] = upper
        out[f"{column}_passed"] = passed
        out[f"{column}_total"] = total
    return out


def format_mistag_value(value, lower_err=None, upper_err=None):
    """Format a mistag percentage rounded to the precision its uncertainty
    actually supports, so tiny mistag rates are never shown with false
    precision (e.g. not "0.01734%" when the uncertainty is +-0.02%).
    """
    if value is None:
        return "n/a"
    errors = [err for err in (lower_err, upper_err) if err is not None and err > 0]
    if not errors:
        return f"{value:.3f}%"
    magnitude = max(errors)
    digits = max(0, min(6, 1 - int(math.floor(math.log10(magnitude)))))
    if lower_err is not None and upper_err is not None and abs(lower_err - upper_err) > 0.5 * 10 ** (-digits):
        return f"{value:.{digits}f}% (+{upper_err:.{digits}f}/-{lower_err:.{digits}f})"
    return f"{value:.{digits}f}±{magnitude:.{digits}f}%"


def controller_objective_mistag(metrics):
    values = [value for value in fixed_working_point_values(metrics).values() if value is not None]
    if values:
        return sum(values) / len(values)
    value = metrics.get("validation_working_point_mistag_percent")
    return None if value is None else float(value)


def training_chunk_samples(manifest):
    shared = manifest.get("config", {}).get("shared", {})
    samples_per_epoch = int(shared.get("samples_per_epoch", 0) or 0)
    epochs_per_generation = int(shared.get("epochs_per_generation", 1) or 1)
    return samples_per_epoch * epochs_per_generation


def training_dataset_size(manifest):
    shared = manifest.get("config", {}).get("shared", {})
    for key in (
        "training_dataset_size",
        "training_dataset_rows",
        "training_dataset_rows_total",
        "train_rows_total",
        "full_training_dataset_size",
    ):
        value = shared.get(key)
        if value is not None:
            value = int(value)
            return value if value > 0 else None
    datasets = manifest.get("datasets") or (manifest.get("run") or {}).get("datasets") or {}
    value = datasets.get("training_dataset_size") or datasets.get("training_rows_total")
    if value is not None:
        value = int(value)
        return value if value > 0 else None
    return None


def _worker_samples_seen(worker, cumulative, chunk_samples):
    explicit = worker.get("samples_seen") or worker.get("training_samples_seen")
    if explicit is not None:
        return int(explicit)
    increment = worker.get("samples_trained") or worker.get("training_samples") or chunk_samples
    return int(cumulative) + int(increment or 0)


def evaluation_metadata(manifest):
    shared = manifest.get("config", {}).get("shared", {})
    datasets = manifest.get("datasets") or (manifest.get("run") or {}).get("datasets") or {}
    proxy = shared.get("proxy_validation") or datasets.get("proxy_validation") or {}
    suffix = shared.get("validation_suffix") or datasets.get("validation_suffix")
    dataset = shared.get("validation_dataset") or datasets.get("validation_dataset") or shared.get("dataset") or datasets.get("train_dataset")
    sample_count = shared.get("samples_per_epoch_val")
    if proxy:
        active_subset = proxy.get("active_subset", "control")
        sample_count = proxy.get(f"{active_subset}_rows_total", sample_count)
    if manifest.get("config", {}).get("smoke"):
        evaluation_type = "smoke"
    elif proxy and proxy.get("active_subset") != "full":
        evaluation_type = "proxy"
    elif suffix and "tail" in str(suffix):
        evaluation_type = "proxy"
    else:
        evaluation_type = "full"
    return {
        "validation_dataset": dataset,
        "validation_suffix": suffix,
        "validation_sample_count": None if sample_count is None else int(sample_count),
        "evaluation_type": evaluation_type,
    }


def _metric_from_row(row):
    return row.get("optimization_metric_value")


def evaluation_rows(manifest):
    metric = _metric_name(manifest)
    mode = _metric_mode(manifest)
    best = None
    rows = []
    metadata = evaluation_metadata(manifest)
    baseline = baseline_record(manifest)
    if baseline and baseline.get("metric_value") is not None:
        best = float(baseline["metric_value"])
    chunk_samples = training_chunk_samples(manifest)
    dataset_size = training_dataset_size(manifest)
    cumulative_by_trial = {}
    for generation in sorted(manifest.get("generations", []), key=lambda item: item.get("index", 0)):
        training_chunk = int(generation.get("index", 0))
        for trial, worker in sorted((generation.get("workers") or {}).items()):
            metrics = worker.get("metrics") or {}
            value = metrics.get(metric)
            if value is None:
                continue
            value = float(value)
            samples_seen = _worker_samples_seen(worker, cumulative_by_trial.get(trial, 0), chunk_samples)
            cumulative_by_trial[trial] = samples_seen
            epoch_fraction = None if dataset_size is None else samples_seen / dataset_size
            if _better(mode, value, best):
                best = value
            rows.append(
                {
                    "generation": training_chunk,
                    "training_chunk": training_chunk,
                    "samples_seen": samples_seen,
                    "epoch_fraction": epoch_fraction,
                    "trial": trial,
                    "LR": float(worker["lr"]) if worker.get("lr") is not None else None,
                    "optimization_metric_name": metric,
                    "optimization_metric_value": value,
                    "optimization_metric_mode": mode,
                    CONTROLLER_OBJECTIVE_COLUMN: controller_objective_mistag(metrics),
                    "validation_working_point_mistag_percent": metrics.get("validation_working_point_mistag_percent"),
                    **fixed_working_point_values(metrics),
                    **fixed_working_point_uncertainties(metrics),
                    "validation_accuracy": metrics.get("validation_accuracy"),
                    "validation_auc": metrics.get("validation_auc"),
                    "validation_loss": metrics.get("validation_loss"),
                    "best_so_far": best,
                    "training_loss": metrics.get("train_loss"),
                    "validation_shutdown_warning": bool(metrics.get("validation_shutdown_warning")),
                    **metadata,
                }
            )
    return rows


def refresh_metrics_csv(run_dir, manifest):
    ensure_run_layout(run_dir)
    rows = evaluation_rows(manifest)
    path = Path(run_dir) / METRICS_NAME
    return write_atomic_csv(path, METRICS_COLUMNS, rows)


def read_metrics_rows(run_dir):
    path = Path(run_dir) / METRICS_NAME
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            converted = dict(row)
            for key in (
                "generation",
                "training_chunk",
                "samples_seen",
                "validation_sample_count",
                *FIXED_WORKING_POINT_COUNT_COLUMNS,
            ):
                converted[key] = int(float(converted[key])) if converted.get(key) else None
            for key in (
                "epoch_fraction",
                "LR",
                "optimization_metric_value",
                CONTROLLER_OBJECTIVE_COLUMN,
                "validation_working_point_mistag_percent",
                *FIXED_WORKING_POINT_COLUMNS,
                *FIXED_WORKING_POINT_ERROR_COLUMNS,
                "validation_accuracy",
                "validation_auc",
                "validation_loss",
                "best_so_far",
                "training_loss",
            ):
                converted[key] = float(converted[key]) if converted.get(key) else None
            rows.append(converted)
        return rows


def baseline_record(manifest):
    metric = _metric_name(manifest)
    initial = manifest.get("initial_evaluation") or {}
    initial_metrics = initial.get("metrics") or {}
    if initial.get("status") == "completed" and initial_metrics.get(metric) is not None:
        return {
            "source": "initial_evaluation",
            "kind": "measured",
            "metric": metric,
            "metric_value": float(initial_metrics[metric]),
            "checkpoint": initial.get("checkpoint"),
            "metrics": initial_metrics,
        }
    return None


def configured_baseline_record(manifest):
    metric = _metric_name(manifest)
    pbt = manifest.get("config", {}).get("pbt", {})
    value = pbt.get("configured_baseline_metric_value", pbt.get("baseline_metric_value"))
    if value is None:
        return None
    return {
        "source": "config",
        "kind": "configured",
        "metric": metric,
        "metric_value": float(value),
        "checkpoint": (manifest.get("initial_resume") or {}).get("state_path")
        or manifest.get("checkpoint", {}).get("path"),
    }


def final_best_row(rows, mode):
    if not rows:
        return None
    selector = max if mode == "max" else min
    return selector(rows, key=lambda row: row["optimization_metric_value"])


def relative_change(mode, baseline, value):
    if baseline in (None, 0) or value is None:
        return None
    if mode == "max":
        return (value - baseline) / baseline
    return (baseline - value) / baseline


def write_exploit_table(run_dir, events):
    path = Path(run_dir) / EXPLOIT_TABLE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {}
    for event in events:
        key = (event.get("generation"), event.get("donor"), event.get("recipient"))
        if event.get("event_type") == "exploit":
            by_key.setdefault(key, {}).update(event)
        elif event.get("event_type") == "weight_copy":
            row = by_key.setdefault(key, {})
            row["weight_source_path"] = event.get("source_path")
            row["weight_destination_path"] = event.get("destination_path")
            row["weight_copied"] = event.get("copied")
        elif event.get("event_type") == "optimizer_copy":
            row = by_key.setdefault(key, {})
            row["optimizer_source_path"] = event.get("source_path")
            row["optimizer_destination_path"] = event.get("destination_path")
            row["optimizer_copied"] = event.get("copied")
    columns = (
        "generation", "donor", "recipient", "donor_metric", "recipient_metric",
        "weight_source", "optimizer_source", "old_lr", "new_lr",
        "mutation", "significance_margin_sigma", "significance_sigma_required",
        "pbt_proposed_lr", "final_lr", "controller_applied", "reason",
        "weight_copied", "weight_source_path", "weight_destination_path",
        "optimizer_copied", "optimizer_source_path", "optimizer_destination_path",
    )
    rows = sorted(by_key.values(), key=lambda item: (item.get("generation") is None, item.get("generation") or -1, item.get("recipient") or ""))
    return write_atomic_csv(path, columns, rows)


def write_skipped_exploits_table(run_dir, events):
    """Every donor->recipient replacement significance gating declined to
    apply -- audit trail for "we deliberately did not overwrite this member
    because the win wasn't statistically meaningful", not just what was done.
    """
    path = Path(run_dir) / SKIPPED_EXPLOIT_TABLE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [event for event in events if event.get("event_type") == "exploit_skipped"]
    columns = (
        "generation", "donor", "recipient", "donor_metric", "recipient_metric",
        "margin_sigma", "required_sigma", "reason",
    )
    rows = sorted(rows, key=lambda item: (item.get("generation") or -1, item.get("recipient") or ""))
    return write_atomic_csv(path, columns, rows)


def _tiered_round_samples_seen(manifest, generation):
    if generation is None or generation < 0:
        return 0
    return training_chunk_samples(manifest) * (int(generation) + 1)


def tiered_evaluation_rows(manifest):
    """Flatten manifest["tiered_evaluations"] into one row per
    (generation, tier, member), preserving the tier's full rank for that
    round so `member`+`generation` observations can be joined across
    control/monitor/full for paired comparison (see requirement: paired
    control<->monitor<->full observations, not just aggregate trends).
    """
    rows = []
    for round_record in manifest.get("tiered_evaluations", []):
        tier = round_record.get("tier")
        generation = round_record.get("generation")
        dataset = round_record.get("dataset")
        suffix = round_record.get("suffix")
        metric_name = round_record.get("metric_name")
        members = round_record.get("members") or {}
        ranking = round_record.get("ranking") or []
        rank_by_member = {name: index + 1 for index, name in enumerate(ranking)}
        for member, record in members.items():
            metrics = record.get("metrics") or {}
            rows.append(
                {
                    "generation": generation,
                    "samples_seen": _tiered_round_samples_seen(manifest, generation),
                    "tier": tier,
                    "member": member,
                    "dataset": dataset,
                    "suffix": suffix,
                    "status": record.get("status"),
                    "rank": rank_by_member.get(member),
                    "population_size": len(members),
                    "metric_name": metric_name,
                    "metric_value": metrics.get(metric_name),
                    CONTROLLER_OBJECTIVE_COLUMN: controller_objective_mistag(metrics) if metrics else None,
                    "validation_working_point_mistag_percent": metrics.get("validation_working_point_mistag_percent"),
                    **fixed_working_point_values(metrics),
                    **fixed_working_point_uncertainties(metrics),
                }
            )
    return rows


def write_tiered_metrics_csv(run_dir, manifest):
    ensure_run_layout(run_dir)
    rows = tiered_evaluation_rows(manifest)
    path = Path(run_dir) / TIERED_METRICS_NAME
    rows = sorted(
        rows,
        key=lambda item: (
            item["generation"] if item["generation"] is not None else -999,
            TIER_ORDER.index(item["tier"]) if item["tier"] in TIER_ORDER else 99,
            item["member"] or "",
        ),
    )
    return write_atomic_csv(path, TIERED_METRICS_COLUMNS, rows)

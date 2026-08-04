#!/usr/bin/env python3
"""Canonical structured artifacts for PBT run directories."""

import csv
import glob
import json
import math
import os
from pathlib import Path

import yaml

from training.runtime import atomic_json, data_paths, git_metadata, sha256, utc_now


EVENTS_NAME = "events.jsonl"
METRICS_NAME = "metrics.csv"
SUMMARY_NAME = "summary.json"
REPORT_NAME = "report.md"
PLOT_NAMES = {
    "training_evolution": "training_evolution.png",
    "working_point_evolution": "working_point_evolution.png",
    "baseline_comparison": "baseline_vs_selected.png",
    "proxy_diagnostics": "proxy_diagnostics.png",
}
CONDITIONAL_PLOT_NAMES = ("baseline_comparison", "proxy_diagnostics")
EXPLOIT_TABLE_NAME = "plots/report/exploit_table.csv"
SKIPPED_EXPLOIT_TABLE_NAME = "plots/report/skipped_exploits.csv"
TIERED_METRICS_NAME = "tiered_metrics.csv"
FIXED_WORKING_POINTS = (
    {"tag": "b", "efficiency": 0.80, "background": "c", "column": "btag_c_mistag_percent_at_0p80", "label": "c bkg, b-eff 80%"},
    {"tag": "b", "efficiency": 0.80, "background": "d", "column": "btag_d_mistag_percent_at_0p80", "label": "d bkg, b-eff 80%"},
    {"tag": "b", "efficiency": 0.90, "background": "c", "column": "btag_c_mistag_percent_at_0p90", "label": "c bkg, b-eff 90%"},
    {"tag": "b", "efficiency": 0.90, "background": "d", "column": "btag_d_mistag_percent_at_0p90", "label": "d bkg, b-eff 90%"},
    {"tag": "c", "efficiency": 0.50, "background": "b", "column": "ctag_b_mistag_percent_at_0p50", "label": "b bkg, c-eff 50%"},
    {"tag": "c", "efficiency": 0.50, "background": "d", "column": "ctag_d_mistag_percent_at_0p50", "label": "d bkg, c-eff 50%"},
    {"tag": "c", "efficiency": 0.80, "background": "b", "column": "ctag_b_mistag_percent_at_0p80", "label": "b bkg, c-eff 80%"},
    {"tag": "c", "efficiency": 0.80, "background": "d", "column": "ctag_d_mistag_percent_at_0p80", "label": "d bkg, c-eff 80%"},
)
FIXED_WORKING_POINT_COLUMNS = tuple(point["column"] for point in FIXED_WORKING_POINTS)
# Wilson-interval bookkeeping columns for each fixed-WP mistag value: the
# asymmetric 68.27% (~1 sigma) confidence half-widths plus the raw
# background pass/total counts they were derived from, kept for auditability.
FIXED_WORKING_POINT_UNCERTAINTY_SUFFIXES = ("err_low", "err_high", "passed", "total")
FIXED_WORKING_POINT_UNCERTAINTY_COLUMNS = tuple(
    f"{point['column']}_{suffix}"
    for point in FIXED_WORKING_POINTS
    for suffix in FIXED_WORKING_POINT_UNCERTAINTY_SUFFIXES
)
FIXED_WORKING_POINT_ERROR_COLUMNS = tuple(
    column for column in FIXED_WORKING_POINT_UNCERTAINTY_COLUMNS if column.endswith(("_err_low", "_err_high"))
)
FIXED_WORKING_POINT_COUNT_COLUMNS = tuple(
    column for column in FIXED_WORKING_POINT_UNCERTAINTY_COLUMNS if column.endswith(("_passed", "_total"))
)
# Background-flavour colors, shared across every fixed-WP plot (b-tag and
# c-tag panels alike) so a given color always means the same mistagged flavour.
FLAVOR_COLORS = {
    "b": "#4c78a8",
    "c": "#59a14f",
    "d": "#e15759",
}


def _working_point_style_ranks():
    ranks = {}
    for tag in sorted({point["tag"] for point in FIXED_WORKING_POINTS}):
        efficiencies = sorted({point["efficiency"] for point in FIXED_WORKING_POINTS if point["tag"] == tag})
        for rank, efficiency in enumerate(efficiencies):
            ranks[(tag, efficiency)] = rank
    return ranks


# Marker/linestyle rank (0 = lower efficiency, 1 = higher efficiency) within
# each tag, so working points are distinguished by shape/style, not color.
WORKING_POINT_STYLE_RANK = _working_point_style_ranks()
WORKING_POINT_MARKERS = ("o", "s")
WORKING_POINT_LINESTYLES = ("-", "--")
CONTROLLER_OBJECTIVE_COLUMN = "controller_objective_mistag_percent"
METRICS_COLUMNS = (
    "generation",
    "training_chunk",
    "samples_seen",
    "epoch_fraction",
    "trial",
    "LR",
    "optimization_metric_name",
    "optimization_metric_value",
    "optimization_metric_mode",
    CONTROLLER_OBJECTIVE_COLUMN,
    "validation_working_point_mistag_percent",
    *FIXED_WORKING_POINT_COLUMNS,
    *FIXED_WORKING_POINT_UNCERTAINTY_COLUMNS,
    "validation_accuracy",
    "validation_auc",
    "validation_loss",
    "best_so_far",
    "training_loss",
    "validation_shutdown_warning",
    "validation_dataset",
    "validation_suffix",
    "validation_sample_count",
    "evaluation_type",
)


def ensure_run_layout(run_dir):
    run_dir = Path(run_dir)
    for relative in ("logs", "checkpoints", "plots"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_resolved_config(run_dir, config):
    path = Path(run_dir) / "resolved_config.yaml"
    atomic_text(path, yaml.safe_dump(config, sort_keys=False))
    return path


def _split_labeled_path(value):
    text = str(value)
    return text.split(":", 1) if ":" in text else (None, text)


def _expanded_labeled_paths(values):
    rows = []
    for value in values:
        label, pattern = _split_labeled_path(value)
        matches = sorted(glob.glob(pattern))
        rows.append(
            {
                "label": label,
                "pattern": pattern,
                "files": matches,
                "missing": not matches,
            }
        )
    return rows


def resolved_input_data_files(config):
    shared = config["shared"]
    paths = data_paths(
        shared["dataset"],
        shared.get("data_extension", "root"),
        shared.get("validation_dataset"),
        shared.get("train_suffix"),
        shared.get("validation_suffix"),
    )
    return {split: _expanded_labeled_paths(values) for split, values in paths.items()}


def configured_intervals(config):
    shared = config["shared"]
    pbt = config["pbt"]
    epochs_per_generation = int(shared["epochs_per_generation"])
    samples_per_epoch = int(shared["samples_per_epoch"])
    chunk_samples = epochs_per_generation * samples_per_epoch
    strategy = pbt.get("strategy", "exploit_mutate")
    evaluation_chunks = int(pbt.get("evaluation_interval_generations") or pbt.get("evaluation_interval") or 1)
    configured_exploit_chunks = pbt.get("exploit_interval_generations") or pbt.get("exploit_interval")
    exploit_chunks = None if strategy == "fixed_lr_grid" else int(configured_exploit_chunks or evaluation_chunks)
    return {
        "training_interval": {
            "epochs_per_generation": epochs_per_generation,
            "samples_per_epoch": samples_per_epoch,
            "samples_per_trial_chunk": chunk_samples,
        },
        "evaluation_interval": {
            "training_chunks": evaluation_chunks,
            "epochs": epochs_per_generation * evaluation_chunks,
            "samples_per_trial": chunk_samples * evaluation_chunks,
            "samples_per_epoch_val": int(shared["samples_per_epoch_val"]),
        },
        "exploit_interval": {
            "enabled": strategy != "fixed_lr_grid",
            "training_chunks": exploit_chunks,
            "epochs": None if exploit_chunks is None else epochs_per_generation * exploit_chunks,
            "samples_per_trial": None if exploit_chunks is None else chunk_samples * exploit_chunks,
            "exploit_fraction": pbt.get("exploit_fraction"),
        },
    }


def metric_definition(metric):
    try:
        from reports.write_metrics_summary import METRIC_DEFINITIONS, SHOWCASE_METRIC_DEFINITION

        if metric in METRIC_DEFINITIONS:
            return METRIC_DEFINITIONS[metric]
        if metric == "validation_working_point_mistag_percent":
            return SHOWCASE_METRIC_DEFINITION
    except Exception:
        pass
    return {
        "display_name": metric,
        "formula": "See the configured Weaver metric implementation.",
        "note": "Metric definition was not found in the reporting registry.",
    }


def run_contract(config, command, backend_name):
    shared = config["shared"]
    pbt = config["pbt"]
    checkpoint = Path(shared["checkpoint"])
    initial_state = Path(shared["initial_state"]) if shared.get("initial_state") else None
    initial_optimizer = Path(shared["initial_optimizer"]) if shared.get("initial_optimizer") else None
    datasets = {
        "train_dataset": shared.get("dataset"),
        "validation_dataset": shared.get("validation_dataset") or shared.get("dataset"),
        "data_extension": shared.get("data_extension"),
        "train_suffix": shared.get("train_suffix"),
        "validation_suffix": shared.get("validation_suffix"),
        "proxy_validation": shared.get("proxy_validation"),
        "resolved_files": resolved_input_data_files(config),
    }
    return {
        "method_name": pbt.get("strategy", "exploit_mutate"),
        "backend": backend_name,
        "command": list(command or []),
        "timestamp": utc_now(),
        "git": git_metadata(),
        "seed": int(shared["seed"]),
        "gpus": [slot.get("label", slot.get("gpu")) if isinstance(slot, dict) else str(slot) for slot in config.get("slots", [])],
        "datasets": datasets,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint) if checkpoint.is_file() else None,
            "initial_state": None if initial_state is None else {
                "path": str(initial_state),
                "sha256": sha256(initial_state) if initial_state.is_file() else None,
                "epoch": int(shared.get("initial_epoch", -1)),
            },
        },
        "optimizer_checkpoint": None if initial_optimizer is None else {
            "path": str(initial_optimizer),
            "sha256": sha256(initial_optimizer) if initial_optimizer.is_file() else None,
            "mode": shared.get("initial_optimizer_mode", "raw"),
            "damping": shared.get("initial_optimizer_damping", 0.1),
        },
        "metric": {
            "name": pbt["metric"],
            "mode": pbt["mode"],
            "definition": metric_definition(pbt["metric"]),
        },
        "baseline_evaluation": {
            "configured_metric_value": pbt.get("baseline_metric_value"),
            "configured_source": "config" if pbt.get("baseline_metric_value") is not None else None,
            "measured_metric_value": pbt.get("runtime_baseline_metric_value"),
            "measured_source": "initial_evaluation" if pbt.get("runtime_baseline_metric_value") is not None else None,
            "guard_tolerance": pbt.get("baseline_guard_tolerance"),
        },
        "schedule": configured_intervals(config),
    }


def append_event(run_dir, event_type, payload):
    ensure_run_layout(run_dir)
    event = {
        "time": utc_now(),
        "event_type": event_type,
        **payload,
    }
    path = Path(run_dir) / EVENTS_NAME
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return event


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


def write_atomic_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    os.replace(temporary, path)
    return path


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


def read_events(run_dir):
    path = Path(run_dir) / EVENTS_NAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def build_summary(run_dir, manifest):
    rows = read_metrics_rows(run_dir)
    events = read_events(run_dir)
    metric = _metric_name(manifest)
    mode = _metric_mode(manifest)
    baseline = baseline_record(manifest)
    configured_baseline = configured_baseline_record(manifest)
    best = manifest.get("best")
    best_value = None if best is None else best.get("metric_value")
    final_row = rows[-1] if rows else None
    final_generation = None
    if manifest.get("generations"):
        final_generation = max(manifest["generations"], key=lambda item: item.get("index", -1))
    final_best = None
    if final_generation:
        final_rows = [row for row in rows if row["generation"] == final_generation.get("index")]
        final_best = final_best_row(final_rows, mode)
    baseline_value = None if baseline is None else baseline.get("metric_value")
    return {
        "schema_version": 1,
        "experiment": manifest.get("experiment"),
        "status": manifest.get("status"),
        "method": manifest.get("method") or manifest.get("run", {}).get("method_name"),
        "metric": {
            "name": metric,
            "mode": mode,
            "definition": metric_definition(metric),
        },
        "starting_checkpoint": manifest.get("initial_resume") or manifest.get("checkpoint"),
        "dataset": manifest.get("datasets") or manifest.get("run", {}).get("datasets"),
        "population": sorted(manifest.get("members", {})),
        "schedule": (manifest.get("run") or {}).get("schedule"),
        "baseline": baseline,
        "configured_baseline": configured_baseline,
        "best": best,
        "final_best": final_best,
        "winning_trial": None if best is None else best.get("member"),
        "best_improvement_vs_baseline": relative_change(mode, baseline_value, best_value),
        "final_improvement_vs_baseline": relative_change(
            mode,
            baseline_value,
            None if final_best is None else final_best.get("optimization_metric_value"),
        ),
        "lr_trajectory": {
            trial: [
                {"generation": row["generation"], "samples_seen": row["samples_seen"], "LR": row["LR"]}
                for row in rows
                if row["trial"] == trial and row["LR"] is not None
            ]
            for trial in sorted({row["trial"] for row in rows})
        },
        "exploit_history": [
            event
            for event in events
            if event.get("event_type") in {"exploit", "weight_copy", "optimizer_copy", "lr_change"}
        ],
        "event_counts": {
            event_type: sum(1 for event in events if event.get("event_type") == event_type)
            for event_type in sorted({event.get("event_type") for event in events})
        },
        "evaluation": evaluation_metadata(manifest),
        "plots": {
            **{
                name: str(Path("plots") / filename)
                for name, filename in PLOT_NAMES.items()
                if name not in CONDITIONAL_PLOT_NAMES or (Path(run_dir) / "plots" / filename).is_file()
            },
            "physics_performance": str(Path("plots") / "report" / "physics_performance.png"),
            "background_efficiency_curves": str(Path("plots") / "diagnostics" / "background_efficiency_curves.png"),
            "btag_mistag_table_csv": str(Path("plots") / "report" / "btag_mistag_tables.csv"),
            "ctag_mistag_table_csv": str(Path("plots") / "report" / "ctag_mistag_tables.csv"),
            "exploit_table_csv": EXPLOIT_TABLE_NAME,
        },
        "checkpoints": {
            "global_best_state": None if best is None else best.get("state_path"),
            "global_best_optimizer": None if best is None else best.get("optimizer_path"),
            "global_best_metadata": None if best is None else best.get("metadata_path"),
        },
    }


def write_summary_json(run_dir, manifest):
    ensure_run_layout(run_dir)
    path = Path(run_dir) / SUMMARY_NAME
    summary = build_summary(run_dir, manifest)
    atomic_json(path, summary)
    return path


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generation_sample_map(rows):
    out = {}
    for row in rows:
        generation = row.get("generation")
        samples_seen = row.get("samples_seen")
        if generation is None or samples_seen is None:
            continue
        out[generation] = max(samples_seen, out.get(generation, 0))
    return out


def _compact_trial(name):
    return str(name).replace("member_", "m")


def _controller_value(row):
    value = row.get(CONTROLLER_OBJECTIVE_COLUMN)
    if value is not None:
        return value
    return row.get("validation_working_point_mistag_percent")


def selected_generation_rows(rows, mode):
    """Per-generation row actually chosen by the configured selection metric.

    This must track the real PBT ranking (same metric/mode as
    `best_worker_in_generation` in metrics.py), not the HEP controller
    objective, so historical max-mode runs plot the trial the algorithm
    truly selected rather than whichever trial happens to have the best
    fixed-WP mistag mean that generation.
    """
    selected = []
    for generation in sorted({row.get("generation") for row in rows if row.get("generation") is not None}):
        row = final_best_row([item for item in rows if item.get("generation") == generation], mode)
        if row is not None:
            selected.append(row)
    return selected


def _row_for_checkpoint(rows, checkpoint):
    if not checkpoint:
        return None
    generation = checkpoint.get("generation")
    member = checkpoint.get("member")
    for row in rows:
        if row.get("generation") == generation and row.get("trial") == member:
            return row
    return None


def _baseline_controller_record(manifest):
    initial = manifest.get("initial_evaluation") or {}
    metrics = initial.get("metrics") or {}
    if initial.get("status") != "completed" or not metrics:
        return None
    value = controller_objective_mistag(metrics)
    if value is None:
        return None
    return {"samples_seen": 0, "trial": "pretrained", "controller_objective": value}


def _completed_initial_evaluation_metrics(manifest):
    initial = manifest.get("initial_evaluation") or {}
    metrics = initial.get("metrics") or {}
    if initial.get("status") != "completed" or not metrics:
        return None
    return metrics


def _baseline_fixed_working_point_values(manifest):
    metrics = _completed_initial_evaluation_metrics(manifest)
    return None if metrics is None else fixed_working_point_values(metrics)


def _baseline_fixed_working_point_uncertainties(manifest):
    metrics = _completed_initial_evaluation_metrics(manifest)
    return None if metrics is None else fixed_working_point_uncertainties(metrics)


def _global_best_metrics(manifest):
    metrics = (manifest.get("best") or {}).get("metrics") or {}
    return metrics or None


def _mark_checkpoint(ax, row, label, marker, color, y_key=None):
    if row is None or row.get("samples_seen") is None:
        return
    value = _controller_value(row) if y_key is None else row.get(y_key)
    if value is None:
        return
    ax.scatter([row["samples_seen"]], [value], marker=marker, s=100, color=color, edgecolor="black", zorder=6, label=label)


def _set_log_if_positive(ax, values):
    values = [value for value in values if value is not None and value > 0]
    if values and max(values) / min(values) >= 8.0:
        ax.set_yscale("log")


def plot_training_evolution(run_dir, manifest, rows, events):
    plt = _plot_setup()
    mode = _metric_mode(manifest)
    selected = selected_generation_rows(rows, mode)
    best_row = _row_for_checkpoint(rows, manifest.get("best") or {})
    final_row = selected[-1] if selected else None
    baseline = _baseline_controller_record(manifest)
    evaluation = evaluation_metadata(manifest)
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.2), sharex=True, gridspec_kw={"height_ratios": [1.45, 1.05, 1.15]})
    fig.subplots_adjust(left=0.08, right=0.82, top=0.88, bottom=0.07, hspace=0.30)
    ax_objective, ax_lr, ax_events = axes

    if selected:
        xs = [row["samples_seen"] for row in selected]
        ys = [_controller_value(row) for row in selected]
        ax_objective.plot(xs, ys, marker="o", markersize=5.5, linestyle=":", linewidth=1.3, color="#2f5aa0", label="selected trial")
    if baseline:
        ax_objective.scatter([0], [baseline["controller_objective"]], marker="o", s=90, facecolor="white", edgecolor="#2f5aa0", linewidth=1.6, zorder=6, label="pretrained start")
    _mark_checkpoint(ax_objective, best_row, "global best", "*", "black")
    _mark_checkpoint(ax_objective, final_row, "final checkpoint", "s", "#8fb7dc")
    ax_objective.set_ylabel("mean fixed-WP mistag [%]")
    ax_objective.set_title("Controller objective (mean fixed-WP mistag, lower = better)", loc="left", fontsize=11, fontweight="bold")
    ax_objective.grid(True, color="0.9", linewidth=0.6)
    ax_objective.legend(frameon=False, fontsize=8.3, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    for trial in sorted({row["trial"] for row in rows}):
        series = [row for row in rows if row["trial"] == trial and row.get("LR") is not None and row.get("samples_seen") is not None]
        if not series:
            continue
        ax_lr.plot([row["samples_seen"] for row in series], [row["LR"] for row in series], marker="o", markersize=4.5, linestyle=":", linewidth=1.1, alpha=0.8, label=_compact_trial(trial))
    if best_row and best_row.get("LR") is not None:
        ax_lr.scatter([best_row["samples_seen"]], [best_row["LR"]], marker="*", s=110, color="black", zorder=6)
    if final_row and final_row.get("LR") is not None:
        ax_lr.scatter([final_row["samples_seen"]], [final_row["LR"]], marker="s", s=82, color="#8fb7dc", edgecolor="black", zorder=6)
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.set_title("Learning-rate trajectories", loc="left", fontsize=11, fontweight="bold")
    ax_lr.grid(True, color="0.9", linewidth=0.6)
    ax_lr.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    trials = sorted({row["trial"] for row in rows})
    y_by_trial = {trial: index for index, trial in enumerate(trials)}
    sample_by_generation = generation_sample_map(rows)
    for trial, y in y_by_trial.items():
        ax_events.hlines(y, 0, max(sample_by_generation.values(), default=1), color="0.88", linewidth=1.0, zorder=1)
    for row in selected:
        trial = row.get("trial")
        if trial in y_by_trial:
            ax_events.scatter([row["samples_seen"]], [y_by_trial[trial]], marker="o", s=42, color="#2f5aa0", zorder=4)
    for event in events:
        if event.get("event_type") not in {"exploit", "weight_copy", "optimizer_copy"}:
            continue
        donor = event.get("donor")
        recipient = event.get("recipient")
        generation = event.get("generation")
        x = sample_by_generation.get(generation)
        if x is None:
            continue
        if donor in y_by_trial:
            ax_events.scatter([x], [y_by_trial[donor]], marker="^", color="#cf6f2e", s=54, zorder=5)
        if donor in y_by_trial and recipient in y_by_trial:
            ax_events.annotate("", xy=(x, y_by_trial[recipient]), xytext=(x, y_by_trial[donor]), arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 1.2}, zorder=3)
    if best_row and best_row.get("trial") in y_by_trial:
        ax_events.scatter([best_row["samples_seen"]], [y_by_trial[best_row["trial"]]], marker="*", s=120, color="black", zorder=6, label="global best")
    if final_row and final_row.get("trial") in y_by_trial:
        ax_events.scatter([final_row["samples_seen"]], [y_by_trial[final_row["trial"]]], marker="s", s=82, color="#8fb7dc", edgecolor="black", zorder=6, label="final checkpoint")
    ax_events.set_yticks(list(y_by_trial.values()))
    ax_events.set_yticklabels([_compact_trial(trial) for trial in trials])
    ax_events.set_ylabel("selected/donor trial")
    ax_events.set_xlabel("samples seen")
    ax_events.set_title("Selected trials and exploit/copy events", loc="left", fontsize=11, fontweight="bold")
    ax_events.grid(True, axis="x", color="0.9", linewidth=0.6)
    ax_events.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    fig.suptitle("PBT training evolution", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.955,
        f"{manifest.get('experiment', Path(run_dir).name)} | evaluation: {evaluation.get('evaluation_type', 'n/a')} | "
        f"PBT selection metric: {_metric_name(manifest)} ({mode})\n"
        "Objective panel below is a separate, always lower-is-better HEP presentation quantity -- not the selection metric.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES["training_evolution"]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_working_point_evolution(run_dir, manifest, rows):
    plt = _plot_setup()
    mode = _metric_mode(manifest)
    selected = selected_generation_rows(rows, mode)
    best_row = _row_for_checkpoint(rows, manifest.get("best") or {})
    baseline_values = _baseline_fixed_working_point_values(manifest)
    baseline_uncertainties = _baseline_fixed_working_point_uncertainties(manifest)
    evaluation = evaluation_metadata(manifest)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.79, top=0.85, bottom=0.08, hspace=0.32)
    groups = (
        ("b", "b-tag fixed-efficiency mistag", axes[0]),
        ("c", "c-tag fixed-efficiency mistag", axes[1]),
    )
    for tag, title, ax in groups:
        plotted = []
        for point in FIXED_WORKING_POINTS:
            if point["tag"] != tag:
                continue
            column = point["column"]
            rank = WORKING_POINT_STYLE_RANK[(tag, point["efficiency"])]
            marker = WORKING_POINT_MARKERS[rank]
            linestyle = WORKING_POINT_LINESTYLES[rank]
            color = FLAVOR_COLORS[point["background"]]

            xs = [row["samples_seen"] for row in selected if row.get(column) is not None]
            ys = [row[column] for row in selected if row.get(column) is not None]
            lower = [row.get(f"{column}_err_low") or 0.0 for row in selected if row.get(column) is not None]
            upper = [row.get(f"{column}_err_high") or 0.0 for row in selected if row.get(column) is not None]

            baseline_value = (baseline_values or {}).get(column)
            if baseline_value is not None:
                baseline_lower = (baseline_uncertainties or {}).get(f"{column}_err_low") or 0.0
                baseline_upper = (baseline_uncertainties or {}).get(f"{column}_err_high") or 0.0
                xs = [0, *xs]
                ys = [baseline_value, *ys]
                lower = [baseline_lower, *lower]
                upper = [baseline_upper, *upper]

            if not xs:
                continue
            plotted.extend(ys)
            ax.errorbar(
                xs,
                ys,
                yerr=[lower, upper],
                marker=marker,
                markersize=5.5,
                linestyle=linestyle,
                linewidth=1.1,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.5,
                alpha=0.92,
                label=point["label"],
            )
        if best_row and best_row.get("samples_seen") is not None:
            ax.axvline(best_row["samples_seen"], color="0.3", linestyle=":", linewidth=1.0, alpha=0.6)
        _set_log_if_positive(ax, plotted)
        ax.set_ylabel("mistag [%]")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8.2, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0, handlelength=2.2)
    axes[-1].set_xlabel("samples seen")

    fig.suptitle("Fixed working-point mistag evolution", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.935,
        f"{manifest.get('experiment', Path(run_dir).name)} | evaluation: {evaluation.get('evaluation_type', 'n/a')}\n"
        "Markers = measured checkpoints; error bars = 68% Wilson interval; lines guide the eye only; "
        "dotted vertical line = selected/global-best checkpoint.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES["working_point_evolution"]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


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


def write_existing_physics_reports(run_dir, manifest):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    outputs = {}
    from reports.plot_background_efficiency_curves import plot_manifest as plot_background_efficiency
    from reports.plot_mistag_tables import collect_tables, write_csv
    from reports.plot_physics_performance import plot_manifest as plot_physics_performance

    physics_path = plot_physics_performance(manifest_path)
    outputs["physics_performance"] = str(physics_path)
    manifest["physics_performance_plot"] = str(physics_path)

    curves_path = plot_background_efficiency(manifest_path)
    outputs["background_efficiency_curves"] = str(curves_path)
    manifest["background_efficiency_curves_plot"] = str(curves_path)

    for tag, efficiencies in {"c": (0.5, 0.8), "b": (0.8, 0.9)}.items():
        tables = collect_tables(
            [(manifest.get("experiment", run_dir.name), manifest_path)],
            tag=tag,
            efficiencies=efficiencies,
            member="best_physics",
            manifests={manifest_path: manifest},
        )
        csv_path = run_dir / "plots" / "report" / f"{tag}tag_mistag_tables.csv"
        write_csv(csv_path, tables, tag)
        key = f"{tag}tag_mistag_table_csv"
        outputs[key] = str(csv_path)
        manifest[key] = str(csv_path)
    return outputs


def plot_baseline_comparison(run_dir, manifest):
    """HEP observable comparison: pretrained baseline vs. the selected
    (global-best) checkpoint at every fixed working point, absolute mistag
    plus the relative gain from training. Skipped (returns None) unless both
    a measured baseline and a global-best checkpoint with metrics exist.
    """
    baseline_metrics = _completed_initial_evaluation_metrics(manifest)
    selected_metrics = _global_best_metrics(manifest)
    if not baseline_metrics or not selected_metrics:
        return None

    plt = _plot_setup()
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 7.4), gridspec_kw={"width_ratios": [1.55, 1.0], "hspace": 0.48, "wspace": 0.30}
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.11)

    best = manifest.get("best") or {}
    selected_label = f"selected ({best.get('member', 'global best')}, gen {best.get('generation', 'n/a')})"

    for tag, (ax_abs, ax_delta) in zip(("b", "c"), axes):
        points = [point for point in FIXED_WORKING_POINTS if point["tag"] == tag]
        labels = [f"{point['background']} bkg\n{tag}-eff {int(round(point['efficiency'] * 100))}%" for point in points]
        colors = [FLAVOR_COLORS[point["background"]] for point in points]
        baseline_vals = [_mistag_percent(baseline_metrics, tag, point["efficiency"], point["background"]) for point in points]
        baseline_errs = [
            fixed_working_point_uncertainty(baseline_metrics, tag, point["efficiency"], point["background"]) for point in points
        ]
        selected_vals = [_mistag_percent(selected_metrics, tag, point["efficiency"], point["background"]) for point in points]
        selected_errs = [
            fixed_working_point_uncertainty(selected_metrics, tag, point["efficiency"], point["background"]) for point in points
        ]

        x_positions = list(range(len(points)))
        width = 0.36
        baseline_x = [x - width / 2 for x in x_positions]
        selected_x = [x + width / 2 for x in x_positions]
        ax_abs.bar(
            baseline_x, [value or 0.0 for value in baseline_vals], width=width,
            color=colors, alpha=0.40, hatch="//", edgecolor="0.3", linewidth=0.6, label="pretrained baseline",
        )
        ax_abs.bar(
            selected_x, [value or 0.0 for value in selected_vals], width=width,
            color=colors, alpha=0.95, edgecolor="0.2", linewidth=0.6, label=selected_label,
        )
        for x, value, err in zip(baseline_x, baseline_vals, baseline_errs):
            if value is None:
                continue
            lower, upper = err[:2] if err else (0.0, 0.0)
            ax_abs.errorbar([x], [value], yerr=[[lower], [upper]], fmt="none", ecolor="0.2", elinewidth=0.9, capsize=2.5, zorder=5)
        for x, value, err in zip(selected_x, selected_vals, selected_errs):
            if value is None:
                continue
            lower, upper = err[:2] if err else (0.0, 0.0)
            ax_abs.errorbar([x], [value], yerr=[[lower], [upper]], fmt="none", ecolor="0.2", elinewidth=0.9, capsize=2.5, zorder=5)
        ax_abs.set_xticks(x_positions)
        ax_abs.set_xticklabels(labels, fontsize=8)
        ax_abs.set_ylabel("mistag [%]")
        ax_abs.set_title(f"{tag}-tag: baseline vs. selected", loc="left", fontsize=10.5, fontweight="bold")
        ax_abs.grid(True, axis="y", color="0.9", linewidth=0.6)
        peak = max(
            [(v or 0.0) + ((e[1] if e else 0.0)) for v, e in zip(baseline_vals, baseline_errs)]
            + [(v or 0.0) + ((e[1] if e else 0.0)) for v, e in zip(selected_vals, selected_errs)]
            or [1.0]
        )
        ax_abs.set_ylim(0, peak * 1.28 if peak > 0 else 1.0)

        deltas = []
        for base, selected in zip(baseline_vals, selected_vals):
            if not base or selected is None:
                deltas.append(None)
            else:
                deltas.append(100.0 * (base - selected) / base)
        bar_colors = ["#2ca02c" if (delta is not None and delta >= 0) else "#d62728" for delta in deltas]
        ax_delta.bar(x_positions, [delta or 0.0 for delta in deltas], color=bar_colors, alpha=0.85, edgecolor="0.25", linewidth=0.6)
        for x, delta in zip(x_positions, deltas):
            if delta is None:
                continue
            ax_delta.text(x, delta, f"{delta:+.0f}%", ha="center", va="bottom" if delta >= 0 else "top", fontsize=7.4)
        ax_delta.axhline(0, color="0.3", linewidth=0.8)
        ax_delta.set_xticks(x_positions)
        ax_delta.set_xticklabels(labels, fontsize=8)
        ax_delta.set_ylabel("relative gain [%]")
        ax_delta.set_title("mistag reduction vs. baseline", loc="left", fontsize=10.5, fontweight="bold")
        ax_delta.grid(True, axis="y", color="0.9", linewidth=0.6)

    fig.suptitle("Baseline vs. selected-model mistag", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.915,
        f"{manifest.get('experiment', Path(run_dir).name)} | positive gain = lower mistag after training; "
        "hatched = pretrained, solid = selected checkpoint",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES["baseline_comparison"]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


TIERED_METRICS_COLUMNS = (
    "generation",
    "samples_seen",
    "tier",
    "member",
    "dataset",
    "suffix",
    "status",
    "rank",
    "population_size",
    "metric_name",
    "metric_value",
    CONTROLLER_OBJECTIVE_COLUMN,
    "validation_working_point_mistag_percent",
    *FIXED_WORKING_POINT_COLUMNS,
    *FIXED_WORKING_POINT_UNCERTAINTY_COLUMNS,
)
TIER_ORDER = ("control", "monitor", "full", "full_holdout")


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


def _paired_tier_values(manifest, tier_a, tier_b):
    """(x, y, member, generation) tuples for every (member, generation) that
    has a valid metric under both tier_a and tier_b -- the paired
    observations proxy-vs-monitor/full correlation requires.
    """
    by_key = {}
    for round_record in manifest.get("tiered_evaluations", []):
        tier = round_record.get("tier")
        if tier not in (tier_a, tier_b):
            continue
        metric_name = round_record.get("metric_name")
        generation = round_record.get("generation")
        for member, record in (round_record.get("members") or {}).items():
            value = (record.get("metrics") or {}).get(metric_name)
            if value is None or not math.isfinite(float(value)):
                continue
            by_key.setdefault((generation, member), {})[tier] = float(value)
    pairs = []
    for (generation, member), values in sorted(by_key.items(), key=lambda item: (item[0][0] or -999, item[0][1] or "")):
        if tier_a in values and tier_b in values:
            pairs.append((values[tier_a], values[tier_b], member, generation))
    return pairs


def tier_correlation(manifest, tier_a, tier_b):
    """Pearson r and Spearman rho between tier_a and tier_b's metric values
    at every (member, generation) evaluated on both. Returns None (with a
    reason) rather than a number when there are too few paired points for a
    correlation to mean anything -- never fabricate significance from n<3.
    """
    pairs = _paired_tier_values(manifest, tier_a, tier_b)
    n = len(pairs)
    if n < 3:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "insufficient_paired_observations"}
    try:
        from scipy import stats
    except ImportError:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "scipy_unavailable"}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    pearson_r, pearson_p = stats.pearsonr(xs, ys)
    spearman_rho, spearman_p = stats.spearmanr(xs, ys)
    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "reason": None,
    }


def _paired_round_rankings(manifest, tier_a, tier_b):
    """(generation, ranking_a, ranking_b, shared_members) for every
    generation where both tiers recorded a round -- the full-ordering data
    ranking-agreement/top-k-agreement is computed from.
    """
    rounds_by_generation = {}
    for round_record in manifest.get("tiered_evaluations", []):
        rounds_by_generation.setdefault(round_record.get("generation"), {})[round_record.get("tier")] = round_record
    paired = []
    for generation, by_tier in sorted(rounds_by_generation.items(), key=lambda item: item[0] if item[0] is not None else -999):
        if tier_a not in by_tier or tier_b not in by_tier:
            continue
        ranking_a = by_tier[tier_a].get("ranking") or []
        ranking_b = by_tier[tier_b].get("ranking") or []
        shared = [name for name in ranking_a if name in ranking_b]
        if len(shared) < 2:
            continue
        paired.append((generation, ranking_a, ranking_b, shared))
    return paired


def ranking_agreement(manifest, tier_a, tier_b):
    """Per paired-generation ranking agreement between tier_a and tier_b:
    Spearman rho of rank positions (restricted to members ranked by both),
    plus top-1 and top-min(3,n) overlap -- "does the proxy pick the same
    winner/leaders as monitor/full", not just "are the raw values correlated".
    """
    paired = _paired_round_rankings(manifest, tier_a, tier_b)
    results = []
    for generation, ranking_a, ranking_b, shared in paired:
        rank_a = {name: index for index, name in enumerate(ranking_a) if name in shared}
        rank_b = {name: index for index, name in enumerate(ranking_b) if name in shared}
        ordered = sorted(shared, key=lambda name: rank_a[name])
        ranks_a = [rank_a[name] for name in ordered]
        ranks_b = [rank_b[name] for name in ordered]
        spearman_rho = None
        if len(ordered) >= 3:
            try:
                from scipy import stats

                spearman_rho, _ = stats.spearmanr(ranks_a, ranks_b)
                spearman_rho = float(spearman_rho)
            except ImportError:
                spearman_rho = None
        top1_a = ranking_a[0] if ranking_a else None
        top1_b = ranking_b[0] if ranking_b else None
        k = min(3, len(shared))
        top_k_a = set(ranking_a[:k])
        top_k_b = set(ranking_b[:k])
        results.append(
            {
                "generation": generation,
                "members_compared": len(shared),
                "spearman_rho": spearman_rho,
                "top1_agrees": bool(top1_a is not None and top1_a == top1_b),
                "top_k": k,
                "top_k_overlap_fraction": len(top_k_a & top_k_b) / k if k else None,
            }
        )
    return results


def best_checkpoint_by_tier(manifest):
    """The (generation, member, metric_value) with the best metric under
    each tier, across every recorded round of that tier -- so we can ask
    "does control's favorite checkpoint match monitor/full's".
    """
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    best_by_tier = {}
    for round_record in manifest.get("tiered_evaluations", []):
        tier = round_record.get("tier")
        metric_name = round_record.get("metric_name")
        for member, record in (round_record.get("members") or {}).items():
            value = (record.get("metrics") or {}).get(metric_name)
            if value is None or not math.isfinite(float(value)):
                continue
            value = float(value)
            candidate = {"generation": round_record.get("generation"), "member": member, "metric_value": value}
            current = best_by_tier.get(tier)
            if current is None or _better(mode, value, current["metric_value"]):
                best_by_tier[tier] = candidate
    return best_by_tier


def proxy_selected_checkpoint_other_tiers(manifest):
    """What do monitor/full say about the checkpoint control-based PBT
    actually selected as global best? None entries mean that checkpoint was
    never evaluated on that tier.
    """
    best = manifest.get("best") or {}
    generation = best.get("generation")
    member = best.get("member")
    out = {}
    for round_record in manifest.get("tiered_evaluations", []):
        if round_record.get("generation") != generation or round_record.get("tier") == "control":
            continue
        record = (round_record.get("members") or {}).get(member)
        if record is None:
            continue
        metric_name = round_record.get("metric_name")
        out[round_record.get("tier")] = {
            "metric_value": (record.get("metrics") or {}).get(metric_name),
            "status": record.get("status"),
        }
    return {"generation": generation, "member": member, "tiers": out}


def proxy_overfitting_cases(manifest, tier_b="monitor"):
    """(member, generation_from, generation_to) triples where control
    improved but tier_b (monitor by default) did not, for the same member
    across the same pair of paired generations -- the explicit "proxy got
    better, physics didn't" signal this whole diagnostic exists to catch.
    A control-only improvement must never be reported as a confirmed
    physics result; this is the concrete evidence for when that would be wrong.
    """
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    paired_generations = sorted(
        {round_record.get("generation") for round_record in manifest.get("tiered_evaluations", []) if round_record.get("tier") == tier_b}
    )
    by_generation_tier = {}
    for round_record in manifest.get("tiered_evaluations", []):
        by_generation_tier.setdefault(round_record.get("generation"), {})[round_record.get("tier")] = round_record

    def metric_for(generation, tier, member):
        round_record = by_generation_tier.get(generation, {}).get(tier)
        if not round_record:
            return None
        metric_name = round_record.get("metric_name")
        value = (round_record.get("members", {}).get(member, {}).get("metrics") or {}).get(metric_name)
        return None if value is None else float(value)

    cases = []
    for previous_generation, current_generation in zip(paired_generations, paired_generations[1:]):
        control_prev = by_generation_tier.get(previous_generation, {}).get("control")
        control_curr = by_generation_tier.get(current_generation, {}).get("control")
        if not control_prev or not control_curr:
            continue
        members = set(control_prev.get("members", {})) & set(control_curr.get("members", {}))
        for member in sorted(members):
            control_before = metric_for(previous_generation, "control", member)
            control_after = metric_for(current_generation, "control", member)
            other_before = metric_for(previous_generation, tier_b, member)
            other_after = metric_for(current_generation, tier_b, member)
            if None in (control_before, control_after, other_before, other_after):
                continue
            control_improved = _better(mode, control_after, control_before)
            other_improved = _better(mode, other_after, other_before)
            if control_improved and not other_improved:
                cases.append(
                    {
                        "member": member,
                        "generation_from": previous_generation,
                        "generation_to": current_generation,
                        "control_before": control_before,
                        "control_after": control_after,
                        f"{tier_b}_before": other_before,
                        f"{tier_b}_after": other_after,
                    }
                )
    return cases


def plot_proxy_diagnostics(run_dir, manifest):
    """Does the control proxy actually track monitor/full? Evolution per
    tier, paired correlation, ranking agreement, and explicit
    proxy-overfitting cases. Returns None (no plot) if no monitor/full
    rounds were ever recorded -- nothing to diagnose.
    """
    rounds = manifest.get("tiered_evaluations", [])
    if not any(round_record.get("tier") in ("monitor", "full", "full_holdout") for round_record in rounds):
        return None

    plt = _plot_setup()
    events = read_events(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.08, hspace=0.38, wspace=0.28)
    ax_evolution, ax_control_monitor, ax_control_holdout, ax_agreement = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    tier_colors = {"control": "#2f5aa0", "monitor": "#cf6f2e", "full": "#59a14f", "full_holdout": "#b07aa1"}
    for tier in TIER_ORDER:
        tier_rounds = [r for r in rounds if r.get("tier") == tier]
        if not tier_rounds:
            continue
        xs, means, mins, maxs = [], [], [], []
        for round_record in sorted(tier_rounds, key=lambda item: item.get("generation") if item.get("generation") is not None else -999):
            metric_name = round_record.get("metric_name")
            values = [
                float((record.get("metrics") or {}).get(metric_name))
                for record in (round_record.get("members") or {}).values()
                if (record.get("metrics") or {}).get(metric_name) is not None
                and math.isfinite(float((record.get("metrics") or {}).get(metric_name)))
            ]
            if not values:
                continue
            xs.append(_tiered_round_samples_seen(manifest, round_record.get("generation")))
            means.append(sum(values) / len(values))
            mins.append(min(values))
            maxs.append(max(values))
        if not xs:
            continue
        color = tier_colors.get(tier, "0.4")
        ax_evolution.plot(xs, means, marker="o", markersize=5, linestyle="-", linewidth=1.4, color=color, label=f"{tier} (population mean)")
        ax_evolution.fill_between(xs, mins, maxs, color=color, alpha=0.15, linewidth=0)
    for event in events:
        if event.get("event_type") != "exploit":
            continue
        x = _tiered_round_samples_seen(manifest, event.get("generation"))
        ax_evolution.axvline(x, color="0.75", linestyle=":", linewidth=0.8, zorder=1)
    ax_evolution.set_xlabel("samples seen")
    ax_evolution.set_ylabel("metric value (population min/mean/max)")
    ax_evolution.set_title("Tiered validation evolution", loc="left", fontsize=10.5, fontweight="bold")
    ax_evolution.grid(True, color="0.9", linewidth=0.6)
    ax_evolution.legend(frameon=False, fontsize=8, loc="best")

    # Fidelity diagnostics deliberately use full_holdout, not plain "full":
    # full contains the exact control/monitor events (see the dataset
    # suitability note), so it is not an independent check of the proxy.
    for ax, tier_b, label in (
        (ax_control_monitor, "monitor", "control vs. monitor"),
        (ax_control_holdout, "full_holdout", "control vs. full_holdout (independent)"),
    ):
        correlation = tier_correlation(manifest, "control", tier_b)
        pairs = _paired_tier_values(manifest, "control", tier_b)
        if pairs:
            xs = [pair[0] for pair in pairs]
            ys = [pair[1] for pair in pairs]
            ax.scatter(xs, ys, s=28, color=tier_colors.get(tier_b, "0.4"), edgecolor="0.2", linewidth=0.4, zorder=3)
            lo, hi = min(xs + ys), max(xs + ys)
            if hi > lo:
                ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=0.9, zorder=2, label="y = x")
        if correlation["reason"] == "insufficient_paired_observations":
            caption = f"n={correlation['n']} paired points -- too few for a meaningful correlation"
        elif correlation["reason"]:
            caption = f"n={correlation['n']}, correlation unavailable ({correlation['reason']})"
        else:
            caption = f"n={correlation['n']}  Pearson r={correlation['pearson_r']:.2f}  Spearman rho={correlation['spearman_rho']:.2f}"
        ax.set_xlabel(f"control {rounds[0].get('metric_name') if rounds else ''}")
        ax.set_ylabel(f"{tier_b} {rounds[0].get('metric_name') if rounds else ''}")
        ax.set_title(label, loc="left", fontsize=10.5, fontweight="bold")
        ax.text(0.02, 0.98, caption, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.3")
        ax.grid(True, color="0.9", linewidth=0.6)

    agreement_rows = ranking_agreement(manifest, "control", "monitor") or ranking_agreement(manifest, "control", "full_holdout")
    if agreement_rows:
        xs = [row["generation"] for row in agreement_rows]
        overlap = [row["top_k_overlap_fraction"] for row in agreement_rows]
        top1 = [1.0 if row["top1_agrees"] else 0.0 for row in agreement_rows]
        ax_agreement.plot(xs, overlap, marker="o", markersize=5, color="#4c78a8", label="top-k overlap fraction")
        ax_agreement.scatter(xs, top1, marker="s", s=36, color="#e15759", label="top-1 (winner) agrees", zorder=4)
        ax_agreement.set_ylim(-0.05, 1.05)
        ax_agreement.set_xlabel("generation")
        ax_agreement.set_ylabel("agreement")
        ax_agreement.legend(frameon=False, fontsize=8, loc="lower left")
    else:
        ax_agreement.text(0.5, 0.5, "no paired control/monitor(-or-full_holdout)\nranking rounds recorded yet", ha="center", va="center", transform=ax_agreement.transAxes, fontsize=9, color="0.4")
    ax_agreement.set_title("Ranking agreement (control vs. monitor/full_holdout)", loc="left", fontsize=10.5, fontweight="bold")
    ax_agreement.grid(True, color="0.9", linewidth=0.6)

    fig.suptitle("Proxy validation diagnostics", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.935,
        f"{manifest.get('experiment', Path(run_dir).name)} | control drives PBT decisions; monitor/full are read-only checks, never fed back\n"
        "Shaded bands = population min-max; dotted vertical lines = exploit events. A control-only improvement is provisional, not confirmed.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES["proxy_diagnostics"]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_plots(run_dir, manifest):
    ensure_run_layout(run_dir)
    rows = read_metrics_rows(run_dir)
    events = read_events(run_dir)
    plots = {
        "training_evolution": str(plot_training_evolution(run_dir, manifest, rows, events)),
        "working_point_evolution": str(plot_working_point_evolution(run_dir, manifest, rows)),
    }
    diagnostics_path = plot_proxy_diagnostics(run_dir, manifest)
    if diagnostics_path is not None:
        plots["proxy_diagnostics"] = str(diagnostics_path)
    comparison_path = plot_baseline_comparison(run_dir, manifest)
    if comparison_path is not None:
        plots["baseline_comparison"] = str(comparison_path)
    return plots


def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _tier_metric_for_member_generation(manifest, tier, generation, member):
    for round_record in manifest.get("tiered_evaluations", []):
        if round_record.get("tier") != tier or round_record.get("generation") != generation:
            continue
        metric_name = round_record.get("metric_name")
        record = (round_record.get("members") or {}).get(member)
        if record is None:
            return None
        value = (record.get("metrics") or {}).get(metric_name)
        return None if value is None else float(value)
    return None


def corroboration_status(manifest):
    """provisional / monitor-corroborated / full-corroborated for the
    control-selected global-best checkpoint: same-direction agreement with
    monitor/full measured at that SAME (member, generation) checkpoint.

    Deliberately not a significance test and not an error-bar-overlap rule
    (see write_report's effect-size/uncertainty lines, reported alongside
    this label, not folded into it) -- this label only says which tiers of
    evidence exist and agree on direction; the human reader judges magnitude.
    """
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    best = manifest.get("best") or {}
    generation, member = best.get("generation"), best.get("member")
    status = "provisional"
    details = {}
    for tier in ("monitor", "full"):
        baseline_tier = _tier_metric_for_member_generation(manifest, tier, -1, "initial_resume")
        selected_tier = _tier_metric_for_member_generation(manifest, tier, generation, member)
        if baseline_tier is None or selected_tier is None:
            details[tier] = {"available": False}
            continue
        improved = _better(mode, selected_tier, baseline_tier)
        details[tier] = {"available": True, "baseline": baseline_tier, "selected": selected_tier, "improved": improved}
        if improved:
            status = f"{tier}-corroborated"
    return status, details


def _proxy_diagnostics_report_lines(manifest, plots, proxy_diagnostics_path):
    lines = ["", "## Proxy Validation Diagnostics", f"- [Proxy validation diagnostics]({proxy_diagnostics_path})"]
    for label, correlation in (
        ("control vs. monitor", tier_correlation(manifest, "control", "monitor")),
        ("control vs. full_holdout (independent, excludes control+monitor)", tier_correlation(manifest, "control", "full_holdout")),
    ):
        if correlation["reason"] == "insufficient_paired_observations":
            lines.append(f"- {label} correlation: n={correlation['n']} paired observations -- too few for a meaningful correlation")
        elif correlation["reason"]:
            lines.append(f"- {label} correlation: unavailable ({correlation['reason']})")
        else:
            lines.append(
                f"- {label} correlation: n={correlation['n']}, Pearson r={correlation['pearson_r']:.3f}, "
                f"Spearman rho={correlation['spearman_rho']:.3f}"
            )

    best_by_tier = best_checkpoint_by_tier(manifest)
    if best_by_tier:
        bits = ", ".join(
            f"{tier}: `{info['member']}` gen {info['generation']} ({_fmt(info['metric_value'])})"
            for tier, info in best_by_tier.items()
        )
        lines.append(f"- Best checkpoint by tier: {bits}")
        if len(best_by_tier) > 1:
            agree = len({(info["member"], info["generation"]) for info in best_by_tier.values()}) == 1
            lines.append(f"- Best-checkpoint agreement across tiers: {'AGREE' if agree else 'DISAGREE'}")

    selected_other_tiers = proxy_selected_checkpoint_other_tiers(manifest)
    if selected_other_tiers["tiers"]:
        bits = ", ".join(f"{tier}: {_fmt(info.get('metric_value'))}" for tier, info in selected_other_tiers["tiers"].items())
        lines.append(
            f"- Control-selected global best (`{selected_other_tiers.get('member')}`, gen {selected_other_tiers.get('generation')}) "
            f"measured on other tiers: {bits}"
        )
    else:
        lines.append("- Control-selected global best has not been evaluated on monitor/full yet.")

    status, details = corroboration_status(manifest)
    lines.append(f"- Corroboration status: **{status}**")
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    for tier, info in details.items():
        if not info.get("available"):
            lines.append(f"  - {tier}: not available (baseline or selected checkpoint not evaluated on this tier)")
            continue
        delta = relative_change(mode, info["baseline"], info["selected"])
        lines.append(
            f"  - {tier}: baseline {_fmt(info['baseline'])} -> selected {_fmt(info['selected'])} "
            f"({'improved' if info['improved'] else 'not improved'}, "
            f"{_fmt(None if delta is None else 100.0 * delta)}% relative change)"
        )

    overfitting = proxy_overfitting_cases(manifest)
    if overfitting:
        lines.append(f"- **{len(overfitting)} proxy-overfitting case(s) detected** (control improved, monitor did not):")
        for case in overfitting[:10]:
            lines.append(
                f"  - `{case['member']}` gen {case['generation_from']}->{case['generation_to']}: "
                f"control {_fmt(case['control_before'])}->{_fmt(case['control_after'])}, "
                f"monitor {_fmt(case['monitor_before'])}->{_fmt(case['monitor_after'])}"
            )
        if len(overfitting) > 10:
            lines.append(f"  - ... and {len(overfitting) - 10} more (see tiered_metrics.csv)")
    else:
        lines.append("- No proxy-overfitting cases detected (control improved while monitor did not) in the paired generations evaluated so far.")
    return lines


def _shutdown_warning_summary(manifest):
    count = 0
    total = 0

    def scan(metrics):
        nonlocal count, total
        if metrics is None:
            return
        total += 1
        if metrics.get("validation_shutdown_warning"):
            count += 1

    scan((manifest.get("initial_evaluation") or {}).get("metrics"))
    for generation in manifest.get("generations", []):
        for worker in (generation.get("workers") or {}).values():
            scan(worker.get("metrics"))
    for round_record in manifest.get("tiered_evaluations", []):
        for record in (round_record.get("members") or {}).values():
            scan(record.get("metrics"))
    if count == 0:
        return f"No data-loader shutdown-race warnings observed across {total} evaluation(s)."
    return (
        f"Data-loader shutdown-race warning (validation_shutdown_warning) observed in {count}/{total} "
        "evaluation(s) -- treat affected metrics with extra caution."
    )


def write_report(run_dir, manifest, summary):
    path = Path(run_dir) / REPORT_NAME
    metric = summary["metric"]
    baseline = summary.get("baseline") or {}
    configured_baseline = summary.get("configured_baseline") or {}
    best = summary.get("best") or {}
    final_best = summary.get("final_best") or {}
    improvement = summary.get("best_improvement_vs_baseline")
    evaluation = summary.get("evaluation") or {}
    schedule = summary.get("schedule") or {}
    eval_schedule = schedule.get("evaluation_interval") or {}
    exploit_schedule = schedule.get("exploit_interval") or {}
    provenance = manifest.get("run") or {}
    git = provenance.get("git") or manifest.get("git") or {}
    plots = summary.get("plots") or {}

    lines = [
        f"# {summary.get('experiment')}",
        "",
        "## Results",
        f"- Evaluation type: `{evaluation.get('evaluation_type', 'n/a')}`",
        f"- Validation dataset: `{evaluation.get('validation_dataset', 'n/a')}`",
        f"- Validation suffix: `{evaluation.get('validation_suffix', 'n/a')}`",
        f"- Validation sample count: {_fmt(evaluation.get('validation_sample_count'))}",
        "- Controller objective: mean predefined fixed-WP mistag percent (lower is better; not a HEP metric)",
        f"- Configured PBT selection metric: `{metric['name']}` ({metric['mode']})",
        f"- Measured baseline: {_fmt(baseline.get('metric_value'))}",
        f"- Configured reference: {_fmt(configured_baseline.get('metric_value'))}",
        f"- Final checkpoint controller objective: {_fmt(final_best.get(CONTROLLER_OBJECTIVE_COLUMN))} by `{final_best.get('trial', 'n/a')}`",
        f"- Global best configured metric: {_fmt(best.get('metric_value'))} by `{best.get('member', 'n/a')}`",
        f"- Delta vs measured baseline: {_fmt(None if improvement is None else 100.0 * improvement)}%",
        f"- Best checkpoint: `{(summary.get('checkpoints') or {}).get('global_best_state')}`",
        "",
        "## Training Evolution",
        f"- [Training evolution]({plots.get('training_evolution', 'plots/training_evolution.png')})",
        f"- [Working-point evolution]({plots.get('working_point_evolution', 'plots/working_point_evolution.png')})",
    ]
    for trial, values in (summary.get("lr_trajectory") or {}).items():
        rendered = ", ".join(f"{item['samples_seen']}:{_fmt(item['LR'], 3)}" for item in values)
        lines.append(f"- `{trial}` samples_seen:LR = {rendered}")

    lines.extend(["", "## Exploit History", f"- [Exploit table]({plots.get('exploit_table_csv', EXPLOIT_TABLE_NAME)})"])
    exploits = [event for event in summary.get("exploit_history", []) if event.get("event_type") == "exploit"]
    if exploits:
        for event in exploits:
            lines.append(
                "- generation {generation}: `{donor}` -> `{recipient}`, donor metric {donor_metric}, recipient metric {recipient_metric}, LR {old_lr} -> {new_lr}, mutation `{mutation}`, weight `{weight_source}`, optimizer `{optimizer_source}`".format(
                    generation=event.get("generation"),
                    donor=event.get("donor"),
                    recipient=event.get("recipient"),
                    donor_metric=_fmt(event.get("donor_metric")),
                    recipient_metric=_fmt(event.get("recipient_metric")),
                    old_lr=_fmt(event.get("old_lr"), 3),
                    new_lr=_fmt(event.get("new_lr"), 3),
                    mutation=event.get("mutation"),
                    weight_source=event.get("weight_source"),
                    optimizer_source=event.get("optimizer_source"),
                )
            )
    else:
        lines.append("- No exploit events recorded.")
    skipped = [event for event in read_events(run_dir) if event.get("event_type") == "exploit_skipped"]
    lines.append(
        f"- [Skipped exploits (significance gating)]({plots.get('skipped_exploit_table_csv', SKIPPED_EXPLOIT_TABLE_NAME)}) -- {len(skipped)} donor->recipient replacement(s) declined for insufficient significance"
    )

    proxy_diagnostics_path = plots.get("proxy_diagnostics")
    if proxy_diagnostics_path:
        lines.extend(_proxy_diagnostics_report_lines(manifest, plots, proxy_diagnostics_path))

    lines.extend(
        [
            "",
            "## Physics Performance",
            f"- [Physics performance]({plots.get('physics_performance', 'plots/report/physics_performance.png')})",
            f"- [Background efficiency curves]({plots.get('background_efficiency_curves', 'plots/diagnostics/background_efficiency_curves.png')})",
            f"- [B-tag mistag CSV]({plots.get('btag_mistag_table_csv', 'plots/report/btag_mistag_tables.csv')})",
            f"- [C-tag mistag CSV]({plots.get('ctag_mistag_table_csv', 'plots/report/ctag_mistag_tables.csv')})",
        ]
    )
    baseline_comparison_path = plots.get("baseline_comparison")
    if baseline_comparison_path:
        lines.extend(
            [
                "",
                "## Baseline vs. Selected Model",
                f"- [Baseline vs. selected mistag]({baseline_comparison_path})",
            ]
        )
    pbt_config = manifest.get("config", {}).get("pbt", {})
    significance_sigma = pbt_config.get("exploit_significance_sigma")
    burn_in = pbt_config.get("burn_in_generations", 0)
    tiered_config = pbt_config.get("tiered_validation") or {}
    lines.extend(
        [
            "",
            "## Method",
            f"- Method: `{summary.get('method')}`",
            f"- Population: {len(summary.get('population') or [])} trials",
            f"- Training interval: {schedule.get('training_interval', {}).get('samples_per_trial_chunk', 'n/a')} samples/trial chunk ({schedule.get('training_interval', {}).get('epochs_per_generation', 'n/a')}x samples_per_epoch)",
            f"- Evaluation interval: every {eval_schedule.get('training_chunks', 'n/a')} training chunk(s), {eval_schedule.get('samples_per_epoch_val', 'n/a')} validation samples",
            f"- Exploit interval: {('disabled' if not exploit_schedule.get('enabled') else 'every ' + str(exploit_schedule.get('training_chunks', 'n/a')) + ' training chunk(s)')}",
            f"- Exploit significance gating: {'disabled (nominal rank order only)' if significance_sigma is None else f'{significance_sigma} sigma (combined uncertainty) required before a donor replaces a recipient'}",
            f"- Burn-in: {burn_in} generation(s) (observe-only, no exploit/controller LR action applied)",
            f"- Monitor-tier cadence: {tiered_config.get('monitor_interval_generations') or 'disabled'} generation(s), all population members, read-only",
            f"- Full-tier cadence: {tiered_config.get('full_interval_generations') or 'disabled'} generation(s), all population members, read-only",
            "",
            "## Provenance",
            f"- Starting checkpoint: `{(summary.get('starting_checkpoint') or {}).get('state_path') or (summary.get('starting_checkpoint') or {}).get('path')}`",
            f"- Git commit: `{git.get('commit')}`",
            f"- Git dirty: `{git.get('dirty')}`",
            f"- Launch command: `{provenance.get('command') or manifest.get('command')}`",
            "- [manifest.json](manifest.json)",
            "- [resolved_config.yaml](resolved_config.yaml)",
            "- [events.jsonl](events.jsonl)",
            "- [metrics.csv](metrics.csv)",
            "- [tiered_metrics.csv](tiered_metrics.csv)",
            "- [summary.json](summary.json)",
            "",
            "## Caveats",
            "- Proxy, smoke, and full validation results are reported as distinct evaluation types and should not be mixed in one scorecard.",
            "- Configured reference values are not treated as measured baselines unless a successful runtime initial evaluation exists.",
            "- Control-tier evidence alone is 'provisional' -- see Proxy Validation Diagnostics above. It is never a substitute for monitor/full corroboration.",
            f"- {_shutdown_warning_summary(manifest)}",
        ]
    )
    atomic_text(path, "\n".join(lines) + "\n")
    return path


def write_canonical_outputs(run_dir, manifest):
    ensure_run_layout(run_dir)
    run_dir = Path(run_dir)
    atomic_json(run_dir / "manifest.json", manifest)
    write_resolved_config(run_dir, manifest.get("config", {}))
    refresh_metrics_csv(run_dir, manifest)
    physics_outputs = write_existing_physics_reports(run_dir, manifest)
    tiered_metrics_path = write_tiered_metrics_csv(run_dir, manifest)
    events = read_events(run_dir)
    exploit_table = write_exploit_table(run_dir, events)
    skipped_exploit_table = write_skipped_exploits_table(run_dir, events)
    plots = write_plots(run_dir, manifest)
    manifest["canonical_artifacts"] = {
        "events": str(run_dir / EVENTS_NAME),
        "metrics": str(run_dir / METRICS_NAME),
        "tiered_metrics": str(tiered_metrics_path),
        "summary": str(run_dir / SUMMARY_NAME),
        "report": str(run_dir / REPORT_NAME),
        "plots": {
            **plots,
            **physics_outputs,
            "exploit_table_csv": str(exploit_table),
            "skipped_exploit_table_csv": str(skipped_exploit_table),
        },
        "resolved_config": str(run_dir / "resolved_config.yaml"),
    }
    manifest["updated_at"] = utc_now()
    atomic_json(run_dir / "manifest.json", manifest)
    summary_path = write_summary_json(run_dir, manifest)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_path = write_report(run_dir, manifest, summary)
    manifest["canonical_artifacts"]["report"] = str(report_path)
    atomic_json(run_dir / "manifest.json", manifest)
    return manifest["canonical_artifacts"]


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

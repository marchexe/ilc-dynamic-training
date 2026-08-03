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
    "physics_metric_over_time": "physics_metric_over_time.png",
    "learning_rate_over_time": "learning_rate_over_time.png",
    "pbt_lineage": "pbt_lineage.png",
    "best_model_progress": "best_model_progress.png",
    "final_summary": "final_summary.png",
}
METRICS_COLUMNS = (
    "step",
    "training_chunk",
    "samples_seen",
    "epoch_fraction",
    "trial",
    "LR",
    "proxy_metric",
    "best_so_far",
    "training_loss",
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


def evaluation_rows(manifest):
    metric = _metric_name(manifest)
    mode = _metric_mode(manifest)
    best = None
    rows = []
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
                    "step": training_chunk,
                    "training_chunk": training_chunk,
                    "samples_seen": samples_seen,
                    "epoch_fraction": epoch_fraction,
                    "trial": trial,
                    "LR": float(worker["lr"]) if worker.get("lr") is not None else None,
                    "proxy_metric": value,
                    "best_so_far": best,
                    "training_loss": metrics.get("train_loss"),
                }
            )
    return rows


def refresh_metrics_csv(run_dir, manifest):
    ensure_run_layout(run_dir)
    rows = evaluation_rows(manifest)
    path = Path(run_dir) / METRICS_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRICS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in METRICS_COLUMNS})
    os.replace(temporary, path)
    return path


def read_metrics_rows(run_dir):
    path = Path(run_dir) / METRICS_NAME
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            converted = dict(row)
            for key in ("step", "training_chunk", "samples_seen"):
                converted[key] = int(float(converted[key])) if converted.get(key) else None
            for key in ("epoch_fraction", "LR", "proxy_metric", "best_so_far", "training_loss"):
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
    return selector(rows, key=lambda row: row["proxy_metric"])


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
        final_rows = [row for row in rows if row["step"] == final_generation.get("index")]
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
            None if final_best is None else final_best.get("proxy_metric"),
        ),
        "lr_trajectory": {
            trial: [
                {"step": row["step"], "LR": row["LR"]}
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
        "plots": {
            **{name: str(Path("plots") / filename) for name, filename in PLOT_NAMES.items()},
            "physics_performance": str(Path("plots") / "report" / "physics_performance.png"),
            "background_efficiency_curves": str(Path("plots") / "diagnostics" / "background_efficiency_curves.png"),
            "btag_mistag_table_csv": str(Path("plots") / "report" / "btag_mistag_tables.csv"),
            "ctag_mistag_table_csv": str(Path("plots") / "report" / "ctag_mistag_tables.csv"),
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


def _finite_points(rows, x_key, y_key):
    return [
        (row[x_key], row[y_key])
        for row in rows
        if row.get(x_key) is not None and row.get(y_key) is not None and math.isfinite(float(row[y_key]))
    ]


def plot_physics_metric(run_dir, manifest, rows):
    plt = _plot_setup()
    metric = _metric_name(manifest)
    fig, ax = plt.subplots(figsize=(9, 5))
    for trial in sorted({row["trial"] for row in rows}):
        series = [row for row in rows if row["trial"] == trial]
        points = _finite_points(series, "step", "proxy_metric")
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=trial)
    baseline = baseline_record(manifest)
    if baseline and baseline.get("metric_value") is not None:
        ax.axhline(float(baseline["metric_value"]), color="black", linestyle="--", linewidth=1.2, label="pretrained baseline")
    best_points = _finite_points(rows, "step", "best_so_far")
    if best_points:
        ax.plot([x for x, _ in best_points], [y for _, y in best_points], color="crimson", linewidth=2.0, label="best so far")
    ax.set_xlabel("PBT step")
    ax.set_ylabel(metric)
    ax.set_title("Physics metric over time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = Path(run_dir) / "plots" / PLOT_NAMES["physics_metric_over_time"]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_learning_rate(run_dir, rows, events):
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    for trial in sorted({row["trial"] for row in rows}):
        series = [row for row in rows if row["trial"] == trial]
        points = _finite_points(series, "step", "LR")
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=trial)
    for event in events:
        if event.get("event_type") != "lr_change":
            continue
        step = event.get("generation")
        new_lr = event.get("new_lr")
        if step is not None and new_lr is not None:
            ax.scatter([step], [new_lr], marker="x", color="crimson", zorder=5)
    ax.set_xlabel("PBT step")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title("Learning rate over time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = Path(run_dir) / "plots" / PLOT_NAMES["learning_rate_over_time"]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_lineage(run_dir, manifest, events):
    plt = _plot_setup()
    members = sorted(manifest.get("members", {}))
    member_y = {name: index for index, name in enumerate(members)}
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(members) + 2)))
    max_generation = max([event.get("generation", 0) or 0 for event in events] + [0])
    for name, y in member_y.items():
        ax.hlines(y, 0, max_generation + 1, color="lightgray", linewidth=1)
        ax.text(-0.05, y, name, ha="right", va="center")
    exploit_events = [event for event in events if event.get("event_type") == "exploit"]
    for event in exploit_events:
        donor = event.get("donor")
        recipient = event.get("recipient")
        generation = event.get("generation", 0) or 0
        if donor not in member_y or recipient not in member_y:
            continue
        ax.annotate(
            "",
            xy=(generation + 0.85, member_y[recipient]),
            xytext=(generation + 0.15, member_y[donor]),
            arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 1.5},
        )
    if not exploit_events:
        ax.text(0.5, 0.5, "No exploit/copy events", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(-0.25, max_generation + 1.25)
    ax.set_ylim(-1, len(members))
    ax.set_yticks([])
    ax.set_xlabel("PBT step")
    ax.set_title("PBT lineage")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path = Path(run_dir) / "plots" / PLOT_NAMES["pbt_lineage"]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_best_progress(run_dir, manifest, rows):
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    points = _finite_points(rows, "step", "best_so_far")
    if points:
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", color="crimson")
    baseline = baseline_record(manifest)
    if baseline and baseline.get("metric_value") is not None:
        ax.axhline(float(baseline["metric_value"]), color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("PBT step")
    ax.set_ylabel(_metric_name(manifest))
    ax.set_title("Best model progress")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = Path(run_dir) / "plots" / PLOT_NAMES["best_model_progress"]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_final_summary(run_dir, manifest, rows):
    plt = _plot_setup()
    summary = build_summary(run_dir, manifest)
    labels = []
    values = []
    baseline = summary.get("baseline")
    if baseline and baseline.get("metric_value") is not None:
        labels.append("pretrained baseline")
        values.append(float(baseline["metric_value"]))
    final = summary.get("final_best")
    if final and final.get("proxy_metric") is not None:
        labels.append("final best")
        values.append(float(final["proxy_metric"]))
    best = summary.get("best")
    if best and best.get("metric_value") is not None:
        labels.append("global best")
        values.append(float(best["metric_value"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if values:
        colors = ["#6b7280", "#2563eb", "#dc2626"][: len(values)]
        ax.bar(labels, values, color=colors)
    else:
        ax.text(0.5, 0.5, "No completed evaluations", transform=ax.transAxes, ha="center", va="center")
    ax.set_ylabel(_metric_name(manifest))
    ax.set_title("Final summary")
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()
    path = Path(run_dir) / "plots" / PLOT_NAMES["final_summary"]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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


def write_plots(run_dir, manifest):
    ensure_run_layout(run_dir)
    rows = read_metrics_rows(run_dir)
    events = read_events(run_dir)
    return {
        "physics_metric_over_time": str(plot_physics_metric(run_dir, manifest, rows)),
        "learning_rate_over_time": str(plot_learning_rate(run_dir, rows, events)),
        "pbt_lineage": str(plot_lineage(run_dir, manifest, events)),
        "best_model_progress": str(plot_best_progress(run_dir, manifest, rows)),
        "final_summary": str(plot_final_summary(run_dir, manifest, rows)),
    }


def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_report(run_dir, manifest, summary):
    path = Path(run_dir) / REPORT_NAME
    metric = summary["metric"]
    baseline = summary.get("baseline") or {}
    configured_baseline = summary.get("configured_baseline") or {}
    best = summary.get("best") or {}
    improvement = summary.get("best_improvement_vs_baseline")
    schedule = summary.get("schedule") or {}
    eval_schedule = schedule.get("evaluation_interval") or {}
    exploit_schedule = schedule.get("exploit_interval") or {}
    lines = [
        f"# {summary.get('experiment')}",
        "",
        "## Method",
        f"- Method: `{summary.get('method')}`",
        f"- Metric: `{metric['name']}` ({metric['mode']})",
        f"- Population: {len(summary.get('population') or [])} trials",
        f"- Training interval: {schedule.get('training_interval', {}).get('epochs_per_generation', 'n/a')} epoch(s), {schedule.get('training_interval', {}).get('samples_per_trial_chunk', 'n/a')} samples/trial chunk",
        f"- Evaluation interval: every {eval_schedule.get('training_chunks', 'n/a')} training chunk(s), {eval_schedule.get('samples_per_epoch_val', 'n/a')} validation samples",
        f"- Exploit interval: {('disabled' if not exploit_schedule.get('enabled') else 'every ' + str(exploit_schedule.get('training_chunks', 'n/a')) + ' training chunk(s)')}",
        "",
        "## Inputs",
        f"- Starting checkpoint: `{(summary.get('starting_checkpoint') or {}).get('state_path') or (summary.get('starting_checkpoint') or {}).get('path')}`",
        f"- Dataset/proxy: `{summary.get('dataset')}`",
        "",
        "## Result",
        f"- Measured baseline: {_fmt(baseline.get('metric_value'))}",
        f"- Configured baseline/reference: {_fmt(configured_baseline.get('metric_value'))}",
        f"- Best: {_fmt(best.get('metric_value'))}",
        f"- Improvement vs baseline: {_fmt(None if improvement is None else 100.0 * improvement)}%",
        f"- Winning trial: `{summary.get('winning_trial')}`",
        f"- Best checkpoint: `{(summary.get('checkpoints') or {}).get('global_best_state')}`",
        "",
        "## LR Trajectory",
    ]
    for trial, values in (summary.get("lr_trajectory") or {}).items():
        rendered = ", ".join(f"{item['step']}:{_fmt(item['LR'], 3)}" for item in values)
        lines.append(f"- `{trial}`: {rendered}")
    lines.extend(["", "## Exploit History"])
    exploits = [event for event in summary.get("exploit_history", []) if event.get("event_type") == "exploit"]
    if exploits:
        for event in exploits:
            lines.append(
                "- generation {generation}: `{donor}` -> `{recipient}`, metric {donor_metric} -> {recipient_metric}, LR {old_lr} -> {new_lr}, mutation `{mutation}`".format(
                    generation=event.get("generation"),
                    donor=event.get("donor"),
                    recipient=event.get("recipient"),
                    donor_metric=_fmt(event.get("donor_metric")),
                    recipient_metric=_fmt(event.get("recipient_metric")),
                    old_lr=_fmt(event.get("old_lr"), 3),
                    new_lr=_fmt(event.get("new_lr"), 3),
                    mutation=event.get("mutation"),
                )
            )
    else:
        lines.append("- No exploit events recorded.")
    lines.extend(
        [
            "",
            "## Plots",
            "- [Physics metric over time](plots/physics_metric_over_time.png)",
            "- [Learning rate over time](plots/learning_rate_over_time.png)",
            "- [PBT lineage](plots/pbt_lineage.png)",
            "- [Best model progress](plots/best_model_progress.png)",
            "- [Final summary](plots/final_summary.png)",
            "- [Physics performance](plots/report/physics_performance.png)",
            "- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)",
            "- [B-tag mistag CSV](plots/report/btag_mistag_tables.csv)",
            "- [C-tag mistag CSV](plots/report/ctag_mistag_tables.csv)",
            "",
            "## Structured Files",
            "- [manifest.json](manifest.json)",
            "- [resolved_config.yaml](resolved_config.yaml)",
            "- [events.jsonl](events.jsonl)",
            "- [metrics.csv](metrics.csv)",
            "- [summary.json](summary.json)",
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
    plots = write_plots(run_dir, manifest)
    manifest["canonical_artifacts"] = {
        "events": str(run_dir / EVENTS_NAME),
        "metrics": str(run_dir / METRICS_NAME),
        "summary": str(run_dir / SUMMARY_NAME),
        "report": str(run_dir / REPORT_NAME),
        "plots": {**plots, **physics_outputs},
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
        "mutation": mutation,
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

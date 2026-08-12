#!/usr/bin/env python3
"""Low-level run-dir I/O primitives: atomic writes, the event log, and the
run contract (config/checkpoint/dataset metadata recorded once per run).

No dependency on any other pbt/reporting/ submodule -- this is the leaf of
the subpackage's dependency graph, so metrics_rows/plots/markdown_report can
all build on it without a cycle back to canonical.py's orchestrator.
"""

import csv
import glob
import json
import os
from pathlib import Path

import yaml

from training.runtime import data_paths, git_metadata, sha256, utc_now
from training.pbt.reporting.constants import EVENTS_NAME

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
    weaver_epochs_per_generation = int(shared["weaver_epochs_per_generation"])
    samples_per_epoch = int(shared["samples_per_epoch"])
    chunk_samples = weaver_epochs_per_generation * samples_per_epoch
    strategy = pbt.get("strategy", "exploit_mutate")
    evaluation_chunks = int(pbt.get("evaluation_interval_generations") or pbt.get("evaluation_interval") or 1)
    configured_exploit_chunks = pbt.get("exploit_interval_generations") or pbt.get("exploit_interval")
    exploit_chunks = None if strategy == "fixed_lr_grid" else int(configured_exploit_chunks or evaluation_chunks)
    return {
        "training_interval": {
            "weaver_epochs_per_generation": weaver_epochs_per_generation,
            "samples_per_epoch": samples_per_epoch,
            "samples_per_trial_chunk": chunk_samples,
        },
        "evaluation_interval": {
            "training_chunks": evaluation_chunks,
            "epochs": weaver_epochs_per_generation * evaluation_chunks,
            "samples_per_trial": chunk_samples * evaluation_chunks,
            "samples_per_epoch_val": int(shared["samples_per_epoch_val"]),
        },
        "exploit_interval": {
            "enabled": strategy != "fixed_lr_grid",
            "training_chunks": exploit_chunks,
            "epochs": None if exploit_chunks is None else weaver_epochs_per_generation * exploit_chunks,
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


def write_atomic_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    os.replace(temporary, path)
    return path


def read_events(run_dir):
    path = Path(run_dir) / EVENTS_NAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

#!/usr/bin/env python3
"""Small manifest-reading helpers reused by the report-facing plot layer,
plus the existing-physics-reports bridge into scripts/reports/ (physics
performance overview, background efficiency curves, mistag tables) -- the
one place that resolves which checkpoint role (global_best vs. best_physics)
those figures show."""

from pathlib import Path

from training.runtime import atomic_json
from training.pbt.reporting.metrics_rows import (
    _metric_mode,
    _metric_name,
    evaluation_metadata,
    fixed_working_point_uncertainties,
    fixed_working_point_values,
)


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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


def write_existing_physics_reports(run_dir, manifest):
    """Physics performance overview, background efficiency curves, and the
    fixed-tag mistag CSVs -- all for one resolved checkpoint role. Prefers
    the manifest's real global-selected checkpoint (manifest["best"]) over
    "best_physics" (a reporting-only arithmetic-mean-of-8-working-points
    selection, scripts/reports/plot_physics_performance.py::worker_for_report,
    independent of and not necessarily equal to the PBT's own selection);
    falls back to best_physics only if no global best has been recorded yet
    (e.g. an early/partial run). Records the resolved role, and whether it
    agrees with best_physics, in manifest["checkpoint_selection_for_report"]
    so report.md can state it explicitly rather than leaving "best"
    ambiguous."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    outputs = {}
    from reports.plot_background_efficiency_curves import plot_manifest as plot_background_efficiency
    from reports.plot_mistag_tables import collect_tables, write_csv
    from reports.plot_physics_performance import CHECKPOINT_ROLE_LABELS
    from reports.plot_physics_performance import plot_manifest as plot_physics_performance
    from reports.plot_physics_performance import worker_for_report

    role = "global_best"
    fallback_reason = None
    try:
        _worker, generation, member_name, _score = worker_for_report(manifest, "global_best")
    except RuntimeError as error:
        role = "best_physics"
        fallback_reason = str(error)
        _worker, generation, member_name, _score = worker_for_report(manifest, "best_physics")

    best_physics_member = best_physics_generation = None
    agrees_with_best_physics = None
    if role == "global_best":
        try:
            _bp_worker, bp_generation, bp_member_name, _bp_score = worker_for_report(manifest, "best_physics")
            best_physics_member = bp_member_name
            best_physics_generation = bp_generation.get("index")
            agrees_with_best_physics = bp_member_name == member_name and bp_generation.get("index") == generation.get("index")
        except RuntimeError:
            pass

    evaluation = evaluation_metadata(manifest)
    manifest["checkpoint_selection_for_report"] = {
        "role": role,
        "role_label": CHECKPOINT_ROLE_LABELS.get(role, role),
        "member": member_name,
        "generation": generation.get("index"),
        "selection_metric_name": _metric_name(manifest),
        "selection_metric_mode": _metric_mode(manifest),
        "best_physics_member": best_physics_member,
        "best_physics_generation": best_physics_generation,
        "agrees_with_best_physics": agrees_with_best_physics,
        "fallback_reason": fallback_reason,
        "validation_dataset": evaluation.get("validation_dataset"),
        "validation_suffix": evaluation.get("validation_suffix"),
        "validation_sample_count": evaluation.get("validation_sample_count"),
    }

    physics_path = plot_physics_performance(manifest_path, member=role)
    outputs["physics_performance"] = str(physics_path)
    manifest["physics_performance_plot"] = str(physics_path)

    curves_path = plot_background_efficiency(manifest_path, member=role)
    outputs["background_efficiency_curves"] = str(curves_path)
    manifest["background_efficiency_curves_plot"] = str(curves_path)

    for tag, efficiencies in {"c": (0.5, 0.8), "b": (0.8, 0.9)}.items():
        tables = collect_tables(
            [(manifest.get("experiment", run_dir.name), manifest_path)],
            tag=tag,
            efficiencies=efficiencies,
            member=role,
            manifests={manifest_path: manifest},
        )
        csv_path = run_dir / "plots" / "report" / f"{tag}tag_mistag_tables.csv"
        write_csv(csv_path, tables, tag)
        key = f"{tag}tag_mistag_table_csv"
        outputs[key] = str(csv_path)
        manifest[key] = str(csv_path)

    outputs["checkpoint_selection_metadata"] = manifest["checkpoint_selection_for_report"]
    return outputs

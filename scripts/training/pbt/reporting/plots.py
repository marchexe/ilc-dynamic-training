#!/usr/bin/env python3
"""matplotlib report plots (training evolution, working-point evolution,
baseline comparison, proxy diagnostics) plus the existing-physics-reports
bridge into scripts/reports/."""

import csv
import json
import math
from pathlib import Path

from training.runtime import atomic_json
from training.pbt.reporting.constants import (
    CONTROLLER_OBJECTIVE_COLUMN,
    FIXED_WORKING_POINTS,
    FLAVOR_COLORS,
    PLOT_NAMES,
    TIER_ORDER,
    WORKING_POINT_LINESTYLES,
    WORKING_POINT_MARKERS,
    WORKING_POINT_STYLE_RANK,
)
from training.pbt.reporting.io import ensure_run_layout, read_events
from training.pbt.reporting.metrics_rows import (
    _metric_mode,
    _metric_name,
    _mistag_percent,
    _tiered_round_samples_seen,
    controller_objective_mistag,
    evaluation_metadata,
    final_best_row,
    fixed_working_point_uncertainties,
    fixed_working_point_uncertainty,
    fixed_working_point_values,
    read_metrics_rows,
)
from training.pbt.reporting.statistics import _paired_tier_values, ranking_agreement, tier_correlation

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

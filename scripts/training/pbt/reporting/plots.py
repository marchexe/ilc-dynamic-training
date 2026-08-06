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
    BTAG_SCORE_COLUMN,
    BTAG_SCORE_WORKING_POINTS,
    CTAG_SCORE_COLUMN,
    CTAG_SCORE_WORKING_POINTS,
    FIXED_WORKING_POINTS,
    FLAVOR_COLORS,
    PLOT_NAMES,
    TIER_ORDER,
    TOTAL_SCORE_COLUMN,
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
    evaluation_metadata,
    final_best_row,
    fixed_working_point_uncertainties,
    fixed_working_point_uncertainty,
    fixed_working_point_values,
    read_metrics_rows,
)
from training.pbt.reporting.statistics import _paired_tier_values, ranking_agreement, tier_correlation

# Group scores are visually distinct from the 8 raw working points: total is
# the canonical PBT ranking metric (bold, black, on top), ctag/btag are its
# two components (thinner, colored, never hidden).
GROUP_SCORE_STYLE = {
    TOTAL_SCORE_COLUMN: {"color": "black", "linewidth": 2.4, "marker": "D", "zorder": 6, "label": "total_mistag_score (PBT ranking metric)"},
    CTAG_SCORE_COLUMN: {"color": "#59a14f", "linewidth": 1.3, "marker": "o", "zorder": 4, "label": "ctag_score"},
    BTAG_SCORE_COLUMN: {"color": "#4c78a8", "linewidth": 1.3, "marker": "s", "zorder": 4, "label": "btag_score"},
}

# The four fixed working points shown individually (not blended into one
# "mean of 8" number) in the top panel of training_evolution.png -- one
# background per (tag, efficiency) pair, picked with the user on 2026-08-05
# as the physically primary combination at that working point.
TRAINING_EVOLUTION_HIGHLIGHT_COLUMNS = (
    "ctag_b_mistag_percent_at_0p50",
    "ctag_d_mistag_percent_at_0p80",
    "btag_c_mistag_percent_at_0p80",
    "btag_d_mistag_percent_at_0p90",
)

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


def _rolling_mean(values, window):
    """Light centered smoothing for display only -- proxy-eval noise makes
    raw per-generation per-member lines unreadable at high generation
    counts; this does not touch any decision-making value."""
    if window <= 1 or len(values) <= 1:
        return list(values)
    half = window // 2
    return [
        sum(values[max(0, index - half):min(len(values), index + half + 1)])
        / len(values[max(0, index - half):min(len(values), index + half + 1)])
        for index in range(len(values))
    ]


def _mean_of_columns(source, columns):
    """Mean of whichever of `columns` are present and non-None in `source`
    (a flattened metrics-row dict, i.e. `fixed_working_point_values(...)` or
    a `read_metrics_rows()` row -- both use the same column names). None if
    none of them are present."""
    values = [float(value) for value in (source.get(column) for column in columns) if value is not None]
    return sum(values) / len(values) if values else None


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


def _exploit_event_samples(rows, events):
    """De-duplicated x-positions (samples_seen) of every generation that had
    at least one exploit/weight-copy event, for marking shared vertical
    reference lines across panels."""
    sample_by_generation = generation_sample_map(rows)
    generations = {
        event.get("generation")
        for event in events
        if event.get("event_type") in {"exploit", "weight_copy", "optimizer_copy"}
        and event.get("generation") is not None
    }
    return sorted({sample_by_generation[g] for g in generations if g in sample_by_generation})


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


def _mark_mean_checkpoint(ax, row, columns, label, marker, color):
    if row is None or row.get("samples_seen") is None:
        return
    value = _mean_of_columns(row, columns)
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
    baseline_values = _baseline_fixed_working_point_values(manifest)
    evaluation = evaluation_metadata(manifest)
    exploit_samples = _exploit_event_samples(rows, events)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]})
    fig.subplots_adjust(left=0.08, right=0.82, top=0.86, bottom=0.08, hspace=0.28)
    ax_metrics, ax_lr = axes

    plotted_values = []
    for column in TRAINING_EVOLUTION_HIGHLIGHT_COLUMNS:
        point = next(candidate for candidate in FIXED_WORKING_POINTS if candidate["column"] == column)
        rank = WORKING_POINT_STYLE_RANK[(point["tag"], point["efficiency"])]
        xs = [row["samples_seen"] for row in selected if row.get(column) is not None]
        ys = [row[column] for row in selected if row.get(column) is not None]
        if baseline_values and baseline_values.get(column) is not None:
            xs = [0, *xs]
            ys = [baseline_values[column], *ys]
        if not xs:
            continue
        plotted_values.extend(ys)
        ax_metrics.plot(
            xs, ys,
            marker=WORKING_POINT_MARKERS[rank], markersize=5, linestyle=WORKING_POINT_LINESTYLES[rank],
            linewidth=1.2, color=FLAVOR_COLORS[point["background"]], alpha=0.95,
            label=f"{point['tag']}-tag, {point['label']}",
        )

    mean_points = [(row["samples_seen"], _mean_of_columns(row, TRAINING_EVOLUTION_HIGHLIGHT_COLUMNS)) for row in selected]
    mean_points = [(x, y) for x, y in mean_points if y is not None]
    baseline_mean = _mean_of_columns(baseline_values, TRAINING_EVOLUTION_HIGHLIGHT_COLUMNS) if baseline_values else None
    if baseline_mean is not None:
        mean_points = [(0, baseline_mean), *mean_points]
    if mean_points:
        mean_xs, mean_ys = zip(*mean_points)
        plotted_values.extend(mean_ys)
        ax_metrics.plot(mean_xs, mean_ys, marker="D", markersize=5, linestyle="--", linewidth=2.0, color="0.25", alpha=0.9, zorder=5, label="mean of the 4 above")
    _mark_mean_checkpoint(ax_metrics, best_row, TRAINING_EVOLUTION_HIGHLIGHT_COLUMNS, "global best", "*", "black")
    _mark_mean_checkpoint(ax_metrics, final_row, TRAINING_EVOLUTION_HIGHLIGHT_COLUMNS, "final checkpoint", "s", "#8fb7dc")
    for x in exploit_samples:
        ax_metrics.axvline(x, color="0.75", linestyle=":", linewidth=0.8, zorder=1)
    _set_log_if_positive(ax_metrics, plotted_values)
    ax_metrics.set_ylabel("mistag [%]")
    ax_metrics.set_title("Fixed working-point mistag, per point (selected trial)", loc="left", fontsize=11, fontweight="bold")
    ax_metrics.grid(True, color="0.9", linewidth=0.6)
    ax_metrics.legend(frameon=False, fontsize=8.3, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    for trial in sorted({row["trial"] for row in rows}):
        series = [row for row in rows if row["trial"] == trial and row.get("LR") is not None and row.get("samples_seen") is not None]
        if not series:
            continue
        ax_lr.plot([row["samples_seen"] for row in series], [row["LR"] for row in series], marker="o", markersize=4.5, linestyle=":", linewidth=1.1, alpha=0.8, label=_compact_trial(trial))
    if best_row and best_row.get("LR") is not None:
        ax_lr.scatter([best_row["samples_seen"]], [best_row["LR"]], marker="*", s=110, color="black", zorder=6)
    if final_row and final_row.get("LR") is not None:
        ax_lr.scatter([final_row["samples_seen"]], [final_row["LR"]], marker="s", s=82, color="#8fb7dc", edgecolor="black", zorder=6)
    for x in exploit_samples:
        ax_lr.axvline(x, color="0.75", linestyle=":", linewidth=0.8, zorder=1)
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("samples seen")
    ax_lr.set_title("Learning-rate trajectories", loc="left", fontsize=11, fontweight="bold")
    ax_lr.grid(True, color="0.9", linewidth=0.6)
    ax_lr.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    fig.suptitle("PBT training evolution", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.945,
        f"{manifest.get('experiment', Path(run_dir).name)} | evaluation: {evaluation.get('evaluation_type', 'n/a')} | "
        f"PBT selection metric: {_metric_name(manifest)} ({mode})\n"
        "Top panel: 4 individual fixed working points for the per-generation winning trial, plus their mean -- "
        "none of these is the PBT selection metric above. Dotted vertical lines = exploit copy events "
        "(who copied whom: plots/report/exploit_table.csv).",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES["training_evolution"]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _draw_fixed_efficiency_panel(ax, tag, selected, best_row, baseline_values, baseline_uncertainties):
    """The one shared per-tag drawing routine behind both canonical raw
    plots (plot_ctag_fixed_efficiency_mistag / plot_btag_fixed_efficiency_mistag)
    -- consumes already-computed row/baseline dicts (fixed_working_point_values
    output), never re-derives a rejection->mistag conversion itself. Members
    iterated in FIXED_WORKING_POINTS' fixed, deterministic order, so legend
    entries are always in the same order across runs.
    """
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
            label=f"{point['score_label']} ({point['label']})",
        )
    if best_row and best_row.get("samples_seen") is not None:
        ax.axvline(best_row["samples_seen"], color="0.3", linestyle=":", linewidth=1.0, alpha=0.6)
    # Preserve the existing appropriate log scaling for raw mistag values
    # (they can span orders of magnitude); never applied to the aggregate
    # score plot below, where it would reduce interpretability of a single
    # bounded ranking curve.
    _set_log_if_positive(ax, plotted)
    ax.set_ylabel("mistag [%]")
    ax.set_xlabel("samples seen")
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0, handlelength=2.2)


def _plot_fixed_efficiency_mistag(run_dir, manifest, rows, tag, title, plot_name_key):
    plt = _plot_setup()
    mode = _metric_mode(manifest)
    selected = selected_generation_rows(rows, mode)
    best_row = _row_for_checkpoint(rows, manifest.get("best") or {})
    baseline_values = _baseline_fixed_working_point_values(manifest)
    baseline_uncertainties = _baseline_fixed_working_point_uncertainties(manifest)
    evaluation = evaluation_metadata(manifest)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
    fig.subplots_adjust(left=0.09, right=0.75, top=0.83, bottom=0.13)
    _draw_fixed_efficiency_panel(ax, tag, selected, best_row, baseline_values, baseline_uncertainties)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")

    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.90,
        f"{manifest.get('experiment', Path(run_dir).name)} | evaluation: {evaluation.get('evaluation_type', 'n/a')}\n"
        "Markers = measured checkpoints; error bars = 68% Wilson interval; lines guide the eye only; "
        "dotted vertical line = selected/global-best checkpoint.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES[plot_name_key]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_ctag_fixed_efficiency_mistag(run_dir, manifest, rows):
    return _plot_fixed_efficiency_mistag(run_dir, manifest, rows, "c", "C-tag fixed-efficiency mistag", "ctag_fixed_efficiency_mistag")


def plot_btag_fixed_efficiency_mistag(run_dir, manifest, rows):
    return _plot_fixed_efficiency_mistag(run_dir, manifest, rows, "b", "B-tag fixed-efficiency mistag", "btag_fixed_efficiency_mistag")


def plot_geometric_mistag_scores(run_dir, manifest, rows):
    """The three canonical aggregate scores together -- ctag_score,
    btag_score, and total_mistag_score (the canonical PBT ranking metric,
    drawn bold/black/on-top so it's visually identifiable without hiding
    the two component curves). Consumes only already-computed
    ctag_score/btag_score/total_mistag_score row columns (see
    metrics_rows.py::group_score_row) -- no formula is reimplemented here.
    """
    plt = _plot_setup()
    mode = _metric_mode(manifest)
    selected = selected_generation_rows(rows, mode)
    best_row = _row_for_checkpoint(rows, manifest.get("best") or {})
    evaluation = evaluation_metadata(manifest)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
    fig.subplots_adjust(left=0.09, right=0.78, top=0.83, bottom=0.13)

    plotted = []
    # Deterministic series order: total first (drawn last/on top via
    # zorder, but always listed first here and therefore first in the
    # legend), then ctag, then btag.
    for column in (TOTAL_SCORE_COLUMN, CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN):
        style = GROUP_SCORE_STYLE[column]
        xs = [row["samples_seen"] for row in selected if row.get(column) is not None]
        ys = [row[column] for row in selected if row.get(column) is not None]
        if not xs:
            continue
        plotted.extend(ys)
        ax.plot(
            xs, ys,
            marker=style["marker"], markersize=6 if column == TOTAL_SCORE_COLUMN else 5,
            linestyle="-", linewidth=style["linewidth"], color=style["color"],
            alpha=0.95, zorder=style["zorder"], label=style["label"],
        )
    if best_row and best_row.get("samples_seen") is not None:
        ax.axvline(best_row["samples_seen"], color="0.3", linestyle=":", linewidth=1.0, alpha=0.6)
    # Deliberately no log scale here: total_mistag_score is a single
    # bounded ranking curve, not a multi-order-of-magnitude raw quantity --
    # log scaling would reduce, not improve, interpretability.
    ax.set_ylabel("mistag score [%]")
    ax.set_xlabel("samples seen")
    ax.set_title("Geometric mistag scores", loc="left", fontsize=12.5, fontweight="bold")
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.6, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    fig.suptitle("Geometric mistag scores", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.90,
        f"{manifest.get('experiment', Path(run_dir).name)} | evaluation: {evaluation.get('evaluation_type', 'n/a')}\n"
        "total_mistag_score = sqrt(ctag_score * btag_score) is the canonical PBT ranking metric (bold); "
        "ctag_score/btag_score are its two components, shown for diagnosis, never used for ranking themselves "
        "unless a legacy configuration explicitly selects one of them.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES["geometric_mistag_scores"]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


_ANCHOR_DECISION_STYLE = {
    "accepted_new_anchor": {"marker": "^", "color": "#2ca02c", "label": "winner (accepted_new_anchor)"},
    "reused_previous_anchor": {"marker": "o", "color": "0.35", "label": "winner (reused_previous_anchor)"},
    "rewound_to_previous_anchor": {"marker": "v", "color": "#d62728", "label": "winner (rewound_to_previous_anchor)"},
}


def plot_pbt_decision_evolution(run_dir, manifest, rows):
    """The anchor_copy_lr_recenter "PBT decision plot": for every
    generation, every member's total_mistag_score, the winner (marker shape
    tied to the accept/reuse/rewind decision), the LR center, and every
    member's respread LR -- so the causal chain (why a member won -> what
    became the anchor -> how the center moved -> how the next population
    was assigned) is visible in one figure. Strategy-specific by nature
    (only this strategy has anchor/decision state to show), but consumes
    only already-computed manifest/row data -- generation_record
    ["anchor_copy_lr_recenter"] (written once by the planner) and `rows`'
    already-computed total_mistag_score column -- never recomputing a
    metric or a decision here. Returns None (no plot) for every other
    strategy or a run with no recorded decisions yet, the same
    "skip, don't fail" convention as plot_baseline_comparison/
    plot_proxy_diagnostics.

    x-axis is "generation", not "samples seen": the decision, LR center,
    and respread LR values are all inherently one-per-generation quantities
    (never a within-generation checkpoint quantity), so generation is the
    correct axis per this plot's own data granularity, not samples seen.
    """
    if manifest.get("config", {}).get("pbt", {}).get("strategy") != "anchor_copy_lr_recenter":
        return None
    decisions = sorted(
        (
            (generation["index"], generation["anchor_copy_lr_recenter"])
            for generation in manifest.get("generations", [])
            if (generation.get("anchor_copy_lr_recenter") or {}).get("decision") in _ANCHOR_DECISION_STYLE
        ),
        key=lambda item: item[0],
    )
    if not decisions:
        return None

    plt = _plot_setup()
    fig, (ax_score, ax_lr) = plt.subplots(
        2, 1, figsize=(11.0, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    fig.subplots_adjust(left=0.08, right=0.80, top=0.86, bottom=0.08, hspace=0.30)

    # Deterministic member order/colors: alphabetical, same convention as
    # plot_training_evolution's per-trial LR series.
    members = sorted({name for _, info in decisions for name in (info.get("assigned_lrs") or {})})
    member_colors = ("#4c78a8", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2")
    color_by_member = {name: member_colors[index % len(member_colors)] for index, name in enumerate(members)}
    row_lookup = {(row.get("generation"), row.get("trial")): row for row in rows}

    for member in members:
        xs = [g for g, _ in decisions if row_lookup.get((g, member), {}).get(TOTAL_SCORE_COLUMN) is not None]
        ys = [row_lookup[(g, member)][TOTAL_SCORE_COLUMN] for g in xs]
        if xs:
            ax_score.plot(
                xs, ys, marker="o", markersize=4.2, linestyle="-", linewidth=1.0, alpha=0.7,
                color=color_by_member[member], label=_compact_trial(member),
            )

    plotted_decision_labels = set()
    for generation, info in decisions:
        winner = info.get("winner")
        row = row_lookup.get((generation, winner))
        if row is None or row.get(TOTAL_SCORE_COLUMN) is None:
            continue
        style = _ANCHOR_DECISION_STYLE[info["decision"]]
        label = style["label"] if style["label"] not in plotted_decision_labels else None
        plotted_decision_labels.add(style["label"])
        ax_score.scatter(
            [generation], [row[TOTAL_SCORE_COLUMN]], marker=style["marker"], s=130,
            color=style["color"], edgecolor="black", linewidth=0.7, zorder=6, label=label,
        )
    ax_score.set_ylabel("total_mistag_score [%]")
    ax_score.set_title("Winner selection and anchor decision, per generation", loc="left", fontsize=11, fontweight="bold")
    ax_score.grid(True, color="0.9", linewidth=0.6)
    ax_score.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    center_xs = [generation for generation, _ in decisions]
    center_ys = [info.get("new_lr_center") for _, info in decisions]
    ax_lr.plot(center_xs, center_ys, marker="D", markersize=6, linestyle="-", linewidth=2.2, color="black", zorder=6, label="LR center")
    decision_by_generation = dict(decisions)
    for member in members:
        xs = [g for g, info in decisions if (info.get("assigned_lrs") or {}).get(member) is not None]
        ys = [decision_by_generation[g]["assigned_lrs"][member] for g in xs]
        if xs:
            ax_lr.plot(
                xs, ys, marker="o", markersize=3.8, linestyle=":", linewidth=1.0, alpha=0.7,
                color=color_by_member[member], label=_compact_trial(member),
            )
    collapsed_generations = [generation for generation, info in decisions if info.get("spread_collapsed")]
    for generation in collapsed_generations:
        ax_lr.axvline(generation, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.45, zorder=1)
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("LR")
    ax_lr.set_xlabel("generation")
    ax_lr.set_title("LR center and per-member respreading", loc="left", fontsize=11, fontweight="bold")
    ax_lr.grid(True, color="0.9", linewidth=0.6)
    ax_lr.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.03), borderaxespad=0.0)

    fig.suptitle("PBT total-score and LR evolution (anchor_copy_lr_recenter)", x=0.02, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.02,
        0.90,
        f"{manifest.get('experiment', Path(run_dir).name)}\n"
        "Top: every member's total_mistag_score; marker shape/color at the winner = this generation's anchor "
        "decision. Bottom: LR center (bold black) and every member's respread LR; dotted red vertical line = "
        "a generation where min_lr/max_lr clamping collapsed two or more members onto the same LR.",
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.35",
    )
    path = Path(run_dir) / "plots" / PLOT_NAMES["pbt_decision"]
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
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.subplots_adjust(left=0.055, right=0.98, top=0.86, bottom=0.08, hspace=0.42, wspace=0.32)
    ax_control, ax_tiers, ax_control_monitor = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_control_holdout, ax_agreement = axes[1, 0], axes[1, 1]
    axes[1, 2].axis("off")

    tier_colors = {"control": "#2f5aa0", "monitor": "#cf6f2e", "full": "#59a14f", "full_holdout": "#b07aa1"}
    exploit_xs = sorted(
        {
            _tiered_round_samples_seen(manifest, event.get("generation"))
            for event in events
            if event.get("event_type") == "exploit"
        }
    )

    # Left: control broken out per member -- control is the tier that
    # actually drives PBT decisions, so a population-mean-only view hides
    # which member is dragging it up or down. Raw per-generation proxy
    # values are noisy, so member lines are lightly smoothed for display
    # (rolling_mean does not touch any decision-making value).
    control_rounds = sorted(
        (r for r in rounds if r.get("tier") == "control"),
        key=lambda item: item.get("generation") if item.get("generation") is not None else -999,
    )
    if control_rounds:
        member_series = {}
        mean_xs, mean_ys = [], []
        for round_record in control_rounds:
            metric_name = round_record.get("metric_name")
            x = _tiered_round_samples_seen(manifest, round_record.get("generation"))
            round_values = []
            for member_name, record in sorted((round_record.get("members") or {}).items()):
                value = (record.get("metrics") or {}).get(metric_name)
                if value is None or not math.isfinite(float(value)):
                    continue
                value = float(value)
                xs_series, ys_series = member_series.setdefault(member_name, ([], []))
                xs_series.append(x)
                ys_series.append(value)
                round_values.append(value)
            if round_values:
                mean_xs.append(x)
                mean_ys.append(sum(round_values) / len(round_values))
        # Order by each member's configured start_lr (not alphabetically --
        # "lr_14e-6" < "lr_3e-6" as strings, which reads as nonsense in the
        # legend), so the legend and colors go smallest-to-largest LR.
        start_lr_by_member = {
            member["name"]: member.get("start_lr")
            for member in (manifest.get("config", {}).get("population") or [])
        }
        ordered_members = sorted(
            member_series.items(),
            key=lambda item: (start_lr_by_member.get(item[0]) is None, start_lr_by_member.get(item[0]), item[0]),
        )
        # Four clearly distinct hues -- this panel no longer shares an axes
        # with monitor/full/full_holdout (those moved to the tier-comparison
        # panel), so members no longer need to be tinted as "one blue family".
        member_colors = ("#4c78a8", "#e15759", "#59a14f", "#f28e2b")
        for index, (member_name, (xs_series, ys_series)) in enumerate(ordered_members):
            color = member_colors[index % len(member_colors)]
            ax_control.plot(xs_series, _rolling_mean(ys_series, 5), linestyle="-", linewidth=1.3, alpha=0.85, color=color, label=_compact_trial(member_name))
        if mean_ys:
            ax_control.plot(mean_xs, _rolling_mean(mean_ys, 5), linestyle="-", linewidth=2.2, color="black", zorder=5, label="mean")
    for x in exploit_xs:
        ax_control.axvline(x, color="0.8", linestyle=":", linewidth=0.8, zorder=1)
    ax_control.set_xlabel("samples seen")
    ax_control.set_ylabel("control metric value")
    ax_control.set_title("Control, per member (5-pt smoothed)", loc="left", fontsize=10.5, fontweight="bold")
    ax_control.grid(True, color="0.9", linewidth=0.6)
    ax_control.legend(frameon=False, fontsize=7.6, loc="best")

    # Middle: clean tier-level comparison (population mean + min-max band,
    # no per-member detail) -- the overview that answers "does control track
    # monitor/full", kept uncluttered by the per-member noise on the left.
    for tier in TIER_ORDER:
        tier_rounds = sorted(
            (r for r in rounds if r.get("tier") == tier),
            key=lambda item: item.get("generation") if item.get("generation") is not None else -999,
        )
        if not tier_rounds:
            continue
        xs, means, mins, maxs = [], [], [], []
        for round_record in tier_rounds:
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
        ax_tiers.plot(xs, means, linestyle="-", linewidth=1.4, color=color, label=f"{tier} (mean)")
        ax_tiers.fill_between(xs, mins, maxs, color=color, alpha=0.15, linewidth=0)
    for x in exploit_xs:
        ax_tiers.axvline(x, color="0.8", linestyle=":", linewidth=0.8, zorder=1)
    ax_tiers.set_xlabel("samples seen")
    ax_tiers.set_ylabel("metric value (population mean)")
    ax_tiers.set_title("Tier comparison (population means)", loc="left", fontsize=10.5, fontweight="bold")
    ax_tiers.grid(True, color="0.9", linewidth=0.6)
    ax_tiers.legend(frameon=False, fontsize=8, loc="best")

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
        "Left: control per member, lightly smoothed for readability. Middle: clean tier comparison (population mean + "
        "min-max band, all 4 tiers). Dotted vertical lines = exploit events. A control-only improvement is provisional, not confirmed.",
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
        "ctag_fixed_efficiency_mistag": str(plot_ctag_fixed_efficiency_mistag(run_dir, manifest, rows)),
        "btag_fixed_efficiency_mistag": str(plot_btag_fixed_efficiency_mistag(run_dir, manifest, rows)),
        "geometric_mistag_scores": str(plot_geometric_mistag_scores(run_dir, manifest, rows)),
    }
    diagnostics_path = plot_proxy_diagnostics(run_dir, manifest)
    if diagnostics_path is not None:
        plots["proxy_diagnostics"] = str(diagnostics_path)
    comparison_path = plot_baseline_comparison(run_dir, manifest)
    if comparison_path is not None:
        plots["baseline_comparison"] = str(comparison_path)
    decision_path = plot_pbt_decision_evolution(run_dir, manifest, rows)
    if decision_path is not None:
        plots["pbt_decision"] = str(decision_path)
    return plots

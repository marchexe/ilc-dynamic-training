#!/usr/bin/env python3
"""Standalone, publication-quality research figures for PBT runs.

Three layers, kept strictly separate:

    build_member_metric_rows / build_generation_decision_rows
        -- assemble already-computed manifest/CSV data into flat rows.
        Never reimplement a metric formula; never parse the manifest
        independently of the existing metrics_rows.py layer.

    validate_metric_rows
        -- drop rows missing/non-finite/negative in a required column,
        with a warning naming exactly what was wrong. Never fabricate a
        substitute value.

    plot_* functions
        -- pure rendering. Every plot_* function takes already-built,
        already-validated rows and returns {"pdf": path, "png": path,
        "warnings": [...], "generations": n, "members": n} -- never a bare
        path, so callers (report generation, tests) can inspect what was
        actually plotted without re-deriving it.

Every figure is written as both a vector PDF (papers/notes) and a raster
PNG (quick inspection), under <run_dir>/plots/research/. Deliberately not
combined into one contact-sheet or dashboard image -- each answers one
specific research question on its own (see each function's docstring).

Plain scientific style throughout: white background, restrained grid,
colorblind-safe (Okabe-Ito) markers with shape distinguishing every
semantically important state (winner/anchor/baseline/decision), so figures
remain legible printed in grayscale.
"""

import math
from pathlib import Path

from training.pbt.reporting.constants import (
    BTAG_SCORE_COLUMN,
    BTAG_SCORE_WORKING_POINTS,
    CB_PALETTE,
    CTAG_SCORE_COLUMN,
    CTAG_SCORE_WORKING_POINTS,
    DECISION_MARKER_STYLE,
    FIXED_WORKING_POINTS,
    RESEARCH_PLOT_NAMES,
    RESEARCH_PLOTS_SUBDIR,
    ROLE_MARKER_STYLE,
    TOTAL_SCORE_COLUMN,
)
from training.pbt.reporting.metrics_rows import (
    _metric_mode,
    _metric_name,
    evaluation_metadata,
    evaluation_rows,
    final_best_row,
    fixed_working_point_uncertainty,
    fixed_working_point_values,
    group_score_row,
)
from training.pbt.reporting.plots import (
    _completed_initial_evaluation_metrics,
    _global_best_metrics,
    _plot_setup,
)

RESOLVED_ANCHOR_DECISIONS = tuple(DECISION_MARKER_STYLE)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


def build_member_metric_rows(manifest):
    """One row per (generation, member): every raw working-point value,
    ctag_score/btag_score/total_mistag_score, LR, samples_seen, plus an
    is_winner flag -- based on the run's actual configured selection
    metric (optimization_metric_value/mode), never re-derived from
    total_mistag_score even when the two happen to coincide.

    Thin wrapper over evaluation_rows() (the existing authoritative
    row-preparation layer, reporting/metrics_rows.py) -- adds only the
    winner flag; every metric value already comes from there unmodified.
    """
    rows = evaluation_rows(manifest)
    mode = _metric_mode(manifest)
    by_generation = {}
    for row in rows:
        by_generation.setdefault(row["generation"], []).append(row)
    winner_by_generation = {}
    for generation, group in by_generation.items():
        best = final_best_row(group, mode)
        winner_by_generation[generation] = best["trial"] if best else None
    for row in rows:
        row["is_winner"] = row["trial"] == winner_by_generation.get(row["generation"])
    return rows


def build_generation_decision_rows(manifest, member_rows):
    """One row per generation with a resolved anchor_copy_lr_recenter
    decision. Empty for any other strategy, or a run with no decisions
    recorded yet -- callers must treat that as "nothing to plot", not an
    error.

    `anchor_row` is the *resulting* (post-decision) active-anchor's full
    metric row -- carried forward unchanged across reused_previous_anchor/
    rewound_to_previous_anchor generations, refreshed only on
    accepted_new_anchor -- so plotting its trajectory across generations
    shows exactly which checkpoint state is live at each point in time.
    `anchor_total_score_before_decision` is the *previous* anchor's own
    total_mistag_score, captured before this generation's outcome is
    applied -- what the winner was actually compared against -- None for
    the first-ever decision (no anchor existed yet, so it is unconditionally
    accepted, by definition).
    """
    if manifest.get("config", {}).get("pbt", {}).get("strategy") != "anchor_copy_lr_recenter":
        return []
    row_lookup = {(row["generation"], row["trial"]): row for row in member_rows}
    decisions = []
    anchor_row = None
    anchor_member = None
    for generation in sorted(manifest.get("generations", []), key=lambda item: item.get("index", 0)):
        info = generation.get("anchor_copy_lr_recenter")
        if not info or info.get("decision") not in RESOLVED_ANCHOR_DECISIONS:
            continue
        anchor_score_before = anchor_row.get(TOTAL_SCORE_COLUMN) if anchor_row else None
        winner_row = row_lookup.get((generation["index"], info.get("winner")))
        if info["decision"] == "accepted_new_anchor":
            anchor_row = winner_row
            anchor_member = info.get("winner")
        decisions.append(
            {
                "generation": generation["index"],
                "winner": info.get("winner"),
                "winner_row": winner_row,
                "decision": info["decision"],
                "anchor_member": anchor_member,
                "anchor_row": anchor_row,
                "anchor_total_score_before_decision": anchor_score_before,
                "winner_lr": info.get("winner_lr"),
                "previous_lr_center": info.get("previous_lr_center"),
                "new_lr_center": info.get("new_lr_center"),
                "assigned_lrs": info.get("assigned_lrs") or {},
                "unclamped_lrs": info.get("unclamped_lrs") or {},
                "spread_collapsed": bool(info.get("spread_collapsed")),
                "duplicate_lr_groups": info.get("duplicate_lr_groups") or [],
            }
        )
    return decisions


def validate_metric_rows(rows, required_columns):
    """(valid_rows, warnings): rows missing, non-finite, or negative in any
    of `required_columns` are excluded from valid_rows -- a valid zero is
    kept, never treated as missing. Each excluded row produces one warning
    naming the member, generation, and exactly which column(s) were
    invalid, never a silently dropped point."""
    valid = []
    warnings = []
    for row in rows:
        problems = []
        for column in required_columns:
            value = row.get(column)
            if value is None:
                problems.append(f"{column}=missing")
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                problems.append(f"{column}=invalid")
                continue
            if not math.isfinite(value):
                problems.append(f"{column}=non-finite")
            elif value < 0:
                problems.append(f"{column}=negative")
        if problems:
            warnings.append(
                f"member={row.get('trial', row.get('member', 'n/a'))} "
                f"generation={row.get('generation')}: {', '.join(problems)}"
            )
        else:
            valid.append(row)
    return valid, warnings


def _baseline_row(manifest):
    """The pretrained/fixed-LR baseline's raw+aggregate metrics in the same
    shape as a member_metric_row -- (row, missing_labels), row is None if
    no measured baseline evaluation exists (never a fabricated baseline)."""
    metrics = _completed_initial_evaluation_metrics(manifest)
    if metrics is None:
        return None, []
    ctag_score, btag_score, total_score, missing = group_score_row(metrics)
    row = {
        "trial": "baseline",
        "generation": None,
        "samples_seen": 0,
        "LR": None,
        CTAG_SCORE_COLUMN: ctag_score,
        BTAG_SCORE_COLUMN: btag_score,
        TOTAL_SCORE_COLUMN: total_score,
        **fixed_working_point_values(metrics),
    }
    return row, missing


def _final_selected_row(manifest):
    """The global-best (final selected) checkpoint's raw+aggregate metrics
    -- (row, missing_labels), row is None if no global best has been
    recorded yet."""
    metrics = _global_best_metrics(manifest)
    if metrics is None:
        return None, []
    best = manifest.get("best") or {}
    ctag_score, btag_score, total_score, missing = group_score_row(metrics)
    row = {
        "trial": best.get("member", "final"),
        "generation": best.get("generation"),
        "samples_seen": None,
        "LR": best.get("lr"),
        CTAG_SCORE_COLUMN: ctag_score,
        BTAG_SCORE_COLUMN: btag_score,
        TOTAL_SCORE_COLUMN: total_score,
        **fixed_working_point_values(metrics),
    }
    return row, missing


def _run_caption(manifest):
    evaluation = evaluation_metadata(manifest)
    return (
        f"run: {manifest.get('experiment', 'n/a')}  |  "
        f"validation: {evaluation.get('validation_dataset', 'n/a')} ({evaluation.get('validation_suffix', 'n/a')})  |  "
        f"selection metric: {_metric_name(manifest)} ({_metric_mode(manifest)})"
    )


def _save_both(fig, run_dir, plot_name_key):
    base = RESEARCH_PLOT_NAMES[plot_name_key]
    directory = Path(run_dir) / "plots" / RESEARCH_PLOTS_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / f"{base}.pdf"
    png_path = directory / f"{base}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def _dedup_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen[label] = handle
    if seen:
        ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=7.6, **kwargs)


# ---------------------------------------------------------------------------
# 1/2. Raw working-point evolution (c-tag / b-tag)
# ---------------------------------------------------------------------------


def _draw_working_point_panel(ax, point, member_rows, decision_rows, baseline_row):
    column = point["column"]
    err_low_key = f"{column}_err_low"
    err_high_key = f"{column}_err_high"
    valid_rows, warnings = validate_metric_rows(member_rows, [column])
    winner_rows = [row for row in valid_rows if row.get("is_winner")]
    other_rows = [row for row in valid_rows if not row.get("is_winner")]

    def _errorbar(rows, style, **kwargs):
        if not rows:
            return
        xs = [row["samples_seen"] for row in rows]
        ys = [row[column] for row in rows]
        has_uncertainty = all(row.get(err_low_key) is not None for row in rows)
        yerr = (
            [[row.get(err_low_key) or 0.0 for row in rows], [row.get(err_high_key) or 0.0 for row in rows]]
            if has_uncertainty
            else None
        )
        ax.errorbar(
            xs, ys, yerr=yerr, fmt=style["marker"], color=style["color"],
            elinewidth=0.7, capsize=1.8, linestyle="none", **kwargs,
        )

    _errorbar(other_rows, ROLE_MARKER_STYLE["member"], markersize=3.2, alpha=0.35, zorder=2, label=None)
    _errorbar(winner_rows, ROLE_MARKER_STYLE["winner"], markersize=7.5, zorder=5, label=ROLE_MARKER_STYLE["winner"]["label"])

    if decision_rows:
        anchor_xs, anchor_ys = [], []
        for decision in decision_rows:
            anchor_row = decision.get("anchor_row")
            if anchor_row is None or anchor_row.get(column) is None:
                continue
            if anchor_row.get("samples_seen") is None:
                continue
            anchor_xs.append(anchor_row["samples_seen"])
            anchor_ys.append(anchor_row[column])
        if anchor_xs:
            ax.plot(
                anchor_xs, anchor_ys, marker=ROLE_MARKER_STYLE["anchor"]["marker"],
                color=ROLE_MARKER_STYLE["anchor"]["color"], markersize=5.5, linewidth=1.1,
                linestyle="-", zorder=4, label=ROLE_MARKER_STYLE["anchor"]["label"],
            )

    if baseline_row is not None and baseline_row.get(column) is not None:
        ax.axhline(
            baseline_row[column], color=ROLE_MARKER_STYLE["baseline"]["color"], linestyle="--",
            linewidth=1.0, zorder=1, label=ROLE_MARKER_STYLE["baseline"]["label"],
        )

    ax.set_title(point["score_label"], fontsize=10.5, fontweight="bold", loc="left")
    ax.set_xlabel("Samples seen")
    ax.set_ylabel("Mistag [%]")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return warnings, len({row["trial"] for row in valid_rows}), len({row["generation"] for row in valid_rows})


def _plot_working_point_group(run_dir, manifest, member_rows, decision_rows, working_points, title, plot_name_key):
    plt = _plot_setup()
    baseline_row, baseline_missing = _baseline_row(manifest)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), sharex=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.10, hspace=0.32, wspace=0.28)

    warnings = list(f"baseline: {label} unavailable" for label in baseline_missing)
    members_plotted, generations_plotted = set(), set()
    for ax, point in zip(axes.flat, working_points):
        panel_warnings, n_members, n_generations = _draw_working_point_panel(ax, point, member_rows, decision_rows, baseline_row)
        warnings.extend(panel_warnings)
        members_plotted.add(n_members)
        generations_plotted.add(n_generations)

    _dedup_legend(axes.flat[1], loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.005, _run_caption(manifest), ha="left", va="bottom", fontsize=7.4, color="0.3")

    paths = _save_both(fig, run_dir, plot_name_key)
    plt.close(fig)
    return {
        **paths,
        "warnings": warnings,
        "generations": max(generations_plotted, default=0),
        "members": max(members_plotted, default=0),
        "metric_keys": [point["column"] for point in working_points],
    }


def plot_ctag_working_points(run_dir, manifest, member_rows, decision_rows):
    """Research question: which c-tag background-rejection components
    improve or degrade during training? 2x2 panels: cb@0.5, cd@0.5, cb@0.8,
    cd@0.8 -- never combined into one shared axis."""
    return _plot_working_point_group(
        run_dir, manifest, member_rows, decision_rows,
        CTAG_SCORE_WORKING_POINTS, "C-tag raw working-point evolution", "ctag_working_points",
    )


def plot_btag_working_points(run_dir, manifest, member_rows, decision_rows):
    """Research question: which b-tag background-rejection components
    improve or degrade during training? 2x2 panels: bc@0.8, bd@0.8, bc@0.9,
    bd@0.9."""
    return _plot_working_point_group(
        run_dir, manifest, member_rows, decision_rows,
        BTAG_SCORE_WORKING_POINTS, "B-tag raw working-point evolution", "btag_working_points",
    )


# ---------------------------------------------------------------------------
# 3. Aggregate score evolution
# ---------------------------------------------------------------------------


def plot_aggregate_scores(run_dir, manifest, member_rows, decision_rows):
    """Research question: does total performance improve, and how do
    b-tag and c-tag each contribute? Three vertically stacked panels
    (ctag_score / btag_score / total_mistag_score) sharing one x-axis, so
    a change in the combined score can be attributed to one or both
    components at a glance."""
    plt = _plot_setup()
    baseline_row, baseline_missing = _baseline_row(manifest)
    columns = (CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN, TOTAL_SCORE_COLUMN)
    titles = ("C-tag score", "B-tag score", "Total geometric mistag score")

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.6), sharex=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.78, top=0.92, bottom=0.07, hspace=0.30)

    warnings = list(f"baseline: {label} unavailable" for label in baseline_missing)
    members_plotted, generations_plotted = set(), set()
    decision_by_generation = {decision["generation"]: decision for decision in decision_rows}

    for ax, column, title in zip(axes, columns, titles):
        valid_rows, panel_warnings = validate_metric_rows(member_rows, [column])
        warnings.extend(panel_warnings)
        winner_rows = [row for row in valid_rows if row.get("is_winner")]
        other_rows = [row for row in valid_rows if not row.get("is_winner")]
        members_plotted.add(len({row["trial"] for row in valid_rows}))
        generations_plotted.add(len({row["generation"] for row in valid_rows}))

        if other_rows:
            ax.plot(
                [row["samples_seen"] for row in other_rows], [row[column] for row in other_rows],
                marker=ROLE_MARKER_STYLE["member"]["marker"], color=ROLE_MARKER_STYLE["member"]["color"],
                linestyle="none", markersize=4, alpha=0.35, zorder=2,
            )

        if decision_rows:
            # Winners marked by their generation's resolved decision (marker
            # shape, not just color) instead of a single undifferentiated
            # "winner" marker -- this panel is the one place accept/reuse/
            # rewind is shown directly against the score that drove it.
            for decision in decision_rows:
                winner_row = decision.get("winner_row")
                if winner_row is None or winner_row.get(column) is None:
                    continue
                style = DECISION_MARKER_STYLE[decision["decision"]]
                ax.scatter(
                    [winner_row["samples_seen"]], [winner_row[column]], marker=style["marker"],
                    color=style["color"], s=70, edgecolor="black", linewidth=0.5, zorder=6, label=style["label"],
                )
            anchor_xs = [decision["anchor_row"]["samples_seen"] for decision in decision_rows if decision.get("anchor_row") and decision["anchor_row"].get(column) is not None]
            anchor_ys = [decision["anchor_row"][column] for decision in decision_rows if decision.get("anchor_row") and decision["anchor_row"].get(column) is not None]
            if anchor_xs:
                ax.plot(
                    anchor_xs, anchor_ys, marker=ROLE_MARKER_STYLE["anchor"]["marker"], color=ROLE_MARKER_STYLE["anchor"]["color"],
                    markersize=5, linewidth=1.1, linestyle="-", zorder=4, label=ROLE_MARKER_STYLE["anchor"]["label"],
                )
        elif winner_rows:
            ax.scatter(
                [row["samples_seen"] for row in winner_rows], [row[column] for row in winner_rows],
                marker=ROLE_MARKER_STYLE["winner"]["marker"], color=ROLE_MARKER_STYLE["winner"]["color"],
                s=70, zorder=5, label=ROLE_MARKER_STYLE["winner"]["label"],
            )

        if baseline_row is not None and baseline_row.get(column) is not None:
            ax.axhline(baseline_row[column], color=ROLE_MARKER_STYLE["baseline"]["color"], linestyle="--", linewidth=1.0, zorder=1, label=ROLE_MARKER_STYLE["baseline"]["label"])

        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.set_ylabel("Mistag score [%]\n(lower is better)")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        _dedup_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    axes[-1].set_xlabel("Samples seen")

    fig.suptitle("Geometric mistag score evolution", x=0.02, y=0.975, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.005, _run_caption(manifest) + "  |  total_mistag_score = sqrt(ctag_score * btag_score)", ha="left", va="bottom", fontsize=7.4, color="0.3")

    paths = _save_both(fig, run_dir, "aggregate_scores")
    plt.close(fig)
    return {
        **paths, "warnings": warnings,
        "generations": max(generations_plotted, default=0), "members": max(members_plotted, default=0),
        "metric_keys": list(columns),
    }


# ---------------------------------------------------------------------------
# 4. C-tag vs b-tag trade-off
# ---------------------------------------------------------------------------


def plot_tag_tradeoff(run_dir, manifest, member_rows, decision_rows):
    """Research question: is the total-score improvement balanced, or does
    one tagger improve at the expense of the other? Scatter of
    (ctag_score, btag_score), one point per (generation, member); position
    relative to constant-total isolines answers the question directly."""
    plt = _plot_setup()
    valid_rows, warnings = validate_metric_rows(member_rows, [CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN])
    baseline_row, baseline_missing = _baseline_row(manifest)
    final_row, final_missing = _final_selected_row(manifest)
    warnings.extend(f"baseline: {label} unavailable" for label in baseline_missing)
    warnings.extend(f"final selected: {label} unavailable" for label in final_missing)

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.11)

    generations = sorted({row["generation"] for row in valid_rows})
    if generations and len(generations) > 1:
        colormap = plt.get_cmap("viridis")
        color_by_generation = {
            generation: colormap(index / (len(generations) - 1)) for index, generation in enumerate(generations)
        }
    else:
        color_by_generation = {generation: CB_PALETTE["blue"] for generation in generations}

    non_winner = [row for row in valid_rows if not row.get("is_winner")]
    winners = [row for row in valid_rows if row.get("is_winner")]
    if non_winner:
        ax.scatter(
            [row[CTAG_SCORE_COLUMN] for row in non_winner], [row[BTAG_SCORE_COLUMN] for row in non_winner],
            c=[color_by_generation[row["generation"]] for row in non_winner], marker=ROLE_MARKER_STYLE["member"]["marker"],
            s=26, alpha=0.55, zorder=2,
        )
    if winners:
        ax.scatter(
            [row[CTAG_SCORE_COLUMN] for row in winners], [row[BTAG_SCORE_COLUMN] for row in winners],
            c=[color_by_generation[row["generation"]] for row in winners], marker=ROLE_MARKER_STYLE["winner"]["marker"],
            s=110, edgecolor="black", linewidth=0.6, zorder=5, label=ROLE_MARKER_STYLE["winner"]["label"],
        )

    if decision_rows:
        anchor_points = [
            (decision["anchor_row"][CTAG_SCORE_COLUMN], decision["anchor_row"][BTAG_SCORE_COLUMN])
            for decision in decision_rows
            if decision.get("anchor_row") and decision["anchor_row"].get(CTAG_SCORE_COLUMN) is not None and decision["anchor_row"].get(BTAG_SCORE_COLUMN) is not None
        ]
        if anchor_points:
            ax.plot(
                [point[0] for point in anchor_points], [point[1] for point in anchor_points],
                marker=ROLE_MARKER_STYLE["anchor"]["marker"], color=ROLE_MARKER_STYLE["anchor"]["color"],
                markersize=7, linewidth=1.0, linestyle="-", zorder=4, label=ROLE_MARKER_STYLE["anchor"]["label"],
            )

    if baseline_row is not None and baseline_row.get(CTAG_SCORE_COLUMN) is not None and baseline_row.get(BTAG_SCORE_COLUMN) is not None:
        ax.scatter(
            [baseline_row[CTAG_SCORE_COLUMN]], [baseline_row[BTAG_SCORE_COLUMN]], marker=ROLE_MARKER_STYLE["baseline"]["marker"],
            color=ROLE_MARKER_STYLE["baseline"]["color"], s=140, zorder=6, label=ROLE_MARKER_STYLE["baseline"]["label"],
        )
    if final_row is not None and final_row.get(CTAG_SCORE_COLUMN) is not None and final_row.get(BTAG_SCORE_COLUMN) is not None:
        ax.scatter(
            [final_row[CTAG_SCORE_COLUMN]], [final_row[BTAG_SCORE_COLUMN]], marker=ROLE_MARKER_STYLE["final"]["marker"],
            color=ROLE_MARKER_STYLE["final"]["color"], s=110, edgecolor="black", linewidth=0.6, zorder=6, label=ROLE_MARKER_STYLE["final"]["label"],
        )

    # Constant-total isolines: total_score = sqrt(ctag*btag) => btag = total^2 / ctag.
    # Drawn only if the axis span is well-behaved (a handful of clean,
    # readable curves) -- skipped rather than crammed in if it wouldn't be.
    all_ctag = [row[CTAG_SCORE_COLUMN] for row in valid_rows if row[CTAG_SCORE_COLUMN] > 0]
    all_btag = [row[BTAG_SCORE_COLUMN] for row in valid_rows if row[BTAG_SCORE_COLUMN] > 0]
    if all_ctag and all_btag:
        totals = sorted({row[TOTAL_SCORE_COLUMN] for row in valid_rows if row.get(TOTAL_SCORE_COLUMN)})
        if totals:
            levels = sorted({totals[0], totals[len(totals) // 2], totals[-1]})
            ctag_min, ctag_max = min(all_ctag) * 0.85, max(all_ctag) * 1.15
            xs = [ctag_min + (ctag_max - ctag_min) * index / 200 for index in range(201)]
            for level in levels:
                ys = [level * level / x if x > 0 else float("nan") for x in xs]
                ax.plot(xs, ys, color="0.75", linestyle=":", linewidth=0.8, zorder=1)
                ax.text(xs[-1], ys[-1], f"  total={level:.3g}", fontsize=6.5, color="0.5", va="center")

    ax.set_xlabel("C-tag score [%] (lower is better)")
    ax.set_ylabel("B-tag score [%] (lower is better)")
    ax.set_title("C-tag vs. B-tag trade-off", fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _dedup_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.text(0.02, 0.01, _run_caption(manifest) + "  |  point color = generation (viridis)  |  dotted = constant total_mistag_score", ha="left", va="bottom", fontsize=7.2, color="0.3")

    paths = _save_both(fig, run_dir, "tag_tradeoff")
    plt.close(fig)
    return {
        **paths, "warnings": warnings,
        "generations": len(generations), "members": len({row["trial"] for row in valid_rows}),
        "metric_keys": [CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN],
    }


# ---------------------------------------------------------------------------
# 5. Total score vs. learning rate
# ---------------------------------------------------------------------------


def plot_score_vs_lr(run_dir, manifest, member_rows, decision_rows, max_generations=12):
    """Research question: what local LR region produces the best validation
    score, and does that optimum move during training? Small multiples --
    one panel per generation (or a representative, evenly-spaced subset
    when there are more than `max_generations`) -- points within a
    generation connected in LR order since that ordering is well-defined;
    never connected across generations, which would imply a false
    continuous function."""
    plt = _plot_setup()
    valid_rows, warnings = validate_metric_rows(member_rows, [TOTAL_SCORE_COLUMN, "LR"])
    by_generation = {}
    for row in valid_rows:
        by_generation.setdefault(row["generation"], []).append(row)
    all_generations = sorted(by_generation)
    if len(all_generations) > max_generations:
        step = max(1, len(all_generations) // max_generations)
        shown_generations = all_generations[::step][:max_generations]
        omitted = len(all_generations) - len(shown_generations)
        warnings.append(f"{omitted} generation(s) omitted from the small-multiples layout (showing {len(shown_generations)} of {len(all_generations)}, evenly spaced)")
    else:
        shown_generations = all_generations

    decision_by_generation = {decision["generation"]: decision for decision in decision_rows}
    n = max(1, len(shown_generations))
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols) if n else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.7 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.11, hspace=0.45, wspace=0.35)

    flat_axes = axes.flat
    for index, generation in enumerate(shown_generations):
        ax = flat_axes[index]
        rows = sorted(by_generation[generation], key=lambda row: row["LR"])
        ax.plot([row["LR"] for row in rows], [row[TOTAL_SCORE_COLUMN] for row in rows], color=CB_PALETTE["grey"], linewidth=0.9, alpha=0.6, zorder=2)
        ax.scatter([row["LR"] for row in rows], [row[TOTAL_SCORE_COLUMN] for row in rows], marker=ROLE_MARKER_STYLE["member"]["marker"], s=26, color=ROLE_MARKER_STYLE["member"]["color"], zorder=3)
        winner_rows = [row for row in rows if row.get("is_winner")]
        if winner_rows:
            ax.scatter([row["LR"] for row in winner_rows], [row[TOTAL_SCORE_COLUMN] for row in winner_rows], marker=ROLE_MARKER_STYLE["winner"]["marker"], s=90, color=ROLE_MARKER_STYLE["winner"]["color"], zorder=5)
        decision = decision_by_generation.get(generation)
        if decision and decision.get("new_lr_center") is not None:
            ax.axvline(decision["new_lr_center"], color=ROLE_MARKER_STYLE["anchor"]["color"], linestyle=":", linewidth=1.0, zorder=1)
        ax.set_xscale("log")
        ax.set_title(f"gen {generation}", fontsize=8.5, loc="left")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7)
    for index in range(len(shown_generations), n_rows * n_cols):
        flat_axes[index].axis("off")

    fig.supxlabel("Learning rate", fontsize=9)
    fig.supylabel("Total mistag score [%]", fontsize=9)
    fig.suptitle("Total score vs. learning rate, per generation", x=0.02, y=0.975, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.02, 0.005,
        _run_caption(manifest) + "  |  • member, ★ winner, dotted vertical = LR center. Lines connect one generation's own members only.",
        ha="left", va="bottom", fontsize=7.2, color="0.3",
    )

    paths = _save_both(fig, run_dir, "score_vs_lr")
    plt.close(fig)
    return {
        **paths, "warnings": warnings,
        "generations": len(shown_generations), "members": len({row["trial"] for row in valid_rows}),
        "metric_keys": [TOTAL_SCORE_COLUMN, "LR"],
    }


# ---------------------------------------------------------------------------
# 6. Learning-rate population evolution
# ---------------------------------------------------------------------------


def plot_lr_population(run_dir, manifest, member_rows, decision_rows):
    """Research question: how does PBT move and explore the LR region?
    Returns None (nothing to plot) for any strategy other than
    anchor_copy_lr_recenter, or a run with no decisions recorded yet --
    there is no shared LR center to show otherwise."""
    if not decision_rows:
        return None
    plt = _plot_setup()
    pbt_config = manifest.get("config", {}).get("pbt", {})
    min_lr, max_lr = pbt_config.get("min_lr"), pbt_config.get("max_lr")

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.4))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.09, right=0.78, top=0.87, bottom=0.13)

    generations = [decision["generation"] for decision in decision_rows]
    envelope_lo, envelope_hi = [], []
    for decision in decision_rows:
        assigned = list((decision.get("assigned_lrs") or {}).values())
        gen = decision["generation"]
        if assigned:
            ax.scatter([gen] * len(assigned), assigned, marker=ROLE_MARKER_STYLE["member"]["marker"], s=22, color=ROLE_MARKER_STYLE["member"]["color"], alpha=0.6, zorder=3)
            envelope_lo.append(min(assigned))
            envelope_hi.append(max(assigned))
        else:
            envelope_lo.append(None)
            envelope_hi.append(None)
    valid_envelope = [(g, lo, hi) for g, lo, hi in zip(generations, envelope_lo, envelope_hi) if lo is not None]
    if valid_envelope:
        ax.fill_between(
            [item[0] for item in valid_envelope], [item[1] for item in valid_envelope], [item[2] for item in valid_envelope],
            color=CB_PALETTE["sky_blue"], alpha=0.15, zorder=1, label="min-max population envelope",
        )

    center_xs = [decision["generation"] for decision in decision_rows if decision.get("new_lr_center") is not None]
    center_ys = [decision["new_lr_center"] for decision in decision_rows if decision.get("new_lr_center") is not None]
    ax.plot(center_xs, center_ys, color=CB_PALETTE["black"], linewidth=2.0, zorder=5, label="LR center")
    for decision in decision_rows:
        if decision.get("new_lr_center") is None:
            continue
        style = DECISION_MARKER_STYLE[decision["decision"]]
        ax.scatter([decision["generation"]], [decision["new_lr_center"]], marker=style["marker"], color=style["color"], s=70, edgecolor="black", linewidth=0.5, zorder=6, label=style["label"])

    winner_xs = [decision["generation"] for decision in decision_rows if decision.get("winner_lr") is not None]
    winner_ys = [decision["winner_lr"] for decision in decision_rows if decision.get("winner_lr") is not None]
    if winner_xs:
        ax.scatter(winner_xs, winner_ys, marker=ROLE_MARKER_STYLE["winner"]["marker"], color=ROLE_MARKER_STYLE["winner"]["color"], s=90, zorder=6, label=ROLE_MARKER_STYLE["winner"]["label"])

    if min_lr is not None:
        ax.axhline(min_lr, color=CB_PALETTE["vermillion"], linestyle="--", linewidth=0.9, zorder=1, label="min_lr / max_lr")
    if max_lr is not None:
        ax.axhline(max_lr, color=CB_PALETTE["vermillion"], linestyle="--", linewidth=0.9, zorder=1)

    collapsed = [decision["generation"] for decision in decision_rows if decision.get("spread_collapsed")]
    for generation in collapsed:
        ax.axvline(generation, color=CB_PALETTE["orange"], linestyle=":", linewidth=1.1, alpha=0.6, zorder=2)
    if collapsed:
        ax.plot([], [], color=CB_PALETTE["orange"], linestyle=":", linewidth=1.1, label="spread_collapsed")

    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate population evolution", fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _dedup_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.text(0.02, 0.01, _run_caption(manifest), ha="left", va="bottom", fontsize=7.4, color="0.3")

    paths = _save_both(fig, run_dir, "lr_population")
    plt.close(fig)
    return {
        **paths, "warnings": [],
        "generations": len(decision_rows), "members": len({name for decision in decision_rows for name in (decision.get("assigned_lrs") or {})}),
        "metric_keys": ["assigned_lrs", "new_lr_center", "min_lr", "max_lr"],
    }


# ---------------------------------------------------------------------------
# 7. Baseline vs. final selected model
# ---------------------------------------------------------------------------


def plot_baseline_ratio(run_dir, manifest):
    """Research question: which physical performance components improved
    relative to baseline? final/baseline ratio per metric (ratio < 1 =
    improvement, lower-is-better metrics) instead of a grouped bar plot,
    which would be misleading across values spanning orders of magnitude."""
    plt = _plot_setup()
    baseline_row, baseline_missing = _baseline_row(manifest)
    final_row, final_missing = _final_selected_row(manifest)
    warnings = [f"baseline: {label} unavailable" for label in baseline_missing]
    warnings.extend(f"final selected: {label} unavailable" for label in final_missing)
    if baseline_row is None or final_row is None:
        warnings.append("baseline and/or final selected model unavailable -- ratio plot not produced")
        return {"pdf": None, "png": None, "warnings": warnings, "generations": 0, "members": 0, "metric_keys": []}

    raw_points = list(FIXED_WORKING_POINTS)
    labels, ratios, omitted = [], [], []
    for point in raw_points:
        column = point["column"]
        base_value, final_value = baseline_row.get(column), final_row.get(column)
        if base_value is None or final_value is None or base_value <= 0:
            omitted.append(point["score_label"])
            continue
        labels.append(point["score_label"])
        ratios.append(final_value / base_value)
    if omitted:
        warnings.append(f"ratio omitted for: {', '.join(omitted)} (baseline value missing/non-positive)")

    group_labels, group_ratios = [], []
    for column, label in ((CTAG_SCORE_COLUMN, "ctag_score"), (BTAG_SCORE_COLUMN, "btag_score"), (TOTAL_SCORE_COLUMN, "total_mistag_score")):
        base_value, final_value = baseline_row.get(column), final_row.get(column)
        if base_value is None or final_value is None or base_value <= 0:
            warnings.append(f"ratio omitted for: {label} (baseline value missing/non-positive)")
            continue
        group_labels.append(label)
        group_ratios.append(final_value / base_value)

    fig, (ax_raw, ax_group) = plt.subplots(1, 2, figsize=(11.0, 5.0), gridspec_kw={"width_ratios": [2.2, 1.0]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.16, wspace=0.35)

    for ax, x_labels, y_values, title in ((ax_raw, labels, ratios, "Raw working points"), (ax_group, group_labels, group_ratios, "Aggregate scores")):
        positions = list(range(len(x_labels)))
        colors = [CB_PALETTE["green"] if value < 1 else (CB_PALETTE["grey"] if value == 1 else CB_PALETTE["vermillion"]) for value in y_values]
        ax.scatter(positions, y_values, marker="D", s=70, color=colors, zorder=3)
        for position, value in zip(positions, y_values):
            ax.plot([position, position], [1.0, value], color="0.7", linewidth=1.0, zorder=1)
        ax.axhline(1.0, color=CB_PALETTE["black"], linewidth=1.0, zorder=2, label="unchanged")
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("final / baseline ratio")
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle("Final selected model vs. baseline", x=0.02, y=0.975, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.02, _run_caption(manifest) + "  |  ratio < 1 = improvement (lower-is-better metrics), > 1 = degradation", ha="left", va="bottom", fontsize=7.4, color="0.3")

    paths = _save_both(fig, run_dir, "baseline_ratio")
    plt.close(fig)
    return {
        **paths, "warnings": warnings, "generations": 1, "members": 1,
        "metric_keys": [point["column"] for point in raw_points] + [CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN, TOTAL_SCORE_COLUMN],
    }


# ---------------------------------------------------------------------------
# 8. PBT decision history
# ---------------------------------------------------------------------------


def plot_decision_history(run_dir, manifest, decision_rows):
    """Research question: what decisions did the algorithm make and on
    what measured basis? Two panels: winner vs. pre-decision anchor total
    score (top), previous/new LR center (bottom) -- decisions annotated by
    marker shape, not decorative dashboard styling. Returns None for any
    non-anchor_copy_lr_recenter run or a run with no decisions recorded."""
    if not decision_rows:
        return None
    plt = _plot_setup()
    fig, (ax_score, ax_lr) = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.09, hspace=0.28)

    generations = [decision["generation"] for decision in decision_rows]
    winner_scores = [
        decision["winner_row"][TOTAL_SCORE_COLUMN] if decision.get("winner_row") else None for decision in decision_rows
    ]
    anchor_before = [decision.get("anchor_total_score_before_decision") for decision in decision_rows]

    ax_score.plot(
        [g for g, v in zip(generations, anchor_before) if v is not None], [v for v in anchor_before if v is not None],
        color=ROLE_MARKER_STYLE["anchor"]["color"], linestyle="--", linewidth=1.2, marker="D", markersize=5,
        zorder=4, label="anchor total score (before this generation's decision)",
    )
    ax_score.plot(
        [g for g, v in zip(generations, winner_scores) if v is not None], [v for v in winner_scores if v is not None],
        color=CB_PALETTE["grey"], linestyle=":", linewidth=0.9, zorder=2,
    )
    for decision, score in zip(decision_rows, winner_scores):
        if score is None:
            continue
        style = DECISION_MARKER_STYLE[decision["decision"]]
        ax_score.scatter([decision["generation"]], [score], marker=style["marker"], color=style["color"], s=90, edgecolor="black", linewidth=0.5, zorder=6, label=style["label"])
    ax_score.set_ylabel("Total mistag score [%]")
    ax_score.set_title("Winner vs. anchor total score, and the resulting decision", fontsize=10.5, fontweight="bold", loc="left")
    ax_score.grid(True, alpha=0.3, linewidth=0.5)
    _dedup_legend(ax_score, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    prev_center = [decision.get("previous_lr_center") for decision in decision_rows]
    new_center = [decision.get("new_lr_center") for decision in decision_rows]
    ax_lr.plot(generations, prev_center, color=CB_PALETTE["sky_blue"], linewidth=1.3, linestyle="--", marker="o", markersize=4, zorder=3, label="previous LR center")
    ax_lr.plot(generations, new_center, color=CB_PALETTE["black"], linewidth=1.8, marker="D", markersize=5, zorder=4, label="new LR center")
    collapsed = [decision["generation"] for decision in decision_rows if decision.get("spread_collapsed")]
    for generation in collapsed:
        ax_lr.axvline(generation, color=CB_PALETTE["orange"], linestyle=":", linewidth=1.1, alpha=0.6, zorder=1)
    if collapsed:
        ax_lr.plot([], [], color=CB_PALETTE["orange"], linestyle=":", linewidth=1.1, label="spread_collapsed")
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Generation")
    ax_lr.set_ylabel("LR center")
    ax_lr.grid(True, alpha=0.3, linewidth=0.5)
    _dedup_legend(ax_lr, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    fig.suptitle("PBT decision history", x=0.02, y=0.975, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.005, _run_caption(manifest), ha="left", va="bottom", fontsize=7.4, color="0.3")

    paths = _save_both(fig, run_dir, "decision_history")
    plt.close(fig)
    return {
        **paths, "warnings": [], "generations": len(decision_rows),
        "members": len({decision["winner"] for decision in decision_rows if decision.get("winner")}),
        "metric_keys": [TOTAL_SCORE_COLUMN, "previous_lr_center", "new_lr_center"],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_research_plots(run_dir, manifest):
    """Generate the full standalone research-figure set for one run.
    Returns {plot_name_key: result_dict}; a None-valued plot (strategy has
    nothing to show, e.g. lr_population/decision_history outside
    anchor_copy_lr_recenter) is simply omitted from the returned dict,
    matching every other conditional plot in this subpackage."""
    member_rows = build_member_metric_rows(manifest)
    decision_rows = build_generation_decision_rows(manifest, member_rows)

    results = {
        "ctag_working_points": plot_ctag_working_points(run_dir, manifest, member_rows, decision_rows),
        "btag_working_points": plot_btag_working_points(run_dir, manifest, member_rows, decision_rows),
        "aggregate_scores": plot_aggregate_scores(run_dir, manifest, member_rows, decision_rows),
        "tag_tradeoff": plot_tag_tradeoff(run_dir, manifest, member_rows, decision_rows),
        "score_vs_lr": plot_score_vs_lr(run_dir, manifest, member_rows, decision_rows),
        "baseline_ratio": plot_baseline_ratio(run_dir, manifest),
    }
    lr_population = plot_lr_population(run_dir, manifest, member_rows, decision_rows)
    if lr_population is not None:
        results["lr_population"] = lr_population
    decision_history = plot_decision_history(run_dir, manifest, decision_rows)
    if decision_history is not None:
        results["decision_history"] = decision_history
    return results

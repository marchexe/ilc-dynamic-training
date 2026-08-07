#!/usr/bin/env python3
"""The report-facing plot set for a PBT run: population/winner overview,
LR lineage, physics-score evolution, the LR-vs-mistag-score population
correlation, and the conditional proxy-validation check (reporting/style.py
for the shared visual system; research_plots.py for the data layer these
all consume). Every function here returns {"png": path|None, "warnings":
[...], "generations": n, "members": n, "metric_keys": [...]} -- never a
bare path -- and takes only already-built, already-validated rows/events;
none of these functions parses the manifest or events.jsonl on its own
beyond what's documented per-function.
"""

import math
from pathlib import Path

from matplotlib.ticker import MaxNLocator

from training.pbt.reporting.constants import (
    BTAG_SCORE_COLUMN,
    CB_PALETTE,
    CTAG_SCORE_COLUMN,
    DECISION_MARKER_STYLE,
    REPORT_PLOT_NAMES,
    ROLE_MARKER_STYLE,
    TOTAL_GEOMEAN_METRIC_KEY,
    TOTAL_SCORE_COLUMN,
)
from training.pbt.reporting.io import read_events
from training.pbt.reporting.metrics_rows import _metric_mode, _metric_name
from training.pbt.reporting.research_plots import (
    _baseline_row,
    _dedup_legend,
    _final_selected_row,
    build_generation_decision_rows,
    build_member_metric_rows,
    generation_winner_member,
    shared_lr_center_series,
    validate_metric_rows,
)
from training.pbt.reporting.statistics import _paired_tier_values, lr_mistag_correlation, ranking_agreement, tier_correlation
from training.pbt.reporting.style import compact_trial, member_color, member_order, plot_setup

# Redefined independently rather than imported -- see planning/
# anchor_copy_lr_recenter.py:35 and state/transitions.py:17, which already
# each carry their own copy of this literal rather than a shared import,
# specifically to avoid training.pbt.state.transitions's own module-level
# dependency back on training.pbt.reporting (a real circular import: this
# module is imported from reporting/canonical.py, which is imported from
# reporting/__init__.py before that package finishes initializing).
ANCHOR_PSEUDO_RECIPIENT = "__anchor__"


def _save_png(fig, run_dir, plot_name_key):
    base = REPORT_PLOT_NAMES[plot_name_key]
    directory = Path(run_dir) / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{base}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return {"png": str(png_path)}


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _contiguous_runs(rows):
    """Split rows (already sorted or not) into maximal runs of consecutive
    `generation` values -- used to draw a population member's trajectory as
    separate line segments rather than bridging a missing/non-finite
    generation with a straight (fabricated) interpolation."""
    rows = sorted(rows, key=lambda row: row["generation"])
    runs = []
    current = []
    previous_generation = None
    for row in rows:
        if previous_generation is not None and row["generation"] != previous_generation + 1:
            runs.append(current)
            current = []
        current.append(row)
        previous_generation = row["generation"]
    if current:
        runs.append(current)
    return runs


# ---------------------------------------------------------------------------
# 1. PBT population and selection
# ---------------------------------------------------------------------------


def plot_pbt_population_selection(run_dir, manifest, member_rows, decision_rows):
    """Research question: how did every population member behave, and who
    was selected winner each generation? Two vertically stacked panels
    sharing one generation axis -- population trajectories on top (every
    member, one stable color each, missing/non-finite generations left as
    a genuine gap rather than interpolated across), a compact winner
    timeline strip below (one cell per generation, colored by the winning
    member, decision-marker overlay for anchor_copy_lr_recenter runs).
    Winner is the authoritative decision winner (row["is_winner"], from
    generation_winner_member/best_worker_in_generation), never re-derived
    from total_mistag_score."""
    order = member_order(manifest)
    mode = _metric_mode(manifest)
    metric_name = _metric_name(manifest)

    rows_by_member = {}
    for row in member_rows:
        value = _finite(row.get("optimization_metric_value"))
        if value is None:
            continue
        rows_by_member.setdefault(row["trial"], []).append({**row, "optimization_metric_value": value})
    if not rows_by_member:
        return {"png": None, "warnings": ["no completed generations to plot"], "generations": 0, "members": 0, "metric_keys": []}

    winner_by_generation = {row["generation"]: row["trial"] for row in member_rows if row.get("is_winner")}
    all_generations = sorted({row["generation"] for rows in rows_by_member.values() for row in rows})

    plt = plot_setup()
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10.0, 7.0), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )

    for member in order:
        rows = rows_by_member.get(member)
        if not rows:
            continue
        color = member_color(member, order)
        for run in _contiguous_runs(rows):
            xs = [row["generation"] for row in run]
            ys = [row["optimization_metric_value"] for row in run]
            ax_top.plot(xs, ys, color=color, linewidth=1.6, alpha=0.9, zorder=2, label=member)

    winner_points = [
        (row["generation"], value)
        for row in member_rows
        if row.get("is_winner") and (value := _finite(row.get("optimization_metric_value"))) is not None
    ]
    if winner_points:
        ax_top.scatter(
            [point[0] for point in winner_points], [point[1] for point in winner_points],
            marker=ROLE_MARKER_STYLE["winner"]["marker"], color=ROLE_MARKER_STYLE["winner"]["color"],
            s=170, zorder=5, label=ROLE_MARKER_STYLE["winner"]["label"], edgecolor="white", linewidth=0.4,
        )

    direction = "lower is better" if mode != "max" else "higher is better"
    ax_top.set_ylabel(f"{metric_name}\n({direction})")
    ax_top.set_title("Population trajectories", fontsize=11, fontweight="bold", loc="left")
    ax_top.grid(True, alpha=0.3, linewidth=0.5)
    _dedup_legend(ax_top, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    # Per-cell text only when there's room to render it legibly -- beyond
    # this many cells, a fixed-size label starts overlapping its neighbors
    # (see spread_collapse's fix history: never shrink font to solve
    # overlap). Color still carries member identity via the top panel's
    # legend; this only removes redundant, no-longer-readable text.
    label_cells = len(all_generations) <= 25
    for generation in all_generations:
        winner = winner_by_generation.get(generation)
        color = member_color(winner, order) if winner else CB_PALETTE["grey"]
        ax_bottom.bar([generation], [1.0], width=0.92, bottom=0.0, color=color, zorder=2)
        if winner and label_cells:
            ax_bottom.text(
                generation, 0.5, compact_trial(winner),
                ha="center", va="center", fontsize=7.2, color="white", fontweight="bold", zorder=3,
            )
    for decision in decision_rows:
        style = DECISION_MARKER_STYLE.get(decision.get("decision"))
        if style is None:
            continue
        ax_bottom.scatter(
            [decision["generation"]], [1.32], marker=style["marker"], color=style["color"],
            s=60, edgecolor="black", linewidth=0.4, zorder=4, label=style["label"],
        )
    ax_bottom.set_ylim(0.0, 1.6 if decision_rows else 1.0)
    ax_bottom.set_yticks([])
    ax_bottom.set_xlabel("Generation")
    ax_bottom.set_title("Winner timeline", fontsize=11, fontweight="bold", loc="left")
    ax_bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
    if decision_rows:
        _dedup_legend(ax_bottom, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    result = _save_png(fig, run_dir, "pbt_population_selection")
    plt.close(fig)
    return {
        **result,
        "warnings": [],
        "generations": len(all_generations),
        "members": len(rows_by_member),
        "metric_keys": ["optimization_metric_value"],
    }


# ---------------------------------------------------------------------------
# 2. Mistag score evolution
# ---------------------------------------------------------------------------


def _dense_generation_series(winner_by_generation, column):
    generations = sorted(winner_by_generation)
    if not generations:
        return [], []
    xs = list(range(generations[0], generations[-1] + 1))
    ys = [_finite((winner_by_generation.get(x) or {}).get(column)) for x in xs]
    ys = [float("nan") if value is None else value for value in ys]
    return xs, ys


def plot_mistag_score_evolution(run_dir, manifest, member_rows):
    """Research question: how did ctag_score, btag_score, and
    total_mistag_score evolve, for the generation winner, across the run?
    One panel: thin blue ctag_score, thin green btag_score, thick dark
    total_mistag_score (always the strongest visual accent) for the
    authoritative decision winner each generation; every member's total
    score as very light grey background context; baseline (if measured) as
    a point to the left of generation 0; final selected checkpoint marked
    separately. Reads only already-computed ctag_score/btag_score/
    total_mistag_score columns -- never recomputes the geometric-mean
    formula."""
    winner_by_generation = {row["generation"]: row for row in member_rows if row.get("is_winner")}
    warnings = []

    plt = plot_setup()
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 5.6), constrained_layout=True)

    context_valid, context_warnings = validate_metric_rows(member_rows, [TOTAL_SCORE_COLUMN])
    warnings.extend(context_warnings)
    if context_valid:
        ax.scatter(
            [row["generation"] for row in context_valid], [row[TOTAL_SCORE_COLUMN] for row in context_valid],
            color=CB_PALETTE["grey"], s=10, alpha=0.25, zorder=1,
        )

    xs, ctag_ys = _dense_generation_series(winner_by_generation, CTAG_SCORE_COLUMN)
    _, btag_ys = _dense_generation_series(winner_by_generation, BTAG_SCORE_COLUMN)
    _, total_ys = _dense_generation_series(winner_by_generation, TOTAL_SCORE_COLUMN)
    if xs:
        ax.plot(xs, ctag_ys, color=CB_PALETTE["blue"], linewidth=1.3, zorder=3, label="ctag_score (winner)")
        ax.plot(xs, btag_ys, color=CB_PALETTE["green"], linewidth=1.3, zorder=3, label="btag_score (winner)")
        ax.plot(xs, total_ys, color=CB_PALETTE["black"], linewidth=3.0, zorder=4, label="total_mistag_score (winner)")

    has_baseline_point = False
    baseline_row, baseline_missing = _baseline_row(manifest)
    warnings.extend(f"baseline: {label} unavailable" for label in baseline_missing)
    if baseline_row is not None and xs:
        baseline_x = xs[0] - 1
        for column, color in (
            (CTAG_SCORE_COLUMN, CB_PALETTE["blue"]),
            (BTAG_SCORE_COLUMN, CB_PALETTE["green"]),
            (TOTAL_SCORE_COLUMN, CB_PALETTE["black"]),
        ):
            baseline_value = _finite(baseline_row.get(column))
            first_value = _finite((winner_by_generation.get(xs[0]) or {}).get(column))
            if baseline_value is None:
                continue
            has_baseline_point = True
            ax.scatter([baseline_x], [baseline_value], marker=ROLE_MARKER_STYLE["baseline"]["marker"], color=color, s=55, zorder=4)
            if first_value is not None:
                ax.plot([baseline_x, xs[0]], [baseline_value, first_value], color=color, linewidth=1.0, linestyle="--", zorder=2)
        if has_baseline_point:
            generation_ticks = [
                tick for tick in MaxNLocator(integer=True, nbins=8).tick_values(xs[0], xs[-1]) if xs[0] <= tick <= xs[-1]
            ]
            ticks = sorted({baseline_x, *generation_ticks})
            ax.set_xticks(ticks)
            ax.set_xticklabels(["baseline" if abs(tick - baseline_x) < 1e-9 else f"{int(round(tick))}" for tick in ticks])

    final_row, final_missing = _final_selected_row(manifest)
    warnings.extend(f"final selected: {label} unavailable" for label in final_missing)
    final_generation = final_row.get("generation") if final_row else None
    final_value = _finite(final_row.get(TOTAL_SCORE_COLUMN)) if final_row else None
    if final_generation is not None and final_value is not None:
        ax.scatter(
            [final_generation], [final_value], marker=ROLE_MARKER_STYLE["final"]["marker"],
            color=ROLE_MARKER_STYLE["final"]["color"], s=90, edgecolor="black", linewidth=0.6,
            zorder=5, label=ROLE_MARKER_STYLE["final"]["label"],
        )

    if not has_baseline_point:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mistag score [%] (lower is better)")
    ax.set_title("Mistag score evolution", fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _dedup_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    result = _save_png(fig, run_dir, "mistag_score_evolution")
    plt.close(fig)
    return {
        **result,
        "warnings": warnings,
        "generations": len(winner_by_generation),
        "members": len({row["trial"] for row in member_rows}),
        "metric_keys": [CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN, TOTAL_SCORE_COLUMN],
        "ranking_metric_is_total_score": _metric_name(manifest) == TOTAL_GEOMEAN_METRIC_KEY,
        "has_baseline_point": has_baseline_point,
    }


# ---------------------------------------------------------------------------
# 3. Learning-rate lineage
# ---------------------------------------------------------------------------


def plot_learning_rate_lineage(run_dir, manifest, member_rows, decision_rows, center_series, events):
    """Research question: which branch did each next-generation member
    descend from, and what LR did it get? A lineage/DAG, not a plain
    LR-vs-generation line plot: one node per (member, generation) at its
    actual training LR; a heavy edge donor(g)->recipient(g+1) for every
    applied exploit copy (events.jsonl, applied=True only -- unapplied/
    skipped events never draw an edge), a light edge member(g)->member(g+1)
    for self-continuation. The __anchor__ pseudo-recipient is drawn as a
    separate thin diamond line from center_series (generation exploit
    records / anchor_copy_lr_recenter decisions), never from events.jsonl,
    since that pseudo-event is never written there (see
    state/transitions.py::apply_exploit's anchor-bundle special case)."""
    order = member_order(manifest)
    pbt_config = manifest.get("config", {}).get("pbt", {})
    min_lr = _finite(pbt_config.get("min_lr"))
    max_lr = _finite(pbt_config.get("max_lr"))

    rows_by_member = {}
    for row in member_rows:
        value = _finite(row.get("LR"))
        if value is None or value <= 0:
            continue
        rows_by_member.setdefault(row["trial"], {})[row["generation"]] = value
    if not rows_by_member:
        return {"png": None, "warnings": ["no LR data to plot"], "generations": 0, "members": 0, "metric_keys": ["LR"]}

    winner_by_generation = {row["generation"]: row["trial"] for row in member_rows if row.get("is_winner")}

    applied_donor = {}
    for event in events:
        if event.get("event_type") != "exploit" or not event.get("applied"):
            continue
        recipient, donor, generation = event.get("recipient"), event.get("donor"), event.get("generation")
        if recipient in (None, ANCHOR_PSEUDO_RECIPIENT) or donor in (None, ANCHOR_PSEUDO_RECIPIENT) or generation is None:
            continue
        applied_donor[(recipient, int(generation) + 1)] = donor

    edges = []
    for member, generations in rows_by_member.items():
        for generation in sorted(generations):
            next_generation = generation + 1
            if next_generation not in generations:
                continue
            donor = applied_donor.get((member, next_generation))
            if donor is None or donor == member:
                edges.append((member, generation, member, next_generation, "self"))
            elif donor in rows_by_member and generation in rows_by_member[donor]:
                edges.append((donor, generation, member, next_generation, "copy"))
            # else: an applied copy happened but the donor's own LR at this
            # generation isn't available to plot from -- leave disconnected
            # rather than guess where the edge should start.

    plt = plot_setup()
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.0), constrained_layout=True)

    for from_member, from_gen, to_member, to_gen, kind in edges:
        from_value = rows_by_member[from_member][from_gen]
        to_value = rows_by_member[to_member][to_gen]
        if kind == "copy":
            ax.plot(
                [from_gen, to_gen], [from_value, to_value], color=member_color(from_member, order),
                linewidth=2.4, alpha=0.9, zorder=3,
            )
        else:
            ax.plot(
                [from_gen, to_gen], [from_value, to_value], color=member_color(to_member, order),
                linewidth=0.9, alpha=0.45, zorder=2,
            )

    for member, generations in rows_by_member.items():
        color = member_color(member, order)
        xs = sorted(generations)
        ys = [generations[x] for x in xs]
        ax.scatter(xs, ys, color=color, s=26, zorder=4, label=member)
        winner_xs = [x for x in xs if winner_by_generation.get(x) == member]
        if winner_xs:
            # zorder above the anchor diamond below (7 > 6): the winner and
            # the active anchor frequently sit at the exact same LR for
            # anchor_copy_lr_recenter (a newly-accepted anchor's LR *is*
            # the winner's LR), so the winner star must never be painted
            # over by the anchor marker at that shared point.
            ax.scatter(
                winner_xs, [generations[x] for x in winner_xs], marker=ROLE_MARKER_STYLE["winner"]["marker"],
                color=ROLE_MARKER_STYLE["winner"]["color"], s=170, zorder=7,
                label=ROLE_MARKER_STYLE["winner"]["label"], edgecolor="white", linewidth=0.4,
            )

    if center_series:
        center_xs = [item[0] for item in center_series]
        center_ys = [item[1] for item in center_series]
        ax.plot(center_xs, center_ys, color=ROLE_MARKER_STYLE["anchor"]["color"], linestyle="--", linewidth=1.1, zorder=5, label="LR center")
        ax.scatter(
            center_xs, center_ys, marker=ROLE_MARKER_STYLE["anchor"]["marker"], color=ROLE_MARKER_STYLE["anchor"]["color"],
            s=60, zorder=6, label=ROLE_MARKER_STYLE["anchor"]["label"],
        )

    if min_lr is not None:
        ax.axhline(min_lr, color=CB_PALETTE["vermillion"], linestyle=":", linewidth=0.9, alpha=0.7, zorder=1, label="min_lr / max_lr")
    if max_lr is not None:
        ax.axhline(max_lr, color=CB_PALETTE["vermillion"], linestyle=":", linewidth=0.9, alpha=0.7, zorder=1)

    collapsed_generations = sorted({decision["generation"] for decision in decision_rows if decision.get("spread_collapsed")})
    for generation in collapsed_generations:
        ax.axvline(generation, color=CB_PALETTE["orange"], linestyle=":", linewidth=1.1, alpha=0.6, zorder=1)
    if collapsed_generations:
        ax.plot([], [], color=CB_PALETTE["orange"], linestyle=":", linewidth=1.1, label="spread_collapsed")

    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate lineage", fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _dedup_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    result = _save_png(fig, run_dir, "learning_rate_lineage")
    plt.close(fig)
    return {
        **result,
        "warnings": [],
        "generations": len({generation for generations in rows_by_member.values() for generation in generations}),
        "members": len(rows_by_member),
        "metric_keys": ["LR"],
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# 4. Learning rate vs. mistag score
# ---------------------------------------------------------------------------


def plot_learning_rate_mistag_correlation(run_dir, manifest, member_rows):
    """Research question: does the LR a member trained at correlate with how
    good its mistag score turned out to be, across the whole explored
    population -- not just the generation winners? One scatter, every
    (member, generation) observation with a finite LR and
    total_mistag_score, colored by generation on a sequential colormap
    (generation is an ordered magnitude here, not a categorical identity
    like member -- so it does not reuse the member-identity CB_PALETTE).
    Winner points (row["is_winner"], the same authoritative flag every
    other report figure uses) get a black ring overlay so they stay
    identifiable without hiding their generation color. Pearson r (on
    log10(LR) -- LR is explored on a log scale, so any real relationship is
    expected to be multiplicative, not additive) and Spearman rho
    (rank-based, invariant to that transform) are read directly from
    statistics.py::lr_mistag_correlation, never recomputed here.
    """
    valid_rows, warnings = validate_metric_rows(member_rows, ["LR", TOTAL_SCORE_COLUMN])
    valid_rows = [row for row in valid_rows if row["LR"] > 0]
    if not valid_rows:
        return {
            "png": None,
            "warnings": warnings or ["no LR/mistag score data to plot"],
            "generations": 0,
            "members": 0,
            "metric_keys": ["LR", TOTAL_SCORE_COLUMN],
        }

    plt = plot_setup()
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.8), constrained_layout=True)

    lrs = [row["LR"] for row in valid_rows]
    values = [row[TOTAL_SCORE_COLUMN] for row in valid_rows]
    generations = [row["generation"] for row in valid_rows]
    scatter = ax.scatter(
        lrs, values, c=generations, cmap="viridis", s=42, alpha=0.85,
        edgecolor="white", linewidth=0.3, zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Generation")
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()

    winner_rows = [row for row in valid_rows if row.get("is_winner")]
    if winner_rows:
        ax.scatter(
            [row["LR"] for row in winner_rows], [row[TOTAL_SCORE_COLUMN] for row in winner_rows],
            marker=ROLE_MARKER_STYLE["winner"]["marker"], s=210, facecolor="none",
            edgecolor=CB_PALETTE["black"], linewidth=1.4, zorder=5, label=ROLE_MARKER_STYLE["winner"]["label"],
        )

    correlation = lr_mistag_correlation(valid_rows)
    if correlation["reason"] == "insufficient_paired_observations":
        caption = f"n={correlation['n']} -- too few for a meaningful correlation"
    elif correlation["reason"]:
        caption = f"n={correlation['n']}, correlation unavailable ({correlation['reason']})"
    else:
        caption = (
            f"n={correlation['n']}  Pearson r (log10 LR)={correlation['pearson_r']:.2f}  "
            f"Spearman rho={correlation['spearman_rho']:.2f}"
        )
    ax.text(0.02, 0.98, caption, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.3")

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Total mistag score [%] (lower is better)")
    ax.set_title("Learning rate vs. mistag score", fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if winner_rows:
        ax.legend(frameon=False, fontsize=8, loc="best")

    result = _save_png(fig, run_dir, "learning_rate_mistag_correlation")
    plt.close(fig)
    return {
        **result,
        "warnings": warnings,
        "generations": len({row["generation"] for row in valid_rows}),
        "members": len({row["trial"] for row in valid_rows}),
        "metric_keys": ["LR", TOTAL_SCORE_COLUMN],
        "correlation": correlation,
    }


# ---------------------------------------------------------------------------
# 5. Proxy validation
# ---------------------------------------------------------------------------


def plot_proxy_validation(run_dir, manifest):
    """Research question: does the control proxy that drives PBT agree
    with an independent check? Only produced when the run recorded a real
    monitor or full_holdout tiered evaluation (None otherwise -- never an
    empty image for a short smoke run). Left panel: the decision winner's
    score by generation, per tier, points only at generations actually
    evaluated on that tier (no interpolation, no smoothing). Right panel:
    control-vs-independent-tier paired scatter with y=x, n, Pearson r,
    Spearman rho, and a top-1 agreement annotation (statistics.py helpers,
    never recomputed here). full_holdout is preferred as the independent
    check; if only monitor is available, the panel is honestly labeled as
    not independent rather than presented as a final verification."""
    rounds = manifest.get("tiered_evaluations") or []
    if not any(round_record.get("tier") in ("monitor", "full_holdout") for round_record in rounds):
        return None

    tier_b = "full_holdout" if any(round_record.get("tier") == "full_holdout" for round_record in rounds) else "monitor"
    independent = tier_b == "full_holdout"
    metric_name = rounds[0].get("metric_name") if rounds else None

    member_rows = build_member_metric_rows(manifest)
    winner_by_generation = {row["generation"]: row["trial"] for row in member_rows if row.get("is_winner")}
    tier_colors = {"control": CB_PALETTE["blue"], "monitor": CB_PALETTE["orange"], "full_holdout": CB_PALETTE["purple"]}

    plt = plot_setup()
    fig, (ax_series, ax_scatter) = plt.subplots(1, 2, figsize=(11.4, 5.0), constrained_layout=True)

    # Exposed on the result dict as "winner_series" purely for test
    # introspection (structured-data assertions instead of pixel
    # inspection), matching learning_rate_lineage's "edges" convention.
    winner_series = {}
    for tier in ("control", "monitor", "full_holdout"):
        tier_rounds = sorted(
            (round_record for round_record in rounds if round_record.get("tier") == tier),
            key=lambda item: item.get("generation") if item.get("generation") is not None else -999,
        )
        if not tier_rounds:
            continue
        xs, ys = [], []
        for round_record in tier_rounds:
            generation = round_record.get("generation")
            winner = winner_by_generation.get(generation)
            if winner is None:
                continue
            record = (round_record.get("members") or {}).get(winner) or {}
            value = _finite((record.get("metrics") or {}).get(round_record.get("metric_name")))
            if value is None:
                continue
            xs.append(generation)
            ys.append(value)
        if xs:
            ax_series.plot(xs, ys, marker="o", markersize=4.5, linewidth=1.4, color=tier_colors.get(tier, "0.4"), label=tier)
            winner_series[tier] = list(zip(xs, ys))
    ax_series.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_series.set_xlabel("Generation")
    ax_series.set_ylabel(metric_name or "metric value")
    ax_series.set_title("Selected-winner score by generation", fontsize=10.5, fontweight="bold", loc="left")
    ax_series.grid(True, alpha=0.3, linewidth=0.5)
    ax_series.legend(frameon=False, fontsize=8, loc="best")

    correlation = tier_correlation(manifest, "control", tier_b)
    pairs = _paired_tier_values(manifest, "control", tier_b)
    if pairs:
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        ax_scatter.scatter(xs, ys, s=32, color=tier_colors.get(tier_b, "0.4"), edgecolor="0.2", linewidth=0.4, zorder=3)
        lo, hi = min(xs + ys), max(xs + ys)
        if hi > lo:
            ax_scatter.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=0.9, zorder=2, label="y = x")
        ax_scatter.legend(frameon=False, fontsize=8, loc="lower right")

    agreement_rows = ranking_agreement(manifest, "control", tier_b)
    if correlation.get("reason") == "insufficient_paired_observations":
        caption = f"n={correlation['n']} paired points -- too few for a meaningful correlation"
    elif correlation.get("reason"):
        caption = f"n={correlation['n']}, correlation unavailable ({correlation['reason']})"
    else:
        caption = f"n={correlation['n']}  Pearson r={correlation['pearson_r']:.2f}  Spearman rho={correlation['spearman_rho']:.2f}"
    if agreement_rows:
        top1_fraction = sum(1 for row in agreement_rows if row["top1_agrees"]) / len(agreement_rows)
        caption += f"\ntop-1 agreement: {top1_fraction:.0%} ({len(agreement_rows)} generation(s))"
    ax_scatter.text(0.02, 0.98, caption, transform=ax_scatter.transAxes, ha="left", va="top", fontsize=8, color="0.3")
    ax_scatter.set_xlabel(f"control {metric_name or ''}")
    ax_scatter.set_ylabel(f"{tier_b} {metric_name or ''}")
    ax_scatter.set_title(
        "control vs. full_holdout (independent)" if independent else "control vs. monitor (not independent)",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax_scatter.grid(True, alpha=0.3, linewidth=0.5)
    if not independent:
        fig.text(
            0.5, -0.02,
            "No full_holdout data available -- shown against monitor as a partial check, not a final independent verification.",
            ha="center", va="top", fontsize=7.6, color="0.35", transform=fig.transFigure,
        )

    result = _save_png(fig, run_dir, "proxy_validation")
    plt.close(fig)
    return {
        **result,
        "warnings": [],
        "generations": len({round_record.get("generation") for round_record in rounds if round_record.get("generation") is not None}),
        "members": len({member for round_record in rounds for member in (round_record.get("members") or {})}),
        "metric_keys": [metric_name] if metric_name else [],
        "independent": independent,
        "tier_b": tier_b,
        "winner_series": winner_series,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_report_plots(run_dir, manifest):
    """Generate the full report-facing plot set for one run. Returns
    {plot_name_key: result_dict}; proxy_validation is simply omitted from
    the returned dict when there's nothing to show (no monitor/full_holdout
    tiered evaluations), matching every other conditional plot in this
    subpackage."""
    member_rows = build_member_metric_rows(manifest)
    decision_rows = build_generation_decision_rows(manifest, member_rows)
    center_series = shared_lr_center_series(manifest)
    events = read_events(run_dir)

    results = {
        "pbt_population_selection": plot_pbt_population_selection(run_dir, manifest, member_rows, decision_rows),
        "mistag_score_evolution": plot_mistag_score_evolution(run_dir, manifest, member_rows),
        "learning_rate_lineage": plot_learning_rate_lineage(run_dir, manifest, member_rows, decision_rows, center_series, events),
        "learning_rate_mistag_correlation": plot_learning_rate_mistag_correlation(run_dir, manifest, member_rows),
    }
    proxy_validation = plot_proxy_validation(run_dir, manifest)
    if proxy_validation is not None:
        results["proxy_validation"] = proxy_validation
    return results

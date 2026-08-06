#!/usr/bin/env python3
"""Data layer shared by the report-facing plots (reporting/report_plots.py).

Kept strictly separate from rendering:

    build_member_metric_rows / build_generation_decision_rows
        -- assemble already-computed manifest/CSV data into flat rows.
        Never reimplement a metric formula; never parse the manifest
        independently of the existing metrics_rows.py layer.

    generation_winner_member / shared_lr_center_series
        -- small authoritative-data adapters: the real per-generation
        decision winner (never re-derived from total_mistag_score), and
        the strategy-specific shared LR center, read from whichever
        manifest field actually carries it for that strategy.

    validate_metric_rows
        -- drop rows missing/non-finite/negative in a required column,
        with a warning naming exactly what was wrong. Never fabricate a
        substitute value.

Rendering itself (report_plots.py) takes only already-built,
already-validated rows from this module -- it never parses the manifest or
events.jsonl on its own.
"""

import math

from training.pbt.reporting.constants import (
    BTAG_SCORE_COLUMN,
    CTAG_SCORE_COLUMN,
    DECISION_MARKER_STYLE,
    TOTAL_SCORE_COLUMN,
)
from training.pbt.reporting.metrics_rows import (
    _metric_mode,
    _metric_name,
    evaluation_metadata,
    evaluation_rows,
    fixed_working_point_values,
    group_score_row,
)
from training.pbt.reporting.plots import (
    _completed_initial_evaluation_metrics,
    _global_best_metrics,
)

RESOLVED_ANCHOR_DECISIONS = tuple(DECISION_MARKER_STYLE)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


def generation_winner_member(manifest, generation_index):
    """The authoritative per-generation decision winner: the member whose
    ranking actually drove this generation's exploit/anchor/global-best
    outcome, i.e. manifest["generations"][i]["ranking"][0] via the same
    metrics.py::best_worker_in_generation the runner itself uses to decide
    manifest["best"] -- confidence-aware incumbent persistence included for
    strategies that have it. Never re-derived from total_mistag_score or
    any other metric than the run's actually configured one. None if the
    generation doesn't exist or has no workers recorded yet.

    Imports metrics.py locally (not at module top level): training.pbt.metrics
    itself imports training.pbt.reporting at module scope (for
    record_new_best), so a top-level import here would be a circular import.
    """
    generation_record = next(
        (item for item in manifest.get("generations", []) if item.get("index") == generation_index),
        None,
    )
    if not generation_record or not generation_record.get("workers"):
        return None
    from training.pbt.metrics import best_worker_in_generation

    try:
        name, _value, _metrics = best_worker_in_generation(manifest.get("config", {}), generation_record)
    except (KeyError, IndexError):
        return None
    return name


def shared_lr_center_series(manifest):
    """(generation, center_lr) pairs for strategies with a genuinely shared
    LR center -- anchor_copy_lr_recenter and anchored_lr_sweep only; empty
    for every other strategy, which have no such concept (never fabricate a
    center line for them).

    Each strategy persists its center at a different manifest path, so
    this is a small strategy-aware adapter over generation exploit records
    -- never a metric recomputation:

      anchor_copy_lr_recenter: generation["anchor_copy_lr_recenter"]["new_lr_center"],
        the once-per-generation decision summary (planning/anchor_copy_lr_recenter.py).
      anchored_lr_sweep: generation["exploit"][*]["lr_center"] -- every
        member's plan event for that generation carries the same shared
        center value (planning/anchored_lr_sweep.py); events.jsonl's own
        "exploit" event type does not carry lr_center (only the manifest's
        generation-embedded raw plan records do), so this reads from
        generation["exploit"], not from events.jsonl.
    """
    strategy = manifest.get("config", {}).get("pbt", {}).get("strategy")
    generations = sorted(manifest.get("generations", []), key=lambda item: item.get("index", 0))
    series = []
    if strategy == "anchor_copy_lr_recenter":
        for generation in generations:
            info = generation.get("anchor_copy_lr_recenter")
            if info and info.get("new_lr_center") is not None:
                series.append((generation["index"], float(info["new_lr_center"])))
    elif strategy == "anchored_lr_sweep":
        for generation in generations:
            for event in generation.get("exploit") or []:
                if event.get("source") == "anchored_lr_sweep" and event.get("lr_center") is not None:
                    series.append((generation["index"], float(event["lr_center"])))
                    break
    return series


def build_member_metric_rows(manifest):
    """One row per (generation, member): every raw working-point value,
    ctag_score/btag_score/total_mistag_score, LR, samples_seen, plus an
    is_winner flag -- the authoritative per-generation decision winner
    (generation_winner_member above), never re-derived from
    total_mistag_score even when the two happen to coincide.

    Thin wrapper over evaluation_rows() (the existing authoritative
    row-preparation layer, reporting/metrics_rows.py) -- adds only the
    winner flag; every metric value already comes from there unmodified.
    """
    rows = evaluation_rows(manifest)
    winner_by_generation = {}
    for row in rows:
        generation = row["generation"]
        if generation not in winner_by_generation:
            winner_by_generation[generation] = generation_winner_member(manifest, generation)
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


def _dedup_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen[label] = handle
    if seen:
        ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=7.6, **kwargs)

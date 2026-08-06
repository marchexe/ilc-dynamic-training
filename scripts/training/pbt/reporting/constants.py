#!/usr/bin/env python3
"""Shared constants for the pbt/reporting/ subpackage: file names, fixed
working-point definitions, plot styling, and CSV column schemas."""

EVENTS_NAME = "events.jsonl"


METRICS_NAME = "metrics.csv"


SUMMARY_NAME = "summary.json"


REPORT_NAME = "report.md"


PLOT_NAMES = {
    "training_evolution": "training_evolution.png",
    "ctag_fixed_efficiency_mistag": "ctag_fixed_efficiency_mistag.png",
    "btag_fixed_efficiency_mistag": "btag_fixed_efficiency_mistag.png",
    "geometric_mistag_scores": "geometric_mistag_scores.png",
    "baseline_comparison": "baseline_vs_selected.png",
    "proxy_diagnostics": "proxy_diagnostics.png",
    # Only produced for pbt.strategy == "anchor_copy_lr_recenter" runs (see
    # reporting/plots.py::plot_pbt_decision_evolution) -- every other
    # strategy has no anchor/decision state to show.
    "pbt_decision": "pbt_total_score_and_lr_evolution.png",
}


CONDITIONAL_PLOT_NAMES = ("baseline_comparison", "proxy_diagnostics", "pbt_decision")


EXPLOIT_TABLE_NAME = "plots/report/exploit_table.csv"


SKIPPED_EXPLOIT_TABLE_NAME = "plots/report/skipped_exploits.csv"


TIERED_METRICS_NAME = "tiered_metrics.csv"


_FIXED_WORKING_POINTS_BASE = (
    {"tag": "b", "efficiency": 0.80, "background": "c", "column": "btag_c_mistag_percent_at_0p80", "label": "c bkg, b-eff 80%"},
    {"tag": "b", "efficiency": 0.80, "background": "d", "column": "btag_d_mistag_percent_at_0p80", "label": "d bkg, b-eff 80%"},
    {"tag": "b", "efficiency": 0.90, "background": "c", "column": "btag_c_mistag_percent_at_0p90", "label": "c bkg, b-eff 90%"},
    {"tag": "b", "efficiency": 0.90, "background": "d", "column": "btag_d_mistag_percent_at_0p90", "label": "d bkg, b-eff 90%"},
    {"tag": "c", "efficiency": 0.50, "background": "b", "column": "ctag_b_mistag_percent_at_0p50", "label": "b bkg, c-eff 50%"},
    {"tag": "c", "efficiency": 0.50, "background": "d", "column": "ctag_d_mistag_percent_at_0p50", "label": "d bkg, c-eff 50%"},
    {"tag": "c", "efficiency": 0.80, "background": "b", "column": "ctag_b_mistag_percent_at_0p80", "label": "b bkg, c-eff 80%"},
    {"tag": "c", "efficiency": 0.80, "background": "d", "column": "ctag_d_mistag_percent_at_0p80", "label": "d bkg, c-eff 80%"},
)


def _with_canonical_labels(point):
    # "pair" is tag+background (e.g. tag="c", background="b" -> "cb" --
    # matches training.runtime.WORKING_POINT_DEFINITION's convention
    # exactly: cb is a *c-tag* working point evaluated against b
    # background, never to be confused with bc, a *b-tag* working point
    # evaluated against c background). "score_label" is the canonical
    # "cb@0.5"-style display label -- derived here, not hand-duplicated
    # anywhere else, so pair/label/efficiency can never drift apart.
    pair = f"{point['tag']}{point['background']}"
    return {**point, "pair": pair, "score_label": f"{pair}@{point['efficiency']:g}"}


# The one canonical, shared definition of every fixed working point used
# anywhere in PBT reporting -- CSV columns, plot labels, and the
# ctag_score/btag_score groupings below are all derived from this single
# tuple, never redefined independently.
FIXED_WORKING_POINTS = tuple(_with_canonical_labels(point) for point in _FIXED_WORKING_POINTS_BASE)


# The four working points behind ctag_score / btag_score respectively --
# exactly training.runtime.CTAG_REFERENCE_WORKING_POINTS /
# BTAG_REFERENCE_WORKING_POINTS, expressed as the same FIXED_WORKING_POINTS
# entries (so callers get column/label/score_label for free) rather than as
# a second, separately-maintained list of (pair, efficiency) tuples.
CTAG_SCORE_WORKING_POINTS = tuple(point for point in FIXED_WORKING_POINTS if point["tag"] == "c")
BTAG_SCORE_WORKING_POINTS = tuple(point for point in FIXED_WORKING_POINTS if point["tag"] == "b")


# Canonical aggregate-score metric keys -- the one place these strings are
# spelled out; every consumer (runtime.py's live computation, the
# reporting-layer reconstruction path, plots, reports) imports them rather
# than re-typing the key name.
CTAG_GEOMEAN_METRIC_KEY = "validation_ctag_reference_mistag_geomean_percent"
BTAG_GEOMEAN_METRIC_KEY = "validation_btag_reference_mistag_geomean_percent"
TOTAL_GEOMEAN_METRIC_KEY = "validation_total_reference_mistag_geomean_percent"


FIXED_WORKING_POINT_COLUMNS = tuple(point["column"] for point in FIXED_WORKING_POINTS)


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


WORKING_POINT_STYLE_RANK = _working_point_style_ranks()


WORKING_POINT_MARKERS = ("o", "s")


WORKING_POINT_LINESTYLES = ("-", "--")


CONTROLLER_OBJECTIVE_COLUMN = "controller_objective_mistag_percent"


# The three canonical geometric-mean scores (see
# reporting/metrics_rows.py::group_score_row) plus a warning column that is
# empty when the row is complete and otherwise names exactly which raw
# working-point value(s) were missing/invalid for this row -- never a
# silently-invented 0/NaN/other-metric substitute.
CTAG_SCORE_COLUMN = "ctag_score"
BTAG_SCORE_COLUMN = "btag_score"
TOTAL_SCORE_COLUMN = "total_mistag_score"
GROUP_SCORE_WARNING_COLUMN = "group_score_warning"
GROUP_SCORE_COLUMNS = (CTAG_SCORE_COLUMN, BTAG_SCORE_COLUMN, TOTAL_SCORE_COLUMN, GROUP_SCORE_WARNING_COLUMN)


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
    *GROUP_SCORE_COLUMNS,
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
    *GROUP_SCORE_COLUMNS,
    *FIXED_WORKING_POINT_COLUMNS,
    *FIXED_WORKING_POINT_UNCERTAINTY_COLUMNS,
)


TIER_ORDER = ("control", "monitor", "full", "full_holdout")

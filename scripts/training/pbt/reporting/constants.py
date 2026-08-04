#!/usr/bin/env python3
"""Shared constants for the pbt/reporting/ subpackage: file names, fixed
working-point definitions, plot styling, and CSV column schemas."""

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

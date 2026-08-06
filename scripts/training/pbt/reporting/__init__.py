#!/usr/bin/env python3
"""Canonical structured artifacts for PBT run directories.

Split by concern: `constants` (file names, fixed-WP definitions, plot
styling, CSV schemas), `io` (low-level atomic run-dir I/O + the run
contract), `events` (event-log writers), `metrics_rows` (fixed-WP physics
math + metrics/tiered/exploit CSV writers), `statistics` (cross-tier
correlation/ranking-agreement/overfitting diagnostics), `plots`
(matplotlib report plots), `markdown_report` (report.md + summary.json),
and `canonical` (write_canonical_outputs, the top-level orchestrator that
ties all of the above together).
"""

from training.pbt.reporting.canonical import write_canonical_outputs
from training.pbt.reporting.events import (
    record_anchor_decision,
    record_controller_lr_change,
    record_evaluation,
    record_exploit_application,
    record_initial_evaluation,
    record_new_best,
    record_skipped_exploit,
    record_tiered_evaluation_round,
    record_train_finish,
    record_train_start,
)
from training.pbt.reporting.io import append_event, ensure_run_layout, run_contract, write_resolved_config
from training.pbt.reporting.metrics_rows import (
    evaluation_rows,
    fixed_working_point_uncertainty,
    format_mistag_value,
    refresh_metrics_csv,
    tiered_evaluation_rows,
    wilson_interval,
    write_tiered_metrics_csv,
)
from training.pbt.reporting.plots import selected_generation_rows
from training.pbt.reporting.statistics import (
    best_checkpoint_by_tier,
    corroboration_status,
    proxy_overfitting_cases,
    proxy_selected_checkpoint_other_tiers,
    ranking_agreement,
    tier_correlation,
)

__all__ = [
    "append_event",
    "best_checkpoint_by_tier",
    "corroboration_status",
    "ensure_run_layout",
    "evaluation_rows",
    "fixed_working_point_uncertainty",
    "format_mistag_value",
    "proxy_overfitting_cases",
    "proxy_selected_checkpoint_other_tiers",
    "ranking_agreement",
    "record_anchor_decision",
    "record_controller_lr_change",
    "record_evaluation",
    "record_exploit_application",
    "record_initial_evaluation",
    "record_new_best",
    "record_skipped_exploit",
    "record_tiered_evaluation_round",
    "record_train_finish",
    "record_train_start",
    "refresh_metrics_csv",
    "run_contract",
    "selected_generation_rows",
    "tier_correlation",
    "tiered_evaluation_rows",
    "wilson_interval",
    "write_canonical_outputs",
    "write_resolved_config",
    "write_tiered_metrics_csv",
]

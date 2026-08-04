#!/usr/bin/env python3
"""write_canonical_outputs(): the top-level orchestrator that ties every
other pbt/reporting/ submodule together into one run-directory artifact
set. This is the highest-level module in the subpackage -- it is the only
one allowed to import from all the others."""

import json
from pathlib import Path

import yaml

from training.runtime import atomic_json, utc_now
from training.pbt.reporting.constants import EVENTS_NAME, METRICS_NAME, REPORT_NAME, SUMMARY_NAME
from training.pbt.reporting.io import ensure_run_layout, read_events, write_resolved_config
from training.pbt.reporting.markdown_report import write_report, write_summary_json
from training.pbt.reporting.metrics_rows import (
    refresh_metrics_csv,
    write_exploit_table,
    write_skipped_exploits_table,
    write_tiered_metrics_csv,
)
from training.pbt.reporting.plots import write_existing_physics_reports, write_plots

def write_canonical_outputs(run_dir, manifest):
    ensure_run_layout(run_dir)
    run_dir = Path(run_dir)
    atomic_json(run_dir / "manifest.json", manifest)
    write_resolved_config(run_dir, manifest.get("config", {}))
    refresh_metrics_csv(run_dir, manifest)
    physics_outputs = write_existing_physics_reports(run_dir, manifest)
    tiered_metrics_path = write_tiered_metrics_csv(run_dir, manifest)
    events = read_events(run_dir)
    exploit_table = write_exploit_table(run_dir, events)
    skipped_exploit_table = write_skipped_exploits_table(run_dir, events)
    plots = write_plots(run_dir, manifest)
    manifest["canonical_artifacts"] = {
        "events": str(run_dir / EVENTS_NAME),
        "metrics": str(run_dir / METRICS_NAME),
        "tiered_metrics": str(tiered_metrics_path),
        "summary": str(run_dir / SUMMARY_NAME),
        "report": str(run_dir / REPORT_NAME),
        "plots": {
            **plots,
            **physics_outputs,
            "exploit_table_csv": str(exploit_table),
            "skipped_exploit_table_csv": str(skipped_exploit_table),
        },
        "resolved_config": str(run_dir / "resolved_config.yaml"),
    }
    manifest["updated_at"] = utc_now()
    atomic_json(run_dir / "manifest.json", manifest)
    summary_path = write_summary_json(run_dir, manifest)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_path = write_report(run_dir, manifest, summary)
    manifest["canonical_artifacts"]["report"] = str(report_path)
    atomic_json(run_dir / "manifest.json", manifest)
    return manifest["canonical_artifacts"]

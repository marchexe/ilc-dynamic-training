#!/usr/bin/env python3
"""The human-readable run report (report.md) and its JSON summary
(summary.json)."""

import json
from pathlib import Path

from training.runtime import atomic_json
from training.pbt.reporting.constants import (
    BTAG_SCORE_COLUMN,
    CONDITIONAL_REPORT_PLOT_NAMES,
    CONTROLLER_OBJECTIVE_COLUMN,
    CTAG_SCORE_COLUMN,
    EXPLOIT_TABLE_NAME,
    FIXED_WORKING_POINTS,
    GROUP_SCORE_WARNING_COLUMN,
    REPORT_NAME,
    REPORT_PLOT_NAMES,
    SKIPPED_EXPLOIT_TABLE_NAME,
    SUMMARY_NAME,
    TOTAL_GEOMEAN_METRIC_KEY,
    TOTAL_SCORE_COLUMN,
)
from training.pbt.reporting.io import atomic_text, ensure_run_layout, metric_definition, read_events
from training.pbt.reporting.metrics_rows import (
    _metric_mode,
    _metric_name,
    baseline_record,
    configured_baseline_record,
    evaluation_metadata,
    final_best_row,
    read_metrics_rows,
    relative_change,
)
from training.pbt.reporting.statistics import (
    best_checkpoint_by_tier,
    corroboration_status,
    lr_mistag_correlation,
    proxy_overfitting_cases,
    proxy_selected_checkpoint_other_tiers,
    tier_correlation,
)

def build_summary(run_dir, manifest):
    rows = read_metrics_rows(run_dir)
    events = read_events(run_dir)
    metric = _metric_name(manifest)
    mode = _metric_mode(manifest)
    baseline = baseline_record(manifest)
    configured_baseline = configured_baseline_record(manifest)
    best = manifest.get("best")
    best_value = None if best is None else best.get("metric_value")
    final_row = rows[-1] if rows else None
    final_generation = None
    if manifest.get("generations"):
        final_generation = max(manifest["generations"], key=lambda item: item.get("index", -1))
    final_best = None
    if final_generation:
        final_rows = [row for row in rows if row["generation"] == final_generation.get("index")]
        final_best = final_best_row(final_rows, mode)
    baseline_value = None if baseline is None else baseline.get("metric_value")
    return {
        "schema_version": 1,
        "experiment": manifest.get("experiment"),
        "status": manifest.get("status"),
        "method": manifest.get("method") or manifest.get("run", {}).get("method_name"),
        "metric": {
            "name": metric,
            "mode": mode,
            "definition": metric_definition(metric),
        },
        "starting_checkpoint": manifest.get("initial_resume") or manifest.get("checkpoint"),
        "dataset": manifest.get("datasets") or manifest.get("run", {}).get("datasets"),
        "population": sorted(manifest.get("members", {})),
        "schedule": (manifest.get("run") or {}).get("schedule"),
        "baseline": baseline,
        "configured_baseline": configured_baseline,
        "best": best,
        "final_best": final_best,
        "winning_trial": None if best is None else best.get("member"),
        "best_improvement_vs_baseline": relative_change(mode, baseline_value, best_value),
        "final_improvement_vs_baseline": relative_change(
            mode,
            baseline_value,
            None if final_best is None else final_best.get("optimization_metric_value"),
        ),
        "lr_trajectory": {
            trial: [
                {"generation": row["generation"], "samples_seen": row["samples_seen"], "LR": row["LR"]}
                for row in rows
                if row["trial"] == trial and row["LR"] is not None
            ]
            for trial in sorted({row["trial"] for row in rows})
        },
        "exploit_history": [
            event
            for event in events
            if event.get("event_type") in {"exploit", "weight_copy", "optimizer_copy", "lr_change"}
        ],
        "event_counts": {
            event_type: sum(1 for event in events if event.get("event_type") == event_type)
            for event_type in sorted({event.get("event_type") for event in events})
        },
        "evaluation": evaluation_metadata(manifest),
        "checkpoint_selection": manifest.get("checkpoint_selection_for_report"),
        "plots": {
            **{
                name: str(Path("plots") / f"{filename}.png")
                for name, filename in REPORT_PLOT_NAMES.items()
                if name not in CONDITIONAL_REPORT_PLOT_NAMES or (Path(run_dir) / "plots" / f"{filename}.png").is_file()
            },
            "physics_performance": str(Path("plots") / "report" / "physics_performance.png"),
            "background_efficiency_curves": str(Path("plots") / "diagnostics" / "background_efficiency_curves.png"),
            "btag_mistag_table_csv": str(Path("plots") / "report" / "btag_mistag_tables.csv"),
            "ctag_mistag_table_csv": str(Path("plots") / "report" / "ctag_mistag_tables.csv"),
            "exploit_table_csv": EXPLOIT_TABLE_NAME,
        },
        "checkpoints": {
            "global_best_state": None if best is None else best.get("state_path"),
            "global_best_optimizer": None if best is None else best.get("optimizer_path"),
            "global_best_metadata": None if best is None else best.get("metadata_path"),
        },
    }


def write_summary_json(run_dir, manifest):
    ensure_run_layout(run_dir)
    path = Path(run_dir) / SUMMARY_NAME
    summary = build_summary(run_dir, manifest)
    atomic_json(path, summary)
    return path


def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _report_plot_result(manifest, key):
    return ((manifest.get("canonical_artifacts") or {}).get("plots") or {}).get(key) or {}


def _final_physics_performance_section_lines(manifest, plots):
    """## Final Physics Performance -- background_efficiency_curves.png (the
    required report-facing figure) plus physics_performance.png and the two
    mistag CSVs, all resolved to the same checkpoint role
    (write_existing_physics_reports / manifest["checkpoint_selection_for_report"]).
    States the role/member/generation/selection-metric explicitly -- never
    a bare "best" -- and, when it differs, names the best-physics-score
    checkpoint separately so the two are never conflated."""
    selection = manifest.get("checkpoint_selection_for_report") or {}
    lines = ["", "## Final Physics Performance"]
    lines.append(f"- [Background efficiency curves]({plots.get('background_efficiency_curves', 'plots/diagnostics/background_efficiency_curves.png')})")
    if selection:
        lines.append(
            f"- Checkpoint: **{selection.get('role_label', selection.get('role', 'n/a'))}** "
            f"(`{selection.get('member', 'n/a')}`, generation {selection.get('generation', 'n/a')}), "
            f"selection metric: `{selection.get('selection_metric_name', 'n/a')}` ({selection.get('selection_metric_mode', 'n/a')})"
        )
        if selection.get("fallback_reason"):
            lines.append(
                f"  - Using `{selection.get('role')}` instead of the global-selected checkpoint: {selection['fallback_reason']}"
            )
        elif selection.get("agrees_with_best_physics") is False:
            lines.append(
                f"  - Differs from the separate best-physics-score checkpoint (`{selection.get('best_physics_member', 'n/a')}`, "
                f"generation {selection.get('best_physics_generation', 'n/a')}) -- these are two distinct selection criteria, "
                "not the same checkpoint."
            )
        dataset = selection.get("validation_dataset")
        if dataset:
            lines.append(
                f"  - Validation: `{dataset}` (`{selection.get('validation_suffix', 'n/a')}`), "
                f"{_fmt(selection.get('validation_sample_count'))} samples"
            )
    lines.append(f"- [Physics performance]({plots.get('physics_performance', 'plots/report/physics_performance.png')})")
    lines.append(f"- [C-tag mistag CSV]({plots.get('ctag_mistag_table_csv', 'plots/report/ctag_mistag_tables.csv')})")
    lines.append(f"- [B-tag mistag CSV]({plots.get('btag_mistag_table_csv', 'plots/report/btag_mistag_tables.csv')})")
    return lines


def _pbt_population_selection_section_lines(manifest, plots):
    """## PBT Population and Selection -- pbt_population_selection.png:
    every member's trajectory plus each generation's winner, both by the
    run's actual configured selection metric."""
    result = _report_plot_result(manifest, "pbt_population_selection")
    png = plots.get("pbt_population_selection") or result.get("png")
    lines = ["", "## PBT Population and Selection"]
    if not png:
        lines.append("- No completed generations to plot yet.")
        return lines
    pbt_config = manifest.get("config", {}).get("pbt", {})
    lines.append(f"- [Population and selection]({png})")
    lines.append(
        f"- Ranking metric: `{pbt_config.get('metric', 'n/a')}` ({pbt_config.get('mode', 'n/a')}); "
        "winner = each generation's authoritative decision winner (the member that actually drove that generation's "
        "exploit/anchor/global-best outcome), never re-derived from total_mistag_score."
    )
    if pbt_config.get("strategy") == "anchor_copy_lr_recenter":
        lines.append(
            "- Winner-timeline decision markers: `^` accepted_new_anchor, `o` reused_previous_anchor, "
            "`v` rewound_to_previous_anchor."
        )
    if result.get("warnings"):
        lines.extend(["", "**Data-quality warnings:**"])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return lines


def _mistag_score_evolution_section_lines(manifest, plots):
    """## Mistag Score Evolution -- mistag_score_evolution.png: ctag_score /
    btag_score / total_mistag_score for the generation winner, with the
    total line always the dominant visual accent."""
    result = _report_plot_result(manifest, "mistag_score_evolution")
    png = plots.get("mistag_score_evolution") or result.get("png")
    lines = ["", "## Mistag Score Evolution"]
    if not png:
        lines.append("- No completed generations to plot yet.")
        return lines
    lines.append(f"- [Mistag score evolution]({png})")
    if result.get("ranking_metric_is_total_score"):
        lines.append(
            "- **`total_mistag_score` (sqrt(ctag_score * btag_score)) is this run's PBT ranking metric** -- "
            "the thick line above is the ranking metric itself, not just a diagnostic summary."
        )
    else:
        metric_name = manifest.get("config", {}).get("pbt", {}).get("metric", "n/a")
        lines.append(
            f"- Physics score summary; PBT ranking metric: `{metric_name}`. The winner marked on each line is still "
            "selected by that real configured metric, never by total_mistag_score."
        )
    lines.append(
        "- Baseline point: "
        + ("shown (measured pretrained baseline available)." if result.get("has_baseline_point") else "not available for this run.")
    )
    if result.get("warnings"):
        lines.extend(["", "**Data-quality warnings:**"])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return lines


def _learning_rate_lineage_section_lines(manifest, plots):
    """## Learning-Rate Lineage -- learning_rate_lineage.png: which branch
    each next-generation member descended from and what LR it got."""
    result = _report_plot_result(manifest, "learning_rate_lineage")
    png = plots.get("learning_rate_lineage") or result.get("png")
    lines = ["", "## Learning-Rate Lineage"]
    if not png:
        lines.append("- No learning-rate data to plot yet.")
        return lines
    lines.append(f"- [Learning-rate lineage]({png})")
    lines.append(
        "- Heavy edge = an applied donor->recipient checkpoint copy (events.jsonl, applied=True only); "
        "light edge = a member continuing its own branch."
    )
    strategy = manifest.get("config", {}).get("pbt", {}).get("strategy")
    if strategy == "fixed_lr_grid":
        lines.append("- `fixed_lr_grid`: independent branches, no copy events -- by design, not a data gap.")
    if result.get("warnings"):
        lines.extend(["", "**Data-quality warnings:**"])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return lines


def _learning_rate_mistag_correlation_section_lines(manifest, plots, rows):
    """## Learning Rate vs. Mistag Score Correlation --
    learning_rate_mistag_correlation.png: two independent panels. Left is
    training dynamics only (population median + min/max band and the
    generation winner, against generation -- no LR on this panel). Right
    is the within-generation LR analysis only (log10(LR) vs. each
    observation's residual against its own generation's median, plus an
    OLS trend line) -- a direct, by-eye view of the exact quantity the
    population-wide, generation-confound-controlled correlation below is
    computed from (statistics.py::lr_mistag_correlation, detrended by each
    generation's median score so ordinary training progress isn't mistaken
    for an LR effect). `rows` is the already-persisted metrics.csv
    round-trip (same rows `_model_selection_score_table_lines` uses below)
    -- LR and total_mistag_score are both plain CSV columns, so this never
    needs the manifest-derived member_rows the plot itself was built
    from."""
    result = _report_plot_result(manifest, "learning_rate_mistag_correlation")
    png = plots.get("learning_rate_mistag_correlation") or result.get("png")
    lines = ["", "## Learning Rate vs. Mistag Score Correlation"]
    if not png:
        lines.append("- No generation winners with LR/mistag score to plot yet.")
        return lines
    lines.append(f"- [Training dynamics and within-generation LR analysis]({png})")
    correlation = lr_mistag_correlation(rows)
    if correlation["reason"] == "insufficient_paired_observations":
        lines.append(
            f"- Population-wide, generation-controlled correlation: n={correlation['n']} paired observations -- "
            "too few for a meaningful correlation"
        )
    elif correlation["reason"]:
        lines.append(f"- Population-wide, generation-controlled correlation: unavailable ({correlation['reason']})")
    else:
        pearson_ci = correlation.get("pearson_r_ci")
        spearman_ci = correlation.get("spearman_rho_ci")
        pearson_ci_text = f" (95% CI {pearson_ci[0]:.3f} to {pearson_ci[1]:.3f})" if pearson_ci else ""
        spearman_ci_text = f" (95% CI {spearman_ci[0]:.3f} to {spearman_ci[1]:.3f})" if spearman_ci else ""
        lines.append(
            f"- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by "
            f"each generation's median): n={correlation['n']}, Pearson r={correlation['pearson_r']:.3f}"
            f"{pearson_ci_text}, Spearman rho={correlation['spearman_rho']:.3f}{spearman_ci_text}"
        )
        if not pearson_ci:
            lines.append(
                "- CI unavailable (too few generations to block-bootstrap -- see statistics.py::"
                "_block_bootstrap_correlation); point estimate above only."
            )
        lines.append(
            "- Detrending removes the ordinary training-progress trend (score improves over generations regardless "
            "of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign "
            "convention: positive means higher LR associates with a worse-than-typical (for that generation) score; "
            "negative means better-than-typical. Not a causal claim."
        )
    if result.get("warnings"):
        lines.extend(["", "**Data-quality warnings:**"])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return lines


def _proxy_validation_section_lines(manifest, plots):
    """## Proxy Validation -- proxy_validation.png (only when a monitor or
    full_holdout tiered evaluation was actually recorded) plus the existing
    cross-tier correlation/ranking-agreement/overfitting-case diagnostics.
    The header is always emitted; runs with no corroboration-tier data get
    an explicit one-line explanation instead of a broken image link."""
    result = _report_plot_result(manifest, "proxy_validation")
    png = plots.get("proxy_validation") or result.get("png")
    lines = ["", "## Proxy Validation"]
    if not png:
        lines.append("No corroboration-tier evaluation was scheduled during this short run.")
        return lines
    lines.append(f"- [Proxy validation]({png})")
    if result.get("independent") is False:
        lines.append(
            "- Shown against `monitor`, not `full_holdout` -- this is a partial check, not a final independent verification."
        )

    for label, correlation in (
        ("control vs. monitor", tier_correlation(manifest, "control", "monitor")),
        ("control vs. full_holdout (independent, excludes control+monitor)", tier_correlation(manifest, "control", "full_holdout")),
    ):
        if correlation["reason"] == "insufficient_paired_observations":
            lines.append(f"- {label} correlation: n={correlation['n']} paired observations -- too few for a meaningful correlation")
        elif correlation["reason"]:
            lines.append(f"- {label} correlation: unavailable ({correlation['reason']})")
        else:
            lines.append(
                f"- {label} correlation: n={correlation['n']}, Pearson r={correlation['pearson_r']:.3f}, "
                f"Spearman rho={correlation['spearman_rho']:.3f}"
            )

    best_by_tier = best_checkpoint_by_tier(manifest)
    if best_by_tier:
        bits = ", ".join(
            f"{tier}: `{info['member']}` gen {info['generation']} ({_fmt(info['metric_value'])})"
            for tier, info in best_by_tier.items()
        )
        lines.append(f"- Best checkpoint by tier: {bits}")
        if len(best_by_tier) > 1:
            agree = len({(info["member"], info["generation"]) for info in best_by_tier.values()}) == 1
            lines.append(f"- Best-checkpoint agreement across tiers: {'AGREE' if agree else 'DISAGREE'}")

    selected_other_tiers = proxy_selected_checkpoint_other_tiers(manifest)
    if selected_other_tiers["tiers"]:
        bits = ", ".join(f"{tier}: {_fmt(info.get('metric_value'))}" for tier, info in selected_other_tiers["tiers"].items())
        lines.append(
            f"- Control-selected global best (`{selected_other_tiers.get('member')}`, gen {selected_other_tiers.get('generation')}) "
            f"measured on other tiers: {bits}"
        )
    else:
        lines.append("- Control-selected global best has not been evaluated on monitor/full yet.")

    status, details = corroboration_status(manifest)
    lines.append(f"- Corroboration status: **{status}**")
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    for tier, info in details.items():
        if not info.get("available"):
            lines.append(f"  - {tier}: not available (baseline or selected checkpoint not evaluated on this tier)")
            continue
        delta = relative_change(mode, info["baseline"], info["selected"])
        lines.append(
            f"  - {tier}: baseline {_fmt(info['baseline'])} -> selected {_fmt(info['selected'])} "
            f"({'improved' if info['improved'] else 'not improved'}, "
            f"{_fmt(None if delta is None else 100.0 * delta)}% relative change)"
        )

    overfitting = proxy_overfitting_cases(manifest)
    if overfitting:
        lines.append(f"- **{len(overfitting)} proxy-overfitting case(s) detected** (control improved, monitor did not):")
        for case in overfitting[:10]:
            lines.append(
                f"  - `{case['member']}` gen {case['generation_from']}->{case['generation_to']}: "
                f"control {_fmt(case['control_before'])}->{_fmt(case['control_after'])}, "
                f"monitor {_fmt(case['monitor_before'])}->{_fmt(case['monitor_after'])}"
            )
        if len(overfitting) > 10:
            lines.append(f"  - ... and {len(overfitting) - 10} more (see tiered_metrics.csv)")
    else:
        lines.append("- No proxy-overfitting cases detected (control improved while monitor did not) in the paired generations evaluated so far.")
    return lines


def _shutdown_warning_summary(manifest):
    count = 0
    total = 0

    def scan(metrics):
        nonlocal count, total
        if metrics is None:
            return
        total += 1
        if metrics.get("validation_shutdown_warning"):
            count += 1

    scan((manifest.get("initial_evaluation") or {}).get("metrics"))
    for generation in manifest.get("generations", []):
        for worker in (generation.get("workers") or {}).values():
            scan(worker.get("metrics"))
    for round_record in manifest.get("tiered_evaluations", []):
        for record in (round_record.get("members") or {}).values():
            scan(record.get("metrics"))
    if count == 0:
        return f"No data-loader shutdown-race warnings observed across {total} evaluation(s)."
    return (
        f"Data-loader shutdown-race warning (validation_shutdown_warning) observed in {count}/{total} "
        "evaluation(s) -- treat affected metrics with extra caution."
    )


def _model_selection_score_table_lines(manifest, rows):
    """Per-member summary table for the final generation: all 8 raw
    working-point mistag values, the 3 canonical aggregate scores, LR, and
    winner/anchor status. Consumes `rows` (already built by
    evaluation_rows/group_score_row) directly -- no metric is recomputed
    here. Deterministic member ordering (alphabetical by trial name)."""
    if not manifest.get("generations"):
        return []
    final_generation = max(manifest["generations"], key=lambda item: item.get("index", -1))
    final_index = final_generation.get("index")
    final_rows = sorted(
        (row for row in rows if row.get("generation") == final_index),
        key=lambda row: row.get("trial") or "",
    )
    if not final_rows:
        return []
    winner_row = final_best_row(final_rows, _metric_mode(manifest))
    winner_name = winner_row.get("trial") if winner_row else None
    anchor_member = (manifest.get("anchor") or {}).get("member")

    header = [
        "member",
        *(point["score_label"] for point in FIXED_WORKING_POINTS),
        "ctag_score", "btag_score", "total_mistag_score", "LR", "status",
    ]
    lines = [
        "",
        "## Model Selection Scores",
        f"- Final generation: {final_index}",
        "- All mistag/score values in percent (lower is better); status marks the generation's winner and/or the persisted anchor member.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in final_rows:
        status_bits = []
        if row.get("trial") == winner_name:
            status_bits.append("winner")
        if row.get("trial") == anchor_member:
            status_bits.append("anchor")
        cells = [
            row.get("trial", "n/a"),
            *(_fmt(row.get(point["column"]), 4) for point in FIXED_WORKING_POINTS),
            _fmt(row.get(CTAG_SCORE_COLUMN), 4),
            _fmt(row.get(BTAG_SCORE_COLUMN), 4),
            _fmt(row.get(TOTAL_SCORE_COLUMN), 4),
            _fmt(row.get("LR"), 3),
            ", ".join(status_bits) if status_bits else "-",
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")

    warnings = sorted({row[GROUP_SCORE_WARNING_COLUMN] for row in final_rows if row.get(GROUP_SCORE_WARNING_COLUMN)})
    if warnings:
        lines.extend(["", "**Score data quality warnings** (aggregate score excluded/reconstructed with missing inputs):"])
        lines.extend(f"- {warning}" for warning in warnings)
    return lines


def _pbt_decision_summary_lines(manifest, rows):
    """Generation-by-generation decision summary for anchor_copy_lr_recenter
    runs only -- empty for every other strategy. Consumes
    generation_record["anchor_copy_lr_recenter"] (already computed by the
    planner) and `rows`' already-computed scores; no recomputation."""
    if manifest.get("config", {}).get("pbt", {}).get("strategy") != "anchor_copy_lr_recenter":
        return []
    decisions = sorted(
        (
            (generation["index"], generation["anchor_copy_lr_recenter"])
            for generation in manifest.get("generations", [])
            if (generation.get("anchor_copy_lr_recenter") or {}).get("decision")
            in ("accepted_new_anchor", "reused_previous_anchor", "rewound_to_previous_anchor")
        ),
        key=lambda item: item[0],
    )
    if not decisions:
        return []
    row_lookup = {(row.get("generation"), row.get("trial")): row for row in rows}
    header = [
        "generation", "winner", "winner total_mistag_score", "winner ctag_score", "winner btag_score",
        "winner LR", "previous LR center", "new LR center", "decision", "spread_collapsed",
    ]
    lines = [
        "",
        "## PBT Decision Summary (anchor_copy_lr_recenter)",
        "- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for generation, info in decisions:
        winner = info.get("winner")
        winner_row = row_lookup.get((generation, winner)) or {}
        cells = [
            generation,
            winner or "n/a",
            _fmt(winner_row.get(TOTAL_SCORE_COLUMN), 4),
            _fmt(winner_row.get(CTAG_SCORE_COLUMN), 4),
            _fmt(winner_row.get(BTAG_SCORE_COLUMN), 4),
            _fmt(info.get("winner_lr"), 4),
            _fmt(info.get("previous_lr_center"), 4),
            _fmt(info.get("new_lr_center"), 4),
            info.get("decision", "n/a"),
            "yes" if info.get("spread_collapsed") else "no",
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return lines


def write_report(run_dir, manifest, summary):
    path = Path(run_dir) / REPORT_NAME
    metric = summary["metric"]
    baseline = summary.get("baseline") or {}
    configured_baseline = summary.get("configured_baseline") or {}
    best = summary.get("best") or {}
    final_best = summary.get("final_best") or {}
    improvement = summary.get("best_improvement_vs_baseline")
    evaluation = summary.get("evaluation") or {}
    schedule = summary.get("schedule") or {}
    eval_schedule = schedule.get("evaluation_interval") or {}
    exploit_schedule = schedule.get("exploit_interval") or {}
    provenance = manifest.get("run") or {}
    git = provenance.get("git") or manifest.get("git") or {}
    plots = summary.get("plots") or {}
    rows = read_metrics_rows(run_dir)

    lines = [
        f"# {summary.get('experiment')}",
        "",
        "## Results",
        f"- Evaluation type: `{evaluation.get('evaluation_type', 'n/a')}`",
        f"- Validation dataset: `{evaluation.get('validation_dataset', 'n/a')}`",
        f"- Validation suffix: `{evaluation.get('validation_suffix', 'n/a')}`",
        f"- Validation sample count: {_fmt(evaluation.get('validation_sample_count'))}",
        "- Controller objective: mean predefined fixed-WP mistag percent (lower is better; not a HEP metric)",
        f"- Configured PBT selection metric: `{metric['name']}` ({metric['mode']})",
    ]
    if metric["name"] == TOTAL_GEOMEAN_METRIC_KEY:
        lines.append(
            "- **`total_mistag_score` (sqrt(ctag_score * btag_score)) is this run's PBT ranking metric** -- "
            "ctag_score/btag_score are its two components, shown for diagnosis, never used for ranking on their own."
        )
    lines.extend(
        [
            f"- Measured baseline: {_fmt(baseline.get('metric_value'))}",
            f"- Configured reference: {_fmt(configured_baseline.get('metric_value'))}",
            f"- Final checkpoint controller objective: {_fmt(final_best.get(CONTROLLER_OBJECTIVE_COLUMN))} by `{final_best.get('trial', 'n/a')}`",
            f"- Global best configured metric: {_fmt(best.get('metric_value'))} by `{best.get('member', 'n/a')}`",
            f"- Delta vs measured baseline: {_fmt(None if improvement is None else 100.0 * improvement)}%",
            f"- Best checkpoint: `{(summary.get('checkpoints') or {}).get('global_best_state')}`",
        ]
    )

    lines.extend(_final_physics_performance_section_lines(manifest, plots))
    lines.extend(_pbt_population_selection_section_lines(manifest, plots))
    lines.extend(_mistag_score_evolution_section_lines(manifest, plots))
    lines.extend(_learning_rate_lineage_section_lines(manifest, plots))
    lines.extend(_learning_rate_mistag_correlation_section_lines(manifest, plots, rows))
    lines.extend(_proxy_validation_section_lines(manifest, plots))

    lines.extend(_model_selection_score_table_lines(manifest, rows))
    lines.extend(_pbt_decision_summary_lines(manifest, rows))

    lines.extend(["", "## Exploit History", f"- [Exploit table]({plots.get('exploit_table_csv', EXPLOIT_TABLE_NAME)})"])
    exploits = [event for event in summary.get("exploit_history", []) if event.get("event_type") == "exploit"]
    if exploits:
        for event in exploits:
            lines.append(
                "- generation {generation}: `{donor}` -> `{recipient}`, donor metric {donor_metric}, recipient metric {recipient_metric}, LR {old_lr} -> {new_lr}, mutation `{mutation}`, weight `{weight_source}`, optimizer `{optimizer_source}`".format(
                    generation=event.get("generation"),
                    donor=event.get("donor"),
                    recipient=event.get("recipient"),
                    donor_metric=_fmt(event.get("donor_metric")),
                    recipient_metric=_fmt(event.get("recipient_metric")),
                    old_lr=_fmt(event.get("old_lr"), 3),
                    new_lr=_fmt(event.get("new_lr"), 3),
                    mutation=event.get("mutation"),
                    weight_source=event.get("weight_source"),
                    optimizer_source=event.get("optimizer_source"),
                )
            )
    else:
        lines.append("- No exploit events recorded.")
    skipped = [event for event in read_events(run_dir) if event.get("event_type") == "exploit_skipped"]
    lines.append(
        f"- [Skipped exploits (significance gating)]({plots.get('skipped_exploit_table_csv', SKIPPED_EXPLOIT_TABLE_NAME)}) -- {len(skipped)} donor->recipient replacement(s) declined for insufficient significance"
    )

    pbt_config = manifest.get("config", {}).get("pbt", {})
    significance_sigma = pbt_config.get("exploit_significance_sigma")
    burn_in = pbt_config.get("burn_in_generations", 0)
    tiered_config = pbt_config.get("tiered_validation") or {}
    lines.extend(
        [
            "",
            "## Method",
            f"- Method: `{summary.get('method')}`",
            f"- Population: {len(summary.get('population') or [])} trials",
            f"- Training interval: {schedule.get('training_interval', {}).get('samples_per_trial_chunk', 'n/a')} samples/trial chunk ({schedule.get('training_interval', {}).get('weaver_epochs_per_generation', 'n/a')}x samples_per_epoch)",
            f"- Evaluation interval: every {eval_schedule.get('training_chunks', 'n/a')} training chunk(s), {eval_schedule.get('samples_per_epoch_val', 'n/a')} validation samples",
            f"- Exploit interval: {('disabled' if not exploit_schedule.get('enabled') else 'every ' + str(exploit_schedule.get('training_chunks', 'n/a')) + ' training chunk(s)')}",
            f"- Exploit significance gating: {'disabled (nominal rank order only)' if significance_sigma is None else f'{significance_sigma} sigma (combined uncertainty) required before a donor replaces a recipient'}",
            f"- Burn-in: {burn_in} generation(s) (observe-only, no exploit/controller LR action applied)",
            f"- Monitor-tier cadence: {tiered_config.get('monitor_interval_generations') or 'disabled'} generation(s), all population members, read-only",
            f"- Full-tier cadence: {tiered_config.get('full_interval_generations') or 'disabled'} generation(s), all population members, read-only",
            "",
            "## Provenance",
            f"- Starting checkpoint: `{(summary.get('starting_checkpoint') or {}).get('state_path') or (summary.get('starting_checkpoint') or {}).get('path')}`",
            f"- Git commit: `{git.get('commit')}`",
            f"- Git dirty: `{git.get('dirty')}`",
            f"- Launch command: `{provenance.get('command') or manifest.get('command')}`",
            "- [manifest.json](manifest.json)",
            "- [resolved_config.yaml](resolved_config.yaml)",
            "- [events.jsonl](events.jsonl)",
            "- [metrics.csv](metrics.csv)",
            "- [tiered_metrics.csv](tiered_metrics.csv)",
            "- [summary.json](summary.json)",
            "",
            "## Caveats",
            "- Proxy, smoke, and full validation results are reported as distinct evaluation types and should not be mixed in one scorecard.",
            "- Configured reference values are not treated as measured baselines unless a successful runtime initial evaluation exists.",
            "- Control-tier evidence alone is 'provisional' -- see Proxy Validation above. It is never a substitute for monitor/full corroboration.",
            f"- {_shutdown_warning_summary(manifest)}",
        ]
    )
    atomic_text(path, "\n".join(lines) + "\n")
    return path

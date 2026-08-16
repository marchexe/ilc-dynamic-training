#!/usr/bin/env python3
"""Compare the LR-vs-mistag-score correlation (statistics.py::
lr_mistag_correlation) across multiple PBT runs.

Each single-run report.md already states this correlation with a
generation-block bootstrap CI (markdown_report.py::
_learning_rate_mistag_correlation_section_lines), but a single run can't
answer questions phrased as "does a longer run / bigger validation set /
bigger per-generation training budget make this trend clearer" -- that
needs several runs' correlations laid out side by side, e.g. the
48-generation showcase run against reruns that vary generations,
weaver_epochs_per_generation, or proxy_validation.control_rows_per_class
one at a time. This module builds that side-by-side view; it never
re-derives the correlation itself (see statistics.py for that).
"""

import argparse
import json
import sys
from pathlib import Path

# Running this file directly (`python .../reporting/compare_runs.py`) makes
# Python auto-insert its own directory (reporting/) at sys.path[0] -- which
# shadows the stdlib `statistics` module with this package's own
# statistics.py (imported below, transitively, via metrics_rows). Must be
# fixed before any training.pbt.reporting import, not in a __main__ guard.
_REPORTING_DIR = Path(__file__).resolve().parent
if sys.path and sys.path[0] == str(_REPORTING_DIR):
    sys.path.pop(0)
_SCRIPTS_DIR = _REPORTING_DIR.parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from training.pbt.reporting.metrics_rows import read_metrics_rows  # noqa: E402
from training.pbt.reporting.statistics import lr_mistag_correlation  # noqa: E402


def _run_config_summary(manifest):
    shared = (manifest.get("config") or {}).get("shared") or {}
    proxy = shared.get("proxy_validation") or {}
    return {
        "generations": shared.get("generations"),
        "weaver_epochs_per_generation": shared.get("weaver_epochs_per_generation"),
        "control_rows_per_class": proxy.get("control_rows_per_class"),
    }


def load_run_correlation(run_path, label=None, n_boot=1000, bootstrap_seed=0):
    """One comparison row for the run at `run_path` (a run directory or a
    manifest.json path directly): label, the three config knobs the
    supervisor's longer-run / bigger-validation / bigger-per-generation
    axes vary, and the population-wide LR-vs-mistag-score correlation with
    its generation-block-bootstrap CI (statistics.py::lr_mistag_correlation).
    Reads metrics.csv via read_metrics_rows -- the same already-persisted
    rows markdown_report.py's own correlation section reads -- not the raw
    manifest, so this never re-derives per-row metric extraction
    independently from the single-run report path.
    """
    run_path = Path(run_path)
    manifest_path = run_path / "manifest.json" if run_path.is_dir() else run_path
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_metrics_rows(run_dir)
    correlation = lr_mistag_correlation(rows, n_boot=n_boot, bootstrap_seed=bootstrap_seed)
    row = {"label": label or manifest.get("experiment", run_dir.name)}
    row.update(_run_config_summary(manifest))
    row.update(
        {
            "n": correlation["n"],
            "pearson_r": correlation["pearson_r"],
            "pearson_r_ci": correlation["pearson_r_ci"],
            "spearman_rho": correlation["spearman_rho"],
            "spearman_rho_ci": correlation["spearman_rho_ci"],
            "reason": correlation["reason"],
        }
    )
    return row


def _format_ci(value, ci):
    if value is None:
        return "n/a"
    if ci is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def render_comparison_table(comparison_rows):
    """Markdown table, one row per run -- direct side-by-side read of a
    multi-run comparison plan (e.g. a 48-generation baseline against
    reruns that vary generation count, weaver_epochs_per_generation, or
    validation size one at a time) instead of cross-referencing several
    separate report.md files by hand."""
    header = "| Run | Generations | weaver_epochs/gen | Val rows/class | n | Pearson r [95% CI] | Spearman rho [95% CI] |"
    separator = "|---|---|---|---|---|---|---|"
    lines = [header, separator]
    for row in comparison_rows:
        if row["reason"]:
            pearson_text = spearman_text = row["reason"]
        else:
            pearson_text = _format_ci(row["pearson_r"], row["pearson_r_ci"])
            spearman_text = _format_ci(row["spearman_rho"], row["spearman_rho_ci"])
        lines.append(
            "| {label} | {generations} | {weaver_epochs_per_generation} | {control_rows_per_class} | {n} | "
            "{pearson} | {spearman} |".format(
                label=row["label"],
                generations=row.get("generations", "n/a"),
                weaver_epochs_per_generation=row.get("weaver_epochs_per_generation", "n/a"),
                control_rows_per_class=row.get("control_rows_per_class", "n/a"),
                n=row["n"],
                pearson=pearson_text,
                spearman=spearman_text,
            )
        )
    return "\n".join(lines)


def plot_run_comparison(comparison_rows, output_path):
    """Forest plot: one horizontal row per run, Pearson r as a point with
    its 95% CI as a whisker, a vertical zero-line for reference. The
    visual this comparison is actually for -- "did the CI get tighter
    and/or move away from zero" is a shape a reader sees at a glance here,
    not something read off a table of numbers. Runs with no CI (too few
    generations to block-bootstrap, or no point estimate at all) get a
    single point / an annotated gap rather than a fabricated whisker.
    Returns the output path, or None if no run has a point estimate to
    plot.
    """
    from training.pbt.reporting.constants import CB_PALETTE
    from training.pbt.reporting.style import plot_setup

    plottable = [row for row in comparison_rows if row["reason"] is None and row["pearson_r"] is not None]
    if not plottable:
        return None

    plt = plot_setup()
    fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(comparison_rows) + 1.2), constrained_layout=True)
    ax.axvline(0.0, color=CB_PALETTE["grey"], linewidth=1.0, linestyle="--", zorder=1)

    y_positions = list(range(len(comparison_rows)))[::-1]
    labels = []
    for y, row in zip(y_positions, comparison_rows):
        labels.append(row["label"])
        if row["reason"] is not None or row["pearson_r"] is None:
            ax.annotate(row["reason"] or "n/a", (0.0, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=CB_PALETTE["grey"])
            continue
        r = row["pearson_r"]
        ci = row["pearson_r_ci"]
        if ci is not None:
            ax.plot([ci[0], ci[1]], [y, y], color=CB_PALETTE["blue"], linewidth=1.6, zorder=2)
        ax.scatter([r], [y], color=CB_PALETTE["blue"], s=45, zorder=3, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson r (log10 LR vs. detrended total_mistag_score)")
    ax.set_title("LR-vs-mistag-score correlation across runs")

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the LR-vs-mistag-score correlation across multiple PBT runs."
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or manifest.json paths, in display order")
    parser.add_argument("--label", action="append", default=[], help="Label for the run at the same position (repeatable); defaults to the run's experiment name")
    parser.add_argument("--plot", type=Path, default=None, help="Write a forest-plot PNG comparing Pearson r + CI across runs to this path")
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = args.label + [None] * (len(args.runs) - len(args.label))
    comparison_rows = [
        load_run_correlation(run, label=label, n_boot=args.n_boot) for run, label in zip(args.runs, labels)
    ]
    print(render_comparison_table(comparison_rows))
    if args.plot:
        written = plot_run_comparison(comparison_rows, args.plot)
        if written:
            print(f"\nWrote {written}")
        else:
            print("\nNo run had a computable correlation to plot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

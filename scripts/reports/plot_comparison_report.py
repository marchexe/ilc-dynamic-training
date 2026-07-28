#!/usr/bin/env python3
"""Compare fixed working-point mistag metrics across PBT runs."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reports.plot_physics_performance import (  # noqa: E402
    REFERENCE_WORKING_POINTS,
    TAG_BACKGROUNDS,
    load_manifest,
    mistag_percent,
    physics_mistag_score,
    sample_summary,
    worker_for_report,
)


POINTS = (
    ("c", 0.50, "b"),
    ("c", 0.50, "d"),
    ("c", 0.80, "b"),
    ("c", 0.80, "d"),
    ("b", 0.80, "c"),
    ("b", 0.80, "d"),
    ("b", 0.90, "c"),
    ("b", 0.90, "d"),
)
COLORS = {
    "current": "#255f9f",
    "final": "#7aa6d9",
    "previous": "#6b7280",
    "other": "#9aa7b8",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a compact fixed-WP comparison report.")
    parser.add_argument("manifest", type=Path, help="Current PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Extra comparison item as label=path[:mode]. Mode defaults to best_physics.",
    )
    return parser.parse_args()


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "report" / "comparison_report.png"


def default_csv(output):
    return Path(output).with_suffix(".csv")


def repo_root():
    return Path(__file__).resolve().parents[2]


def parse_compare(value):
    if "=" in value:
        label, rest = value.split("=", 1)
    else:
        path = Path(value)
        label, rest = (path.parent.name if path.name == "manifest.json" else path.name), value
    if ":" in rest:
        path, mode = rest.rsplit(":", 1)
    else:
        path, mode = rest, "best_physics"
    return label, Path(path), mode


def default_items(current_manifest_path):
    items = []
    previous = repo_root() / "runs" / "pbt" / "ray_anchored_lr_sweep_parquet" / "manifest.json"
    if previous.exists() and previous.resolve() != current_manifest_path.resolve():
        items.append(("previous ray sweep", previous, "best_physics", "previous"))
    items.append(("this run best", current_manifest_path, "best_physics", "current"))
    items.append(("this run final", current_manifest_path, "best_final", "final"))
    return items


def metric_row(label, path, mode, role="other"):
    manifest, resolved_path = load_manifest(path)
    worker, generation, member, score = worker_for_report(manifest, mode)
    metrics = worker.get("metrics") or {}
    if score is None:
        score = physics_mistag_score(metrics)
    values = {
        f"{tag}{int(round(eff * 100))}_{background}": mistag_percent(metrics, tag, eff, background)
        for tag, eff, background in POINTS
    }
    return {
        "label": label,
        "role": role,
        "experiment": manifest.get("experiment", resolved_path.parent.name),
        "source": sample_summary(manifest),
        "mode": mode,
        "generation": generation["index"],
        "member": member.replace("member_", "m"),
        "avg_mistag_percent": score,
        **values,
    }


def collect_rows(current_manifest_path, extra_items=None):
    current_manifest_path = Path(current_manifest_path)
    if current_manifest_path.is_dir():
        current_manifest_path = current_manifest_path / "manifest.json"
    items = default_items(current_manifest_path)
    items.extend(extra_items or [])
    rows = []
    seen = set()
    for label, path, mode, role in items:
        key = (Path(path).resolve(), mode, label)
        if key in seen:
            continue
        seen.add(key)
        try:
            rows.append(metric_row(label, path, mode, role))
        except Exception as error:
            rows.append(
                {
                    "label": label,
                    "role": role,
                    "experiment": Path(path).parent.name,
                    "source": f"skipped: {type(error).__name__}: {error}",
                    "mode": mode,
                    "generation": None,
                    "member": "-",
                    "avg_mistag_percent": None,
                }
            )
    return rows


def format_percent(value):
    return "-" if value is None else f"{value:.3f}"


def write_csv_report(path, rows):
    fieldnames = [
        "label",
        "experiment",
        "mode",
        "generation",
        "member",
        "avg_mistag_percent",
        *[f"{tag}{int(round(eff * 100))}_{background}" for tag, eff, background in POINTS],
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return path


def draw_summary_table(ax, rows):
    ax.axis("off")
    col_labels = [
        "model",
        "checkpoint",
        "avg [%]",
        "c50 b",
        "c50 d",
        "c80 b",
        "c80 d",
        "b80 c",
        "b80 d",
        "b90 c",
        "b90 d",
    ]
    cell_text = []
    for row in rows:
        checkpoint = "-" if row["generation"] is None else f"{row['member']} / gen {row['generation']}"
        cell_text.append(
            [
                row["label"],
                checkpoint,
                format_percent(row.get("avg_mistag_percent")),
                *[format_percent(row.get(f"{tag}{int(round(eff * 100))}_{background}")) for tag, eff, background in POINTS],
            ]
        )
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.17, 0.13, 0.08, *([0.0775] * 8)],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.42)
    for (row_index, _col_index), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#255f9f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f3f7fc" if row_index % 2 else "#e3edf8")


def draw_avg_bars(ax, rows):
    valid_rows = [row for row in rows if row.get("avg_mistag_percent") is not None]
    xs = list(range(len(valid_rows)))
    values = [row["avg_mistag_percent"] for row in valid_rows]
    colors = [COLORS.get(row.get("role"), COLORS["other"]) for row in valid_rows]
    bars = ax.bar(xs, values, color=colors, width=0.58)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([row["label"] for row in valid_rows], rotation=0)
    ax.set_ylabel("average mistag [%]")
    ax.set_title("Fixed-WP selection metric", loc="left", fontsize=11, fontweight="bold")
    ymax = max(values or [1.0])
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    ax.grid(axis="y", color="0.88", linewidth=0.6)


def draw_point_panel(ax, rows, tag):
    valid_rows = [row for row in rows if row.get("avg_mistag_percent") is not None]
    backgrounds = TAG_BACKGROUNDS[tag]
    efficiencies = REFERENCE_WORKING_POINTS[tag]
    point_labels = [f"{int(round(eff * 100))}% {background}" for eff in efficiencies for background in backgrounds]
    keys = [f"{tag}{int(round(eff * 100))}_{background}" for eff in efficiencies for background in backgrounds]
    x_base = list(range(len(point_labels)))
    width = min(0.24, 0.70 / max(len(valid_rows), 1))
    for row_index, row in enumerate(valid_rows):
        offset = (row_index - (len(valid_rows) - 1) / 2) * width
        values = [row.get(key) for key in keys]
        ax.bar(
            [x + offset for x in x_base],
            [0.0 if value is None else value for value in values],
            width=width,
            label=row["label"],
            color=COLORS.get(row.get("role"), COLORS["other"]),
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(point_labels)
    ax.set_ylabel("mistag [%]")
    ax.set_title(f"{tag}-tag target working points", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_rows(rows, output):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(12.2, 7.1), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.88,
        hspace=0.50,
        wspace=0.22,
        height_ratios=[0.72, 1.0, 1.0],
    )
    draw_summary_table(fig.add_subplot(grid[0, :]), rows)
    draw_avg_bars(fig.add_subplot(grid[1, 0]), rows)
    draw_point_panel(fig.add_subplot(grid[1, 1]), rows, "c")
    draw_point_panel(fig.add_subplot(grid[2, :]), rows, "b")
    fig.suptitle("Fixed working-point comparison", x=0.06, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.06,
        0.935,
        "Mistag is shown in percent at HEP-style reference operating points; lower is better.",
        ha="left",
        va="top",
        fontsize=9.5,
        color="0.35",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_manifest(manifest_path, output=None, csv_path=None, compare_items=None):
    manifest, resolved_manifest_path = load_manifest(manifest_path)
    del manifest
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    csv_path = Path(csv_path) if csv_path is not None else default_csv(output)
    rows = collect_rows(resolved_manifest_path, compare_items)
    write_csv_report(csv_path, rows)
    return plot_rows(rows, output), csv_path


def main():
    args = parse_args()
    compare_items = [(*parse_compare(value), "other") for value in args.compare]
    output, csv_path = plot_manifest(args.manifest, args.output, args.csv, compare_items)
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot background rejection curves for the physics-selected checkpoint."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reports.plot_physics_performance import (
    PAIR_COLORS,
    PAIR_LABELS,
    REFERENCE_WORKING_POINTS,
    TAG_PAIRS,
    load_manifest,
    log_tick_label,
    sample_summary,
    worker_for_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot diagnostic background rejection curves.")
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--member", default="best_physics", help="best_physics, best_final, global_best, or a member name")
    return parser.parse_args()


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "diagnostics" / "background_rejection_curves.png"


def draw_rejection(ax, metrics, tag):
    curves = metrics.get("validation_bkg_rejection_at_eff") or {}
    efficiencies = [float(value) for value in curves.get("efficiencies") or []]
    pairs = curves.get("pairs") or {}
    if not efficiencies or not all(pair in pairs for pair in TAG_PAIRS[tag]):
        ax.text(0.5, 0.5, "no rejection curves", ha="center", va="center")
        ax.axis("off")
        return

    for pair in TAG_PAIRS[tag]:
        ax.plot(
            efficiencies,
            [float(value) for value in pairs[pair]],
            marker="o",
            markersize=4.2,
            linewidth=2.0,
            color=PAIR_COLORS[pair],
            label=PAIR_LABELS[pair],
        )
    for working_point in REFERENCE_WORKING_POINTS[tag]:
        ax.axvline(working_point, color="0.70", linestyle=":", linewidth=1.0)
    ax.set_title(f"{tag}-tag", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{tag}-tag efficiency")
    ax.set_ylabel("background rejection")
    ax.set_yscale("log")
    ax.set_xlim(0.18, 1.02)
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker").ticker.FuncFormatter(log_tick_label))
    ax.grid(axis="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")


def plot_manifest(manifest_path, output=None, member="best_physics"):
    import matplotlib.pyplot as plt

    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    worker, generation, member_name, physics_score = worker_for_report(manifest, member)
    metrics = worker.get("metrics") or {}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.78, wspace=0.18)
    draw_rejection(axes[0], metrics, "c")
    draw_rejection(axes[1], metrics, "b")
    score_text = "" if physics_score is None else f" | avg fixed-WP mistag {physics_score:.3f}%"
    fig.suptitle(
        f"Background rejection curves: {member_name}, generation {generation['index']}{score_text}",
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.905,
        f"{manifest.get('experiment', resolved_manifest_path.parent.name)} | {sample_summary(manifest)}",
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    print(plot_manifest(args.manifest, args.output, args.member))


if __name__ == "__main__":
    main()

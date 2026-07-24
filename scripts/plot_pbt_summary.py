#!/usr/bin/env python3
"""Create a compact summary plot from a PBT manifest."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


PAIR_METRICS = [
    ("bc", "validation_bkg_rejection_bc_score"),
    ("bd", "validation_bkg_rejection_bd_score"),
    ("cb", "validation_bkg_rejection_cb_score"),
    ("cd", "validation_bkg_rejection_cd_score"),
]
SUMMARY_METRICS = [
    ("b tag", "validation_b_tag_rejection_score"),
    ("c tag", "validation_c_tag_rejection_score"),
    ("all", "validation_bkg_rejection_score"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def completed_generations(manifest):
    generations = []
    for generation in manifest.get("generations", []):
        if generation.get("status") == "completed":
            generations.append(generation)
    return generations


def metric_points(generations, member, metric):
    xs = []
    ys = []
    for generation in generations:
        worker = generation.get("workers", {}).get(member, {})
        metrics = worker.get("metrics") or {}
        value = metrics.get(metric)
        if value is not None:
            xs.append(generation["index"])
            ys.append(float(value))
    return xs, ys


def final_generation(generations):
    for generation in reversed(generations):
        workers = generation.get("workers", {})
        if any((record.get("metrics") or {}).get("validation_bkg_rejection_score") is not None
               for record in workers.values()):
            return generation
    return None


def annotate_last(ax, xs, ys, label):
    if xs and ys:
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )


def compact_member_name(name):
    return name.replace("member_", "m")


def plot_manifest(manifest_path, output=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(output) if output is not None else manifest_path.with_name("summary.png")
    generations = completed_generations(manifest)
    if not generations:
        raise RuntimeError("manifest has no completed generations to plot")

    members = list(manifest.get("members", {}).keys())
    metric = manifest.get("config", {}).get("pbt", {}).get(
        "metric", "validation_bkg_rejection_score"
    )
    experiment = manifest.get("experiment", manifest_path.parent.name)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), gridspec_kw={"width_ratios": [1.5, 1.3, 1]})
    ax_score, ax_pairs, ax_summary = axes
    colors = plt.get_cmap("tab10")

    for index, member in enumerate(members):
        xs, ys = metric_points(generations, member, metric)
        if not xs:
            continue
        short_name = compact_member_name(member)
        ax_score.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=colors(index),
            label=short_name,
        )
        annotate_last(ax_score, xs, ys, short_name)
    ax_score.set_title("(a) PBT ranking metric", loc="left")
    ax_score.set_xlabel("generation")
    ax_score.set_ylabel(metric.replace("validation_", ""))
    ax_score.grid(axis="both", color="0.88", linewidth=0.6)
    ax_score.set_xticks([generation["index"] for generation in generations])

    generation = final_generation(generations)
    if generation is None:
        raise RuntimeError("manifest has no final metrics to plot")

    width = 0.8 / max(1, len(members))
    base_x = list(range(len(PAIR_METRICS)))
    for member_index, member in enumerate(members):
        worker = generation.get("workers", {}).get(member, {})
        metrics = worker.get("metrics") or {}
        offset = (member_index - (len(members) - 1) / 2) * width
        values = [metrics.get(metric_name) for _, metric_name in PAIR_METRICS]
        if any(value is not None for value in values):
            ax_pairs.bar(
                [x + offset for x in base_x],
                [0.0 if value is None else float(value) for value in values],
                width=width,
                color=colors(member_index),
                label=compact_member_name(member),
            )
    ax_pairs.set_title(f"(b) Final pair scores, gen {generation['index']}", loc="left")
    ax_pairs.set_xticks(base_x, [label for label, _ in PAIR_METRICS])
    ax_pairs.set_xlabel("tag/background")
    ax_pairs.set_ylabel("mean log(BGrej)")
    ax_pairs.grid(axis="y", color="0.88", linewidth=0.6)
    ax_pairs.legend(frameon=False, ncols=2)

    rows = []
    for member in members:
        worker = generation.get("workers", {}).get(member, {})
        metrics = worker.get("metrics") or {}
        if metrics:
            rows.append(
                [
                    compact_member_name(member),
                    *(f"{float(metrics[key]):.3f}" if metrics.get(key) is not None else "-"
                      for _, key in SUMMARY_METRICS),
                    f"{float(metrics['validation_accuracy']):.4f}"
                    if metrics.get("validation_accuracy") is not None else "-",
                ]
            )
    ax_summary.axis("off")
    ax_summary.set_title("(c) Final validation", loc="left")
    if rows:
        table = ax_summary.table(
            cellText=rows,
            colLabels=["id", "b tag", "c tag", "all", "acc"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)

    fig.suptitle(experiment, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    output = plot_manifest(args.manifest, args.output)
    print(output)


if __name__ == "__main__":
    main()

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
OBJECTIVE_LABELS = {
    "validation_bkg_rejection_score": "PBT objective: mean ln(BGrej), all pairs",
    "validation_b_tag_rejection_score": "PBT objective: mean ln(BGrej), b-tag pairs",
    "validation_c_tag_rejection_score": "PBT objective: mean ln(BGrej), c-tag pairs",
}


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


def rejection_lookup(metrics, tag, eff, background):
    lookup = metrics.get("validation_bkg_rejection_at_eff_lookup") or {}
    row = lookup.get(f"{tag}_tag_eff_{eff:.2f}") or {}
    return row.get(f"{background}_bkg_rejection")


def format_rejection(value):
    return f"{float(value):.1f}" if value is not None else "-"


def objective_label(metric):
    return OBJECTIVE_LABELS.get(metric, metric.replace("validation_", ""))


def plot_manifest(manifest_path, output=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(output) if output is not None else manifest_path.parent / "plots" / "pbt_objective_diagnostics.png"
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
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax_score = plt.subplots(figsize=(8.8, 4.8))
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
            linewidth=1.5,
            markersize=4.5,
            color=colors(index),
            label=short_name,
        )
        annotate_last(ax_score, xs, ys, short_name)

    best = manifest.get("best") or {}
    if best:
        ax_score.axvline(
            best["generation"],
            color="0.20",
            linestyle="--",
            linewidth=1.0,
        )
        ax_score.scatter(
            [best["generation"]],
            [best["metric_value"]],
            marker="*",
            s=115,
            color="0.05",
            zorder=5,
            label="global best",
        )
        ax_score.annotate(
            f"global best\ngen {best['generation']} / {compact_member_name(best['member'])}",
            (best["generation"], best["metric_value"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax_score.set_title("PBT objective by generation", loc="left")
    ax_score.set_xlabel("generation")
    ax_score.set_ylabel(objective_label(metric))
    ax_score.grid(axis="both", color="0.88", linewidth=0.6)
    ax_score.set_xticks([generation["index"] for generation in generations])
    ax_score.legend(frameon=False, ncols=3)

    fig.suptitle(experiment, x=0.02, ha="left", fontsize=13, fontweight="bold")
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

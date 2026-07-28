#!/usr/bin/env python3
"""Create compact training diagnostics from a PBT manifest."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SHOWCASE_WORKING_POINTS = (
    ("b", 0.80, "c"),
    ("b", 0.80, "d"),
    ("b", 0.90, "c"),
    ("b", 0.90, "d"),
    ("c", 0.50, "b"),
    ("c", 0.50, "d"),
    ("c", 0.80, "b"),
    ("c", 0.80, "d"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create compact PBT training diagnostics.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def completed_generations(manifest):
    return [
        generation
        for generation in manifest.get("generations", [])
        if generation.get("status") == "completed"
    ]


def rejection_lookup(metrics, tag, eff, background):
    lookup = metrics.get("validation_bkg_rejection_at_eff_lookup") or {}
    row = lookup.get(f"{tag}_tag_eff_{eff:.2f}") or {}
    value = row.get(f"{background}_bkg_rejection")
    if value is not None:
        return float(value)

    curves = metrics.get("validation_bkg_rejection_at_eff") or {}
    efficiencies = curves.get("efficiencies") or []
    pairs = curves.get("pairs") or {}
    if eff not in efficiencies:
        return None
    index = efficiencies.index(eff)
    pair = f"{tag}{background}"
    values = pairs.get(pair) or []
    return float(values[index]) if index < len(values) else None


def mistag_percent(metrics, tag, eff, background):
    rejection = rejection_lookup(metrics, tag, eff, background)
    if rejection is None or rejection <= 0 or not math.isfinite(rejection):
        return None
    return 100.0 / rejection


def fixed_wp_mistag_score(metrics):
    values = []
    for tag, eff, background in SHOWCASE_WORKING_POINTS:
        value = mistag_percent(metrics, tag, eff, background)
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def worker_lr(worker):
    if worker.get("lr") is not None:
        return float(worker["lr"])
    command = worker.get("command") or []
    if "--start-lr" in command:
        index = command.index("--start-lr")
        if index + 1 < len(command):
            return float(command[index + 1])
    return None


def compact_member_name(name):
    return name.replace("member_", "m")


def best_physics_for_generation(generation):
    candidates = []
    for member, worker in generation.get("workers", {}).items():
        score = fixed_wp_mistag_score(worker.get("metrics") or {})
        if score is not None:
            candidates.append((score, member, worker))
    if not candidates:
        return None
    score, member, worker = min(candidates, key=lambda item: item[0])
    return {
        "generation": generation["index"],
        "member": member,
        "worker": worker,
        "mistag_score": score,
        "lr": worker_lr(worker),
    }


def best_physics_overall(rows):
    if not rows:
        return None
    return min(rows, key=lambda row: row["mistag_score"])


def training_logic_points(best_rows):
    if not best_rows:
        return None, None, None
    start = best_rows[0]
    best = best_physics_overall(best_rows)
    final = best_rows[-1]
    return start, best, final


def drift_percent(from_row, to_row):
    if from_row is None or to_row is None:
        return None
    return to_row["mistag_score"] - from_row["mistag_score"]


def exact_uncertainty_available(row):
    if row is None:
        return False
    metrics = row.get("worker", {}).get("metrics") or {}
    return bool(metrics.get("validation_bkg_rejection_at_eff_counts"))


def member_lr_rows(manifest, generations):
    rows = []
    for generation in generations:
        for member, worker in sorted(generation.get("workers", {}).items()):
            lr = worker_lr(worker)
            if lr is not None:
                rows.append({"generation": generation["index"], "member": member, "lr": lr})
    return rows


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "report" / "training_diagnostics.png"


def plot_manifest(manifest_path, output=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(output) if output is not None else default_output(manifest_path)
    generations = completed_generations(manifest)
    if not generations:
        raise RuntimeError("manifest has no completed generations to plot")

    best_rows = [row for generation in generations for row in [best_physics_for_generation(generation)] if row]
    start, selected, final = training_logic_points(best_rows)
    if selected is None:
        raise RuntimeError("manifest has no fixed working-point mistag metrics to plot")
    start_delta = drift_percent(start, selected)
    final_drift = drift_percent(selected, final)
    uncertainty_note = "exact stat. uncertainty available" if exact_uncertainty_available(selected) else "exact stat. uncertainty unavailable"

    lr_rows = member_lr_rows(manifest, generations)
    members = sorted({row["member"] for row in lr_rows})
    experiment = manifest.get("experiment", manifest_path.parent.name)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (ax_mistag, ax_lr) = plt.subplots(
        2,
        1,
        figsize=(9.4, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )

    xs = [row["generation"] for row in best_rows]
    ys = [row["mistag_score"] for row in best_rows]
    ax_mistag.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, color="#2f5597")
    marker_specs = [
        (start, "start", "o", "#ffffff", "#2f5597", (-8, -24)),
        (selected, "best", "*", "#111111", "#111111", (10, 12)),
        (final, "final", "s", "#7aa6d9", "#255f9f", (-34, 10)),
    ]
    for row, label, marker, face, edge, offset in marker_specs:
        if row is None:
            continue
        size = 135 if label == "best" else 70
        ax_mistag.scatter(
            [row["generation"]],
            [row["mistag_score"]],
            marker=marker,
            s=size,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
            zorder=5,
        )
        ax_mistag.annotate(
            f"{label}\n{row['mistag_score']:.3f}%",
            (row["generation"], row["mistag_score"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold" if label == "best" else "normal",
        )
    ax_mistag.axvline(selected["generation"], color="0.25", linestyle="--", linewidth=1.0)
    if start_delta is not None and final_drift is not None:
        ax_mistag.text(
            0.995,
            0.06,
            f"start -> best: {start_delta:+.3f}%   best -> final: {final_drift:+.3f}%\n{uncertainty_note}",
            transform=ax_mistag.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color="0.25",
        )
    ax_mistag.set_title("Mistag at target working points", loc="left")
    ax_mistag.set_ylabel("Mistag [%]")
    ax_mistag.grid(axis="both", color="0.88", linewidth=0.6)

    selected_lr = None
    for member in members:
        rows = [row for row in lr_rows if row["member"] == member]
        ax_lr.plot(
            [row["generation"] for row in rows],
            [row["lr"] for row in rows],
            linewidth=1.0,
            color="0.70",
            alpha=0.8,
            zorder=1,
        )
        if member == selected["member"]:
            for row in rows:
                if row["generation"] == selected["generation"]:
                    selected_lr = row["lr"]
                    break
    if selected_lr is not None:
        ax_lr.scatter(
            [selected["generation"]],
            [selected_lr],
            marker="*",
            s=115,
            color="#111111",
            zorder=4,
        )
        ax_lr.annotate(
            f"lr {selected_lr:.3g}",
            (selected["generation"], selected_lr),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_lr.axvline(selected["generation"], color="0.25", linestyle="--", linewidth=1.0)
    ax_lr.set_title("Learning rate", loc="left")
    ax_lr.set_xlabel("Generation")
    ax_lr.set_ylabel("LR")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax_lr.grid(axis="both", color="0.88", linewidth=0.6)
    ax_lr.legend(
        handles=[
            Line2D([0], [0], color="0.70", linewidth=1.0, label="all trials"),
            Line2D([0], [0], color="none", marker="*", markerfacecolor="#111111", markeredgecolor="#111111", label=f"selected checkpoint ({compact_member_name(selected['member'])})", markersize=9),
        ],
        frameon=False,
        loc="best",
    )
    ax_lr.set_xticks([generation["index"] for generation in generations])

    fig.suptitle(
        experiment,
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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

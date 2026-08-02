#!/usr/bin/env python3
"""Plot which PBT member was selected by target-WP mistag in each generation."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reports.plot_pbt_summary import (
    best_physics_for_generation,
    best_physics_overall,
    checkpoint_baseline_row,
    compact_member_name,
    completed_generations,
)
from reports.plot_physics_performance import load_manifest


MEMBER_COLORS = {
    "member_00": "#4c78a8",
    "member_01": "#f58518",
    "member_02": "#54a24b",
    "member_03": "#e45756",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot best-member timeline by target working-point mistag.")
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "diagnostics" / "selection_timeline.png"


def selected_rows(manifest):
    rows = [
        row
        for generation in completed_generations(manifest)
        for row in [best_physics_for_generation(generation)]
        if row
    ]
    selected = best_physics_overall(rows)
    if selected is None:
        raise RuntimeError("manifest has no target working-point mistag metrics")
    return rows, selected


def plot_manifest(manifest_path, output=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    rows, selected = selected_rows(manifest)
    baseline = checkpoint_baseline_row(manifest, resolved_manifest_path)
    members = sorted({row["member"] for row in rows})
    member_y = {member: len(members) - 1 - index for index, member in enumerate(members)}
    xs = [row["generation"] for row in rows]

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
    fig, (ax_member, ax_score) = plt.subplots(
        2,
        1,
        figsize=(9.6, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.82, 1.18]},
    )

    ys = [member_y[row["member"]] for row in rows]
    ax_member.step(xs, ys, where="mid", color="0.78", linewidth=1.2, zorder=1)
    for row in rows:
        is_selected = row["generation"] == selected["generation"] and row["member"] == selected["member"]
        ax_member.scatter(
            row["generation"],
            member_y[row["member"]],
            s=140 if is_selected else 58,
            marker="*" if is_selected else "o",
            color="#111111" if is_selected else MEMBER_COLORS.get(row["member"], "#4c78a8"),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax_member.axvline(selected["generation"], color="0.25", linestyle="--", linewidth=1.0)
    ax_member.set_yticks(range(len(members)))
    ax_member.set_yticklabels([compact_member_name(member) for member in reversed(members)])
    ax_member.set_ylabel("Chosen trial")
    ax_member.set_title("Chosen trial per generation", loc="left")
    ax_member.grid(axis="x", color="0.90", linewidth=0.6)
    ax_member.set_ylim(-0.5, len(members) - 0.35)

    ax_score.plot(
        xs,
        [row["mistag_score"] for row in rows],
        color="0.70",
        linewidth=1.2,
        zorder=1,
    )
    if baseline is not None:
        ax_score.axhline(
            baseline["mistag_score"],
            color="#5f6f82",
            linestyle=":",
            linewidth=1.2,
            zorder=2,
        )
        ax_score.annotate(
            f"checkpoint {baseline['mistag_score']:.3f}%",
            (xs[0], baseline["mistag_score"]),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8.2,
            color="#4c5868",
        )
    for row in rows:
        is_selected = row["generation"] == selected["generation"] and row["member"] == selected["member"]
        ax_score.scatter(
            row["generation"],
            row["mistag_score"],
            s=135 if is_selected else 46,
            marker="*" if is_selected else "o",
            color="#111111" if is_selected else MEMBER_COLORS.get(row["member"], "#4c78a8"),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax_score.axvline(selected["generation"], color="0.25", linestyle="--", linewidth=1.0)
    ax_score.annotate(
        f"best {selected['mistag_score']:.3f}%",
        (selected["generation"], selected["mistag_score"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
    )
    ax_score.set_title("Mistag of chosen trial", loc="left")
    ax_score.set_xlabel("Generation")
    ax_score.set_ylabel("Average mistag [%]")
    ax_score.grid(axis="both", color="0.88", linewidth=0.6)
    ax_score.set_xticks(xs)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MEMBER_COLORS.get(member, "#4c78a8"), markeredgecolor="white", label=compact_member_name(member), markersize=7)
        for member in members
    ]
    if baseline is not None:
        handles.append(Line2D([0], [0], color="#5f6f82", linestyle=":", linewidth=1.2, label="checkpoint mistag"))
    handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111", markeredgecolor="#111111", label="overall best", markersize=10))
    ax_score.legend(handles=handles, frameon=False, ncols=min(5, len(handles)), loc="best")

    fig.suptitle(
        manifest.get("experiment", resolved_manifest_path.parent.name),
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    print(plot_manifest(args.manifest, args.output))


if __name__ == "__main__":
    main()

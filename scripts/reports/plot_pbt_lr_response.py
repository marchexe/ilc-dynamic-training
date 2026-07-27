#!/usr/bin/env python3
"""Plot LR response of PBT workers at fixed b-tag working points."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_B_EFFICIENCIES = (0.8, 0.9)
BACKGROUNDS = ("c", "d")
MEMBER_MARKERS = {
    "member_00": "o",
    "member_01": "s",
    "member_02": "^",
    "member_03": "D",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot learning-rate response using b-tag mistag working points."
    )
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--b-eff",
        default=",".join(str(value) for value in DEFAULT_B_EFFICIENCIES),
        help="Comma-separated b-tag efficiencies, e.g. 0.8,0.9.",
    )
    return parser.parse_args()


def load_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def completed_generations(manifest):
    return [
        generation
        for generation in manifest.get("generations", [])
        if generation.get("status") == "completed"
    ]


def worker_lr(worker):
    if worker.get("lr") is not None:
        return float(worker["lr"])
    command = worker.get("command") or []
    if "--start-lr" in command:
        index = command.index("--start-lr")
        if index + 1 < len(command):
            return float(command[index + 1])
    return None


def rejection_at(metrics, b_eff, background):
    lookup = metrics.get("validation_bkg_rejection_at_eff_lookup") or {}
    row = lookup.get(f"b_tag_eff_{b_eff:.2f}") or {}
    value = row.get(f"{background}_bkg_rejection")
    if value is not None:
        return float(value)

    curves = metrics.get("validation_bkg_rejection_at_eff") or {}
    efficiencies = curves.get("efficiencies") or []
    pairs = curves.get("pairs") or {}
    if b_eff not in efficiencies:
        return None
    index = efficiencies.index(b_eff)
    pair = "bc" if background == "c" else "bd"
    values = pairs.get(pair) or []
    return float(values[index]) if index < len(values) else None


def mistag_percent(metrics, b_eff, background):
    rejection = rejection_at(metrics, b_eff, background)
    if rejection is None or rejection <= 0 or not math.isfinite(rejection):
        return None
    return 100.0 / rejection


def collect_points(manifest, b_efficiencies=DEFAULT_B_EFFICIENCIES):
    best = manifest.get("best") or {}
    points = []
    for generation in completed_generations(manifest):
        for member, worker in sorted(generation.get("workers", {}).items()):
            lr = worker_lr(worker)
            metrics = worker.get("metrics") or {}
            if lr is None or not metrics:
                continue
            point = {
                "generation": generation["index"],
                "epoch": generation.get("epoch"),
                "member": member,
                "lr": lr,
                "is_global_best": generation.get("index") == best.get("generation") and member == best.get("member"),
                "mistag_percent": {},
            }
            for b_eff in b_efficiencies:
                point["mistag_percent"][b_eff] = {
                    background: mistag_percent(metrics, b_eff, background)
                    for background in BACKGROUNDS
                }
            points.append(point)
    return points


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "pbt_lr_response.png"


def compact_member(member):
    return member.replace("member_", "m")


def plot_manifest(manifest_path, output=None, b_efficiencies=DEFAULT_B_EFFICIENCIES):
    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    points = collect_points(manifest, b_efficiencies=b_efficiencies)
    if not points:
        raise RuntimeError("manifest has no worker LR/mistag points to plot")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(len(b_efficiencies), len(BACKGROUNDS), figsize=(10.8, 6.2), sharex=True, constrained_layout=True)
    if len(b_efficiencies) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    generations = sorted({point["generation"] for point in points})
    gen_min, gen_max = min(generations), max(generations)
    denominator = max(1, gen_max - gen_min)

    for row_index, b_eff in enumerate(b_efficiencies):
        for col_index, background in enumerate(BACKGROUNDS):
            ax = axes[row_index][col_index]
            for point in points:
                value = point["mistag_percent"].get(b_eff, {}).get(background)
                if value is None:
                    continue
                color = cmap((point["generation"] - gen_min) / denominator)
                marker = MEMBER_MARKERS.get(point["member"], "o")
                ax.scatter(
                    point["lr"],
                    value,
                    s=88 if point["is_global_best"] else 42,
                    marker="*" if point["is_global_best"] else marker,
                    color="black" if point["is_global_best"] else color,
                    edgecolor="black" if point["is_global_best"] else "none",
                    linewidth=0.8,
                    alpha=1.0 if point["is_global_best"] else 0.78,
                    zorder=5 if point["is_global_best"] else 3,
                )
            ax.set_title(f"{background} mistag / b-eff={b_eff:.2f}", loc="left")
            ax.set_yscale("log")
            ax.grid(axis="both", color="0.88", linewidth=0.6)
            ax.set_ylabel("mistag [%]")
            if row_index == len(b_efficiencies) - 1:
                ax.set_xlabel("learning rate")
    member_handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor="0.35", label=compact_member(member), markersize=6)
        for member, marker in MEMBER_MARKERS.items()
    ]
    best_handle = Line2D([0], [0], marker="*", color="none", markerfacecolor="black", markeredgecolor="black", label="global best", markersize=10)
    axes[0][-1].legend(handles=member_handles + [best_handle], frameon=False, loc="best")
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gen_min, vmax=gen_max))
    colorbar = fig.colorbar(scalar, ax=[axis for row in axes for axis in row], shrink=0.82, pad=0.02)
    colorbar.set_label("generation")
    fig.suptitle(
        f"{manifest.get('experiment', resolved_manifest_path.parent.name)}: LR response at b-tag working points",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    b_efficiencies = tuple(float(value.strip()) for value in args.b_eff.split(",") if value.strip())
    output = plot_manifest(args.manifest, output=args.output, b_efficiencies=b_efficiencies)
    print(output)


if __name__ == "__main__":
    main()

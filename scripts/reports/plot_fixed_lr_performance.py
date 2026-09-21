#!/usr/bin/env python3
"""Plot recorded fixed-LR performance, optionally stitching continuations.

This is deliberately separate from the PBT lineage figures: fixed-LR runs
have no checkpoint-copy branches or LR mutations. The script only reads
completed manifests and never invokes training or evaluation.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reports.export_research_result import load_manifest  # noqa: E402
from reports.plot_physics_performance import FIGURE_SIZE_INCHES, OUTPUT_DPI  # noqa: E402
from training.pbt.reporting.constants import CB_PALETTE  # noqa: E402
from training.pbt.reporting.style import plot_setup  # noqa: E402


COLORS = [
    CB_PALETTE["orange"],
    CB_PALETTE["sky_blue"],
    CB_PALETTE["green"],
    CB_PALETTE["purple"],
    CB_PALETTE["blue"],
    CB_PALETTE["vermillion"],
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+", help="One or more fixed-LR runs to compare")
    parser.add_argument(
        "--prefix-run", type=Path, action="append", default=[],
        help="Earlier segment shared by continuation members; repeat in chronological order",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _completed_rows(manifest):
    rows = sorted(manifest.get("generations") or [], key=lambda row: row.get("epoch", -1))
    if not rows or any(row.get("status") != "completed" for row in rows):
        raise ValueError("Fixed-LR comparison requires completed generation records")
    return rows


def _metric_name(manifest):
    metric = ((manifest.get("config") or {}).get("pbt") or {}).get("metric")
    if not metric:
        raise ValueError("Manifest has no configured PBT metric")
    return metric


def fixed_lr_data(runs, prefix_runs=()):
    primary = [load_manifest(path)[0] for path in runs]
    prefixes = [load_manifest(path)[0] for path in prefix_runs]
    manifests = [*prefixes, *primary]
    if any(manifest.get("method") != "fixed_lr_grid" for manifest in manifests):
        raise ValueError("All inputs must be fixed_lr_grid runs")
    metrics = {_metric_name(manifest) for manifest in manifests}
    if len(metrics) != 1:
        raise ValueError("Fixed-LR runs use different selection metrics")
    metric = metrics.pop()

    # With explicit continuation runs, plot only the members they continue;
    # the prefix may contain additional sweep members that were not extended.
    members = sorted(
        {member for manifest in primary for member in (manifest.get("members") or {})},
        key=lambda name: float(next(
            manifest["members"][name]["lr"] for manifest in primary if name in (manifest.get("members") or {})
        )),
    )
    curves = {}
    lrs = {}
    boundaries = set()
    for member in members:
        values = []
        segment_lengths = []
        member_lr = None
        for manifest in manifests:
            if member not in (manifest.get("members") or {}):
                continue
            rows = _completed_rows(manifest)
            segment = []
            for row in rows:
                worker = (row.get("workers") or {}).get(member) or {}
                value = (worker.get("metrics") or {}).get(metric)
                lr = worker.get("lr")
                if value is None or lr is None or not math.isfinite(float(value)) or not math.isfinite(float(lr)):
                    raise ValueError(f"Missing/non-finite fixed-LR evidence for {member}")
                if member_lr is None:
                    member_lr = float(lr)
                elif not math.isclose(member_lr, float(lr), rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError(f"{member} changed LR inside a fixed-LR comparison")
                segment.append(float(value))
            if segment:
                values.extend(segment)
                segment_lengths.append(len(segment))
        if not values:
            continue
        curves[member] = values
        lrs[member] = member_lr
        offset = 0
        for length in segment_lengths[:-1]:
            offset += length
            boundaries.add(offset)
    if not curves:
        raise ValueError("No fixed-LR curves found")
    lengths = {len(values) for values in curves.values()}
    if len(lengths) != 1:
        raise ValueError("Compared fixed-LR branches have different recorded horizons")
    horizon = lengths.pop()
    if horizon < 10:
        raise ValueError("At least ten recorded epochs are required")
    winner = min(curves, key=lambda member: mean(curves[member][-10:]))
    best_member, best_index = min(
        ((member, index) for member, values in curves.items() for index in range(horizon)),
        key=lambda item: curves[item[0]][item[1]],
    )
    return {
        "metric": metric,
        "epochs": list(range(1, horizon + 1)),
        "curves": curves,
        "lrs": lrs,
        "boundaries": sorted(boundaries),
        "winner": winner,
        "best_member": best_member,
        "best_epoch": best_index + 1,
        "best_value": curves[best_member][best_index],
        "final10": {member: mean(values[-10:]) for member, values in curves.items()},
    }


def draw(plt, data):
    fig = plt.figure(figsize=FIGURE_SIZE_INCHES)
    ax = fig.add_axes([0.08, 0.20, 0.88, 0.61])
    members = sorted(data["curves"], key=lambda member: data["lrs"][member])
    colors = {member: COLORS[index % len(COLORS)] for index, member in enumerate(members)}
    for member in members:
        highlighted = member == data["winner"]
        ax.plot(
            data["epochs"], data["curves"][member],
            color=colors[member], linewidth=3.0 if highlighted else 1.7,
            alpha=1.0 if highlighted else 0.72,
            label=f"{data['lrs'][member] * 1e6:g}e-6",
            zorder=4 if highlighted else 2,
        )
    for boundary in data["boundaries"]:
        ax.axvline(boundary + 0.5, color="#75838C", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
        ax.text(
            boundary + 0.5, 0.02, "continuation",
            transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="bottom",
            fontsize=8.5, color="#75838C",
        )
    ax.scatter(
        data["best_epoch"], data["best_value"], marker="*", s=180,
        color=colors[data["best_member"]], edgecolors="white", linewidths=1.0,
        zorder=7, clip_on=False,
    )
    ranking = sorted(data["final10"].items(), key=lambda item: item[1])
    summary = ["FINAL 10 EPOCHS"]
    summary.extend(
        f"{data['lrs'][member] * 1e6:g}e-6:  {value:.6f}%"
        for member, value in ranking
    )
    ax.text(
        0.985, 0.96, "\n".join(summary), transform=ax.transAxes,
        ha="right", va="top", fontsize=9.5, linespacing=1.35, color="#26343C",
        bbox=dict(boxstyle="round,pad=.55", fc="white", ec="#D6DFE5"),
    )
    horizon = data["epochs"][-1]
    ax.set_xlim(0, horizon)
    ax.set_xticks(range(0, horizon + 1, 10))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Composite mistag (%)")
    ax.grid(axis="y", color="#E5E9EC", linewidth=0.6)
    fig.legend(
        loc="lower left", bbox_to_anchor=(0.075, 0.10), ncol=min(len(members), 5),
        frameon=False, fontsize=9.5,
    )
    fig.text(0.08, 0.94, "Fixed-LR performance comparison", fontsize=22, fontweight="bold", color="#172B3A")
    fig.text(
        0.08, 0.895,
        f"{len(members)} independent fixed-rate branches · no checkpoint copying or LR mutation",
        fontsize=11.5, color="#53616B",
    )
    fig.text(
        0.08, 0.045,
        f"★ Best recorded checkpoint: {data['lrs'][data['best_member']] * 1e6:g}e-6 at E{data['best_epoch']} "
        f"({data['best_value']:.6f}%). Thick line: best final-10 mean.",
        fontsize=9.5, color=colors[data["best_member"]],
    )
    return fig


def plot_runs(runs, prefix_runs=(), output=None):
    runs = [Path(path) for path in runs]
    output = Path(output) if output else runs[0] / "plots" / "fixed_lr_performance_comparison.png"
    data = fixed_lr_data(runs, prefix_runs)
    plt = plot_setup()
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = draw(plt, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=OUTPUT_DPI, facecolor="white")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    print(plot_runs(args.run, args.prefix_run, args.output))


if __name__ == "__main__":
    main()

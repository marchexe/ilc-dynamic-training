#!/usr/bin/env python3
"""Plot dynamic-controller signals saved in a PBT manifest."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


ACTION_COLORS = {
    "keep": "0.45",
    "lr_mul_0_95": "#f58518",
    "lr_mul_0_9": "#e45756",
    "lr_mul_1_05": "#54a24b",
    "lr_mul_1_1": "#2f5597",
    "flag_review": "#b279a2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot dynamic PBT controller diagnostics.")
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "diagnostics" / "controller_diagnostics.png"


def completed_generations(manifest):
    return [
        generation
        for generation in manifest.get("generations", [])
        if generation.get("status") == "completed"
    ]


def _float_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _compact_member(member):
    return str(member).replace("member_", "m")


def collect_rows(manifest):
    rows = []
    for generation in completed_generations(manifest):
        observations = generation.get("controller_observations") or {}
        actions = generation.get("controller_actions") or {}
        for member, observation in sorted(observations.items()):
            action = actions.get(member) or {}
            x_value = observation.get("epoch_fraction")
            if x_value is None:
                x_value = generation.get("index")
            rows.append(
                {
                    "generation": generation.get("index"),
                    "x": _float_or_none(x_value),
                    "member": member,
                    "metric_name": observation.get("metric_name"),
                    "metric_value": _float_or_none(observation.get("metric_value")),
                    "metric_ema": _float_or_none(observation.get("metric_ema")),
                    "metric_uncertainty": _float_or_none(observation.get("metric_uncertainty")),
                    "metric_delta_sigma": _float_or_none(observation.get("metric_delta_sigma")),
                    "baseline_metric_value": _float_or_none(observation.get("baseline_metric_value")),
                    "baseline_delta": _float_or_none(observation.get("baseline_delta")),
                    "lr": _float_or_none(observation.get("lr")),
                    "train_loss_ema": _float_or_none(observation.get("train_loss_ema")),
                    "grad_norm": _float_or_none(observation.get("grad_norm")),
                    "adaptive_direction_norm": _float_or_none(observation.get("adaptive_direction_norm")),
                    "optimizer_step": _float_or_none(observation.get("optimizer_step")),
                    "state_label": action.get("state_label"),
                    "action": action.get("action", "keep"),
                    "safety_check": action.get("safety_check"),
                    "applied": bool(action.get("applied", False)),
                }
            )
    return [row for row in rows if row["x"] is not None]


def _plot_member_lines(ax, rows, members, value_key, *, ylabel, title, logy=False):
    for member in members:
        member_rows = [row for row in rows if row["member"] == member and row.get(value_key) is not None]
        if not member_rows:
            continue
        ax.plot(
            [row["x"] for row in member_rows],
            [row[value_key] for row in member_rows],
            marker="o",
            markersize=3.4,
            linewidth=1.1,
            label=_compact_member(member),
        )
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", color="0.88", linewidth=0.6)


def _first_baseline(rows):
    for row in rows:
        if row.get("baseline_metric_value") is not None:
            return row["baseline_metric_value"]
    return None


def plot_manifest(manifest_path, output=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    rows = collect_rows(manifest)
    if not rows:
        raise RuntimeError("manifest has no dynamic-controller observations to plot")

    members = sorted({row["member"] for row in rows})
    baseline = _first_baseline(rows)

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
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.8, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 1.0]},
    )

    ax_metric, ax_lr, ax_train, ax_grad = axes
    _plot_member_lines(
        ax_metric,
        rows,
        members,
        "metric_ema",
        ylabel="metric EMA",
        title="Controller metric signal",
    )
    raw_rows = [row for row in rows if row["metric_value"] is not None]
    ax_metric.scatter(
        [row["x"] for row in raw_rows],
        [row["metric_value"] for row in raw_rows],
        s=12,
        color="0.20",
        alpha=0.24,
        label="raw metric",
        zorder=2,
    )
    if baseline is not None:
        ax_metric.axhline(
            baseline,
            color="#5f6f82",
            linestyle=":",
            linewidth=1.1,
            label="checkpoint baseline",
        )

    _plot_member_lines(ax_lr, rows, members, "lr", ylabel="LR", title="Learning-rate actions")
    for row in rows:
        if row["lr"] is None:
            continue
        ax_lr.scatter(
            row["x"],
            row["lr"],
            s=54 if row["applied"] else 30,
            color=ACTION_COLORS.get(row["action"], "0.45"),
            edgecolor="black" if row["applied"] else "white",
            linewidth=0.7,
            zorder=4,
        )
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    _plot_member_lines(
        ax_train,
        rows,
        members,
        "train_loss_ema",
        ylabel="loss EMA",
        title="Training loss EMA",
    )
    _plot_member_lines(
        ax_grad,
        rows,
        members,
        "grad_norm",
        ylabel="grad norm",
        title="Gradient and optimizer pressure",
        logy=True,
    )
    adaptive_rows = [row for row in rows if row["adaptive_direction_norm"] is not None]
    if adaptive_rows:
        ax_adaptive = ax_grad.twinx()
        ax_adaptive.plot(
            [row["x"] for row in adaptive_rows],
            [row["adaptive_direction_norm"] for row in adaptive_rows],
            color="#b279a2",
            linewidth=1.0,
            alpha=0.72,
            label="adaptive direction",
        )
        ax_adaptive.set_ylabel("adaptive direction")

    action_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            label=action,
            markersize=6,
        )
        for action, color in ACTION_COLORS.items()
        if any(row["action"] == action for row in rows)
    ]
    applied_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        label="applied",
        markersize=7,
    )
    ax_lr.legend(handles=action_handles + [applied_handle], frameon=False, loc="best", ncols=4)
    ax_metric.legend(frameon=False, loc="best", ncols=min(5, len(members) + 2))
    ax_grad.set_xlabel("Epoch fraction")

    fig.suptitle(
        f"{manifest.get('experiment', resolved_manifest_path.parent.name)}: dynamic controller diagnostics",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    print(plot_manifest(args.manifest, args.output))


if __name__ == "__main__":
    main()

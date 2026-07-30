#!/usr/bin/env python3
"""Plot background efficiency at fixed b-tag efficiency from PBT manifests."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


DEFAULT_B_EFFICIENCIES = (0.8, 0.9)
BACKGROUND_LABELS = {
    "c": "c mistag",
    "d": "d mistag",
}
BACKGROUND_COLORS = {
    ("c", 0.8): "#d7191c",
    ("d", 0.8): "#0000cc",
    ("c", 0.9): "#a000b0",
    ("d", 0.9): "#66c7f4",
    ("c", 1.0): "#e67e22",
    ("d", 1.0): "#2ca25f",
}
BACKGROUND_MARKERS = {
    ("c", 0.8): "o",
    ("d", 0.8): "s",
    ("c", 0.9): "o",
    ("d", 0.9): "s",
    ("c", 1.0): "^",
    ("d", 1.0): "v",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot background efficiency versus training events at fixed b-tag efficiency."
        )
    )
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--member",
        default="best",
        help="Member to plot, or 'best' for the best member in each completed generation.",
    )
    parser.add_argument(
        "--b-eff",
        default=",".join(str(value) for value in DEFAULT_B_EFFICIENCIES),
        help="Comma-separated fixed b-tag efficiencies, e.g. 0.8,0.9,1.0.",
    )
    parser.add_argument("--title")
    parser.add_argument("--labels", action="store_true", help="Annotate training-size labels on points.")
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


def generation_events(manifest, generation):
    samples_per_epoch = int(manifest.get("config", {}).get("shared", {}).get("samples_per_epoch", 0))
    epoch = int(generation.get("epoch", generation.get("index", 0)))
    return (epoch + 1) * samples_per_epoch


def generation_member(manifest, generation, member):
    workers = generation.get("workers", {})
    if member != "best":
        return member if member in workers else None
    metric = manifest.get("config", {}).get("pbt", {}).get("metric", "validation_bkg_rejection_score")
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    candidates = []
    for name, worker in workers.items():
        value = (worker.get("metrics") or {}).get(metric)
        if value is not None:
            candidates.append((name, float(value)))
    if not candidates:
        return None
    return (max if mode == "max" else min)(candidates, key=lambda item: item[1])[0]


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


def collect_series(manifest, b_efficiencies, member="best"):
    series = {(background, b_eff): {"x": [], "y": []} for b_eff in b_efficiencies for background in ("c", "d")}
    point_labels = []
    for generation in completed_generations(manifest):
        member_name = generation_member(manifest, generation, member)
        if member_name is None:
            continue
        metrics = (generation.get("workers", {}).get(member_name, {}) or {}).get("metrics") or {}
        events = generation_events(manifest, generation)
        used = False
        for b_eff in b_efficiencies:
            for background in ("c", "d"):
                rejection = rejection_at(metrics, b_eff, background)
                if rejection is None or rejection <= 0 or not math.isfinite(rejection):
                    continue
                series[(background, b_eff)]["x"].append(events)
                series[(background, b_eff)]["y"].append(1.0 / rejection)
                used = True
        if used:
            point_labels.append((events, member_name, generation["index"]))
    return series, point_labels


def event_label(events):
    if events >= 1_000_000:
        value = events / 1_000_000
        return f"{value:g}M"
    if events >= 1_000:
        value = events / 1_000
        return f"{value:g}k"
    return str(events)


def milestone_events(point_labels):
    events = sorted({events for events, _, _ in point_labels})
    if len(events) <= 8:
        return set(events)
    indices = {0, 1, 2, min(4, len(events) - 1), len(events) // 2, len(events) - 1}
    return {events[index] for index in indices}


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "diagnostics" / "btag_background_efficiency_vs_training_size.png"


def log_tick_label(value, _position):
    if value <= 0:
        return ""
    exponent = round(math.log10(value))
    if not math.isclose(value, 10**exponent, rel_tol=1e-8, abs_tol=1e-12):
        return ""
    return "1" if exponent == 0 else rf"$10^{{{exponent}}}$"


def plot_manifest(manifest_path, output=None, member="best", b_efficiencies=DEFAULT_B_EFFICIENCIES, title=None, annotate=True):
    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    series, point_labels = collect_series(manifest, b_efficiencies, member=member)
    if not any(values["x"] for values in series.values()):
        raise RuntimeError("manifest has no b-tag efficiency rejection curves to plot")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.5), constrained_layout=True)
    line_styles = {
        0.8: "-",
        0.9: "-",
        1.0: "--",
    }

    for b_eff in b_efficiencies:
        for background in ("c", "d"):
            values = series[(background, b_eff)]
            if not values["x"]:
                continue
            label_background = background
            marker = BACKGROUND_MARKERS.get((background, b_eff), "o")
            markerface = "white" if b_eff >= 0.9 else BACKGROUND_COLORS.get((background, b_eff))
            ax.plot(
                values["x"],
                values["y"],
                marker=marker,
                markersize=4.8,
                markerfacecolor=markerface,
                markeredgewidth=1.0,
                linewidth=2.0,
                linestyle=line_styles.get(b_eff, "-"),
                color=BACKGROUND_COLORS.get((background, b_eff)),
                label=f"{label_background} bkg, b-eff {b_eff:.2f}",
            )

    if annotate:
        labelled_events = milestone_events(point_labels)
        seen = set()
        for events, _, _ in point_labels:
            if events in seen or events not in labelled_events:
                continue
            seen.add(events)
            local_values = [
                y
                for values in series.values()
                for x, y in zip(values["x"], values["y"])
                if x == events
            ]
            if local_values:
                ax.annotate(
                    event_label(events),
                    (events, max(local_values)),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=8.5,
                    color="0.15",
                    ha="center",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                )

    ax.set_title(title or "Background efficiency at fixed b-tag efficiency", loc="left", pad=8)
    ax.set_xlabel("Training size [events]")
    ax.set_ylabel("Background efficiency")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_label))
    ax.grid(axis="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output

def main():
    args = parse_args()
    b_efficiencies = tuple(float(value.strip()) for value in args.b_eff.split(",") if value.strip())
    output = plot_manifest(
        args.manifest,
        output=args.output,
        member=args.member,
        b_efficiencies=b_efficiencies,
        title=args.title,
        annotate=args.labels,
    )
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot BGrej-vs-efficiency evolution for PBT generation winners."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

TAG_PAIRS = {
    "b": ("bc", "bd"),
    "c": ("cb", "cd"),
}
PAIR_LABELS = {
    "bc": "b tag / c bkg",
    "bd": "b tag / d bkg",
    "cb": "c tag / b bkg",
    "cd": "c tag / d bkg",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot signal efficiency vs background rejection or mistag for PBT generation winners."
    )
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag", choices=sorted(TAG_PAIRS), default="b")
    parser.add_argument("--quantity", choices=("rejection", "mistag"), default="rejection")
    parser.add_argument("--member", default="winner", help="'winner' or a fixed member name")
    parser.add_argument("--title")
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


def metric_value(worker, metric):
    value = (worker.get("metrics") or {}).get(metric)
    return float(value) if value is not None else None


def generation_member(manifest, generation, member):
    workers = generation.get("workers", {})
    if member != "winner":
        return member if member in workers else None
    metric = manifest.get("config", {}).get("pbt", {}).get("metric", "validation_bkg_rejection_score")
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    candidates = []
    for name, worker in workers.items():
        value = metric_value(worker, metric)
        if value is not None:
            candidates.append((name, value))
    if not candidates:
        return None
    return (max if mode == "max" else min)(candidates, key=lambda item: item[1])[0]


def worker_lr(worker):
    if worker.get("lr") is not None:
        return float(worker["lr"])
    command = worker.get("command") or []
    if "--start-lr" in command:
        index = command.index("--start-lr")
        if index + 1 < len(command):
            return float(command[index + 1])
    return None


def collect_curves(manifest, tag="b", member="winner"):
    metric = manifest.get("config", {}).get("pbt", {}).get("metric", "validation_bkg_rejection_score")
    rows = []
    for generation in completed_generations(manifest):
        member_name = generation_member(manifest, generation, member)
        if member_name is None:
            continue
        worker = generation.get("workers", {}).get(member_name, {})
        metrics = worker.get("metrics") or {}
        curves = metrics.get("validation_bkg_rejection_at_eff") or {}
        efficiencies = curves.get("efficiencies") or []
        pairs = curves.get("pairs") or {}
        if not efficiencies or not all(pair in pairs for pair in TAG_PAIRS[tag]):
            continue
        rows.append(
            {
                "generation": generation["index"],
                "epoch": generation.get("epoch"),
                "member": member_name,
                "lr": worker_lr(worker),
                "objective": metric_value(worker, metric),
                "efficiencies": [float(value) for value in efficiencies],
                "pairs": {pair: [float(value) for value in pairs[pair]] for pair in TAG_PAIRS[tag]},
            }
        )
    return rows


def default_output(manifest_path, tag="b", quantity="rejection"):
    filename = f"{tag}tag_rejection_evolution.png" if quantity == "rejection" else f"{tag}tag_mistag_evolution.png"
    return Path(manifest_path).parent / "plots" / filename


def log_tick_label(value, _position):
    if value <= 0:
        return ""
    exponent = round(math.log10(value))
    if not math.isclose(value, 10**exponent, rel_tol=1e-8, abs_tol=1e-12):
        return ""
    return "1" if exponent == 0 else rf"$10^{{{exponent}}}$"


def values_for_quantity(rejection_values, quantity):
    if quantity == "rejection":
        return rejection_values
    return [100.0 / value if value > 0 and math.isfinite(value) else float("nan") for value in rejection_values]


def quantity_ylabel(quantity):
    return "background rejection" if quantity == "rejection" else "mistag [%]"


def quantity_title(tag, quantity):
    tag_name = f"{tag}-tag"
    if quantity == "rejection":
        return f"{tag_name} background rejection evolution"
    return f"{tag_name} mistag evolution"


def lower_is_better_text(quantity):
    return "higher is better" if quantity == "rejection" else "lower is better"


def label_for_row(row):
    lr = "?" if row["lr"] is None else f"{row['lr']:.3g}"
    return f"gen {row['generation']} / {row['member'].replace('member_', 'm')} / lr {lr}"


def plot_manifest(manifest_path, output=None, tag="b", quantity="rejection", member="winner", title=None):
    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path, tag, quantity)
    rows = collect_curves(manifest, tag=tag, member=member)
    if not rows:
        raise RuntimeError("manifest has no completed BGrej curves to plot")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

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
    pairs = TAG_PAIRS[tag]
    fig, axes = plt.subplots(1, len(pairs), figsize=(11.0, 4.5), sharex=True, sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    colors = plt.get_cmap("viridis")
    denominator = max(1, len(rows) - 1)
    best = manifest.get("best") or {}

    for pair_index, pair in enumerate(pairs):
        ax = axes[pair_index]
        for row_index, row in enumerate(rows):
            is_global_best = (
                row["generation"] == best.get("generation")
                and row["member"] == best.get("member")
            )
            ax.plot(
                row["efficiencies"],
                values_for_quantity(row["pairs"][pair], quantity),
                marker="o",
                markersize=4.0 if not is_global_best else 5.5,
                linewidth=1.4 if not is_global_best else 2.4,
                color="black" if is_global_best else colors(row_index / denominator),
                alpha=1.0 if is_global_best else 0.78,
                label=("global best: " if is_global_best else "") + label_for_row(row),
            )
        ax.set_title(PAIR_LABELS[pair], loc="left")
        for working_point in (0.8, 0.9):
            ax.axvline(working_point, color="0.72", linestyle=":", linewidth=0.9)
        ax.set_xlabel(f"{tag}-tag efficiency")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(log_tick_label))
        ax.grid(axis="both", color="0.88", linewidth=0.6)
        ax.set_xticks(rows[-1]["efficiencies"])
    axes[0].set_ylabel(quantity_ylabel(quantity))
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        title or f"{manifest.get('experiment', resolved_manifest_path.parent.name)}: {quantity_title(tag, quantity)} ({lower_is_better_text(quantity)})",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    output = plot_manifest(
        args.manifest,
        output=args.output,
        tag=args.tag,
        quantity=args.quantity,
        member=args.member,
        title=args.title,
    )
    print(output)


if __name__ == "__main__":
    main()

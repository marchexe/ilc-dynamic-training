#!/usr/bin/env python3
"""Render fixed tag-efficiency mistag tables from Weaver/PBT metrics."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


TAG_BACKGROUNDS = {
    "b": ("c", "d"),
    "c": ("b", "d"),
}
PAIR_FOR = {
    ("b", "c"): "bc",
    ("b", "d"): "bd",
    ("c", "b"): "cb",
    ("c", "d"): "cd",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create mistag percentage tables at fixed b-tag or c-tag efficiencies."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Manifest/run path, optionally label=path. Each input becomes one table row.",
    )
    parser.add_argument("--output", type=Path, help="PNG output path")
    parser.add_argument("--csv", type=Path, help="CSV output path")
    parser.add_argument("--tag", choices=sorted(TAG_BACKGROUNDS), default="c")
    parser.add_argument("--eff", default="0.5,0.8", help="Comma-separated fixed tag efficiencies")
    parser.add_argument("--member", default="global_best", help="global_best, best_final, or a member name")
    parser.add_argument("--title")
    return parser.parse_args()


def parse_input(value):
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.parent.name if path.name == "manifest.json" else path.name, path


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


def worker_for_row(manifest, member):
    generations = completed_generations(manifest)
    if not generations:
        raise RuntimeError("manifest has no completed generations")
    if member == "global_best":
        best = manifest.get("best") or {}
        generation_index = best.get("generation")
        member_name = best.get("member")
        if generation_index is None or member_name is None:
            raise RuntimeError("manifest has no global best record")
        generation = next(item for item in generations if item["index"] == generation_index)
        return generation.get("workers", {})[member_name], generation, member_name
    if member == "best_final":
        generation = generations[-1]
        metric = manifest.get("config", {}).get("pbt", {}).get("metric", "validation_bkg_rejection_score")
        mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
        candidates = [
            (name, float((worker.get("metrics") or {}).get(metric)))
            for name, worker in generation.get("workers", {}).items()
            if (worker.get("metrics") or {}).get(metric) is not None
        ]
        if not candidates:
            raise RuntimeError("final generation has no ranking metrics")
        member_name = (max if mode == "max" else min)(candidates, key=lambda item: item[1])[0]
        return generation["workers"][member_name], generation, member_name
    for generation in reversed(generations):
        worker = generation.get("workers", {}).get(member)
        if worker and worker.get("metrics"):
            return worker, generation, member
    raise RuntimeError(f"member not found or has no metrics: {member}")


def rejection_at(metrics, tag, eff, background):
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
    pair = PAIR_FOR[(tag, background)]
    values = pairs.get(pair) or []
    index = efficiencies.index(eff)
    return float(values[index]) if index < len(values) else None


def mistag_percent(metrics, tag, eff, background):
    rejection = rejection_at(metrics, tag, eff, background)
    if rejection is None or rejection <= 0 or not math.isfinite(rejection):
        return None
    return 100.0 / rejection


def collect_tables(inputs, tag, efficiencies, member, manifests=None):
    backgrounds = TAG_BACKGROUNDS[tag]
    tables = {eff: [] for eff in efficiencies}
    manifests = manifests or {}
    for label, path in inputs:
        manifest = manifests.get(path)
        if manifest is None:
            manifest, _ = load_manifest(path)
        worker, generation, member_name = worker_for_row(manifest, member)
        metrics = worker.get("metrics") or {}
        for eff in efficiencies:
            row = {
                "label": label,
                "generation": generation["index"],
                "member": member_name,
            }
            for background in backgrounds:
                row[f"{background}_bkg_percent"] = mistag_percent(metrics, tag, eff, background)
            tables[eff].append(row)
    return tables


def format_percent(value):
    return "-" if value is None else f"{value:.3f}%"


def write_csv(path, tables, tag):
    backgrounds = TAG_BACKGROUNDS[tag]
    fieldnames = ["fixed_tag", "label", "generation", "member", *[f"{background}_bkg_percent" for background in backgrounds]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for eff, rows in tables.items():
            for row in rows:
                writer.writerow({"fixed_tag": f"{tag}_eff_{eff:.2f}", **row})



def plot_tables(tables, tag, output, title=None):
    import matplotlib.pyplot as plt

    backgrounds = TAG_BACKGROUNDS[tag]
    fig_height = max(2.0, 1.0 + 1.0 * len(tables) + 0.32 * max(len(rows) for rows in tables.values()))
    fig, axes = plt.subplots(len(tables), 1, figsize=(6.3, fig_height), squeeze=False)
    axes = axes[:, 0]
    header_color = "#178f78"
    row_color = "#e9f3ef"
    alt_row_color = "#d8ebe4"

    for ax, (eff, rows) in zip(axes, tables.items()):
        ax.axis("off")
        cell_text = []
        for row in rows:
            cell_text.append(
                [
                    row["label"],
                    *[format_percent(row[f"{background}_bkg_percent"]) for background in backgrounds],
                ]
            )
        col_labels = [f"{tag}-tag {int(round(eff * 100))}% eff.", *[f"{background} bkg." for background in backgrounds]]
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=[0.48, *([0.26] * len(backgrounds))],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.0, 1.55)
        for (row_index, _col_index), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            if row_index == 0:
                cell.set_facecolor(header_color)
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(alt_row_color if row_index % 2 else row_color)
    if title:
        fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def default_output_path(first_input, tag):
    _, path = first_input
    path = Path(path)
    if path.is_dir():
        return path / f"{tag}_mistag_tables.png"
    return path.with_name(f"{tag}_mistag_tables.png")


def main():
    args = parse_args()
    inputs = [parse_input(value) for value in args.inputs]
    efficiencies = tuple(float(value.strip()) for value in args.eff.split(",") if value.strip())
    tables = collect_tables(inputs, args.tag, efficiencies, args.member)
    output = args.output or default_output_path(inputs[0], args.tag)
    csv_path = args.csv or output.with_suffix(".csv")
    write_csv(csv_path, tables, args.tag)
    print(plot_tables(tables, args.tag, output, title=args.title))
    print(csv_path)


if __name__ == "__main__":
    main()

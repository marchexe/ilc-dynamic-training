#!/usr/bin/env python3
"""Plot background rejection curves from existing Weaver/PBT logs."""

import argparse
import ast
import json
import re
from pathlib import Path


EFFICIENCY_POINTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PAIR_LABELS = {
    "bc": "b tag / c bkg",
    "bd": "b tag / d bkg",
    "cb": "c tag / b bkg",
    "cd": "c tag / d bkg",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot BGrej vs signal efficiency from a completed training log or PBT manifest."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a generation log, a run directory, or a PBT manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path. Defaults to <run>/bgrej_curves.png for manifests.",
    )
    parser.add_argument(
        "--generation",
        type=int,
        help="PBT generation to plot. Defaults to the recorded global best generation.",
    )
    parser.add_argument(
        "--member",
        help="PBT member to plot. Defaults to the recorded global best member.",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Defaults to experiment/member/generation metadata.",
    )
    return parser.parse_args()


def parse_bgrej_curves(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"- bkg_rejection_at_eff:\s*\n(?P<payload>\{.*?\})(?=\n\s+- |\n\[|\Z)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"No bkg_rejection_at_eff block found in {log_path}")
    curves = ast.literal_eval(matches[-1].group("payload"))
    missing = set(PAIR_LABELS) - set(curves)
    if missing:
        raise ValueError(f"Missing pair curves in {log_path}: {sorted(missing)}")
    for pair, values in curves.items():
        if len(values) != len(EFFICIENCY_POINTS):
            raise ValueError(
                f"Pair {pair} in {log_path} has {len(values)} points, "
                f"expected {len(EFFICIENCY_POINTS)}"
            )
    return curves


def load_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if path.name != "manifest.json":
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def completed_generation(manifest, index):
    for generation in manifest.get("generations", []):
        if generation.get("index") == index:
            if generation.get("status") != "completed":
                raise ValueError(f"Generation {index} is not completed")
            return generation
    raise ValueError(f"Generation {index} not found in manifest")


def log_from_manifest(manifest, manifest_path, generation_index=None, member_name=None):
    best = manifest.get("best") or {}
    if generation_index is None:
        generation_index = best.get("generation")
    if member_name is None:
        member_name = best.get("member")
    if generation_index is None or member_name is None:
        raise ValueError("Specify --generation and --member, or use a manifest with a recorded best")

    generation = completed_generation(manifest, int(generation_index))
    worker = generation.get("workers", {}).get(member_name)
    if worker is None:
        raise ValueError(f"Member {member_name!r} not found in generation {generation_index}")
    log_path = Path(worker["log"])
    if not log_path.is_absolute():
        log_path = manifest_path.parent / log_path
    return log_path, int(generation_index), member_name


def default_output(input_path, manifest_path=None):
    if manifest_path is not None:
        return manifest_path.with_name("bgrej_curves.png")
    input_path = Path(input_path)
    return input_path.with_suffix(".bgrej_curves.png")


def plot_curves(curves, output, title):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    colors = plt.get_cmap("tab10")
    for index, pair in enumerate(PAIR_LABELS):
        ax.plot(
            EFFICIENCY_POINTS,
            curves[pair],
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            color=colors(index),
            label=PAIR_LABELS[pair],
        )
    ax.set_title(title, loc="left")
    ax.set_xlabel("signal efficiency")
    ax.set_ylabel("background rejection")
    ax.set_yscale("log")
    ax.set_xticks(EFFICIENCY_POINTS)
    ax.grid(axis="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    manifest, manifest_path = load_manifest(args.input)
    if manifest is None:
        log_path = args.input
        generation = args.generation
        member = args.member
        experiment = log_path.parent.parent.name
    else:
        log_path, generation, member = log_from_manifest(
            manifest,
            manifest_path,
            generation_index=args.generation,
            member_name=args.member,
        )
        experiment = manifest.get("experiment", manifest_path.parent.name)

    curves = parse_bgrej_curves(log_path)
    if args.title:
        title = args.title
    elif generation is not None and member:
        title = f"{experiment}: {member}, generation {generation}"
    else:
        title = f"{experiment}: {log_path.name}"
    output = args.output or default_output(args.input, manifest_path)
    print(plot_curves(curves, output, title))


if __name__ == "__main__":
    main()

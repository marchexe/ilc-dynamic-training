#!/usr/bin/env python3
"""Build the supervisor-milestone figures from completed PBT manifests only.

The script is deliberately read-only with respect to run directories.  It
accepts the four matched cadence-control manifests, validates their completed
shape, and writes a compact JSON evidence table plus three deterministic PNGs.
It never imports or invokes training, evaluation, or PBT execution code.
"""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.pbt.reporting.constants import CB_PALETTE
from training.pbt.reporting.style import plot_setup
from training.runtime import sha256


PROJECT_DIR = Path(__file__).resolve().parents[2]
METRIC = "validation_total_reference_mistag_geomean_percent"
DEFAULT_RUNS = {
    (12345, 1): PROJECT_DIR / "runs/pbt/cadenced_pbt_v1_50epochs/manifest.json",
    (12345, 5): PROJECT_DIR / "runs/pbt/cadenced_pbt_v1_cadence5_50epochs/manifest.json",
    (22345, 1): PROJECT_DIR / "runs/pbt/cadenced_pbt_v1_seed22345_50epochs/manifest.json",
    (22345, 5): PROJECT_DIR / "runs/pbt/cadenced_pbt_v1_cadence5_seed22345_50epochs/manifest.json",
}
DEFAULT_OUTPUT = PROJECT_DIR / "docs/_static/results"
OUTPUT_NAMES = (
    "pbt_interim_final10_matched_seeds.png",
    "pbt_interim_current_best_trajectories.png",
    "pbt_interim_behavior_summary.png",
    "pbt_interim_metrics.json",
)


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()), path.resolve()


def summarize_run(path, expected_seed, expected_interval):
    """Validate and summarize one completed 50-generation cadenced run."""
    manifest, source = _load(path)
    config = manifest.get("config", {})
    shared = config.get("shared", {})
    pbt = config.get("pbt", {})
    generations = manifest.get("generations", [])
    members = sorted(manifest.get("members", {}))
    if manifest.get("status") != "completed" or manifest.get("next_generation") != 50:
        raise ValueError(f"incomplete run: {source}")
    if len(generations) != 50 or [g.get("index") for g in generations] != list(range(50)):
        raise ValueError(f"run does not contain 50 contiguous generations: {source}")
    if pbt.get("strategy") != "cadenced_pbt_v1" or pbt.get("metric") != METRIC:
        raise ValueError(f"unexpected strategy or metric: {source}")
    if shared.get("seed") != expected_seed or pbt.get("exploit_interval_generations") != expected_interval:
        raise ValueError(f"seed/cadence does not match requested pair: {source}")
    if shared.get("weaver_epochs_per_generation") != 1 or len(members) != 5:
        raise ValueError(f"not a five-member full-epoch run: {source}")

    trajectory = []
    ancestors = {name: name for name in members}
    surviving_ancestries = []
    collapse_epoch = None
    exploits = []
    for expected_index, generation in enumerate(generations):
        workers = generation.get("workers", {})
        if generation.get("status") != "completed" or set(workers) != set(members):
            raise ValueError(f"incomplete generation {expected_index}: {source}")
        scores = []
        for name in members:
            worker = workers[name]
            value = (worker.get("metrics") or {}).get(METRIC)
            if worker.get("status") != "completed" or worker.get("returncode") != 0:
                raise ValueError(f"failed worker {expected_index}/{name}: {source}")
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"non-finite metric {expected_index}/{name}: {source}")
            scores.append(float(value))
        trajectory.append(min(scores))
        before = dict(ancestors)
        for event in generation.get("exploit", []):
            if not event.get("applied"):
                raise ValueError(f"unapplied exploit at generation {expected_index}: {source}")
            ancestors[event["recipient"]] = before[event["donor"]]
            exploits.append(event)
        survivors = len(set(ancestors.values()))
        surviving_ancestries.append(survivors)
        if survivors == 1 and collapse_epoch is None:
            collapse_epoch = expected_index + 1

    final_control = (manifest.get("final_evaluations") or {}).get("control") or {}
    if len(final_control) != 6 or any(item.get("status") != "completed" for item in final_control.values()):
        raise ValueError(f"final evaluations are incomplete: {source}")
    if generations[-1].get("exploit") or not generations[-1]["cadenced_pbt_v1"].get("terminal"):
        raise ValueError(f"terminal generation is not suppressed: {source}")

    best = manifest.get("best") or {}
    if not math.isclose(float(best.get("metric_value")), min(trajectory), rel_tol=0, abs_tol=1e-15):
        raise ValueError(f"global-best record disagrees with trajectory: {source}")
    run_record = manifest.get("run") or {}
    return {
        "run": str(source.parent),
        "manifest": str(source),
        "manifest_sha256": sha256(source),
        "experiment_name": config.get("experiment_name"),
        "training_seed": expected_seed,
        "exploit_interval_generations": expected_interval,
        "trajectory": trajectory,
        "final10_current_best_mean": mean(trajectory[-10:]),
        "global_best": float(best["metric_value"]),
        "global_best_member": best.get("member"),
        "global_best_generation": int(best.get("generation")),
        "exploit_count": len(exploits),
        "lineage_collapse_epoch": collapse_epoch,
        "surviving_initial_ancestries": surviving_ancestries,
        "git": manifest.get("git"),
        "resolved_config_sha256": manifest.get("fingerprint"),
        "source_hashes": run_record.get("source_hashes"),
        "dataset_fingerprints": (manifest.get("datasets") or {}).get("fingerprints"),
        "initial_checkpoint_sha256": (manifest.get("checkpoint") or {}).get("sha256"),
    }


def compare_pair(cadence1, cadence5):
    if cadence1["training_seed"] != cadence5["training_seed"]:
        raise ValueError("paired runs must use the same training seed")
    wins = {"cadence1": 0, "cadence5": 0, "ties": 0}
    for left, right in zip(cadence1["trajectory"], cadence5["trajectory"]):
        if math.isclose(left, right, rel_tol=0, abs_tol=1e-15):
            wins["ties"] += 1
        elif left < right:
            wins["cadence1"] += 1
        else:
            wins["cadence5"] += 1
    difference = cadence5["final10_current_best_mean"] - cadence1["final10_current_best_mean"]
    return {
        "seed": cadence1["training_seed"],
        "cadence1_final10": cadence1["final10_current_best_mean"],
        "cadence5_final10": cadence5["final10_current_best_mean"],
        "cadence5_minus_cadence1": difference,
        "cadence5_relative_difference_percent": 100 * difference / cadence1["final10_current_best_mean"],
        "trajectory_wins": wins,
    }


def collect(run_paths=None):
    run_paths = run_paths or DEFAULT_RUNS
    runs = {
        (seed, cadence): summarize_run(path, seed, cadence)
        for (seed, cadence), path in sorted(run_paths.items())
    }
    checkpoint_hashes = {run["initial_checkpoint_sha256"] for run in runs.values()}
    dataset_fingerprints = {json.dumps(run["dataset_fingerprints"], sort_keys=True) for run in runs.values()}
    if len(checkpoint_hashes) != 1 or len(dataset_fingerprints) != 1:
        raise ValueError("matched runs do not share checkpoint and dataset fingerprints")
    pairs = {seed: compare_pair(runs[(seed, 1)], runs[(seed, 5)]) for seed in (12345, 22345)}
    return {"metric": METRIC, "lower_is_better": True, "runs": runs, "pairs": pairs}


def _figure_final10(plt, data):
    seeds = (12345, 22345)
    x = list(range(len(seeds)))
    blue, orange = CB_PALETTE["blue"], CB_PALETTE["vermillion"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    left = [data["runs"][(seed, 1)]["final10_current_best_mean"] for seed in seeds]
    right = [data["runs"][(seed, 5)]["final10_current_best_mean"] for seed in seeds]
    for index in x:
        ax.plot([index, index], [left[index], right[index]], color="#AAB4BA", lw=2, zorder=1)
    ax.scatter(x, left, s=85, color=blue, label="Variant A: adapt every epoch", zorder=3)
    ax.scatter(x, right, s=85, color=orange, label="Cadence5 control", zorder=3)
    floor = min(left + right) - 0.0006
    ax.set_ylim(floor, max(left + right) + 0.00065)
    ax.set_xlim(-.35, 1.35)
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Final-10 current-best mean (%) ↓")
    ax.set_title("Matched-seed cadence control\nOne-epoch generations; lower is better",
                 loc="left", fontweight="bold", linespacing=1.5)
    ax.grid(axis="y", color="#E5E9EC", linewidth=.7)
    ax.legend(frameon=False, loc="upper left")
    for index, seed in enumerate(seeds):
        delta = data["pairs"][seed]["cadence5_minus_cadence1"]
        ax.text(index, left[index] - .00012, f"{left[index]:.6f}",
                ha="center", va="top", fontsize=8, color=blue)
        ax.text(index, right[index] + .00010, f"{right[index]:.6f}",
                ha="center", va="bottom", fontsize=8, color=orange)
        ax.text(index, floor + .00008, f"cadence5 − cadence1 = +{delta:.6f} pp",
                ha="center", va="bottom", fontsize=8.5, color="#26343C")
    fig.text(.99, .005, "Two deterministic seeds; no statistical-significance claim",
             ha="right", va="bottom", fontsize=7.5, color="#68757D")
    return fig


def _figure_trajectories(plt, data):
    blue, orange = CB_PALETTE["blue"], CB_PALETTE["vermillion"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, seed in zip(axes, (12345, 22345)):
        epochs = list(range(1, 51))
        ax.plot(epochs, data["runs"][(seed, 1)]["trajectory"], color=blue, lw=2,
                label="Variant A: every epoch")
        ax.plot(epochs, data["runs"][(seed, 5)]["trajectory"], color=orange, lw=2,
                label="Cadence5 control")
        wins = data["pairs"][seed]["trajectory_wins"]
        ax.set_title(f"Training seed {seed}", loc="left", fontweight="bold")
        ax.text(.98, .97,
                f"Lower epochs: A {wins['cadence1']} · C5 {wins['cadence5']} · ties {wins['ties']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#D6DFE5"))
        ax.set_xlabel("Completed full epoch")
        ax.grid(color="#E5E9EC", linewidth=.7)
    axes[0].set_ylabel("Current-best full-reference metric (%) ↓")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Current-best full-reference trajectories", x=.02, ha="left", fontweight="bold")
    return fig


def _figure_behavior(plt, data):
    seeds = (12345, 22345)
    x = list(range(len(seeds)))
    width = .34
    blue, orange = CB_PALETTE["blue"], CB_PALETTE["vermillion"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, field, title, ylabel in (
        (axes[0], "exploit_count", "Exploit actions", "Applied copy actions"),
        (axes[1], "lineage_collapse_epoch", "Lineage collapse", "First epoch with one ancestry"),
    ):
        values1 = [data["runs"][(seed, 1)][field] for seed in seeds]
        values5 = [data["runs"][(seed, 5)][field] for seed in seeds]
        bars1 = ax.bar([v - width / 2 for v in x], values1, width, color=blue,
                       label="Variant A: every epoch")
        bars5 = ax.bar([v + width / 2 for v in x], values5, width, color=orange,
                       label="Cadence5 control")
        ax.set_xticks(x, [str(seed) for seed in seeds])
        ax.set_xlabel("Training seed")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E9EC", linewidth=.7)
        ax.bar_label(bars1, padding=3)
        ax.bar_label(bars5, padding=3)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("PBT behavior under matched one-epoch controls", x=.02, ha="left", fontweight="bold")
    return fig


def export(output=DEFAULT_OUTPUT, run_paths=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = collect(run_paths)
    plt = plot_setup()
    figures = (
        (OUTPUT_NAMES[0], _figure_final10(plt, data)),
        (OUTPUT_NAMES[1], _figure_trajectories(plt, data)),
        (OUTPUT_NAMES[2], _figure_behavior(plt, data)),
    )
    for name, figure in figures:
        figure.savefig(output / name, dpi=220, metadata={"Software": "matplotlib"})
        plt.close(figure)

    serializable = {
        "schema_version": 1,
        "metric": data["metric"],
        "lower_is_better": data["lower_is_better"],
        "runs": {
            f"seed{seed}_cadence{cadence}": run
            for (seed, cadence), run in sorted(data["runs"].items())
        },
        "pairs": {str(seed): pair for seed, pair in sorted(data["pairs"].items())},
    }
    (output / OUTPUT_NAMES[3]).write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    return [output / name for name in OUTPUT_NAMES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in export(args.output):
        print(path)


if __name__ == "__main__":
    main()

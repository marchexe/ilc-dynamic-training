#!/usr/bin/env python3
"""Create a single HEP-style physics performance overview plot from a PBT manifest."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

TAG_BACKGROUNDS = {
    "b": ("c", "d"),
    "c": ("b", "d"),
}
TAG_PAIRS = {
    "b": ("bc", "bd"),
    "c": ("cb", "cd"),
}
PAIR_LABELS = {
    "bc": "c background",
    "bd": "d background",
    "cb": "b background",
    "cd": "d background",
}
# Background-flavour colors, shared across every fixed-WP figure (this file,
# artifacts.py's evolution/comparison plots, and the background-efficiency
# diagnostic) so a given color always means the same mistagged flavour.
FLAVOR_COLORS = {
    "b": "#4c78a8",
    "c": "#59a14f",
    "d": "#e15759",
}
REFERENCE_WORKING_POINTS = {
    "b": (0.8, 0.9),
    "c": (0.5, 0.8),
}
# Short human-readable label per worker_for_report() role -- used in figure
# titles so a reader never has to guess whether "best" means the PBT's own
# selection or this reporting-only physics-score pick, which can disagree
# (see worker_for_report's "best_physics" vs "global_best" branches).
CHECKPOINT_ROLE_LABELS = {
    "best_physics": "best physics (mean of 8 fixed WPs)",
    "global_best": "global best (PBT selection)",
    "best_final": "best of final generation",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a compact HEP-style physics performance report.")
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--member", default="best_physics", help="best_physics, best_final, global_best, or a member name")
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


def worker_for_report(manifest, member):
    generations = completed_generations(manifest)
    if not generations:
        raise RuntimeError("manifest has no completed generations")
    if member == "best_physics":
        candidates = []
        for generation in generations:
            for member_name, worker in generation.get("workers", {}).items():
                score = physics_mistag_score(worker.get("metrics") or {})
                if score is not None:
                    candidates.append((score, worker, generation, member_name))
        if not candidates:
            raise RuntimeError("manifest has no fixed working-point mistag metrics")
        score, worker, generation, member_name = min(candidates, key=lambda item: item[0])
        return worker, generation, member_name, score
    if member == "global_best":
        best = manifest.get("best") or {}
        generation_index = best.get("generation")
        member_name = best.get("member")
        if generation_index is None or member_name is None:
            raise RuntimeError("manifest has no global best record")
        generation = next((item for item in generations if item["index"] == generation_index), None)
        # generation_index == -1 (member "initial_resume") means the
        # baseline guard kept the pretrained checkpoint itself as global
        # best -- it has no entry in `generations` (that's pre-training),
        # so there is nothing to plot fixed-WP curves from here. Any other
        # missing-generation case is equally unplottable. Raise the same
        # RuntimeError family as the branches above so callers (e.g.
        # reporting/plots.py::write_existing_physics_reports) can fall back
        # to another role instead of crashing on an unhandled StopIteration.
        if generation is None or member_name not in generation.get("workers", {}):
            raise RuntimeError(
                f"manifest's global best (generation {generation_index}, member {member_name!r}) "
                "has no plottable worker -- likely the baseline-guard-kept pretrained checkpoint"
            )
        worker = generation["workers"][member_name]
        return worker, generation, member_name, physics_mistag_score(worker.get("metrics") or {})
    if member == "best_final":
        generation = generations[-1]
        candidates = []
        for member_name, worker in generation.get("workers", {}).items():
            score = physics_mistag_score(worker.get("metrics") or {})
            if score is not None:
                candidates.append((score, worker, member_name))
        if not candidates:
            raise RuntimeError("final generation has no fixed working-point mistag metrics")
        score, worker, member_name = min(candidates, key=lambda item: item[0])
        return worker, generation, member_name, score
    for generation in reversed(generations):
        worker = generation.get("workers", {}).get(member)
        if worker and worker.get("metrics"):
            return worker, generation, member, physics_mistag_score(worker.get("metrics") or {})
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
    pair = f"{tag}{background}"
    values = pairs.get(pair) or []
    index = efficiencies.index(eff)
    return float(values[index]) if index < len(values) else None


def mistag_percent(metrics, tag, eff, background):
    rejection = rejection_at(metrics, tag, eff, background)
    if rejection is None or rejection <= 0 or not math.isfinite(rejection):
        return None
    return 100.0 / rejection


def working_point_counts(metrics, tag, eff, background):
    rows = (metrics.get("validation_bkg_rejection_at_eff_counts") or {}).get(f"{tag}{background}") or []
    for row in rows:
        if abs(float(row.get("signal_efficiency", -1.0)) - float(eff)) < 1.0e-6:
            return row.get("background_passed"), row.get("background_total")
    return None, None


def wilson_uncertainty_percent(passed, total, z=1.0):
    """Asymmetric ~68% (1 sigma) Wilson score interval half-widths, in mistag
    percent units. Stays well-behaved (non-negative, asymmetric) for the
    small mistag rates fixed working points usually sit at, unlike a naive
    sqrt(p(1-p)/n) normal approximation.
    """
    if passed is None or total is None:
        return None
    total = int(total)
    passed = int(passed)
    if total <= 0 or passed < 0 or passed > total:
        return None
    p = passed / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    lower = max(0.0, centre - half)
    upper = min(1.0, centre + half)
    return 100.0 * max(0.0, p - lower), 100.0 * max(0.0, upper - p)


def mistag_uncertainty(metrics, tag, eff, background):
    passed, total = working_point_counts(metrics, tag, eff, background)
    return wilson_uncertainty_percent(passed, total)


def format_percent(value, err=None):
    if value is None:
        return "-"
    errors = [e for e in (err or ()) if e is not None and e > 0]
    if not errors:
        return f"{value:.3f}%"
    magnitude = max(errors)
    digits = max(0, min(6, 1 - int(math.floor(math.log10(magnitude)))))
    lower, upper = err
    if abs(lower - upper) > 0.5 * 10 ** (-digits):
        return f"{value:.{digits}f}% (+{upper:.{digits}f}/-{lower:.{digits}f})"
    return f"{value:.{digits}f}±{magnitude:.{digits}f}%"


def physics_mistag_score(metrics):
    values = []
    for tag, efficiencies in REFERENCE_WORKING_POINTS.items():
        for eff in efficiencies:
            for background in TAG_BACKGROUNDS[tag]:
                value = mistag_percent(metrics, tag, eff, background)
                if value is not None:
                    values.append(value)
    return sum(values) / len(values) if values else None


def selected_rows(rows, best, max_curves=5):
    if max_curves <= 0 or len(rows) <= max_curves:
        return rows
    candidate_indices = [round(index * (len(rows) - 1) / float(max_curves - 1)) for index in range(max_curves)]
    keep = {0, len(rows) - 1, *candidate_indices}
    for index, row in enumerate(rows):
        if row["generation"] == best.get("generation") and row["member"] == best.get("member"):
            keep.add(index)
            break
    return [rows[index] for index in sorted(keep)[:max_curves]]


def collect_evolution_rows(manifest, tag):
    rows = []
    metric = manifest.get("config", {}).get("pbt", {}).get("metric", "validation_bkg_rejection_score")
    for generation in completed_generations(manifest):
        member_name = generation_member(manifest, generation, "winner")
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
                "member": member_name,
                "objective": metric_value(worker, metric),
                "efficiencies": [float(value) for value in efficiencies],
                "pairs": {pair: [float(value) for value in pairs[pair]] for pair in TAG_PAIRS[tag]},
            }
        )
    return rows


def default_output(manifest_path):
    return Path(manifest_path).parent / "plots" / "report" / "physics_performance.png"


def log_tick_label(value, _position):
    if value <= 0:
        return ""
    exponent = round(math.log10(value))
    if not math.isclose(value, 10**exponent, rel_tol=1e-8, abs_tol=1e-12):
        return ""
    return "1" if exponent == 0 else rf"$10^{{{exponent}}}$"


def draw_table(ax, metrics, tag):
    ax.axis("off")
    backgrounds = TAG_BACKGROUNDS[tag]
    rows = []
    for eff in REFERENCE_WORKING_POINTS[tag]:
        rows.append(
            [
                f"{int(round(eff * 100))}%",
                *[
                    format_percent(mistag_percent(metrics, tag, eff, background), mistag_uncertainty(metrics, tag, eff, background))
                    for background in backgrounds
                ],
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["eff.", *[f"{background} bkg." for background in backgrounds]],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.20, 0.40, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.34)
    for (row_index, _col_index), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#255f9f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f3f7fc" if row_index % 2 else "#e3edf8")
    ax.set_title(f"{tag}-tag", loc="left", fontsize=11, fontweight="bold", pad=4)


def draw_mistag_bars(ax, metrics, tag):
    backgrounds = TAG_BACKGROUNDS[tag]
    efficiencies = REFERENCE_WORKING_POINTS[tag]
    x_positions = list(range(len(efficiencies)))
    width = 0.34
    colors = FLAVOR_COLORS

    for background_index, background in enumerate(backgrounds):
        values = [mistag_percent(metrics, tag, eff, background) for eff in efficiencies]
        errors = [mistag_uncertainty(metrics, tag, eff, background) for eff in efficiencies]
        offsets = [x + (background_index - 0.5) * width for x in x_positions]
        bars = ax.bar(
            offsets,
            [0.0 if value is None else value for value in values],
            width=width,
            color=colors[background],
            label=f"{background} background",
        )
        lower = [0.0 if (value is None or err is None) else err[0] for value, err in zip(values, errors)]
        upper = [0.0 if (value is None or err is None) else err[1] for value, err in zip(values, errors)]
        ax.errorbar(
            offsets,
            [0.0 if value is None else value for value in values],
            yerr=[lower, upper],
            fmt="none",
            ecolor="0.2",
            elinewidth=1.0,
            capsize=3,
            zorder=5,
        )
        for bar, value, err in zip(bars, values, errors):
            if value is None:
                continue
            top = value + (err[1] if err else 0.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top,
                format_percent(value, err),
                ha="center",
                va="bottom",
                fontsize=7.4,
            )

    ax.set_title(f"{tag}-tag mistag at fixed efficiency", loc="left", fontsize=11, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(round(eff * 100))}%" for eff in efficiencies])
    ax.set_xlabel(f"{tag}-tag efficiency")
    ax.set_ylabel("mistag [%]")
    ymax = max(
        [
            (mistag_percent(metrics, tag, eff, background) or 0.0)
            + ((mistag_uncertainty(metrics, tag, eff, background) or (0.0, 0.0))[1])
            for eff in efficiencies
            for background in backgrounds
        ]
        or [1.0]
    )
    ax.set_ylim(0, ymax * 1.3 if ymax > 0 else 1.0)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def plot_manifest(manifest_path, output=None, member="best_physics"):
    import matplotlib.pyplot as plt

    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else default_output(resolved_manifest_path)
    worker, generation, member_name, physics_score = worker_for_report(manifest, member)
    metrics = worker.get("metrics") or {}
    if not metrics.get("validation_bkg_rejection_at_eff") and not metrics.get("validation_bkg_rejection_at_eff_lookup"):
        raise RuntimeError("selected worker has no fixed-efficiency rejection metrics")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(12.0, 8.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.80,
        hspace=0.40,
        wspace=0.20,
        height_ratios=[0.52, 1.48],
    )
    fig.text(0.06, 0.835, "Fixed working-point mistag [%]", ha="left", va="bottom", fontsize=12, fontweight="bold")
    draw_table(fig.add_subplot(grid[0, 0]), metrics, "c")
    draw_table(fig.add_subplot(grid[0, 1]), metrics, "b")
    draw_mistag_bars(fig.add_subplot(grid[1, 0]), metrics, "c")
    draw_mistag_bars(fig.add_subplot(grid[1, 1]), metrics, "b")
    if member == "global_best":
        best = manifest.get("best") or {}
        score_text = "" if best.get("metric_value") is None else f" | {best.get('metric')} {best['metric_value']:.4g}"
    else:
        score_text = "" if physics_score is None else f" | avg fixed-WP mistag {physics_score:.3f}%"
    role_label = CHECKPOINT_ROLE_LABELS.get(member, f"member `{member}`")
    fig.suptitle(
        f"Checkpoint: {role_label} ({member_name}, generation {generation['index']}){score_text}",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    args = parse_args()
    print(plot_manifest(args.manifest, args.output, args.member))


if __name__ == "__main__":
    main()

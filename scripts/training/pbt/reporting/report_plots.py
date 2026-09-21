#!/usr/bin/env python3
"""Conditional proxy-validation plot for a PBT run.

The primary performance/LR lineage pair and the two retained physics figures
are produced by their dedicated report scripts. Redundant population, score,
LR-lineage, and LR-correlation candidates intentionally do not live here.
"""

import math
from pathlib import Path

from matplotlib.ticker import MaxNLocator

from training.pbt.reporting.constants import CB_PALETTE, REPORT_PLOT_NAMES
from training.pbt.reporting.research_plots import build_member_metric_rows
from training.pbt.reporting.statistics import _paired_tier_values, ranking_agreement, tier_correlation
from training.pbt.reporting.style import plot_setup


def _save_png(fig, run_dir, plot_name_key):
    directory = Path(run_dir) / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{REPORT_PLOT_NAMES[plot_name_key]}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    return {"png": str(png_path)}


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def plot_proxy_validation(run_dir, manifest):
    """Plot control-vs-monitor/full-holdout agreement when such data exist."""
    rounds = manifest.get("tiered_evaluations") or []
    if not any(record.get("tier") in ("monitor", "full_holdout") for record in rounds):
        return None

    tier_b = "full_holdout" if any(record.get("tier") == "full_holdout" for record in rounds) else "monitor"
    independent = tier_b == "full_holdout"
    metric_name = rounds[0].get("metric_name") if rounds else None
    member_rows = build_member_metric_rows(manifest)
    winner_by_generation = {
        row["generation"]: row["trial"] for row in member_rows if row.get("is_winner")
    }
    tier_colors = {
        "control": CB_PALETTE["blue"],
        "monitor": CB_PALETTE["orange"],
        "full_holdout": CB_PALETTE["purple"],
    }

    plt = plot_setup()
    fig, (ax_series, ax_scatter) = plt.subplots(
        1, 2, figsize=(11.4, 5.0), constrained_layout=True
    )
    winner_series = {}
    for tier in ("control", "monitor", "full_holdout"):
        tier_rounds = sorted(
            (record for record in rounds if record.get("tier") == tier),
            key=lambda item: item.get("generation") if item.get("generation") is not None else -999,
        )
        xs, ys = [], []
        for record in tier_rounds:
            generation = record.get("generation")
            winner = winner_by_generation.get(generation)
            member = (record.get("members") or {}).get(winner) or {}
            value = _finite((member.get("metrics") or {}).get(record.get("metric_name")))
            if winner is not None and value is not None:
                xs.append(generation)
                ys.append(value)
        if xs:
            ax_series.plot(
                xs, ys, marker="o", markersize=4.5, linewidth=1.4,
                color=tier_colors.get(tier, "0.4"), label=tier,
            )
            winner_series[tier] = list(zip(xs, ys))
    ax_series.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_series.set_xlabel("Generation")
    ax_series.set_ylabel(metric_name or "metric value")
    ax_series.set_title("Selected-winner score by generation", fontsize=10.5, fontweight="bold", loc="left")
    ax_series.grid(True, alpha=0.3, linewidth=0.5)
    ax_series.legend(frameon=False, fontsize=8, loc="best")

    correlation = tier_correlation(manifest, "control", tier_b)
    pairs = _paired_tier_values(manifest, "control", tier_b)
    if pairs:
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        ax_scatter.scatter(xs, ys, s=32, color=tier_colors[tier_b], edgecolor="0.2", linewidth=0.4, zorder=3)
        lo, hi = min(xs + ys), max(xs + ys)
        if hi > lo:
            ax_scatter.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=0.9, zorder=2, label="y = x")
            ax_scatter.legend(frameon=False, fontsize=8, loc="lower right")

    agreement_rows = ranking_agreement(manifest, "control", tier_b)
    if correlation.get("reason") == "insufficient_paired_observations":
        caption = f"n={correlation['n']} paired points -- too few for a meaningful correlation"
    elif correlation.get("reason"):
        caption = f"n={correlation['n']}, correlation unavailable ({correlation['reason']})"
    else:
        caption = f"n={correlation['n']}  Pearson r={correlation['pearson_r']:.2f}  Spearman rho={correlation['spearman_rho']:.2f}"
    if agreement_rows:
        top1 = sum(1 for row in agreement_rows if row["top1_agrees"]) / len(agreement_rows)
        caption += f"\ntop-1 agreement: {top1:.0%} ({len(agreement_rows)} generation(s))"
    ax_scatter.text(0.02, 0.98, caption, transform=ax_scatter.transAxes, ha="left", va="top", fontsize=8, color="0.3")
    ax_scatter.set_xlabel(f"control {metric_name or ''}")
    ax_scatter.set_ylabel(f"{tier_b} {metric_name or ''}")
    ax_scatter.set_title(
        "control vs. full_holdout (independent)" if independent else "control vs. monitor (not independent)",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax_scatter.grid(True, alpha=0.3, linewidth=0.5)
    if not independent:
        fig.text(
            0.5, -0.02,
            "No full_holdout data available -- shown against monitor as a partial check, not a final independent verification.",
            ha="center", va="top", fontsize=7.6, color="0.35", transform=fig.transFigure,
        )

    result = _save_png(fig, run_dir, "proxy_validation")
    plt.close(fig)
    return {
        **result,
        "warnings": [],
        "generations": len({record.get("generation") for record in rounds if record.get("generation") is not None}),
        "members": len({member for record in rounds for member in (record.get("members") or {})}),
        "metric_keys": [metric_name] if metric_name else [],
        "independent": independent,
        "tier_b": tier_b,
        "winner_series": winner_series,
    }


def write_report_plots(run_dir, manifest):
    """Generate only the conditional proxy-validation diagnostic."""
    proxy = plot_proxy_validation(run_dir, manifest)
    return {} if proxy is None else {"proxy_validation": proxy}
